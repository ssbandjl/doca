/*
 * Copyright (c) 2015-2019 Nicira, Inc.
 * Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <config.h>
#include <ctype.h>
#include <errno.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/icmp6.h>
#include <string.h>

#include "bitmap.h"
#include "conntrack.h"
#include "conntrack-offload.h"
#include "conntrack-private.h"
#include "conntrack-tp.h"
#include "coverage.h"
#include "csum.h"
#include "ct-dist-private.h"
#include "ct-dist.h"
#include "ct-dist-thread.h"
#include "ct-dpif.h"
#include "dp-packet.h"
#include "dpif-netdev-private.h"
#include "flow.h"
#include "netdev.h"
#include "odp-netlink.h"
#include "openvswitch/hmap.h"
#include "openvswitch/vlog.h"
#include "ovs-rcu.h"
#include "ovs-thread.h"
#include "random.h"
#include "timeval.h"

VLOG_DEFINE_THIS_MODULE(ctd);

COVERAGE_DEFINE(ctd_full);
COVERAGE_DEFINE(ctd_l3csum_err);
COVERAGE_DEFINE(ctd_l4csum_err);
COVERAGE_DEFINE(ctd_lookup_natted_miss);

enum ftp_ctl_pkt {
    /* Control packets with address and/or port specifiers. */
    CT_FTP_CTL_INTEREST,
    /* Control packets without address and/or port specifiers. */
    CT_FTP_CTL_OTHER,
    CT_FTP_CTL_INVALID,
};

enum ct_alg_mode {
    CT_FTP_MODE_ACTIVE,
    CT_FTP_MODE_PASSIVE,
    CT_TFTP_MODE,
};

enum ct_alg_ctl_type {
    CT_ALG_CTL_NONE,
    CT_ALG_CTL_FTP,
    CT_ALG_CTL_TFTP,
    /* SIP is not enabled through Openflow and presently only used as
     * an example of an alg that allows a wildcard src ip. */
    CT_ALG_CTL_SIP,
};

struct zone_limit {
    struct cmap_node node;
    struct conntrack_zone_limit czl;
};

static void conn_key_reverse(struct conn_key *);
static bool valid_new(struct dp_packet *pkt, struct conn_key *);
static struct conn *new_conn(struct conntrack *ct, struct dp_packet *pkt,
                             struct conn_key *, long long now,
                             uint32_t tp_id);
static void delete_conn_cmn(struct conn *);
static void delete_conn(struct conn *);
static enum ct_update_res conn_update(struct conntrack *ct, struct conn *conn,
                                      struct dp_packet *pkt,
                                      struct conn_lookup_ctx *ctx,
                                      long long now);
static bool conn_expired(struct conn *, long long now);
static void set_mark(struct dp_packet *, struct conn *conn,
                     uint32_t, uint32_t)
    OVS_REQUIRES(conn->lock);
static void set_label(struct dp_packet *, struct conn *conn,
                      const struct ovs_key_ct_labels *,
                      const struct ovs_key_ct_labels *)
    OVS_REQUIRES(conn->lock);

static uint8_t
reverse_icmp_type(uint8_t type);
static uint8_t
reverse_icmp6_type(uint8_t type);
static inline bool
extract_l3_ipv4(struct conn_key *key, const void *data, size_t size,
                const char **new_data, bool validate_checksum);
static inline bool
extract_l3_ipv6(struct conn_key *key, const void *data, size_t size,
                const char **new_data);
static struct alg_exp_node *
expectation_lookup(struct hmap *alg_expectations, const struct conn_key *key,
                   uint32_t basis, bool src_ip_wc);

static int
repl_ftp_v4_addr(struct dp_packet *pkt, ovs_be32 v4_addr_rep,
                 char *ftp_data_v4_start,
                 size_t addr_offset_from_ftp_data_start, size_t addr_size);

static enum ftp_ctl_pkt
process_ftp_ctl_v4(struct conntrack *ct,
                   struct dp_packet *pkt,
                   const struct conn *conn_for_expectation,
                   ovs_be32 *v4_addr_rep,
                   char **ftp_data_v4_start,
                   size_t *addr_offset_from_ftp_data_start,
                   size_t *addr_size);

static enum ftp_ctl_pkt
detect_ftp_ctl_type(const struct conn_lookup_ctx *ctx,
                    struct dp_packet *pkt);

static void
expectation_clean(struct conntrack *ct, const struct conn_key *parent_key);

static void
handle_ftp_ctl(struct conntrack *ct, const struct conn_lookup_ctx *ctx,
               struct dp_packet *pkt, struct conn *ec, long long now,
               enum ftp_ctl_pkt ftp_ctl, bool nat);

static void
handle_tftp_ctl(struct conntrack *ct,
                const struct conn_lookup_ctx *ctx OVS_UNUSED,
                struct dp_packet *pkt, struct conn *conn_for_expectation,
                long long now OVS_UNUSED, enum ftp_ctl_pkt ftp_ctl OVS_UNUSED,
                bool nat OVS_UNUSED);

#define IS_PAT_PROTO(_nw_proto) \
    ((_nw_proto) == IPPROTO_TCP || (_nw_proto) == IPPROTO_UDP)

static void
ctd_nat_rev_key_init(struct dp_packet *pkt, const struct conn *conn);

typedef void (*alg_helper)(struct conntrack *ct,
                           const struct conn_lookup_ctx *ctx,
                           struct dp_packet *pkt,
                           struct conn *conn_for_expectation,
                           long long now, enum ftp_ctl_pkt ftp_ctl,
                           bool nat);

static alg_helper alg_helpers[] = {
    [CT_ALG_CTL_NONE] = NULL,
    [CT_ALG_CTL_FTP] = handle_ftp_ctl,
    [CT_ALG_CTL_TFTP] = handle_tftp_ctl,
};

/* The maximum TCP or UDP port number. */
#define CT_MAX_L4_PORT 65535
/* String buffer used for parsing FTP string messages.
 * This is sized about twice what is needed to leave some
 * margin of error. */
#define LARGEST_FTP_MSG_OF_INTEREST 128
/* FTP port string used in active mode. */
#define FTP_PORT_CMD "PORT"
/* FTP pasv string used in passive mode. */
#define FTP_PASV_REPLY_CODE "227"
/* Maximum decimal digits for port in FTP command.
 * The port is represented as two 3 digit numbers with the
 * high part a multiple of 256. */
#define MAX_FTP_PORT_DGTS 3

/* FTP extension EPRT string used for active mode. */
#define FTP_EPRT_CMD "EPRT"
/* FTP extension EPSV string used for passive mode. */
#define FTP_EPSV_REPLY "EXTENDED PASSIVE"
/* Maximum decimal digits for port in FTP extended command. */
#define MAX_EXT_FTP_PORT_DGTS 5
/* FTP extended command code for IPv6. */
#define FTP_AF_V6 '2'
/* Used to indicate a wildcard L4 source port number for ALGs.
 * This is used for port numbers that we cannot predict in
 * expectations. */
#define ALG_WC_SRC_PORT 0

/* If the total number of connections goes above this value, no new connections
 * are accepted; this is for CT_CONN_TYPE_DEFAULT connections. */
#define DEFAULT_N_CONN_LIMIT 3000000

/* Does a member by member comparison of two conn_keys; this
 * function must be kept in sync with struct conn_key; returns 0
 * if the keys are equal or 1 if the keys are not equal. */
static int
conn_key_cmp(const struct conn_key *key1, const struct conn_key *key2)
{
    if (!memcmp(&key1->src.addr, &key2->src.addr, sizeof key1->src.addr) &&
        !memcmp(&key1->dst.addr, &key2->dst.addr, sizeof key1->dst.addr) &&
        (key1->src.icmp_id == key2->src.icmp_id) &&
        (key1->src.icmp_type == key2->src.icmp_type) &&
        (key1->src.icmp_code == key2->src.icmp_code) &&
        (key1->dst.icmp_id == key2->dst.icmp_id) &&
        (key1->dst.icmp_type == key2->dst.icmp_type) &&
        (key1->dst.icmp_code == key2->dst.icmp_code) &&
        (key1->dl_type == key2->dl_type) &&
        (key1->zone == key2->zone) &&
        (key1->nw_proto == key2->nw_proto)) {

        return 0;
    }
    return 1;
}

static void
ct_print_conn_info(const struct conn *c, const char *log_msg,
                   enum vlog_level vll, bool force, bool rl_on)
{
#define CT_VLOG(RL_ON, LEVEL, ...)                                          \
    do {                                                                    \
        if (RL_ON) {                                                        \
            static struct vlog_rate_limit rl_ = VLOG_RATE_LIMIT_INIT(5, 5); \
            vlog_rate_limit(&this_module, LEVEL, &rl_, __VA_ARGS__);        \
        } else {                                                            \
            vlog(&this_module, LEVEL, __VA_ARGS__);                         \
        }                                                                   \
    } while (0)

    if (OVS_UNLIKELY(force || vlog_is_enabled(&this_module, vll))) {
        if (c->key.dl_type == htons(ETH_TYPE_IP)) {
            CT_VLOG(rl_on, vll, "%s: src ip "IP_FMT" dst ip "IP_FMT" rev src "
                    "ip "IP_FMT" rev dst ip "IP_FMT" src/dst ports "
                    "%"PRIu16"/%"PRIu16" rev src/dst ports "
                    "%"PRIu16"/%"PRIu16" zone/rev zone "
                    "%"PRIu16"/%"PRIu16" nw_proto/rev nw_proto "
                    "%"PRIu8"/%"PRIu8, log_msg,
                    IP_ARGS(c->key.src.addr.ipv4),
                    IP_ARGS(c->key.dst.addr.ipv4),
                    IP_ARGS(c->rev_key.src.addr.ipv4),
                    IP_ARGS(c->rev_key.dst.addr.ipv4),
                    ntohs(c->key.src.port), ntohs(c->key.dst.port),
                    ntohs(c->rev_key.src.port), ntohs(c->rev_key.dst.port),
                    c->key.zone, c->rev_key.zone, c->key.nw_proto,
                    c->rev_key.nw_proto);
        } else {
            char ip6_s[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &c->key.src.addr.ipv6, ip6_s, sizeof ip6_s);
            char ip6_d[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &c->key.dst.addr.ipv6, ip6_d, sizeof ip6_d);
            char ip6_rs[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &c->rev_key.src.addr.ipv6, ip6_rs,
                      sizeof ip6_rs);
            char ip6_rd[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &c->rev_key.dst.addr.ipv6, ip6_rd,
                      sizeof ip6_rd);

            CT_VLOG(rl_on, vll, "%s: src ip %s dst ip %s rev src ip %s"
                    " rev dst ip %s src/dst ports %"PRIu16"/%"PRIu16
                    " rev src/dst ports %"PRIu16"/%"PRIu16" zone/rev zone "
                    "%"PRIu16"/%"PRIu16" nw_proto/rev nw_proto "
                    "%"PRIu8"/%"PRIu8, log_msg, ip6_s, ip6_d, ip6_rs,
                    ip6_rd, ntohs(c->key.src.port), ntohs(c->key.dst.port),
                    ntohs(c->rev_key.src.port), ntohs(c->rev_key.dst.port),
                    c->key.zone, c->rev_key.zone, c->key.nw_proto,
                    c->rev_key.nw_proto);
        }
    }
}

static uint32_t
zone_key_hash(int32_t zone, uint32_t basis)
{
    size_t hash = hash_int((OVS_FORCE uint32_t) zone, basis);
    return hash;
}

static struct zone_limit *
zone_limit_lookup(struct conntrack *ct, int32_t zone)
{
    uint32_t hash = zone_key_hash(zone, ct->hash_basis);
    struct zone_limit *zl;
    CMAP_FOR_EACH_WITH_HASH (zl, node, hash, &ct->zone_limits) {
        if (zl->czl.zone == zone) {
            return zl;
        }
    }
    return NULL;
}

static struct zone_limit *
zone_limit_lookup_or_default(struct conntrack *ct, int32_t zone)
{
    struct zone_limit *zl = zone_limit_lookup(ct, zone);
    return zl ? zl : zone_limit_lookup(ct, DEFAULT_ZONE);
}

static void
conn_do_delete(struct conn *conn,
               void (*delete_cb)(struct conn *))
{
    ovsrcu_gc(delete_cb, conn, gc_node);
}

bool
conn_unref(struct conn *conn)
{
    if (ovs_refcount_unref(&conn->exp.refcount) == 1) {
        conn_do_delete(conn, delete_conn);
        return true;
    }
    return false;
}

static void
conn_force_expire(struct conn *conn)
{
    atomic_store_relaxed(&conn->expiration, 0);
}



static bool
conn_key_lookup(struct conntrack *ct, const struct conn_key *key,
                uint32_t hash, long long now, struct conn **conn_out,
                bool *reply)
{
    struct conn *conn;
    bool found = false;
    bool is_reply = false;

    CMAP_FOR_EACH_WITH_HASH (conn, cm_node, hash, &ct->conns) {
        if (conn_key_cmp(&conn->key, key)) {
            if (conn_key_cmp(&conn->rev_key, key)) {
                continue;
            } else {
                is_reply = true;
            }
        }

        if (conn_expired(conn, now)) {
            break;
        }

        found = true;
        if (reply) {
            *reply = is_reply;
        }
        break;
    }

    if (found && conn_out) {
        *conn_out = conn;
    } else if (conn_out) {
        *conn_out = NULL;
    }
    return found;
}

static bool
conn_lookup(struct conntrack *ct, const struct conn_key *key,
            long long now, struct conn **conn_out, bool *reply)
{
    uint32_t hash = conn_key_hash(key, ct->hash_basis);
    return conn_key_lookup(ct, key, hash, now, conn_out, reply);
}

static void
write_ct_md_key(struct dp_packet *pkt, const struct conn_key *key)
{

    pkt->md.ct_orig_tuple_ipv6 = false;

    if (key) {
        if (key->dl_type == htons(ETH_TYPE_IP)) {
            pkt->md.ct_orig_tuple.ipv4 = (struct ovs_key_ct_tuple_ipv4) {
                key->src.addr.ipv4,
                key->dst.addr.ipv4,
                key->nw_proto != IPPROTO_ICMP
                ? key->src.port : htons(key->src.icmp_type),
                key->nw_proto != IPPROTO_ICMP
                ? key->dst.port : htons(key->src.icmp_code),
                key->nw_proto,
            };
        } else {
            pkt->md.ct_orig_tuple_ipv6 = true;
            pkt->md.ct_orig_tuple.ipv6 = (struct ovs_key_ct_tuple_ipv6) {
                key->src.addr.ipv6,
                key->dst.addr.ipv6,
                key->nw_proto != IPPROTO_ICMPV6
                ? key->src.port : htons(key->src.icmp_type),
                key->nw_proto != IPPROTO_ICMPV6
                ? key->dst.port : htons(key->src.icmp_code),
                key->nw_proto,
            };
        }
    } else {
        memset(&pkt->md.ct_orig_tuple, 0, sizeof pkt->md.ct_orig_tuple);
    }
}

static void
write_ct_md_conn(struct dp_packet *pkt, uint16_t zone,
                 const struct conn *conn)
    OVS_REQUIRES(conn->lock)
{
    const struct conn_key *key;

    pkt->md.ct_state |= CS_TRACKED;
    pkt->md.ct_zone = zone;

    pkt->md.ct_mark = conn->mark;
    pkt->md.ct_label = conn->label;

    /* Use the original direction tuple if we have it. */
    if (conn->alg_related) {
        key = &conn->parent_key;
    } else {
        key = &conn->key;
    }

    write_ct_md_key(pkt, key);
}

static void
write_ct_md_alg_exp(struct dp_packet *pkt, uint16_t zone,
                    const struct conn_key *key, const struct alg_exp_node *alg_exp)
{
    pkt->md.ct_state |= CS_TRACKED;
    pkt->md.ct_zone = zone;

    if (alg_exp) {
        pkt->md.ct_mark = alg_exp->parent_mark;
        pkt->md.ct_label = alg_exp->parent_label;
        key = &alg_exp->parent_key;
    } else {
        pkt->md.ct_mark = 0;
        pkt->md.ct_label = OVS_U128_ZERO;
    }

    write_ct_md_key(pkt, key);
}

static uint8_t
get_ip_proto(const struct dp_packet *pkt)
{
    uint8_t ip_proto;
    struct eth_header *l2 = dp_packet_eth(pkt);
    if (l2->eth_type == htons(ETH_TYPE_IPV6)) {
        struct ovs_16aligned_ip6_hdr *nh6 = dp_packet_l3(pkt);
        ip_proto = nh6->ip6_ctlun.ip6_un1.ip6_un1_nxt;
    } else {
        struct ip_header *l3_hdr = dp_packet_l3(pkt);
        ip_proto = l3_hdr->ip_proto;
    }

    return ip_proto;
}

static bool
is_ftp_ctl(const enum ct_alg_ctl_type ct_alg_ctl)
{
    return ct_alg_ctl == CT_ALG_CTL_FTP;
}

static enum ct_alg_ctl_type
get_alg_ctl_type(const struct dp_packet *pkt, ovs_be16 tp_src, ovs_be16 tp_dst,
                 const char *helper)
{
    /* CT_IPPORT_FTP/TFTP is used because IPPORT_FTP/TFTP in not defined
     * in OSX, at least in in.h. Since these values will never change, remove
     * the external dependency. */
    enum { CT_IPPORT_FTP = 21 };
    enum { CT_IPPORT_TFTP = 69 };
    uint8_t ip_proto = get_ip_proto(pkt);
    struct udp_header *uh = dp_packet_l4(pkt);
    struct tcp_header *th = dp_packet_l4(pkt);
    ovs_be16 ftp_src_port = htons(CT_IPPORT_FTP);
    ovs_be16 ftp_dst_port = htons(CT_IPPORT_FTP);
    ovs_be16 tftp_dst_port = htons(CT_IPPORT_TFTP);

    if (OVS_UNLIKELY(tp_dst)) {
        if (helper && !strncmp(helper, "ftp", strlen("ftp"))) {
            ftp_dst_port = tp_dst;
        } else if (helper && !strncmp(helper, "tftp", strlen("tftp"))) {
            tftp_dst_port = tp_dst;
        }
    } else if (OVS_UNLIKELY(tp_src)) {
        if (helper && !strncmp(helper, "ftp", strlen("ftp"))) {
            ftp_src_port = tp_src;
        }
    }

    if (ip_proto == IPPROTO_UDP && uh->udp_dst == tftp_dst_port) {
        return CT_ALG_CTL_TFTP;
    } else if (ip_proto == IPPROTO_TCP &&
               (th->tcp_src == ftp_src_port || th->tcp_dst == ftp_dst_port)) {
        return CT_ALG_CTL_FTP;
    }
    return CT_ALG_CTL_NONE;
}

static bool
alg_src_ip_wc(enum ct_alg_ctl_type alg_ctl_type)
{
    if (alg_ctl_type == CT_ALG_CTL_SIP) {
        return true;
    }
    return false;
}

static void
handle_alg_ctl(struct conntrack *ct, const struct conn_lookup_ctx *ctx,
               struct dp_packet *pkt, enum ct_alg_ctl_type ct_alg_ctl,
               struct conn *conn, long long now, bool nat)
{
    /* ALG control packet handling with expectation creation. */
    if (OVS_UNLIKELY(alg_helpers[ct_alg_ctl] && conn && conn->alg)) {
        conn_lock(conn);
        alg_helpers[ct_alg_ctl](ct, ctx, pkt, conn, now, CT_FTP_CTL_INTEREST,
                                nat);
        conn_unlock(conn);
    }
}

static void
pat_packet(struct dp_packet *pkt, const struct conn_key *key)
{
    if (key->nw_proto == IPPROTO_TCP) {
        packet_set_tcp_port(pkt, key->dst.port, key->src.port);
    } else if (key->nw_proto == IPPROTO_UDP) {
        packet_set_udp_port(pkt, key->dst.port, key->src.port);
    }
}

static uint16_t
nat_action_reverse(uint16_t nat_action)
{
    if (nat_action & NAT_ACTION_SRC) {
        nat_action ^= NAT_ACTION_SRC;
        nat_action |= NAT_ACTION_DST;
    } else if (nat_action & NAT_ACTION_DST) {
        nat_action ^= NAT_ACTION_DST;
        nat_action |= NAT_ACTION_SRC;
    }
    return nat_action;
}

static void
nat_packet_ipv4(struct dp_packet *pkt, const struct conn_key *key,
                uint16_t nat_action)
{
    struct ip_header *nh = dp_packet_l3(pkt);

    if (nat_action & NAT_ACTION_SRC) {
        packet_set_ipv4_addr(pkt, &nh->ip_src, key->dst.addr.ipv4);
    } else if (nat_action & NAT_ACTION_DST) {
        packet_set_ipv4_addr(pkt, &nh->ip_dst, key->src.addr.ipv4);
    }
}

static void
nat_packet_ipv6(struct dp_packet *pkt, const struct conn_key *key,
                uint16_t nat_action)
{
    struct ovs_16aligned_ip6_hdr *nh6 = dp_packet_l3(pkt);

    if (nat_action & NAT_ACTION_SRC) {
        packet_set_ipv6_addr(pkt, key->nw_proto, nh6->ip6_src.be32,
                             &key->dst.addr.ipv6, true);
    } else if (nat_action & NAT_ACTION_DST) {
        packet_set_ipv6_addr(pkt, key->nw_proto, nh6->ip6_dst.be32,
                             &key->src.addr.ipv6, true);
    }
}

static void
nat_inner_packet(struct dp_packet *pkt, struct conn_key *key,
                 uint16_t nat_action)
{
    char *tail = dp_packet_tail(pkt);
    uint16_t pad = dp_packet_l2_pad_size(pkt);
    struct conn_key inner_key;
    const char *inner_l4 = NULL;
    uint16_t orig_l3_ofs = pkt->l3_ofs;
    uint16_t orig_l4_ofs = pkt->l4_ofs;

    void *l3 = dp_packet_l3(pkt);
    void *l4 = dp_packet_l4(pkt);
    void *inner_l3;
    /* These calls are already verified to succeed during the code path from
     * 'conn_key_extract()' which calls
     * 'extract_l4_icmp()'/'extract_l4_icmp6()'. */
    if (key->dl_type == htons(ETH_TYPE_IP)) {
        inner_l3 = (char *) l4 + sizeof(struct icmp_header);
        extract_l3_ipv4(&inner_key, inner_l3, tail - ((char *) inner_l3) - pad,
                        &inner_l4, false);
    } else {
        inner_l3 = (char *) l4 + sizeof(struct icmp6_data_header);
        extract_l3_ipv6(&inner_key, inner_l3, tail - ((char *) inner_l3) - pad,
                        &inner_l4);
    }
    pkt->l3_ofs += (char *) inner_l3 - (char *) l3;
    pkt->l4_ofs += inner_l4 - (char *) l4;

    /* Reverse the key for inner packet. */
    struct conn_key rev_key = *key;
    conn_key_reverse(&rev_key);

    pat_packet(pkt, &rev_key);

    if (key->dl_type == htons(ETH_TYPE_IP)) {
        nat_packet_ipv4(pkt, &rev_key, nat_action);

        struct icmp_header *icmp = (struct icmp_header *) l4;
        icmp->icmp_csum = 0;
        icmp->icmp_csum = csum(icmp, tail - (char *) icmp - pad);
    } else {
        nat_packet_ipv6(pkt, &rev_key, nat_action);

        struct icmp6_data_header *icmp6 = (struct icmp6_data_header *) l4;
        icmp6->icmp6_base.icmp6_cksum = 0;
        icmp6->icmp6_base.icmp6_cksum =
            packet_csum_upperlayer6(l3, icmp6, IPPROTO_ICMPV6,
                                    tail - (char *) icmp6 - pad);
    }

    pkt->l3_ofs = orig_l3_ofs;
    pkt->l4_ofs = orig_l4_ofs;
}

static void
nat_packet(struct dp_packet *pkt, struct conn *conn, bool reply, bool related)
{
    struct conn_key *key = reply ? &conn->key : &conn->rev_key;
    uint16_t nat_action = reply ? nat_action_reverse(conn->nat_action)
                                : conn->nat_action;

    /* Update ct_state. */
    if (nat_action & NAT_ACTION_SRC) {
        pkt->md.ct_state |= CS_SRC_NAT;
    } else if (nat_action & NAT_ACTION_DST) {
        pkt->md.ct_state |= CS_DST_NAT;
    }

    /* Reverse the key for outer header. */
    if (key->dl_type == htons(ETH_TYPE_IP)) {
        nat_packet_ipv4(pkt, key, nat_action);
    } else {
        nat_packet_ipv6(pkt, key, nat_action);
    }

    if (nat_action & NAT_ACTION_SRC || nat_action & NAT_ACTION_DST) {
        if (OVS_UNLIKELY(related)) {
            nat_action = nat_action_reverse(nat_action);
            nat_inner_packet(pkt, key, nat_action);
        } else {
            pat_packet(pkt, key);
        }
    }
}

static void
conn_seq_skew_set(struct conntrack *ct, const struct conn *conn_in,
                  long long now, int seq_skew, bool seq_skew_dir)
    OVS_NO_THREAD_SAFETY_ANALYSIS
{
    struct conn *conn;
    conn_unlock(conn_in);
    conn_lookup(ct, &conn_in->key, now, &conn, NULL);
    conn_lock(conn_in);

    if (conn && seq_skew) {
        conn->seq_skew = seq_skew;
        conn->seq_skew_dir = seq_skew_dir;
    }
}

static bool
ct_verify_helper(const char *helper, enum ct_alg_ctl_type ct_alg_ctl)
{
    if (ct_alg_ctl == CT_ALG_CTL_NONE) {
        return true;
    } else if (helper) {
        if ((ct_alg_ctl == CT_ALG_CTL_FTP) &&
             !strncmp(helper, "ftp", strlen("ftp"))) {
            return true;
        } else if ((ct_alg_ctl == CT_ALG_CTL_TFTP) &&
                   !strncmp(helper, "tftp", strlen("tftp"))) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

static void
ctd_nat_conn_init(struct dp_packet *pkt,
                  struct conn *conn,
                  struct conn *nat_conn)
{
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;

    memcpy(&nat_conn->key, &conn->rev_key, sizeof nat_conn->key);
    memcpy(&nat_conn->rev_key, &conn->key, sizeof nat_conn->rev_key);
    nat_conn->conn_type = CT_CONN_TYPE_UN_NAT;
    nat_conn->nat_action = 0;
    nat_conn->alg = NULL;
    nat_conn->nat_conn = NULL;
    nat_conn->master_conn = conn;
    conntrack_lock(m->ct);
    cmap_insert(&m->ct->conns, &nat_conn->cm_node, e->nli.hash);
    conntrack_unlock(m->ct);
}

static void
ctd_nat_reorder_packet_from_orig(struct dp_packet *pkt, struct conn *conn)
{
    struct ctd_msg *m = &pkt->cme.hdr;

    /* Early bail out for standard in-order packets. */
    if (!conn->reordering) {
        return;
    }

    /* When the response_pkt arrives, handle it and set it as NULL, marking
     * we now wait for the resume_pkt.
     */
    if (pkt == conn->response_pkt) {
        conn->response_pkt = NULL;
        return;
    }

    /* When the resume_pkt arrives, handle it and cancel the reordering
     * state.
     */
    if (pkt == conn->resume_pkt) {
        conn->reordering = false;
        conn->resume_pkt = NULL;
        return;
    }

    /* In any other case, defer the packet. */
    ctd_msg_fate_set(m, CTD_MSG_FATE_SELF);
}

static struct conn *
ctd_conn_not_found(struct conntrack *ct, struct dp_packet *pkt,
                   struct conn_lookup_ctx *ctx, bool commit, long long now,
                   const struct nat_action_info_t *nat_action_info,
                   const char *helper, const struct alg_exp_node *alg_exp,
                   enum ct_alg_ctl_type ct_alg_ctl, uint32_t tp_id)
{
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;
    struct conn *nat_conn = NULL;
    struct zone_limit *zl = NULL;
    struct nat_lookup_info *nli;
    struct conn *nc = NULL;

    nli = &e->nli;

    if (commit) {
        zl = zone_limit_lookup_or_default(ct, ctx->key.zone);
    }

    if (m->msg_type == CTD_MSG_NAT_CANDIDATE_RESPONSE) {
        goto CTD_MSG_NAT_CANDIDATE_RESPONSE;
    }

    if (!valid_new(pkt, &ctx->key)) {
        pkt->md.ct_state = CS_INVALID;
        return nc;
    }

    pkt->md.ct_state = CS_NEW;

    if (alg_exp) {
        pkt->md.ct_state |= CS_RELATED;
    }

    if (commit) {
        if (zl && atomic_count_get(&zl->czl.count) >= zl->czl.limit) {
            return nc;
        }

        unsigned int n_conn_limit;
        atomic_read_relaxed(&ct->n_conn_limit, &n_conn_limit);
        if (atomic_count_get(&ct->n_conn) >= n_conn_limit) {
            COVERAGE_INC(ctd_full);
            return nc;
        }

        nc = new_conn(ct, pkt, &ctx->key, now, tp_id);
        nc->conn_type = CT_CONN_TYPE_DEFAULT;
        memcpy(&nc->key, &ctx->key, sizeof nc->key);
        memcpy(&nc->rev_key, &nc->key, sizeof nc->rev_key);
        conn_key_reverse(&nc->rev_key);

        if (ct_verify_helper(helper, ct_alg_ctl)) {
            nc->alg = nullable_xstrdup(helper);
        }

        if (alg_exp) {
            nc->alg_related = true;
            nc->mark = alg_exp->parent_mark;
            nc->label = alg_exp->parent_label;
            nc->parent_key = alg_exp->parent_key;
        }

        if (nat_action_info) {
            nc->nat_action = nat_action_info->nat_action;

            if (alg_exp) {
                if (alg_exp->nat_rpl_dst) {
                    nc->rev_key.dst.addr = alg_exp->alg_nat_repl_addr;
                    nc->nat_action = NAT_ACTION_SRC;
                } else {
                    nc->rev_key.src.addr = alg_exp->alg_nat_repl_addr;
                    nc->nat_action = NAT_ACTION_DST;
                }
                nli->hash = conn_key_hash(&nc->rev_key, ct->hash_basis);
                nat_conn = xzalloc(sizeof *nat_conn);
                ctd_nat_conn_init(pkt, nc, nat_conn);
            } else {
                memcpy(&nli->rev_key, &nc->rev_key, sizeof nli->rev_key);
                ctd_nat_rev_key_init(pkt, nc);
                nli->hash = conn_key_hash(&nli->rev_key, ct->hash_basis);

                ctx->conn = nc;
                conn_lock_init(nc);
                nc->reordering = true;
                nc->resume_pkt = pkt;
                atomic_flag_clear(&nc->reclaimed);
                conntrack_lock(ct);
                cmap_insert(&ct->conns, &nc->cm_node, ctx->hash);
                conntrack_unlock(ct);
                ctd_msg_type_set(m, CTD_MSG_NAT_CANDIDATE);
                ctd_msg_dest_set(m, nli->hash);
                ctd_msg_fate_set(m, CTD_MSG_FATE_CTD);
                return NULL;

CTD_MSG_NAT_CANDIDATE_RESPONSE:
                ovs_assert(ctx->conn);
                nat_conn = nli->nat_conn;
                nc = ctx->conn;
                if (!nat_conn) {
                    delete_conn_cmn(nc);
                    ctx->conn = NULL;
                    return NULL;
                }
                ctd_nat_reorder_packet_from_orig(pkt, nc);
                if (conn_expired(nc, now)) {
                    ctd_msg_conn_clean_send(ct, nc, ctx->hash);
                    return NULL;
                }
                ctd_msg_type_set(m, CTD_MSG_EXEC);
            }
        }

        nc->nat_conn = nat_conn;
        conn_lock_init(nc);
        /* In case of nat and not alg, the conn was already inserted before
         * sending the candidate message, so no don't insert it again.
         */
        if (!(nat_action_info && !alg_exp)) {
            atomic_flag_clear(&nc->reclaimed);
            conntrack_lock(ct);
            cmap_insert(&ct->conns, &nc->cm_node, ctx->hash);
            conntrack_unlock(ct);
        }
        conn_expire_push_back(ct, nc);
        atomic_count_inc(&ct->n_conn);
        atomic_count_inc(&ct->l4_counters[ctx->key.nw_proto]);
        ctx->conn = nc; /* For completeness. */

        if (zl) {
            nc->admit_zone = zl->czl.zone;
            nc->zone_limit_seq = zl->czl.zone_limit_seq;
            atomic_count_inc(&zl->czl.count);
        } else {
            nc->admit_zone = INVALID_ZONE;
        }
    }

    return nc;
}

static bool
conn_update_state(struct conntrack *ct, struct dp_packet *pkt,
                  struct conn_lookup_ctx *ctx, struct conn *conn,
                  long long now)
{
    ovs_assert(conn->conn_type == CT_CONN_TYPE_DEFAULT);
    bool create_new_conn = false;

    if (ctx->icmp_related) {
        pkt->md.ct_state |= CS_RELATED;
        if (ctx->reply) {
            pkt->md.ct_state |= CS_REPLY_DIR;
        }
    } else {
        if (conn->alg_related) {
            pkt->md.ct_state |= CS_RELATED;
        }

        enum ct_update_res res = conn_update(ct, conn, pkt, ctx, now);

        switch (res) {
        case CT_UPDATE_VALID:
            pkt->md.ct_state |= CS_ESTABLISHED;
            pkt->md.ct_state &= ~CS_NEW;
            if (ctx->reply) {
                pkt->md.ct_state |= CS_REPLY_DIR;
            }
            break;
        case CT_UPDATE_INVALID:
            pkt->md.ct_state = CS_INVALID;
            break;
        case CT_UPDATE_NEW:
            if (conn_lookup(ct, &conn->key, now, NULL, NULL)) {
                conn_force_expire(conn);
            }
            create_new_conn = true;
            break;
        case CT_UPDATE_VALID_NEW:
            pkt->md.ct_state |= CS_NEW;
            break;
        default:
            OVS_NOT_REACHED();
        }
    }
    return create_new_conn;
}

static void
handle_nat(struct dp_packet *pkt, struct conn *conn,
           uint16_t zone, bool reply, bool related)
{
    if (conn->nat_action &&
        (!(pkt->md.ct_state & (CS_SRC_NAT | CS_DST_NAT)) ||
          (pkt->md.ct_state & (CS_SRC_NAT | CS_DST_NAT) &&
           zone != pkt->md.ct_zone))) {

        if (pkt->md.ct_state & (CS_SRC_NAT | CS_DST_NAT)) {
            pkt->md.ct_state &= ~(CS_SRC_NAT | CS_DST_NAT);
        }

        nat_packet(pkt, conn, reply, related);
    }
}

static bool
check_orig_tuple(struct conntrack *ct, struct dp_packet *pkt,
                 struct conn_lookup_ctx *ctx_in, long long now,
                 struct conn **conn,
                 const struct nat_action_info_t *nat_action_info)
{
    if (!(pkt->md.ct_state & (CS_SRC_NAT | CS_DST_NAT)) ||
        (ctx_in->key.dl_type == htons(ETH_TYPE_IP) &&
         !pkt->md.ct_orig_tuple.ipv4.ipv4_proto) ||
        (ctx_in->key.dl_type == htons(ETH_TYPE_IPV6) &&
         !pkt->md.ct_orig_tuple.ipv6.ipv6_proto) ||
        nat_action_info) {
        return false;
    }

    struct conn_key key;
    memset(&key, 0 , sizeof key);

    if (ctx_in->key.dl_type == htons(ETH_TYPE_IP)) {
        key.src.addr.ipv4 = pkt->md.ct_orig_tuple.ipv4.ipv4_src;
        key.dst.addr.ipv4 = pkt->md.ct_orig_tuple.ipv4.ipv4_dst;

        if (ctx_in->key.nw_proto == IPPROTO_ICMP) {
            key.src.icmp_id = ctx_in->key.src.icmp_id;
            key.dst.icmp_id = ctx_in->key.dst.icmp_id;
            uint16_t src_port = ntohs(pkt->md.ct_orig_tuple.ipv4.src_port);
            key.src.icmp_type = (uint8_t) src_port;
            key.dst.icmp_type = reverse_icmp_type(key.src.icmp_type);
        } else {
            key.src.port = pkt->md.ct_orig_tuple.ipv4.src_port;
            key.dst.port = pkt->md.ct_orig_tuple.ipv4.dst_port;
        }
        key.nw_proto = pkt->md.ct_orig_tuple.ipv4.ipv4_proto;
    } else {
        key.src.addr.ipv6 = pkt->md.ct_orig_tuple.ipv6.ipv6_src;
        key.dst.addr.ipv6 = pkt->md.ct_orig_tuple.ipv6.ipv6_dst;

        if (ctx_in->key.nw_proto == IPPROTO_ICMPV6) {
            key.src.icmp_id = ctx_in->key.src.icmp_id;
            key.dst.icmp_id = ctx_in->key.dst.icmp_id;
            uint16_t src_port = ntohs(pkt->md.ct_orig_tuple.ipv6.src_port);
            key.src.icmp_type = (uint8_t) src_port;
            key.dst.icmp_type = reverse_icmp6_type(key.src.icmp_type);
        } else {
            key.src.port = pkt->md.ct_orig_tuple.ipv6.src_port;
            key.dst.port = pkt->md.ct_orig_tuple.ipv6.dst_port;
        }
        key.nw_proto = pkt->md.ct_orig_tuple.ipv6.ipv6_proto;
    }

    key.dl_type = ctx_in->key.dl_type;
    key.zone = pkt->md.ct_zone;
    conn_lookup(ct, &key, now, conn, NULL);
    return *conn ? true : false;
}

static bool
conn_update_state_alg(struct conntrack *ct, struct dp_packet *pkt,
                      struct conn_lookup_ctx *ctx, struct conn *conn,
                      const struct nat_action_info_t *nat_action_info,
                      enum ct_alg_ctl_type ct_alg_ctl, long long now,
                      bool *create_new_conn)
{
    if (is_ftp_ctl(ct_alg_ctl)) {
        /* Keep sequence tracking in sync with the source of the
         * sequence skew. */
        conn_lock(conn);
        if (ctx->reply != conn->seq_skew_dir) {
            handle_ftp_ctl(ct, ctx, pkt, conn, now, CT_FTP_CTL_OTHER,
                           !!nat_action_info);
            /* conn_update_state locks for unrelated fields, so unlock. */
            conn_unlock(conn);
            *create_new_conn = conn_update_state(ct, pkt, ctx, conn, now);
        } else {
            /* conn_update_state locks for unrelated fields, so unlock. */
            conn_unlock(conn);
            *create_new_conn = conn_update_state(ct, pkt, ctx, conn, now);
            conn_lock(conn);
            if (*create_new_conn == false) {
                handle_ftp_ctl(ct, ctx, pkt, conn, now, CT_FTP_CTL_OTHER,
                               !!nat_action_info);
            }
            conn_unlock(conn);
        }
        return true;
    }
    return false;
}

static void
set_cached_conn(const struct nat_action_info_t *nat_action_info,
                const struct conn_lookup_ctx *ctx, struct conn *conn,
                struct dp_packet *pkt)
{
    if (OVS_LIKELY(!nat_action_info)) {
        pkt->md.conn = conn;
        pkt->md.reply = ctx->reply;
        pkt->md.icmp_related = ctx->icmp_related;
    } else {
        pkt->md.conn = NULL;
    }
}

static void
process_one_fast(uint16_t zone, const uint32_t *setmark,
                 const struct ovs_key_ct_labels *setlabel,
                 const struct nat_action_info_t *nat_action_info,
                 struct conn *conn, struct dp_packet *pkt)
{
    if (nat_action_info) {
        handle_nat(pkt, conn, zone, pkt->md.reply, pkt->md.icmp_related);
        pkt->md.conn = NULL;
    }

    pkt->md.ct_zone = zone;
    conn_lock(conn);
    pkt->md.ct_mark = conn->mark;
    pkt->md.ct_label = conn->label;

    if (setmark) {
        set_mark(pkt, conn, setmark[0], setmark[1]);
    }

    if (setlabel) {
        set_label(pkt, conn, &setlabel[0], &setlabel[1]);
    }

    conn_unlock(conn);
}

static void
initial_conn_lookup(struct conntrack *ct, struct conn_lookup_ctx *ctx,
                    long long now, bool natted)
{
    if (natted) {
        /* If the packet has been already natted (e.g. a previous
         * action took place), retrieve it performing a lookup of its
         * reverse key. */
        conn_key_reverse(&ctx->key);
    }

    conn_key_lookup(ct, &ctx->key, ctx->hash, now, &ctx->conn, &ctx->reply);

    if (natted) {
        if (OVS_LIKELY(ctx->conn)) {
            ctx->reply = !ctx->reply;
            ctx->key = ctx->reply ? ctx->conn->rev_key : ctx->conn->key;
            ctx->hash = conn_key_hash(&ctx->key, ct->hash_basis);
        } else {
            /* A lookup failure does not necessarily imply that an
             * error occurred, it may simply indicate that a conn got
             * removed during the recirculation. */
            COVERAGE_INC(ctd_lookup_natted_miss);
            conn_key_reverse(&ctx->key);
        }
    }
}

static struct conn *
ctd_process_one_init(struct dp_packet *pkt)
{
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;
    struct conn_lookup_ctx *ctx;
    struct conntrack *ct;
    struct conn *conn;
    uint16_t zone;
    long long now;
    bool force;

    ct = m->ct;
    zone = e->zone;
    force = e->force;
    now = m->timestamp_ms;
    ctx = &e->ct_lookup_ctx;

    /* Reset ct_state whenever entering a new zone. */
    if (pkt->md.ct_state && pkt->md.ct_zone != zone) {
        pkt->md.ct_state = 0;
    }

    initial_conn_lookup(ct, ctx, now, !!(pkt->md.ct_state &
                                         (CS_SRC_NAT | CS_DST_NAT)));
    conn = ctx->conn;

    /* Delete found entry if in wrong direction. 'force' implies commit. */
    if (OVS_UNLIKELY(force && ctx->reply && conn)) {
        if (conn_lookup(ct, &conn->key, now, NULL, NULL)) {
            conn_force_expire(conn);
        }
        conn = NULL;
    }

    return conn;
}

static struct conn *
ctd_process_conn_type_un_nat(struct dp_packet *pkt, struct conn *conn)
{
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;
    struct conn_lookup_ctx *ctx;
    struct conntrack *ct;
    uint16_t zone;
    long long now;

    ovs_assert(conn);
    ct = m->ct;
    zone = e->zone;
    now = m->timestamp_ms;
    ctx = &e->ct_lookup_ctx;

    ctx->reply = true;
    struct conn *rev_conn = conn;  /* Save for debugging. */
    uint32_t hash = conn_key_hash(&conn->rev_key, ct->hash_basis);
    conn_key_lookup(ct, &ctx->key, hash, now, &conn, &ctx->reply);

    if (!conn) {
        pkt->md.ct_state |= CS_INVALID;
        write_ct_md_alg_exp(pkt, zone, NULL, NULL);
        char *log_msg = xasprintf("Missing parent conn %p", rev_conn);
        ct_print_conn_info(rev_conn, log_msg, VLL_INFO, true, true);
        free(log_msg);
    }

    return conn;
}

static void
ctd_process_one(struct dp_packet *pkt)
{
    const struct nat_action_info_t *nat_action_info;
    const struct ovs_key_ct_labels *setlabel;
    const struct alg_exp_node *alg_exp;
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;
    enum ct_alg_ctl_type ct_alg_ctl;
    bool create_new_conn = false;
    struct nat_lookup_info *nli;
    struct conn_lookup_ctx *ctx;
    struct conn *conn = NULL;
    const uint32_t *setmark;
    struct conntrack *ct;
    const char *helper;
    ovs_be16 tp_src;
    ovs_be16 tp_dst;
    uint32_t tp_id;
    uint16_t zone;
    long long now;
    bool commit;

    ct = m->ct;
    zone = e->zone;
    commit = e->commit;
    now = m->timestamp_ms;
    setmark = e->setmark;
    setlabel = e->setlabel;
    nat_action_info = e->nat_action_info_ref;
    tp_src = e->tp_src;
    tp_dst = e->tp_dst;
    helper = e->helper;
    tp_id = e->tp_id;
    ctx = &e->ct_lookup_ctx;
    nli = &e->nli;

    if (m->msg_type == CTD_MSG_NAT_CANDIDATE_RESPONSE) {
        alg_exp = NULL;
        ct_alg_ctl = CT_ALG_CTL_NONE;
        goto CTD_MSG_NAT_CANDIDATE_RESPONSE;
    }

    if (m->msg_type == CTD_MSG_EXEC) {
        conn = ctd_process_one_init(pkt);

        if (OVS_LIKELY(conn)) {
            if (conn->conn_type == CT_CONN_TYPE_UN_NAT) {
                ctx->reply = true;
                nli->hash = conn_key_hash(&conn->rev_key, ct->hash_basis);
                ctd_msg_type_set(m, CTD_MSG_EXEC_NAT);
                ctd_msg_dest_set(m, nli->hash);
                ctd_msg_fate_set(m, CTD_MSG_FATE_CTD);
                return;
            }

            if (nat_action_info) {
                ctd_nat_reorder_packet_from_orig(pkt, conn);
                if (m->msg_fate == CTD_MSG_FATE_SELF) {
                    return;
                }
            }
        }
    } else if (m->msg_type == CTD_MSG_EXEC_NAT) {
        conn = ctd_process_conn_type_un_nat(pkt, ctx->conn);
        if (!conn) {
            ctd_msg_fate_set(m, CTD_MSG_FATE_PMD);
            return;
        }
    }

    ct_alg_ctl = get_alg_ctl_type(pkt, tp_src, tp_dst, helper);

    if (OVS_LIKELY(conn)) {
        if (OVS_LIKELY(!conn_update_state_alg(ct, pkt, ctx, conn,
                                              nat_action_info,
                                              ct_alg_ctl, now,
                                              &create_new_conn))) {
            create_new_conn = conn_update_state(ct, pkt, ctx, conn, now);
        }
        if (nat_action_info && !create_new_conn) {
            handle_nat(pkt, conn, zone, ctx->reply, ctx->icmp_related);
        }

    } else if (check_orig_tuple(ct, pkt, ctx, now, &conn, nat_action_info)) {
        create_new_conn = conn_update_state(ct, pkt, ctx, conn, now);
    } else {
        if (ctx->icmp_related) {
            /* An icmp related conn should always be found; no new
               connection is created based on an icmp related packet. */
            pkt->md.ct_state = CS_INVALID;
        } else {
            create_new_conn = true;
        }
    }

    alg_exp = NULL;
    struct alg_exp_node alg_exp_entry;

    if (OVS_UNLIKELY(create_new_conn)) {

        ovs_rwlock_rdlock(&ct->resources_lock);
        alg_exp = expectation_lookup(&ct->alg_expectations, &ctx->key,
                                     ct->hash_basis,
                                     alg_src_ip_wc(ct_alg_ctl));
        if (alg_exp) {
            memcpy(&alg_exp_entry, alg_exp, sizeof alg_exp_entry);
            alg_exp = &alg_exp_entry;
        }
        ovs_rwlock_unlock(&ct->resources_lock);

        if (!conn_lookup(ct, &ctx->key, now, NULL, NULL)) {
CTD_MSG_NAT_CANDIDATE_RESPONSE:
            conn = ctd_conn_not_found(ct, pkt, ctx, commit, now,
                                      nat_action_info, helper, alg_exp,
                                      ct_alg_ctl, tp_id);
            if (m->msg_fate == CTD_MSG_FATE_CTD) {
                return;
            }
        }
    }

    if (conn) {
        conn_lock(conn);

        write_ct_md_conn(pkt, zone, conn);
        if (setmark) {
            set_mark(pkt, conn, setmark[0], setmark[1]);
        }
        if (setlabel) {
            set_label(pkt, conn, &setlabel[0], &setlabel[1]);
        }

        conn_unlock(conn);
    } else {
        write_ct_md_alg_exp(pkt, zone, &ctx->key, alg_exp);
    }

    if (alg_exp) {
        handle_alg_ctl(ct, ctx, pkt, ct_alg_ctl, conn, now, !!nat_action_info);
    }

    set_cached_conn(nat_action_info, ctx, conn, pkt);
    ctd_msg_fate_set(m, CTD_MSG_FATE_PMD);
}

/* Sends the packets in '*pkt_batch' through the connection tracker 'ct'.  All
 * the packets must have the same 'dl_type' (IPv4 or IPv6) and should have
 * the l3 and and l4 offset properly set.  Performs fragment reassembly with
 * the help of ipf_preprocess_conntrack().
 *
 * If 'commit' is true, the packets are allowed to create new entries in the
 * connection tables.  'setmark', if not NULL, should point to a two
 * elements array containing a value and a mask to set the connection mark.
 * 'setlabel' behaves similarly for the connection label.*/
int
ctd_conntrack_execute(struct dp_packet *pkt)
{
    const struct nat_action_info_t *nat_action_info;
    const struct ovs_key_ct_labels *setlabel;
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;
    struct dp_packet_batch pkt_batch;
    struct conn_lookup_ctx *ctx;
    const uint32_t *setmark;
    struct conntrack *ct;
    const char *helper;
    struct conn *conn;
    long long now_us;
    long long now_ms;
    ovs_be16 dl_type;
    ovs_be16 tp_src;
    ovs_be16 tp_dst;
    uint16_t zone;
    bool force;

    ct = m->ct;
    dp_packet_batch_init_packet(&pkt_batch, pkt);
    dl_type = e->dl_type;
    zone = e->zone;
    force = e->force;
    setmark = e->setmark;
    setlabel = e->setlabel;
    tp_src = e->tp_src;
    tp_dst = e->tp_dst;
    helper = e->helper;
    ctx = &e->ct_lookup_ctx;
    nat_action_info = e->nat_action_info_ref;
    now_ms = m->timestamp_ms;

    now_us = now_ms * 1000;
    if (m->msg_type == CTD_MSG_EXEC) {
        ipf_preprocess_conntrack(ct->ipf, &pkt_batch, now_ms, dl_type, zone,
                                 ct->hash_basis);
        ctx->conn = NULL;
    }

    conn = pkt->md.conn;

    if (OVS_UNLIKELY(pkt->md.ct_state == CS_INVALID)) {
        write_ct_md_alg_exp(pkt, zone, NULL, NULL);
    } else if (conn && conn->key.zone == zone && !force
               && !get_alg_ctl_type(pkt, tp_src, tp_dst, helper)) {
        process_one_fast(zone, setmark, setlabel, nat_action_info,
                         conn, pkt);
    } else if (OVS_UNLIKELY(!ctx->valid)) {
        pkt->md.ct_state = CS_INVALID;
        write_ct_md_alg_exp(pkt, zone, NULL, NULL);
    } else {
        ctd_process_one(pkt);
    }
    conn = pkt->md.conn ? pkt->md.conn : ctx->conn;
    process_one_ct_offload(ct, pkt, conn, ctx->reply, now_us);

    ipf_postprocess_conntrack(ct->ipf, &pkt_batch, now_ms, dl_type);

    return 0;
}

static void
set_mark(struct dp_packet *pkt, struct conn *conn, uint32_t val, uint32_t mask)
    OVS_REQUIRES(conn->lock)
{
    if (conn->alg_related) {
        pkt->md.ct_mark = conn->mark;
    } else {
        pkt->md.ct_mark = val | (pkt->md.ct_mark & ~(mask));
        conn->mark = pkt->md.ct_mark;
    }
}

static void
set_label(struct dp_packet *pkt, struct conn *conn,
          const struct ovs_key_ct_labels *val,
          const struct ovs_key_ct_labels *mask)
    OVS_REQUIRES(conn->lock)
{
    if (conn->alg_related) {
        pkt->md.ct_label = conn->label;
    } else {
        ovs_u128 v, m;

        memcpy(&v, val, sizeof v);
        memcpy(&m, mask, sizeof m);

        pkt->md.ct_label.u64.lo = v.u64.lo
                              | (pkt->md.ct_label.u64.lo & ~(m.u64.lo));
        pkt->md.ct_label.u64.hi = v.u64.hi
                              | (pkt->md.ct_label.u64.hi & ~(m.u64.hi));
        conn->label = pkt->md.ct_label;
    }
}

/* 'Data' is a pointer to the beginning of the L3 header and 'new_data' is
 * used to store a pointer to the first byte after the L3 header.  'Size' is
 * the size of the packet beyond the data pointer. */
static inline bool
extract_l3_ipv4(struct conn_key *key, const void *data, size_t size,
                const char **new_data, bool validate_checksum)
{
    if (OVS_UNLIKELY(size < IP_HEADER_LEN)) {
        return false;
    }

    const struct ip_header *ip = data;
    size_t ip_len = IP_IHL(ip->ip_ihl_ver) * 4;

    if (OVS_UNLIKELY(ip_len < IP_HEADER_LEN)) {
        return false;
    }

    if (OVS_UNLIKELY(size < ip_len)) {
        return false;
    }

    if (IP_IS_FRAGMENT(ip->ip_frag_off)) {
        return false;
    }

    if (validate_checksum && csum(data, ip_len) != 0) {
        COVERAGE_INC(ctd_l3csum_err);
        return false;
    }

    if (new_data) {
        *new_data = (char *) data + ip_len;
    }

    key->src.addr.ipv4 = get_16aligned_be32(&ip->ip_src);
    key->dst.addr.ipv4 = get_16aligned_be32(&ip->ip_dst);
    key->nw_proto = ip->ip_proto;

    return true;
}

/* 'Data' is a pointer to the beginning of the L3 header and 'new_data' is
 * used to store a pointer to the first byte after the L3 header.  'Size' is
 * the size of the packet beyond the data pointer. */
static inline bool
extract_l3_ipv6(struct conn_key *key, const void *data, size_t size,
                const char **new_data)
{
    const struct ovs_16aligned_ip6_hdr *ip6 = data;

    if (OVS_UNLIKELY(size < sizeof *ip6)) {
        return false;
    }

    data = ip6 + 1;
    size -=  sizeof *ip6;
    uint8_t nw_proto = ip6->ip6_nxt;
    uint8_t nw_frag = 0;

    const struct ovs_16aligned_ip6_frag *frag_hdr;
    if (!parse_ipv6_ext_hdrs(&data, &size, &nw_proto, &nw_frag, &frag_hdr)) {
        return false;
    }

    if (nw_frag) {
        return false;
    }

    if (new_data) {
        *new_data = data;
    }

    memcpy(&key->src.addr.ipv6, &ip6->ip6_src, sizeof key->src.addr);
    memcpy(&key->dst.addr.ipv6, &ip6->ip6_dst, sizeof key->dst.addr);
    key->nw_proto = nw_proto;

    return true;
}

static inline bool
checksum_valid(const struct conn_key *key, const void *data, size_t size,
               const void *l3)
{
    bool valid;

    if (key->dl_type == htons(ETH_TYPE_IP)) {
        uint32_t csum = packet_csum_pseudoheader(l3);
        valid = (csum_finish(csum_continue(csum, data, size)) == 0);
    } else if (key->dl_type == htons(ETH_TYPE_IPV6)) {
        valid = (packet_csum_upperlayer6(l3, data, key->nw_proto, size) == 0);
    } else {
        valid = false;
    }

    if (!valid) {
        COVERAGE_INC(ctd_l4csum_err);
    }

    return valid;
}

static inline bool
check_l4_tcp(const struct conn_key *key, const void *data, size_t size,
             const void *l3, bool validate_checksum)
{
    const struct tcp_header *tcp = data;
    if (size < sizeof *tcp) {
        return false;
    }

    size_t tcp_len = TCP_OFFSET(tcp->tcp_ctl) * 4;
    if (OVS_UNLIKELY(tcp_len < TCP_HEADER_LEN || tcp_len > size)) {
        return false;
    }

    return validate_checksum ? checksum_valid(key, data, size, l3) : true;
}

static inline bool
check_l4_udp(const struct conn_key *key, const void *data, size_t size,
             const void *l3, bool validate_checksum)
{
    const struct udp_header *udp = data;
    if (size < sizeof *udp) {
        return false;
    }

    size_t udp_len = ntohs(udp->udp_len);
    if (OVS_UNLIKELY(udp_len < UDP_HEADER_LEN || udp_len > size)) {
        return false;
    }

    /* Validation must be skipped if checksum is 0 on IPv4 packets */
    return (udp->udp_csum == 0 && key->dl_type == htons(ETH_TYPE_IP))
           || (validate_checksum ? checksum_valid(key, data, size, l3) : true);
}

static inline bool
check_l4_icmp(const void *data, size_t size, bool validate_checksum)
{
    if (validate_checksum && csum(data, size) != 0) {
        COVERAGE_INC(ctd_l4csum_err);
        return false;
    } else {
        return true;
    }
}

static inline bool
check_l4_icmp6(const struct conn_key *key, const void *data, size_t size,
               const void *l3, bool validate_checksum)
{
    return validate_checksum ? checksum_valid(key, data, size, l3) : true;
}

static inline bool
extract_l4_tcp(struct conn_key *key, const void *data, size_t size,
               size_t *chk_len)
{
    if (OVS_UNLIKELY(size < (chk_len ? *chk_len : TCP_HEADER_LEN))) {
        return false;
    }

    const struct tcp_header *tcp = data;
    key->src.port = tcp->tcp_src;
    key->dst.port = tcp->tcp_dst;

    /* Port 0 is invalid */
    return key->src.port && key->dst.port;
}

static inline bool
extract_l4_udp(struct conn_key *key, const void *data, size_t size,
               size_t *chk_len)
{
    if (OVS_UNLIKELY(size < (chk_len ? *chk_len : UDP_HEADER_LEN))) {
        return false;
    }

    const struct udp_header *udp = data;
    key->src.port = udp->udp_src;
    key->dst.port = udp->udp_dst;

    /* Port 0 is invalid */
    return key->src.port && key->dst.port;
}

static inline bool extract_l4(struct conn_key *key, const void *data,
                              size_t size, bool *related, const void *l3,
                              bool validate_checksum, size_t *chk_len);

static uint8_t
reverse_icmp_type(uint8_t type)
{
    switch (type) {
    case ICMP4_ECHO_REQUEST:
        return ICMP4_ECHO_REPLY;
    case ICMP4_ECHO_REPLY:
        return ICMP4_ECHO_REQUEST;

    case ICMP4_TIMESTAMP:
        return ICMP4_TIMESTAMPREPLY;
    case ICMP4_TIMESTAMPREPLY:
        return ICMP4_TIMESTAMP;

    case ICMP4_INFOREQUEST:
        return ICMP4_INFOREPLY;
    case ICMP4_INFOREPLY:
        return ICMP4_INFOREQUEST;
    default:
        OVS_NOT_REACHED();
    }
}

/* If 'related' is not NULL and the function is processing an ICMP
 * error packet, extract the l3 and l4 fields from the nested header
 * instead and set *related to true.  If 'related' is NULL we're
 * already processing a nested header and no such recursion is
 * possible */
static inline int
extract_l4_icmp(struct conn_key *key, const void *data, size_t size,
                bool *related, size_t *chk_len)
{
    if (OVS_UNLIKELY(size < (chk_len ? *chk_len : ICMP_HEADER_LEN))) {
        return false;
    }

    const struct icmp_header *icmp = data;

    switch (icmp->icmp_type) {
    case ICMP4_ECHO_REQUEST:
    case ICMP4_ECHO_REPLY:
    case ICMP4_TIMESTAMP:
    case ICMP4_TIMESTAMPREPLY:
    case ICMP4_INFOREQUEST:
    case ICMP4_INFOREPLY:
        if (icmp->icmp_code != 0) {
            return false;
        }
        /* Separate ICMP connection: identified using id */
        key->src.icmp_id = key->dst.icmp_id = icmp->icmp_fields.echo.id;
        key->src.icmp_type = icmp->icmp_type;
        key->dst.icmp_type = reverse_icmp_type(icmp->icmp_type);
        break;
    case ICMP4_DST_UNREACH:
    case ICMP4_TIME_EXCEEDED:
    case ICMP4_PARAM_PROB:
    case ICMP4_SOURCEQUENCH:
    case ICMP4_REDIRECT: {
        /* ICMP packet part of another connection. We should
         * extract the key from embedded packet header */
        struct conn_key inner_key;
        const char *l3 = (const char *) (icmp + 1);
        const char *tail = (const char *) data + size;
        const char *l4;

        if (!related) {
            return false;
        }

        memset(&inner_key, 0, sizeof inner_key);
        inner_key.dl_type = htons(ETH_TYPE_IP);
        bool ok = extract_l3_ipv4(&inner_key, l3, tail - l3, &l4, false);
        if (!ok) {
            return false;
        }

        if (inner_key.src.addr.ipv4 != key->dst.addr.ipv4) {
            return false;
        }

        key->src = inner_key.src;
        key->dst = inner_key.dst;
        key->nw_proto = inner_key.nw_proto;
        size_t check_len = ICMP_ERROR_DATA_L4_LEN;

        ok = extract_l4(key, l4, tail - l4, NULL, l3, false, &check_len);
        if (ok) {
            conn_key_reverse(key);
            *related = true;
        }
        return ok;
    }
    default:
        return false;
    }

    return true;
}

static uint8_t
reverse_icmp6_type(uint8_t type)
{
    switch (type) {
    case ICMP6_ECHO_REQUEST:
        return ICMP6_ECHO_REPLY;
    case ICMP6_ECHO_REPLY:
        return ICMP6_ECHO_REQUEST;
    default:
        OVS_NOT_REACHED();
    }
}

/* If 'related' is not NULL and the function is processing an ICMP
 * error packet, extract the l3 and l4 fields from the nested header
 * instead and set *related to true.  If 'related' is NULL we're
 * already processing a nested header and no such recursion is
 * possible */
static inline bool
extract_l4_icmp6(struct conn_key *key, const void *data, size_t size,
                 bool *related)
{
    const struct icmp6_header *icmp6 = data;

    /* All the messages that we support need at least 4 bytes after
     * the header */
    if (size < sizeof *icmp6 + 4) {
        return false;
    }

    switch (icmp6->icmp6_type) {
    case ICMP6_ECHO_REQUEST:
    case ICMP6_ECHO_REPLY:
        if (icmp6->icmp6_code != 0) {
            return false;
        }
        /* Separate ICMP connection: identified using id */
        key->src.icmp_id = key->dst.icmp_id = *(ovs_be16 *) (icmp6 + 1);
        key->src.icmp_type = icmp6->icmp6_type;
        key->dst.icmp_type = reverse_icmp6_type(icmp6->icmp6_type);
        break;
    case ICMP6_DST_UNREACH:
    case ICMP6_PACKET_TOO_BIG:
    case ICMP6_TIME_EXCEEDED:
    case ICMP6_PARAM_PROB: {
        /* ICMP packet part of another connection. We should
         * extract the key from embedded packet header */
        struct conn_key inner_key;
        const char *l3 = (const char *) icmp6 + 8;
        const char *tail = (const char *) data + size;
        const char *l4 = NULL;

        if (!related) {
            return false;
        }

        memset(&inner_key, 0, sizeof inner_key);
        inner_key.dl_type = htons(ETH_TYPE_IPV6);
        bool ok = extract_l3_ipv6(&inner_key, l3, tail - l3, &l4);
        if (!ok) {
            return false;
        }

        /* pf doesn't do this, but it seems a good idea */
        if (!ipv6_addr_equals(&inner_key.src.addr.ipv6,
                              &key->dst.addr.ipv6)) {
            return false;
        }

        key->src = inner_key.src;
        key->dst = inner_key.dst;
        key->nw_proto = inner_key.nw_proto;

        ok = extract_l4(key, l4, tail - l4, NULL, l3, false, NULL);
        if (ok) {
            conn_key_reverse(key);
            *related = true;
        }
        return ok;
    }
    default:
        return false;
    }

    return true;
}

/* Extract l4 fields into 'key', which must already contain valid l3
 * members.
 *
 * If 'related' is not NULL and an ICMP error packet is being
 * processed, the function will extract the key from the packet nested
 * in the ICMP payload and set '*related' to true.
 *
 * 'size' here is the layer 4 size, which can be a nested size if parsing
 * an ICMP or ICMP6 header.
 *
 * If 'related' is NULL, it means that we're already parsing a header nested
 * in an ICMP error.  In this case, we skip the checksum and some length
 * validations. */
static inline bool
extract_l4(struct conn_key *key, const void *data, size_t size, bool *related,
           const void *l3, bool validate_checksum, size_t *chk_len)
{
    if (key->nw_proto == IPPROTO_TCP) {
        return (!related || check_l4_tcp(key, data, size, l3,
                validate_checksum))
               && extract_l4_tcp(key, data, size, chk_len);
    } else if (key->nw_proto == IPPROTO_UDP) {
        return (!related || check_l4_udp(key, data, size, l3,
                validate_checksum))
               && extract_l4_udp(key, data, size, chk_len);
    } else if (key->dl_type == htons(ETH_TYPE_IP)
               && key->nw_proto == IPPROTO_ICMP) {
        return (!related || check_l4_icmp(data, size, validate_checksum))
               && extract_l4_icmp(key, data, size, related, chk_len);
    } else if (key->dl_type == htons(ETH_TYPE_IPV6)
               && key->nw_proto == IPPROTO_ICMPV6) {
        return (!related || check_l4_icmp6(key, data, size, l3,
                validate_checksum))
               && extract_l4_icmp6(key, data, size, related);
    }

    /* For all other protocols we do not have L4 keys, so keep them zero. */
    return true;
}

static uint32_t
ct_addr_hash_add(uint32_t hash, const union ct_addr *addr)
{
    BUILD_ASSERT_DECL(sizeof *addr % 4 == 0);
    return hash_add_bytes32(hash, (const uint32_t *) addr, sizeof *addr);
}

static uint32_t
ct_endpoint_hash_add(uint32_t hash, const struct ct_endpoint *ep)
{
    BUILD_ASSERT_DECL(sizeof *ep % 4 == 0);
    return hash_add_bytes32(hash, (const uint32_t *) ep, sizeof *ep);
}

/* Symmetric */
uint32_t
conn_key_hash(const struct conn_key *key, uint32_t basis)
{
    uint32_t hsrc, hdst, hash;
    hsrc = hdst = basis;
    hsrc = ct_endpoint_hash_add(hsrc, &key->src);
    hdst = ct_endpoint_hash_add(hdst, &key->dst);

    /* Even if source and destination are swapped the hash will be the same. */
    hash = hsrc ^ hdst;

    /* Hash the rest of the key(L3 and L4 types and zone). */
    return hash_words((uint32_t *) (&key->dst + 1),
                      (uint32_t *) (key + 1) - (uint32_t *) (&key->dst + 1),
                      hash);
}

static void
conn_key_reverse(struct conn_key *key)
{
    struct ct_endpoint tmp = key->src;
    key->src = key->dst;
    key->dst = tmp;
}

static uint32_t
nat_ipv6_addrs_delta(const struct in6_addr *ipv6_min,
                     const struct in6_addr *ipv6_max)
{
    const uint8_t *ipv6_min_hi = &ipv6_min->s6_addr[0];
    const uint8_t *ipv6_min_lo = &ipv6_min->s6_addr[0] +  sizeof(uint64_t);
    const uint8_t *ipv6_max_hi = &ipv6_max->s6_addr[0];
    const uint8_t *ipv6_max_lo = &ipv6_max->s6_addr[0] + sizeof(uint64_t);

    ovs_be64 addr6_64_min_hi;
    ovs_be64 addr6_64_min_lo;
    memcpy(&addr6_64_min_hi, ipv6_min_hi, sizeof addr6_64_min_hi);
    memcpy(&addr6_64_min_lo, ipv6_min_lo, sizeof addr6_64_min_lo);

    ovs_be64 addr6_64_max_hi;
    ovs_be64 addr6_64_max_lo;
    memcpy(&addr6_64_max_hi, ipv6_max_hi, sizeof addr6_64_max_hi);
    memcpy(&addr6_64_max_lo, ipv6_max_lo, sizeof addr6_64_max_lo);

    uint64_t diff;

    if (addr6_64_min_hi == addr6_64_max_hi &&
        ntohll(addr6_64_min_lo) <= ntohll(addr6_64_max_lo)) {
        diff = ntohll(addr6_64_max_lo) - ntohll(addr6_64_min_lo);
    } else if (ntohll(addr6_64_min_hi) + 1 == ntohll(addr6_64_max_hi) &&
               ntohll(addr6_64_min_lo) > ntohll(addr6_64_max_lo)) {
        diff = UINT64_MAX - (ntohll(addr6_64_min_lo) -
                             ntohll(addr6_64_max_lo) - 1);
    } else {
        /* Limit address delta supported to 32 bits or 4 billion approximately.
         * Possibly, this should be visible to the user through a datapath
         * support check, however the practical impact is probably nil. */
        diff = 0xfffffffe;
    }

    if (diff > 0xfffffffe) {
        diff = 0xfffffffe;
    }
    return diff;
}

/* This function must be used in tandem with nat_ipv6_addrs_delta(), which
 * restricts the input parameters. */
static void
nat_ipv6_addr_increment(struct in6_addr *ipv6, uint32_t increment)
{
    uint8_t *ipv6_hi = &ipv6->s6_addr[0];
    uint8_t *ipv6_lo = &ipv6->s6_addr[0] + sizeof(ovs_be64);
    ovs_be64 addr6_64_hi;
    ovs_be64 addr6_64_lo;
    memcpy(&addr6_64_hi, ipv6_hi, sizeof addr6_64_hi);
    memcpy(&addr6_64_lo, ipv6_lo, sizeof addr6_64_lo);

    if (UINT64_MAX - increment >= ntohll(addr6_64_lo)) {
        addr6_64_lo = htonll(increment + ntohll(addr6_64_lo));
    } else if (addr6_64_hi != OVS_BE64_MAX) {
        addr6_64_hi = htonll(1 + ntohll(addr6_64_hi));
        addr6_64_lo = htonll(increment - (UINT64_MAX -
                                          ntohll(addr6_64_lo) + 1));
    } else {
        OVS_NOT_REACHED();
    }

    memcpy(ipv6_hi, &addr6_64_hi, sizeof addr6_64_hi);
    memcpy(ipv6_lo, &addr6_64_lo, sizeof addr6_64_lo);
}

static uint32_t
nat_range_hash(const struct conn *conn, uint32_t basis,
               const struct nat_action_info_t *nat_info)
{
    uint32_t hash = basis;

    hash = ct_addr_hash_add(hash, &nat_info->min_addr);
    hash = ct_addr_hash_add(hash, &nat_info->max_addr);
    hash = hash_add(hash,
                    ((uint32_t) nat_info->max_port << 16)
                    | nat_info->min_port);
    hash = ct_endpoint_hash_add(hash, &conn->key.src);
    hash = ct_endpoint_hash_add(hash, &conn->key.dst);
    hash = hash_add(hash, (OVS_FORCE uint32_t) conn->key.dl_type);
    hash = hash_add(hash, conn->key.nw_proto);
    hash = hash_add(hash, conn->key.zone);

    /* The purpose of the second parameter is to distinguish hashes of data of
     * different length; our data always has the same length so there is no
     * value in counting. */
    return hash_finish(hash, 0);
}

/* Ports are stored in host byte order for convenience. */
static void
set_sport_range(const struct nat_action_info_t *ni, const struct conn_key *k,
                uint32_t hash, uint16_t *curr, uint16_t *min,
                uint16_t *max)
{
    if (((ni->nat_action & NAT_ACTION_SNAT_ALL) == NAT_ACTION_SRC) ||
        ((ni->nat_action & NAT_ACTION_DST))) {
        *curr = ntohs(k->src.port);
        if (*curr < 512) {
            *min = 1;
            *max = 511;
        } else if (*curr < 1024) {
            *min = 600;
            *max = 1023;
        } else {
            *min = MIN_NAT_EPHEMERAL_PORT;
            *max = MAX_NAT_EPHEMERAL_PORT;
        }
    } else {
        *min = ni->min_port;
        *max = ni->max_port;
        *curr = *min + (hash % ((*max - *min) + 1));
    }
}

static void
set_dport_range(const struct nat_action_info_t *ni, const struct conn_key *k,
                uint32_t hash, uint16_t *curr, uint16_t *min,
                uint16_t *max)
{
    if (ni->nat_action & NAT_ACTION_DST_PORT) {
        *min = ni->min_port;
        *max = ni->max_port;
        *curr = *min + (hash % ((*max - *min) + 1));
    } else {
        *curr = ntohs(k->dst.port);
        *min = *max = *curr;
    }
}

/* Gets an in range address based on the hash.
 * Addresses are kept in network order. */
static void
get_addr_in_range(union ct_addr *min, union ct_addr *max,
                  union ct_addr *curr, uint32_t hash, bool ipv4)
{
    uint32_t offt, range;

    if (ipv4) {
        range = (ntohl(max->ipv4) - ntohl(min->ipv4)) + 1;
        offt = hash % range;
        curr->ipv4 = htonl(ntohl(min->ipv4) + offt);
    } else {
        range = nat_ipv6_addrs_delta(&min->ipv6, &max->ipv6) + 1;
        /* Range must be within 32 bits for full hash coverage. A 64 or
         * 128 bit hash is unnecessary and hence not used here. Most code
         * is kept common with V4; nat_ipv6_addrs_delta() will do the
         * enforcement via max_ct_addr. */
        offt = hash % range;
        curr->ipv6 = min->ipv6;
        nat_ipv6_addr_increment(&curr->ipv6, offt);
    }
}

static void
find_addr(const struct conn *conn, union ct_addr *min,
          union ct_addr *max, union ct_addr *curr,
          uint32_t hash, bool ipv4,
          const struct nat_action_info_t *nat_info)
{
    const union ct_addr zero_ip = {0};

    /* All-zero case. */
    if (!memcmp(min, &zero_ip, sizeof *min)) {
        if (nat_info->nat_action & NAT_ACTION_SRC) {
            *curr = conn->key.src.addr;
        } else if (nat_info->nat_action & NAT_ACTION_DST) {
            *curr = conn->key.dst.addr;
        }
    } else {
        get_addr_in_range(min, max, curr, hash, ipv4);
    }
}

static void
store_addr_to_key(union ct_addr *addr, struct conn_key *key,
                  uint16_t action)
{
    if (action & NAT_ACTION_SRC) {
        key->dst.addr = *addr;
    } else {
        key->src.addr = *addr;
    }
}

static bool
ctd_nat_next_candidate_l4(struct nat_lookup_info *nli,
                          struct ctd_msg *m,
                          bool is_snat)
{
    static const unsigned int max_attempts = 128;
    uint16_t *curr, min, max;
    uint16_t range;

    if (is_snat) {
        min = nli->sport.min;
        max = nli->sport.max;
        curr = &nli->sport.curr;
    } else {
        min = nli->dport.min;
        max = nli->dport.max;
        curr = &nli->dport.curr;
    }

    range = max - min + 1;

    /* Set the next candidate. In case the candidate search is over, set
     * the type to CTD_MSG_NAT_CANDIDATE_RESPONSE.
     * In case it is not, it is left as CTD_TYPE_NAT_CANDIDATE. The "fate"
     * will be "DELEGATED" in both cases.
     */
    if (!nli->port) {
        nli->port = is_snat ? &nli->rev_key.dst.port : &nli->rev_key.src.port;
        nli->attempts = range;
        if (nli->attempts > max_attempts) {
            nli->attempts = max_attempts;
        }
        if (nli->attempts > N_PORT_ATTEMPTS(*curr, min, max)) {
            nli->attempts = N_PORT_ATTEMPTS(*curr, min, max);
        }
        nli->port_iter = 1;
        NEXT_PORT_IN_RANGE(*curr, min, max);
        *nli->port = htons(*curr);
        return false;
    }

    *nli->port = htons(*curr);

    if (nli->port_iter++ < nli->attempts) {
        NEXT_PORT_IN_RANGE(*curr, min, max);
        return false;
    }

    if (nli->attempts < range && nli->attempts >= 16) {
        nli->port_iter = 0;
        nli->attempts /= 2;
        *curr = min + (random_uint32() % range);
        return false;
    }

    ctd_msg_type_set(m, CTD_MSG_NAT_CANDIDATE_RESPONSE);
    return false;
}

/* This function tries to get a unique tuple.
 * Every iteration checks that the reverse tuple doesn't
 * collide with any existing one.
 *
 * In case of SNAT:
 *    - Pick a src IP address in the range.
 *        - Try to find a source port in range (if any).
 *        - If no port range exists, use the whole
 *          ephemeral range (after testing the port
 *          used by the sender), otherwise use the
 *          specified range.
 *
 * In case of DNAT:
 *    - Pick a dst IP address in the range.
 *        - For each dport in range (if any) tries to find
 *          an unique tuple.
 *        - Eventually, if the previous attempt fails,
 *          tries to find a source port in the ephemeral
 *          range (after testing the port used by the sender).
 *
 * If none can be found, return exhaustion to the caller. */
static bool
ctd_nat_get_candidate_tuple(struct conntrack *ct, const struct conn *conn,
                            const struct nat_action_info_t *nat_info,
                            struct ctd_msg *m,
                            struct nat_lookup_info *nli)
{
    if (!IS_PAT_PROTO(conn->key.nw_proto)) {
        ctd_msg_type_set(m, CTD_MSG_NAT_CANDIDATE_RESPONSE);
        if (!conn_lookup(ct, &nli->rev_key, time_msec(), NULL, NULL)) {
            return true;
        }

        return false;
    }

    if (!conn_lookup(ct, &nli->rev_key, time_msec(), NULL, NULL)) {
        ctd_msg_type_set(m, CTD_MSG_NAT_CANDIDATE_RESPONSE);
        return true;
    }

    bool found = false;
    if (nat_info->nat_action & NAT_ACTION_DST_PORT) {
        found = ctd_nat_next_candidate_l4(nli, m, false);
    }

    if (!found) {
        found = ctd_nat_next_candidate_l4(nli, m, true);
    }

    if (found) {
        return true;
    }

    nli->hash = conn_key_hash(&nli->rev_key, ct->hash_basis);
    return false;
}

static enum ct_update_res
conn_update(struct conntrack *ct, struct conn *conn, struct dp_packet *pkt,
            struct conn_lookup_ctx *ctx, long long now)
{
    conn_lock(conn);
    enum ct_update_res update_res =
        l4_protos[conn->key.nw_proto]->conn_update(ct, conn, pkt, ctx->reply,
                                                   now);
    conn_unlock(conn);
    return update_res;
}

long long int
conn_expiration(const struct conn *conn)
{
    long long int hw_expiration;
    long long int expiration;

    atomic_read_relaxed(&conn->expiration, &expiration);
    atomic_read_relaxed(&conn->hw_expiration, &hw_expiration);

    return MAX(expiration, hw_expiration);
}

static bool
conn_expired(struct conn *conn, long long now)
{
    if (conn->conn_type == CT_CONN_TYPE_DEFAULT) {
        return now >= conn_expiration(conn);
    }
    return false;
}

static bool
valid_new(struct dp_packet *pkt, struct conn_key *key)
{
    return l4_protos[key->nw_proto]->valid_new(pkt);
}

static struct conn *
new_conn(struct conntrack *ct, struct dp_packet *pkt, struct conn_key *key,
         long long now, uint32_t tp_id)
{
    return l4_protos[key->nw_proto]->new_conn(ct, pkt, now, tp_id);
}

static void
delete_conn_cmn(struct conn *conn)
{
    free(conn->alg);
    free(conn);
}

static void
delete_conn(struct conn *conn)
{
    ovs_assert(conn->conn_type == CT_CONN_TYPE_DEFAULT);
    conn_lock_destroy(conn);
    free(conn->nat_conn);
    delete_conn_cmn(conn);
}

/* Convert an IP address 'a' into a conntrack address 'b' based on 'dl_type'.
 *
 * Note that 'dl_type' should be either "ETH_TYPE_IP" or "ETH_TYPE_IPv6"
 * in network-byte order. */
static void
ct_dpif_inet_addr_to_ct_endpoint(const union ct_dpif_inet_addr *a,
                                 union ct_addr *b, ovs_be16 dl_type)
{
    if (dl_type == htons(ETH_TYPE_IP)) {
        b->ipv4 = a->ip;
    } else if (dl_type == htons(ETH_TYPE_IPV6)){
        b->ipv6 = a->in6;
    }
}

static void
tuple_to_conn_key(const struct ct_dpif_tuple *tuple, uint16_t zone,
                  struct conn_key *key)
{
    if (tuple->l3_type == AF_INET) {
        key->dl_type = htons(ETH_TYPE_IP);
    } else if (tuple->l3_type == AF_INET6) {
        key->dl_type = htons(ETH_TYPE_IPV6);
    }
    key->nw_proto = tuple->ip_proto;
    ct_dpif_inet_addr_to_ct_endpoint(&tuple->src, &key->src.addr,
                                     key->dl_type);
    ct_dpif_inet_addr_to_ct_endpoint(&tuple->dst, &key->dst.addr,
                                     key->dl_type);

    if (tuple->ip_proto == IPPROTO_ICMP || tuple->ip_proto == IPPROTO_ICMPV6) {
        key->src.icmp_id = tuple->icmp_id;
        key->src.icmp_type = tuple->icmp_type;
        key->src.icmp_code = tuple->icmp_code;
        key->dst.icmp_id = tuple->icmp_id;
        key->dst.icmp_type = reverse_icmp_type(tuple->icmp_type);
        key->dst.icmp_code = tuple->icmp_code;
    } else {
        key->src.port = tuple->src_port;
        key->dst.port = tuple->dst_port;
    }
    key->zone = zone;
}

int
ctd_flush(struct conntrack *ct, const uint16_t *zone)
{
    struct conn *conn;
    uint32_t hash;

    if (ct->n_threads == 0) {
        return conntrack_flush(ct, zone);
    }

    CMAP_FOR_EACH (conn, cm_node, &ct->conns) {
        if ((!zone || *zone == conn->key.zone) &&
            conn->conn_type == CT_CONN_TYPE_DEFAULT) {
            /* Pass NAT conn, they will be cleaned when
             * their master conn is removed.
             */
            hash = conn_key_hash(&conn->key, ct->hash_basis);
            ctd_msg_conn_clean_send(ct, conn, hash);
        }
    }

    return 0;
}

int
ctd_flush_tuple(struct conntrack *ct,
                const struct ct_dpif_tuple *tuple,
                uint16_t zone)
{
    int error = 0;
    struct conn_key key;
    struct conn *conn;
    uint32_t hash;

    if (ct->n_threads == 0) {
        return conntrack_flush_tuple(ct, tuple, zone);
    }

    memset(&key, 0, sizeof(key));
    tuple_to_conn_key(tuple, zone, &key);

    conn_lookup(ct, &key, time_msec(), &conn, NULL);
    if (conn && conn->conn_type == CT_CONN_TYPE_DEFAULT) {
        hash = conn_key_hash(&conn->key, ct->hash_basis);
        ctd_msg_conn_clean_send(ct, conn, hash);
    } else {
        VLOG_WARN("Must flush tuple using the original pre-NATed tuple");
        error = ENOENT;
    }

    return error;
}

/* This function must be called with the ct->resources read lock taken. */
static struct alg_exp_node *
expectation_lookup(struct hmap *alg_expectations, const struct conn_key *key,
                   uint32_t basis, bool src_ip_wc)
{
    struct conn_key check_key;
    memcpy(&check_key, key, sizeof check_key);
    check_key.src.port = ALG_WC_SRC_PORT;

    if (src_ip_wc) {
        memset(&check_key.src.addr, 0, sizeof check_key.src.addr);
    }

    struct alg_exp_node *alg_exp_node;

    HMAP_FOR_EACH_WITH_HASH (alg_exp_node, node,
                             conn_key_hash(&check_key, basis),
                             alg_expectations) {
        if (!conn_key_cmp(&alg_exp_node->key, &check_key)) {
            return alg_exp_node;
        }
    }
    return NULL;
}

/* This function must be called with the ct->resources write lock taken. */
static void
expectation_remove(struct hmap *alg_expectations,
                   const struct conn_key *key, uint32_t basis)
{
    struct alg_exp_node *alg_exp_node;

    HMAP_FOR_EACH_WITH_HASH (alg_exp_node, node, conn_key_hash(key, basis),
                             alg_expectations) {
        if (!conn_key_cmp(&alg_exp_node->key, key)) {
            hmap_remove(alg_expectations, &alg_exp_node->node);
            break;
        }
    }
}

/* This function must be called with the ct->resources read lock taken. */
static struct alg_exp_node *
expectation_ref_lookup_unique(const struct hindex *alg_expectation_refs,
                              const struct conn_key *parent_key,
                              const struct conn_key *alg_exp_key,
                              uint32_t basis)
{
    struct alg_exp_node *alg_exp_node;

    HINDEX_FOR_EACH_WITH_HASH (alg_exp_node, node_ref,
                               conn_key_hash(parent_key, basis),
                               alg_expectation_refs) {
        if (!conn_key_cmp(&alg_exp_node->parent_key, parent_key) &&
            !conn_key_cmp(&alg_exp_node->key, alg_exp_key)) {
            return alg_exp_node;
        }
    }
    return NULL;
}

/* This function must be called with the ct->resources write lock taken. */
static void
expectation_ref_create(struct hindex *alg_expectation_refs,
                       struct alg_exp_node *alg_exp_node,
                       uint32_t basis)
{
    if (!expectation_ref_lookup_unique(alg_expectation_refs,
                                       &alg_exp_node->parent_key,
                                       &alg_exp_node->key, basis)) {
        hindex_insert(alg_expectation_refs, &alg_exp_node->node_ref,
                      conn_key_hash(&alg_exp_node->parent_key, basis));
    }
}

static void
expectation_clean(struct conntrack *ct, const struct conn_key *parent_key)
{
    ovs_rwlock_wrlock(&ct->resources_lock);

    struct alg_exp_node *node;
    HINDEX_FOR_EACH_WITH_HASH_SAFE (node, node_ref,
                                    conn_key_hash(parent_key, ct->hash_basis),
                                    &ct->alg_expectation_refs) {
        if (!conn_key_cmp(&node->parent_key, parent_key)) {
            expectation_remove(&ct->alg_expectations, &node->key,
                               ct->hash_basis);
            hindex_remove(&ct->alg_expectation_refs, &node->node_ref);
            free(node);
        }
    }

    ovs_rwlock_unlock(&ct->resources_lock);
}

static void
expectation_create(struct conntrack *ct, ovs_be16 dst_port,
                   const struct conn *parent_conn, bool reply, bool src_ip_wc,
                   bool skip_nat)
{
    union ct_addr src_addr;
    union ct_addr dst_addr;
    union ct_addr alg_nat_repl_addr;
    struct alg_exp_node *alg_exp_node = xzalloc(sizeof *alg_exp_node);

    if (reply) {
        src_addr = parent_conn->key.src.addr;
        dst_addr = parent_conn->key.dst.addr;
        alg_exp_node->nat_rpl_dst = true;
        if (skip_nat) {
            alg_nat_repl_addr = dst_addr;
        } else if (parent_conn->nat_action & NAT_ACTION_DST) {
            alg_nat_repl_addr = parent_conn->rev_key.src.addr;
            alg_exp_node->nat_rpl_dst = false;
        } else {
            alg_nat_repl_addr = parent_conn->rev_key.dst.addr;
        }
    } else {
        src_addr = parent_conn->rev_key.src.addr;
        dst_addr = parent_conn->rev_key.dst.addr;
        alg_exp_node->nat_rpl_dst = false;
        if (skip_nat) {
            alg_nat_repl_addr = src_addr;
        } else if (parent_conn->nat_action & NAT_ACTION_DST) {
            alg_nat_repl_addr = parent_conn->key.dst.addr;
            alg_exp_node->nat_rpl_dst = true;
        } else {
            alg_nat_repl_addr = parent_conn->key.src.addr;
        }
    }
    if (src_ip_wc) {
        memset(&src_addr, 0, sizeof src_addr);
    }

    alg_exp_node->key.dl_type = parent_conn->key.dl_type;
    alg_exp_node->key.nw_proto = parent_conn->key.nw_proto;
    alg_exp_node->key.zone = parent_conn->key.zone;
    alg_exp_node->key.src.addr = src_addr;
    alg_exp_node->key.dst.addr = dst_addr;
    alg_exp_node->key.src.port = ALG_WC_SRC_PORT;
    alg_exp_node->key.dst.port = dst_port;
    alg_exp_node->parent_mark = parent_conn->mark;
    alg_exp_node->parent_label = parent_conn->label;
    memcpy(&alg_exp_node->parent_key, &parent_conn->key,
           sizeof alg_exp_node->parent_key);
    /* Take the write lock here because it is almost 100%
     * likely that the lookup will fail and
     * expectation_create() will be called below. */
    ovs_rwlock_wrlock(&ct->resources_lock);
    struct alg_exp_node *alg_exp = expectation_lookup(
        &ct->alg_expectations, &alg_exp_node->key, ct->hash_basis, src_ip_wc);
    if (alg_exp) {
        free(alg_exp_node);
        ovs_rwlock_unlock(&ct->resources_lock);
        return;
    }

    alg_exp_node->alg_nat_repl_addr = alg_nat_repl_addr;
    hmap_insert(&ct->alg_expectations, &alg_exp_node->node,
                conn_key_hash(&alg_exp_node->key, ct->hash_basis));
    expectation_ref_create(&ct->alg_expectation_refs, alg_exp_node,
                           ct->hash_basis);
    ovs_rwlock_unlock(&ct->resources_lock);
}

static void
replace_substring(char *substr, uint8_t substr_size,
                  uint8_t total_size, char *rep_str,
                  uint8_t rep_str_size)
{
    memmove(substr + rep_str_size, substr + substr_size,
            total_size - substr_size);
    memcpy(substr, rep_str, rep_str_size);
}

static void
repl_bytes(char *str, char c1, char c2)
{
    while (*str) {
        if (*str == c1) {
            *str = c2;
        }
        str++;
    }
}

static void
modify_packet(struct dp_packet *pkt, char *pkt_str, size_t size,
              char *repl_str, size_t repl_size,
              uint32_t orig_used_size)
{
    replace_substring(pkt_str, size,
                      (const char *) dp_packet_tail(pkt) - pkt_str,
                      repl_str, repl_size);
    dp_packet_set_size(pkt, orig_used_size + (int) repl_size - (int) size);
}

/* Replace IPV4 address in FTP message with NATed address. */
static int
repl_ftp_v4_addr(struct dp_packet *pkt, ovs_be32 v4_addr_rep,
                 char *ftp_data_start,
                 size_t addr_offset_from_ftp_data_start,
                 size_t addr_size OVS_UNUSED)
{
    enum { MAX_FTP_V4_NAT_DELTA = 8 };

    /* Do conservative check for pathological MTU usage. */
    uint32_t orig_used_size = dp_packet_size(pkt);
    if (orig_used_size + MAX_FTP_V4_NAT_DELTA >
        dp_packet_get_allocated(pkt)) {

        static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(5, 5);
        VLOG_WARN_RL(&rl, "Unsupported effective MTU %u used with FTP V4",
                     dp_packet_get_allocated(pkt));
        return 0;
    }

    char v4_addr_str[INET_ADDRSTRLEN] = {0};
    ovs_assert(inet_ntop(AF_INET, &v4_addr_rep, v4_addr_str,
                         sizeof v4_addr_str));
    repl_bytes(v4_addr_str, '.', ',');
    modify_packet(pkt, ftp_data_start + addr_offset_from_ftp_data_start,
                  addr_size, v4_addr_str, strlen(v4_addr_str),
                  orig_used_size);
    return (int) strlen(v4_addr_str) - (int) addr_size;
}

static char *
skip_non_digits(char *str)
{
    while (!isdigit(*str) && *str != 0) {
        str++;
    }
    return str;
}

static char *
terminate_number_str(char *str, uint8_t max_digits)
{
    uint8_t digits_found = 0;
    while (isdigit(*str) && digits_found <= max_digits) {
        str++;
        digits_found++;
    }

    *str = 0;
    return str;
}


static void
get_ftp_ctl_msg(struct dp_packet *pkt, char *ftp_msg)
{
    struct tcp_header *th = dp_packet_l4(pkt);
    char *tcp_hdr = (char *) th;
    uint32_t tcp_payload_len = dp_packet_get_tcp_payload_length(pkt);
    size_t tcp_payload_of_interest = MIN(tcp_payload_len,
                                         LARGEST_FTP_MSG_OF_INTEREST);
    size_t tcp_hdr_len = TCP_OFFSET(th->tcp_ctl) * 4;

    ovs_strlcpy(ftp_msg, tcp_hdr + tcp_hdr_len,
                tcp_payload_of_interest);
}

static enum ftp_ctl_pkt
detect_ftp_ctl_type(const struct conn_lookup_ctx *ctx,
                    struct dp_packet *pkt)
{
    char ftp_msg[LARGEST_FTP_MSG_OF_INTEREST + 1] = {0};
    get_ftp_ctl_msg(pkt, ftp_msg);

    if (ctx->key.dl_type == htons(ETH_TYPE_IPV6)) {
        if (strncasecmp(ftp_msg, FTP_EPRT_CMD, strlen(FTP_EPRT_CMD)) &&
            !strcasestr(ftp_msg, FTP_EPSV_REPLY)) {
            return CT_FTP_CTL_OTHER;
        }
    } else {
        if (strncasecmp(ftp_msg, FTP_PORT_CMD, strlen(FTP_PORT_CMD)) &&
            strncasecmp(ftp_msg, FTP_PASV_REPLY_CODE,
                        strlen(FTP_PASV_REPLY_CODE))) {
            return CT_FTP_CTL_OTHER;
        }
    }

    return CT_FTP_CTL_INTEREST;
}

static enum ftp_ctl_pkt
process_ftp_ctl_v4(struct conntrack *ct,
                   struct dp_packet *pkt,
                   const struct conn *conn_for_expectation,
                   ovs_be32 *v4_addr_rep,
                   char **ftp_data_v4_start,
                   size_t *addr_offset_from_ftp_data_start,
                   size_t *addr_size)
{
    struct tcp_header *th = dp_packet_l4(pkt);
    size_t tcp_hdr_len = TCP_OFFSET(th->tcp_ctl) * 4;
    char *tcp_hdr = (char *) th;
    *ftp_data_v4_start = tcp_hdr + tcp_hdr_len;
    char ftp_msg[LARGEST_FTP_MSG_OF_INTEREST + 1] = {0};
    get_ftp_ctl_msg(pkt, ftp_msg);
    char *ftp = ftp_msg;
    enum ct_alg_mode mode;

    if (!strncasecmp(ftp, FTP_PORT_CMD, strlen(FTP_PORT_CMD))) {
        ftp = ftp_msg + strlen(FTP_PORT_CMD);
        mode = CT_FTP_MODE_ACTIVE;
    } else {
        ftp = ftp_msg + strlen(FTP_PASV_REPLY_CODE);
        mode = CT_FTP_MODE_PASSIVE;
    }

    /* Find first space. */
    ftp = strchr(ftp, ' ');
    if (!ftp) {
        return CT_FTP_CTL_INVALID;
    }

    /* Find the first digit, after space. */
    ftp = skip_non_digits(ftp);
    if (*ftp == 0) {
        return CT_FTP_CTL_INVALID;
    }

    char *ip_addr_start = ftp;
    *addr_offset_from_ftp_data_start = ip_addr_start - ftp_msg;

    uint8_t comma_count = 0;
    while (comma_count < 4 && *ftp) {
        if (*ftp == ',') {
            comma_count++;
            if (comma_count == 4) {
                *ftp = 0;
            } else {
                *ftp = '.';
            }
        }
        ftp++;
    }
    if (comma_count != 4) {
        return CT_FTP_CTL_INVALID;
    }

    struct in_addr ip_addr;
    int rc2 = inet_pton(AF_INET, ip_addr_start, &ip_addr);
    if (rc2 != 1) {
        return CT_FTP_CTL_INVALID;
    }

    *addr_size = ftp - ip_addr_start - 1;
    char *save_ftp = ftp;
    ftp = terminate_number_str(ftp, MAX_FTP_PORT_DGTS);
    if (!ftp) {
        return CT_FTP_CTL_INVALID;
    }
    int value;
    if (!str_to_int(save_ftp, 10, &value)) {
        return CT_FTP_CTL_INVALID;
    }

    /* This is derived from the L4 port maximum is 65535. */
    if (value > 255) {
        return CT_FTP_CTL_INVALID;
    }

    uint16_t port_hs = value;
    port_hs <<= 8;

    /* Skip over comma. */
    ftp++;
    save_ftp = ftp;
    bool digit_found = false;
    while (isdigit(*ftp)) {
        ftp++;
        digit_found = true;
    }
    if (!digit_found) {
        return CT_FTP_CTL_INVALID;
    }
    *ftp = 0;
    if (!str_to_int(save_ftp, 10, &value)) {
        return CT_FTP_CTL_INVALID;
    }

    if (value > 255) {
        return CT_FTP_CTL_INVALID;
    }

    port_hs |= value;
    ovs_be16 port = htons(port_hs);
    ovs_be32 conn_ipv4_addr;

    switch (mode) {
    case CT_FTP_MODE_ACTIVE:
        *v4_addr_rep = conn_for_expectation->rev_key.dst.addr.ipv4;
        conn_ipv4_addr = conn_for_expectation->key.src.addr.ipv4;
        break;
    case CT_FTP_MODE_PASSIVE:
        *v4_addr_rep = conn_for_expectation->key.dst.addr.ipv4;
        conn_ipv4_addr = conn_for_expectation->rev_key.src.addr.ipv4;
        break;
    case CT_TFTP_MODE:
    default:
        OVS_NOT_REACHED();
    }

    ovs_be32 ftp_ipv4_addr;
    ftp_ipv4_addr = ip_addr.s_addr;
    /* Although most servers will block this exploit, there may be some
     * less well managed. */
    if (ftp_ipv4_addr != conn_ipv4_addr && ftp_ipv4_addr != *v4_addr_rep) {
        return CT_FTP_CTL_INVALID;
    }

    expectation_create(ct, port, conn_for_expectation,
                       !!(pkt->md.ct_state & CS_REPLY_DIR), false, false);
    return CT_FTP_CTL_INTEREST;
}

static char *
skip_ipv6_digits(char *str)
{
    while (isxdigit(*str) || *str == ':' || *str == '.') {
        str++;
    }
    return str;
}

static enum ftp_ctl_pkt
process_ftp_ctl_v6(struct conntrack *ct,
                   struct dp_packet *pkt,
                   const struct conn *conn_for_expectation,
                   union ct_addr *v6_addr_rep, char **ftp_data_start,
                   size_t *addr_offset_from_ftp_data_start,
                   size_t *addr_size, enum ct_alg_mode *mode)
{
    struct tcp_header *th = dp_packet_l4(pkt);
    size_t tcp_hdr_len = TCP_OFFSET(th->tcp_ctl) * 4;
    char *tcp_hdr = (char *) th;
    char ftp_msg[LARGEST_FTP_MSG_OF_INTEREST + 1] = {0};
    get_ftp_ctl_msg(pkt, ftp_msg);
    *ftp_data_start = tcp_hdr + tcp_hdr_len;
    char *ftp = ftp_msg;
    struct in6_addr ip6_addr;

    if (!strncasecmp(ftp, FTP_EPRT_CMD, strlen(FTP_EPRT_CMD))) {
        ftp = ftp_msg + strlen(FTP_EPRT_CMD);
        ftp = skip_non_digits(ftp);
        if (*ftp != FTP_AF_V6 || isdigit(ftp[1])) {
            return CT_FTP_CTL_INVALID;
        }
        /* Jump over delimiter. */
        ftp += 2;

        memset(&ip6_addr, 0, sizeof ip6_addr);
        char *ip_addr_start = ftp;
        *addr_offset_from_ftp_data_start = ip_addr_start - ftp_msg;
        ftp = skip_ipv6_digits(ftp);
        *ftp = 0;
        *addr_size = ftp - ip_addr_start;
        int rc2 = inet_pton(AF_INET6, ip_addr_start, &ip6_addr);
        if (rc2 != 1) {
            return CT_FTP_CTL_INVALID;
        }
        ftp++;
        *mode = CT_FTP_MODE_ACTIVE;
    } else {
        ftp = ftp_msg + strcspn(ftp_msg, "(");
        ftp = skip_non_digits(ftp);
        if (!isdigit(*ftp)) {
            return CT_FTP_CTL_INVALID;
        }

        /* Not used for passive mode. */
        *addr_offset_from_ftp_data_start = 0;
        *addr_size = 0;

        *mode = CT_FTP_MODE_PASSIVE;
    }

    char *save_ftp = ftp;
    ftp = terminate_number_str(ftp, MAX_EXT_FTP_PORT_DGTS);
    if (!ftp) {
        return CT_FTP_CTL_INVALID;
    }

    int value;
    if (!str_to_int(save_ftp, 10, &value)) {
        return CT_FTP_CTL_INVALID;
    }
    if (value > CT_MAX_L4_PORT) {
        return CT_FTP_CTL_INVALID;
    }

    uint16_t port_hs = value;
    ovs_be16 port = htons(port_hs);

    switch (*mode) {
    case CT_FTP_MODE_ACTIVE:
        *v6_addr_rep = conn_for_expectation->rev_key.dst.addr;
        /* Although most servers will block this exploit, there may be some
         * less well managed. */
        if (memcmp(&ip6_addr, &v6_addr_rep->ipv6, sizeof ip6_addr) &&
            memcmp(&ip6_addr, &conn_for_expectation->key.src.addr.ipv6,
                   sizeof ip6_addr)) {
            return CT_FTP_CTL_INVALID;
        }
        break;
    case CT_FTP_MODE_PASSIVE:
        *v6_addr_rep = conn_for_expectation->key.dst.addr;
        break;
    case CT_TFTP_MODE:
    default:
        OVS_NOT_REACHED();
    }

    expectation_create(ct, port, conn_for_expectation,
                       !!(pkt->md.ct_state & CS_REPLY_DIR), false, false);
    return CT_FTP_CTL_INTEREST;
}

static int
repl_ftp_v6_addr(struct dp_packet *pkt, union ct_addr v6_addr_rep,
                 char *ftp_data_start,
                 size_t addr_offset_from_ftp_data_start,
                 size_t addr_size, enum ct_alg_mode mode)
{
    /* This is slightly bigger than really possible. */
    enum { MAX_FTP_V6_NAT_DELTA = 45 };

    if (mode == CT_FTP_MODE_PASSIVE) {
        return 0;
    }

    /* Do conservative check for pathological MTU usage. */
    uint32_t orig_used_size = dp_packet_size(pkt);
    if (orig_used_size + MAX_FTP_V6_NAT_DELTA >
        dp_packet_get_allocated(pkt)) {

        static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(5, 5);
        VLOG_WARN_RL(&rl, "Unsupported effective MTU %u used with FTP V6",
                     dp_packet_get_allocated(pkt));
        return 0;
    }

    char v6_addr_str[INET6_ADDRSTRLEN] = {0};
    ovs_assert(inet_ntop(AF_INET6, &v6_addr_rep.ipv6, v6_addr_str,
                         sizeof v6_addr_str));
    modify_packet(pkt, ftp_data_start + addr_offset_from_ftp_data_start,
                  addr_size, v6_addr_str, strlen(v6_addr_str),
                  orig_used_size);
    return (int) strlen(v6_addr_str) - (int) addr_size;
}

/* Increment/decrement a TCP sequence number. */
static void
adj_seqnum(ovs_16aligned_be32 *val, int32_t inc)
{
    put_16aligned_be32(val, htonl(ntohl(get_16aligned_be32(val)) + inc));
}

static void
handle_ftp_ctl(struct conntrack *ct, const struct conn_lookup_ctx *ctx,
               struct dp_packet *pkt, struct conn *ec, long long now,
               enum ftp_ctl_pkt ftp_ctl, bool nat)
{
    struct ip_header *l3_hdr = dp_packet_l3(pkt);
    ovs_be32 v4_addr_rep = 0;
    union ct_addr v6_addr_rep;
    size_t addr_offset_from_ftp_data_start = 0;
    size_t addr_size = 0;
    char *ftp_data_start;
    enum ct_alg_mode mode = CT_FTP_MODE_ACTIVE;

    if (detect_ftp_ctl_type(ctx, pkt) != ftp_ctl) {
        return;
    }

    struct ovs_16aligned_ip6_hdr *nh6 = dp_packet_l3(pkt);
    int64_t seq_skew = 0;

    if (ftp_ctl == CT_FTP_CTL_INTEREST) {
        enum ftp_ctl_pkt rc;
        if (ctx->key.dl_type == htons(ETH_TYPE_IPV6)) {
            rc = process_ftp_ctl_v6(ct, pkt, ec,
                                    &v6_addr_rep, &ftp_data_start,
                                    &addr_offset_from_ftp_data_start,
                                    &addr_size, &mode);
        } else {
            rc = process_ftp_ctl_v4(ct, pkt, ec,
                                    &v4_addr_rep, &ftp_data_start,
                                    &addr_offset_from_ftp_data_start,
                                    &addr_size);
        }
        if (rc == CT_FTP_CTL_INVALID) {
            static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(5, 5);
            VLOG_WARN_RL(&rl, "Invalid FTP control packet format");
            pkt->md.ct_state |= CS_TRACKED | CS_INVALID;
            return;
        } else if (rc == CT_FTP_CTL_INTEREST) {
            uint16_t ip_len;

            if (ctx->key.dl_type == htons(ETH_TYPE_IPV6)) {
                if (nat) {
                    seq_skew = repl_ftp_v6_addr(pkt, v6_addr_rep,
                                   ftp_data_start,
                                   addr_offset_from_ftp_data_start,
                                   addr_size, mode);
                }

                if (seq_skew) {
                    ip_len = ntohs(nh6->ip6_ctlun.ip6_un1.ip6_un1_plen) +
                        seq_skew;
                    nh6->ip6_ctlun.ip6_un1.ip6_un1_plen = htons(ip_len);
                }
            } else {
                if (nat) {
                    seq_skew = repl_ftp_v4_addr(pkt, v4_addr_rep,
                                   ftp_data_start,
                                   addr_offset_from_ftp_data_start,
                                   addr_size);
                }
                if (seq_skew) {
                    ip_len = ntohs(l3_hdr->ip_tot_len) + seq_skew;
                    if (!dp_packet_hwol_is_ipv4(pkt)) {
                        l3_hdr->ip_csum = recalc_csum16(l3_hdr->ip_csum,
                                                        l3_hdr->ip_tot_len,
                                                        htons(ip_len));
                    }
                    l3_hdr->ip_tot_len = htons(ip_len);
                }
            }
        } else {
            OVS_NOT_REACHED();
        }
    }

    struct tcp_header *th = dp_packet_l4(pkt);

    if (nat && ec->seq_skew != 0) {
        ctx->reply != ec->seq_skew_dir ?
            adj_seqnum(&th->tcp_ack, -ec->seq_skew) :
            adj_seqnum(&th->tcp_seq, ec->seq_skew);
    }

    th->tcp_csum = 0;
    if (!dp_packet_hwol_tx_l4_checksum(pkt)) {
        if (ctx->key.dl_type == htons(ETH_TYPE_IPV6)) {
            th->tcp_csum = packet_csum_upperlayer6(nh6, th, ctx->key.nw_proto,
                               dp_packet_l4_size(pkt));
        } else {
            uint32_t tcp_csum = packet_csum_pseudoheader(l3_hdr);
            th->tcp_csum = csum_finish(
                 csum_continue(tcp_csum, th, dp_packet_l4_size(pkt)));
        }
    }

    if (seq_skew) {
        conn_seq_skew_set(ct, ec, now, seq_skew + ec->seq_skew,
                          ctx->reply);
    }
}

static void
handle_tftp_ctl(struct conntrack *ct,
                const struct conn_lookup_ctx *ctx OVS_UNUSED,
                struct dp_packet *pkt, struct conn *conn_for_expectation,
                long long now OVS_UNUSED, enum ftp_ctl_pkt ftp_ctl OVS_UNUSED,
                bool nat OVS_UNUSED)
{
    expectation_create(ct, conn_for_expectation->key.src.port,
                       conn_for_expectation,
                       !!(pkt->md.ct_state & CS_REPLY_DIR), false, false);
}

static void
ctd_nat_rev_key_init(struct dp_packet *pkt, const struct conn *conn)
{
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;
    struct nat_action_info_t *nai;
    struct nat_lookup_info *nli;
    union ct_addr addr = {0};
    uint32_t hash;

    nli = &e->nli;
    nai = e->nat_action_info_ref;

    hash = nat_range_hash(conn, m->ct->hash_basis, nai);
    find_addr(conn, &nai->min_addr, &nai->max_addr, &addr, hash,
              (conn->key.dl_type == htons(ETH_TYPE_IP)), nai);

    set_sport_range(nai, &conn->key, hash, &nli->sport.curr, &nli->sport.min,
                    &nli->sport.max);
    set_dport_range(nai, &conn->key, hash, &nli->dport.curr, &nli->dport.min,
                    &nli->dport.max);

    if (IS_PAT_PROTO(conn->key.nw_proto)) {
        nli->rev_key.src.port = htons(nli->dport.curr);
        nli->rev_key.dst.port = htons(nli->sport.curr);
    }

    store_addr_to_key(&addr, &nli->rev_key, nai->nat_action);
}

void
ctd_nat_candidate(struct dp_packet *pkt)
{
    const struct nat_action_info_t *nat_action_info;
    struct ctd_msg *m = &pkt->cme.hdr;
    struct ctd_msg_exec *e = &pkt->cme;
    struct nat_lookup_info *nli;
    struct conn *nat_conn = NULL;
    struct conn_lookup_ctx *ctx;
    struct conntrack *ct;
    struct conn *nc;
    bool nat_res;

    ct = m->ct;
    nat_action_info = e->nat_action_info_ref;
    ctx = &e->ct_lookup_ctx;
    nli = &e->nli;

    nc = ctx->conn;

    nat_res = ctd_nat_get_candidate_tuple(ct, nc, nat_action_info, m, nli);
    if (m->msg_type != CTD_MSG_NAT_CANDIDATE_RESPONSE) {
        /* In case the tuple is not valid, but still the search is not
         * exhausted, try a new tuple. Send it to the relevant thread.
         */
        ctd_msg_dest_set(m, nli->hash);
        ctd_msg_fate_set(m, CTD_MSG_FATE_CTD);
        return;
    }

    /* Search is exhausted. Send it back with a NULL nat-conn as a response. */
    if (!nat_res) {
        /* This would be a user error or a DOS attack.  A user error is prevented
         * by allocating enough combinations of NAT addresses when combined with
         * ephemeral ports.  A DOS attack should be protected against with
         * firewall rules or a separate firewall.  Also using zone partitioning
         * can limit DoS impact. */

        static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(5, 5);
        VLOG_WARN_RL(&rl, "Unable to NAT due to tuple space exhaustion - "
                     "if DoS attack, use firewalling and/or zone partitioning.");
        ctd_msg_fate_set(m, CTD_MSG_FATE_PMD);
        return;
    }

    nat_conn = xzalloc(sizeof *nat_conn);
    memcpy(nat_conn, nc, sizeof *nat_conn);
    /* Update conn with nat adjustments. */
    memcpy(&nc->rev_key, &nli->rev_key, sizeof nc->rev_key);
    ctd_nat_conn_init(pkt, nc, nat_conn);

    nli->nat_conn = nat_conn;
    ctd_msg_dest_set(m, ctx->hash);
    ctd_msg_fate_set(m, CTD_MSG_FATE_CTD);
}

void
ctd_conn_clean(struct ctd_msg_conn_clean *msg)
{
    struct ctd_msg *m = &msg->hdr;
    struct zone_limit *zl;
    struct conntrack *ct;
    struct conn *conn;
    uint32_t hash;

    ct = m->ct;
    conn = msg->conn;

    /* If the connection is pending for a nat response, pend its cleaning. */
    if (conn->reordering) {
        ctd_msg_fate_set(&msg->hdr, CTD_MSG_FATE_SELF);
        return;
    }

    if (conn->conn_type == CT_CONN_TYPE_UN_NAT) {
        conn_lock(conn);
        conntrack_lock(ct);

        cmap_remove(&ct->conns, &conn->cm_node, m->dest_hash);

        conntrack_unlock(ct);
        conn_unlock(conn);
        ctd_msg_fate_set(m, CTD_MSG_FATE_FREE);
        return;
    }

    if (atomic_flag_test_and_set(&conn->reclaimed)) {
        ctd_msg_fate_set(m, CTD_MSG_FATE_FREE);
        return;
    }

    conn_lock(conn);
    conntrack_lock(ct);

    if (conn->alg) {
        expectation_clean(ct, &conn->key);
    }

    conntrack_offload_del_conn(ct, conn, false);

    hash = conn_key_hash(&conn->key, ct->hash_basis);
    cmap_remove(&ct->conns, &conn->cm_node, hash);

    conntrack_unlock(ct);
    conn_unlock(conn);

    zl = zone_limit_lookup(ct, conn->admit_zone);
    if (zl && zl->czl.zone_limit_seq == conn->zone_limit_seq) {
        atomic_count_dec(&zl->czl.count);
    }

    if (conn->nat_conn) {
        hash = conn_key_hash(&conn->nat_conn->key, ct->hash_basis);
        ctd_msg_dest_set(m, hash);
        ctd_msg_fate_set(m, CTD_MSG_FATE_CTD);
    } else {
        ctd_msg_fate_set(m, CTD_MSG_FATE_FREE);
    }
    conn_unref(conn);
    atomic_count_dec(&ct->n_conn);
    atomic_count_dec(&ct->l4_counters[conn->key.nw_proto]);
}

void
ctd_init(struct conntrack *ct, const struct smap *ovs_other_config)
{
    ctd_n_threads = smap_get_ullong(ovs_other_config, "n-ct-threads",
                                DEFAULT_CT_DIST_THREAD_NB);
    if (ctd_n_threads > MAX_CT_DIST_THREAD_NB) {
        VLOG_WARN("Invalid number of threads requested: %u. Limiting to %u",
                  ctd_n_threads, MAX_CT_DIST_THREAD_NB);
        ctd_n_threads = MAX_CT_DIST_THREAD_NB;
    }

    ct->n_threads = ctd_n_threads;
    if (ctd_n_threads) {
        ctd_thread_create(ct);
    }
}

bool
ctd_exec(struct conntrack *conntrack,
         struct dp_netdev_pmd_thread *pmd,
         const struct flow *flow,
         struct dp_packet_batch *packets_,
         const struct nlattr *ct_action,
         struct dp_netdev_flow *dp_flow OVS_UNUSED,
         const struct nlattr *actions OVS_UNUSED,
         size_t actions_len OVS_UNUSED,
         uint32_t depth OVS_UNUSED)
{
    const struct ovs_key_ct_labels *setlabel = NULL;
    struct nat_action_info_t *nat_action_info_ref;
    struct nat_action_info_t nat_action_info = {0};
    struct dp_packet OVS_UNUSED *packet;
    const uint32_t *setmark = NULL;
    const char *helper = NULL;
    bool nat_config = false;
    const struct nlattr *b;
    bool commit = false;
    bool force = false;
    uint32_t tp_id = 0;
    unsigned int left;
    uint16_t zone = 0;

    nat_action_info_ref = NULL;
    NL_ATTR_FOR_EACH_UNSAFE (b, left, nl_attr_get(ct_action),
                             nl_attr_get_size(ct_action)) {
        enum ovs_ct_attr sub_type = nl_attr_type(b);

        switch(sub_type) {
        case OVS_CT_ATTR_FORCE_COMMIT:
            force = true;
            /* fall through. */
        case OVS_CT_ATTR_COMMIT:
            commit = true;
            break;
        case OVS_CT_ATTR_ZONE:
            zone = nl_attr_get_u16(b);
            break;
        case OVS_CT_ATTR_HELPER:
            helper = nl_attr_get_string(b);
            break;
        case OVS_CT_ATTR_MARK:
            setmark = nl_attr_get(b);
            break;
        case OVS_CT_ATTR_LABELS:
            setlabel = nl_attr_get(b);
            break;
        case OVS_CT_ATTR_EVENTMASK:
            /* Silently ignored, as userspace datapath does not generate
             * netlink events. */
            break;
        case OVS_CT_ATTR_TIMEOUT:
            if (!str_to_uint(nl_attr_get_string(b), 10, &tp_id)) {
                VLOG_WARN("Invalid Timeout Policy ID: %s.",
                          nl_attr_get_string(b));
                tp_id = DEFAULT_TP_ID;
            }
            break;
        case OVS_CT_ATTR_NAT: {
            const struct nlattr *b_nest;
            unsigned int left_nest;
            bool ip_min_specified = false;
            bool proto_num_min_specified = false;
            bool ip_max_specified = false;
            bool proto_num_max_specified = false;

            memset(&nat_action_info, 0, sizeof nat_action_info);
            nat_action_info_ref = &nat_action_info;

            NL_NESTED_FOR_EACH_UNSAFE (b_nest, left_nest, b) {
                enum ovs_nat_attr sub_type_nest = nl_attr_type(b_nest);

                switch (sub_type_nest) {
                case OVS_NAT_ATTR_SRC:
                case OVS_NAT_ATTR_DST:
                    nat_config = true;
                    nat_action_info.nat_action |=
                        ((sub_type_nest == OVS_NAT_ATTR_SRC)
                            ? NAT_ACTION_SRC : NAT_ACTION_DST);
                    break;
                case OVS_NAT_ATTR_IP_MIN:
                    memcpy(&nat_action_info.min_addr,
                           nl_attr_get(b_nest),
                           nl_attr_get_size(b_nest));
                    ip_min_specified = true;
                    break;
                case OVS_NAT_ATTR_IP_MAX:
                    memcpy(&nat_action_info.max_addr,
                           nl_attr_get(b_nest),
                           nl_attr_get_size(b_nest));
                    ip_max_specified = true;
                    break;
                case OVS_NAT_ATTR_PROTO_MIN:
                    nat_action_info.min_port =
                        nl_attr_get_u16(b_nest);
                    proto_num_min_specified = true;
                    break;
                case OVS_NAT_ATTR_PROTO_MAX:
                    nat_action_info.max_port =
                        nl_attr_get_u16(b_nest);
                    proto_num_max_specified = true;
                    break;
                case OVS_NAT_ATTR_PERSISTENT:
                case OVS_NAT_ATTR_PROTO_HASH:
                case OVS_NAT_ATTR_PROTO_RANDOM:
                    break;
                case OVS_NAT_ATTR_UNSPEC:
                case __OVS_NAT_ATTR_MAX:
                    OVS_NOT_REACHED();
                }
            }

            if (ip_min_specified && !ip_max_specified) {
                nat_action_info.max_addr = nat_action_info.min_addr;
            }
            if (proto_num_min_specified && !proto_num_max_specified) {
                nat_action_info.max_port = nat_action_info.min_port;
            }
            if (proto_num_min_specified || proto_num_max_specified) {
                if (nat_action_info.nat_action & NAT_ACTION_SRC) {
                    nat_action_info.nat_action |= NAT_ACTION_SRC_PORT;
                } else if (nat_action_info.nat_action & NAT_ACTION_DST) {
                    nat_action_info.nat_action |= NAT_ACTION_DST_PORT;
                }
            }
            break;
        }
        case OVS_CT_ATTR_UNSPEC:
        case __OVS_CT_ATTR_MAX:
            OVS_NOT_REACHED();
        }
    }

    /* We won't be able to function properly in this case, hence
     * complain loudly. */
    if (nat_config && !commit) {
        static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(5, 5);
        VLOG_WARN_RL(&rl, "NAT specified without commit.");
    }

    if (conntrack->n_threads == 0) {
        conntrack_execute(conntrack, packets_, flow->dl_type, force,
                          commit, zone, setmark, setlabel, flow->tp_src,
                          flow->tp_dst, helper, nat_action_info_ref,
                          pmd->ctx.now, tp_id);
        return false;
    }

    /* Each packet is sent separately on a message to the appropriate
     * ct-thread (by its ct-hash).
     * Batching it is TBD.
     */
    DP_PACKET_BATCH_FOR_EACH (i, packet, packets_) {
        struct ctd_msg *m = &packet->cme.hdr;
        struct ctd_msg_exec *e = &packet->cme;

        ctd_msg_type_set(m, CTD_MSG_EXEC);
        ctd_msg_fate_set(m, CTD_MSG_FATE_TBD);
        m->timestamp_ms = pmd->ctx.now / 1000;
        m->ct = conntrack,
        *e = (struct ctd_msg_exec) {
            .dl_type = flow->dl_type,
            .force = force,
            .commit = commit,
            .zone = zone,
            .setmark = setmark,
            .setlabel = setlabel,
            .tp_src = flow->tp_src,
            .tp_dst = flow->tp_dst,
            .helper = helper,
            .nat_action_info = nat_action_info,
            .nat_action_info_ref = NULL,
            .tp_id = tp_id,
            .pmd = pmd,
            .flow = dp_flow,
            .actions_len = actions_len,
            .depth = depth,
        };
        if (nat_action_info_ref) {
            e->nat_action_info_ref = &e->nat_action_info;
        }
        conn_key_extract(conntrack, packet, e->dl_type, &e->ct_lookup_ctx,
                         e->zone);

        if (dp_flow) {
            dp_netdev_flow_ref(dp_flow);
        }
        memcpy(e->actions_buf, actions, actions_len);
        ctd_send_msg_to_thread(m, ctd_h2tid(e->ct_lookup_ctx.hash));
    }

    /* Empty the batch, to stop its processing in this context.
     * It will be completed in the ct2pmd context.
     */
    dp_packet_batch_init(packets_);

    return true;
}
