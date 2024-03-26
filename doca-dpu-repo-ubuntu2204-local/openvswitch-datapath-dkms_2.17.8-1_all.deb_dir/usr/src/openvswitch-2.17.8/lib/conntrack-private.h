/*
 * Copyright (c) 2015-2019 Nicira, Inc.
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

#ifndef CONNTRACK_PRIVATE_H
#define CONNTRACK_PRIVATE_H 1

#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/ip6.h>

#include "cmap.h"
#include "conntrack.h"
#include "conntrack-offload.h"
#include "ct-dpif.h"
#include "ipf.h"
#include "openvswitch/hmap.h"
#include "openvswitch/list.h"
#include "openvswitch/types.h"
#include "ovs-atomic.h"
#include "packets.h"
#include "rculist.h"
#include "unaligned.h"
#include "dp-packet.h"

/* This is used for alg expectations; an expectation is a
 * context created in preparation for establishing a data
 * connection. The expectation is created by the control
 * connection. */
struct alg_exp_node {
    /* Node in alg_expectations. */
    struct hmap_node node;
    /* Node in alg_expectation_refs. */
    struct hindex_node node_ref;
    /* Key of data connection to be created. */
    struct conn_key key;
    /* Corresponding key of the control connection. */
    struct conn_key parent_key;
    /* The NAT replacement address to be used by the data connection. */
    union ct_addr alg_nat_repl_addr;
    /* The data connection inherits the parent control
     * connection label and mark. */
    ovs_u128 parent_label;
    uint32_t parent_mark;
    /* True if for NAT application, the alg replaces the dest address;
     * otherwise, the source address is replaced.  */
    bool nat_rpl_dst;
};

/* Timeouts: all the possible timeout states passed to update_expiration()
 * are listed here. The name will be prefix by CT_TM_ and the value is in
 * milliseconds */
#define CT_TIMEOUTS \
    CT_TIMEOUT(TCP_FIRST_PACKET) \
    CT_TIMEOUT(TCP_OPENING) \
    CT_TIMEOUT(TCP_ESTABLISHED) \
    CT_TIMEOUT(TCP_CLOSING) \
    CT_TIMEOUT(TCP_FIN_WAIT) \
    CT_TIMEOUT(TCP_CLOSED) \
    CT_TIMEOUT(OTHER_FIRST) \
    CT_TIMEOUT(OTHER_MULTIPLE) \
    CT_TIMEOUT(OTHER_BIDIR) \
    CT_TIMEOUT(ICMP_FIRST) \
    CT_TIMEOUT(ICMP_REPLY)

#define NAT_ACTION_SNAT_ALL (NAT_ACTION_SRC | NAT_ACTION_SRC_PORT)
#define NAT_ACTION_DNAT_ALL (NAT_ACTION_DST | NAT_ACTION_DST_PORT)

enum ct_ephemeral_range {
    MIN_NAT_EPHEMERAL_PORT = 1024,
    MAX_NAT_EPHEMERAL_PORT = 65535
};

#define IN_RANGE(curr, min, max) \
    (curr >= min && curr <= max)

#define NEXT_PORT_IN_RANGE(curr, min, max) \
    (curr = (!IN_RANGE(curr, min, max) || curr == max) ? min : curr + 1)

/* If the current port is out of range increase the attempts by
 * one so that in the worst case scenario the current out of
 * range port plus all the in-range ports get tested.
 * Note that curr can be an out of range port only in case of
 * source port (SNAT with port range unspecified or DNAT),
 * furthermore the source port in the packet has to be less than
 * MIN_NAT_EPHEMERAL_PORT. */
#define N_PORT_ATTEMPTS(curr, min, max) \
    ((!IN_RANGE(curr, min, max)) ? (max - min) + 2 : (max - min) + 1)

/* Loose in-range check, the first curr port can be any port out of
 * the range. */
#define FOR_EACH_PORT_IN_RANGE__(curr, min, max, INAME) \
    for (uint16_t INAME = N_PORT_ATTEMPTS(curr, min, max); \
        INAME > 0; INAME--, NEXT_PORT_IN_RANGE(curr, min, max))

#define FOR_EACH_PORT_IN_RANGE(curr, min, max) \
    FOR_EACH_PORT_IN_RANGE__(curr, min, max, OVS_JOIN(idx, __COUNTER__))

enum ct_timeout {
#define CT_TIMEOUT(NAME) CT_TM_##NAME,
    CT_TIMEOUTS
#undef CT_TIMEOUT
    N_CT_TM
};

enum OVS_PACKED_ENUM ct_conn_type {
    CT_CONN_TYPE_DEFAULT,
    CT_CONN_TYPE_UN_NAT,
};

static inline int
ct_get_packet_dir(bool reply)
{
    return reply ? CT_DIR_REP : CT_DIR_INIT;
}

struct conn_expire {
    struct mpsc_queue_node node;
    /* Timeout state of the connection.
     * It follows the connection state updates.
     */
    enum ct_timeout tm;
    struct ovs_refcount refcount;
    atomic_flag reschedule;
};

struct conn {
    /* Immutable data. */
    struct conn_key key;
    struct conn_key rev_key;
    struct conn_key parent_key; /* Only used for orig_tuple support. */
    struct cmap_node cm_node;
    uint16_t nat_action;
    char *alg;
    struct conn *nat_conn; /* The NAT 'conn' context, if there is one. */
    struct conn *master_conn; /* The master 'conn' context if this is a NAT
                               * 'conn' */
    atomic_flag reclaimed; /* False during the lifetime of the connection,
                            * True as soon as a thread has started freeing
                            * its memory. */

    struct ovsrcu_gc_node gc_node;

    /* Inserted once by a PMD, then managed by the 'ct_clean' thread. */
    struct conn_expire exp;

    /* Mutable data. */
    struct ovs_spin lock; /* Guards all mutable fields. */
    ovs_u128 label;
    atomic_llong expiration;
    long long prev_query;
    uint32_t mark;
    int seq_skew;

    /* Immutable data. */
    int32_t admit_zone; /* The zone for managing zone limit counts. */
    uint32_t zone_limit_seq; /* Used to disambiguate zone limit counts. */

    /* Mutable data. */
    bool seq_skew_dir; /* TCP sequence skew direction due to NATTing of FTP
                        * control messages; true if reply direction. */

    /* Immutable data. */
    bool alg_related; /* True if alg data connection. */
    enum ct_conn_type conn_type;

    uint32_t tp_id; /* Timeout policy ID. */
    struct ct_offloads offloads;
    struct {
        atomic_llong hw_expiration;
        bool reordering;
        struct dp_packet *response_pkt;
        struct dp_packet *resume_pkt;
    };
};

#define conn_lock_init(conn) do { \
    ovs_spin_init(&(conn)->lock); \
} while (0)
#define conn_lock_destroy(conn) do { \
    ovs_spin_destroy(&(conn)->lock); \
} while (0)
#define conn_lock(conn) do { \
    ovs_spin_lock(&(conn)->lock); \
} while (0)
#define conn_unlock(conn) do { \
    ovs_spin_unlock(&(conn)->lock); \
} while (0)

enum ct_update_res {
    CT_UPDATE_INVALID,
    CT_UPDATE_VALID,
    CT_UPDATE_NEW,
    CT_UPDATE_VALID_NEW,
};

struct ct_thread;

struct conntrack {
    struct ct_thread *threads;
    unsigned int n_threads;

    struct ovs_spin ct_lock; /* Protects 2 following fields. */
    struct cmap conns OVS_GUARDED;
    struct mpsc_queue exp_lists[N_CT_TM] OVS_GUARDED;
    struct cmap zone_limits OVS_GUARDED;
    struct cmap timeout_policies OVS_GUARDED;

    atomic_count l4_counters[UINT8_MAX + 1];

    uint32_t hash_basis; /* Salt for hashing a connection key. */
    pthread_t clean_thread; /* Periodically cleans up connection tracker. */
    struct latch clean_thread_exit; /* To destroy the 'clean_thread'. */

    /* Counting connections. */
    atomic_count n_conn; /* Number of connections currently tracked. */
    atomic_uint n_conn_limit; /* Max connections tracked. */

    /* Expectations for application level gateways (created by control
     * connections to help create data connections, e.g. for FTP). */
    struct ovs_rwlock resources_lock; /* Protects fields below. */
    struct hmap alg_expectations OVS_GUARDED; /* Holds struct
                                               * alg_exp_nodes. */
    struct hindex alg_expectation_refs OVS_GUARDED; /* For lookup from
                                                     * control context.  */

    struct ipf *ipf; /* Fragmentation handling context. */
    uint32_t zone_limit_seq; /* Used to disambiguate zone limit counts. */
    atomic_bool tcp_seq_chk; /* Check TCP sequence numbers. */
    void *dp; /* DP handler for offloads. */

    /* Holding HW offload callbacks */
    OVSRCU_TYPE(struct conntrack_offload_class *) offload_class;
};

#define conntrack_lock_init(ct) do { \
    ovs_spin_init(&(ct)->ct_lock); \
} while (0)
#define conntrack_lock_destroy(ct) do { \
    ovs_spin_destroy(&(ct)->ct_lock); \
} while (0)
#define conntrack_lock(ct) do { \
    ovs_spin_lock(&(ct)->ct_lock); \
} while (0)
#define conntrack_unlock(ct) do { \
    ovs_spin_unlock(&(ct)->ct_lock); \
} while (0)

/* Lock acquisition order:
 *    1. 'conn->lock'
 *    2. 'ct_lock'
 *    3. 'resources_lock'
 */

extern struct ct_l4_proto ct_proto_tcp;
extern struct ct_l4_proto ct_proto_other;
extern struct ct_l4_proto ct_proto_icmp4;
extern struct ct_l4_proto ct_proto_icmp6;
extern struct ct_l4_proto *l4_protos[UINT8_MAX + 1];

struct ct_l4_proto {
    struct conn *(*new_conn)(struct conntrack *ct, struct dp_packet *pkt,
                             long long now, uint32_t tp_id);
    bool (*valid_new)(struct dp_packet *pkt);
    enum ct_update_res (*conn_update)(struct conntrack *ct, struct conn *conn,
                                      struct dp_packet *pkt, bool reply,
                                      long long now);
    void (*conn_get_protoinfo)(const struct conn *,
                               struct ct_dpif_protoinfo *);
    enum ct_timeout (*get_tm)(struct conn *conn);
};


static inline void
conn_expire_push_back(struct conntrack *ct, struct conn *conn)
{
    if (ovs_refcount_try_ref_rcu(&conn->exp.refcount)) {
        atomic_flag_clear(&conn->exp.reschedule);
        mpsc_queue_insert(&ct->exp_lists[conn->exp.tm], &conn->exp.node);
    }
}

static inline void
conn_expire_push_front(struct conntrack *ct, struct conn *conn)
    OVS_REQUIRES(ct->exp_lists[conn->exp.tm].read_lock)
{
    if (ovs_refcount_try_ref_rcu(&conn->exp.refcount)) {
        /* Do not change 'reschedule' state, if this expire node is put
         * at the tail of the list, it will be re-examined next sweep.
         */
        mpsc_queue_push_front(&ct->exp_lists[conn->exp.tm], &conn->exp.node);
    }
}

#endif /* conntrack-private.h */
