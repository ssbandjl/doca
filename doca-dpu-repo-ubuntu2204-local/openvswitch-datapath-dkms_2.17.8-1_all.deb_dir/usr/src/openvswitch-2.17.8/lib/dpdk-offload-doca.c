/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <doca_flow.h>
#include <rte_flow.h>
#include <sys/types.h>

#include "conntrack-offload.h"
#include "coverage.h"
#include "dp-packet.h"
#include "dpdk-offload-provider.h"
#include "id-fpool.h"
#include "offload-metadata.h"
#include "openvswitch/vlog.h"
#include "openvswitch/list.h"
#include "ovs-doca.h"
#include "netdev-dpdk.h"
#include "netdev-vport.h"
#include "timeval.h"
#include "util.h"

/*
 * DOCA offload implementation for DPDK provider.
 *
 * The CT offload implementation over basic pipes is designed as such:
 *
 * +---------------------------------------------------------------------------------------------+
 * | Control pipes                                                                               |
 * |                                                                                             |
 * |                 ,-[ CT Zone X ]-----.                                                       |
 * |                 |  ,-[ CT Zone Y ]-----.                                                    |
 * |                 |  |  ,-[ CT Zone Z ]-----.                                                 |
 * |                 |  |  |                   |                                                 |
 * |                 |  |  |  ,---------.      |                                                 |
 * |                 |  |  |  |ct_zone=Z+------+--------------------------------------.          |
 * |                 |  |  |  `---------'hit   |                                      |          |
 * |                 |  |  |                   | +----------------------------+       |          |
 * |                 |  |  |                   | | Basic pipes                |       |          |
 * |                 |  |  |                   | |                            |       |          |
 * |                 |  |  |                   | |  ,-[ CT IPv4 x UDP ]-.     |       |          |
 * |                 |  |  |    ,----------.   | |  |                   |     |       |          |
 * |                 |  |  |    |IPv4 + UDP+---+--->|  ,-[ CT IPv4 x TCP ]-.  |       v          |
 * |  ,-[ Pre-CT ]-. |  |  |    `----------'hit| |  |  |                   |  | ,-[ Post-CT ]--. |
 * |  |            +------>|    ,----------.   | |  |  | ,-------------.   |--->|              | |
 * |  |            | |  |  |    |IPv4 + TCP+---+------>| | CT entries  +---+--->|              | |
 * |  `------------' |  |  |    `----------'hit| |  |  | `-------------'hit|  | `--------------' |
 * |                 |  |  |       ,---------. | |  |  |   ,---------.     |  |                  |
 * |                 |  |  |       |Catch-all| | |  |  |   |Catch-all|     |  |                  |
 * |                 `--|  |       `----+----' | |  `--|   `----+----'     |  |                  |
 * |                    `--|            |      | |     `--------+----------'  |                  |
 * |                       `------------+------' |           |  |             |                  |
 * |                                    |        +-----------|--|-------------+                  |
 * |                                    v                    |  |                                |
 * |                             ,-[ Miss pipe ]-------------v--v-------.                        |
 * |                             |         Go to software datapath      |                        |
 * |                             `--------------------------------------'                        |
 * +---------------------------------------------------------------------------------------------+
 *
 * This model is replicated once per eswitch.
 *
 * A megaflow that contains a 'ct()' action is split
 * into its 'pre-CT' and 'post-CT' part. The pre-CT is inserted
 * into the eswitch root pipe, and contains the megaflow original
 * match.
 *
 * On match, to execute CT, the packet is sent to the 'CT-zone' pipes,
 * one pipe per CT zone ID. If the ct_zone value is already set on the packet
 * and the value matches that of the current CT-zone pipe, then CT is known
 * to have already been executed. The packet is thus immediately forwarded to
 * post-CT. Post-CT contains the rest of the original megaflow that was not
 * used in pre-CT.
 *
 * If this ct_zone match fails, then either CT was never executed, or
 * it was executed in a different CT zone. If it matches the currently
 * supported CT (network x protocol) tuple, then its ct_zone is set and
 * it is forwarded to the corresponding CT pipe. If no (network x protocol)
 * tuple matches, then CT is not supported for this flow and the packet
 * goes to software.
 *
 * The CT pipe is a basic pipe with a single action type, which writes to
 *
 *  * The packet registers used for CT metadata.
 *  * The packet 5-tuple.
 *
 * For plain CT, the 5-tuple is overwritten with its own values.
 * For NAT, the translations are written instead where relevant.
 *
 * In both cases, all fields are written anyway.
 * This way, the number of template used by the CT pipe is minimal.
 * During performance tests, no impact was measured due to the
 * superfluous writes.
 *
 * If a CT entry matches the packet, the CT pipe action is executed
 * and the packet is then forwarded to post-CT. Otherwise, the packet
 * goes to the miss pipe and is then handed over to the software
 * datapath.
 *
 * IPv6 Connection Tracking Implementation
 * =======================================
 *
 * By default IPv6 connection offloading is disabled.
 * Set 'other_config:hw-offload-ct-ipv6-enabled=true' to enable.
 *
 * The diagram below shows how IPv6 connection tracking is implemented in the
 * hardware datapath. IPv6 CT rules are too large to fit into single steering
 * hardware objects (STE) and must be split.
 *
 * Note:
 * This is HW specific.
 * In BF3 and above, jumbo STE is supported and the rules can match in a
 * single STE. However, as OVS is HW agnostic, and to support < BF3 cards,
 * this split is done.
 *
 * The IPv6 tuple is divided into a prefix and a suffix, each inserted into
 * their respective pipe. A packet has to match both prefix a suffix rules
 * for a complete IPv6 5-tuple match and continue into the common post-CT pipe.
 *
 * This implementation complements and integrates with the IPv4 model.
 * The distinction is made during the CT-zone stage, matching on L3 protocol
 * to steer the packet toward the relevant basic pipe. Beside splitting match
 * into two, the same logic applies as for the IPv4 implementation.
 *
 *               +-----------------------------------------------+
 *               |IPv6 CT basic pipes                            |
 *               |                         +-[CT suffix TCP]-+   |
 *               |                         |                 |   |
 *               |                         +-----------------+   |
 *               |                         |prefix_id +      |   |
 *               |                         |IPv6.dst +       +-+ |
 *               |                         |TCP ports        | | |
 *               | +[CT prefix pipe]+      +-----------------+ | |
 *+[CT Zone X]+  | |                |      |                 | | |
 *|           |  | |                | +--->|                 | | |
 *|           |  | |                | |    |                 | | |
 *|           |  | |+-------------+ | |    |  +------------+ | | |
 *|+---------+|  | ||IPv6.src+TCP +-+-+ +--+--+ Miss fwd   | | | | +-[Post CT]+
 *||IPv6+UDP ++--+>|+-------------+ |   |  |  +------------+ | | | |          |
 *|+---------+|  | ||IPv6.src+UDP +-+-+ |  |                 | | | |          |
 *||IPv6+TCP ++--+>|+-------------+ | | |  +-----------------+ | | |          |
 *|+---------+|  | |                | | |                      +-+>|          |
 *|           |  | |                | | |  +-[CT suffix UDP]-+ | | |          |
 *|           |  | |                | +-+->|                 | | | |          |
 *|           |  | |+-------------+ |   |  |                 | | | |          |
 *|           |  | ||  Miss fwd   | |   |  +-----------------+ | | |          |
 *|           |  | |+-------+-----+ |   |  |Prefix_id +      | | | |          |
 *|           |  | |        |       |   |  |IPv6.dst +       +-+ | |          |
 *+-----------+  | |        |       |   |  |UDP ports        |   | |          |
 *               | +--------+-------+   |  +-----------------+   | +----------+
 *               |          |           |  |  +------------+ |   |
 *               |          |           | ++--+ Miss fwd   | |   |
 *               |          |           | ||  +------------+ |   |
 *               |          |           | ||                 |   |
 *               |          |           | |+-----------------+   |
 *               +----------+-----------+-+----------------------+
 *                          |           | |
 *                          |    +------v-v---------+
 *                          +--->|     Miss Pipe    |
 *                               +------------------+
 *
 * Meter action post processing
 * ============================
 *
 * +-[pre-CT]------+
 * |               |
 * | +-[CT-zones]--+--+
 * | |                |
 * | | +-[post-CT]----+-+         +-[normal-tables]-+        +-[POST-METER]-------+
 * | | |                |         |                 |        |                    |
 * | | |                |         |                 |        |  +--------------+  |
 * | | |                |         |       ...       |    +---+->|action=meter2 +--+--+
 * | | |                |         |                 |    |   |  +--------------+  |  |
 * | | |                |         | +------------+  |    |   |        ...         |  |
 * | | |                +-------->| |action=meter+--+----+   |  +--------------+  |  |
 * | | |                |         | +------------+  |        |  |action=meterN |<-+--+
 * +-+ |                |         |                 |        |  +-------+------+  |
 *   | |                |         |       ...       |        |          |         |      +------+
 *   +-+                |         |                 |        |          +---------+----->| port |
 *     |                |         |                 |        |                    |      +------+
 *     +----------------+         +-----------------+        +--------------------+
 *
 * Single meter action forwards to the POSTMETER_TABLE where matching on
 * red/green color is done and green packets get forwarded to the real
 * destination of the original flow.
 *
 * If there are more then one meter action, then there is a loop over
 * POSTMETER_TABLE to perform second meter action, then third meter action and
 * so on. Each iteration drops red packets and forwards green packets to the
 * next meter in POSTMETER_TABLE. When last meter action is reached, the next
 * match in POSTMETER_TABLE forwards green packets to the real destination.
 *
 * Multiple split flow
 * ===================
 *
 * If an original flow is too big to fit in a single STE, split the matches of
 * such flow to up to 10 different flows, in which each split flow takes some of
 * the original flow's matches. each match is mapped to an id "prefix_id" which
 * is then matched in the following split flow.
 *
 * Notes:
 * If an original flow has a tunnel header match, the split is done forcefully
 * over the outer header first then the inner.
 *
 * +-[split_depth(0)]---+    +-[split_depth(1)]---+       +-[split_depth(n)]---+
 * |                    |    |                    |       |                    |
 * |match:              |    |match:              |       |match:              |
 * | M(0)               |    | M(1)               |       | M(n)               |
 * |                    |    | prefix_id(1)       |       | prefix_id(n-1)     |
 * |                    |    |                    |       |                    |
 * |actions:            +--->|actions:            | . . . |actions:            |
 * | set prefix_id(1)   |    | set prefix_id(2)   |       | orig_actions       |
 * | jump split_depth(1)|    | jump split_depth(2)|       |                    |
 * |                    |    |                    |       |                    |
 * +--------------------+    +--------------------+       +--------------------+
 *
 * The diagrams were initially drawn with https://asciiflow.com/ and edited in VIM.
 * The resulting extended ASCII chars should however be avoided.
 */

COVERAGE_DEFINE(doca_async_queue_full);
COVERAGE_DEFINE(doca_async_queue_blocked);
COVERAGE_DEFINE(doca_async_add_failed);
COVERAGE_DEFINE(doca_pipe_resize);
COVERAGE_DEFINE(doca_pipe_resize_over_10_ms);

/* This is hardware-specific.
 * Current NIC must abide by this limit.
 *
 * Need a feature to query it from the actual HW.
 *
 * The value reflects the max number of bytes we can match on
 * in single rule where each field in the match contributes to
 * the total matched bytes according to his size.
 */
#define MAX_FIELD_BYTES 32

#define ENTRY_PROCESS_TIMEOUT_MS 1000
/* TBD until doca can support insertion from more than one queue */
#define MAX_OFFLOAD_QUEUE_NB MAX_OFFLOAD_THREAD_NB
#define AUX_QUEUE 0

#define MAX_GENEVE_OPT 1

#define SHARED_MTR_N_IDS (OVS_DOCA_MAX_MEGAFLOWS_COUNTERS * \
                          DPDK_OFFLOAD_MAX_METERS_PER_FLOW)
#define MIN_SHARED_MTR_FLOW_ID 1

#define PIPE_RESIZE_ELAPSED_MAX_MS 10

#define DOCA_RESIZED_PIPE_CONGESTION_LEVEL 50

VLOG_DEFINE_THIS_MODULE(dpdk_offload_doca);
static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(600, 600);

#define SPLIT_FIELD(field, type, proto_type) \
                    {type, proto_type, \
                     offsetof(struct doca_flow_header_format, field), \
                     MEMBER_SIZEOF(struct doca_flow_header_format, field)}

enum split_field_type {
    FIELD_TYPE_INVALID,
    FIELD_TYPE_NONE,
    FIELD_TYPE_SRC,
    FIELD_TYPE_DST,
};

struct split_field {
    int type;
    int proto_type;
    uint16_t offset;
    size_t size;
};

enum split_l2_field_names {
    FIELD_ETH_MAC_SRC,
    FIELD_ETH_MAC_DST,
    FIELD_ETH_TYPE,
    FIELD_ETH_VLAN_TCI,
    NUM_L2_FIELDS,
};

enum split_l3_field_names {
    FIELD_L3_TYPE,
    FIELD_IP4_SRC,
    FIELD_IP4_DST,
    FIELD_IP4_VER_IHL,
    FIELD_IP4_DSCP_ECN,
    FIELD_IP4_NXT_PROTO,
    FIELD_IP4_TTL,
    FIELD_IP6_SRC,
    FIELD_IP6_DST,
    FIELD_IP6_DSCP_ECN,
    FIELD_IP6_NXT_PROTO,
    FIELD_IP6_HOP_LIMIT,
    NUM_L3_FIELDS,
};

enum split_l4_field_names {
    FIELD_L4_TYPE,
    FIELD_UDP_SRC,
    FIELD_UDP_DSR,
    FIELD_TCP_SRC,
    FIELD_TCP_DST,
    FIELD_TCP_FLAGS,
    FIELD_ICMP_TYPE,
    FIELD_ICMP_CODE,
    NUM_L4_FIELDS,
};

enum split_field_layer {
    L2_HEADERS,
    L3_HEADERS,
    L4_HEADERS,
};

static struct split_field split_fields[][NUM_L3_FIELDS] = {
    [L2_HEADERS] = {
        SPLIT_FIELD(eth.src_mac, FIELD_TYPE_SRC, 0),
        SPLIT_FIELD(eth.dst_mac, FIELD_TYPE_DST, 0),
        SPLIT_FIELD(eth.type, FIELD_TYPE_NONE, 0),
        SPLIT_FIELD(eth_vlan[0].tci, FIELD_TYPE_NONE, 0),
    },
    [L3_HEADERS] = {
        SPLIT_FIELD(ip4.src_ip, FIELD_TYPE_SRC, DOCA_FLOW_L3_TYPE_IP4),
        SPLIT_FIELD(ip4.dst_ip, FIELD_TYPE_DST, DOCA_FLOW_L3_TYPE_IP4),
        SPLIT_FIELD(ip4.version_ihl, FIELD_TYPE_NONE, DOCA_FLOW_L3_TYPE_IP4),
        SPLIT_FIELD(ip4.dscp_ecn, FIELD_TYPE_NONE, DOCA_FLOW_L3_TYPE_IP4),
        SPLIT_FIELD(ip4.next_proto, FIELD_TYPE_NONE, DOCA_FLOW_L3_TYPE_IP4),
        SPLIT_FIELD(ip4.ttl, FIELD_TYPE_NONE, DOCA_FLOW_L3_TYPE_IP4),
        SPLIT_FIELD(ip6.src_ip, FIELD_TYPE_SRC, DOCA_FLOW_L3_TYPE_IP6),
        SPLIT_FIELD(ip6.dst_ip, FIELD_TYPE_DST, DOCA_FLOW_L3_TYPE_IP6),
        SPLIT_FIELD(ip6.dscp_ecn, FIELD_TYPE_NONE, DOCA_FLOW_L3_TYPE_IP6),
        SPLIT_FIELD(ip6.next_proto, FIELD_TYPE_NONE, DOCA_FLOW_L3_TYPE_IP6),
        SPLIT_FIELD(ip6.hop_limit, FIELD_TYPE_NONE, DOCA_FLOW_L3_TYPE_IP6),
    },
    [L4_HEADERS] = {
        SPLIT_FIELD(udp.l4_port.src_port, FIELD_TYPE_SRC,
                    DOCA_FLOW_L4_TYPE_EXT_UDP),
        SPLIT_FIELD(udp.l4_port.dst_port, FIELD_TYPE_DST,
                    DOCA_FLOW_L4_TYPE_EXT_UDP),
        SPLIT_FIELD(tcp.l4_port.src_port, FIELD_TYPE_SRC,
                    DOCA_FLOW_L4_TYPE_EXT_TCP),
        SPLIT_FIELD(tcp.l4_port.dst_port, FIELD_TYPE_DST,
                    DOCA_FLOW_L4_TYPE_EXT_TCP),
        SPLIT_FIELD(tcp.flags, FIELD_TYPE_NONE, DOCA_FLOW_L4_TYPE_EXT_TCP),
        SPLIT_FIELD(icmp.type, FIELD_TYPE_NONE, DOCA_FLOW_L4_TYPE_EXT_ICMP),
        SPLIT_FIELD(icmp.code, FIELD_TYPE_NONE, DOCA_FLOW_L4_TYPE_EXT_ICMP),
        SPLIT_FIELD(icmp.type, FIELD_TYPE_NONE, DOCA_FLOW_L4_TYPE_EXT_ICMP6),
        SPLIT_FIELD(icmp.code, FIELD_TYPE_NONE, DOCA_FLOW_L4_TYPE_EXT_ICMP6),
    },
};

enum ct_zone_cls_flow_type {
    CT_ZONE_FLOW_REVISIT,
    CT_ZONE_FLOW_UPHOLD_IP4_UDP,
    CT_ZONE_FLOW_UPHOLD_IP4_TCP,
    CT_ZONE_FLOW_UPHOLD_IP6_UDP,
    CT_ZONE_FLOW_UPHOLD_IP6_TCP,
    CT_ZONE_FLOW_MISS,
    CT_ZONE_FLOWS_NUM,
};

enum ct_nw_type {
    CT_NW_IP4, /* CT on IPv4 networks. */
    CT_NW_IP6, /* CT on IPv6 networks. */
    NUM_CT_NW,
};

enum ct_tp_type {
    CT_TP_UDP, /* CT on UDP datagrams. */
    CT_TP_TCP, /* CT on TCP streams. */
    NUM_CT_TP,
};

enum hash_pipe_type {
    HASH_TYPE_IPV4_UDP,
    HASH_TYPE_IPV4_TCP,
    HASH_TYPE_IPV4_L3,
    HASH_TYPE_IPV6_UDP_SUF,
    HASH_TYPE_IPV6_TCP_SUF,
    HASH_TYPE_IPV6_L3_SUF,
    HASH_TYPE_IPV6_UDP_PRE,
    HASH_TYPE_IPV6_TCP_PRE,
    HASH_TYPE_IPV6_L3_PRE,
    NUM_HASH_PIPE_TYPE,
};
/* The SUF types must be in lower positions than the PRE ones, as this is the
 * order of initialization, and release in the opposite order.
 */
BUILD_ASSERT_DECL(HASH_TYPE_IPV6_UDP_SUF < HASH_TYPE_IPV6_UDP_PRE);
BUILD_ASSERT_DECL(HASH_TYPE_IPV6_TCP_SUF < HASH_TYPE_IPV6_TCP_PRE);
BUILD_ASSERT_DECL(HASH_TYPE_IPV6_L3_SUF < HASH_TYPE_IPV6_L3_PRE);

struct doca_basic_pipe_ctx {
    struct doca_flow_pipe *pipe;
    struct doca_ctl_pipe_ctx *fwd_pipe_ctx;
    struct doca_ctl_pipe_ctx *miss_pipe_ctx;
};

/* +--------+   +-------------+
 * |IPv4-UDP|-->|HASH-IPv4-UDP|
 * |        |   +-------------+
 * |        |   +-------------+
 * |IPv4-TCP|-->|HASH-IPv4-TCP|
 * +---+----+   +-------------+
 *     | miss   +-------------+
 *     +------->|HASH-IPv4-L3 |
 *              +-------------+
 * +--------+   +-----------------+   +-----------------+
 * |IPv6-UDP|-->|HASH-IPv6-UDP_PRE|-->|HASH-IPv6-UDP_SUF|
 * |        |   +-----------------+   +-----------------+
 * |        |   +-----------------+   +-----------------+
 * |IPv6-TCP|-->|HASH-IPv6-TCP_PRE|-->|HASH-IPv6-TCP_SUF|
 * +---+----+   +-----------------+   +-----------------+
 *     | miss   +-----------------+   +-----------------+
 *     +------->|HASH-IPv6-L3_PRE |-->|HASH-IPv6-L3_SUF |
 *              +-----------------+   +-----------------+
 * OVS always matches on ether type.
 * We only need to know TCP/UDP, or miss to simple L3.
 * With IPv6, full IP match cannot fit in a single STE, thus a split is done.
 * In PRE, the SRC IP/port are hashed. In SUF, the hash result from the PRE
 * and the DST IP/port are hashed, to create the final hash.
 */
enum hash_tp_type {
    HASH_TP_UDP,
    HASH_TP_TCP,
    NUM_HASH_TP,
};

enum hash_nw_type {
    HASH_NW_IP4, /* CT on IPv4 networks. */
    HASH_NW_IP6, /* CT on IPv6 networks. */
    NUM_HASH_NW,
};

struct doca_hash_pipe_ctx {
    struct {
        struct doca_flow_pipe *pipe;
        struct doca_flow_pipe_entry *entry;
    } hashes[NUM_HASH_PIPE_TYPE];
    struct {
        struct doca_flow_pipe *pipe;
        struct doca_flow_pipe_entry *tcpudp[NUM_HASH_TP];
    } classifier[NUM_HASH_NW];
    struct doca_ctl_pipe_ctx *post_hash_pipe_ctx;
    struct netdev *netdev;
};

struct doca_pipe_resize_ctx {
    struct ovs_list resized_list_node;
    void *esw_ctx;
    struct netdev *netdev;
    uint32_t group_id;
    atomic_bool resizing;
};

struct doca_ctl_pipe_ctx {
    struct doca_flow_pipe *pipe;
    struct doca_pipe_resize_ctx resize_ctx;
    uint32_t miss_flow_id;
};

struct doca_async_entry {
    struct netdev *netdev; /* If set, port that posted this entry. */
    struct dpdk_offload_handle *doh; /* Corresponding handle for this entry. */
    unsigned int index; /* Index of this entry within the aync_state array. */
};

struct doca_async_state {
    PADDED_MEMBERS(CACHE_LINE_SIZE,
        unsigned int n_entries;
        struct doca_async_entry entries[OVS_DOCA_QUEUE_DEPTH];
    );
};

struct gnv_opt_parser {
    struct ovsthread_once once;
    struct doca_flow_parser *parser;
};

OVS_ASSERT_PACKED(struct doca_eswitch_ctx,
    struct doca_flow_port *esw_port;
    struct doca_ctl_pipe_ctx *root_pipe_ctx;
    struct gnv_opt_parser gnv_opt_parser;
    struct doca_async_state async_state[MAX_OFFLOAD_QUEUE_NB];
    struct doca_basic_pipe_ctx ct_pipes[NUM_CT_NW][NUM_CT_TP];
    struct fixed_rule zone_cls_flows[2][CT_ZONE_FLOWS_NUM][MAX_ZONE_ID + 1];
    struct doca_basic_pipe_ctx ct_ip6_prefix;
    struct id_fpool *shared_counter_id_pool;
    struct id_fpool *shared_ct_counter_id_pool;
    struct id_fpool *shared_mtr_flow_id_pool;
    uint32_t esw_id;
    struct ovs_refcount pipe_resizing;
    struct ovs_list resized_pipe_lists[MAX_OFFLOAD_QUEUE_NB];
    struct doca_ctl_pipe_ctx *post_meter_pipe_ctx;
    struct doca_hash_pipe_ctx *hash_pipe_ctx;
);

OVS_ASSERT_PACKED(struct doca_ctl_pipe_key,
    uint32_t group_id;
    uint32_t esw_mgr_port_id;
);

struct doca_ctl_pipe_arg {
    struct netdev *netdev;
    uint32_t group_id;
};

struct meter_info {
    struct ovs_list list_node;
    uint32_t id;
    uint32_t flow_id;
};

struct doca_act_vars {
    uint32_t flow_id;
    uint32_t mtr_flow_id;
    struct ovs_list next_meters;
    const struct hash_data *hash_data;
};

struct doca_meter_ctx {
    struct ovs_list list_node;
    uint32_t post_meter_flow_id;
    struct doca_flow_pipe_entry *post_meter_red_entry;
    struct doca_flow_pipe_entry *post_meter_green_entry;
};

static int
destroy_dpdk_offload_handle(struct netdev *netdev,
                            struct dpdk_offload_handle *doh,
                            unsigned int queue_id,
                            struct rte_flow_error *error);
static struct id_fpool *esw_id_pool;

static bool doca_ct_offload_enabled;

static struct doca_eswitch_ctx *
doca_eswitch_ctx_get(struct netdev *netdev);

static int
shared_mtr_flow_id_alloc(struct doca_eswitch_ctx *esw_ctx,
                         struct rte_flow_error *error);

static void
shared_mtr_flow_id_free(struct doca_eswitch_ctx *esw_ctx, uint32_t id);

static struct doca_ctl_pipe_ctx *
doca_ctl_pipe_ctx_ref(struct netdev *netdev, uint32_t group_id);

static void
doca_ctl_pipe_ctx_unref(struct doca_ctl_pipe_ctx *ctx);

/* From an async entry in the descriptor queue kept in an
 * eswitch context, find back through pointer arithmetic the
 * containing eswitch context. */
static inline struct doca_eswitch_ctx *
doca_eswitch_ctx_from_async_entry(struct doca_async_entry *dae,
                                  unsigned int qid)
{
    struct doca_async_state *das, *async_state;
    struct doca_async_entry *entries;

    entries = dae - dae->index;
    das = CONTAINER_OF(entries, struct doca_async_state, entries);
    async_state = das - qid;
    return CONTAINER_OF(async_state, struct doca_eswitch_ctx, async_state);
}

static inline enum ct_nw_type
l3_to_nw_type(enum doca_flow_l3_type l3_type)
{
    switch (l3_type) {
    case DOCA_FLOW_L3_TYPE_IP4: return CT_NW_IP4;
    case DOCA_FLOW_L3_TYPE_IP6: return CT_NW_IP6;
    case DOCA_FLOW_L3_TYPE_NONE: return NUM_CT_NW;
    };
    return NUM_CT_NW;
}

static inline enum ct_tp_type
l4_to_tp_type(enum doca_flow_l4_type_ext l4_type)
{
    switch (l4_type) {
    case DOCA_FLOW_L4_TYPE_EXT_TCP: return CT_TP_TCP;
    case DOCA_FLOW_L4_TYPE_EXT_UDP: return CT_TP_UDP;
    case DOCA_FLOW_L4_TYPE_EXT_ICMP:
    case DOCA_FLOW_L4_TYPE_EXT_ICMP6:
    case DOCA_FLOW_L4_TYPE_EXT_TRANSPORT:
    case DOCA_FLOW_L4_TYPE_EXT_NONE: return NUM_CT_TP;
    }
    return NUM_CT_TP;
}

static bool
is_ct_zone_group_id(uint32_t group)
{
    return ((group >= CT_TABLE_ID + MIN_ZONE_ID &&
             group <= CT_TABLE_ID + MAX_ZONE_ID) ||
            (group >= CTNAT_TABLE_ID + MIN_ZONE_ID &&
             group <= CTNAT_TABLE_ID + MAX_ZONE_ID));
}

#define MIN_SPLIT_PREFIX_ID 1
#define MAX_SPLIT_PREFIX_ID (reg_fields[REG_FIELD_SCRATCH].mask - 1)
#define NUM_SPLIT_PREFIX_ID (MAX_SPLIT_PREFIX_ID - MIN_SPLIT_PREFIX_ID + 1)
static struct offload_metadata *split_prefix_md;
static struct id_fpool *split_prefix_id_pool = NULL;

OVS_ASSERT_PACKED(struct doca_split_prefix_key,
    struct doca_flow_match spec;
    struct doca_flow_match mask;
    struct doca_flow_pipe *prefix_pipe;
);

struct doca_split_prefix_ctx {
    struct netdev *netdev;
    struct dpdk_offload_handle doh;
};

struct doca_split_prefix_arg {
    struct netdev *netdev;
    struct doca_flow_match *spec;
    struct doca_flow_match *mask;
    int prefix_pipe_type;
    struct doca_flow_pipe *prefix_pipe;
    struct doca_flow_pipe *suffix_pipe;
    uint32_t set_flow_info_id;
    uint32_t prio;
};

static int
doca_ctl_pipe_ctx_init(void *ctx_, void *arg_, uint32_t id OVS_UNUSED)
{
    struct doca_ctl_pipe_ctx *ctx = ctx_;
    struct doca_ctl_pipe_arg *arg = arg_;
    struct doca_flow_port *doca_port;
    struct doca_eswitch_ctx *esw_ctx;
    struct doca_flow_pipe_cfg cfg;
    char pipe_name[50];
    uint32_t group_id;
    bool is_root;
    int ret;

    /* The pipe for recirc = 0 without any tunnel involved is
     * global and shared among devices on the esw. It is a root pipe.
     */
    group_id = arg->group_id;
    is_root = group_id == 0;
    snprintf(pipe_name, sizeof pipe_name, "OVS_CTL_PIPE_%" PRIu32, group_id);
    doca_port = netdev_dpdk_doca_port_get(arg->netdev);

    memset(&cfg, 0, sizeof cfg);
    cfg.attr.name = pipe_name;
    cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
    cfg.attr.is_root = is_root;
    cfg.attr.enable_strict_matching = true;
    cfg.port = doca_flow_port_switch_get(doca_port);

    if (is_ct_zone_group_id(group_id)) {
        cfg.attr.nb_flows = CT_ZONE_FLOWS_NUM;
    } else if (group_id == MISS_TABLE_ID) {
        cfg.attr.nb_flows = 1;
    } else {
        if (group_id == 0 || group_id == POSTHASH_TABLE_ID ||
            group_id == POSTCT_TABLE_ID || group_id == POSTMETER_TABLE_ID ||
            IS_SPLIT_TABLE_ID(group_id) || group_id == MISS_TABLE_ID) {
            cfg.attr.nb_flows = ctl_pipe_infra_size;
        } else {
            cfg.attr.nb_flows = ctl_pipe_size;
        }
        cfg.attr.is_resizable = true;

        ctx->resize_ctx.netdev = arg->netdev;
        ctx->resize_ctx.group_id = arg->group_id;
        ovs_list_init(&ctx->resize_ctx.resized_list_node);
        esw_ctx = doca_eswitch_ctx_get(arg->netdev);
        if (!esw_ctx) {
            VLOG_WARN("%s: Failed to create ctl pipe: esw_ctx not found",
                      netdev_get_name(arg->netdev));
            return EINVAL;
        }
        ctx->resize_ctx.esw_ctx = esw_ctx;
        cfg.attr.user_ctx = ctx;
        cfg.attr.congestion_level_threshold = doca_congestion_threshold;
    }

    ret = doca_flow_pipe_create(&cfg, NULL, NULL, &ctx->pipe);
    if (ret) {
        VLOG_ERR("%s: Failed to create ctl pipe: %d (%s)",
                 netdev_get_name(arg->netdev), ret, doca_error_get_descr(ret));
    }
    return ret;
}

static void
doca_hash_pipe_ctx_uninit(struct doca_hash_pipe_ctx *ctx)
{
    unsigned int queue_id = netdev_offload_thread_id();
    enum hash_nw_type nw_type;
    int i;

    if (ctx == NULL) {
        return;
    }

    /* Destroy classifiers. */
    for (nw_type = 0; nw_type < NUM_HASH_NW; nw_type++) {
        for (i = 0; i < NUM_HASH_TP; i++) {
            if (ctx->classifier[nw_type].tcpudp[i]) {
                doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT,
                                        ctx->classifier[nw_type].tcpudp[i]);
                dpdk_offload_counter_dec(ctx->netdev);
            }
        }
        if (ctx->classifier[nw_type].pipe) {
            doca_flow_pipe_destroy(ctx->classifier[nw_type].pipe);
        }
    }

    /* Release hash pipes in the reverse order of their dependency.
     * Ordering is described and enforced at the `hash_pipe_type` definition.
     */
    for (i = NUM_HASH_PIPE_TYPE - 1; i >= 0; i--) {
        if (ctx->hashes[i].entry) {
            doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT,
                                    ctx->hashes[i].entry);
            dpdk_offload_counter_dec(ctx->netdev);
        }
        if (ctx->hashes[i].pipe) {
            doca_flow_pipe_destroy(ctx->hashes[i].pipe);
        }
    }

    doca_ctl_pipe_ctx_unref(ctx->post_hash_pipe_ctx);

    free(ctx);
}

static void
doca_ctl_pipe_ctx_uninit(void *ctx_)
{
    struct doca_ctl_pipe_ctx *ctx = ctx_;

    doca_flow_pipe_destroy(ctx->pipe);
    ctx->pipe = NULL;
}

static struct ds *
dump_doca_ctl_pipe_ctx(struct ds *s, void *key_, void *ctx_, void *arg_ OVS_UNUSED)
{
    struct doca_ctl_pipe_key *key = key_;
    struct doca_ctl_pipe_ctx *ctx = ctx_;

    if (ctx) {
        ds_put_format(s, "ctl_pipe=%p, ", ctx->pipe);
    }
    ds_put_format(s, "group_id=%"PRIu32", ", key->group_id);

    return s;
}

static struct offload_metadata *doca_ctl_pipe_md;

static void
doca_ctl_pipe_md_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .priv_size = sizeof(struct doca_ctl_pipe_ctx),
            .priv_init = doca_ctl_pipe_ctx_init,
            .priv_uninit = doca_ctl_pipe_ctx_uninit,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        doca_ctl_pipe_md = offload_metadata_create(nb_thread, "doca_ctl_pipe",
                                                   sizeof(struct doca_ctl_pipe_key),
                                                   dump_doca_ctl_pipe_ctx,
                                                   params);

        ovsthread_once_done(&init_once);
    }
}

static struct doca_ctl_pipe_ctx *
doca_ctl_pipe_ctx_ref(struct netdev *netdev, uint32_t group_id)
{
    struct doca_ctl_pipe_key key = {
        .group_id = group_id,
        .esw_mgr_port_id = netdev_dpdk_get_esw_mgr_port_id(netdev),
    };
    struct doca_ctl_pipe_arg arg = {
        .netdev = netdev,
        .group_id = group_id,
    };

    doca_ctl_pipe_md_init();
    return offload_metadata_priv_get(doca_ctl_pipe_md, &key, &arg, NULL, true);
}

static void
doca_ctl_pipe_ctx_unref(struct doca_ctl_pipe_ctx *ctx)
{
    if (ctx == NULL) {
        return;
    }

    doca_ctl_pipe_md_init();
    offload_metadata_priv_unref(doca_ctl_pipe_md,
                                netdev_offload_thread_id(),
                                ctx);
}

static bool
doca_ctl_pipe_resizing(struct doca_ctl_pipe_ctx *ctx)
{
    bool resizing;

    atomic_read(&ctx->resize_ctx.resizing, &resizing);
    return resizing;
}

static void
doca_ctl_pipe_ctx_resize_begin(struct doca_ctl_pipe_ctx *ctx)
{
    long long int resize_start_ms, elapsed_ms;
    struct doca_eswitch_ctx *esw_ctx;

    esw_ctx = doca_eswitch_ctx_get(ctx->resize_ctx.netdev);
    ovs_refcount_ref(&esw_ctx->pipe_resizing);
    doca_ctl_pipe_ctx_ref(ctx->resize_ctx.netdev, ctx->resize_ctx.group_id);
    resize_start_ms = time_msec();
    doca_flow_pipe_resize(ctx->pipe, DOCA_RESIZED_PIPE_CONGESTION_LEVEL,
                          NULL, NULL);
    elapsed_ms = time_msec() - resize_start_ms;
    COVERAGE_INC(doca_pipe_resize);
    if (elapsed_ms > PIPE_RESIZE_ELAPSED_MAX_MS) {
        COVERAGE_INC(doca_pipe_resize_over_10_ms);
    }
    atomic_store_relaxed(&ctx->resize_ctx.resizing, true);
}

static void
doca_ctl_pipe_ctx_resize_end_defer(struct doca_ctl_pipe_ctx *ctx)
{
    unsigned int tid = netdev_offload_thread_id();
    struct doca_eswitch_ctx *esw_ctx;

    esw_ctx = doca_eswitch_ctx_get(ctx->resize_ctx.netdev);
    ovs_list_push_back(&esw_ctx->resized_pipe_lists[tid],
                       &ctx->resize_ctx.resized_list_node);
}

static void
doca_ctl_pipe_ctx_resize_end(struct doca_ctl_pipe_ctx *ctx)
{
    struct doca_eswitch_ctx *esw_ctx;

    atomic_store(&ctx->resize_ctx.resizing, false);
    esw_ctx = doca_eswitch_ctx_get(ctx->resize_ctx.netdev);
    ovs_refcount_unref(&esw_ctx->pipe_resizing);
    doca_ctl_pipe_ctx_unref(ctx);
}

static struct reg_field reg_fields[] = {
    [REG_FIELD_CT_STATE] = {
        .type = REG_TYPE_TAG,
        .index = 0,
        .offset = 0,
        .mask = 0x000000FF,
    },
    [REG_FIELD_CT_ZONE] = {
        .type = REG_TYPE_TAG,
        .index = 0,
        .offset = 8,
        .mask = 0x000000FF,
    },
    /* Re-use REG_FIELD_CT_CTX register as hash is not supported concurrently
     * with CT.
     */
    [REG_FIELD_DP_HASH] = {
        .type = REG_TYPE_META,
        .index = 0,
        .offset = 12,
        .mask = 0x0000000F,
    },
    [REG_FIELD_TUN_INFO] = {
        .type = REG_TYPE_TAG,
        .index = 0,
        .offset = 16,
        .mask = 0x0000FFFF,
    },
    [REG_FIELD_CT_MARK] = {
        .type = REG_TYPE_TAG,
        .index = 1,
        .offset = 0,
        .mask = 0xFFFFFFFF,
    },
    [REG_FIELD_CT_LABEL_ID] = {
        .type = REG_TYPE_TAG,
        .index = 2,
        .offset = 0,
        .mask = 0xFFFFFFFF,
    },
    [REG_FIELD_CT_CTX] = {
        .type = REG_TYPE_META,
        .index = 0,
        .offset = 0,
        .mask = 0x0000FFFF,
    },
    /* sFlow is currently not supported for DOCA. */
    [REG_FIELD_SFLOW_CTX] = {
        .type = REG_TYPE_META,
        .index = 0,
        .offset = 0,
        .mask = 0x0000FFFF,
    },
    /* Since recirc-reg and CT will not work concurrently is it safe
     * to have the reg_fields use the same bits for RECIRC and CT_MARK.
     */
    [REG_FIELD_RECIRC] = {
        .type = REG_TYPE_TAG,
        .index = 1,
        .offset = 0,
        .mask = 0xFFFFFFFF,
    },
    [REG_FIELD_FLOW_INFO] = {
        .type = REG_TYPE_META,
        .index = 0,
        .offset = 16,
        .mask = 0x0000FFFF,
    },
    /* Re-use ct label id register to store scratch info:
     * - prefix id for split flows.
     * - dp-hash ipv6 split hash result.
     * - dp-hash seed.
     * For a split flow we make sure that ct label match will be included
     * in the prefix flow before we overwrite it with the prefix id and
     * the prefix flow always points to a suffix flow where the prefix id
     * will be matched and no longer needed - therefore, it can then
     * be re-used to store ct label id (in case the suffix flow includes
     * a ct action).
     * For dp-hash as hash is not supported concurrently with CT.
     */
    [REG_FIELD_SCRATCH] = {
        .type = REG_TYPE_TAG,
        .index = 2,
        .offset = 0,
        .mask = 0xFFFFFFFF,
    },
};

/* Register usage:
 * +------+--------+--------+--------+--------+--------+--------+--------+--------+
 * |      |  31-28 |  27-24 |  23-20 |  19-16 |  15-12 |  11-8  |   7-4  |   3-0  |
 * |      +-----------------+-----------------+-----------------+-----------------+
 * |      |      31-24      |      23-16      |      15-8       |       7-0       |
 * |      +-----------------+-----------------+-----------------+-----------------+
 * |      |                                 31-0                                  |
 * +------+-----------------+-----------------+-----------------+-----------------+
 * | meta |               FLOW_INFO           |            CT_MISS_CTX            |
 * |      | - Pre/post CT: flow-id. In case   | ID to restore CT information      |
 * |      |   of CT miss, it has a            | on miss after CT operation:       |
 * |      |   flow_miss_ctx to recover.       | - CT state.                       |
 * |      | - Post-hash: flow-id.             | - CT zone.                        |
 * |      | - Meters: meter-flow-id. As a     | - CT mark.                        |
 * |      |   temp id in the post-meter loop. | - CT label.                       |
 * |      |   Last iteration restores to the  |                                   |
 * |      |   flow-id.                        |                                   |
 * |      |                                   +--------+--------+--------+--------+
 * |      |                                   |dp-hash |                          |
 * +------+-----------------+-----------------+-----------------+-----------------+
 * | tag0 |           TUNNEL_INFO             |     CT_ZONE     |    CT_STATE     |
 * |      | - recirc_id(0) matches on the     |                 |                 |
 * |      |   packet's fields, and decap. The |                 |                 |
 * |      |   tunnel match is mapped.         |                 |                 |
 * |      |   Following recircs map on this   |                 |                 |
 * |      |   ID.                             |                 |                 |
 * +------+-----------------+-----------------+-----------------+-----------------+
 * | tag1 |                               CT_MARK                                 |
 * |      |                                   /                                   |
 * |      |                      recirc in case CT is disabled.                   |
 * +------+-----------------+-----------------+-----------------+-----------------+
 * | tag2 |                      CT_LABEL. 32 low bits or mapped.                 |
 * |      |                                   /                                   |
 * |      |                                SCRATCH                                |
 * |      | - Prefix id for split flows.                                          |
 * |      | - dp-hash ipv6 split hash result.                                     |
 * |      | - dp-hash seed.                                                       |
 * +------+-----------------+-----------------+-----------------+-----------------+
 */

static struct reg_field *
dpdk_offload_doca_get_reg_fields(void)
{
    return reg_fields;
}

static uint32_t
doca_get_reg_val(struct doca_flow_meta *dmeta, uint8_t reg_field_id)
{
    struct reg_field *reg_field = &reg_fields[reg_field_id];
    uint32_t val;

    val = reg_field->type == REG_TYPE_TAG
        ? dmeta->u32[reg_field->index]
        : dmeta->pkt_meta;
    val >>= reg_field->offset;

    return val & reg_field->mask;
}

static void
doca_set_reg_val(struct doca_flow_meta *dmeta, uint8_t reg_field_id,
                 uint32_t val)
{
    struct reg_field *reg_field = &reg_fields[reg_field_id];
    uint32_t offset = reg_field->offset;
    uint32_t mask = reg_field->mask;

    val = (val & mask) << offset;
    if (reg_field->type == REG_TYPE_TAG) {
        dmeta->u32[reg_field->index] |= val;
    } else {
        dmeta->pkt_meta |= val;
    }
}

static void
doca_set_reg_mask(struct doca_flow_meta *dmeta, uint8_t reg_field_id)
{
    doca_set_reg_val(dmeta, reg_field_id, UINT32_MAX);
}

static void
doca_set_reg_val_mask(struct doca_flow_meta *dmeta,
                      struct doca_flow_meta *dmeta_mask,
                      uint8_t reg_field_id,
                      uint32_t val)
{
    doca_set_reg_val(dmeta, reg_field_id, val);
    doca_set_reg_mask(dmeta_mask, reg_field_id);
}

static uint32_t
doca_get_reg_bit_offset(uint8_t reg_field_id)
{
    struct reg_field *reg_field = &reg_fields[reg_field_id];
    uint32_t offset;

    offset = reg_field->offset;

    if (reg_field->type == REG_TYPE_TAG) {
        offset += offsetof(struct doca_flow_meta, u32[reg_field->index]) * 8;
    } else {
        offset += offsetof(struct doca_flow_meta, pkt_meta) * 8;
    }

    return offset;
}

static uint32_t
doca_get_reg_width(uint8_t reg_field_id)
{
    struct reg_field *reg_field = &reg_fields[reg_field_id];

    return ffsl(~((uint64_t) reg_field->mask)) - 1;
}

static void
doca_spec_mask(void *dspec__, void *dmask__, size_t size)
{
    char *dspec = dspec__;
    char *dmask = dmask__;
    int i;

    if (!dspec || !dmask) {
        return;
    }

    for (i = 0; i < size; i++) {
        dspec[i] &= dmask[i];
    }
}

static void
doca_copy_mask_field__(void *dspec, void *dmask,
                       const void *spec, const void *mask,
                       size_t size)
{
    memcpy(dspec, spec, size);
    memcpy(dmask, mask, size);
    doca_spec_mask(dspec, dmask, size);
}

static void
doca_translate_gre_key_item(const struct rte_flow_item *item,
                            struct doca_flow_match *doca_spec,
                            struct doca_flow_match *doca_mask)
{
    const rte_be32_t *key_spec, *key_mask;

    doca_spec->tun.type = DOCA_FLOW_TUN_GRE;
    doca_mask->tun.type = DOCA_FLOW_TUN_GRE;

    key_spec = item->spec;
    key_mask = item->mask;

    if (item->spec) {
        doca_spec->tun.gre_key = *key_spec;
    }
    if (item->mask) {
        doca_mask->tun.gre_key = *key_mask;
    }

    doca_spec_mask(&doca_spec->tun.gre_key, &doca_mask->tun.gre_key,
                   sizeof doca_spec->tun.gre_key);
}

static void
doca_translate_gre_item(const struct rte_flow_item *item,
                        struct doca_flow_match *doca_spec,
                        struct doca_flow_match *doca_mask)
{
    const struct rte_gre_hdr *greh_spec, *greh_mask;

    doca_spec->tun.type = DOCA_FLOW_TUN_GRE;
    doca_mask->tun.type = DOCA_FLOW_TUN_GRE;

    greh_spec = (struct rte_gre_hdr *) item->spec;
    greh_mask = (struct rte_gre_hdr *) item->mask;

    if (item->spec) {
        doca_spec->tun.key_present = greh_spec->k;
    }
    if (item->mask) {
        doca_mask->tun.key_present = greh_mask->k;
    }

    doca_spec_mask(&doca_spec->tun.key_present, &doca_mask->tun.key_present,
                   sizeof doca_spec->tun.key_present);
}

static void
doca_translate_geneve_item(const struct rte_flow_item *item,
                           struct doca_flow_match *doca_spec,
                           struct doca_flow_match *doca_mask)
{
    const struct rte_flow_item_geneve *gnv_spec = item->spec;
    const struct rte_flow_item_geneve *gnv_mask = item->mask;

    if (!item->spec || !item->mask) {
        return;
    }

    doca_spec->tun.type = DOCA_FLOW_TUN_GENEVE;
    doca_spec->tun.geneve.vni =
        get_unaligned_be32(ALIGNED_CAST(ovs_be32 *, gnv_spec->vni));

    doca_mask->tun.type = DOCA_FLOW_TUN_GENEVE;
    doca_mask->tun.geneve.vni =
        get_unaligned_be32(ALIGNED_CAST(ovs_be32 *, gnv_mask->vni));

    doca_spec_mask(&doca_spec->tun.geneve.vni, &doca_mask->tun.geneve.vni,
                   sizeof doca_spec->tun.geneve.vni);
}

static int
doca_init_geneve_opt_parser(struct netdev *netdev,
                            const struct rte_flow_item *item)
{
    struct doca_flow_parser_geneve_opt_cfg opt_cfg[MAX_GENEVE_OPT];
    const struct rte_flow_item_geneve_opt *geneve_opt_spec;
    struct doca_eswitch_ctx *esw_ctx;
    int ret;

    esw_ctx = doca_eswitch_ctx_get(netdev);
    if (!esw_ctx) {
        VLOG_ERR("%s: Failed to create geneve_opt parser - esw_ctx is NULL",
                 netdev_get_name(netdev));
        return -1;
    }

    if (!ovsthread_once_start(&esw_ctx->gnv_opt_parser.once)) {
        return 0;
    }
    geneve_opt_spec = item->spec;

    memset(&opt_cfg[0], 0, sizeof(opt_cfg[0]));
    opt_cfg[0].match_on_class_mode =
        DOCA_FLOW_PARSER_GENEVE_OPT_MODE_MATCHABLE;
    opt_cfg[0].option_len = geneve_opt_spec->option_len;
    opt_cfg[0].option_class = geneve_opt_spec->option_class;
    opt_cfg[0].option_type = geneve_opt_spec->option_type;
    BUILD_ASSERT_DECL(MEMBER_SIZEOF(struct doca_flow_parser_geneve_opt_cfg,
                                    data_mask[0]) ==
                      MEMBER_SIZEOF(struct rte_flow_item_geneve_opt, data[0]));
    memset(&opt_cfg[0].data_mask[0], UINT32_MAX,
           sizeof(opt_cfg[0].data_mask[0]) * geneve_opt_spec->option_len);

    ret = doca_flow_parser_geneve_opt_create(esw_ctx->esw_port, opt_cfg,
                                             MAX_GENEVE_OPT,
                                             &esw_ctx->gnv_opt_parser.parser);
    if (ret) {
        VLOG_DBG_RL(&rl, "%s: Create geneve_opt parser failed - doca call failure "
                         "rc %d, (%s)",netdev_get_name(netdev), ret,
                         doca_error_get_descr(ret));
        ovsthread_once_reset(&esw_ctx->gnv_opt_parser.once);
        return -1;
    }
    ovsthread_once_done(&esw_ctx->gnv_opt_parser.once);
    return 0;
}

static void
doca_translate_geneve_opt_item(const struct rte_flow_item *item,
                               struct doca_flow_match *doca_spec,
                               struct doca_flow_match *doca_mask)
{
    union doca_flow_geneve_option *doca_opt_spec, *doca_opt_mask;
    const struct rte_flow_item_geneve_opt *geneve_opt_spec;
    const struct rte_flow_item_geneve_opt *geneve_opt_mask;

    geneve_opt_spec = item->spec;
    geneve_opt_mask = item->mask;
    doca_opt_spec = &doca_spec->tun.geneve_options[0];
    doca_opt_mask = &doca_mask->tun.geneve_options[0];

    doca_opt_spec->length = geneve_opt_spec->option_len;
    doca_opt_spec->class_id = geneve_opt_spec->option_class;
    doca_opt_spec->type = geneve_opt_spec->option_type;
    doca_opt_mask->length = geneve_opt_mask->option_len;
    doca_opt_mask->class_id = geneve_opt_mask->option_class;
    doca_opt_mask->type = geneve_opt_mask->option_type;
    doca_spec_mask(&doca_opt_spec->length, &doca_opt_mask->length,
                   sizeof doca_opt_spec->length);
    doca_spec_mask(&doca_opt_spec->class_id, &doca_opt_mask->class_id,
                   sizeof doca_opt_spec->class_id);
    doca_spec_mask(&doca_opt_spec->type, &doca_opt_mask->type,
                   sizeof doca_opt_spec->type);

    /* doca_flow represents the geneve option header as an array of a union of
     * 32 bits, the array's first element is the type/class/len and this
     * option's data starts from the next element in the array up to option_len
     */
    doca_opt_spec++;
    doca_opt_mask++;
    BUILD_ASSERT_DECL(sizeof(doca_opt_spec->data) ==
                      sizeof(geneve_opt_spec->data[0]));
    memcpy(&doca_opt_spec->data, &geneve_opt_spec->data[0],
           sizeof(doca_opt_spec->data) * geneve_opt_spec->option_len);
    memcpy(&doca_opt_mask->data, &geneve_opt_mask->data[0],
           sizeof(doca_opt_mask->data) * geneve_opt_spec->option_len);
    doca_spec_mask(&doca_opt_spec->data, &doca_opt_mask->data,
                   sizeof(doca_opt_spec->data) * geneve_opt_spec->option_len);
}

static void
doca_translate_vxlan_item(const struct rte_flow_item *item,
                          struct doca_flow_match *doca_spec,
                          struct doca_flow_match *doca_mask)
{
    const struct rte_flow_item_vxlan *vxlan_spec = item->spec;
    const struct rte_flow_item_vxlan *vxlan_mask = item->mask;
    ovs_be32 spec_vni, mask_vni;

    doca_spec->tun.type = DOCA_FLOW_TUN_VXLAN;
    if (item->spec) {
        spec_vni = get_unaligned_be32(ALIGNED_CAST(ovs_be32 *,
                    vxlan_spec->vni));
        doca_spec->tun.vxlan_tun_id = spec_vni;
    }

    doca_mask->tun.type = DOCA_FLOW_TUN_VXLAN;
    if (item->mask) {
        mask_vni = get_unaligned_be32(ALIGNED_CAST(ovs_be32 *,
                    vxlan_mask->vni));
        doca_mask->tun.vxlan_tun_id = mask_vni;
    }

    doca_spec_mask(&doca_spec->tun.vxlan_tun_id, &doca_mask->tun.vxlan_tun_id,
                   sizeof doca_spec->tun.vxlan_tun_id);
}

static int
doca_translate_items(struct netdev *netdev,
                     const struct rte_flow_attr *attr,
                     const struct rte_flow_item *items,
                     struct doca_flow_match *doca_spec,
                     struct doca_flow_match *doca_mask)
{
    struct doca_flow_header_format *doca_hdr_spec, *doca_hdr_mask;

#define doca_copy_mask_field(dfield, field)                   \
    if (spec && mask) {                                       \
        doca_copy_mask_field__(&doca_hdr_spec->dfield,        \
                               &doca_hdr_mask->dfield,        \
                               &spec->field,                  \
                               &mask->field,                  \
                               sizeof doca_hdr_spec->dfield); \
    }

    /* Start by filling out outer header match and
     * switch to inner in case we encounter a tnl proto.
     */
    doca_hdr_spec = &doca_spec->outer;
    doca_hdr_mask = &doca_mask->outer;

    for (; items->type != RTE_FLOW_ITEM_TYPE_END; items++) {
        int item_type = items->type;

        if (item_type == RTE_FLOW_ITEM_TYPE_PORT_ID) {
            const struct rte_flow_item_port_id *spec = items->spec;

            /* Only recirc_id 0 (group_id == 0) may hold flows
             * from different source ports since it's the root table.
             * For every other recirc_id we have a table per port and
             * therefore we can skip matching on port id for those
             * tables.
             */
            if (attr->group > 0) {
                continue;
            }

            doca_spec->parser_meta.port_meta = spec->id;
            doca_mask->parser_meta.port_meta = 0xFFFFFFFF;
        } else if (item_type == RTE_FLOW_ITEM_TYPE_ETH) {
            const struct rte_flow_item_eth *spec = items->spec;
            const struct rte_flow_item_eth *mask = items->mask;

            if (!spec || !mask) {
                continue;
            }

            doca_copy_mask_field(eth.src_mac, src);
            doca_copy_mask_field(eth.dst_mac, dst);
            doca_copy_mask_field(eth.type, type);

            if (mask->has_vlan) {
                if (doca_hdr_spec == &doca_spec->outer) {
                    doca_spec->parser_meta.outer_l2_type =
                        spec->has_vlan
                        ? DOCA_FLOW_L2_META_SINGLE_VLAN
                        : DOCA_FLOW_L2_META_NO_VLAN;
                    doca_mask->parser_meta.outer_l2_type = UINT32_MAX;
                } else {
                    doca_spec->parser_meta.inner_l2_type =
                        spec->has_vlan
                        ? DOCA_FLOW_L2_META_SINGLE_VLAN
                        : DOCA_FLOW_L2_META_NO_VLAN;
                    doca_mask->parser_meta.inner_l2_type = UINT32_MAX;
                }
            }
        } else if (item_type == RTE_FLOW_ITEM_TYPE_VLAN) {
            const struct rte_flow_item_vlan *spec = items->spec;
            const struct rte_flow_item_vlan *mask = items->mask;

            /* HW supports match on one Ethertype, the Ethertype following the
             * last VLAN tag of the packet (see PRM). DOCA API has only that one.
             * Add a match on it as part of the doca eth header.
             */
            doca_copy_mask_field(eth.type, inner_type);
            doca_copy_mask_field(eth_vlan[0].tci, tci);
            doca_hdr_spec->l2_valid_headers = DOCA_FLOW_L2_VALID_HEADER_VLAN_0;
            doca_hdr_mask->l2_valid_headers = DOCA_FLOW_L2_VALID_HEADER_VLAN_0;
        /* L3 */
        } else if (item_type == RTE_FLOW_ITEM_TYPE_IPV4) {
            const struct rte_flow_item_ipv4 *spec = items->spec;
            const struct rte_flow_item_ipv4 *mask = items->mask;

            doca_hdr_spec->l3_type = DOCA_FLOW_L3_TYPE_IP4;
            doca_hdr_mask->l3_type = DOCA_FLOW_L3_TYPE_IP4;

            doca_copy_mask_field(ip4.next_proto, hdr.next_proto_id);
            doca_copy_mask_field(ip4.src_ip, hdr.src_addr);
            doca_copy_mask_field(ip4.dst_ip, hdr.dst_addr);
            doca_copy_mask_field(ip4.ttl, hdr.time_to_live);
            doca_copy_mask_field(ip4.dscp_ecn, hdr.type_of_service);

            /* DOCA API does not provide distinguishment between first/later
             * frags. Reject both.
             */
            if (items->last || (spec && spec->hdr.fragment_offset)) {
                return -1;
            }

            if (doca_hdr_spec == &doca_spec->outer) {
                doca_mask->parser_meta.outer_ip_fragmented = UINT8_MAX;
            } else {
                doca_mask->parser_meta.inner_ip_fragmented = UINT8_MAX;
                doca_mask->parser_meta.outer_ip_fragmented = 0;
            }
        } else if (item_type == RTE_FLOW_ITEM_TYPE_IPV6) {
            const struct rte_flow_item_ipv6 *spec = items->spec;
            const struct rte_flow_item_ipv6 *mask = items->mask;

            doca_hdr_spec->l3_type = DOCA_FLOW_L3_TYPE_IP6;
            doca_hdr_mask->l3_type = DOCA_FLOW_L3_TYPE_IP6;

            doca_copy_mask_field(ip6.src_ip, hdr.src_addr);
            doca_copy_mask_field(ip6.dst_ip, hdr.dst_addr);
            doca_copy_mask_field(ip6.next_proto, hdr.proto);
            doca_copy_mask_field(ip6.dscp_ecn, hdr.vtc_flow);
            doca_copy_mask_field(ip6.hop_limit, hdr.hop_limits);

            if (doca_hdr_spec == &doca_spec->outer) {
                doca_mask->parser_meta.outer_ip_fragmented = UINT8_MAX;
            } else {
                doca_mask->parser_meta.inner_ip_fragmented = UINT8_MAX;
                doca_mask->parser_meta.outer_ip_fragmented = 0;
            }
        /* L4 */
        } else if (item_type == RTE_FLOW_ITEM_TYPE_UDP) {
            const struct rte_flow_item_udp *spec = items->spec;
            const struct rte_flow_item_udp *mask = items->mask;

            doca_hdr_spec->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
            doca_hdr_mask->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

            doca_copy_mask_field(udp.l4_port.src_port, hdr.src_port);
            doca_copy_mask_field(udp.l4_port.dst_port, hdr.dst_port);
        } else if (item_type ==  RTE_FLOW_ITEM_TYPE_TCP) {
            const struct rte_flow_item_tcp *spec = items->spec;
            const struct rte_flow_item_tcp *mask = items->mask;

            doca_hdr_spec->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
            doca_hdr_mask->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;

            doca_copy_mask_field(tcp.l4_port.src_port, hdr.src_port);
            doca_copy_mask_field(tcp.l4_port.dst_port, hdr.dst_port);
            doca_copy_mask_field(tcp.flags, hdr.tcp_flags);
        } else if (item_type == RTE_FLOW_ITEM_TYPE_VXLAN) {
            doca_translate_vxlan_item(items, doca_spec, doca_mask);

            doca_hdr_spec = &doca_spec->inner;
            doca_hdr_mask = &doca_mask->inner;
        } else if (item_type == RTE_FLOW_ITEM_TYPE_GRE) {
            doca_translate_gre_item(items, doca_spec, doca_mask);

            doca_hdr_spec = &doca_spec->inner;
            doca_hdr_mask = &doca_mask->inner;
        } else if (item_type == RTE_FLOW_ITEM_TYPE_GRE_KEY) {
            doca_translate_gre_key_item(items, doca_spec, doca_mask);

            doca_hdr_spec = &doca_spec->inner;
            doca_hdr_mask = &doca_mask->inner;
        } else if (item_type == RTE_FLOW_ITEM_TYPE_GENEVE) {
            doca_translate_geneve_item(items, doca_spec, doca_mask);

            doca_hdr_spec = &doca_spec->inner;
            doca_hdr_mask = &doca_mask->inner;
        } else if (item_type == RTE_FLOW_ITEM_TYPE_GENEVE_OPT) {
            if (doca_init_geneve_opt_parser(netdev, items)) {
                return -1;
            }
            doca_translate_geneve_opt_item(items, doca_spec, doca_mask);
        } else if (item_type == RTE_FLOW_ITEM_TYPE_ICMP) {
            const struct rte_flow_item_icmp *spec = items->spec;
            const struct rte_flow_item_icmp *mask = items->mask;

            doca_hdr_spec->l4_type_ext  = DOCA_FLOW_L4_TYPE_EXT_ICMP;
            doca_hdr_mask->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_ICMP;

            doca_copy_mask_field(icmp.type, hdr.icmp_type);
            doca_copy_mask_field(icmp.code, hdr.icmp_code);
        } else if (item_type == RTE_FLOW_ITEM_TYPE_ICMP6) {
            const struct rte_flow_item_icmp6 *spec = items->spec;
            const struct rte_flow_item_icmp6 *mask = items->mask;

            doca_hdr_spec->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_ICMP6;
            doca_hdr_mask->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_ICMP6;

            doca_copy_mask_field(icmp.code, code);
            doca_copy_mask_field(icmp.type, type);
        } else if (item_type == RTE_FLOW_ITEM_TYPE_TAG) {
            const struct rte_flow_item_tag *spec = items->spec;
            const struct rte_flow_item_tag *mask = items->mask;

            if (spec && mask) {
                doca_spec->meta.u32[spec->index] |= spec->data & mask->data;
                doca_mask->meta.u32[spec->index] |= mask->data;
            }
        } else if (item_type == OVS_RTE_FLOW_ITEM_TYPE(FLOW_INFO)) {
            uint32_t reg_offset = reg_fields[REG_FIELD_FLOW_INFO].offset;
            uint32_t reg_mask = reg_fields[REG_FIELD_FLOW_INFO].mask;
            const struct rte_flow_item_mark *spec = items->spec;

            if (spec) {
                doca_spec->meta.pkt_meta |= (spec->id & reg_mask) << reg_offset;
                doca_mask->meta.pkt_meta |= reg_mask << reg_offset;
            }
        } else if (item_type == OVS_RTE_FLOW_ITEM_TYPE(HASH)) {
            struct reg_field *reg_field = &reg_fields[REG_FIELD_DP_HASH];
            const struct rte_flow_item_mark *hash_spec = items->spec;
            const struct rte_flow_item_mark *hash_mask = items->mask;

            /* Disable dp-hash offload if CT is configured. */
            if (doca_ct_offload_enabled) {
                return -1;
            }

            /* In case of non-IPv4, the first flow with the hash function is
             * not offloaded, so there is no point to offload this flow as it
             * will never be hit.
             */
            if (doca_mask->outer.l3_type != DOCA_FLOW_L3_TYPE_IP4 &&
                doca_mask->outer.l3_type != DOCA_FLOW_L3_TYPE_IP6) {
                return -1;
            }
            if (!hash_spec || !hash_mask || hash_mask->id & ~reg_field->mask) {
                /* Can't support larger mask. */
                return -1;
            }

            doca_spec->meta.pkt_meta |=
                (hash_spec->id & reg_field->mask) << reg_field->offset;
            doca_mask->meta.pkt_meta |=
                (hash_mask->id & reg_field->mask) << reg_field->offset;
        } else {
            VLOG_DBG_RL(&rl, "item %d is not supported", item_type);
            return -1;
        }
    }

    return 0;
}

static int
doca_translate_geneve_encap(const struct genevehdr *geneve,
                            struct doca_flow_actions *dacts)
{
    struct doca_flow_encap_action *encap = &dacts->encap;

    encap->tun.type = DOCA_FLOW_TUN_GENEVE;
    encap->tun.geneve.ver_opt_len = geneve->opt_len;
    encap->tun.geneve.ver_opt_len |= geneve->ver << 6;
    encap->tun.geneve.o_c = geneve->critical << 6;
    encap->tun.geneve.o_c |= geneve->oam << 7;
    encap->tun.geneve.next_proto = geneve->proto_type;
    encap->tun.geneve.vni = get_16aligned_be32(&geneve->vni);

    if (geneve->options[0].length) {
        encap->tun.geneve_options[0].class_id = geneve->options[0].opt_class;
        encap->tun.geneve_options[0].type = geneve->options[0].type;
        encap->tun.geneve_options[0].length = geneve->options[0].length;

        /* doca_flow represents the geneve option header as an array of a union
         * of 32 bits, the array's first element is the type/class/len and this
         * option's data starts from the next element in the array up to option_len
         */
        BUILD_ASSERT_DECL(sizeof(encap->tun.geneve_options[1].data) ==
                          sizeof(geneve->options[1]));
        memcpy(&encap->tun.geneve_options[1].data, &geneve->options[1],
               sizeof(encap->tun.geneve_options[1].data) * geneve->options[0].length);
    }

    dacts->has_encap = true;

    return 0;
}

static int
doca_translate_gre_encap(const struct gre_base_hdr *gre,
                         struct doca_flow_actions *dacts)
{
    struct doca_flow_encap_action *encap = &dacts->encap;
    const void *gre_key;

    encap->tun.protocol = gre->protocol;
    encap->tun.type = DOCA_FLOW_TUN_GRE;
    encap->tun.key_present = !!(gre->flags & htons(GRE_KEY));

    gre_key = gre + 1;
    if (encap->tun.key_present) {
        const uint32_t *key = gre_key;

        encap->tun.gre_key = *key;
    }

    dacts->has_encap = true;

    return 0;
}

static int
doca_translate_vxlan_encap(const struct vxlanhdr *vxlan,
                           struct doca_flow_actions *dacts)
{
    struct doca_flow_encap_action *encap = &dacts->encap;

    encap->tun.type = DOCA_FLOW_TUN_VXLAN;
    memcpy(&encap->tun.vxlan_tun_id, &vxlan->vx_vni, sizeof vxlan->vx_vni);

    dacts->has_encap = true;

    return 0;
}

static int
doca_translate_raw_encap(const struct rte_flow_action *action,
                         struct doca_flow_actions *dacts)
{
    struct doca_flow_header_format *outer = &dacts->encap.outer;
    const struct raw_encap_data *data = action->conf;
    struct ovs_16aligned_ip6_hdr *ip6;
    struct vlan_header *vlan;
    struct udp_header *udp;
    struct eth_header *eth;
    struct ip_header *ip;
    uint16_t proto;
    void *l4;

    /* L2 */
    eth = find_raw_encap_spec(data, RTE_FLOW_ITEM_TYPE_ETH);
    if (!eth) {
        return -1;
    }

    memcpy(&outer->eth.src_mac, &eth->eth_src, DOCA_ETHER_ADDR_LEN);
    memcpy(&outer->eth.dst_mac, &eth->eth_dst, DOCA_ETHER_ADDR_LEN);
    outer->eth.type = eth->eth_type;

    proto = eth->eth_type;
    if (proto == htons(ETH_TYPE_VLAN_8021Q)) {
        vlan = ALIGNED_CAST(struct vlan_header *, (uint8_t *) (eth + 1));
        outer->eth_vlan[0].tci = vlan->vlan_tci;
        outer->l2_valid_headers = DOCA_FLOW_L2_VALID_HEADER_VLAN_0;
        proto = vlan->vlan_next_type;
    }

    /* L3 */
    if (proto == htons(ETH_TYPE_IP)) {
        ip = find_raw_encap_spec(data, RTE_FLOW_ITEM_TYPE_IPV4);
        if (!ip) {
            return -1;
        }

        outer->l3_type = DOCA_FLOW_L3_TYPE_IP4;
        outer->ip4.src_ip = get_16aligned_be32(&ip->ip_src);
        outer->l3_type = DOCA_FLOW_L3_TYPE_IP4;
        outer->ip4.dst_ip = get_16aligned_be32(&ip->ip_dst);
        outer->ip4.ttl = ip->ip_ttl;
        outer->ip4.dscp_ecn = ip->ip_tos;
        l4 = ip + 1;
    } else if (proto == htons(ETH_TYPE_IPV6)) {
        ip6 = find_raw_encap_spec(data, RTE_FLOW_ITEM_TYPE_IPV6);
        if (!ip6) {
            return -1;
        }

        outer->l3_type = DOCA_FLOW_L3_TYPE_IP6;
        memcpy(&outer->ip6.src_ip, &ip6->ip6_src, sizeof ip6->ip6_src);
        outer->l3_type = DOCA_FLOW_L3_TYPE_IP6;
        memcpy(&outer->ip6.dst_ip, &ip6->ip6_dst, sizeof ip6->ip6_dst);
        outer->ip6.hop_limit = ip6->ip6_hlim;
        outer->ip6.dscp_ecn = ntohl(get_16aligned_be32(&ip6->ip6_flow)) >> 20;
        l4 = ip6 + 1;
    } else {
        return -1;
    }

    /* Tunnel */
    if (data->tnl_type == OVS_VPORT_TYPE_GRE
        || data->tnl_type == OVS_VPORT_TYPE_IP6GRE) {
        return doca_translate_gre_encap(l4, dacts);
    }

    udp = l4;
    /* L4 */
    outer->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
    outer->udp.l4_port.src_port = udp->udp_src;
    outer->udp.l4_port.dst_port = udp->udp_dst;

    if (data->tnl_type == OVS_VPORT_TYPE_GENEVE) {
        return doca_translate_geneve_encap((void *) (udp + 1), dacts);
    }

    if (data->tnl_type == OVS_VPORT_TYPE_VXLAN) {
        return doca_translate_vxlan_encap((void *) (udp + 1), dacts);
    }

    return -1;
}

static int
doca_hash_pipe_init(struct netdev *netdev,
                    unsigned int queue_id,
                    struct doca_hash_pipe_ctx *hash_pipe_ctx,
                    enum hash_pipe_type type)
{
    struct doca_flow_pipe *next_pipe = hash_pipe_ctx->post_hash_pipe_ctx->pipe;
    struct doca_flow_match hash_matches[NUM_HASH_PIPE_TYPE] = {
        [HASH_TYPE_IPV4_UDP] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP4,
            .outer.ip4.src_ip = UINT32_MAX,
            .outer.ip4.dst_ip = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP,
            .outer.udp.l4_port.src_port = UINT16_MAX,
            .outer.udp.l4_port.dst_port = UINT16_MAX,
        },
        [HASH_TYPE_IPV4_TCP] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP4,
            .outer.ip4.src_ip = UINT32_MAX,
            .outer.ip4.dst_ip = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
            .outer.tcp.l4_port.src_port = UINT16_MAX,
            .outer.tcp.l4_port.dst_port = UINT16_MAX,
        },
        [HASH_TYPE_IPV4_L3] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP4,
            .outer.ip4.src_ip = UINT32_MAX,
            .outer.ip4.dst_ip = UINT32_MAX,
        },
        [HASH_TYPE_IPV6_UDP_SUF] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.dst_ip[0] = UINT32_MAX,
            .outer.ip6.dst_ip[1] = UINT32_MAX,
            .outer.ip6.dst_ip[2] = UINT32_MAX,
            .outer.ip6.dst_ip[3] = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP,
            .outer.udp.l4_port.dst_port = UINT16_MAX,
        },
        [HASH_TYPE_IPV6_TCP_SUF] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.dst_ip[0] = UINT32_MAX,
            .outer.ip6.dst_ip[1] = UINT32_MAX,
            .outer.ip6.dst_ip[2] = UINT32_MAX,
            .outer.ip6.dst_ip[3] = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
            .outer.tcp.l4_port.dst_port = UINT16_MAX,
        },
        [HASH_TYPE_IPV6_L3_SUF] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.dst_ip[0] = UINT32_MAX,
            .outer.ip6.dst_ip[1] = UINT32_MAX,
            .outer.ip6.dst_ip[2] = UINT32_MAX,
            .outer.ip6.dst_ip[3] = UINT32_MAX,
        },
        [HASH_TYPE_IPV6_UDP_PRE] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.src_ip[0] = UINT32_MAX,
            .outer.ip6.src_ip[1] = UINT32_MAX,
            .outer.ip6.src_ip[2] = UINT32_MAX,
            .outer.ip6.src_ip[3] = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP,
            .outer.udp.l4_port.src_port = UINT16_MAX,
        },
        [HASH_TYPE_IPV6_TCP_PRE] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.src_ip[0] = UINT32_MAX,
            .outer.ip6.src_ip[1] = UINT32_MAX,
            .outer.ip6.src_ip[2] = UINT32_MAX,
            .outer.ip6.src_ip[3] = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
            .outer.tcp.l4_port.src_port = UINT16_MAX,
        },
        [HASH_TYPE_IPV6_L3_PRE] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.src_ip[0] = UINT32_MAX,
            .outer.ip6.src_ip[1] = UINT32_MAX,
            .outer.ip6.src_ip[2] = UINT32_MAX,
            .outer.ip6.src_ip[3] = UINT32_MAX,
        },
    };
    struct doca_flow_actions actions, *actions_arr[1];
    struct doca_flow_action_descs *descs_arr[1];
    struct doca_flow_pipe_entry **pentry;
    struct doca_flow_action_descs descs;
    struct doca_flow_action_desc desc;
    struct doca_eswitch_ctx *esw_ctx;
    struct doca_flow_fwd fwd, miss;
    struct doca_flow_pipe_cfg cfg;
    struct doca_flow_pipe **ppipe;
    char pipe_name[50];
    int ret;

    esw_ctx = doca_eswitch_ctx_get(netdev);

    ppipe = &hash_pipe_ctx->hashes[type].pipe;
    pentry = &hash_pipe_ctx->hashes[type].entry;

    snprintf(pipe_name, sizeof pipe_name, "OVS_HASH_PIPE_type_%u", type);

    memset(&cfg, 0, sizeof cfg);
    memset(&fwd, 0, sizeof(fwd));
    memset(&miss, 0, sizeof(miss));
    memset(&descs, 0, sizeof(descs));
    memset(&actions, 0, sizeof(actions));
    memset(&desc, 0, sizeof desc);

    cfg.attr.name = pipe_name;
    cfg.attr.type = DOCA_FLOW_PIPE_HASH;
    cfg.attr.enable_strict_matching = true;
    cfg.port = esw_ctx->esw_port;
    cfg.match_mask = &hash_matches[type];
    cfg.attr.nb_flows = 1;
    descs_arr[0] = &descs;
    cfg.action_descs = descs_arr;
    descs.desc_array = &desc;
    descs.nb_action_desc = 1;
    cfg.actions = actions_arr;
    cfg.attr.nb_actions = 1;
    actions_arr[0] = &actions;

    desc.type = DOCA_FLOW_ACTION_COPY;
    desc.field_op.src.field_string = "parser_meta.hash.result";
    desc.field_op.src.bit_offset = 0;
    desc.field_op.dst.field_string = "meta.data";
    desc.field_op.dst.bit_offset = doca_get_reg_bit_offset(REG_FIELD_DP_HASH);
    desc.field_op.width = doca_get_reg_width(REG_FIELD_DP_HASH);

    if (type >= HASH_TYPE_IPV6_UDP_PRE && type <= HASH_TYPE_IPV6_L3_PRE) {
        uint32_t offset = HASH_TYPE_IPV6_UDP_SUF - HASH_TYPE_IPV6_UDP_PRE;

        next_pipe = hash_pipe_ctx->hashes[type + offset].pipe;

        desc.field_op.dst.bit_offset =
            doca_get_reg_bit_offset(REG_FIELD_SCRATCH);
        desc.field_op.width = doca_get_reg_width(REG_FIELD_SCRATCH);
    }

    /* For SUF pipes, it's the hash result of the PRE pipes. For others it's
     * the seed.
     */
    doca_set_reg_mask(&hash_matches[type].meta, REG_FIELD_SCRATCH);

    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = next_pipe;
    miss.type = DOCA_FLOW_FWD_DROP;

    ret = doca_flow_pipe_create(&cfg, &fwd, &miss, ppipe);
    if (ret) {
        VLOG_ERR("Failed to create hash pipe: %d (%s)", ret,
                 doca_error_get_descr(ret));
        return ret;
    }

    ret = doca_flow_pipe_hash_add_entry(queue_id, *ppipe, 0, NULL, NULL, NULL,
                                        DOCA_FLOW_NO_WAIT, NULL, pentry);
    if (ret) {
        VLOG_ERR("Failed to create hash pipe entry. Error: %d (%s)", ret,
                 doca_error_get_descr(ret));
        return ret;
    }
    dpdk_offload_counter_inc(netdev);

    ret = doca_flow_entries_process(cfg.port, queue_id,
                                    ENTRY_PROCESS_TIMEOUT_MS, 0);
    if (ret) {
        VLOG_ERR("Failed to process hash pipe entry. Error: %d (%s)", ret,
                 doca_error_get_descr(ret));
        return ret;
    }

    return 0;
}

static struct doca_hash_pipe_ctx *
doca_hash_pipe_ctx_init(struct netdev *netdev)
{
    unsigned int queue_id = netdev_offload_thread_id();
    struct doca_hash_pipe_ctx *hash_pipe_ctx;
    struct doca_flow_pipe_entry **pentry;
    struct doca_eswitch_ctx *esw_ctx;
    struct doca_flow_pipe_cfg cfg;
    struct doca_flow_match spec;
    struct doca_flow_match mask;
    struct doca_flow_fwd miss;
    enum hash_nw_type nw_type;
    struct doca_flow_fwd fwd;
    char pipe_name[50];
    doca_error_t err;
    int type;

    esw_ctx = doca_eswitch_ctx_get(netdev);
    if (esw_ctx == NULL) {
        return NULL;
    }

    hash_pipe_ctx = xzalloc(sizeof *hash_pipe_ctx);
    hash_pipe_ctx->netdev = netdev;

    hash_pipe_ctx->post_hash_pipe_ctx =
        doca_ctl_pipe_ctx_ref(netdev, POSTHASH_TABLE_ID);
    if (hash_pipe_ctx->post_hash_pipe_ctx == NULL) {
        VLOG_ERR("%s: Failed to create post-hash", netdev_get_name(netdev));
        goto err;
    }

    for (type = 0; type < NUM_HASH_PIPE_TYPE; type++) {
        int ret;

        ret = doca_hash_pipe_init(netdev, queue_id, hash_pipe_ctx, type);
        if (ret) {
            VLOG_ERR("%s: Failed to create hash pipe ctx",
                     netdev_get_name(netdev));
            goto err;
        }
    }

    /* Classifier pipe. */
    memset(&cfg, 0, sizeof cfg);

    cfg.attr.name = pipe_name;
    cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
    cfg.attr.enable_strict_matching = true;
    cfg.port = esw_ctx->esw_port;
    cfg.attr.nb_flows = 2;
    cfg.match = &mask;
    cfg.match_mask = &mask;

    for (nw_type = 0; nw_type < NUM_HASH_NW; nw_type++) {
        struct doca_flow_pipe **ppipe;

        memset(&mask, 0, sizeof mask);
        memset(&spec, 0, sizeof spec);
        memset(&fwd, 0, sizeof fwd);
        fwd.type = DOCA_FLOW_FWD_PIPE;
        memset(&miss, 0, sizeof miss);
        miss.type = DOCA_FLOW_FWD_PIPE;

        if (nw_type == HASH_NW_IP4) {
            mask.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
            mask.outer.ip4.next_proto = 0xFF;
            miss.next_pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV4_L3].pipe;
            snprintf(pipe_name, sizeof pipe_name, "OVS_HASH_IPv4_CLASSIFIER_PIPE");
        } else {
            mask.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
            mask.outer.ip6.next_proto = 0xFF;
            miss.next_pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_L3_PRE].pipe;
            snprintf(pipe_name, sizeof pipe_name, "OVS_HASH_IPv6_CLASSIFIER_PIPE");
        }

        ppipe = &hash_pipe_ctx->classifier[nw_type].pipe;

        err = doca_flow_pipe_create(&cfg, &fwd, &miss, ppipe);
        if (err) {
            VLOG_ERR("%s: Failed to create hash classifier pipe: %d (%s)",
                     netdev_get_name(netdev), err, doca_error_get_descr(err));
            goto err;
        }

        /* TCP/UDP entries. */
        pentry = &hash_pipe_ctx->classifier[nw_type].tcpudp[HASH_TP_UDP];
        if (nw_type == HASH_NW_IP4) {
            spec.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
            spec.outer.ip4.next_proto = IPPROTO_UDP;
            fwd.next_pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV4_UDP].pipe;
        } else {
            spec.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
            spec.outer.ip6.next_proto = IPPROTO_UDP;
            fwd.next_pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_UDP_PRE].pipe;
        }
        err = doca_flow_pipe_add_entry(queue_id, *ppipe, &spec, NULL, NULL,
                                       &fwd, DOCA_FLOW_NO_WAIT, NULL, pentry);
        if (err) {
            VLOG_ERR("%s: Failed to create UDP classifier entry: %d (%s)",
                     netdev_get_name(netdev), err, doca_error_get_descr(err));
            goto err;
        }
        dpdk_offload_counter_inc(netdev);

        pentry = &hash_pipe_ctx->classifier[nw_type].tcpudp[HASH_TP_TCP];
        if (nw_type == HASH_NW_IP4) {
            spec.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
            spec.outer.ip4.next_proto = IPPROTO_TCP;
            fwd.next_pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV4_TCP].pipe;
        } else {
            spec.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
            spec.outer.ip6.next_proto = IPPROTO_TCP;
            fwd.next_pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_TCP_PRE].pipe;
        }
        err = doca_flow_pipe_add_entry(queue_id, *ppipe, &spec, NULL, NULL,
                                       &fwd, DOCA_FLOW_NO_WAIT, NULL, pentry);
        if (err) {
            VLOG_ERR("%s: Failed to create TCP classifier entry: %d (%s)",
                     netdev_get_name(netdev), err, doca_error_get_descr(err));
            goto err;
        }
        dpdk_offload_counter_inc(netdev);

        err = doca_flow_entries_process(esw_ctx->esw_port, queue_id,
                                        ENTRY_PROCESS_TIMEOUT_MS, 0);
        if (err) {
            VLOG_ERR("%s: Failed to poll classifier completion: queue %u. "
                     "Error: %d (%s)", netdev_get_name(netdev), queue_id, err,
                     doca_error_get_descr(err));
            goto err;
        }
    }

    return hash_pipe_ctx;
err:
    doca_hash_pipe_ctx_uninit(hash_pipe_ctx);
    return NULL;
}

static struct doca_flow_pipe *
get_ctl_pipe_root(struct doca_hash_pipe_ctx *hash_pipe_ctx,
                  struct doca_ctl_pipe_ctx *next_pipe_ctx,
                  struct doca_flow_match *spec,
                  struct doca_flow_actions *dacts,
                  bool has_dp_hash)
{
    if (!has_dp_hash) {
        return next_pipe_ctx->pipe;
    }

    if (dacts->has_encap) {
        if (dacts->encap.outer.l3_type == DOCA_FLOW_L3_TYPE_IP4) {
            if (dacts->encap.tun.type == DOCA_FLOW_TUN_VXLAN ||
                dacts->encap.tun.type == DOCA_FLOW_TUN_GENEVE) {
                return hash_pipe_ctx->hashes[HASH_TYPE_IPV4_UDP].pipe;
            }
            return hash_pipe_ctx->hashes[HASH_TYPE_IPV4_L3].pipe;
        }

        if (dacts->encap.outer.l3_type == DOCA_FLOW_L3_TYPE_IP6) {
            if (dacts->encap.tun.type == DOCA_FLOW_TUN_VXLAN ||
                dacts->encap.tun.type == DOCA_FLOW_TUN_GENEVE) {
                return hash_pipe_ctx->hashes[HASH_TYPE_IPV6_UDP_PRE].pipe;
            }
            return hash_pipe_ctx->hashes[HASH_TYPE_IPV6_L3_PRE].pipe;
        }
        return NULL;
    }

    if (spec->outer.l3_type == DOCA_FLOW_L3_TYPE_IP4) {
        if (spec->outer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_TCP) {
            return hash_pipe_ctx->hashes[HASH_TYPE_IPV4_TCP].pipe;
        } else if (spec->outer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_UDP) {
            return hash_pipe_ctx->hashes[HASH_TYPE_IPV4_UDP].pipe;
        }
        return hash_pipe_ctx->classifier[HASH_NW_IP4].pipe;
    }

    if (spec->outer.l3_type == DOCA_FLOW_L3_TYPE_IP6) {
        if (spec->outer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_TCP) {
            return hash_pipe_ctx->hashes[HASH_TYPE_IPV6_TCP_PRE].pipe;
        } else if (spec->outer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_UDP) {
            return hash_pipe_ctx->hashes[HASH_TYPE_IPV6_UDP_PRE].pipe;
        }
        return hash_pipe_ctx->classifier[HASH_NW_IP6].pipe;
    }

    return NULL;
}

static int
doca_translate_actions(struct netdev *netdev,
                       struct doca_flow_match *spec,
                       const struct rte_flow_action *actions,
                       struct doca_flow_actions *dacts,
                       struct doca_flow_actions *dacts_masks,
                       struct doca_flow_action_descs *dacts_descs,
                       struct doca_flow_fwd *fwd,
                       struct doca_flow_monitor *monitor,
                       struct doca_flow_handle_resources *flow_res,
                       struct doca_act_vars *dact_vars)
{
    struct doca_flow_header_format *outer_masks = &dacts_masks->outer;
    struct doca_flow_header_format *outer = &dacts->outer;
    struct doca_eswitch_ctx *esw_ctx;
    bool vlan_act_push = false;

    esw_ctx = doca_eswitch_ctx_get(netdev);
    for (; actions->type != RTE_FLOW_ACTION_TYPE_END; actions++) {
        int act_type = actions->type;

        if (act_type == RTE_FLOW_ACTION_TYPE_DROP) {
            fwd->type = DOCA_FLOW_FWD_DROP;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_MAC_SRC) {
            const struct action_set_data *asd = actions->conf;

            memcpy(&outer->eth.src_mac, asd->value, asd->size);
            memcpy(&outer_masks->eth.src_mac, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_MAC_DST) {
            const struct action_set_data *asd = actions->conf;

            memcpy(&outer->eth.dst_mac, asd->value, asd->size);
            memcpy(&outer_masks->eth.dst_mac, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_OF_SET_VLAN_VID) {
            const struct rte_flow_action_of_set_vlan_vid *rte_vlan_vid;

            rte_vlan_vid = actions->conf;
            /* If preceeded by vlan push action, this is a new
             * vlan tag. Otherwise, perfrom vlan modification.
             */
            if (vlan_act_push) {
                dacts->push.type = DOCA_FLOW_PUSH_ACTION_VLAN;
                dacts->push.vlan.tci = rte_vlan_vid->vlan_vid;
                dacts->has_push = true;
            } else {
                outer->eth_vlan[0].tci = rte_vlan_vid->vlan_vid;
                outer->l2_valid_headers = DOCA_FLOW_L2_VALID_HEADER_VLAN_0;
                memset(&outer_masks->eth_vlan[0].tci, 0xFF,
                       sizeof outer_masks->eth_vlan[0].tci);
                outer_masks->l2_valid_headers =
                    DOCA_FLOW_L2_VALID_HEADER_VLAN_0;
            }
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV4_SRC) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP4;
            memcpy(&outer->ip4.src_ip, asd->value, asd->size);
            memcpy(&outer_masks->ip4.src_ip, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV4_DST) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP4;
            memcpy(&outer->ip4.dst_ip, asd->value, asd->size);
            memcpy(&outer_masks->ip4.dst_ip, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV4_TTL) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP4;
            memcpy(&outer->ip4.ttl, asd->value, asd->size);
            memcpy(&outer_masks->ip4.ttl, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV4_DSCP) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP4;
            memcpy(&outer->ip4.dscp_ecn, asd->value, asd->size);
            memcpy(&outer_masks->ip4.dscp_ecn, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV6_HOP) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP6;
            memcpy(&outer->ip6.hop_limit, asd->value, asd->size);
            memcpy(&outer_masks->ip6.hop_limit, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV6_SRC) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP6;
            memcpy(&outer->ip6.src_ip, asd->value, asd->size);
            memcpy(&outer_masks->ip6.src_ip, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV6_DST) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP6;
            memcpy(&outer->ip6.dst_ip, asd->value, asd->size);
            memcpy(&outer_masks->ip6.dst_ip, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV6_DSCP) {
            const struct action_set_data *asd = actions->conf;

            outer->l3_type = DOCA_FLOW_L3_TYPE_IP6;
            memcpy(&outer->ip6.dscp_ecn, asd->value, asd->size);
            memcpy(&outer_masks->ip6.dscp_ecn, asd->mask, asd->size);
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_SRC)) {
            const struct action_set_data *asd = actions->conf;

            outer->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
            memcpy(&outer->udp.l4_port.src_port, asd->value, asd->size);
            memcpy(&outer_masks->udp.l4_port.src_port, asd->mask, asd->size);
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_DST)) {
            const struct action_set_data *asd = actions->conf;

            outer->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
            memcpy(&outer->udp.l4_port.dst_port, asd->value, asd->size);
            memcpy(&outer_masks->udp.l4_port.dst_port, asd->mask, asd->size);
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_SRC)) {
            const struct action_set_data *asd = actions->conf;

            outer->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
            memcpy(&outer->tcp.l4_port.src_port, asd->value, asd->size);
            memcpy(&outer_masks->tcp.l4_port.src_port, asd->mask, asd->size);
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_DST)) {
            const struct action_set_data *asd = actions->conf;

            outer->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
            memcpy(&outer->tcp.l4_port.dst_port, asd->value, asd->size);
            memcpy(&outer_masks->tcp.l4_port.dst_port, asd->mask, asd->size);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_PORT_ID) {
            const struct rte_flow_action_port_id *port_id = actions->conf;

            fwd->type = DOCA_FLOW_FWD_PORT;
            fwd->port_id = port_id->id;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_NVGRE_DECAP) {
            dacts->decap = true;
            dacts_descs->desc_array[0].type = DOCA_FLOW_ACTION_DECAP_ENCAP;
            dacts_descs->desc_array[0].decap_encap.is_l2 = true;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_VXLAN_DECAP)  {
            dacts->decap = true;
            dacts_descs->desc_array[0].type = DOCA_FLOW_ACTION_DECAP_ENCAP;
            dacts_descs->desc_array[0].decap_encap.is_l2 = true;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_RAW_DECAP) {
            dacts->decap = true;
            dacts_descs->desc_array[0].type = DOCA_FLOW_ACTION_DECAP_ENCAP;
            dacts_descs->desc_array[0].decap_encap.is_l2 = true;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_COUNT) {
            monitor->counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_JUMP) {
            const struct rte_flow_action_jump *jump = actions->conf;
            struct doca_ctl_pipe_ctx *next_pipe_ctx;

            next_pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, jump->group);
            if (!next_pipe_ctx) {
                return -1;
            }

            fwd->type = DOCA_FLOW_FWD_PIPE;
            fwd->next_pipe = get_ctl_pipe_root(esw_ctx->hash_pipe_ctx,
                                               next_pipe_ctx, spec, dacts,
                                               !!dact_vars->hash_data);
            if (!fwd->next_pipe) {
                return -1;
            }
            flow_res->next_pipe_ctx = next_pipe_ctx;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_RAW_ENCAP) {
            if (doca_translate_raw_encap(actions, dacts)) {
                return -1;
            }
            dacts_descs->desc_array[0].type = DOCA_FLOW_ACTION_DECAP_ENCAP;
            dacts_descs->desc_array[0].decap_encap.is_l2 = true;
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(FLOW_INFO)) {
            const struct rte_flow_action_mark *mark = actions->conf;

            doca_set_reg_val_mask(&dacts->meta, &dacts_masks->meta,
                                  REG_FIELD_FLOW_INFO, mark->id);
            dact_vars->flow_id = mark->id;
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE_CT_INFO) {
            const struct rte_flow_action_set_meta *set_meta = actions->conf;

            doca_set_reg_val(&dacts->meta, REG_FIELD_CT_CTX, set_meta->data);
            doca_set_reg_val(&dacts_masks->meta, REG_FIELD_CT_CTX,
                             set_meta->mask);
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_TAG) {
            const struct rte_flow_action_set_tag *set_tag = actions->conf;
            uint8_t index = set_tag->index;

            dacts->meta.u32[index] |= set_tag->data;
            dacts_masks->meta.u32[index] |= set_tag->mask;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_OF_POP_VLAN) {
            /* Current support is for a single VLAN tag */
            if (dacts->pop) {
                return -1;
            }
            dacts->pop = true;
            dacts_masks->pop = true;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_OF_PUSH_VLAN) {
            if (vlan_act_push) {
                return -1;
            }
            vlan_act_push = true;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_OF_SET_VLAN_PCP) {
            continue;
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(HASH)) {
            const struct hash_data *hash_data = actions->conf;

            /* Disable dp-hash offload if CT is configured. */
            if (doca_ct_offload_enabled) {
                return -1;
            }

            doca_set_reg_val_mask(&dacts->meta, &dacts_masks->meta,
                                  REG_FIELD_FLOW_INFO, hash_data->flow_id);
            doca_set_reg_val_mask(&dacts->meta, &dacts_masks->meta,
                                  REG_FIELD_SCRATCH, hash_data->seed);
            dact_vars->hash_data = hash_data;
        } else if (act_type == RTE_FLOW_ACTION_TYPE_METER) {
            const struct meter_data *mtr_data = actions->conf;
            uint32_t doca_mtr_id, flow_id;

            flow_id = shared_mtr_flow_id_alloc(esw_ctx, NULL);
            if (flow_id == -1) {
                return -1;
            }

            /* id is determine by both the upper layer id, and the esw_id. */
            doca_mtr_id = ovs_doca_meter_id(mtr_data->conf.mtr_id,
                                            esw_ctx->esw_id);

            if (monitor->meter_type == DOCA_FLOW_RESOURCE_TYPE_NONE) {
                /* first meter in a multi-meter action */
                monitor->meter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
                monitor->shared_meter.shared_meter_id = doca_mtr_id;
                dact_vars->mtr_flow_id = flow_id;
                dact_vars->flow_id = mtr_data->flow_id;
                doca_set_reg_val_mask(&dacts->meta, &dacts_masks->meta,
                                      REG_FIELD_FLOW_INFO, flow_id);
            } else {
                struct meter_info *mtr_info = xzalloc(sizeof *mtr_info);

                mtr_info->id = doca_mtr_id;
                mtr_info->flow_id = flow_id;
                ovs_list_push_back(&dact_vars->next_meters,
                                   &mtr_info->list_node);
            }
        } else if (act_type == RTE_FLOW_ACTION_TYPE_VOID) {
            continue;
        } else {
            return -1;
        }
    }

    return 0;
}

static void
dpdk_offload_doca_upkeep_queue(struct netdev *netdev, bool quiescing,
                               unsigned int qid)
{
    struct doca_ctl_pipe_ctx *resized_pipe_ctx;
    struct doca_eswitch_ctx *esw_ctx;
    unsigned int n_entries;
    bool pipe_resizing;
    doca_error_t err;

    if (netdev == NULL || !netdev_dpdk_is_ethdev(netdev)) {
        return;
    }

    esw_ctx = doca_eswitch_ctx_get(netdev);
    /* vports won't take an esw_ctx ref. */
    if (esw_ctx == NULL) {
        return;
    }

    pipe_resizing = ovs_refcount_read(&esw_ctx->pipe_resizing) > 1;
    n_entries = esw_ctx->async_state[qid].n_entries;
    if ((n_entries == 0 || (!quiescing && n_entries < OVS_DOCA_QUEUE_DEPTH))) {
        /* Unless a pipe is resizing and requires processing to be called until
         * finished, early bail-out if the queue has no entry or if it is not
         * full and we are not preparing for a long sleep. */
        if (!pipe_resizing) {
            return;
        }
    }

    /* Use 'max_processed_entries' == 0 to always attempt processing
     * the full length of the queue. */
    err = doca_flow_entries_process(esw_ctx->esw_port, qid,
                                    ENTRY_PROCESS_TIMEOUT_MS, 0);
    if (err) {
        VLOG_WARN_RL(&rl, "%s: Failed to process entries in queue %u. "
                     "Error: %d (%s)", netdev_get_name(netdev), qid,
                     err, doca_error_get_descr(err));
    }

    LIST_FOR_EACH_POP (resized_pipe_ctx, resize_ctx.resized_list_node,
                       &esw_ctx->resized_pipe_lists[qid]) {
        doca_ctl_pipe_ctx_resize_end(resized_pipe_ctx);
    }
}

static void
dpdk_offload_doca_upkeep(struct netdev *netdev, bool quiescing)
{
    dpdk_offload_doca_upkeep_queue(netdev, quiescing,
                                   netdev_offload_thread_id());
}

void
ovs_doca_entry_process_cb(struct doca_flow_pipe_entry *entry, uint16_t qid,
                          enum doca_flow_entry_status status,
                          enum doca_flow_entry_op op, void *aux)
{
    struct doca_eswitch_ctx *esw;
    struct doca_async_entry *dae;
    struct doca_flow_handle *dfh;
    struct netdev *netdev;

    if (aux == NULL) {
        /* 'aux' is NULL if the operation is synchronous. This is the
         * case for all control pipe changes, as well as CT if the user
         * requested it.
         * In this case, everything is handled in the calling function,
         * nothing to do. */
        return;
    }

    switch (op) {
    case DOCA_FLOW_ENTRY_OP_ADD:
        dae = aux;
        if (dae->doh == NULL) {
            /* Previous queue completion might have finished
             * before completing the whole queue due to timeout.
             * In that case, some 'dae' might have already been
             * processed and have their handle set to NULL.
             * Skip them. */
            return;
        }
        dfh = &dae->doh->dfh;
        netdev = dae->netdev;
        esw = doca_eswitch_ctx_from_async_entry(dae, qid);
        if (status == DOCA_FLOW_ENTRY_STATUS_SUCCESS) {
            dpdk_offload_counter_inc(netdev);
            dfh->flow = entry;
        } else if (status == DOCA_FLOW_ENTRY_STATUS_ERROR) {
            /* dfh->flow remains NULL. */
            COVERAGE_INC(doca_async_add_failed);
            VLOG_WARN_RL(&rl, "%s: Insertion failed for handle %p",
                         netdev_get_name(netdev), dfh);
        }
        dae->netdev = NULL;
        dae->doh = NULL;
        esw->async_state[qid].n_entries--;
        break;
    case DOCA_FLOW_ENTRY_OP_DEL:
        /* Deletion is always synchronous. */
        break;
    case DOCA_FLOW_ENTRY_OP_AGED:
    case DOCA_FLOW_ENTRY_OP_UPD:
        /* Not used by this implementation. */
        OVS_NOT_REACHED();
        break;
    }
}

void
ovs_doca_pipe_process_cb(struct doca_flow_pipe *pipe OVS_UNUSED,
                         enum doca_flow_pipe_status status OVS_UNUSED,
                         enum doca_flow_pipe_op op, void *user_ctx)
{
    struct doca_ctl_pipe_ctx *ctl_pipe_ctx = user_ctx;

    /* This case can be called for every pipe, even those
     * non-resizable. For them, 'user_ctx' will never have been
     * set. Ignore them. */
    if (user_ctx == NULL) {
        return;
    }

    switch (op) {
    case DOCA_FLOW_PIPE_OP_CONGESTION_REACHED:
        doca_ctl_pipe_ctx_resize_begin(ctl_pipe_ctx);
        break;
    case DOCA_FLOW_PIPE_OP_RESIZED:
        /* Register this context to finish its resizing outside of this callback. */
        doca_ctl_pipe_ctx_resize_end_defer(ctl_pipe_ctx);
        break;
    case DOCA_FLOW_PIPE_OP_DESTROYED:
    default:
        break;
    }
}


static struct doca_async_entry *
doca_async_entry_find(struct netdev *netdev,
                      struct doca_eswitch_ctx *esw,
                      unsigned int qid)
{
    struct doca_async_entry *dae = NULL;
    unsigned int *n_entries;

    n_entries = &esw->async_state[qid].n_entries;

    /* If the queue is currently full, do not try to
     * take a pointer to an entry. Trigger the linear scan,
     * and if really full, process it before attempting again. */
    if ((*n_entries) != OVS_DOCA_QUEUE_DEPTH) {
        dae = &esw->async_state[qid].entries[(*n_entries)];
    }

    /* The queue is not completed in any guaranteed order, meaning
     * that n_entries might not always point to a 'free' entry.
     * When it happens, linearly scan for an available descriptor. */
    if (dae == NULL || dae->doh != NULL) {
        unsigned int retry_count = 0;

        dae = NULL;
        while (dae == NULL) {
            int i;

            if (retry_count++ > 10) {
                COVERAGE_INC(doca_async_queue_blocked);
                return NULL;
            }
            for (i = 0; i < OVS_DOCA_QUEUE_DEPTH; i++) {
                if (esw->async_state[qid].entries[i].doh == NULL) {
                    dae = &esw->async_state[qid].entries[i];
                    break;
                }
            }
            if (i == OVS_DOCA_QUEUE_DEPTH) {
                COVERAGE_INC(doca_async_queue_full);
                if (netdev == NULL) {
                    /* We cannot hope to flush that netdev queue
                     * if it's NULL, report that we didn't find an entry. */
                    return NULL;
                }
                dpdk_offload_doca_upkeep_queue(netdev, true, qid);
            }
        }
    }

    (*n_entries)++;
    return dae;
}

static int
create_doca_basic_flow_entry(struct netdev *netdev,
                             unsigned int queue_id,
                             struct doca_flow_pipe *pipe,
                             struct doca_flow_match *spec,
                             struct doca_flow_actions *actions,
                             struct doca_flow_monitor *monitor,
                             struct doca_flow_fwd *fwd,
                             struct dpdk_offload_handle *doh,
                             struct rte_flow_error *error)
{
    enum doca_flow_flags_type doca_flags;
    struct doca_flow_pipe_entry *entry;
    struct doca_eswitch_ctx *esw_ctx;
    struct doca_async_entry *dae;
    doca_error_t err;

    doca_flags = DOCA_FLOW_NO_WAIT;
    dae = NULL;

    esw_ctx = doca_eswitch_ctx_get(netdev);
    if (ovs_doca_async) {
        dae = doca_async_entry_find(netdev, esw_ctx, queue_id);
        if (dae != NULL) {
            unsigned int n_entries;

            /* No reference is taken on the netdev.
             * When a netdev is removed from the datapath, a blocking
             * 'flush' command is issued. This command should take care
             * of emptying the offload queue, leaving no dangling netdev
             * reference before removing that specific port.
             */
            dae->netdev = netdev;
            dae->doh = doh;
            n_entries = esw_ctx->async_state[queue_id].n_entries;
            if (n_entries < OVS_DOCA_QUEUE_DEPTH) {
                doca_flags = DOCA_FLOW_WAIT_FOR_BATCH;
            }
        }
    }

    err = doca_flow_pipe_add_entry(queue_id, pipe, spec, actions, monitor, fwd,
                                   doca_flags, dae, &entry);
    if (err) {
        VLOG_WARN_RL(&rl, "%s: Failed to create basic pipe entry. Error: %d (%s)",
                     netdev_get_name(netdev), err, doca_error_get_descr(err));
        error->type = RTE_FLOW_ERROR_TYPE_HANDLE;
        error->message = doca_error_get_descr(err);
        return -1;
    }

    if (dae == NULL) {
        err = doca_flow_entries_process(esw_ctx->esw_port, queue_id,
                                        ENTRY_PROCESS_TIMEOUT_MS, 0);
        if (err) {
            VLOG_WARN_RL(&rl, "%s: Failed to poll completion of pipe queue %u."
                         " Error: %d (%s)", netdev_get_name(netdev), queue_id,
                         err, doca_error_get_descr(err));
            error->type = RTE_FLOW_ERROR_TYPE_HANDLE;
            error->message = doca_error_get_descr(err);
            return -1;
        }
        dpdk_offload_counter_inc(netdev);
        doh->dfh.flow = entry;
    }

    return 0;
}

static struct doca_flow_pipe_entry *
create_doca_ctl_flow_entry(struct netdev *netdev,
                           unsigned int queue_id,
                           struct doca_ctl_pipe_ctx *self_pipe_ctx,
                           uint32_t prio,
                           struct doca_flow_match *spec,
                           struct doca_flow_match *mask,
                           struct doca_flow_actions *actions,
                           struct doca_flow_actions *actions_masks,
                           struct doca_flow_action_descs *dacts_descs,
                           struct doca_flow_monitor *monitor,
                           struct doca_flow_fwd *fwd,
                           struct rte_flow_error *error)
{
    struct doca_flow_pipe *pipe = self_pipe_ctx->pipe;
    struct doca_flow_pipe_entry *entry;
    int upkeep_retries = 100;
    doca_error_t err;

    while (doca_ctl_pipe_resizing(self_pipe_ctx) && upkeep_retries-- > 0) {
        dpdk_offload_doca_upkeep_queue(netdev, false, queue_id);
    }
    if (!upkeep_retries) {
        VLOG_WARN("%s: Exhausted attempts to complete resize of pipe 0x%08x "
                  "before rule insertion", netdev_get_name(netdev),
                  self_pipe_ctx->resize_ctx.group_id);
    }

    err = doca_flow_pipe_control_add_entry(queue_id, prio, pipe, spec, mask,
                                           NULL, actions, actions_masks,
                                           dacts_descs, monitor, fwd, NULL,
                                           &entry);
    if (err) {
        VLOG_WARN_RL(&rl, "%s: Failed to create ctl pipe entry. Error: %d (%s)",
                     netdev_get_name(netdev), err, doca_error_get_descr(err));
        error->type = (enum rte_flow_error_type) err;
        error->message = doca_error_get_descr(err);
        return NULL;
    }

    dpdk_offload_counter_inc(netdev);
    ovs_assert(entry);

    return entry;
}

static struct doca_flow_pipe *
doca_get_ct_pipe(struct doca_eswitch_ctx *ctx,
                 struct doca_flow_match *spec)
{
    enum ct_nw_type nw_type;
    enum ct_tp_type tp_type;

    if (ctx == NULL) {
        return NULL;
    }

    nw_type = l3_to_nw_type(spec->outer.l3_type);
    if (nw_type >= NUM_CT_NW) {
        VLOG_DBG_RL(&rl, "Unsupported CT network type.");
        return NULL;
    }

    if (nw_type == CT_NW_IP6) {
        return ctx->ct_ip6_prefix.pipe;
    }

    tp_type = l4_to_tp_type(spec->outer.l4_type_ext);
    if (tp_type >= NUM_CT_TP) {
        VLOG_DBG_RL(&rl, "Unsupported CT protocol type.");
        return NULL;
    }

    return ctx->ct_pipes[nw_type][tp_type].pipe;
}

static uint32_t
split_prefix_id_alloc(void)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t prefix_id;

    if (id_fpool_new_id(split_prefix_id_pool, tid, &prefix_id)) {
        return prefix_id;
    }
    return 0;
}

static void
split_prefix_id_free(uint32_t prefix_id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(split_prefix_id_pool, tid, prefix_id);
}

static int
doca_split_prefix_ctx_init(void *ctx_, void *arg_, uint32_t id)
{
    struct doca_flow_actions dacts, dacts_mask;
    struct doca_split_prefix_ctx *ctx = ctx_;
    struct doca_split_prefix_arg *arg = arg_;
    struct doca_flow_handle *hndl;
    struct rte_flow_error error;
    struct doca_flow_fwd fwd;
    unsigned int queue_id;
    doca_error_t derr;

    memset(&dacts, 0, sizeof dacts);
    memset(&dacts_mask, 0, sizeof dacts_mask);
    memset(&fwd, 0, sizeof fwd);

    doca_set_reg_val_mask(&dacts.meta, &dacts_mask.meta,
                          REG_FIELD_SCRATCH, id);
    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = arg->suffix_pipe;
    queue_id = netdev_offload_thread_id();

    if (arg->prefix_pipe_type == DOCA_FLOW_PIPE_BASIC) {
        if (create_doca_basic_flow_entry(arg->netdev, queue_id,
                                         arg->prefix_pipe, arg->spec, &dacts,
                                         NULL, &fwd, &ctx->doh, &error)) {
            VLOG_WARN_RL(&rl, "%s: Failed to create basic split prefix entry:"
                         " Error %d (%s)", netdev_get_name(arg->netdev),
                         error.type, error.message);
            return -1;
        }
    } else {
        if (arg->set_flow_info_id) {
            /* Set flow info id to pkt meta.
             * If packet misses on suffix table the flow info
             * is already stored for proper recovery.
             */
            doca_set_reg_val_mask(&dacts.meta, &dacts_mask.meta,
                                  REG_FIELD_FLOW_INFO, arg->set_flow_info_id);
        }
        hndl = &ctx->doh.dfh;
        derr = doca_flow_pipe_control_add_entry(queue_id, arg->prio,
                                                arg->prefix_pipe, arg->spec,
                                                arg->mask, NULL, &dacts,
                                                &dacts_mask, NULL, NULL,
                                                &fwd, NULL, &hndl->flow);
        if (derr) {
            VLOG_WARN_RL(&rl, "%s: Failed to create ctl split prefix entry: "
                         "Error %d (%s)", netdev_get_name(arg->netdev), derr,
                         doca_error_get_descr(derr));
            return -1;
        }

        dpdk_offload_counter_inc(arg->netdev);
    }

    ctx->netdev = arg->netdev;

    return 0;
}

static void
doca_split_prefix_ctx_uninit(void *ctx_)
{
    unsigned int queue_id = netdev_offload_thread_id();
    struct doca_split_prefix_ctx *ctx = ctx_;
    struct rte_flow_error error;

    destroy_dpdk_offload_handle(ctx->netdev, &ctx->doh, queue_id, &error);
}

static struct ds *
dump_split_prefix_id(struct ds *s, void *key_, void *ctx_, void *arg OVS_UNUSED)
{
    struct doca_split_prefix_key *key = key_;
    struct doca_split_prefix_ctx *ctx = ctx_;
    struct doca_flow_handle *hndl;

    if (ctx) {
        hndl = &ctx->doh.dfh;
        ds_put_format(s, "prefix_flow=%p, ", hndl->flow);
    }
    ds_put_format(s, "prefix_pipe=%p, ", key->prefix_pipe);

    return s;
}

static void
split_prefix_id_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = split_prefix_id_alloc,
            .id_free = split_prefix_id_free,
            .priv_size = sizeof(struct doca_split_prefix_ctx),
            .priv_init = doca_split_prefix_ctx_init,
            .priv_uninit = doca_split_prefix_ctx_uninit,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();
        size_t data_size = sizeof(struct doca_split_prefix_key);

        split_prefix_id_pool = id_fpool_create(nb_thread, MIN_SPLIT_PREFIX_ID,
                                               NUM_SPLIT_PREFIX_ID);
        split_prefix_md = offload_metadata_create(nb_thread, "prefix_id",
                                                  data_size,
                                                  dump_split_prefix_id,
                                                  params);

        ovsthread_once_done(&init_once);
    }
}

static struct doca_split_prefix_ctx *
split_prefix_ctx_ref(struct doca_split_prefix_key *key,
                     struct doca_split_prefix_arg *args,
                     uint32_t *prefix_id)
{
    split_prefix_id_init();
    return offload_metadata_priv_get(split_prefix_md, key, args, prefix_id,
                                     true);
}

static void
split_prefix_ctx_unref(struct doca_split_prefix_ctx *ctx)
{
    offload_metadata_priv_unref(split_prefix_md, netdev_offload_thread_id(),
                                ctx);
}

static void
split_match_header_fields(struct doca_flow_header_format *spec_outer_,
                          struct doca_flow_header_format *mask_outer_,
                          struct doca_flow_header_format *pspec_outer_,
                          struct doca_flow_header_format *pmask_outer_,
                          struct doca_flow_header_format *smask_outer_,
                          enum split_field_type avoid_type,
                          size_t *match_bytes)
{
    char *pspec_outer, *pmask_outer;
    char *spec_outer, *mask_outer;
    enum split_field_layer layer;
    struct split_field *field;
    int i, proto, num_fields;
    char *smask_outer;

    spec_outer = (void *) spec_outer_;
    mask_outer = (void *) mask_outer_;
    pspec_outer = (void *) pspec_outer_;
    pmask_outer = (void *) pmask_outer_;
    smask_outer = (void *) smask_outer_;

    for (layer = L2_HEADERS;
         layer <= L4_HEADERS && *match_bytes < MAX_FIELD_BYTES; layer++) {
        proto = layer == L4_HEADERS ? spec_outer_->l4_type_ext :
                layer == L3_HEADERS ? spec_outer_->l3_type :
                                      0;
        num_fields = layer == L4_HEADERS ? NUM_L4_FIELDS :
                     layer == L3_HEADERS ? NUM_L3_FIELDS :
                                           NUM_L2_FIELDS;

        for (i = 0; i < num_fields && *match_bytes < MAX_FIELD_BYTES; i++) {
            field = &split_fields[layer][i];
            if (field->type != avoid_type &&
                field->proto_type == proto &&
                !is_all_zeros(mask_outer + field->offset, field->size) &&
                *match_bytes + field->size <= MAX_FIELD_BYTES) {
                memcpy(pspec_outer + field->offset, spec_outer + field->offset,
                       field->size);
                memcpy(pmask_outer + field->offset, mask_outer + field->offset,
                       field->size);
                memset(smask_outer + field->offset, 0, field->size);
                *match_bytes += field->size;
            }
        }
    }
}

static int
split_tunnel_header_fields(struct doca_flow_match *spec,
                           struct doca_flow_match *mask,
                           struct doca_flow_match *pspec,
                           struct doca_flow_match *pmask,
                           struct doca_flow_match *smask,
                           size_t *match_bytes)
{
    /* Assume the tunnel match will fit into the prefix header */
    if (spec->tun.type == DOCA_FLOW_TUN_GENEVE) {
        pspec->tun.type = DOCA_FLOW_TUN_GENEVE;
        pmask->tun.type = DOCA_FLOW_TUN_GENEVE;
        pspec->tun.geneve.vni = spec->tun.geneve.vni;
        pmask->tun.geneve.vni = mask->tun.geneve.vni;
        smask->tun.type = DOCA_FLOW_TUN_NONE;
        smask->tun.geneve.vni = 0;
        *match_bytes += sizeof spec->tun.geneve.vni;
        memcpy(pspec->tun.geneve_options, spec->tun.geneve_options,
               sizeof pspec->tun.geneve_options);
        memcpy(pmask->tun.geneve_options, mask->tun.geneve_options,
               sizeof pmask->tun.geneve_options);
        memset(smask->tun.geneve_options, 0, sizeof smask->tun.geneve_options);
        *match_bytes += spec->tun.geneve_options->length * 4;
    } else if (spec->tun.type == DOCA_FLOW_TUN_VXLAN) {
        pspec->tun.type = DOCA_FLOW_TUN_VXLAN;
        pmask->tun.type = DOCA_FLOW_TUN_VXLAN;
        pspec->tun.vxlan_tun_id = spec->tun.vxlan_tun_id;
        pmask->tun.vxlan_tun_id = mask->tun.vxlan_tun_id;
        smask->tun.type = DOCA_FLOW_TUN_NONE;
        smask->tun.vxlan_tun_id = 0;
        *match_bytes += sizeof spec->tun.vxlan_tun_id;
    } else if (spec->tun.type == DOCA_FLOW_TUN_GRE) {
        pspec->tun.type = DOCA_FLOW_TUN_GRE;
        pmask->tun.type = DOCA_FLOW_TUN_GRE;
        if (mask->tun.gre_key) {
            pspec->tun.gre_key = spec->tun.gre_key;
            pmask->tun.gre_key = mask->tun.gre_key;
            smask->tun.type = DOCA_FLOW_TUN_NONE;
            smask->tun.gre_key = 0;
            *match_bytes += sizeof spec->tun.gre_key;
        }
    } else {
        return -1;
    }

    return 0;
}

static int
split_doca_flow_match(struct netdev *netdev,
                      struct doca_flow_match *spec,
                      struct doca_flow_match *mask,
                      struct doca_flow_match *pspec,
                      struct doca_flow_match *pmask,
                      struct doca_flow_match *sspec,
                      struct doca_flow_match *smask,
                      uint8_t depth)
{
    enum split_field_type avoid_type;
    bool has_tunnel, is_n2h;
    size_t match_bytes = 0;
    int i;

    memset(pspec, 0, sizeof *pspec);
    memset(pmask, 0, sizeof *pmask);

    has_tunnel = spec->tun.type != DOCA_FLOW_TUN_NONE;
    is_n2h = netdev_dpdk_is_esw_mgr(netdev);

    memcpy(sspec, spec, sizeof *spec);
    memcpy(smask, mask, sizeof *mask);

    /* Collect meta matches to prefix first and assume we
     * always have enough room in prefix for all meta.
     * Port meta match always exists and takes 4 bytes and
     * we set it for both prefix and suffix to utilize the
     * doca flow direction optimization and reduce number
     * of flows.
     */
    pspec->parser_meta.port_meta = spec->parser_meta.port_meta;
    pmask->parser_meta.port_meta = mask->parser_meta.port_meta;
    match_bytes += sizeof spec->parser_meta.port_meta;

    if (mask->meta.pkt_meta) {
        pspec->meta.pkt_meta = spec->meta.pkt_meta;
        pmask->meta.pkt_meta = mask->meta.pkt_meta;
        smask->meta.pkt_meta = 0;
        match_bytes += sizeof spec->meta.pkt_meta;
    }

    for (i = 0; i < ARRAY_SIZE(spec->meta.u32); i++) {
        if (mask->meta.u32[i]) {
            pspec->meta.u32[i] = spec->meta.u32[i];
            pmask->meta.u32[i] = mask->meta.u32[i];
            smask->meta.u32[i] = 0;
            match_bytes += sizeof spec->meta.u32[i];
        }
    }

    pspec->outer.l2_valid_headers = spec->outer.l2_valid_headers;
    pmask->outer.l2_valid_headers = mask->outer.l2_valid_headers;
    pspec->outer.l3_type = spec->outer.l3_type;
    pmask->outer.l3_type = mask->outer.l3_type;
    pspec->outer.l4_type_ext = spec->outer.l4_type_ext;
    pmask->outer.l4_type_ext = mask->outer.l4_type_ext;

    if (has_tunnel) {
        if (split_tunnel_header_fields(spec, mask, pspec, pmask,
                                       smask, &match_bytes)) {
            return -1;
        }

        /* Tunnel outer header is common among flows so collect
         * them into the prefix flow.
         */
        split_match_header_fields(&spec->outer, &mask->outer, &pspec->outer,
                                  &pmask->outer, &smask->outer,
                                  FIELD_TYPE_INVALID, &match_bytes);
        if (depth) {
            pspec->inner.l3_type = spec->inner.l3_type;
            pmask->inner.l3_type = mask->inner.l3_type;
            pspec->inner.l4_type_ext = spec->inner.l4_type_ext;
            pmask->inner.l4_type_ext = mask->inner.l4_type_ext;
            split_match_header_fields(&spec->inner, &mask->inner, &pspec->inner,
                                      &pmask->inner, &smask->inner,
                                      FIELD_TYPE_SRC, &match_bytes);
        }
    } else {
        /* In case there's no tunnel, collect fields to prefix based
         * on the flow's direction.
         * is_n2h will indicate the flow is network 2 host flow and
         * therefore, dst addresses are more likely to be common among
         * these flows and we have a better chance to create a prefix
         * flow that will be used by multiple n2h split flows.
         * The other option is that the flow is host 2 network
         * and in such case, the src addresses are taken into the prefix flow
         * as they are the common addresses among these flows.
         */
        avoid_type = is_n2h ? FIELD_TYPE_SRC : FIELD_TYPE_DST;
        split_match_header_fields(&spec->outer, &mask->outer, &pspec->outer,
                                  &pmask->outer, &smask->outer, avoid_type,
                                  &match_bytes);
    }

    return 0;
}

static struct doca_flow_pipe_entry *
create_split_doca_flow_entry(struct netdev *netdev,
                             unsigned int queue_id,
                             uint32_t prio,
                             struct doca_ctl_pipe_ctx *pipe,
                             struct doca_flow_match *spec,
                             struct doca_flow_match *mask,
                             struct doca_flow_actions *actions,
                             struct doca_flow_actions *actions_masks,
                             struct doca_flow_action_descs *dacts_descs,
                             struct doca_flow_monitor *monitor,
                             struct doca_flow_fwd *fwd,
                             struct doca_flow_handle_resources *flow_res,
                             uint8_t depth,
                             struct rte_flow_error *error)
{
    struct doca_split_prefix_key prefix_key;
    struct doca_ctl_pipe_ctx *suffix_pipe;
    struct doca_split_prefix_ctx *pctx;
    struct doca_flow_match suffix_spec;
    struct doca_flow_match suffix_mask;
    struct doca_flow_pipe_entry *entry;
    struct doca_split_prefix_arg args;
    uint32_t prefix_id;

    if (unlikely(depth == MAX_SPLIT_DEPTH)) {
        VLOG_DBG_RL(&rl, "Exceeded max split depth %d", MAX_SPLIT_DEPTH);
        return NULL;
    }

    memset(&suffix_spec, 0, sizeof(suffix_spec));
    memset(&suffix_mask, 0, sizeof(suffix_mask));
    /* Split the original match to prefix and suffix keys */
    if (split_doca_flow_match(netdev, spec, mask,
                              &prefix_key.spec, &prefix_key.mask,
                              &suffix_spec, &suffix_mask, depth)) {
        error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
        error->message = "Could not split flow";
        return NULL;
    }

    suffix_pipe = doca_ctl_pipe_ctx_ref(netdev, SPLIT_DEPTH_TABLE_ID(depth));
    if (!suffix_pipe) {
        error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
        error->message = "Could not get suffix table for split flow";
        return NULL;
    }

    prefix_key.prefix_pipe = pipe->pipe;
    args.netdev = netdev;
    args.spec = &prefix_key.spec;
    args.mask = &prefix_key.mask;
    args.prefix_pipe_type = DOCA_FLOW_PIPE_CONTROL;
    args.prefix_pipe = pipe->pipe;
    args.suffix_pipe = suffix_pipe->pipe;
    args.set_flow_info_id = pipe->miss_flow_id;
    args.prio = prio;

    /* Get prefix flow id. */
    pctx = split_prefix_ctx_ref(&prefix_key, &args, &prefix_id);
    if (!pctx) {
        error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
        error->message = "Could not get split flow prefix id";
        goto err_prefix;
    }

    if (depth > 0) {
        /* If this is not the first split attempt, suffix_spec and mask are
         * copied from the previous split suffix which included a match on the
         * previous prefix_id, mask that match before setting a match on
         * the current prefix_id
         */
        memset(&suffix_spec.meta, 0, sizeof(suffix_spec.meta));
    }

    /* Add prefix ID match to suffix rule. */
    doca_set_reg_val_mask(&suffix_spec.meta, &suffix_mask.meta,
                          REG_FIELD_SCRATCH, prefix_id);

    /* Insert suffix rule. */
    entry = create_doca_ctl_flow_entry(netdev, queue_id, suffix_pipe, prio,
                                       &suffix_spec, &suffix_mask, actions,
                                       actions_masks, dacts_descs, monitor,
                                       fwd, error);
    if (!entry) {
        if (depth != MAX_SPLIT_DEPTH &&
            (doca_error_t) error->type == DOCA_ERROR_TOO_BIG) {
            VLOG_DBG_RL(&rl, "%s: Split attempt %d flow entry is too big to "
                             "insert directly. Attempt another flow split.",
                        netdev_get_name(netdev), depth);
            entry = create_split_doca_flow_entry(netdev, queue_id, prio,
                                                 suffix_pipe, &suffix_spec,
                                                 &suffix_mask, actions,
                                                 actions_masks, dacts_descs,
                                                 monitor, fwd, flow_res,
                                                 depth + 1, error);

            if (entry) {
                goto out;
            }
        }
        error->type = RTE_FLOW_ERROR_TYPE_HANDLE;
        error->message = "Could not insert split flow suffix rule";
        goto err_suffix;
    }

out:
    flow_res->split_ctx[depth].curr_split_ctx = pctx;
    flow_res->split_ctx[depth].next_split_pipe_ctx = suffix_pipe;

    error->type = (enum rte_flow_error_type) DOCA_SUCCESS;
    return entry;

err_suffix:
    split_prefix_ctx_unref(pctx);
err_prefix:
    doca_ctl_pipe_ctx_unref(suffix_pipe);

    return NULL;
}

static struct doca_flow_handle *
create_doca_flow_handle(struct netdev *netdev,
                        unsigned int queue_id,
                        uint32_t prio,
                        uint32_t group,
                        uint32_t miss_flow_id,
                        struct doca_flow_match *spec,
                        struct doca_flow_match *mask,
                        struct doca_flow_actions *actions,
                        struct doca_flow_actions *actions_masks,
                        struct doca_flow_action_descs *dacts_descs,
                        struct doca_flow_monitor *monitor,
                        struct doca_flow_fwd *fwd,
                        struct doca_flow_handle_resources *flow_res,
                        struct dpdk_offload_handle *doh,
                        struct rte_flow_error *error)
{
    struct doca_ctl_pipe_ctx *pipe_ctx = NULL;
    struct doca_flow_handle *hndl;

    hndl = &doh->dfh;

    /* get self table pointer */
    pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, group);
    if (!pipe_ctx) {
        error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
        error->message = "Could not create table";
        goto err_pipe;
    }
    /* insert rule */
    if (!spec || spec->tun.type == DOCA_FLOW_TUN_NONE) {
        hndl->flow = create_doca_ctl_flow_entry(netdev, queue_id, pipe_ctx,
                                                prio, spec, mask, actions,
                                                actions_masks, dacts_descs,
                                                monitor, fwd, error);
    } else {
        hndl->flow = NULL;
        error->type = (enum rte_flow_error_type) DOCA_ERROR_TOO_BIG;
    }
    if (!hndl->flow) {
        if ((doca_error_t) error->type == DOCA_ERROR_TOO_BIG) {
            VLOG_DBG_RL(&rl, "%s: Flow entry is too big to insert directly. "
                        "Attempt flow split.",
                        netdev_get_name(netdev));
            hndl->flow = create_split_doca_flow_entry(netdev, queue_id, prio,
                                                      pipe_ctx, spec, mask,
                                                      actions, actions_masks,
                                                      dacts_descs, monitor,
                                                      fwd, flow_res, 0, error);
        } else {
            error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
            error->message = "Could not create offload rule";
        }
    }

    if (!hndl->flow) {
        goto err_insert;
    }

    memcpy(&hndl->flow_res, flow_res, sizeof *flow_res);
    hndl->flow_res.self_pipe_ctx = pipe_ctx;
    /* miss_flow_id will be non 0 only for pipe miss flows
     * and should be set once per pipe when the miss flow
     * is created. To avoid overwriting it when regular flows
     * are inserted to the pipe, we perform the check for
     * non 0 value.
     */
    if (miss_flow_id) {
        pipe_ctx->miss_flow_id = miss_flow_id;
    }

    return hndl;

err_insert:
    doca_ctl_pipe_ctx_unref(pipe_ctx);
err_pipe:
    return NULL;
}

static struct doca_flow_pipe_entry *
add_doca_post_meter_red_entry(struct netdev *netdev,
                              unsigned int queue_id,
                              uint32_t shared_meter_id,
                              uint32_t flow_id,
                              struct rte_flow_error *error)
{
    struct doca_ctl_pipe_ctx *post_meter_pipe_ctx;
    struct doca_flow_pipe_entry *entry;
    struct doca_eswitch_ctx *esw_ctx;
    struct doca_flow_monitor monitor;
    struct doca_flow_match red_match;
    struct doca_flow_match red_mask;
    struct doca_flow_fwd fwd;

    esw_ctx = doca_eswitch_ctx_get(netdev);
    post_meter_pipe_ctx = esw_ctx->post_meter_pipe_ctx;

    memset(&red_match, 0, sizeof(red_match));
    memset(&red_mask, 0, sizeof(red_mask));
    memset(&monitor, 0, sizeof(monitor));
    memset(&fwd, 0, sizeof fwd);

    fwd.type = DOCA_FLOW_FWD_DROP;
    monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
    monitor.shared_counter.shared_counter_id =
        ovs_doca_get_post_meter_counter_id(shared_meter_id,
                                           DOCA_FLOW_METER_COLOR_RED);

    doca_set_reg_val_mask(&red_match.meta, &red_mask.meta,
                          REG_FIELD_FLOW_INFO, flow_id);

    /* Insert red rule with low priority, lower than the corresponding
     * green rule, optimizing for the case when traffic stays below metered
     * rate.
     */
    entry = create_doca_ctl_flow_entry(netdev, queue_id, post_meter_pipe_ctx,
                                       DPDK_OFFLOAD_PRIORITY_MED, &red_match,
                                       &red_mask, NULL, NULL, NULL, &monitor,
                                       &fwd, error);
    if (!entry) {
        VLOG_ERR_RL(&rl,
                    "%s: Failed to create shared meter red rule for flow ID %u",
                    netdev_get_name(netdev), flow_id);
        return NULL;
    }

    return entry;
}

static struct doca_flow_pipe_entry *
add_doca_post_meter_green_entry(struct netdev *netdev,
                                unsigned int queue_id,
                                uint32_t shared_meter_id,
                                uint32_t flow_id,
                                uint32_t restore_flow_id,
                                struct meter_info *next_meter,
                                struct doca_flow_fwd *orig_fwd,
                                struct rte_flow_error *error)
{
    struct doca_ctl_pipe_ctx *post_meter_pipe_ctx;
    struct doca_flow_actions dacts, dacts_masks;
    struct doca_flow_fwd *fwd = orig_fwd;
    struct doca_flow_pipe_entry *entry;
    struct doca_flow_match green_match;
    struct doca_flow_match green_mask;
    struct doca_flow_fwd next_mtr_fwd;
    struct doca_eswitch_ctx *esw_ctx;
    struct doca_flow_monitor monitor;
    uint32_t fid = restore_flow_id;

    esw_ctx = doca_eswitch_ctx_get(netdev);
    post_meter_pipe_ctx = esw_ctx->post_meter_pipe_ctx;

    memset(&next_mtr_fwd, 0, sizeof next_mtr_fwd);
    memset(&green_match, 0, sizeof(green_match));
    memset(&dacts_masks, 0, sizeof dacts_masks);
    memset(&green_mask, 0, sizeof(green_mask));
    memset(&monitor, 0, sizeof(monitor));
    memset(&dacts, 0, sizeof dacts);

    monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
    monitor.shared_counter.shared_counter_id =
        ovs_doca_get_post_meter_counter_id(shared_meter_id,
                                           DOCA_FLOW_METER_COLOR_GREEN);

    doca_set_reg_val_mask(&green_match.meta, &green_mask.meta,
                          REG_FIELD_FLOW_INFO, flow_id);
    green_match.parser_meta.meter_color = DOCA_FLOW_METER_COLOR_GREEN;
    green_mask.parser_meta.meter_color = 0xff;

    if (next_meter) {
        monitor.meter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
        monitor.shared_meter.shared_meter_id = next_meter->id;
        fid = next_meter->flow_id;
        next_mtr_fwd.type = DOCA_FLOW_FWD_PIPE;
        next_mtr_fwd.next_pipe = post_meter_pipe_ctx->pipe;
        fwd = &next_mtr_fwd;
    }

    doca_set_reg_val_mask(&dacts.meta, &dacts_masks.meta,
                          REG_FIELD_FLOW_INFO, fid);

    /* Insert green rule with high prio, which is higher than the corresponding
     * red rule, optimizing for the case when traffic stays below metered rate.
     */
    entry = create_doca_ctl_flow_entry(netdev, queue_id, post_meter_pipe_ctx,
                                       DPDK_OFFLOAD_PRIORITY_HIGH, &green_match,
                                       &green_mask, &dacts, &dacts_masks, NULL,
                                       &monitor, fwd, error);
    if (!entry) {
        VLOG_ERR_RL(&rl, "%s: Failed to create shared meter green rule for mtr"
                    " ID %u, flow ID %u", netdev_get_name(netdev),
                    shared_meter_id, flow_id);
        return NULL;
    }

    if (!next_meter) {
        /* replace original fwd with the internal meter pipe */
        memset(orig_fwd, 0, sizeof *orig_fwd);
        orig_fwd->type = DOCA_FLOW_FWD_PIPE;
        orig_fwd->next_pipe = post_meter_pipe_ctx->pipe;
    }

    return entry;
}

static int
destroy_meter_hierarchy(struct netdev *netdev, unsigned int queue_id,
                        struct doca_flow_handle_resources *flow_res,
                        struct rte_flow_error *error)
{
    struct doca_eswitch_ctx *esw_ctx = doca_eswitch_ctx_get(netdev);
    struct doca_meter_ctx *mtr_ctx;
    int err;

    if (!flow_res->meters_ctx) {
        return 0;
    }

    LIST_FOR_EACH_SAFE (mtr_ctx, list_node, flow_res->meters_ctx) {
        /* RED entry */
        if (mtr_ctx->post_meter_red_entry) {
            err = doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT,
                                          mtr_ctx->post_meter_red_entry);
            if (err) {
                if (error) {
                    error->type = RTE_FLOW_ERROR_TYPE_HANDLE;
                    error->message = doca_error_get_descr(err);
                }
                return -1;
            }
            dpdk_offload_counter_dec(netdev);
        }

        /* GREEN entry */
        if (mtr_ctx->post_meter_green_entry) {
            err = doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT,
                                          mtr_ctx->post_meter_green_entry);
            if (err) {
                if (error) {
                    error->type = RTE_FLOW_ERROR_TYPE_HANDLE;
                    error->message = doca_error_get_descr(err);
                }
                return -1;
            }
            dpdk_offload_counter_dec(netdev);
        }

        /* Meter internal flow ID */
        if (mtr_ctx->post_meter_flow_id) {
            shared_mtr_flow_id_free(esw_ctx, mtr_ctx->post_meter_flow_id);
        }

        ovs_list_remove(&mtr_ctx->list_node);
        free(mtr_ctx);
    }

    free(flow_res->meters_ctx);
    flow_res->meters_ctx = NULL;

    return 0;
}

static void
destroy_post_hash_entry(struct netdev *netdev, unsigned int queue_id,
                        struct doca_flow_handle_resources *flow_res)
{
    if (!flow_res->post_hash_entry) {
        return;
    }

    doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT,
                            flow_res->post_hash_entry);
    dpdk_offload_counter_dec(netdev);
    flow_res->post_hash_entry = NULL;
}

static struct doca_flow_pipe_entry *
create_post_hash_entry(struct netdev *netdev, unsigned int queue_id,
                       uint32_t flow_id,
                       struct doca_ctl_pipe_ctx *next_pipe_ctx,
                       struct rte_flow_error *error)
{
    struct doca_eswitch_ctx *esw_ctx = doca_eswitch_ctx_get(netdev);
    struct doca_ctl_pipe_ctx *post_hash_pipe_ctx;
    struct doca_flow_pipe_entry *entry;
    struct doca_flow_match spec;
    struct doca_flow_match mask;
    struct doca_flow_fwd fwd;

    memset(&mask, 0, sizeof mask);
    memset(&spec, 0, sizeof spec);
    memset(&fwd, 0, sizeof fwd);

    post_hash_pipe_ctx = esw_ctx->hash_pipe_ctx->post_hash_pipe_ctx;

    doca_set_reg_val_mask(&spec.meta, &mask.meta, REG_FIELD_FLOW_INFO, flow_id);
    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = next_pipe_ctx->pipe;

    entry = create_doca_ctl_flow_entry(netdev, queue_id, post_hash_pipe_ctx, 0,
                                       &spec, &mask, NULL, NULL, NULL, NULL,
                                       &fwd, error);
    return entry;
}

static int
create_meter_hierarchy(struct netdev *netdev, unsigned int queue_id,
                       uint32_t shared_meter_id,
                       struct doca_act_vars *dact_vars,
                       struct doca_flow_handle_resources *flow_res,
                       struct doca_flow_fwd *fwd, struct rte_flow_error *error)
{
    struct meter_info *next_mtr = NULL;
    struct doca_meter_ctx *mtr_ctx;
    struct meter_info *cur_mtr;

    if (!flow_res->meters_ctx) {
        flow_res->meters_ctx = xzalloc(sizeof *flow_res->meters_ctx);
        ovs_list_init(flow_res->meters_ctx);
    }

    /* Meter entries are added in reverse order to have the full hierarchy
     * already in place at time when the first packet arrives. The entries are
     * "chained" by the means of mark and match on meter_info->flow_id.
     */
    LIST_FOR_EACH_REVERSE (cur_mtr, list_node, &dact_vars->next_meters) {
        mtr_ctx = xzalloc(sizeof *mtr_ctx);

        /* RED entry */
        mtr_ctx->post_meter_red_entry =
            add_doca_post_meter_red_entry(netdev, queue_id, cur_mtr->id,
                                          cur_mtr->flow_id, error);
        if (!mtr_ctx->post_meter_red_entry) {
            if (error) {
                error->type = RTE_FLOW_ERROR_TYPE_ACTION;
                error->message = "Could not create red post multi-meter rule";
            }
            goto err;
        }

        /* GREEN entry */
        mtr_ctx->post_meter_green_entry =
            add_doca_post_meter_green_entry(netdev, queue_id, cur_mtr->id,
                                            cur_mtr->flow_id,
                                            dact_vars->flow_id, next_mtr, fwd,
                                            error);
        if (!mtr_ctx->post_meter_green_entry) {
            if (error) {
                error->type = RTE_FLOW_ERROR_TYPE_ACTION;
                error->message = "Could not create green post multi-meter rule";
            }
            doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT,
                                    mtr_ctx->post_meter_red_entry);
            goto err;
        }

        mtr_ctx->post_meter_flow_id = cur_mtr->flow_id;
        ovs_list_push_front(flow_res->meters_ctx, &mtr_ctx->list_node);
        next_mtr = cur_mtr;
    }

    mtr_ctx = xzalloc(sizeof *mtr_ctx);

    /* First RED entry */
    mtr_ctx->post_meter_red_entry =
        add_doca_post_meter_red_entry(netdev, queue_id, shared_meter_id,
                                      dact_vars->mtr_flow_id, error);
    if (!mtr_ctx->post_meter_red_entry) {
        if (error) {
            error->type = RTE_FLOW_ERROR_TYPE_ACTION;
            error->message = "Could not create red post meter rule";
        }
        goto err;
    }

    /* First GREEN entry */
    mtr_ctx->post_meter_green_entry =
        add_doca_post_meter_green_entry(netdev, queue_id, shared_meter_id,
                                            dact_vars->mtr_flow_id,
                                            dact_vars->flow_id, next_mtr,
                                            fwd, error);
    if (!mtr_ctx->post_meter_green_entry) {
        if (error) {
            error->type = RTE_FLOW_ERROR_TYPE_ACTION;
            error->message = "Could not create green post meter rule";
        }
        doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT,
                                mtr_ctx->post_meter_red_entry);
        goto err;
    }

    mtr_ctx->post_meter_flow_id = dact_vars->mtr_flow_id;
    ovs_list_push_front(flow_res->meters_ctx, &mtr_ctx->list_node);

    return 0;

err:
    free(mtr_ctx);
    destroy_meter_hierarchy(netdev, queue_id, flow_res, NULL);
    if (error) {
        ovs_assert(error->message);
        ovs_assert(error->type);
    }
    return -1;
}

static int
dpdk_offload_doca_create(struct netdev *netdev,
                         const struct rte_flow_attr *attr,
                         struct rte_flow_item *items,
                         struct rte_flow_action *actions,
                         struct dpdk_offload_handle *doh,
                         struct rte_flow_error *error)
{
    struct doca_eswitch_ctx *esw_ctx = doca_eswitch_ctx_get(netdev);
    unsigned int tid = netdev_offload_thread_id();
    struct doca_flow_actions dacts, dacts_masks;
    struct doca_flow_handle_resources flow_res;
    struct doca_flow_action_descs dacts_descs;
    struct doca_flow_action_desc desc_array;
    struct doca_flow_monitor monitor;
    struct doca_act_vars dact_vars;
    struct doca_flow_handle *hndl;
    struct doca_flow_match mask;
    struct doca_flow_match spec;
    struct meter_info *next_mtr;
    unsigned int queue_id = tid;
    uint32_t miss_flow_id = 0;
    struct doca_flow_fwd fwd;

    memset(error, 0, sizeof *error);
    memset(&dact_vars, 0, sizeof dact_vars);
    ovs_list_init(&dact_vars.next_meters);

    /* If it's a post ct rule, check for eswitch ct offload support */
    if (attr->group == POSTCT_TABLE_ID &&
        esw_ctx->shared_ct_counter_id_pool == NULL) {
        error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
        error->message = "CT offload disabled on this eswitch";
        goto err;
    }

    memset(&dacts_masks, 0, sizeof dacts_masks);
    memset(&dacts_descs, 0, sizeof dacts_descs);
    memset(&desc_array, 0, sizeof desc_array);
    memset(&flow_res, 0, sizeof flow_res);
    memset(&monitor, 0, sizeof monitor);
    memset(&dacts, 0, sizeof dacts);
    memset(&mask, 0, sizeof mask);
    memset(&spec, 0, sizeof spec);
    memset(&fwd, 0, sizeof fwd);

    if (doca_translate_items(netdev, attr, items, &spec, &mask)) {
        error->type = RTE_FLOW_ERROR_TYPE_ITEM;
        error->message = "Could not create items";
        goto err;
    }

    /* parse actions */
    dacts_descs.desc_array = &desc_array;
    dacts_descs.nb_action_desc = 1;
    if (doca_translate_actions(netdev, &spec, actions, &dacts, &dacts_masks,
                               &dacts_descs, &fwd, &monitor,
                               &flow_res, &dact_vars)) {
        error->type = RTE_FLOW_ERROR_TYPE_ACTION;
        error->message = "Could not create actions";
        goto err;
    }

    /* Detect a miss flow which sets flow id for the
     * Ctl pipe and store it for split flows usage.
     */
    if (attr->priority == DPDK_OFFLOAD_PRIORITY_MISS && dact_vars.flow_id) {
        miss_flow_id = dact_vars.flow_id;
    }

    /* Create post-hash entry */
    if (dact_vars.hash_data) {
        struct doca_ctl_pipe_ctx *next_pipe_ctx = flow_res.next_pipe_ctx;
        uint32_t flow_id = dact_vars.hash_data->flow_id;

        flow_res.post_hash_entry =
            create_post_hash_entry(netdev, queue_id, flow_id, next_pipe_ctx,
                                   error);
        if (flow_res.post_hash_entry == NULL) {
            goto err;
        }
    }

    if (monitor.meter_type == DOCA_FLOW_RESOURCE_TYPE_SHARED) {
        if (create_meter_hierarchy(netdev, queue_id, monitor.shared_meter.shared_meter_id,
                                   &dact_vars, &flow_res, &fwd, error)) {
            goto err;
        }
    }

    hndl = create_doca_flow_handle(netdev, queue_id, attr->priority, attr->group,
                                   miss_flow_id, &spec, &mask, &dacts,
                                   &dacts_masks, &dacts_descs, &monitor, &fwd,
                                   &flow_res, doh, error);
    if (!hndl) {
        /* change to free doca flow resources function */
        destroy_post_hash_entry(netdev, queue_id, &flow_res);
        doca_ctl_pipe_ctx_unref(flow_res.next_pipe_ctx);
        if (monitor.meter_type == DOCA_FLOW_RESOURCE_TYPE_SHARED) {
            destroy_meter_hierarchy(netdev, queue_id, &flow_res, NULL);
        }
        goto err;
    }

    LIST_FOR_EACH_POP (next_mtr, list_node, &dact_vars.next_meters) {
        free(next_mtr);
    }

    return 0;

err:
    LIST_FOR_EACH_POP (next_mtr, list_node, &dact_vars.next_meters) {
        free(next_mtr);
    }
    ovs_assert(error->message);
    ovs_assert(error->type);
    doh->rte_flow = NULL;
    return -1;
}

static int
destroy_dpdk_offload_handle(struct netdev *netdev,
                            struct dpdk_offload_handle *doh,
                            unsigned int queue_id,
                            struct rte_flow_error *error)
{
    int upkeep_retries = 10;
    doca_error_t err;

    while (doh->dfh.flow == NULL && upkeep_retries-- > 0) {
        /* Force polling completions, this handle
         * was not yet completed. */
        dpdk_offload_doca_upkeep_queue(netdev, true, queue_id);
    }

    /* It should have been completed by now, or something is wrong. */
    if (doh->dfh.flow == NULL) {
        if (error) {
            error->type = RTE_FLOW_ERROR_TYPE_HANDLE;
            error->message = "Failed to delete entry, "
                             "async insertion never completed";
        }
        return -1;
    }

    /* Deletion is always synchronous.
     *
     * If async deletion is implemented, aux-table uninit calls deleting
     * entries will use the offload queues in conflict with offload threads
     * polling them during upkeep. It should result in a crash or
     * in a lockup of the queues. */
    err = doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT, doh->dfh.flow);
    if (err) {
        if (error) {
            error->type = RTE_FLOW_ERROR_TYPE_HANDLE;
            error->message = doca_error_get_descr(err);
        }
        return -1;
    }

    if (doh->dfh.flow_res.meters_ctx) {
        err = destroy_meter_hierarchy(netdev, queue_id, &doh->dfh.flow_res,
                                      error);
        if (err) {
            return -1;
        }
    }

    if (doh->dfh.flow_res.split_ctx[0].curr_split_ctx) {
        for (int i = MAX_SPLIT_DEPTH - 1; i >= 0; i--) {
            split_prefix_ctx_unref(doh->dfh.flow_res.split_ctx[i].curr_split_ctx);
            doca_ctl_pipe_ctx_unref(doh->dfh.flow_res.split_ctx[i].next_split_pipe_ctx);
        }
    }

    destroy_post_hash_entry(netdev, queue_id, &doh->dfh.flow_res);

    /* Netdev can only be NULL during aux tables uninit. */
    if (netdev) {
        dpdk_offload_counter_dec(netdev);
    }

    doca_ctl_pipe_ctx_unref(doh->dfh.flow_res.next_pipe_ctx);
    doca_ctl_pipe_ctx_unref(doh->dfh.flow_res.self_pipe_ctx);

    return 0;
}

static int
dpdk_offload_doca_destroy(struct netdev *netdev,
                          struct dpdk_offload_handle *doh,
                          struct rte_flow_error *error,
                          bool esw_port_id OVS_UNUSED)
{
    unsigned int queue_id = netdev_offload_thread_id();

    return destroy_dpdk_offload_handle(netdev, doh, queue_id, error);
}

static int
dpdk_offload_doca_query_count(struct netdev *netdev,
                              struct dpdk_offload_handle *doh,
                              struct rte_flow_query_count *query,
                              struct rte_flow_error *error)
{
    struct doca_flow_pipe_entry *doca_flow;
    struct doca_flow_handle *hndl;
    struct doca_flow_query stats;
    doca_error_t err;

    hndl = &doh->dfh;
    doca_flow = hndl->flow;

    memset(query, 0, sizeof *query);
    memset(&stats, 0, sizeof stats);

    if (doca_flow == NULL) {
        /* The async entry has not yet been completed,
         * it cannot have done anything yet. */
        return 0;
    }

    err = doca_flow_query_entry(doca_flow, &stats);
    if (err) {
        VLOG_WARN_RL(&rl, "%s: Failed to query doca_flow: %p. Error %d (%s)",
                     netdev_get_name(netdev), doca_flow, err,
                     doca_error_get_descr(err));
        error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
        error->message = doca_error_get_descr(err);
        return -1;
    }

    query->hits = stats.total_pkts;
    query->bytes = stats.total_bytes;

    return 0;
}

static int
dpdk_offload_doca_shared_create(struct netdev *netdev,
                                struct indirect_ctx *ctx,
                                const struct rte_flow_action *action,
                                struct rte_flow_error *error)
{
    struct doca_eswitch_ctx *esw_ctx = doca_eswitch_ctx_get(netdev);
    unsigned int tid = netdev_offload_thread_id();
    uint32_t id;

    if (action->type != RTE_FLOW_ACTION_TYPE_COUNT) {
        return -1;
    }

    error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
    error->message = NULL;

    switch (ctx->res_type) {
    case OVS_SHARED_COUNT:
        if (!id_fpool_new_id(esw_ctx->shared_counter_id_pool, tid, &id)) {
            error->message = "Flow counter exhausted: no free ID";
        }
        if (error->message) {
            /* Flow-related message are not expected to be great in number,
             * no need to rate-limit errors. */
            VLOG_ERR("%s", error->message);
            return -1;
        }
        break;
    case OVS_SHARED_CT_COUNT:
        if (!esw_ctx->shared_ct_counter_id_pool) {
            error->message = "Shared CT counters not enabled on this eswitch";
        } else if (!id_fpool_new_id(esw_ctx->shared_ct_counter_id_pool,
                                    tid, &id)) {
            error->message = "CT counter exhausted: no free ID";
        }
        if (error->message) {
            /* CT-related message can be in high-volume, do not DoS our
             * syslog with errors. */
            VLOG_ERR_RL(&rl, "%s", error->message);
            return -1;
        }
        break;
    case OVS_SHARED_UNDEFINED:
        error->message = "Unsupported shared resource type requested";
        return -1;
    }

    ctx->res_id = id;
    ctx->act_type = action->type;

    return 0;
}

static int
dpdk_offload_doca_shared_destroy(struct indirect_ctx *ctx,
                                 struct rte_flow_error *error OVS_UNUSED)
{
    struct doca_eswitch_ctx *esw_ctx = doca_eswitch_ctx_get(ctx->netdev);
    unsigned int tid = netdev_offload_thread_id();

    switch (ctx->res_type) {
    case OVS_SHARED_COUNT:
        id_fpool_free_id(esw_ctx->shared_counter_id_pool, tid, ctx->res_id);
        break;
    case OVS_SHARED_CT_COUNT:
        if (esw_ctx->shared_ct_counter_id_pool) {
            id_fpool_free_id(esw_ctx->shared_ct_counter_id_pool, tid,
                             ctx->res_id);
        }
        break;
    case OVS_SHARED_UNDEFINED:
        VLOG_ERR("Unsupported shared resource type deletion");
        return -1;
    }

    return 0;
}

static int
dpdk_offload_doca_shared_query(struct indirect_ctx *ctx,
                               void *data,
                               struct rte_flow_error *error)
{
    struct doca_flow_shared_resource_result query_results;
    struct rte_flow_query_count *query;
    struct doca_flow_query *stats;
    doca_error_t ret;
    uint32_t cnt_id;

    /* Only shared counter supported at the moment */
    if (ctx->act_type != RTE_FLOW_ACTION_TYPE_COUNT) {
        return -1;
    }

    query = (struct rte_flow_query_count *) data;
    memset(query, 0, sizeof *query);
    memset(&query_results, 0, sizeof query_results);

    cnt_id = ctx->res_id;
    ret = doca_flow_shared_resources_query(DOCA_FLOW_SHARED_RESOURCE_COUNT,
                                           &cnt_id, &query_results, 1);
    if (ret != DOCA_SUCCESS) {
        VLOG_ERR("Failed to query shared counter id 0x%.8x: %s",
                 ctx->res_id, doca_error_get_descr(ret));
        error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
        error->message = doca_error_get_descr(ret);
        return -1;
    }

    stats = &query_results.counter;
    query->hits = stats->total_pkts;
    query->bytes = stats->total_bytes;

    return 0;
}

static int
ipv4_packet_hw_hash(struct doca_hash_pipe_ctx *hash_pipe_ctx,
                    struct dp_packet *packet,
                    uint32_t seed,
                    uint32_t *hash)
{
    struct doca_flow_match field_values;
    struct doca_flow_pipe *pipe;
    struct udp_header *udp;
    struct tcp_header *tcp;
    struct ip_header *ip;

    ip = dp_packet_l3(packet);

    memset(&field_values.meta, 0, sizeof field_values.meta);
    doca_set_reg_val(&field_values.meta, REG_FIELD_SCRATCH, seed);
    if (ip->ip_proto == IPPROTO_UDP) {
        pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV4_UDP].pipe;
        udp = (struct udp_header *) (ip + 1);
        field_values.outer.udp.l4_port.src_port = udp->udp_src;
        field_values.outer.udp.l4_port.dst_port = udp->udp_dst;
        field_values.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
    } else if (ip->ip_proto == IPPROTO_TCP) {
        pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV4_TCP].pipe;
        tcp = (struct tcp_header *) (ip + 1);
        field_values.outer.tcp.l4_port.src_port = tcp->tcp_src;
        field_values.outer.tcp.l4_port.dst_port = tcp->tcp_dst;
        field_values.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
    } else {
        pipe = hash_pipe_ctx->hashes[HASH_TYPE_IPV4_L3].pipe;
    }

    field_values.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    field_values.outer.ip4.src_ip = get_16aligned_be32(&ip->ip_src);
    field_values.outer.ip4.dst_ip = get_16aligned_be32(&ip->ip_dst);

    return doca_flow_pipe_calc_hash(pipe, &field_values, hash);
}

static int
ipv6_packet_hw_hash(struct doca_hash_pipe_ctx *hash_pipe_ctx,
                    struct dp_packet *packet,
                    uint32_t seed,
                    uint32_t *hash)
{
    struct doca_flow_match field_values;
    struct ovs_16aligned_ip6_hdr *ip6;
    struct doca_flow_pipe *pre, *suf;
    struct udp_header *udp;
    struct tcp_header *tcp;
    int rv;

    ip6 = dp_packet_l3(packet);

    memset(&field_values.meta, 0, sizeof field_values.meta);
    doca_set_reg_val(&field_values.meta, REG_FIELD_SCRATCH, seed);
    if (ip6->ip6_nxt == IPPROTO_UDP) {
        pre = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_UDP_PRE].pipe;
        suf = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_UDP_SUF].pipe;
        udp = (struct udp_header *) (ip6 + 1);
        field_values.outer.udp.l4_port.src_port = udp->udp_src;
        field_values.outer.udp.l4_port.dst_port = udp->udp_dst;
        field_values.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
    } else if (ip6->ip6_nxt == IPPROTO_TCP) {
        pre = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_TCP_PRE].pipe;
        suf = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_TCP_SUF].pipe;
        tcp = (struct tcp_header *) (ip6 + 1);
        field_values.outer.tcp.l4_port.src_port = tcp->tcp_src;
        field_values.outer.tcp.l4_port.dst_port = tcp->tcp_dst;
        field_values.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
    } else {
        pre = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_L3_PRE].pipe;
        suf = hash_pipe_ctx->hashes[HASH_TYPE_IPV6_L3_SUF].pipe;
    }

    field_values.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
    memcpy(&field_values.outer.ip6.src_ip, &ip6->ip6_src, sizeof ip6->ip6_src);
    memcpy(&field_values.outer.ip6.dst_ip, &ip6->ip6_dst, sizeof ip6->ip6_dst);

    rv = doca_flow_pipe_calc_hash(pre, &field_values, hash);

    if (rv) {
        return rv;
    }

    memset(&field_values.meta, 0, sizeof field_values.meta);
    doca_set_reg_val(&field_values.meta, REG_FIELD_SCRATCH, *hash);

    return doca_flow_pipe_calc_hash(suf, &field_values, hash);
}

static int
dpdk_offload_doca_packet_hw_hash(struct netdev *netdev,
                                 struct dp_packet *packet,
                                 uint32_t seed,
                                 uint32_t *hash)
{
    struct doca_hash_pipe_ctx *hash_pipe_ctx;
    struct doca_eswitch_ctx *esw_ctx;
    ovs_be16 eth_proto;
    int rv = -1;

    esw_ctx = doca_eswitch_ctx_get(netdev);
    if (!esw_ctx) {
        return -1;
    }
    hash_pipe_ctx = esw_ctx->hash_pipe_ctx;

    parse_tcp_flags(packet, &eth_proto, NULL, NULL);
    if (eth_proto == htons(ETH_TYPE_IP)) {
        rv = ipv4_packet_hw_hash(hash_pipe_ctx, packet, seed, hash);
    } else if (eth_proto == htons(ETH_TYPE_IPV6)) {
        rv = ipv6_packet_hw_hash(hash_pipe_ctx, packet, seed, hash);
    } else {
        /* Only IPv4/IPv6 are supported. */
        return -1;
    }

    if (rv) {
        *hash = 0;
    } else {
        *hash &= 0x0000FFFF;
        *hash |= 0xd0ca0000;
    }
    return rv;
}

static void
dpdk_offload_doca_get_pkt_recover_info(struct dp_packet *p,
                                       struct dpdk_offload_recovery_info *info)
{
    memset(info, 0, sizeof *info);
    if (dpdk_offload_get_reg_field(p, REG_FIELD_FLOW_INFO,
                                   &info->flow_miss_id)) {
        dp_packet_set_flow_mark(p, info->flow_miss_id);
        if (doca_ct_offload_enabled) {
            dpdk_offload_get_reg_field(p, REG_FIELD_CT_CTX, &info->ct_miss_id);
        } else {
            dpdk_offload_get_reg_field(p, REG_FIELD_DP_HASH, &info->dp_hash);
        }
    }
}

static void
dpdk_offload_doca_update_stats(struct dpif_flow_stats *stats,
                               struct dpif_flow_attrs *attrs,
                               struct rte_flow_query_count *query)
{
    if (attrs) {
        attrs->dp_layer = "doca";
    }

    if (!query) {
        return;
    }

    if (stats->n_packets != query->hits) {
        query->hits_set = 1;
        query->bytes_set = 1;
    }

    stats->n_packets = query->hits;
    stats->n_bytes = query->bytes;
}

static void
doca_fixed_rule_uninit(struct netdev *netdev, struct fixed_rule *fr)
{
    if (!fr->doh.dfh.flow) {
        return;
    }

    destroy_dpdk_offload_handle(netdev, &fr->doh, AUX_QUEUE, NULL);
    fr->doh.dfh.flow = NULL;
}

static void
doca_ct_zones_uninit(struct netdev *netdev, struct doca_eswitch_ctx *ctx)
{
    struct fixed_rule *fr;
    uint32_t zone_id;
    int nat, i;

    if (netdev_is_zone_tables_disabled()) {
        VLOG_ERR("Disabling ct zones is not supported with doca");
        return;
    }

    for (nat = 0; nat < 2; nat++) {
        for (i = 0; i < CT_ZONE_FLOWS_NUM; i++) {
            for (zone_id = MIN_ZONE_ID; zone_id <= MAX_ZONE_ID; zone_id++) {
                fr = &ctx->zone_cls_flows[nat][i][zone_id];
                doca_fixed_rule_uninit(netdev, fr);
            }
        }
    }
}

static int
doca_create_ct_zone_revisit_rule(struct netdev *netdev, uint32_t group,
                                 uint16_t zone, int nat,
                                 struct dpdk_offload_handle *doh)
{
    struct doca_flow_handle_resources flow_res;
    struct doca_ctl_pipe_ctx *next_pipe_ctx;
    uint32_t ct_state_spec, ct_state_mask;
    struct doca_flow_handle *hndl;
    struct rte_flow_error error;
    struct doca_flow_match mask;
    struct doca_flow_match spec;
    struct doca_flow_fwd fwd;

    memset(&flow_res, 0, sizeof flow_res);
    memset(&mask, 0, sizeof mask);
    memset(&spec, 0, sizeof spec);
    memset(&fwd, 0, sizeof fwd);

    /* If the zone is the same, and already visited ct/ct-nat, skip
     * ct/ct-nat and jump directly to post-ct.
     */
    ct_state_spec = OVS_CS_F_TRACKED;
    if (nat) {
        ct_state_spec |= OVS_CS_F_NAT_MASK;
    }
    ct_state_mask = ct_state_spec;

    /* Merge ct_zone and ct_state matches in a single item. */
    doca_set_reg_val_mask(&spec.meta, &mask.meta, REG_FIELD_CT_ZONE, zone);
    doca_set_reg_val(&spec.meta, REG_FIELD_CT_STATE, ct_state_spec);
    doca_set_reg_val(&mask.meta, REG_FIELD_CT_STATE, ct_state_mask);

    next_pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, POSTCT_TABLE_ID);
    if (!next_pipe_ctx) {
        return -1;
    }

    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = next_pipe_ctx->pipe;
    flow_res.next_pipe_ctx = next_pipe_ctx;

    hndl = create_doca_flow_handle(netdev, AUX_QUEUE,
                                   DPDK_OFFLOAD_PRIORITY_HIGH, group, 0, &spec,
                                   &mask, NULL, NULL, NULL, NULL, &fwd,
                                   &flow_res, doh, &error);
    if (!hndl) {
        return -1;
    }
    return 0;
}

static int
doca_create_ct_zone_uphold_rule(struct netdev *netdev,
                                struct doca_eswitch_ctx *ctx, uint32_t group,
                                uint16_t zone,
                                enum ct_zone_cls_flow_type type,
                                struct dpdk_offload_handle *doh)
{
    struct doca_flow_actions dacts, dacts_masks;
    struct doca_flow_handle_resources flow_res;
    struct doca_flow_handle *hndl;
    struct doca_flow_match spec;
    struct doca_flow_match mask;
    struct rte_flow_error error;
    struct doca_flow_fwd fwd;

    memset(&dacts_masks, 0, sizeof dacts_masks);
    memset(&flow_res, 0, sizeof flow_res);
    memset(&dacts, 0, sizeof dacts);
    memset(&fwd, 0, sizeof fwd);
    memset(&spec, 0, sizeof spec);
    memset(&mask, 0, sizeof mask);

    if (type == CT_ZONE_FLOW_UPHOLD_IP6_TCP ||
        type == CT_ZONE_FLOW_UPHOLD_IP6_UDP) {
        spec.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
        mask.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
    } else {
        spec.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
        mask.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    }

    if (type == CT_ZONE_FLOW_UPHOLD_IP4_TCP ||
        type == CT_ZONE_FLOW_UPHOLD_IP6_TCP) {
        spec.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
        mask.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
        /* Ensure that none of SYN | RST | FIN flag is set in
         * packets going to CT: they must miss and go to SW. */
        mask.outer.tcp.flags = TCP_SYN | TCP_RST | TCP_FIN;
    } else {
        spec.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
        mask.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
    }

    doca_set_reg_val_mask(&dacts.meta, &dacts_masks.meta, REG_FIELD_CT_ZONE, zone);

    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = doca_get_ct_pipe(ctx, &spec);
    flow_res.next_pipe_ctx = NULL;

    hndl = create_doca_flow_handle(netdev, AUX_QUEUE,
                                   DPDK_OFFLOAD_PRIORITY_MED, group, 0, &spec,
                                   &mask, &dacts, &dacts_masks, NULL, NULL,
                                   &fwd, &flow_res, doh, &error);
    if (!hndl) {
        return -1;
    }
    return 0;
}

static int
doca_create_ct_zone_miss_rule(struct netdev *netdev, uint32_t group,
                              struct dpdk_offload_handle *doh)
{
    struct doca_flow_handle_resources flow_res;
    struct doca_ctl_pipe_ctx *pipe_ctx;
    struct doca_flow_handle *hndl;
    struct rte_flow_error error;
    struct doca_flow_fwd fwd;

    memset(&flow_res, 0, sizeof flow_res);
    memset(&fwd, 0, sizeof fwd);

    pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, MISS_TABLE_ID);
    if (!pipe_ctx) {
        return -1;
    }

    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = pipe_ctx->pipe;
    flow_res.next_pipe_ctx = pipe_ctx;

    hndl = create_doca_flow_handle(netdev, AUX_QUEUE,
                                   DPDK_OFFLOAD_PRIORITY_LOW, group, 0, NULL,
                                   NULL, NULL, NULL, NULL, NULL, &fwd,
                                   &flow_res, doh, &error);
    if (!hndl) {
        return -1;
    }
    return 0;
}

static int
doca_ct_zones_init(struct netdev *netdev, struct doca_eswitch_ctx *ctx)
{
    struct fixed_rule *fr;
    uint32_t base_group;
    uint32_t zone_id;
    int nat;

    if (netdev_is_zone_tables_disabled()) {
        VLOG_ERR("Disabling ct zones is not supported with doca");
        return -1;
    }

    /* Merge the tag match for zone and state only if they are
     * at the same index. */
    ovs_assert(reg_fields[REG_FIELD_CT_ZONE].index == reg_fields[REG_FIELD_CT_STATE].index);

    for (nat = 0; nat < 2; nat++) {
        base_group = nat ? CTNAT_TABLE_ID : CT_TABLE_ID;

        for (zone_id = MIN_ZONE_ID; zone_id <= MAX_ZONE_ID; zone_id++) {
            /* If the zone is already set, then CT for this zone has already
             * been executed: skip to post-ct. */

            fr = &ctx->zone_cls_flows[nat][CT_ZONE_FLOW_REVISIT][zone_id];
            if (doca_create_ct_zone_revisit_rule(netdev, base_group + zone_id,
                                                 zone_id, nat, &fr->doh)) {
                goto err;
            }

            /* Otherwise, set the zone and go to CT/CT-NAT. */

            fr = &ctx->zone_cls_flows[nat][CT_ZONE_FLOW_UPHOLD_IP4_UDP][zone_id];
            if (doca_create_ct_zone_uphold_rule(netdev, ctx,
                                                base_group + zone_id, zone_id,
                                                CT_ZONE_FLOW_UPHOLD_IP4_UDP,
                                                &fr->doh)) {
                goto err;
            }

            fr = &ctx->zone_cls_flows[nat][CT_ZONE_FLOW_UPHOLD_IP4_TCP][zone_id];
            if (doca_create_ct_zone_uphold_rule(netdev, ctx,
                                                base_group + zone_id,
                                                zone_id,
                                                CT_ZONE_FLOW_UPHOLD_IP4_TCP,
                                                &fr->doh)) {
                goto err;
            }

            /* If the CT-zone was never visited, but the packet does
             * not match either TCP(!SFR) or UDP, miss and go to SW. */

            fr = &ctx->zone_cls_flows[nat][CT_ZONE_FLOW_MISS][zone_id];
            if (doca_create_ct_zone_miss_rule(netdev, base_group + zone_id,
                                              &fr->doh)) {
                goto err;
            }

            if (!conntrack_offload_ipv6_is_enabled()) {
                continue;
            }

            fr = &ctx->zone_cls_flows[nat][CT_ZONE_FLOW_UPHOLD_IP6_UDP][zone_id];
            if (doca_create_ct_zone_uphold_rule(netdev, ctx,
                                                base_group + zone_id,
                                                zone_id,
                                                CT_ZONE_FLOW_UPHOLD_IP6_UDP,
                                                &fr->doh)) {
                goto err;
            }

            fr = &ctx->zone_cls_flows[nat][CT_ZONE_FLOW_UPHOLD_IP6_TCP][zone_id];
            if (doca_create_ct_zone_uphold_rule(netdev, ctx,
                                                base_group + zone_id,
                                                zone_id,
                                                CT_ZONE_FLOW_UPHOLD_IP6_TCP,
                                                &fr->doh)) {
                goto err;
            }
        }
    }

    return 0;

err:
    doca_ct_zones_uninit(netdev, ctx);
    return -1;
}

static void
doca_ct_pipe_destroy(struct doca_basic_pipe_ctx *pipe_ctx)
{
    doca_ctl_pipe_ctx_unref(pipe_ctx->fwd_pipe_ctx);
    pipe_ctx->fwd_pipe_ctx = NULL;

    doca_ctl_pipe_ctx_unref(pipe_ctx->miss_pipe_ctx);
    pipe_ctx->miss_pipe_ctx = NULL;

    doca_flow_pipe_destroy(pipe_ctx->pipe);
    pipe_ctx->pipe = NULL;
}

static void
doca_ct_pipes_destroy(struct doca_eswitch_ctx *ctx)
{
    struct doca_basic_pipe_ctx *pipe_ctx;
    int i, j;

    doca_ct_pipe_destroy(&ctx->ct_ip6_prefix);

    for (i = 0; i < NUM_CT_NW; i++) {
        for (j = 0; j < NUM_CT_TP; j++) {
            pipe_ctx = &ctx->ct_pipes[i][j];
            doca_ct_pipe_destroy(pipe_ctx);
        }
    }
}

static void
doca_basic_pipe_name(struct ds *s, struct netdev *netdev,
                     enum ct_nw_type nw_type,
                     enum ct_tp_type tp_type)
{
    ds_put_format(s, "OVS_BASIC_CT_PIPE_%d",
                  netdev_dpdk_get_esw_mgr_port_id(netdev));

    switch (nw_type) {
    case CT_NW_IP4:
        ds_put_cstr(s, "_IP4");
        break;
    case CT_NW_IP6:
        ds_put_cstr(s, "_IP6_SUFFIX");
        break;
    case NUM_CT_NW:
       OVS_NOT_REACHED();
    }

    switch (tp_type) {
    case CT_TP_UDP:
        ds_put_cstr(s, "_UDP");
        break;
    case CT_TP_TCP:
        ds_put_cstr(s, "_TCP");
        break;
    case NUM_CT_TP:
       OVS_NOT_REACHED();
    }
}

static struct doca_flow_match ct_matches[NUM_CT_NW][NUM_CT_TP] = {
    [CT_NW_IP4] = {
        [CT_TP_UDP] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP4,
            .outer.ip4.src_ip = UINT32_MAX,
            .outer.ip4.dst_ip = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP,
            .outer.udp.l4_port.src_port = UINT16_MAX,
            .outer.udp.l4_port.dst_port = UINT16_MAX,
            /* For pure functionality this match is not needed, instead use it
             * as a WA to take advantage of the lower level optimization.
             */
            .parser_meta.port_meta = UINT32_MAX,
        },
        [CT_TP_TCP] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP4,
            .outer.ip4.src_ip = UINT32_MAX,
            .outer.ip4.dst_ip = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
            .outer.tcp.l4_port.src_port = UINT16_MAX,
            .outer.tcp.l4_port.dst_port = UINT16_MAX,
            /* For pure functionality this match is not needed, instead use it
             * as a WA to take advantage of the lower level optimization.
             */
            .parser_meta.port_meta = UINT32_MAX,
        },
    },
    [CT_NW_IP6] = {
        [CT_TP_UDP] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.dst_ip[0] = UINT32_MAX,
            .outer.ip6.dst_ip[1] = UINT32_MAX,
            .outer.ip6.dst_ip[2] = UINT32_MAX,
            .outer.ip6.dst_ip[3] = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP,
            .outer.udp.l4_port.src_port = UINT16_MAX,
            .outer.udp.l4_port.dst_port = UINT16_MAX,
            /* For pure functionality this match is not needed, instead use it
             * as a WA to take advantage of the lower level optimization.
             */
            .parser_meta.port_meta = UINT32_MAX,
        },
        [CT_TP_TCP] = {
            .outer.l3_type = DOCA_FLOW_L3_TYPE_IP6,
            .outer.ip6.dst_ip[0] = UINT32_MAX,
            .outer.ip6.dst_ip[1] = UINT32_MAX,
            .outer.ip6.dst_ip[2] = UINT32_MAX,
            .outer.ip6.dst_ip[3] = UINT32_MAX,
            .outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
            .outer.tcp.l4_port.src_port = UINT16_MAX,
            .outer.tcp.l4_port.dst_port = UINT16_MAX,
            /* For pure functionality this match is not needed, instead use it
             * as a WA to take advantage of the lower level optimization.
             */
            .parser_meta.port_meta = UINT32_MAX,
        },
    },
};

static int
doca_ct_ip6_prefix_pipe_init(struct netdev *netdev,
                             struct doca_eswitch_ctx *ctx)
{
    struct doca_ctl_pipe_ctx *miss_pipe_ctx = NULL;
    struct doca_flow_actions *actions_masks_list;
    struct doca_flow_match match, match_mask;
    struct doca_flow_actions actions_masks;
    struct doca_flow_actions *actions_list;
    struct doca_basic_pipe_ctx *pipe_ctx;
    struct doca_flow_actions actions;
    struct doca_flow_pipe *miss_pipe;
    struct doca_flow_port *doca_port;
    struct doca_flow_pipe_cfg cfg;
    struct doca_flow_fwd miss;
    struct doca_flow_fwd fwd;
    struct ds pipe_name;
    int ret;

    pipe_ctx = &ctx->ct_ip6_prefix;

    memset(&cfg, 0, sizeof cfg);
    memset(&fwd, 0, sizeof fwd);
    memset(&miss, 0, sizeof miss);
    memset(&match, 0, sizeof match);
    memset(&actions, 0, sizeof actions);
    memset(&match_mask, 0, sizeof match_mask);
    memset(&actions_masks, 0, sizeof actions_masks);

    ds_init(&pipe_name);
    ds_put_format(&pipe_name, "OVS_BASIC_CT_PIPE_%d_IP6_PREFIX",
                  netdev_dpdk_get_esw_mgr_port_id(netdev));

    actions_list = &actions;
    actions_masks_list = &actions_masks;

    doca_set_reg_mask(&actions.meta, REG_FIELD_SCRATCH);
    doca_set_reg_mask(&actions_masks.meta, REG_FIELD_SCRATCH);

    /* Set ip6 ct tuple match template (zone + src_ip + l4_type) */
    doca_set_reg_mask(&match.meta, REG_FIELD_CT_ZONE);
    doca_set_reg_mask(&match_mask.meta, REG_FIELD_CT_ZONE);
    match.parser_meta.port_meta = UINT32_MAX;
    match_mask.parser_meta.port_meta = UINT32_MAX;
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
    match_mask.outer.l3_type = DOCA_FLOW_L3_TYPE_IP6;
    memset(&match.outer.ip6.src_ip, 0xFF, sizeof match.outer.ip6.src_ip);
    memset(&match_mask.outer.ip6.src_ip, 0xFF, sizeof match.outer.ip6.src_ip);
    match.outer.ip6.next_proto = UINT8_MAX;
    match_mask.outer.ip6.next_proto = UINT8_MAX;

    doca_port = netdev_dpdk_doca_port_get(netdev);

    cfg.attr.name = ds_cstr(&pipe_name);
    cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
    cfg.attr.is_root = false;
    cfg.attr.nb_actions = 1;
    cfg.attr.nb_flows = ovs_doca_max_ct_rules();
    cfg.attr.enable_strict_matching = true;
    cfg.port = doca_flow_port_switch_get(doca_port);
    cfg.match = &match;
    cfg.match_mask = &match_mask;
    cfg.actions = &actions_list;
    cfg.actions_masks = &actions_masks_list;

    /* Next pipe will be determined per connection based on L4 type */
    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = NULL;

    miss_pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, MISS_TABLE_ID);
    ds_destroy(&pipe_name);
    if (miss_pipe_ctx == NULL) {
        VLOG_ERR("%s: Failed to take a reference on miss table",
                 netdev_get_name(netdev));
        return -1;
    }
    miss_pipe = miss_pipe_ctx->pipe;
    miss.type = DOCA_FLOW_FWD_PIPE;
    miss.next_pipe = miss_pipe;

    ret = doca_flow_pipe_create(&cfg, &fwd, &miss, &pipe_ctx->pipe);
    if (ret) {
        VLOG_ERR("%s: Failed to create ct prefix basic pipe: %d (%s)",
                 netdev_get_name(netdev), ret,
                 doca_error_get_descr(ret));
        doca_ctl_pipe_ctx_unref(miss_pipe_ctx);
        return ret;
    }

    pipe_ctx->miss_pipe_ctx = miss_pipe_ctx;

    return 0;
}

static int
doca_ct_pipe_init(struct netdev *netdev, struct doca_eswitch_ctx *ctx,
                  enum ct_nw_type nw_type, enum ct_tp_type tp_type)
{
    struct doca_ctl_pipe_ctx *miss_pipe_ctx = NULL;
    struct doca_ctl_pipe_ctx *fwd_pipe_ctx = NULL;
    struct doca_flow_actions *actions_masks_list;
    struct doca_flow_header_format *outer_masks;
    struct doca_flow_actions actions_masks;
    struct doca_flow_header_format *outer;
    struct doca_flow_actions *actions_list;
    struct doca_basic_pipe_ctx *pipe_ctx;
    struct doca_flow_match match_mask;
    struct doca_flow_actions actions;
    struct doca_flow_pipe *miss_pipe;
    struct doca_flow_port *doca_port;
    struct doca_flow_monitor monitor;
    enum dpdk_reg_id set_tags[] = {
        REG_FIELD_CT_STATE,
        REG_FIELD_CT_MARK,
        REG_FIELD_CT_LABEL_ID,
    };
    struct doca_flow_pipe_cfg cfg;
    struct doca_flow_fwd miss;
    struct doca_flow_fwd fwd;
    struct reg_field *ct_reg;
    struct ds pipe_name;
    int ret, i;

    pipe_ctx = &ctx->ct_pipes[nw_type][tp_type];

    /* Do not re-init a pipe if already done. */
    if (pipe_ctx->pipe != NULL) {
        return 0;
    }

    /* Do not initialize IPv6 pipes if not enabled. */
    if (!conntrack_offload_ipv6_is_enabled() &&
        nw_type == CT_NW_IP6) {
        return 0;
    }

    memset(&cfg, 0, sizeof cfg);
    memset(&fwd, 0, sizeof fwd);
    memset(&miss, 0, sizeof miss);
    memset(&actions, 0, sizeof actions);
    memset(&actions_masks, 0, sizeof actions_masks);
    memset(&monitor, 0, sizeof monitor);

    ds_init(&pipe_name);
    doca_basic_pipe_name(&pipe_name, netdev, nw_type, tp_type);

    actions_list = &actions;
    actions_masks_list = &actions_masks;

    outer = &actions.outer;
    outer_masks = &actions_masks.outer;

    /* Write the CT-NAT action template. */
    if (nw_type == CT_NW_IP4) {
        outer->l3_type = DOCA_FLOW_L3_TYPE_IP4;
        outer->ip4.src_ip = UINT32_MAX;
        outer->ip4.dst_ip = UINT32_MAX;
    } else if (nw_type == CT_NW_IP6) {
        outer->l3_type = DOCA_FLOW_L3_TYPE_IP6;
        memset(&outer->ip6.src_ip, UINT8_MAX, sizeof outer->ip6.src_ip) ;
        memset(&outer->ip6.dst_ip, UINT8_MAX, sizeof outer->ip6.dst_ip) ;
    } else {
        OVS_NOT_REACHED();
    }

    if (tp_type == CT_TP_UDP) {
        outer->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
        outer->udp.l4_port.src_port = UINT16_MAX;
        outer->udp.l4_port.dst_port = UINT16_MAX;
    } else {
        outer->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
        outer->tcp.l4_port.src_port = UINT16_MAX;
        outer->tcp.l4_port.dst_port = UINT16_MAX;
    }
    memcpy(outer_masks, outer, sizeof *outer_masks);

    ct_reg = &reg_fields[REG_FIELD_CT_CTX];
    /* Use 0xFFFs values to set pkt_meta in the action upon pipe create
     * and have the mask in the actions_mask
     */
    if (ct_reg->type == REG_TYPE_TAG) {
        actions.meta.u32[ct_reg->index] = UINT32_MAX;
    } else {
        actions.meta.pkt_meta = UINT32_MAX;
    }
    doca_set_reg_mask(&actions_masks.meta, REG_FIELD_CT_CTX);
    for (i = 0; i < ARRAY_SIZE(set_tags); i++) {
        ct_reg = &reg_fields[set_tags[i]];
        /* Use 0xFFFs values to set meta.u32 in the action upon pipe create
         * and have the mask in the actions_mask
         */
        if (ct_reg->type == REG_TYPE_TAG) {
            actions.meta.u32[ct_reg->index] = UINT32_MAX;
        } else {
            actions.meta.pkt_meta = UINT32_MAX;
        }
        doca_set_reg_mask(&actions_masks.meta, set_tags[i]);
    }

    /* Finalize the match templates. */
    doca_set_reg_mask(&ct_matches[CT_NW_IP4][CT_TP_UDP].meta, REG_FIELD_CT_ZONE);
    doca_set_reg_mask(&ct_matches[CT_NW_IP4][CT_TP_TCP].meta, REG_FIELD_CT_ZONE);
    /* Add prefix id match to IPv6 pipes */
    doca_set_reg_mask(&ct_matches[CT_NW_IP6][CT_TP_UDP].meta, REG_FIELD_SCRATCH);
    doca_set_reg_mask(&ct_matches[CT_NW_IP6][CT_TP_TCP].meta, REG_FIELD_SCRATCH);
    /* The mask is identical to the match itself. */
    match_mask = ct_matches[nw_type][tp_type];
    doca_port = netdev_dpdk_doca_port_get(netdev);

    monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
    monitor.shared_counter.shared_counter_id = UINT32_MAX;

    cfg.attr.name = ds_cstr(&pipe_name);
    cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
    cfg.attr.is_root = false;
    cfg.attr.nb_actions = 1;
    cfg.attr.nb_flows = ovs_doca_max_ct_rules();
    cfg.attr.enable_strict_matching = true;
    cfg.port = doca_flow_port_switch_get(doca_port);
    cfg.match = &ct_matches[nw_type][tp_type];
    cfg.match_mask = &match_mask;
    cfg.actions = &actions_list;
    cfg.actions_masks = &actions_masks_list;
    cfg.monitor = &monitor;

    fwd_pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, POSTCT_TABLE_ID);
    if (fwd_pipe_ctx == NULL) {
        VLOG_ERR("%s: Failed to take a reference on post-ct table",
                 netdev_get_name(netdev));
        return -1;
    }
    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = fwd_pipe_ctx->pipe;

    miss_pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, MISS_TABLE_ID);
    if (miss_pipe_ctx == NULL) {
        VLOG_ERR("%s: Failed to take a reference on miss table",
                 netdev_get_name(netdev));
        return -1;
    }
    miss_pipe = miss_pipe_ctx->pipe;
    miss.type = DOCA_FLOW_FWD_PIPE;
    miss.next_pipe = miss_pipe;

    ret = doca_flow_pipe_create(&cfg, &fwd, &miss, &pipe_ctx->pipe);
    if (ret) {
        VLOG_ERR("%s: Failed to create basic pipe: %d (%s)",
                 netdev_get_name(netdev), ret,
                 doca_error_get_descr(ret));
        goto error;
    }

    pipe_ctx->fwd_pipe_ctx = fwd_pipe_ctx;
    pipe_ctx->miss_pipe_ctx = miss_pipe_ctx;
    ds_destroy(&pipe_name);
    return 0;

error:
    doca_ctl_pipe_ctx_unref(fwd_pipe_ctx);
    doca_ctl_pipe_ctx_unref(miss_pipe_ctx);
    ds_destroy(&pipe_name);
    return ret;
}

static int
doca_ct_pipes_init(struct netdev *netdev, struct doca_eswitch_ctx *ctx)
{
    int i, j;

    for (i = 0; i < NUM_CT_NW; i++) {
        for (j = 0; j < NUM_CT_TP; j++) {
            if (doca_ct_pipe_init(netdev, ctx, i, j)) {
                goto error;
            }
        }
    }

    if (conntrack_offload_ipv6_is_enabled()) {
        if (doca_ct_ip6_prefix_pipe_init(netdev, ctx)) {
            goto error;
        }
    }

    return 0;

error:
    /* Rollback any pipe creation. */
    doca_ct_pipes_destroy(ctx);
    return -1;
}

/* Init the shared counter id map for the first
 * eswitch context that requests it. This is the only
 * eswitch that will support CT offload for now.
 * After DOCA adds proper support this limitation should
 * be lifted and support shared counters for every eswitch
 * will be added.
 */
static void
shared_counter_id_pool_init(struct doca_eswitch_ctx *ctx)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;
    unsigned int n_thread = netdev_offload_thread_nb();
    uint32_t base_id;

    if (ovsthread_once_start(&init_once)) {
        esw_id_pool = id_fpool_create(1, 0, OVS_DOCA_MAX_ESW);
        ovsthread_once_done(&init_once);
    }
    if (!esw_id_pool || !id_fpool_new_id(esw_id_pool, 0, &ctx->esw_id)) {
        VLOG_ERR("Failed to alloc a new esw id");
        return;
    }

    base_id = ovs_doca_max_shared_counters_per_esw() * ctx->esw_id;
    if (ovs_doca_max_ct_counters_per_esw() > 0) {
        ctx->shared_ct_counter_id_pool =
            id_fpool_create(n_thread, base_id,
                            ovs_doca_max_ct_counters_per_esw());
    } else {
        ctx->shared_ct_counter_id_pool = NULL;
    }

    base_id += ovs_doca_max_ct_counters_per_esw();
    ctx->shared_counter_id_pool =
        id_fpool_create(n_thread, base_id,
                        OVS_DOCA_MAX_METER_COUNTERS_PER_ESW);
}

static int
doca_bind_shared_cntrs(struct doca_eswitch_ctx *ctx)
{
    struct doca_flow_shared_resource_cfg cfg =
        { .domain = DOCA_FLOW_PIPE_DOMAIN_DEFAULT };
    uint32_t base_id;
    uint32_t *ids;
    int i, ret;

    ids = xcalloc(ovs_doca_max_shared_counters_per_esw(), sizeof *ids);

    base_id = ovs_doca_max_shared_counters_per_esw() * ctx->esw_id;
    for (i = 0; i < ovs_doca_max_shared_counters_per_esw(); i++) {
        ids[i] = base_id + i;
        ret = doca_flow_shared_resource_cfg(DOCA_FLOW_SHARED_RESOURCE_COUNT,
                                            ids[i], &cfg);
        if (ret != DOCA_SUCCESS) {
            VLOG_ERR("Failed to config shared counter id %d, err %d - %s",
                     ids[i], ret, doca_error_get_descr(ret));
            goto free_ids;
        }
    }

    ret = doca_flow_shared_resources_bind(DOCA_FLOW_SHARED_RESOURCE_COUNT, ids,
                                       ovs_doca_max_shared_counters_per_esw(),
                                       ctx->esw_port);
    if (ret != DOCA_SUCCESS) {
        VLOG_ERR("Shared counters binding failed, ids %d-%d, err %d - %s",
                 ids[0], ids[ovs_doca_max_shared_counters_per_esw() - 1], ret,
                 doca_error_get_descr(ret));
        goto free_ids;
    }

free_ids:
    free(ids);
    return ret ? -1 : 0;
}

static int
shared_mtr_flow_id_alloc(struct doca_eswitch_ctx *esw_ctx,
                         struct rte_flow_error *error)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t id;

    if (!esw_ctx->shared_mtr_flow_id_pool) {
        if (error) {
            error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
            error->message = "Could not allocate meter flow ID, id-pool is not"
                             " initialized";
        }
        goto err;
    }

    if (!id_fpool_new_id(esw_ctx->shared_mtr_flow_id_pool, tid, &id)) {
        VLOG_ERR("Failed to alloc a new shared meter flow id");
        if (error) {
            error->type = RTE_FLOW_ERROR_TYPE_UNSPECIFIED;
            error->message = "Failed to alloc a new shared meter flow id";
        }
        goto err;
    }

    return id;
err:
    if (error) {
        ovs_assert(error->message);
        ovs_assert(error->type);
    }
    return -1;
}

static void
shared_mtr_flow_id_free(struct doca_eswitch_ctx *esw_ctx, uint32_t id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(esw_ctx->shared_mtr_flow_id_pool, tid, id);
}

static void
shared_mtr_flow_id_pool_init(struct doca_eswitch_ctx *ctx)
{
    uint32_t base_id;

    base_id = SHARED_MTR_N_IDS * ctx->esw_id + MIN_SHARED_MTR_FLOW_ID;
    ctx->shared_mtr_flow_id_pool = id_fpool_create(netdev_offload_thread_nb(),
                                                   base_id, SHARED_MTR_N_IDS);
}

static int
doca_bind_shared_meters(struct doca_eswitch_ctx *ctx)
{
    struct doca_flow_shared_resource_cfg dummy_cfg = {
        .domain = DOCA_FLOW_PIPE_DOMAIN_DEFAULT,
        .meter_cfg.limit_type = DOCA_FLOW_METER_LIMIT_TYPE_BYTES,
        .meter_cfg.cir = 125000,
        .meter_cfg.cbs = 12500,
    };
    uint32_t ids[OVS_DOCA_MAX_METERS_PER_ESW];
    int i, id, ret;

    /* DOCA allows meter IDs to start from 0, but it's problematic to have a
     * meter with ID 0 because in such case it will be impossible to disable
     * shared meter in doca_flow_monitor struct later, so meter with ID 0 is
     * not configured and not bound to avoid this issue.
     *
     * Total number of shared meters is OVS_DOCA_MAX_METERS_PER_ESW-1 because
     * meter ID 0 is not used.
     */
    for (i = 0, id = 1; i < OVS_DOCA_MAX_METERS_PER_ESW - 1; i++, id++) {
        ids[i] = ovs_doca_meter_id(id, ctx->esw_id);
        /* DOCA will fail to bind a shared meter if it's unconfigured, which is
         * a bug, so a dummy configuration is used as a W/A; actual meter
         * configuration will be set by the user when OVS meter is added with
         * `ovs-ofctl add-meter` command.
         */
        ret = doca_flow_shared_resource_cfg(DOCA_FLOW_SHARED_RESOURCE_METER,
                                            ids[i], &dummy_cfg);
        if (ret != DOCA_SUCCESS) {
            VLOG_ERR("Failed to init shared meter (id %d), err %d - %s",
                    ids[i], ret, doca_error_get_descr(ret));
            return -1;
        }
    }

    ret = doca_flow_shared_resources_bind(DOCA_FLOW_SHARED_RESOURCE_METER, ids,
                                          OVS_DOCA_MAX_METERS_PER_ESW - 1,
                                          ctx->esw_port);
    if (ret != DOCA_SUCCESS) {
        VLOG_ERR("Shared meters binding failed, ids %d-%d, err %d - %s",
                 ids[0], ids[OVS_DOCA_MAX_METERS_PER_ESW - 2], ret,
                 doca_error_get_descr(ret));
        return -1;
    }

    return 0;
}

static int
doca_post_meter_pipe_init(struct netdev *netdev, struct doca_eswitch_ctx *ctx)
{
    struct doca_ctl_pipe_ctx *pipe_ctx;

    pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, POSTMETER_TABLE_ID);
    if (!pipe_ctx) {
        return -1;
    }
    ctx->post_meter_pipe_ctx = pipe_ctx;

    return 0;
}

static struct offload_metadata *doca_eswitch_md;

static void
doca_eswitch_ctx_uninit(void *ctx_)
{
    struct doca_eswitch_ctx *ctx = ctx_;

    atomic_store_explicit(&ovs_doca_eswitch_active_ids[ctx->esw_id], false,
                          memory_order_release);

    /* The fixed rule insertions were counted in the counters of
     * the netdev that issued the eswitch context init.
     *
     * Destroying the fixed rule is done only when the last netdev
     * using this eswitch context is being removed.
     *
     * We cannot keep track of the original init netdev without inducing
     * a circular dependency.
     *
     * So remove the fixed rules without counting the deletions
     * in the uninit netdev. As all netdevs related to this eswitch
     * are meant to be removed after this, the original counts will
     * have been removed once the uninit has finished.
     */
    doca_ctl_pipe_ctx_unref(ctx->post_meter_pipe_ctx);

    if (ctx->shared_ct_counter_id_pool) {
        doca_ct_zones_uninit(NULL, ctx);
        doca_ct_pipes_destroy(ctx);
    }

    doca_ctl_pipe_ctx_unref(ctx->root_pipe_ctx);
    if (ctx->gnv_opt_parser.parser) {
        doca_flow_parser_geneve_opt_destroy(ctx->gnv_opt_parser.parser);
        ctx->gnv_opt_parser.parser = NULL;
        if (ovsthread_once_start(&ctx->gnv_opt_parser.once)) {
            ovsthread_once_reset(&ctx->gnv_opt_parser.once);
        }
        ovs_mutex_destroy(&ctx->gnv_opt_parser.once.mutex);
    }
    ctx->root_pipe_ctx = NULL;
    /* DOCA doesn't provide an api to unbind shared counters
     * and they will remain bound until the port is destroyed.
     */
    if (ctx->shared_counter_id_pool) {
        id_fpool_destroy(ctx->shared_counter_id_pool);
        ctx->shared_counter_id_pool = NULL;
    }
    if (ctx->shared_ct_counter_id_pool) {
        id_fpool_destroy(ctx->shared_ct_counter_id_pool);
        ctx->shared_ct_counter_id_pool = NULL;
    }
    if (ctx->shared_mtr_flow_id_pool) {
        /* DOCA doesn't provide an api to unbind shared meters
         * and they will remain bound until the port is destroyed.
         */
        id_fpool_destroy(ctx->shared_mtr_flow_id_pool);
        ctx->shared_mtr_flow_id_pool = NULL;
    }
    if (esw_id_pool) {
        id_fpool_free_id(esw_id_pool, 0, ctx->esw_id);
    }

    doca_hash_pipe_ctx_uninit(ctx->hash_pipe_ctx);
    ctx->esw_port = NULL;
}

static int
doca_eswitch_ctx_init(void *ctx_, void *arg_, uint32_t id OVS_UNUSED)
{
    struct netdev *netdev = (struct netdev *) arg_;
    struct netdev_offload_dpdk_data *data;
    struct doca_eswitch_ctx *ctx = ctx_;
    struct doca_flow_port *doca_port;

    /* Set the esw-ctx reference early.
     * It is required by some inner calls.
     * Beware that is it incomplete and should be handled
     * with care. */
    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    data->eswitch_ctx = ctx;

    /* Write the constant offsets of each async entries of the eswitch,
     * used to back reference this context from any entry. */
    for (unsigned int qid = 0; qid < MAX_OFFLOAD_QUEUE_NB; qid++) {
        for (unsigned int idx = 0; idx < OVS_DOCA_QUEUE_DEPTH; idx++) {
            ctx->async_state[qid].entries[idx].index = idx;
        }
    }

    for (unsigned int qid = 0; qid < MAX_OFFLOAD_QUEUE_NB; qid++) {
        ovs_list_init(&ctx->resized_pipe_lists[qid]);
    }

    ctx->root_pipe_ctx = doca_ctl_pipe_ctx_ref(netdev, 0);
    if (ctx->root_pipe_ctx == NULL) {
        goto error;
    }

    shared_counter_id_pool_init(ctx);
    shared_mtr_flow_id_pool_init(ctx);

    doca_port = netdev_dpdk_doca_port_get(netdev);
    ctx->esw_port = doca_flow_port_switch_get(doca_port);
    ctx->gnv_opt_parser.once =
        (struct ovsthread_once) OVSTHREAD_ONCE_INITIALIZER;

    if (ctx->shared_ct_counter_id_pool) {
        if (doca_ct_pipes_init(netdev, ctx)) {
            goto error;
        }

        if (doca_ct_zones_init(netdev, ctx)) {
            goto error;
        }
    }

    ovs_assert(ctx->shared_counter_id_pool);
    if (doca_bind_shared_cntrs(ctx)) {
        goto error;
    }

    if (doca_bind_shared_meters(ctx)) {
        goto error;
    }

    if (doca_post_meter_pipe_init(netdev, ctx)) {
        goto error;
    }

    ctx->hash_pipe_ctx = doca_hash_pipe_ctx_init(netdev);
    if (!ctx->hash_pipe_ctx) {
        goto error;
    }
    ovs_refcount_init(&ctx->pipe_resizing);
    atomic_store_explicit(&ovs_doca_eswitch_active_ids[ctx->esw_id], true,
                          memory_order_release);

    return 0;

error:
    /* Roll-back esw-ctx access on error. */
    data->eswitch_ctx = NULL;
    VLOG_ERR("%s: Failed to init eswitch %d",
             netdev_get_name(netdev),
             netdev_dpdk_get_esw_mgr_port_id(netdev));
    doca_eswitch_ctx_uninit(ctx);
    return -1;
}

static struct ds *
dump_doca_eswitch(struct ds *s, void *key_, void *ctx_, void *arg_ OVS_UNUSED)
{
    struct doca_flow_port *esw_port = key_;
    struct doca_eswitch_ctx *ctx = ctx_;

    if (ctx) {
        ds_put_format(s, "ct_zone_rules_array=%p",
                      ctx->zone_cls_flows);
    }
    ds_put_format(s, "esw_port=%p, ", esw_port);

    return s;
}

static void
doca_eswitch_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .priv_size = sizeof(struct doca_eswitch_ctx),
            .priv_init = doca_eswitch_ctx_init,
            .priv_uninit = doca_eswitch_ctx_uninit,
        };

        /* Only one thread (main) handles the eswitch offload metadata. */
        doca_eswitch_md = offload_metadata_create(1, "doca_eswitch",
                                                  sizeof(struct doca_flow_port *),
                                                  dump_doca_eswitch, params);

        doca_ct_offload_enabled = conntrack_offload_size() > 0;

        ovsthread_once_done(&init_once);
    }
}

/* Get the current eswitch context for this netdev,
 * /!\ without taking a reference, and without creating it!
 * The eswitch context must have been initialized once
 * beforehand using 'doca_eswitch_ctx_ref()' for this netdev.
 */
static struct doca_eswitch_ctx *
doca_eswitch_ctx_get(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    return data->eswitch_ctx;
}

static struct doca_eswitch_ctx *
doca_eswitch_ctx_ref(struct netdev *netdev)
{
    struct doca_flow_port *doca_port = netdev_dpdk_doca_port_get(netdev);
    struct doca_flow_port *esw_port;

    esw_port = doca_flow_port_switch_get(doca_port);

    doca_eswitch_init();
    return offload_metadata_priv_get(doca_eswitch_md, &esw_port, netdev,
                                     NULL, true);
}

static void
doca_eswitch_ctx_unref(struct doca_eswitch_ctx *ctx)
{
    offload_metadata_priv_unref(doca_eswitch_md,
                                netdev_offload_thread_id(),
                                ctx);
}

static void
dpdk_offload_doca_aux_tables_uninit(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;

    if (!netdev_dpdk_is_ethdev(netdev)) {
        return;
    }

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);

    doca_eswitch_ctx_unref(data->eswitch_ctx);
}

static int
dpdk_offload_doca_aux_tables_init(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;
    struct doca_eswitch_ctx *ctx;

    if (!netdev_dpdk_is_ethdev(netdev)) {
        return 0;
    }

    ctx = doca_eswitch_ctx_ref(netdev);
    if (!ctx) {
        VLOG_ERR("%s: Failed to get doca eswitch ctx", netdev_get_name(netdev));
        return -1;
    }

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    data->eswitch_ctx = ctx;

    return 0;
}

static void
log_conn_rule(uint32_t group,
              enum ct_nw_type nw_type,
              enum ct_tp_type tp_type,
              struct doca_flow_match *dspec,
              struct doca_flow_actions *dacts)
{
    struct doca_flow_header_format *dhdr;
    uint16_t sport, dport;
    struct ds s;

    if (VLOG_DROP_DBG(&rl)) {
        return;
    }

    ds_init(&s);

    dhdr = &dspec->outer;

    if (tp_type == CT_TP_TCP) {
        sport = dhdr->tcp.l4_port.src_port;
        dport = dhdr->tcp.l4_port.dst_port;
    } else {
        sport = dhdr->udp.l4_port.src_port;
        dport = dhdr->udp.l4_port.dst_port;
    }

    if (nw_type == CT_NW_IP4) {
        ds_put_format(&s, IP_FMT":%"PRIu16"->"IP_FMT":%"PRIu16,
                      IP_ARGS(dhdr->ip4.src_ip), ntohs(sport),
                      IP_ARGS(dhdr->ip4.dst_ip), ntohs(dport));
    } else {
        char saddr_str[INET6_ADDRSTRLEN];
        char daddr_str[INET6_ADDRSTRLEN];
        struct in6_addr src, dst;

        memcpy(&src, &dhdr->ip6.src_ip, sizeof src);
        memcpy(&dst, &dhdr->ip6.dst_ip, sizeof src);
        ipv6_string_mapped(saddr_str, &src);
        ipv6_string_mapped(daddr_str, &dst);

        ds_put_format(&s, "%s:%"PRIu16"->%s:%"PRIu16,
                      saddr_str, ntohs(sport),
                      daddr_str, ntohs(dport));
    }

    ds_put_format(&s, " zone_map=%d",
                  doca_get_reg_val(&dspec->meta, REG_FIELD_CT_ZONE));

    /* CT MARK */
    ds_put_format(&s, " mark=0x%08x",
                  doca_get_reg_val(&dacts->meta, REG_FIELD_CT_MARK));

    /* CT LABEL */
    ds_put_format(&s, " label=0x%08x",
                  doca_get_reg_val(&dacts->meta, REG_FIELD_CT_LABEL_ID));

    /* CT STATE */
    ds_put_format(&s, " state=0x%02x",
                  doca_get_reg_val(&dacts->meta, REG_FIELD_CT_STATE));

    /* CT CTX */
    ds_put_format(&s, " ctx=0x%02x",
                  doca_get_reg_val(&dacts->meta, REG_FIELD_CT_CTX));

    dhdr = &dacts->outer;

    if (group == CTNAT_TABLE_ID) {
        ds_put_format(&s, " NAT: ");
        if (tp_type == CT_TP_TCP) {
            sport = dhdr->tcp.l4_port.src_port;
            dport = dhdr->tcp.l4_port.dst_port;
        } else {
            sport = dhdr->udp.l4_port.src_port;
            dport = dhdr->udp.l4_port.dst_port;
        }
        ds_put_format(&s, IP_FMT":%"PRIu16"->"IP_FMT":%"PRIu16,
                      IP_ARGS(dhdr->ip4.src_ip), ntohs(sport),
                      IP_ARGS(dhdr->ip4.dst_ip), ntohs(dport));
    }

    VLOG_DBG("conn create: %s", ds_cstr(&s));

    ds_destroy(&s);
}

static struct doca_split_prefix_ctx *
doca_insert_ip6_prefix_conn(struct netdev *netdev,
                            const struct ct_match *ct_match,
                            uint32_t ct_match_zone_id,
                            enum ct_tp_type tp_type,
                            uint32_t *prefix_id)
{
    struct doca_split_prefix_key prefix_key;
    struct doca_split_prefix_ctx *pctx;
    struct doca_split_prefix_arg args;
    struct doca_flow_match *dspec;
    struct doca_eswitch_ctx *ctx;

    memset(&prefix_key, 0, sizeof(prefix_key));
    dspec = &prefix_key.spec;

    /* IPv6 CT prefix match: port_id + zone + src_ip + next_proto. */
    doca_set_reg_val(&dspec->meta, REG_FIELD_CT_ZONE, ct_match_zone_id);
    dspec->parser_meta.port_meta = netdev_dpdk_get_port_id(netdev);

    dspec->outer.ip6.next_proto = tp_type == CT_TP_TCP ?
                                  IPPROTO_TCP :
                                  IPPROTO_UDP;
    memcpy(dspec->outer.ip6.src_ip, &ct_match->key.src.addr.ipv6,
           sizeof dspec->outer.ip6.src_ip);

    ctx = doca_eswitch_ctx_get(netdev);
    args.netdev = netdev;
    args.spec = dspec;
    args.mask = &prefix_key.mask;
    args.prefix_pipe_type = DOCA_FLOW_PIPE_BASIC;
    args.prefix_pipe = ctx->ct_ip6_prefix.pipe;
    args.suffix_pipe = ctx->ct_pipes[CT_NW_IP6][tp_type].pipe;

    prefix_key.prefix_pipe = args.prefix_pipe;

    pctx = split_prefix_ctx_ref(&prefix_key, &args, prefix_id);
    if (!pctx) {
        VLOG_WARN_RL(&rl, "%s: Failed to associate a CT IPv6 prefix with a "
                     "prefix ID", netdev_get_name(netdev));
        return NULL;
    }

    return pctx;
}

static int
dpdk_offload_doca_insert_conn(struct netdev *netdev,
                              struct ct_flow_offload_item ct_offload[1],
                              uint32_t ct_match_zone_id,
                              uint32_t ct_action_label_id,
                              struct indirect_ctx *shared_count_ctx,
                              uint32_t ct_miss_ctx_id,
                              struct flow_item *fi)
{
    struct doca_split_prefix_ctx *pctx = NULL;
    struct doca_flow_header_format *dhdr;
    const struct ct_match *ct_match;
    struct doca_flow_actions dacts;
    struct doca_flow_monitor dmon;
    struct doca_flow_match dspec;
    struct doca_eswitch_ctx *ctx;
    struct doca_flow_pipe *pipe;
    struct rte_flow_error error;
    enum ct_nw_type nw_type;
    enum ct_tp_type tp_type;
    unsigned int prefix_id;
    uint32_t ct_state_spec;
    unsigned int queue_id;
    bool is_ct;

    ct_match = &ct_offload->ct_match;

    tp_type = ct_match->key.nw_proto == IPPROTO_TCP ? CT_TP_TCP : CT_TP_UDP;
    if (ct_match->key.dl_type == htons(ETH_TYPE_IP)) {
        nw_type = CT_NW_IP4;
    } else if (ct_match->key.dl_type == htons(ETH_TYPE_IPV6)) {
        nw_type = CT_NW_IP6;
        /* No IPv6 conn request should be generated if not enabled. */
        ovs_assert(conntrack_offload_ipv6_is_enabled());
    } else {
        VLOG_DBG_RL(&rl, "Unsupported CT network type.");
        return -1;
    }

    memset(&dspec.meta, 0, sizeof dspec.meta);

    dhdr = &dspec.outer;

    /* IPv4 */
    if (nw_type == CT_NW_IP4) {
        dhdr->ip4.src_ip = ct_match->key.src.addr.ipv4;
        dhdr->ip4.dst_ip = ct_match->key.dst.addr.ipv4;
        dhdr->ip4.next_proto = ct_match->key.nw_proto;
    /* IPv6 */
    } else {
        /* ip6 src match will be in the prefix rule while dst match in the
         * suffix rule
         */
        pctx = doca_insert_ip6_prefix_conn(netdev, ct_match, ct_match_zone_id,
                                           tp_type, &prefix_id);
        if (!pctx) {
            return -1;
        }

        memcpy(dhdr->ip6.dst_ip, &ct_match->key.dst.addr.ipv6,
               sizeof dhdr->ip6.dst_ip);
        doca_set_reg_val(&dspec.meta, REG_FIELD_SCRATCH, prefix_id);
    }

    if (tp_type == CT_TP_TCP) {
        dhdr->tcp.l4_port.src_port = ct_match->key.src.port;
        dhdr->tcp.l4_port.dst_port = ct_match->key.dst.port;
    } else {
        dhdr->udp.l4_port.src_port = ct_match->key.src.port;
        dhdr->udp.l4_port.dst_port = ct_match->key.dst.port;
    }

    doca_set_reg_val(&dspec.meta, REG_FIELD_CT_ZONE, ct_match_zone_id);

    dspec.parser_meta.port_meta = netdev_dpdk_get_port_id(netdev);

    dhdr = &dacts.outer;

    /* Common part for all CT, plain and NAT. */

    if (nw_type == CT_NW_IP4) {
        dhdr->l3_type = DOCA_FLOW_L3_TYPE_IP4;
        dhdr->ip4.src_ip = ct_match->key.src.addr.ipv4;
        dhdr->ip4.dst_ip = ct_match->key.dst.addr.ipv4;
    } else {
        dhdr->l3_type = DOCA_FLOW_L3_TYPE_IP6;
        memcpy(&dhdr->ip6.src_ip, &ct_match->key.src.addr.ipv6,
               sizeof dhdr->ip6.src_ip);
        memcpy(&dhdr->ip6.dst_ip, &ct_match->key.dst.addr.ipv6,
               sizeof dhdr->ip6.dst_ip);
    }

    if (tp_type == CT_TP_TCP) {
        dhdr->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
        dhdr->tcp.l4_port.src_port = ct_match->key.src.port;
        dhdr->tcp.l4_port.dst_port = ct_match->key.dst.port;
    } else {
        dhdr->l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
        dhdr->udp.l4_port.src_port = ct_match->key.src.port;
        dhdr->udp.l4_port.dst_port = ct_match->key.dst.port;
    }

    /* For NAT translate the relevant fields. */
    if (ct_offload->nat.mod_flags) {
        is_ct = false;
        if (nw_type == CT_NW_IP4) {
            if (ct_offload->nat.mod_flags & NAT_ACTION_SRC) {
                dhdr->ip4.src_ip = ct_offload->nat.key.src.addr.ipv4;
            } else if (ct_offload->nat.mod_flags & NAT_ACTION_DST) {
                dhdr->ip4.dst_ip = ct_offload->nat.key.dst.addr.ipv4;
            }
        } else {
            if (ct_offload->nat.mod_flags & NAT_ACTION_SRC) {
                memcpy(&dhdr->ip6.src_ip, &ct_offload->nat.key.src.addr.ipv6,
                       sizeof dhdr->ip6.src_ip);
            } else if (ct_offload->nat.mod_flags & NAT_ACTION_DST) {
                memcpy(&dhdr->ip6.dst_ip, &ct_offload->nat.key.dst.addr.ipv6,
                       sizeof dhdr->ip6.dst_ip);
            }
        }
        if (ct_offload->nat.mod_flags & NAT_ACTION_SRC_PORT) {
            if (tp_type == CT_TP_TCP) {
                dhdr->tcp.l4_port.src_port = ct_offload->nat.key.src.port;
            } else {
                dhdr->udp.l4_port.src_port = ct_offload->nat.key.src.port;
            }
        } else if (ct_offload->nat.mod_flags & NAT_ACTION_DST_PORT) {
            if (tp_type == CT_TP_TCP) {
                dhdr->tcp.l4_port.dst_port = ct_offload->nat.key.dst.port;
            } else {
                dhdr->udp.l4_port.dst_port = ct_offload->nat.key.dst.port;
            }
        }
    } else {
        is_ct = true;
    }

    memset(&dmon, 0, sizeof dmon);
    dmon.counter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
    dmon.shared_counter.shared_counter_id = shared_count_ctx->res_id;

    memset(&dacts.meta, 0, sizeof dacts.meta);

    /* CT MARK */
    doca_set_reg_val(&dacts.meta, REG_FIELD_CT_MARK, ct_offload->mark_key);

    /* CT LABEL */
    doca_set_reg_val(&dacts.meta, REG_FIELD_CT_LABEL_ID, ct_action_label_id);

    /* CT STATE */
    ct_state_spec = ct_offload->ct_state;
    /* If any of the NAT bits is set, set both of them.
     * As the NAT action is executed, the 5-tuple of the packet is modified.
     * A revisit with the post-NAT tuple would miss. */
    if (ct_offload->ct_state & OVS_CS_F_NAT_MASK) {
        ct_state_spec |= OVS_CS_F_NAT_MASK;
    }
    doca_set_reg_val(&dacts.meta, REG_FIELD_CT_STATE, ct_state_spec);

    /* CT CTX */
    doca_set_reg_val(&dacts.meta, REG_FIELD_CT_CTX, ct_miss_ctx_id);

    ctx = doca_eswitch_ctx_get(netdev);
    queue_id = netdev_offload_thread_id();
    memset(fi, 0, sizeof *fi);

    log_conn_rule(is_ct ? CT_TABLE_ID : CTNAT_TABLE_ID, nw_type, tp_type,
                  &dspec, &dacts);

    dacts.action_idx = 0;
    pipe = ctx->ct_pipes[nw_type][tp_type].pipe;
    if (create_doca_basic_flow_entry(netdev, queue_id, pipe, &dspec,
                                     &dacts, &dmon, NULL, &fi->doh[0],
                                     &error)) {
        VLOG_WARN_RL(&rl, "%s: Failed to create ct entry: Error %d (%s)",
                     netdev_get_name(netdev), error.type, error.message);
        return -1;
    }
    fi->doh[0].valid = true;
    fi->doh[0].dfh.flow_res.split_ctx[0].curr_split_ctx = pctx;

    return 0;
}

static int
dpdk_offload_doca_packet_hw_entropy(struct netdev *netdev,
                                    struct dp_packet *packet,
                                    uint16_t *entropy)
{
    struct doca_flow_entropy_format header;
    struct doca_flow_port *port;
    uint8_t ip_proto = 0;
    ovs_be16 eth_proto;
    doca_error_t err;
    void *nh;
    void *l4;

    parse_tcp_flags(packet, &eth_proto, NULL, NULL);
    nh = dp_packet_l3(packet);
    l4 = dp_packet_l4(packet);

    if (!netdev_dpdk_is_ethdev(netdev)) {
        return -1;
    }

    memset(&header, 0, sizeof header);

    if (eth_proto == htons(ETH_TYPE_IP)) {
        struct ip_header *ip = nh;

        header.l3_type = DOCA_FLOW_L3_TYPE_IP4;
        header.ip4.src_ip = get_16aligned_be32(&ip->ip_src);
        header.ip4.dst_ip = get_16aligned_be32(&ip->ip_dst);
        ip_proto = ip->ip_proto;
    } else if (eth_proto == htons(ETH_TYPE_IPV6)) {
        struct ovs_16aligned_ip6_hdr *ip6 = nh;

        header.l3_type = DOCA_FLOW_L3_TYPE_IP6;
        memcpy(header.ip6.src_ip, &ip6->ip6_src, sizeof(ovs_be32[4]));
        memcpy(header.ip6.dst_ip, &ip6->ip6_dst, sizeof(ovs_be32[4]));
        ip_proto = ip6->ip6_nxt;
    } else {
        header.l3_type = DOCA_FLOW_L3_TYPE_NONE;
    }

    if (ip_proto == IPPROTO_TCP) {
        struct tcp_header *tcp = l4;

        header.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
        header.transport.src_port = tcp->tcp_src;
        header.transport.dst_port = tcp->tcp_dst;
    } else if (ip_proto == IPPROTO_UDP) {
        struct udp_header *udp = l4;

        header.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
        header.transport.src_port = udp->udp_src;
        header.transport.dst_port = udp->udp_dst;
    } else if (ip_proto == IPPROTO_ICMP) {
        header.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_ICMP;
    } else if (ip_proto == IPPROTO_ICMPV6) {
        header.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_ICMP6;
    } else {
        header.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_NONE;
    }

    port = netdev_dpdk_doca_port_get(netdev);
    err = doca_flow_port_calc_entropy(port, &header, entropy);
    if (err != DOCA_SUCCESS) {
        return -1;
    }

    return 0;
}

struct dpdk_offload_api dpdk_offload_api_doca = {
    .upkeep = dpdk_offload_doca_upkeep,
    .create = dpdk_offload_doca_create,
    .destroy = dpdk_offload_doca_destroy,
    .query_count = dpdk_offload_doca_query_count,
    .get_packet_recover_info = dpdk_offload_doca_get_pkt_recover_info,
    .insert_conn = dpdk_offload_doca_insert_conn,
    .reg_fields = dpdk_offload_doca_get_reg_fields,
    .update_stats = dpdk_offload_doca_update_stats,
    .aux_tables_init = dpdk_offload_doca_aux_tables_init,
    .aux_tables_uninit = dpdk_offload_doca_aux_tables_uninit,
    .shared_create = dpdk_offload_doca_shared_create,
    .shared_destroy = dpdk_offload_doca_shared_destroy,
    .shared_query = dpdk_offload_doca_shared_query,
    .packet_hw_hash = dpdk_offload_doca_packet_hw_hash,
    .packet_hw_entropy = dpdk_offload_doca_packet_hw_entropy,
};
