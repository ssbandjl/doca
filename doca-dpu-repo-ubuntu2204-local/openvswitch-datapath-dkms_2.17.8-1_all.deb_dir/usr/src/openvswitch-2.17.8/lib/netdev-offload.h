/*
 * Copyright (c) 2008, 2009, 2010, 2011, 2012, 2013 Nicira, Inc.
 * Copyright (c) 2019 Samsung Electronics Co.,Ltd.
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

#ifndef NETDEV_OFFLOAD_H
#define NETDEV_OFFLOAD_H 1

#include "conntrack-offload.h"
#include "openvswitch/netdev.h"
#include "openvswitch/types.h"
#include "dp-packet.h"
#include "dpif.h"
#include "ovs-atomic.h"
#include "ovs-rcu.h"
#include "ovs-thread.h"
#include "packets.h"
#include "flow.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define DEFAULT_OFFLOAD_THREAD_NB 1
#define MAX_OFFLOAD_METERS 4

struct netdev_class;
struct netdev_rxq;
struct netdev_saved_flags;
struct ofpbuf;
struct in_addr;
struct in6_addr;
struct smap;
struct sset;
struct ovs_action_push_tnl;


/* Offload-capable (HW) netdev information */
struct netdev_hw_info {
    bool oor;		/* Out of Offload Resources ? */
    atomic_bool miss_api_supported;  /* hw_miss_packet_recover() supported.*/
    int offload_count;  /* Pending (non-offloaded) flow count */
    int pending_count;  /* Offloaded flow count */
    OVSRCU_TYPE(void *) offload_data; /* Offload metadata. */
};

enum hw_info_type {
    HW_INFO_TYPE_OOR = 1,		/* OOR state */
    HW_INFO_TYPE_PEND_COUNT = 2,	/* Pending(non-offloaded) flow count */
    HW_INFO_TYPE_OFFL_COUNT = 3		/* Offloaded flow count */
};

/* Attributes for offload UFIDs, generated by uuid_set_bits_v4(uuid, attr). */
enum offload_uuid_attr {
    UUID_ATTR_0,	/* Reserved for non-offloads. */
    UUID_ATTR_1,
    UUID_ATTR_2,
    UUID_ATTR_3,
    UUID_ATTR_4,
};

struct netdev_flow_dump {
    struct netdev *netdev;
    odp_port_t port;
    bool terse;
    struct nl_dump *nl_dump;
};

/* Generic statistics of the offload provider of a netdev.
 * It is not related to the 'offload_count' or 'pending_count'
 * stored within the 'netdev_hw_info' and managed entirely
 * by the upcall handler. */
struct netdev_offload_stats {
    uint64_t n_inserted;
    uint64_t n_flows;
    uint64_t n_conns;
};

static inline void
netdev_offload_stats_add(struct netdev_offload_stats *dst,
                         struct netdev_offload_stats src)
{
    dst->n_inserted = ovs_u64_safeadd(dst->n_inserted, src.n_inserted);
    dst->n_flows = ovs_u64_safeadd(dst->n_flows, src.n_flows);
    dst->n_conns = ovs_u64_safeadd(dst->n_conns, src.n_conns);
}

#define OFFLOAD_FLOWS_COUNTER_KEY_SIZE  E2E_CACHE_MAX_TRACE

/* This is a maximal required buffer size for output argument
 * of netdev_flow_counter_key_to_string().
 */
#define OFFLOAD_FLOWS_COUNTER_KEY_STRING_SIZE \
    (OFFLOAD_FLOWS_COUNTER_KEY_SIZE * 35 + 3)

OVS_ASSERT_PACKED(struct flows_counter_key,
    union {
        uintptr_t ptr_key;
        ovs_u128  ufid_key[OFFLOAD_FLOWS_COUNTER_KEY_SIZE];
    };
);

/* Flow offloading. */
struct offload_info {
    bool recirc_id_shared_with_tc;  /* Indicates whever tc chains will be in
                                     * sync with datapath recirc ids. */

    /*
     * The flow mark id assigened to the flow. If any pkts hit the flow,
     * it will be in the pkt meta data.
     */
    uint32_t flow_mark;

    bool tc_modify_flow_deleted; /* Indicate the tc modify flow put success
                                  * to delete the original flow. */
    odp_port_t orig_in_port; /* Originating in_port for tnl flows. */
    /* Indicates if flow is for e2e cache*/
    bool is_e2e_cache_flow;
    bool is_ct_conn;

    uintptr_t ct_counter_key;
    struct flows_counter_key flows_counter_key;
    uint32_t police_ids[MAX_OFFLOAD_METERS]; /* police ids of the offloaded
                                              * meters in the flow */
};

DECLARE_EXTERN_PER_THREAD_DATA(unsigned int, netdev_offload_thread_id);
#define MAX_OFFLOAD_THREAD_NB 10

unsigned int netdev_offload_thread_nb(void);
unsigned int netdev_offload_thread_init(unsigned int);
unsigned int netdev_offload_ufid_to_thread_id(const ovs_u128 ufid);

static inline unsigned int
netdev_offload_thread_id(void)
{
    unsigned int id = *netdev_offload_thread_id_get();

    if (OVS_UNLIKELY(id == OVSTHREAD_ID_UNSET)) {
        id = netdev_offload_thread_init(OVSTHREAD_ID_UNSET);
    }

    return id;
}

#define INVALID_FLOW_MARK 0
#define HAIRPIN_FLOW_MARK 1
#define MIN_FLOW_MARK 2
#define MAX_FLOW_MARK (UINT32_MAX - 1)
#define NB_FLOW_MARK (MAX_FLOW_MARK - MIN_FLOW_MARK + 1)

int netdev_flow_flush(struct netdev *);
int netdev_flow_dump_create(struct netdev *, struct netdev_flow_dump **dump,
                            bool terse);
int netdev_flow_dump_destroy(struct netdev_flow_dump *);
bool netdev_flow_dump_next(struct netdev_flow_dump *, struct match *,
                          struct nlattr **actions, struct dpif_flow_stats *,
                          struct dpif_flow_attrs *, ovs_u128 *ufid,
                          struct ofpbuf *rbuffer, struct ofpbuf *wbuffer);
int netdev_flow_put(struct netdev *, struct match *, struct nlattr *actions,
                    size_t actions_len, const ovs_u128 *,
                    struct offload_info *, struct dpif_flow_stats *);
int netdev_hw_miss_packet_recover(struct netdev *, struct dp_packet *,
                                  uint8_t *, struct dpif_sflow_attr *);
int netdev_flow_get(struct netdev *, struct match *, struct nlattr **actions,
                    const ovs_u128 *, struct dpif_flow_stats *,
                    struct dpif_flow_attrs *, struct ofpbuf *wbuffer,
                    long long now);
int netdev_flow_del(struct netdev *, const ovs_u128 *,
                    struct dpif_flow_stats *);
int netdev_init_flow_api(struct netdev *);
int netdev_ct_counter_query(struct netdev *, uintptr_t, long long, long long,
                            struct dpif_flow_stats *);
void netdev_uninit_flow_api(struct netdev *);
uint32_t netdev_get_block_id(struct netdev *);
int netdev_get_hw_info(struct netdev *, int);
void netdev_set_hw_info(struct netdev *, int, int);
bool netdev_any_oor(void);
bool netdev_is_flow_api_enabled(void);
void netdev_set_flow_api_enabled(const struct smap *ovs_other_config);
bool netdev_is_offload_rebalance_policy_enabled(void);
int netdev_flow_get_n_flows(struct netdev *netdev, uint64_t *n_flows);
int netdev_flow_get_n_offloads(struct netdev *netdev,
                               uint64_t *n_offloads);
int netdev_offload_get_stats(struct netdev *netdev,
                             struct netdev_offload_stats *stats);
bool netdev_is_e2e_cache_enabled(void);
uint32_t netdev_get_e2e_cache_size(void);
bool netdev_is_flow_counter_key_zero(const struct flows_counter_key *);
char *netdev_flow_counter_key_to_string(const struct flows_counter_key *,
                                        char *, size_t);
bool netdev_is_ct_labels_mapping_enabled(void);
bool netdev_is_zone_tables_disabled(void);

int netdev_conn_add(struct netdev *, struct ct_flow_offload_item[1]);
int netdev_conn_del(struct netdev *, struct ct_flow_offload_item[1]);
int netdev_conn_stats(struct netdev *, struct ct_flow_offload_item[1],
                      struct dpif_flow_stats *, struct dpif_flow_attrs *,
                      long long int);

/* Upkeep a single netdev, if supported.
 * If 'quiescing' is true, the calling thread is signaling
 * that this is the last upkeep call before starting to wait
 * on more work.
 */
void netdev_offload_upkeep(struct netdev *netdev, bool quiescing);
/* Upkeep all netdev-offload ports. */
void netdev_ports_upkeep(bool quiescing);

struct dpif_port;
int netdev_ports_insert(struct netdev *, struct dpif_port *);
struct netdev *netdev_ports_get(odp_port_t port, const char *dpif_type);
int netdev_ports_remove(odp_port_t port, const char *dpif_type);
odp_port_t netdev_ifindex_to_odp_port(int ifindex);

/* Make 'netdev' visible for 'netdev_ports_get'.
 * If set to 'false', this netdev will not be returned on lookup.
 */
void netdev_ports_set_visible(struct netdev *netdev, bool visible);

/* For each of the ports with dpif_type, call cb with the netdev and port
 * number of the port, and an opaque user argument.
 * The returned value is used to continue traversing upon false or stop if
 * true.
 */
void netdev_ports_traverse(const char *dpif_type,
                           bool (*cb)(struct netdev *, odp_port_t, void *),
                           void *aux);
struct netdev_flow_dump **netdev_ports_flow_dump_create(
                                        const char *dpif_type,
                                        int *ports,
                                        bool terse);
void netdev_ports_flow_flush(const char *dpif_type);
int netdev_ports_flow_del(const char *dpif_type, const ovs_u128 *ufid,
                          struct dpif_flow_stats *stats);
int netdev_ports_flow_get(const char *dpif_type, struct match *match,
                          struct nlattr **actions,
                          const ovs_u128 *ufid,
                          struct dpif_flow_stats *stats,
                          struct dpif_flow_attrs *attrs,
                          struct ofpbuf *buf);
int netdev_ports_get_n_flows(const char *dpif_type,
                             odp_port_t port_no, uint64_t *n_flows);
uint32_t netdev_offload_flow_mark_alloc(void);
void netdev_offload_flow_mark_free(uint32_t mark);

extern bool netdev_offload_ct_on_ct_nat;

const struct dpif_offload_sflow_attr *dpif_offload_sflow_attr_find(uint32_t id);

int netdev_packet_hw_hash(struct netdev *netdev,
                          struct dp_packet *packet,
                          uint32_t seed,
                          uint32_t *hash);

int netdev_packet_hw_entropy(struct netdev *netdev,
                             struct dp_packet *packet,
                             uint16_t *entropy);

#ifdef  __cplusplus
}
#endif

#endif /* netdev-offload.h */
