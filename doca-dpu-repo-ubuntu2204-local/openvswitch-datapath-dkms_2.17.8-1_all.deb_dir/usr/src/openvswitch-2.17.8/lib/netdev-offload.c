/*
 * Copyright (c) 2008 - 2014, 2016, 2017 Nicira, Inc.
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

#include <config.h>
#include "netdev-offload.h"

#include <errno.h>
#include <inttypes.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cmap.h"
#include "coverage.h"
#include "dpif.h"
#include "dp-packet.h"
#include "openvswitch/dynamic-string.h"
#include "fatal-signal.h"
#include "hash.h"
#include "id-fpool.h"
#include "openvswitch/list.h"
#include "netdev-offload-provider.h"
#include "netdev-provider.h"
#include "netdev-vport.h"
#include "odp-netlink.h"
#include "openflow/openflow.h"
#include "packets.h"
#include "openvswitch/ofp-print.h"
#include "openvswitch/poll-loop.h"
#include "seq.h"
#include "openvswitch/shash.h"
#include "smap.h"
#include "socket-util.h"
#include "sset.h"
#include "svec.h"
#include "openvswitch/vlog.h"
#include "flow.h"
#include "util.h"
#ifdef __linux__
#include "tc.h"
#endif

VLOG_DEFINE_THIS_MODULE(netdev_offload);


static bool netdev_flow_api_enabled = false;
static bool e2e_cache_enabled = false;
static struct id_fpool *flow_mark_pool;
static uint32_t e2e_cache_size = 0;
bool netdev_offload_ct_on_ct_nat = false;
bool ct_labels_mapping = false;
bool disable_zone_tables = false;

static unsigned int offload_thread_nb = DEFAULT_OFFLOAD_THREAD_NB;
DEFINE_EXTERN_PER_THREAD_DATA(netdev_offload_thread_id, OVSTHREAD_ID_UNSET);

/* Protects 'netdev_flow_apis'.  */
static struct ovs_mutex netdev_flow_api_provider_mutex = OVS_MUTEX_INITIALIZER;

/* Contains 'struct netdev_registered_flow_api's. */
static struct cmap netdev_flow_apis = CMAP_INITIALIZER;

struct netdev_registered_flow_api {
    struct cmap_node cmap_node; /* In 'netdev_flow_apis', by flow_api->type. */
    const struct netdev_flow_api *flow_api;

    /* Number of references: one for the flow_api itself and one for every
     * instance of the netdev that uses it. */
    struct ovs_refcount refcnt;
};

static struct netdev_registered_flow_api *
netdev_lookup_flow_api(const char *type)
{
    struct netdev_registered_flow_api *rfa;
    CMAP_FOR_EACH_WITH_HASH (rfa, cmap_node, hash_string(type, 0),
                             &netdev_flow_apis) {
        if (!strcmp(type, rfa->flow_api->type)) {
            return rfa;
        }
    }
    return NULL;
}

/* Registers a new netdev flow api provider. */
int
netdev_register_flow_api_provider(const struct netdev_flow_api *new_flow_api)
    OVS_EXCLUDED(netdev_flow_api_provider_mutex)
{
    int error = 0;

    if (!new_flow_api->init_flow_api) {
        VLOG_WARN("attempted to register invalid flow api provider: %s",
                   new_flow_api->type);
        error = EINVAL;
    }

    ovs_mutex_lock(&netdev_flow_api_provider_mutex);
    if (netdev_lookup_flow_api(new_flow_api->type)) {
        VLOG_WARN("attempted to register duplicate flow api provider: %s",
                   new_flow_api->type);
        error = EEXIST;
    } else {
        struct netdev_registered_flow_api *rfa;

        rfa = xmalloc(sizeof *rfa);
        cmap_insert(&netdev_flow_apis, &rfa->cmap_node,
                    hash_string(new_flow_api->type, 0));
        rfa->flow_api = new_flow_api;
        ovs_refcount_init(&rfa->refcnt);
        VLOG_DBG("netdev: flow API '%s' registered.", new_flow_api->type);
    }
    ovs_mutex_unlock(&netdev_flow_api_provider_mutex);

    return error;
}

/* Unregisters a netdev flow api provider.  'type' must have been previously
 * registered and not currently be in use by any netdevs.  After unregistration
 * netdev flow api of that type cannot be used for netdevs.  (However, the
 * provider may still be accessible from other threads until the next RCU grace
 * period, so the caller must not free or re-register the same netdev_flow_api
 * until that has passed.) */
int
netdev_unregister_flow_api_provider(const char *type)
    OVS_EXCLUDED(netdev_flow_api_provider_mutex)
{
    struct netdev_registered_flow_api *rfa;
    int error;

    ovs_mutex_lock(&netdev_flow_api_provider_mutex);
    rfa = netdev_lookup_flow_api(type);
    if (!rfa) {
        VLOG_WARN("attempted to unregister a flow api provider that is not "
                  "registered: %s", type);
        error = EAFNOSUPPORT;
    } else if (ovs_refcount_unref(&rfa->refcnt) != 1) {
        ovs_refcount_ref(&rfa->refcnt);
        VLOG_WARN("attempted to unregister in use flow api provider: %s",
                  type);
        error = EBUSY;
    } else  {
        cmap_remove(&netdev_flow_apis, &rfa->cmap_node,
                    hash_string(rfa->flow_api->type, 0));
        ovsrcu_postpone(free, rfa);
        error = 0;
    }
    ovs_mutex_unlock(&netdev_flow_api_provider_mutex);

    return error;
}

bool
netdev_flow_api_equals(const struct netdev *netdev1,
                       const struct netdev *netdev2)
{
    const struct netdev_flow_api *netdev_flow_api1 =
        ovsrcu_get(const struct netdev_flow_api *, &netdev1->flow_api);
    const struct netdev_flow_api *netdev_flow_api2 =
        ovsrcu_get(const struct netdev_flow_api *, &netdev2->flow_api);

    return netdev_flow_api1 == netdev_flow_api2;
}

static int
netdev_assign_flow_api(struct netdev *netdev)
{
    struct netdev_registered_flow_api *rfa;
    int ret;

    CMAP_FOR_EACH (rfa, cmap_node, &netdev_flow_apis) {
        ret = rfa->flow_api->init_flow_api(netdev);
        if (ret == EAGAIN) {
            VLOG_INFO("%s: flow API '%s' is not ready. Will try again",
                      netdev_get_name(netdev), rfa->flow_api->type);
            return ret;
        }
        if (!ret) {
            ovs_refcount_ref(&rfa->refcnt);
            atomic_store_relaxed(&netdev->hw_info.miss_api_supported, true);
            ovsrcu_set(&netdev->flow_api, rfa->flow_api);
            VLOG_INFO("%s: Assigned flow API '%s'.",
                      netdev_get_name(netdev), rfa->flow_api->type);
            return 0;
        }
        VLOG_DBG("%s: flow API '%s' is not suitable.",
                 netdev_get_name(netdev), rfa->flow_api->type);
    }
    atomic_store_relaxed(&netdev->hw_info.miss_api_supported, false);
    VLOG_INFO("%s: No suitable flow API found.", netdev_get_name(netdev));

    return EOPNOTSUPP;
}

int
netdev_flow_flush(struct netdev *netdev)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    netdev_offload_upkeep(netdev, false);

    return (flow_api && flow_api->flow_flush)
           ? flow_api->flow_flush(netdev)
           : EOPNOTSUPP;
}

int
netdev_flow_dump_create(struct netdev *netdev, struct netdev_flow_dump **dump,
                        bool terse)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->flow_dump_create)
           ? flow_api->flow_dump_create(netdev, dump, terse)
           : EOPNOTSUPP;
}

int
netdev_flow_dump_destroy(struct netdev_flow_dump *dump)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &dump->netdev->flow_api);

    return (flow_api && flow_api->flow_dump_destroy)
           ? flow_api->flow_dump_destroy(dump)
           : EOPNOTSUPP;
}

bool
netdev_flow_dump_next(struct netdev_flow_dump *dump, struct match *match,
                      struct nlattr **actions, struct dpif_flow_stats *stats,
                      struct dpif_flow_attrs *attrs, ovs_u128 *ufid,
                      struct ofpbuf *rbuffer, struct ofpbuf *wbuffer)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &dump->netdev->flow_api);

    return (flow_api && flow_api->flow_dump_next)
           ? flow_api->flow_dump_next(dump, match, actions, stats, attrs,
                                      ufid, rbuffer, wbuffer)
           : false;
}

int
netdev_flow_put(struct netdev *netdev, struct match *match,
                struct nlattr *actions, size_t act_len,
                const ovs_u128 *ufid, struct offload_info *info,
                struct dpif_flow_stats *stats)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    netdev_offload_upkeep(netdev, false);

    return (flow_api && flow_api->flow_put)
           ? flow_api->flow_put(netdev, match, actions, act_len, ufid,
                                info, stats)
           : EOPNOTSUPP;
}

int
netdev_hw_miss_packet_recover(struct netdev *netdev,
                              struct dp_packet *packet,
                              uint8_t *skip_actions,
                              struct dpif_sflow_attr *sflow_attr)
{
    const struct netdev_flow_api *flow_api;
    bool miss_api_supported;
    int rv;

    atomic_read_relaxed(&netdev->hw_info.miss_api_supported,
                        &miss_api_supported);
    if (!miss_api_supported) {
        return EOPNOTSUPP;
    }

    flow_api = ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);
    if (!flow_api || !flow_api->hw_miss_packet_recover) {
        return EOPNOTSUPP;
    }

    rv = flow_api->hw_miss_packet_recover(netdev, packet, skip_actions,
                                          sflow_attr);
    if (rv == EOPNOTSUPP) {
        /* API unsupported by the port; avoid subsequent calls. */
        atomic_store_relaxed(&netdev->hw_info.miss_api_supported, false);
    }

    packet->orig_netdev = netdev;
    return rv;
}

int
netdev_flow_get(struct netdev *netdev, struct match *match,
                struct nlattr **actions, const ovs_u128 *ufid,
                struct dpif_flow_stats *stats,
                struct dpif_flow_attrs *attrs, struct ofpbuf *buf,
                long long now)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->flow_get)
           ? flow_api->flow_get(netdev, match, actions, ufid,
                                stats, attrs, buf, now)
           : EOPNOTSUPP;
}

int
netdev_flow_del(struct netdev *netdev, const ovs_u128 *ufid,
                struct dpif_flow_stats *stats)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    netdev_offload_upkeep(netdev, false);

    return (flow_api && flow_api->flow_del)
           ? flow_api->flow_del(netdev, ufid, stats)
           : EOPNOTSUPP;
}

int
netdev_conn_add(struct netdev *netdev,
                struct ct_flow_offload_item ct_offload[1])
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    netdev_offload_upkeep(netdev, false);

    return (flow_api && flow_api->conn_add)
           ? flow_api->conn_add(netdev, ct_offload)
           : EOPNOTSUPP;
}

int
netdev_conn_del(struct netdev *netdev,
                struct ct_flow_offload_item ct_offload[1])
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    netdev_offload_upkeep(netdev, false);

    return (flow_api && flow_api->conn_del)
           ? flow_api->conn_del(netdev, ct_offload)
           : EOPNOTSUPP;
}

int
netdev_conn_stats(struct netdev *netdev,
                  struct ct_flow_offload_item ct_offload[1],
                  struct dpif_flow_stats *stats,
                  struct dpif_flow_attrs *attrs,
                  long long int now)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->conn_stats)
           ? flow_api->conn_stats(netdev, ct_offload, stats, attrs, now)
           : EOPNOTSUPP;
}

void
netdev_offload_upkeep(struct netdev *netdev, bool quiescing)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    if (flow_api && flow_api->upkeep) {
        flow_api->upkeep(netdev, quiescing);
    }
}

static struct ovs_list *mark_release_lists;

struct mark_release_item {
    struct ovs_list node;
    long long int timestamp;
    uint32_t mark;
};

static void
mark_delayed_release(uint32_t mark)
{
    struct mark_release_item *item = xzalloc(sizeof *item);
    unsigned int tid = netdev_offload_thread_id();

    item->mark = mark;
    item->timestamp = time_msec();
    ovs_list_push_back(&mark_release_lists[tid], &item->node);
    VLOG_DBG("%s: mark=%d, timestamp=%llu", __func__, item->mark,
             item->timestamp);
}

#define DELAYED_RELEASE_TIMEOUT_MS 1000
/* This timeout is to reserve the flow mark ID for a while after the flow is
 * deleted, so that ID will not be allocated to another flow while packets
 * matching the deleted flow are still in HW queues. We assume this timeout
 * is enough for such packets to already be processed.
 */

static void
do_mark_delayed_release(unsigned int tid)
{
    struct ovs_list *mark_release_list = &mark_release_lists[tid];
    struct mark_release_item *item;
    struct ovs_list *list;
    long long int now;

    now = time_msec();
    while (!ovs_list_is_empty(mark_release_list)) {
        list = ovs_list_front(mark_release_list);
        item = CONTAINER_OF(list, struct mark_release_item, node);
        if (now < item->timestamp + DELAYED_RELEASE_TIMEOUT_MS) {
            break;
        }
        VLOG_DBG("%s: mark=%d, timestamp=%llu, now=%llu", __func__, item->mark,
                 item->timestamp, now);
        id_fpool_free_id(flow_mark_pool, tid, item->mark);
        ovs_list_remove(list);
        free(item);
    }
}

uint32_t
netdev_offload_flow_mark_alloc(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;
    unsigned int tid = netdev_offload_thread_id();
    uint32_t mark;

    if (ovsthread_once_start(&init_once)) {
        unsigned int nb_thread = netdev_offload_thread_nb();
        size_t i;

        /* Haven't initiated yet, do it here */
        flow_mark_pool = id_fpool_create(nb_thread, MIN_FLOW_MARK, NB_FLOW_MARK);
        mark_release_lists = xcalloc(nb_thread, sizeof *mark_release_lists);
        for (i = 0; i < nb_thread; i++) {
            ovs_list_init(&mark_release_lists[i]);
        }

        ovsthread_once_done(&init_once);
    }

    do_mark_delayed_release(tid);
    if (id_fpool_new_id(flow_mark_pool, tid, &mark)) {
        return mark;
    }

    return INVALID_FLOW_MARK;
}

void
netdev_offload_flow_mark_free(uint32_t mark)
{
    mark_delayed_release(mark);
}

int
netdev_flow_get_n_flows(struct netdev *netdev, uint64_t *n_flows)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->flow_get_n_flows)
           ? flow_api->flow_get_n_flows(netdev, n_flows)
           : EOPNOTSUPP;
}

int
netdev_flow_get_n_offloads(struct netdev *netdev, uint64_t *n_flows)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->flow_get_n_offloads)
           ? flow_api->flow_get_n_offloads(netdev, n_flows)
           : EOPNOTSUPP;
}

int
netdev_offload_get_stats(struct netdev *netdev,
                         struct netdev_offload_stats *stats)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->get_stats)
           ? flow_api->get_stats(netdev, stats)
           : EOPNOTSUPP;
}

int
netdev_init_flow_api(struct netdev *netdev)
{
    if (!netdev_is_flow_api_enabled()) {
        return EOPNOTSUPP;
    }

    if (ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api)) {
        return 0;
    }

    return netdev_assign_flow_api(netdev);
}

void
netdev_uninit_flow_api(struct netdev *netdev)
{
    struct netdev_registered_flow_api *rfa;
    const struct netdev_flow_api *flow_api =
            ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    if (!flow_api) {
        return;
    }

    if (flow_api->uninit_flow_api) {
        flow_api->uninit_flow_api(netdev);
    }

    ovsrcu_set(&netdev->flow_api, NULL);
    rfa = netdev_lookup_flow_api(flow_api->type);
    ovs_refcount_unref(&rfa->refcnt);
}

int
netdev_ct_counter_query(struct netdev *netdev,
                        uintptr_t counter,
                        long long now,
                        long long prev_now,
                        struct dpif_flow_stats *stats)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->ct_counter_query)
           ? flow_api->ct_counter_query(netdev, counter, now, prev_now, stats)
           : EOPNOTSUPP;
}

uint32_t
netdev_get_block_id(struct netdev *netdev)
{
    const struct netdev_class *class = netdev->netdev_class;

    return (class->get_block_id
            ? class->get_block_id(netdev)
            : 0);
}

/*
 * Get the value of the hw info parameter specified by type.
 * Returns the value on success (>= 0). Returns -1 on failure.
 */
int
netdev_get_hw_info(struct netdev *netdev, int type)
{
    int val = -1;

    switch (type) {
    case HW_INFO_TYPE_OOR:
        val = netdev->hw_info.oor;
        break;
    case HW_INFO_TYPE_PEND_COUNT:
        val = netdev->hw_info.pending_count;
        break;
    case HW_INFO_TYPE_OFFL_COUNT:
        val = netdev->hw_info.offload_count;
        break;
    default:
        break;
    }

    return val;
}

/*
 * Set the value of the hw info parameter specified by type.
 */
void
netdev_set_hw_info(struct netdev *netdev, int type, int val)
{
    switch (type) {
    case HW_INFO_TYPE_OOR:
        if (val == 0) {
            VLOG_DBG("Offload rebalance: netdev: %s is not OOR", netdev->name);
        }
        netdev->hw_info.oor = val;
        break;
    case HW_INFO_TYPE_PEND_COUNT:
        netdev->hw_info.pending_count = val;
        break;
    case HW_INFO_TYPE_OFFL_COUNT:
        netdev->hw_info.offload_count = val;
        break;
    default:
        break;
    }
}

/* Protects below port hashmaps. */
static struct ovs_rwlock ifindex_to_port_rwlock = OVS_RWLOCK_INITIALIZER;
static struct ovs_rwlock port_to_netdev_rwlock
    OVS_ACQ_BEFORE(ifindex_to_port_rwlock) = OVS_RWLOCK_INITIALIZER;

static struct hmap port_to_netdev OVS_GUARDED_BY(port_to_netdev_rwlock)
    = HMAP_INITIALIZER(&port_to_netdev);
static struct hmap ifindex_to_port OVS_GUARDED_BY(ifindex_to_port_rwlock)
    = HMAP_INITIALIZER(&ifindex_to_port);

struct port_to_netdev_data {
    struct hmap_node portno_node; /* By (dpif_type, dpif_port.port_no). */
    struct hmap_node ifindex_node; /* By (dpif_type, ifindex). */
    struct netdev *netdev;
    struct dpif_port dpif_port;
    int ifindex;
    atomic_bool visible; /* Is this pairing visible externally. */
};

/*
 * Find if any netdev is in OOR state. Return true if there's at least
 * one netdev that's in OOR state; otherwise return false.
 */
bool
netdev_any_oor(void)
    OVS_EXCLUDED(port_to_netdev_rwlock)
{
    struct port_to_netdev_data *data;
    bool oor = false;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        struct netdev *dev = data->netdev;

        if (dev->hw_info.oor) {
            oor = true;
            break;
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);

    return oor;
}

bool
netdev_is_flow_api_enabled(void)
{
    return netdev_flow_api_enabled;
}

unsigned int
netdev_offload_thread_nb(void)
{
    return offload_thread_nb;
}

bool
netdev_is_e2e_cache_enabled(void)
{
    return e2e_cache_enabled;
}

uint32_t
netdev_get_e2e_cache_size(void)
{
    return e2e_cache_size;
}

bool
netdev_is_ct_labels_mapping_enabled(void)
{
    return ct_labels_mapping;
}

bool
netdev_is_zone_tables_disabled(void)
{
    return disable_zone_tables;
}

unsigned int
netdev_offload_ufid_to_thread_id(const ovs_u128 ufid)
{
    uint32_t ufid_hash;

    if (netdev_offload_thread_nb() == 1) {
        return 0;
    }

    ufid_hash = hash_words64_inline(
            (const uint64_t [2]){ ufid.u64.lo,
                                  ufid.u64.hi }, 2, 1);
    return ufid_hash % netdev_offload_thread_nb();
}

unsigned int
netdev_offload_thread_init(unsigned int tid)
{
    bool thread_is_hw_offload;
    bool thread_is_ct_clean;
    bool thread_is_main;
    bool thread_is_rcu;

    thread_is_hw_offload = !strncmp(get_subprogram_name(),
                                    "hw_offload", strlen("hw_offload"));
    thread_is_rcu = !strncmp(get_subprogram_name(), "urcu", strlen("urcu"));
    thread_is_ct_clean = !strncmp(get_subprogram_name(), "ct_clean",
                                  strlen("ct_clean"));
    thread_is_main = *ovsthread_id_get() == 0;

    /* Panic if any other thread besides offload and RCU tries
     * to initialize their thread ID. */
    ovs_assert(thread_is_hw_offload || thread_is_rcu || thread_is_ct_clean ||
               thread_is_main);

    if (*netdev_offload_thread_id_get() == OVSTHREAD_ID_UNSET) {
        unsigned int id;

        if (thread_is_main) {
            /* Main thread does the aux-tables init/uninit. It uses 0 thread-id
             * as it is always a valid offload thread.
             */
            return 0;
        }
        if (thread_is_ct_clean) {
            id = netdev_offload_thread_nb();
            return *netdev_offload_thread_id_get() = id;
        }
        if (thread_is_rcu) {
            /* RCU will compete with other threads for shared object access.
             * Reclamation functions using a thread ID must be thread-safe.
             * For that end, and because RCU must consider all potential shared
             * objects anyway, its thread-id can be whichever, so return 0.
             */
            id = 0;
        } else {
            /* Only the actual offload threads have their own ID. */
            id = tid;
        }
        /* Panic if any offload thread is getting a spurious ID. */
        ovs_assert(id < netdev_offload_thread_nb());
        return *netdev_offload_thread_id_get() = id;
    } else {
        return *netdev_offload_thread_id_get();
    }
}

void
netdev_ports_flow_flush(const char *dpif_type)
{
    struct port_to_netdev_data *data;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        if (netdev_get_dpif_type(data->netdev) == dpif_type) {
            netdev_flow_flush(data->netdev);
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);
}

void
netdev_ports_upkeep(bool quiescing)
{
    struct port_to_netdev_data *data;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        netdev_offload_upkeep(data->netdev, quiescing);
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);
}

void
netdev_ports_traverse(const char *dpif_type,
                      bool (*cb)(struct netdev *, odp_port_t, void *),
                      void *aux)
{
    struct port_to_netdev_data *data;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        if (netdev_get_dpif_type(data->netdev) == dpif_type) {
            bool visible;

            atomic_read_explicit(&data->visible, &visible,
                                 memory_order_acquire);
            if (visible && cb(data->netdev, data->dpif_port.port_no, aux)) {
                break;
            }
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);
}

struct netdev_flow_dump **
netdev_ports_flow_dump_create(const char *dpif_type, int *ports, bool terse)
{
    struct port_to_netdev_data *data;
    struct netdev_flow_dump **dumps;
    int count = 0;
    int i = 0;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        if (netdev_get_dpif_type(data->netdev) == dpif_type) {
            count++;
        }
    }

    dumps = count ? xzalloc(sizeof *dumps * count) : NULL;
    if (!dumps) {
        ovs_rwlock_unlock(&port_to_netdev_rwlock);
        return dumps;
    }

    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        if (netdev_get_dpif_type(data->netdev) == dpif_type) {
            if (netdev_flow_dump_create(data->netdev, &dumps[i], terse)) {
                continue;
            }

            dumps[i]->port = data->dpif_port.port_no;
            i++;
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);

    *ports = i;
    return dumps;
}

int
netdev_ports_flow_del(const char *dpif_type, const ovs_u128 *ufid,
                      struct dpif_flow_stats *stats)
{
    struct port_to_netdev_data *data;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        if (netdev_get_dpif_type(data->netdev) == dpif_type
            && !netdev_flow_del(data->netdev, ufid, stats)) {
            ovs_rwlock_unlock(&port_to_netdev_rwlock);
            return 0;
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);

    return ENOENT;
}

int
netdev_ports_flow_get(const char *dpif_type, struct match *match,
                      struct nlattr **actions, const ovs_u128 *ufid,
                      struct dpif_flow_stats *stats,
                      struct dpif_flow_attrs *attrs, struct ofpbuf *buf)
{
    struct port_to_netdev_data *data;
    long long now = time_msec();

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        if (netdev_get_dpif_type(data->netdev) == dpif_type
            && !netdev_flow_get(data->netdev, match, actions,
                                ufid, stats, attrs, buf, now)) {
            ovs_rwlock_unlock(&port_to_netdev_rwlock);
            return 0;
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);
    return ENOENT;
}

static uint32_t
netdev_ports_hash(odp_port_t port, const char *dpif_type)
{
    return hash_int(odp_to_u32(port), hash_pointer(dpif_type, 0));
}

static struct port_to_netdev_data *
netdev_ports_lookup(odp_port_t port_no, const char *dpif_type)
    OVS_REQ_RDLOCK(port_to_netdev_rwlock)
{
    struct port_to_netdev_data *data;

    HMAP_FOR_EACH_WITH_HASH (data, portno_node,
                             netdev_ports_hash(port_no, dpif_type),
                             &port_to_netdev) {
        if (netdev_get_dpif_type(data->netdev) == dpif_type
            && data->dpif_port.port_no == port_no) {
            return data;
        }
    }
    return NULL;
}

int
netdev_ports_insert(struct netdev *netdev, struct dpif_port *dpif_port)
{
    const char *dpif_type = netdev_get_dpif_type(netdev);
    struct port_to_netdev_data *data;
    int ifindex = netdev_get_ifindex(netdev);
    int ret;

    ovs_assert(dpif_type);

    ovs_rwlock_wrlock(&port_to_netdev_rwlock);
    data = netdev_ports_lookup(dpif_port->port_no, dpif_type);
    if (data) {
        atomic_store_relaxed(&data->visible, true);
        ovs_rwlock_unlock(&port_to_netdev_rwlock);
        return EEXIST;
    }

    data = xzalloc(sizeof *data);
    data->netdev = netdev_ref(netdev);
    atomic_store_relaxed(&data->visible, true);
    dpif_port_clone(&data->dpif_port, dpif_port);

    if (ifindex >= 0) {
        data->ifindex = ifindex;
        ovs_rwlock_wrlock(&ifindex_to_port_rwlock);
        hmap_insert(&ifindex_to_port, &data->ifindex_node, ifindex);
        ovs_rwlock_unlock(&ifindex_to_port_rwlock);
    } else {
        data->ifindex = -1;
    }

    hmap_insert(&port_to_netdev, &data->portno_node,
                netdev_ports_hash(dpif_port->port_no, dpif_type));
    ovs_rwlock_unlock(&port_to_netdev_rwlock);

    ret = netdev_init_flow_api(netdev);
    if (ret == EAGAIN) {
        netdev_ports_remove(dpif_port->port_no, dpif_type);
        return ret;
    }

    return 0;
}

void
netdev_ports_set_visible(struct netdev *netdev, bool visible)
{
    struct port_to_netdev_data *data;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
        if (data->netdev == netdev) {
            atomic_store_relaxed(&data->visible, visible);
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);
}

struct netdev *
netdev_ports_get(odp_port_t port_no, const char *dpif_type)
{
    struct port_to_netdev_data *data;
    struct netdev *ret = NULL;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    data = netdev_ports_lookup(port_no, dpif_type);
    if (data) {
        bool visible;

        atomic_read_explicit(&data->visible, &visible, memory_order_acquire);
        if (visible) {
            ret = netdev_ref(data->netdev);
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);

    return ret;
}

int
netdev_ports_remove(odp_port_t port_no, const char *dpif_type)
{
    struct port_to_netdev_data *data;
    int ret = ENOENT;

    ovs_rwlock_wrlock(&port_to_netdev_rwlock);
    data = netdev_ports_lookup(port_no, dpif_type);
    if (data) {
        dpif_port_destroy(&data->dpif_port);
        netdev_close(data->netdev); /* unref and possibly close */
        hmap_remove(&port_to_netdev, &data->portno_node);
        if (data->ifindex >= 0) {
            ovs_rwlock_wrlock(&ifindex_to_port_rwlock);
            hmap_remove(&ifindex_to_port, &data->ifindex_node);
            ovs_rwlock_unlock(&ifindex_to_port_rwlock);
        }
        free(data);
        ret = 0;
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);

    return ret;
}

int
netdev_ports_get_n_flows(const char *dpif_type, odp_port_t port_no,
                         uint64_t *n_flows)
{
    struct port_to_netdev_data *data;
    int ret = EOPNOTSUPP;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    data = netdev_ports_lookup(port_no, dpif_type);
    if (data) {
        uint64_t thread_n_flows[MAX_OFFLOAD_THREAD_NB] = {0};
        unsigned int tid;

        ret = netdev_flow_get_n_flows(data->netdev, thread_n_flows);
        *n_flows = 0;
        if (!ret) {
            for (tid = 0; tid < netdev_offload_thread_nb(); tid++) {
                *n_flows += thread_n_flows[tid];
            }
        }
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);
    return ret;
}

odp_port_t
netdev_ifindex_to_odp_port(int ifindex)
{
    struct port_to_netdev_data *data;
    odp_port_t ret = 0;

    ovs_rwlock_rdlock(&ifindex_to_port_rwlock);
    HMAP_FOR_EACH_WITH_HASH (data, ifindex_node, ifindex, &ifindex_to_port) {
        if (data->ifindex == ifindex) {
            ret = data->dpif_port.port_no;
            break;
        }
    }
    ovs_rwlock_unlock(&ifindex_to_port_rwlock);

    return ret;
}

static bool netdev_offload_rebalance_policy = false;

bool
netdev_is_offload_rebalance_policy_enabled(void)
{
    return netdev_offload_rebalance_policy;
}

bool
netdev_is_flow_counter_key_zero(const struct flows_counter_key *k)
{
    size_t i;

    for (i = 0; i < ARRAY_SIZE(k->ufid_key); i++) {
        if (k->ufid_key[i].u64.hi != 0 || k->ufid_key[i].u64.lo != 0) {
            return false;
        }
    }
    return true;
}

char*
netdev_flow_counter_key_to_string(const struct flows_counter_key *k,
                                  char *string, size_t sz)
{
    char *ptr = string;
    size_t i;

    if (OVS_UNLIKELY(!sz)) {
        return string;
    }
    sz--;

    if (OVS_LIKELY(sz)) {
        *ptr = '[';
        ptr++;
        sz--;
    } else {
        goto out;
    }
    for (i = 0; i < ARRAY_SIZE(k->ufid_key); i++) {
        const ovs_u128 *ufid = &k->ufid_key[i];
        int n;

        if (ufid->u64.hi == 0 && ufid->u64.lo == 0) {
            break;
        }
        if (OVS_UNLIKELY(sz < 35)) {
            goto out;
        }
        if (i > 0) {
            *ptr = ',';
            ptr++;
            sz--;
        }
        n = sprintf(ptr, "0x%"PRIx64"%"PRIx64, ufid->u64.hi, ufid->u64.lo);
        if (OVS_UNLIKELY(n <= 0)) {
            goto out;
        }
        ptr += (unsigned int)n;
        sz -= (unsigned int)n;
    }
    if (OVS_LIKELY(sz)) {
        *ptr = ']';
        ptr++;
    }
out:
    *ptr = '\0';
    return string;
}

static void
netdev_ports_flow_init(void)
{
    struct port_to_netdev_data *data;

    ovs_rwlock_rdlock(&port_to_netdev_rwlock);
    HMAP_FOR_EACH (data, portno_node, &port_to_netdev) {
       netdev_init_flow_api(data->netdev);
    }
    ovs_rwlock_unlock(&port_to_netdev_rwlock);
}

void
netdev_set_flow_api_enabled(const struct smap *ovs_other_config)
{
    if (smap_get_bool(ovs_other_config, "hw-offload", false)) {
        static struct ovsthread_once once = OVSTHREAD_ONCE_INITIALIZER;

        if (ovsthread_once_start(&once)) {
            netdev_flow_api_enabled = true;

            offload_thread_nb = smap_get_ullong(ovs_other_config,
                                                "n-offload-threads",
                                                DEFAULT_OFFLOAD_THREAD_NB);
            if (offload_thread_nb > MAX_OFFLOAD_THREAD_NB) {
                VLOG_WARN("netdev: Invalid number of threads requested: %u",
                          offload_thread_nb);
                offload_thread_nb = DEFAULT_OFFLOAD_THREAD_NB;
            }

            if (smap_get(ovs_other_config, "n-offload-threads")) {
                VLOG_INFO("netdev: Flow API Enabled, using %u thread%s",
                          offload_thread_nb,
                          offload_thread_nb > 1 ? "s" : "");
            } else {
                VLOG_INFO("netdev: Flow API Enabled");
            }

#ifdef __linux__
            tc_set_policy(smap_get_def(ovs_other_config, "tc-policy",
                                       TC_POLICY_DEFAULT));
#endif

            if (smap_get_bool(ovs_other_config, "offload-rebalance", false)) {
                netdev_offload_rebalance_policy = true;
            }

            netdev_ports_flow_init();

            ovsthread_once_done(&once);
        }
    } else {
        VLOG_INFO_ONCE("netdev: Flow API Disabled. Sub-offload configurations"
                       " are ignored.");
        return;
    }

    if (smap_get_bool(ovs_other_config, "e2e-enable", false)) {
        static struct ovsthread_once once_e2e = OVSTHREAD_ONCE_INITIALIZER;

        if (ovsthread_once_start(&once_e2e)) {
            e2e_cache_enabled = true;
            if (e2e_cache_enabled) {
                e2e_cache_size = smap_get_int(ovs_other_config, "e2e-size",
                                              32000);
                VLOG_INFO("E2E cache size is %"PRIu32, e2e_cache_size);
            }
            ovsthread_once_done(&once_e2e);
        }
    }

    {
        bool req_conf = smap_get_bool(ovs_other_config,
                                      "ct-action-on-nat-conns", false);

        if (req_conf && smap_get_bool(ovs_other_config, "doca-init", false)) {
            VLOG_WARN_ONCE("ct-action-on-nat-conns is not supported by OVS-DOCA.");
        } else if (netdev_offload_ct_on_ct_nat != req_conf) {
            netdev_offload_ct_on_ct_nat = req_conf;
            VLOG_INFO("offloads CT on NAT connections: %s",
                      netdev_offload_ct_on_ct_nat ? "enabled" : "disabled");
        }
    }

    if (smap_get_bool(ovs_other_config, "ct-labels-mapping", false)) {
        static struct ovsthread_once once = OVSTHREAD_ONCE_INITIALIZER;

        if (ovsthread_once_start(&once)) {
            ct_labels_mapping = true;
            VLOG_INFO("CT offloads: labels mapping enabled");
            ovsthread_once_done(&once);
        }
    }

    if (smap_get_bool(ovs_other_config, "disable-zone-tables", false)) {
        static struct ovsthread_once once = OVSTHREAD_ONCE_INITIALIZER;

        if (ovsthread_once_start(&once)) {
            if (!smap_get_bool(ovs_other_config, "doca-init", false)) {
                disable_zone_tables = true;
                VLOG_INFO("CT offloads: zone tables disabled");
            } else {
                VLOG_WARN("disable-zone-tables flag is ignored with doca");
            }
            ovsthread_once_done(&once);
        }
    }
}

int
netdev_packet_hw_hash(struct netdev *netdev,
                      struct dp_packet *packet,
                      uint32_t seed,
                      uint32_t *hash)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->packet_hw_hash)
        ? flow_api->packet_hw_hash(netdev, packet, seed, hash)
        : EOPNOTSUPP;
}

int
netdev_packet_hw_entropy(struct netdev *netdev,
                         struct dp_packet *packet,
                         uint16_t *entropy)
{
    const struct netdev_flow_api *flow_api =
        ovsrcu_get(const struct netdev_flow_api *, &netdev->flow_api);

    return (flow_api && flow_api->packet_hw_entropy)
        ? flow_api->packet_hw_entropy(netdev, packet, entropy)
        : EOPNOTSUPP;
}
