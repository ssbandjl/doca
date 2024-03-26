/*
 * Copyright (c) 2014, 2015, 2016, 2017 Nicira, Inc.
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/types.h>
#include <netinet/ip6.h>
#include <rte_flow.h>
#include <rte_gre.h>

#include "cmap.h"
#include "conntrack-offload.h"
#include "dpdk-offload-provider.h"
#include "dpif-netdev.h"
#include "id-fpool.h"
#include "netdev-offload.h"
#include "netdev-offload-dpdk.h"
#include "netdev-offload-provider.h"
#include "netdev-provider.h"
#include "netdev-vport.h"
#include "ovs-atomic.h"
#include "ovs-doca.h"
#include "odp-util.h"
#include "offload-metadata.h"
#include "openvswitch/match.h"
#include "openvswitch/vlog.h"
#include "ovs-rcu.h"
#include "packets.h"
#include "salloc.h"
#include "uuid.h"

VLOG_DEFINE_THIS_MODULE(netdev_offload_dpdk);
static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(600, 600);

static bool netdev_offload_dpdk_ct_labels_mapping = false;
static bool netdev_offload_dpdk_disable_zone_tables = false;

static struct dpdk_offload_api *offload;

/* Thread-safety
 * =============
 *
 * Below API is NOT thread safe in following terms:
 *
 *  - The caller must be sure that 'netdev' will not be destructed/deallocated.
 *
 *  - The caller must be sure that 'netdev' configuration will not be changed.
 *    For example, simultaneous call of 'netdev_reconfigure()' for the same
 *    'netdev' is forbidden.
 *
 * For current implementation all above restrictions are fulfilled by
 * read-locking the datapath 'port_rwlock' in lib/dpif-netdev.c.  */

/*
 * A mapping from ufid to dpdk rte_flow.
 */

struct per_thread {
PADDED_MEMBERS(CACHE_LINE_SIZE,
    char scratch[10000];
    struct salloc *s;
    struct ovs_list conn_list;
);
};

static struct per_thread per_threads[MAX_OFFLOAD_THREAD_NB];

static void ct_ctx_init(void);
static void flow_miss_ctx_init(void);
static void label_id_init(void);
static void sflow_id_init(void);
static void shared_age_init(void);
static void shared_count_init(void);
static void table_id_init(void);
static void tnl_md_init(void);
static void zone_id_init(void);
static int
get_netdev_by_port(struct netdev *netdev,
                   const struct nlattr *nla,
                   int *outdev_id,
                   struct netdev **outdev);

static void
per_thread_init(void)
{
    struct per_thread *pt = &per_threads[netdev_offload_thread_id()];

    if (pt->s == NULL) {
        pt->s = salloc_init(pt->scratch, sizeof pt->scratch);
        ovs_list_init(&pt->conn_list);
    }
    salloc_reset(pt->s);
}

static void *
per_thread_xmalloc(size_t n)
{
    struct per_thread *pt = &per_threads[netdev_offload_thread_id()];
    void *p = salloc(pt->s, n);

    if (p == NULL) {
        p = xmalloc(n);
    }

    return p;
}

static void *
per_thread_xzalloc(size_t n)
{
    struct per_thread *pt = &per_threads[netdev_offload_thread_id()];
    void *p = szalloc(pt->s, n);

    if (p == NULL) {
        p = xzalloc(n);
    }

    return p;
}

static void *
per_thread_xcalloc(size_t n, size_t sz)
{
    struct per_thread *pt = &per_threads[netdev_offload_thread_id()];
    void *p = scalloc(pt->s, n, sz);

    if (p == NULL) {
        p = xcalloc(n, sz);
    }

    return p;
}

static void *
per_thread_xrealloc(void *old_p, size_t old_size, size_t new_size)
{
    struct per_thread *pt = &per_threads[netdev_offload_thread_id()];
    void *new_p = NULL;

    if (salloc_contains(pt->s, old_p)) {
        new_p = srealloc(pt->s, old_p, new_size);
        if (new_p == NULL) {
            new_p = xmalloc(new_size);
            if (new_p) {
                memcpy(new_p, old_p, old_size);
            }
        }
    } else {
        new_p = xrealloc(old_p, new_size);
    }

    return new_p;
}

static void
per_thread_free(void *p)
{
    struct per_thread *pt = &per_threads[netdev_offload_thread_id()];

    if (salloc_contains(pt->s, p)) {
        /* The only freeing done in the scratch allocator is when resetting it.
         * However, realloc has a chance to shrink, so still attempt it. */
        srealloc(pt->s, p, 0);
    } else {
        free(p);
    }
}

struct esw_members_aux {
    struct netdev *esw_netdev;
    int ret;
    /* Operation to perform on ESW members */
    bool (*op)(struct netdev *, odp_port_t, void *);
};

static inline bool
offload_data_is_conn(struct act_resources *act_resources)
{
    return !!act_resources->ct_miss_ctx_id;
}

static void
offload_data_destroy__(struct netdev_offload_dpdk_data *data)
{
    ovs_mutex_destroy(&data->map_lock);
    free(data->offload_counters);
    free(data->flow_counters);
    free(data->conn_counters);
    free(data);
}

static int
offload_data_init(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;

    ct_ctx_init();
    flow_miss_ctx_init();
    label_id_init();
    sflow_id_init();
    shared_age_init();
    shared_count_init();
    table_id_init();
    tnl_md_init();
    zone_id_init();

    data = xzalloc(sizeof *data);
    ovs_mutex_init(&data->map_lock);
    cmap_init(&data->ufid_to_rte_flow);
    /* Configure cmap to never shrink. */
    cmap_set_min_load(&data->ufid_to_rte_flow, 0.0);
    data->offload_counters = xcalloc(netdev_offload_thread_nb(),
                                     sizeof *data->offload_counters);
    data->flow_counters = xcalloc(netdev_offload_thread_nb(),
                                  sizeof *data->flow_counters);
    data->conn_counters = xcalloc(netdev_offload_thread_nb(),
                                  sizeof *data->conn_counters);

    ovsrcu_set(&netdev->hw_info.offload_data, (void *) data);
    if (offload->aux_tables_init(netdev)) {
        VLOG_WARN("aux_tables_init failed for netdev=%s",
                  netdev_get_name(netdev));
        offload_data_destroy__(data);
        return EAGAIN;
    }

    return 0;
}

static void
offload_data_destroy(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;
    struct ufid_to_rte_flow_data *node;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    if (data == NULL) {
        return;
    }

    if (!cmap_is_empty(&data->ufid_to_rte_flow)) {
        VLOG_ERR("Incomplete flush: %s contains rte_flow elements",
                 netdev_get_name(netdev));
        CMAP_FOR_EACH (node, node, &data->ufid_to_rte_flow) {
            ovsrcu_postpone(free, node);
        }
    }

    offload->aux_tables_uninit(netdev);
    cmap_destroy(&data->ufid_to_rte_flow);
    ovsrcu_postpone(offload_data_destroy__, data);

    ovsrcu_set(&netdev->hw_info.offload_data, NULL);
}

static void
offload_data_lock(struct netdev *netdev)
    OVS_NO_THREAD_SAFETY_ANALYSIS
{
    struct netdev_offload_dpdk_data *data;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    if (!data) {
        return;
    }
    ovs_mutex_lock(&data->map_lock);
}

static void
offload_data_unlock(struct netdev *netdev)
    OVS_NO_THREAD_SAFETY_ANALYSIS
{
    struct netdev_offload_dpdk_data *data;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    if (!data) {
        return;
    }
    ovs_mutex_unlock(&data->map_lock);
}

static struct cmap *
offload_data_map(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);

    return data ? &data->ufid_to_rte_flow : NULL;
}

/* Find rte_flow with @ufid. */
static struct ufid_to_rte_flow_data *
ufid_to_rte_flow_data_find(struct netdev *netdev,
                           const ovs_u128 *ufid, bool warn)
{
    size_t hash = hash_bytes(ufid, sizeof *ufid, 0);
    struct ufid_to_rte_flow_data *data;
    struct cmap *map = offload_data_map(netdev);

    if (!map) {
        return NULL;
    }

    CMAP_FOR_EACH_WITH_HASH (data, node, hash, map) {
        if (ovs_u128_equals(*ufid, data->ufid)) {
            return data;
        }
    }

    if (warn) {
        VLOG_WARN_RL(&rl, "ufid "UUID_FMT" is not associated with an rte flow",
                     UUID_ARGS((struct uuid *) ufid));
    }

    return NULL;
}

static inline void
ufid_to_rte_flow_set(struct ufid_to_rte_flow_data *data,
                     struct netdev *netdev,
                     struct netdev *physdev,
                     struct act_resources *act_resources)
{
    if (data->netdev) {
        ovs_assert(data->netdev == netdev);
    } else {
        data->netdev = netdev_ref(netdev);
    }
    if (data->physdev) {
        ovs_assert(data->physdev == physdev);
    } else {
        data->physdev = netdev != physdev ? netdev_ref(physdev) : physdev;
    }
    memcpy(&data->act_resources, act_resources, sizeof data->act_resources);
}

static inline int
ufid_to_rte_flow_associate(const ovs_u128 *ufid, struct netdev *netdev,
                           struct netdev *physdev,
                           struct ufid_to_rte_flow_data *data,
                           struct act_resources *act_resources)
{
    size_t hash = hash_bytes(ufid, sizeof *ufid, 0);
    unsigned int tid = netdev_offload_thread_id();
    struct cmap *map = offload_data_map(netdev);

    if (!map) {
        return ENODEV;
    }

    /* We already checked before that no one inserted an
     * rte_flow for this ufid. As the ufid dispatch is per
     * thread id, this ufid would have been serviced by
     * this thread and sync is implicit. */

    data->ufid = *ufid;
    data->creation_tid = tid;
    ovs_mutex_init(&data->lock);
    ufid_to_rte_flow_set(data, netdev, physdev, act_resources);

    offload_data_lock(netdev);
    cmap_insert(map, CONST_CAST(struct cmap_node *, &data->node), hash);
    offload_data_unlock(netdev);

    rte_flow_data_active_set(data, true);

    return 0;
}

static void
rte_flow_data_gc(struct ufid_to_rte_flow_data *data)
{
    ovs_mutex_destroy(&data->lock);
    /* Objects for CT are not allocated, but provided. Skip them. */
    if (offload_data_is_conn(&data->act_resources)) {
        return;
    }
    free(data);
}

static inline void
ufid_to_rte_flow_disassociate(struct ufid_to_rte_flow_data *data)
    OVS_REQUIRES(data->lock)
{
    size_t hash = hash_bytes(&data->ufid, sizeof data->ufid, 0);
    struct cmap *map = offload_data_map(data->netdev);

    if (!map) {
        return;
    }

    offload_data_lock(data->netdev);
    cmap_remove(map, CONST_CAST(struct cmap_node *, &data->node), hash);
    offload_data_unlock(data->netdev);

    if (data->netdev != data->physdev) {
        netdev_close(data->netdev);
    }
    netdev_close(data->physdev);
    ovsrcu_gc(rte_flow_data_gc, data, gc_node);
}

static inline void
conn_unlink(struct ufid_to_rte_flow_data *data)
    OVS_REQUIRES(data->lock)
{
    ovs_list_remove(&data->list_node);
    netdev_close(data->physdev);
    ovsrcu_postpone(rte_flow_data_gc, data);
}

static inline int
conn_link(struct netdev *netdev,
          struct ufid_to_rte_flow_data *data,
          struct act_resources *act_resources)
{
    unsigned int tid = netdev_offload_thread_id();

    data->netdev = netdev_ref(netdev);
    data->physdev = netdev;
    data->creation_tid = tid;
    ovs_mutex_init(&data->lock);
    memcpy(&data->act_resources, act_resources, sizeof data->act_resources);
    ovs_list_push_back(&per_threads[tid].conn_list, &data->list_node);

    rte_flow_data_active_set(data, true);

    return 0;
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
    /* Since sFlow and CT will not work concurrently is it safe
     * to have the reg_fields use the same bits for SFLOW_CTX and CT_CTX.
     */
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
        .type = REG_TYPE_MARK,
        .index = 0,
        .offset = 0,
        .mask = 0x00FFFFFF,
    },
};

static struct reg_field *
rte_get_reg_fields(void)
{
    return reg_fields;
}

OVS_ASSERT_PACKED(struct table_id_data,
    struct netdev *netdev;
    odp_port_t vport;
    uint32_t recirc_id;
);

static struct ds *
dump_table_id(struct ds *s, void *data,
              void *priv OVS_UNUSED, void *priv_arg OVS_UNUSED)
{
    struct table_id_data *table_id_data = data;
    const char *netdev_name;

    netdev_name = table_id_data->netdev
                  ? netdev_get_name(table_id_data->netdev)
                  : NULL;
    ds_put_format(s, "%s, vport=%"PRIu32", recirc_id=%"PRIu32,
                  netdev_name, table_id_data->vport, table_id_data->recirc_id);
    return s;
}

static struct ds *
dump_label_id(struct ds *s, void *data,
              void *priv OVS_UNUSED, void *priv_arg OVS_UNUSED)
{
    ovs_u128 not_mapped_ct_label = *(ovs_u128 *) data;

    ds_put_format(s, "label = %x%x%x%x", not_mapped_ct_label.u32[3],
                                         not_mapped_ct_label.u32[2],
                                         not_mapped_ct_label.u32[1],
                                         not_mapped_ct_label.u32[0]);
    return s;
}

#define MIN_LABEL_ID 1
#define MAX_LABEL_ID (offload->reg_fields()[REG_FIELD_CT_LABEL_ID].mask - 1)
#define NUM_LABEL_ID (MAX_LABEL_ID - MIN_LABEL_ID + 1)

static struct id_fpool *label_id_pool = NULL;
static struct offload_metadata *label_id_md;

static void label_id_init(void);

static bool
esw_members_cb(struct netdev *netdev,
               odp_port_t odp_port OVS_UNUSED,
               void *aux_);
static bool
flush_esw_members_op(struct netdev *netdev,
                     odp_port_t odp_port OVS_UNUSED,
                     void *aux_);
static bool
init_esw_members_op(struct netdev *netdev,
                    odp_port_t odp_port OVS_UNUSED,
                    void *aux_ OVS_UNUSED);
static bool
uninit_esw_members_op(struct netdev *netdev,
                      odp_port_t odp_port OVS_UNUSED,
                      void *aux_ OVS_UNUSED);

static uint32_t
label_id_alloc(void)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t label_id;

    if (id_fpool_new_id(label_id_pool, tid, &label_id)) {
        return label_id;
    }
    return 0;
}

static void
label_id_free(uint32_t label_id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(label_id_pool, tid, label_id);
}

static void
label_id_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = label_id_alloc,
            .id_free = label_id_free,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        label_id_pool = id_fpool_create(nb_thread, MIN_LABEL_ID, NUM_LABEL_ID);
        label_id_md = offload_metadata_create(nb_thread, "label_id",
                                              sizeof(ovs_u128), dump_label_id,
                                              params);

        ovsthread_once_done(&init_once);
    }
}

static int
get_label_id(ovs_u128 *ct_label, uint32_t *ct_label_id)
{
    if (is_all_zeros(ct_label, sizeof *ct_label)) {
        *ct_label_id = 0;
        return 0;
    }

    return offload_metadata_id_ref(label_id_md, ct_label,
                       NULL, ct_label_id);
}

static void
put_label_id(uint32_t label_id)
{
    offload_metadata_id_unref(label_id_md,
                              netdev_offload_thread_id(),
                              label_id);
}


static struct id_fpool *zone_id_pool = NULL;
static struct offload_metadata *zone_id_md;

static struct ds *
dump_zone_id(struct ds *s, void *data,
             void *priv OVS_UNUSED, void *priv_arg OVS_UNUSED)
{
    uint16_t not_mapped_ct_zone = *(uint16_t *) data;

    ds_put_format(s, "zone = %d", not_mapped_ct_zone);
    return s;
}

static uint32_t
zone_id_alloc(void)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t zone_id;

    if (id_fpool_new_id(zone_id_pool, tid, &zone_id)) {
        return zone_id;
    }
    return 0;
}

static void
zone_id_free(uint32_t zone_id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(zone_id_pool, tid, zone_id);
}

static void
zone_id_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = zone_id_alloc,
            .id_free = zone_id_free,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        zone_id_pool = id_fpool_create(nb_thread, MIN_ZONE_ID, NUM_ZONE_ID);
        zone_id_md = offload_metadata_create(nb_thread, "zone_id",
                                             sizeof(uint16_t), dump_zone_id,
                                             params);

        ovsthread_once_done(&init_once);
    }
}

static int
get_zone_id(uint16_t ct_zone, uint32_t *ct_zone_id)
{
    return offload_metadata_id_ref(zone_id_md, &ct_zone, NULL, ct_zone_id);
}

static void
put_zone_id(uint32_t zone_id)
{
    offload_metadata_id_unref(zone_id_md, netdev_offload_thread_id(), zone_id);
}

static struct id_fpool *table_id_pool = NULL;
static struct offload_metadata *table_id_md;

static uint32_t
table_id_alloc(void)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t id;

    if (id_fpool_new_id(table_id_pool, tid, &id)) {
        return id;
    }
    return 0;
}

static int
add_miss_flow(struct netdev *netdev,
              uint32_t src_table_id,
              uint32_t dst_table_id,
              uint32_t recirc_id,
              uint32_t mark_id,
              struct dpdk_offload_handle *doh);

static int
netdev_offload_dpdk_destroy_flow(struct netdev *netdev,
                                 struct dpdk_offload_handle *doh,
                                 const ovs_u128 *ufid, bool is_esw);

static void
table_id_free(uint32_t id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(table_id_pool, tid, id);
}

struct table_id_ctx_priv {
    struct netdev *netdev;
    struct dpdk_offload_handle miss_flow;
};

static int
get_table_id(odp_port_t vport,
             uint32_t recirc_id,
             struct netdev *physdev,
             bool is_e2e_cache,
             uint32_t *table_id);

static int
table_id_ctx_init(void *priv_, void *priv_arg_, uint32_t table_id)
{
    struct table_id_data *priv_arg = priv_arg_;
    struct table_id_ctx_priv *priv = priv_;
    uint32_t e2e_table_id;

    if (priv->netdev != NULL) {
        return 0;
    }

    if (!netdev_is_e2e_cache_enabled() || priv_arg->recirc_id != 0) {
       return 0;
    }

    if (get_table_id(priv_arg->vport, 0, priv_arg->netdev, true,
                     &e2e_table_id)) {
        return -1;
    }
    priv->netdev = netdev_ref(priv_arg->netdev);
    if (add_miss_flow(priv->netdev, e2e_table_id, table_id, 0, 0,
                      &priv->miss_flow)) {
        priv->netdev = NULL;
        netdev_close(priv->netdev);
        return -1;
    }
    return 0;
}

static void
table_id_ctx_uninit(void *priv_)
{
    struct table_id_ctx_priv *priv = priv_;

    if (!netdev_is_e2e_cache_enabled() || !priv->netdev) {
       return;
    }

    netdev_offload_dpdk_destroy_flow(priv->netdev, &priv->miss_flow, NULL, true);
    netdev_close(priv->netdev);
    priv->netdev = NULL;
}

static void
table_id_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = table_id_alloc,
            .id_free = table_id_free,
            .priv_size = sizeof(struct table_id_ctx_priv),
            .priv_init = table_id_ctx_init,
            .priv_uninit = table_id_ctx_uninit,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        table_id_pool = id_fpool_create(nb_thread, MIN_TABLE_ID, NUM_TABLE_ID);
        table_id_md = offload_metadata_create(nb_thread, "table_id",
                                              sizeof(struct table_id_data),
                                              dump_table_id, params);

        ovsthread_once_done(&init_once);
    }
}

static int
get_table_id(odp_port_t vport,
             uint32_t recirc_id,
             struct netdev *physdev,
             bool is_e2e_cache,
             uint32_t *table_id)
{
    struct table_id_data table_id_data = {
        .netdev = physdev,
        .vport = vport,
        .recirc_id = recirc_id,
    };
    struct table_id_data priv_arg = table_id_data;

    if (vport == ODPP_NONE && recirc_id == 0 &&
        !(netdev_is_e2e_cache_enabled() && !is_e2e_cache)) {
        *table_id = 0;
        return 0;
    }

    if (is_e2e_cache) {
        *table_id = E2E_BASE_TABLE_ID | vport;
        return 0;
    }

    if (vport != ODPP_NONE) {
        table_id_data.netdev = NULL;
    }
    if (recirc_id && conntrack_offload_size() == 0) {
        table_id_data.recirc_id = 1;
    }

    return offload_metadata_id_ref(table_id_md, &table_id_data, &priv_arg,
                                   table_id);
}

static void
put_table_id(uint32_t table_id)
{
    if (table_id > MAX_TABLE_ID) {
        return;
    }
    offload_metadata_id_unref(table_id_md,
                              netdev_offload_thread_id(),
                              table_id);
}

OVS_ASSERT_PACKED(struct sflow_ctx,
    struct dpif_sflow_attr sflow_attr;
    struct user_action_cookie cookie;
    struct flow_tnl sflow_tnl;
);

static struct ds *
dump_sflow_id(struct ds *s, void *data,
              void *priv OVS_UNUSED, void *priv_arg OVS_UNUSED)
{
    struct sflow_ctx *sflow_ctx = data;
    struct user_action_cookie *cookie;

    cookie = &sflow_ctx->cookie;
    ds_put_format(s, "sFlow cookie %p, ofproto_uuid "UUID_FMT,
                  cookie, UUID_ARGS(&cookie->ofproto_uuid));
    return s;
}

#define MIN_SFLOW_ID 1
#define MAX_SFLOW_ID (offload->reg_fields()[REG_FIELD_SFLOW_CTX].mask - 1)
#define NUM_SFLOW_ID (MAX_SFLOW_ID - MIN_SFLOW_ID + 1)

static struct id_fpool *sflow_id_pool = NULL;
static struct offload_metadata *sflow_id_md;
static void sflow_id_init(void);

static uint32_t
sflow_id_alloc(void)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t id;

    if (id_fpool_new_id(sflow_id_pool, tid, &id)) {
        return id;
    }
    return 0;
}

static void
sflow_id_free(uint32_t sflow_id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(sflow_id_pool, tid, sflow_id);
}

static void
sflow_id_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = sflow_id_alloc,
            .id_free = sflow_id_free,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        sflow_id_pool = id_fpool_create(nb_thread, MIN_SFLOW_ID, NUM_SFLOW_ID);
        sflow_id_md = offload_metadata_create(nb_thread, "sflow_id",
                                              sizeof(struct sflow_ctx),
                                              dump_sflow_id, params);

        ovsthread_once_done(&init_once);
    }
}

static int
get_sflow_id(struct sflow_ctx *sflow_ctx, uint32_t *sflow_id)
{
    return offload_metadata_id_ref(sflow_id_md, sflow_ctx, NULL, sflow_id);
}

static void
put_sflow_id(uint32_t sflow_id)
{
    offload_metadata_id_unref(sflow_id_md,
                              netdev_offload_thread_id(),
                              sflow_id);
}

static int
find_sflow_ctx(int sflow_id, struct sflow_ctx *ctx)
{
    return offload_metadata_data_from_id(sflow_id_md, sflow_id, ctx);
}

#define MIN_CT_CTX_ID 1
#define MAX_CT_CTX_ID (offload->reg_fields()[REG_FIELD_CT_CTX].mask - 1)
#define NUM_CT_CTX_ID (MAX_CT_CTX_ID - MIN_CT_CTX_ID + 1)

static struct id_fpool *ct_ctx_pool = NULL;
static struct offload_metadata *ct_ctx_md;

static uint32_t
ct_ctx_id_alloc(void)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t id;

    if (id_fpool_new_id(ct_ctx_pool, tid, &id)) {
        return id;
    }
    return 0;
}

static void
ct_ctx_id_free(uint32_t id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(ct_ctx_pool, tid, id);
}

OVS_ASSERT_PACKED(struct ct_miss_ctx,
    ovs_u128 label;
    uint32_t mark;
    uint16_t zone;
    uint8_t state;
    /* Manual padding must be used instead of PADDED_MEMBERS. */
    uint8_t pad[1];
);

static struct ds *
dump_ct_ctx_id(struct ds *s, void *data,
               void *priv OVS_UNUSED, void *priv_arg OVS_UNUSED)
{
    struct ct_miss_ctx *ct_ctx_data = data;
    ovs_be128 label;

    label = hton128(ct_ctx_data->label);
    ds_put_format(s, "ct_state=0x%"PRIx8", zone=%d, ct_mark=0x%"PRIx32
                  ", ct_label=", ct_ctx_data->state, ct_ctx_data->zone,
                  ct_ctx_data->mark);
    ds_put_hex(s, &label, sizeof label);
    return s;
}

/* In ofproto/ofproto-dpif-rid.c, function recirc_run. Timeout for expired
 * flows is 250 msec. Set this timeout the same.
 */
#define DELAYED_RELEASE_TIMEOUT_MS 250

static void
ct_ctx_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = ct_ctx_id_alloc,
            .id_free = ct_ctx_id_free,
            .release_delay_ms = DELAYED_RELEASE_TIMEOUT_MS,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        ct_ctx_pool = id_fpool_create(nb_thread, MIN_CT_CTX_ID, NUM_CT_CTX_ID);
        ct_ctx_md = offload_metadata_create(nb_thread, "ct_miss_ctx",
                                            sizeof(struct ct_miss_ctx),
                                            dump_ct_ctx_id, params);

        ovsthread_once_done(&init_once);
    }
}

static int
get_ct_ctx_id(struct ct_miss_ctx *ct_miss_ctx_data, uint32_t *ct_ctx_id)
{
    return offload_metadata_id_ref(ct_ctx_md, ct_miss_ctx_data,
                                   NULL, ct_ctx_id);
}

static void
put_ct_ctx_id(uint32_t ct_ctx_id)
{
    offload_metadata_id_unref(ct_ctx_md,
                              netdev_offload_thread_id(),
                              ct_ctx_id);
}

static int
find_ct_miss_ctx(int ct_ctx_id, struct ct_miss_ctx *ctx)
{
    return offload_metadata_data_from_id(ct_ctx_md, ct_ctx_id, ctx);
}

#define MIN_TUNNEL_ID 1
#define MAX_TUNNEL_ID (offload->reg_fields()[REG_FIELD_TUN_INFO].mask - 1)
#define NUM_TUNNEL_ID (MAX_TUNNEL_ID - MIN_TUNNEL_ID + 1)

static struct id_fpool *tnl_id_pool = NULL;
static struct offload_metadata *tnl_md;

static uint32_t
tnl_id_alloc(void)
{
    unsigned int tid = netdev_offload_thread_id();
    uint32_t id;

    if (id_fpool_new_id(tnl_id_pool, tid, &id)) {
        return id;
    }
    return 0;
}

static void
tnl_id_free(uint32_t id)
{
    unsigned int tid = netdev_offload_thread_id();

    id_fpool_free_id(tnl_id_pool, tid, id);
}

static struct ds *
dump_tnl_id(struct ds *s, void *data,
            void *priv OVS_UNUSED, void *priv_arg OVS_UNUSED)
{
    struct flow_tnl *tnl = data;

    ds_put_format(s, IP_FMT" -> "IP_FMT", tun_id=%"PRIu64,
                  IP_ARGS(tnl->ip_src), IP_ARGS(tnl->ip_dst),
                  ntohll(tnl->tun_id));
    return s;
}

static void
tnl_md_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = tnl_id_alloc,
            .id_free = tnl_id_free,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        tnl_id_pool = id_fpool_create(nb_thread, MIN_TUNNEL_ID, NUM_TUNNEL_ID);
        tnl_md = offload_metadata_create(nb_thread, "tunnel",
                                         2 * sizeof(struct flow_tnl),
                                         dump_tnl_id, params);

        ovsthread_once_done(&init_once);
    }
}

static void
get_tnl_masked(struct flow_tnl *dst_key, struct flow_tnl *dst_mask,
               struct flow_tnl *src_key, struct flow_tnl *src_mask)
{
    char *psrc_key;
    char *pdst_key, *pdst_mask;
    int i;

    if (dst_mask) {
        memcpy(dst_mask, src_mask, sizeof *dst_mask);
        memset(&dst_mask->metadata, 0, sizeof dst_mask->metadata);

        pdst_key = (char *)dst_key;
        psrc_key = (char *)src_key;
        pdst_mask = (char *)dst_mask;
        for (i = 0; i < sizeof *dst_key; i++) {
            *pdst_key++ = *psrc_key++ & *pdst_mask++;
        }
    } else {
        memcpy(dst_key, src_key, sizeof *dst_key);
    }
}

static int
get_tnl_id(struct flow_tnl *tnl_key, struct flow_tnl *tnl_mask,
           uint32_t *tnl_id)
{
    struct flow_tnl tnl_tmp[2];

    get_tnl_masked(&tnl_tmp[0], &tnl_tmp[1], tnl_key, tnl_mask);
    if (is_all_zeros(&tnl_tmp, sizeof tnl_tmp)) {
        *tnl_id = 0;
        return 0;
    }

    return offload_metadata_id_ref(tnl_md, tnl_tmp, NULL, tnl_id);
}

static void
put_tnl_id(uint32_t tnl_id)
{
    offload_metadata_id_unref(tnl_md, netdev_offload_thread_id(), tnl_id);
}

OVS_ASSERT_PACKED(struct flow_miss_ctx,
    /* Manual padding must be used instead of PADDED_MEMBERS. */
    odp_port_t vport;
    uint32_t recirc_id;
    uint8_t skip_actions;
    uint8_t has_dp_hash;
    uint8_t pad0[6];
    struct flow_tnl tnl;
);

static struct ds *
dump_flow_ctx_id(struct ds *s, void *data,
                 void *priv, void *priv_arg)
{
    struct flow_miss_ctx *flow_ctx_data = data;

    ds_put_format(s, "vport=%"PRIu32", recirc_id=%"PRIu32", ",
                  flow_ctx_data->vport, flow_ctx_data->recirc_id);
    dump_tnl_id(s, &flow_ctx_data->tnl, priv, priv_arg);

    return s;
}

struct flow_miss_ctx_priv_arg {
    struct netdev *netdev;
    uint32_t table_id;
    uint32_t recirc_id;
};

struct flow_miss_ctx_priv {
    struct netdev *netdev;
    struct dpdk_offload_handle miss_flow;
};

static int
flow_miss_ctx_priv_init(void *priv_, void *priv_arg_, uint32_t mark_id)
{
    struct flow_miss_ctx_priv_arg *priv_arg = priv_arg_;
    struct flow_miss_ctx_priv *priv = priv_;

    if (priv->netdev != NULL) {
        return 0;
    }

    priv->netdev = netdev_ref(priv_arg->netdev);
    if (add_miss_flow(priv->netdev, priv_arg->table_id, MISS_TABLE_ID,
                      priv_arg->recirc_id, mark_id, &priv->miss_flow)) {
        netdev_close(priv->netdev);
        priv->netdev = NULL;
        return -1;
    }

    return 0;
}

static void
flow_miss_ctx_priv_uninit(void *priv_)
{
    struct flow_miss_ctx_priv *priv = priv_;

    if (!priv->netdev) {
       return;
    }

    netdev_offload_dpdk_destroy_flow(priv->netdev, &priv->miss_flow, NULL, true);
    netdev_close(priv->netdev);
    priv->netdev = NULL;
}

static struct offload_metadata *flow_miss_ctx_md;

static void
flow_miss_ctx_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .id_alloc = netdev_offload_flow_mark_alloc,
            .id_free = netdev_offload_flow_mark_free,
            .priv_size = sizeof(struct flow_miss_ctx_priv),
            .priv_init = flow_miss_ctx_priv_init,
            .priv_uninit = flow_miss_ctx_priv_uninit,
            .release_delay_ms = DELAYED_RELEASE_TIMEOUT_MS,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        flow_miss_ctx_md = offload_metadata_create(nb_thread, "flow_miss_ctx",
                                                sizeof(struct flow_miss_ctx),
                                                dump_flow_ctx_id, params);

        ovsthread_once_done(&init_once);
    }
}

static int
get_flow_miss_ctx_id(struct flow_miss_ctx *flow_ctx_data,
                     struct netdev *netdev,
                     uint32_t table_id,
                     uint32_t recirc_id,
                     uint32_t *miss_ctx_id)
{
    struct flow_miss_ctx_priv_arg priv_arg = {
        .netdev = netdev,
        .table_id = table_id,
        .recirc_id = recirc_id,
    };

    return offload_metadata_id_ref(flow_miss_ctx_md, flow_ctx_data, &priv_arg,
                       miss_ctx_id);
}

static void
put_flow_miss_ctx_id(uint32_t flow_ctx_id)
{
    offload_metadata_id_unref(flow_miss_ctx_md,
                              netdev_offload_thread_id(),
                              flow_ctx_id);
}

static int
associate_flow_id(uint32_t flow_id, struct flow_miss_ctx *flow_ctx_data)
{
    offload_metadata_id_set(flow_miss_ctx_md, flow_ctx_data, flow_id);
    return 0;
}

static int
disassociate_flow_id(uint32_t flow_id)
{
    offload_metadata_id_unset(flow_miss_ctx_md,
                              netdev_offload_thread_id(),
                              flow_id);
    return 0;
}

struct indirect_ctx_init_arg {
    struct netdev *netdev;
    struct rte_flow_action *action;
    enum ovs_shared_type type;
};

static int
indirect_ctx_init(void *ctx_, void *arg_, uint32_t id OVS_UNUSED)
{
    struct indirect_ctx_init_arg *arg = arg_;
    struct indirect_ctx *ctx = ctx_;
    struct rte_flow_error error;

    ctx->res_type = arg->type;

    if (offload->shared_create(arg->netdev, ctx, arg->action, &error)) {
        ctx->port_id = -1;
        return -1;
    }

    ctx->netdev = arg->netdev;
    ctx->port_id = netdev_dpdk_get_esw_mgr_port_id(arg->netdev);

    return 0;
}

static void
indirect_ctx_uninit(void *ctx_)
{
    struct indirect_ctx *ctx = ctx_;
    struct rte_flow_error error;

    if (!ctx) {
        return;
    }

    offload->shared_destroy(ctx, &error);
    ctx->port_id = -1;
}

static struct ds *
dump_shared_age(struct ds *s, void *data,
                void *priv, void *priv_arg)
{
    struct indirect_ctx_init_arg *arg = priv_arg;
    struct indirect_ctx *ctx = priv;
    uintptr_t *key = data;

    if (key) {
        ds_put_format(s, "netdev=%s, key=0x%"PRIxPTR", ",
                      arg ? netdev_get_name(arg->netdev) : "nil",
                      *((uintptr_t *) key));
    }
    if (ctx) {
        ds_put_format(s, "ctx->port_id=%d, ctx->act_hdl=%p",
                      ctx->port_id, ctx->act_hdl);
    } else {
        ds_put_cstr(s, "ctx=NULL");
    }
    return s;
}

static struct offload_metadata *shared_age_md;

static void
shared_age_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .priv_size = sizeof(struct indirect_ctx),
            .priv_init = indirect_ctx_init,
            .priv_uninit = indirect_ctx_uninit,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        shared_age_md = offload_metadata_create(nb_thread, "shared-age",
                                                sizeof(uintptr_t),
                                                dump_shared_age,
                                                params);

        ovsthread_once_done(&init_once);
    }
}

static struct indirect_ctx *
get_indirect_age_ctx(struct netdev *netdev,
                     uintptr_t app_counter_key,
                     bool create)
{
    struct rte_flow_action_age age_conf = {
        .timeout = 0xFFFFFF,
    };
    struct rte_flow_action action = {
        .type = RTE_FLOW_ACTION_TYPE_AGE,
        .conf = &age_conf,
    };
    struct indirect_ctx_init_arg arg = {
        .netdev = netdev,
        .action = &action,
    };

    return offload_metadata_priv_get(shared_age_md, &app_counter_key, &arg,
                                     NULL, create);
}

static void
free_indirect_age_ctx(void *priv)
{
    offload_metadata_priv_unref(shared_age_md,
                                netdev_offload_thread_id(),
                                priv);
}

static struct ds *
dump_shared_count(struct ds *s, void *data, void *priv, void *priv_arg)
{
    struct indirect_ctx_init_arg *arg = priv_arg;
    struct flows_counter_key *fck = data;
    struct indirect_ctx *ctx = priv;
    const char *type;

    ovs_assert(arg);

    switch (arg->type) {
    case OVS_SHARED_COUNT:
        type = "FLOW";
        break;
    case OVS_SHARED_CT_COUNT:
        type = "CT";
        break;
    case OVS_SHARED_UNDEFINED:
        type = "UNDEFINED";
        break;
    default:
        OVS_NOT_REACHED();
    }

    if (fck) {
        ovs_u128 *last_ufid_key;

        last_ufid_key = &fck->ufid_key[OFFLOAD_FLOWS_COUNTER_KEY_SIZE - 1];
        if (ovs_u128_is_zero(*last_ufid_key)) {
            ds_put_format(s, "netdev=%s, type=%s key=0x%"PRIxPTR", ",
                          arg ? netdev_get_name(arg->netdev) : "nil",
                          type, fck->ptr_key);
        } else {
            char key_str[OFFLOAD_FLOWS_COUNTER_KEY_STRING_SIZE];

            netdev_flow_counter_key_to_string(fck, key_str, sizeof key_str);
            ds_put_format(s, "netdev=%s, type=%s, key=%s, ",
                          arg ? netdev_get_name(arg->netdev) : "nil",
                          type, key_str);
        }
    }
    if (ctx) {
        ds_put_format(s, "ctx->port_id=%d, type=%s, ctx->act_hdl=%p",
                      ctx->port_id, type, ctx->act_hdl);
    } else {
        ds_put_cstr(s, "ctx=NULL");
    }
    return s;
}

static struct offload_metadata *shared_count_md;

static void
shared_count_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        struct offload_metadata_parameters params = {
            .priv_size = sizeof(struct indirect_ctx),
            .priv_init = indirect_ctx_init,
            .priv_uninit = indirect_ctx_uninit,
            /* Disable shrinking on CT counter CMAPs.
             * Otherwise they might re-expand afterward,
             * adding latency jitter. */
            .disable_map_shrink = true,
        };
        unsigned int nb_thread = netdev_offload_thread_nb();

        shared_count_md = offload_metadata_create(nb_thread, "shared-count",
                                            sizeof(struct flows_counter_key),
                                            dump_shared_count, params);

        ovsthread_once_done(&init_once);
    }
}


static struct indirect_ctx *
get_indirect_count_ctx(struct netdev *netdev,
                       struct flows_counter_key *key,
                       enum ovs_shared_type type,
                       bool create)
{
    struct rte_flow_action action = {
        .type = RTE_FLOW_ACTION_TYPE_COUNT,
    };
    struct indirect_ctx_init_arg arg = {
        .netdev = netdev,
        .action = &action,
        .type = type,
    };

    return offload_metadata_priv_get(shared_count_md, key, &arg,
                                     NULL, create);
}

static void
free_indirect_count_ctx(void *priv)
{
    offload_metadata_priv_unref(shared_count_md,
                                netdev_offload_thread_id(),
                                priv);
}

static void
put_action_resources(struct act_resources *act_resources)
{
    int i;

    put_table_id(act_resources->self_table_id);
    put_table_id(act_resources->next_table_id);
    put_flow_miss_ctx_id(act_resources->flow_miss_ctx_id);
    put_tnl_id(act_resources->tnl_id);
    if (act_resources->associated_flow_id) {
        disassociate_flow_id(act_resources->flow_id);
    }
    put_ct_ctx_id(act_resources->ct_miss_ctx_id);
    put_zone_id(act_resources->ct_match_zone_id);
    put_zone_id(act_resources->ct_action_zone_id);
    put_label_id(act_resources->ct_match_label_id);
    put_label_id(act_resources->ct_action_label_id);
    free_indirect_age_ctx(act_resources->shared_age_ctx);
    free_indirect_count_ctx(act_resources->shared_count_ctx);
    put_sflow_id(act_resources->sflow_id);
    for (i = 0; i < DPDK_OFFLOAD_MAX_METERS_PER_FLOW &&
         act_resources->meter_ids[i] != 0; i++) {
        netdev_dpdk_meter_unref(act_resources->meter_ids[i]);
    }
}

static int
find_flow_miss_ctx(int flow_ctx_id, struct flow_miss_ctx *ctx)
{
    return offload_metadata_data_from_id(flow_miss_ctx_md, flow_ctx_id, ctx);
}

static void
netdev_offload_dpdk_upkeep(struct netdev *netdev, bool quiescing)
{
    unsigned int tid = netdev_offload_thread_id();
    long long int now = time_msec();

    if (offload->upkeep) {
        offload->upkeep(netdev, quiescing);
    }

    offload_metadata_upkeep(label_id_md, tid, now);
    offload_metadata_upkeep(zone_id_md, tid, now);
    offload_metadata_upkeep(table_id_md, tid, now);
    offload_metadata_upkeep(sflow_id_md, tid, now);
    offload_metadata_upkeep(ct_ctx_md, tid, now);
    offload_metadata_upkeep(tnl_md, tid, now);
    offload_metadata_upkeep(flow_miss_ctx_md, tid, now);
    offload_metadata_upkeep(shared_age_md, tid, now);
    offload_metadata_upkeep(shared_count_md, tid, now);
}

/*
 * To avoid individual xrealloc calls for each new element, a 'curent_max'
 * is used to keep track of current allocated number of elements. Starts
 * by 8 and doubles on each xrealloc call.
 */
struct flow_patterns {
    struct rte_flow_item *items;
    int cnt;
    int current_max;
    struct netdev *physdev;
    /* tnl_pmd_items is the opaque array of items returned by the PMD. */
    struct rte_flow_item *tnl_pmd_items;
    uint32_t tnl_pmd_items_cnt;
    struct ds s_tnl;
    /* Tag matches must be merged per-index. Keep track of
     * each index and use a single item for each. */
    struct rte_flow_item_tag *tag_spec[REG_TAG_INDEX_NUM];
    struct rte_flow_item_tag *tag_mask[REG_TAG_INDEX_NUM];
};

struct flow_actions {
    struct rte_flow_action *actions;
    int cnt;
    int current_max;
    struct netdev *tnl_netdev;
    /* tnl_pmd_actions is the opaque array of actions returned by the PMD. */
    struct rte_flow_action *tnl_pmd_actions;
    uint32_t tnl_pmd_actions_cnt;
    /* tnl_pmd_actions_pos is where the tunnel actions starts within the
     * 'actions' field.
     */
    int tnl_pmd_actions_pos;
    struct ds s_tnl;
    int shared_age_action_pos;
    int shared_count_action_pos;
};

#define IS_REWRITE_ACTION(act_type) (\
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_MAC_SRC) || \
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_MAC_DST) || \
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_IPV4_SRC) || \
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_IPV4_DST) || \
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_IPV4_TTL) || \
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_IPV6_SRC) || \
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_IPV6_DST) || \
    ((act_type) == RTE_FLOW_ACTION_TYPE_SET_IPV6_HOP) || \
    ((act_type) == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_SRC)) || \
    ((act_type) == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_DST)) || \
    ((act_type) == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_SRC)) || \
    ((act_type) == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_DST)))

static void
dump_flow_attr(struct ds *s, struct ds *s_extra,
               const struct rte_flow_attr *attr,
               struct flow_patterns *flow_patterns,
               struct flow_actions *flow_actions)
{
    if (flow_actions->tnl_pmd_actions_cnt) {
        ds_clone(s_extra, &flow_actions->s_tnl);
    } else if (flow_patterns->tnl_pmd_items_cnt) {
        ds_clone(s_extra, &flow_patterns->s_tnl);
    }
    ds_put_format(s, "%s%spriority %"PRIu32" group 0x%"PRIx32" %s%s%s",
                  attr->ingress  ? "ingress " : "",
                  attr->egress   ? "egress " : "", attr->priority, attr->group,
                  attr->transfer ? "transfer " : "",
                  flow_actions->tnl_pmd_actions_cnt ? "tunnel_set 1 " : "",
                  flow_patterns->tnl_pmd_items_cnt ? "tunnel_match 1 " : "");
}

/* Adds one pattern item 'field' with the 'mask' to dynamic string 's' using
 * 'testpmd command'-like format. */
#define DUMP_PATTERN_ITEM(mask, has_last, field, fmt, spec_pri, mask_pri, \
                          last_pri) \
    if (has_last) { \
        ds_put_format(s, field " spec " fmt " " field " mask " fmt " " field \
                      " last " fmt " ", spec_pri, mask_pri, last_pri); \
    } else if (is_all_ones(&mask, sizeof mask)) { \
        ds_put_format(s, field " is " fmt " ", spec_pri); \
    } else if (!is_all_zeros(&mask, sizeof mask)) { \
        ds_put_format(s, field " spec " fmt " " field " mask " fmt " ", \
                      spec_pri, mask_pri); \
    }

static void
dump_flow_pattern(struct ds *s,
                  struct flow_patterns *flow_patterns,
                  int pattern_index)
{
    const struct rte_flow_item *item = &flow_patterns->items[pattern_index];

    if (item->type == RTE_FLOW_ITEM_TYPE_END) {
        ds_put_cstr(s, "end ");
    } else if (flow_patterns->tnl_pmd_items_cnt &&
               pattern_index < flow_patterns->tnl_pmd_items_cnt) {
        return;
    } else if (item->type == RTE_FLOW_ITEM_TYPE_ETH) {
        const struct rte_flow_item_eth *eth_spec = item->spec;
        const struct rte_flow_item_eth *eth_mask = item->mask;
        uint8_t ea[ETH_ADDR_LEN];

        ds_put_cstr(s, "eth ");
        if (eth_spec) {
            uint32_t has_vlan_mask;

            if (!eth_mask) {
                eth_mask = &rte_flow_item_eth_mask;
            }
            DUMP_PATTERN_ITEM(eth_mask->src, false, "src", ETH_ADDR_FMT,
                              ETH_ADDR_BYTES_ARGS(eth_spec->src.addr_bytes),
                              ETH_ADDR_BYTES_ARGS(eth_mask->src.addr_bytes),
                              ETH_ADDR_BYTES_ARGS(ea));
            DUMP_PATTERN_ITEM(eth_mask->dst, false, "dst", ETH_ADDR_FMT,
                              ETH_ADDR_BYTES_ARGS(eth_spec->dst.addr_bytes),
                              ETH_ADDR_BYTES_ARGS(eth_mask->dst.addr_bytes),
                              ETH_ADDR_BYTES_ARGS(ea));
            DUMP_PATTERN_ITEM(eth_mask->type, false, "type", "0x%04"PRIx16,
                              ntohs(eth_spec->type),
                              ntohs(eth_mask->type), 0);
            has_vlan_mask = eth_mask->has_vlan ? UINT32_MAX : 0;
            DUMP_PATTERN_ITEM(has_vlan_mask, NULL, "has_vlan", "%d",
                              eth_spec->has_vlan, eth_mask->has_vlan, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_VLAN) {
        const struct rte_flow_item_vlan *vlan_spec = item->spec;
        const struct rte_flow_item_vlan *vlan_mask = item->mask;

        ds_put_cstr(s, "vlan ");
        if (vlan_spec) {
            if (!vlan_mask) {
                vlan_mask = &rte_flow_item_vlan_mask;
            }
            DUMP_PATTERN_ITEM(vlan_mask->inner_type, false, "inner_type",
                              "0x%"PRIx16, ntohs(vlan_spec->inner_type),
                              ntohs(vlan_mask->inner_type), 0);
            DUMP_PATTERN_ITEM(vlan_mask->tci, false, "tci", "0x%"PRIx16,
                              ntohs(vlan_spec->tci), ntohs(vlan_mask->tci), 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_IPV4) {
        const struct rte_flow_item_ipv4 *ipv4_spec = item->spec;
        const struct rte_flow_item_ipv4 *ipv4_mask = item->mask;
        const struct rte_flow_item_ipv4 *ipv4_last = item->last;

        ds_put_cstr(s, "ipv4 ");
        if (ipv4_spec) {
            ovs_be16 fragment_offset_mask;

            if (!ipv4_mask) {
                ipv4_mask = &rte_flow_item_ipv4_mask;
            }
            if (!ipv4_last) {
                ipv4_last = &rte_flow_item_ipv4_mask;
            }
            DUMP_PATTERN_ITEM(ipv4_mask->hdr.src_addr, false, "src", IP_FMT,
                              IP_ARGS(ipv4_spec->hdr.src_addr),
                              IP_ARGS(ipv4_mask->hdr.src_addr), IP_ARGS(0));
            DUMP_PATTERN_ITEM(ipv4_mask->hdr.dst_addr, false, "dst", IP_FMT,
                              IP_ARGS(ipv4_spec->hdr.dst_addr),
                              IP_ARGS(ipv4_mask->hdr.dst_addr), IP_ARGS(0));
            DUMP_PATTERN_ITEM(ipv4_mask->hdr.next_proto_id, false, "proto",
                              "0x%"PRIx8, ipv4_spec->hdr.next_proto_id,
                              ipv4_mask->hdr.next_proto_id, 0);
            DUMP_PATTERN_ITEM(ipv4_mask->hdr.type_of_service, false, "tos",
                              "0x%"PRIx8, ipv4_spec->hdr.type_of_service,
                              ipv4_mask->hdr.type_of_service, 0);
            DUMP_PATTERN_ITEM(ipv4_mask->hdr.time_to_live, false, "ttl",
                              "0x%"PRIx8, ipv4_spec->hdr.time_to_live,
                              ipv4_mask->hdr.time_to_live, 0);
            fragment_offset_mask = ipv4_mask->hdr.fragment_offset ==
                                   htons(RTE_IPV4_HDR_OFFSET_MASK |
                                         RTE_IPV4_HDR_MF_FLAG)
                                   ? OVS_BE16_MAX
                                   : ipv4_mask->hdr.fragment_offset;
            DUMP_PATTERN_ITEM(fragment_offset_mask, item->last,
                              "fragment_offset", "0x%"PRIx16,
                              ntohs(ipv4_spec->hdr.fragment_offset),
                              ntohs(ipv4_mask->hdr.fragment_offset),
                              ntohs(ipv4_last->hdr.fragment_offset));
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_UDP) {
        const struct rte_flow_item_udp *udp_spec = item->spec;
        const struct rte_flow_item_udp *udp_mask = item->mask;

        ds_put_cstr(s, "udp ");
        if (udp_spec) {
            if (!udp_mask) {
                udp_mask = &rte_flow_item_udp_mask;
            }
            DUMP_PATTERN_ITEM(udp_mask->hdr.src_port, false, "src", "%"PRIu16,
                              ntohs(udp_spec->hdr.src_port),
                              ntohs(udp_mask->hdr.src_port), 0);
            DUMP_PATTERN_ITEM(udp_mask->hdr.dst_port, false, "dst", "%"PRIu16,
                              ntohs(udp_spec->hdr.dst_port),
                              ntohs(udp_mask->hdr.dst_port), 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_SCTP) {
        const struct rte_flow_item_sctp *sctp_spec = item->spec;
        const struct rte_flow_item_sctp *sctp_mask = item->mask;

        ds_put_cstr(s, "sctp ");
        if (sctp_spec) {
            if (!sctp_mask) {
                sctp_mask = &rte_flow_item_sctp_mask;
            }
            DUMP_PATTERN_ITEM(sctp_mask->hdr.src_port, false, "src", "%"PRIu16,
                              ntohs(sctp_spec->hdr.src_port),
                              ntohs(sctp_mask->hdr.src_port), 0);
            DUMP_PATTERN_ITEM(sctp_mask->hdr.dst_port, false, "dst", "%"PRIu16,
                              ntohs(sctp_spec->hdr.dst_port),
                              ntohs(sctp_mask->hdr.dst_port), 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_ICMP) {
        const struct rte_flow_item_icmp *icmp_spec = item->spec;
        const struct rte_flow_item_icmp *icmp_mask = item->mask;

        ds_put_cstr(s, "icmp ");
        if (icmp_spec) {
            if (!icmp_mask) {
                icmp_mask = &rte_flow_item_icmp_mask;
            }
            DUMP_PATTERN_ITEM(icmp_mask->hdr.icmp_type, false, "type",
                              "%"PRIu8, icmp_spec->hdr.icmp_type,
                              icmp_mask->hdr.icmp_type, 0);
            DUMP_PATTERN_ITEM(icmp_mask->hdr.icmp_code, false, "code",
                              "%"PRIu8, icmp_spec->hdr.icmp_code,
                              icmp_mask->hdr.icmp_code, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_ICMP6) {
        const struct rte_flow_item_icmp6 *icmp6_spec = item->spec;
        const struct rte_flow_item_icmp6 *icmp6_mask = item->mask;

        ds_put_cstr(s, "icmp6 ");
        if (icmp6_spec) {
            if (!icmp6_mask) {
                icmp6_mask = &rte_flow_item_icmp6_mask;
            }
            DUMP_PATTERN_ITEM(icmp6_mask->type, false, "type", "%"PRIu8,
                              icmp6_spec->type, icmp6_mask->type, 0);
            DUMP_PATTERN_ITEM(icmp6_mask->code, false, "code", "%"PRIu8,
                              icmp6_spec->code, icmp6_mask->code, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_TCP) {
        const struct rte_flow_item_tcp *tcp_spec = item->spec;
        const struct rte_flow_item_tcp *tcp_mask = item->mask;

        ds_put_cstr(s, "tcp ");
        if (tcp_spec) {
            if (!tcp_mask) {
                tcp_mask = &rte_flow_item_tcp_mask;
            }
            DUMP_PATTERN_ITEM(tcp_mask->hdr.src_port, false, "src", "%"PRIu16,
                              ntohs(tcp_spec->hdr.src_port),
                              ntohs(tcp_mask->hdr.src_port), 0);
            DUMP_PATTERN_ITEM(tcp_mask->hdr.dst_port, false, "dst", "%"PRIu16,
                              ntohs(tcp_spec->hdr.dst_port),
                              ntohs(tcp_mask->hdr.dst_port), 0);
            DUMP_PATTERN_ITEM(tcp_mask->hdr.tcp_flags, false, "flags",
                              "0x%"PRIx8, tcp_spec->hdr.tcp_flags,
                              tcp_mask->hdr.tcp_flags, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_IPV6) {
        const struct rte_flow_item_ipv6 *ipv6_spec = item->spec;
        const struct rte_flow_item_ipv6 *ipv6_mask = item->mask;

        char addr_str[INET6_ADDRSTRLEN];
        char mask_str[INET6_ADDRSTRLEN];
        struct in6_addr addr, mask;

        ds_put_cstr(s, "ipv6 ");
        if (ipv6_spec) {
            uint8_t has_frag_ext_mask;

            if (!ipv6_mask) {
                ipv6_mask = &rte_flow_item_ipv6_mask;
            }
            memcpy(&addr, ipv6_spec->hdr.src_addr, sizeof addr);
            memcpy(&mask, ipv6_mask->hdr.src_addr, sizeof mask);
            ipv6_string_mapped(addr_str, &addr);
            ipv6_string_mapped(mask_str, &mask);
            DUMP_PATTERN_ITEM(mask, false, "src", "%s",
                              addr_str, mask_str, "");

            memcpy(&addr, ipv6_spec->hdr.dst_addr, sizeof addr);
            memcpy(&mask, ipv6_mask->hdr.dst_addr, sizeof mask);
            ipv6_string_mapped(addr_str, &addr);
            ipv6_string_mapped(mask_str, &mask);
            DUMP_PATTERN_ITEM(mask, false, "dst", "%s",
                              addr_str, mask_str, "");

            DUMP_PATTERN_ITEM(ipv6_mask->hdr.proto, false, "proto", "%"PRIu8,
                              ipv6_spec->hdr.proto, ipv6_mask->hdr.proto, 0);
            DUMP_PATTERN_ITEM(ipv6_mask->hdr.vtc_flow, false,
                              "tc", "0x%"PRIx32,
                              ntohl(ipv6_spec->hdr.vtc_flow),
                              ntohl(ipv6_mask->hdr.vtc_flow), 0);
            DUMP_PATTERN_ITEM(ipv6_mask->hdr.hop_limits, false,
                              "hop", "%"PRIu8,
                              ipv6_spec->hdr.hop_limits,
                              ipv6_mask->hdr.hop_limits, 0);
            has_frag_ext_mask = ipv6_mask->has_frag_ext ? UINT8_MAX : 0;
            DUMP_PATTERN_ITEM(has_frag_ext_mask, false, "has_frag_ext",
                              "%"PRIu8, ipv6_spec->has_frag_ext,
                              ipv6_mask->has_frag_ext, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_IPV6_FRAG_EXT) {
        const struct rte_flow_item_ipv6_frag_ext *ipv6_frag_spec = item->spec;
        const struct rte_flow_item_ipv6_frag_ext *ipv6_frag_mask = item->mask;
        const struct rte_flow_item_ipv6_frag_ext *ipv6_frag_last = item->last;
        const struct rte_flow_item_ipv6_frag_ext ipv6_frag_def = {
            .hdr.next_header = 0, .hdr.frag_data = 0};

        ds_put_cstr(s, "ipv6_frag_ext ");
        if (ipv6_frag_spec) {
            if (!ipv6_frag_mask) {
                ipv6_frag_mask = &ipv6_frag_def;
            }
            if (!ipv6_frag_last) {
                ipv6_frag_last = &ipv6_frag_def;
            }
            DUMP_PATTERN_ITEM(ipv6_frag_mask->hdr.next_header, item->last,
                              "next_hdr", "%"PRIu8,
                              ipv6_frag_spec->hdr.next_header,
                              ipv6_frag_mask->hdr.next_header,
                              ipv6_frag_last->hdr.next_header);
            DUMP_PATTERN_ITEM(ipv6_frag_mask->hdr.frag_data, item->last,
                              "frag_data", "0x%"PRIx16,
                              ntohs(ipv6_frag_spec->hdr.frag_data),
                              ntohs(ipv6_frag_mask->hdr.frag_data),
                              ntohs(ipv6_frag_last->hdr.frag_data));
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_VXLAN) {
        const struct rte_flow_item_vxlan *vxlan_spec = item->spec;
        const struct rte_flow_item_vxlan *vxlan_mask = item->mask;
        ovs_be32 spec_vni, mask_vni;

        ds_put_cstr(s, "vxlan ");
        if (vxlan_spec) {
            if (!vxlan_mask) {
                vxlan_mask = &rte_flow_item_vxlan_mask;
            }
            spec_vni = get_unaligned_be32(ALIGNED_CAST(ovs_be32 *,
                                                       vxlan_spec->vni));
            mask_vni = get_unaligned_be32(ALIGNED_CAST(ovs_be32 *,
                                                       vxlan_mask->vni));
            DUMP_PATTERN_ITEM(vxlan_mask->vni, false, "vni", "%"PRIu32,
                              ntohl(spec_vni) >> 8, ntohl(mask_vni) >> 8, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_GRE) {
        const struct rte_flow_item_gre *gre_spec = item->spec;
        const struct rte_flow_item_gre *gre_mask = item->mask;
        const struct rte_gre_hdr *greh_spec, *greh_mask;
        uint8_t c_bit_spec, c_bit_mask;
        uint8_t k_bit_spec, k_bit_mask;

        ds_put_cstr(s, "gre ");
        if (gre_spec) {
            if (!gre_mask) {
                gre_mask = &rte_flow_item_gre_mask;
            }
            greh_spec = (struct rte_gre_hdr *) gre_spec;
            greh_mask = (struct rte_gre_hdr *) gre_mask;

            c_bit_spec = greh_spec->c;
            c_bit_mask = greh_mask->c ? UINT8_MAX : 0;
            DUMP_PATTERN_ITEM(c_bit_mask, false, "c_bit", "%"PRIu8,
                              c_bit_spec, c_bit_mask, 0);

            k_bit_spec = greh_spec->k;
            k_bit_mask = greh_mask->k ? UINT8_MAX : 0;
            DUMP_PATTERN_ITEM(k_bit_mask, false, "k_bit", "%"PRIu8,
                              k_bit_spec, k_bit_mask, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_GRE_KEY) {
        const rte_be32_t gre_mask = RTE_BE32(UINT32_MAX);
        const rte_be32_t *key_spec = item->spec;
        const rte_be32_t *key_mask = item->mask;

        ds_put_cstr(s, "gre_key ");
        if (key_spec) {
            if (!key_mask) {
                key_mask = &gre_mask;
            }
            DUMP_PATTERN_ITEM(*key_mask, false, "value", "%"PRIu32,
                              ntohl(*key_spec), ntohl(*key_mask), 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_TAG) {
        const struct rte_flow_item_tag *tag_spec = item->spec;
        const struct rte_flow_item_tag *tag_mask = item->mask;

        ds_put_cstr(s, "tag ");
        if (tag_spec) {
            if (!tag_mask) {
                tag_mask = &rte_flow_item_tag_mask;
            }
            DUMP_PATTERN_ITEM(tag_mask->index, false, "index", "%"PRIu8,
                              tag_spec->index, tag_mask->index, 0);
            DUMP_PATTERN_ITEM(tag_mask->data, false, "data", "0x%"PRIx32,
                              tag_spec->data, tag_mask->data, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_META) {
        const struct rte_flow_item_meta *meta_spec = item->spec;
        const struct rte_flow_item_meta *meta_mask = item->mask;

        ds_put_cstr(s, "meta ");
        if (meta_spec) {
            if (!meta_mask) {
                meta_mask = &rte_flow_item_meta_mask;
            }
            DUMP_PATTERN_ITEM(meta_mask->data, false, "data", "0x%"PRIx32,
                              meta_spec->data, meta_mask->data, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_MARK) {
        const struct rte_flow_item_mark *mark_spec = item->spec;
        const struct rte_flow_item_mark *mark_mask = item->mask;

        ds_put_cstr(s, "mark ");
        if (mark_spec) {
            ds_put_format(s, "id spec %d ", mark_spec->id);
        }
        if (mark_mask) {
            ds_put_format(s, "id mask %d ", mark_mask->id);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == OVS_RTE_FLOW_ITEM_TYPE(FLOW_INFO)) {
        const struct rte_flow_item_mark *mark_spec = item->spec;
        const struct rte_flow_item_mark *mark_mask = item->mask;

        ds_put_cstr(s, "flow-info ");
        if (mark_spec) {
            ds_put_format(s, "id spec %d ", mark_spec->id);
        }
        if (mark_mask) {
            ds_put_format(s, "id mask %d ", mark_mask->id);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_GENEVE) {
        const struct rte_flow_item_geneve *gnv_spec = item->spec;
        const struct rte_flow_item_geneve *gnv_mask = item->mask;
        ovs_be32 spec_vni, mask_vni;

        ds_put_cstr(s, "geneve ");
        if (gnv_spec) {
            if (!gnv_mask) {
                gnv_mask = &rte_flow_item_geneve_mask;
            }
            spec_vni = get_unaligned_be32(ALIGNED_CAST(ovs_be32 *,
                                                       gnv_spec->vni));
            mask_vni = get_unaligned_be32(ALIGNED_CAST(ovs_be32 *,
                                                       gnv_mask->vni));
            DUMP_PATTERN_ITEM(gnv_mask->vni, false, "vni", "%"PRIu32,
                              ntohl(spec_vni) >> 8, ntohl(mask_vni) >> 8, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_GENEVE_OPT) {
        const struct rte_flow_item_geneve_opt *opt_spec = item->spec;
        const struct rte_flow_item_geneve_opt *opt_mask = item->mask;
        uint8_t len, len_mask;
        int i;

        ds_put_cstr(s, "geneve-opt ");
        if (opt_spec) {
            if (!opt_mask) {
                opt_mask = &rte_flow_item_geneve_opt_mask;
            }
            DUMP_PATTERN_ITEM(opt_mask->option_class, false, "class",
                              "0x%"PRIx16, opt_spec->option_class,
                              opt_mask->option_class, 0);
            DUMP_PATTERN_ITEM(opt_mask->option_type, false, "type",
                              "0x%"PRIx8,opt_spec->option_type,
                              opt_mask->option_type, 0);
            len = opt_spec->option_len;
            len_mask = opt_mask->option_len;
            DUMP_PATTERN_ITEM(len_mask, false, "length", "0x%"PRIx8,
                              len, len_mask, 0);
            if (is_all_ones(opt_mask->data,
                            sizeof (uint32_t) * opt_spec->option_len)) {
                ds_put_cstr(s, "data is 0x");
                for (i = 0; i < opt_spec->option_len; i++) {
                    ds_put_format(s,"%"PRIx32"", htonl(opt_spec->data[i]));
                }
            } else if (!is_all_zeros(opt_mask->data,
                        sizeof (uint32_t) * opt_spec->option_len)) {
                ds_put_cstr(s, "data spec 0x");
                for (i = 0; i < opt_spec->option_len; i++) {
                    ds_put_format(s,"%"PRIx32"", htonl(opt_spec->data[i]));
                }
                ds_put_cstr(s, "data mask 0x");
                for (i = 0; i < opt_spec->option_len; i++) {
                    ds_put_format(s,"%"PRIx32"", htonl(opt_mask->data[i]));
                }
            }
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_PORT_ID) {
        const struct rte_flow_item_port_id *port_id_spec = item->spec;
        const struct rte_flow_item_port_id *port_id_mask = item->mask;

        ds_put_cstr(s, "port_id ");
        if (port_id_spec) {
            if (!port_id_mask) {
                port_id_mask = &rte_flow_item_port_id_mask;
            }
            DUMP_PATTERN_ITEM(port_id_mask->id, false, "id", "%"PRIu32,
                              port_id_spec->id, port_id_mask->id, 0);
        }
        ds_put_cstr(s, "/ ");
    } else if (item->type == RTE_FLOW_ITEM_TYPE_VOID) {
        ds_put_cstr(s, "void / ");
    } else if (item->type == OVS_RTE_FLOW_ITEM_TYPE(HASH)) {
        const struct rte_flow_item_mark *hash_spec = item->spec;
        const struct rte_flow_item_mark *hash_mask = item->mask;

        ds_put_cstr(s, "hash ");
        if (hash_spec) {
            ds_put_format(s, "id spec 0x%08x ", hash_spec->id);
        }
        if (hash_mask) {
            ds_put_format(s, "id mask 0x%x ", hash_mask->id);
        }
        ds_put_cstr(s, "/ ");
    } else {
        ds_put_format(s, "unknown rte flow pattern (%d)\n", item->type);
    }
}

static void
dump_raw_encap(struct ds *s,
               struct ds *s_extra,
               const struct rte_flow_action_raw_encap *raw_encap)
{
    int i;

    ds_put_cstr(s, "raw_encap index 0 / ");
    if (raw_encap) {
        ds_put_format(s_extra, "Raw-encap size=%ld set raw_encap 0 raw "
                      "pattern is ", raw_encap->size);
        for (i = 0; i < raw_encap->size; i++) {
            ds_put_format(s_extra, "%02x", raw_encap->data[i]);
        }
        ds_put_cstr(s_extra, " / end_set; ");
    }
}

static void
dump_port_id(struct ds *s, const void *conf)
{
    const struct rte_flow_action_port_id *port_id = conf;

    ds_put_cstr(s, "port_id ");
    if (port_id) {
        ds_put_format(s, "original %d id %d ", port_id->original,
                      port_id->id);
    }
}

static void
dump_flow_action(struct ds *s, struct ds *s_extra,
                 struct flow_actions *flow_actions, int act_index)
{
    const struct rte_flow_action *actions = &flow_actions->actions[act_index];

    if (actions->type == RTE_FLOW_ACTION_TYPE_END) {
        ds_put_cstr(s, "end");
    } else if (flow_actions->tnl_pmd_actions_cnt &&
               act_index >= flow_actions->tnl_pmd_actions_pos &&
               act_index < flow_actions->tnl_pmd_actions_pos +
                           flow_actions->tnl_pmd_actions_cnt) {
        /* Opaque PMD tunnel actions are skipped. */
        return;
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_MARK) {
        const struct rte_flow_action_mark *mark = actions->conf;

        ds_put_cstr(s, "mark ");
        if (mark) {
            ds_put_format(s, "id %d ", mark->id);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == OVS_RTE_FLOW_ACTION_TYPE(FLOW_INFO)) {
        const struct rte_flow_action_mark *mark = actions->conf;

        ds_put_cstr(s, "flow-info ");
        if (mark) {
            ds_put_format(s, "id %d ", mark->id);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_COUNT) {
        ds_put_cstr(s, "count / ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_PORT_ID) {
        dump_port_id(s, actions->conf);
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_DROP) {
        ds_put_cstr(s, "drop / ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_MAC_SRC ||
               actions->type == RTE_FLOW_ACTION_TYPE_SET_MAC_DST) {
        const struct action_set_data *asd = actions->conf;
        const struct rte_flow_action_set_mac *set_mac;

        char *dirstr = actions->type == RTE_FLOW_ACTION_TYPE_SET_MAC_DST
                       ? "dst" : "src";

        set_mac = (const struct rte_flow_action_set_mac *) asd->value;
        ds_put_format(s, "set_mac_%s ", dirstr);
        ds_put_format(s, "mac_addr "ETH_ADDR_FMT,
                ETH_ADDR_BYTES_ARGS(set_mac->mac_addr));
        if (is_all_ones(asd->mask, asd->size)) {
            ds_put_cstr(s, " ");
        } else {
            set_mac = (const struct rte_flow_action_set_mac *) asd->mask;
            ds_put_format(s, "/"ETH_ADDR_FMT" ",
                    ETH_ADDR_BYTES_ARGS(set_mac->mac_addr));
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV4_SRC ||
               actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV4_DST) {
        const struct action_set_data *asd = actions->conf;
        const struct rte_flow_action_set_ipv4 *set_ipv4;

        char *dirstr = actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV4_DST
                       ? "dst" : "src";

        set_ipv4 = ALIGNED_CAST(const struct rte_flow_action_set_ipv4 *,
                                asd->value);
        ds_put_format(s, "set_ipv4_%s ", dirstr);
        ds_put_format(s, "ipv4_addr "IP_FMT, IP_ARGS(set_ipv4->ipv4_addr));
        if (is_all_ones(asd->mask, asd->size)) {
            ds_put_cstr(s, " ");
        } else {
            set_ipv4 = ALIGNED_CAST(const struct rte_flow_action_set_ipv4 *,
                                    asd->mask);
            ds_put_format(s, "/"IP_FMT" ", IP_ARGS(set_ipv4->ipv4_addr));
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV4_TTL) {
        const struct action_set_data *asd = actions->conf;
        const struct rte_flow_action_set_ttl *set_ttl;

        set_ttl = (const struct rte_flow_action_set_ttl *) asd->value;
        ds_put_cstr(s, "set_ipv4_ttl ");
        ds_put_format(s, "ttl_value %d", set_ttl->ttl_value);
        if (is_all_ones(asd->mask, asd->size)) {
            ds_put_cstr(s, " ");
        } else {
            set_ttl = (const struct rte_flow_action_set_ttl *) asd->mask;
            ds_put_format(s, "/%d ", set_ttl->ttl_value);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV4_DSCP) {
        const struct action_set_data *asd = actions->conf;
        const struct rte_flow_action_set_dscp *set_dscp;

        set_dscp = (const struct rte_flow_action_set_dscp *) asd->value;
        ds_put_cstr(s, "set_ipv4_dscp ");
        ds_put_format(s, "dscp_value 0x%02x", set_dscp->dscp);
        if (is_all_ones(asd->mask, asd->size)) {
            ds_put_cstr(s, " ");
        } else {
            set_dscp = (const struct rte_flow_action_set_dscp *) asd->mask;
            ds_put_format(s, "/0x%02x ", set_dscp->dscp);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV6_HOP) {
        const struct action_set_data *asd = actions->conf;
        const struct rte_flow_action_set_ttl *set_ttl;

        ds_put_cstr(s, "set_ipv6_hop ");
        set_ttl = (const struct rte_flow_action_set_ttl *) asd->value;
        ds_put_format(s, "hop_value %d", set_ttl->ttl_value);
        if (is_all_ones(asd->mask, asd->size)) {
            ds_put_cstr(s, " ");
        } else {
            set_ttl = (const struct rte_flow_action_set_ttl *) asd->mask;
            ds_put_format(s, "/%d ", set_ttl->ttl_value);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV6_DSCP) {
        const struct action_set_data *asd = actions->conf;
        const struct rte_flow_action_set_dscp *set_dscp;

        set_dscp = (const struct rte_flow_action_set_dscp *) asd->value;
        ds_put_cstr(s, "set_ipv6_dscp ");
        ds_put_format(s, "dscp_value 0x%02x", set_dscp->dscp);
        if (is_all_ones(asd->mask, asd->size)) {
            ds_put_cstr(s, " ");
        } else {
            set_dscp = (const struct rte_flow_action_set_dscp *) asd->mask;
            ds_put_format(s, "/0x%02x ", set_dscp->dscp);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_TP_SRC ||
               actions->type == RTE_FLOW_ACTION_TYPE_SET_TP_DST ||
               actions->type == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_SRC) ||
               actions->type == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_SRC) ||
               actions->type == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_DST) ||
               actions->type == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_DST)) {
        const struct action_set_data *asd = actions->conf;
        const struct rte_flow_action_set_tp *set_tp;

        char *dirstr = actions->type == RTE_FLOW_ACTION_TYPE_SET_TP_DST ||
                       actions->type == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_DST) ||
                       actions->type == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_DST)
                       ? "dst" : "src";

        ds_put_format(s, "set_tp_%s", dirstr);
        set_tp = ALIGNED_CAST(const struct rte_flow_action_set_tp *, asd->value);
        ds_put_format(s, "port %"PRIu16, ntohs(set_tp->port));
        if (is_all_ones(asd->mask, asd->size)) {
            ds_put_cstr(s, " ");
        } else {
            set_tp = ALIGNED_CAST(const struct rte_flow_action_set_tp *, asd->mask);
            ds_put_format(s, "%"PRIu16" ", ntohs(set_tp->port));
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_OF_PUSH_VLAN) {
        const struct rte_flow_action_of_push_vlan *of_push_vlan =
            actions->conf;

        ds_put_cstr(s, "of_push_vlan ");
        if (of_push_vlan) {
            ds_put_format(s, "ethertype 0x%"PRIx16" ",
                          ntohs(of_push_vlan->ethertype));
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_OF_SET_VLAN_PCP) {
        const struct rte_flow_action_of_set_vlan_pcp *of_set_vlan_pcp =
            actions->conf;

        ds_put_cstr(s, "of_set_vlan_pcp ");
        if (of_set_vlan_pcp) {
            ds_put_format(s, "vlan_pcp %"PRIu8" ", of_set_vlan_pcp->vlan_pcp);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_OF_SET_VLAN_VID) {
        const struct rte_flow_action_of_set_vlan_vid *of_set_vlan_vid =
            actions->conf;

        ds_put_cstr(s, "of_set_vlan_vid ");
        if (of_set_vlan_vid) {
            ds_put_format(s, "vlan_vid %"PRIu16" ",
                          ntohs(of_set_vlan_vid->vlan_vid));
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_OF_POP_VLAN) {
        ds_put_cstr(s, "of_pop_vlan / ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV6_SRC ||
               actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV6_DST) {
        const struct rte_flow_action_set_ipv6 *set_ipv6 = actions->conf;

        char *dirstr = actions->type == RTE_FLOW_ACTION_TYPE_SET_IPV6_DST
                       ? "dst" : "src";

        ds_put_format(s, "set_ipv6_%s ", dirstr);
        if (set_ipv6) {
            struct in6_addr addr;

            ds_put_cstr(s, "ipv6_addr ");
            memcpy(&addr, set_ipv6->ipv6_addr, sizeof addr);
            ipv6_format_addr(&addr, s);
            ds_put_cstr(s, " ");
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_RAW_ENCAP) {
        const struct rte_flow_action_raw_encap *raw_encap = actions->conf;

        dump_raw_encap(s, s_extra, raw_encap);
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_JUMP) {
        const struct rte_flow_action_jump *jump = actions->conf;

        ds_put_cstr(s, "jump ");
        if (jump) {
            ds_put_format(s, "group 0x%"PRIx32" ", jump->group);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_VXLAN_DECAP) {
        ds_put_cstr(s, "vxlan_decap / ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_NVGRE_DECAP) {
        ds_put_cstr(s, "nvgre_decap / ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_TAG) {
        const struct rte_flow_action_set_tag *set_tag = actions->conf;

        ds_put_cstr(s, "set_tag ");
        if (set_tag) {
            ds_put_format(s, "index %u data 0x%08x mask 0x%08x ",
                          set_tag->index, set_tag->data, set_tag->mask);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SET_META) {
        const struct rte_flow_action_set_meta *meta = actions->conf;

        ds_put_cstr(s, "set_meta ");
        if (meta) {
            ds_put_format(s, "data 0x%08x mask 0x%08x ", meta->data,
                          meta->mask);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == OVS_RTE_FLOW_ACTION_TYPE(CT_INFO)) {
        const struct rte_flow_action_set_meta *meta = actions->conf;

        ds_put_cstr(s, "ct-info ");
        if (meta) {
            ds_put_format(s, "data 0x%08x mask 0x%08x ", meta->data,
                          meta->mask);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_INDIRECT) {
        ds_put_format(s, "indirect %p / ", actions->conf);
        ds_put_format(s_extra, "flow indirect_action 0 create transfer"
                      " action_id %p action ",
                      actions->conf);
        if (act_index == flow_actions->shared_age_action_pos &&
            flow_actions->shared_count_action_pos !=
            flow_actions->shared_age_action_pos) {
            ds_put_cstr(s_extra, "age timeout 0xffffff / end;");
        } else if (act_index == flow_actions->shared_count_action_pos) {
            ds_put_cstr(s_extra, "count / end;");
        } else {
            ds_put_cstr(s_extra, "UNKONWN / end;");
        }
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_RAW_DECAP) {
        const struct rte_flow_action_raw_decap *raw_decap = actions->conf;

        ds_put_cstr(s, "raw_decap index 0 / ");
        if (raw_decap) {
            ds_put_format(s_extra, "%s", ds_cstr(&flow_actions->s_tnl));
        }
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_SAMPLE) {
        const struct rte_flow_action_sample *sample = actions->conf;

        if (sample) {
            const struct rte_flow_action *rte_actions;

            rte_actions = sample->actions;
            ds_put_format(s_extra, "set sample_actions %d ", act_index);
            while (rte_actions &&
                   rte_actions->type != RTE_FLOW_ACTION_TYPE_END) {
                if (rte_actions->type == RTE_FLOW_ACTION_TYPE_PORT_ID) {
                    dump_port_id(s_extra, rte_actions->conf);
                } else if (rte_actions->type ==
                           RTE_FLOW_ACTION_TYPE_RAW_ENCAP) {
                    const struct rte_flow_action_raw_encap *raw_encap =
                        rte_actions->conf;

                    dump_raw_encap(s, s_extra, raw_encap);
                    ds_put_format(s_extra, "raw_encap index 0 / ");
                } else {
                    ds_put_format(s, "unknown rte flow action (%d)\n",
                                  rte_actions->type);
                }
                rte_actions++;
            }
            ds_put_cstr(s_extra, "/ end; ");
            ds_put_format(s, "sample ratio %d index %d / ", sample->ratio,
                          act_index);
        }
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_METER) {
        const struct meter_data *meter = actions->conf;

        ds_put_cstr(s, "meter ");
        if (meter) {
            ds_put_format(s, "mtr_id %d (flow_id %d) ",
                          meter->conf.mtr_id, meter->flow_id);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_VOID) {
        ds_put_cstr(s, "void / ");
    } else if (actions->type == RTE_FLOW_ACTION_TYPE_QUEUE) {
        const struct rte_flow_action_queue *queue = actions->conf;

        ds_put_cstr(s, "queue ");
        if (queue) {
            ds_put_format(s, "index %d ", queue->index);
        }
        ds_put_cstr(s, "/ ");
    } else if (actions->type == OVS_RTE_FLOW_ACTION_TYPE(PRE_CT_END)) {
        /* This is only a dummy action used to split pre and post CT. It
         * should never be actually used.
         */
        OVS_NOT_REACHED();
    } else if (actions->type == OVS_RTE_FLOW_ACTION_TYPE(HASH)) {
        const struct hash_data *hash_data = actions->conf;

        ds_put_cstr(s, "hash ");
        if (hash_data) {
            ds_put_format(s, "(flow_id %d) ", hash_data->flow_id);
        }
        ds_put_cstr(s, "/ ");
    } else {
        ds_put_format(s, "unknown rte flow action (%d)\n", actions->type);
    }
}

static struct ds *
dump_flow(struct ds *s, struct ds *s_extra,
          const struct rte_flow_attr *attr,
          struct flow_patterns *flow_patterns,
          struct flow_actions *flow_actions)
{
    int i;

    if (attr) {
        dump_flow_attr(s, s_extra, attr, flow_patterns, flow_actions);
    }
    ds_put_cstr(s, "pattern ");
    for (i = 0; i < flow_patterns->cnt; i++) {
        dump_flow_pattern(s, flow_patterns, i);
    }
    ds_put_cstr(s, "actions ");
    for (i = 0; i < flow_actions->cnt; i++) {
        dump_flow_action(s, s_extra, flow_actions, i);
    }
    return s;
}

enum ct_mode {
    CT_MODE_NONE,
    CT_MODE_CT,
    CT_MODE_CT_NAT,
    CT_MODE_CT_CONN,
};

struct act_vars {
    enum ct_mode ct_mode;
    uint8_t pre_ct_cnt;
    odp_port_t vport;
    uint32_t recirc_id;
    struct flow_tnl *tnl_key;
    struct flow_tnl tnl_mask;
    bool is_e2e_cache;
    uintptr_t ct_counter_key;
    struct flows_counter_key flows_counter_key;
    enum tun_type tun_type;
    bool is_outer_ipv4;
    uint8_t gnv_opts_cnt;
    bool is_ct_conn;
    rte_be16_t vlan_tpid;
    uint8_t vlan_pcp;
    uint8_t proto;
    bool has_dp_hash;
    odp_port_t tnl_push_out_port;
    bool has_known_fate;
};

static int
dpdk_offload_rte_create(struct netdev *netdev,
                        const struct rte_flow_attr *attr,
                        struct rte_flow_item *items,
                        struct rte_flow_action *actions,
                        struct dpdk_offload_handle *doh,
                        struct rte_flow_error *error)
{
    struct rte_flow_action *a;
    struct rte_flow_item *it;

    for (it = items; it->type != RTE_FLOW_ITEM_TYPE_END; it++) {
        if (it->type == OVS_RTE_FLOW_ITEM_TYPE(FLOW_INFO)) {
            it->type = RTE_FLOW_ITEM_TYPE_MARK;
        } else if (it->type == OVS_RTE_FLOW_ITEM_TYPE(HASH)) {
            return -1;
        }
    }

    for (a = actions; a->type != RTE_FLOW_ACTION_TYPE_END; a++) {
        int act_type = a->type;

        if (act_type == OVS_RTE_FLOW_ACTION_TYPE(FLOW_INFO)) {
            a->type = RTE_FLOW_ACTION_TYPE_MARK;
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(CT_INFO)) {
            struct rte_flow_action_set_meta *set_meta;
            struct reg_field *reg_field;

            a->type = RTE_FLOW_ACTION_TYPE_SET_META;
            reg_field = &reg_fields[REG_FIELD_CT_CTX];
            set_meta = CONST_CAST(struct rte_flow_action_set_meta *, a->conf);
            set_meta->data = (set_meta->data & reg_field->mask) << reg_field->offset;
            set_meta->mask = reg_field->mask << reg_field->offset;
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(PRE_CT_END)) {
            /* This is only a dummy action used to split pre and post CT. It
             * should never be actually used.
             */
            OVS_NOT_REACHED();
        } else if (act_type == OVS_RTE_FLOW_ACTION_TYPE(HASH)) {
            return -1;
        } else if (IS_REWRITE_ACTION(act_type)) {
            struct action_set_data *asd = (struct action_set_data *) a->conf;

            /* DPDK does not support partially masked set actions. In such
             * case, fail the offload.
             */
            if (!is_all_ones(asd->mask, asd->size)) {
                VLOG_DBG_RL(&rl, "Partial mask is not supported");
                return -1;
            }
            if (act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_DST) ||
                act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_DST)) {
                a->type = RTE_FLOW_ACTION_TYPE_SET_TP_DST;
            }
            if (act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_SRC) ||
                act_type == OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_SRC)) {
                a->type = RTE_FLOW_ACTION_TYPE_SET_TP_SRC;
            }
        } else if (act_type == RTE_FLOW_ACTION_TYPE_SET_IPV4_DSCP ||
                   act_type == RTE_FLOW_ACTION_TYPE_SET_IPV6_DSCP) {
            return -1;
        }
    }

    doh->rte_flow = netdev_dpdk_rte_flow_create(netdev, attr, items, actions,
                                                error);
    if (doh->rte_flow != NULL) {
        dpdk_offload_counter_inc(netdev);
    }
    return doh->rte_flow == NULL ? -1 : 0;
}

static int
create_rte_flow(struct netdev *netdev,
                const struct rte_flow_attr *attr,
                struct flow_patterns *flow_patterns,
                struct flow_actions *flow_actions,
                struct dpdk_offload_handle *doh,
                struct rte_flow_error *error)
{
    struct rte_flow_action *actions = flow_actions->actions;
    struct rte_flow_item *items = flow_patterns->items;
    struct ds s_extra = DS_EMPTY_INITIALIZER;
    struct ds s = DS_EMPTY_INITIALIZER;
    char *extra_str;
    int rv;

    rv = offload->create(netdev, attr, items, actions, doh, error);
    if (rv != 0) {
        enum vlog_level level = VLL_WARN;

        if (error->type == RTE_FLOW_ERROR_TYPE_ACTION) {
            level = VLL_DBG;
        }
        VLOG_RL(&rl, level, "%s: rte_flow creation failed: %d (%s).",
                netdev_get_name(netdev), error->type, error->message);
        if (!vlog_should_drop(&this_module, level, &rl)) {
            dump_flow(&s, &s_extra, attr, flow_patterns, flow_actions);
            extra_str = ds_cstr(&s_extra);
            VLOG_RL(&rl, level, "%s: Failed flow: %s  flow create %d %s",
                    netdev_get_name(netdev), extra_str,
                    attr->transfer
                    ? netdev_dpdk_get_esw_mgr_port_id(netdev)
                    : netdev_dpdk_get_port_id(netdev), ds_cstr(&s));
        }
    } else {
        doh->valid = true;
        if (!VLOG_DROP_DBG(&rl)) {
            dump_flow(&s, &s_extra, attr, flow_patterns, flow_actions);
            extra_str = ds_cstr(&s_extra);
            VLOG_DBG_RL(&rl, "%s: %s  flow create %d user_id 0x%"PRIxPTR" %s",
                        netdev_get_name(netdev), extra_str,
                        attr->transfer
                        ? netdev_dpdk_get_esw_mgr_port_id(netdev)
                        : netdev_dpdk_get_port_id(netdev),
                        (intptr_t) doh->rte_flow, ds_cstr(&s));
        }
    }

    ds_destroy(&s);
    ds_destroy(&s_extra);
    return rv;
}

static void
add_flow_pattern(struct flow_patterns *patterns, enum rte_flow_item_type type,
                 const void *spec, const void *mask, const void *last)
{
    int cnt = patterns->cnt;

    if (cnt == 0) {
        patterns->current_max = 8;
        patterns->items = per_thread_xcalloc(patterns->current_max,
                                             sizeof *patterns->items);
    } else if (cnt == patterns->current_max) {
        patterns->current_max *= 2;
        patterns->items = per_thread_xrealloc(patterns->items,
                                              patterns->current_max / 2 *
                                              sizeof *patterns->items,
                                              patterns->current_max *
                                              sizeof *patterns->items);
    }

    patterns->items[cnt].type = type;
    patterns->items[cnt].spec = spec;
    patterns->items[cnt].mask = mask;
    patterns->items[cnt].last = last;
    patterns->cnt++;
}

static void
add_flow_action(struct flow_actions *actions, enum rte_flow_action_type type,
                const void *conf)
{
    int cnt = actions->cnt;

    if (cnt == 0) {
        actions->current_max = 8;
        actions->actions = per_thread_xcalloc(actions->current_max,
                                              sizeof *actions->actions);
    } else if (cnt == actions->current_max) {
        actions->current_max *= 2;
        actions->actions = per_thread_xrealloc(actions->actions,
                                               actions->current_max / 2 *
                                               sizeof *actions->actions,
                                               actions->current_max *
                                               sizeof *actions->actions);
    }

    actions->actions[cnt].type = type;
    actions->actions[cnt].conf = conf;
    actions->cnt++;
}

OVS_UNUSED
static void
add_flow_tnl_actions(struct flow_actions *actions,
                     struct netdev *tnl_netdev,
                     struct rte_flow_action *tnl_pmd_actions,
                     uint32_t tnl_pmd_actions_cnt)
{
    int i;

    actions->tnl_netdev = tnl_netdev;
    actions->tnl_pmd_actions_pos = actions->cnt;
    actions->tnl_pmd_actions = tnl_pmd_actions;
    actions->tnl_pmd_actions_cnt = tnl_pmd_actions_cnt;
    for (i = 0; i < tnl_pmd_actions_cnt; i++) {
        add_flow_action(actions, tnl_pmd_actions[i].type,
                        tnl_pmd_actions[i].conf);
    }
}

static void
add_flow_tnl_items(struct flow_patterns *patterns,
                   struct netdev *physdev,
                   struct rte_flow_item *tnl_pmd_items,
                   uint32_t tnl_pmd_items_cnt)
{
    int i;

    patterns->physdev = physdev;
    patterns->tnl_pmd_items = tnl_pmd_items;
    patterns->tnl_pmd_items_cnt = tnl_pmd_items_cnt;
    for (i = 0; i < tnl_pmd_items_cnt; i++) {
        add_flow_pattern(patterns, tnl_pmd_items[i].type,
                         tnl_pmd_items[i].spec, tnl_pmd_items[i].mask, NULL);
    }
}

static void
free_flow_patterns(struct flow_patterns *patterns)
{
    struct rte_flow_error error;
    int i;

    if (patterns->tnl_pmd_items) {
        struct rte_flow_item *tnl_pmd_items = patterns->tnl_pmd_items;
        uint32_t tnl_pmd_items_cnt = patterns->tnl_pmd_items_cnt;
        struct netdev *physdev = patterns->physdev;

        if (netdev_dpdk_rte_flow_tunnel_item_release(physdev, tnl_pmd_items,
                                                     tnl_pmd_items_cnt,
                                                     &error)) {
            VLOG_DBG_RL(&rl, "%s: netdev_dpdk_rte_flow_tunnel_item_release "
                        "failed: %d (%s).", netdev_get_name(physdev),
                        error.type, error.message);
        }
    }

    for (i = patterns->tnl_pmd_items_cnt; i < patterns->cnt; i++) {
        if (patterns->items[i].spec) {
            per_thread_free(CONST_CAST(void *, patterns->items[i].spec));
        }
        if (patterns->items[i].mask) {
            per_thread_free(CONST_CAST(void *, patterns->items[i].mask));
        }
        if (patterns->items[i].last) {
            per_thread_free(CONST_CAST(void *, patterns->items[i].last));
        }
    }
    per_thread_free(patterns->items);
    patterns->items = NULL;
    patterns->cnt = 0;
    ds_destroy(&patterns->s_tnl);
}

static void
flow_actions_create_from(struct flow_actions *flow_actions,
                         const struct rte_flow_action *actions)
{
    memset(flow_actions, 0, sizeof *flow_actions);

    for (; actions && actions->type != RTE_FLOW_ACTION_TYPE_END; actions++) {
        add_flow_action(flow_actions, actions->type, actions->conf);
    }
}

static void
free_flow_actions(struct flow_actions *actions, bool free_confs)
{
    struct rte_flow_error error;
    int i;

    for (i = 0; free_confs && i < actions->cnt; i++) {
        if (actions->tnl_pmd_actions_cnt &&
            i == actions->tnl_pmd_actions_pos) {
            if (netdev_dpdk_rte_flow_tunnel_action_decap_release(
                    actions->tnl_netdev, actions->tnl_pmd_actions,
                    actions->tnl_pmd_actions_cnt, &error)) {
                VLOG_DBG_RL(&rl, "%s: "
                            "netdev_dpdk_rte_flow_tunnel_action_decap_release "
                            "failed: %d (%s).",
                            netdev_get_name(actions->tnl_netdev),
                            error.type, error.message);
            }
            i += actions->tnl_pmd_actions_cnt - 1;
            continue;
        }
        if (actions->actions[i].type == RTE_FLOW_ACTION_TYPE_INDIRECT) {
            continue;
        }
        if (actions->actions[i].type == RTE_FLOW_ACTION_TYPE_SAMPLE) {
            const struct rte_flow_action_sample *sample;
            struct flow_actions sample_flow_actions;

            sample = actions->actions[i].conf;
            flow_actions_create_from(&sample_flow_actions, sample->actions);
            free_flow_actions(&sample_flow_actions, free_confs);
        }
        if (actions->actions[i].conf) {
            per_thread_free(CONST_CAST(void *, actions->actions[i].conf));
        }
    }
    per_thread_free(actions->actions);
    actions->actions = NULL;
    actions->cnt = 0;
    ds_destroy(&actions->s_tnl);
}

OVS_UNUSED
static int
vport_to_rte_tunnel(struct netdev *vport,
                    struct rte_flow_tunnel *tunnel,
                    struct netdev *netdev,
                    struct ds *s_tnl)
{
    const struct netdev_tunnel_config *tnl_cfg;

    memset(tunnel, 0, sizeof *tunnel);

    tnl_cfg = netdev_get_tunnel_config(vport);
    if (!tnl_cfg) {
        return -1;
    }

    if (!IN6_IS_ADDR_V4MAPPED(&tnl_cfg->ipv6_dst)) {
        tunnel->is_ipv6 = true;
    }

    if (!strcmp(netdev_get_type(vport), "vxlan")) {
        tunnel->type = RTE_FLOW_ITEM_TYPE_VXLAN;
        tunnel->tp_dst = tnl_cfg->dst_port;
        if (!VLOG_DROP_DBG(&rl)) {
            ds_put_format(s_tnl, "flow tunnel create %d type vxlan; ",
                          netdev_dpdk_get_port_id(netdev));
        }
    } else if (!strcmp(netdev_get_type(vport), "gre")) {
        tunnel->type = RTE_FLOW_ITEM_TYPE_GRE;
        if (!VLOG_DROP_DBG(&rl)) {
            ds_put_format(s_tnl, "flow tunnel create %d type gre; ",
                          netdev_dpdk_get_port_id(netdev));
        }
    } else {
        VLOG_DBG_RL(&rl, "vport type '%s' is not supported",
                    netdev_get_type(vport));
        return -1;
    }

    return 0;
}

static int
add_vport_match(struct flow_patterns *patterns,
                odp_port_t orig_in_port,
                struct netdev *tnldev)
{
    struct netdev *physdev;

    physdev = netdev_ports_get(orig_in_port, tnldev->dpif_type);
    if (physdev == NULL) {
        return -1;
    }

    add_flow_tnl_items(patterns, physdev, NULL, 0);

    netdev_close(physdev);
    return 0;
}

static int
netdev_offload_dpdk_destroy_flow(struct netdev *netdev,
                                 struct dpdk_offload_handle *doh,
                                 const ovs_u128 *ufid, bool is_esw)
{
    struct uuid ufid0 = UUID_ZERO;
    struct rte_flow_error error;
    int ret;

    ret = offload->destroy(netdev, doh, &error, is_esw);
    if (!ret) {
        VLOG_DBG_RL(&rl, "%s: flow destroy %d user_id rule 0x%"PRIxPTR" ufid "
                    UUID_FMT, netdev_get_name(netdev),
                    is_esw ? netdev_dpdk_get_esw_mgr_port_id(netdev)
                           : netdev_dpdk_get_port_id(netdev),
                    (intptr_t) doh->rte_flow,
                    UUID_ARGS(ufid ? (struct uuid *) ufid : &ufid0));
        /* In case of modification, this handle could be re-used.
         * Clear up relevant fields to make sure it is usable. */
        doh->rte_flow = NULL;
        doh->valid = false;
    } else {
        VLOG_ERR("Failed: %s: flow destroy %d user_id rule 0x%"PRIxPTR" ufid "
                 UUID_FMT " %s (%u)", netdev_get_name(netdev),
                 is_esw ? netdev_dpdk_get_esw_mgr_port_id(netdev)
                        : netdev_dpdk_get_port_id(netdev),
                 (intptr_t) doh->rte_flow,
                 UUID_ARGS(ufid ? (struct uuid *) ufid : &ufid0),
                 error.message, error.type);
        return -1;
    }

    return ret;
}

static int
parse_tnl_ip_match(struct flow_patterns *patterns,
                   struct match *match,
                   uint8_t proto)
{
    struct flow *consumed_masks;

    consumed_masks = &match->wc.masks;
    /* IP v4 */
    if (match->wc.masks.tunnel.ip_src || match->wc.masks.tunnel.ip_dst) {
        struct rte_flow_item_ipv4 *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.type_of_service = match->flow.tunnel.ip_tos;
        spec->hdr.time_to_live    = match->flow.tunnel.ip_ttl;
        spec->hdr.next_proto_id   = proto;
        spec->hdr.src_addr        = match->flow.tunnel.ip_src;
        spec->hdr.dst_addr        = match->flow.tunnel.ip_dst;

        mask->hdr.type_of_service = match->wc.masks.tunnel.ip_tos;
        mask->hdr.time_to_live    = match->wc.masks.tunnel.ip_ttl;
        mask->hdr.next_proto_id   = UINT8_MAX;
        mask->hdr.src_addr        = match->wc.masks.tunnel.ip_src;
        mask->hdr.dst_addr        = match->wc.masks.tunnel.ip_dst;

        consumed_masks->tunnel.ip_tos = 0;
        consumed_masks->tunnel.ip_ttl = 0;
        consumed_masks->tunnel.ip_src = 0;
        consumed_masks->tunnel.ip_dst = 0;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_IPV4, spec, mask, NULL);
    } else if (!is_all_zeros(&match->wc.masks.tunnel.ipv6_src,
                             sizeof(struct in6_addr)) ||
               !is_all_zeros(&match->wc.masks.tunnel.ipv6_dst,
                             sizeof(struct in6_addr))) {
        /* IP v6 */
        struct rte_flow_item_ipv6 *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.proto = proto;
        spec->hdr.hop_limits = match->flow.tunnel.ip_ttl;
        spec->hdr.vtc_flow = htonl((uint32_t) match->flow.tunnel.ip_tos <<
                                   RTE_IPV6_HDR_TC_SHIFT);
        memcpy(spec->hdr.src_addr, &match->flow.tunnel.ipv6_src,
               sizeof spec->hdr.src_addr);
        memcpy(spec->hdr.dst_addr, &match->flow.tunnel.ipv6_dst,
               sizeof spec->hdr.dst_addr);

        mask->hdr.proto = UINT8_MAX;
        mask->hdr.hop_limits = match->wc.masks.tunnel.ip_ttl;
        mask->hdr.vtc_flow = htonl((uint32_t) match->wc.masks.tunnel.ip_tos <<
                                   RTE_IPV6_HDR_TC_SHIFT);
        memcpy(mask->hdr.src_addr, &match->wc.masks.tunnel.ipv6_src,
               sizeof mask->hdr.src_addr);
        memcpy(mask->hdr.dst_addr, &match->wc.masks.tunnel.ipv6_dst,
               sizeof mask->hdr.dst_addr);

        consumed_masks->tunnel.ip_tos = 0;
        consumed_masks->tunnel.ip_ttl = 0;
        memset(&consumed_masks->tunnel.ipv6_src, 0,
               sizeof consumed_masks->tunnel.ipv6_src);
        memset(&consumed_masks->tunnel.ipv6_dst, 0,
               sizeof consumed_masks->tunnel.ipv6_dst);

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_IPV6, spec, mask, NULL);
    } else if (!(match->wc.masks.tunnel.flags & FLOW_TNL_F_EXPLICIT)) {
        VLOG_ERR_RL(&rl, "Tunnel L3 protocol is neither IPv4 nor IPv6");
        return -1;
    }

    return 0;
}

static void
parse_tnl_udp_match(struct flow_patterns *patterns,
                    struct match *match)
{
    struct flow *consumed_masks;
    struct rte_flow_item_udp *spec, *mask;

    consumed_masks = &match->wc.masks;

    spec = per_thread_xzalloc(sizeof *spec);
    mask = per_thread_xzalloc(sizeof *mask);

    spec->hdr.src_port = match->flow.tunnel.tp_src;
    spec->hdr.dst_port = match->flow.tunnel.tp_dst;

    mask->hdr.src_port = match->wc.masks.tunnel.tp_src;
    mask->hdr.dst_port = match->wc.masks.tunnel.tp_dst;

    consumed_masks->tunnel.tp_src = 0;
    consumed_masks->tunnel.tp_dst = 0;

    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_UDP, spec, mask, NULL);
}

static int
parse_vxlan_match(struct flow_patterns *patterns,
                  struct match *match)
{
    struct rte_flow_item_vxlan *vx_spec, *vx_mask;
    struct flow *consumed_masks;
    int ret;

    ret = parse_tnl_ip_match(patterns, match, IPPROTO_UDP);
    if (ret) {
        return -1;
    }
    parse_tnl_udp_match(patterns, match);

    consumed_masks = &match->wc.masks;
    /* VXLAN */
    vx_spec = per_thread_xzalloc(sizeof *vx_spec);
    vx_mask = per_thread_xzalloc(sizeof *vx_mask);

    put_unaligned_be32(ALIGNED_CAST(ovs_be32 *, vx_spec->vni),
                       htonl(ntohll(match->flow.tunnel.tun_id) << 8));
    put_unaligned_be32(ALIGNED_CAST(ovs_be32 *, vx_mask->vni),
                       htonl(ntohll(match->wc.masks.tunnel.tun_id) << 8));

    consumed_masks->tunnel.tun_id = 0;
    consumed_masks->tunnel.flags = 0;

    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_VXLAN, vx_spec, vx_mask,
                     NULL);
    return 0;
}

static int
parse_gre_match(struct flow_patterns *patterns,
                struct match *match)
{
    struct rte_flow_item_gre *gre_spec, *gre_mask;
    struct rte_gre_hdr *greh_spec, *greh_mask;
    rte_be32_t *key_spec, *key_mask;
    struct flow *consumed_masks;
    int ret;


    ret = parse_tnl_ip_match(patterns, match, IPPROTO_GRE);
    if (ret) {
        return -1;
    }

    gre_spec = per_thread_xzalloc(sizeof *gre_spec);
    gre_mask = per_thread_xzalloc(sizeof *gre_mask);
    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_GRE, gre_spec, gre_mask,
                     NULL);

    consumed_masks = &match->wc.masks;

    greh_spec = (struct rte_gre_hdr *) gre_spec;
    greh_mask = (struct rte_gre_hdr *) gre_mask;

    if (match->wc.masks.tunnel.flags & FLOW_TNL_F_CSUM) {
        greh_spec->c = !!(match->flow.tunnel.flags & FLOW_TNL_F_CSUM);
        greh_mask->c = 1;
        consumed_masks->tunnel.flags &= ~FLOW_TNL_F_CSUM;
    }

    if (match->wc.masks.tunnel.flags & FLOW_TNL_F_KEY) {
        greh_spec->k = !!(match->flow.tunnel.flags & FLOW_TNL_F_KEY);
        greh_mask->k = 1;

        if (greh_spec->k) {
            key_spec = per_thread_xzalloc(sizeof *key_spec);
            key_mask = per_thread_xzalloc(sizeof *key_mask);

            *key_spec = htonl(ntohll(match->flow.tunnel.tun_id));
            *key_mask = htonl(ntohll(match->wc.masks.tunnel.tun_id));

            add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_GRE_KEY, key_spec,
                             key_mask, NULL);
        }
        consumed_masks->tunnel.tun_id = 0;
        consumed_masks->tunnel.flags &= ~FLOW_TNL_F_KEY;
    }

    consumed_masks->tunnel.flags &= ~FLOW_TNL_F_DONT_FRAGMENT;

    return 0;
}

static void
parse_geneve_opt_match(struct flow *consumed_masks,
                       struct flow_patterns *patterns,
                       struct match *match,
                       struct act_vars *act_vars)
{
    int len, opt_idx;
    uint8_t idx;
    struct geneve_opt curr_opt_spec, curr_opt_mask;
    struct gnv_opts {
        struct rte_flow_item_geneve_opt opts[TUN_METADATA_NUM_OPTS];
        uint32_t options_data[TUN_METADATA_NUM_OPTS];
    } *gnv_opts;
    BUILD_ASSERT_DECL(offsetof(struct gnv_opts, opts) == 0);
    struct gnv_opts_mask {
        struct rte_flow_item_geneve_opt opts_mask[TUN_METADATA_NUM_OPTS];
        uint32_t options_data_mask[TUN_METADATA_NUM_OPTS];
    } *gnv_opts_mask;
    BUILD_ASSERT_DECL(offsetof(struct gnv_opts_mask, opts_mask) == 0);

    len = match->flow.tunnel.metadata.present.len;
    idx = 0;
    opt_idx = 0;
    curr_opt_spec = match->flow.tunnel.metadata.opts.gnv[opt_idx];
    curr_opt_mask = match->wc.masks.tunnel.metadata.opts.gnv[opt_idx];

    if (!is_all_zeros(match->wc.masks.tunnel.metadata.opts.gnv,
                      sizeof *match->wc.masks.tunnel.metadata.opts.gnv) &&
        match->flow.tunnel.metadata.present.len) {
        while (len) {
            gnv_opts = per_thread_xzalloc(sizeof *gnv_opts);
            gnv_opts_mask = per_thread_xzalloc(sizeof *gnv_opts_mask);
            memcpy(&gnv_opts->opts[idx].option_class,
                   &curr_opt_spec.opt_class, sizeof curr_opt_spec.opt_class);
            memcpy(&gnv_opts_mask->opts_mask[idx].option_class,
                   &curr_opt_mask.opt_class,
                   sizeof curr_opt_mask.opt_class);

            gnv_opts->opts[idx].option_type = curr_opt_spec.type;
            gnv_opts_mask->opts_mask[idx].option_type = curr_opt_mask.type;

            gnv_opts->opts[idx].option_len = curr_opt_spec.length;
            gnv_opts_mask->opts_mask[idx].option_len = curr_opt_mask.length;

            /* According to the Geneve protocol
            * https://tools.ietf.org/html/draft-gross-geneve-00#section-3.1
            * Length (5 bits):  Length of the option, expressed in four byte
            * multiples excluding the option header
            * (tunnel.metadata.opts.gnv.length).
            * Opt Len (6 bits):  The length of the options fields, expressed
            * in four byte multiples, not including the eight byte
            * fixed tunnel header (tunnel.metadata.present.len).
            */
            opt_idx++;
            memcpy(&gnv_opts->options_data[opt_idx - 1],
                   &match->flow.tunnel.metadata.opts.gnv[opt_idx],
                   sizeof gnv_opts->options_data[opt_idx - 1] *
                   curr_opt_spec.length * 4);
            memcpy(&gnv_opts_mask->options_data_mask[opt_idx - 1],
                   &match->wc.masks.tunnel.metadata.opts.gnv[opt_idx],
                   sizeof gnv_opts_mask->options_data_mask[opt_idx - 1] *
                   curr_opt_spec.length * 4);

            gnv_opts->opts[opt_idx - 1].data = gnv_opts->options_data;
            gnv_opts_mask->opts_mask[opt_idx - 1].data =
            gnv_opts_mask->options_data_mask;

            add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_GENEVE_OPT,
                             &gnv_opts->opts[idx],
                             &gnv_opts_mask->opts_mask[idx], NULL);

            len -= sizeof(struct geneve_opt) + curr_opt_spec.length * 4;
            opt_idx += sizeof(struct geneve_opt) / 4 +
                curr_opt_spec.length - 1;
            idx++;
        }
        memset(&consumed_masks->tunnel.metadata.opts.gnv, 0,
               sizeof consumed_masks->tunnel.metadata.opts.gnv);
    }
    act_vars->gnv_opts_cnt = idx;
}

static int
parse_geneve_match(struct flow_patterns *patterns,
                   struct match *match,
                   struct act_vars *act_vars)
{
    struct rte_flow_item_geneve *gnv_spec, *gnv_mask;
    struct flow *consumed_masks;
    int ret;

    ret = parse_tnl_ip_match(patterns, match, IPPROTO_UDP);
    if (ret) {
        return -1;
    }

    parse_tnl_udp_match(patterns, match);

    consumed_masks = &match->wc.masks;
    /* GENEVE */
    gnv_spec = per_thread_xzalloc(sizeof *gnv_spec);
    gnv_mask = per_thread_xzalloc(sizeof *gnv_mask);

    put_unaligned_be32(ALIGNED_CAST(ovs_be32 *, gnv_spec->vni),
                       htonl(ntohll(match->flow.tunnel.tun_id) << 8));
    put_unaligned_be32(ALIGNED_CAST(ovs_be32 *, gnv_mask->vni),
                       htonl(ntohll(match->wc.masks.tunnel.tun_id) << 8));

    consumed_masks->tunnel.tun_id = 0;
    consumed_masks->tunnel.flags = 0;

    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_GENEVE, gnv_spec, gnv_mask,
                     NULL);
    parse_geneve_opt_match(consumed_masks, patterns, match, act_vars);

    /* tunnel.metadata.present.len value indicates the number of
     * options, it's mask does not indicate any match on the packet,
     * thus masked.
     */
    memset(&consumed_masks->tunnel.metadata.present, 0,
           sizeof consumed_masks->tunnel.metadata.present);

    return 0;
}

static int OVS_UNUSED
parse_flow_tnl_match(struct netdev *tnldev,
                     struct flow_patterns *patterns,
                     odp_port_t orig_in_port,
                     struct match *match,
                     struct act_vars *act_vars)
{
    int ret;

    ret = add_vport_match(patterns, orig_in_port, tnldev);
    if (ret) {
        return ret;
    }

    if (is_all_zeros(&match->wc.masks.tunnel, sizeof match->wc.masks.tunnel)) {
        return 0;
    }

    if (!eth_addr_is_zero(match->wc.masks.tunnel.eth_src) ||
        !eth_addr_is_zero(match->wc.masks.tunnel.eth_dst)) {
        struct rte_flow_item_eth *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        memcpy(&spec->dst, &match->flow.tunnel.eth_dst, sizeof spec->dst);
        memcpy(&spec->src, &match->flow.tunnel.eth_src, sizeof spec->src);

        memcpy(&mask->dst, &match->wc.masks.tunnel.eth_dst, sizeof mask->dst);
        memcpy(&mask->src, &match->wc.masks.tunnel.eth_src, sizeof mask->src);

        memset(&match->wc.masks.tunnel.eth_dst, 0,
               sizeof match->wc.masks.tunnel.eth_dst);
        memset(&match->wc.masks.tunnel.eth_src, 0,
               sizeof match->wc.masks.tunnel.eth_src);

        spec->has_vlan = 0;
        mask->has_vlan = 1;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_ETH, spec, mask, NULL);
    }

    if (act_vars->tun_type == TUN_TYPE_VXLAN ||
        !strcmp(netdev_get_type(tnldev), "vxlan")) {
        act_vars->tun_type = TUN_TYPE_VXLAN;
        ret = parse_vxlan_match(patterns, match);
    }
    else if (act_vars->tun_type == TUN_TYPE_GRE ||
             !strcmp(netdev_get_type(tnldev), "gre") ||
             !strcmp(netdev_get_type(tnldev), "ip6gre")) {
        act_vars->tun_type = TUN_TYPE_GRE;
        ret = parse_gre_match(patterns, match);
    }
    if (act_vars->tun_type == TUN_TYPE_GENEVE ||
        !strcmp(netdev_get_type(tnldev), "geneve")) {
        act_vars->tun_type = TUN_TYPE_GENEVE;
        return parse_geneve_match(patterns, match, act_vars);
    }

    return ret;
}

bool
dpdk_offload_get_reg_field(struct dp_packet *packet,
                           enum dpdk_reg_id reg_id,
                           uint32_t *val)
{
    struct reg_field *reg_field = &offload->reg_fields()[reg_id];
    uint32_t mark = 0;
    uint32_t meta;

    if (reg_field->type == REG_TYPE_META) {
        if (!dp_packet_get_meta(packet, &meta)) {
            return false;
        }

        /* An error should be returned above if meta is 0.
         * We perform another validation and alert on unexpected
         * DPDK behavior.
         */
        if (meta == 0) {
            dp_packet_has_flow_mark(packet, &mark);
            VLOG_ERR_RL(&rl, "port %d, recirc=%d, mark=%d, has meta 0",
                        packet->md.in_port.odp_port, packet->md.recirc_id,
                        mark);
            return false;
        }

        meta >>= reg_field->offset;
        meta &= reg_field->mask;

        *val = meta;
        return true;
    }

    if (reg_field->type == REG_TYPE_MARK) {
        if (dp_packet_has_flow_mark(packet, &mark)) {
            *val = mark;
            return true;
        }
        return false;
    }

    OVS_NOT_REACHED();
}

static int
add_pattern_match_reg_field(struct flow_patterns *patterns,
                            uint8_t reg_field_id, uint32_t val, uint32_t mask)
{
    struct rte_flow_item_meta *meta_spec, *meta_mask;
    struct rte_flow_item_tag *tag_spec, *tag_mask;
    struct reg_field *reg_field;
    uint32_t reg_spec, reg_mask;
    uint8_t reg_index;

    if (reg_field_id >= REG_FIELD_NUM) {
        VLOG_ERR("unkonwn reg id %d", reg_field_id);
        return -1;
    }
    reg_field = &offload->reg_fields()[reg_field_id];
    if (val != (val & reg_field->mask)) {
        VLOG_ERR("value 0x%"PRIx32" is out of range for reg id %d", val,
                 reg_field_id);
        return -1;
    }

    reg_spec = (val & reg_field->mask) << reg_field->offset;
    reg_mask = (mask & reg_field->mask) << reg_field->offset;
    reg_index = reg_field->index;
    ovs_assert(reg_index < REG_TAG_INDEX_NUM);
    switch (reg_field->type) {
    case REG_TYPE_TAG:
        if (patterns->tag_spec[reg_index] == NULL) {
            tag_spec = per_thread_xzalloc(sizeof *tag_spec);
            tag_spec->index = reg_index;
            patterns->tag_spec[reg_index] = tag_spec;

            tag_mask = per_thread_xzalloc(sizeof *tag_mask);
            tag_mask->index = 0xFF;
            patterns->tag_mask[reg_index] = tag_mask;

            add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_TAG,
                             tag_spec, tag_mask, NULL);
        } else {
            tag_spec = patterns->tag_spec[reg_index];
            tag_mask = patterns->tag_mask[reg_index];
        }
        tag_spec->data |= reg_spec;
        tag_mask->data |= reg_mask;
        break;
    case REG_TYPE_META:
        meta_spec = per_thread_xzalloc(sizeof *meta_spec);
        meta_spec->data = reg_spec;

        meta_mask = per_thread_xzalloc(sizeof *meta_mask);
        meta_mask->data = reg_mask;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_META, meta_spec,
                         meta_mask, NULL);
        break;
    case REG_TYPE_MARK:
    default:
        VLOG_ERR("unkonwn reg type (%d) for reg field %d", reg_field->type,
                 reg_field_id);
        return -1;
    }

    return 0;
}

static int
add_action_set_reg_field(struct flow_actions *actions,
                         uint8_t reg_field_id, uint32_t val, uint32_t mask)
{
    struct rte_flow_action_set_meta *set_meta;
    struct rte_flow_action_set_tag *set_tag;
    struct reg_field *reg_field;
    uint32_t reg_spec, reg_mask;

    if (reg_field_id >= REG_FIELD_NUM) {
        VLOG_ERR("unkonwn reg id %d", reg_field_id);
        return -1;
    }
    reg_field = &offload->reg_fields()[reg_field_id];
    if (val != (val & reg_field->mask)) {
        VLOG_ERR_RL(&rl, "value 0x%"PRIx32" is out of range for reg id %d",
                          val, reg_field_id);
        return -1;
    }

    reg_spec = (val & reg_field->mask) << reg_field->offset;
    reg_mask = (mask & reg_field->mask) << reg_field->offset;
    switch (reg_field->type) {
    case REG_TYPE_TAG:
        set_tag = per_thread_xzalloc(sizeof *set_tag);
        set_tag->index = reg_field->index;
        set_tag->data = reg_spec;
        set_tag->mask = reg_mask;
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_SET_TAG, set_tag);
        break;
    case REG_TYPE_META:
        set_meta = per_thread_xzalloc(sizeof *set_meta);
        set_meta->data = reg_spec;
        set_meta->mask = reg_mask;
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_SET_META, set_meta);
        break;
    case REG_TYPE_MARK:
    default:
        VLOG_ERR("unkonwn reg type (%d) for reg field %d", reg_field->type,
                 reg_field_id);
        return -1;
    }

    return 0;
}

static int
parse_tnl_match_recirc(struct flow_patterns *patterns,
                       struct match *match,
                       struct act_resources *act_resources)
{
    if (get_tnl_id(&match->flow.tunnel, &match->wc.masks.tunnel,
                   &act_resources->tnl_id)) {
        return -1;
    }
    if (add_pattern_match_reg_field(patterns, REG_FIELD_TUN_INFO,
                                    act_resources->tnl_id, 0xFFFFFFFF)) {
        return -1;
    }
    memset(&match->wc.masks.tunnel, 0, sizeof match->wc.masks.tunnel);
    return 0;
}

/* DPDK/DOCA allow decap offload only if at least the tunnel type is matched.
 * For flows that does not have any tunnel matches but have tun_decap action,
 * add a tunnel match to allow the offload.
 */
static int
add_tunnel_match(struct netdev *netdev,
                 struct flow_patterns *patterns,
                 struct nlattr *nl_actions,
                 size_t nl_actions_len,
                 bool has_udp,
                 struct act_vars *act_vars)
{
    struct nlattr *nla;
    int rv = 0;
    int left;

    NL_ATTR_FOR_EACH_UNSAFE (nla, left, nl_actions, nl_actions_len) {
        if (nl_attr_type(nla) == OVS_ACTION_ATTR_TUN_DECAP) {
            struct netdev *tundev;

            if (get_netdev_by_port(netdev, nla, NULL, &tundev)) {
                continue;
            }

            if (!strcmp(netdev_get_type(tundev), "vxlan")) {
                act_vars->tun_type = TUN_TYPE_VXLAN;
                if (!has_udp) {
                    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_UDP, NULL,
                                     NULL, NULL);
                }
                add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_VXLAN, NULL,
                                 NULL, NULL);
            } else if (!strcmp(netdev_get_type(tundev), "geneve")) {
                act_vars->tun_type = TUN_TYPE_GENEVE;
                if (!has_udp) {
                    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_UDP, NULL,
                                     NULL, NULL);
                }

                add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_GENEVE, NULL,
                                 NULL, NULL);
            } else if (!strcmp(netdev_get_type(tundev), "gre")) {
                act_vars->tun_type = TUN_TYPE_GRE;
                add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_GRE, NULL, NULL,
                                 NULL);
            } else {
                rv = -1;
            }
            netdev_close(tundev);
            break;
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_TUNNEL_PUSH) {
            return 0;
        }
    }

    return rv;
}

static int
parse_flow_match(struct netdev *netdev,
                 odp_port_t orig_in_port OVS_UNUSED,
                 struct rte_flow_attr *flow_attr,
                 struct flow_patterns *patterns,
                 struct match *match,
                 struct nlattr *nl_actions,
                 size_t actions_len,
                 struct act_resources *act_resources,
                 struct act_vars *act_vars)
{
    struct rte_flow_item_eth *eth_spec = NULL, *eth_mask = NULL;
    struct rte_flow_item_port_id *port_id_spec;
    struct flow *consumed_masks;
    bool has_udp = false;
    uint8_t proto = 0;

    consumed_masks = &match->wc.masks;

    if (!is_nd(&match->flow, NULL)) {
        memset(&match->wc.masks.nd_target, 0,
               sizeof match->wc.masks.nd_target);
        if (!is_arp(&match->flow)) {
            memset(&match->wc.masks.arp_sha, 0,
                   sizeof match->wc.masks.arp_sha);
            memset(&match->wc.masks.arp_tha, 0,
                   sizeof match->wc.masks.arp_tha);
        }
        if (!is_igmp(&match->flow, NULL)) {
            match->wc.masks.igmp_group_ip4 = 0;
        }
    }

    patterns->physdev = netdev;
    act_vars->tun_type = TUN_TYPE_NONE;
    if (match->wc.masks.tunnel.type) {
        act_vars->tun_type = match->flow.tunnel.type;
        consumed_masks->tunnel.type = 0;
    }
#ifdef ALLOW_EXPERIMENTAL_API /* Packet restoration API required. */
    if (act_vars->tun_type != TUN_TYPE_NONE ||
        netdev_vport_is_vport_class(netdev->netdev_class)) {
        if (act_vars->tun_type == TUN_TYPE_NONE) {
            act_vars->vport = match->flow.in_port.odp_port;
        }
        act_vars->tnl_key = &match->flow.tunnel;
        act_vars->tnl_mask = match->wc.masks.tunnel;
        act_vars->is_outer_ipv4 = match->wc.masks.tunnel.ip_src ||
                                  match->wc.masks.tunnel.ip_dst;
        /* In case of a tunnel, pre-ct flow decapsulates the tunnel and sets
         * the tunnel info (matches) in a register. Following tunnel flows
         * (recirc_id>0) don't match the tunnel outer headers, as they are
         * already decapsulated, but on the tunnel info register.
         *
         * CT2CT is applied after a pre-ct flow, so tunnel match should be done
         * on the tunnel info register, as recirc_id>0 flows.
         */
        if ((match->flow.recirc_id ||
             (act_vars->is_e2e_cache &&
              act_resources->flow_id != INVALID_FLOW_MARK)) &&
            parse_tnl_match_recirc(patterns, match, act_resources)) {
            return -1;
        }
        if (parse_flow_tnl_match(netdev, patterns, orig_in_port, match,
                                 act_vars)) {
            return -1;
        }
    } else if (!strcmp(netdev_get_type(netdev), "tap")) {
        act_vars->vport = match->flow.in_port.odp_port;
        patterns->physdev = netdev_ports_get(orig_in_port, netdev->dpif_type);
        if (patterns->physdev == NULL) {
            return -1;
        }
        if (netdev_dpdk_get_esw_mgr_port_id(patterns->physdev) < 0) {
            netdev_close(patterns->physdev);
            return -1;
        }
        netdev_close(patterns->physdev);
    }
#endif

    consumed_masks->tunnel.flags &= ~FLOW_TNL_F_EXPLICIT;

    port_id_spec = per_thread_xzalloc(sizeof *port_id_spec);
    port_id_spec->id = netdev_dpdk_get_port_id(patterns->physdev);
    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_PORT_ID, port_id_spec, NULL,
                     NULL);

    if (get_table_id(act_vars->vport, match->flow.recirc_id,
                     patterns->physdev, act_vars->is_e2e_cache,
                     &act_resources->self_table_id)) {
        return -1;
    }
    act_vars->recirc_id = match->flow.recirc_id;

    memset(&consumed_masks->in_port, 0, sizeof consumed_masks->in_port);
    consumed_masks->recirc_id = 0;
    consumed_masks->packet_type = 0;

    if (act_vars->recirc_id && conntrack_offload_size() == 0 &&
        add_pattern_match_reg_field(patterns, REG_FIELD_RECIRC,
                                    act_vars->recirc_id,
                                    offload->reg_fields()[REG_FIELD_RECIRC].mask)) {
        return -1;
    }

    /* Eth */
    if (act_vars->is_ct_conn) {
        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_ETH, NULL, NULL, NULL);
    } else if (match->wc.masks.dl_type ||
        !eth_addr_is_zero(match->wc.masks.dl_src) ||
        !eth_addr_is_zero(match->wc.masks.dl_dst)) {
        struct rte_flow_item_eth *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        memcpy(&spec->dst, &match->flow.dl_dst, sizeof spec->dst);
        memcpy(&spec->src, &match->flow.dl_src, sizeof spec->src);
        spec->type = match->flow.dl_type;

        memcpy(&mask->dst, &match->wc.masks.dl_dst, sizeof mask->dst);
        memcpy(&mask->src, &match->wc.masks.dl_src, sizeof mask->src);
        mask->type = match->wc.masks.dl_type;

        memset(&consumed_masks->dl_dst, 0, sizeof consumed_masks->dl_dst);
        memset(&consumed_masks->dl_src, 0, sizeof consumed_masks->dl_src);
        consumed_masks->dl_type = 0;

        spec->has_vlan = 0;
        mask->has_vlan = 1;
        eth_spec = spec;
        eth_mask = mask;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_ETH, spec, mask, NULL);
        /* Eth traffic that is not further classified to IPv4/6 gets low
         * priority, To improve performance of IP traffic.
         */
        flow_attr->priority = DPDK_OFFLOAD_PRIORITY_LOW;
    }

    /* VLAN */
    if (match->wc.masks.vlans[0].tci && match->flow.vlans[0].tci) {
        struct rte_flow_item_vlan *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->tci = match->flow.vlans[0].tci & ~htons(VLAN_CFI);
        mask->tci = match->wc.masks.vlans[0].tci & ~htons(VLAN_CFI);

        if (eth_spec && eth_mask) {
            eth_spec->has_vlan = 1;
            eth_mask->has_vlan = 1;
            spec->inner_type = eth_spec->type;
            mask->inner_type = eth_mask->type;
            eth_spec->type = match->flow.vlans[0].tpid;
            eth_mask->type = match->wc.masks.vlans[0].tpid;
        }

        act_vars->vlan_tpid = match->flow.vlans[0].tpid;
        act_vars->vlan_pcp = vlan_tci_to_pcp(match->flow.vlans[0].tci);
        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_VLAN, spec, mask, NULL);
    }
    /* For untagged matching match->wc.masks.vlans[0].tci is 0xFFFF and
     * match->flow.vlans[0].tci is 0. Consuming is needed outside of the if
     * scope to handle that.
     */
    memset(&consumed_masks->vlans[0], 0, sizeof consumed_masks->vlans[0]);

    /* IP v4 */
    if (match->flow.dl_type == htons(ETH_TYPE_IP)) {
        struct rte_flow_item_ipv4 *spec, *mask, *last = NULL;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.type_of_service = match->flow.nw_tos;
        spec->hdr.time_to_live    = match->flow.nw_ttl;
        spec->hdr.next_proto_id   = match->flow.nw_proto;
        spec->hdr.src_addr        = match->flow.nw_src;
        spec->hdr.dst_addr        = match->flow.nw_dst;

        mask->hdr.type_of_service = match->wc.masks.nw_tos;
        mask->hdr.time_to_live    = match->wc.masks.nw_ttl;
        mask->hdr.next_proto_id   = match->wc.masks.nw_proto;
        mask->hdr.src_addr        = match->wc.masks.nw_src;
        mask->hdr.dst_addr        = match->wc.masks.nw_dst;

        consumed_masks->nw_tos = 0;
        consumed_masks->nw_ttl = 0;
        consumed_masks->nw_proto = 0;
        consumed_masks->nw_src = 0;
        consumed_masks->nw_dst = 0;

        if (match->wc.masks.nw_frag & FLOW_NW_FRAG_ANY) {
            if (!(match->flow.nw_frag & FLOW_NW_FRAG_ANY)) {
                /* frag=no. */
                spec->hdr.fragment_offset = 0;
                mask->hdr.fragment_offset = htons(RTE_IPV4_HDR_OFFSET_MASK
                                                  | RTE_IPV4_HDR_MF_FLAG);
            } else if (match->wc.masks.nw_frag & FLOW_NW_FRAG_LATER) {
                if (!(match->flow.nw_frag & FLOW_NW_FRAG_LATER)) {
                    /* frag=first. */
                    spec->hdr.fragment_offset = htons(RTE_IPV4_HDR_MF_FLAG);
                    mask->hdr.fragment_offset = htons(RTE_IPV4_HDR_OFFSET_MASK
                                                      | RTE_IPV4_HDR_MF_FLAG);
                } else {
                    /* frag=later. */
                    last = per_thread_xzalloc(sizeof *last);
                    spec->hdr.fragment_offset =
                        htons(1 << RTE_IPV4_HDR_FO_SHIFT);
                    mask->hdr.fragment_offset =
                        htons(RTE_IPV4_HDR_OFFSET_MASK);
                    last->hdr.fragment_offset =
                        htons(RTE_IPV4_HDR_OFFSET_MASK);
                }
            } else {
                VLOG_WARN_RL(&rl, "Unknown IPv4 frag (0x%x/0x%x)",
                             match->flow.nw_frag, match->wc.masks.nw_frag);
                return -1;
            }
            consumed_masks->nw_frag = 0;
        }

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_IPV4, spec, mask, last);

        /* Save proto for L4 protocol setup. */
        proto = spec->hdr.next_proto_id &
                mask->hdr.next_proto_id;
        /* IPv4 gets the highest priority, to improve its performance. */
        flow_attr->priority = DPDK_OFFLOAD_PRIORITY_HIGH;
    }

    /* IP v6 */
    if (match->flow.dl_type == htons(ETH_TYPE_IPV6)) {
        struct rte_flow_item_ipv6 *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.proto = match->flow.nw_proto;
        spec->hdr.hop_limits = match->flow.nw_ttl;
        spec->hdr.vtc_flow =
            htonl((uint32_t) match->flow.nw_tos << RTE_IPV6_HDR_TC_SHIFT);
        memcpy(spec->hdr.src_addr, &match->flow.ipv6_src,
               sizeof spec->hdr.src_addr);
        memcpy(spec->hdr.dst_addr, &match->flow.ipv6_dst,
               sizeof spec->hdr.dst_addr);
        if ((match->wc.masks.nw_frag & FLOW_NW_FRAG_ANY)
            && (match->flow.nw_frag & FLOW_NW_FRAG_ANY)) {
            spec->has_frag_ext = 1;
        }

        mask->hdr.proto = match->wc.masks.nw_proto;
        mask->hdr.hop_limits = match->wc.masks.nw_ttl;
        mask->hdr.vtc_flow =
            htonl((uint32_t) match->wc.masks.nw_tos << RTE_IPV6_HDR_TC_SHIFT);
        memcpy(mask->hdr.src_addr, &match->wc.masks.ipv6_src,
               sizeof mask->hdr.src_addr);
        memcpy(mask->hdr.dst_addr, &match->wc.masks.ipv6_dst,
               sizeof mask->hdr.dst_addr);

        consumed_masks->nw_ttl = 0;
        consumed_masks->nw_tos = 0;
        memset(&consumed_masks->ipv6_src, 0, sizeof consumed_masks->ipv6_src);
        memset(&consumed_masks->ipv6_dst, 0, sizeof consumed_masks->ipv6_dst);

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_IPV6, spec, mask, NULL);

        /* Save proto for L4 protocol setup. */
        proto = spec->hdr.proto & mask->hdr.proto;

        if (spec->has_frag_ext) {
            struct rte_flow_item_ipv6_frag_ext *frag_spec, *frag_mask,
                *frag_last = NULL;

            frag_spec = per_thread_xzalloc(sizeof *frag_spec);
            frag_mask = per_thread_xzalloc(sizeof *frag_mask);

            if (match->wc.masks.nw_frag & FLOW_NW_FRAG_LATER) {
                if (!(match->flow.nw_frag & FLOW_NW_FRAG_LATER)) {
                    /* frag=first. */
                    frag_spec->hdr.frag_data = htons(RTE_IPV6_EHDR_MF_MASK);
                    frag_mask->hdr.frag_data = htons(RTE_IPV6_EHDR_MF_MASK |
                                                     RTE_IPV6_EHDR_FO_MASK);
                    /* Move the proto match to the extension item. */
                    frag_spec->hdr.next_header = match->flow.nw_proto;
                    frag_mask->hdr.next_header = match->wc.masks.nw_proto;
                    spec->hdr.proto = 0;
                    mask->hdr.proto = 0;
                } else {
                    /* frag=later. */
                    frag_last = per_thread_xzalloc(sizeof *frag_last);
                    frag_spec->hdr.frag_data =
                        htons(1 << RTE_IPV6_EHDR_FO_SHIFT);
                    frag_mask->hdr.frag_data = htons(RTE_IPV6_EHDR_FO_MASK);
                    frag_last->hdr.frag_data = htons(RTE_IPV6_EHDR_FO_MASK);
                    /* There can't be a proto for later frags. */
                    spec->hdr.proto = 0;
                    mask->hdr.proto = 0;
                }
            } else {
                VLOG_WARN_RL(&rl, "Unknown IPv6 frag (0x%x/0x%x)",
                             match->flow.nw_frag, match->wc.masks.nw_frag);
                return -1;
            }

            add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_IPV6_FRAG_EXT,
                             frag_spec, frag_mask, frag_last);
        }
        if (match->wc.masks.nw_frag) {
            /* frag=no is indicated by spec->has_frag_ext=0. */
            mask->has_frag_ext = 1;
            consumed_masks->nw_frag = 0;
        }
        consumed_masks->nw_proto = 0;
        /* IPv6 gets the 2nd highest priority (highest is IPv4), to improve
         * its performance over non-IP traffic.
         */
        flow_attr->priority = DPDK_OFFLOAD_PRIORITY_MED;
    }

    if (!act_vars->is_ct_conn &&
        proto != IPPROTO_ICMP && proto != IPPROTO_UDP  &&
        proto != IPPROTO_SCTP && proto != IPPROTO_TCP  &&
        proto != IPPROTO_ICMPV6 &&
        (match->wc.masks.tp_src ||
         match->wc.masks.tp_dst ||
         match->wc.masks.tcp_flags)) {
        VLOG_DBG("L4 Protocol (%u) not supported", proto);
        return -1;
    }

    act_vars->proto = proto;

    if (proto == IPPROTO_TCP) {
        struct rte_flow_item_tcp *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.src_port  = match->flow.tp_src;
        spec->hdr.dst_port  = match->flow.tp_dst;
        spec->hdr.data_off  = ntohs(match->flow.tcp_flags) >> 8;
        spec->hdr.tcp_flags = ntohs(match->flow.tcp_flags) & 0xff;

        mask->hdr.src_port  = match->wc.masks.tp_src;
        mask->hdr.dst_port  = match->wc.masks.tp_dst;
        mask->hdr.data_off  = ntohs(match->wc.masks.tcp_flags) >> 8;
        mask->hdr.tcp_flags = ntohs(match->wc.masks.tcp_flags) & 0xff;

        consumed_masks->tp_src = 0;
        consumed_masks->tp_dst = 0;
        consumed_masks->tcp_flags = 0;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_TCP, spec, mask, NULL);
    } else if (proto == IPPROTO_UDP) {
        struct rte_flow_item_udp *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.src_port = match->flow.tp_src;
        spec->hdr.dst_port = match->flow.tp_dst;

        mask->hdr.src_port = match->wc.masks.tp_src;
        mask->hdr.dst_port = match->wc.masks.tp_dst;

        consumed_masks->tp_src = 0;
        consumed_masks->tp_dst = 0;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_UDP, spec, mask, NULL);
        has_udp = true;
    } else if (proto == IPPROTO_SCTP) {
        struct rte_flow_item_sctp *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.src_port = match->flow.tp_src;
        spec->hdr.dst_port = match->flow.tp_dst;

        mask->hdr.src_port = match->wc.masks.tp_src;
        mask->hdr.dst_port = match->wc.masks.tp_dst;

        consumed_masks->tp_src = 0;
        consumed_masks->tp_dst = 0;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_SCTP, spec, mask, NULL);
    } else if (proto == IPPROTO_ICMP) {
        struct rte_flow_item_icmp *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.icmp_type = (uint8_t) ntohs(match->flow.tp_src);
        spec->hdr.icmp_code = (uint8_t) ntohs(match->flow.tp_dst);

        mask->hdr.icmp_type = (uint8_t) ntohs(match->wc.masks.tp_src);
        mask->hdr.icmp_code = (uint8_t) ntohs(match->wc.masks.tp_dst);

        consumed_masks->tp_src = 0;
        consumed_masks->tp_dst = 0;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_ICMP, spec, mask, NULL);
    } else if (proto == IPPROTO_ICMPV6) {
        struct rte_flow_item_icmp6 *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->type = (uint8_t) ntohs(match->flow.tp_src);
        spec->code = (uint8_t) ntohs(match->flow.tp_dst);

        mask->type = (uint8_t) ntohs(match->wc.masks.tp_src);
        mask->code = (uint8_t) ntohs(match->wc.masks.tp_dst);

        consumed_masks->tp_src = 0;
        consumed_masks->tp_dst = 0;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_ICMP6, spec, mask, NULL);
    }

    if (act_vars->tun_type == TUN_TYPE_NONE &&
        add_tunnel_match(netdev, patterns, nl_actions, actions_len, has_udp,
                         act_vars)) {
        return -1;
    }

    /* ct-state */
    if (match->wc.masks.ct_state &&
        !((match->wc.masks.ct_state & CS_NEW) &&
          (match->flow.ct_state & CS_NEW)) &&
        !(match->wc.masks.ct_state & OVS_CS_F_NAT_MASK)) {

        if (act_vars->proto && act_vars->proto != IPPROTO_UDP &&
            act_vars->proto != IPPROTO_TCP) {
            VLOG_DBG_RL(&rl, "Unsupported CT offload for L4 protocol: 0x%02"
                        PRIx8, act_vars->proto);
            return -1;
        }

        if ((!match->flow.recirc_id &&
             !(match->wc.masks.ct_state & match->flow.ct_state)) ||
            !add_pattern_match_reg_field(patterns, REG_FIELD_CT_STATE,
                                         match->flow.ct_state,
                                         match->wc.masks.ct_state)) {
            consumed_masks->ct_state = 0;
        }
    }
    /* ct-zone */
    if (match->wc.masks.ct_zone &&
        (!get_zone_id(match->flow.ct_zone,
                      &act_resources->ct_match_zone_id) &&
         !add_pattern_match_reg_field(patterns,
                                      REG_FIELD_CT_ZONE,
                                      act_resources->ct_match_zone_id,
                                      offload->reg_fields()[REG_FIELD_CT_ZONE].mask))) {
        consumed_masks->ct_zone = 0;
    }
    /* ct-mark */
    if (match->wc.masks.ct_mark) {
        if ((!match->flow.recirc_id &&
             !(match->flow.ct_mark & match->wc.masks.ct_mark)) ||
            !add_pattern_match_reg_field(patterns, REG_FIELD_CT_MARK,
                                         match->flow.ct_mark,
                                         match->wc.masks.ct_mark)) {
            consumed_masks->ct_mark = 0;
        }
    }
    /* ct-label */
    if (!act_vars->is_ct_conn &&
        !is_all_zeros(&match->wc.masks.ct_label,
                      sizeof match->wc.masks.ct_label)) {
        uint32_t value, mask;

        if (netdev_offload_dpdk_ct_labels_mapping) {
            ovs_u128 tmp_u128;

            tmp_u128.u64.lo = match->flow.ct_label.u64.lo &
                              match->wc.masks.ct_label.u64.lo;
            tmp_u128.u64.hi = match->flow.ct_label.u64.hi &
                              match->wc.masks.ct_label.u64.hi;
            if (get_label_id(&tmp_u128, &act_resources->ct_match_label_id)) {
                return -1;
            }
            value = act_resources->ct_match_label_id;
            mask = offload->reg_fields()[REG_FIELD_CT_LABEL_ID].mask;
        } else {
            if (match->wc.masks.ct_label.u32[1] ||
                match->wc.masks.ct_label.u32[2] ||
                match->wc.masks.ct_label.u32[3]) {
                return -1;
            }

            value = match->flow.ct_label.u32[0];
            mask = match->wc.masks.ct_label.u32[0];
        }

        if (!add_pattern_match_reg_field(patterns, REG_FIELD_CT_LABEL_ID,
                                         value, mask)) {
            memset(&consumed_masks->ct_label, 0,
                   sizeof consumed_masks->ct_label);
        }
    }

    if (match->wc.masks.dp_hash) {
        struct rte_flow_item_mark *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);
        spec->id = match->flow.dp_hash;
        mask->id = match->wc.masks.dp_hash;
        add_flow_pattern(patterns, OVS_RTE_FLOW_ITEM_TYPE(HASH), spec, mask,
                         NULL);
        match->wc.masks.dp_hash = 0;
    }

    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_END, NULL, NULL, NULL);

    /* A CT conn offload is assured to be fully matched.
     * Verify full match only for other offloads. */
    if (!act_vars->is_ct_conn &&
        !is_all_zeros(consumed_masks, sizeof *consumed_masks)) {
        return -1;
    }
    return 0;
}

static void
add_empty_sample_action(int ratio,
                        struct flow_actions *actions)
{
    struct sample_data {
        struct rte_flow_action_sample sample;
        struct rte_flow_action end_action;
    } *sample_data;
    BUILD_ASSERT_DECL(offsetof(struct sample_data, sample) == 0);

    sample_data = xzalloc(sizeof *sample_data);
    sample_data->end_action.type = RTE_FLOW_ACTION_TYPE_END;
    sample_data->sample.actions = &sample_data->end_action;
    sample_data->sample.ratio = ratio;

    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_SAMPLE,
                    sample_data);
}

static int
map_sflow_attr(struct flow_actions *actions,
               const struct nlattr *nl_actions,
               struct dpif_sflow_attr *sflow_attr,
               struct act_resources *act_resources,
               struct act_vars *act_vars)
{
    struct sflow_ctx sflow_ctx;
    const struct nlattr *nla;
    unsigned int left;

    NL_NESTED_FOR_EACH_UNSAFE (nla, left, nl_actions) {
        if (nl_attr_type(nla) == OVS_USERSPACE_ATTR_USERDATA) {
            const struct user_action_cookie *cookie;

            cookie = nl_attr_get(nla);
            if (cookie->type == USER_ACTION_COOKIE_SFLOW) {
                sflow_attr->userdata_len = nl_attr_get_size(nla);
                memset(&sflow_ctx, 0, sizeof sflow_ctx);
                sflow_ctx.sflow_attr = *sflow_attr;
                sflow_ctx.cookie = *cookie;
                if (act_vars->tun_type != TUN_TYPE_NONE) {
                    memcpy(&sflow_ctx.sflow_tnl, act_vars->tnl_key,
                           sizeof sflow_ctx.sflow_tnl);
                }
                if (!get_sflow_id(&sflow_ctx, &act_resources->sflow_id) &&
                    !add_action_set_reg_field(actions, REG_FIELD_SFLOW_CTX,
                                              act_resources->sflow_id,
                                              UINT32_MAX)) {
                    return 0;
                }
            }
        }
    }

    VLOG_DBG_RL(&rl, "no sFlow cookie");
    return -1;
}

static int
parse_userspace_action(struct flow_actions *actions,
                       const struct nlattr *nl_actions,
                       struct dpif_sflow_attr *sflow_attr,
                       struct act_resources *act_resources,
                       struct act_vars *act_vars)
{
    const struct nlattr *nla;
    unsigned int left;

    NL_NESTED_FOR_EACH_UNSAFE (nla, left, nl_actions) {
        if (nl_attr_type(nla) == OVS_ACTION_ATTR_USERSPACE) {
            return map_sflow_attr(actions, nla, sflow_attr,
                                  act_resources, act_vars);
        }
    }

    VLOG_DBG_RL(&rl, "no OVS_ACTION_ATTR_USERSPACE attribute");
    return -1;
}

static int
parse_sample_action(struct flow_actions *actions,
                    const struct nlattr *nl_actions,
                    struct dpif_sflow_attr *sflow_attr,
                    struct act_resources *act_resources,
                    struct act_vars *act_vars)
{
    const struct nlattr *nla;
    unsigned int left;
    int ratio = 0;

    sflow_attr->sflow = nl_actions;
    sflow_attr->sflow_len = nl_actions->nla_len;

    NL_NESTED_FOR_EACH_UNSAFE (nla, left, nl_actions) {
        if (nl_attr_type(nla) == OVS_SAMPLE_ATTR_ACTIONS) {
            if (parse_userspace_action(actions, nla,
                                       sflow_attr, act_resources, act_vars)) {
                return -1;
            }
        } else if (nl_attr_type(nla) == OVS_SAMPLE_ATTR_PROBABILITY) {
            ratio = UINT32_MAX / nl_attr_get_u32(nla);
        } else {
            return -1;
        }
    }

    add_empty_sample_action(ratio, actions);
    return 0;
}

static int
add_count_action(struct netdev *netdev,
                 struct flow_actions *actions,
                 struct act_resources *act_resources,
                 struct act_vars *act_vars)
{
    struct rte_flow_action_count *count = per_thread_xzalloc(sizeof *count);
    struct indirect_ctx *ctx;

    /* e2e flows don't use mark. ct2ct do. we can share only e2e, not ct2ct. */
    if (act_vars->is_e2e_cache &&
        act_resources->flow_id == INVALID_FLOW_MARK &&
        !netdev_is_flow_counter_key_zero(&act_vars->flows_counter_key)) {
        ctx = get_indirect_count_ctx(netdev, &act_vars->flows_counter_key,
                                     OVS_SHARED_COUNT, true);
        if (!ctx) {
            return -1;
        }
        act_resources->shared_count_ctx = ctx;
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_INDIRECT, ctx->act_hdl);
        actions->shared_count_action_pos = actions->cnt - 1;
    } else if (act_vars->is_ct_conn) {
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_INDIRECT, NULL);
        actions->shared_count_action_pos = actions->cnt - 1;
    } else {
        /* For normal flows, add a standard count action. */
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_COUNT, count);
    }

    /* e2e flows don't use mark. ct2ct do. we can share only e2e, not ct2ct. */
    if (act_vars->is_e2e_cache && act_vars->ct_counter_key &&
        act_resources->flow_id == INVALID_FLOW_MARK) {
        ctx = get_indirect_age_ctx(netdev, act_vars->ct_counter_key, true);
        if (!ctx) {
            return -1;
        }
        act_resources->shared_age_ctx = ctx;
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_INDIRECT, ctx->act_hdl);
        actions->shared_age_action_pos = actions->cnt - 1;
    }

    return 0;
}

static void
add_port_id_action(struct flow_actions *actions,
                   int outdev_id)
{
    struct rte_flow_action_port_id *port_id;

    port_id = per_thread_xzalloc(sizeof *port_id);
    port_id->id = outdev_id;
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_PORT_ID, port_id);
}

static void
add_hairpin_action(struct flow_actions *actions)
{
    struct rte_flow_action_mark *mark = per_thread_xzalloc(sizeof *mark);
    struct rte_flow_action_jump *jump = per_thread_xzalloc (sizeof *jump);

    mark->id = HAIRPIN_FLOW_MARK;
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_MARK, mark);

    jump->group = MISS_TABLE_ID;
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_JUMP, jump);
}

static int
get_netdev_by_port(struct netdev *netdev,
                   const struct nlattr *nla,
                   int *outdev_id,
                   struct netdev **outdev)
{
    odp_port_t port;

    port = nl_attr_get_odp_port(nla);
    *outdev = netdev_ports_get(port, netdev->dpif_type);
    if (!*outdev) {
        VLOG_DBG_RL(&rl, "Cannot find netdev for odp port %"PRIu32, port);
        return -1;
    }
    if (!netdev_flow_api_equals(netdev, *outdev)) {
        goto err;
    }
    if (!outdev_id) {
        return 0;
    }
    *outdev_id = netdev_dpdk_get_port_id(*outdev);
    if (*outdev_id < 0) {
        goto err;
    }
    return 0;
err:
    VLOG_DBG_RL(&rl, "%s: Output to port \'%s\' cannot be offloaded.",
                netdev_get_name(netdev), netdev_get_name(*outdev));
    netdev_close(*outdev);
    return -1;
}

static int
add_output_action(struct netdev *netdev,
                  struct flow_actions *actions,
                  const struct nlattr *nla)
{
    struct netdev *outdev;
    int outdev_id;
    int ret = 0;

    if (get_netdev_by_port(netdev, nla, &outdev_id, &outdev)) {
        return -1;
    }
    if (netdev == outdev) {
        add_hairpin_action(actions);
    } else {
        add_port_id_action(actions, outdev_id);
    }

    netdev_close(outdev);
    return ret;
}

static int
add_set_flow_action__(struct flow_actions *actions,
                      const void *value, void *mask,
                      const size_t size, enum rte_flow_action_type type)
{
    struct action_set_data *asd;

    if (mask && is_all_zeros(mask, size)) {
        return 0;
    }

    asd = per_thread_xzalloc(sizeof *asd);
    memcpy(asd->value, value, size);
    if (mask) {
        memcpy(asd->mask, mask, size);
    } else {
        memset(asd->mask, 0xFF, sizeof asd->mask);
    }
    asd->size = size;
    add_flow_action(actions, type, asd);

    /* Clear used mask for later checking. */
    if (mask) {
        memset(mask, 0, size);
    }
    return 0;
}

static void
add_full_set_action(struct flow_actions *actions,
                    enum rte_flow_action_type type,
                    const void *value, size_t size)
{
    struct action_set_data *asd;

    asd = per_thread_xzalloc(sizeof *asd);
    memcpy(asd->value, value, size);
    memset(asd->mask, 0xFF, size);
    asd->size = size;
    add_flow_action(actions, type, asd);
}

BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_mac) ==
                  MEMBER_SIZEOF(struct ovs_key_ethernet, eth_src));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_mac) ==
                  MEMBER_SIZEOF(struct ovs_key_ethernet, eth_dst));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_ipv4) ==
                  MEMBER_SIZEOF(struct ovs_key_ipv4, ipv4_src));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_ipv4) ==
                  MEMBER_SIZEOF(struct ovs_key_ipv4, ipv4_dst));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_ttl) ==
                  MEMBER_SIZEOF(struct ovs_key_ipv4, ipv4_ttl));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_ipv6) ==
                  MEMBER_SIZEOF(struct ovs_key_ipv6, ipv6_src));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_ipv6) ==
                  MEMBER_SIZEOF(struct ovs_key_ipv6, ipv6_dst));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_ttl) ==
                  MEMBER_SIZEOF(struct ovs_key_ipv6, ipv6_hlimit));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_tp) ==
                  MEMBER_SIZEOF(struct ovs_key_tcp, tcp_src));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_tp) ==
                  MEMBER_SIZEOF(struct ovs_key_tcp, tcp_dst));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_tp) ==
                  MEMBER_SIZEOF(struct ovs_key_udp, udp_src));
BUILD_ASSERT_DECL(sizeof(struct rte_flow_action_set_tp) ==
                  MEMBER_SIZEOF(struct ovs_key_udp, udp_dst));

static int
parse_set_actions(struct flow_actions *actions,
                  const struct nlattr *set_actions,
                  const size_t set_actions_len,
                  bool masked, uint8_t proto)
{
    const struct nlattr *sa;
    unsigned int sleft;

#define add_set_flow_action(field, type)                                      \
    if (add_set_flow_action__(actions, &key->field,                           \
                              mask ? CONST_CAST(void *, &mask->field) : NULL, \
                              sizeof key->field, type)) {                     \
        return -1;                                                            \
    }

    NL_ATTR_FOR_EACH_UNSAFE (sa, sleft, set_actions, set_actions_len) {
        if (nl_attr_type(sa) == OVS_KEY_ATTR_ETHERNET) {
            const struct ovs_key_ethernet *key = nl_attr_get(sa);
            const struct ovs_key_ethernet *mask = masked ? key + 1 : NULL;

            add_set_flow_action(eth_src, RTE_FLOW_ACTION_TYPE_SET_MAC_SRC);
            add_set_flow_action(eth_dst, RTE_FLOW_ACTION_TYPE_SET_MAC_DST);

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported ETHERNET set action");
                return -1;
            }
        } else if (nl_attr_type(sa) == OVS_KEY_ATTR_IPV4) {
            const struct ovs_key_ipv4 *key = nl_attr_get(sa);
            const struct ovs_key_ipv4 *mask = masked ? key + 1 : NULL;

            add_set_flow_action(ipv4_src, RTE_FLOW_ACTION_TYPE_SET_IPV4_SRC);
            add_set_flow_action(ipv4_dst, RTE_FLOW_ACTION_TYPE_SET_IPV4_DST);
            add_set_flow_action(ipv4_ttl, RTE_FLOW_ACTION_TYPE_SET_IPV4_TTL);
            add_set_flow_action(ipv4_tos, RTE_FLOW_ACTION_TYPE_SET_IPV4_DSCP);

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported IPv4 set action");
                return -1;
            }
        } else if (nl_attr_type(sa) == OVS_KEY_ATTR_IPV6) {
            const struct ovs_key_ipv6 *key = nl_attr_get(sa);
            const struct ovs_key_ipv6 *mask = masked ? key + 1 : NULL;
            bool modify_ip_header;

            modify_ip_header = !mask ||
                ipv6_addr_is_set(&mask->ipv6_src) ||
                ipv6_addr_is_set(&mask->ipv6_dst);

            add_set_flow_action(ipv6_src, RTE_FLOW_ACTION_TYPE_SET_IPV6_SRC);
            add_set_flow_action(ipv6_dst, RTE_FLOW_ACTION_TYPE_SET_IPV6_DST);
            add_set_flow_action(ipv6_hlimit, RTE_FLOW_ACTION_TYPE_SET_IPV6_HOP);
            add_set_flow_action(ipv6_tclass, RTE_FLOW_ACTION_TYPE_SET_IPV6_DSCP);

            if ((mask && !is_all_zeros(mask, sizeof *mask)) ||
                (proto != IPPROTO_TCP &&
                 proto != IPPROTO_UDP &&
                 proto != IPPROTO_ICMP &&
                 modify_ip_header)) {
                VLOG_DBG_RL(&rl, "Unsupported IPv6 set action");
                return -1;
            }
        } else if (nl_attr_type(sa) == OVS_KEY_ATTR_TCP) {
            const struct ovs_key_tcp *key = nl_attr_get(sa);
            const struct ovs_key_tcp *mask = masked ? key + 1 : NULL;

            add_set_flow_action(tcp_src, OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_SRC));
            add_set_flow_action(tcp_dst, OVS_RTE_FLOW_ACTION_TYPE(SET_TCP_DST));

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported TCP set action");
                return -1;
            }
        } else if (nl_attr_type(sa) == OVS_KEY_ATTR_UDP) {
            const struct ovs_key_udp *key = nl_attr_get(sa);
            const struct ovs_key_udp *mask = masked ? key + 1 : NULL;

            add_set_flow_action(udp_src, OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_SRC));
            add_set_flow_action(udp_dst, OVS_RTE_FLOW_ACTION_TYPE(SET_UDP_DST));

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported UDP set action");
                return -1;
            }
        } else {
            VLOG_DBG_RL(&rl,
                        "Unsupported set action type %d", nl_attr_type(sa));
            return -1;
        }
    }

    return 0;
}

static int
parse_vlan_push_action(struct flow_actions *actions,
                       const struct ovs_action_push_vlan *vlan_push,
                       struct act_vars *act_vars)
{
    struct rte_flow_action_of_push_vlan *rte_push_vlan;
    struct rte_flow_action_of_set_vlan_pcp *rte_vlan_pcp;
    struct rte_flow_action_of_set_vlan_vid *rte_vlan_vid;
    struct rte_flow_action *last_action = NULL;

    if (actions->cnt > 0) {
        last_action = &actions->actions[actions->cnt - 1];
    }
    if (last_action && last_action->type == RTE_FLOW_ACTION_TYPE_OF_POP_VLAN &&
        act_vars->vlan_tpid == vlan_push->vlan_tpid &&
        act_vars->vlan_pcp == vlan_tci_to_pcp(vlan_push->vlan_tci)) {
        actions->cnt--;
    } else {
        rte_push_vlan = per_thread_xzalloc(sizeof *rte_push_vlan);
        rte_push_vlan->ethertype = vlan_push->vlan_tpid;
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_OF_PUSH_VLAN, rte_push_vlan);

        rte_vlan_pcp = per_thread_xzalloc(sizeof *rte_vlan_pcp);
        rte_vlan_pcp->vlan_pcp = vlan_tci_to_pcp(vlan_push->vlan_tci);
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_OF_SET_VLAN_PCP,
                        rte_vlan_pcp);
    }

    rte_vlan_vid = per_thread_xzalloc(sizeof *rte_vlan_vid);
    rte_vlan_vid->vlan_vid = htons(vlan_tci_to_vid(vlan_push->vlan_tci));
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_OF_SET_VLAN_VID,
                    rte_vlan_vid);
    return 0;
}

static int
add_meter_action(struct flow_actions *actions,
                 const struct nlattr *nla,
                 struct act_resources *act_resources)
{
    /* Compensate for ovs-ofctl (meter_ID - 1) adjustment */
    uint32_t mtr_id = nl_attr_get_u32(nla) + 1;
    struct meter_data *mtr_data;
    int mtr_cnt = 0;

    while (mtr_cnt < DPDK_OFFLOAD_MAX_METERS_PER_FLOW &&
           act_resources->meter_ids[mtr_cnt] != 0) {
        mtr_cnt++;
    }

    /* DPDK supports only single meter per flow. */
    if (mtr_cnt > 0 && !ovs_doca_enabled()) {
        return -1;
    }
    if (mtr_cnt >= DPDK_OFFLOAD_MAX_METERS_PER_FLOW) {
        VLOG_ERR("Failed to add meter action, max supported is %u",
                 DPDK_OFFLOAD_MAX_METERS_PER_FLOW);
        return -1;
    }

    if (!netdev_dpdk_meter_ref(mtr_id)) {
        return -1;
    }
    act_resources->meter_ids[mtr_cnt] = mtr_id;

    mtr_data = per_thread_xzalloc(sizeof *mtr_data);
    mtr_data->conf.mtr_id = mtr_id;
    mtr_data->flow_id = act_resources->flow_id;
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_METER, mtr_data);

    return 0;
}

static void
add_jump_action(struct flow_actions *actions, uint32_t group)
{
    struct rte_flow_action_jump *jump = per_thread_xzalloc (sizeof *jump);

    jump->group = group;
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_JUMP, jump);
}

static int
add_miss_flow(struct netdev *netdev,
              uint32_t src_table_id,
              uint32_t dst_table_id,
              uint32_t recirc_id,
              uint32_t mark_id,
              struct dpdk_offload_handle *doh)
{
    struct rte_flow_attr miss_attr = {
        .transfer = 1,
        .priority = DPDK_OFFLOAD_PRIORITY_MISS,
    };
    struct rte_flow_item_port_id port_id;
    struct rte_flow_item_tag tag_spec;
    struct rte_flow_item_tag tag_mask;
    struct flow_patterns miss_patterns = {
        .items = (struct rte_flow_item []) {
            { .type = RTE_FLOW_ITEM_TYPE_TAG, .spec = &tag_spec,
              .mask = &tag_mask },
            { .type = RTE_FLOW_ITEM_TYPE_PORT_ID, .spec = &port_id, },
            { .type = RTE_FLOW_ITEM_TYPE_ETH, },
            { .type = RTE_FLOW_ITEM_TYPE_END, },
        },
        .cnt = 4,
    };
    struct rte_flow_action_jump miss_jump = { .group = dst_table_id, };
    struct rte_flow_action_mark miss_mark;
    struct flow_actions miss_actions = {
        .actions = (struct rte_flow_action []) {
            { .type = OVS_RTE_FLOW_ACTION_TYPE(FLOW_INFO), .conf = &miss_mark },
            { .type = RTE_FLOW_ACTION_TYPE_JUMP, .conf = &miss_jump },
            { .type = RTE_FLOW_ACTION_TYPE_END, },
        },
        .cnt = 3,
    };
    struct rte_flow_error error;
    struct reg_field *reg_field;

    miss_attr.group = src_table_id;
    miss_mark.id = mark_id;
    if (mark_id == INVALID_FLOW_MARK) {
        miss_actions.actions++;
        miss_actions.cnt--;
    }

    if (recirc_id == 0 || conntrack_offload_size() > 0) {
        miss_patterns.items++;
        miss_patterns.cnt--;
    } else {
        reg_field = &offload->reg_fields()[REG_FIELD_RECIRC];
        memset(&tag_spec, 0, sizeof(tag_spec));
        memset(&tag_mask, 0, sizeof(tag_mask));
        tag_spec.index = reg_field->index;
        tag_spec.data = recirc_id;
        tag_mask.index = 0xFF;
        tag_mask.data = reg_field->mask << reg_field->offset;
    }

    port_id.id = netdev_dpdk_get_port_id(netdev);
    if (!create_rte_flow(netdev, &miss_attr, &miss_patterns, &miss_actions,
                         doh, &error)) {
        return 0;
    }
    return -1;
}

static int OVS_UNUSED
add_tnl_pop_action(struct netdev *netdev,
                   struct flow_actions *actions,
                   const struct nlattr *nla,
                   struct act_resources *act_resources,
                   struct act_vars *act_vars)
{
    struct flow_miss_ctx miss_ctx;
    odp_port_t port;

    if (act_resources->sflow_id) {
        /* Reject flows with sample and jump actions */
        VLOG_DBG_RL(&rl, "cannot offload sFlow with jump");
        return -1;
    }
    port = nl_attr_get_odp_port(nla);
    memset(&miss_ctx, 0, sizeof miss_ctx);
    miss_ctx.vport = port;
    miss_ctx.skip_actions = act_vars->pre_ct_cnt;
    if (get_table_id(port, 0, netdev, act_vars->is_e2e_cache,
                     &act_resources->next_table_id)) {
        return -1;
    }
    if (!act_vars->is_e2e_cache &&
        get_flow_miss_ctx_id(&miss_ctx, netdev, act_resources->next_table_id,
                             0, &act_resources->flow_miss_ctx_id)) {
        return -1;
    }
    add_jump_action(actions, act_resources->next_table_id);
    return 0;
}

static int
add_recirc_action(struct netdev *netdev,
                  struct flow_actions *actions,
                  const struct nlattr *nla,
                  struct act_resources *act_resources,
                  struct act_vars *act_vars)
{
    struct flow_miss_ctx miss_ctx;

    if (act_resources->sflow_id) {
        /* Reject flows with sample and jump actions */
        VLOG_DBG_RL(&rl, "cannot offload sFlow with jump");
        return -1;
    }
    memset(&miss_ctx, 0, sizeof miss_ctx);
    if (act_vars->tnl_push_out_port != ODPP_NONE) {
        miss_ctx.vport = act_vars->tnl_push_out_port;
    } else {
        miss_ctx.vport = act_vars->vport;
    }
    miss_ctx.recirc_id = nl_attr_get_u32(nla);
    miss_ctx.skip_actions = act_vars->pre_ct_cnt;
    miss_ctx.has_dp_hash = act_vars->has_dp_hash;
    if (act_vars->tun_type != TUN_TYPE_NONE) {
        get_tnl_masked(&miss_ctx.tnl, NULL, act_vars->tnl_key,
                       &act_vars->tnl_mask);
    }
    if (get_table_id(miss_ctx.vport, miss_ctx.recirc_id, netdev,
                     act_vars->is_e2e_cache, &act_resources->next_table_id)) {
        return -1;
    }
    if (!act_vars->is_e2e_cache &&
        get_flow_miss_ctx_id(&miss_ctx, netdev, act_resources->next_table_id,
                             miss_ctx.recirc_id,
                             &act_resources->flow_miss_ctx_id)) {
        return -1;
    }
    if (act_vars->tun_type != TUN_TYPE_NONE && act_vars->recirc_id == 0) {
        if (get_tnl_id(act_vars->tnl_key, &act_vars->tnl_mask,
                       &act_resources->tnl_id)) {
            return -1;
        }
        if (add_action_set_reg_field(actions, REG_FIELD_TUN_INFO,
                                     act_resources->tnl_id, 0xFFFFFFFF)) {
            return -1;
        }
    }
    if (conntrack_offload_size() == 0) {
        add_action_set_reg_field(actions, REG_FIELD_RECIRC, miss_ctx.recirc_id,
                                 offload->reg_fields()[REG_FIELD_RECIRC].mask);
    }
    add_jump_action(actions, act_resources->next_table_id);
    return 0;
}

static void
dump_raw_decap(struct ds *s_extra,
               struct act_vars *act_vars)
{
    int i;

    ds_init(s_extra);
    ds_put_format(s_extra, "set raw_decap eth / udp / ");
    if (act_vars->is_outer_ipv4) {
        ds_put_format(s_extra, "ipv4 / ");
    } else {
        ds_put_format(s_extra, "ipv6 / ");
    }
    ds_put_format(s_extra, "geneve / ");
    for (i = 0; i < act_vars->gnv_opts_cnt; i++) {
        ds_put_format(s_extra, "geneve-opt / ");
    }
    ds_put_format(s_extra, "end_set");
}

static int
add_vxlan_decap_action(struct flow_actions *actions)
{
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_VXLAN_DECAP, NULL);
    return 0;
}

static int
add_geneve_decap_action(struct flow_actions *actions,
                        struct act_vars *act_vars)
{
    struct rte_flow_action_raw_decap *conf;

    conf = per_thread_xmalloc(sizeof (struct rte_flow_action_raw_decap));
    /* MLX5 PMD supports only one option of size 32 bits
     * which is the minimum size of options (if exists)
     * in case a flow exists with an option decapsulate 32 bits
     * from the header for the geneve options.
     */
    conf->size = sizeof (struct eth_header) +
                 sizeof (struct udp_header) +
                 sizeof (struct geneve_opt) +
                 (act_vars->is_outer_ipv4 ?
                  sizeof (struct ip_header) :
                  sizeof (struct ovs_16aligned_ip6_hdr)) +
                 (act_vars->gnv_opts_cnt ?
                   sizeof (uint32_t) : 0);

    conf->data = NULL;

    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_RAW_DECAP, conf);
    if (VLOG_IS_DBG_ENABLED()) {
        dump_raw_decap(&actions->s_tnl, act_vars);
    }
    return 0;
}

static int
add_gre_decap_action(struct flow_actions *actions)
{
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_NVGRE_DECAP, NULL);
    return 0;
}

static int
add_tnl_decap_action(struct flow_actions *actions,
                     struct act_vars *act_vars)
{
    struct rte_flow_action *actions_iter;
    int cnt;

    if (actions->cnt > 0) {
        struct rte_flow_action *last_action;

        last_action = &actions->actions[actions->cnt - 1];
        if (last_action->type == RTE_FLOW_ACTION_TYPE_RAW_ENCAP) {
            actions->cnt--;
            return 0;
        }
    }

    /* Avoid rewrite actions before decap action, as they have no meaning as
     * the outer header is going to be decapsulated.
     */
    for (cnt = 0, actions_iter = actions->actions; cnt < actions->cnt;
         cnt++, actions_iter++) {
        if (!IS_REWRITE_ACTION(actions_iter->type)) {
            continue;
        }

        actions_iter->type = RTE_FLOW_ACTION_TYPE_VOID;
    }

    if (act_vars->tun_type == TUN_TYPE_VXLAN) {
        return add_vxlan_decap_action(actions);
    }
    if (act_vars->tun_type == TUN_TYPE_GENEVE) {
        return add_geneve_decap_action(actions, act_vars);
    }
    if (act_vars->tun_type == TUN_TYPE_GRE) {
        return add_gre_decap_action(actions);
    }
    return -1;
}

static int
parse_ct_actions(struct netdev *netdev,
                 struct flow_actions *actions,
                 const struct nlattr *ct_actions,
                 const size_t ct_actions_len,
                 struct act_resources *act_resources,
                 struct act_vars *act_vars)
{
    struct ct_miss_ctx ct_miss_ctx;
    const struct nlattr *cta;
    unsigned int ctleft;

    /* Not Supported */
    if (act_vars->proto && act_vars->proto != IPPROTO_UDP &&
        act_vars->proto != IPPROTO_TCP) {
        VLOG_DBG_RL(&rl, "Unsupported CT offload for L4 protocol: 0x02%" PRIx8,
                    act_vars->proto);
        return -1;
    }

    memset(&ct_miss_ctx, 0, sizeof ct_miss_ctx);
    act_vars->ct_mode = CT_MODE_CT;
    NL_ATTR_FOR_EACH_UNSAFE (cta, ctleft, ct_actions, ct_actions_len) {
        if (nl_attr_type(cta) == OVS_CT_ATTR_ZONE) {
            if (!netdev_offload_dpdk_disable_zone_tables) {
                if (act_resources->ct_action_zone_id) {
                    put_zone_id(act_resources->ct_action_zone_id);
                    act_resources->ct_action_zone_id = 0;
                }
                if (act_resources->flow_id != INVALID_FLOW_MARK &&
                    get_zone_id(nl_attr_get_u16(cta),
                                &act_resources->ct_action_zone_id)) {
                    VLOG_DBG_RL(&rl, "Could not create zone id");
                    return -1;
                }
            } else {
                const uint32_t ct_zone_mask = offload->reg_fields()[REG_FIELD_CT_ZONE].mask;

                if (act_resources->flow_id != INVALID_FLOW_MARK &&
                    (get_zone_id(nl_attr_get_u16(cta),
                                 &act_resources->ct_action_zone_id) ||
                     add_action_set_reg_field(actions, REG_FIELD_CT_ZONE,
                                              act_resources->ct_action_zone_id,
                                              ct_zone_mask))) {
                    VLOG_DBG_RL(&rl, "Could not create zone id");
                    return -1;
                }
            }

            ct_miss_ctx.zone = nl_attr_get_u16(cta);
        } else if (nl_attr_type(cta) == OVS_CT_ATTR_MARK) {
            const uint32_t *key = nl_attr_get(cta);
            const uint32_t *mask = key + 1;

            add_action_set_reg_field(actions, REG_FIELD_CT_MARK, *key, *mask);
            ct_miss_ctx.mark = *key;
        } else if (nl_attr_type(cta) == OVS_CT_ATTR_LABELS) {
            const ovs_32aligned_u128 *key = nl_attr_get(cta);
            const ovs_32aligned_u128 *mask = key + 1;
            uint32_t set_value, set_mask;

            if (netdev_offload_dpdk_ct_labels_mapping) {
                ovs_u128 tmp_key, tmp_mask;

                tmp_key.u32[0] = key->u32[0];
                tmp_key.u32[1] = key->u32[1];
                tmp_key.u32[2] = key->u32[2];
                tmp_key.u32[3] = key->u32[3];

                tmp_mask.u32[0] = mask->u32[0];
                tmp_mask.u32[1] = mask->u32[1];
                tmp_mask.u32[2] = mask->u32[2];
                tmp_mask.u32[3] = mask->u32[3];

                tmp_key.u64.lo &= tmp_mask.u64.lo;
                tmp_key.u64.hi &= tmp_mask.u64.hi;
                if (get_label_id(&tmp_key, &act_resources->ct_action_label_id)) {
                    return -1;
                }
                set_value = act_resources->ct_action_label_id;
                set_mask = offload->reg_fields()[REG_FIELD_CT_LABEL_ID].mask;
            } else {
                if (!act_vars->is_ct_conn) {
                    if (key->u32[1] & mask->u32[1] ||
                        key->u32[2] & mask->u32[2] ||
                        key->u32[3] & mask->u32[3]) {
                        return -1;
                    }
                }
                set_value = key->u32[0] & mask->u32[0];
                set_mask = mask->u32[0];
            }

            if (add_action_set_reg_field(actions, REG_FIELD_CT_LABEL_ID,
                                         set_value, set_mask)) {
                VLOG_DBG_RL(&rl, "Could not create label id");
                return -1;
            }
            ct_miss_ctx.label.u32[0] = key->u32[0];
            ct_miss_ctx.label.u32[1] = key->u32[1];
            ct_miss_ctx.label.u32[2] = key->u32[2];
            ct_miss_ctx.label.u32[3] = key->u32[3];
        } else if (nl_attr_type(cta) == OVS_CT_ATTR_NAT) {
            act_vars->ct_mode = CT_MODE_CT_NAT;
        } else if (nl_attr_type(cta) == OVS_CT_ATTR_HELPER) {
            struct rte_flow_action_set_meta *set_meta;
            const char *helper = nl_attr_get(cta);
            uintptr_t ctid_key;

            if (strncmp(helper, "offl", strlen("offl"))) {
                continue;
            }

            if (!ovs_scan(helper, "offl,st(0x%"SCNx8"),id_key(0x%"SCNxPTR")",
                          &ct_miss_ctx.state, &ctid_key)) {
                VLOG_ERR("Invalid offload helper: '%s'", helper);
                return -1;
            }
            if (act_resources->flow_id == INVALID_FLOW_MARK) {
                struct flows_counter_key counter_id_key = {
                    .ptr_key = ctid_key,
                };
                struct indirect_ctx *ctx;
                struct rte_flow_action *ia;

                ctx = get_indirect_count_ctx(netdev, &counter_id_key,
                                             OVS_SHARED_CT_COUNT, true);
                if (!ctx) {
                    return -1;
                }
                act_resources->shared_count_ctx = ctx;
                ia = &actions->actions[actions->shared_count_action_pos];
                ovs_assert(ia->type == RTE_FLOW_ACTION_TYPE_INDIRECT &&
                           ia->conf == NULL);
                ia->conf = ctx->act_hdl;
            }

            act_vars->ct_mode = CT_MODE_CT_CONN;
            if (get_ct_ctx_id(&ct_miss_ctx, &act_resources->ct_miss_ctx_id)) {
                return -1;
            }
            add_action_set_reg_field(actions, REG_FIELD_CT_STATE,
                                     ct_miss_ctx.state, 0xFF);
            set_meta = per_thread_xzalloc(sizeof *set_meta);
            set_meta->data = act_resources->ct_miss_ctx_id;
            add_flow_action(actions, OVS_RTE_FLOW_ACTION_TYPE(CT_INFO), set_meta);
            if (act_resources->flow_id != INVALID_FLOW_MARK) {
                struct rte_flow_action_mark *mark =
                    per_thread_xzalloc(sizeof *mark);

                mark->id = act_resources->flow_id;
                add_flow_action(actions, OVS_RTE_FLOW_ACTION_TYPE(FLOW_INFO),
                                mark);
            }
            add_jump_action(actions, POSTCT_TABLE_ID);
            act_vars->has_known_fate = true;
        } else {
            VLOG_DBG_RL(&rl,
                        "Ignored nested action inside ct(), action type: %d",
                        nl_attr_type(cta));
            continue;
        }
    }
    return 0;
}

static int
parse_flow_actions(struct netdev *flowdev,
                   struct netdev *netdev,
                   struct flow_actions *actions,
                   struct nlattr *nl_actions,
                   size_t nl_actions_len,
                   struct act_resources *act_resources,
                   struct act_vars *act_vars,
                   uint8_t nest_level);

static int
add_sample_embedded_action(struct netdev *flowdev,
                           struct netdev *netdev,
                           struct flow_actions *actions,
                           struct nlattr *nl_actions,
                           size_t nl_actions_len,
                           struct act_resources *act_resources,
                           struct act_vars *act_vars,
                           uint8_t nest_level)
{
    struct rte_flow_action_sample *sample;
    struct flow_actions *sample_actions;

    sample_actions = per_thread_xzalloc(sizeof *sample_actions);

    if (parse_flow_actions(flowdev, netdev, sample_actions, nl_actions,
                           nl_actions_len, act_resources, act_vars,
                           nest_level + 1)) {
        goto err;
    }
    add_flow_action(sample_actions, RTE_FLOW_ACTION_TYPE_END, NULL);
    sample = per_thread_xzalloc(sizeof *sample);
    sample->ratio = 1;
    sample->actions = sample_actions->actions;
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_SAMPLE, sample);

    return 0;
err:
    per_thread_free(sample_actions);
    return -1;
}

static void
set_ct_ctnat_conf(const struct rte_flow_action *actions,
                  struct rte_flow_action_set_tag *ct_state,
                  struct rte_flow_action_set_tag *ctnat_state,
                  const void **ct_conf, const void **ctnat_conf)
{
    const struct rte_flow_action_set_tag *set_tag = actions->conf;
    struct reg_field *rf = &offload->reg_fields()[REG_FIELD_CT_STATE];

    *ct_conf = actions->conf;
    *ctnat_conf = actions->conf;

    /* In case this is not a ct-state set, no need for further changes. */
    if (actions->type != RTE_FLOW_ACTION_TYPE_SET_TAG ||
        set_tag->index != rf->index ||
        set_tag->mask != (rf->mask << rf->offset)) {
        return;
    }

    /* For ct-state set, clear NAT bits in ct_conf, and set in ctnat_conf.
     * Hops following this one will then be able to know that a CT-NAT action
     * has been executed in the past.
     */
    *ct_state = *(const struct rte_flow_action_set_tag *) *ct_conf;
    *ctnat_state =
        *(const struct rte_flow_action_set_tag *) *ctnat_conf;
    ct_state->data &= ((~(uint32_t) OVS_CS_F_NAT_MASK) << rf->offset);

    ctnat_state->data |= OVS_CS_F_NAT_MASK << rf->offset;
    *ct_conf = ct_state;
    *ctnat_conf = ctnat_state;
}

static void
split_ct_conn_actions(const struct rte_flow_action *actions,
                      struct flow_actions *ct_actions,
                      struct flow_actions *nat_actions,
                      struct rte_flow_action_set_tag *ct_state,
                      struct rte_flow_action_set_tag *ctnat_state)
{
    const void *ct_conf, *ctnat_conf;

    for (; actions && actions->type != RTE_FLOW_ACTION_TYPE_END; actions++) {
        /* This is only a dummy action used to split pre and post CT. It
         * should never be actually used.
         */
        if (actions->type == OVS_RTE_FLOW_ACTION_TYPE(PRE_CT_END)) {
            continue;
        }
        set_ct_ctnat_conf(actions, ct_state, ctnat_state, &ct_conf, &ctnat_conf);
        if (actions->type != RTE_FLOW_ACTION_TYPE_SET_IPV4_SRC &&
            actions->type != RTE_FLOW_ACTION_TYPE_SET_IPV4_DST &&
            actions->type != RTE_FLOW_ACTION_TYPE_SET_IPV6_SRC &&
            actions->type != RTE_FLOW_ACTION_TYPE_SET_IPV6_DST &&
            actions->type != RTE_FLOW_ACTION_TYPE_SET_TP_SRC &&
            actions->type != RTE_FLOW_ACTION_TYPE_SET_TP_DST) {
            add_flow_action(ct_actions, actions->type, ct_conf);
        }
        add_flow_action(nat_actions, actions->type, ctnat_conf);
    }
    add_flow_action(ct_actions, RTE_FLOW_ACTION_TYPE_END, NULL);
    add_flow_action(nat_actions, RTE_FLOW_ACTION_TYPE_END, NULL);
}

static int
create_ct_conn(struct netdev *netdev,
               struct flow_patterns *flow_patterns,
               struct flow_actions *flow_actions,
               struct rte_flow_error *error,
               struct act_resources *act_resources,
               struct flow_item *fi)
{
    struct flow_actions nat_actions = { .actions = NULL, .cnt = 0 };
    struct flow_actions ct_actions = { .actions = NULL, .cnt = 0 };
    struct rte_flow_action_set_tag ct_state, ctnat_state;
    struct rte_flow_attr attr = { .transfer = 1 };
    int ret = -1;
    int pos = 0;
    bool is_ct;

    fi->doh[0].rte_flow = fi->doh[1].rte_flow = NULL;

    split_ct_conn_actions(flow_actions->actions, &ct_actions, &nat_actions,
                          &ct_state, &ctnat_state);
    is_ct = ct_actions.cnt == nat_actions.cnt;

    if (act_resources) {
        put_table_id(act_resources->self_table_id);
        act_resources->self_table_id = 0;
    }
    pos = netdev_offload_ct_on_ct_nat;

    if (netdev_offload_ct_on_ct_nat || !is_ct) {
        attr.group = CTNAT_TABLE_ID;
        ret = create_rte_flow(netdev, &attr, flow_patterns, &nat_actions,
                              &fi->doh[pos], error);
        if (ret) {
            goto out;
        }
    }

    if (netdev_offload_ct_on_ct_nat || is_ct) {
        attr.group = CT_TABLE_ID;
        ret = create_rte_flow(netdev, &attr, flow_patterns, &ct_actions,
                              &fi->doh[0], error);
        if (ret) {
            goto ct_err;
        }
    }
    goto out;

ct_err:
    if (netdev_offload_ct_on_ct_nat) {
        netdev_offload_dpdk_destroy_flow(netdev, &fi->doh[1], NULL, true);
    }
out:
    free_flow_actions(&ct_actions, false);
    free_flow_actions(&nat_actions, false);
    return ret;
}

static void
split_pre_post_ct_actions(const struct rte_flow_action *actions,
                          struct flow_actions *pre_ct_actions,
                          struct flow_actions *post_ct_actions)
{
    struct flow_actions *split = pre_ct_actions;

    while (actions && actions->type != RTE_FLOW_ACTION_TYPE_END) {
        if (actions->type == RTE_FLOW_ACTION_TYPE_COUNT) {
            add_flow_action(post_ct_actions, actions->type, actions->conf);
        } else if (actions->type == OVS_RTE_FLOW_ACTION_TYPE(PRE_CT_END)) {
            split = post_ct_actions;
        } else {
            add_flow_action(split, actions->type, actions->conf);
        }
        actions++;
    }
}

static int
create_pre_post_ct(struct netdev *netdev,
                   const struct rte_flow_attr *attr,
                   struct flow_patterns *flow_patterns,
                   struct flow_actions *flow_actions,
                   struct rte_flow_error *error,
                   struct act_resources *act_resources,
                   struct act_vars *act_vars,
                   struct flow_item *fi)
{
    struct flow_actions post_ct_actions = { .actions = NULL, .cnt = 0 };
    struct flow_actions pre_ct_actions = { .actions = NULL, .cnt = 0 };
    struct rte_flow_item_port_id port_id;
    struct rte_flow_item_mark post_ct_mark;
    struct flow_patterns post_ct_patterns = {
        .items = (struct rte_flow_item []) {
            { .type = RTE_FLOW_ITEM_TYPE_PORT_ID, .spec = &port_id, },
            { .type = OVS_RTE_FLOW_ITEM_TYPE(FLOW_INFO), .spec = &post_ct_mark, },
            { .type = RTE_FLOW_ITEM_TYPE_END, },
        },
        .cnt = 3,
    };
    struct rte_flow_action_mark pre_ct_mark;
    struct rte_flow_action_jump pre_ct_jump;
    struct flow_miss_ctx pre_ct_miss_ctx;
    struct rte_flow_attr post_ct_attr;
    uint32_t ct_table_id;
    int ret;

    port_id.id = netdev_dpdk_get_port_id(netdev);

    /* post-ct */
    post_ct_mark.id = act_resources->flow_id;
    memcpy(&post_ct_attr, attr, sizeof post_ct_attr);
    post_ct_attr.group = POSTCT_TABLE_ID;
    split_pre_post_ct_actions(flow_actions->actions, &pre_ct_actions,
                              &post_ct_actions);
    add_flow_action(&post_ct_actions, RTE_FLOW_ACTION_TYPE_END, NULL);
    ret = create_rte_flow(netdev, &post_ct_attr, &post_ct_patterns,
                          &post_ct_actions, &fi->doh[1], error);
    if (ret) {
        goto out;
    }

    /* pre-ct */
    if (act_vars->ct_mode == CT_MODE_CT) {
        ct_table_id = CT_TABLE_ID;
    } else {
        ct_table_id = CTNAT_TABLE_ID;
    }
    if (!netdev_offload_dpdk_disable_zone_tables) {
        ct_table_id += act_resources->ct_action_zone_id;
    }
    pre_ct_miss_ctx.vport = act_vars->vport;
    pre_ct_miss_ctx.recirc_id = act_vars->recirc_id;
    if (act_vars->vport != ODPP_NONE) {
        get_tnl_masked(&pre_ct_miss_ctx.tnl, NULL, act_vars->tnl_key,
                       &act_vars->tnl_mask);
    } else {
        memset(&pre_ct_miss_ctx.tnl, 0, sizeof pre_ct_miss_ctx.tnl);
    }
    pre_ct_miss_ctx.skip_actions = act_vars->pre_ct_cnt;
    if (!act_resources->associated_flow_id) {
        if (associate_flow_id(act_resources->flow_id, &pre_ct_miss_ctx)) {
            goto pre_ct_err;
        }
        act_resources->associated_flow_id = true;
    }
    pre_ct_mark.id = act_resources->flow_id;
    add_flow_action(&pre_ct_actions, OVS_RTE_FLOW_ACTION_TYPE(FLOW_INFO),
                    &pre_ct_mark);
    pre_ct_jump.group = ct_table_id;
    add_flow_action(&pre_ct_actions, RTE_FLOW_ACTION_TYPE_JUMP, &pre_ct_jump);
    add_flow_action(&pre_ct_actions, RTE_FLOW_ACTION_TYPE_END, NULL);
    ret = create_rte_flow(netdev, attr, flow_patterns, &pre_ct_actions,
                          &fi->doh[0], error);
    if (ret) {
        goto pre_ct_err;
    }
    goto out;

pre_ct_err:
    netdev_offload_dpdk_destroy_flow(netdev, &fi->doh[1], NULL, true);
out:
    free_flow_actions(&pre_ct_actions, false);
    free_flow_actions(&post_ct_actions, false);
    return ret;
}

static int
netdev_offload_dpdk_flow_create(struct netdev *netdev,
                                const struct rte_flow_attr *attr,
                                struct flow_patterns *flow_patterns,
                                struct flow_actions *flow_actions,
                                struct rte_flow_error *error,
                                struct act_resources *act_resources,
                                struct act_vars *act_vars,
                                struct flow_item *fi)
{
    int ret = 0;

    fi->flow_offload = true;

    switch (act_vars->ct_mode) {
    case CT_MODE_NONE:
        ret = create_rte_flow(netdev, attr, flow_patterns, flow_actions,
                              &fi->doh[0], error);
        break;
    case CT_MODE_CT:
        /* fallthrough */
    case CT_MODE_CT_NAT:
        ret = create_pre_post_ct(netdev, attr, flow_patterns, flow_actions,
                                 error, act_resources, act_vars, fi);
        break;
    case CT_MODE_CT_CONN:
        fi->flow_offload = false;
        ret = create_ct_conn(netdev, flow_patterns, flow_actions, error,
                             act_resources, fi);
        break;
    default:
        OVS_NOT_REACHED();
    }

    return ret;
}

void *
find_raw_encap_spec(const struct raw_encap_data *raw_encap_data,
                    enum rte_flow_item_type type)
{
    struct ovs_16aligned_ip6_hdr *ipv6;
    struct udp_header *udp = NULL;
    struct vlan_header *vlan;
    struct eth_header *eth;
    struct ip_header *ipv4;
    uint8_t *next_hdr;
    uint16_t proto;

    eth = ALIGNED_CAST(struct eth_header *, raw_encap_data->conf.data);
    if (type == RTE_FLOW_ITEM_TYPE_ETH) {
        return eth;
    }

    next_hdr = (uint8_t *) (eth + 1);
    proto = htons(eth->eth_type);
    /* VLAN skipping */
    while (eth_type_vlan(ntohs(proto))) {
        vlan = ALIGNED_CAST(struct vlan_header *, next_hdr);
        proto = htons(vlan->vlan_next_type);
        next_hdr += sizeof *vlan;
    }

    if (proto == RTE_ETHER_TYPE_IPV4) {
        ipv4 = ALIGNED_CAST(struct ip_header *, next_hdr);
        if (type == RTE_FLOW_ITEM_TYPE_IPV4) {
            return ipv4;
        }
        if (ipv4->ip_proto != IPPROTO_UDP) {
            return NULL;
        }
        udp = (struct udp_header *) (ipv4 + 1);
    }

    if (proto == RTE_ETHER_TYPE_IPV6) {
        ipv6 = ALIGNED_CAST(struct ovs_16aligned_ip6_hdr *, next_hdr);
        if (type == RTE_FLOW_ITEM_TYPE_IPV6) {
            return ipv6;
        }
        if (ipv6->ip6_nxt != IPPROTO_UDP) {
            return NULL;
        }
        udp = (struct udp_header *) (ipv6 + 1);
    }

    if (udp && type == RTE_FLOW_ITEM_TYPE_UDP) {
        return udp;
    }

    return NULL;
}

static void *
find_encap_spec(struct raw_encap_data *raw_encap_data,
                enum rte_flow_item_type type)
{
    void *rv;

    if (raw_encap_data) {
        rv = find_raw_encap_spec(raw_encap_data, type);
        if (rv == NULL) {
            VLOG_DBG_RL(&rl, "Could not find raw_encap spec type=%d", type);
        }
        return rv;
    }

    OVS_NOT_REACHED();
}

static void
set_encap_field__(const void *value, void *hdr, void *mask, const size_t size)
{
    const uint8_t *v;
    uint8_t *h, *m;
    uint32_t i;

    if (!mask) {
        memcpy(hdr, value, size);
        return;
    }

    for (i = 0, h = hdr, v = value, m = mask; i < size; i++) {
        *h = (*h & ~*m) | (*v & *m);
        h++;
        v++;
        *m++ = 0;
    }
}

static int
outer_encap_set_actions(struct raw_encap_data *raw_encap_data,
                        const struct nlattr *set_actions,
                        const size_t set_actions_len,
                        bool masked)
{
    const struct nlattr *sa;
    unsigned int sleft;

#define set_encap_field(field, hdr)                                           \
    set_encap_field__(&key->field, hdr,                                       \
                    mask ? CONST_CAST(void *, &mask->field) : NULL,           \
                    sizeof key->field)

    NL_ATTR_FOR_EACH_UNSAFE (sa, sleft, set_actions, set_actions_len) {
        if (nl_attr_type(sa) == OVS_KEY_ATTR_ETHERNET) {
            const struct ovs_key_ethernet *key = nl_attr_get(sa);
            const struct ovs_key_ethernet *mask = masked ? key + 1 : NULL;
            struct rte_flow_item_eth *spec;

            spec = find_encap_spec(raw_encap_data, RTE_FLOW_ITEM_TYPE_ETH);
            if (!spec) {
                return -1;
            }

            set_encap_field(eth_src, spec->src.addr_bytes);
            set_encap_field(eth_dst, spec->dst.addr_bytes);

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported ETHERNET set action");
                return -1;
            }
        } else if (nl_attr_type(sa) == OVS_KEY_ATTR_IPV4) {
            const struct ovs_key_ipv4 *key = nl_attr_get(sa);
            const struct ovs_key_ipv4 *mask = masked ? key + 1 : NULL;
            struct rte_flow_item_ipv4 *spec;

            spec = find_encap_spec(raw_encap_data, RTE_FLOW_ITEM_TYPE_IPV4);
            if (!spec) {
                return -1;
            }

            set_encap_field(ipv4_src, &spec->hdr.src_addr);
            set_encap_field(ipv4_dst, &spec->hdr.dst_addr);
            set_encap_field(ipv4_ttl, &spec->hdr.time_to_live);

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported IPv4 set action");
                return -1;
            }
        } else if (nl_attr_type(sa) == OVS_KEY_ATTR_IPV6) {
            const struct ovs_key_ipv6 *key = nl_attr_get(sa);
            const struct ovs_key_ipv6 *mask = masked ? key + 1 : NULL;
            struct rte_flow_item_ipv6 *spec;

            spec = find_encap_spec(raw_encap_data, RTE_FLOW_ITEM_TYPE_IPV6);
            if (!spec) {
                return -1;
            }

            set_encap_field(ipv6_src, &spec->hdr.src_addr);
            set_encap_field(ipv6_dst, &spec->hdr.dst_addr);
            set_encap_field(ipv6_hlimit, &spec->hdr.hop_limits);

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported IPv6 set action");
                return -1;
            }
        } else if (nl_attr_type(sa) == OVS_KEY_ATTR_UDP) {
            const struct ovs_key_udp *key = nl_attr_get(sa);
            const struct ovs_key_udp *mask = masked ? key + 1 : NULL;
            struct rte_flow_item_udp *spec;

            spec = find_encap_spec(raw_encap_data, RTE_FLOW_ITEM_TYPE_UDP);
            if (!spec) {
                return -1;
            }

            set_encap_field(udp_src, &spec->hdr.src_port);
            set_encap_field(udp_dst, &spec->hdr.dst_port);

            if (mask && !is_all_zeros(mask, sizeof *mask)) {
                VLOG_DBG_RL(&rl, "Unsupported UDP set action");
                return -1;
            }
        } else {
            VLOG_DBG_RL(&rl,
                        "Unsupported set action type %d", nl_attr_type(sa));
            return -1;
        }
    }

    return 0;
}

static int
parse_flow_actions(struct netdev *flowdev,
                   struct netdev *netdev,
                   struct flow_actions *actions,
                   struct nlattr *nl_actions,
                   size_t nl_actions_len,
                   struct act_resources *act_resources,
                   struct act_vars *act_vars,
                   uint8_t nest_level)
{
    struct raw_encap_data *raw_encap_data = NULL;
    struct nlattr *nla;
    int left;

    if (nest_level == 0) {
        if (nl_actions_len != 0 &&
            act_vars->tun_type != TUN_TYPE_NONE &&
            act_vars->recirc_id == 0 &&
            act_vars->vport != ODPP_NONE &&
            add_tnl_decap_action(actions, act_vars)) {
            return -1;
        }
        if (add_count_action(netdev, actions, act_resources, act_vars)) {
            return -1;
        }
    }

    NL_ATTR_FOR_EACH_UNSAFE (nla, left, nl_actions, nl_actions_len) {
        if (act_vars->has_dp_hash &&
            nl_attr_type(nla) != OVS_ACTION_ATTR_RECIRC) {
            return -1;
        }

        if (nl_attr_type(nla) == OVS_ACTION_ATTR_OUTPUT) {
            /* The last output should use port-id action, while previous
             * outputs should embed the port-id action inside a sample action.
             */
            if (left <= NLA_ALIGN(nla->nla_len)) {
                if (add_output_action(flowdev, actions, nla)) {
                   return -1;
                }
            } else {
                if (add_sample_embedded_action(flowdev, netdev, actions, nla,
                                               nl_attr_get_size(nla),
                                               act_resources, act_vars,
                                               nest_level)) {
                    return -1;
                }
            }
            act_vars->tnl_push_out_port = ODPP_NONE;
            act_vars->has_known_fate = true;
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_DROP) {
            add_flow_action(actions, RTE_FLOW_ACTION_TYPE_DROP, NULL);
            act_vars->has_known_fate = true;
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_SET ||
                   nl_attr_type(nla) == OVS_ACTION_ATTR_SET_MASKED) {
            const struct nlattr *set_actions = nl_attr_get(nla);
            const size_t set_actions_len = nl_attr_get_size(nla);
            bool masked = nl_attr_type(nla) == OVS_ACTION_ATTR_SET_MASKED;

            if (raw_encap_data &&
                !outer_encap_set_actions(raw_encap_data, set_actions,
                                         set_actions_len, masked)) {
                continue;
            }

            if (parse_set_actions(actions, set_actions, set_actions_len,
                                  masked, act_vars->proto)) {
                return -1;
            }
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_PUSH_VLAN) {
            const struct ovs_action_push_vlan *vlan = nl_attr_get(nla);
            struct vlan_eth_header *veh;

            if (!raw_encap_data) {
                if (parse_vlan_push_action(actions, vlan, act_vars)) {
                    return -1;
                }
                continue;
            }

            /* Insert new 802.1Q header. */
            raw_encap_data->conf.data -= VLAN_HEADER_LEN;
            if (raw_encap_data->conf.data < raw_encap_data->headroom) {
                return -1;
            }
            raw_encap_data->conf.size += VLAN_HEADER_LEN;
            veh = ALIGNED_CAST(struct vlan_eth_header *,
                               raw_encap_data->conf.data);
            memmove(veh, (char *)veh + VLAN_HEADER_LEN, 2 * ETH_ADDR_LEN);
            veh->veth_type = vlan->vlan_tpid;
            veh->veth_tci = vlan->vlan_tci & htons(~VLAN_CFI);
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_POP_VLAN) {
            add_flow_action(actions, RTE_FLOW_ACTION_TYPE_OF_POP_VLAN, NULL);
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_TUNNEL_PUSH) {
            const struct ovs_action_push_tnl *tnl_push = nl_attr_get(nla);
            struct rte_flow_action *last_action = NULL;

            if (actions->cnt > 0) {
                last_action = &actions->actions[actions->cnt - 1];
            }

            act_vars->tnl_push_out_port = tnl_push->out_port;

            if (last_action &&
                (last_action->type == RTE_FLOW_ACTION_TYPE_NVGRE_DECAP ||
                 last_action->type == RTE_FLOW_ACTION_TYPE_RAW_DECAP ||
                 (tnl_push->tnl_type == OVS_VPORT_TYPE_VXLAN &&
                  last_action->type == RTE_FLOW_ACTION_TYPE_VXLAN_DECAP))) {
                actions->cnt--;
                continue;
            }

            raw_encap_data = per_thread_xzalloc(sizeof *raw_encap_data);
            memcpy(raw_encap_data->data, tnl_push->header,
                   tnl_push->header_len);
            raw_encap_data->tnl_type = tnl_push->tnl_type;
            raw_encap_data->conf.data = raw_encap_data->data;
            raw_encap_data->conf.preserve = NULL;
            raw_encap_data->conf.size = tnl_push->header_len;
            add_flow_action(actions, RTE_FLOW_ACTION_TYPE_RAW_ENCAP,
                            raw_encap_data);
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_CLONE) {
            size_t clone_actions_len = nl_attr_get_size(nla);
            struct nlattr *clone_actions;

            clone_actions = CONST_CAST(struct nlattr *, nl_attr_get(nla));
            /* The last cloned action is parsed and actions are applied
             * natively, while previous ones are parsed and the actions are
             * applied embedded in a sample action.
             */
            if (left <= NLA_ALIGN(nla->nla_len)) {
                if (parse_flow_actions(flowdev, netdev, actions, clone_actions,
                                       clone_actions_len, act_resources,
                                       act_vars, nest_level + 1)) {
                    return -1;
                }
            } else {
                if (add_sample_embedded_action(flowdev, netdev, actions,
                                               clone_actions,
                                               clone_actions_len,
                                               act_resources, act_vars,
                                               nest_level)) {
                    return -1;
                }
            }
#ifdef ALLOW_EXPERIMENTAL_API /* Packet restoration API required. */
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_TUNNEL_POP) {
            if (add_tnl_pop_action(netdev, actions, nla, act_resources,
                                   act_vars)) {
                return -1;
            }
            act_vars->has_known_fate = true;
#endif
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_RECIRC) {
            if (add_recirc_action(netdev, actions, nla, act_resources,
                                  act_vars)) {
                return -1;
            }
            act_vars->has_known_fate = true;
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_CT) {
            const struct nlattr *ct_actions = nl_attr_get(nla);
            size_t ct_actions_len = nl_attr_get_size(nla);

            if (parse_ct_actions(netdev, actions, ct_actions, ct_actions_len,
                                 act_resources, act_vars)) {
                return -1;
            }
            add_flow_action(actions, OVS_RTE_FLOW_ACTION_TYPE(PRE_CT_END), NULL);
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_SAMPLE) {
            struct dpif_sflow_attr sflow_attr;

            memset(&sflow_attr, 0, sizeof sflow_attr);
            if (parse_sample_action(actions, nla,
                                    &sflow_attr, act_resources, act_vars)) {
                return -1;
            }
            act_vars->has_known_fate = true;
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_USERSPACE) {
            struct dpif_sflow_attr sflow_attr;

            memset(&sflow_attr, 0, sizeof sflow_attr);
            /* Cases where the sFlow sampling rate is 1 the ovs action
             * is translated into OVS_ACTION_ATTR_USERSPACE and not
             * OVS_ACTION_ATTR_SAMPLE, this requires only mapping the
             * sFlow cookie.
             */
            sflow_attr.sflow = nla;
            sflow_attr.sflow_len = nla->nla_len;
            if (map_sflow_attr(actions, nla, &sflow_attr,
                               act_resources, act_vars)) {
                return -1;
            }
            add_empty_sample_action(1, actions);
            act_vars->has_known_fate = true;
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_METER) {
            if (add_meter_action(actions, nla, act_resources)) {
                return -1;
            }
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_CT_CLEAR) {
            if (act_resources->ct_action_label_id) {
                put_label_id(act_resources->ct_action_label_id);
                act_resources->ct_action_label_id = 0;
            }
            if (act_resources->ct_action_zone_id) {
                put_zone_id(act_resources->ct_action_zone_id);
                act_resources->ct_action_zone_id = 0;
            }
            if (get_zone_id(0, &act_resources->ct_action_zone_id)) {
                return -1;
            }
            add_action_set_reg_field(actions, REG_FIELD_CT_STATE, 0,
                                     offload->reg_fields()[REG_FIELD_CT_STATE].mask);
            add_action_set_reg_field(actions, REG_FIELD_CT_ZONE,
                                     act_resources->ct_action_zone_id,
                                     offload->reg_fields()[REG_FIELD_CT_ZONE].mask);
            add_action_set_reg_field(actions, REG_FIELD_CT_MARK, 0,
                                     offload->reg_fields()[REG_FIELD_CT_MARK].mask);
            add_action_set_reg_field(actions, REG_FIELD_CT_LABEL_ID, 0,
                                     offload->reg_fields()[REG_FIELD_CT_LABEL_ID].mask);
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_HASH) {
            const struct ovs_action_hash *hash_act = nl_attr_get(nla);
            struct hash_data *hash_data;

            act_vars->has_dp_hash = true;

            hash_data = per_thread_xzalloc(sizeof *hash_data);
            hash_data->flow_id = act_resources->flow_id;
            hash_data->seed = hash_act->hash_basis;
            add_flow_action(actions, OVS_RTE_FLOW_ACTION_TYPE(HASH), hash_data);
        } else if (nl_attr_type(nla) == OVS_ACTION_ATTR_TUN_DECAP) {
            if (add_tnl_decap_action(actions, act_vars)) {
                VLOG_DBG_RL(&rl, "Cannot decap non-tunnel");
                return -1;
            }
        } else {
            VLOG_DBG_RL(&rl, "Unsupported action type %d", nl_attr_type(nla));
            return -1;
        }
        if (act_vars->ct_mode == CT_MODE_NONE) {
            act_vars->pre_ct_cnt++;
        }
    }

    if (act_vars->ct_mode == CT_MODE_NONE) {
        act_vars->pre_ct_cnt = 0;
    }

    if (nl_actions_len == 0) {
        VLOG_DBG_RL(&rl, "No actions provided");
        return -1;
    }

    if (!act_vars->has_known_fate) {
        add_flow_action(actions, RTE_FLOW_ACTION_TYPE_DROP, NULL);
    }

    return 0;
}

static int
netdev_offload_dpdk_actions(struct netdev *flowdev,
                            struct netdev *netdev,
                            struct rte_flow_attr *flow_attr,
                            struct flow_patterns *patterns,
                            struct nlattr *nl_actions,
                            size_t actions_len,
                            struct act_resources *act_resources,
                            struct act_vars *act_vars,
                            struct flow_item *fi)
{
    struct flow_actions actions = {
        .actions = NULL,
        .cnt = 0,
        .s_tnl = DS_EMPTY_INITIALIZER,
    };
    struct rte_flow_error error;
    int ret;

    ret = parse_flow_actions(flowdev, netdev, &actions, nl_actions,
                             actions_len, act_resources, act_vars, 0);
    if (ret) {
        goto out;
    }
    add_flow_action(&actions, RTE_FLOW_ACTION_TYPE_END, NULL);
    flow_attr->group = act_resources->self_table_id;
    ret = netdev_offload_dpdk_flow_create(netdev, flow_attr, patterns,
                                          &actions, &error, act_resources,
                                          act_vars, fi);

    if (ret == 0) {
        unsigned int tid = netdev_offload_thread_id();
        struct netdev_offload_dpdk_data *data;

        /* The flowdev's counters are updated, not the netdev's ones. */
        data = (struct netdev_offload_dpdk_data *)
            ovsrcu_get(void *, &flowdev->hw_info.offload_data);

        if (fi->flow_offload) {
            data->flow_counters[tid]++;
        } else {
            data->conn_counters[tid]++;
        }
    }
out:
    free_flow_actions(&actions, true);
    return ret;
}

static struct ufid_to_rte_flow_data *
netdev_offload_dpdk_add_flow(struct netdev *netdev,
                             struct match *match,
                             struct nlattr *nl_actions,
                             size_t actions_len,
                             const ovs_u128 *ufid,
                             struct offload_info *info,
                             struct ufid_to_rte_flow_data *old_rte_flow_data)
{
    struct act_resources act_resources = { .flow_id = info->flow_mark };
    struct rte_flow_attr flow_attr = {
        .transfer = 1,
        .priority = DPDK_OFFLOAD_PRIORITY_LOW,
    };
    struct flow_patterns patterns = {
        .items = NULL,
        .cnt = 0,
        .s_tnl = DS_EMPTY_INITIALIZER,
    };
    struct act_vars act_vars = {
        .vport = ODPP_NONE,
        .tnl_push_out_port = ODPP_NONE,
    };
    struct ufid_to_rte_flow_data *flows_data = NULL;
    int ret;

    act_vars.is_e2e_cache = info->is_e2e_cache_flow;
    act_vars.is_ct_conn = info->is_ct_conn;
    act_vars.ct_counter_key = info->ct_counter_key;
    memcpy(&act_vars.flows_counter_key, &info->flows_counter_key,
           sizeof info->flows_counter_key);
    ret = parse_flow_match(netdev, info->orig_in_port, &flow_attr, &patterns,
                           match, nl_actions, actions_len, &act_resources,
                           &act_vars);
    if (ret) {
        if (OVS_UNLIKELY(!VLOG_DROP_DBG((&rl)))) {
            struct ds match_ds = DS_EMPTY_INITIALIZER;

            match_format(match, NULL, &match_ds, OFP_DEFAULT_PRIORITY);
            VLOG_DBG("%s: some matches of ufid "UUID_FMT" are not supported: %s",
                     netdev_get_name(netdev), UUID_ARGS((struct uuid *) ufid),
                     ds_cstr(&match_ds));
            ds_destroy(&match_ds);
        }
        goto out;
    }

    flows_data = old_rte_flow_data ? old_rte_flow_data : xzalloc(sizeof *flows_data);
    ret = netdev_offload_dpdk_actions(netdev, patterns.physdev, &flow_attr,
                                      &patterns, nl_actions, actions_len,
                                      &act_resources, &act_vars,
                                      &flows_data->flow_item);
    if (ret) {
        goto out;
    }

    if (old_rte_flow_data) {
        ovs_mutex_lock(&flows_data->lock);
        ufid_to_rte_flow_set(flows_data, netdev, patterns.physdev, &act_resources);
        rte_flow_data_active_set(flows_data, true);
        ovs_mutex_unlock(&flows_data->lock);
    } else {
        if (offload_data_is_conn(&act_resources)) {
            conn_link(netdev, flows_data, &act_resources);
        } else if (ufid_to_rte_flow_associate(ufid, netdev, patterns.physdev,
                                              flows_data, &act_resources)) {
            ret = -1;
            goto out;
        }
    }
    VLOG_DBG("%s/%s: installed flow %p/%p by ufid "UUID_FMT,
             netdev_get_name(netdev), netdev_get_name(patterns.physdev),
             flows_data->flow_item.doh[0].rte_flow,
             flows_data->flow_item.doh[1].rte_flow,
             UUID_ARGS((struct uuid *) ufid));

out:
    if (ret) {
        put_action_resources(&act_resources);
        if (!old_rte_flow_data) {
            free(flows_data);
        }
        flows_data = NULL;
    }
    free_flow_patterns(&patterns);
    return flows_data;
}

static int
netdev_offload_dpdk_remove_flows(struct ufid_to_rte_flow_data *rte_flow_data, bool modification)
{
    unsigned int tid = netdev_offload_thread_id();
    struct netdev_offload_dpdk_data *data;
    struct dpdk_offload_handle *doh;
    void *handles[NUM_HANDLE_PER_ITEM] = {
        rte_flow_data->flow_item.doh[0].rte_flow,
        rte_flow_data->flow_item.doh[1].rte_flow,
    };
    struct netdev *physdev;
    struct netdev *netdev;
    ovs_u128 *ufid;
    int ret = -1;
    int i;

    if (!rte_flow_data_active(rte_flow_data)) {
        return 0;
    }

    ovs_mutex_lock(&rte_flow_data->lock);

    if (!rte_flow_data_active(rte_flow_data)) {
        ovs_mutex_unlock(&rte_flow_data->lock);
        return 0;
    }

    rte_flow_data_active_set(rte_flow_data, false);

    physdev = rte_flow_data->physdev;
    netdev = rte_flow_data->netdev;
    ufid = &rte_flow_data->ufid;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    for (i = 0; i < NUM_HANDLE_PER_ITEM; i++) {
        doh = &rte_flow_data->flow_item.doh[i];

        if (!doh->valid) {
            continue;
        }

        ret = netdev_offload_dpdk_destroy_flow(physdev, doh, ufid, true);
        if (ret) {
            break;
        }
    }

    if (ret == 0) {
        if (rte_flow_data->flow_item.flow_offload) {
            data->flow_counters[tid]--;
        } else {
            data->conn_counters[tid]--;
        }
        put_action_resources(&rte_flow_data->act_resources);
        VLOG_DBG_RL(&rl, "%s/%s: removed flows 0x%"PRIxPTR"/0x%"PRIxPTR
                    " associated with ufid " UUID_FMT,
                    netdev_get_name(netdev), netdev_get_name(physdev),
                    (intptr_t) handles[0], (intptr_t) handles[1],
                    UUID_ARGS((struct uuid *) ufid));

        if (!modification) {
            if (offload_data_is_conn(&rte_flow_data->act_resources)) {
                conn_unlink(rte_flow_data);
            } else {
                ufid_to_rte_flow_disassociate(rte_flow_data);
            }
        }
    } else {
        VLOG_ERR("Failed flow destroy: %s/%s ufid " UUID_FMT,
                 netdev_get_name(netdev), netdev_get_name(physdev),
                 UUID_ARGS((struct uuid *) ufid));
    }

    ovs_mutex_unlock(&rte_flow_data->lock);

    return ret;
}

struct get_netdev_odp_aux {
    struct netdev *netdev;
    odp_port_t odp_port;
};

static bool
get_netdev_odp_cb(struct netdev *netdev,
                  odp_port_t odp_port,
                  void *aux_)
{
    struct get_netdev_odp_aux *aux = aux_;

    if (netdev == aux->netdev) {
        aux->odp_port = odp_port;
        return true;
    }
    return false;
}

static int
netdev_offload_dpdk_flow_put(struct netdev *netdev, struct match *match,
                             struct nlattr *actions, size_t actions_len,
                             const ovs_u128 *ufid, struct offload_info *info,
                             struct dpif_flow_stats *stats)
{
    struct ufid_to_rte_flow_data *old_rte_flow_data;
    struct ufid_to_rte_flow_data *rte_flow_data;
    struct dpif_flow_stats old_stats;
    bool modification = false;
    int ret;

    /*
     * If an old rte_flow exists, it means it's a flow modification.
     * Here destroy the old rte flow first before adding a new one.
     * Keep the stats for the newly created rule.
     */
    old_rte_flow_data = ufid_to_rte_flow_data_find(netdev, ufid, false);
    if (old_rte_flow_data && old_rte_flow_data->flow_item.doh[0].rte_flow) {
        struct get_netdev_odp_aux aux = {
            .netdev = old_rte_flow_data->physdev,
            .odp_port = ODPP_NONE,
        };

        /* Extract the orig_in_port from physdev as in case of modify the one
         * provided by upper layer cannot be used.
         */
        netdev_ports_traverse(old_rte_flow_data->physdev->dpif_type,
                              get_netdev_odp_cb, &aux);
        info->orig_in_port = aux.odp_port;
        old_stats = old_rte_flow_data->stats;
        modification = true;
        ret = netdev_offload_dpdk_remove_flows(old_rte_flow_data, true);
        if (ret < 0) {
            return ret;
        }
    } else {
        old_rte_flow_data = NULL;
    }

    per_thread_init();

    rte_flow_data = netdev_offload_dpdk_add_flow(netdev, match, actions,
                                                 actions_len, ufid, info, old_rte_flow_data);
    if (!rte_flow_data) {
        if (modification) {
            ovs_mutex_lock(&old_rte_flow_data->lock);
            ufid_to_rte_flow_disassociate(old_rte_flow_data);
            ovs_mutex_unlock(&old_rte_flow_data->lock);
        }
        return -1;
    }
    if (modification) {
        rte_flow_data->stats = old_stats;
    }
    if (stats) {
        *stats = rte_flow_data->stats;
    }
    return 0;
}

static int
netdev_offload_dpdk_flow_del(struct netdev *netdev OVS_UNUSED,
                             const ovs_u128 *ufid,
                             struct dpif_flow_stats *stats)
{
    struct ufid_to_rte_flow_data *rte_flow_data;

    rte_flow_data = ufid_to_rte_flow_data_find(netdev, ufid, true);
    if (!rte_flow_data || !rte_flow_data->flow_item.doh[0].valid) {
        return -1;
    }

    if (stats) {
        memset(stats, 0, sizeof *stats);
    }
    return netdev_offload_dpdk_remove_flows(rte_flow_data, false);
}

static void
offload_provider_api_init(void)
{
    static struct ovsthread_once init_once = OVSTHREAD_ONCE_INITIALIZER;

    if (ovsthread_once_start(&init_once)) {
        if (ovs_doca_enabled()) {
            offload = &dpdk_offload_api_doca;
        } else {
            offload = &dpdk_offload_api_rte;
        }

        ovsthread_once_done(&init_once);
    }
}

static int
netdev_offload_dpdk_init_flow_api(struct netdev *netdev)
{
    int ret = EOPNOTSUPP;
    struct esw_members_aux aux = {
        .esw_netdev = netdev,
        .op = init_esw_members_op,
    };

    if ((netdev_vport_is_vport_class(netdev->netdev_class) ||
         !strcmp(netdev_get_type(netdev), "tap"))
        && !strcmp(netdev_get_dpif_type(netdev), "system")) {
        VLOG_DBG("%s: vport belongs to the system datapath. Skipping.",
                 netdev_get_name(netdev));
        return EOPNOTSUPP;
    }

    /* VDPA ports which are added without representor are not backed by ethdev
     * in DPDK and are not compatible with the offload API so do not init it.
     */
    if (!netdev_dpdk_is_ethdev(netdev) &&
        !netdev_vport_is_vport_class(netdev->netdev_class) &&
        strcmp(netdev_get_type(netdev), "tap")) {
        return EOPNOTSUPP;
    }

    offload_provider_api_init();

    if (netdev_dpdk_flow_api_supported(netdev)) {
        ret = offload_data_init(netdev);
    }

    /* If the netdev is an ESW manager, init its members too. */
    if (netdev_dpdk_is_esw_mgr(netdev)) {
        netdev_ports_traverse(netdev->dpif_type, esw_members_cb, &aux);
    }

    netdev_offload_dpdk_ct_labels_mapping = netdev_is_ct_labels_mapping_enabled();
    netdev_offload_dpdk_disable_zone_tables = netdev_is_zone_tables_disabled();

    return ret;
}

static void
netdev_offload_dpdk_uninit_flow_api(struct netdev *netdev)
{
    struct esw_members_aux aux = {
        .esw_netdev = netdev,
        .op = uninit_esw_members_op,
    };

    /* If the netdev is an ESW manager, uninit its members too. */
    if (netdev_dpdk_is_esw_mgr(netdev)) {
        netdev_ports_traverse(netdev->dpif_type, esw_members_cb, &aux);
    }

    if (netdev_dpdk_flow_api_supported(netdev)) {
        offload_data_destroy(netdev);
    }
}

static void
netdev_offload_dpdk_update_stats(struct dpif_flow_stats *stats,
                                 struct dpif_flow_attrs *attrs,
                                 struct rte_flow_query_count *query)
{
    if (attrs) {
        attrs->dp_layer = "dpdk";
    }

    if (!query) {
        return;
    }

    stats->n_packets += (query->hits_set) ? query->hits : 0;
    stats->n_bytes += (query->bytes_set) ? query->bytes : 0;
}

static int
netdev_offload_dpdk_flow_get(struct netdev *netdev,
                             struct match *match OVS_UNUSED,
                             struct nlattr **actions OVS_UNUSED,
                             const ovs_u128 *ufid,
                             struct dpif_flow_stats *stats,
                             struct dpif_flow_attrs *attrs,
                             struct ofpbuf *buf OVS_UNUSED,
                             long long now)
{
    struct rte_flow_query_count query = { .reset = 1 };
    struct ufid_to_rte_flow_data *rte_flow_data;
    struct dpdk_offload_handle *doh;
    struct rte_flow_error error;
    struct indirect_ctx *ctx;
    int ret = 0;

    attrs->dp_extra_info = NULL;

    rte_flow_data = ufid_to_rte_flow_data_find(netdev, ufid, false);
    if (rte_flow_data) {
        attrs->offloaded = true;
    }

    if (!rte_flow_data || !rte_flow_data->flow_item.doh[0].rte_flow ||
        !rte_flow_data_active(rte_flow_data) || ovs_mutex_trylock(&rte_flow_data->lock)) {
        if (rte_flow_data) {
            offload->update_stats(&rte_flow_data->stats, attrs, NULL);
            /* The node might be in the process of being deleted.
             * It should be fine to read from it as actual freeing is done
             * after an RCU grace period. */
            memcpy(stats, &rte_flow_data->stats, sizeof *stats);
            return 0;
        }
        return -1;
    }

    /* Check again whether the data is inactive, as it could have been
     * updated while the lock was not yet taken. The first check above
     * was only to avoid unnecessary locking if possible.
     */
    if (!rte_flow_data_active(rte_flow_data)) {
        goto out;
    }

    doh = rte_flow_data->flow_item.doh[1].rte_flow
        ? &rte_flow_data->flow_item.doh[1]
        : &rte_flow_data->flow_item.doh[0];

    if (!rte_flow_data->act_resources.shared_count_ctx) {
        ret = offload->query_count(rte_flow_data->physdev, doh, &query, &error);
        if (ret) {
            VLOG_DBG_RL(&rl, "%s: Failed to query ufid "UUID_FMT" flow: %p. "
                        "%d (%s)", netdev_get_name(netdev),
                        UUID_ARGS((struct uuid *) ufid), doh->rte_flow,
                        error.type, error.message);
            goto out;
        }
    } else {
        ctx = rte_flow_data->act_resources.shared_count_ctx;
        ret = offload->shared_query(ctx, &query, &error);
        if (ret) {
            VLOG_DBG_RL(&rl, "port-id=%d: Failed to query ufid "UUID_FMT
                        " action %p. %d (%s)", ctx->port_id,
                        UUID_ARGS((struct uuid *) ufid), ctx->act_hdl,
                        error.type, error.message);
            goto out;
        }
    }

    offload->update_stats(&rte_flow_data->stats, attrs, &query);
    if (query.hits_set && query.hits) {
        rte_flow_data->stats.used = now;
    }
    memcpy(stats, &rte_flow_data->stats, sizeof *stats);
out:
    ovs_mutex_unlock(&rte_flow_data->lock);
    return ret;
}

static void
rte_aux_tables_uninit(struct netdev *netdev);

static int
flush_netdev_flows_in_related(struct netdev *netdev, struct netdev *related)
{
    unsigned int tid = netdev_offload_thread_id();
    struct cmap *map = offload_data_map(related);
    struct ufid_to_rte_flow_data *data;

    if (!map) {
        return -1;
    }

    CMAP_FOR_EACH (data, node, map) {
        if (data->netdev != netdev && data->physdev != netdev) {
            continue;
        }
        if (data->creation_tid == tid) {
            netdev_offload_dpdk_remove_flows(data, false);
        }
    }
    LIST_FOR_EACH_SAFE (data, list_node, &per_threads[tid].conn_list) {
        if (data->physdev != netdev) {
            continue;
        }
        netdev_offload_dpdk_remove_flows(data, false);
    }

    return 0;
}

/* Upon flushing the ESW manager, its members netdevs should be flushed too,
 * as their offloads are done on it.
 */
static bool
flush_esw_members_op(struct netdev *netdev,
                     odp_port_t odp_port OVS_UNUSED,
                     void *aux_)
{
    struct cmap *map = offload_data_map(netdev);
    struct esw_members_aux *aux = aux_;

    if (!cmap_is_empty(map)) {
        VLOG_ERR("Incomplete flush: %s should have been empty",
                 netdev_get_name(netdev));
    }

    if (flush_netdev_flows_in_related(netdev, netdev)) {
        aux->ret = -1;
        return true;
    }

    return false;
}

static bool
esw_members_cb(struct netdev *netdev,
               odp_port_t odp_port OVS_UNUSED,
               void *aux_)
{
    struct esw_members_aux *aux = aux_;
    struct netdev *esw_netdev;
    int netdev_esw_mgr_pid;
    int esw_mgr_pid;

    esw_netdev = aux->esw_netdev;
    esw_mgr_pid = netdev_dpdk_get_esw_mgr_port_id(aux->esw_netdev);

    /* Skip the ESW netdev itself. */
    if (netdev == esw_netdev) {
        return false;
    }

    netdev_esw_mgr_pid = netdev_dpdk_get_esw_mgr_port_id(netdev);

    /* Skip a non-member. */
    if (netdev_esw_mgr_pid == -1 || netdev_esw_mgr_pid != esw_mgr_pid) {
        return false;
    }

    return aux->op(netdev, odp_port, aux);
}

static bool
uninit_esw_members_op(struct netdev *netdev,
                      odp_port_t odp_port OVS_UNUSED,
                      void *aux_ OVS_UNUSED)
{
    netdev_uninit_flow_api(netdev);

    return false;
}

static bool
init_esw_members_op(struct netdev *netdev,
                    odp_port_t odp_port OVS_UNUSED,
                    void *aux_ OVS_UNUSED)
{
    netdev_init_flow_api(netdev);

    return false;
}

static bool
flush_in_vport_cb(struct netdev *vport,
                  odp_port_t odp_port OVS_UNUSED,
                  void *aux)
{
    struct netdev *netdev = aux;

    if (!netdev_dpdk_is_ethdev(vport)) {
        flush_netdev_flows_in_related(netdev, vport);
    }

    return false;
}

static int
netdev_offload_dpdk_flow_flush(struct netdev *netdev)
{
    struct esw_members_aux aux = {
        .esw_netdev = netdev,
        .ret = 0,
        .op = flush_esw_members_op,
    };

    per_thread_init();

    if (flush_netdev_flows_in_related(netdev, netdev)) {
        return -1;
    }

    if (netdev_dpdk_is_ethdev(netdev)) {
        /* If the flushed netdev is an ESW manager, flush its members too. */
        if (netdev_dpdk_is_esw_mgr(netdev)) {
            netdev_ports_traverse(netdev->dpif_type, esw_members_cb, &aux);
        }
        netdev_ports_traverse(netdev->dpif_type, flush_in_vport_cb, netdev);
    }

    return aux.ret;
}

struct get_vport_netdev_aux {
    struct rte_flow_tunnel *tunnel;
    odp_port_t *odp_port;
    struct netdev *vport;
    const char *type;
};

static bool
get_vport_netdev_cb(struct netdev *netdev,
                    odp_port_t odp_port,
                    void *aux_)
{
    const struct netdev_tunnel_config *tnl_cfg;
    struct get_vport_netdev_aux *aux = aux_;

    if (!aux->type || strcmp(netdev_get_type(netdev), aux->type)) {
        return false;
    }
    if (!strcmp(netdev_get_type(netdev), "gre")) {
        goto out;
    }

    tnl_cfg = netdev_get_tunnel_config(netdev);
    if (!tnl_cfg) {
        VLOG_ERR_RL(&rl, "Cannot get a tunnel config for netdev %s",
                    netdev_get_name(netdev));
        return false;
    }

    if (tnl_cfg->dst_port != aux->tunnel->tp_dst) {
        return false;
    }

out:
    /* Found the netdev. Store the results and stop the traversing. */
    aux->vport = netdev_ref(netdev);
    *aux->odp_port = odp_port;

    return true;
}

OVS_UNUSED
static struct netdev *
get_vport_netdev(const char *dpif_type,
                 struct rte_flow_tunnel *tunnel,
                 odp_port_t *odp_port)
{
    struct get_vport_netdev_aux aux = {
        .tunnel = tunnel,
        .odp_port = odp_port,
        .vport = NULL,
        .type = NULL,
    };

    if (tunnel->type == RTE_FLOW_ITEM_TYPE_VXLAN) {
        aux.type = "vxlan";
    } else if (tunnel->type == RTE_FLOW_ITEM_TYPE_GRE) {
        aux.type = "gre";
    }
    netdev_ports_traverse(dpif_type, get_vport_netdev_cb, &aux);

    return aux.vport;
}

#define PKT_DUMP_MAX_LEN    80

static void
log_packet_err(struct netdev *netdev, struct dp_packet *pkt, char *str)
{
    struct ds s;

    ds_init(&s);

    VLOG_ERR("%s: %s. %s", netdev_get_name(netdev), str,
             ds_cstr(dp_packet_ds_put_hex(&s, pkt, PKT_DUMP_MAX_LEN)));

    ds_destroy(&s);
}

static void
rte_get_packet_recovery_info(struct dp_packet *packet,
                             struct dpdk_offload_recovery_info *info)
{
    memset(info, 0, sizeof *info);
    if (dpdk_offload_get_reg_field(packet, REG_FIELD_FLOW_INFO,
                                   &info->flow_miss_id)) {
        dpdk_offload_get_reg_field(packet, REG_FIELD_CT_CTX,
                                   &info->ct_miss_id);
    } else {
        dpdk_offload_get_reg_field(packet, REG_FIELD_SFLOW_CTX,
                                   &info->sflow_id);
    }
}

static int
netdev_offload_dpdk_packet_hw_hash(struct netdev *netdev,
                                   struct dp_packet *packet,
                                   uint32_t seed,
                                   uint32_t *hash)
{
    return offload->packet_hw_hash(netdev, packet, seed, hash);
}

static int
netdev_offload_dpdk_packet_hw_entropy(struct netdev *netdev,
                                      struct dp_packet *packet,
                                      uint16_t *entropy)
{
    return offload->packet_hw_entropy(netdev, packet, entropy);
}

static int
netdev_offload_dpdk_hw_miss_packet_recover(struct netdev *netdev,
                                           struct dp_packet *packet,
                                           uint8_t *skip_actions,
                                           struct dpif_sflow_attr *sflow_attr)
{
    struct dpdk_offload_recovery_info info;
    struct flow_miss_ctx flow_miss_ctx;
    struct ct_miss_ctx ct_miss_ctx;
    struct sflow_ctx sflow_ctx;
    struct netdev *vport_netdev;

    offload->get_packet_recover_info(packet, &info);

    if (info.sflow_id) {
        /* Since sFlow does not work with CT, offloaded sampled packets
         * cannot have mark. If a packet without a mark reaches SW it
         * is either a sampled packet if a cookie is found or a datapath one.
         */
        if (find_sflow_ctx(info.sflow_id, &sflow_ctx)) {
            log_packet_err(netdev, packet, "sFlow id not found");
            return 0;
        }
        memcpy(sflow_attr->userdata, &sflow_ctx.cookie,
               sflow_ctx.sflow_attr.userdata_len);
        if (!is_all_zeros(&sflow_ctx.sflow_tnl, sizeof sflow_ctx.sflow_tnl)) {
            memcpy(sflow_attr->tunnel, &sflow_ctx.sflow_tnl,
                   sizeof *sflow_attr->tunnel);
        } else {
            sflow_attr->tunnel = NULL;
        }
        sflow_attr->sflow = sflow_ctx.sflow_attr.sflow;
        sflow_attr->sflow_len = sflow_ctx.sflow_attr.sflow_len;
        sflow_attr->userdata_len = sflow_ctx.sflow_attr.userdata_len;
        return EIO;
    }

    if (info.flow_miss_id) {
        if (find_flow_miss_ctx(info.flow_miss_id, &flow_miss_ctx)) {
            log_packet_err(netdev, packet, "flow miss ctx id not found");
            return 0;
        }
        *skip_actions = flow_miss_ctx.skip_actions;
        packet->md.recirc_id = flow_miss_ctx.recirc_id;
        if (flow_miss_ctx.has_dp_hash) {
            /* OVS will not match on dp-hash unless it has a non-zero value.
             * The 4 LSBs are matched, but zero value is valid. In order to
             * have a non-zero value in the dp-hash of the packet, set the 5th
             * bit.
             */
            packet->md.dp_hash = info.dp_hash | 0x10;
        }

        if (flow_miss_ctx.vport != ODPP_NONE) {
            vport_netdev = netdev_ports_get(flow_miss_ctx.vport,
                                            netdev->dpif_type);
            if (!vport_netdev) {
                return -1;
            }
            if (is_all_zeros(&flow_miss_ctx.tnl, sizeof flow_miss_ctx.tnl)) {
                if (vport_netdev->netdev_class->pop_header) {
                    parse_tcp_flags(packet, NULL, NULL, NULL);
                    if (!vport_netdev->netdev_class->pop_header(packet, false)) {
                        netdev_close(vport_netdev);
                        return -1;
                    }
                }
            } else {
                memcpy(&packet->md.tunnel, &flow_miss_ctx.tnl,
                       sizeof packet->md.tunnel);
            }
            packet->md.in_port.odp_port = flow_miss_ctx.vport;
            netdev_close(vport_netdev);
        }
    }

    if (info.ct_miss_id) {
        if (find_ct_miss_ctx(info.ct_miss_id, &ct_miss_ctx)) {
            log_packet_err(netdev, packet, "ct miss ctx id not found");
            return 0;
        }
        packet->md.ct_state = ct_miss_ctx.state;
        packet->md.ct_zone = ct_miss_ctx.zone;
        packet->md.ct_mark = ct_miss_ctx.mark;
        packet->md.ct_label = ct_miss_ctx.label;
    }

    return 0;
}

static int
netdev_offload_dpdk_get_n_flows(struct netdev *netdev,
                                uint64_t *n_flows)
{
    struct netdev_offload_dpdk_data *data;
    unsigned int tid;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    if (!data) {
        return -1;
    }

    for (tid = 0; tid < netdev_offload_thread_nb(); tid++) {
        n_flows[tid] = data->flow_counters[tid];
    }

    return 0;
}

static int
netdev_offload_dpdk_get_n_offloads(struct netdev *netdev,
                                   uint64_t *n_offloads)
{
    struct netdev_offload_dpdk_data *data;
    unsigned int tid;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    if (!data) {
        return -1;
    }

    for (tid = 0; tid < netdev_offload_thread_nb(); tid++) {
        n_offloads[tid] = data->offload_counters[tid];
    }

    return 0;
}

static int
netdev_offload_dpdk_get_stats(struct netdev *netdev,
                              struct netdev_offload_stats *stats)
{
    struct netdev_offload_dpdk_data *data;
    unsigned int tid;

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    if (!data) {
        return -1;
    }

    for (tid = 0; tid < netdev_offload_thread_nb(); tid++) {
        stats[tid].n_inserted = data->offload_counters[tid];
        stats[tid].n_flows = data->flow_counters[tid];
        stats[tid].n_conns = data->conn_counters[tid];
    }

    return 0;
}

static int
netdev_offload_dpdk_ct_counter_query(struct netdev *netdev,
                                     uintptr_t counter_key,
                                     long long now,
                                     long long prev_now,
                                     struct dpif_flow_stats *stats)
{
    struct rte_flow_query_age query_age;
    struct indirect_ctx *ctx;
    struct rte_flow_error error;
    int ret;

    memset(stats, 0, sizeof *stats);

    ctx = get_indirect_age_ctx(netdev, counter_key, false);
    if (ctx == NULL) {
        VLOG_ERR_RL(&rl, "Could not get shared age ctx for "
                    "counter_key=0x%"PRIxPTR, counter_key);
        return -1;
    }

    ret = netdev_dpdk_indirect_action_query(ctx->port_id, ctx->act_hdl,
                                            &query_age, &error);
    if (!ret && query_age.sec_since_last_hit_valid &&
        (query_age.sec_since_last_hit * 1000) <= (now - prev_now)) {
        stats->used = now;
    }
    return ret;
}

static void
fixed_rule_uninit(struct netdev *netdev, struct fixed_rule *fr, bool is_esw)
{
    if (!fr->doh.rte_flow) {
        return;
    }

    netdev_offload_dpdk_destroy_flow(netdev, &fr->doh, NULL, is_esw);
    fr->doh.rte_flow = NULL;
}

static void
ct_nat_miss_uninit(struct netdev *netdev, struct fixed_rule *fr)
{
    fixed_rule_uninit(netdev, fr, true);
}

static int
ct_nat_miss_init(struct netdev *netdev, struct fixed_rule *fr)
{
    if (add_miss_flow(netdev, CTNAT_TABLE_ID, CT_TABLE_ID, 0, 0, &fr->doh)) {
        return -1;
    }
    return 0;
}

static void
ct_zones_uninit(struct netdev *netdev, struct netdev_offload_dpdk_data *data)
{
    struct fixed_rule *fr;
    uint32_t zone_id;
    int nat, i;

    if (netdev_offload_dpdk_disable_zone_tables) {
        return;
    }

    for (nat = 0; nat < 2; nat++) {
        for (i = 0; i < 2; i++) {
            for (zone_id = MIN_ZONE_ID; zone_id <= MAX_ZONE_ID; zone_id++) {
                fr = &data->zone_flows[nat][i][zone_id];

                fixed_rule_uninit(netdev, fr, true);
            }
        }
    }
}

static int
ct_zones_init(struct netdev *netdev, struct netdev_offload_dpdk_data *data)
{
    struct rte_flow_action_set_tag set_tag;
    struct rte_flow_item_port_id port_id;
    struct rte_flow_item_tag tag_spec;
    struct rte_flow_item_tag tag_mask;
    struct rte_flow_action_jump jump;
    struct rte_flow_attr attr = {
        .transfer = 1,
    };
    struct flow_patterns patterns = {
        .items = (struct rte_flow_item []) {
            { .type = RTE_FLOW_ITEM_TYPE_PORT_ID, .spec = &port_id, },
            { .type = RTE_FLOW_ITEM_TYPE_ETH, },
            { .type = RTE_FLOW_ITEM_TYPE_TAG, .spec = &tag_spec,
              .mask = &tag_mask },
            { .type = RTE_FLOW_ITEM_TYPE_END, },
        },
        .cnt = 4,
    };
    struct dpdk_offload_handle doh;
    struct flow_actions actions = {
        .actions = (struct rte_flow_action []) {
            { .type = RTE_FLOW_ACTION_TYPE_SET_TAG, .conf = &set_tag, },
            { .type = RTE_FLOW_ACTION_TYPE_JUMP, .conf = &jump, },
            { .type = RTE_FLOW_ACTION_TYPE_END, .conf = NULL, },
        },
        .cnt = 3,
    };
    struct reg_field *reg_field;
    struct rte_flow_error error;
    struct fixed_rule *fr;
    uint32_t base_group;
    uint32_t zone_id;
    int nat;

    if (netdev_offload_dpdk_disable_zone_tables) {
        return 0;
    }

    memset(&set_tag, 0, sizeof(set_tag));
    memset(&jump, 0, sizeof(jump));
    memset(&port_id, 0, sizeof(port_id));
    memset(&tag_spec, 0, sizeof(tag_spec));
    memset(&tag_mask, 0, sizeof(tag_mask));

    port_id.id = netdev_dpdk_get_port_id(netdev);

    /* Merge the tag match for zone and state only if they are
     * at the same index. */
    ovs_assert(offload->reg_fields()[REG_FIELD_CT_ZONE].index == offload->reg_fields()[REG_FIELD_CT_STATE].index);

    for (nat = 0; nat < 2; nat++) {
        base_group = nat ? CTNAT_TABLE_ID : CT_TABLE_ID;

        for (zone_id = MIN_ZONE_ID; zone_id <= MAX_ZONE_ID; zone_id++) {
            uint32_t ct_zone_spec, ct_zone_mask;
            uint32_t ct_state_spec, ct_state_mask;

            attr.group = base_group + zone_id;
            jump.group = base_group;

            fr = &data->zone_flows[nat][0][zone_id];
            attr.priority = 0;
            /* If the zone is the same, and already visited ct/ct-nat, skip
             * ct/ct-nat and jump directly to post-ct.
             */
            reg_field = &offload->reg_fields()[REG_FIELD_CT_ZONE];
            ct_zone_spec = zone_id << reg_field->offset;
            ct_zone_mask = reg_field->mask << reg_field->offset;
            reg_field = &offload->reg_fields()[REG_FIELD_CT_STATE];
            ct_state_spec = OVS_CS_F_TRACKED;
            if (nat) {
                ct_state_spec |= OVS_CS_F_NAT_MASK;
            }
            ct_state_spec <<= reg_field->offset;
            ct_state_mask = ct_state_spec;

            /* Merge ct_zone and ct_state matches in a single item. */
            tag_spec.index = reg_field->index;
            tag_spec.data = ct_zone_spec | ct_state_spec;
            tag_mask.index = 0xFF;
            tag_mask.data = ct_zone_mask | ct_state_mask;
            patterns.items[2].type = RTE_FLOW_ITEM_TYPE_TAG;
            actions.actions[0].type = RTE_FLOW_ACTION_TYPE_VOID;
            jump.group = POSTCT_TABLE_ID;
            if (create_rte_flow(netdev, &attr, &patterns, &actions, &doh,
                                &error)) {
                goto err;
            }
            fr->doh.rte_flow = doh.rte_flow;

            fr = &data->zone_flows[nat][1][zone_id];
            attr.priority = 1;
            /* Otherwise, set the zone and go to CT/CT-NAT. */
            reg_field = &offload->reg_fields()[REG_FIELD_CT_STATE];
            tag_spec.index = reg_field->index;
            tag_spec.data = 0;
            tag_mask.index = 0xFF;
            tag_mask.data = reg_field->mask << reg_field->offset;
            patterns.items[2].type = RTE_FLOW_ITEM_TYPE_VOID;
            reg_field = &offload->reg_fields()[REG_FIELD_CT_ZONE];
            set_tag.index = reg_field->index;
            set_tag.data = zone_id << reg_field->offset;
            set_tag.mask = reg_field->mask << reg_field->offset;
            actions.actions[0].type = RTE_FLOW_ACTION_TYPE_SET_TAG;
            jump.group = base_group;
            if (create_rte_flow(netdev, &attr, &patterns, &actions, &doh,
                                &error)) {
                goto err;
            }
            fr->doh.rte_flow = doh.rte_flow;
        }
    }

    return 0;

err:
    ct_zones_uninit(netdev, data);
    return -1;
}

static void
hairpin_uninit(struct netdev *netdev, struct fixed_rule *fr)
{
    fixed_rule_uninit(netdev, fr, false);
}

static int
hairpin_init(struct netdev *netdev, struct fixed_rule *fr)
{
    struct rte_flow_attr attr = { .ingress = 1, };
    struct rte_flow_item_mark hp_mark;
    struct flow_patterns patterns = {
        .items = (struct rte_flow_item []) {
            { .type = OVS_RTE_FLOW_ITEM_TYPE(FLOW_INFO), .spec = &hp_mark, },
            { .type = RTE_FLOW_ITEM_TYPE_END, },
        },
        .cnt = 2,
    };
    struct rte_flow_action_queue hp_queue;
    struct dpdk_offload_handle doh;
    struct flow_actions actions = {
        .actions = (struct rte_flow_action []) {
            { .type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &hp_queue, },
            { .type = RTE_FLOW_ACTION_TYPE_END, },
        },
        .cnt = 2,
    };
    struct rte_flow_error error;

    hp_mark.id = HAIRPIN_FLOW_MARK;
    hp_queue.index = netdev->n_rxq;

    if (create_rte_flow(netdev, &attr, &patterns, &actions, &doh, &error)) {
        return -1;
    }
    fr->doh.rte_flow = doh.rte_flow;
    return 0;
}

static void
rte_aux_tables_uninit(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;

    if (!netdev_dpdk_is_ethdev(netdev)) {
        return;
    }

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);

    ct_nat_miss_uninit(netdev, &data->ct_nat_miss);
    ct_zones_uninit(netdev, data);
    hairpin_uninit(netdev, &data->hairpin);
}

static int
rte_aux_tables_init(struct netdev *netdev)
{
    struct netdev_offload_dpdk_data *data;
    int ret = 0;

    if (!netdev_dpdk_is_ethdev(netdev)) {
        return 0;
    }

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);

    ret = ct_nat_miss_init(netdev, &data->ct_nat_miss);
    if (!ret) {
        ret = ct_zones_init(netdev, data);
    }
    if (!ret) {
        ret = hairpin_init(netdev, &data->hairpin);
    }
    if (ret) {
        VLOG_WARN("Cannot apply init flows for netdev %s",
                  netdev_get_name(netdev));
        rte_aux_tables_uninit(netdev);
    }
    return ret;
}

static int
conn_build_patterns(struct netdev *netdev,
                    const struct ct_match *ct_match,
                    struct flow_patterns *patterns,
                    uint32_t ct_match_zone_id,
                    struct act_vars *act_vars)
{
    struct rte_flow_item_port_id *port_id_spec;
    uint8_t proto = 0;

    act_vars->ct_mode = CT_MODE_CT_CONN;
    act_vars->is_ct_conn = true;

    patterns->physdev = netdev;

    port_id_spec = per_thread_xzalloc(sizeof *port_id_spec);
    port_id_spec->id = netdev_dpdk_get_port_id(patterns->physdev);
    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_PORT_ID, port_id_spec,
                     NULL, NULL);

    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_ETH, NULL, NULL, NULL);

    /* IPv4 */
    if (ct_match->key.dl_type == htons(ETH_TYPE_IP)) {
        struct rte_flow_item_ipv4 *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *spec);

        spec->hdr.src_addr = ct_match->key.src.addr.ipv4;
        spec->hdr.dst_addr = ct_match->key.dst.addr.ipv4;
        spec->hdr.next_proto_id = ct_match->key.nw_proto;

        mask->hdr.src_addr = OVS_BE32_MAX;
        mask->hdr.dst_addr = OVS_BE32_MAX;
        mask->hdr.next_proto_id = UINT8_MAX;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_IPV4, spec, mask, NULL);

        /* Save proto for L4 protocol setup. */
        proto = spec->hdr.next_proto_id;
    }

    /* IPv6 */
    if (ct_match->key.dl_type == htons(ETH_TYPE_IPV6)) {
        struct rte_flow_item_ipv6 *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        memcpy(spec->hdr.src_addr, &ct_match->key.src.addr.ipv6,
               sizeof spec->hdr.src_addr);
        memcpy(spec->hdr.dst_addr, &ct_match->key.dst.addr.ipv6,
               sizeof spec->hdr.dst_addr);
        spec->hdr.proto = ct_match->key.nw_proto;

        memset(&mask->hdr.src_addr, 0xff, sizeof mask->hdr.src_addr);
        memset(&mask->hdr.dst_addr, 0xff, sizeof mask->hdr.dst_addr);
        mask->hdr.proto = UINT8_MAX;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_IPV6, spec, mask, NULL);

        /* Save proto for L4 protocol setup. */
        proto = spec->hdr.proto;
    }

    if (proto == IPPROTO_TCP) {
        struct rte_flow_item_tcp *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.src_port  = ct_match->key.src.port;
        spec->hdr.dst_port  = ct_match->key.dst.port;

        mask->hdr.src_port  = OVS_BE16_MAX;
        mask->hdr.dst_port  = OVS_BE16_MAX;
        /* Any segment syn, rst or fin must miss and go to SW to trigger
         * a connection state change. */
        mask->hdr.tcp_flags = (TCP_SYN | TCP_RST | TCP_FIN) & 0xff;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_TCP, spec, mask, NULL);
    } else if (proto == IPPROTO_UDP) {
        struct rte_flow_item_udp *spec, *mask;

        spec = per_thread_xzalloc(sizeof *spec);
        mask = per_thread_xzalloc(sizeof *mask);

        spec->hdr.src_port  = ct_match->key.src.port;
        spec->hdr.dst_port  = ct_match->key.dst.port;

        mask->hdr.src_port  = OVS_BE16_MAX;
        mask->hdr.dst_port  = OVS_BE16_MAX;

        add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_UDP, spec, mask, NULL);
    } else {
        VLOG_ERR_RL(&rl, "Unsupported L4 protocol: 0x02%" PRIx8, proto);
        return -1;
    }

    if (add_pattern_match_reg_field(patterns, REG_FIELD_CT_ZONE,
                                    ct_match_zone_id,
                                    offload->reg_fields()[REG_FIELD_CT_ZONE].mask)) {
        VLOG_ERR_RL(&rl, "Failed to add the CT zone %"PRIu16" register match",
                    ct_match->key.zone);
        return -1;
    }

    add_flow_pattern(patterns, RTE_FLOW_ITEM_TYPE_END, NULL, NULL, NULL);
    return 0;
}

static int
conn_build_actions(struct ct_flow_offload_item ct_offload[1],
                   struct flow_actions *actions,
                   uint32_t ct_action_label_id,
                   struct indirect_ctx *shared_count_ctx,
                   uint32_t ct_miss_ctx_id)
{
    struct rte_flow_action_set_meta *set_meta;
    struct rte_flow_action *ia;
    size_t size;

    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_INDIRECT, NULL);
    actions->shared_count_action_pos = actions->cnt - 1;

    /* NAT */
    if (ct_offload->nat.mod_flags) {
        if (ct_offload->ct_match.key.dl_type == htons(ETH_TYPE_IP)) {
            /* IPv4 */
            size = sizeof ct_offload->nat.key.src.addr.ipv4;
            if (ct_offload->nat.mod_flags & NAT_ACTION_SRC) {
                add_full_set_action(actions,
                    RTE_FLOW_ACTION_TYPE_SET_IPV4_SRC,
                    &ct_offload->nat.key.src.addr.ipv4, size);
            }
            if (ct_offload->nat.mod_flags & NAT_ACTION_DST) {
                add_full_set_action(actions,
                    RTE_FLOW_ACTION_TYPE_SET_IPV4_DST,
                    &ct_offload->nat.key.dst.addr.ipv4, size);
            }
        } else {
            /* IPv6 */
            size = sizeof ct_offload->nat.key.src.addr.ipv6;
            if (ct_offload->nat.mod_flags & NAT_ACTION_SRC) {
                add_full_set_action(actions,
                    RTE_FLOW_ACTION_TYPE_SET_IPV6_SRC,
                    &ct_offload->nat.key.src.addr.ipv6, size);
            }
            if (ct_offload->nat.mod_flags & NAT_ACTION_DST) {
                add_full_set_action(actions,
                    RTE_FLOW_ACTION_TYPE_SET_IPV6_DST,
                    &ct_offload->nat.key.dst.addr.ipv6, size);
            }
        }
        /* TCP | UDP */
        if (ct_offload->nat.mod_flags & NAT_ACTION_SRC_PORT ||
            ct_offload->nat.mod_flags & NAT_ACTION_DST_PORT) {
            size = sizeof ct_offload->nat.key.src.port;
            if (ct_offload->nat.mod_flags & NAT_ACTION_SRC_PORT) {
                add_full_set_action(actions,
                    RTE_FLOW_ACTION_TYPE_SET_TP_SRC,
                    &ct_offload->nat.key.src.port, size);
            }
            if (ct_offload->nat.mod_flags & NAT_ACTION_DST_PORT) {
                add_full_set_action(actions,
                    RTE_FLOW_ACTION_TYPE_SET_TP_DST,
                    &ct_offload->nat.key.dst.port, size);
            }
        }
    }

    /* CT MARK */
    add_action_set_reg_field(actions, REG_FIELD_CT_MARK,
                             ct_offload->mark_key,
                             offload->reg_fields()[REG_FIELD_CT_MARK].mask);

    /* CT LABEL */
    add_action_set_reg_field(actions, REG_FIELD_CT_LABEL_ID,
                             ct_action_label_id,
                             offload->reg_fields()[REG_FIELD_CT_LABEL_ID].mask);

    /* CT STATE */
    add_action_set_reg_field(actions, REG_FIELD_CT_STATE,
                             ct_offload->ct_state, 0xFF);

    /* Shared counter. */
    ia = &actions->actions[actions->shared_count_action_pos];
    ovs_assert(ia->type == RTE_FLOW_ACTION_TYPE_INDIRECT &&
               ia->conf == NULL);
    ia->conf = shared_count_ctx->act_hdl;

    set_meta = per_thread_xzalloc(sizeof *set_meta);
    set_meta->data = ct_miss_ctx_id;
    add_flow_action(actions, OVS_RTE_FLOW_ACTION_TYPE(CT_INFO), set_meta);

    /* Last CT action is to go to Post-CT. */
    add_jump_action(actions, POSTCT_TABLE_ID);
    add_flow_action(actions, RTE_FLOW_ACTION_TYPE_END, NULL);

    return 0;
}

static int
dpdk_offload_insert_conn_rte(struct netdev *netdev,
                             struct ct_flow_offload_item ct_offload[1],
                             uint32_t ct_match_zone_id,
                             uint32_t ct_action_label_id,
                             struct indirect_ctx *shared_count_ctx,
                             uint32_t ct_miss_ctx_id,
                             struct flow_item *fi)
{
    struct flow_patterns patterns = {
        .items = NULL,
        .cnt = 0,
        .s_tnl = DS_EMPTY_INITIALIZER,
    };
    struct flow_actions actions = {
        .actions = NULL,
        .cnt = 0,
        .s_tnl = DS_EMPTY_INITIALIZER,
    };
    struct rte_flow_attr flow_attr;
    struct rte_flow_error error;
    struct act_vars act_vars;
    int ret;

    memset(&act_vars, 0, sizeof act_vars);
    act_vars.vport = ODPP_NONE;

    ret = conn_build_patterns(netdev, &ct_offload->ct_match, &patterns,
                              ct_match_zone_id, &act_vars);
    if (ret) {
        goto free_patterns;
    }

    ret = conn_build_actions(ct_offload, &actions, ct_action_label_id,
                             shared_count_ctx, ct_miss_ctx_id);
    if (ret) {
        goto free_actions;
    }

    memset(&flow_attr, 0, sizeof flow_attr);
    flow_attr.transfer = 1;

    ret = netdev_offload_dpdk_flow_create(netdev, &flow_attr, &patterns,
                                          &actions, &error, NULL, &act_vars,
                                          fi);

free_actions:
    free_flow_actions(&actions, true);
free_patterns:
    free_flow_patterns(&patterns);

    return ret;
}

static int
conn_get_resources(struct netdev *netdev,
                   struct ct_flow_offload_item ct_offload[1],
                   struct act_resources *act_resources)
{
    struct flows_counter_key counter_id_key;
    struct ct_miss_ctx miss_ctx;
    struct indirect_ctx *ctx;

    memset(&miss_ctx, 0, sizeof miss_ctx);

    /* CT ZONE */
    if (get_zone_id(ct_offload->ct_match.key.zone,
                    &act_resources->ct_match_zone_id)) {
        VLOG_ERR_RL(&rl, "Unable to find the offload zone id mapped to the "
                    "CT zone %" PRIu16, ct_offload->ct_match.key.zone);
        return -1;
    }
    miss_ctx.zone = ct_offload->ct_match.key.zone;


    /* CT MARK */
    miss_ctx.mark = ct_offload->mark_key;

    /* CT LABEL */
    if (netdev_offload_dpdk_ct_labels_mapping) {
        if (get_label_id(&ct_offload->label_key,
                         &act_resources->ct_action_label_id)) {
            VLOG_ERR_RL(&rl, "Failed to generate a label ID");
            return -1;
        }
    }
    memcpy(&miss_ctx.label, &ct_offload->label_key, sizeof miss_ctx.label);

    miss_ctx.state = ct_offload->ct_state;

    if (get_ct_ctx_id(&miss_ctx, &act_resources->ct_miss_ctx_id)) {
        VLOG_ERR("Could not get a CT context ID");
        return -1;
    }

    /* Shared counter. */
    memset(&counter_id_key, 0, sizeof counter_id_key);
    counter_id_key.ptr_key = ct_offload->ctid_key;

    ctx = get_indirect_count_ctx(netdev, &counter_id_key,
                                 OVS_SHARED_CT_COUNT, true);
    if (!ctx) {
        return -1;
    }
    act_resources->shared_count_ctx = ctx;

    put_table_id(act_resources->self_table_id);
    act_resources->self_table_id = 0;

    return 0;
}

static int
netdev_offload_dpdk_conn_add(struct netdev *netdev,
                             struct ct_flow_offload_item ct_offload[1])
{
    struct act_resources act_resources = { .flow_id = INVALID_FLOW_MARK, };
    unsigned int tid = netdev_offload_thread_id();
    struct ufid_to_rte_flow_data *rte_flow_data;
    struct netdev_offload_dpdk_data *data;
    uint32_t ct_action_label_id;

    if (!netdev_dpdk_is_ethdev(netdev)) {
        return ENOTSUP;
    }

    rte_flow_data = ct_offload->offload_data;
    if (!rte_flow_data) {
        return EINVAL;
    }
    if (rte_flow_data->flow_item.doh[0].rte_flow) {
        /* Conn offload modification is not supported. */
        return EEXIST;
    }

    per_thread_init();

    if (conn_get_resources(netdev, ct_offload, &act_resources)) {
        return EINVAL;
    }

    if (netdev_offload_dpdk_ct_labels_mapping) {
        ct_action_label_id = act_resources.ct_action_label_id;
    } else {
        ct_action_label_id = ct_offload->label_key.u32[0];
    }

    if (offload->insert_conn(netdev, ct_offload,
                             act_resources.ct_match_zone_id,
                             ct_action_label_id,
                             act_resources.shared_count_ctx,
                             act_resources.ct_miss_ctx_id,
                             &rte_flow_data->flow_item)) {
        put_action_resources(&act_resources);
        return EINVAL;
    }

    data = (struct netdev_offload_dpdk_data *)
        ovsrcu_get(void *, &netdev->hw_info.offload_data);
    data->conn_counters[tid]++;

    conn_link(netdev, rte_flow_data, &act_resources);

    /* Mark this conn as offloaded by exposing the offload data to the
     * conntrack module. */
    ct_dir_info_data_set(ct_offload->conn_dir_info, ct_offload->offload_data);

    return 0;
}

static int
netdev_offload_dpdk_conn_del(struct netdev *netdev OVS_UNUSED,
                             struct ct_flow_offload_item ct_offload[1])
{
    struct ufid_to_rte_flow_data *rte_flow_data;

    rte_flow_data = ct_offload->offload_data;
    if (!rte_flow_data || !rte_flow_data->flow_item.doh[0].valid) {
        return ENODATA;
    }

    return netdev_offload_dpdk_remove_flows(rte_flow_data, false);
}

static int
netdev_offload_dpdk_conn_stats(struct netdev *netdev,
                               struct ct_flow_offload_item ct_offload[1],
                               struct dpif_flow_stats *stats,
                               struct dpif_flow_attrs *attrs,
                               long long int now)
{
    struct rte_flow_query_count query = { .reset = 1 };
    struct ufid_to_rte_flow_data *rte_flow_data;
    const ovs_u128 *ufid = &ct_offload->ufid;
    struct dpdk_offload_handle *doh;
    struct rte_flow_error error;
    struct indirect_ctx *ctx;
    int ret = 0;

    rte_flow_data = ct_offload->offload_data;
    if (!rte_flow_data || !rte_flow_data->flow_item.doh[0].rte_flow ||
        !rte_flow_data_active(rte_flow_data) || ovs_mutex_trylock(&rte_flow_data->lock)) {
        return ENODATA;
    }

    /* Check again whether the data is dead, as it could have been
     * updated while the lock was not yet taken. The first check above
     * was only to avoid unnecessary locking if possible.
     */
    if (!rte_flow_data_active(rte_flow_data)) {
        ret = ENODATA;
        goto out;
    }

    if (attrs) {
        attrs->dp_extra_info = NULL;
        attrs->offloaded = true;
        attrs->dp_layer = "dpdk";
    }

    doh = rte_flow_data->flow_item.doh[1].rte_flow
        ? &rte_flow_data->flow_item.doh[1]
        : &rte_flow_data->flow_item.doh[0];

    if (!rte_flow_data->act_resources.shared_count_ctx) {
        ret = offload->query_count(rte_flow_data->physdev, doh, &query, &error);
    } else {
        ctx = rte_flow_data->act_resources.shared_count_ctx;
        ret = offload->shared_query(ctx, &query, &error);
    }
    if (ret) {
        VLOG_DBG_RL(&rl, "%s: Failed to query ufid "UUID_FMT" flow: %p",
                    netdev_get_name(netdev), UUID_ARGS((struct uuid *) ufid),
                    doh->rte_flow);
        goto out;
    }
    offload->update_stats(&rte_flow_data->stats, attrs, &query);
    if (query.hits_set && query.hits) {
        rte_flow_data->stats.used = now;
    }

    if (stats) {
        memcpy(stats, &rte_flow_data->stats, sizeof *stats);
    }
out:
    ovs_mutex_unlock(&rte_flow_data->lock);
    return ret;
}

static int
dpdk_offload_rte_destroy(struct netdev *netdev,
                         struct dpdk_offload_handle *doh,
                         struct rte_flow_error *error,
                         bool esw_port_id)
{
    int ret;

    ret = netdev_dpdk_rte_flow_destroy(netdev, doh->rte_flow, error,
                                       esw_port_id);
    if (ret == 0) {
        dpdk_offload_counter_dec(netdev);
    }

    return ret;
}

static int
dpdk_offload_rte_query_count(struct netdev *netdev,
                             struct dpdk_offload_handle *doh,
                             struct rte_flow_query_count *query,
                             struct rte_flow_error *error)
{
    return netdev_dpdk_rte_flow_query_count(netdev, doh->rte_flow, query,
                                            error);
}

static int
dpdk_offload_rte_shared_create(struct netdev *netdev,
                               struct indirect_ctx *ctx,
                               const struct rte_flow_action *action,
                               struct rte_flow_error *error)
{
    ovs_assert(ctx->act_hdl == NULL);

    ctx->act_hdl = netdev_dpdk_indirect_action_create(netdev, action, error);
    if (!ctx->act_hdl) {
        VLOG_DBG("%s: netdev_dpdk_indirect_action_create failed: %d (%s)",
                 netdev_get_name(netdev), error->type, error->message);
        return -1;
    }

    return 0;
}

static int
dpdk_offload_rte_shared_destroy(struct indirect_ctx *ctx,
                                struct rte_flow_error *error)
{
    int ret;

    if (!ctx->act_hdl) {
        return 0;
    }

    ret = netdev_dpdk_indirect_action_destroy(ctx->port_id, ctx->act_hdl, error);
    if (ret) {
        return ret;
    }

    ctx->act_hdl = NULL;

    return 0;
}

static int
dpdk_offload_rte_shared_query(struct indirect_ctx *ctx, void *data,
                              struct rte_flow_error *error)
{
    return netdev_dpdk_indirect_action_query(ctx->port_id, ctx->act_hdl, data,
                                             error);
}

static int
dpdk_offload_rte_packet_hw_hash(struct netdev *netdev OVS_UNUSED,
                                struct dp_packet *packet OVS_UNUSED,
                                uint32_t seed OVS_UNUSED,
                                uint32_t *hash OVS_UNUSED)
{
    return -1;
}

static int
dpdk_offload_rte_packet_hw_entropy(struct netdev *netdev OVS_UNUSED,
                                   struct dp_packet *packet OVS_UNUSED,
                                   uint16_t *hash OVS_UNUSED)
{
    return -1;
}

struct dpdk_offload_api dpdk_offload_api_rte = {
    .create = dpdk_offload_rte_create,
    .destroy = dpdk_offload_rte_destroy,
    .query_count = dpdk_offload_rte_query_count,
    .shared_create = dpdk_offload_rte_shared_create,
    .shared_destroy = dpdk_offload_rte_shared_destroy,
    .shared_query = dpdk_offload_rte_shared_query,
    .get_packet_recover_info = rte_get_packet_recovery_info,
    .insert_conn = dpdk_offload_insert_conn_rte,
    .reg_fields = rte_get_reg_fields,
    .update_stats = netdev_offload_dpdk_update_stats,
    .aux_tables_init = rte_aux_tables_init,
    .aux_tables_uninit = rte_aux_tables_uninit,
    .packet_hw_hash = dpdk_offload_rte_packet_hw_hash,
    .packet_hw_entropy = dpdk_offload_rte_packet_hw_entropy,
};

const struct netdev_flow_api netdev_offload_dpdk = {
    .type = "dpdk_flow_api",
    .flow_put = netdev_offload_dpdk_flow_put,
    .flow_del = netdev_offload_dpdk_flow_del,
    .init_flow_api = netdev_offload_dpdk_init_flow_api,
    .uninit_flow_api = netdev_offload_dpdk_uninit_flow_api,
    .flow_get = netdev_offload_dpdk_flow_get,
    .flow_flush = netdev_offload_dpdk_flow_flush,
    .hw_miss_packet_recover = netdev_offload_dpdk_hw_miss_packet_recover,
    .flow_get_n_flows = netdev_offload_dpdk_get_n_flows,
    .flow_get_n_offloads = netdev_offload_dpdk_get_n_offloads,
    .get_stats = netdev_offload_dpdk_get_stats,
    .ct_counter_query = netdev_offload_dpdk_ct_counter_query,
    .conn_add = netdev_offload_dpdk_conn_add,
    .conn_del = netdev_offload_dpdk_conn_del,
    .conn_stats = netdev_offload_dpdk_conn_stats,
    .upkeep = netdev_offload_dpdk_upkeep,
    .packet_hw_hash = netdev_offload_dpdk_packet_hw_hash,
    .packet_hw_entropy = netdev_offload_dpdk_packet_hw_entropy,
};
