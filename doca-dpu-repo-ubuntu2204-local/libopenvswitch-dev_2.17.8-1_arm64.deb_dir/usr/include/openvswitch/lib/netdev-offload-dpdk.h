/*
 * Copyright (c) 2023 NVIDIA Corporation.
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NETDEV_OFFLOAD_DPDK_H
#define NETDEV_OFFLOAD_DPDK_H

#include <sys/types.h>

#include "cmap.h"
#include "conntrack.h"
#include "dpdk-offload-provider.h"
#include "dpif.h"
#include "openvswitch/list.h"
#include "ovs-atomic.h"
#include "ovs-rcu.h"

struct act_resources {
    uint32_t next_table_id;
    uint32_t self_table_id;
    uint32_t flow_miss_ctx_id;
    uint32_t tnl_id;
    uint32_t flow_id;
    bool associated_flow_id;
    uint32_t ct_miss_ctx_id;
    uint32_t ct_match_zone_id;
    uint32_t ct_action_zone_id;
    uint32_t ct_match_label_id;
    uint32_t ct_action_label_id;
    struct indirect_ctx *shared_age_ctx;
    struct indirect_ctx *shared_count_ctx;
    uint32_t sflow_id;
    uint32_t meter_ids[DPDK_OFFLOAD_MAX_METERS_PER_FLOW];
};

struct ufid_to_rte_flow_data {
    union {
        struct cmap_node node;
        struct ovs_list list_node;
    };
    ovs_u128 ufid;
    struct netdev *netdev;
    struct flow_item flow_item;
    struct dpif_flow_stats stats;
    struct netdev *physdev;
    struct ovs_mutex lock;
    unsigned int creation_tid;
    struct ovsrcu_gc_node gc_node;
    atomic_bool active;
    struct act_resources act_resources;
};

static inline bool
rte_flow_data_active(struct ufid_to_rte_flow_data *data)
{
    bool active;

    if (!data) {
        return false;
    }
    atomic_read(&data->active, &active);
    return active;
}

static inline void
rte_flow_data_active_set(struct ufid_to_rte_flow_data *data, bool state)
{
    atomic_store(&data->active, state);
}

struct ct_offload_handle {
    struct ovs_refcount refcnt;
    struct ufid_to_rte_flow_data dir[CT_DIR_NUM];
};

#endif /* NETDEV_OFFLOAD_DPDK_H */
