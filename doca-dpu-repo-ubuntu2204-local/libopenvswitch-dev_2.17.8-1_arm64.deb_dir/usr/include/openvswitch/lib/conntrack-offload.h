/*
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

#ifndef CONNTRACK_OFFLOAD_H
#define CONNTRACK_OFFLOAD_H

#include "conntrack.h"
#include "openvswitch/types.h"
#include "smap.h"

enum ct_timeout;
struct conn;
struct conntrack;
struct conntrack_offload_class;
struct dp_packet;
struct netdev;
struct ct_offload_handle;

enum ct_offload_flag {
    CT_OFFLOAD_NONE = 0,
    CT_OFFLOAD_INIT = 0x1 << 0,
    CT_OFFLOAD_REP  = 0x1 << 1,
    CT_OFFLOAD_SKIP = 0x1 << 2,
    CT_OFFLOAD_BOTH = (CT_OFFLOAD_INIT | CT_OFFLOAD_REP),
    CT_OFFLOAD_TERMINATED = 0x1 << 3,
};

struct ct_dir_info {
    odp_port_t port;
    ovs_u128 ufid;
    void *dp;
    uint8_t pkt_ct_state;
    uint32_t pkt_ct_mark;
    ovs_u128 pkt_ct_label;
    bool e2e_flow;
    uint8_t e2e_seen_pkts;
    /* This pointer is atomic as it is sync
     * between PMD, ct_clean and hw_offload threads. */
    OVSRCU_TYPE(void *) offload_data;
};

static inline void *
ct_dir_info_data_get(const struct ct_dir_info *info)
{
    return ovsrcu_get(void *, &info->offload_data);
}

static inline void
ct_dir_info_data_set(struct ct_dir_info *info, void *data)
{
    ovsrcu_set(&info->offload_data, data);
}

struct ct_offloads {
    uint8_t flags;
    struct ct_offload_handle *coh;
    struct ct_dir_info dir_info[CT_DIR_NUM];
};

struct ct_flow_offload_item {
    int  op;
    ovs_u128 ufid;
    void *dp;
    uintptr_t ctid_key;
    long long int timestamp;

    /* matches */
    struct ct_match ct_match;

    /* actions */
    uint8_t ct_state;
    ovs_u128 label_key;
    uint32_t mark_key;

    /* Pre-created CT actions */
    bool ct_actions_set;
    struct nlattr *actions;
    size_t actions_size;

    struct {
        uint8_t mod_flags;
        struct conn_key  key;
    } nat;

    /* refcnt is used to handle a scenario in which a connection issued an
     * offload request and was removed before the offload request is processed.
     */
    struct ovs_refcount *refcnt;
    struct ct_dir_info *conn_dir_info;
    void *offload_data;
};

/* hw-offload callbacks */
struct conntrack_offload_class {
    void (*conn_get_ufid)(ovs_u128 *);
    void (*conn_add)(struct ct_flow_offload_item *);
    void (*conn_del)(struct ct_flow_offload_item *);
    int (*conn_active)(struct ct_flow_offload_item *, long long now,
                       long long prev_now);
    void (*conn_e2e_add)(struct ct_flow_offload_item *);
    void (*conn_e2e_del)(ovs_u128 *, void *dp, long long int now);
    bool (*queue_full)(void);
};

void
process_one_ct_offload(struct conntrack *ct,
                       struct dp_packet *packet,
                       struct conn *conn,
                       bool reply,
                       long long now_us);
int
conn_hw_update(struct conntrack *ct,
               struct conntrack_offload_class *offload_class,
               struct conn *conn,
               enum ct_timeout *ptm,
               long long now);
void
conntrack_set_offload_class(struct conntrack *,
                            struct conntrack_offload_class *);
void
conntrack_offload_del_conn(struct conntrack *ct,
                           struct conn *conn,
                           bool flush);

unsigned int conntrack_offload_size(void);
bool conntrack_offload_is_enabled(void);
bool conntrack_offload_ipv6_is_enabled(void);
void conntrack_offload_config(const struct smap *other_config);
void
conntrack_offload_netdev_flush(struct conntrack *ct, struct netdev *netdev);

#endif /* CONNTRACK_OFFLOAD_H */
