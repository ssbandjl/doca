/*
 * Copyright (c) 2010, 2011, 2014 Nicira, Inc.
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DPIF_NETLINK_H
#define DPIF_NETLINK_H 1

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "flow.h"

struct ofpbuf;

struct dpif_netlink_vport {
    /* Generic Netlink header. */
    uint8_t cmd;

    /* ovs_vport header. */
    int dp_ifindex;
    int netnsid;                           /* Network Namespace ID. */
    odp_port_t port_no;                    /* ODPP_NONE if unknown. */
    enum ovs_vport_type type;

    /* Attributes.
     *
     * The 'stats' member points to 64-bit data that might only be aligned on
     * 32-bit boundaries, so use get_unaligned_u64() to access its values.
     */
    const char *name;                      /* OVS_VPORT_ATTR_NAME. */
    uint32_t n_upcall_pids;
    const uint32_t *upcall_pids;           /* OVS_VPORT_ATTR_UPCALL_PID. */
    const struct ovs_vport_stats *stats;   /* OVS_VPORT_ATTR_STATS. */
    const struct nlattr *options;          /* OVS_VPORT_ATTR_OPTIONS. */
    size_t options_len;
};

void dpif_netlink_vport_init(struct dpif_netlink_vport *);

int dpif_netlink_vport_transact(const struct dpif_netlink_vport *request,
                                struct dpif_netlink_vport *reply,
                                struct ofpbuf **bufp);
int dpif_netlink_vport_get(const char *name, struct dpif_netlink_vport *reply,
                           struct ofpbuf **bufp);

bool dpif_netlink_is_internal_device(const char *name);

enum ovs_vport_type netdev_to_ovs_vport_type(const char *type);

#endif /* dpif-netlink.h */
