/*
 * Copyright (c) 2016 Nicira, Inc.
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

#ifndef DPDK_H
#define DPDK_H

#include <stdbool.h>

#ifdef DPDK_NETDEV

#include <rte_config.h>
#include <rte_lcore.h>

#define NON_PMD_CORE_ID LCORE_ID_ANY

#else

#define NON_PMD_CORE_ID UINT32_MAX

#endif /* DPDK_NETDEV */

struct smap;
struct ovsrec_open_vswitch;
struct ovs_dpdk_mempool;

void dpdk_init(const struct smap *ovs_other_config);
bool dpdk_attach_thread(unsigned cpu);
void dpdk_detach_thread(void);
const char *dpdk_get_vhost_sock_dir(void);
bool dpdk_vhost_iommu_enabled(void);
bool dpdk_vhost_postcopy_enabled(void);
bool dpdk_per_port_memory(void);
bool dpdk_available(void);
void print_dpdk_version(void);
const char *dpdk_get_version(void);
void dpdk_status(const struct ovsrec_open_vswitch *);

void ovs_dpdk_mempool_destroy(struct ovs_dpdk_mempool *odmp);
struct ovs_dpdk_mempool *ovs_dpdk_mempool_create(unsigned n, unsigned elt_size);
void ovs_dpdk_mempool_free(struct ovs_dpdk_mempool *odmp, void *obj);
int ovs_dpdk_mempool_alloc(struct ovs_dpdk_mempool *odmp, void **obj_p);

#endif /* dpdk.h */
