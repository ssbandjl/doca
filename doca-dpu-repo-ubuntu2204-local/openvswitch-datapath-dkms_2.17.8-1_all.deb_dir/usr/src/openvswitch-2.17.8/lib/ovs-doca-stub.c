/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <config.h>

#include "ovs-doca.h"
#include "ovs-thread.h"
#include "openvswitch/vlog.h"
#include "smap.h"
#include "vswitch-idl.h"

#ifdef DPDK_NETDEV
#include "dpdk-offload-provider.h"
struct dpdk_offload_api dpdk_offload_api_doca = {0};
#endif /* DPDK_NETDEV */

VLOG_DEFINE_THIS_MODULE(doca);

int
ovs_doca_init(const struct smap *ovs_other_config OVS_UNUSED)
{
    return 0;
}

void *
ovs_doca_port_create(uint16_t port_id OVS_UNUSED)
{
    return NULL;
}

int
ovs_doca_port_destroy(void *port OVS_UNUSED)
{
    return 0;
}

void
ovs_doca_status(const struct ovsrec_open_vswitch *cfg)
{
    if (cfg) {
        ovsrec_open_vswitch_set_doca_initialized(cfg, false);
        ovsrec_open_vswitch_set_doca_version(cfg, "none");
    }
}

bool
ovs_doca_enabled(void)
{
    return false;
}

bool
ovs_doca_initialized(void)
{
    return false;
}

void
print_doca_version(void)
{
}

const char *
ovs_doca_get_version(void)
{
    return "none";
}

int
ovs_doca_create_meter(uint32_t meter_id OVS_UNUSED,
                      struct ofputil_meter_config *config OVS_UNUSED,
                      struct rte_mtr_error *error OVS_UNUSED)
{
    return -1;
}

int
ovs_doca_delete_meter(uint16_t port_id OVS_UNUSED,
                      uint32_t meter_id OVS_UNUSED,
                      struct rte_mtr_error *error OVS_UNUSED)
{
    return -1;
}

int
ovs_doca_mtr_stats_read(uint16_t port_id OVS_UNUSED,
                        uint32_t mtr_id OVS_UNUSED,
                        struct rte_mtr_stats *stats OVS_UNUSED,
                        struct rte_mtr_error *error OVS_UNUSED)
{
    return 0;
}
