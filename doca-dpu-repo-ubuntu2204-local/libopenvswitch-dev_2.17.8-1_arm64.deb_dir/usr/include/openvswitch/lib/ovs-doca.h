/*
 * Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#ifndef OVS_DOCA_H_
#define OVS_DOCA_H_

#include <config.h>

#include "dpif-netdev.h"
#include "smap.h"
#include "vswitch-idl.h"

#if DOCA_OFFLOAD
#include <doca_flow.h>

/* DOCA requires everything upfront. As a WA we define max number of ESW. */
#define OVS_DOCA_MAX_ESW    2

#define OVS_DOCA_QUEUE_DEPTH 32

/*
 * Shared counter IDs allocation scheme.
 *
 *                                 ovs_doca_max_shared_counters
 * --------------------------------------------------------------------------------------------->
 *
 *                ovs_doca_max_shared_counters_per_esw
 * ---------------------------------------------------------------------->
 *                                   ESW_0                                            ESW_N
 * ,----------------------------------+----------------------------------.     ,-------+---------.
 * |                CT                |               METER              |     |       |         |
 * |                                  | ,------------------------------. |     |       |         |
 * |                                  | |    GREEN,RED,GREEN,RED,...   | | ... |  CT   |  METER  |
 * |                                  | `------------------------------' |     |       |         |
 * |                                  |                                  |     |       |         |
 * `----------------------------------+----------------------------------'     `-------+---------'
 *   ovs_doca_max_ct_counters_per_esw | OVS_DOCA_MAX_METER_COUNTERS_PER_ESW
 * ---------------------------------->|--------------------------------->
 *                                    |
 *                                    v
 *                                    OVS_DOCA_METER_COUNTERS_PER_ESW_BASE_ID
 */

/* Using shared counters we need 2 per meter */
#define OVS_DOCA_MAX_METER_COLORS 2 /* green and red */
#define OVS_DOCA_MAX_METERS_PER_ESW MAX_METERS
#define OVS_DOCA_MAX_METERS (OVS_DOCA_MAX_METERS_PER_ESW * OVS_DOCA_MAX_ESW)
#define OVS_DOCA_MAX_METER_COUNTERS_PER_ESW \
    (OVS_DOCA_MAX_METERS_PER_ESW * OVS_DOCA_MAX_METER_COLORS)
#define OVS_DOCA_MAX_METER_COUNTERS (OVS_DOCA_MAX_METER_COUNTERS_PER_ESW * \
                                     OVS_DOCA_MAX_ESW)
/* Estimated maximum number of megaflows */
#define OVS_DOCA_MAX_MEGAFLOWS_COUNTERS (1 << 16)

unsigned int ovs_doca_max_ct_conns(void);

/* Connections are offloaded with one hardware rule per direction.
 * The netdev-offload layer manages offloads rule-wise, so a
 * connection is handled in two parts. This discrepancy can be
 * misleading.
 * This macro expresses the number of hardware rules required
 * to handle the number of CT connections supported by ovs-doca.
 */
static inline unsigned int
ovs_doca_max_ct_rules(void)
{
    return ovs_doca_max_ct_conns() * 2;
}

/* Using shared counters we need 1 per connection */
static inline unsigned int
ovs_doca_max_ct_counters_per_esw(void)
{
    return ovs_doca_max_ct_conns();
}

/* Total shared counters */
static inline unsigned int
ovs_doca_max_shared_counters_per_esw(void)
{
    return ovs_doca_max_ct_counters_per_esw() +
           OVS_DOCA_MAX_METER_COUNTERS_PER_ESW;
}

static inline unsigned int
ovs_doca_max_shared_counters(void)
{
    return ovs_doca_max_shared_counters_per_esw() * OVS_DOCA_MAX_ESW;
}

extern bool ovs_doca_async;
extern uint32_t ctl_pipe_size;
extern uint32_t ctl_pipe_infra_size;
extern unsigned int doca_congestion_threshold;
extern atomic_bool ovs_doca_eswitch_active_ids[OVS_DOCA_MAX_ESW];

static inline int
ovs_doca_eswitch_next_active(int next)
{
    bool active;

    while (next < OVS_DOCA_MAX_ESW) {
        atomic_read_explicit(&ovs_doca_eswitch_active_ids[next], &active,
                             memory_order_consume);
        if (active) {
            return next;
        }
        next++;
    }
    return next;
}

#define OVS_DOCA_FOREACH_ACTIVE_ESWITCH(ID) \
    for (ID = ovs_doca_eswitch_next_active(0); \
         ID < OVS_DOCA_MAX_ESW; \
         ID = ovs_doca_eswitch_next_active(ID + 1))

void
ovs_doca_entry_process_cb(struct doca_flow_pipe_entry *entry, uint16_t qid,
                          enum doca_flow_entry_status status,
                          enum doca_flow_entry_op op, void *aux);
void
ovs_doca_pipe_process_cb(struct doca_flow_pipe *pipe,
                         enum doca_flow_pipe_status status,
                         enum doca_flow_pipe_op op, void *user_ctx);

uint32_t
ovs_doca_get_post_meter_counter_id(uint32_t meter_id,
                                   enum doca_flow_meter_color meter_color);
uint32_t
ovs_doca_meter_id(uint32_t dp_meter_id, uint32_t esw_id);
#endif /* DOCA_OFFLOAD */

struct ofputil_meter_config;
struct rte_mtr_error;
struct rte_mtr_stats;

/* Whether DOCA support is compiled-in and user configuration
 * requested DOCA to be enabled. DOCA might not yet be initialized,
 * but the user expects a DOCA execution at some point. */
bool
ovs_doca_enabled(void);

/* Both 'ovs_doca_enabled() == true' and DOCA has successfully initialized. */
bool
ovs_doca_initialized(void);

int
ovs_doca_init(const struct smap *ovs_other_config);

void *
ovs_doca_port_create(uint16_t port_id);

int
ovs_doca_port_destroy(void *port);

void
ovs_doca_status(const struct ovsrec_open_vswitch *cfg);

void
print_doca_version(void);

const char *
ovs_doca_get_version(void);

int
ovs_doca_create_meter(uint32_t meter_id,
                      struct ofputil_meter_config *config,
                      struct rte_mtr_error *error);

int
ovs_doca_delete_meter(uint16_t port_id, uint32_t meter_id,
                      struct rte_mtr_error *error);

int
ovs_doca_mtr_stats_read(uint16_t port_id,
                        uint32_t mtr_id,
                        struct rte_mtr_stats *stats,
                        struct rte_mtr_error *error);
#endif
