/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef VIRTNET_DPA_DEV_DBG_H
#define VIRTNET_DPA_DEV_DBG_H

#include "virtnet_dpa_common.h"

/* How to use below APIs to debug latency:
 * 1. Add VIRTNET_LAT_TX_xxx before VIRTNET_LAT_TX_MAX_xxx for tx or
 *    VIRTNET_LAT_RX_XXX before VIRTNET_LAT_RX_MAX_xxx for rx in
 *    virtnet_dpa_common_dbg.h.
 *
 * 2. Get the maximum between VIRTNET_LAT_TX_MAX_xxx and
 *    VIRTNET_LAT_RX_xxx and set it to LAT_MAX_xxx.
 *
 * 3. Add print name in virtnet_tx_latency_xxx/virtnet_rx_latency_xxx in
 *    virtnet_dpa_vq.c.
 *
 * 4. Call below APIs to debug latency.
 *    ctx is struct virtnet_dpa_event_handler_ctx.
 *    pre_cycle is gotten from virtnet_dpa_lat_cyc_get().
 *    cnt_idx is in [0..VIRTNET_LAT_TX_MAX_xxx] or [0..VIRTNET_LAT_RX_MAX_xxx].
 *
 *    For cycles:
 *      Define local variables as below:
 *        tx: uint32_t latency_cycle[VIRTNET_LAT_TX_MAX_CYC_CNT];
 *        rx: uint32_t latency_cycle[VIRTNET_LAT_RX_MAX_CYC_CNT];
 *      Get CPU cycles before the target code to be captured,
 *        virtnet_dpa_lat_cyc_get(&latency_cycle[VIRTNET_LAT_TX_xxx]);
 *      Put CPU cycles after the target code,
 *        virtnet_dpa_lat_cyc_put(ctx,
 *                                latency_cycle[VIRTNET_LAT_TX_xxx],
 *                                VIRTNET_LAT_TX_xxx);
 *
 *    For invoke counter:
 *      virtnet_dpa_lat_invoke_incr(ctx, VIRTNET_LAT_TX_xxx);
 *
 *    For batch:
 *      virtnet_dpa_lat_batch_set(ctx, batch, VIRTNET_LAT_TX_xxx);
 *
 * 5. trigger the code to be called,
 *    use cmd " virtnet query -p 0 --latency_stats" to output the results.
 */
#ifdef LATENCY_DEBUG
#include "virtnet_dpa_dev.h"
static inline void virtnet_dpa_lat_cyc_get(uint32_t *cycles)
{
	*cycles = virtnet_dpa_cpu_cyc_get();
}

static inline void
virtnet_dpa_lat_cyc_put(struct virtnet_dpa_event_handler_ctx *ctx,
			uint32_t pre_cycle, uint32_t cnt_idx)
{
	uint32_t post_cycle;
	virtnet_dpa_lat_cyc_get(&post_cycle);
	ctx->lat_stats.cycle_counter[cnt_idx].cycles = post_cycle - pre_cycle;
	ctx->lat_stats.cycle_counter[cnt_idx].events++;
	ctx->lat_stats.cycle_counter[cnt_idx].total_cycles += post_cycle - pre_cycle;
}

static inline void
virtnet_dpa_lat_invoke_incr(struct virtnet_dpa_event_handler_ctx *ctx,
			    uint32_t cnt_idx)
{
	ctx->lat_stats.invoke_counter[cnt_idx]++;
}

static inline void
virtnet_dpa_lat_batch_set(struct virtnet_dpa_event_handler_ctx *ctx,
			  uint32_t batch, uint32_t cnt_idx)
{
	ctx->lat_stats.batch_value[cnt_idx].events++;
	ctx->lat_stats.batch_value[cnt_idx].total_value += batch;
}

#else
static inline void
virtnet_dpa_lat_cyc_get(uint32_t *cycles __attribute__((unused))) {}

static inline void
virtnet_dpa_lat_cyc_put(struct virtnet_dpa_event_handler_ctx *ctx
			__attribute__((unused)),
			uint32_t pre_cycle __attribute__((unused)),
			uint32_t cnt_idx __attribute__((unused))) {}

static inline void
virtnet_dpa_lat_invoke_incr(struct virtnet_dpa_event_handler_ctx *ctx
			    __attribute__((unused)),
			    uint32_t cnt_idx __attribute__((unused))) {}

static inline void
virtnet_dpa_lat_batch_set(struct virtnet_dpa_event_handler_ctx *ctx
			  __attribute__((unused)),
			  uint32_t value __attribute__((unused)),
			  uint32_t cnt_idx __attribute__((unused))) {}

#endif /* LATENCY_DEBUG */
#endif /* VIRTNET_DPA_DEV_DBG_H */
