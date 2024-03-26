/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef __VIRTNET_DPA_H__
#define __VIRTNET_DPA_H__

#include <libflexio/flexio.h>
#include <infiniband/mlx5dv.h>
#include "virtnet_dpa_common.h"

#define COREDUMP_FILE	"/opt/mellanox/mlnx_virtnet/dpa_coredump"
#define VIRTNET_MAX_HARTS_PER_CORE  16

struct virtnet_prov_init_attr;
struct virtnet_prov_emu_dev_init_attr;
struct virtnet_prov_caps;

enum virtnet_dpa_device_handler_type {
	VIRTNET_DPA_DEVICE_MSIX_SEND = 0,
	VIRTNET_DPA_DEVICE_RX_DMA_Q_POOL,
	VIRTNET_DPA_DEVICE_TX_DMA_Q_POOL,
	VIRTNET_DPA_DEVICE_RX_AUX_HANDLER,
	VIRTNET_DPA_DEVICE_TX_AUX_HANDLER,
	VIRTNET_DPA_DEVICE_RX_DMA_Q_NUM_HANDLER,
	VIRTNET_DPA_DEVICE_RX_DIM_STATE_HANDLER,
	VIRTNET_DPA_DEVICE_MAX
};

/* Hart 0-4 of each core is reserved for Aux handlers.
 * Hart 5 - 15 of each core is reserved for primary handlers.
 */
enum virtnet_dpa_handler_core_pos {
	VIRTNET_AUX_HDLR_HART_START = 0,
	VIRTNET_PRI_HDLR_HART_START = 5
};

struct virtnet_dpa_rx_shadow_desc_tbl {
	struct virtnet_dpa_vq_desc descs[POW2(VIRTNET_DPA_RX_DESC_LOG_DEPTH)];
};

struct virtnet_dpa_shadow_desc_tbl {
	struct virtnet_dpa_vq_desc descs[POW2(VIRTNET_DPA_TX_DESC_LOG_DEPTH)];
};

struct virtnet_dpa_rx_dma_q_pool {
	flexio_uintptr_t virtnet_shadow_vq_ctx_daddr;
	struct flexio_mkey *virtnet_shadow_vq_mkey;
	flexio_uintptr_t vnet_avail_ring_daddr;
	/* The device address for below 'dev_access' */
	flexio_uintptr_t pool_daddr;
	struct virtnet_dpa_rx_dma_q_dev_access dev_access;
	struct virtnet_dpa_cq qp_rqcq[VIRTNET_DPA_RX_DMA_QP_SIZE];
};

struct virtnet_dpa_tx_dma_q_pool {
	flexio_uintptr_t vnet_hdr_rctx_daddr;
	struct flexio_mkey *vnet_hdr_rctx_mkey;
	flexio_uintptr_t virtnet_shadow_vq_ctx_daddr;
	struct flexio_mkey *virtnet_shadow_vq_mkey;
	flexio_uintptr_t vnet_avail_ring_daddr;
	/* The device address for below 'dev_access' */
	flexio_uintptr_t pool_daddr;
	struct virtnet_dpa_tx_dma_q_dev_access dev_access;
	struct virtnet_dpa_cq qp_sqcq[VIRTNET_DPA_TX_DMA_QP_SIZE];
};

struct virtnet_dpa_dma_q_health {
	struct virtnet_dpa_dma_q_state *state;
	struct ibv_mr *mr;
	pthread_t tid;
	volatile bool enable;
};

struct virtnet_dpa_dim {
	struct virtnet_dpa_vq_dim_stats *stats;
	struct ibv_mr *mr;
	pthread_mutex_t stat_lock;
	pthread_t tid;
	volatile bool enable;
};

struct virtnet_dpa_ctx {
	struct flexio_process *flexio_process;
	struct flexio_uar *flexio_uar;
	uint32_t emu_mng_gvmi;
	struct flexio_window *window;
	void *elf_buf;
	uint16_t dpa_core_start;
	uint16_t dpa_core_end;
	uint8_t aux_hdlr_hart_pos;
	uint8_t aux_hdlr_core_idx;
	uint8_t pri_hdlr_hart_pos;
	uint8_t pri_hdlr_core_idx;
	struct virtnet_dpa_vq_data *vq_data;
	struct ibv_mr *vq_counter_mr;
	struct virtnet_dpa_rx_dma_q_pool rx_dma_q_pool;
	struct virtnet_dpa_tx_dma_q_pool tx_dma_q_pool;
	struct virtnet_dpa_rx_aux_handler_pool *handler_pool;
	struct virtnet_dpa_tx_aux_handler_pool *tx_handler_pool;
	struct flexio_app *app;
	flexio_uintptr_t rx_aux_ctx_pool;
	flexio_uintptr_t rx_aux_stack_addr;
	flexio_uintptr_t rx_aux_err_attr;
	flexio_uintptr_t tx_aux_ctx_pool;
	flexio_uintptr_t tx_aux_stack_addr;
	flexio_uintptr_t tx_aux_err_attr;
	uint16_t emu_mgr_vhca_id;
	pthread_mutex_t hart_lock;
	struct flexio_msg_stream *stream;
	struct virtnet_dpa_dim dim;
	struct virtnet_dpa_dma_q_health dma_q_health;
};

struct virtnet_dpa_msix {
	atomic32_t msix_refcount;
	uint32_t cqn;
	uint32_t eqn;
	struct mlx5dv_devx_obj *obj;
	struct mlx5dv_devx_obj *alias_eq_obj;
	struct virtnet_dpa_cq alias_cq;
};

struct virtnet_dpa_rx_aux_handler_pool {
	struct virtnet_dpa_rx_aux_handler dpa_handler;
};

struct virtnet_dpa_tx_aux_handler_pool {
	struct virtnet_dpa_tx_aux_handler dpa_handler;
};

struct virtnet_dpa_emu_dev_ctx {
	struct virtnet_dpa_ctx *dpa_ctx;
	uint32_t *heap_mkey;
	struct flexio_uar *flexio_ext_uar;
	flexio_uintptr_t dev_ctx_daddr;
	struct flexio_mkey *dmem_key;
	uint16_t msix_config_vector;
	struct virtnet_dpa_msix *msix;
	struct virtnet_device *dev;
	pthread_mutex_t msix_lock;
};

#ifdef DEBUG
#define virtnet_dpa_dev_print_init flexio_msg_stream_create
#define virtnet_dpa_dev_print_uninit flexio_msg_stream_destroy
#else
static inline flexio_status
virtnet_dpa_dev_print_init(struct flexio_process *process,
			   struct flexio_log_dev_attr *log_dev_fattr, FILE *out,
			   pthread_t *ppthread,
			   struct flexio_msg_stream **stream)
{
	return 0;
}

static inline void
virtnet_dpa_dev_print_uninit(struct flexio_msg_stream *stream){}
#endif

#define virtnet_process_call(process, func, ret, err, ...) \
	do { \
		(*(err)) = flexio_process_call(process, func, ret, \
					       __VA_ARGS__); \
		virtnet_dpa_coredump(process, COREDUMP_FILE); \
	} while (0)

void virtnet_dpa_rpc_pack_func(void *arg_buf, va_list pa);
int virtnet_dpa_coredump(struct flexio_process *process, const char *corefile);
int virtnet_dpa_init(const struct virtnet_prov_init_attr *attr, void **out);
void virtnet_dpa_uninit(void *in);
int virtnet_dpa_emu_dev_init(const struct virtnet_prov_emu_dev_init_attr *attr,
			     void **out);
void virtnet_dpa_emu_dev_uninit(void *emu_dev_handler);
int virtnet_dpa_caps_query(void *dev, struct virtnet_prov_caps *caps_out);
int virtnet_dpa_device_msix_send(void *handler);
int
virtnet_dpa_rx_aux_handler_pool_create(struct virtnet_dpa_ctx *dpa_ctx,
				       struct ibv_context *emu_mgr_ibv_ctx);
void
virtnet_dpa_rx_aux_handler_pool_destroy(struct virtnet_dpa_ctx *dpa_ctx,
					struct flexio_process *process);
int
virtnet_dpa_tx_aux_handler_pool_create(struct virtnet_dpa_ctx *dpa_ctx,
				       struct ibv_context *emu_mgr_ibv_ctx);
void
virtnet_dpa_tx_aux_handler_pool_destroy(struct virtnet_dpa_ctx *dpa_ctx,
					struct flexio_process *process);
#endif
