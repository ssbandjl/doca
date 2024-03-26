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

#ifndef __VIRTNET_DPA_VQ_H__
#define __VIRTNET_DPA_VQ_H__

#include <libflexio/flexio.h>
#include <infiniband/mlx5dv.h>
#include "virtnet_dpa_common.h"

struct virtnet_dpa_ctx;
struct virtnet_prov_vq_init_attr;
enum virtnet_dpa_vq_state;

struct virtnet_dpa_nw_sq {
	flexio_uintptr_t wq_ring_daddr;
	struct virtnet_dpa_cq cq;
	struct flexio_sq *sq;
};

struct virtnet_dpa_nw_rq {
	flexio_uintptr_t wq_ring_daddr;
	flexio_uintptr_t wq_dbr_daddr;
	struct virtnet_dpa_cq cq;
	struct flexio_rq *rq;
};

enum virtnet_dpa_vq_type {
	VIRTNET_DPA_VQ_RQ = 0,
	VIRTNET_DPA_VQ_SQ = 1,
	VIRTNET_DPA_VQ_CTRL = 2,
	VIRTNET_DPA_VQ_ADMIN = 3,
	VIRTNET_DPA_VQ_MAX
};

struct virtnet_emu_db_to_cq_ctx {
	uint32_t emu_db_to_cq_id;
	struct mlx5dv_devx_obj *devx_emu_db_to_cq_ctx;
};

struct virtnet_dpa_cq_attr {
	uint8_t overrun_ignore;
	uint8_t always_armed;
};

struct virtnet_dpa_vq {
	struct flexio_event_handler *db_handler;
	union {
		struct flexio_event_handler *rq_nw_handler;
		struct flexio_event_handler *rq_dma_q_handler;
	};
	struct virtnet_emu_db_to_cq_ctx guest_db_to_cq_ctx;
	struct virtnet_dpa_cq db_cq;
	struct virtnet_dpa_cq dma_q_rqcq;
	struct virtnet_dpa_cq dma_q_sqcq;
	flexio_uintptr_t heap_memory;
	flexio_uintptr_t dpa_err_daddr;
	flexio_uintptr_t dpa_nw_rq_err_daddr;
	union {
		struct virtnet_dpa_nw_sq nw_sq;
		struct virtnet_dpa_nw_rq nw_rq;
	};
	struct virtnet_dpa_ctx *dpa_ctx;
	struct virtnet_dpa_emu_dev_ctx *emu_dev_ctx;
	int idx;
	enum virtnet_dpa_vq_type vq_type;
	struct flexio_msix *msix;
	struct virtnet_dpa_vq_counters *host_vq_counters;
	uint32_t sf_mkey;
	uint32_t emu_dev_xmkey;
	struct {
		struct flexio_qp *qp;
		struct flexio_mkey *rqd_mkey;
		flexio_uintptr_t buff_daddr;
		flexio_uintptr_t rq_daddr;
		flexio_uintptr_t sq_daddr;
		flexio_uintptr_t dbr_daddr;
		flexio_uintptr_t rx_wqe_buff;
		int qp_num;
	} dma_qp;
	uint16_t msix_vector;
	uint16_t db_hdlr_hart;
	uint16_t nw_hdlr_hart;
	uint8_t default_outbox_type;
	flexio_uintptr_t aux_shared_mem_ctx;
	uint64_t dim_stat_addr;
	flexio_uintptr_t rq_shadow_daddr;
};

struct virtnet_msix_init_attr {
	struct ibv_context *emu_mgr_ibv_ctx;
	uint16_t emu_dev_vhca_id;
	struct ibv_context *sf_ib_ctx;
	uint16_t sf_vhca_id;
	uint16_t msix_vector;
};

int virtnet_dpa_vq_init(struct virtnet_dpa_vq *dpa_vq,
			struct virtnet_dpa_ctx *dpa_ctx,
			struct virtnet_dpa_emu_dev_ctx *emu_dev_ctx,
			flexio_func_t *vq_handler_func,
			flexio_uintptr_t *dpa_daddr,
			int qsize);
void virtnet_dpa_vq_uninit(struct virtnet_dpa_vq *dpa_vq);

int virtnet_dpa_db_cq_create(struct virtnet_dpa_ctx *dpa_ctx,
			     struct ibv_context *emu_mgr_ibv_ctx,
			     struct flexio_event_handler *event_handler,
			     struct virtnet_dpa_cq_attr *dpa_cq_attr,
			     struct virtnet_dpa_cq *dpa_cq);
void virtnet_dpa_db_cq_destroy(struct virtnet_dpa_vq *dpa_vq);

int virtnet_dpa_vq_state_modify(struct virtnet_dpa_vq *dpa_vq,
				enum virtnet_dpa_vq_state vq_state);
int
virtnet_dpa_tunnel_vq_event_handler_init(const struct virtnet_dpa_vq *dpa_vq,
					 struct virtnet_dpa_ctx *dpa_ctx,
					 struct virtnet_prov_vq_init_attr *attr,
					 struct virtnet_dpa_emu_dev_ctx *emu_dev_ctx);
int
virtnet_dpa_vq_event_handler_init(const struct virtnet_dpa_vq *dpa_vq,
				  struct virtnet_dpa_ctx *dpa_ctx,
				  struct virtnet_prov_vq_init_attr *attr,
				  struct virtnet_dpa_emu_dev_ctx *emu_dev_ctx,
				  struct virtnet_dpa_tunnel_event_handler_ctx *tun_ctx);
int virtnet_dpa_msix_create(struct virtnet_dpa_vq *dpa_vq,
			    struct flexio_process *process,
			    struct virtnet_msix_init_attr *attr,
			    struct virtnet_dpa_emu_dev_ctx *emu_dev_ctx,
			    int max_msix);

void virtnet_dpa_msix_destroy(uint16_t msix_vector,
			      struct virtnet_dpa_emu_dev_ctx *emu_dev_ctx);
int
virtnet_dpa_rx_aux_handler_create(struct virtnet_dpa_ctx *dpa_ctx,
				  struct virtnet_dpa_rx_aux_handler *handler,
				  struct ibv_context *emu_mgr_ibv_ctx,
				  flexio_uintptr_t rx_aux_ctx,
				  flexio_uintptr_t rx_aux_err_attr);
void
virtnet_dpa_rx_aux_handler_destroy(struct flexio_process *process,
				   struct virtnet_dpa_rx_aux_handler *handler);
int
virtnet_dpa_tx_aux_handler_create(struct virtnet_dpa_ctx *dpa_ctx,
				  struct virtnet_dpa_tx_aux_handler *handler,
				  struct ibv_context *emu_mgr_ibv_ctx,
				  flexio_uintptr_t tx_aux_ctx,
				  flexio_uintptr_t tx_aux_err_attr);
void
virtnet_dpa_tx_aux_handler_destroy(struct flexio_process *process,
				   struct virtnet_dpa_tx_aux_handler *handler);
#endif
