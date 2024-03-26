/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef __VIRTNET_DPA_MM_H__
#define __VIRTNET_DPA_MM_H__

#include <libflexio/flexio.h>
#include <infiniband/mlx5dv.h>
#include "virtnet_dpa_common.h"

enum virtnet_dpa_ring_size {
	VIRTNET_DPA_CQE_BSIZE = 6,
	VIRTNET_DPA_SWQE_BSIZE = VIRTNET_DPA_CQE_BSIZE,
};

enum virtnet_dpa_buff_size {
	VIRTNET_DPA_QP_RQ_BUFF_SIZE = 64,
};

int virtnet_dpa_mm_zalloc(struct flexio_process *process, size_t buff_bsize,
			  flexio_uintptr_t *dest_daddr_p);
int virtnet_dpa_mm_free(struct flexio_process *process, flexio_uintptr_t daddr);
flexio_uintptr_t virtnet_dpa_mm_dbr_alloc(struct flexio_process *process);
int virtnet_dpa_mm_cq_alloc(struct flexio_process *process, int log_depth,
			    struct virtnet_dpa_cq *cq);
void virtnet_dpa_mm_cq_free(struct flexio_process *process,
			    struct virtnet_dpa_cq *cq);
flexio_uintptr_t virtnet_dpa_mm_cq_ring_alloc(struct flexio_process *process,
					      int log_depth);
void virtnet_dpa_mm_cq_ring_free(struct flexio_process *process,
				 flexio_uintptr_t ring_addr);
int virtnet_dpa_mm_db_cq_alloc(struct flexio_process *process,
			       struct virtnet_dpa_cq *cq);
void virtnet_dpa_mm_db_cq_free(struct flexio_process *process,
			       struct virtnet_dpa_cq *cq);
int virtnet_dpa_mm_sq_alloc(struct flexio_process *process, int log_depth,
			    struct virtnet_dpa_nw_sq *nw_sq);
void virtnet_dpa_mm_sq_free(struct flexio_process *process,
			    struct virtnet_dpa_nw_sq *nw_sq);
int virtnet_dpa_mm_rq_alloc(struct flexio_process *process, int log_depth,
			    struct virtnet_dpa_nw_rq *nw_rq);
void virtnet_dpa_mm_rq_free(struct flexio_process *process,
			    struct virtnet_dpa_nw_rq *nw_rq);
flexio_uintptr_t virtnet_dpa_mm_qp_buff_alloc(struct flexio_process *process,
					      int log_rq_depth,
					      flexio_uintptr_t *rq_daddr,
					      int log_sq_depth,
					      flexio_uintptr_t *sq_daddr);
void virtnet_dpa_mm_qp_buff_free(struct flexio_process *process,
				 flexio_uintptr_t buff_daddr);
int virtnet_dpa_init_qp_rx_ring(struct virtnet_dpa_vq *dpa_vq,
				flexio_uintptr_t *rq_daddr,
				uint32_t num_of_wqes,
				uint32_t wqe_stride,
				uint32_t mkey_id);
int virtnet_dpa_mkey_create(struct virtnet_dpa_vq *dpa_vq,
			    struct flexio_qp_attr *qp_attr,
			    uint32_t data_bsize);
void virtnet_dpa_mkey_destroy(struct virtnet_dpa_vq *dpa_vq);
#endif
