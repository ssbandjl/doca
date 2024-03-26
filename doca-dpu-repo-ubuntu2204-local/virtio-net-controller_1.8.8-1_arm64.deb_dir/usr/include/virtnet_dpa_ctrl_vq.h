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

#ifndef __VIRTNET_DPA_CTRL_VQ_H__
#define __VIRTNET_DPA_CTRL_VQ_H__

struct virtnet_dpa_ctx;
struct virtnet_dpa_vq;
struct virtnet_dpa_emu_dev_ctx;

int virtnet_dpa_ctrl_vq_create(struct virtnet_dpa_vq *dpa_vq,
			       struct virtnet_dpa_ctx *dpa_ctx,
			       struct virtnet_dpa_emu_dev_ctx *emu_dev_ctx,
			       struct virtnet_prov_vq_init_attr *attr);
void virtnet_dpa_ctrl_vq_destroy(struct virtnet_dpa_vq *dpa_vq);
int virtnet_dpa_dma_q_init2rts(struct flexio_qp *qp,
			       struct flexio_qp_attr *qp_attr,
			       struct flexio_qp_attr_opt_param_mask *qp_mask);

#endif
