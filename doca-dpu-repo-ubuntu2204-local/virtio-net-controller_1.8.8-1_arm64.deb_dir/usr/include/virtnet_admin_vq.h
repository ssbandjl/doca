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

#ifndef __VIRTNET_DPA_ADMIN_VQ_H__
#define __VIRTNET_DPA_ADMIN_VQ_H__

int
virtnet_dpa_admin_vq_create(struct virtnet_device *dev, int idx);
void
virtnet_dpa_admin_vq_destroy(struct virtnet_device *dev);
void
virtnet_dpa_admin_cmd_new(struct virtnet_device *dev,
			  const struct virtnet_admin_request_header *req);
int virtnet_dpa_admin_cmd_init(struct virtnet_device *dev, int size);
void virtnet_dpa_admin_cmd_deinit(struct virtnet_device *dev);
struct snap_dma_q *virtnet_dpa_admin_get_dma_q(struct virtnet_admin_vq *vq);
struct snap_dma_q *virtnet_dpa_admin_get_dma_q_by_cmd(struct snap_vq_cmd *cmd);
struct ibv_cq *virtnet_dpa_admin_get_dma_cq(struct virtnet_admin_vq *vq);
void virtnet_dpa_get_admin_q(struct virtnet_device *dev,
			     struct virtnet_prov_vq *vq);
struct snap_virtio_adm_cmd_layout *
virtnet_dpa_get_admin_cmd_layout(struct snap_vq_cmd *cmd);
void
virtnet_dpa_admin_cmd_complete(struct snap_vq_cmd *cmd,
			       enum snap_virtio_adm_status status,
			       enum snap_virtio_adm_status_qualifier status_qualifier,
			       bool dnr);
size_t virtnet_dpa_admin_cmd_len_get(struct snap_vq_cmd *cmd);
void **virtnet_dpa_admin_cmd_priv(struct snap_vq_cmd *cmd);
int virtnet_dpa_admin_cmd_layout_data_read(struct snap_vq_cmd *cmd,
					   size_t total_len,
					   void *lbuf, uint32_t lbuf_mkey,
					   snap_vq_cmd_done_cb_t done_fn,
					   size_t layout_offset);
int virtnet_dpa_admin_cmd_layout_data_write(struct snap_vq_cmd *cmd,
					    size_t total_len,
					    void *lbuf, uint32_t lbuf_mkey,
					    snap_vq_cmd_done_cb_t done_fn);
void
__virtnet_dpa_admin_cmd_complete(struct virtnet_admin_cmd *cmd,
				 enum snap_virtio_adm_status status,
				 enum snap_virtio_adm_status_qualifier status_qualifier);
int virtnet_dpa_admin_cmd_layout_data_read_int(struct virtnet_admin_cmd *cmd,
					       size_t total_len,
					       void *lbuf, uint32_t lbuf_mkey,
					       virtnet_admin_cmd_done_cb_t done_fn,
					       size_t layout_offset);
int virtnet_dpa_admin_cmd_layout_data_write_int(struct virtnet_admin_cmd *cmd,
						size_t total_len,
						void *lbuf, uint32_t lbuf_mkey,
						virtnet_admin_cmd_done_cb_t done_fn);
size_t virtnet_dpa_admin_cmd_get_total_len(struct virtnet_admin_cmd *cmd);
#endif
