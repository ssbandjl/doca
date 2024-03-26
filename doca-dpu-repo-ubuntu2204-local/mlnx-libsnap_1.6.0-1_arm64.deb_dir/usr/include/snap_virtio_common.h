/*
 * Copyright © 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef SNAP_VIRTIO_COMMON_H
#define SNAP_VIRTIO_COMMON_H

#include <stdlib.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <pthread.h>
#include <linux/types.h>
#include <linux/virtio_ring.h>
#include "snap_dma.h"
#include "snap.h"

enum snap_virtq_type {
	SNAP_VIRTQ_SPLIT_MODE	= 1 << 0,
	SNAP_VIRTQ_PACKED_MODE	= 1 << 1,
};

enum snap_virtq_event_mode {
	SNAP_VIRTQ_NO_MSIX_MODE	= 1 << 0,
	SNAP_VIRTQ_QP_MODE	= 1 << 1,
	SNAP_VIRTQ_MSIX_MODE	= 1 << 2,
};

enum snap_virtq_offload_type {
	SNAP_VIRTQ_OFFLOAD_ETH_FRAME	= 1 << 0,
	SNAP_VIRTQ_OFFLOAD_DESC_TUNNEL	= 1 << 1,
};

enum snap_virtq_state {
	SNAP_VIRTQ_STATE_INIT		= 1 << 0,
	SNAP_VIRTQ_STATE_RDY		= 1 << 1,
	SNAP_VIRTQ_STATE_SUSPEND	= 1 << 2,
	SNAP_VIRTQ_STATE_ERR		= 1 << 3,
};

enum snap_virtq_error_type {
	SNAP_VIRTQ_ERROR_TYPE_NO_ERROR                      = 0x0,
	SNAP_VIRTQ_ERROR_TYPE_NETWORK_ERROR                 = 0x1,
	SNAP_VIRTQ_ERROR_TYPE_BAD_DESCRIPTOR                = 0x2,
	SNAP_VIRTQ_ERROR_TYPE_INVALID_BUFFER                = 0x3,
	SNAP_VIRTQ_ERROR_TYPE_DESCRIPTOR_LIST_EXCEED_LIMIT  = 0x4,
	SNAP_VIRTQ_ERROR_TYPE_INTERNAL_ERROR                = 0x5,
};

enum snap_virtio_features {
	SNAP_VIRTIO_NET_F_CSUM		= 1ULL << 0,
	SNAP_VIRTIO_NET_F_GUEST_CSUM	= 1ULL << 1,
	SNAP_VIRTIO_NET_F_HOST_TSO4	= 1ULL << 11,
	SNAP_VIRTIO_NET_F_HOST_TSO6	= 1ULL << 12,
	SNAP_VIRTIO_NET_F_MRG_RXBUF     = 1ULL << 15,
	SNAP_VIRTIO_NET_F_CTRL_VQ	= 1ULL << 17,
	SNAP_VIRTIO_F_VERSION_1		= 1ULL << 32,
};

enum snap_virtq_period_mode {
	SNAP_VIRTQ_PERIOD_DEFAULT	= 0,
	SNAP_VIRTQ_PERIOD_UPON_EVENT	= 1,
	SNAP_VIRTQ_PERIOD_UPON_CQE	= 2,
};

struct snap_virtio_queue_attr {
	enum snap_virtq_type		type;
	enum snap_virtq_event_mode	ev_mode;
	bool				full_emulation;
	bool				virtio_version_1_0;
	enum snap_virtq_offload_type	offload_type;
	uint16_t			max_tunnel_desc;
	uint32_t			event_qpn_or_msix;
	uint16_t			idx;
	uint16_t			size;
	uint16_t			msix_vector;
	uint16_t			enable;
	uint16_t			notify_off;
	uint64_t			desc;
	uint64_t			driver;
	uint64_t			device;
	struct ibv_pd			*pd;
	uint32_t			ctrs_obj_id;
	uint16_t			hw_available_index;
	uint16_t			hw_used_index;
	uint32_t			tisn_or_qpn;
	uint16_t			vhca_id;

	enum snap_virtq_state		state; /* query and modify */
	uint8_t				error_type;

	enum snap_virtq_period_mode	queue_period_mode;
	uint16_t			queue_period;
	uint16_t			queue_max_count;
	uint16_t			reset;

	/* lm */
	bool				dirty_map_dump_enable;
	bool				dirty_map_mode;
	uint32_t			dirty_map_mkey;
	uint32_t			dirty_map_size;
	uint64_t			dirty_map_addr;
	uint8_t				vhost_log_page;

	/* Query: */
	uint32_t			dma_mkey;
};

struct snap_virtio_queue_counters_attr {
	uint64_t			received_desc;
	uint64_t			completed_desc;
	uint32_t			error_cqes;
	uint32_t			bad_desc_errors;
	uint32_t			exceed_max_chain;
	uint32_t			invalid_buffer;
};

struct snap_virtio_queue_debugstat {
	uint32_t				qid;
	uint32_t				hw_available_index;
	uint32_t				sw_available_index;
	uint32_t				hw_used_index;
	uint32_t				sw_used_index;
	uint64_t				hw_received_descs;
	uint64_t				hw_completed_descs;
};

struct snap_virtio_ctrl_debugstat {
	uint32_t				network_error;
	uint32_t				bad_descriptor_error;
	uint32_t				invalid_buffer;
	uint32_t				desc_list_exceed_limit;
	uint32_t				internal_error;
	size_t					num_queues;
	struct snap_virtio_queue_debugstat	*queues;
};

struct snap_virtio_common_queue_attr {
	uint64_t			modifiable_fields;//mask of snap_virtio_queue_modify
	struct ibv_qp		*qp;
	uint16_t			hw_available_index;
	uint16_t			hw_used_index;

	struct snap_virtio_queue_attr   vattr;
	int					q_provider;
	struct snap_dma_q	*dma_q;
};

struct virtq_split_tunnel_req;

struct virtq_q_ops {
	struct snap_virtio_queue *(*create)(struct snap_device *sdev,
			struct snap_virtio_common_queue_attr *attr);
	int (*destroy)(struct snap_virtio_queue *vq);
	int (*query)(struct snap_virtio_queue *vq,
			struct snap_virtio_common_queue_attr *attr);
	int (*modify)(struct snap_virtio_queue *vq,
			uint64_t mask, struct snap_virtio_common_queue_attr *attr);
	/* extended ops */
	int (*poll)(struct snap_virtio_queue *vq, struct virtq_split_tunnel_req *req, int num_reqs);
	int (*complete)(struct snap_virtio_queue *vq, struct vring_used_elem *comp);
	int (*send_completions)(struct snap_virtio_queue *vq);
};

struct snap_virtio_queue {
	uint32_t				idx;
	struct mlx5_snap_devx_obj		*virtq;
	struct snap_umem			umem[3];
	uint64_t				mod_allowed_mask;
	struct mlx5_snap_devx_obj		*ctrs_obj;

	struct virtq_q_ops		*q_ops;
};

enum snap_virtio_dev_modify {
	SNAP_VIRTIO_MOD_DEV_STATUS = 1 << 0,
	SNAP_VIRTIO_MOD_LINK_STATUS = 1 << 1,
	SNAP_VIRTIO_MOD_RESET = 1 << 2,
	SNAP_VIRTIO_MOD_PCI_COMMON_CFG = 1 << 3,
	SNAP_VIRTIO_MOD_DEV_CFG = 1 << 4,
	SNAP_VIRTIO_MOD_ALL = 1 << 6,
	SNAP_VIRTIO_MOD_QUEUE_CFG = 1 << 7,
	SNAP_VIRTIO_MOD_NUM_MSIX = 1 << 8,
	SNAP_VIRTIO_MOD_DYN_MSIX_RESET = 1 << 9,
	SNAP_VIRTIO_MOD_PCI_HOTPLUG_STATE = 1 << 10,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_SIZE = 1 << 11,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_MSIX_VECTOR = 1 << 12,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_ENABLE = 1 << 13,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_DESC = 1 << 14,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_DRIVER = 1 << 15,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_DEVICE = 1 << 16,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_NOTIFY_OFF = 1 << 17,
	SNAP_VIRTIO_MOD_VQ_CFG_Q_RESET = 1 << 18,
};

struct snap_virtio_device_attr {
	uint64_t			device_feature;
	uint64_t			driver_feature;
	uint16_t			msix_config;
	uint16_t			max_queues;
	uint16_t			max_queue_size;
	uint16_t			pci_bdf;
	uint8_t				status;
	bool				enabled;
	bool				reset;
	bool				dynamic_vf_msix_reset;
	uint16_t			num_msix;
	uint16_t			num_free_dynamic_vfs_msix;
	uint16_t			num_of_vfs;
	uint8_t				config_generation;
	uint8_t				device_feature_select;
	uint8_t				driver_feature_select;
	uint16_t			queue_select;
	uint8_t				pci_hotplug_state;
	uint16_t			q_conf_list_size;
	uint16_t			admin_queue_index;
	uint16_t			admin_queue_num;
};


#define VQ_TABLE_REC 5
/**
 * struct virtq_split_tunnel_req_hdr - header of command received from FW
 *
 * Struct uses 2 rsvd so it will be aligned to 4B (and not 8B)
 */
struct virtq_split_tunnel_req_hdr {
	uint16_t descr_head_idx;
	uint16_t num_desc;
	uint32_t dpa_vq_table_flag;
	uint32_t rsvd2;
};

struct virtq_split_tunnel_req {
	struct virtq_split_tunnel_req_hdr hdr;
	struct vring_desc *tunnel_descs;
};

enum {
	SNAP_HW_Q_PROVIDER = 0,
	SNAP_SW_Q_PROVIDER = 1,
	SNAP_DPA_Q_PROVIDER = 2,
};

#define SNAP_QUEUE_PROVIDER   "SNAP_QUEUE_PROVIDER"

static inline struct snap_virtio_common_queue_attr*
to_common_queue_attr(struct snap_virtio_queue_attr *vattr)
{
	return container_of(vattr, struct snap_virtio_common_queue_attr,
			    vattr);
}

void snap_virtio_get_queue_attr(struct snap_virtio_queue_attr *vattr,
	void *q_configuration);
void snap_virtio_get_queue_attr_v2(struct snap_virtio_queue_attr *vattr,
				   void *q_configuration_v2);
void snap_virtio_get_device_attr(struct snap_device *sdev,
				 struct snap_virtio_device_attr *vattr,
				 void *device_configuration);
int snap_virtio_query_device(struct snap_device *sdev,
	enum snap_emulation_type type, uint8_t *out, int outlen);
int snap_virtio_modify_device(struct snap_device *sdev,
		enum snap_emulation_type type, uint64_t mask,
		struct snap_virtio_device_attr *attr);


struct mlx5_snap_devx_obj*
snap_virtio_create_queue_counters(struct snap_device *sdev);
int snap_virtio_query_queue(struct snap_virtio_queue *virtq,
	struct snap_virtio_queue_attr *vattr);
int snap_virtio_modify_queue(struct snap_virtio_queue *virtq, uint64_t mask,
	struct snap_virtio_queue_attr *vattr);
int snap_virtio_get_mod_fields_queue(struct snap_virtio_queue *virtq);

int snap_virtio_create_hw_queue(struct snap_device *sdev,
				struct snap_virtio_queue *vq,
				struct snap_virtio_caps *caps,
				struct snap_virtio_queue_attr *vattr);
int snap_virtio_destroy_hw_queue(struct snap_virtio_queue *vq);

int snap_virtio_get_vring_indexes_from_host(struct ibv_pd *pd, uint64_t drv_addr,
					    uint64_t dev_addr, uint32_t dma_mkey,
					    struct vring_avail *vra,
					    struct vring_used *vru);
int snap_virtio_get_avail_index_from_host(struct snap_dma_q *dma_q,
		uint64_t drv_addr, uint32_t dma_mkey, uint16_t *hw_avail, int *flush_ret);
int snap_virtio_get_used_index_from_host(struct snap_dma_q *dma_q,
		uint64_t dev_addr, uint32_t dma_mkey, uint16_t *hw_used, int *flush_ret);
int snap_virtio_query_queue_counters(struct mlx5_snap_devx_obj *counters_obj,
				struct snap_virtio_queue_counters_attr *attr);
int snap_virtio_common_queue_config(struct snap_virtio_common_queue_attr *common_attr,
		uint16_t hw_available_index, uint16_t hw_used_index, struct snap_dma_q *dma_q);
struct virtq_q_ops *snap_virtio_queue_provider(void);
#endif
