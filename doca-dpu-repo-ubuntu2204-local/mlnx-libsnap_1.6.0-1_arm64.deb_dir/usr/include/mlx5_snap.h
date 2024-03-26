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

#ifndef MLX5_SNAP_H
#define MLX5_SNAP_H

#include <stdlib.h>
#if !defined(__DPA)
#include <pthread.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <linux/types.h>
#endif

#define SNAP_FT_ROOT_LEVEL 5
#define SNAP_FT_LOG_SIZE 10

enum mlx5_snap_flow_group_type {
	SNAP_FG_MATCH	= 1 << 0,
	SNAP_FG_MISS	= 1 << 1,
};

struct snap_event;
struct snap_pci_attr;

struct mlx5_snap_device;
struct mlx5_snap_flow_group;
struct mlx5_snap_flow_table_entry;

struct mlx5_snap_context {
	uint8_t				max_ft_level;
	uint8_t				log_max_ft_size;

	bool				nvme_need_tunnel;
	bool				virtio_net_need_tunnel;
	bool				virtio_blk_need_tunnel;
	bool				virtio_fs_need_tunnel;
};

struct mlx5_snap_pci {
	int				vhca_id;
};

struct mlx5_snap_devx_obj {
	struct mlx5dv_devx_obj		*obj;
	uint32_t			obj_id;
	struct snap_device		*sdev;
	int (*consume_event)(struct mlx5_snap_devx_obj *obj,
			     struct snap_event *event);

	/* destructor for tunneld objects */
	struct mlx5_snap_devx_obj	*vtunnel;
	void				*dtor_in;
	int				inlen;
	void				*dtor_out;
	int				outlen;
};

struct mlx5_snap_flow_table {
	struct mlx5_snap_devx_obj		*ft;
	uint32_t				table_id;
	uint32_t				table_type;
	uint8_t					level;
	uint64_t				ft_size;

	pthread_mutex_t				lock;

	TAILQ_HEAD(, mlx5_snap_flow_group)	fg_list;

	struct mlx5_snap_flow_table_entry	*ftes;
};

struct mlx5_snap_flow_group {
	struct mlx5_snap_devx_obj		*fg;
	uint32_t				group_id;
	uint32_t				start_idx;
	uint32_t				end_idx;
	enum mlx5_snap_flow_group_type		type;

	pthread_mutex_t				lock;
	uint8_t					*fte_bitmap;

	TAILQ_ENTRY(mlx5_snap_flow_group)	entry;
	struct mlx5_snap_flow_table		*ft;
};

struct mlx5_snap_flow_table_entry {
	struct mlx5_snap_devx_obj		*fte;
	uint32_t				idx;

	struct mlx5_snap_flow_group		*fg;
};

struct mlx5_snap_hw_qp {
	struct mlx5_snap_devx_obj		*mqp;
	struct mlx5_snap_flow_group		*fg_tx;
	struct mlx5_snap_flow_table_entry	*fte_tx;
	struct mlx5_snap_flow_group		*fg_rx;
	struct mlx5_snap_flow_table_entry	*fte_rx;

	struct mlx5dv_flow_matcher		*rdma_matcher;
	struct ibv_flow				*rdma_flow;

	struct ibv_qp				*parent_qp;
};

struct mlx5_snap_device {
	struct ibv_context			*context;
	struct mlx5_snap_devx_obj		*device_emulation;
	struct mlx5dv_devx_event_channel	*channel;

	/* for BF-1 usage only */
	struct mlx5_snap_devx_obj		*vtunnel;
	struct mlx5_snap_flow_table		*tx;
	struct mlx5_snap_flow_group		*fg_tx;
	struct mlx5_snap_flow_table		*rx;
	struct mlx5_snap_flow_group		*fg_rx;
	struct mlx5_snap_flow_group		*fg_rx_miss;
	struct mlx5_snap_devx_obj		*tunneled_pd;
	uint32_t				pd_id;
	pthread_mutex_t				rdma_lock;
	struct ibv_context			*rdma_dev;
	struct mlx5_snap_flow_table_entry	*rdma_fte_rx;
	struct mlx5_snap_flow_table_entry	*fte_rx_miss;
	struct mlx5_snap_flow_table		*rdma_ft_rx;
	struct mlx5_snap_flow_group		*rdma_fg_rx;
	unsigned int				rdma_dev_users;
};

/* The following functions are to be used by NVMe/VirtIO libraries only */
int snap_init_device(struct snap_device *sdev);
int snap_teardown_device(struct snap_device *sdev);
struct mlx5_snap_devx_obj*
snap_devx_obj_create(struct snap_device *sdev, void *in, size_t inlen,
		void *out, size_t outlen, struct mlx5_snap_devx_obj *vtunnel,
		size_t dtor_inlen, size_t dtor_outlen);
int snap_devx_obj_destroy(struct mlx5_snap_devx_obj *snap_obj);
int snap_devx_obj_modify(struct mlx5_snap_devx_obj *snap_obj, void *in,
			 size_t inlen, void *out, size_t outlen);
int snap_devx_obj_query(struct mlx5_snap_devx_obj *snap_obj, void *in,
			size_t inlen, void *out, size_t outlen);
void snap_get_pci_attr(struct snap_pci_attr *pci_attr,
		void *pci_params_out);
uint16_t snap_get_vhca_id(struct snap_device *sdev);
uint16_t snap_get_dev_vhca_id(struct ibv_context *context);
void snap_put_rdma_dev(struct snap_device *sdev, struct ibv_context *context);
struct ibv_context *snap_find_get_rdma_dev(struct snap_device *sdev,
		struct ibv_context *context);
struct mlx5_snap_hw_qp *snap_create_hw_qp(struct snap_device *sdev,
		struct ibv_qp *qp);
int snap_destroy_hw_qp(struct mlx5_snap_hw_qp *hw_qp);

#endif
