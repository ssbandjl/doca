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

#ifndef SNAP_VIRTIO_NET_H
#define SNAP_VIRTIO_NET_H

#include <sys/time.h>
#include "snap_virtio_common.h"

struct snap_virtio_net_device;

enum snap_virtio_net_queue_modify {
	SNAP_VIRTIO_NET_QUEUE_MOD_STATE	           = 1 << 0,
	SNAP_VIRTIO_NET_QUEUE_MOD_DIRTY_MAP_PARAM  = 1 << 3,
	SNAP_VIRTIO_NET_QUEUE_MOD_DIRTY_MAP_ENABLE = 1 << 4,
	SNAP_VIRTIO_NET_QUEUE_PERIOD               = 1 << 5,
	SNAP_VIRTIO_NET_QUEUE_DESC_USED_AVAIL_ADDR = 1 << 6,
	SNAP_VIRTIO_NET_QUEUE_HW_AVAIL_IDX         = 1 << 7,
	SNAP_VIRTIO_NET_QUEUE_HW_USED_IDX          = 1 << 8,
};

struct snap_virtio_net_queue_attr {
	/* create: */
	uint16_t			vhca_id;
	uint32_t			tisn_or_qpn;
	uint64_t			features;
	/* query result: */
	uint64_t			modifiable_fields;//mask of snap_virtio_net_queue_modify
	uint16_t			hw_available_index;
	uint16_t			hw_used_index;
	struct snap_virtio_queue_attr   vattr;
};

struct snap_virtio_net_queue {
	struct snap_virtio_queue	virtq;

	struct snap_virtio_net_device	*vndev;
};

struct snap_virtio_net_device_attr {
	uint64_t				mac;
	uint16_t				status;
	uint16_t				max_queue_pairs;
	uint16_t				mtu;
	struct snap_virtio_device_attr		vattr;
	struct snap_virtio_net_queue_attr	*q_attrs;
	unsigned int				queues;

	uint64_t				modifiable_fields;//mask of snap_virtio_dev_modify

	uint32_t				crossed_vhca_mkey;
};

struct snap_virtnet_migration_log {
	uint16_t			flag;
	uint16_t			mode;

	uint32_t			guest_page_size;
	uint64_t			log_base;
	uint32_t			log_size;
	uint32_t			num_sge;

	uint32_t			dirty_map_mkey;
	struct mlx5_klm			*klm_array;
	struct snap_indirect_mkey	*indirect_mkey;
	struct snap_cross_mkey		*crossing_mkey;
	struct ibv_mr			*mr;
};

struct snap_virtio_net_device {
	uint32_t				num_queues;
	struct snap_virtnet_migration_log	lattr;
	struct snap_virtio_net_queue		*virtqs;
};

int snap_virtio_net_init_device(struct snap_device *sdev);
int snap_virtio_net_teardown_device(struct snap_device *sdev);
int snap_virtio_net_query_device(struct snap_device *sdev,
	struct snap_virtio_net_device_attr *attr);
int snap_virtio_net_modify_device(struct snap_device *sdev, uint64_t mask,
		struct snap_virtio_net_device_attr *attr);
struct snap_virtio_net_queue*
snap_virtio_net_create_queue(struct snap_device *sdev,
	struct snap_virtio_net_queue_attr *attr);
int snap_virtio_net_destroy_queue(struct snap_virtio_net_queue *vnq);
int snap_virtio_net_query_queue(struct snap_virtio_net_queue *vnq,
		struct snap_virtio_net_queue_attr *attr);
int snap_virtio_net_modify_queue(struct snap_virtio_net_queue *vnq,
		uint64_t mask, struct snap_virtio_net_queue_attr *attr);
int snap_virtio_net_query_counters(struct snap_virtio_net_queue *vnq,
				struct snap_virtio_queue_counters_attr *q_cnt);

static inline struct snap_virtio_net_queue_attr*
to_net_queue_attr(struct snap_virtio_queue_attr *vattr)
{
	return container_of(vattr, struct snap_virtio_net_queue_attr,
			    vattr);
}

static inline struct snap_virtio_net_queue*
to_net_queue(struct snap_virtio_queue *virtq)
{
	return container_of(virtq, struct snap_virtio_net_queue, virtq);
}

static inline struct snap_virtio_net_device_attr*
to_net_device_attr(struct snap_virtio_device_attr *vattr)
{
	return container_of(vattr, struct snap_virtio_net_device_attr, vattr);
}

static inline void eth_random_addr(uint8_t *addr)
{
	struct timeval t;
	uint64_t rand;

	gettimeofday(&t, NULL);
	srandom(t.tv_sec + t.tv_usec);
	rand = random();

	rand = rand << 32 | random();

	memcpy(addr, (uint8_t *)&rand, 6);
	addr[0] &= 0xfe;        /* clear multicast bit */
	addr[0] |= 0x02;        /* set local assignment bit (IEEE802) */
}

void snap_virtio_net_pci_functions_cleanup(struct snap_context *sctx);
#endif
