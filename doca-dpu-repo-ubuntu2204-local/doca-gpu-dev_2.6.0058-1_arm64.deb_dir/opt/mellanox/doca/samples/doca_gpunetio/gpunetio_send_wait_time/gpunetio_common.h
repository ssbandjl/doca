/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#ifndef GPUNETIO_SEND_WAIT_TIME_COMMON_H_
#define GPUNETIO_SEND_WAIT_TIME_COMMON_H_

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <bsd/string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_error.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_eth_txq.h>
#include <doca_buf_array.h>

#include "common.h"

#define WARP_SIZE 32
#define NUM_BURST_SEND 8
#define NUM_PACKETS_X_BURST WARP_SIZE
#define PACKET_SIZE 1024
#define DELTA_NS 50000000 /* 50ms of delta before sending the first burst */
#define ETHER_ADDR_LEN 6
#define MAX_SQ_DESCR_NUM 8192

/* Application configuration structure */
struct sample_send_wait_cfg {
	char gpu_pcie_addr[DOCA_DEVINFO_PCI_ADDR_SIZE];	/* GPU PCIe address */
	char nic_pcie_addr[DOCA_DEVINFO_PCI_ADDR_SIZE];	/* Network card PCIe address */
	uint32_t time_interval_ns;			/* Nanoseconds between sends */
};

/* Tx buffer, used to send HTTP responses */
struct tx_buf {
	struct doca_gpu *gpu_dev;		/* GPU device */
	struct doca_dev *ddev;			/* Network DOCA device */
	uint32_t num_packets;			/* Number of packets in the buffer */
	uint32_t max_pkt_sz;			/* Max size of each packet in the buffer */
	uint32_t pkt_nbytes;			/* Effective bytes in each packet */
	uint8_t *gpu_pkt_addr;			/* GPU memory address of the buffer */
	struct doca_mmap *mmap;			/* DOCA mmap around GPU memory buffer for the DOCA device */
	struct doca_buf_arr *buf_arr;		/* DOCA buffer array object around GPU memory buffer */
	struct doca_gpu_buf_arr *buf_arr_gpu;	/* DOCA buffer array GPU handle */
	int dmabuf_fd;				/* GPU memory dmabuf file descriptor */
};

/* Send queues objects */
struct txq_queue {
	struct doca_gpu *gpu_dev;		/* GPUNetio handler associated to queues*/
	struct doca_dev *ddev;			/* DOCA device handler associated to queues */

	struct doca_ctx *eth_txq_ctx;		/* DOCA Ethernet send queue context */
	struct doca_eth_txq *eth_txq_cpu;	/* DOCA Ethernet send queue CPU handler */
	struct doca_gpu_eth_txq *eth_txq_gpu;	/* DOCA Ethernet send queue GPU handler */

	struct tx_buf txbuf;			/* GPU memory buffer for HTTP index page */
};

struct ether_hdr {
	uint8_t d_addr_bytes[ETHER_ADDR_LEN];	/* Destination addr bytes in tx order */
	uint8_t s_addr_bytes[ETHER_ADDR_LEN];	/* Source addr bytes in tx order */
	uint16_t ether_type;			/* Frame type */
} __attribute__((__packed__));

/*
 * Launch GPUNetIO send wait on time sample
 *
 * @sample_cfg [in]: Sample config parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_send_wait_time(struct sample_send_wait_cfg *sample_cfg);


#if __cplusplus
extern "C" {
#endif

/*
 * Launch a CUDA kernel to send packets with wait on time feature.
 *
 * @stream [in]: CUDA stream to launch the kernel
 * @txq [in]: DOCA Eth Tx queue to use to send packets
 * @intervals_gpu [in]: at which time each burst of packet must be sent
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t kernel_send_wait_on_time(cudaStream_t stream, struct txq_queue *txq, uint64_t *intervals_gpu);

#if __cplusplus
}
#endif
#endif
