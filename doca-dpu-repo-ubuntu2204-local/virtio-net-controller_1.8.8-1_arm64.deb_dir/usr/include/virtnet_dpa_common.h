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

#ifndef __VIRTNET_DPA_COMMON_H__
#define __VIRTNET_DPA_COMMON_H__

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <libutil/util.h>
#include <libutil/atomic.h>
#include <virtnet_dpa_util.h>
#include <virtnet_dpa_common_dbg.h>

typedef uint64_t flexio_uintptr_t;

#define MAX_FETCHED_DESC	32
#define MAX_FETCHED_DESC_BIG_VQ	16
#define MAX_EHDR_LEN		142
#define MAX_VIRTQ_SIZE		256
#define MAX_VIRTQ_NUM		64
#define VIRTNET_DPA_SYNC_TIMEOUT 1000
#define FLEXIO_QP_QPC_RQ_TYPE_ZERO_SIZE_RQ 0x3
#define BIT_MODE_BYTE_LOG 3
#define PAGE_SIZE_START_LOG 12
#define DESC_PER_BATCH		32
#define VIRTNET_DPA_DIM_CQ_COUNT 1024
#define MAX_MEM_DUMP_NUM_DWORD 1024

/* Copied from linux/virtio_net.h as dpa redefined virtio_net_hdr */
enum {
	VIRTIO_NET_F_CSUM = 0,
	VIRTIO_NET_F_GUEST_CSUM = 1,
	VIRTIO_NET_F_CTRL_GUEST_OFFLOADS = 2,
	VIRTIO_NET_F_MTU = 3,
	VIRTIO_NET_F_MAC = 5,
	VIRTIO_NET_F_GUEST_TSO4 = 7,
	VIRTIO_NET_F_GUEST_TSO6 = 8,
	VIRTIO_NET_F_GUEST_ECN = 9,
	VIRTIO_NET_F_GUEST_UFO = 10,
	VIRTIO_NET_F_HOST_TSO4 = 11,
	VIRTIO_NET_F_HOST_TSO6 = 12,
	VIRTIO_NET_F_HOST_ECN = 13,
	VIRTIO_NET_F_HOST_UFO = 14,
	VIRTIO_NET_F_MRG_RXBUF = 15,
	VIRTIO_NET_F_STATUS = 16,
	VIRTIO_NET_F_CTRL_VQ = 17,
	VIRTIO_NET_F_CTRL_RX = 18,
	VIRTIO_NET_F_CTRL_VLAN = 19,
	VIRTIO_NET_F_CTRL_RX_EXTRA = 20,
	VIRTIO_NET_F_GUEST_ANNOUNCE = 21,
	VIRTIO_NET_F_MQ = 22,
	VIRTIO_NET_F_CTRL_MAC_ADDR = 23,
	VIRTIO_NET_F_HASH_REPORT = 57,
	VIRTIO_NET_F_GUEST_HDRLEN = 59,
	VIRTIO_NET_F_RSS = 60,
	VIRTIO_NET_F_RSC_EXT = 61,
	VIRTIO_NET_F_STANDBY = 62,
	VIRTIO_NET_F_SPEED_DUPLEX = 63,
};

enum {
	/* MAX size is from MAX_DESC_CHAIN */
	VIRTNET_DPA_MAX_WQE_LOG_SIZE = 2,
	VIRTNET_DPA_RX_WQE_STRIDE_LOG_SIZE = 4,
	VIRTNET_DPA_MAX_RX_WQE_STRIDE_LOG_SIZE =
	(VIRTNET_DPA_RX_WQE_STRIDE_LOG_SIZE + VIRTNET_DPA_MAX_WQE_LOG_SIZE),
};

enum {
	LM_DIRTY_MAP_MODE_BIT     = 0x0,
	LM_DIRTY_MAP_MODE_BYTE    = 0x1,
};

enum {
	VIRTNET_DB_CQ_LOG_DEPTH = 2,
	VIRTNET_DPA_CVQ_SQ_DEPTH = 6,
	VIRTNET_DPA_CVQ_RQ_DEPTH = 6,
	/* One sqe_seg is 16B and One WQEBB is 64B.
	 * The algorithm to calculate sq size is as below:
	 * for each batch, the max sqe_seg number used is in LSO case,
	 * virtnet_dpa_ctrl_seg_set_lso() - 1 sqe_seg
	 * virtnet_dpa_eth_seg_set_lso() - 1 sqe_seg
	 * virtnet_dpa_inhdr_seg_set_lso() - 5 sqe_seg
	 * virtnet_dpa_data_segs_set_lso() - 4 sqe_seg
	 * A total of 11 DPA sqe_segs for LSO case which translates to
	 * up to 3 network WQEBBs and for a batch size of 32 we will need
	 * at least 66 network WQEBBs and rounding up
	 * it will be 128 network WQEBBs
	 */
	VIRTNET_NW_SQ_LOG_DEPTH = 7,
	VIRTNET_NW_SQ_CQ_LOG_DEPTH = 0,
	VIRTNET_FETCH_CQ_LOG_DEPTH = 3,
	VIRTNET_FETCH_SQ_LOG_DEPTH = 7,
	VIRTNET_DPA_TX_DESC_LOG_DEPTH = 8,
	VIRTNET_DPA_RX_DESC_LOG_DEPTH = 8,
};

enum {
	VIRTNET_MAX_TUNNEL_TX_POST_DESC_BUDGET = 1,
	VIRTNET_MAX_TX_POST_DESC_BUDGET = 256,
	VIRTNET_MAX_TX_INLINE_POST_DESC_BUDGET = 128,
	VIRTNET_MAX_RX_POST_DESC_BUDGET = 96,
	VIRTNET_MAX_RX_POLL_PKT_BUDGET = 96,
};

#define VIRTNET_DPA_RX_AUX_HANDLER_POOL_SIZE 32
#define VIRTNET_DPA_TX_AUX_HANDLER_POOL_SIZE 32
/* The first 32 is vq-index based, remaining 32 is MRU(Most Recently
 * Used) based.
 */
#define VIRTNET_DPA_TX_DMA_QP_SIZE           64
/* dma_qp allocation with vq index, we call "fast" mode */
#define VIRTNET_DPA_DMA_QP_FAST_POOL_SIZE    32
#define VIRTNET_DPA_DMA_QP_MRU_POOL_SIZE     (VIRTNET_DPA_TX_DMA_QP_SIZE - \
					      VIRTNET_DPA_DMA_QP_FAST_POOL_SIZE)

#define VIRTNET_DPA_RX_DMA_QP_SIZE           64
/* dma_qp allocation with vq index, we call "fast" mode */
#define VIRTNET_DPA_RX_DMA_QP_FAST_POOL_SIZE    32
#define VIRTNET_DPA_RX_DMA_QP_MRU_POOL_SIZE     (VIRTNET_DPA_RX_DMA_QP_SIZE - \
					      VIRTNET_DPA_RX_DMA_QP_FAST_POOL_SIZE)

enum {
	VIRTIO_VERSION_NONE = 0,
	VIRTIO_VERSION_0_95 = 1,
	VIRTIO_VERSION_1_0 = 2,
};

enum {
	NET_HDR_LEN = 10,
	NET_HDR_MRG_RXBUF_LEN = 12,
};

enum dpa_sync_state_t{
	VIRTNET_DPA_SYNC_HOST_RDY = 1,
	VIRTNET_DPA_SYNC_DEV_RDY = 2,
};

enum {
	VIRTNET_DPA_AUX_OWNERSHIP = 0,
	VIRTNET_DPA_PRI_NETHDR_SKIP = 1,
	VIRTNET_DPA_PRI_NETHDR_POLL = 2,
};

enum virtnet_outbox_type {
	EMU_MNG_OUTBOX = 0,
	SF_OUTBOX = 1,
};

enum {
	VIRTNET_SQ_DB_HANDLER = 0,
	VIRTNET_SQ_AUX_HANDLER = 1,
	VIRTNET_RQ_DB_HANDLER = 2,
	VIRTNET_RQ_AUX_HANDLER = 3,
	VIRTNET_NW_RQ_HANDLER = 4,
	VIRTNET_CTRL_VQ_HANDLER = 5,
	VIRTNET_ADMIN_VQ_HANDLER = 6,
};

struct refcount {
	atomic32_t cnt;
	uint8_t valid;		/* when not set, try_get will abort */
};

struct virtnet_dpa_cq {
	uint32_t cq_num;
	uint32_t log_cq_depth;
	flexio_uintptr_t cq_ring_daddr;
	flexio_uintptr_t cq_dbr_daddr;
	struct flexio_cq *cq;
};

struct virtnet_dpa_error_handler_attr {
	flexio_uintptr_t event_handler_ctx;
	uint32_t event_handler_type;
};

struct virtnet_dpa_vq_counters {
	uint64_t received_desc;
	uint64_t completed_desc;
	/* dma_q_used is tx/rx specific */
	uint32_t dma_q_used;
	uint32_t handler_schd_num;
	uint16_t max_post_desc_num;
	uint8_t batch_num;
	uint32_t aux_handler_schd_num;
	uint32_t nw_handler_schd_num;
	uint64_t total_bytes;
};

struct virtnet_dpa_rx_shadow_avail_ring {
	uint16_t avail[POW2(VIRTNET_DPA_RX_DESC_LOG_DEPTH)];
};

struct virtnet_dpa_shadow_avail_ring {
	uint16_t avail[POW2(VIRTNET_DPA_TX_DESC_LOG_DEPTH)];
};

struct virtnet_dpa_vq_desc {
	uint64_t addr;
	uint32_t len;
	uint16_t flags;
	uint16_t next;
};

struct virtnet_dpa_cq_ctx {
	uint32_t cqn;
	uint32_t ci;
	struct virtnet_dev_cqe64 *ring;
	struct virtnet_dev_cqe64 *cqe;
	uint32_t comp_wqe_idx;
	uint32_t *dbr;
	uint8_t hw_owner_bit;
};

struct virtnet_dpa_ring_ctx {
	uint32_t num;
	/* Stores the q number which is right shifted by 8 bits to directly
	 * write into the WQE
	 */
	uint32_t num_shift;
	union flexio_dev_sqe_seg *ring;
	uint32_t wqe_seg_idx;
	uint32_t *dbr;
	uint32_t pi;
	uint32_t ci;
};

/* virtnet_dpa_vq_state values:
 *
 * @VIRTNET_DPA_VQ_STATE_INIT - VQ is created, but cannot handle doorbells.
 * @VIRTNET_DPA_VQ_STATE_SUSPEND - VQ is suspended, no outgoing DMA, can be restarted.
 * @VIRTNET_DPA_VQ_STATE_RDY - Can handle doorbells.
 * @VIRTNET_DPA_VQ_STATE_ERR - VQ is in error state.
 */
enum virtnet_dpa_vq_state {
	VIRTNET_DPA_VQ_STATE_INIT = 1 << 0,
	VIRTNET_DPA_VQ_STATE_RDY = 1 << 1,
	VIRTNET_DPA_VQ_STATE_SUSPEND = 1 << 2,
	VIRTNET_DPA_VQ_STATE_ERR = 1 << 3,
};

enum virtq_types {
	VIRTNET_DPA_SX = 0,
	VIRTNET_DPA_RX = 1,
	VIRTNET_DPA_CTRL_Q = 2,
	VIRTNET_DPA_ADMIN_Q = 3,
};

struct virtnet_dpa_avail_ring {
	uint64_t base_addr;
	uint16_t next_index;
	uint16_t done;
	uint32_t reserved;
};

struct virtnet_dpa_used_ring {
	uint64_t base_addr;
	uint16_t next_index;
	uint16_t reserved[3];
};

struct virtnet_dpa_desc_table {
	uint64_t base_addr;
};

struct virtnet_dpa_splitq {
	struct virtnet_dpa_avail_ring avail_ring;
	struct virtnet_dpa_desc_table desc_table;
	struct virtnet_dpa_used_ring used_ring;
};

struct virtnet_window_dev_config {
	uint32_t mkey;
	flexio_uintptr_t haddr;
	flexio_uintptr_t heap_memory;
} __attribute__((__packed__, aligned(8)));

struct virtnet_dpa_vq_ctx {
	struct virtnet_dpa_splitq splitq;
	uint32_t sf_crossing_mkey;
	uint32_t emu_crossing_mkey;
	uint32_t dumem_mkey;
	bool need_wait_sq_cq;
} __attribute__((__packed__, aligned(8)));

struct virtnet_dpa_sqrq {
	struct virtnet_dpa_vq_ctx vq_ctx;
	uint8_t wqe_log_size;
};

struct virtnet_dpa_vq_features {
	uint8_t tx_csum;
	uint8_t rx_csum;
	uint8_t tso_ipv4;
	uint8_t tso_ipv6;
	uint8_t hdr_len;
	uint8_t notify_data;
};

struct virtnet_dpa_device_ctx {
	uint16_t used_idx_cache[MAX_VIRTQ_NUM];
};

#define CYCLE_NAME_LEN 16
struct virtnet_dpa_cpu_counter {
	uint32_t cycles;
	uint32_t events;
	uint64_t total_cycles;
};

#define MAX_CPU_COUNTERS 8
struct virtnet_dpa_cpu_counters {
	struct virtnet_dpa_cpu_counter counter[MAX_CPU_COUNTERS];
};

struct virtio_hdr_flags_gso_type {
	uint8_t flags;
	uint8_t gso_type;
} __attribute__((packed));

union virtio_net_hdr_u16 {
	uint16_t flags_gso_type_raw;
	struct virtio_hdr_flags_gso_type type;
};

/* Smaller virtio_net_hdr for tx side which doesn't need to use fields starting
 * csum_start. We only need first 6 bytes, adding 2 bytes for padding so we can
 * copy hdr in one instruction.
 */
struct virtio_net_hdr_tx {
	union virtio_net_hdr_u16 flags_gso;
	uint16_t hdr_len;
	uint16_t gso_size;
	uint16_t rsvd;
} __attribute__((packed));

struct virtnet_dpa_vnet_hdr_rctx {
	struct virtio_net_hdr_tx hdrs[VIRTNET_MAX_TX_POST_DESC_BUDGET];
};

union virtnet_len_key {
	uint64_t data;
	struct {
		be32_t len;
		be32_t key;
	};
};

struct virtnet_tx_dma_q {
	be32_t vnet_hdr_rctx_mkey;
	struct virtnet_dpa_vnet_hdr_rctx *vnet_hdr_rctx;
	struct virtnet_dpa_shadow_avail_ring *avail_ring;
};

struct virtnet_rx_dma_q {
	struct virtnet_dpa_rx_shadow_avail_ring *avail_ring;
};

/* This will contain VIRTNET_DPA_TX_DMA_QP_SIZE +
 * VIRTNET_DPA_RX_DMA_QP_SIZE
 */
struct virtnet_dpa_dma_q_state {
	uint32_t qp_num;
	uint8_t qp_in_error;
};

struct virtnet_dma_q {
	struct flexio_qp *qp;
	struct virtnet_dpa_cq_ctx fetch_cq_ctx;
	struct virtnet_dpa_vq_desc *desc_table;

	flexio_uintptr_t qp_dbr_daddr;
	flexio_uintptr_t qp_sq_daddr;

	uint32_t hw_qp_sq_pi;
	uint32_t qp_num;
	uint16_t hw_qp_depth;
	uint8_t err_status;
	atomic32_t fast_pool_used;
	be32_t virtnet_shadow_vq_mkey;
	union {
		struct virtnet_tx_dma_q tx_q;
		struct virtnet_rx_dma_q rx_q;
	};
	uint64_t health_addr;
	uint32_t health_lkey;
};

struct virtnet_dpa_dirtymap_para {
	uint32_t vhost_log_page:5;	/* Log (base 2) of page size */
	uint32_t mode:2;		/* 0-bitmap. 1-bytemap */
	uint32_t pad:25;		/* padding */
	uint32_t mkey;			/* mkey to log (a.k.a dirty_bitmap) */
	uint64_t addr;			/* addr of dirty_bitmap */
};

struct virtnet_dpa_shared_mem_ctx {
	uint16_t depth[VIRTNET_MAX_TX_POST_DESC_BUDGET];
	uint16_t avail[VIRTNET_MAX_TX_POST_DESC_BUDGET];
	/* Number of desc to process per batch */
	uint8_t batch_flag[VIRTNET_MAX_TX_POST_DESC_BUDGET/DESC_PER_BATCH];
};

struct virtnet_dpa_rq_shadow_elem {
	/* All rq specific shadow memory should be put here */
	uint16_t used_idx;
};

struct virtnet_dpa_event_handler_ctx {
	struct virtnet_dpa_device_ctx *dev_ctx;
	uint8_t net_hdr_len;
	uint8_t type;
	uint8_t batch;
	uint8_t is_chain;
	struct refcount refcount;
	union virtnet_len_key len_key;

	struct virtnet_dpa_cq_ctx guest_db_cq_ctx;
	struct virtnet_dpa_cq_ctx nw_cq_ctx;
	struct virtnet_dpa_ring_ctx ring_ctx;
	struct virtnet_dpa_vq_counters eh_vq_counters;
	struct virtnet_dpa_vq_counters *host_vq_counters;

	struct virtnet_dpa_sqrq sqrq;
	uint32_t emu_mng_gvmi;
	uint32_t sf_gvmi;
	struct virtnet_dpa_vq_features features;
	uint32_t emu_db_to_cq_id;
	uint32_t msix_cqn;
	flexio_uintptr_t window_base_addr;
	uint16_t hw_used_index;
	uint16_t vq_index;
	uint16_t vq_depth;
#ifdef LATENCY_DEBUG
	struct virtnet_dpa_cpu_counters cpu_cnts;
	struct virtnet_dpa_latency_stats lat_stats;
#endif
	struct {
		/* Primary handler sets these fields before invoking auxiliary
		 * handler.
		 */
		uint8_t aux_handler_running;
		uint8_t pri_handler_running;
		uint8_t aux_handler_bailout;
		uint8_t avail_req_poll_done;
		uint8_t desc_req_poll_done;
	} aux_handler;
	struct {
		union {
			struct virtnet_dpa_tx_aux_handler_ctx *tx_aux_ctx;
			struct virtnet_dpa_rx_aux_handler_ctx *rx_aux_ctx;
		} aux_ctx;
		struct virtnet_dma_q *dma_q;
	} error_handling;
	/* This will tell us if we have net_hdr data updated */
	bool use_qp_based_net_hdr;
	uint8_t cur_outbox_type;
	uint8_t stats_clear;
	struct virtnet_dpa_dirtymap_para dirty_map_para;
	struct virtnet_dpa_shared_mem_ctx *shared_mem_ctx;
	uint64_t dim_stats_addr;
	uint32_t dim_mr_lkey;
	bool dirty_log_enable;
	uint32_t cqe_err;
	uint8_t last_syndrome;
	uint8_t last_vendor_err_synd;
	uint8_t emu_ctx_pi_valid;
	/* It's an array with size of qsize */
	struct virtnet_dpa_rq_shadow_elem *rq_shadow;
};

struct virtnet_dpa_tunnel_event_handler_ctx {
	struct virtnet_dpa_event_handler_ctx ctx;
	struct {
		struct virtnet_dpa_cq qp_rqcq;
		uint32_t hw_qp_sq_pi;
		uint32_t hw_qp_cq_ci;
		uint32_t hw_qp_depth;
		uint16_t max_tunnel_desc;
		uint16_t qp_num;
		flexio_uintptr_t qp_sq_buff;
		flexio_uintptr_t qp_rq_buff;
		flexio_uintptr_t dbr_daddr;
	} dma_qp;
};

struct virtnet_dpa_vq_data {
	uint32_t dump_mem[MAX_MEM_DUMP_NUM_DWORD];
	struct virtnet_dpa_vq_counters vq_counters;
	struct virtnet_dpa_event_handler_ctx ehctx;
	enum dpa_sync_state_t state;
	flexio_uintptr_t db_cq_ring_addr;
	uint8_t err;
} __attribute__((__packed__, aligned(8)));

/* DIM stats per ms */
struct virtnet_dpa_vq_dim_ms {
	uint64_t ppms;
	uint64_t bpms;
	uint64_t epms;
};

struct virtnet_dpa_vq_dim_stats_sample {
	uint64_t time;
	uint64_t num_pkts;
	uint64_t num_bytes;
	uint64_t num_events;
};

struct virtnet_dpa_vq_dim_stats {
	struct flexio_cq *cq;

	uint8_t tune_state;
	uint8_t steps_left;
	uint8_t steps_right;
	uint8_t tired;
	uint8_t profile_ix;
	volatile uint8_t in_use;
	uint32_t count;


	struct virtnet_dpa_vq_dim_ms old_stats_per_ms;
	struct virtnet_dpa_vq_dim_stats_sample old_stats;
	struct virtnet_dpa_vq_dim_stats_sample new_stats;
};

struct virtnet_dpa_msix_send {
	uint32_t sf_gvmi;
	uint32_t cqn;
};

struct virtnet_dpa_rx_aux_handler_ctx {
	/* Primary handler will use this cqn to wake up auxiliary handler */
	struct virtnet_dpa_cq_ctx cq_ctx;
	/* Primary handler will update the following that
	 * auxiliary handler will read to fetch desc
	 */
	struct virtnet_dpa_event_handler_ctx *pri_ehctx;
	struct virtnet_dma_q *dma_q;
	uint32_t avail_ring_start_idx;
	uint32_t desc_count;
	int hart_num;

	/* When a secondary handler is allocated, by the caller,
	 * in_use is set, indicating it is allocated. When usage is done,
	 * its cleared (freed) for next usage.
	 */
	volatile bool in_use;
};

typedef void (virtnet_func_t) (void);
struct virtnet_dpa_rx_aux_handler {
	struct flexio_event_handler *db_handler;
	struct virtnet_dpa_cq db_cq;
	/* Points to virtnet_dpa_rx_aux_handler_ctx structure. */
	flexio_uintptr_t rx_aux_ctx;
};

struct virtnet_dpa_rx_aux_handler_attr {
	flexio_uintptr_t rx_aux_ctx_pool;
	flexio_uintptr_t rx_aux_stack_addr;
	uint32_t rx_aux_pool_size;
};

struct virtnet_dpa_tx_aux_handler_ctx {
	/* Primary handler will use this cqn to wake up auxiliary handler */
	struct virtnet_dpa_cq_ctx cq_ctx;
	/* Primary handler will update the following that
	 * auxiliary handler will read to fetch desc
	 */
	struct virtnet_dpa_event_handler_ctx *pri_ehctx;
	struct virtnet_dma_q *dma_q;
	uint32_t emu_mng_outbox;
	uint32_t desc_count;
	int hart_num;
	/* When a secondary handler is allocated, by the caller,
	 * in_use is set, indicating it is allocated. When usage is done,
	 * its cleared (freed) for next usage.
	 */
	volatile bool in_use;
};

struct virtnet_dpa_tx_aux_handler {
	struct flexio_event_handler *db_handler;
	struct virtnet_dpa_cq db_cq;
	/* Points to virtnet_dpa_tx_aux_handler_ctx structure. */
	flexio_uintptr_t tx_aux_ctx;
};

struct virtnet_dpa_tx_aux_handler_attr {
	flexio_uintptr_t tx_aux_ctx_pool;
	flexio_uintptr_t tx_aux_stack_addr;
	uint32_t tx_aux_pool_size;
};

struct virtnet_dpa_tx_dma_q_dev_access {
	struct virtnet_dma_q qps[VIRTNET_DPA_TX_DMA_QP_SIZE];
	flexio_uintptr_t stack_daddr;
};

struct virtnet_dpa_rx_dma_q_dev_access {
	struct virtnet_dma_q qps[VIRTNET_DPA_RX_DMA_QP_SIZE];
	flexio_uintptr_t stack_daddr;
};

struct virtnet_dev_cqe64 {
	uint32_t rsvd0[7];	/* 00h..06h - Reserved */
	uint8_t csum_ok;	/* 07h 24..26 - checksum ok bits */
	uint8_t l4_hdr_type;	/* 07h 20..22 - l4 hdr type bits */
	uint8_t rsvd30[2];	/* 07h 0..15 - Reserved */
	uint32_t srqn_uidx;	/* 08h - SRQ number or user index */
	uint32_t rsvd36[2];	/* 09h..0Ah - Reserved */
	uint32_t byte_cnt;	/* 0Bh - Byte count */
	uint32_t rsvd48[2];	/* 0Ch..0Dh - Reserved */
	uint32_t qpn;		/* 0Eh - QPN */
	uint16_t wqe_counter;	/* 0Fh 16..31 - WQE counter */
	uint8_t signature;	/* 0Fh 8..15 - Signature */
	uint8_t op_own;		/* 0Fh 0 - Ownership bit */
} __attribute__((packed, aligned(8)));
#endif
