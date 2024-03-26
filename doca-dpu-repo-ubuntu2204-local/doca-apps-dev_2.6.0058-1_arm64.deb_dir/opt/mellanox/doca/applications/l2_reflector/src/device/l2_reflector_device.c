/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <stddef.h>

#include <libflexio-libc/stdio.h>
#include <libflexio-libc/string.h>
#include <libflexio-dev/flexio_dev.h>
#include <libflexio-dev/flexio_dev_err.h>
#include <libflexio-dev/flexio_dev_queue_access.h>
#include <dpaintrin.h>

#include "../common/l2_reflector_common.h"

flexio_dev_rpc_handler_t l2_reflector_dev_init;		/* Device initialization function */
flexio_dev_event_handler_t l2_reflector_event_handler;	/* Event handler function */

/* CQ Context */
struct cq_ctx_t {
	uint32_t cq_number;			/* CQ number */
	struct flexio_dev_cqe64 *cq_ring;	/* CQEs buffer */
	struct flexio_dev_cqe64 *cqe;		/* Current CQE */
	uint32_t cq_idx;			/* Current CQE IDX */
	uint8_t cq_hw_owner_bit;		/* HW/SW ownership */
	uint32_t *cq_dbr;			/* CQ doorbell record */
};

/* RQ Context */
struct rq_ctx_t {
	uint32_t rq_number;				/* RQ number */
	struct flexio_dev_wqe_rcv_data_seg *rq_ring;	/* WQEs buffer */
	uint32_t *rq_dbr;				/* RQ doorbell record */
};

/* SQ Context */
struct sq_ctx_t {
	uint32_t sq_number;			/* SQ number */
	uint32_t sq_wqe_seg_idx;		/* WQE segment index */
	union flexio_dev_sqe_seg *sq_ring;	/* SQEs buffer */
	uint32_t *sq_dbr;			/* SQ doorbell record */
	uint32_t sq_pi;				/* SQ producer index */
};

/* SQ data buffer */
struct dt_ctx_t {
	void *sq_tx_buff;	/* SQ TX buffer */
	uint32_t tx_buff_idx;	/* TX buffer index */
};

/* Device context */
static struct {
	uint32_t lkey;			/* Local memory key */
	uint32_t is_initalized;		/* Initialization flag */
	struct cq_ctx_t rqcq_ctx;	/* RQ CQ context */
	struct cq_ctx_t sqcq_ctx;	/* SQ CQ context */
	struct rq_ctx_t rq_ctx;		/* RQ context */
	struct sq_ctx_t sq_ctx;		/* SQ context */
	struct dt_ctx_t dt_ctx;		/* DT context */
	uint32_t packets_count;		/* Number of processed packets */
} dev_ctx = {0};

/*
 * Initialize the CQ context
 *
 * @app_cq [in]: CQ HW context
 * @ctx [out]: CQ context
 */
static void
init_cq(const struct app_transfer_cq app_cq, struct cq_ctx_t *ctx)
{
	ctx->cq_number = app_cq.cq_num;
	ctx->cq_ring = (struct flexio_dev_cqe64 *)app_cq.cq_ring_daddr;
	ctx->cq_dbr = (uint32_t *)app_cq.cq_dbr_daddr;

	ctx->cqe = ctx->cq_ring; /* Points to the first CQE */
	ctx->cq_idx = 0;
	ctx->cq_hw_owner_bit = 0x1;
}

/*
 * Initialize the RQ context
 *
 * @app_rq [in]: RQ HW context
 * @ctx [out]: RQ context
 */
static void
init_rq(const struct app_transfer_wq app_rq, struct rq_ctx_t *ctx)
{
	ctx->rq_number = app_rq.wq_num;
	ctx->rq_ring = (struct flexio_dev_wqe_rcv_data_seg *)app_rq.wq_ring_daddr;
	ctx->rq_dbr = (uint32_t *)app_rq.wq_dbr_daddr;
}

/*
 * Initialize the SQ context
 *
 * @app_sq [in]: SQ HW context
 * @ctx [out]: SQ context
 */
static void
init_sq(const struct app_transfer_wq app_sq, struct sq_ctx_t *ctx)
{
	ctx->sq_number = app_sq.wq_num;
	ctx->sq_ring = (union flexio_dev_sqe_seg *)app_sq.wq_ring_daddr;
	ctx->sq_dbr = (uint32_t *)app_sq.wq_dbr_daddr;

	ctx->sq_wqe_seg_idx = 0;
	ctx->sq_dbr++;
}

/*
 * Get next data buffer entry
 *
 * @dt_ctx [in]: Data transfer context
 * @dt_idx_mask [in]: Data transfer segment index mask
 * @log_dt_entry_sz [in]: Log of data transfer entry size
 * @return: Data buffer entry
 */
static void *
get_next_dte(struct dt_ctx_t *dt_ctx, uint32_t dt_idx_mask, uint32_t log_dt_entry_sz)
{
	uint32_t mask = ((dt_ctx->tx_buff_idx++ & dt_idx_mask) << log_dt_entry_sz);
	char *buff_p =  (char *) dt_ctx->sq_tx_buff;

	return buff_p + mask;
}

/*
 * Get next SQE from the SQ ring
 *
 * @sq_ctx [in]: SQ context
 * @sq_idx_mask [in]: SQ index mask
 * @return: pointer to next SQE
 */
static void *
get_next_sqe(struct sq_ctx_t *sq_ctx, uint32_t sq_idx_mask)
{
	return &sq_ctx->sq_ring[sq_ctx->sq_wqe_seg_idx++ & sq_idx_mask];
}

/*
 * Increase consumer index of the CQ,
 * Once a CQE is polled, the consumer index is increased.
 * Upon completing a CQ epoch, the HW owner bit is flipped.
 *
 * @cq_ctx [in]: CQ context
 * @cq_idx_mask [in]: CQ index mask which indicates when the CQ is full
 */
static void
step_cq(struct cq_ctx_t *cq_ctx, uint32_t cq_idx_mask)
{
	cq_ctx->cq_idx++;
	cq_ctx->cqe = &cq_ctx->cq_ring[cq_ctx->cq_idx & cq_idx_mask];
	/* check for wrap around */
	if (!(cq_ctx->cq_idx & cq_idx_mask))
		cq_ctx->cq_hw_owner_bit = !cq_ctx->cq_hw_owner_bit;

	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_dbr_cq_set_ci(cq_ctx->cq_dbr, cq_ctx->cq_idx);
}

/*
 * This is the main function of the L2 reflector device, called on each packet from l2_reflector_device_event_handler()
 * Packet are received from the RQ, processed by changing MAC addresses and transmitted to the SQ.
 *
 * @dtctx [in]: This thread context
 */
static void
process_packet(struct flexio_dev_thread_ctx *dtctx)
{
	uint32_t rq_wqe_idx;
	struct flexio_dev_wqe_rcv_data_seg *rwqe;
	uint32_t data_sz;
	char *rq_data;
	char *sq_data;
	union flexio_dev_sqe_seg *swqe;
	const uint16_t mss = 0, checksum = 0;
	char tmp;
	/* MAC address has 6 bytes: ff:ff:ff:ff:ff:ff */
	const int nb_mac_address_bytes = 6;


	/* Extract relevant data from CQE */
	rq_wqe_idx = flexio_dev_cqe_get_wqe_counter(dev_ctx.rqcq_ctx.cqe);
	data_sz = flexio_dev_cqe_get_byte_cnt(dev_ctx.rqcq_ctx.cqe);

	/* Get RQ WQE pointed by CQE */
	rwqe = &dev_ctx.rq_ctx.rq_ring[rq_wqe_idx & L2_RQ_IDX_MASK];

	/* Extract data (whole packet) pointed by RQ WQE */
	rq_data = flexio_dev_rwqe_get_addr(rwqe);

	/* Take next entry from data ring */
	sq_data = get_next_dte(&dev_ctx.dt_ctx, L2_DATA_IDX_MASK, L2_LOG_WQ_DATA_ENTRY_BSIZE);

	/* Copy received packet to sq_data as is */
	memcpy(sq_data, rq_data, data_sz);

	/* swap mac addresses */
	for (int byte = 0; byte < nb_mac_address_bytes; byte++) {
		tmp = sq_data[byte];
		sq_data[byte] = sq_data[byte + nb_mac_address_bytes];
		/* dst and src MACs are aligned one after the other in the ether header */
		sq_data[byte + nb_mac_address_bytes] = tmp;
	}

	/* Take first segment for SQ WQE (3 segments will be used) */
	swqe = get_next_sqe(&dev_ctx.sq_ctx, L2_SQ_IDX_MASK);

	/* Fill out 1-st segment (Control) */
	flexio_dev_swqe_seg_ctrl_set(swqe, dev_ctx.sq_ctx.sq_pi, dev_ctx.sq_ctx.sq_number,
				     MLX5_CTRL_SEG_CE_CQE_ON_CQE_ERROR, FLEXIO_CTRL_SEG_SEND_EN);

	/* Fill out 2-nd segment (Ethernet) */
	swqe = get_next_sqe(&dev_ctx.sq_ctx, L2_SQ_IDX_MASK);
	flexio_dev_swqe_seg_eth_set(swqe, mss, checksum, 0, NULL);

	/* Fill out 3-rd segment (Data) */
	swqe = get_next_sqe(&dev_ctx.sq_ctx, L2_SQ_IDX_MASK);
	flexio_dev_swqe_seg_mem_ptr_data_set(swqe, data_sz, dev_ctx.lkey, (uint64_t)sq_data);

	/* Send WQE is 4 WQEBBs need to skip the 4-th segment */
	swqe = get_next_sqe(&dev_ctx.sq_ctx, L2_SQ_IDX_MASK);

	/* Ring DB */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	dev_ctx.sq_ctx.sq_pi++;
	flexio_dev_qp_sq_ring_db(dtctx, dev_ctx.sq_ctx.sq_pi, dev_ctx.sq_ctx.sq_number);
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_dbr_rq_inc_pi(dev_ctx.rq_ctx.rq_dbr);
}

/*
 * Called by host to initialize the device context
 *
 * @data [in]: pointer to the device context from the host
 * @return: This function always returns 0
 */
__dpa_rpc__ uint64_t
l2_reflector_device_init(uint64_t data)
{
	struct l2_reflector_data *shared_data = (struct l2_reflector_data *)data;

	dev_ctx.lkey = shared_data->sq_data.wqd_mkey_id;
	init_cq(shared_data->rq_cq_data, &dev_ctx.rqcq_ctx);
	init_rq(shared_data->rq_data, &dev_ctx.rq_ctx);
	init_cq(shared_data->sq_cq_data, &dev_ctx.sqcq_ctx);
	init_sq(shared_data->sq_data, &dev_ctx.sq_ctx);

	dev_ctx.dt_ctx.sq_tx_buff = (void *)shared_data->sq_data.wqd_daddr;
	dev_ctx.dt_ctx.tx_buff_idx = 0;

	dev_ctx.is_initalized = 1;
	return 0;
}

/*
 * This function is called when a new packet is received to RQ's CQ.
 * Upon receiving a packet, the function will iterate over all received packets and process them.
 * Once all packets in the CQ are processed, the CQ will be rearmed to receive new packets events.
 */
void
__dpa_global__ l2_reflector_device_event_handler(uint64_t __unused arg0)
{
	struct flexio_dev_thread_ctx *dtctx;

	flexio_dev_get_thread_ctx(&dtctx);

	if (dev_ctx.is_initalized == 0)
		flexio_dev_thread_reschedule();

	while (flexio_dev_cqe_get_owner(dev_ctx.rqcq_ctx.cqe) != dev_ctx.rqcq_ctx.cq_hw_owner_bit) {
		__dpa_thread_fence(__DPA_MEMORY, __DPA_R, __DPA_R);
		process_packet(dtctx);
		step_cq(&dev_ctx.rqcq_ctx, L2_CQ_IDX_MASK);
	}
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_cq_arm(dtctx, dev_ctx.rqcq_ctx.cq_idx, dev_ctx.rqcq_ctx.cq_number);
	flexio_dev_thread_reschedule();
}
