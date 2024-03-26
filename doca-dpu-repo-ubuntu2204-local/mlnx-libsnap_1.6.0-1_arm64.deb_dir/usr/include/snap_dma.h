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

#ifndef SNAP_DMA_H
#define SNAP_DMA_H

#if !defined(__DPA)
#include <sys/uio.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <sys/queue.h>
#else
#include "../dpa/snap_dma_compat.h"
#endif

#include "snap_mr.h"
#include "snap_qp.h"
#include "snap_dpa_common.h"
#include "snap_dma_stat.h"

#define SNAP_DMA_Q_OPMODE        "SNAP_DMA_Q_OPMODE"
#define SNAP_DMA_Q_IOV_SUPP      "SNAP_DMA_Q_IOV_SUPP"
#define SNAP_DMA_Q_CRYPTO_SUPP   "SNAP_DMA_Q_CRYPTO_SUPP"
#define SNAP_DMA_Q_DBMODE        "SNAP_DMA_Q_DBMODE"

#define SNAP_DMA_Q_MAX_IOV_CNT		128
#define SNAP_DMA_Q_MAX_SGE_NUM		20
#define SNAP_DMA_Q_MAX_WR_CNT		128
#define SNAP_DMA_Q_POST_RECV_BUF_FACTOR	2

#define SNAP_CRYPTO_KEYTAG_SIZE              8

struct snap_dma_q;
struct snap_dma_completion;

/**
 * typedef snap_dma_rx_cb_t - receive callback
 * @q:        dma queue
 * @data:     received data. The buffer belongs to the queue, once the
 *            callback is completed the buffer content is going to
 *            be overwritten
 * @data_len: size of the received data
 * @imm_data: immediate data in the network order as defined in the IB spec
 *
 * The callback is called from within snap_dma_q_progress() when a new data
 * is received from the emulated device.
 *
 * The layout of the @data as well as the validity of @imm_data field depends
 * on the emulated device.  For example, in case of the NVMe emulation queue
 * @data will be a nvme sqe and @imm_data will be undefined.
 *
 * It is safe to initiate data transfers from within the callback. However
 * it is not safe to destroy or modify the dma queue.
 */
typedef void (*snap_dma_rx_cb_t)(struct snap_dma_q *q, const void *data,
		uint32_t data_len, uint32_t imm_data);

/**
 * typedef snap_dma_comp_cb_t - DMA operation completion callback
 * @comp:   user owned dma completion which was given  to the snap_dma_q_write()
 *          or to the snap_dma_q_read() function
 * @status: IBV_WC_SUCCESS (0) on success or anything else on error.
 *          See enum ibv_wc_status.
 *
 * The callback is called when dma operation is completed. It means
 * that either data has been successfully copied to the host memory or
 * an error has occurred.
 *
 * It is safe to initiate data transfers from within the callback. However
 * it is not safe to destroy or modify the dma queue.
 */
typedef void (*snap_dma_comp_cb_t)(struct snap_dma_completion *comp, int status);

/**
 * typedef free_dma_q_resources - Free DMA Q resources callback
 * @dpa_cq: CQ for which resources have to cleaned
 *
 * This cb is set when the user issues CQ destroy command and dma q is not
 * yet drained.
 * Cllabck will be invoked from the rx poller once the DMA Q has been
 * drained completely
 */
typedef void (*free_dma_q_resources)(void *dpa_cq);

/**
 * struct snap_dma_completion - completion handle and callback
 *
 * This structure should be allocated by the user and can be passed to communication
 * primitives. User has to initializes both fields of the structure.
 *
 * If snap_dma_q_write() or snap_dma_q_read() returns 0, this structure will be
 * in use until the DMA operation completes. When the DMA completes, @count
 * field is decremented by 1, and whenever it reaches 0 - the callback is called.
 *
 * Notes:
 *  - The same structure can be passed multiple times to communication functions
 *    without the need to wait for completion.
 *  - If the number of operations is smaller than the initial value of the counter,
 *    the callback will not be called at all, so it may be left undefined.
 */
struct snap_dma_completion {
	/** @func: callback function. See &typedef snap_dma_comp_cb_t */
	snap_dma_comp_cb_t func;
	/** @count: completion counter */
	int                count;
};

struct mlx5_dma_opaque;

struct snap_rx_completion {
	void *data;
	uint32_t imm_data;
	uint32_t byte_len;
	struct snap_dma_q *q;
};

struct snap_dv_dma_completion {
	int n_outstanding;
	void *read_payload;
	struct snap_dma_completion *comp;
};

enum snap_db_ring_flag {
	SNAP_DB_RING_BATCH = 0,
	SNAP_DB_RING_IMM   = 1,
	SNAP_DB_RING_API   = 2
};

struct snap_dv_qp {
	struct snap_hw_qp hw_qp;
	int n_outstanding;
	uint32_t opaque_lkey;
	uint32_t dpa_mkey;
	struct snap_dv_dma_completion *comps;
	/* used to hold GGA data */
	struct mlx5_dma_opaque     *opaque_buf;
	struct ibv_mr              *opaque_mr;
	/* true if tx db is in the non cacheable memory */
	bool tx_db_nc;
	enum snap_db_ring_flag db_flag;
	bool tx_need_ring_db;
	struct mlx5_wqe_ctrl_seg *ctrl;
	struct snap_dv_qp_stat stat;
};

struct snap_dma_ibv_qp {
	/* used when working in devx mode */
	struct snap_hw_cq dv_tx_cq;
	struct snap_hw_cq dv_rx_cq;
	struct snap_dv_qp dv_qp;

	struct snap_qp *qp;
	struct snap_cq *tx_cq;
	struct snap_cq *rx_cq;
	struct ibv_mr  *rx_mr;
	char           *rx_buf;
	int            mode;
	struct {
		struct snap_dpa_memh *rx_mr;
		struct snap_dpa_mkeyh *mkey;
	} dpa;
};

enum {
	SNAP_DMA_Q_IO_TYPE_IOV      = 0x1,
	SNAP_DMA_Q_IO_TYPE_ENCRYPTO = 0x2,
};

struct snap_dma_q_io_attr {
	int io_type;
	size_t len;

	/* for IOV TYPE IO */
	uint32_t *lkey;
	struct iovec *liov;
	int liov_cnt;
	uint32_t *rkey;
	struct iovec *riov;
	int riov_cnt;

	/* for ENCRYPTO IO */
	uint32_t dek_obj_id;
	uint32_t enc_order;
	uint64_t xts_initial_tweak;
};

enum snap_dma_q_mode {
	SNAP_DMA_Q_MODE_AUTOSELECT = 0,
	SNAP_DMA_Q_MODE_VERBS = 1,
	SNAP_DMA_Q_MODE_DV = 2,
	SNAP_DMA_Q_MODE_GGA = 3
};

struct snap_dma_q_ops {
	enum snap_dma_q_mode mode;

	int (*write)(struct snap_dma_q *q, void *src_buf, size_t len,
		     uint32_t lkey, uint64_t dstaddr, uint32_t rmkey,
		     struct snap_dma_completion *comp);
	int (*writev2v)(struct snap_dma_q *q, struct snap_dma_q_io_attr *io_attr,
		     struct snap_dma_completion *comp, int *n_bb);
	int (*writec)(struct snap_dma_q *q, struct snap_dma_q_io_attr *io_attr,
		     struct snap_dma_completion *comp, int *n_bb);
	int (*write_short)(struct snap_dma_q *q, void *src_buf, size_t len,
			   uint64_t dstaddr, uint32_t rmkey, int *n_bb);
	int (*read)(struct snap_dma_q *q, void *dst_buf, size_t len,
		    uint32_t lkey, uint64_t srcaddr, uint32_t rmkey,
		    struct snap_dma_completion *comp);
	int (*readv2v)(struct snap_dma_q *q, struct snap_dma_q_io_attr *io_attr,
		    struct snap_dma_completion *comp, int *n_bb);
	int (*readc)(struct snap_dma_q *q, struct snap_dma_q_io_attr *io_attr,
		    struct snap_dma_completion *comp, int *n_bb);
	int (*read_short)(struct snap_dma_q *q, void *dst_buf,
			 size_t len, uint64_t srcaddr, uint32_t rmkey,
			 struct snap_dma_completion *comp);
	int (*send_completion)(struct snap_dma_q *q, void *src_buf,
			size_t len, int *n_bb);
	int (*send)(struct snap_dma_q *q, void *in_buf, size_t in_len,
		    uint64_t addr, int len, uint32_t key,
		    int *n_bb, uint32_t *imm);
	int (*progress_tx)(struct snap_dma_q *q, int max_tx_comp);
	void (*complete_tx)(struct snap_dma_q *q);
	int (*progress_rx)(struct snap_dma_q *q);
	int (*flush)(struct snap_dma_q *q);
	int (*flush_nowait)(struct snap_dma_q *q, struct snap_dma_completion *comp, int *n_bb);
	bool (*empty)(struct snap_dma_q *q);
	int (*arm)(struct snap_dma_q *q);
	int (*poll)(struct snap_dma_q *q);
	int (*poll_rx)(struct snap_dma_q *q, struct snap_rx_completion *rx_completions, int max_completions);
	int (*poll_tx)(struct snap_dma_q *q, struct snap_dma_completion **comp, int max_completions);
	const struct snap_dv_qp_stat* (*stat)(const struct snap_dma_q *q);
};

struct snap_dma_q_iov_ctx {
	struct snap_dma_q *q;

	int n_bb;

	struct snap_dma_completion comp;
	void *uctx;

	TAILQ_ENTRY(snap_dma_q_iov_ctx) entry;
};

struct snap_dma_q_crypto_ctx {
	struct snap_dma_q *q;

	struct snap_indirect_mkey *l_klm_mkey;
	struct snap_indirect_mkey *r_klm_mkey;

	struct snap_dma_completion comp;
	void *uctx;

	TAILQ_ENTRY(snap_dma_q_crypto_ctx) entry;
};

/* dma inline recv ctx, only used for VERBS mode */
struct snap_dma_q_ir_ctx {
	struct snap_dma_q *q;

	void *buf;
	uint32_t mkey;
	struct snap_dma_completion comp;

	void *user_buf;
	size_t len;
	void *uctx;

	TAILQ_ENTRY(snap_dma_q_ir_ctx) entry;
};

struct snap_dma_fw_qp {
	struct snap_dma_ibv_qp fw_qp;
	struct ibv_qp fake_verbs_qp;
	bool use_devx;
};

/**
 * struct snap_dma_q - DMA queue
 *
 * DMA queue is a connected pair of the IB queue pais (QPs). One QP
 * can be passed to the FW emulation objects such as NVMe
 * submission queue or VirtIO queue. Another QP can be used to:
 *
 *  - receive protocol related data. E.x. NVMe submission queue entry or SGL
 *  - send completion notifications. E.x. NVMe completion queue entry
 *  - read/write data from/to the host memory
 *
 * DMA queue is not thread safe. A caller must take care of the proper locking
 * if DMA queue is used by different threads.
 * However it is guaranteed that each queue is independent of others. It means
 * that no locking is needed as long as each queue is always used in the same
 * thread.
 */
struct snap_dma_q {
	/* private: */
	/* TODO: for dpa/ep we don't need all fields. Group all frequently used
	 * fields so that they all fit into 2 cachelines
	 */
	struct snap_dma_ibv_qp sw_qp;
	int                    tx_available;
	int                    tx_qsize;
	int                    tx_elem_size;
	int                    rx_elem_size;
	snap_dma_rx_cb_t       rx_cb;

	const struct snap_dma_q_ops  *ops;

	struct snap_dma_q_iov_ctx *iov_ctx;
	struct snap_dma_q_crypto_ctx *crypto_ctx;
	void *ir_buf;
	struct ibv_mr *ir_mr;
	struct snap_dma_q_ir_ctx *ir_ctx;

	TAILQ_HEAD(, snap_dma_q_iov_ctx) free_iov_ctx;

	TAILQ_HEAD(, snap_dma_q_crypto_ctx) free_crypto_ctx;

	TAILQ_HEAD(, snap_dma_q_ir_ctx) free_ir_ctx;

	SLIST_ENTRY(snap_dma_q) entry;

	struct snap_dma_q_ops  *custom_ops;
	struct snap_dma_worker *worker;

	/* public: */
	/** @uctx:  user supplied context */
	void                  *uctx;
	bool                  iov_support;
	bool                  crypto_support;
	bool                  no_events;
	int                   rx_qsize;

#if !defined(__DPA)
	pthread_mutex_t lock;
	bool destroy_done;
	int flush_count;
	free_dma_q_resources free_dma_q_resources_cb;
#endif
	struct snap_dma_fw_qp *fw_qp;
	int n_crypto_ctx;
	int crypto_place;
};

enum {
	SNAP_DMA_Q_DPA_MODE_NONE = 0,
	SNAP_DMA_Q_DPA_MODE_POLLING,
	SNAP_DMA_Q_DPA_MODE_EVENT,
	SNAP_DMA_Q_DPA_MODE_TRIGGER,
	SNAP_DMA_Q_DPA_MODE_MSIX_TRIGGER
};

enum {
	SNAP_DMA_Q_CRYPTO_ON_DEST,
	SNAP_DMA_Q_CRYPTO_ON_SRC
};

enum {
	/* wait for umr completion before using crypto key */
	SNAP_DMA_Q_CRYPTO_UMR_WAIT,
	/* immediately use crypto key, add small fence to the wqe */
	SNAP_DMA_Q_CRYPTO_UMR_FENCE,
	/* use separate qp to post umr, wait for umr completion. The qp
	 * can be shared with several data qps
	 */
	SNAP_DMA_Q_CRYPTO_UMR_QP
};

/* this is a reasonable default value. It takes a significant time to setup
 * a crypto context, which adds up when we have many dma queues with crypto
 */
#define SNAP_DMA_Q_CRYPTO_CTX_MAX 64
struct snap_dma_q_crypto_attr {
	int crypto_ctx_max;  // max number of crypto contexts per qp
	int crypto_place;
	int crypto_umr_engine; // how to post crypto umr
	int crypto_block_size;
	struct snap_dma_q *umr_q;
};

/**
 * struct snap_dma_q_create_attr - DMA queue creation attributes
 * @tx_qsize:     send queue size of the software qp
 * @tx_elem_size: size of the completion. The size is emulation specific.
 *                For example 16 bytes for NVMe
 * @rx_qsize:     receive queue size of the software qp. In case if the qp is
 *                used with the NVMe SQ, @rx_qsize must be no less than the
 *                SQ size.
 * @rx_elem_size: size of the receive element. The size is emulation specific.
 *                For example 64 bytes for NVMe
 * @uctx:         user supplied context
 * @mode:         choose dma implementation:
 *                 SNAP_DMA_Q_MODE_AUTOSELECT - select best option automatically
 *                 SNAP_DMA_Q_MODE_VERBS - verbs, standard API, safest, slowest
 *                 SNAP_DMA_Q_MODE_DV    - dv, direct hw access, faster than verbs
 *                 SNAP_DMA_Q_MODE_GGA   - dv, plus uses hw dma engine directly to
 *                                         do rdma read or write. Fastest, best bandwidth.
 *                Mode choice can be overridden at runtime by setting SNAP_DMA_Q_OPMODE
 *                environment variable: 0 - autoselect, 1 - verbs, 2 - dv, 3 - gga.
 * @rx_cb:        receive callback. See &typedef snap_dma_rx_cb_t
 * @iov_enable:   enable/disable this dma queue to use readv/writev API
 * @crypto_enable:enable/disable this dma queue to use crypto rw API
 * @crypto_attr:  parameters that configure crypto engine
 * @comp_channel: receive and DMA completion channel. See
 *                man ibv_create_comp_channel
 * @comp_vector:  completion vector
 * @comp_context: completion context that will be returned by the
 *                ibv_get_cq_event(). See man ibv_get_cq_event
 * @sw_use_devx:  use DEVX to create CQs and QP for sw instead of mlx5dv/verbs api. Works
 *                only if @mode is dv or gga
 * @fw_use_devx:  use DEVX to create CQs and QP for fw instead of mlx5dv/verbs api. Works
 *                only if @mode is dv or gga
 * @wk:           if not NULL, the dma_queue will be attached to the given
 *                worker. In such case worker progress/polling functions must
 *                be used instead of queue progress/polling functions.
 * @dpa_mode:     if non zero, create dma queue on the DPA. Valid only with snap_dma_ep_create()
 *                Possible values are:
 *                SNAP_DMA_Q_DPA_MODE_NONE or 0 - regular queue is created
 *                SNAP_DMA_Q_DPA_MODE_POLLING   - QP and CQ are in dpa memory.
 *                                                CQs are bound to dummy dpa EQ
 *                SNAP_DMA_Q_DPA_MODE_EVENT     - QP and CQ are in dpa memory.
 *                                                CQs are bound to the @dpa_thread
 *                SNAP_DMA_Q_DPA_MODE_TRIGGER   - QP is in host memory, CQs are in dpa memory
 *                                                CQs are bound to the @dpa_thread.
 *                SNAP_DMA_Q_DPA_MODE_MSIX_TRIGGER - QP is in host memory, CQs are in dpa memory,
 *                                                   CQs are bound to the emulated device EQ given
 *                                                   by @emu_dev_eqn.
 *                In all DPA modes, CQs have doorbell records on host memory. It means
 *                that they can be also armed from host.
 *                Basically polling mode is for polling dpa application, event
 *                mode is used to wake up and schedule dpa thread, trigger mode
 *                is used to wake up and schedule dpa thread from the dpu side.
 *                MSIX trigger mode can be used to raise interrupt from the DPU
 *                side when coordination with DPA is needed.
 * @dpa_proc:     snap dpa process context. Must be valid if @dpa_mode is SNAP_DMA_Q_DPA_MODE_POLLING
 *                or SNAP_DMA_Q_DPA_MODE_MSIX_TRIGGER.
 * @dpa_thread:   snap dpa thread context. Must be valid if @dpa_mode is
 *                SNAP_DMA_Q_DPA_MODE_EVENT or SMAP_DMA_Q_DPA_MODE_TRIGGER
 * @use_emu_dev_eqn: If true CQs will be connected to the emulated device msix EQ
 *                given by @emu_dev_eqn. CQs are always armed and completions will
 *                trigger MSIX interrupt.
 * @emu_dev_eqn:  emulated device msix EQ number
 */
struct snap_dma_q_create_attr {
	uint32_t tx_qsize;
	uint32_t tx_elem_size;
	uint32_t rx_qsize;
	uint32_t rx_elem_size;
	void  *uctx;
	int   mode;
	bool  iov_enable;
	bool  crypto_enable;
	snap_dma_rx_cb_t rx_cb;

	struct ibv_comp_channel *comp_channel;
	int                      comp_vector;
	void                    *comp_context;

	bool sw_use_devx;
	bool fw_use_devx;
	struct snap_dma_worker *wk;

	int  dpa_mode;
	union {
		struct snap_dpa_ctx *dpa_proc;
		struct snap_dpa_thread *dpa_thread;
	};

	bool use_emu_dev_eqn;
	uint32_t emu_dev_eqn;

	struct snap_dma_q_crypto_attr crypto_attr;
};

/* TODO add support for worker mode single and SRQ*/
enum snap_dma_worker_mode {
	/* shared cq size is exp_queue_num * exp_queue_rx_size */
	SNAP_DMA_WORKER_MODE_SHARED_CQ,
	SNAP_DMA_WORKER_MODE_SHARED_CQ_TX_ONLY,
	SNAP_DMA_WORKER_MODE_SHARED_CQ_RX_ONLY,

	/* TODO: Remove the below? */

	/* cq per qp, suitable for small numbers of qps */
	SNAP_DMA_WORKER_MODE_SINGLE,
	/* cq pool, rx cq size is exp_queue_num * exp_queue_rx_size */
	SNAP_DMA_WORKER_MODE_CQ_POOL,
	/* use to receive */
	SNAP_DMA_WORKER_MODE_SRQ
};

struct snap_dma_worker_create_attr {
	enum snap_dma_worker_mode mode;
	int exp_queue_num; /* hint to the worker: how many queues it is going to serve */
	int exp_queue_rx_size; /* hint to the worker: queue rx size */
	int id;
};

struct snap_dma_worker {
	/* used when working in devx mode */
	struct snap_hw_cq dv_tx_cq;
	struct snap_hw_cq dv_rx_cq;

	struct snap_cq *rx_cq;
	struct snap_cq *tx_cq;
	enum snap_dma_worker_mode mode;
	int max_queues;

	SLIST_HEAD(, snap_dma_q) pending_dbs;
	struct snap_dma_q *queues[0];
};
struct snap_dma_worker *snap_dma_worker_create(struct ibv_pd *pd,
		const struct snap_dma_worker_create_attr *attr);
void snap_dma_worker_destroy(struct snap_dma_worker *wk);
int snap_dma_worker_flush(struct snap_dma_worker *wk);

/* progress receives, dma_q rx callbacks will be called */
int snap_dma_worker_progress_rx(struct snap_dma_worker *wk);
/* progress tx, dma_q tx completion callbacks will be called */
int snap_dma_worker_progress_tx(struct snap_dma_worker *wk);

struct snap_dma_q *snap_dma_q_create(struct ibv_pd *pd,
		const struct snap_dma_q_create_attr *attr);
void snap_dma_q_destroy(struct snap_dma_q *q);
void snap_dma_ep_destroy(struct snap_dma_q *q);
int snap_dma_q_write(struct snap_dma_q *q, void *src_buf, size_t len,
		uint32_t lkey, uint64_t dstaddr, uint32_t rmkey,
		struct snap_dma_completion *comp);
int snap_dma_q_writev2v(struct snap_dma_q *q,
		uint32_t *lkey, struct iovec *src_iov, int src_iovcnt,
		uint32_t *rkey, struct iovec *dst_iov, int dst_iovcnt,
		bool share_src_mkey, bool share_dst_mkey,
		struct snap_dma_completion *comp);
int snap_dma_q_writec(struct snap_dma_q *q, void *src_buf, uint32_t lkey,
		struct iovec *iov, int iov_cnt, uint32_t rmkey,
		uint32_t dek_obj_id, uint64_t tweak, struct snap_dma_completion *comp);
int snap_dma_q_writev2vc(struct snap_dma_q *q,
		uint32_t *lkey, struct iovec *src_iov, int src_iov_cnt,
		uint32_t rmkey, struct iovec *dst_iov, int dst_iov_cnt,
		uint32_t dek_obj_id, uint64_t tweak, struct snap_dma_completion *comp);
int snap_dma_q_write_short(struct snap_dma_q *q, void *src_buf, size_t len,
		uint64_t dstaddr, uint32_t rmkey);
int snap_dma_q_read(struct snap_dma_q *q, void *dst_buf, size_t len,
		uint32_t lkey, uint64_t srcaddr, uint32_t rmkey,
		struct snap_dma_completion *comp);
int snap_dma_q_readv2v(struct snap_dma_q *q,
		uint32_t *lkey, struct iovec *dst_iov, int dst_iovcnt,
		uint32_t *rkey, struct iovec *src_iov, int src_iovcnt,
		bool share_dst_mkey, bool share_src_mkey,
		struct snap_dma_completion *comp);
int snap_dma_q_readc(struct snap_dma_q *q, void *dst_buf, uint32_t lkey,
		struct iovec *iov, int iov_cnt, uint32_t rmkey,
	    uint32_t dek_obj_id, uint64_t tweak, struct snap_dma_completion *comp);
int snap_dma_q_read_short(struct snap_dma_q *q, void *dst_buf,
		    size_t len, uint64_t srcaddr, uint32_t rmkey,
		    struct snap_dma_completion *comp);
int snap_dma_q_send_completion(struct snap_dma_q *q, void *src_buf, size_t len);
int snap_dma_q_progress(struct snap_dma_q *q);
int snap_dma_q_poll_rx(struct snap_dma_q *q, struct snap_rx_completion *rx_completions, int max_completions);
int snap_dma_q_poll_tx(struct snap_dma_q *q, struct snap_dma_completion **comp, int max_completions);
int snap_dma_q_flush(struct snap_dma_q *q);
int snap_dma_q_flush_nowait(struct snap_dma_q *q, struct snap_dma_completion *comp);
bool snap_dma_q_empty(struct snap_dma_q *q);
int snap_dma_q_arm(struct snap_dma_q *q);
struct ibv_qp *snap_dma_q_get_fw_qp(struct snap_dma_q *q);
struct snap_dma_q *snap_dma_ep_create(struct ibv_pd *pd,
	const struct snap_dma_q_create_attr *attr);
int snap_dma_ep_connect(struct snap_dma_q *q1, struct snap_dma_q *q2);
int snap_dma_ep_connect_remote_qpn(struct snap_dma_q *q1, int remote_qp2_num);
int snap_dma_q_send(struct snap_dma_q *q, void *in_buf, size_t in_len,
		uint64_t addr, size_t len, uint32_t key, uint32_t *imm);
int snap_dma_q_post_recv(struct snap_dma_q *q);

int snap_dma_q_modify_to_err_state(struct snap_dma_q *q);

struct snap_dma_ep_copy_cmd {
	struct snap_dpa_cmd base;
	struct snap_dma_q q;
};

static inline uint32_t snap_dma_q_dpa_mkey(struct snap_dma_q *q)
{
	return q->sw_qp.dv_qp.dpa_mkey;
}

int snap_dma_ep_dpa_copy_sync(struct snap_dpa_thread *thr, struct snap_dma_q *q);

/**
 * snap_dma_q_ctx - get queue context
 * @q: dma queue
 *
 * Returns: dma queue context
 */
static inline void *snap_dma_q_ctx(struct snap_dma_q *q)
{
	return q->uctx;
}

static inline const struct snap_dv_qp_stat *snap_dma_q_stat(const struct snap_dma_q *q)
{
	return q->ops->stat ? q->ops->stat(q) : NULL;
}

static inline int snap_dma_q_dv_get_tx_avail_max(struct snap_dma_q *q)
{
	/**
	 * if wqe_cnt > cqe_cnt we can have cq overrun because most operations take
	 * exactly one wqe. Notable exceptions are inline sends (>48b) and sgls.
	 * If qp is not created via devx, rdma-core always allocates enough wqes
	 * to accommodate worst possible scenrio. For example if inline is 64b it
	 * will allocate 2xwqe(s) than the requested tx_qsize.
	 *
	 * Limit number of outstanding wqes by cq size, in the future
	 * consider counting cq space separately.
	 */
	return snap_min(q->sw_qp.dv_qp.hw_qp.sq.wqe_cnt, q->sw_qp.dv_tx_cq.cqe_cnt);
}

/* how many tx and rx completions to process during a single progress call */
#define SNAP_DMA_MAX_TX_COMPLETIONS  128
#define SNAP_DMA_MAX_RX_COMPLETIONS  128

/* Number of completions for Shared RX CQ in single poll cycle */
#define SNAP_DMA_MAX_SHARED_RX_CQ_COMPLETIONS	64

/* align start of the receive buffer on 4k boundary */
#define SNAP_DMA_RX_BUF_ALIGN    4096
#define SNAP_DMA_BUF_ALIGN       4096

/* create params */
#define SNAP_DMA_FW_QP_MIN_SEND_WR 32

/* INIT state params */
#define SNAP_DMA_QP_PKEY_INDEX  0
#define SNAP_DMA_QP_PORT_NUM    1

/* RTR state params */
#define SNAP_DMA_QP_RQ_PSN              0x4242
#define SNAP_DMA_QP_MAX_DEST_RD_ATOMIC      16
#define SNAP_DMA_QP_RNR_TIMER               12
#define SNAP_DMA_QP_HOP_LIMIT               64
#define SNAP_DMA_QP_GID_INDEX                0

/* RTS state params */
#define SNAP_DMA_QP_TIMEOUT            14
#define SNAP_DMA_QP_RETRY_COUNT         7
#define SNAP_DMA_QP_RNR_RETRY           7
#define SNAP_DMA_QP_MAX_RD_ATOMIC      16
#define SNAP_DMA_QP_SQ_PSN         0x4242

#endif
