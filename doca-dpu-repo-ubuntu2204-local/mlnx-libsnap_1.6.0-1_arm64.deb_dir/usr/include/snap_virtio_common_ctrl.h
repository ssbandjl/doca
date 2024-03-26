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

#ifndef SNAP_VIRTIO_COMMON_CTRL_H
#define SNAP_VIRTIO_COMMON_CTRL_H

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include "snap.h"
#include "snap_virtio_common.h"
#include "snap_poll_groups.h"

struct snap_virtio_ctrl;
struct snap_virtio_ctrl_queue;

enum snap_virtio_ctrl_type {
	SNAP_VIRTIO_BLK_CTRL,
	SNAP_VIRTIO_NET_CTRL,
	SNAP_VIRTIO_FS_CTRL,
};

/*
 * Device status field according to virtio spec v1.1 (section 2.1)
 *
 * The virtio device state is discussed between device and driver
 * over the `device_status` PCI bar register, in a "bitmask" mode;
 * a.k.a multiple "statuses" can be configured simultaneously.
 *
 * Full description of statuses can be found on virtio spec ducomentation.
 *
 * NOTE: RESET status is unique, as instead of raising a bit in register,
 *       driver *unsets* all bits on register.
 */
enum snap_virtio_common_device_status {
	SNAP_VIRTIO_DEVICE_S_RESET = 0,
	SNAP_VIRTIO_DEVICE_S_ACKNOWLEDGE = 1 << 0,
	SNAP_VIRTIO_DEVICE_S_DRIVER = 1 << 1,
	SNAP_VIRTIO_DEVICE_S_DRIVER_OK = 1 << 2,
	SNAP_VIRTIO_DEVICE_S_FEATURES_OK = 1 << 3,
	SNAP_VIRTIO_DEVICE_S_DEVICE_NEEDS_RESET = 1 << 6,
	SNAP_VIRTIO_DEVICE_S_FAILED = 1 << 7,
};

/*
 * Driver may choose to reset device for numerous reasons:
 * during initialization, on error, or during FLR.
 * Driver executes reset by writing `0` to `device_status` bar register.
 * According to virtio v0.95 spec., driver is not obligated to wait
 * for device to finish the RESET command, which may cause race conditions
 * to occur between driver and controller.
 * Issue is solved by using the extra internal `reset` bit:
 *  - FW set bit to `1` on driver reset.
 *  - Controller set it back to `0` once finished.
 */
#define SNAP_VIRTIO_CTRL_RESET_DETECTED(vctrl) \
			(vctrl->bar_curr->reset)

#define SNAP_VIRTIO_CTRL_FLR_DETECTED(vctrl) \
		(!vctrl->bar_curr->enabled)

/*
 * DRIVER_OK bit indicates that the driver is set up and ready to drive the
 * device. Only at this point, device is considered "live".
 * Prior to that, it is not promised that any driver resource is available
 * for the device to use.
 */
#define SNAP_VIRTIO_CTRL_LIVE_DETECTED(vctrl) \
		!!(vctrl->bar_curr->status & SNAP_VIRTIO_DEVICE_S_DRIVER_OK)

/**
 * enum snap_virtio_ctrl_state - Virtio controller internal state
 *
 *  @STOPPED: All on-demand resources (virtqueues) are cleaned.
 *    Can be reached from various contexts:
 *    - initial state after controller creation.
 *    - state to move after error/flr detection.
 *    - state to move when closing application. In this case we must update
 *      host driver we are no longer operational by raising DEVICE_NEEDS_RESET
 *      bit in `device_status`.
 *    - state to move after virtio RESET detection. In this case we must update
 *      FW ctrl is stopped by writing back `0` to `device_status`.
 *
 *  @STARTED: Controller is live and ready to handle IO requests. All
 *    requested on-demand resources (virtqueues) are created successfully.
 *
 *  @SUSPENED: All enabled queues are flushed and suspended. Internal controller
 *    state will stay constant. DMA access to the host memory is stopped. The
 *    state is equivalent of doing quiesce+freeze in live migration terms.
 *    In order to do a safe shutdown, application should put controller in the
 *    suspended state before controller is stopped.
 *    NOTE: going to the suspened state is an async operation. Reverse operation
 *    (resume) is a sync operation.
 *
 *  @SUSPENDING: indicates that suspend operation is in progress
 *
 *  Normal flow:
 *  STOPPED -> STARTED - [SUSPENDING] -> SUSPENDED -> STOPPED
 *
 *  Allowed transitions:
 *  STOPPED -> STARTED, SUSPENDED
 *  STARTED - [SUSPENDING] -> SUSPENDED
 *	  -> STOPPED  NOTE: this is not a safe transition. If there is outstanding
 *			    io, the controller may crash.
 *  SUSPENDED -> STOPPED, STARTED
 */
enum snap_virtio_ctrl_state {
	SNAP_VIRTIO_CTRL_STOPPED,
	SNAP_VIRTIO_CTRL_STARTED,
	SNAP_VIRTIO_CTRL_SUSPENDED,
	SNAP_VIRTIO_CTRL_SUSPENDING
};

struct snap_virtio_ctrl_bar_cbs {
	int (*validate)(void *cb_ctx);
	int (*start)(void *cb_ctx);
	int (*stop)(void *cb_ctx);
	int (*num_vfs_changed)(void *cb_ctx, uint16_t new_numvfs);
	int (*pre_flr)(void *cb_ctx);
	int (*post_flr)(void *cb_ctx);
};

struct snap_virtio_ctrl_attr {
	struct ibv_context *context;
	enum snap_virtio_ctrl_type type;
	enum snap_pci_type pci_type;
	int pf_id;
	int vf_id;
	bool event;
	void *cb_ctx;
	struct snap_virtio_ctrl_bar_cbs *bar_cbs;
	struct ibv_pd *pd;
	uint32_t npgs;
	bool force_in_order;
	bool suspended;
	bool recover;
	bool vf_dynamic_msix_supported;
	bool force_recover;
	bool db_cq_map_supported;
	bool eq_in_sw_supported;
};

struct snap_virtio_ctrl_queue {
	struct snap_virtio_ctrl *ctrl;
	int index;
	struct snap_pg *pg;
	struct snap_pg_q_entry pg_q;
	bool log_writes_to_host;

	TAILQ_ENTRY(snap_virtio_ctrl_queue) entry;
	int thread_id;
};

struct snap_virtio_ctrl_queue_counter {
	uint64_t total;
	uint64_t success;
	uint64_t fail;
	uint64_t unordered;
	uint64_t merged_desc;
	uint64_t long_desc_chain;
	uint64_t large_in_buf;

};

/**
 * struct virtq_cmd_ctrs - virtq commands counters
 * @outstanding_total:		active commands - sent from host to DPU (new incoming commands)
 * @outstanding_in_bdev:	active commands - sent to back-end device
 * @outstanding_to_host:	active commands - sent to host (e.g. completions or fetch descriptors)
 * @fatal:			fatal commands counter
 */
struct snap_virtio_ctrl_queue_out_counter {
	uint32_t outstanding_total;
	uint32_t outstanding_in_bdev;
	uint32_t outstanding_to_host;
	uint32_t fatal;
};

struct snap_virtio_ctrl_queue_stats {
	struct snap_virtio_ctrl_queue_counter read;
	struct snap_virtio_ctrl_queue_counter write;
	struct snap_virtio_ctrl_queue_counter flush;
	struct snap_virtio_ctrl_queue_out_counter outstanding;
};

struct snap_virtio_ctrl_queue_state;

struct snap_virtio_queue_ops {
	struct snap_virtio_ctrl_queue *(*create)(struct snap_virtio_ctrl *ctrl,
						 int index);
	void (*destroy)(struct snap_virtio_ctrl_queue *queue);
	int (*progress)(struct snap_virtio_ctrl_queue *queue);
	void (*start)(struct snap_virtio_ctrl_queue *queue);
	void (*suspend)(struct snap_virtio_ctrl_queue *queue);
	bool (*is_suspended)(struct snap_virtio_ctrl_queue *queue);
	int (*resume)(struct snap_virtio_ctrl_queue *queue);
	int (*get_state)(struct snap_virtio_ctrl_queue *queue,
			 struct snap_virtio_ctrl_queue_state *state);
	const struct snap_virtio_ctrl_queue_stats *
			(*get_io_stats)(struct snap_virtio_ctrl_queue *queue);
	bool  (*is_admin)(struct snap_virtio_ctrl_queue *queue);
};

struct snap_virtio_ctrl_bar_ops {
	struct snap_virtio_device_attr *(*create)(struct snap_virtio_ctrl *ctrl);
	void (*destroy)(struct snap_virtio_device_attr *ctrl);
	void (*copy)(struct snap_virtio_device_attr *orig,
		     struct snap_virtio_device_attr *copy);
	int (*update)(struct snap_virtio_ctrl *ctrl,
		      struct snap_virtio_device_attr *attr);
	int (*modify)(struct snap_virtio_ctrl *ctrl,
		      uint64_t mask, struct snap_virtio_device_attr *attr);
	struct snap_virtio_queue_attr *(*get_queue_attr)(
			struct snap_virtio_device_attr *vbar, int index);
	size_t (*get_state_size)(struct snap_virtio_ctrl *ctrl);
	int (*get_state)(struct snap_virtio_ctrl *ctrl,
			 struct snap_virtio_device_attr *attr, void *buf,
			 size_t len);
	int (*set_state)(struct snap_virtio_ctrl *ctrl,
			 struct snap_virtio_device_attr *attr,
			 const struct snap_virtio_ctrl_queue_state *queue_state,
			 const void *buf, int len);
	void (*dump_state)(struct snap_virtio_ctrl *ctrl, const void *buf, int len);
	bool (*queue_attr_valid)(struct snap_virtio_device_attr *attr);
	int (*get_attr)(struct snap_virtio_ctrl *ctrl, struct snap_virtio_device_attr *attr);

};

struct snap_virtio_ctrl {
	enum snap_virtio_ctrl_type type;
	enum snap_virtio_ctrl_state state;
	pthread_mutex_t progress_lock;
	struct snap_device *sdev;
	size_t max_queues;
	size_t enabled_queues;
	struct snap_virtio_ctrl_queue **queues;
	struct snap_virtio_queue_ops *q_ops;
	void *cb_ctx; /* bar callback context */
	struct snap_virtio_ctrl_bar_cbs bar_cbs;
	struct snap_virtio_ctrl_bar_ops *bar_ops;
	struct snap_virtio_device_attr *bar_curr;
	struct snap_virtio_device_attr *bar_prev;
	struct ibv_pd *lb_pd;
	struct snap_pg_ctx pg_ctx;
	bool log_writes_to_host;
	struct snap_channel *lm_channel;
	/* true if reset was requested while some queues are not suspended */
	bool pending_reset;
	bool ignore_reset;
	/* true if completion (commands handled by queues) should be sent in order */
	bool force_in_order;
	/* true if FLR was requested */
	bool pending_flr;
	struct snap_device_attr sdev_attr;
	int lm_state;
	struct snap_cross_mkey *xmkey;
	bool is_quiesce;
	struct snap_vq_cmd *quiesce_cmd;
	/* true if ctrl resume was requested while ctrl was still suspending */
	bool pending_resume;
	struct snap_dp_bmap *dp_map;
	struct snap_cross_mkey *pf_xmkey;
	uint16_t spec_version;
};

bool snap_virtio_ctrl_is_stopped(struct snap_virtio_ctrl *ctrl);
int snap_virtio_ctrl_start(struct snap_virtio_ctrl *ctrl);
int snap_virtio_ctrl_stop(struct snap_virtio_ctrl *ctrl);

bool snap_virtio_ctrl_is_configurable(struct snap_virtio_ctrl *ctrl);

bool snap_virtio_ctrl_is_suspended(struct snap_virtio_ctrl *ctrl);
int snap_virtio_ctrl_suspend(struct snap_virtio_ctrl *ctrl);
int snap_virtio_ctrl_resume(struct snap_virtio_ctrl *ctrl);

bool snap_virtio_ctrl_critical_bar_change_detected(struct snap_virtio_ctrl *ctrl);
void snap_virtio_ctrl_progress(struct snap_virtio_ctrl *ctrl);
void snap_virtio_ctrl_progress_lock(struct snap_virtio_ctrl *ctrl);
void snap_virtio_ctrl_progress_unlock(struct snap_virtio_ctrl *ctrl);
int snap_virtio_ctrl_io_progress(struct snap_virtio_ctrl *ctrl);
int snap_virtio_ctrl_pg_io_progress(struct snap_virtio_ctrl *ctrl, int pg_id);
int snap_virtio_ctrl_open(struct snap_virtio_ctrl *ctrl,
			  struct snap_virtio_ctrl_bar_ops *bar_ops,
			  struct snap_virtio_queue_ops *q_ops,
			  struct snap_context *sctx,
			  const struct snap_virtio_ctrl_attr *attr);
void snap_virtio_ctrl_close(struct snap_virtio_ctrl *ctrl);

/* live migration support */

/**
 * Virtio Controller State
 *
 * The virtio controller state consists of pci_common, device and queue
 * configuration sections.
 *
 * Device configuration and part of the queue configuration are controller
 * specific and should be filled by the controller implementation.
 *
 * Controller implementation is also going to be responsible for the restoring
 * device specific state and queues.
 *
 * State format:
 * <global_hdr><section_hdr><section>...<section_hdr><section>
 *
 * Each header and section are in the little endian (x86) order.
 */

/**
 * struct snap_virtio_ctrl_section - state section header
 *
 * @len:   section length, including section header
 * @name:  symbolic section name
 */
struct snap_virtio_ctrl_section {
	uint16_t   len;
	char       name[16];
} __attribute__((packed));

/**
 * struct snap_virtio_ctrl_common_state - pci_common state
 *
 * The struct defines controller pci_common state as described
 * in the virtio spec.
 * NOTE: that device and driver features bits are expanded
 *
 * @ctlr_state:  this is an internal controller state. We keep it in order to
 *	       validate state restore operation.
 */
struct snap_virtio_ctrl_common_state {
	uint32_t device_feature_select;
	uint64_t device_feature;
	uint32_t driver_feature_select;
	uint64_t driver_feature;
	uint16_t msix_config;
	uint16_t num_queues;
	uint16_t queue_select;
	uint8_t device_status;
	uint8_t config_generation;

	enum snap_virtio_ctrl_state ctrl_state;
} __attribute__((packed));

/**
 * struct snap_virtio_ctrl_queue_state - queue state
 *
 * The struct defines controller queue state as described in the
 * virtio spec. In addition available and used indexes are saved.
 *
 * The queue state section consists of the array of queues, the
 * size of the array is &struct snap_virtio_ctrl_common_state.num_queues
 *
 * @hw_available_index:  queue available index as reported by the controller.
 *		       It is always less or equal to the driver available index
 *		       because some commands may not have been processed by
 *		       the controller.
 * @hw_used_index:       queue used index as reported by the controller.
 */
struct snap_virtio_ctrl_queue_state {
	uint16_t queue_size;
	uint16_t queue_msix_vector;
	uint16_t queue_enable;
	uint16_t queue_notify_off;
	uint64_t queue_desc;
	uint64_t queue_driver;
	uint64_t queue_device;

	uint16_t hw_available_index;
	uint16_t hw_used_index;
} __attribute__((packed));

/**
 * enum snap_virtio_ctrl_lm_state - Virtio controller live migration state
 *
 * The enum define live migration state.
 */
enum snap_virtio_ctrl_lm_state {
	SNAP_VIRTIO_CTRL_LM_INIT,
	SNAP_VIRTIO_CTRL_LM_NORMAL, // Keep for anyone using snap_channel
	SNAP_VIRTIO_CTRL_LM_RUNNING = SNAP_VIRTIO_CTRL_LM_NORMAL,
	SNAP_VIRTIO_CTRL_LM_QUIESCED,
	SNAP_VIRTIO_CTRL_LM_FREEZED,
};

/* defaults to v2 */
int snap_virtio_ctrl_state_size(struct snap_virtio_ctrl *ctrl, size_t *common_cfg_len,
				size_t *queue_cfg_len, size_t *dev_cfg_len);
int snap_virtio_ctrl_state_save(struct snap_virtio_ctrl *ctrl, void *buf, size_t len);
int snap_virtio_ctrl_state_restore(struct snap_virtio_ctrl *ctrl,
				   const void *buf, size_t len);

int snap_virtio_ctrl_state_size_v1(struct snap_virtio_ctrl *ctrl, size_t *common_cfg_len,
				size_t *queue_cfg_len, size_t *dev_cfg_len);
int snap_virtio_ctrl_state_save_v1(struct snap_virtio_ctrl *ctrl, void *buf, size_t len);
int snap_virtio_ctrl_state_restore_v1(struct snap_virtio_ctrl *ctrl,
				   const void *buf, size_t len);

int snap_virtio_ctrl_state_size_v2(struct snap_virtio_ctrl *ctrl, size_t *common_cfg_len,
				size_t *queue_cfg_len, size_t *dev_cfg_len);
int snap_virtio_ctrl_state_save_v2(struct snap_virtio_ctrl *ctrl, void *buf, size_t len);
int snap_virtio_ctrl_state_restore_v2(struct snap_virtio_ctrl *ctrl,
				   const void *buf, size_t len);

int snap_virtio_ctrl_provision_queue(struct snap_virtio_ctrl *ctrl,
				     struct snap_virtio_ctrl_queue_state *qst,
				     uint32_t vq_index);
int snap_virtio_ctrl_query_queue(struct snap_virtio_ctrl *ctrl,
				     struct snap_virtio_ctrl_queue_state *qst,
				     uint32_t vq_index);
void snap_virtio_ctrl_log_writes(struct snap_virtio_ctrl *ctrl, bool enable);

int snap_virtio_ctrl_lm_enable(struct snap_virtio_ctrl *ctrl, const char *name);
void snap_virtio_ctrl_lm_disable(struct snap_virtio_ctrl *ctrl);

int  snap_virtio_ctrl_recover(struct snap_virtio_ctrl *ctrl,
			      struct snap_virtio_device_attr *attr);
int snap_virtio_ctrl_should_recover(struct snap_virtio_ctrl *ctrl);

const struct snap_virtio_ctrl_queue_stats *
snap_virtio_ctrl_q_io_stats(struct snap_virtio_ctrl *ctrl, uint16_t q_idx);

int snap_virtio_ctrl_hotunplug(struct snap_virtio_ctrl *ctrl);


int snap_virtio_ctrl_quiesce_adm(void *data);
int snap_virtio_ctrl_freeze(void *data);
int snap_virtio_ctrl_unquiesce(void *data);
int snap_virtio_ctrl_unfreeze(void *data);
/* v1 is now internal for use in the migration channel only */
int snap_virtio_ctrl_get_state_size_v2(void *data);
enum snap_virtio_ctrl_lm_state snap_virtio_ctrl_get_lm_state(void *data);
/**
 * snap_virtio_ctrl_set_lm_state() - Set the live migtration state
 * @ctrl: virtio controller
 * @lm_state: the lm state value set to
 *
 * This function should be used to set the controller live migration state.
 */
static inline void
snap_virtio_ctrl_set_lm_state(struct snap_virtio_ctrl *ctrl,
			      enum snap_virtio_ctrl_lm_state lm_state)
{
	ctrl->lm_state = lm_state;
}

int snap_virtio_ctrl_start_dirty_pages_track(void *data);
int snap_virtio_ctrl_stop_dirty_pages_track(void *data);
int snap_virtio_ctrl_get_dirty_pages_size(void *data);
int snap_virtio_ctrl_serialize_dirty_pages(void *data, void *buffer, size_t length);
int snap_virtio_ctrl_clear_reset(struct snap_virtio_ctrl *ctrl);

#endif
