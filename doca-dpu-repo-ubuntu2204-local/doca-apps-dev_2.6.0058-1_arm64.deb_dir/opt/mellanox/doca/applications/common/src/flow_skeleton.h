/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#ifndef COMMON_FLOW_SKELETON_H_
#define COMMON_FLOW_SKELETON_H_

#include <doca_flow.h>

#ifdef __cplusplus
extern "C" {
#endif

struct flow_skeleton_entry_ctx {
	enum doca_flow_entry_op op;				/* doca flow entry operation */
	enum doca_flow_pipe_type type;				/* pipe type for adding the entry */
	struct doca_flow_pipe *pipe;				/* pointer to the pipe of the entry */
	uint32_t priority;					/* priority value */
	struct doca_flow_match *match;				/* pointer to match */
	struct doca_flow_match *match_mask;			/* pointer to match mask */
	struct doca_flow_actions *actions;			/* pointer to modify actions */
	struct doca_flow_actions *actions_mask;			/* pointer to modify actions' mask */
	const struct doca_flow_action_descs *action_descs;	/* pointer to action descriptions */
	struct doca_flow_monitor *monitor;			/* pointer to monitor actions */
	struct doca_flow_fwd *fwd;				/* pointer to fwd actions */
	uint32_t idx;						/* unique entry index for ordered list entry */
	const struct doca_flow_ordered_list *ordered_list;	/* pointer to ordered list struct */
	void *usr_ctx;						/* pointer to user context */
	struct doca_flow_pipe_entry **entry;			/* created pipe entry handler */
};

struct flow_skeleton_entry_mem {
	struct doca_flow_match match;				/* match struct, indicate specific packet match information */
	struct doca_flow_match match_mask;			/* match mask information */
	struct doca_flow_actions actions;			/* modify actions, indicate specific modify information */
	const struct doca_flow_action_descs action_descs;	/* action descriptions */
	struct doca_flow_monitor monitor;			/* monitor actions, indicate specific monitor information */
	struct doca_flow_fwd fwd;				/* fwd actions */
	const struct doca_flow_ordered_list ordered_list;	/* ordered list with pointers to actions and monitor */

};

struct flow_skeleton_entry {
	struct flow_skeleton_entry_ctx ctx;			/* context to sent to add entry API */
	struct flow_skeleton_entry_mem mem;			/* memory to fill */
};

struct flow_skeleton_aging_op {
	bool to_remove;						/* Whether to remove the aged entry or not */
};

/**
 * Process callback for add and remove ops
 */
typedef void (*flow_skeleton_process_cb)(struct doca_flow_pipe_entry *entry, uint16_t pipe_queue,
	enum doca_flow_entry_status status, void *user_ctx, void *program_ctx);

/**
 * Entries acquisition callback
 */
typedef void (*flow_skeleton_entries_acquisition_cb)(struct flow_skeleton_entry *entries,
	uint16_t port_id, void *program_ctx, uint32_t *nb_entries);

/**
 * Initialization callback
 */
typedef void (*flow_skeleton_initialize_cb)(struct flow_skeleton_entry *entries,
	uint16_t port_id, void *program_ctx, uint32_t *nb_entries);

/**
 * Aging callback
 */
typedef void (*flow_skeleton_aging_cb)(struct doca_flow_pipe_entry *entry, struct flow_skeleton_aging_op *aging_op);

/**
 * Failure callback
 */
typedef void (*flow_skeleton_failure_cb)(void);

struct flow_skeleton_cfg {
	uint32_t nb_ports;						/* number of ports */
	uint32_t nb_entries;						/* number of entries to fill in entries acquisition cb */
	bool handle_aging;						/* true if application wants to handle aging */
	flow_skeleton_process_cb add_cb;				/* process callback for add operation */
	flow_skeleton_process_cb remove_cb;				/* process callback for remove operation */
	flow_skeleton_entries_acquisition_cb entries_acquisition_cb;	/* entries acquisition callback */
	flow_skeleton_initialize_cb init_cb;				/* initialization callback */
	flow_skeleton_aging_cb aging_cb;				/* aging callback */
	flow_skeleton_failure_cb failure_cb;				/* Failure callback */
};

struct main_loop_params {
	bool initialization;		/* Whether to call the init_cb on this core or not */
	uint16_t pipe_queue;		/* pipe queue for adding entries - lcore ID */
	void *program_ctx;		/* application context */
	int nb_ports;			/* number of initialized ports */
	struct doca_flow_port *ports[];	/* array of DOCA Flow ports */
};

/*
 * Initialize DOCA Flow and the skeleton resources
 *
 * @flow_cfg [in]: DOCA Flow configuration struct
 * @skeleton_cfg [in]: skeleton configuration struct
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t flow_skeleton_init(struct doca_flow_cfg *flow_cfg, struct flow_skeleton_cfg *skeleton_cfg);

/*
 * Destroy DOCA Flow and free the skeleton resources
 */
void flow_skeleton_destroy(void);

/*
 * Run the skeleton main loop
 *
 * @main_loop_params [in]: pointer of type struct flow_skeleton_main_loop_params
 */
void flow_skeleton_main_loop(void *main_loop_params);

/*
 * Notify the skeleton to exit from main loop
 */
void flow_skeleton_notify_exit(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COMMON_FLOW_SKELETON_H_ */
