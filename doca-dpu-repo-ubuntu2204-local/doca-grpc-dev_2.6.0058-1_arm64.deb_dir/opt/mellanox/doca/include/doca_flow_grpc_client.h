/*
 * Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

/**
 * @file doca_flow_grpc_client.h
 * @page doca flow grpc client
 * @defgroup GRPC Flow
 * DOCA flow grpc API to run remote HW offload with flow library.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */

#ifndef DOCA_FLOW_GRPC_CLIENT_H_
#define DOCA_FLOW_GRPC_CLIENT_H_

#include <stdint.h>

#include <doca_error.h>
#include <doca_flow_ct.h>

#ifdef __cplusplus
extern "C" {
#endif

struct doca_flow_ct_cfg;

/**
 * @brief pipeline configuration wrapper
 */
struct doca_flow_grpc_pipe_cfg {
	struct doca_flow_pipe_cfg *cfg;
	/**< doca_flow_pipe_cfg struct */
	uint16_t port_id;
	/**< port id */
};

/**
 * @brief forwarding configuration wrapper
 */
struct doca_flow_grpc_fwd {
	struct doca_flow_fwd *fwd;
	/**< doca flow fwd struct */
	uint64_t next_pipe_id;
	/**< next pipe id */
};

/**
 * @brief doca flow grpc bindable object types
 */
enum doca_flow_grpc_bindable_obj_type {
	DOCA_FLOW_GRPC_BIND_TYPE_PIPE,
	/**< bind resource to a pipe */
	DOCA_FLOW_GRPC_BIND_TYPE_PORT,
	/**< bind resource to a port */
	DOCA_FLOW_GRPC_BIND_TYPE_NULL,
	/**< bind resource globally */
};

/**
 * @brief bindable object configuration
 */
struct doca_flow_grpc_bindable_obj {
	enum doca_flow_grpc_bindable_obj_type type;
	/**< bindable object type */
	union {
		uint32_t port_id;
		/**< port id if type is port */
		uint64_t pipe_id;
		/**< pipe id if type is pipe */
	};
};

/**
 * @brief Initialize a channel to DOCA flow grpc server.
 *
 * Must be invoked first before any other function in this API.
 * this is a one time call, used for grpc channel initialization.
 *
 * @param grpc_address
 * String representing the service ip, i.e. "127.0.0.1" or "192.168.100.3:5050".
 * If no port is provided, it will use the service default port.
 */
DOCA_EXPERIMENTAL
void doca_flow_grpc_client_create(const char *grpc_address);

/**
 * @brief RPC call for doca_flow_init().
 *
 * @param cfg
 * Program configuration, see doca_flow_cfg for details.
 * @param ct_cfg
 * ct configuration if required, otherwise NULL. see doce_flow_ct_cfg for details
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_init(const struct doca_flow_cfg *cfg, const struct doca_flow_ct_cfg *ct_cfg);

/**
 * @brief RPC call for doca_flow_port_start().
 *
 * @param cfg
 * Port configuration, see doca_flow_port_cfg for details.
 * @param port_id
 * Created port ID on success.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_port_start(const struct doca_flow_port_cfg *cfg, uint16_t *port_id);

/**
 * @brief RPC call for doca_flow_port_stop().
 *
 * @param port_id
 * Port ID.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_port_stop(uint16_t port_id);

/**
 * @brief RPC call for doca_flow_port_pair().
 *
 * @param port_id
 * port ID.
 * @param pair_port_id
 * pair port ID.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_port_pair(uint16_t port_id, uint16_t pair_port_id);

/**
 * @brief RPC call for doca_flow_shared_resource_cfg().
 *
 * @param type
 * Shared resource type.
 * @param id
 * Shared resource id.
 * @param cfg
 * Pointer to a shared resource configuration.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_shared_resource_cfg(enum doca_flow_shared_resource_type type,
		uint32_t id, struct doca_flow_shared_resource_cfg *cfg);

/**
 * @brief RPC call for doca_flow_shared_resources_bind().
 *
 * @param type
 * Shared resource type.
 * @param res_array
 * Array of shared resource IDs.
 * @param res_array_len
 * Shared resource IDs array length.
 * @param bindable_obj_id
 * Pointer to a bindable object ID.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_shared_resources_bind(enum doca_flow_shared_resource_type type,
		uint32_t *res_array, uint32_t res_array_len, struct doca_flow_grpc_bindable_obj *bindable_obj_id);

/**
 * @brief RPC call for doca_flow_shared_resources_query().
 *
 * @param type
 * Shared object type.
 * @param res_array
 * Array of shared objects IDs to query.
 * @param query_results_array
 * Data array retrieved by the query.
 * @param array_len
 * Number of objects and their query results in their arrays (same number).
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_shared_resources_query(enum doca_flow_shared_resource_type type,
		uint32_t *res_array, struct doca_flow_shared_resource_result *query_results_array, uint32_t array_len);

/**
 * @brief RPC call for doca_flow_pipe_create().
 *
 * @param cfg
 * Pipe configuration, see doca_flow_grpc_pipe_cfg for details.
 * @param fwd
 * Fwd configuration for the pipe.
 * @param fwd_miss
 * Fwd_miss configuration for the pipe. NULL for no fwd_miss.
 * When creating a pipe if there is a miss and fwd_miss configured,
 * packet steering should jump to it.
 * @param pipe_id
 * Created pipe ID on success.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_create(const struct doca_flow_grpc_pipe_cfg *cfg,
					const struct doca_flow_grpc_fwd *fwd,
					const struct doca_flow_grpc_fwd *fwd_miss,
					uint64_t *pipe_id);

/**
 * @brief RPC call for doca_flow_pipe_add_entry().
 *
 * @param pipe_queue
 * Queue identifier.
 * @param pipe_id
 * Pipe ID.
 * @param match
 * Pointer to match, indicate specific packet match information.
 * @param actions
 * Pointer to modify actions, indicate specific modify information.
 * @param monitor
 * Pointer to monitor actions.
 * @param client_fwd
 * Pointer to fwd actions.
 * @param flags
 * Flow entry will be pushed to hw immediately or not. enum doca_flow_flags_type.
 * @param entry_id
 * Created entry ID on success.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_add_entry(uint16_t pipe_queue, uint64_t pipe_id, const struct doca_flow_match *match,
		const struct doca_flow_actions *actions, const struct doca_flow_monitor *monitor,
		const struct doca_flow_grpc_fwd *client_fwd, uint32_t flags, uint64_t *entry_id);

/**
 * @brief RPC call for doca_flow_pipe_control_add_entry().
 *
 * @param pipe_queue
 * Queue identifier.
 * @param priority
 * Priority value..
 * @param pipe_id
 * Pipe ID.
 * @param match
 * Pointer to match, indicate specific packet match information.
 * @param match_mask
 * Pointer to match mask information.
 * @param actions
 * Pointer to actions
 * @param actions_mask
 * Pointer to actions' mask
 * @param actions_descs
 * Pointer to actions descriptions
 * @param monitor
 * Pointer to monitor
 * @param client_fwd
 * Pointer to fwd actions.
 * @param entry_id
 * Created entry ID on success.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_control_add_entry(uint16_t pipe_queue, uint8_t priority,
		uint64_t pipe_id, const struct doca_flow_match *match, const struct doca_flow_match *match_mask,
		const struct doca_flow_actions *actions, const struct doca_flow_actions *actions_mask,
		const struct doca_flow_action_descs *actions_descs,
		const struct doca_flow_monitor *monitor,
		const struct doca_flow_grpc_fwd *client_fwd, uint64_t *entry_id);

/**
 * @brief RPC call for doca_flow_pipe_lpm_add_entry().
 *
 * @param pipe_queue
 * Queue identifier.
 * @param pipe_id
 * Pipe ID.
 * @param match
 * Pointer to match, indicate specific packet match information.
 * @param match_mask
 * Pointer to match mask information.
 * @param actions
 * Pointer to modify actions, indicate specific modify information.
 * @param monitor
 * Pointer to monitor actions.
 * @param client_fwd
 * Pointer to fwd actions.
 * @param flag
 * Flow entry will be pushed to hw immediately or not. enum doca_flow_flags_type.
 * @param entry_id
 * Created entry ID on success.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_lpm_add_entry(uint16_t pipe_queue, uint64_t pipe_id, const struct doca_flow_match *match,
		const struct doca_flow_match *match_mask, const struct doca_flow_actions *actions,
		const struct doca_flow_monitor *monitor, const struct doca_flow_grpc_fwd *client_fwd,
		const enum doca_flow_flags_type flag, uint64_t *entry_id);

/**
 * @brief RPC call for doca_flow_grpc_pipe_rm_entry().
 *
 * @param pipe_queue
 * Queue identifier.
 * @param entry_id
 * The entry ID to be removed.
 * @param flags
 * Flow entry will removed from hw immediately or not. enum doca_flow_flags_type.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_rm_entry(uint16_t pipe_queue, uint64_t entry_id, uint32_t flags);

/**
 * @brief RPC call for doca_flow_pipe_create().
 *
 * @param pipe_id
 * Pipe ID.
 * @param new_congestion_level
 * New congestino level percentage.
 * @param nr_entries
 * Number of entries on resize success.
 * @param pipe_ctx
 * Pipe user context of the resized pipe.
 *
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_resize(uint64_t pipe_id,
					uint32_t new_congestion_level,
					uint32_t *nr_entries,
					uint64_t *pipe_ctx);
/**
 * @brief RPC call for doca_flow_pipe_destroy().
 *
 * @param pipe_id
 * Pipe ID.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_destroy(uint64_t pipe_id);

/**
 * @brief RPC call for doca_flow_port_pipes_flush().
 *
 * @param port_id
 * Port ID.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_port_pipes_flush(uint16_t port_id);

/**
 * @brief RPC call for doca_flow_port_pipes_dump().
 *
 * @param port_id
 * Port ID.
 * @param f
 * The output file of the pipe information.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_port_pipes_dump(uint16_t port_id, FILE *f);

/**
 * @brief RPC call for doca_flow_pipe_dump().
 *
 * @param pipe_id
 * pipe ID.
 * @param f
 * The output file of the pipe information.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_dump(uint64_t pipe_id, FILE *f);

/**
 * @brief RPC call for doca_flow_query_entry().
 *
 * @param entry_id
 * The pipe entry ID to query.
 * @param query_stats
 * Data retrieved by the query.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_query_entry(uint64_t entry_id, struct doca_flow_query *query_stats);

/**
 * @brief RPC call for doca_flow_query_pipe_miss().
 *
 * @param pipe_id
 * The pipe ID to query.
 * @param query_stats
 * Data retrieved by the query.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t doca_flow_grpc_query_pipe_miss(uint64_t pipe_id, struct doca_flow_query *query_stats);

/**
 * @brief RPC call for doca_flow_aging_handle().
 *
 * @param port_id
 * Port id to handle aging
 * @param queue
 * Queue identifier.
 * @param quota
 * Max time quota in micro seconds for this function to handle aging.
 * @param max_entries
 * Max entries for this function to handle aging, 0: no limit.
 * @param entries_id
 * User input entries array for the aged flows.
 * @param len
 * User input length of entries array.
 * @param nb_aged_flow
 * Number of aged flow.
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_aging_handle(uint16_t port_id, uint16_t queue,
		uint64_t quota, uint64_t max_entries, uint64_t *entries_id, int len, int *nb_aged_flow);

/**
 * @brief RPC call for doca_flow_grpc_entries_process().
 *
 * @param port_id
 * Port ID
 * @param pipe_queue
 * Queue identifier.
 * @param timeout
 * Max time in micro seconds for this function to process entries.
 * Process once if timeout is 0
 * @param max_processed_entries
 * Flow entries number to process
 * If it is 0, it will proceed until timeout.
 * @param num_processed_entries
 * Number of entries processed
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_entries_process(uint16_t port_id, uint16_t pipe_queue, uint64_t timeout,
		uint32_t max_processed_entries, int *num_processed_entries);

/**
 * @brief RPC call for doca_flow_pipe_entry_get_status()
 *
 * @param entry_id
 * pipe entry ID
 * @param entry_status
 * entry's status
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_pipe_entry_get_status(uint64_t entry_id, enum doca_flow_entry_status *entry_status);

/**
 * @brief RPC call for doca_flow_port_switch_get()
 *
 * @param port_id
 * Switch port ID
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 *
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_port_switch_get(uint16_t *port_id);

/**
 * @brief RPC call for doca_flow_destroy().
 *
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_destroy(void);

/**
 * @brief Add new entry to doca flow CT table.
 *
 * @param [in] queue_id
 * queue ID, offset from doca_flow.nb_queues.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] flags
 * operation flags, see DOCA_FLOW_CT_ENTRY_FLAGS_xxx.
 * @param [in] match_origin
 * match pattern in origin direction.
 * @param [in] match_reply
 * match pattern in reply direction, default to reverse of origin pattern.
 * @param [in] meta_origin
 * meta to set on origin direction
 * @param [in] meta_reply
 * meta to set on reply direction
 * @param [in] timeout_s
 * aging timeout in second, 0 to disable aging
 * @param [in] usr_ctx
 * user context data to associate to entry
 * @param [in] actions_origin
 * actions to set on origin direction
 * @param [in] actions_reply
 * actions to set on reply direction
 * @param [out] entry_id
 * new netry ID
 * @return
 * DOCA_SUCCESS - in case of success.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_ct_add_entry(uint16_t queue_id, uint64_t pipe_id, uint32_t flags,
		struct doca_flow_ct_match *match_origin, struct doca_flow_ct_match *match_reply,
		uint32_t meta_origin, uint32_t meta_reply, uint32_t timeout_s, uint64_t usr_ctx,
		const struct doca_flow_ct_actions *actions_origin, const struct doca_flow_ct_actions *actions_reply,
		uint64_t *entry_id);

/**
 * @brief Add new direction rule to doca flow CT entry.
 *
 * @param [in] queue_id
 * queue ID, offset from doca_flow.nb_queues.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] flags
 * operation flags, see DOCA_FLOW_CT_ENTRY_FLAGS_xxx.
 * @param [in] match
 * match pattern.
 * @param [in] meta
 * meta to set
 * @param [in] entry_id
 * netry ID
 * @return
 * DOCA_SUCCESS - in case of success.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_ct_entry_add_dir(uint16_t queue_id, uint64_t pipe_id, uint32_t flags,
		struct doca_flow_ct_match *match, uint32_t meta, uint64_t entry_id);

/**
 * @brief Update an existing entry doca flow CT table.
 *
 * @param [in] queue_id
 * queue ID, offset from doca_flow.nb_queues.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] entry_id
 * entry ID
 * @param [in] flags
 * operation flags, see DOCA_FLOW_CT_ENTRY_FLAGS_xxx.
 * @param [in] meta_origin
 * meta to set on origin direction
 * @param [in] meta_reply
 * meta to set on reply direction
 * @param [in] timeout_s
 * aging timeout in second, 0 to disable aging
 * @param [in] actions_origin
 * actions to set on origin direction
 * @param [in] actions_reply
 * actions to set on reply direction
 * @return
 * DOCA_SUCCESS - in case of success.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_ct_update_entry(uint16_t queue_id, uint64_t pipe_id, uint64_t entry_id,
		uint32_t flags, uint32_t meta_origin, uint32_t meta_reply, uint32_t timeout_s,
		const struct doca_flow_ct_actions *actions_origin, const struct doca_flow_ct_actions *actions_reply);

/**
 * @brief remove an existing entry doca flow CT table.
 *
 * @param [in] queue_id
 * queue ID, offset from doca_flow.nb_queues.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] entry_id
 * entry ID
 * @param [in] flags
 * operation flags, see DOCA_FLOW_CT_ENTRY_FLAGS_xxx.
 * @return
 * DOCA_SUCCESS - in case of success.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_ct_rm_entry(uint16_t queue_id, uint64_t pipe_id, uint64_t entry_id,
		uint32_t flags);

/**
 * @brief get match info of an existing entry.
 *
 * @param [in] queue_id
 * queue ID, offset from doca_flow.nb_queues.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] entry_id
 * entry ID
 * @param [in] flags
 * operation flags, see DOCA_FLOW_CT_ENTRY_FLAGS_xxx.
 * @param [out] match_origin
 * meta to set on origin direction
 * @param [out] match_reply
 * meta to set on reply direction
 * @param [out] entry_flags
 * Entry flags, see DOCA_FLOW_CT_ENTRY_FLAGS_xxx.
 * @return
 * DOCA_SUCCESS - in case of success.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_flow_grpc_ct_get_entry(uint16_t queue_id, uint64_t pipe_id, uint64_t entry_id,
		uint32_t flags, struct doca_flow_ct_match *match_origin,
		struct doca_flow_ct_match *match_reply, uint64_t *entry_flags);

/**
 * @brief Add shared modify-action
 *
 * @param [in] ctrl_queue_id
 * control queue id.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] actions
 * list of actions data, each updated with action id
 * @param [in] nb_actions
 * number of actions to create ​
 * @param [out] actions_handles
 * list of handles allocated for the input actions
 * @return
 * DOCA_SUCCESS - in case of success
 */
DOCA_EXPERIMENTAL
doca_error_t
doca_flow_grpc_ct_actions_add_shared(uint16_t ctrl_queue_id, uint64_t pipe_id, const struct doca_flow_ct_actions actions[],
				uint32_t nb_actions, uint32_t actions_handles[]);

/**
 * @brief Remove shared modify-action
 *
 * @param [in] ctrl_queue_id
 * control ctrl queue id.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] actions_handles
 * list of action ids
 * @param [in] nb_actions
 * number of actions to create ​
 * @return
 * DOCA_SUCCESS - in case of success
 */
DOCA_EXPERIMENTAL
doca_error_t
doca_flow_grpc_ct_actions_rm_shared(uint16_t ctrl_queue_id, uint64_t pipe_id, uint32_t actions_handles[], uint32_t nb_actions);

/**
 * @brief RPC call for doca_flow_ct_flow_log().
 *
 * @param [in] queue_id
 * queue ID, offset from doca_flow.nb_queues.
 * @param [in] pipe_id
 * CT pipe ID.
 * @param [in] nb_max_entries
 * Max entries to get in the output arrays
 * @param [out] nb_entries
 * Number of entries returned
 * @param [out] entry_id
 * Array of entries ids
 * @param [out] last_hit_s
 * Last hit in seconds
 * @param [out] origin_stats
 * Array of origin direction statistics
 * @param [out] reply_stats
 * Array of reply direction statistics
 * @return
 * DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
DOCA_EXPERIMENTAL
doca_error_t
doca_flow_grpc_ct_flow_log(uint16_t queue_id, uint64_t pipe_id, uint32_t nb_max_entries,
			uint32_t *nb_entries, uint64_t *entry_id, uint64_t *last_hit_s,
			struct doca_flow_query *origin_stats, struct doca_flow_query *reply_stats);

#ifdef __cplusplus
} /* extern "C" */
#endif

/** @} */

#endif /* DOCA_FLOW_GRPC_CLIENT_H_ */
