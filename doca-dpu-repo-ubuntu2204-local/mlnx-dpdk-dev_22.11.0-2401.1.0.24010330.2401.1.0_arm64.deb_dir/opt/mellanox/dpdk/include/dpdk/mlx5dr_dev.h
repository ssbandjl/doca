/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2022 NVIDIA Corporation & Affiliates
 */

#ifndef MLX5DR_DEV_H_
#define MLX5DR_DEV_H_

struct mlx5dr_dev_context;
struct mlx5dr_dev_matcher;
struct mlx5dr_dev_action;
struct mlx5dr_dev_rule;

enum mlx5dr_dev_rule_modify_header_mode {
	/* no modify-header action at all */
	MLX5DR_DEV_RULE_MODE_DATA_NONE,
	/* One action only, will be inserted inline the WQE */
	MLX5DR_DEV_RULE_MODE_DATA_OPTIMIZED,
	/* Set of actions, the data was written before rule insertion call */
	MLX5DR_DEV_RULE_MODE_DATA_PRE_WRITTEN,
	/* Set of actions, the data was not written yet */
	MLX5DR_DEV_RULE_MODE_DATA_INLINE,
};

struct mlx5dr_dev_action_ct {
	struct mlx5dr_dev_action *ctr;
	enum mlx5dr_dev_rule_modify_header_mode mh_mode;
	struct mlx5dr_dev_action *modify;
	struct mlx5dr_dev_action *modify_reverse;
	union {
		struct {
			uint32_t modify_value;
			uint32_t modify_reverse_value;
		} optimized;
		struct {
			uint32_t modify_offset;
			uint32_t modify_reverse_offset;
		} pre_written;
		struct {
			uint32_t modify_offset;
			uint32_t modify_reverse_offset;
			uint16_t data_size;
			uint8_t *modify_data;
			uint8_t *modify_reverse_data;
		} inline_mh;
	};
	struct mlx5dr_dev_action *dest_table;
	struct mlx5dr_dev_action *tag;
	uint32_t ctr_offset;
	uint32_t tag_value;
};

struct mlx5dr_dev_rule_match_ctv4 {
	__be16 src_port;
	__be16 dst_port;
	__be32 src_addr;
	__be32 dst_addr;
	__be32 metadata;
	uint8_t protocol;
};

struct mlx5dr_dev_rule_match_ctv6 {
	__be16 src_port;
	__be16 dst_port;
	uint8_t src_addr[16];
	uint8_t dst_addr[16];
	__be32 metadata;
	uint8_t protocol;
};

struct mlx5dr_dev_rule_match_ct {
	union {
		struct mlx5dr_dev_rule_match_ctv4 ctv4;
		struct mlx5dr_dev_rule_match_ctv6 ctv6;
	};
};

struct mlx5dr_dev_rule_attr_ct {
	uint16_t queue_id;
	void *user_data;
	/* Valid if matcher optimize_using_rule_idx is set */
	uint32_t rule_idx;
	uint32_t burst:1;
	/* CT use case is 2 direction under the same rule handle,
	 * The match is done A->B and B->A both of the rules are
	 * inserted to RX side of the FDB. TX side should be unused.
	 */
	uint8_t bi_direction;

};

enum mlx5dr_dev_send_op_status {
	MLX5DR_DEV_SEND_OP_SUCCESS,
	MLX5DR_DEV_SEND_OP_ERROR,
};

struct mlx5dr_dev_send_op_result {
	enum mlx5dr_dev_send_op_status status;
	void *user_data;
};

/* Bind thread context to thread. This initialization should be called once per
 * new thread to allow rule create/destroy/poll operations over the provided
 * queue id. For DPA flexio_dev_outbox_config must be called prior to this call.
 * NOTE: This function should only be called from the device.
 *
 * @param[in] dev_ctx - Device context.
 * @return void
 */
void mlx5dr_dev_send_bind_to_thread(struct mlx5dr_dev_context *dev_ctx,
				    uint8_t queue_id);

/* Create CT rule.
 * NOTE: This function should only be called from the device.
 *
 * @param[in] dev_matcher - FDB DPA device matcher supporting CT.
 * @param[in] dev_match - Match values for rule creation.
 * @param[in] dev_actions - CT DPA device actions.
 * @param[in] attr - Rule creation attributes.
 * @param[in, out] dev_rule - Rule handle.
 * @return 0 on success non zero otherwise.
 */
int mlx5dr_dev_rule_ct_create(struct mlx5dr_dev_matcher *dev_matcher,
			      struct mlx5dr_dev_rule_match_ct *dev_match,
			      struct mlx5dr_dev_action_ct *dev_actions,
			      struct mlx5dr_dev_rule_attr_ct *attr,
			      struct mlx5dr_dev_rule *dev_rule);

/* Enqueue update actions on an existing dev rule.
 * NOTE: This function should only be called from the device.

 * @param[in] dev_matcher - FDB DPA device matcher supporting CT.
 * @param[in] dev_match - Match values used for rule identification.
 * @param[in] dev_actions - CT DPA device actions to be used on exiting rule.
 * @param[in] attr - Rule update attributes.
 * @param[in, out] dev_rule - Existing rule handle.
 * @return 0 on successful enqueue non zero otherwise.
 */
int mlx5dr_dev_rule_ct_action_update(struct mlx5dr_dev_matcher *dev_matcher,
				     struct mlx5dr_dev_rule_match_ct *dev_match,
				     struct mlx5dr_dev_action_ct *dev_actions,
				     struct mlx5dr_dev_rule_attr_ct *attr,
				     struct mlx5dr_dev_rule *dev_rule);

/* Destroy CT rule.
 * NOTE: This function should only be called from the device.
 *
 * @param[in] dev_matcher - Device matcher.
 * @param[in] dev_match - Match values required for rule deletion.
 * @param[in] attr - Rule creation attributes.
 * @param[in] dev_rule- Rule handle to destroy.
 * @return 0 on success non zero otherwise.
 */
int mlx5dr_dev_rule_ct_destroy(struct mlx5dr_dev_matcher *dev_matcher,
			       struct mlx5dr_dev_rule_match_ct *dev_match,
			       struct mlx5dr_dev_rule_attr_ct *attr,
			       struct mlx5dr_dev_rule *dev_rule);

/* Poll queue for rule creation and deletions completions.
 * NOTE: This function should only be called from the device.
 *
 * @param[in] dev_ctx - Device context.
 * @param[in] queue_id - The id of the queue to poll.
 * @param[in, out] res - Completion array.
 * @param[in] res_nb - Maximum number of results to return.
 * @return negative number on failure, the number of completions otherwise.
 */
int mlx5dr_dev_send_queue_poll(struct mlx5dr_dev_context *dev_ctx,
			       uint16_t queue_id,
			       struct mlx5dr_dev_send_op_result res[],
			       uint32_t res_nb);

/* Order queue to process queue rule creation and deletion
 * NOTE: This function should only be called from the device.
 *
 * @param[in] dev_ctx - Device context.
 * @param[in] queue_id - The id of the queue to poll.
 * @return 0 on success non zero otherwise.
 */
int mlx5dr_dev_send_queue_drain(struct mlx5dr_dev_context *dev_ctx,
				uint16_t queue_id);

struct mlx5dr_dev_arg_send_attr {
	uint8_t burst;
	uint8_t queue_id;
};

/* Send ARG data.
 * NOTE: This function should only be called from the device.
 *
 * @param[in] dev_ctx - Device context.
 * @param[in] attr - ARG creation attributes.
 * @param[in] mh_action - Action connected to the ARG.
 * @param[in] arg_offset- The offset in the ARG object.
 * @param[in] arg_data- The data to write in the ARG object.
 * @param[in] data_size- The size of the data.
 * @return 0 on success non zero otherwise.
 */
int mlx5dr_dev_send_arg_data(struct mlx5dr_dev_context *dev_ctx,
			     struct mlx5dr_dev_arg_send_attr *attr,
			     struct mlx5dr_dev_action *mh_action,
			     uint32_t arg_offset,
			     uint8_t *arg_data,
			     uint16_t data_size);

#endif
