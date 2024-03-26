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

#include <string.h>
#include <unistd.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

#define META_U32_BIT_OFFSET(idx) \
	(offsetof(struct doca_flow_meta, u32[(idx)]) << 3)
#define NB_ACTION_DESC (3)
#define IP_TCP_DEFAULT_HDR_LEN 40

DOCA_LOG_REGISTER(FLOW_MATCH_COMPARISON);

/*
 * Create DOCA Flow pipe with changeable match on meta data
 *
 * @port [in]: port of the pipe
 * @port_id [in]: port ID of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_match_meta_pipe(struct doca_flow_port *port, int port_id, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_match match_mask;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&match_mask, 0, sizeof(match_mask));
	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "MATCH_PIPE";
	pipe_cfg.match = &match;
	pipe_cfg.match_mask = &match_mask;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.attr.nb_actions = NB_ACTIONS_ARR;
	pipe_cfg.port = port;

	/* set match_mask on meta */
	match_mask.meta.u32[0] = UINT32_MAX;

	/* forwarding traffic to other port */
	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry with example meta data value to match
 *
 * @pipe [in]: pipe of the entry
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_match_meta_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));

	/* setting match on meta */
	match.meta.u32[0] = IP_TCP_DEFAULT_HDR_LEN;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Flow control pipe for comparison
 *
 * @port [in]: port of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_match_comparsion_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "CONTROL_MATCH_COMPARISON_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
	pipe_cfg.port = port;

	return doca_flow_pipe_create(&pipe_cfg, NULL, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry with comparison to match
 *
 * @pipe [in]: pipe of the entry
 * @next_pipe [in]: next_pipe of the comparison
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_match_comparsion_pipe_entry(struct doca_flow_pipe *pipe,
				struct doca_flow_pipe *next_pipe,
				struct entries_status *status)
{
	struct doca_flow_match_condition condition;
	struct doca_flow_fwd fwd;
	doca_error_t result;

	memset(&condition, 0, sizeof(condition));

	condition.operation = DOCA_FLOW_COMPARE_GT;
	condition.field_op.a.field_string = "meta.data";
	condition.field_op.a.bit_offset = META_U32_BIT_OFFSET(1);
	condition.field_op.b.field_string = "meta.data";
	condition.field_op.b.bit_offset = META_U32_BIT_OFFSET(0);
	condition.field_op.width = 32;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = next_pipe;

	result = doca_flow_pipe_control_add_entry(0, 0, pipe, NULL, NULL, &condition,
						  NULL, NULL, NULL, NULL, &fwd, status, NULL);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Flow pipe with 5 tuple match, and copy & add ipv4.version_ihl tcp.data_offset
 *
 * @port [in]: port of the pipe
 * @next_pipe [in]: pipe to forward the matched packets
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_sum_to_meta_pipe(struct doca_flow_port *port, struct doca_flow_pipe *next_pipe,
			struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_actions *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_action_descs descs;
	struct doca_flow_action_descs *descs_arr[NB_ACTIONS_ARR];
	struct doca_flow_action_desc desc_array[NB_ACTION_DESC] = {0};

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));
	memset(&descs, 0, sizeof(descs));

	pipe_cfg.attr.name = "SUM_TO_META_PIPE";
	pipe_cfg.match = &match;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	descs_arr[0] = &descs;
	descs.nb_action_desc = NB_ACTION_DESC;
	descs.desc_array = desc_array;
	pipe_cfg.action_descs = descs_arr;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.attr.nb_actions = NB_ACTIONS_ARR;
	pipe_cfg.port = port;

	/* copy ipv4.version_ihl to u32[0] */
	desc_array[0].type = DOCA_FLOW_ACTION_COPY;
	desc_array[0].field_op.src.field_string = "outer.ipv4.version_ihl";
	desc_array[0].field_op.src.bit_offset = 0;
	desc_array[0].field_op.dst.field_string = "meta.data";
	/* Set bit_offset to 2 as ihl * 4 = IPv4 hdr len */
	desc_array[0].field_op.dst.bit_offset = META_U32_BIT_OFFSET(0) + 2;
	desc_array[0].field_op.width = 4;

	/* accumulate tcp.data_offset to u32[0] */
	desc_array[1].type = DOCA_FLOW_ACTION_ADD;
	desc_array[1].field_op.src.field_string = "outer.tcp.data_offset";
	desc_array[1].field_op.src.bit_offset = 0;
	desc_array[1].field_op.dst.field_string = "meta.data";
	/* Set bit_offset to 2 as tcp.df * 4 = TCP hdr len */
	desc_array[1].field_op.dst.bit_offset = META_U32_BIT_OFFSET(0) + 2;
	desc_array[1].field_op.width = 4;

	/* add IPv4.total_len to u32[1] */
	desc_array[2].type = DOCA_FLOW_ACTION_ADD;
	desc_array[2].field_op.src.field_string = "outer.ipv4.total_len";
	desc_array[2].field_op.src.bit_offset = 0;
	desc_array[2].field_op.dst.field_string = "meta.data";
	desc_array[2].field_op.dst.bit_offset = META_U32_BIT_OFFSET(1);
	desc_array[2].field_op.width = 16;

	/* 5 tuple match */
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;
	match.outer.tcp.l4_port.src_port = 0xffff;
	match.outer.tcp.l4_port.dst_port = 0xffff;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = next_pipe;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry with sum of ipv4.ihl and tcp.data_offset to meta.
 *
 * @pipe [in]: pipe of the entry
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_sum_to_meta_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	doca_be32_t dst_ip_addr = BE_IPV4_ADDR(8, 8, 8, 8);
	doca_be32_t src_ip_addr = BE_IPV4_ADDR(1, 2, 3, 4);
	doca_be16_t dst_port = rte_cpu_to_be_16(80);
	doca_be16_t src_port = rte_cpu_to_be_16(1234);

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));

	match.outer.ip4.dst_ip = dst_ip_addr;
	match.outer.ip4.src_ip = src_ip_addr;
	match.outer.tcp.l4_port.dst_port = dst_port;
	match.outer.tcp.l4_port.src_port = src_port;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Run flow_match_comparsion sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
flow_match_comparsion(int nb_queues)
{
	const int nb_ports = 2;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_flow_pipe *comparison_pipe;
	struct doca_flow_pipe *match_meta_pipe;
	struct doca_flow_pipe *sum_pipe;
	struct doca_dev *dev_arr[nb_ports];
	struct entries_status status;
	int num_of_entries = 3;
	doca_error_t result;
	int port_id;

	result = init_doca_flow(nb_queues, "vnf,hws", resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA Flow: %s", doca_error_get_descr(result));
		return result;
	}

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	result = init_doca_flow_ports(nb_ports, ports, true, dev_arr);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA ports: %s", doca_error_get_descr(result));
		doca_flow_destroy();
		return result;
	}

	for (port_id = 0; port_id < nb_ports; port_id++) {
		memset(&status, 0, sizeof(status));

		result = create_match_meta_pipe(ports[port_id], port_id, &match_meta_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create match meta pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_match_meta_pipe_entry(match_meta_pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add match meta entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_match_comparsion_pipe(ports[port_id], &comparison_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create comparsion pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_match_comparsion_pipe_entry(comparison_pipe, match_meta_pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add comparsion entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_sum_to_meta_pipe(ports[port_id], comparison_pipe, &sum_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create sum to meta pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_sum_to_meta_pipe_entry(sum_pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add sum to meta entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = doca_flow_entries_process(ports[port_id], 0, DEFAULT_TIMEOUT_US, num_of_entries);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to process entries: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		if (status.nb_processed != num_of_entries || status.failure) {
			DOCA_LOG_ERR("Failed to process entries");
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return DOCA_ERROR_BAD_STATE;
		}
	}

	DOCA_LOG_INFO("Wait few seconds for packets to arrive");
	sleep(5);

	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return DOCA_SUCCESS;
}
