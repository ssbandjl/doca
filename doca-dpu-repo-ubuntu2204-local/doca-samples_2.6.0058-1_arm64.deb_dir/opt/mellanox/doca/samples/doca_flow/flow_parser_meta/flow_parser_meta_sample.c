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

DOCA_LOG_REGISTER(FLOW_PARSER_META);

/*
 * Create DOCA Flow pipe with changeable match on parser meta type
 *
 * @port [in]: port of the pipe
 * @port_id [in]: port ID of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_match_parser_meta_type_pipe(struct doca_flow_port *port, int port_id, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_match match_mask;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&match_mask, 0, sizeof(match_mask));
	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	/* set match_mask value */
	match_mask.parser_meta.inner_l4_type = UINT32_MAX;
	match_mask.parser_meta.outer_l3_type = UINT32_MAX;
	match.parser_meta.inner_l4_type = UINT32_MAX;
	match.parser_meta.outer_l3_type = UINT32_MAX;

	pipe_cfg.attr.name = "MATCH_PARSER_META_TYPE_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	pipe_cfg.match_mask = &match_mask;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.port = port;

	/* forwarding traffic to other port */
	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry with example of parser meta type value to match
 *
 * @pipe [in]: pipe of the entry
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_match_parser_meta_type_pipe_entries(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	memset(&match, 0, sizeof(match));

	/* set match value */
	match.parser_meta.inner_l4_type = DOCA_FLOW_L4_META_UDP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	result = doca_flow_pipe_add_entry(0, pipe, &match, NULL, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	/* set match value */
	match.parser_meta.inner_l4_type = DOCA_FLOW_L4_META_TCP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	result = doca_flow_pipe_add_entry(0, pipe, &match, NULL, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Flow pipe with changeable match on parser meta ok
 *
 * @port [in]: port of the pipe
 * @type_pipe [in]: next pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_match_parser_meta_ok_pipe(struct doca_flow_port *port,
				 struct doca_flow_pipe *type_pipe,
				 struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_match match_mask;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&match, 0, sizeof(match));
	memset(&match_mask, 0, sizeof(match_mask));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	/* set match_mask value */
	match_mask.parser_meta.inner_l3_ok = UINT8_MAX;
	match.parser_meta.inner_l3_ok = UINT8_MAX;
	match_mask.parser_meta.outer_ip4_checksum_ok = UINT8_MAX;
	match.parser_meta.outer_ip4_checksum_ok = UINT8_MAX;

	pipe_cfg.attr.name = "MATCH_PARSER_META_OK_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	pipe_cfg.match_mask = &match_mask;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.port = port;

	/* forwarding traffic to other port */
	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = type_pipe;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry with example of parser meta data integrity to match
 *
 * @pipe [in]: pipe of the entry
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_match_parser_meta_ok_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	memset(&match, 0, sizeof(match));

	/* set match value */
	match.parser_meta.inner_l3_ok = 1;
	match.parser_meta.outer_ip4_checksum_ok = 1;
	result = doca_flow_pipe_add_entry(0, pipe, &match, NULL, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Flow control pipe
 *
 * @port [in]: port of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_fragmented_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "MATCH_PARSER_META_FRAGMENTED_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.port = port;

	return doca_flow_pipe_create(&pipe_cfg, NULL, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entries to the control pipe:
 * - entry with outer IP non fragmented match that forward the matched packet to ok_pipe
 * - entry with outer IP fragmented match that forward to pair port
 *
 * @frag_pipe [in]: pipe of the entry
 * @ok_pipe [in]: pipe to forward non fragmented traffic
 * @port_id [in]: port ID of the pipe
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_fragmented_pipe_entries(struct doca_flow_pipe *frag_pipe, struct doca_flow_pipe *ok_pipe,
			    int port_id, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_match match_mask;
	struct doca_flow_fwd fwd;
	uint8_t priority = 0;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&match_mask, 0, sizeof(match_mask));
	memset(&fwd, 0, sizeof(fwd));

	match.parser_meta.outer_ip_fragmented = 0;
	match_mask.parser_meta.outer_ip_fragmented = UINT8_MAX;
	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = ok_pipe;

	result = doca_flow_pipe_control_add_entry(0, priority, frag_pipe, &match, &match_mask, NULL,
						  NULL, NULL, NULL, NULL, &fwd, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add control pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	memset(&match, 0, sizeof(match));
	memset(&match_mask, 0, sizeof(match_mask));
	memset(&fwd, 0, sizeof(fwd));

	match.parser_meta.outer_ip_fragmented = 1;
	match_mask.parser_meta.outer_ip_fragmented = UINT8_MAX;
	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	result = doca_flow_pipe_control_add_entry(0, priority, frag_pipe, &match, &match_mask, NULL,
						  NULL, NULL, NULL, NULL, &fwd, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add control pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

/*
 * Run flow_parser_meta sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
flow_parser_meta(int nb_queues)
{
	const int nb_ports = 2;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *type_pipe, *ok_pipe, *frag_pipe;
	struct entries_status status;
	int num_of_entries = 5;
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

		result = create_match_parser_meta_type_pipe(ports[port_id], port_id, &type_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create type pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_match_parser_meta_type_pipe_entries(type_pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add type entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_match_parser_meta_ok_pipe(ports[port_id], type_pipe, &ok_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create ok pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_match_parser_meta_ok_pipe_entry(ok_pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add ok entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_fragmented_pipe(ports[port_id], &frag_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create fragmented pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_fragmented_pipe_entries(frag_pipe, ok_pipe, port_id, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add fragmented entries: %s", doca_error_get_descr(result));
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
