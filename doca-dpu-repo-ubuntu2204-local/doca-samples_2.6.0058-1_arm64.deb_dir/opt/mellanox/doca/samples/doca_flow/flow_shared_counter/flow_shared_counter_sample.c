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

#include <string.h>
#include <unistd.h>

#include <rte_byteorder.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

DOCA_LOG_REGISTER(FLOW_SHARED_COUNTER);

/* Set match l4 port */
#define SET_L4_PORT(layer, port, value) \
do {\
	if (match.layer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_TCP)\
		match.layer.tcp.l4_port.port = (value);\
	else if (match.layer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_UDP)\
		match.layer.udp.l4_port.port = (value);\
} while (0)

/*
 * Create DOCA Flow pipe with 5 tuple match and monitor with shared counter ID
 *
 * @port [in]: port of the pipe
 * @port_id [in]: port ID of the pipe
 * @out_l4_type [in]: l4 type to match: UDP/TCP
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_shared_counter_pipe(struct doca_flow_port *port, int port_id, enum doca_flow_l4_type_ext out_l4_type,
			   struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_monitor monitor;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&match, 0, sizeof(match));
	memset(&monitor, 0, sizeof(monitor));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "SHARED_COUNTER_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	pipe_cfg.monitor = &monitor;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.attr.nb_actions = NB_ACTIONS_ARR;
	pipe_cfg.port = port;

	/* 5 tuple match */
	match.outer.l4_type_ext = out_l4_type;
	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;
	SET_L4_PORT(outer, src_port, 0xffff);
	SET_L4_PORT(outer, dst_port, 0xffff);

	/* monitor with changeable shared counter ID */
	monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
	monitor.shared_counter.shared_counter_id = 0xffffffff;

	/* forwarding traffic to other port */
	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry to the shared counter pipe
 *
 * @pipe [in]: pipe of the entry
 * @out_l4_type [in]: l4 type to match: UDP/TCP
 * @shared_counter_id [in]: ID of the shared counter
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_shared_counter_pipe_entry(struct doca_flow_pipe *pipe, enum doca_flow_l4_type_ext out_l4_type,
			      uint32_t shared_counter_id, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_monitor monitor;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	/* example 5-tuple to match */
	doca_be32_t dst_ip_addr = BE_IPV4_ADDR(8, 8, 8, 8);
	doca_be32_t src_ip_addr = BE_IPV4_ADDR(1, 2, 3, 4);
	doca_be16_t dst_port = rte_cpu_to_be_16(80);
	doca_be16_t src_port = rte_cpu_to_be_16(1234);

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&monitor, 0, sizeof(monitor));

	/* set shared counter ID */
	monitor.shared_counter.shared_counter_id = shared_counter_id;

	match.outer.ip4.dst_ip = dst_ip_addr;
	match.outer.ip4.src_ip = src_ip_addr;
	match.outer.l4_type_ext = out_l4_type;
	SET_L4_PORT(outer, dst_port, dst_port);
	SET_L4_PORT(outer, src_port, src_port);

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, &monitor, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

/*
 * Add DOCA Flow control pipe
 *
 * @port [in]: port of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_control_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "CONTROL_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.port = port;

	return doca_flow_pipe_create(&pipe_cfg, NULL, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entries to the control pipe. First entry forwards UDP packets to udp_pipe and the second
 * forwards TCP packets to tcp_pipe
 *
 * @control_pipe [in]: pipe of the entries
 * @tcp_pipe [in]: pointer to the TCP pipe to forward packets to
 * @udp_pipe [in]: pointer to the UDP pipe to forward packets to
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_control_pipe_entries(struct doca_flow_pipe *control_pipe, struct doca_flow_pipe *tcp_pipe,
			 struct doca_flow_pipe *udp_pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_fwd fwd;
	uint8_t priority = 0;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));

	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_UDP;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = udp_pipe;

	result = doca_flow_pipe_control_add_entry(0, priority, control_pipe, &match, NULL,
						  NULL, NULL, NULL, NULL, NULL, &fwd, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add control pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));

	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_TCP;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = tcp_pipe;

	result = doca_flow_pipe_control_add_entry(0, priority, control_pipe, &match, NULL,
						  NULL, NULL, NULL, NULL, NULL, &fwd, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add control pipe entry: %s", doca_error_get_descr(result));
		return result;
	}
	return DOCA_SUCCESS;
}

/*
 * Run flow_shared_counter sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
flow_shared_counter(int nb_queues)
{
	int nb_ports = 2;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *tcp_pipe, *udp_pipe, *pipe;
	int port_id;
	uint32_t shared_counter_ids[] = {0, 1};
	struct doca_flow_shared_resource_result query_results_array[nb_ports];
	struct doca_flow_shared_resource_cfg cfg = {.domain = DOCA_FLOW_PIPE_DOMAIN_DEFAULT};
	struct entries_status status;
	int num_of_entries = 4;
	doca_error_t result;

	nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_COUNT] = 2;

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
		/* config and bind shared counter to port */
		result = doca_flow_shared_resource_cfg(DOCA_FLOW_SHARED_RESOURCE_COUNT, port_id, &cfg);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to configure shared counter to port %d", port_id);
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = doca_flow_shared_resources_bind(DOCA_FLOW_SHARED_RESOURCE_COUNT,
							 &shared_counter_ids[port_id], 1, ports[port_id]);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to bind shared counter to pipe");
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_shared_counter_pipe(ports[port_id], port_id, DOCA_FLOW_L4_TYPE_EXT_TCP, &tcp_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_shared_counter_pipe_entry(tcp_pipe, DOCA_FLOW_L4_TYPE_EXT_TCP, shared_counter_ids[port_id], &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_shared_counter_pipe(ports[port_id], port_id, DOCA_FLOW_L4_TYPE_EXT_UDP, &udp_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_shared_counter_pipe_entry(udp_pipe, DOCA_FLOW_L4_TYPE_EXT_UDP, shared_counter_ids[port_id], &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}
		result = create_control_pipe(ports[port_id], &pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create control pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_control_pipe_entries(pipe, tcp_pipe, udp_pipe, &status);
		if (result != DOCA_SUCCESS) {
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

	result = doca_flow_shared_resources_query(DOCA_FLOW_SHARED_RESOURCE_COUNT, shared_counter_ids,
						  query_results_array, nb_ports);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to query entry: %s", doca_error_get_descr(result));
		stop_doca_flow_ports(nb_ports, ports);
		doca_flow_destroy();
		return result;
	}

	for (port_id = 0; port_id < nb_ports; port_id++) {
		DOCA_LOG_INFO("Port %d:", port_id);
		DOCA_LOG_INFO("Total bytes: %ld", query_results_array[port_id].counter.total_bytes);
		DOCA_LOG_INFO("Total packets: %ld", query_results_array[port_id].counter.total_pkts);
	}

	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return DOCA_SUCCESS;
}
