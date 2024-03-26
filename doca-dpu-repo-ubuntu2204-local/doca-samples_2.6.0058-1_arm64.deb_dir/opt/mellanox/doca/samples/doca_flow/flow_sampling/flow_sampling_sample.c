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

#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <rte_ethdev.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "doca_error.h"
#include "flow_common.h"

DOCA_LOG_REGISTER(FLOW_SAMPLING);

/* The number of bits in random field */
#define RANDOM_WIDTH 16

/* The number of numbers in random field range */
#define RANDOM_TOTAL_RANGE (1 << RANDOM_WIDTH)

/* Get the percentage according to part and total */
#define GET_PERCENTAGE(part, total) (((double)part / (double)total) * 100)

/* The number of seconds app waits for traffic to come */
#define WAITING_TIME 15

/*
 * Create DOCA Flow pipe with changeable 5 tuple match as root
 *
 * @port [in]: port of the pipe
 * @next_pipe [in]: next pipe pointer
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_root_pipe(struct doca_flow_port *port,
				     struct doca_flow_pipe *next_pipe,
				     struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_monitor monitor;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&match, 0, sizeof(match));
	memset(&monitor, 0, sizeof(monitor));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	/* 5 tuple match */
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_TCP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;
	match.outer.tcp.l4_port.src_port = 0xffff;
	match.outer.tcp.l4_port.dst_port = 0xffff;

	/* Add counter to see how many packet arrived before sampling */
	monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	pipe_cfg.attr.name = "ROOT_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.attr.nb_flows = 1;
	pipe_cfg.match = &match;
	pipe_cfg.monitor = &monitor;
	pipe_cfg.port = port;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = next_pipe;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry to the root pipe that forwards the traffic to specific random pipe.
 *
 * @pipe [in]: pipe of the entries.
 * @status [in]: user context for adding entry.
 * @entry [out]: created entry pointer.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t add_root_pipe_entry(struct doca_flow_pipe *pipe,
					struct entries_status *status,
					struct doca_flow_pipe_entry **entry)
{
	struct doca_flow_match match;

	memset(&match, 0, sizeof(match));

	match.outer.ip4.dst_ip = BE_IPV4_ADDR(8, 8, 8, 8);
	match.outer.ip4.src_ip = BE_IPV4_ADDR(1, 1, 1, 1);
	match.outer.tcp.l4_port.dst_port = rte_cpu_to_be_16(80);
	match.outer.tcp.l4_port.src_port = rte_cpu_to_be_16(1234);

	DOCA_LOG_DBG("Adding root pipe entry matching of: "
		     "IPv4(src='1.1.1.1',dst='8.8.8.8') and TCP(src_port=1234,dst_port=80)");

	return doca_flow_pipe_add_entry(0, pipe, &match, NULL, NULL, NULL, 0, status, entry);
}

/*
 * Calculate the random value used to achieve given certain percentage.
 *
 * @percentage [in]: the certain percentage user wish to get in sampling.
 * @return: value to use in match.parser_meta.random field for getting this percentage along with
 *          "parser_meta.random.value" string in condition argument.
 */
static uint16_t get_random_value(double percentage)
{
	uint16_t random_value;
	double temp;

	temp = (percentage / 100) * RANDOM_TOTAL_RANGE;
	random_value = (uint16_t)temp;
	temp = GET_PERCENTAGE(random_value, RANDOM_TOTAL_RANGE);

	DOCA_LOG_DBG("Using random value 0x%04x for sampling %g%% of traffic (%g%% requested)",
		     random_value, temp, percentage);

	return random_value;
}

/*
 * Add DOCA Flow pipe for sampling according to random value
 *
 * @port [in]: port of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_random_sampling_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "SAMPLING_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.port = port;

	return doca_flow_pipe_create(&pipe_cfg, NULL, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry to the random sampling pipe.
 *
 * @pipe [in]: pipe of the entries
 * @status [in]: user context for adding entry
 * @percentage [in]: the certain percentage user wish to get in sampling
 * @entry [out]: created entry pointer.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t add_random_sampling_pipe_entry(struct doca_flow_pipe *pipe,
						   struct entries_status *status,
						   double percentage,
						   struct doca_flow_pipe_entry **entry)
{
	struct doca_flow_match match;
	struct doca_flow_match_condition condition;
	struct doca_flow_monitor monitor;
	struct doca_flow_fwd fwd;

	memset(&match, 0, sizeof(match));
	memset(&condition, 0, sizeof(condition));
	memset(&monitor, 0, sizeof(monitor));
	memset(&fwd, 0, sizeof(fwd));

	condition.operation = DOCA_FLOW_COMPARE_LT;
	/* Argument field is random */
	condition.field_op.a.field_string = "parser_meta.random.value";
	condition.field_op.a.bit_offset = 0;
	/* Base is immediate value, so the string should be NULL and value is taken from match structure */
	condition.field_op.b.field_string = NULL;
	condition.field_op.b.bit_offset = 0;
	condition.field_op.width = RANDOM_WIDTH;

	/*
	 * The immediate value to compare with random field is provided in the match structure.
	 * Match mask structure is not relevant here since it is masked in argument using offset + width.
	 */
	match.parser_meta.random = get_random_value(percentage);

	/* Add counter to see how many packet are sampled */
	monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = 1;

	return doca_flow_pipe_control_add_entry(0 /* queue */,
						0 /* priority */,
						pipe,
						&match,
						NULL /* match_mask */,
						&condition,
						NULL /* actions */,
						NULL /* actions_mask */,
						NULL /* action_descs */,
						&monitor,
						&fwd,
						status,
						entry);
}

/*
 * Get results about cretain percentage sampling.
 *
 * @root_entry [in]: entry sent packets to sampling.
 * @random_entry [in]: entry samples cretain percentage of traffic.
 * @reqested_percentage [in]: the certain percentage user wished to get in sampling.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t random_sampling_results(struct doca_flow_pipe_entry *root_entry,
					    struct doca_flow_pipe_entry *random_entry,
					    double reqested_percentage)
{
	struct doca_flow_query root_query_stats;
	struct doca_flow_query random_query_stats;
	double actuall_percentage;
	uint32_t total_packets;
	uint32_t nb_sampled_packets;
	doca_error_t result;

	result = doca_flow_query_entry(root_entry, &root_query_stats);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to query root entry: %s", doca_error_get_descr(result));
		return result;
	}

	total_packets = root_query_stats.total_pkts;
	if (total_packets == 0)
		return DOCA_SUCCESS;

	result = doca_flow_query_entry(random_entry, &random_query_stats);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to query random entry: %s", doca_error_get_descr(result));
		return result;
	}

	nb_sampled_packets = random_query_stats.total_pkts;
	actuall_percentage = GET_PERCENTAGE(nb_sampled_packets, total_packets);

	DOCA_LOG_INFO("Sampling result information (%g%% is requested):", reqested_percentage);
	DOCA_LOG_INFO("This pipeline samples %u packets which is %g%% of the traffic (%u/%u)",
		      nb_sampled_packets,
		      actuall_percentage,
		      nb_sampled_packets,
		      total_packets);

	return DOCA_SUCCESS;
}

/*
 * Run flow_sampling sample.
 *
 * This sample tests the sampling certain persentage of traffic.
 *
 * @nb_queues [in]: number of queues the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t flow_sampling(int nb_queues)
{
	int nb_ports = 1;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_flow_port *switch_port;
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *root_pipe;
	struct doca_flow_pipe *sampling_pipe;
	struct doca_flow_pipe_entry *root_entry;
	struct doca_flow_pipe_entry *random_entry;
	struct entries_status status;
	double requested_percentage = 35;
	int num_of_entries = 2;
	doca_error_t result;

	memset(&status, 0, sizeof(status));
	resource.nb_counters = num_of_entries;

	result = init_doca_flow(nb_queues, "switch,hws", resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA Flow: %s", doca_error_get_descr(result));
		return result;
	}

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	result = init_doca_flow_ports(nb_ports, ports, false, dev_arr);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA ports: %s", doca_error_get_descr(result));
		doca_flow_destroy();
		return result;
	}

	switch_port = doca_flow_port_switch_get(NULL);

	result = create_random_sampling_pipe(switch_port, &sampling_pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create random sampling pipe: %s", doca_error_get_descr(result));
		stop_doca_flow_ports(nb_ports, ports);
		doca_flow_destroy();
		return result;
	}

	result = add_random_sampling_pipe_entry(sampling_pipe, &status, requested_percentage, &random_entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add random sampling pipe entry: %s", doca_error_get_descr(result));
		stop_doca_flow_ports(nb_ports, ports);
		doca_flow_destroy();
		return result;
	}

	result = create_root_pipe(switch_port, sampling_pipe, &root_pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create root pipe: %s", doca_error_get_descr(result));
		stop_doca_flow_ports(nb_ports, ports);
		doca_flow_destroy();
		return result;
	}

	result = add_root_pipe_entry(root_pipe, &status, &root_entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add root pipe entry: %s", doca_error_get_descr(result));
		stop_doca_flow_ports(nb_ports, ports);
		doca_flow_destroy();
		return result;
	}

	result = doca_flow_entries_process(switch_port, 0, DEFAULT_TIMEOUT_US, 0);
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

	DOCA_LOG_INFO("Wait %u seconds for packets to arrive", WAITING_TIME);
	sleep(WAITING_TIME);

	/* Show the results for sampling */
	result = random_sampling_results(root_entry, random_entry, requested_percentage);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to show sampling results: %s", doca_error_get_descr(result));
		stop_doca_flow_ports(nb_ports, ports);
		doca_flow_destroy();
		return result;
	}

	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return DOCA_SUCCESS;
}
