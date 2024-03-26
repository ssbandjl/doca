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

#include <doca_log.h>
#include <doca_flow.h>

#include "doca_error.h"
#include "flow_common.h"

DOCA_LOG_REGISTER(FLOW_ESP);

/* The number of seconds app waits for traffic to come */
#define WAITING_TIME 5

/*
 * Create DOCA Flow pipe with changeable 5 tuple match as root
 *
 * @port [in]: port of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_root_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	uint32_t sequence_number = 2;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	/* ESP packet match */
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_ESP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.tun.type = DOCA_FLOW_TUN_ESP;
	match.tun.esp_spi = 0xffffffff;
	match.tun.esp_sn = rte_cpu_to_be_32(sequence_number);

	pipe_cfg.attr.name = "ESP_ROOT_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.attr.nb_flows = 2;
	pipe_cfg.match = &match;
	pipe_cfg.port = port;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = NULL;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry to the root pipe that forwards the traffic to specific comparison pipe.
 *
 * @pipe [in]: pipe of the entries.
 * @next_pipe [in]: comparison pipe to forward the matched traffic.
 * @spi [in]: the ESP SPI to match on in this entry.
 * @status [in]: user context for adding entry.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_root_pipe_entry(struct doca_flow_pipe *pipe, struct doca_flow_pipe *next_pipe, uint32_t spi,
		    struct entries_status *status)
{
	struct doca_flow_pipe_entry *entry;
	struct doca_flow_match match;
	struct doca_flow_fwd fwd;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));

	match.tun.esp_spi = rte_cpu_to_be_32(spi);

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = next_pipe;

	return doca_flow_pipe_add_entry(0, pipe, &match, NULL, NULL, &fwd, 0, status, &entry);
}

/*
 * Add DOCA Flow pipe entries to the root pipe that forwards the traffic to comparison pipes.
 *
 * @pipe [in]: pipe of the entries.
 * @status [in]: user context for adding entry
 * @gt_pipe [in]: Greater Than pipe to forward the matched traffic.
 * @lt_pipe [in]: Less Than pipe to forward the matched traffic.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_root_pipe_entries(struct doca_flow_pipe *pipe, struct entries_status *status,
		      struct doca_flow_pipe *gt_pipe, struct doca_flow_pipe *lt_pipe)
{
	doca_error_t result;
	uint32_t spi = 8;

	result = add_root_pipe_entry(pipe, gt_pipe, spi, status);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add entry - go to GT pipe: %s", doca_error_get_descr(result));
		return result;
	}

	spi = 5;
	result = add_root_pipe_entry(pipe, lt_pipe, spi, status);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add entry - go to LT pipe: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

/*
 * Add DOCA Flow pipe use condition API according to ESP sequence number.
 *
 * @port [in]: port of the pipe
 * @name [in]: name of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_comparisom_pipe(struct doca_flow_port *port, const char *name, struct doca_flow_pipe **pipe)
{
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = name;
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.port = port;

	return doca_flow_pipe_create(&pipe_cfg, NULL, NULL, pipe);
}

/*
 * Add DOCA Flow pipe entry to the comparison pipe.
 *
 * @pipe [in]: pipe of the entries.
 * @port_id [in]: port ID for forwarding to.
 * @gt [in]: indicator whether compare operation is "GT (>)" or "LT (<)".
 * @status [in]: user context for adding entry.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
add_comparison_pipe_entry(struct doca_flow_pipe *pipe, uint16_t port_id, bool gt, struct entries_status *status)
{
	struct doca_flow_pipe_entry *entry;
	struct doca_flow_match match;
	struct doca_flow_match_condition condition;
	struct doca_flow_fwd fwd;

	memset(&match, 0, sizeof(match));
	memset(&condition, 0, sizeof(condition));
	memset(&fwd, 0, sizeof(fwd));

	condition.operation = gt ? DOCA_FLOW_COMPARE_GT : DOCA_FLOW_COMPARE_LT;
	/* Argument field is always ESP sequence number */
	condition.field_op.a.field_string = "tunnel.esp.sn";
	condition.field_op.a.bit_offset = 0;
	/* Base is immediate value, so the string should be NULL and value is taken from match structure */
	condition.field_op.b.field_string = NULL;
	condition.field_op.b.bit_offset = 0;
	condition.field_op.width = 32;

	/*
	 * The immediate value to compare with ESP sequence number field is provided in the match structure.
	 * The value is hard-coded 3 (arbitrary).
	 */
	match.tun.esp_sn = 3;

	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	return doca_flow_pipe_control_add_entry(0 /* queue */, 0 /* priority */, pipe,
						&match, NULL /* match_mask */, &condition,
						NULL, NULL, NULL, NULL, &fwd, status, &entry);
}

/*
 * Run flow_esp sample.
 *
 * This sample tests ESP header fields in two ways:
 *  1. Exact match on SPI field in the root pipe.
 *  2. Condition match for sequence number field.
 *
 * @nb_queues [in]: number of queues the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
flow_esp(int nb_queues)
{
	int nb_ports = 2;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *root_pipe;
	struct doca_flow_pipe *gt_pipe;
	struct doca_flow_pipe *lt_pipe;
	struct entries_status status;
	int num_of_entries = 4;
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

		result = create_comparisom_pipe(ports[port_id], "ESP_GT_PIPE", &gt_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create GT comparison pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_comparison_pipe_entry(gt_pipe, port_id, true, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add GT comparison pipe entries: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_comparisom_pipe(ports[port_id], "ESP_LT_PIPE", &lt_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create LT comparison pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_comparison_pipe_entry(lt_pipe, port_id, false, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add LT comparison pipe entries: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_root_pipe(ports[port_id], &root_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create root pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_root_pipe_entries(root_pipe, &status, gt_pipe, lt_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add root pipe entries: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = doca_flow_entries_process(ports[port_id], 0, DEFAULT_TIMEOUT_US, 0);
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

	DOCA_LOG_INFO("Wait %u seconds for packets to arrive", WAITING_TIME);
	sleep(WAITING_TIME);

	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return DOCA_SUCCESS;
}
