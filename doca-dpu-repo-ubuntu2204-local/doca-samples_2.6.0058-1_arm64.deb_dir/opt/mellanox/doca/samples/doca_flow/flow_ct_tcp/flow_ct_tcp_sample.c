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

#include <unistd.h>

#include <rte_ethdev.h>

#include <doca_log.h>
#include <doca_flow.h>
#include <doca_flow_ct.h>

#include "flow_ct_common.h"
#include "flow_common.h"

#define PACKET_BURST 128

DOCA_LOG_REGISTER(FLOW_CT_TCP);

/*
 * Create RSS pipe
 *
 * @port [in]: Pipe port
 * @status [in]: user context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_rss_pipe(struct doca_flow_port *port, struct entries_status *status, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_pipe_cfg cfg;
	struct doca_flow_fwd fwd;
	uint16_t rss_queues[1];
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&cfg, 0, sizeof(cfg));
	memset(&fwd, 0, sizeof(fwd));

	cfg.attr.name = "RSS_PIPE";
	cfg.attr.is_root = true;
	cfg.match = &match;
	cfg.port = port;

	/* RSS queue - send matched traffic to queue 0  */
	rss_queues[0] = 0;
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_inner_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP;
	fwd.num_of_queues = 1;

	result = doca_flow_pipe_create(&cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create RSS pipe: %s", doca_error_get_descr(result));
		return result;
	}

	/* Match on any packet */
	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, &fwd, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add RSS pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process RSS entry: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Create egress pipe
 *
 * @port [in]: Pipe port
 * @port_id [in]: Next pipe port id
 * @status [in]: user context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_egress_pipe(struct doca_flow_port *port, int port_id, struct entries_status *status,
		   struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_pipe_cfg cfg;
	struct doca_flow_fwd fwd;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&cfg, 0, sizeof(cfg));
	memset(&fwd, 0, sizeof(fwd));

	cfg.attr.name = "EGRESS_PIPE";
	cfg.match = &match;
	cfg.port = port;

	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id;

	result = doca_flow_pipe_create(&cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create EGRESS pipe: %s", doca_error_get_descr(result));
		return result;
	}

	/* Match on any packet */
	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, &fwd, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add EGRESS pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process EGRESS entry: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Create CT miss pipe
 *
 * @port [in]: Pipe port
 * @fwd_pipe [in]: Forward pipe pointer
 * @status [in]: user context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_ct_miss_pipe(struct doca_flow_port *port, struct doca_flow_pipe *fwd_pipe, struct entries_status *status,
		    struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_match mask;
	struct doca_flow_pipe_cfg cfg;
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&mask, 0, sizeof(mask));
	memset(&cfg, 0, sizeof(cfg));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd));

	cfg.attr.name = "CT_MISS_PIPE";
	cfg.match = &match;
	cfg.match_mask = &mask;
	cfg.port = port;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	fwd_miss.type = DOCA_FLOW_FWD_PIPE;
	fwd_miss.next_pipe = fwd_pipe;

	result = doca_flow_pipe_create(&cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create CT miss pipe: %s", doca_error_get_descr(result));
		return result;
	}

	/* Match on any packet */
	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, &fwd, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add CT miss pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process CT miss entry: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Create DOCA Flow TCP state pipe to filter state on known TCP session
 *
 * @port [in]: Pipe port
 * @status [in]: User context for adding entry
 * @fwd_pipe [in]: Forward pipe
 * @fwd_miss_pipe [in]: Forward miss pipe
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_tcp_flags_filter_pipe(struct doca_flow_port *port, struct entries_status *status,
			     struct doca_flow_pipe *fwd_pipe, struct doca_flow_pipe *fwd_miss_pipe,
			     struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_match mask;
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&mask, 0, sizeof(mask));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "TCP_FLAGS_FILTER_PIPE";
	pipe_cfg.match = &match;
	pipe_cfg.port = port;

	/* Match on non SYN, FIN and RST packets */
	match.outer.tcp.flags = 0xff;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;

	mask.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
	mask.outer.tcp.flags = DOCA_FLOW_MATCH_TCP_FLAG_SYN | DOCA_FLOW_MATCH_TCP_FLAG_FIN |
			       DOCA_FLOW_MATCH_TCP_FLAG_RST;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	fwd_miss.type = DOCA_FLOW_FWD_PIPE;
	fwd_miss.next_pipe = fwd_miss_pipe;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create TCP_FLAGS_FILTER pipe: %s", doca_error_get_descr(result));
		return result;
	}

	match.outer.tcp.flags = 0;
	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, NULL, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create TCP flags filter pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process TCP flags filter entry: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Create CT pipe
 *
 * @port [in]: Pipe port
 * @fwd_pipe [in]: Forward pipe pointer
 * @fwd_miss_pipe [in]: Forward miss pipe pointer
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_ct_pipe(struct doca_flow_port *port, struct doca_flow_pipe *fwd_pipe, struct doca_flow_pipe *fwd_miss_pipe,
	       struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_match mask;
	struct doca_flow_pipe_cfg cfg;
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&mask, 0, sizeof(mask));
	memset(&cfg, 0, sizeof(cfg));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd));

	cfg.attr.name = "CT_PIPE";
	cfg.attr.type = DOCA_FLOW_PIPE_CT;
	cfg.match = &match;
	cfg.match_mask = &mask;
	cfg.port = port;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	fwd_miss.type = DOCA_FLOW_FWD_PIPE;
	fwd_miss.next_pipe = fwd_miss_pipe;

	result = doca_flow_pipe_create(&cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to create CT pipe: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Create TCP pipe
 *
 * @port [in]: Pipe port
 * @fwd_pipe [in]: Next pipe pointer
 * @status [in]: user context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_tcp_pipe(struct doca_flow_port *port, struct doca_flow_pipe *fwd_pipe, struct entries_status *status,
		struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_pipe_cfg cfg;
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&cfg, 0, sizeof(cfg));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));

	cfg.attr.name = "TCP_PIPE";
	cfg.attr.is_root = true;
	cfg.match = &match;
	cfg.port = port;

	/* Match IPv4 TCP packets */
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_TCP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	/* Drop non TCP packets */
	fwd_miss.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create TCP pipe: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, &fwd, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add TCP pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process TCP entry: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Parse TCP packet to update CT tables
 *
 * @packet [in]: Packet to parse
 * @match_o [out]: Origin match struct to fill
 * @match_r [out]: Reply match struct to fill
 * @tcp_state [out]: Packet TCP state
 */
static void
parse_packet(struct rte_mbuf *packet, struct doca_flow_ct_match *match_o, struct doca_flow_ct_match *match_r,
	     uint8_t *tcp_state)
{
	uint8_t *l4_hdr;
	struct rte_ipv4_hdr *ipv4_hdr;
	const struct rte_tcp_hdr *tcp_hdr;

	ipv4_hdr = rte_pktmbuf_mtod_offset(packet, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));

	match_o->ipv4.src_ip = ipv4_hdr->src_addr;
	match_o->ipv4.dst_ip = ipv4_hdr->dst_addr;
	match_r->ipv4.src_ip = match_o->ipv4.dst_ip;
	match_r->ipv4.dst_ip = match_o->ipv4.src_ip;

	l4_hdr = (typeof(l4_hdr))ipv4_hdr + rte_ipv4_hdr_len(ipv4_hdr);
	tcp_hdr = (typeof(tcp_hdr))l4_hdr;

	match_o->ipv4.l4_port.src_port = tcp_hdr->src_port;
	match_o->ipv4.l4_port.dst_port = tcp_hdr->dst_port;
	match_r->ipv4.l4_port.src_port = match_o->ipv4.l4_port.dst_port;
	match_r->ipv4.l4_port.dst_port = match_o->ipv4.l4_port.src_port;

	match_o->ipv4.next_proto = DOCA_PROTO_TCP;
	match_r->ipv4.next_proto = DOCA_PROTO_TCP;

	*tcp_state = tcp_hdr->tcp_flags;
}

/*
 * Dequeue packets from DPDK queues, parse and update CT tables with new connection 5 tuple
 *
 * @port [in]: Port id to which an entry should be inserted
 * @ct_queue [in]: DOCA Flow CT queue number
 * @ct_status [in]: User context for adding CT entry
 * @entry [in/out]: CT entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
process_packets(struct doca_flow_port *port, uint16_t ct_queue, struct entries_status *ct_status, struct doca_flow_pipe_entry **entry)
{
	struct rte_mbuf *packets[PACKET_BURST];
	struct doca_flow_ct_match match_o;
	struct doca_flow_ct_match match_r;
	uint32_t flags = DOCA_FLOW_CT_ENTRY_FLAGS_NO_WAIT | DOCA_FLOW_CT_ENTRY_FLAGS_DIR_ORIGIN;
	uint8_t tcp_state;
	doca_error_t result;
	int i, nb_packets, nb_process = 0;

	memset(&match_o, 0, sizeof(match_o));
	memset(&match_r, 0, sizeof(match_r));

	nb_packets = rte_eth_rx_burst(0, 0, packets, PACKET_BURST);
	if (nb_packets == 0) {
		DOCA_LOG_INFO("Sample didn't receive packets to process");
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("%d packets received on rx_burst()", nb_packets);
	for (i = 0; i < PACKET_BURST && i < nb_packets; i++) {
		parse_packet(packets[i], &match_o, &match_r, &tcp_state);
		if (tcp_state & DOCA_FLOW_MATCH_TCP_FLAG_SYN) {
			result = doca_flow_ct_add_entry(ct_queue, NULL, flags, &match_o, &match_r, 0, 0, 0, ct_status,
							entry);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to add CT pipe an entry: %s", doca_error_get_descr(result));
				return result;
			}
			nb_process++;
		} else if (tcp_state & DOCA_FLOW_MATCH_TCP_FLAG_FIN || tcp_state & DOCA_FLOW_MATCH_TCP_FLAG_RST) {
			result = doca_flow_ct_rm_entry(ct_queue, NULL, flags, *entry);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to add CT pipe an entry: %s", doca_error_get_descr(result));
				return result;
			}
		}
	}

	while (ct_status->nb_processed != nb_process) {
		result = doca_flow_entries_process(port, ct_queue, DEFAULT_TIMEOUT_US, 0);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to process Flow CT entries: %s", doca_error_get_descr(result));
			return result;
		}

		if (ct_status->failure) {
			DOCA_LOG_ERR("Flow CT entries process returned with a failure");
			return DOCA_ERROR_BAD_STATE;
		}
	}
	ct_status->nb_processed = 0;

	rte_eth_tx_burst(1, 0, packets, nb_packets);

	return DOCA_SUCCESS;
}

/*
 * Run flow_ct_tcp sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @ct_dev [in]: Flow CT device
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
flow_ct_tcp(uint16_t nb_queues, struct doca_dev *ct_dev)
{
	const int nb_ports = 2, nb_entries = 5;
	struct doca_flow_resources resource;
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_pipe_entry *tcp_entry;
	struct doca_flow_pipe *egress_pipe, *ct_pipe, *ct_miss_pipe, *tcp_flags_filter_pipe, *rss_pipe, *tcp_pipe;
	struct doca_flow_port *ports[nb_ports];
	struct doca_flow_meta o_zone_mask, o_modify_mask, r_zone_mask, r_modify_mask;
	struct doca_dev *dev_arr[nb_ports];
	struct entries_status ctrl_status, ct_status;
	uint32_t ct_flags, nb_arm_queues = 1, nb_ctrl_queues = 1, nb_user_actions = 0, nb_ipv4_sessions = 1024,
			   nb_ipv6_sessions = 0; /* On BF2 should always be 0 */
	uint16_t ct_queue = nb_queues;
	doca_error_t result;

	memset(&ctrl_status, 0, sizeof(ctrl_status));
	memset(&ct_status, 0, sizeof(ct_status));
	memset(&resource, 0, sizeof(resource));

	resource.nb_counters = 1;

	result = init_doca_flow(nb_queues, "switch,hws", resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA Flow: %s", doca_error_get_descr(result));
		return result;
	}

	/* Dont use zone masking */
	memset(&o_zone_mask, 0, sizeof(o_zone_mask));
	memset(&o_modify_mask, 0, sizeof(o_modify_mask));
	memset(&r_zone_mask, 0, sizeof(r_zone_mask));
	memset(&r_modify_mask, 0, sizeof(r_modify_mask));

	ct_flags = DOCA_FLOW_CT_FLAG_NO_AGING;
	result = init_doca_flow_ct(ct_dev, ct_flags, nb_arm_queues, nb_ctrl_queues, nb_user_actions, NULL,
				   nb_ipv4_sessions, nb_ipv6_sessions, false, &o_zone_mask, &o_modify_mask, false,
				   &r_zone_mask, &r_modify_mask);
	if (result != DOCA_SUCCESS) {
		doca_flow_destroy();
		return result;
	}

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	dev_arr[0] = ct_dev;
	result = init_doca_flow_ports(nb_ports, ports, false, dev_arr);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA ports: %s", doca_error_get_descr(result));
		doca_flow_ct_destroy();
		doca_flow_destroy();
		return result;
	}

	result = create_rss_pipe(ports[0], &ctrl_status, &rss_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_egress_pipe(ports[0], 1, &ctrl_status, &egress_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_tcp_flags_filter_pipe(ports[0], &ctrl_status, egress_pipe, rss_pipe, &tcp_flags_filter_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_ct_miss_pipe(ports[0], rss_pipe, &ctrl_status, &ct_miss_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_ct_pipe(ports[0], tcp_flags_filter_pipe, ct_miss_pipe, &ct_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_tcp_pipe(ports[0], ct_pipe, &ctrl_status, &tcp_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = doca_flow_entries_process(ports[0], 0, DEFAULT_TIMEOUT_US, nb_entries);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to process Flow CT entries: %s", doca_error_get_descr(result));
		goto cleanup;
	}

	if (ctrl_status.nb_processed != nb_entries || ctrl_status.failure) {
		DOCA_LOG_ERR("Failed to process entries");
		result = DOCA_ERROR_BAD_STATE;
		goto cleanup;
	}

	DOCA_LOG_INFO("Wait few seconds for 'SYN' packet to arrive");
	sleep(5);

	result = process_packets(ports[0], ct_queue, &ct_status, &tcp_entry);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	DOCA_LOG_INFO("TCP session was created, waiting for 'FIN'/'RST' packet before ending the session");
	sleep(7);

	result = process_packets(ports[0], ct_queue, &ct_status, &tcp_entry);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	DOCA_LOG_INFO("TCP session was ended");

cleanup:
	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_ct_destroy();
	doca_flow_destroy();

	return result;
}
