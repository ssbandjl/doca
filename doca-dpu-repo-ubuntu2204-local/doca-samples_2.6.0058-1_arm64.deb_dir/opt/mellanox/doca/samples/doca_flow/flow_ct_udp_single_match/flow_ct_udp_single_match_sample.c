/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

DOCA_LOG_REGISTER(FLOW_CT_UDP_SINGLE_MATCH);

/*
 * Create RSS pipe
 *
 * @port [in]: Pipe port
 * @status [in]: User context for adding entry
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
	fwd.rss_inner_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
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
 * Create UDP pipe
 *
 * @port [in]: Pipe port
 * @fwd_pipe [in]: Next pipe pointer
 * @status [in]: User context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_udp_pipe(struct doca_flow_port *port, struct doca_flow_pipe *fwd_pipe, struct entries_status *status,
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

	cfg.attr.name = "UDP_PIPE";
	cfg.attr.is_root = true;
	cfg.match = &match;
	cfg.port = port;

	/* Match IPv4 UDP packets */
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_UDP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	/* Drop non UDP packets */
	fwd_miss.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create UDP pipe: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, &fwd, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add UDP pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process UDP entry: %s", doca_error_get_descr(result));

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
		DOCA_LOG_ERR("Failed to add CT pipe: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Create VxLAN encapsulation pipe
 *
 * @port [in]: Pipe port
 * @port_id [in]: Forward port ID
 * @status [in]: User context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_vxlan_encap_pipe(struct doca_flow_port *port, int port_id, struct entries_status *status,
			struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_actions *actions_list[] = {&actions};
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_action_descs descs, *descs_arr[NB_ACTIONS_ARR];
	struct doca_flow_action_desc desc_array[NB_ACTIONS_ARR] = {0};
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "VXLAN_ENCAP_PIPE";
	pipe_cfg.attr.is_root = true;
	pipe_cfg.attr.domain = DOCA_FLOW_PIPE_DOMAIN_EGRESS;
	pipe_cfg.match = &match;
	pipe_cfg.actions = actions_list;
	pipe_cfg.attr.nb_actions = 1;
	pipe_cfg.action_descs = descs_arr;
	pipe_cfg.port = port;

	actions.has_encap = true;
	SET_MAC_ADDR(actions.encap.outer.eth.src_mac, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
	SET_MAC_ADDR(actions.encap.outer.eth.dst_mac, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
	actions.encap.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	actions.encap.outer.ip4.src_ip = 0xffffffff;
	actions.encap.outer.ip4.dst_ip = 0xffffffff;
	actions.encap.outer.ip4.ttl = 0xff;
	actions.encap.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
	actions.encap.outer.udp.l4_port.dst_port = RTE_BE16(DOCA_VXLAN_DEFAULT_PORT);
	actions.encap.tun.type = DOCA_FLOW_TUN_VXLAN;
	actions.encap.tun.vxlan_tun_id = 0xffffffff;

	desc_array[0].type = DOCA_FLOW_ACTION_DECAP_ENCAP;
	desc_array[0].decap_encap.is_l2 = true;
	descs.desc_array = desc_array;
	descs.nb_action_desc = NB_ACTIONS_ARR;
	descs_arr[0] = &descs;

	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create VxLAN Encap pipe: %s", doca_error_get_descr(result));
		return result;
	}

	uint8_t src_mac[] = {0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
	uint8_t dst_mac[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

	memset(&actions, 0, sizeof(actions));
	SET_MAC_ADDR(actions.encap.outer.eth.src_mac, src_mac[0], src_mac[1], src_mac[2], src_mac[3], src_mac[4],
		     src_mac[5]);
	SET_MAC_ADDR(actions.encap.outer.eth.dst_mac, dst_mac[0], dst_mac[1], dst_mac[2], dst_mac[3], dst_mac[4],
		     dst_mac[5]);
	actions.encap.outer.ip4.src_ip = BE_IPV4_ADDR(11, 21, 31, 41);
	actions.encap.outer.ip4.dst_ip = BE_IPV4_ADDR(81, 81, 81, 81);
	actions.encap.outer.ip4.ttl = 17;
	actions.encap.tun.vxlan_tun_id = BUILD_VNI(0xadadad);
	actions.action_idx = 0;

	result = doca_flow_pipe_add_entry(0, *pipe, &match, &actions, NULL, NULL, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add VxLAN Encap pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process UDP entry: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Create pipe to count packets based on 5 tuple match
 *
 * @port [in]: Pipe port
 * @fwd_pipe [in]: Next pipe pointer
 * @status [in]: User context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
create_count_pipe(struct doca_flow_port *port, struct doca_flow_pipe *fwd_pipe, struct entries_status *status,
		  struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_monitor monitor;
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&monitor, 0, sizeof(monitor));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "COUNT_PIPE";
	pipe_cfg.match = &match;
	pipe_cfg.monitor = &monitor;
	pipe_cfg.port = port;

	/* 5 tuple match */
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_UDP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;
	match.outer.udp.l4_port.src_port = 0xffff;
	match.outer.udp.l4_port.dst_port = 0xffff;

	monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	fwd_miss.type = DOCA_FLOW_FWD_PIPE;
	fwd_miss.next_pipe = fwd_pipe;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create count pipe: %s", doca_error_get_descr(result));
		return result;
	}

	memset(&match, 0, sizeof(match));
	match.outer.ip4.dst_ip = BE_IPV4_ADDR(8, 8, 8, 8);
	match.outer.ip4.src_ip = BE_IPV4_ADDR(1, 2, 3, 4);
	match.outer.udp.l4_port.dst_port = rte_cpu_to_be_16(80);
	match.outer.udp.l4_port.src_port = rte_cpu_to_be_16(1234);

	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, NULL, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add count pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process count entry: %s", doca_error_get_descr(result));

	return result;
}

/*
 * Parse UDP packet to update CT tables
 *
 * @packet [in]: Packet to parse
 * @match_o [out]: Origin match struct to fill
 */
static void
parse_packet(struct rte_mbuf *packet, struct doca_flow_ct_match *match_o)
{
	uint8_t *l4_hdr;
	struct rte_ipv4_hdr *ipv4_hdr;
	const struct rte_udp_hdr *udp_hdr;

	ipv4_hdr = rte_pktmbuf_mtod_offset(packet, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));

	match_o->ipv4.src_ip = ipv4_hdr->src_addr;
	match_o->ipv4.dst_ip = ipv4_hdr->dst_addr;

	l4_hdr = (typeof(l4_hdr))ipv4_hdr + rte_ipv4_hdr_len(ipv4_hdr);
	udp_hdr = (typeof(udp_hdr))l4_hdr;

	match_o->ipv4.l4_port.src_port = udp_hdr->src_port;
	match_o->ipv4.l4_port.dst_port = udp_hdr->dst_port;

	match_o->ipv4.next_proto = DOCA_PROTO_UDP;
}

/*
 * Dequeue packets from DPDK queues, parse and update CT tables with new connection 5 tuple
 *
 * @port [in]: Port id to which an entry should be inserted
 * @ct_queue [in]: DOCA Flow CT queue number
 * @ct_status [in]: User context for adding CT entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t
process_packets(struct doca_flow_port *port, uint16_t ct_queue, struct entries_status *ct_status)
{
	struct rte_mbuf *packets[PACKET_BURST];
	struct doca_flow_ct_match match_o;
	struct doca_flow_pipe_entry *entry;
	uint32_t flags = DOCA_FLOW_CT_ENTRY_FLAGS_NO_WAIT | DOCA_FLOW_CT_ENTRY_FLAGS_DIR_ORIGIN;
	doca_error_t result;
	int i, nb_packets = 0;

	memset(&match_o, 0, sizeof(match_o));

	nb_packets = rte_eth_rx_burst(0, 0, packets, PACKET_BURST);
	if (nb_packets == 0) {
		DOCA_LOG_INFO("Sample didn't receive packets to process");
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("Sample received %d packets", nb_packets);
	for (i = 0; i < PACKET_BURST && i < nb_packets; i++) {
		parse_packet(packets[i], &match_o);
		/* Add origin match only */
		result = doca_flow_ct_add_entry(ct_queue, NULL, flags, &match_o, NULL, 0, 0, 0, ct_status, &entry);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add CT pipe an entry: %s", doca_error_get_descr(result));
			return result;
		}
	}

	while (ct_status->nb_processed != nb_packets) {
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

	return DOCA_SUCCESS;
}

/*
 * Run flow_ct_udp_single_match sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @ct_dev [in]: Flow CT device
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
flow_ct_udp_single_match(uint16_t nb_queues, struct doca_dev *ct_dev)
{
	const int nb_ports = 2, nb_entries = 4;
	struct doca_flow_resources resource;
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_pipe *rss_pipe, *encap_pipe, *count_pipe, *ct_pipe, *udp_pipe;
	struct doca_flow_port *ports[nb_ports];
	struct doca_flow_meta o_zone_mask, o_modify_mask, r_zone_mask, r_modify_mask;
	struct doca_dev *dev_arr[nb_ports];
	struct entries_status ctrl_status, ct_status;
	uint32_t ct_flags, nb_arm_queues = 1, nb_ctrl_queues = 1, nb_user_actions = 0, nb_ipv4_sessions = 1024,
			   nb_ipv6_sessions = 0; /* On BF2 should always be 0 */
	uint16_t ct_queue = nb_queues;
	doca_error_t result, doca_error;

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

	ct_flags = DOCA_FLOW_CT_FLAG_NO_AGING | DOCA_FLOW_CT_FLAG_NO_COUNTER;
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

	result = create_vxlan_encap_pipe(ports[0], 0, &ctrl_status, &encap_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_count_pipe(ports[0], rss_pipe, &ctrl_status, &count_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_ct_pipe(ports[0], encap_pipe, count_pipe, &ct_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	result = create_udp_pipe(ports[0], ct_pipe, &ctrl_status, &udp_pipe);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	if (ctrl_status.nb_processed != nb_entries || ctrl_status.failure) {
		DOCA_LOG_ERR("Failed to process control path entries");
		result = DOCA_ERROR_BAD_STATE;
		goto cleanup;
	}

	DOCA_LOG_INFO("Wait few seconds for packets to arrive");
	sleep(5);

	result = process_packets(ports[0], ct_queue, &ct_status);
	if (result != DOCA_SUCCESS)
		goto cleanup;

	DOCA_LOG_INFO("Same UDP packet should be resent");
	sleep(5);

cleanup:
	doca_error = result;
	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_ct_destroy();
	doca_flow_destroy();

	return doca_error;
}
