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
#include <stdint.h>
#include <signal.h>

#include <rte_ethdev.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

DOCA_LOG_REGISTER(FLOW_LOOPBACK);

#define PACKET_BURST 128	/* The number of packets in the rx queue */

static bool force_quit = false;

/*
 * Signal handler
 *
 * @signum [in]: The signal received to handle
 */
static void
signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		DOCA_LOG_INFO("Signal %d received, preparing to exit", signum);
		force_quit = true;
	}
}

/*
 * Dequeue packets from DPDK queues
 *
 * @ingress_port [in]: port id for dequeue packets
 */
static void
process_packets(int ingress_port)
{
	struct rte_mbuf *packets[PACKET_BURST];
	int queue_index = 0;
	int nb_packets;
	int i;

	while (!force_quit) {
		nb_packets = rte_eth_rx_burst(ingress_port, queue_index, packets, PACKET_BURST);

		/* Print received packets' meta data */
		for (i = 0; i < nb_packets; i++) {
			if (rte_flow_dynf_metadata_avail())
				DOCA_LOG_INFO("Packet received with meta data %d", *RTE_FLOW_DYNF_METADATA(packets[i]));
		}
		rte_eth_tx_burst(ingress_port, queue_index, packets, PACKET_BURST);
	}
}

/*
 * Create DOCA Flow pipe where the match is on changeable TCP over IPv4 source and destination
 * addresses (IP and port), forward is RSS, and the forward miss is to another pipe.
 *
 * @port [in]: port of the pipe
 * @miss_pipe [in]: the next pipe to use in case of a miss.
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_rss_tcp_ip_pipe(struct doca_flow_port *port, struct doca_flow_pipe *miss_pipe, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;
	uint16_t rss_queues[1];

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "RSS_TCP_IPv4_PIPE";
	pipe_cfg.match = &match;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.attr.nb_actions = NB_ACTIONS_ARR;
	pipe_cfg.port = port;

	/* changeable TCP over IPv4 source and destination addresses */
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_TCP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
	match.outer.tcp.l4_port.src_port = 0xffff;
	match.outer.tcp.l4_port.dst_port = 0xffff;

	/* RSS queue - send matched traffic to queue 0 */
	rss_queues[0] = 0;
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_inner_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP;
	fwd.num_of_queues = 1;

	/* In case of a miss, forward the packet to the next pipe */
	fwd_miss.type = DOCA_FLOW_FWD_PIPE;
	fwd_miss.next_pipe = miss_pipe;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd_miss, pipe);
}

/*
 * Add DOCA Flow pipe entry to the "RSS TCP over IPv4" pipe
 *
 * @pipe [in]: a pointer to the pipe to add the entry to
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
add_rss_tcp_ip_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	/* TCPoTPv4 source and destination addresses */
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
 * Create DOCA Flow pipe where it matches UDP over IPv4 traffic
 * with changeable source and destination IPv4 addresses.
 * The actions of the pipe is only setting a chageable metadata, forward is RSS and forward miss is drop.
 *
 * @port [in]: port of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_rss_udp_ip_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;
	uint16_t rss_queues[1];

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	/* set mask value */
	actions.meta.pkt_meta = UINT32_MAX;

	pipe_cfg.attr.name = "RSS_UDP_IP_PIPE";
	pipe_cfg.match = &match;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.nb_actions = NB_ACTIONS_ARR;
	pipe_cfg.port = port;

	/* changeable IPv4 source and destination addresses */
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;

	/* RSS queue - send matched traffic to queue 0 */
	rss_queues[0] = 0;
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_inner_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
	fwd.num_of_queues = 1;

	/* In case of a miss match, drop the packet */
	fwd_miss.type = DOCA_FLOW_FWD_DROP;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd_miss, pipe);
}

/*
 * Add DOCA Flow pipe entry to the "RSS UDP over IPv4" pipe
 *
 * @pipe [in]: a pointer to the pipe to add the entry to
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
add_rss_rss_udp_ip_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));

	/* IPv4 source and destination addresses */
	match.outer.ip4.dst_ip = BE_IPV4_ADDR(81, 81, 81, 81);
	match.outer.ip4.src_ip = BE_IPV4_ADDR(11, 21, 31, 41);

	/* set meta value */
	actions.meta.pkt_meta = 10;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Flow "loopback" pipe.
 * The "loopback" pipe is created on egress domain, and it
 * matches changeable TCP over IPv4 source and destination addresses (IP and port).
 * It also does changeable VXLAN enxapsulation.
 *
 * @port [in]: port of the pipe
 * @port_id [in]: the port identifier to re-inject the packet to
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_loopback_pipe(struct doca_flow_port *port, uint16_t port_id, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_action_descs descs, *descs_arr[NB_ACTIONS_ARR];
	struct doca_flow_action_desc desc_array[NB_ACTIONS_ARR] = {0};

	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "LOOPBACK_PIPE";
	pipe_cfg.match = &match;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.action_descs = descs_arr;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.attr.domain = DOCA_FLOW_PIPE_DOMAIN_EGRESS;
	pipe_cfg.attr.nb_actions = NB_ACTIONS_ARR;
	pipe_cfg.port = port;

	/* changeable TCP over IPv4 source and destination addresses */
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_TCP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
	match.outer.tcp.l4_port.src_port = 0xffff;
	match.outer.tcp.l4_port.dst_port = 0xffff;

	/* build basic outer VXLAN encap data */
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

	/* In cade of a miss, drop the packet */
	fwd_miss.type = DOCA_FLOW_FWD_DROP;

	return doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd_miss, pipe);
}

/*
 * Add DOCA Flow pipe entry to the "loopback" pipe
 * Note that upon encapsulation the
 * destination MAC address is the same mac address of the port
 * the packet arrived to (needed for the "loopback" or "packet re-injection").
 *
 * @pipe [in]: pipe of the entry
 * @status [in]: user context for adding entry
 * @dst_mac [in]: MAC address used for the encapsulation
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
add_loopback_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status, uint8_t dst_mac[6])
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	doca_be32_t encap_dst_ip_addr = BE_IPV4_ADDR(81, 81, 81, 81);
	doca_be32_t encap_src_ip_addr = BE_IPV4_ADDR(11, 21, 31, 41);
	uint8_t encap_ttl = 17;
	doca_be32_t encap_vxlan_tun_id = BUILD_VNI(0xadadad);
	uint8_t src_mac[] = {0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};

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

	actions.has_encap = true;
	SET_MAC_ADDR(actions.encap.outer.eth.src_mac, src_mac[0], src_mac[1], src_mac[2], src_mac[3], src_mac[4], src_mac[5]);
	SET_MAC_ADDR(actions.encap.outer.eth.dst_mac, dst_mac[0], dst_mac[1], dst_mac[2], dst_mac[3], dst_mac[4], dst_mac[5]);
	actions.encap.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	actions.encap.outer.ip4.src_ip = encap_src_ip_addr;
	actions.encap.outer.ip4.dst_ip = encap_dst_ip_addr;
	actions.encap.outer.ip4.ttl = encap_ttl;
	actions.encap.tun.type = DOCA_FLOW_TUN_VXLAN;
	actions.encap.tun.vxlan_tun_id = encap_vxlan_tun_id;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Run flow_loopback sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @mac_addresses [in]: MAC addresses of each port used for encapsulation
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
flow_loopback(int nb_queues, uint8_t mac_addresses[2][6])
{
	const int nb_ports = 2;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *pipe, *miss_pipe;
	struct entries_status status;
	int num_of_entries = 3;
	doca_error_t result;
	int port_id;

	result = init_doca_flow(nb_queues, "vnf,hws", resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA Flow: %s", doca_error_get_descr(result));
		return -1;
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
		result = create_rss_udp_ip_pipe(ports[port_id], &pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_rss_rss_udp_ip_pipe_entry(pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}
		miss_pipe = pipe;

		result = create_rss_tcp_ip_pipe(ports[port_id], miss_pipe, &pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_rss_tcp_ip_pipe_entry(pipe, &status);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_loopback_pipe(ports[port_id], port_id, &pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_loopback_pipe_entry(pipe, &status, mac_addresses[port_id]);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
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

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	DOCA_LOG_INFO("Wait for packets to arrive");
	for (port_id = 0; port_id < nb_ports; port_id++)
		process_packets(port_id);

	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return DOCA_SUCCESS;
}
