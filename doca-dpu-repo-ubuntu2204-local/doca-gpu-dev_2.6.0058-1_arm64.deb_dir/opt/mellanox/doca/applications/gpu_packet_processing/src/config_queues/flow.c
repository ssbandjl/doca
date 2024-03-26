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

#include <arpa/inet.h>
#include <rte_ethdev.h>
#include <doca_flow.h>

#include "common.h"
#include "dpdk_tcp/tcp_session_table.h"

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING_FLOW);

static uint64_t default_flow_timeout_usec;

struct doca_flow_port *
init_doca_flow(uint16_t port_id, uint8_t rxq_num)
{
	doca_error_t result;
	char port_id_str[MAX_PORT_STR_LEN];
	struct doca_flow_port_cfg port_cfg = {0};
	struct doca_flow_port *df_port;
	struct doca_flow_cfg rxq_flow_cfg = {0};
	int ret = 0;
	struct rte_eth_dev_info dev_info = {0};
	struct rte_eth_conf eth_conf = {
		.rxmode = {
			.mtu = 2048, /* Not really used, just to initialize DPDK */
		},
		.txmode = {
			.offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM,
		},
	};
	struct rte_mempool *mp = NULL;
	struct rte_eth_txconf tx_conf;
	struct rte_flow_error error;

	/*
	 * DPDK should be initialized and started before DOCA Flow.
	 * DPDK doesn't start the device without, at least, one DPDK Rx queue.
	 * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
	 *
	 * Following lines of code can be considered the minimum WAR for this issue.
	 */

	ret = rte_eth_dev_info_get(port_id, &dev_info);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_info_get with: %s", rte_strerror(-ret));
		return NULL;
	}

	ret = rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_configure with: %s", rte_strerror(-ret));
		return NULL;
	}

	mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, MAX_PKT_SIZE, rte_eth_dev_socket_id(port_id));
	if (mp == NULL) {
		DOCA_LOG_ERR("Failed rte_pktmbuf_pool_create with: %s", rte_strerror(-ret));
		return NULL;
	}

	tx_conf = dev_info.default_txconf;
	tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM;

	for (int idx = 0; idx < rxq_num; idx++) {
		ret = rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), NULL, mp);
		if (ret) {
			DOCA_LOG_ERR("Failed rte_eth_rx_queue_setup with: %s", rte_strerror(-ret));
			return NULL;
		}

		ret = rte_eth_tx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), &tx_conf);
		if (ret) {
			DOCA_LOG_ERR("Failed rte_eth_tx_queue_setup with: %s", rte_strerror(-ret));
			return NULL;
		}
	}

	ret = rte_flow_isolate(port_id, 1, &error);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_flow_isolate with: %s", error.message);
		return NULL;
	}

	ret = rte_eth_dev_start(port_id);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_start with: %s", rte_strerror(-ret));
		return NULL;
	}

	/* Initialize doca flow framework */
	rxq_flow_cfg.pipe_queues = rxq_num;
	/*
	 * HWS: Hardware steering
	 * Isolated: don't create RSS rule for DPDK created RX queues
	 */
	rxq_flow_cfg.mode_args = "vnf,hws,isolated";
	rxq_flow_cfg.resource.nb_counters = FLOW_NB_COUNTERS;

	result = doca_flow_init(&rxq_flow_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init doca flow with: %s", doca_error_get_descr(result));
		return NULL;
	}

	/* Start doca flow port */
	port_cfg.port_id = port_id;
	port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
	snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
	port_cfg.devargs = port_id_str;
	result = doca_flow_port_start(&port_cfg, &df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start doca flow port with: %s", doca_error_get_descr(result));
		return NULL;
	}

	default_flow_timeout_usec = 0;

	return df_port;
}

doca_error_t
create_udp_pipe(struct rxq_udp_queues *udp_queues, struct doca_flow_port *port)
{
	doca_error_t result;
	struct doca_flow_match match = {0};
	struct doca_flow_match match_mask = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_fwd miss_fwd = {0};
	struct doca_flow_pipe_cfg pipe_cfg = {0};
	struct doca_flow_pipe_entry *entry;
	uint16_t flow_queue_id;
	uint16_t rss_queues[MAX_QUEUES];
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	if (udp_queues == NULL || port == NULL || udp_queues->numq > MAX_QUEUES)
		return DOCA_ERROR_INVALID_VALUE;

	pipe_cfg.attr.name = "GPU_RXQ_UDP_PIPE";
	pipe_cfg.attr.enable_strict_matching = true;
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.attr.nb_actions = 0;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.match = &match;
	pipe_cfg.match_mask = &match_mask;
	pipe_cfg.monitor = &monitor;
	pipe_cfg.port = port;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

	for (int idx = 0; idx < udp_queues->numq; idx++) {
		doca_eth_rxq_get_flow_queue_id(udp_queues->eth_rxq_cpu[idx], &flow_queue_id);
		rss_queues[idx] = flow_queue_id;
	}

	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
	fwd.num_of_queues = udp_queues->numq;

	miss_fwd.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &(udp_queues->rxq_pipe));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	/* Add HW offload */
	result = doca_flow_pipe_add_entry(0, udp_queues->rxq_pipe, &match, NULL, NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

	return DOCA_SUCCESS;
}

doca_error_t
create_tcp_cpu_pipe(struct rxq_tcp_queues *tcp_queues, struct doca_flow_port *port)
{
	doca_error_t result;
	uint16_t rss_queues[MAX_QUEUES];
	struct doca_flow_match match_mask = {0};

	/*
	 * Setup the TCP pipe 'rxq_pipe_cpu' which forwards unrecognized flows and
	 * TCP SYN/ACK/FIN flags to the CPU - in other words, any TCP packets not
	 * recognized by the GPU TCP pipe.
	 */

	if (tcp_queues == NULL || port == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	/* Init TCP session table */
	tcp_session_table = rte_hash_create(&tcp_session_ht_params);

	struct doca_flow_match match = {
		.outer = {
			.l3_type = DOCA_FLOW_L3_TYPE_IP4,
			.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
		}
	};

	for (int idx = 0; idx < tcp_queues->numq_cpu_rss; idx++)
		rss_queues[idx] = idx;

	struct doca_flow_fwd fwd = {
		.type = DOCA_FLOW_FWD_RSS,
		.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP,
		.rss_queues = rss_queues,
		.num_of_queues = tcp_queues->numq_cpu_rss,
	};

	struct doca_flow_fwd miss_fwd = {
		.type = DOCA_FLOW_FWD_DROP,
	};

	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	struct doca_flow_pipe_cfg pipe_cfg = {
		.attr = {
			.name = "CPU_RXQ_TCP_PIPE",
			.enable_strict_matching = true,
			.type = DOCA_FLOW_PIPE_BASIC,
			.nb_actions = 0,
			.is_root = false,
		},
		.match = &match,
		.match_mask = &match_mask,
		.monitor = &monitor,
		.port = port,
	};

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &tcp_queues->rxq_pipe_cpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_pipe_add_entry(0, tcp_queues->rxq_pipe_cpu, NULL, NULL, NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &tcp_queues->cpu_rss_entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

	return DOCA_SUCCESS;
}

doca_error_t
create_tcp_gpu_pipe(struct rxq_tcp_queues *tcp_queues, struct doca_flow_port *port, bool connection_based_flows)
{
	uint16_t flow_queue_id;
	uint16_t rss_queues[MAX_QUEUES];
	doca_error_t result;
	struct doca_flow_pipe_entry *dummy_entry = NULL;
	struct doca_flow_match match_mask = {0};

	/* The GPU TCP pipe should only forward known flows to the GPU. Others will be dropped */

	if (tcp_queues == NULL || port == NULL || tcp_queues->numq > MAX_QUEUES)
		return DOCA_ERROR_INVALID_VALUE;

	struct doca_flow_match match = {
		.outer = {
			.l3_type = DOCA_FLOW_L3_TYPE_IP4,
			.ip4.next_proto = IPPROTO_TCP,
			.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
		},
	};

	if (connection_based_flows) {
		match.outer.ip4.src_ip = 0xffffffff;
		match.outer.ip4.dst_ip = 0xffffffff;
		match.outer.tcp.l4_port.src_port = 0xffff;
		match.outer.tcp.l4_port.dst_port = 0xffff;
	};

	for (int idx = 0; idx < tcp_queues->numq; idx++) {
		doca_eth_rxq_get_flow_queue_id(tcp_queues->eth_rxq_cpu[idx], &flow_queue_id);
		rss_queues[idx] = flow_queue_id;
	}

	struct doca_flow_fwd fwd = {
		.type = DOCA_FLOW_FWD_RSS,
		.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP,
		.rss_queues = rss_queues,
		.num_of_queues = tcp_queues->numq,
	};

	struct doca_flow_fwd miss_fwd = {
		.type = DOCA_FLOW_FWD_DROP,
	};

	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	struct doca_flow_pipe_cfg pipe_cfg = {
		.attr = {
			.name = "GPU_RXQ_TCP_PIPE",
			.enable_strict_matching = true,
			.type = DOCA_FLOW_PIPE_BASIC,
			.nb_actions = 0,
			.is_root = false,
		},
		.match = &match,
		.match_mask = &match_mask,
		.monitor = &monitor,
		.port = port,
	};

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &tcp_queues->rxq_pipe_gpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	if (!connection_based_flows) {
		// For the non-connection-based configuration, create a dummy flow entry which will enable
		// any TCP packets to be forwarded.
		result = doca_flow_pipe_add_entry(0, tcp_queues->rxq_pipe_gpu, NULL, NULL, NULL, NULL, 0, NULL, &dummy_entry);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("RxQ pipe-entry creation failed with: %s", doca_error_get_descr(result));
			DOCA_GPUNETIO_VOLATILE(force_quit) = true;
			return result;
		}

		result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
			return result;
		}
	}

	DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

	return DOCA_SUCCESS;
}

doca_error_t
create_icmp_gpu_pipe(struct rxq_icmp_queues *icmp_queues, struct doca_flow_port *port)
{
	doca_error_t result;
	struct doca_flow_match match = {0};
	struct doca_flow_match match_mask = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_fwd miss_fwd = {0};
	struct doca_flow_pipe_cfg pipe_cfg = {0};
	struct doca_flow_pipe_entry *entry;
	uint16_t flow_queue_id;
	uint16_t rss_queues[MAX_QUEUES];
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	if (icmp_queues == NULL || port == NULL || icmp_queues->numq > MAX_QUEUES)
		return DOCA_ERROR_INVALID_VALUE;

	pipe_cfg.attr.name = "GPU_RXQ_ICMP_PIPE";
	pipe_cfg.attr.enable_strict_matching = true;
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.attr.nb_actions = 0;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.match = &match;
	pipe_cfg.match_mask = &match_mask;
	pipe_cfg.monitor = &monitor;
	pipe_cfg.port = port;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_ICMP;

	for (int idx = 0; idx < icmp_queues->numq; idx++) {
		doca_eth_rxq_get_flow_queue_id(icmp_queues->eth_rxq_cpu[idx], &flow_queue_id);
		rss_queues[idx] = flow_queue_id;
	}

	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4;
	fwd.num_of_queues = icmp_queues->numq;

	miss_fwd.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &(icmp_queues->rxq_pipe));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	/* Add HW offload */
	result = doca_flow_pipe_add_entry(0, icmp_queues->rxq_pipe, &match, NULL, NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

	return DOCA_SUCCESS;
}

doca_error_t
create_root_pipe(struct rxq_udp_queues *udp_queues, struct rxq_tcp_queues *tcp_queues, struct rxq_icmp_queues *icmp_queues, struct doca_flow_port *port)
{
	uint32_t priority_high = 1;
	uint32_t priority_low = 3;
	doca_error_t result;
	struct doca_flow_match match_mask = {0};
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	if (udp_queues == NULL || tcp_queues == NULL || port == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	struct doca_flow_pipe_cfg pipe_cfg = {
		.attr = {
			.name = "ROOT_PIPE",
			.enable_strict_matching = true,
			.is_root = true,
			.type = DOCA_FLOW_PIPE_CONTROL,
		},
		.port = port,
		.monitor = &monitor,
		.match_mask = &match_mask,
	};

	result = doca_flow_pipe_create(&pipe_cfg, NULL, NULL, &udp_queues->root_pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	struct doca_flow_match udp_match = {
		.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4,
		.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP,
	};

	struct doca_flow_fwd udp_fwd = {
		.type = DOCA_FLOW_FWD_PIPE,
		.next_pipe = udp_queues->rxq_pipe,
	};

	result = doca_flow_pipe_control_add_entry(0, 0, udp_queues->root_pipe, &udp_match, NULL, NULL, NULL, NULL, NULL, NULL,
							&udp_fwd, NULL, &udp_queues->root_udp_entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe UDP entry creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	if (icmp_queues->rxq_pipe) {
		struct doca_flow_match icmp_match_gpu = {
			.outer = {
				.l3_type = DOCA_FLOW_L3_TYPE_IP4,
				.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_ICMP,
			},
		};

		struct doca_flow_fwd icmp_fwd_gpu = {
			.type = DOCA_FLOW_FWD_PIPE,
			.next_pipe = icmp_queues->rxq_pipe,
		};

		result = doca_flow_pipe_control_add_entry(0, priority_low, udp_queues->root_pipe, &icmp_match_gpu, NULL, NULL, NULL, NULL, NULL, NULL,
								&icmp_fwd_gpu, NULL, &udp_queues->root_icmp_entry_gpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Root pipe UDP entry creation failed with: %s", doca_error_get_descr(result));
			return result;
		}
	}

	if (tcp_queues->rxq_pipe_gpu) {
		struct doca_flow_match tcp_match_gpu = {
			.outer = {
				.l3_type = DOCA_FLOW_L3_TYPE_IP4,
				.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
			},
		};

		struct doca_flow_fwd tcp_fwd_gpu = {
			.type = DOCA_FLOW_FWD_PIPE,
			.next_pipe = tcp_queues->rxq_pipe_gpu,
		};

		result = doca_flow_pipe_control_add_entry(0, priority_low, udp_queues->root_pipe, &tcp_match_gpu, NULL, NULL, NULL, NULL, NULL, NULL,
								&tcp_fwd_gpu, NULL, &udp_queues->root_tcp_entry_gpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Root pipe UDP entry creation failed with: %s", doca_error_get_descr(result));
			return result;
		}
	}

	if (tcp_queues->rxq_pipe_cpu) {
		struct doca_flow_match tcp_match_cpu = {
			.outer = {
				.l3_type = DOCA_FLOW_L3_TYPE_IP4,
				.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
			},
		};
		struct doca_flow_fwd tcp_fwd_cpu = {
			.type = DOCA_FLOW_FWD_PIPE,
			.next_pipe = tcp_queues->rxq_pipe_cpu,
		};

		uint8_t individual_tcp_flags[] = {
			DOCA_FLOW_MATCH_TCP_FLAG_SYN,
			DOCA_FLOW_MATCH_TCP_FLAG_RST,
			DOCA_FLOW_MATCH_TCP_FLAG_FIN,
		};

		for (int i = 0; i < 3; i++) {
			tcp_match_cpu.outer.tcp.flags = individual_tcp_flags[i];
			result = doca_flow_pipe_control_add_entry(0, priority_high, udp_queues->root_pipe, &tcp_match_cpu, &tcp_match_cpu, NULL, NULL, NULL, NULL, NULL,
									&tcp_fwd_cpu, NULL, &udp_queues->root_tcp_entry_cpu[i]);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Root pipe TCP entry creation failed with: %s", doca_error_get_descr(result));
				return result;
			}
		}
	}

	result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

	return DOCA_SUCCESS;
}

doca_error_t
enable_tcp_gpu_offload(struct doca_flow_port *port, uint16_t queue_id, struct doca_flow_pipe *gpu_rss_pipe, struct tcp_session_entry *session_entry)
{
	doca_error_t result;
	char src_addr[INET_ADDRSTRLEN];
	char dst_addr[INET_ADDRSTRLEN];

	struct doca_flow_match match = {
		.outer = {
			.l3_type = DOCA_FLOW_L3_TYPE_IP4,
			.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP,
			.tcp.flags = 0,
			.ip4.src_ip = session_entry->key.src_addr,
			.ip4.dst_ip = session_entry->key.dst_addr,
			.tcp.l4_port.src_port = session_entry->key.src_port,
			.tcp.l4_port.dst_port = session_entry->key.dst_port,
		},
	};

	result = doca_flow_pipe_add_entry(queue_id, gpu_rss_pipe, &match, NULL, NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &session_entry->flow);
	if (result != DOCA_SUCCESS) {
		inet_ntop(AF_INET, &session_entry->key.src_addr, src_addr, INET_ADDRSTRLEN);
		inet_ntop(AF_INET, &session_entry->key.dst_addr, dst_addr, INET_ADDRSTRLEN);
		DOCA_LOG_ERR("Failed to create TCP offload session; error = %s, session = %s:%d>%s:%d",
			doca_error_get_descr(result),
			src_addr, htons(session_entry->key.src_port),
			dst_addr, htons(session_entry->key.dst_port));
			return result;
	}

	inet_ntop(AF_INET, &session_entry->key.src_addr, src_addr, INET_ADDRSTRLEN);
	inet_ntop(AF_INET, &session_entry->key.dst_addr, dst_addr, INET_ADDRSTRLEN);
	DOCA_LOG_INFO("Created TCP offload session %s:%d>%s:%d",
		src_addr, htons(session_entry->key.src_port),
		dst_addr, htons(session_entry->key.dst_port));

	result = doca_flow_entries_process(port, queue_id, default_flow_timeout_usec, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t
disable_tcp_gpu_offload(struct doca_flow_port *port, uint16_t queue_id, struct doca_flow_pipe *gpu_rss_pipe, struct tcp_session_entry *session_entry)
{
	doca_error_t result;
	char src_addr[INET_ADDRSTRLEN];
	char dst_addr[INET_ADDRSTRLEN];

	/*
	 * Because those flows tend to be extremely short-lived,
	 * process the queue once more to ensure the newly created
	 * flows have been able to reach a deletable state.
	 */
	result = doca_flow_entries_process(port, queue_id, default_flow_timeout_usec, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_pipe_rm_entry(queue_id, DOCA_FLOW_NO_WAIT, session_entry->flow);

	if (result != DOCA_SUCCESS) {
		inet_ntop(AF_INET, &session_entry->key.src_addr, src_addr, INET_ADDRSTRLEN);
		inet_ntop(AF_INET, &session_entry->key.dst_addr, dst_addr, INET_ADDRSTRLEN);
		DOCA_LOG_ERR("Failed to destroy TCP offload session; error = %s, session = %s:%d>%s:%d",
			doca_error_get_descr(result),
			src_addr, htons(session_entry->key.src_port),
			dst_addr, htons(session_entry->key.dst_port));
	} else {
		inet_ntop(AF_INET, &session_entry->key.src_addr, src_addr, INET_ADDRSTRLEN);
		inet_ntop(AF_INET, &session_entry->key.dst_addr, dst_addr, INET_ADDRSTRLEN);
		DOCA_LOG_INFO("Destroyed TCP offload session %s:%d>%s:%d",
			src_addr, htons(session_entry->key.src_port),
			dst_addr, htons(session_entry->key.dst_port));
	}

	return result;
}


doca_error_t
destroy_flow_queue(uint16_t port_id, struct doca_flow_port *port_df,
			struct rxq_icmp_queues *icmp_queues, struct rxq_udp_queues *udp_queues,
			struct rxq_tcp_queues *tcp_queues,
			bool http_server, struct txq_http_queues *http_queues)
{
	int ret = 0;

	doca_flow_port_stop(port_df);
	doca_flow_destroy();

	destroy_icmp_queues(icmp_queues);
	destroy_udp_queues(udp_queues);
	destroy_tcp_queues(tcp_queues, http_server, http_queues);

	ret = rte_eth_dev_stop(port_id);
	if (ret != 0) {
		DOCA_LOG_ERR("Couldn't stop DPDK port %d err %d", port_id, ret);
		return DOCA_ERROR_DRIVER;
	}

	return DOCA_SUCCESS;
}
