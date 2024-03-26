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

#ifndef DOCA_GPU_PACKET_PROCESSING_TCP_RSS_H
#define DOCA_GPU_PACKET_PROCESSING_TCP_RSS_H

#include <stdbool.h>
#include <arpa/inet.h>

#include <rte_ethdev.h>
#include <rte_malloc.h>

#include <doca_flow.h>
#include <doca_log.h>

#include <common.h>

#define TCP_PACKET_MAX_BURST_SIZE 4096

/*
 * Launch CPU thread to manage TCP 3way handshake
 *
 * @args [in]: thread input args
 * @return: 0 on success and 1 otherwise
 */
int tcp_cpu_rss_func(void *args);

/*
 * Extract the address of the IPv4 TCP header contained in the
 * raw ethernet frame packet buffer if present; otherwise null.
 *
 * @packet [in]: Packet to extract TCP hdr
 * @return: ptr on success and NULL otherwise
 */
const struct rte_tcp_hdr *extract_tcp_hdr(const struct rte_mbuf *packet);

/*
 * Create TCP session
 *
 * @queue_id [in]: DPDK queue id for TCP control packets
 * @pkt [in]: pkt triggering the TCP session creation
 * @port [in]: DOCA Flow port
 * @gpu_rss_pipe [in]: DOCA Flow GPU RSS pipe
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_tcp_session(const uint16_t queue_id, const struct rte_mbuf *pkt,  struct doca_flow_port *port, struct doca_flow_pipe *gpu_rss_pipe);

/*
 * Destroy TCP session
 *
 * @queue_id [in]: DPDK queue id for TCP control packets
 * @pkt [in]: pkt triggering the TCP session destruction
 * @port [in]: DOCA Flow port
 * @gpu_rss_pipe [in]: DOCA Flow GPU RSS pipe
 */
void destroy_tcp_session(const uint16_t queue_id, const struct rte_mbuf *pkt, struct doca_flow_port *port, struct doca_flow_pipe *gpu_rss_pipe);

/*
 * Log TCP flags
 *
 * @packet [in]: Packet to report TCP flags
 * @flags [in]: TCP Flags
 */
void log_tcp_flag(const struct rte_mbuf *packet, const char *flags);

/*
 * Create TCP ACK packet
 *
 * @src_packet [in]: Src packet to use to create ACK packet
 * @tcp_ack_pkt_pool [in]: DPDK mempool to create ACK packets
 * @return: ptr on success and NULL otherwise
 */
struct rte_mbuf *create_ack_packet(const struct rte_mbuf *src_packet, struct rte_mempool *tcp_ack_pkt_pool);

/*
 * Extract TCP session key
 *
 * @packet [in]: Extract session key from this packet
 * @return: object
 */
struct tcp_session_key extract_session_key(const struct rte_mbuf *packet);

#endif

