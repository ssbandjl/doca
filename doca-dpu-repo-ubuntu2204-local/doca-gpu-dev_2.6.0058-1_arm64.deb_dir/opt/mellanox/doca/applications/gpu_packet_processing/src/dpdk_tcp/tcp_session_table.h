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

#ifndef DOCA_GPU_PACKET_PROCESSING_TCP_SESSION_H
#define DOCA_GPU_PACKET_PROCESSING_TCP_SESSION_H

#include <stdint.h>
#include <rte_common.h>
#include <rte_byteorder.h>
#include <rte_hash.h>

#define TCP_SESSION_MAX_ENTRIES 4096

/* TCP session key */
struct tcp_session_key {
	rte_be32_t src_addr;	/* TCP session key src addr */
	rte_be32_t dst_addr;	/* TCP session key dst addr */
	rte_be16_t src_port;	/* TCP session key src port */
	rte_be16_t dst_port;	/* TCP session key dst port */
};

/* TCP session entry */
struct tcp_session_entry {
	struct tcp_session_key key;		/* TCP session key */
	struct doca_flow_pipe_entry *flow;	/* TCP session key DOCA flow entry */
};

/* TCP session params */
extern struct rte_hash_parameters tcp_session_ht_params;
/* TCP session table */
extern struct rte_hash *tcp_session_table;

/*
 * TCP session table CRC
 *
 * @data [in]: Network card PCIe address
 * @data_len [in]: DOCA device
 * @init_val [in]: DPDK port id associated with the DOCA device
 * @return: 0 on success and 1 otherwise
 */
uint32_t tcp_session_table_crc(const void *data, uint32_t data_len, uint32_t init_val);

/*
 * TCP session table CRC
 *
 * @key [in]: TCP session table key
 * @return: ptr on success and NULL otherwise
 */
struct tcp_session_entry *tcp_session_table_find(struct tcp_session_key *key);

/*
 * TCP session table CRC
 *
 * @entry [in]: TCP session table key
 */
void tcp_session_table_delete(struct tcp_session_entry *entry);

/*
 * Establish new TCP session
 *
 * @key [in]: TCP session table key
 * @return: ptr on success and NULL otherwise
 */
struct tcp_session_entry *tcp_session_table_new(struct tcp_session_key *key);

#endif
