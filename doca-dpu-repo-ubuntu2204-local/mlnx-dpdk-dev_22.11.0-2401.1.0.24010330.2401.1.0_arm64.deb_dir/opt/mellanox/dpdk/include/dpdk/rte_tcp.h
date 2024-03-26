/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 1982, 1986, 1990, 1993
 *      The Regents of the University of California.
 * Copyright(c) 2010-2014 Intel Corporation.
 * All rights reserved.
 */

#ifndef _RTE_TCP_H_
#define _RTE_TCP_H_

/**
 * @file
 *
 * TCP-related defines
 */

#include <stdint.h>

#include <rte_byteorder.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TCP Header
 */
__extension__
struct rte_tcp_hdr {
	rte_be16_t src_port; /**< TCP source port. */
	rte_be16_t dst_port; /**< TCP destination port. */
	rte_be32_t sent_seq; /**< TX data sequence number. */
	rte_be32_t recv_ack; /**< RX data acknowledgment sequence number. */
	union {
		uint8_t data_off;
		struct {
#if RTE_BYTE_ORDER == RTE_LITTLE_ENDIAN
			uint8_t rsrv:4;
			uint8_t dt_off:4;   /**< Data offset. */
#elif RTE_BYTE_ORDER == RTE_BIG_ENDIAN
			uint8_t dt_off:4;   /**< Data offset. */
			uint8_t rsrv:4;
#else
#error "setup endian definition"
#endif
		};

	};
	union {
		uint8_t tcp_flags;  /**< TCP flags */
		struct {
#if RTE_BYTE_ORDER == RTE_LITTLE_ENDIAN
			uint8_t fin:1;
			uint8_t syn:1;
			uint8_t rst:1;
			uint8_t psh:1;
			uint8_t ack:1;
			uint8_t urg:1;
			uint8_t ecne:1;
			uint8_t cwr:1;
#elif RTE_BYTE_ORDER == RTE_BIG_ENDIAN
			uint8_t cwr:1;
			uint8_t ecne:1;
			uint8_t urg:1;
			uint8_t ack:1;
			uint8_t psh:1;
			uint8_t rst:1;
			uint8_t syn:1;
			uint8_t fin:1;
#else
#error "setup endian definition"
#endif
		};
	};
	rte_be16_t rx_win;   /**< RX flow control window. */
	rte_be16_t cksum;    /**< TCP checksum. */
	rte_be16_t tcp_urp;  /**< TCP urgent pointer, if any. */
} __rte_packed;

/**
 * TCP Flags
 */
#define RTE_TCP_CWR_FLAG 0x80 /**< Congestion Window Reduced */
#define RTE_TCP_ECE_FLAG 0x40 /**< ECN-Echo */
#define RTE_TCP_URG_FLAG 0x20 /**< Urgent Pointer field significant */
#define RTE_TCP_ACK_FLAG 0x10 /**< Acknowledgment field significant */
#define RTE_TCP_PSH_FLAG 0x08 /**< Push Function */
#define RTE_TCP_RST_FLAG 0x04 /**< Reset the connection */
#define RTE_TCP_SYN_FLAG 0x02 /**< Synchronize sequence numbers */
#define RTE_TCP_FIN_FLAG 0x01 /**< No more data from sender */

enum rte_tcp_state {
	RTE_TCP_ESTABLISHED = 1,
	RTE_TCP_SYN_SENT,
	RTE_TCP_SYN_RECV,
	RTE_TCP_FIN_WAIT1,
	RTE_TCP_FIN_WAIT2,
	RTE_TCP_TIME_WAIT,
	RTE_TCP_CLOSE,
	RTE_TCP_CLOSE_WAIT,
	RTE_TCP_LAST_ACK,
	RTE_TCP_LISTEN,
	RTE_TCP_CLOSING
};

enum rte_tcp_opt {
	RTE_TCP_OPT_END = 0,
	RTE_TCP_OPT_NOP = 1,
	RTE_TCP_OPT_MSS = 2,
	RTE_TCP_OPT_WND_SCALE = 3,
	RTE_TCP_OPT_SACK_PERMITTED = 4,
	RTE_TCP_OPT_SACK = 5,
	RTE_TCP_OPT_TIMESTAMP = 8,
};

static inline const char *
rte_tcp_state_name(int state)
{
	const char *name;

	switch (state) {
	case RTE_TCP_ESTABLISHED:
		name = "ESTABLISHED";
		break;
	case RTE_TCP_SYN_SENT:
		name = "SYN_SENT";
		break;
	case RTE_TCP_SYN_RECV:
		name = "SYN_RECV";
		break;
	case RTE_TCP_FIN_WAIT1:
		name = "FIN_WAIT1";
		break;
	case RTE_TCP_FIN_WAIT2:
		name = "FIN_WAIT2";
		break;
	case RTE_TCP_TIME_WAIT:
		name = "TIME_WAIT";
		break;
	case RTE_TCP_CLOSE:
		name = "CLOSE";
		break;
	case RTE_TCP_CLOSE_WAIT:
		name = "CLOSE_WAIT";
		break;
	case RTE_TCP_LAST_ACK:
		name = "LAST_ACK";
		break;
	case RTE_TCP_LISTEN:
		name = "LISTEN";
		break;
	case RTE_TCP_CLOSING:
		name = "CLOSING";
		break;
	default:
		name = "INVALID";
	}

	return name;
}

/**
 * Get the length of an TCP header.
 *
 * @param tcp_hdr
 *   Pointer to the TCP header.
 * @return
 *   The length of the TCP header (with options if present) in bytes.
 */
static inline uint8_t
rte_tcp_hdr_len(const struct rte_tcp_hdr *tcp_hdr)
{
	return (uint8_t)(tcp_hdr->dt_off * sizeof(int));
}

#define RTE_TCP_MIN_HDR_LEN	20

#ifdef __cplusplus
}
#endif

#endif /* RTE_TCP_H_ */
