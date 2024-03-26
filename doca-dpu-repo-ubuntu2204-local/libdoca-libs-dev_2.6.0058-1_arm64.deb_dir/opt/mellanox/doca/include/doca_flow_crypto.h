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

/**
 * @file doca_flow_crypto.h
 * @page doca flow crypto
 * @defgroup FLOW_CRYPTO flow net define
 * DOCA HW offload flow cryptonet structure define. For more details please refer to
 * the user guide on DOCA devzone.
 *
 * @{
 */

#ifndef DOCA_FLOW_CRYPTO_H_
#define DOCA_FLOW_CRYPTO_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief doca flow crypto operation protocol type
 */
enum doca_flow_crypto_protocol_type {
	DOCA_FLOW_CRYPTO_PROTOCOL_NONE = 0,
	/**< No security protocol engaged */
	DOCA_FLOW_CRYPTO_PROTOCOL_ESP,
	/**< IPsec ESP protocol action */
};

/**
 * @brief doca flow crypto operation action type
 */
enum doca_flow_crypto_action_type {
	DOCA_FLOW_CRYPTO_ACTION_NONE = 0,
	/**< No crypto action performed */
	DOCA_FLOW_CRYPTO_ACTION_ENCRYPT,
	/**< Perform encryption */
	DOCA_FLOW_CRYPTO_ACTION_DECRYPT,
	/**< Perform decryption/authentication */
};

/**
 * @brief doca flow crypto operation reformat type
 */
enum doca_flow_crypto_encap_action_type {
	DOCA_FLOW_CRYPTO_REFORMAT_NONE = 0,
	/**< No reformat action performed */
	DOCA_FLOW_CRYPTO_REFORMAT_ENCAP,
	/**< Perform encapsulation action */
	DOCA_FLOW_CRYPTO_REFORMAT_DECAP,
	/**< Perform decapsulation action */
};

/**
 * @brief doca flow crypto operation encapsulation header type
 */
enum doca_flow_crypto_encap_net_type {
	DOCA_FLOW_CRYPTO_HEADER_NONE = 0,
	/**< No network header involved */
	DOCA_FLOW_CRYPTO_HEADER_ESP_TUNNEL,
	/**< ESP tunnel header type */
	DOCA_FLOW_CRYPTO_HEADER_ESP_OVER_IP,
	/**< IPv4 network header type */
	DOCA_FLOW_CRYPTO_HEADER_UDP_ESP_OVER_IP,
	/**< IPv6 + UDP network header type */
	DOCA_FLOW_CRYPTO_HEADER_ESP_OVER_LAN,
	/**< UDP, TCP or ICMP network header type */
};

#ifdef __cplusplus
} /* extern "C" */
#endif

/** @} */

#endif /* DOCA_FLOW_CRYPTO_H_ */
