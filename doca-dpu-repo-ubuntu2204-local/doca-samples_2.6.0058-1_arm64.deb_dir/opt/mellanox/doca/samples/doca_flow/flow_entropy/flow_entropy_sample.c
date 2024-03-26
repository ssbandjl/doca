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

#include <string.h>
#include <unistd.h>

#include <rte_byteorder.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

DOCA_LOG_REGISTER(FLOW_ENTROPY);

/*
 * Run flow_entropy sample
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
flow_entropy(void)
{
	const int nb_ports = 1;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_entropy_format header;
	uint16_t entropy;
	doca_error_t result;

	result = init_doca_flow(1, "switch,hws", resource, nr_shared_resources);
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

	header.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	header.ip4.src_ip = BE_IPV4_ADDR(8, 8, 8, 8);
	header.ip4.dst_ip = BE_IPV4_ADDR(7, 7, 7, 7);
	header.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
	header.transport.src_port = rte_cpu_to_be_16(1234);
	header.transport.dst_port = rte_cpu_to_be_16(5678);

	doca_flow_port_calc_entropy(ports[0], &header, &entropy);
	/* The enteropy result should be equal to 0xdb9e */
	DOCA_LOG_INFO("The entropy for the given packet header is:0x%x", rte_be_to_cpu_16(entropy));

	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return DOCA_SUCCESS;
}
