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

#include <rte_ethdev.h>

#include "common.h"

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING_DEVICE);

struct rte_eth_dev_info dev_info = {0};
struct rte_eth_conf eth_conf = {0};
struct rte_mempool *mp;

/*
 * Initialize a DOCA device with PCIe address object.
 *
 * @pcie_value [in]: PCIe address object
 * @retval [out]: DOCA device object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR_INVALID_VALUE otherwise
 */
static doca_error_t
open_doca_device_with_pci(const char *pcie_value, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	int res;
	size_t i;
	uint8_t is_addr_equal = 0;

	/* Set default return value */
	*retval = NULL;

	res = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value: %d", res);
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_is_equal_pci_addr(dev_list[i], pcie_value, &is_addr_equal);
		if (res == DOCA_SUCCESS && is_addr_equal) {
			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_destroy_list(dev_list);
				return res;
			}
		}
	}

	DOCA_LOG_ERR("Matching device not found");
	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_destroy_list(dev_list);
	return res;
}

doca_error_t
init_doca_device(char *nic_pcie_addr, struct doca_dev **ddev, uint16_t *dpdk_port_id)
{
	doca_error_t result;
	int ret;
	char *eal_param[3] = {"", "-a", "00:00.0"};

	if (nic_pcie_addr == NULL || ddev == NULL || dpdk_port_id == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	if (strlen(nic_pcie_addr) >= DOCA_DEVINFO_PCI_ADDR_SIZE)
		return DOCA_ERROR_INVALID_VALUE;

	result = open_doca_device_with_pci(nic_pcie_addr, ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to open NIC device based on PCI address");
		return result;
	}

	ret = rte_eal_init(3, eal_param);
	if (ret < 0) {
		DOCA_LOG_ERR("DPDK init failed: %d", ret);
		return DOCA_ERROR_DRIVER;
	}

	/* Enable DOCA Flow HWS mode */
	result = doca_dpdk_port_probe(*ddev, "dv_flow_en=2");
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_dpdk_port_probe returned %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_dpdk_get_first_port_id(*ddev, dpdk_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_dpdk_get_first_port_id returned %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}
