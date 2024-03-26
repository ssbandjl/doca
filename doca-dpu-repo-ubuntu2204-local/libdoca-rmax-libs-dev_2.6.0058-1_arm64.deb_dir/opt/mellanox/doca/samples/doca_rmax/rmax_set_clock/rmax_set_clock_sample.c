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

#include "rmax_common.h"

DOCA_LOG_REGISTER(RMAX_SET_CLOCK);

/*
 * Sets PTP clock device to be used internally in DOCA RMAX
 *
 * @pcie_addr [in]: PCIe address, to set the PTP clock cpability for
 * @state [in]: a place holder for DOCA core related objects
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
set_clock(const char *pcie_addr, struct program_core_objects *state)
{
	doca_error_t result;

	/* open DOCA device with the given PCI address */
	result = open_doca_device_with_pci(pcie_addr, NULL, &state->dev);
	if (result != DOCA_SUCCESS)
		return result;

	/* DOCA RMAX library Initialization */
	result = doca_rmax_init();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to initialize DOCA RMAX library: %s", doca_error_get_descr(result));
		destroy_core_objects(state);
		return result;
	}

	/* Set the device to use for obtaining PTP time */
	result = doca_rmax_set_clock(state->dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to clock for the device to use for obtaining PTP time.: %s", doca_error_get_descr(result));
		doca_rmax_release();
		return result;
	}

	result = doca_rmax_release();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to clock for the device to use for obtaining PTP time.: %s", doca_error_get_descr(result));
		return result;
	}

	return result;
}
