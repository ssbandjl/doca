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

DOCA_LOG_REGISTER(RMAX_SET_AFFINITY);

/*
 * Sets the CPU affinity, through DOCA Rivermax API
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
set_affinity(void)
{
	doca_error_t result;
	struct doca_rmax_cpu_affinity_mask mask;

	memset(&mask, 0, sizeof(mask));
	/* set affinity mask to CPU 0 */
	mask.cpu_bits[0] = 1;

	result  = doca_rmax_set_cpu_affinity_mask(&mask);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set CPU affinity: %s", doca_error_get_descr(result));
		return result;
	}

	/* initialization */
	result = doca_rmax_init();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to initialize DOCA Rivermax: %s", doca_error_get_descr(result));
		return result;
	}

	/* application code here */

	/* deinitialization */
	result = doca_rmax_release();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to deinitialize DOCA Rivermax: %s", doca_error_get_descr(result));
		return result;
	}

	return result;
}
