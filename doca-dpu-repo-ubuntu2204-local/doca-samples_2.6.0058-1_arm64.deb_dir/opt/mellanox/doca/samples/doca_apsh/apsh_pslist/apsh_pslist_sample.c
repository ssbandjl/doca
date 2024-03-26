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
#include <inttypes.h>

#include <doca_apsh.h>
#include <doca_log.h>

#include "apsh_common.h"

DOCA_LOG_REGISTER(PSLIST);

/*
 * Calls the DOCA APSH API function that matches this sample name and prints the result
 *
 * @dma_device_name [in]: IBDEV Name of the device to use for DMA
 * @pci_vuid [in]: VUID of the device exposed to the target system
 * @os_type [in]: Indicates the OS type of the target system
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
pslist(const char *dma_device_name, const char *pci_vuid, enum doca_apsh_system_os os_type)
{
	doca_error_t result;
	int num_processes, i;
	struct doca_apsh_ctx *apsh_ctx;
	struct doca_apsh_system *sys;
	struct doca_apsh_process **pslist;
	/* Hardcoded pathes to the files created by doca_apsh_config tool */
	const char *os_symbols = "/tmp/symbols.json";
	const char *mem_region = "/tmp/mem_regions.json";

	/* Init */
	result = init_doca_apsh(dma_device_name, &apsh_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init the DOCA APSH lib");
		return result;
	}
	DOCA_LOG_INFO("DOCA APSH lib context init successful");

	result = init_doca_apsh_system(apsh_ctx, os_type, os_symbols, mem_region, pci_vuid, &sys);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init the system context");
		return result;
	}
	DOCA_LOG_INFO("DOCA APSH system context created");

	result = doca_apsh_processes_get(sys, &pslist, &num_processes);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create the process list");
		cleanup_doca_apsh(apsh_ctx, sys);
		return result;
	}
	DOCA_LOG_INFO("Successfully performed %s. Host system contains %d processes", __func__, num_processes);

	/* Print some attributes of the processes */
	DOCA_LOG_INFO("First 5 (or less) processes of system:");
	for (i = 0; i < 5 && i < num_processes; ++i) {
		DOCA_LOG_INFO("\tProcess %d  -  name: %s, pid: %u, CPU cycles the process consumed: %lu", i,
			doca_apsh_process_info_get(pslist[i], DOCA_APSH_PROCESS_COMM),
			doca_apsh_process_info_get(pslist[i], DOCA_APSH_PROCESS_PID),
			doca_apsh_process_info_get(pslist[i], DOCA_APSH_PROCESS_CPU_TIME));
	}

	/* Cleanup */
	doca_apsh_processes_free(pslist);
	cleanup_doca_apsh(apsh_ctx, sys);
	return DOCA_SUCCESS;
}
