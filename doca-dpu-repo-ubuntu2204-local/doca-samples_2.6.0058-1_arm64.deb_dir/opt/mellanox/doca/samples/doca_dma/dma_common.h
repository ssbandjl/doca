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

#ifndef DMA_COMMON_H_
#define DMA_COMMON_H_

#include <unistd.h>
#include <stdbool.h>

#include <doca_dma.h>
#include <doca_error.h>

#include "common.h"

#define MAX_USER_ARG_SIZE 256			/* Maximum size of user input argument */
#define MAX_ARG_SIZE (MAX_USER_ARG_SIZE + 1)	/* Maximum size of input argument */
#define MAX_USER_TXT_SIZE 4096			/* Maximum size of user input text */
#define MAX_TXT_SIZE (MAX_USER_TXT_SIZE + 1)	/* Maximum size of input text */
#define PAGE_SIZE sysconf(_SC_PAGESIZE)		/* Page size */
#define NUM_DMA_TASKS (1)			/* DMA tasks number */

/* Configuration struct */
struct dma_config {
	char pci_address[DOCA_DEVINFO_PCI_ADDR_SIZE];	/* PCI device address */
	char cpy_txt[MAX_TXT_SIZE];			/* Text to copy between the two local buffers */
	char export_desc_path[MAX_ARG_SIZE];		/* Path to save/read the exported descriptor file */
	char buf_info_path[MAX_ARG_SIZE];		/* Path to save/read the buffer information file */
};

struct dma_resources {
	struct program_core_objects state;	/* Core objects that manage our "state" */
	struct doca_dma *dma_ctx;		/* DOCA DMA context */
	size_t num_remaining_tasks;		/* Number of remaining tasks to process */
	bool run_main_loop;			/* Should we keep on running the main loop? */
};

/*
 * Register the command line parameters for the DOCA DMA samples
 *
 * @is_remote [in]: Indication for handling configuration parameters which are
 * needed when there is a remote side
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t register_dma_params(bool is_remote);

/*
 * Allocate DOCA DMA resources
 *
 * @pcie_addr [in]: PCIe address of device to open
 * @resources [out]: Structure containing all DMA resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t allocate_dma_resources(const char *pcie_addr, struct dma_resources *resources);

/*
 * Destroy DOCA DMA resources
 *
 * @resources [out]: Structure containing all DMA resources
 * @dma_ctx [in]: DOCA DMA context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t destroy_dma_resources(struct dma_resources *resources);

/*
 * Allocate DOCA DMA host resources
 *
 * @pcie_addr [in]: PCIe address of device to open
 * @state [out]: Structure containing all DOCA core structures
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t allocate_dma_host_resources(const char *pcie_addr, struct program_core_objects *state);

/*
 * Destroy DOCA DMA host resources
 *
 * @state [in]: Structure containing all DOCA core structures
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t destroy_dma_host_resources(struct program_core_objects *state);

/*
 * Check if given device is capable of executing a DMA memcpy task.
 *
 * @devinfo [in]: The DOCA device information
 * @return: DOCA_SUCCESS if the device supports DMA memcpy task and DOCA_ERROR otherwise.
 */
doca_error_t dma_task_is_supported(struct doca_devinfo *devinfo);

#endif
