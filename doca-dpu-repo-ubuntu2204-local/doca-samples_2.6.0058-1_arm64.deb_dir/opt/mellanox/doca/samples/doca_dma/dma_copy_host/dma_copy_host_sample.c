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

#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <doca_dma.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_mmap.h>

#include "dma_common.h"

DOCA_LOG_REGISTER(DMA_COPY_HOST);

/*
 * Saves export descriptor and buffer information into two separate files
 *
 * @export_desc [in]: Export descriptor to write into a file
 * @export_desc_len [in]: Export descriptor length
 * @src_buffer [in]: Source buffer
 * @src_buffer_len [in]: Source buffer length
 * @export_desc_file_path [in]: Export descriptor file path
 * @buffer_info_file_path [in]: Buffer information file path
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
save_config_info_to_files(const void *export_desc, size_t export_desc_len, const char *src_buffer, size_t src_buffer_len,
			  char *export_desc_file_path, char *buffer_info_file_path)
{
	FILE *fp;
	uint64_t buffer_addr = (uintptr_t)src_buffer;
	uint64_t buffer_len = (uint64_t)src_buffer_len;

	fp = fopen(export_desc_file_path, "wb");
	if (fp == NULL) {
		DOCA_LOG_ERR("Failed to create the DMA copy file");
		return DOCA_ERROR_IO_FAILED;
	}

	if (fwrite(export_desc, 1, export_desc_len, fp) != export_desc_len) {
		DOCA_LOG_ERR("Failed to write all data into the file");
		fclose(fp);
		return DOCA_ERROR_IO_FAILED;
	}

	fclose(fp);

	fp = fopen(buffer_info_file_path, "w");
	if (fp == NULL) {
		DOCA_LOG_ERR("Failed to create the DMA copy file");
		return DOCA_ERROR_IO_FAILED;
	}

	fprintf(fp, "%" PRIu64 "\n", buffer_addr);
	fprintf(fp, "%" PRIu64 "", buffer_len);

	fclose(fp);

	return DOCA_SUCCESS;
}

/*
 * Run DOCA DMA Host copy sample
 *
 * @pcie_addr [in]: Device PCI address
 * @src_buffer [in]: Source buffer to copy
 * @src_buffer_size [in]: Buffer size
 * @export_desc_file_path [in]: Export descriptor file path
 * @buffer_info_file_name [in]: Buffer info file path
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
dma_copy_host(const char *pcie_addr, char *src_buffer, size_t src_buffer_size,
		     char *export_desc_file_path, char *buffer_info_file_name)
{
	struct program_core_objects state = {0};
	const void *export_desc;
	size_t export_desc_len;
	int enter = 0;
	doca_error_t result, tmp_result;

	/* Allocate resources */
	result = allocate_dma_host_resources(pcie_addr, &state);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate DMA host resources: %s", doca_error_get_descr(result));
		return result;
	}

	/* Allow exporting the mmap to DPU for read only operations */
	result = doca_mmap_set_permissions(state.src_mmap, DOCA_ACCESS_FLAG_PCI_READ_ONLY);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set mmap permissions: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	/* Populate the memory map with the allocated memory */
	result = doca_mmap_set_memrange(state.src_mmap, src_buffer, src_buffer_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set memory range for source mmap: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	result = doca_mmap_start(state.src_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start source mmap: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	/* Export DOCA mmap to enable DMA on Host*/
	result = doca_mmap_export_pci(state.src_mmap, state.dev, &export_desc, &export_desc_len);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start export source mmap: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	DOCA_LOG_INFO("Please copy %s and %s to the DPU and run DMA Copy DPU sample", export_desc_file_path, buffer_info_file_name);

	/* Saves the export desc and buffer info to files, it is the user responsibility to transfer them to the dpu */
	result = save_config_info_to_files(export_desc, export_desc_len, src_buffer, src_buffer_size,
					   export_desc_file_path, buffer_info_file_name);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to save configurations information: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	/* Wait for enter which means that the requester has finished reading */
	DOCA_LOG_INFO("Wait till the DPU has finished and press enter");
	while (enter != '\r' && enter != '\n')
		enter = getchar();

destroy_resources:
	tmp_result = destroy_dma_host_resources(&state);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to destroy DMA host resources: %s", doca_error_get_descr(tmp_result));
	}

	return result;
}
