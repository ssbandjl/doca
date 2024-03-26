/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <stdlib.h>
#include <string.h>

#include <doca_argp.h>
#include <doca_error.h>
#include <doca_dev.h>
#include <doca_log.h>

#include "dma_common.h"

DOCA_LOG_REGISTER(DPU_LOCAL_DMA_COPY::MAIN);

/* Sample's Logic */
doca_error_t dma_local_copy(const char *pcie_addr, char *dst_buffer, const char *src_buffer, size_t length);

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int
main(int argc, char **argv)
{
	struct dma_config dma_conf;
	char *dst_buffer = NULL, *src_buffer = NULL;
	size_t length;
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_FAILURE;

	/* Set the default configuration values (Example values) */
	strcpy(dma_conf.pci_address, "03:00.0");
	strcpy(dma_conf.cpy_txt, "This is a sample piece of text");
	/* No need to set export_desc_path and buf_info_path which are only needed for DMA across devices */

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		goto sample_exit;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	DOCA_LOG_INFO("Starting the sample");

#ifndef DOCA_ARCH_DPU
	DOCA_LOG_ERR("Local DMA copy can run only on the DPU");
	goto sample_exit;
#endif

	result = doca_argp_init("doca_dma_local_copy", &dma_conf);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}
	result = register_dma_params(false);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register DMA sample parameters: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	length = strlen(dma_conf.cpy_txt) + 1;
	dst_buffer = (char *)calloc(1, length);
	if (dst_buffer == NULL) {
		DOCA_LOG_ERR("Destination buffer allocation failed");
		goto argp_cleanup;
	}

	src_buffer = (char *)malloc(length);
	if (src_buffer == NULL) {
		DOCA_LOG_ERR("Source buffer allocation failed");
		goto dst_buffer_cleanup;
	}

	memcpy(src_buffer, dma_conf.cpy_txt, length);

	result = dma_local_copy(dma_conf.pci_address, dst_buffer, src_buffer, length);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("dma_local_copy() encountered an error: %s", doca_error_get_descr(result));
		goto src_buffer_cleanup;
	}

	exit_status = EXIT_SUCCESS;

src_buffer_cleanup:
	if (src_buffer != NULL)
		free(src_buffer);
dst_buffer_cleanup:
	if (dst_buffer != NULL)
		free(dst_buffer);
argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
