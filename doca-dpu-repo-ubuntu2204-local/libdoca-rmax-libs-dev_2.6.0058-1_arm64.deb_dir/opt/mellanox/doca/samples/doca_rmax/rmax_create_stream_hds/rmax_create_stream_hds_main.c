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

#include <doca_log.h>
#include <doca_rmax.h>

#include "rmax_common.h"
#include "common.h"

DOCA_LOG_REGISTER(RMAX_CREATE_STREAM_HDS::MAIN);

/* Sample's logic */
doca_error_t rmax_create_stream_hds(struct rmax_program_state *state, struct rmax_stream_config *stream_config);

/*
 * Initialize default configurations; stream and flow configurations that should be used when setting their attributes
 *
 * @config [out]: all needed configuration that should be set.
 */
static void
init_config(struct rmax_stream_config *config)
{

	union {
		const struct in_addr ip;
		uint8_t octets[4];
	} src_addr, dst_addr;

	src_addr.octets[0] = 192;
	src_addr.octets[1] = 168;
	src_addr.octets[2] = 105;
	src_addr.octets[3] = 3;

	dst_addr.octets[0] = 192;
	dst_addr.octets[1] = 168;
	dst_addr.octets[2] = 105;
	dst_addr.octets[3] = 2;

	config->type = DOCA_RMAX_IN_STREAM_TYPE_GENERIC;
	config->scatter_type = DOCA_RMAX_IN_STREAM_SCATTER_TYPE_ULP;
	config->src_ip.s_addr = src_addr.ip.s_addr;
	config->dst_ip.s_addr = dst_addr.ip.s_addr;
	config->dst_port = 5200;
	config->hdr_size = 8;
	config->data_size = 150;
	config->num_elements = 1024;

	strcpy(config->pci_address, "03:00.0");
}

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
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_FAILURE;
	struct rmax_program_state state;
	struct rmax_stream_config stream_config;

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

	/* default configurations */
	init_config(&stream_config);

	/* ARGP initialization */
	result = doca_argp_init("doca_rmax_create_stream_hds", &stream_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}

	/* register ARGP parameters of the sample */
	result = register_create_stream_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register ARGP params: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	/* start parsing received arguments */
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	/* Sample's main logic */
	result = rmax_create_stream_hds(&state, &stream_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("rmax_create_stream_hds() encountered an error: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	exit_status = EXIT_SUCCESS;

argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
