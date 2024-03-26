/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <doca_argp.h>
#include <doca_log.h>
#include <doca_dpdk.h>

#include <dpdk_utils.h>

#include "flow_ct_common.h"
#include "common.h"

DOCA_LOG_REGISTER(FLOW_CT_UDP_QUERY::MAIN);

/* Sample's Logic */
doca_error_t
flow_ct_udp_query(uint16_t nb_queues, struct doca_dev *ct_dev);

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
	struct doca_dev *ct_dev = NULL;
	struct ct_config ct_cfg = {0};
	struct application_dpdk_config dpdk_config = {
		.port_config.nb_ports = 2,
		.port_config.nb_queues = 2,
		.reserve_main_thread = false,
	};

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

	result = doca_argp_init("doca_flow_ct_udp_query", &ct_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}
	doca_argp_set_dpdk_program(flow_ct_dpdk_init);

	result = flow_ct_register_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register Flow Ct sample parameters: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = open_doca_device_with_pci(ct_cfg.ct_dev_pci_addr, flow_ct_capable, &ct_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to open Flow CT device: %s", doca_error_get_descr(result));
		goto dpdk_cleanup;
	}

	result = doca_dpdk_port_probe(ct_dev, "dv_flow_en=2,dv_xmeta_en=4,representor=pf[0-1],repr_matching_en=1");
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to open Flow CT device: %s", doca_error_get_descr(result));
		goto device_cleanup;
	}

	/* update queues and ports */
	result = dpdk_queues_and_ports_init(&dpdk_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to update ports and queues");
		goto device_cleanup;
	}

	/* run sample */
	result = flow_ct_udp_query(dpdk_config.port_config.nb_queues, ct_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("flow_ct_udp_query() encountered an error: %s", doca_error_get_descr(result));
		goto dpdk_ports_queues_cleanup;
	}

	exit_status = EXIT_SUCCESS;
dpdk_ports_queues_cleanup:
	dpdk_queues_and_ports_fini(&dpdk_config);
device_cleanup:
	doca_dev_close(ct_dev);
dpdk_cleanup:
	dpdk_fini();
argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
