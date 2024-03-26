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

#include <stdlib.h>
#include <string.h>
#include <rte_ethdev.h>

#include "common.h"
#include "dpdk_tcp/tcp_session_table.h"
#include "dpdk_tcp/tcp_cpu_rss_func.h"

#define SLEEP_IN_NANOS (10 * 1000) /* Sample the PE every 10 microseconds  */

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING);

bool force_quit;
static struct doca_gpu *gpu_dev;
static struct app_gpu_cfg app_cfg = {0};
static struct doca_dev *ddev;
static uint16_t dpdk_dev_port_id;
static struct rxq_udp_queues udp_queues;
static struct rxq_tcp_queues tcp_queues;
static struct rxq_icmp_queues icmp_queues;
static struct txq_http_queues http_queues;
static struct doca_flow_port *df_port;
static struct doca_pe *pe;
static uint64_t icmp_last_ping;

/*
 * DOCA PE callback to be invoked if any Eth Txq get an error
 * sending packets.
 *
 * @event_error [in]: DOCA PE event error handler
 * @event_user_data [in]: custom user data set at registration time
 */
void
error_send_packet_cb(struct doca_eth_txq_gpu_event_error_send_packet *event_error, union doca_data event_user_data)
{
	uint16_t packet_index;

	doca_eth_txq_gpu_event_error_send_packet_get_position(event_error, &packet_index);
	DOCA_LOG_INFO("Error in send queue %ld, packet %d. Gracefully killing the app",
			event_user_data.u64, packet_index);
	DOCA_GPUNETIO_VOLATILE(force_quit) = true;
}

/*
 * DOCA PE callback to be invoked on ICMP Eth Txq to get the debug info
 * when sending packets
 *
 * @event_notify [in]: DOCA PE event debug handler
 * @event_user_data [in]: custom user data set at registration time
 */
void
debug_send_packet_icmp_cb(struct doca_eth_txq_gpu_event_notify_send_packet *event_notify, union doca_data event_user_data)
{
	uint16_t packet_index;
	uint64_t packet_timestamp;
	uint64_t ts_diff = 0;

	doca_eth_txq_gpu_event_notify_send_packet_get_position(event_notify, &packet_index);
	doca_eth_txq_gpu_event_notify_send_packet_get_timestamp(event_notify, &packet_timestamp);

	if (icmp_last_ping != 0)
		ts_diff = packet_timestamp - icmp_last_ping;

	DOCA_LOG_INFO("ICMP debug event: Queue %ld packet %d sent at %ld time from last ICMP is %.6f sec",
		event_user_data.u64, packet_index, packet_timestamp,
		(double)((ts_diff > 0 ? ((double)ts_diff) / 1000000000.0 : 0)));

	icmp_last_ping = packet_timestamp;
}

/*
 * Get timestamp in nanoseconds
 *
 * @sec [out]: seconds
 * @return: UTC timestamp
 */
static uint64_t
get_ns(uint64_t *sec)
{
	struct timespec t;
	int ret;

	ret = clock_gettime(CLOCK_REALTIME, &t);
	if (ret != 0)
		exit(EXIT_FAILURE);

	(*sec) = (uint64_t)t.tv_sec;

	return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

/*
 * CPU thread to print statistics from GPU filtering on the console
 *
 * @args [in]: thread input args
 */
static void
stats_core(void *args)
{
	doca_error_t result = DOCA_SUCCESS;
	enum doca_gpu_semaphore_status status;
	struct stats_udp udp_st[MAX_QUEUES] = {0};
	struct stats_tcp tcp_st[MAX_QUEUES] = {0};
	uint32_t sem_idx_udp[MAX_QUEUES] = {0};
	uint32_t sem_idx_tcp[MAX_QUEUES] = {0};
	uint64_t start_time_sec = 0;
	uint64_t interval_print = 0;
	uint64_t interval_sec = 0;
	struct stats_udp *custom_udp_st;
	struct stats_tcp *custom_tcp_st;

	DOCA_LOG_INFO("Core %u is reporting filter stats", rte_lcore_id());
	get_ns(&start_time_sec);
	interval_print = get_ns(&interval_sec);
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		/* Check UDP packets */
		for (int idxq = 0; idxq < udp_queues.numq; idxq++) {
			result = doca_gpu_semaphore_get_status(udp_queues.sem_cpu[idxq], sem_idx_udp[idxq], &status);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("UDP semaphore error");
				DOCA_GPUNETIO_VOLATILE(force_quit) = true;
				return;
			}

			if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
				result = doca_gpu_semaphore_get_custom_info_addr(udp_queues.sem_cpu[idxq], sem_idx_udp[idxq], (void **)&(custom_udp_st));
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("UDP semaphore get address error");
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				udp_st[idxq].dns += custom_udp_st->dns;
				udp_st[idxq].others += custom_udp_st->others;
				udp_st[idxq].total += custom_udp_st->total;

				result = doca_gpu_semaphore_set_status(udp_queues.sem_cpu[idxq], sem_idx_udp[idxq], DOCA_GPU_SEMAPHORE_STATUS_FREE);
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("UDP semaphore %d error", sem_idx_udp[idxq]);
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				sem_idx_udp[idxq] = (sem_idx_udp[idxq] + 1) % udp_queues.nums;
			}
		}

		/* Check TCP packets */
		for (int idxq = 0; idxq < tcp_queues.numq; idxq++) {
			result = doca_gpu_semaphore_get_status(tcp_queues.sem_cpu[idxq], sem_idx_tcp[idxq], &status);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("TCP semaphore error");
				DOCA_GPUNETIO_VOLATILE(force_quit) = true;
				return;
			}

			if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
				result = doca_gpu_semaphore_get_custom_info_addr(tcp_queues.sem_cpu[idxq], sem_idx_tcp[idxq], (void **)&(custom_tcp_st));
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("TCP semaphore get address error");
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				tcp_st[idxq].http += custom_tcp_st->http;
				tcp_st[idxq].http_head += custom_tcp_st->http_head;
				tcp_st[idxq].http_get += custom_tcp_st->http_get;
				tcp_st[idxq].http_post += custom_tcp_st->http_post;
				tcp_st[idxq].tcp_syn += custom_tcp_st->tcp_syn;
				tcp_st[idxq].tcp_fin += custom_tcp_st->tcp_fin;
				tcp_st[idxq].tcp_ack += custom_tcp_st->tcp_ack;
				tcp_st[idxq].others += custom_tcp_st->others;
				tcp_st[idxq].total += custom_tcp_st->total;

				result = doca_gpu_semaphore_set_status(tcp_queues.sem_cpu[idxq], sem_idx_tcp[idxq], DOCA_GPU_SEMAPHORE_STATUS_FREE);
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("TCP semaphore %d error", sem_idx_tcp[idxq]);
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				sem_idx_tcp[idxq] = (sem_idx_tcp[idxq] + 1) % tcp_queues.nums;
			}
		}

		if ((get_ns(&interval_sec) - interval_print) > 5000000000) {
			printf("\nSeconds %ld\n", interval_sec - start_time_sec);

			for (int idxq = 0; idxq < udp_queues.numq; idxq++) {
				printf("[UDP] QUEUE: %d DNS: %ld OTHER: %ld TOTAL: %ld\n",
					idxq,
					udp_st[idxq].dns,
					udp_st[idxq].others,
					udp_st[idxq].total
				);
			}

			for (int idxq = 0; idxq < tcp_queues.numq; idxq++) {
				printf("[TCP] QUEUE: %d HTTP: %d HTTP HEAD: %d HTTP GET: %d HTTP POST: %d TCP [SYN: %d FIN: %d ACK: %d] OTHER: %d TOTAL: %d\n",
					idxq,
					tcp_st[idxq].http,
					tcp_st[idxq].http_head,
					tcp_st[idxq].http_get,
					tcp_st[idxq].http_post,
					tcp_st[idxq].tcp_syn,
					tcp_st[idxq].tcp_fin,
					tcp_st[idxq].tcp_ack,
					tcp_st[idxq].others,
					tcp_st[idxq].total
				);
			}

			interval_print = get_ns(&interval_sec);
		}
	}
}


/*
 * Signal handler to quit application gracefully
 *
 * @signum [in]: signal received
 */
static void
signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		DOCA_LOG_INFO("Signal %d received, preparing to exit!", signum);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
	}
}

/*
 * GPU packet processing application main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int
main(int argc, char **argv)
{
	doca_error_t result;
	int current_lcore = 0;
	int cuda_id;
	cudaError_t cuda_ret;
	struct doca_log_backend *sdk_log;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	DOCA_LOG_INFO("===========================================================");
	DOCA_LOG_INFO("DOCA version: %s", doca_version());
	DOCA_LOG_INFO("===========================================================");

	/* Basic DPDK initialization */
	result = doca_argp_init("doca_gpu_packet_processing", &app_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = register_application_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("Options enabled:\n\tGPU %s\n\tNIC %s\n\tGPU Rx queues %d\n\tGPU HTTP server enabled %s",
		app_cfg.gpu_pcie_addr, app_cfg.nic_pcie_addr, app_cfg.queue_num,
		(app_cfg.http_server == true ? "Yes" : "No")
	);

	/* In a multi-GPU system, ensure CUDA refers to the right GPU device */
	cuda_ret = cudaDeviceGetByPCIBusId(&cuda_id, app_cfg.gpu_pcie_addr);
	if (cuda_ret != cudaSuccess) {
		DOCA_LOG_ERR("Invalid GPU bus id provided %s", app_cfg.gpu_pcie_addr);
		return DOCA_ERROR_INVALID_VALUE;
	}

	cudaFree(0);
	cudaSetDevice(cuda_id);

	result = init_doca_device(app_cfg.nic_pcie_addr, &ddev, &dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function init_doca_device returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Initialize DOCA GPU instance */
	result = doca_gpu_create(app_cfg.gpu_pcie_addr, &gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_gpu_create returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	df_port = init_doca_flow(dpdk_dev_port_id, app_cfg.queue_num);
	if (df_port == NULL) {
		DOCA_LOG_ERR("FAILED: init_doca_flow");
		return EXIT_FAILURE;
	}

	result = doca_pe_create(&pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create pe queue: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = create_udp_queues(&udp_queues, df_port, gpu_dev, ddev, app_cfg.queue_num, SEMAPHORES_PER_QUEUE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_udp_queues returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = create_tcp_queues(&tcp_queues, df_port, gpu_dev, ddev, app_cfg.queue_num, SEMAPHORES_PER_QUEUE, app_cfg.http_server, &http_queues,
				   pe, &error_send_packet_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_tcp_queues returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = create_icmp_queues(&icmp_queues, df_port, gpu_dev, ddev, MAX_QUEUES_ICMP,
				    pe, &error_send_packet_cb, &debug_send_packet_icmp_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_icmp_queues returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Create root control pipe to route tcp/udp/OS packets */
	result = create_root_pipe(&udp_queues, &tcp_queues, &icmp_queues, df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_root_pipe returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Gracefully terminate app if ctrlc */
	DOCA_GPUNETIO_VOLATILE(force_quit) = false;
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	cudaStream_t rx_tcp_stream, rx_udp_stream, rx_icmp_stream, tx_http_server;
	cudaError_t res_rt = cudaSuccess;
	uint32_t *cpu_exit_condition;
	uint32_t *gpu_exit_condition;

	res_rt = cudaStreamCreateWithFlags(&rx_udp_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return EXIT_FAILURE;
	}

	res_rt = cudaStreamCreateWithFlags(&rx_tcp_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return EXIT_FAILURE;
	}

	res_rt = cudaStreamCreateWithFlags(&rx_icmp_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return EXIT_FAILURE;
	}

	if (app_cfg.http_server) {
		res_rt = cudaStreamCreateWithFlags(&tx_http_server, cudaStreamNonBlocking);
		if (res_rt != cudaSuccess) {
			DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
			return EXIT_FAILURE;
		}
	}

	result = doca_gpu_mem_alloc(gpu_dev, sizeof(uint32_t), 4096, DOCA_GPU_MEM_TYPE_GPU_CPU, (void **)&gpu_exit_condition, (void **)&cpu_exit_condition);
	if (result != DOCA_SUCCESS || gpu_exit_condition == NULL || cpu_exit_condition == NULL) {
		DOCA_LOG_ERR("Function doca_gpu_mem_alloc returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}
	cpu_exit_condition[0] = 0;

	/*
	 * Some GPUs may require an initial warmup without doing any real operation.
	 */
	DOCA_LOG_INFO("Warm up CUDA kernels");
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
	kernel_receive_udp(rx_udp_stream, gpu_exit_condition, &udp_queues);
	kernel_receive_tcp(rx_tcp_stream, gpu_exit_condition, &tcp_queues, app_cfg.http_server);
	kernel_receive_icmp(rx_icmp_stream, gpu_exit_condition, &icmp_queues);
	if (app_cfg.http_server)
		kernel_http_server(tx_http_server, gpu_exit_condition, &tcp_queues, &http_queues);

	cudaStreamSynchronize(rx_udp_stream);
	cudaStreamSynchronize(rx_tcp_stream);
	cudaStreamSynchronize(rx_icmp_stream);
	if (app_cfg.http_server)
		cudaStreamSynchronize(tx_http_server);
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 0;

	DOCA_LOG_INFO("Launching CUDA kernels");

	kernel_receive_udp(rx_udp_stream, gpu_exit_condition, &udp_queues);
	kernel_receive_tcp(rx_tcp_stream, gpu_exit_condition, &tcp_queues, app_cfg.http_server);
	kernel_receive_icmp(rx_icmp_stream, gpu_exit_condition, &icmp_queues);
	if (app_cfg.http_server)
		kernel_http_server(tx_http_server, gpu_exit_condition, &tcp_queues, &http_queues);

	/* Launch stats proxy thread to report pipeline status */
	current_lcore = rte_get_next_lcore(current_lcore, true, false);
	if (rte_eal_remote_launch((void *)stats_core, NULL, current_lcore) != 0) {
		DOCA_LOG_ERR("Remote launch failed");
		goto exit;
	}

	if (app_cfg.http_server) {
		tcp_queues.tcp_ack_pkt_pool = rte_pktmbuf_pool_create("tcp_ack_pkt_pool", 1023, 0, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
		if (!tcp_queues.tcp_ack_pkt_pool) {
			DOCA_LOG_ERR("%s: failed to allocate tcp-ack packet pool", __func__);
			goto exit;
		}

		/* Start the CPU RSS threads to address new TCP connections */
		tcp_queues.lcore_idx_start = rte_get_next_lcore(current_lcore, true, false);
		for (int i = 0; i < tcp_queues.numq_cpu_rss; i++) {
			current_lcore = rte_get_next_lcore(current_lcore, true, false);
			if (rte_eal_remote_launch(tcp_cpu_rss_func, &tcp_queues, current_lcore) != 0) {
				DOCA_LOG_ERR("Remote launch failed");
				goto exit;
			}
		}
	}

	DOCA_LOG_INFO("Waiting for termination");
	/* This loop keeps busy main thread until force_quit is set to 1 (e.g. typing ctrl+c) */
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		doca_pe_progress(pe);
		nanosleep(&ts, &ts);
	}

	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
	cudaStreamSynchronize(rx_udp_stream);
	cudaStreamDestroy(rx_udp_stream);
	cudaStreamSynchronize(rx_tcp_stream);
	cudaStreamDestroy(rx_tcp_stream);
	cudaStreamSynchronize(rx_icmp_stream);
	cudaStreamDestroy(rx_icmp_stream);
	if (app_cfg.http_server) {
		cudaStreamSynchronize(tx_http_server);
		cudaStreamDestroy(tx_http_server);
	}

	doca_gpu_mem_free(gpu_dev, gpu_exit_condition);

	DOCA_LOG_INFO("GPU work ended");

	current_lcore = 0;
	RTE_LCORE_FOREACH_WORKER(current_lcore) {
		if (rte_eal_wait_lcore(current_lcore) < 0) {
			DOCA_LOG_ERR("Bad exit for coreid: %d", current_lcore);
			break;
		}
	}

exit:

	result = destroy_flow_queue(dpdk_dev_port_id, df_port, &icmp_queues, &udp_queues, &tcp_queues, app_cfg.http_server, &http_queues);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function finialize_doca_flow returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_gpu_destroy(gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy GPU: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_pe_destroy(pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_pe_destroy returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("Application finished successfully");

	return EXIT_SUCCESS;
}
