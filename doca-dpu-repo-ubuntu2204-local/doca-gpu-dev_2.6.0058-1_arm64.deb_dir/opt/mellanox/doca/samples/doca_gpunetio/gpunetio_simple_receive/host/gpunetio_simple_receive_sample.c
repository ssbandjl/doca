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
#include <rte_pmd_mlx5.h>
#include <rte_mbuf_dyn.h>

#include <doca_dpdk.h>
#include <doca_rdma_bridge.h>
#include <doca_flow.h>
#include <doca_log.h>

#include "gpunetio_common.h"

#include "common.h"

#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define MBUF_NUM 8192
#define MBUF_SIZE 2048

struct doca_flow_port *df_port;
bool force_quit;

DOCA_LOG_REGISTER(GPU_DMABUF:SAMPLE);

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
 * Initialize a DOCA network device.
 *
 * @nic_pcie_addr [in]: Network card PCIe address
 * @ddev [out]: DOCA device
 * @dpdk_port_id [out]: DPDK port id associated with the DOCA device
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
init_doca_device(char *nic_pcie_addr, struct doca_dev **ddev, uint16_t *dpdk_port_id)
{
	doca_error_t result;
	int ret;
	char *eal_param[3] = {"", "-a", "00:00.0"};

	if (nic_pcie_addr == NULL || ddev == NULL || dpdk_port_id == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	if (strnlen(nic_pcie_addr, DOCA_DEVINFO_PCI_ADDR_SIZE) >= DOCA_DEVINFO_PCI_ADDR_SIZE)
		return DOCA_ERROR_INVALID_VALUE;

	ret = rte_eal_init(3, eal_param);
	if (ret < 0) {
		DOCA_LOG_ERR("DPDK init failed: %d", ret);
		return DOCA_ERROR_DRIVER;
	}

	result = open_doca_device_with_pci(nic_pcie_addr, NULL, ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to open NIC device based on PCI address");
		return result;
	}

	/*
	 * From CX7, tx_pp is not needed anymore.
	 */
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

/*
 * Start DPDK port. This is not needed from CX7 or newer
 *
 * @dpdk_port_id [in]: DPDK port id to start
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
start_dpdk_port(uint16_t dpdk_port_id)
{
	int ret = 0;
	struct rte_eth_dev_info dev_info = {0};
	struct rte_eth_conf eth_conf = {};
	struct rte_flow_error error = {};

	/*
	 * DPDK should be initialized and started before DOCA Flow.
	 * DPDK doesn't start the device without, at least, one DPDK Rx queue.
	 * DOCA Flow needs to specify in advance how many Rx queues will be used by the program.
	 *
	 * Following lines of code can be considered the minimum WAR for this issue.
	 */

	ret = rte_eth_dev_info_get(dpdk_port_id, &dev_info);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_info_get with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

	ret = rte_eth_dev_configure(dpdk_port_id, 0, 0, &eth_conf);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_configure with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

	ret = rte_flow_isolate(dpdk_port_id, 1, &error);
	if (ret) {
		DOCA_LOG_ERR("Failed to configure dpdk port: %s", error.message);
		return DOCA_ERROR_DRIVER;
	}

	ret = rte_eth_dev_start(dpdk_port_id);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_start with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

	/* Maximal length of port name */
	#define MAX_PORT_STR_LEN 128

	struct doca_flow_port_cfg port_cfg = {0};
	struct doca_flow_cfg queue_flow_cfg = {0};
	doca_error_t result;
	char port_id_str[MAX_PORT_STR_LEN];

	/* Initialize doca flow framework */
	queue_flow_cfg.pipe_queues = 1;
	/*
	 * HWS: Hardware steering
	 * Isolated: don't create RSS rule for DPDK created RX queues
	 */
	queue_flow_cfg.mode_args = "vnf,hws,isolated";
	queue_flow_cfg.resource.nb_counters = FLOW_NB_COUNTERS;

	result = doca_flow_init(&queue_flow_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init doca flow with: %s", doca_error_get_descr(result));
		return DOCA_ERROR_NOT_SUPPORTED;
	}

	/* Start doca flow port */
	port_cfg.port_id = dpdk_port_id;
	port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
	snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
	port_cfg.devargs = port_id_str;
	result = doca_flow_port_start(&port_cfg, &df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start doca flow port with: %s", doca_error_get_descr(result));
		return DOCA_ERROR_NOT_SUPPORTED;
	}

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Flow UDP pipeline
 *
 * @rxq [in]: Receive queue handler
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_udp_pipe(struct rxq_queue *rxq)
{
	doca_error_t result;
	struct doca_flow_match match = {0};
	struct doca_flow_match match_mask = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_fwd miss_fwd = {0};
	struct doca_flow_pipe_cfg pipe_cfg = {0};
	struct doca_flow_pipe_entry *entry;
	uint16_t flow_queue_id;
	uint16_t rss_queues[1];
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	if (rxq == NULL || df_port == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	pipe_cfg.attr.name = "GPU_RXQ_UDP_PIPE";
	pipe_cfg.attr.enable_strict_matching = true;
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.attr.nb_actions = 0;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.match = &match;
	pipe_cfg.match_mask = &match_mask;
	pipe_cfg.monitor = &monitor;
	pipe_cfg.port = df_port;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

	doca_eth_rxq_get_flow_queue_id(rxq->eth_rxq_cpu, &flow_queue_id);
	rss_queues[0] = flow_queue_id;

	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
	fwd.num_of_queues = 1;

	miss_fwd.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &(rxq->rxq_pipe));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	/* Add HW offload */
	result = doca_flow_pipe_add_entry(0, rxq->rxq_pipe, &match, NULL, NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(df_port, 0, 0, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Flow root pipeline
 *
 * @rxq [in]: Receive queue handler
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_root_pipe(struct rxq_queue *rxq)
{
	doca_error_t result;
	struct doca_flow_match match_mask = {0};
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	if (rxq == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	struct doca_flow_pipe_cfg pipe_cfg = {
		.attr = {
			.name = "ROOT_PIPE",
			.enable_strict_matching = true,
			.is_root = true,
			.type = DOCA_FLOW_PIPE_CONTROL,
		},
		.port = df_port,
		.monitor = &monitor,
		.match_mask = &match_mask,
	};

	result = doca_flow_pipe_create(&pipe_cfg, NULL, NULL, &rxq->root_pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	struct doca_flow_match udp_match = {
		.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4,
		.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP,
	};

	struct doca_flow_fwd udp_fwd = {
		.type = DOCA_FLOW_FWD_PIPE,
		.next_pipe = rxq->rxq_pipe,
	};

	result = doca_flow_pipe_control_add_entry(0, 0, rxq->root_pipe, &udp_match, NULL, NULL, NULL, NULL, NULL, NULL,
							&udp_fwd, NULL, &rxq->root_udp_entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe UDP entry creation failed with: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(df_port, 0, 0, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe entry process failed with: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

	return DOCA_SUCCESS;
}

/*
 * Destroy DOCA Ethernet Tx queue for GPU
 *
 * @rxq [in]: DOCA Eth Rx queue handler
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
destroy_rxq(struct rxq_queue *rxq)
{
	doca_error_t result;

	if (rxq == NULL) {
		DOCA_LOG_ERR("Can't destroy UDP queues, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	DOCA_LOG_INFO("Destroying Rxq");

	doca_flow_port_stop(df_port);
	doca_flow_destroy();

	if (rxq->eth_rxq_ctx != NULL) {
		result = doca_ctx_stop(rxq->eth_rxq_ctx);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->eth_rxq_cpu != NULL) {
		result = doca_eth_rxq_destroy(rxq->eth_rxq_cpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->pkt_buff_mmap != NULL) {
		result = doca_mmap_destroy(rxq->pkt_buff_mmap);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to destroy mmap: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->gpu_pkt_addr != NULL) {
		result = doca_gpu_mem_free(rxq->gpu_dev, rxq->gpu_pkt_addr);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to free gpu memory: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Ethernet Tx queue for GPU
 *
 * @rxq [in]: DOCA Eth Tx queue handler
 * @gpu_dev [in]: DOCA GPUNetIO device
 * @ddev [in]: DOCA device
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_rxq(struct rxq_queue *rxq, struct doca_gpu *gpu_dev, struct doca_dev *ddev)
{
	doca_error_t result;
	uint32_t cyclic_buffer_size = 0;

	if (rxq == NULL || gpu_dev == NULL || ddev == NULL) {
		DOCA_LOG_ERR("Can't create UDP queues, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	rxq->gpu_dev = gpu_dev;
	rxq->ddev = ddev;
	rxq->port = df_port;

	DOCA_LOG_INFO("Creating Sample Eth Rxq\n");

	result = doca_eth_rxq_create(rxq->ddev, MAX_PKT_NUM, MAX_PKT_SIZE, &(rxq->eth_rxq_cpu));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_create: %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_rxq_set_type(rxq->eth_rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_set_type: %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, MAX_PKT_SIZE, MAX_PKT_NUM, 0, &cyclic_buffer_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to get eth_rxq cyclic buffer size: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_mmap_create(&rxq->pkt_buff_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_mmap_add_dev(rxq->pkt_buff_mmap, rxq->ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add dev to mmap: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_gpu_mem_alloc(rxq->gpu_dev, cyclic_buffer_size, GPU_PAGE_SIZE, DOCA_GPU_MEM_TYPE_GPU, &rxq->gpu_pkt_addr, NULL);
	if (result != DOCA_SUCCESS || rxq->gpu_pkt_addr == NULL) {
		DOCA_LOG_ERR("Failed to allocate gpu memory %s", doca_error_get_descr(result));
		goto exit_error;
	}

	/* Map GPU memory buffer used to receive packets with DMABuf */
	result = doca_gpu_dmabuf_fd(rxq->gpu_dev, rxq->gpu_pkt_addr, cyclic_buffer_size, &(rxq->dmabuf_fd));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB) with nvidia-peermem mode",
			rxq->gpu_pkt_addr, cyclic_buffer_size);

		/* If failed, use nvidia-peermem legacy method */
		result = doca_mmap_set_memrange(rxq->pkt_buff_mmap, rxq->gpu_pkt_addr, cyclic_buffer_size);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to set memrange for mmap %s", doca_error_get_descr(result));
			goto exit_error;
		}
	} else {
		DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB dmabuf fd %d) with dmabuf mode",
			rxq->gpu_pkt_addr, cyclic_buffer_size, rxq->dmabuf_fd);

		result = doca_mmap_set_dmabuf_memrange(rxq->pkt_buff_mmap, rxq->dmabuf_fd, rxq->gpu_pkt_addr, 0, cyclic_buffer_size);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to set dmabuf memrange for mmap %s", doca_error_get_descr(result));
			goto exit_error;
		}
	}

	result = doca_mmap_set_permissions(rxq->pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set permissions for mmap %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_mmap_start(rxq->pkt_buff_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start mmap %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_eth_rxq_set_pkt_buf(rxq->eth_rxq_cpu, rxq->pkt_buff_mmap, 0, cyclic_buffer_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set cyclic buffer  %s", doca_error_get_descr(result));
		goto exit_error;
	}

	rxq->eth_rxq_ctx = doca_eth_rxq_as_doca_ctx(rxq->eth_rxq_cpu);
	if (rxq->eth_rxq_ctx == NULL) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_as_doca_ctx: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_ctx_set_datapath_on_gpu(rxq->eth_rxq_ctx, rxq->gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_ctx_start(rxq->eth_rxq_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_eth_rxq_get_gpu_handle(rxq->eth_rxq_cpu, &(rxq->eth_rxq_gpu));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_get_gpu_handle: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	/* Create UDP based flow pipe */
	result = create_udp_pipe(rxq);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_udp_pipe returned %s", doca_error_get_descr(result));
		goto exit_error;
	}

	/* Create root pipe with UDP pipe as unique entry */
	result = create_root_pipe(rxq);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_root_pipe returned %s", doca_error_get_descr(result));
		goto exit_error;
	}

	return DOCA_SUCCESS;

exit_error:
	destroy_rxq(rxq);
	return DOCA_ERROR_BAD_STATE;
}

/*
 * Launch GPUNetIO simple receive sample
 *
 * @sample_cfg [in]: Sample config parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
gpunetio_simple_receive(struct sample_send_wait_cfg *sample_cfg)
{
	doca_error_t result;
	struct doca_gpu *gpu_dev = NULL;
	struct doca_dev *ddev = NULL;
	uint16_t dpdk_dev_port_id;
	struct rxq_queue rxq = {0};
	cudaStream_t stream;
	cudaError_t res_rt = cudaSuccess;
	uint32_t *cpu_exit_condition;
	uint32_t *gpu_exit_condition;

	result = init_doca_device(sample_cfg->nic_pcie_addr, &ddev, &dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function init_doca_device returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = start_dpdk_port(dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function start_dpdk_port returned %s", doca_error_get_descr(result));
		goto exit;
	}

	/* Gracefully terminate sample if ctrlc */
	DOCA_GPUNETIO_VOLATILE(force_quit) = false;
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	result = doca_gpu_create(sample_cfg->gpu_pcie_addr, &gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_gpu_create returned %s", doca_error_get_descr(result));
		goto exit;
	}

	result = create_rxq(&rxq, gpu_dev, ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_rxq returned %s", doca_error_get_descr(result));
		goto exit;
	}

	res_rt = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return DOCA_ERROR_DRIVER;
	}

	result = doca_gpu_mem_alloc(gpu_dev, sizeof(uint32_t), 4096, DOCA_GPU_MEM_TYPE_GPU_CPU, (void **)&gpu_exit_condition, (void **)&cpu_exit_condition);
	if (result != DOCA_SUCCESS || gpu_exit_condition == NULL || cpu_exit_condition == NULL) {
		DOCA_LOG_ERR("Function doca_gpu_mem_alloc returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}
	cpu_exit_condition[0] = 0;

	DOCA_LOG_INFO("Launching CUDA kernel to receive packets");

	kernel_receive_packets(stream, &rxq, gpu_exit_condition);

	DOCA_LOG_INFO("Waiting for termination");
	/* This loop keeps busy main thread until force_quit is set to 1 (e.g. typing ctrl+c) */
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false)
		;
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;

	DOCA_LOG_INFO("Exiting from sample");

	cudaStreamSynchronize(stream);
exit:

	result = destroy_rxq(&rxq);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function destroy_rxq returned %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("Sample finished successfully");

	return DOCA_SUCCESS;
}
