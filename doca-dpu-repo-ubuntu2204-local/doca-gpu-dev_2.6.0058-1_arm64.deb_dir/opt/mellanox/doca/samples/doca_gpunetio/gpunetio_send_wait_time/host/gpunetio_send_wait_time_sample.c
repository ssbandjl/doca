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

#include "../gpunetio_common.h"

#ifdef DOCA_ARCH_DPU
struct doca_flow_port *df_port;
#endif

DOCA_LOG_REGISTER(GPU_SEND_WAIT_TIME:SAMPLE);

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
	result = doca_dpdk_port_probe(*ddev, "tx_pp=500,txq_inline_max=0,dv_flow_en=2");
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
/*
 * Start DPDK port. This is not needed from CX7 or newer
 *
 * @dpdk_port_id [in]: DPDK port id to start
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
start_dpdk_port(uint16_t dpdk_port_id)
{
	int ret = 0, numq = 1;
	struct rte_eth_dev_info dev_info = {0};
	struct rte_eth_conf eth_conf = {
		.txmode = {
			.offloads = RTE_ETH_TX_OFFLOAD_SEND_ON_TIMESTAMP,
		},
	};

	struct rte_mempool *mp = NULL;
	struct rte_eth_txconf tx_conf;
	int32_t timestamp_offset;
	int32_t dynflag_bitnum;

	static const struct rte_mbuf_dynfield dynfield_desc = {
		RTE_MBUF_DYNFIELD_TIMESTAMP_NAME,
		sizeof(uint64_t),
		.align = __alignof__(uint64_t),
	};

	static const struct rte_mbuf_dynflag dynflag_desc = {
		RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME,
		0,
	};

	timestamp_offset = rte_mbuf_dynfield_register(&dynfield_desc);
	if (timestamp_offset < 0) {
		DOCA_LOG_ERR("Failed rte_eth_dev_info_get with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

	dynflag_bitnum = rte_mbuf_dynflag_register(&dynflag_desc);
	if (dynflag_bitnum == -1) {
		DOCA_LOG_ERR("Failed rte_eth_dev_info_get with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

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

	ret = rte_eth_dev_configure(dpdk_port_id, numq, numq, &eth_conf);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_configure with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

	mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, 2048,
					rte_eth_dev_socket_id(dpdk_port_id));
	if (mp == NULL) {
		DOCA_LOG_ERR("Failed rte_pktmbuf_pool_create with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

	tx_conf = dev_info.default_txconf;
	for (int idx = 0; idx < numq; idx++) {
		ret = rte_eth_rx_queue_setup(dpdk_port_id, idx, 2048,
						rte_eth_dev_socket_id(dpdk_port_id), NULL, mp);
		if (ret) {
			DOCA_LOG_ERR("Failed rte_eth_rx_queue_setup with: %s", rte_strerror(-ret));
			return DOCA_ERROR_DRIVER;
		}

		ret = rte_eth_tx_queue_setup(dpdk_port_id, idx, 2048,
						rte_eth_dev_socket_id(dpdk_port_id), &tx_conf);
		if (ret) {
			DOCA_LOG_ERR("Failed rte_eth_tx_queue_setup with: %s", rte_strerror(-ret));
			return DOCA_ERROR_DRIVER;
		}
	}

	ret = rte_eth_dev_start(dpdk_port_id);
	if (ret) {
		DOCA_LOG_ERR("Failed rte_eth_dev_start with: %s", rte_strerror(-ret));
		return DOCA_ERROR_DRIVER;
	}

#ifdef DOCA_ARCH_DPU
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
#endif
	return DOCA_SUCCESS;
}


/*
 * Get timestamp in nanoseconds
 *
 * @return: UTC timestamp
 */
uint64_t
get_ns(void)
{
	struct timespec t;
	int ret;

	ret = clock_gettime(CLOCK_REALTIME, &t);
	if (ret != 0)
		exit(EXIT_FAILURE);

	return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

/*
 * Create TX buf to send dummy packets to Ethernet broadcast address
 *
 * @txq [in]: DOCA Eth Tx queue with Tx buf
 * @num_packets [in]: Number of packets in the doca_buf_arr of the txbuf
 * @max_pkt_sz [in]: Max packet size
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_tx_buf(struct txq_queue *txq, uint32_t num_packets, uint32_t max_pkt_sz)
{
	doca_error_t status;
	struct tx_buf *buf;

	if (txq == NULL || num_packets == 0 || max_pkt_sz == 0) {
		DOCA_LOG_ERR("Invalid input arguments");
		return DOCA_ERROR_INVALID_VALUE;
	}

	buf = &(txq->txbuf);
	buf->num_packets = num_packets;
	buf->max_pkt_sz = max_pkt_sz;
	buf->gpu_dev = txq->gpu_dev;

	status = doca_mmap_create(&(buf->mmap));
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create doca_buf: failed to create mmap");
		return status;
	}

	status = doca_mmap_add_dev(buf->mmap, txq->ddev);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to add dev to buf: doca mmap internal error");
		return status;
	}

	status = doca_gpu_mem_alloc(buf->gpu_dev, buf->num_packets * buf->max_pkt_sz, 4096,
					DOCA_GPU_MEM_TYPE_GPU, (void **)&(buf->gpu_pkt_addr), NULL);
	if ((status != DOCA_SUCCESS) || (buf->gpu_pkt_addr == NULL)) {
		DOCA_LOG_ERR("Unable to alloc txbuf: failed to allocate gpu memory");
		return status;
	}

	/* Map GPU memory buffer used to send packets with DMABuf */
	status = doca_gpu_dmabuf_fd(buf->gpu_dev, buf->gpu_pkt_addr, buf->num_packets * buf->max_pkt_sz, &(buf->dmabuf_fd));
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_INFO("Mapping send queue buffer (0x%p size %dB) with legacy nvidia-peermem mode",
			buf->gpu_pkt_addr, buf->num_packets * buf->max_pkt_sz);

		/* If failed, use nvidia-peermem legacy method */
		status = doca_mmap_set_memrange(buf->mmap, buf->gpu_pkt_addr, (buf->num_packets * buf->max_pkt_sz));
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to start buf: doca mmap internal error");
			return status;
		}
	} else {
		DOCA_LOG_INFO("Mapping send queue buffer (0x%p size %dB dmabuf fd %d) with dmabuf mode",
			 buf->gpu_pkt_addr, (buf->num_packets * buf->max_pkt_sz), buf->dmabuf_fd);

		status = doca_mmap_set_dmabuf_memrange(buf->mmap, buf->dmabuf_fd, buf->gpu_pkt_addr, 0, (buf->num_packets * buf->max_pkt_sz));
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to set dmabuf memrange for mmap %s", doca_error_get_descr(status));
			return status;
		}
	}

	status = doca_mmap_set_permissions(buf->mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start buf: doca mmap internal error");
		return status;
	}

	status = doca_mmap_start(buf->mmap);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start buf: doca mmap internal error");
		return status;
	}

	status = doca_buf_arr_create(buf->mmap, &buf->buf_arr);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
		return status;
	}

	status = doca_buf_arr_set_target_gpu(buf->buf_arr, buf->gpu_dev);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
		return status;
	}

	status = doca_buf_arr_set_params(buf->buf_arr, buf->max_pkt_sz, buf->num_packets, 0);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
		return status;
	}

	status = doca_buf_arr_start(buf->buf_arr);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
		return status;
	}

	status = doca_buf_arr_get_gpu_handle(buf->buf_arr, &(buf->buf_arr_gpu));
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to get buff_arr GPU handle: %s", doca_error_get_descr(status));
		return status;
	}

	return DOCA_SUCCESS;
}

/*
 * Pre-prepare TX buf filling default values in GPU memory
 *
 * @txq [in]: DOCA Eth Tx queue handler
 * @dpdk_port_id [in]: DPDK port id of the DOCA device to get MAC address interface
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
prepare_tx_buf(struct txq_queue *txq, uint16_t dpdk_port_id)
{
	uint8_t *cpu_pkt_addr;
	uint8_t *pkt;
	struct ether_hdr *hdr;
	cudaError_t res_cuda;
	struct tx_buf *buf;
	struct rte_ether_addr mac_addr;
	int ret;
	uint32_t idx;
	const char *payload = "Sent from DOCA GPUNetIO";

	if (txq == NULL) {
		DOCA_LOG_ERR("Invalid input arguments");
		return DOCA_ERROR_INVALID_VALUE;
	}

	buf = &(txq->txbuf);
	buf->pkt_nbytes = strlen(payload);

	ret = rte_eth_macaddr_get(dpdk_port_id, &mac_addr);
	if (ret != 0)
		return DOCA_ERROR_DRIVER;

	cpu_pkt_addr = (uint8_t *) calloc(buf->num_packets * buf->max_pkt_sz, sizeof(uint8_t));
	if (cpu_pkt_addr == NULL) {
		DOCA_LOG_ERR("Error in txbuf preparation, failed to allocate memory");
		return DOCA_ERROR_NO_MEMORY;
	}

	for (idx = 0; idx < buf->num_packets; idx++) {
		pkt = cpu_pkt_addr + (idx * buf->max_pkt_sz);
		hdr = (struct ether_hdr *) pkt;

		hdr->s_addr_bytes[0] = mac_addr.addr_bytes[0];
		hdr->s_addr_bytes[1] = mac_addr.addr_bytes[1];
		hdr->s_addr_bytes[2] = mac_addr.addr_bytes[2];
		hdr->s_addr_bytes[3] = mac_addr.addr_bytes[3];
		hdr->s_addr_bytes[4] = mac_addr.addr_bytes[4];
		hdr->s_addr_bytes[5] = mac_addr.addr_bytes[5];

		hdr->d_addr_bytes[0] = 0x10;
		hdr->d_addr_bytes[1] = 0x11;
		hdr->d_addr_bytes[2] = 0x12;
		hdr->d_addr_bytes[3] = 0x13;
		hdr->d_addr_bytes[4] = 0x14;
		hdr->d_addr_bytes[5] = 0x15;

		hdr->ether_type = rte_cpu_to_be_16(DOCA_ETHER_TYPE_IPV4);

		/* Assuming no TCP flags needed */
		pkt = pkt + sizeof(struct ether_hdr);

		memcpy(pkt, payload, buf->pkt_nbytes);
	}

	/* Copy the whole list of packets into GPU memory buffer */
	res_cuda = cudaMemcpy(buf->gpu_pkt_addr, cpu_pkt_addr,
				buf->num_packets * buf->max_pkt_sz, cudaMemcpyDefault);
	free(cpu_pkt_addr);
	if (res_cuda != cudaSuccess) {
		DOCA_LOG_ERR("Function CUDA Memcpy cqe_addr failed with %s", cudaGetErrorString(res_cuda));
		return DOCA_ERROR_DRIVER;
	}

	return DOCA_SUCCESS;
}

/*
 * Destroy TX buf
 *
 * @txq [in]: DOCA Eth Tx queue with Tx buf
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
destroy_tx_buf(struct txq_queue *txq)
{
	doca_error_t status;
	struct tx_buf *buf;

	if (txq == NULL) {
		DOCA_LOG_ERR("Invalid input arguments");
		return DOCA_ERROR_INVALID_VALUE;
	}

	buf = &(txq->txbuf);

	/* Tx buf may not be created yet */
	if (buf == NULL)
		return DOCA_SUCCESS;

	if (buf->mmap) {
		status = doca_mmap_destroy(buf->mmap);
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to destroy doca_buf: failed to destroy mmap");
			return status;
		}
	}

	if (buf->gpu_pkt_addr) {
		status = doca_gpu_mem_free(txq->gpu_dev, buf->gpu_pkt_addr);
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to stop buf: failed to free gpu memory");
			return status;
		}
	}

	if (buf->buf_arr) {
		status = doca_buf_arr_stop(buf->buf_arr);
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to stop buf: failed to destroy doca_buf_arr");
			return status;
		}

		status = doca_buf_arr_destroy(buf->buf_arr);
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to stop buf: failed to destroy doca_buf_arr");
			return status;
		}
	}

	return DOCA_SUCCESS;
}

/*
 * Destroy DOCA Ethernet Tx queue for GPU
 *
 * @txq [in]: DOCA Eth Tx queue handler
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
destroy_txq(struct txq_queue *txq)
{
	doca_error_t result;

	if (txq == NULL) {
		DOCA_LOG_ERR("Can't destroy Tx queue, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	result = doca_ctx_stop(txq->eth_txq_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_destroy(txq->eth_txq_cpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

#ifdef DOCA_ARCH_DPU
	doca_flow_port_stop(df_port);
	doca_flow_destroy();
#endif

	result = doca_dev_close(txq->ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_dev_close: %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

/*
 * Create DOCA Ethernet Tx queue for GPU
 *
 * @txq [in]: DOCA Eth Tx queue handler
 * @gpu_dev [in]: DOCA GPUNetIO device
 * @ddev [in]: DOCA device
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
create_txq(struct txq_queue *txq, struct doca_gpu *gpu_dev, struct doca_dev *ddev)
{
	doca_error_t result;

	if (txq == NULL || gpu_dev == NULL || ddev == NULL) {
		DOCA_LOG_ERR("Can't create DOCA Eth Tx queue, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	txq->gpu_dev = gpu_dev;
	txq->ddev = ddev;

	result = doca_eth_txq_create(txq->ddev, MAX_SQ_DESCR_NUM, &(txq->eth_txq_cpu));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_txq_create: %s", doca_error_get_descr(result));
		destroy_txq(txq);
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_set_wait_on_time_offload(txq->eth_txq_cpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set eth_txq l3 offloads: %s", doca_error_get_descr(result));
		destroy_txq(txq);
		return DOCA_ERROR_BAD_STATE;
	}

	txq->eth_txq_ctx = doca_eth_txq_as_doca_ctx(txq->eth_txq_cpu);
	if (txq->eth_txq_ctx == NULL) {
		DOCA_LOG_ERR("Failed doca_eth_txq_as_doca_ctx: %s", doca_error_get_descr(result));
		destroy_txq(txq);
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_ctx_set_datapath_on_gpu(txq->eth_txq_ctx, txq->gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
		destroy_txq(txq);
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_ctx_start(txq->eth_txq_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
		destroy_txq(txq);
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_get_gpu_handle(txq->eth_txq_cpu, &(txq->eth_txq_gpu));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_txq_get_gpu_handle: %s", doca_error_get_descr(result));
		destroy_txq(txq);
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

/*
 * Launch GPUNetIO send wait on time sample
 *
 * @sample_cfg [in]: Sample config parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
gpunetio_send_wait_time(struct sample_send_wait_cfg *sample_cfg)
{
	doca_error_t result;
	uint64_t *intervals_cpu = NULL;
	uint64_t *intervals_gpu = NULL;
	uint64_t time_seed;
	struct doca_gpu *gpu_dev = NULL;
	struct doca_dev *ddev = NULL;
	uint16_t dpdk_dev_port_id;
	struct txq_queue txq = {0};
	enum doca_eth_wait_on_time_type wait_on_time_mode;
	cudaStream_t stream;
	cudaError_t res_rt = cudaSuccess;

	result = init_doca_device(sample_cfg->nic_pcie_addr, &ddev, &dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function init_doca_device returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_eth_txq_cap_get_wait_on_time_offload_supported(ddev, &wait_on_time_mode);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Wait on time offload error, returned %s",
				doca_error_get_descr(result));
		goto exit;
	}

#ifdef DOCA_ARCH_DPU
	result = start_dpdk_port(dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_txq returned %s", doca_error_get_descr(result));
		goto exit;
	}
#else
	if (wait_on_time_mode == DOCA_ETH_WAIT_ON_TIME_TYPE_DPDK) {
		result = start_dpdk_port(dpdk_dev_port_id);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Function create_txq returned %s", doca_error_get_descr(result));
			goto exit;
		}
	}
#endif

	DOCA_LOG_INFO("Wait on time supported mode: %s",
			(wait_on_time_mode == DOCA_ETH_WAIT_ON_TIME_TYPE_DPDK) ? "DPDK" : "Native");

	result = doca_gpu_create(sample_cfg->gpu_pcie_addr, &gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_gpu_create returned %s", doca_error_get_descr(result));
		goto exit;
	}

	result = create_txq(&txq, gpu_dev, ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_txq returned %s", doca_error_get_descr(result));
		goto exit;
	}

	result = create_tx_buf(&txq, NUM_PACKETS_X_BURST * NUM_BURST_SEND, PACKET_SIZE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_tx_buf returned %s", doca_error_get_descr(result));
		goto exit;
	}

	result = prepare_tx_buf(&txq, dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function prepare_tx_buf returned %s", doca_error_get_descr(result));
		goto exit;
	}

	res_rt = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return DOCA_ERROR_DRIVER;
	}

	result = doca_gpu_mem_alloc(gpu_dev, sizeof(uint64_t) * NUM_BURST_SEND, 4096, DOCA_GPU_MEM_TYPE_GPU_CPU,
					(void **)&intervals_gpu, (void **)&intervals_cpu);
	if (result != DOCA_SUCCESS || intervals_gpu == NULL || intervals_cpu == NULL) {
		DOCA_LOG_ERR("Failed to allocate gpu memory %s", doca_error_get_descr(result));
		goto exit;
	}

	time_seed = get_ns() + DELTA_NS;
	for (int idx = 0; idx < NUM_BURST_SEND; idx++) {
		result = doca_eth_txq_calculate_timestamp(txq.eth_txq_cpu, time_seed + (sample_cfg->time_interval_ns * idx), &intervals_cpu[idx]);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to get wait on time value for timestamp %ld, error %s",
					time_seed + (sample_cfg->time_interval_ns * idx), doca_error_get_descr(result));
			goto exit;
		}
	}

	DOCA_LOG_INFO("Launching CUDA kernel to send packets");
	kernel_send_wait_on_time(stream, &txq, intervals_gpu);
	cudaStreamSynchronize(stream);
	/*
	 * This is needed only because it's a synthetic example.
	 * Typical application works in a continuos loop so there is no need to wait.
	 */
	DOCA_LOG_INFO("Waiting 10 sec for %d packets to be sent",
			NUM_BURST_SEND * NUM_PACKETS_X_BURST);
	sleep(10);

exit:
	if (intervals_gpu)
		doca_gpu_mem_free(gpu_dev, intervals_gpu);

	result = destroy_tx_buf(&txq);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_txq returned %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = destroy_txq(&txq);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_txq returned %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("Sample finished successfully");

	return DOCA_SUCCESS;
}
