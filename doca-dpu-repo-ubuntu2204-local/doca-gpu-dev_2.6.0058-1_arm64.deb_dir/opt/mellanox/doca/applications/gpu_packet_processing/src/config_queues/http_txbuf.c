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

#include <arpa/inet.h>
#include <rte_ethdev.h>
#include <doca_flow.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_buf_array.h>

#include "common.h"
#include "packets.h"
#include "dpdk_tcp/tcp_session_table.h"

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING_TXBUF);

const char *payload_page_index = "HTTP/1.1 200 OK\r\n"
	"Date: Sun, 30 Apr 2023 20:30:40 GMT\r\n"
	"Content-Type: text/html; charset=UTF-8\r\n"
	"Content-Length: 158\r\n"
	"Last-Modified: Sun, 30 Apr 2023 22:38:34 GMT\r\n"
	"Server: GPUNetIO\r\n"
	"Accept-Ranges: bytes\r\n"
	"Connection: keep-alive\r\n"
	"Keep-Alive: timeout=5\r\n"
	"\r\n"
	"<html>\r\n"
	"  <head>\r\n"
	"    <title>GPUNetIO index page</title>\r\n"
	"  </head>\r\n"
	"  <body>\r\n"
	"    <p>Hello World, the GPUNetIO server Index page!</p>\r\n"
	"  </body>\r\n"
	"</html>\r\n"
	"\r\n";

const char *payload_page_contacts = "HTTP/1.1 200 OK\r\n"
	"Date: Sun, 30 Apr 2023 20:30:40 GMT\r\n"
	"Content-Type: text/html; charset=UTF-8\r\n"
	"Content-Length: 175\r\n"
	"Last-Modified: Sun, 30 Apr 2023 22:38:34 GMT\r\n"
	"Server: GPUNetIO\r\n"
	"Accept-Ranges: bytes\r\n"
	"Connection: keep-alive\r\n"
	"Keep-Alive: timeout=5\r\n"
	"\r\n"
	"<html>\r\n"
	"  <head>\r\n"
	"    <title>GPUNetIO Contact page</title>\r\n"
	"  </head>\r\n"
	"  <body>\r\n"
	"    <p>For any GPUNetIO question please contact support@nvidia.com</p>\r\n"
	"  </body>\r\n"
	"</html>\r\n"
	"\r\n";

const char *payload_page_not_found = "HTTP/1.1 404 Not Found\r\n"
	"Date: Sun, 30 Apr 2023 20:30:40 GMT\r\n"
	"Content-Type: text/html; charset=UTF-8\r\n"
	"Content-Length: 152\r\n"
	"Last-Modified: Sun, 30 Apr 2023 22:38:34 GMT\r\n"
	"Server: GPUNetIO\r\n"
	"Connection: close\r\n"
	"\r\n"
	"<html>\r\n"
	"  <head>\r\n"
	"    <title>GPUNetIO 404 page</title>\r\n"
	"  </head>\r\n"
	"  <body>\r\n"
	"    <p>Hello! Page you requested doesn't exist!</p>\r\n"
	"  </body>\r\n"
	"</html>\r\n"
	"\r\n";


doca_error_t
create_tx_buf(struct tx_buf *buf, struct doca_gpu *gpu_dev, struct doca_dev *ddev, uint32_t num_packets, uint32_t max_pkt_sz)
{
	doca_error_t status;

	if (buf == NULL || gpu_dev == NULL || ddev == NULL || num_packets == 0 || max_pkt_sz == 0) {
		DOCA_LOG_ERR("Invalid input arguments");
		return DOCA_ERROR_INVALID_VALUE;
	}

	buf->gpu_dev = gpu_dev;
	buf->ddev = ddev;
	buf->num_packets = num_packets;
	buf->max_pkt_sz = max_pkt_sz;

	status = doca_mmap_create(&(buf->mmap));
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create doca_buf: failed to create mmap");
		return status;
	}

	status = doca_mmap_add_dev(buf->mmap, buf->ddev);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to add dev to buf: doca mmap internal error");
		return status;
	}

	status = doca_gpu_mem_alloc(buf->gpu_dev, buf->num_packets * buf->max_pkt_sz, 4096, DOCA_GPU_MEM_TYPE_GPU, (void **)&(buf->gpu_pkt_addr), NULL);
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

	status = doca_mmap_set_permissions(buf->mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
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

doca_error_t
prepare_tx_buf(struct tx_buf *buf, enum http_page_get page_type)
{
	uint8_t *cpu_pkt_addr;
	uint8_t *pkt;
	struct eth_ip_tcp_hdr *hdr;
	const char *payload;
	cudaError_t res_cuda;

	if (page_type == HTTP_GET_INDEX)
		payload = payload_page_index;
	else if (page_type == HTTP_GET_CONTACTS)
		payload = payload_page_contacts;
	else
		payload = payload_page_not_found;

	buf->pkt_nbytes = strlen(payload);

	cpu_pkt_addr = (uint8_t *) calloc(buf->num_packets * buf->max_pkt_sz, sizeof(uint8_t));
	if (cpu_pkt_addr == NULL) {
		DOCA_LOG_ERR("Error in txbuf preparation, failed to allocate memory");
		return DOCA_ERROR_NO_MEMORY;
	}

	for (int idx = 0; idx < buf->num_packets; idx++) {
		pkt = cpu_pkt_addr + (idx * buf->max_pkt_sz);
		hdr = (struct eth_ip_tcp_hdr *) pkt;

		hdr->l2_hdr.ether_type = rte_cpu_to_be_16(DOCA_ETHER_TYPE_IPV4);

		hdr->l3_hdr.version_ihl = 0x45;
		hdr->l3_hdr.type_of_service = 0x0;
		hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr) + buf->pkt_nbytes);
		hdr->l3_hdr.packet_id = 0;
		hdr->l3_hdr.fragment_offset = 0;
		hdr->l3_hdr.time_to_live = 60;
		hdr->l3_hdr.next_proto_id = 6;
		hdr->l3_hdr.hdr_checksum = 0;
		hdr->l3_hdr.src_addr = 0;
		hdr->l3_hdr.dst_addr = 0;

		hdr->l4_hdr.src_port = 0;
		hdr->l4_hdr.dst_port = 0;
		hdr->l4_hdr.sent_seq = 0;
		hdr->l4_hdr.recv_ack = 0;
		/* Assuming no TCP flags needed */
		hdr->l4_hdr.dt_off = 0x50; //5 << 4;
		/* Assuming no TCP flags needed */
		hdr->l4_hdr.tcp_flags = TCP_FLAG_PSH | TCP_FLAG_ACK; //| TCP_FLAG_FIN;
		hdr->l4_hdr.rx_win = BYTE_SWAP16(6000);
		hdr->l4_hdr.cksum = 0;
		hdr->l4_hdr.tcp_urp = 0;

		/* Assuming no TCP flags needed */
		pkt = pkt + sizeof(struct eth_ip_tcp_hdr);

		memcpy(pkt, payload, buf->pkt_nbytes);
	}

	/* Copy the whole list of packets into GPU memory buffer */
	res_cuda = cudaMemcpy(buf->gpu_pkt_addr, cpu_pkt_addr, buf->num_packets * buf->max_pkt_sz, cudaMemcpyDefault);
	free(cpu_pkt_addr);
	if (res_cuda != cudaSuccess) {
		DOCA_LOG_ERR("Function CUDA Memcpy cqe_addr failed with %s", cudaGetErrorString(res_cuda));
		return DOCA_ERROR_DRIVER;
	}

	return DOCA_SUCCESS;
}

doca_error_t
destroy_tx_buf(struct tx_buf *buf)
{
	doca_error_t status;

	if (buf == NULL) {
		DOCA_LOG_ERR("Invalid input arguments");
		return DOCA_ERROR_INVALID_VALUE;
	}

	status = doca_mmap_stop(buf->mmap);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to stop buf: unable to stop mmap");
		return status;
	}

	status = doca_mmap_rm_dev(buf->mmap, buf->ddev);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to remove dev from buf: doca mmap internal error");
		return status;
	}

	status = doca_mmap_destroy(buf->mmap);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to destroy doca_buf: failed to destroy mmap");
		return status;
	}

	status = doca_gpu_mem_free(buf->gpu_dev, buf->gpu_pkt_addr);
	if (status != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to stop buf: failed to free gpu memory");
		return status;
	}

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

	return status;
}
