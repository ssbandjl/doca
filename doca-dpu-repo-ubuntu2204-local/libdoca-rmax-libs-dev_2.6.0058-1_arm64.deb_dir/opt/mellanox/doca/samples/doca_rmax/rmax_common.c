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

#include <unistd.h>

#include "rmax_common.h"

DOCA_LOG_REGISTER(rmax_common);

/*
 * ARGP Callback - Handle PCI device address parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
pci_address_callback(void *param, void *config)
{
	struct rmax_stream_config *cfg = (struct rmax_stream_config *)config;
	char *pci_address = (char *)param;
	int len;

	len = strnlen(pci_address, DOCA_DEVINFO_PCI_ADDR_SIZE);
	if (len == DOCA_DEVINFO_PCI_ADDR_SIZE) {
		DOCA_LOG_ERR("Entered device PCI address exceeding the maximum size of %d", DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}
	strncpy(cfg->pci_address, pci_address, len + 1);
	return DOCA_SUCCESS;
}

doca_error_t
register_create_stream_params(void)
{
	doca_error_t result;
	struct doca_argp_param *pci_param;

	result = doca_argp_param_create(&pci_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(pci_param, "p");
	doca_argp_param_set_long_name(pci_param, "pci-addr");
	doca_argp_param_set_description(pci_param, "DOCA device PCI address");
	doca_argp_param_set_callback(pci_param, pci_address_callback);
	doca_argp_param_set_type(pci_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(pci_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

/*
 * Free callback - free doca_buf allocated pointer
 *
 * @addr [in]: Memory range pointer
 * @len [in]: Memory range length
 * @opaque [in]: An opaque pointer passed to iterator
 */
static void
free_callback(void *addr, size_t len, void *opaque)
{
	(void)len;
	(void)opaque;
	free(addr);
}


doca_error_t
rmax_flow_set_attributes(struct rmax_stream_config *config, struct doca_rmax_flow *flow)
{
	doca_error_t result;

	result = doca_rmax_flow_set_src_ip(flow, &config->src_ip);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_rmax_flow_set_dst_ip(flow, &config->dst_ip);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_rmax_flow_set_dst_port(flow, config->dst_port);
	if (result != DOCA_SUCCESS)
		return result;

	return result;
}

doca_error_t
rmax_stream_set_attributes(struct doca_rmax_in_stream *stream, struct rmax_stream_config *config)
{
	size_t num_buffers = (config->hdr_size > 0) ? 2 : 1;
	uint16_t pkt_size[MAX_BUFFERS];
	doca_error_t result;

	/* fill stream parameters */
	result = doca_rmax_in_stream_set_type(stream, config->type);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_rmax_in_stream_set_scatter_type(stream, config->scatter_type);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_rmax_in_stream_set_elements_count(stream, config->num_elements);
	if (result != DOCA_SUCCESS)
		return result;

	if (num_buffers == 1)
		pkt_size[0] = config->data_size;
	else {
		/* Header-Data Split mode */
		pkt_size[0] = config->hdr_size;
		pkt_size[1] = config->data_size;
	}

	result = doca_rmax_in_stream_set_memblks_count(stream, num_buffers);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_rmax_in_stream_memblk_desc_set_min_size(stream, pkt_size);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_rmax_in_stream_memblk_desc_set_max_size(stream, pkt_size);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

doca_error_t
rmax_stream_start(struct rmax_program_state *state)
{
	doca_error_t result;

	/* allow receiving rmax events using progress engine */
	result = doca_pe_connect_ctx(state->core_objects.pe, state->core_objects.ctx);
	if (result != DOCA_SUCCESS)
		return result;

	/* start the rmax context */
	result = doca_ctx_start(state->core_objects.ctx);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

doca_error_t
rmax_stream_allocate_buf(struct rmax_program_state *state, struct doca_rmax_in_stream *stream, struct rmax_stream_config *config, struct doca_buf **buffer, uint16_t *stride_size)
{
	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t num_buffers = (config->hdr_size > 0) ? 2 : 1;
	size_t size[MAX_BUFFERS] = {0, 0};
	char *ptr_memory = NULL;
	void *ptr[MAX_BUFFERS];
	doca_error_t result;

	/* query buffer size */
	result = doca_rmax_in_stream_get_memblk_size(stream, size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to get memory block size: %s", doca_error_get_descr(result));
		return result;
	}

	/* query stride size */
	result = doca_rmax_in_stream_get_memblk_stride_size(stream, stride_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to get memory block stride size: %s", doca_error_get_descr(result));
		return result;
	}

	/* allocate memory */
	ptr_memory = aligned_alloc(page_size, size[0] + size[1]);
	if (ptr_memory == NULL)
		return DOCA_ERROR_NO_MEMORY;

	result = doca_mmap_set_memrange(state->core_objects.src_mmap, ptr_memory, size[0] + size[1]);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set mmap memory range, ptr %p, size %zu: %s", ptr_memory, size[0] + size[1], doca_error_get_descr(result));
		return result;
	}

	result = doca_mmap_set_free_cb(state->core_objects.src_mmap, free_callback, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set mmap free callback: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_mmap_start(state->core_objects.src_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start mmap: %s", doca_error_get_descr(result));
		return result;
	}

	if (num_buffers == 1) {
		ptr[0] = ptr_memory;
	} else {
		ptr[0] = ptr_memory;		/* header */
		ptr[1] = ptr_memory + size[0];	/* data */
	}

	/* build memory buffer chain */
	for (size_t i = 0; i < num_buffers; ++i) {
		struct doca_buf *buf;

		result = doca_buf_inventory_buf_get_by_addr(state->core_objects.buf_inv,
				state->core_objects.src_mmap, ptr[i], size[i], &buf);
		if (result != DOCA_SUCCESS)
			return result;
		if (i == 0)
			*buffer = buf;
		else {
			/* chain buffers */
			result = doca_buf_chain_list(*buffer, buf);
			if (result != DOCA_SUCCESS)
				return result;
		}
	}

	/* set memory buffer(s) */
	result = doca_rmax_in_stream_set_memblk(stream, *buffer);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set stream memory block(s): %s", doca_error_get_descr(result));
		return result;
	}

	return result;
}

void
rmax_create_stream_cleanup(struct rmax_program_state *state, struct doca_rmax_in_stream *stream, struct doca_rmax_flow *flow, struct doca_buf *buf)
{
	doca_error_t result;

	if (buf != NULL) {
		result = doca_buf_dec_refcount(buf, NULL);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_WARN("Failed to remove buffers: %s", doca_error_get_descr(result));
	}

	result = doca_rmax_flow_destroy(flow);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy DOCA Rmax flow: %s", doca_error_get_descr(result));

	if (state->core_objects.ctx != NULL) {
		result = doca_ctx_stop(state->core_objects.ctx);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("Failed to stop DOCA Rmax in stream context: %s", doca_error_get_descr(result));
	}

	/* destroy stream */
	result = doca_rmax_in_stream_destroy(stream);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy the stream: %s", doca_error_get_descr(result));

	result = destroy_core_objects(&state->core_objects);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy DOCA core related objects: %s", doca_error_get_descr(result));

	result = doca_rmax_release();
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy the DOCA Rivermax: %s", doca_error_get_descr(result));
}
