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

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <netinet/in.h>

#include "rmax_common.h"

DOCA_LOG_REGISTER(RMAX_CREATE_STREAM);

/*
 * Handle a successful Rx data event
 *
 * @event_rx_data [in]: the event that occured. Holds completion data.
 * @event_user_data [in]: user defined data that was previously provided on registeration. Holds rmax_program_state.
 */
void
rx_success_cb(struct doca_rmax_in_stream_event_rx_data *event_rx_data, union doca_data event_user_data)
{
	struct doca_rmax_in_stream_completion *comp = doca_rmax_in_stream_event_rx_data_get_completion(event_rx_data);

	if (comp->elements_count == 0)
		return;

	DOCA_LOG_INFO("Received %4d packet(s), first %lu last %lu",
		      comp->elements_count, comp->ts_first, comp->ts_last);

	struct rmax_program_state *state = (struct rmax_program_state *)event_user_data.ptr;

	/* dump packets */
	uint8_t *data = comp->memblk_ptr_arr[0];

	for (size_t i = 0; i < comp->elements_count; ++i, data += state->stride_size[0]) {
		char *dump = hex_dump(data, state->config->data_size);

		DOCA_LOG_DBG("Packet:\n%s", dump);
		free(dump);
	}
}

/*
 * Handle a failed Rx data event
 *
 * @event_rx_data [in]: the event that occured. Holds error data.
 * @event_user_data [in]: user defined data that was previously provided on registeration. Holds rmax_program_state.
 */
void
rx_error_cb(struct doca_rmax_in_stream_event_rx_data *event_rx_data, union doca_data event_user_data)
{
	struct doca_rmax_stream_error *err = doca_rmax_in_stream_event_rx_data_get_error(event_rx_data);
	struct rmax_program_state *state = (struct rmax_program_state *)event_user_data.ptr;

	DOCA_LOG_ERR("Error in Rx event: code=%d message=%s", err->code, err->message);

	state->run_main_loop = false;
	state->exit_status = DOCA_ERROR_IO_FAILED;
}

/*
 * Run create_stream sample, which creates stream
 *
 * @state [in]: DOCA core related objects
 * @stream_config [in]: stream configurations
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
rmax_create_stream(struct rmax_program_state *state, struct rmax_stream_config *stream_config)
{
	struct doca_rmax_in_stream *stream = NULL;
	struct doca_buf *buf = NULL;
	struct doca_rmax_flow *flow = NULL;
	doca_error_t result;

	memset(state, 0, sizeof(*state));
	state->config = stream_config;

	/* open DOCA device with the given PCI address */
	result = open_doca_device_with_pci(stream_config->pci_address, NULL, &state->core_objects.dev);
	if (result != DOCA_SUCCESS)
		return result;

	/* DOCA RMAX library Initialization */
	result = doca_rmax_init();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to initialize DOCA RMAX library: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* create core DOCA objects */
	result = create_core_objects(&state->core_objects, 1);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA core related objects: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* create stream object */
	result = doca_rmax_in_stream_create(state->core_objects.dev, &stream);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA Rivermax stream: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* set stream attributes, based on received or default configurations */
	result = rmax_stream_set_attributes(stream, stream_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set stream attributes: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* Register Rx data event handlers */
	union doca_data event_user_data;

	event_user_data.ptr = (void *)state;
	result = doca_rmax_in_stream_event_rx_data_register(stream, event_user_data, rx_success_cb, rx_error_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register to Rx data event: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* convert DOCA Rmax stream to DOCA context */
	state->core_objects.ctx = doca_rmax_in_stream_as_ctx(stream);
	if (state->core_objects.ctx == NULL) {
		DOCA_LOG_ERR("Failed to convert stream to a context");
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* allocate buffers for received data */
	result = rmax_stream_allocate_buf(state, stream, stream_config, &buf, state->stride_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate RX buffers: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* connect the rivermax context to a progress engine and start it */
	result = rmax_stream_start(state);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start rmax context: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* create a flow using DOCA Rmax API */
	result = doca_rmax_flow_create(&flow);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create a flow: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* create a DOCA Rmax flow and attach it to the given stream  */
	result = rmax_flow_set_attributes(stream_config, flow);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set flow attributes: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	/* Attach flow to a stream using DOCA Rmax API */
	result = doca_rmax_flow_attach(flow, stream);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to attach a flow: %s", doca_error_get_descr(result));
		rmax_create_stream_cleanup(state, stream, flow, buf);
		return result;
	}

	state->run_main_loop = true;
	state->exit_status = DOCA_SUCCESS;

	while (state->run_main_loop)
		(void)doca_pe_progress(state->core_objects.pe);

	if (state->exit_status != DOCA_SUCCESS)
		DOCA_LOG_ERR("Program exiting with failure to receive data. err=%s",
			     doca_error_get_name(state->exit_status));

	/* detach flow */
	result = doca_rmax_flow_detach(flow, stream);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to detach flow from stream: %s", doca_error_get_descr(result));

	/* Clean and destroy all relevant objects */
	rmax_create_stream_cleanup(state, stream, flow, buf);

	return DOCA_SUCCESS;
}
