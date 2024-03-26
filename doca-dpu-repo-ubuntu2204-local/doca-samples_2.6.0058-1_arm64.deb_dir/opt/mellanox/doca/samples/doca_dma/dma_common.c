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

#include <string.h>
#include <unistd.h>

#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_dma.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_mmap.h>
#include <doca_argp.h>

#include "dma_common.h"

DOCA_LOG_REGISTER(DMA_COMMON);

/*
 * ARGP Callback - Handle PCI device address parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
pci_callback(void *param, void *config)
{
	struct dma_config *conf = (struct dma_config *)config;
	const char *addr = (char *)param;
	int addr_len = strnlen(addr, DOCA_DEVINFO_PCI_ADDR_SIZE);

	/* Check using >= to make static code analysis satisfied */
	if (addr_len >= DOCA_DEVINFO_PCI_ADDR_SIZE) {
		DOCA_LOG_ERR("Entered device PCI address exceeding the maximum size of %d", DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(conf->pci_address, addr, addr_len + 1);

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle text to copy parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
text_callback(void *param, void *config)
{
	struct dma_config *conf = (struct dma_config *)config;
	const char *txt = (char *)param;
	int txt_len = strnlen(txt, MAX_TXT_SIZE);

	/* Check using >= to make static code analysis satisfied */
	if (txt_len >= MAX_TXT_SIZE) {
		DOCA_LOG_ERR("Entered text exceeded buffer size of: %d", MAX_USER_TXT_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(conf->cpy_txt, txt, txt_len + 1);

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle exported descriptor file path parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
descriptor_path_callback(void *param, void *config)
{
	struct dma_config *conf = (struct dma_config *)config;
	const char *path = (char *)param;
	int path_len = strnlen(path, MAX_ARG_SIZE);

	/* Check using >= to make static code analysis satisfied */
	if (path_len >= MAX_ARG_SIZE) {
		DOCA_LOG_ERR("Entered path exceeded buffer size: %d", MAX_USER_ARG_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

#ifdef DOCA_ARCH_DPU
	if (access(path, F_OK | R_OK) != 0) {
		DOCA_LOG_ERR("Failed to find file path pointed by export descriptor: %s", path);
		return DOCA_ERROR_INVALID_VALUE;
	}
#endif

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(conf->export_desc_path, path, path_len + 1);

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle buffer information file path parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
buf_info_path_callback(void *param, void *config)
{
	struct dma_config *conf = (struct dma_config *)config;
	const char *path = (char *)param;
	int path_len = strnlen(path, MAX_ARG_SIZE);

	/* Check using >= to make static code analysis satisfied */
	if (path_len >= MAX_ARG_SIZE) {
		DOCA_LOG_ERR("Entered path exceeded buffer size: %d", MAX_USER_ARG_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

#ifdef DOCA_ARCH_DPU
	if (access(path, F_OK | R_OK) != 0) {
		DOCA_LOG_ERR("Failed to find file path pointed by buffer information: %s", path);
		return DOCA_ERROR_INVALID_VALUE;
	}
#endif

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(conf->buf_info_path, path, path_len + 1);

	return DOCA_SUCCESS;
}

doca_error_t
register_dma_params(bool is_remote)
{
	doca_error_t result;
	struct doca_argp_param *pci_address_param, *cpy_txt_param, *export_desc_path_param, *buf_info_path_param;

	/* Create and register PCI address param */
	result = doca_argp_param_create(&pci_address_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(pci_address_param, "p");
	doca_argp_param_set_long_name(pci_address_param, "pci-addr");
	doca_argp_param_set_description(pci_address_param, "DOCA DMA device PCI address");
	doca_argp_param_set_callback(pci_address_param, pci_callback);
	doca_argp_param_set_type(pci_address_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(pci_address_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	/* Create and register text to copy param */
	result = doca_argp_param_create(&cpy_txt_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(cpy_txt_param, "t");
	doca_argp_param_set_long_name(cpy_txt_param, "text");
	doca_argp_param_set_description(cpy_txt_param,
					"Text to DMA copy from the Host to the DPU (relevant only on the Host side)");
	doca_argp_param_set_callback(cpy_txt_param, text_callback);
	doca_argp_param_set_type(cpy_txt_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(cpy_txt_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	if (is_remote) {
		/* Create and register exported descriptor file path param */
		result = doca_argp_param_create(&export_desc_path_param);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
			return result;
		}
		doca_argp_param_set_short_name(export_desc_path_param, "d");
		doca_argp_param_set_long_name(export_desc_path_param, "descriptor-path");
		doca_argp_param_set_description(export_desc_path_param,
						"Exported descriptor file path to save (Host) or to read from (DPU)");
		doca_argp_param_set_callback(export_desc_path_param, descriptor_path_callback);
		doca_argp_param_set_type(export_desc_path_param, DOCA_ARGP_TYPE_STRING);
		result = doca_argp_register_param(export_desc_path_param);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
			return result;
		}

		/* Create and register buffer information file param */
		result = doca_argp_param_create(&buf_info_path_param);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
			return result;
		}
		doca_argp_param_set_short_name(buf_info_path_param, "b");
		doca_argp_param_set_long_name(buf_info_path_param, "buffer-path");
		doca_argp_param_set_description(buf_info_path_param,
						"Buffer information file path to save (Host) or to read from (DPU)");
		doca_argp_param_set_callback(buf_info_path_param, buf_info_path_callback);
		doca_argp_param_set_type(buf_info_path_param, DOCA_ARGP_TYPE_STRING);
		result = doca_argp_register_param(buf_info_path_param);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
			return result;
		}
	}

	return DOCA_SUCCESS;
}

/*
 * DMA Memcpy task completed callback
 *
 * @dma_task [in]: Completed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void
dma_memcpy_completed_callback(struct doca_dma_task_memcpy *dma_task, union doca_data task_user_data,
			      union doca_data ctx_user_data)
{
	struct dma_resources *resources = (struct dma_resources *)ctx_user_data.ptr;
	doca_error_t *result = (doca_error_t *)task_user_data.ptr;

	/* Assign success to the result */
	*result = DOCA_SUCCESS;
	DOCA_LOG_INFO("DMA task was completed successfully");

	/* Free task */
	doca_task_free(doca_dma_task_memcpy_as_task(dma_task));
	/* Decrement number of remaining tasks */
	--resources->num_remaining_tasks;
	/* Stop context once all tasks are completed */
	if (resources->num_remaining_tasks == 0)
		(void)doca_ctx_stop(resources->state.ctx);
}

/*
 * Memcpy task error callback
 *
 * @dma_task [in]: failed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void
dma_memcpy_error_callback(struct doca_dma_task_memcpy *dma_task, union doca_data task_user_data,
			  union doca_data ctx_user_data)
{
	struct dma_resources *resources = (struct dma_resources *)ctx_user_data.ptr;
	struct doca_task *task = doca_dma_task_memcpy_as_task(dma_task);
	doca_error_t *result = (doca_error_t *)task_user_data.ptr;

	/* Get the result of the task */
	*result = doca_task_get_status(task);
	DOCA_LOG_ERR("DMA task failed: %s", doca_error_get_descr(*result));

	/* Free task */
	doca_task_free(task);
	/* Decrement number of remaining tasks */
	--resources->num_remaining_tasks;
	/* Stop context once all tasks are completed */
	if (resources->num_remaining_tasks == 0)
		(void)doca_ctx_stop(resources->state.ctx);
}

/**
 * Callback triggered whenever DMA context state changes
 *
 * @user_data [in]: User data associated with the DMA context. Will hold struct dma_resources *
 * @ctx [in]: The DMA context that had a state change
 * @prev_state [in]: Previous context state
 * @next_state [in]: Next context state (context is already in this state when the callback is called)
 */
static void
dma_state_changed_callback(const union doca_data user_data, struct doca_ctx *ctx, enum doca_ctx_states prev_state,
				enum doca_ctx_states next_state)
{
	(void)ctx;
	(void)prev_state;

	struct dma_resources *resources = (struct dma_resources *)user_data.ptr;

	switch (next_state) {
	case DOCA_CTX_STATE_IDLE:
		DOCA_LOG_INFO("DMA context has been stopped");
		/* We can stop the main loop */
		resources->run_main_loop = false;
		break;
	case DOCA_CTX_STATE_STARTING:
		/**
		 * The context is in starting state, this is unexpected for DMA.
		 */
		DOCA_LOG_ERR("DMA context entered into starting state. Unexpected transition");
		break;
	case DOCA_CTX_STATE_RUNNING:
		DOCA_LOG_INFO("DMA context is running");
		break;
	case DOCA_CTX_STATE_STOPPING:
		/**
		 * The context is in stopping due to failure encountered in one of the tasks, nothing to do at this stage.
		 * doca_pe_progress() will cause all tasks to be flushed, and finally transition state to idle
		 */
		DOCA_LOG_ERR("DMA context entered into stopping state. All inflight tasks will be flushed");
		break;
	default:
		break;
	}
}

doca_error_t
allocate_dma_resources(const char *pcie_addr, struct dma_resources *resources)
{
	memset(resources, 0, sizeof(*resources));
	/* Two buffers for source and destination */
	uint32_t max_bufs = 2;
	union doca_data ctx_user_data = {0};
	struct program_core_objects *state = &resources->state;
	doca_error_t result, tmp_result;

	result = open_doca_device_with_pci(pcie_addr, &dma_task_is_supported, &state->dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to open DOCA device for DMA: %s", doca_error_get_descr(result));
		return result;
	}

	result = create_core_objects(state, max_bufs);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA core objects: %s", doca_error_get_descr(result));
		goto close_device;
	}

	result = doca_dma_create(state->dev, &resources->dma_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DMA context: %s", doca_error_get_descr(result));
		goto destroy_core_objects;
	}

	state->ctx = doca_dma_as_ctx(resources->dma_ctx);

	result = doca_ctx_set_state_changed_cb(state->ctx, dma_state_changed_callback);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to set DMA state change callback: %s", doca_error_get_descr(result));
		goto destroy_dma;
	}

	result = doca_dma_task_memcpy_set_conf(resources->dma_ctx, dma_memcpy_completed_callback, dma_memcpy_error_callback,
					       NUM_DMA_TASKS);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set configurations for DMA memcpy task: %s", doca_error_get_descr(result));
		goto destroy_dma;
	}

	/* Include resources in user data of context to be used in callbacks */
	ctx_user_data.ptr = resources;
	doca_ctx_set_user_data(state->ctx, ctx_user_data);

	return result;

destroy_dma:
	tmp_result = doca_dma_destroy(resources->dma_ctx);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to destroy DOCA DMA context: %s", doca_error_get_descr(tmp_result));
	}
destroy_core_objects:
	tmp_result = destroy_core_objects(state);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to destroy DOCA core objects: %s", doca_error_get_descr(tmp_result));
	}
close_device:
	tmp_result = doca_dev_close(state->dev);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to close DOCA device: %s", doca_error_get_descr(tmp_result));
	}

	return result;
}

doca_error_t
destroy_dma_resources(struct dma_resources *resources)
{
	doca_error_t result, tmp_result;

	result = doca_dma_destroy(resources->dma_ctx);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy DOCA DMA context: %s", doca_error_get_descr(result));

	tmp_result = destroy_core_objects(&resources->state);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to destroy DOCA core objects: %s", doca_error_get_descr(tmp_result));
	}

	tmp_result = doca_dev_close(resources->state.dev);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to close DOCA device: %s", doca_error_get_descr(tmp_result));
	}

	return result;
}

doca_error_t
allocate_dma_host_resources(const char *pcie_addr, struct program_core_objects *state)
{
	doca_error_t result, tmp_result;

	result = open_doca_device_with_pci(pcie_addr, &dma_task_is_supported, &state->dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to open DOCA device for DMA: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_mmap_create(&state->src_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
		goto close_device;
	}

	result = doca_mmap_add_dev(state->src_mmap, state->dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add device to mmap: %s", doca_error_get_descr(result));
		goto destroy_mmap;
	}

	return result;

destroy_mmap:
	tmp_result = doca_mmap_destroy(state->src_mmap);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to destroy DOCA mmap: %s", doca_error_get_descr(tmp_result));
	}
close_device:
	tmp_result = doca_dev_close(state->dev);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to close DOCA device: %s", doca_error_get_descr(tmp_result));
	}

	return result;
}

doca_error_t
destroy_dma_host_resources(struct program_core_objects *state)
{
	doca_error_t result, tmp_result;

	result = doca_mmap_destroy(state->src_mmap);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to destroy DOCA mmap: %s", doca_error_get_descr(result));

	tmp_result = doca_dev_close(state->dev);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(result, tmp_result);
		DOCA_LOG_ERR("Failed to close DOCA device: %s", doca_error_get_descr(tmp_result));
	}

	return result;
}

doca_error_t
dma_task_is_supported(struct doca_devinfo *devinfo)
{
	return doca_dma_cap_task_memcpy_is_supported(devinfo);
}
