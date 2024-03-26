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

#include <doca_error.h>
#include <doca_log.h>
#include <doca_buf_inventory.h>
#include <doca_buf.h>
#include <doca_argp.h>

#include "rdma_common.h"

#define MAX_BUFF_SIZE			(256)		/* Maximum DOCA buffer size */
#define EXPECTED_IMMEDIATE_VALUE	(0xABCD)	/* Expected immediate value to receive */

DOCA_LOG_REGISTER(RDMA_RECEIVE_IMMEDIATE::SAMPLE);

/*
 * Write the connection details for the sender to read, and read the connection details of the sender
 *
 * @cfg [in]: Configuration parameters
 * @resources [in/out]: RDMA resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
write_read_connection(struct rdma_config *cfg, struct rdma_resources *resources)
{
	int enter = 0;
	doca_error_t result = DOCA_SUCCESS;

	/* Write the RDMA connection details */
	result = write_file(cfg->local_connection_desc_path, (char *)resources->rdma_conn_descriptor,
				resources->rdma_conn_descriptor_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to write the RDMA connection details: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_INFO("You can now copy %s to the sender", cfg->local_connection_desc_path);
	DOCA_LOG_INFO("Please copy %s from the sender and then press enter", cfg->remote_connection_desc_path);

	/* Wait for enter */
	while (enter != '\r' && enter != '\n')
		enter = getchar();

	/* Read the remote RDMA connection details */
	result = read_file(cfg->remote_connection_desc_path, (char **)&resources->remote_rdma_conn_descriptor,
				&resources->remote_rdma_conn_descriptor_size);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to read the remote RDMA connection details: %s", doca_error_get_descr(result));

	return result;
}

/*
 * RDMA receive task completed callback
 *
 * @rdma_receive_task [in]: Completed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void
rdma_receive_completed_callback(struct doca_rdma_task_receive *rdma_receive_task,
				union doca_data task_user_data, union doca_data ctx_user_data)
{
	struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
	void *dst_buf_data;
	doca_be32_t immediate_data;
	enum doca_rdma_opcode op_code;
	doca_error_t *first_encountered_error = (doca_error_t *)task_user_data.ptr;
	doca_error_t result = DOCA_SUCCESS, tmp_result;

	/*
	 * Retrieve immediate data.
	 * Immediate data is only valid if the task result is success and the task contains the correct opcode
	 */
	op_code = doca_rdma_task_receive_get_result_opcode(rdma_receive_task);
	if (op_code != DOCA_RDMA_OPCODE_RECV_SEND_WITH_IMM) {
		result = DOCA_ERROR_UNEXPECTED;
		DOCA_LOG_ERR("Got incorrect opcode: %s", doca_error_get_descr(result));
		goto free_task;
	}

	immediate_data = doca_rdma_task_receive_get_result_immediate_data(rdma_receive_task);
	if (immediate_data != EXPECTED_IMMEDIATE_VALUE) {
		result = DOCA_ERROR_UNEXPECTED;
		DOCA_LOG_ERR("RDMA receive task failed: immediate value that was received isn't the expected immediate value");
		goto free_task;
	}

	DOCA_LOG_INFO("RDMA receive task was done Successfully");

	/* Read the data that was received */
	result = doca_buf_get_data(resources->dst_buf, &dst_buf_data);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to get destination buffer data: %s", doca_error_get_descr(result));
		goto free_task;
	}

	/* Check if dst_buf_data is null terminated and of legal size */
	if (strnlen(dst_buf_data, MAX_BUFF_SIZE) == MAX_BUFF_SIZE) {
		DOCA_LOG_ERR("The message that was received from sender exceeds buffer size %d", MAX_BUFF_SIZE);
		result = DOCA_ERROR_INVALID_VALUE;
		goto free_task;
	}

	DOCA_LOG_INFO("Got from sender: \"%s\" with immediate: %u", (char *)dst_buf_data, immediate_data);

free_task:
	doca_task_free(doca_rdma_task_receive_as_task(rdma_receive_task));
	tmp_result = doca_buf_dec_refcount(resources->dst_buf, NULL);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to decrease dst_buf count: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}

	/* Update that an error was encountered, if any */
	DOCA_ERROR_PROPAGATE(*first_encountered_error, tmp_result);

	resources->num_remaining_tasks--;
	/* Stop context once all tasks are completed */
	if (resources->num_remaining_tasks == 0)
		(void)doca_ctx_stop(resources->rdma_ctx);
}

/*
 * RDMA receive task error callback
 *
 * @rdma_receive_task [in]: failed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void
rdma_receive_error_callback(struct doca_rdma_task_receive *rdma_receive_task,
			    union doca_data task_user_data, union doca_data ctx_user_data)
{
	struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
	struct doca_task *task = doca_rdma_task_receive_as_task(rdma_receive_task);
	doca_error_t *first_encountered_error = (doca_error_t *)task_user_data.ptr;
	doca_error_t result;

	/* Update that an error was encountered */
	result = doca_task_get_status(task);
	DOCA_ERROR_PROPAGATE(*first_encountered_error, result);
	DOCA_LOG_ERR("RDMA receive task failed: %s", doca_error_get_descr(result));

	doca_task_free(task);
	result = doca_buf_dec_refcount(resources->dst_buf, NULL);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to decrease dst_buf count: %s", doca_error_get_descr(result));

	resources->num_remaining_tasks--;
	/* Stop context once all tasks are completed */
	if (resources->num_remaining_tasks == 0)
		(void)doca_ctx_stop(resources->rdma_ctx);
}

/*
 * Export and receive connection details, and connect to the remote RDMA
 *
 * @resources [in]: RDMA resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
rdma_receive_immediate_export_and_connect(struct rdma_resources *resources)
{
	doca_error_t result;

	/* Export RDMA connection details */
	result = doca_rdma_export(resources->rdma, &(resources->rdma_conn_descriptor),
					&(resources->rdma_conn_descriptor_size));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to export RDMA: %s", doca_error_get_descr(result));
		return result;
	}

	/* write and read connection details to the sender */
	result = write_read_connection(resources->cfg, resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to write and read connection details from sender: %s",
				doca_error_get_descr(result));
		return result;
	}

	/* Connect RDMA */
	result = doca_rdma_connect(resources->rdma, resources->remote_rdma_conn_descriptor,
					resources->remote_rdma_conn_descriptor_size);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to connect the receiver's RDMA to the sender's RDMA: %s",
				doca_error_get_descr(result));

	return result;
}

/*
 * Prepare and submit RDMA receive task
 *
 * @resources [in]: RDMA resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
rdma_receive_prepare_and_submit_task(struct rdma_resources *resources)
{
	struct doca_rdma_task_receive *rdma_receive_task = NULL;
	union doca_data task_user_data = {0};
	doca_error_t result, tmp_result;

	/* Add dst buffer to DOCA buffer inventory */
	result = doca_buf_inventory_buf_get_by_addr(resources->buf_inventory, resources->mmap, resources->mmap_memrange,
						MAX_BUFF_SIZE, &resources->dst_buf);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate DOCA buffer to DOCA buffer inventory: %s",
				doca_error_get_descr(result));
		return result;
	}

	/* Include first_encountered_error in user data of task to be used in the callbacks */
	task_user_data.ptr = &(resources->first_encountered_error);
	/* Allocate and construct RDMA receive task */
	result = doca_rdma_task_receive_allocate_init(resources->rdma, resources->dst_buf, task_user_data,
								&rdma_receive_task);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate RDMA receive task: %s", doca_error_get_descr(result));
		goto destroy_dst_buf;
	}

	/* Submit RDMA receive task */
	DOCA_LOG_INFO("Submitting RDMA receive task");
	resources->num_remaining_tasks++;
	result = doca_task_submit(doca_rdma_task_receive_as_task(rdma_receive_task));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to submit RDMA receive task: %s", doca_error_get_descr(result));
		goto free_task;
	}

	return result;

free_task:
	doca_task_free(doca_rdma_task_receive_as_task(rdma_receive_task));
destroy_dst_buf:
	tmp_result = doca_buf_dec_refcount(resources->dst_buf, NULL);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to decrease dst_buf count: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
	return result;
}

/*
 * RDMA receive state change callback
 * This function represents the state machine for this RDMA program
 *
 * @user_data [in]: doca_data from the context
 * @ctx [in]: DOCA context
 * @prev_state [in]: Previous DOCA context state
 * @next_state [in]: Next DOCA context state
 */
static void
rdma_receive_state_change_callback(const union doca_data user_data, struct doca_ctx *ctx,
				   enum doca_ctx_states prev_state, enum doca_ctx_states next_state)
{
	struct rdma_resources *resources = (struct rdma_resources *)user_data.ptr;
	doca_error_t result = DOCA_SUCCESS;

	(void)prev_state;
	(void)ctx;

	switch (next_state) {
	case DOCA_CTX_STATE_STARTING:
		DOCA_LOG_INFO("RDMA context entered starting state");

		result = rdma_receive_immediate_export_and_connect(resources);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("rdma_receive_immediate_export_and_connect() failed: %s", doca_error_get_descr(result));
		else
			DOCA_LOG_INFO("RDMA context finished initialization");
		break;
	case DOCA_CTX_STATE_RUNNING:
		DOCA_LOG_INFO("RDMA context is running");

		result = rdma_receive_prepare_and_submit_task(resources);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("rdma_receive_prepare_and_submit_task() failed: %s", doca_error_get_descr(result));
		break;
	case DOCA_CTX_STATE_STOPPING:
		/*
		 * The context is in stopping due to failure encountered in one of the tasks, nothing to do at this stage.
		 * doca_pe_progress() will cause all tasks to be flushed, and finally transition state to idle
		 */
		DOCA_LOG_ERR("RDMA context entered stopping state. All inflight tasks will be flushed");
		break;
	case DOCA_CTX_STATE_IDLE:
		DOCA_LOG_INFO("RDMA context has been stopped");

		/* We can stop the main loop */
		resources->run_main_loop = false;
		break;
	default:
		break;
	}

	/* If something failed - update that an error was encountered and stop the ctx */
	if (result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(resources->first_encountered_error, result);
		(void)doca_ctx_stop(ctx);
	}
}

/*
 * Receive a message from the sender with immediate
 *
 * @cfg [in]: Configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t
rdma_receive_immediate(struct rdma_config *cfg)
{
	struct rdma_resources resources = {0};
	union doca_data ctx_user_data = {0};
	uint32_t mmap_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE;
	uint32_t rdma_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};
	doca_error_t result, tmp_result;

	/* Allocating resources */
	result = allocate_rdma_resources(cfg, mmap_permissions, rdma_permissions,
					 doca_rdma_cap_task_receive_is_supported, &resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate RDMA Resources: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_rdma_task_receive_set_conf(resources.rdma,
						 rdma_receive_completed_callback,
						 rdma_receive_error_callback,
						 NUM_RDMA_TASKS);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to set configurations for RDMA receive task: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	result = doca_ctx_set_state_changed_cb(resources.rdma_ctx, rdma_receive_state_change_callback);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to set state change callback for RDMA context: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	/* Include the program's resources in user data of context to be used in callbacks */
	ctx_user_data.ptr = &(resources);
	result = doca_ctx_set_user_data(resources.rdma_ctx, ctx_user_data);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set context user data: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	/* Create DOCA buffer inventory */
	result = doca_buf_inventory_create(INVENTORY_NUM_INITIAL_ELEMENTS, &resources.buf_inventory);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA buffer inventory: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	/* Start DOCA buffer inventory */
	result = doca_buf_inventory_start(resources.buf_inventory);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start DOCA buffer inventory: %s", doca_error_get_descr(result));
		goto destroy_buf_inventory;
	}

	/* Start RDMA context */
	result = doca_ctx_start(resources.rdma_ctx);
	/* DOCA_ERROR_IN_PROGRESS is expected and handled by the state change callback function */
	if (result != DOCA_ERROR_IN_PROGRESS) {
		DOCA_LOG_ERR("Failed to start RDMA context: %s", doca_error_get_descr(result));
		goto stop_buf_inventory;
	}

	/*
	 * Run the progress engine which will run the state machine defined in rdma_receive_state_change_callback()
	 * When the context moves to idle, the context change callback call will signal to stop running the progress engine.
	 */
	while (resources.run_main_loop) {
		if (doca_pe_progress(resources.pe) == 0)
			nanosleep(&ts, &ts);
	}

	/* Assign the result we update in the callbacks */
	result = resources.first_encountered_error;

stop_buf_inventory:
	tmp_result = doca_buf_inventory_stop(resources.buf_inventory);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to stop DOCA buffer inventory: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
destroy_buf_inventory:
	tmp_result = doca_buf_inventory_destroy(resources.buf_inventory);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy DOCA buffer inventory: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
destroy_resources:
	tmp_result = destroy_rdma_resources(&resources, cfg);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy DOCA RDMA resources: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
	return result;
}
