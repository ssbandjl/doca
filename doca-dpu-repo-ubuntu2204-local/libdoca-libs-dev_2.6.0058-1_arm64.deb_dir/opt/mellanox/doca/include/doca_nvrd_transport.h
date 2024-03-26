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

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <doca_ctx.h>
#include <doca_error.h>
#include <doca_rdma.h>
#include <doca_pe.h>

#ifdef __cplusplus
extern "C" {
#endif

/** NVRD address handle. Might be changed in the future. */
struct doca_nvrd_transport_address_handle {
	uint64_t id;	/**< Handle identifier */
};

/** NVRD transport RDMA properties */
struct doca_nvrd_transport_rdma_properties {
	uint32_t send_queue_size;			/**< Send queue size */
	uint32_t recv_queue_size;			/**< Receive queue size */
	uint32_t max_send_buf_list_len;			/**< Max send buffer list length */
	enum doca_rdma_transport_type transport_type;	/**< RDMA transport type */
	enum doca_mtu_size mtu_size;			/**< MTU size */
	uint32_t permissions;				/**< RDMA permissions. See doca_access_flag */
	uint32_t recv_buf_list_len;			/**< Receive buffer list length */
	uint16_t gid_index;				/**< GID index */
	uint8_t sl;					/**< Service level */
	uint8_t grh_enabled;				/**< Global routing header enabled. Always enabled for RoCE */
	uint8_t is_mtu_adaptive;			/**< If 1, MTU size will be dictated by the minimal MTU
							  *  between the two peers.
							  *  If 0, MTU was set by the user and may not be changed */
};

/** Opaque structure representing DOCA NVRD transport instance. */
struct doca_nvrd_transport;

/**
 * @brief Creates a DOCA NVRD transport instance.
 *
 * @param [in] gvmi
 * GVMI which will be presented by NVRD transport instance
 * @param [out] nvrd_transport_pptr
 * Pointer to pointer to be set to point to the created doca_nvrd_transport instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - nvrd_transport_pptr argument is a NULL pointer.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 * - DOCA_ERROR_INITIALIZATION - failed to initialize NVRD transport.
 */
DOCA_EXPERIMENTAL doca_error_t
doca_nvrd_transport_create(uint16_t gvmi, struct doca_nvrd_transport **nvrd_transport_pptr);

/**
 * @brief Sets global configuration for NVRD transport
 *
 * @param [in] nvrd_transport_ptr
 * Pointer to doca_nvrd_transport instance.
 * @param [in] gid_index
 * Global GID index to be used as default for NVRD remote instances
 * @param [in] dev_ptr
 * Global DOCA device to be used as default for NVRD remote instances
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - nvrd_transport_ptr argument is a NULL pointer.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 * - DOCA_ERROR_INITIALIZATION - failed to initialize NVRD transport.
 */
DOCA_EXPERIMENTAL doca_error_t
doca_nvrd_transport_set_global_configs(struct doca_nvrd_transport *nvrd_transport_ptr, uint16_t gid_index,
				       struct doca_dev *dev_ptr);


DOCA_EXPERIMENTAL doca_error_t
doca_nvrd_transport_add_remote(struct doca_nvrd_transport *doca_nvrd_transport_ptr,
			       struct doca_dev *doca_dev_ptr,
			       struct doca_nvrd_transport_address_handle *ah_ptr,
			       size_t num_of_qp_connections,
			       const void **vrdma_blob_pptr,
			       size_t *vrdma_blob_size_ptr,
			       struct doca_nvrd_transport_rdma_properties *rmda_properties_ptr,
			       void **nvrd_remote_pptr);

DOCA_EXPERIMENTAL doca_error_t
doca_nvrd_transport_destroy(struct doca_nvrd_transport *doca_nvrd_transport_ptr);

DOCA_EXPERIMENTAL doca_error_t
doca_nvrd_transport_export_connection(struct doca_nvrd_transport *doca_nvrd_transport_ptr,
				      const void **local_connection_details_pptr,
				      size_t *local_connection_details_size_ptr);

/**
 * @brief Connect local NVRD remote object, to remote NVRD remote object
 *
 * @details This function takes, as input, local NVRD remote object, along with remote connection details
 * When this function is called the local NVRD remote is connected to the remote NVRD remote object according
 * to the remote connection details provided
 *
 * @param [in] nvrd_remote_ctx_ptr
 * NVRD remote context pointer.
 * @param [in] remote_nvrd_conn_details_ptr
 * Remote connection details pointer.
 * @param [in] remote_nvrd_conn_details_size
 * Remote connection details size in bytes.
 */
DOCA_EXPERIMENTAL doca_error_t
doca_nvrd_transport_connect_remote(void *nvrd_remote_ctx_ptr,
				   const void *remote_nvrd_conn_details_ptr,
				   size_t remote_nvrd_conn_details_size);

DOCA_EXPERIMENTAL struct doca_ctx *
doca_nvrd_transport_as_ctx(struct doca_nvrd_transport *nvrd_transport_ptr);

/********************************************
 * DOCA NVRD transport task - write	    *
 ********************************************/

/**
 * @brief This task writes data to peer remote memory.
 */
struct doca_nvrd_transport_task_write;

/**
 * @brief Function to execute on completion of a write task.
 *
 * @details This function is called by doca_pe_progress() when a write task is successfully identified as completed.
 * When this function is called the ownership of the task object passes from DOCA back to user.
 * Inside this callback the user may decide on the task object:
 * - re-submit task with doca_task_submit(); task object ownership passed to DOCA
 * - release task with doca_task_free(); task object ownership passed to DOCA
 * - keep the task object for future re-use; user keeps the ownership on the task object
 * Inside this callback the user shouldn't call doca_pe_progress().
 * Please see doca_pe_progress() for details.
 *
 * Any failure/error inside this function should be handled internally or deferred;
 * Since this function is nested in the execution of doca_pe_progress(), this callback doesn't return an error.
 *
 * @note This callback type is utilized for both successful & failed task completions.
 *
 * @param [in] task
 * The completed write task.
 * @note The implementation can assume this value is not NULL.
 * @param [in] task_user_data
 * The doca_data supplied to the task by the application (during task allocation or by a setter).
 * @param [in] ctx_user_data
 * The doca_data supplied to the doca_ctx by the application (using a setter).
 */
typedef void (*doca_nvrd_transport_task_write_completion_cb_t)(struct doca_nvrd_transport_task_write *task,
							       union doca_data task_user_data,
							       union doca_data ctx_user_data);

/**
 * Check if a given device supports executing a write task.
 *
 * @param [in] devinfo
 * The DOCA device information that should be queried.
 *
 * @return
 * DOCA_SUCCESS - in case device supports the task.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - provided devinfo does not support the task.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_nvrd_transport_cap_task_write_is_supported(const struct doca_devinfo *devinfo);

/**
 * Get the maximal buffer list length for a source buffer of a write task, for the given devinfo and transport type.
 *
 * @param [in] devinfo
 * The DOCA device information.
 * @param [in] transport_type
 * The relevant transport type.
 * @param [out] max_buf_list_len
 * The maximal number of local buffers that can be chained with a source buffer of a write task, for the given devinfo
 * and transport type.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_DRIVER - failed to query device capabilities.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_nvrd_transport_cap_task_write_get_max_src_buf_list_len(const struct doca_devinfo *devinfo,
									 enum doca_rdma_transport_type transport_type,
									 uint32_t *max_buf_list_len);

/**
 * @brief This method sets the write tasks configuration.
 *
 * @param [in] nvrd_transport_ptr
 * The NVRD transport instance to config.
 * @param [in] task_completion_cb
 * A callback function for write tasks that were completed successfully.
 * @param [in] task_error_cb
 * A callback function for write tasks that were completed with an error.
 * @param [in] log_num_tasks
 * Log of number of write tasks.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_nvrd_transport_task_write_set_conf(struct doca_nvrd_transport *nvrd_transport_ptr,
						     doca_nvrd_transport_task_write_completion_cb_t task_completion_cb,
						     doca_nvrd_transport_task_write_completion_cb_t task_error_cb,
						     uint8_t log_num_tasks);

/**
 * @brief This method converts a write task to a doca_task.
 *
 * @param [in] task_ptr
 * The task that should be converted.
 *
 * @return
 * The write task converted to doca_task.
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_nvrd_transport_task_write_as_task(struct doca_nvrd_transport_task_write *task_ptr);

/**
 * @brief This method sets the address handle of a write task.
 *
 * @param [in] task_ptr
 * The task to set.
 * @param [in] ah_ptr
 * Address handle pointer.
 *
 */
DOCA_EXPERIMENTAL
void doca_nvrd_transport_task_write_set_address_handle(struct doca_nvrd_transport_task_write *task_ptr,
						       struct doca_nvrd_transport_address_handle *ah_ptr);

/**
 * @brief This method gets the address handle of a write task.
 *
 * @param [in] task_ptr
 * The task that should be queried.
 *
 * @return
 * The task's address handle.
 */
DOCA_EXPERIMENTAL
struct doca_nvrd_transport_address_handle *
doca_nvrd_transport_task_write_get_address_handle(struct doca_nvrd_transport_task_write *task_ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

