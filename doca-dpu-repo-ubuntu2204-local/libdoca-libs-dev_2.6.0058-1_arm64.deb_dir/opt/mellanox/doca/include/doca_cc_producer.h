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

/**
 * @file doca_cc_producer.h
 * @page comm_channel_v2
 * @defgroup DOCA_CC Comm Channel
 *
 * DOCA Communication Channel Producer offers an extension the doca_cc channel for accelerated data transfer between
 * memory on the host and DPU in a FIFO format. An established doca_cc connection is required to negotiate the end
 * points of the FIFO. A producer object can populate buffers advertised by any consumers associated with the same
 * doca_cc connection. The inter-process communication runs over DMA/PCIe and does not affect network bandwidth.
 *
 * @{
 */
#ifndef DOCA_CC_PRODUCER_H_
#define DOCA_CC_PRODUCER_H_

#include <stddef.h>
#include <stdint.h>

#include <doca_compat.h>
#include <doca_error.h>
#include <doca_types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct doca_buf;
struct doca_dev;
struct doca_devinfo;

/* Representantion of a comms channel point to point connection */
struct doca_cc_connection;

/* Instance of a doca_cc producer */
struct doca_cc_producer;

/*********************************************************************************************************************
 * Producer Creation
 *********************************************************************************************************************/

/**
 * Create a DOCA CC producer instance.
 *
 * @param [in] cc_connection
 * An established control channel connection to associate producer with.
 * @param [out] producer
 * Pointer to pointer to be set to created doca_cc_producer instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - input parameter is a NULL pointer.
 * - DOCA_ERROR_NO_MEMORY - failed to alloc doca_cc_producer.
 * - DOCA_ERROR_BAD_STATE - cc_connection is not established.
 * - DOCA_ERROR_NOT_PERMITTED - incompatible version of cc_connection.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_create(struct doca_cc_connection *cc_connection, struct doca_cc_producer **producer);

/**
 * Destroy a DOCA CC producer instance.
 *
 * @param [in] producer
 * Pointer to doca_cc_producer instance to destroy.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - producer argument is a NULL pointer.
 * - DOCA_ERROR_INITIALIZATION - failed to initialise a mutex.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_destroy(struct doca_cc_producer *producer);

/**
 * Check if given device is capable of running a producer.
 *
 * @param [in] devinfo
 * The DOCA device information.
 *
 * @return
 * DOCA_SUCCESS - in case device can implement a producer.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - provided devinfo can not implement a producer.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_cap_is_supported(const struct doca_devinfo *devinfo);

/**
 * Get the id the doca_cc_producer instance.
 *
 * @param [in] producer
 * The doca_cc_producer instance.
 * @param [out] id
 * Per cc_connection unique id associated with the producer instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_get_id(const struct doca_cc_producer *producer, uint32_t *id);

/**
 * Get the max number of tasks supported by the device for a doca_cc_producer instance.
 *
 * @param [in] devinfo
 * Devinfo to query the capability for.
 * @param [out] max_num_tasks
 * The maximum number of tasks that can allocated by the instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_cap_get_max_num_tasks(const struct doca_devinfo *devinfo, uint32_t *max_num_tasks);

/**
 * Get the max size doca_buf that can be sent by a doca_cc_producer instance.
 *
 * @param [in] devinfo
 * Devinfo to query the capability for.
 * @param [out] max_buf_size
 * Maximum sized buffer that can be sent by the producer.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - if producer is not supported on device.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_cap_get_max_buf_size(const struct doca_devinfo *devinfo, uint32_t *max_buf_size);

/**
 * Get the max number of producers that can be associated with a doca_cc_connection.
 *
 * @param [in] devinfo
 * Devinfo to query the capability for.
 * @param [out] max_producers
 * Maximum number of producers that can be added to a doca_cc_connection.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_cap_get_max_producers(const struct doca_devinfo *devinfo, uint32_t *max_producers);


/**
 * Convert doca_cc_producer instance into a generalised context for use with doca core objects.
 *
 * @param [in] producer
 * Doca_cc_producer instance. This must remain valid until after the context is no longer required.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_ctx *doca_cc_producer_as_ctx(struct doca_cc_producer *producer);

/*********************************************************************************************************************
 * Producer Sent Task
 *********************************************************************************************************************/

/* Task instance to for producer buffer send */
struct doca_cc_producer_task_send;

/**
 * Function executed on doca_cc_producer send task completion. Used for both task success and failure.
 *
 * @param [in] task
 * Doca producer send task that has completed.
 * @param [in] task_user_data
 * The task user data.
 * @param [in] ctx_user_data
 * Doca_cc context user data.
 *
 * The implementation can assume this value is not NULL.
 */
typedef void (*doca_cc_producer_task_send_completion_cb_t)(struct doca_cc_producer_task_send *task,
							   union doca_data task_user_data,
							   union doca_data ctx_user_data);

/**
 * Configure the doca_cc_producer send task callback and parameters.
 *
 * @param [in] producer
 * The doca_cc_producer instance.
 * @param [in] task_completion_cb
 * Send task completion callback.
 * @param [in] task_error_cb
 * Send task error callback.
 * @param [in] num_send_tasks
 * Number of send tasks a producer can allocate.
 * Must not exceed value returned by doca_cc_producer_cap_get_max_num_tasks()
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - producer instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_task_send_set_conf(struct doca_cc_producer *producer,
						 doca_cc_producer_task_send_completion_cb_t task_completion_cb,
						 doca_cc_producer_task_send_completion_cb_t task_error_cb,
						 uint32_t num_send_tasks);

/**
 * Allocate and initialise a doca_producer send task.
 *
 * @param [in] producer
 * The doca_cc_producer  instance.
 * @param [in] buf
 * Doca buffer to send to a consumer.
 * @param [in] consumer_id
 * ID of consumer to send the buffer to.
 * @param [out] task
 * Pointer to a doca_cc_producer_send_task instance populated with input parameters.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_NO_MEMORY - no available tasks to allocate.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_producer_task_send_alloc_init(struct doca_cc_producer *producer, struct doca_buf *buf,
						   uint32_t consumer_id, struct doca_cc_producer_task_send **task);

/**
 * Get the doca_buf from the doca_cc_producer_send_task instance.
 *
 * @param [in] task
 * The doca_cc_producer_send_task instance.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_buf *doca_cc_producer_task_send_get_buf(struct doca_cc_producer_task_send *task);

/**
 * Set the doca_buf in a doca_cc_producer_send_task instance.
 *
 * @param [in] task
 * The doca_cc_producer_send_task instance.
 * @param [in] buf
 * Buffer to set in the task.
 */
DOCA_EXPERIMENTAL
void doca_cc_producer_task_send_set_buf(struct doca_cc_producer_task_send *task, struct doca_buf *buf);

/**
 * Get the consumer id from the doca_cc_producer_send_task instance.
 *
 * @param [in] task
 * The doca_cc_producer_send_task instance.
 *
 * @return
 * Consumer_id
 */
DOCA_EXPERIMENTAL
uint32_t doca_cc_producer_task_send_get_consumer_id(struct doca_cc_producer_task_send *task);

/**
 * Set the consumer id in the doca_cc_producer_send_task instance.
 *
 * @param [in] task
 * The doca_cc_producer_send_task instance.
 * @param [in] consumer_id
 * Consumer id to set in the task;
 */
DOCA_EXPERIMENTAL
void doca_cc_producer_task_send_set_consumer_id(struct doca_cc_producer_task_send *task, const uint32_t consumer_id);

/**
 * Convert doca_cc_producer_send_task instance into a generalised task for use with progress engine.
 *
 * @param [in] task
 * Doca_cc_producer_send_task instance.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_cc_producer_task_send_as_task(struct doca_cc_producer_task_send *task);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* DOCA_CC_PRODUCER_H_ */
