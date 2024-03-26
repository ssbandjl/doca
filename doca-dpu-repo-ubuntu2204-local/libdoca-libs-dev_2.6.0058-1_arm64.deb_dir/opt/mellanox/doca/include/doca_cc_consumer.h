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
 * @file doca_cc_consumer.h
 * @page comm_channel_v2
 * @defgroup DOCA_CC Comm Channel
 *
 * DOCA Communication Channel Consumer offers an extension the doca_cc channel for accelerated data transfer between
 * memory on the host and DPU in a FIFO format. An established doca_cc connection is required to negotiate the end
 * points of the FIFO. A consumer object can then post buffers to a remote process that it wishes to receive data on.
 * Completion of a consumer post receive message indicates that data has been populated from a remote producer. The
 * inter-process communication runs over DMA/PCIe and does not affect network bandwidth.
 *
 * @{
 */
#ifndef DOCA_CC_CONSUMER_H_
#define DOCA_CC_CONSUMER_H_

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
struct doca_mmap;

/* Representantion of a comms channel point to point connection */
struct doca_cc_connection;

/* Instance of a doca_cc consumer */
struct doca_cc_consumer;

/*********************************************************************************************************************
 * Consumer Creation
 *********************************************************************************************************************/

/**
 * Create a DOCA CC consumer instance.
 *
 * @param [in] cc_connection
 * An established control channel connection to create consumer across.
 * @param [in] buf_mmap
 * A registered mmap for the memory region the consumer allows buffer writes to.
 * @param [out] consumer
 * Pointer to pointer to be set to created doca_cc_consumer instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - input parameter is a NULL pointer.
 * - DOCA_ERROR_NO_MEMORY - failed to alloc doca_cc_consumer object or id.
 * - DOCA_ERROR_BAD_STATE - cc_connection is not established.
 * - DOCA_ERROR_NOT_PERMITTED - incompatable version of cc_connection.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_create(struct doca_cc_connection *cc_connection, struct doca_mmap *buf_mmap,
				     struct doca_cc_consumer **consumer);

/**
 * Destory a DOCA CC consumer instance.
 *
 * @param [in] consumer
 * Pointer to doca_cc_consumer instance to destroy.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - consumer argument is a NULL pointer.
 * - DOCA_ERROR_INITIALIZATION - failed to initialise a mutex.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_destroy(struct doca_cc_consumer *consumer);

/**
 * Check if given device is capable of running a consumer.
 *
 * @param [in] devinfo
 * The DOCA device information.
 *
 * @return
 * DOCA_SUCCESS - in case device can implement a consumer.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - provided devinfo can not implement a consumer.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_cap_is_supported(const struct doca_devinfo *devinfo);

/**
 * Get the id the doca_cc_consumer instance.
 *
 * @param [in] consumer
 * The doca_cc_consumer instance.
 * @param [out] id
 * Per cc_connection unique id associated with the consumer instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_get_id(const struct doca_cc_consumer *consumer, uint32_t *id);

/**
 * Get the max number of tasks supported by the device for a doca_cc_consumer instance.
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
doca_error_t doca_cc_consumer_cap_get_max_num_tasks(const struct doca_devinfo *devinfo, uint32_t *max_num_tasks);

/**
 * Get the max size doca_buf that can be received by a doca_cc_consumer instance.
 *
 * @param [in] devinfo
 * Devinfo to query the capability for.
 * @param [out] max_buf_size
 * Maximum sized buffer that can be received by the consumer.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - if consumer is not supported on device.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_cap_get_max_buf_size(const struct doca_devinfo *devinfo, uint32_t *max_buf_size);

/**
 * Get the max number of consumers that can be associated with a doca_cc_connection.
 *
 * @param [in] devinfo
 * Devinfo to query the capability for.
 * @param [out] max_consumers
 * Maximum number of consumers that can be added to a doca_cc_connection.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_cap_get_max_consumers(const struct doca_devinfo *devinfo, uint32_t *max_consumers);

/**
 * Convert doca_cc_consumer instance into a generalised context for use with doca core objects.
 *
 * @param [in] consumer
 * Doca_cc_consumer instance. This must remain valid until after the context is no longer required.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_ctx *doca_cc_consumer_as_ctx(struct doca_cc_consumer *consumer);

/*********************************************************************************************************************
 * Consumer Post Receive Task
 *********************************************************************************************************************/

/* Task instance for consumer to do a post receive */
struct doca_cc_consumer_task_post_recv;

/**
 * Function executed on doca_cc_consumer post receive completion. Used for both task success and failure.
 *
 * @param [in] task
 * Doca consumer post recv task that has completed.
 * @param [in] task_user_data
 * The task user data.
 * @param [in] ctx_user_data
 * Doca_cc context user data.
 *
 * The implementation can assume this value is not NULL.
 */
typedef void (*doca_cc_consumer_task_post_recv_completion_cb_t)(struct doca_cc_consumer_task_post_recv *task,
								union doca_data task_user_data,
								union doca_data ctx_user_data);

/**
 * Configure the doca_cc_consumer post receive task callback and parameters.
 *
 * @param [in] consumer
 * The doca_cc_consumer instance.
 * @param [in] task_completion_cb
 * Post receive task completion callback.
 * @param [in] task_error_cb
 * Post receive task error callback.
 * @param [in] num_post_recv_tasks
 * Number of post_recv tasks a consumer can allocate.
 * Must not exceed value returned by doca_cc_consumer_cap_get_max_num_tasks().
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - consumer instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_task_post_recv_set_conf(struct doca_cc_consumer *consumer,
	doca_cc_consumer_task_post_recv_completion_cb_t task_completion_cb,
	doca_cc_consumer_task_post_recv_completion_cb_t task_error_cb, uint32_t num_post_recv_tasks);

/**
 * @brief Allocate and initialise a doca_consumer post receive task.
 *
 * Doca buffer should be located within the registered mmap associated with consumer instance.
 * Completion callback will be triggered whenever the buffer has been populated by a consumer.
 *
 * @param [in] consumer
 * The doca_cc_consumer instance.
 * @param [in] buf
 * Doca buffer available to be populated by producers.
 * @param [out] task
 * Pointer to a doca_cc_consumer_post_recv_task instance populated with input parameters.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_NO_MEMORY - no available tasks to allocate.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_consumer_task_post_recv_alloc_init(struct doca_cc_consumer *consumer, struct doca_buf *buf,
							struct doca_cc_consumer_task_post_recv **task);

/**
 * Get the doca_buf from the doca_cc_consumer_post_recv_task instance.
 *
 * @param [in] task
 * The doca_cc_consumer_post_recv_task instance.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_buf *doca_cc_consumer_task_post_recv_get_buf(struct doca_cc_consumer_task_post_recv *task);

/**
 * Set the doca_buf in a doca_cc_consumer_post_recv_task instance.
 *
 * @param [in] task
 * The doca_cc_consumer_post_recv_task instance.
 * @param [in] buf
 * Buffer to set in the task.
 */
DOCA_EXPERIMENTAL
void doca_cc_consumer_task_post_recv_set_buf(struct doca_cc_consumer_task_post_recv *task, struct doca_buf *buf);

/**
 * Get the producer id from the doca_cc_consumer_post_recv_task instance.
 *
 * Producer id will only be set on post recv completion and indicates the remote producer that has written data to the
 * associated doca_buf.
 *
 * @param [in] task
 * The doca_cc_consumer_post_recv_task instance.
 *
 * @return
 * Producer id upon success, 0 otherwise.
 */
DOCA_EXPERIMENTAL
uint32_t doca_cc_consumer_task_post_recv_get_producer_id(struct doca_cc_consumer_task_post_recv *task);

/**
 * Convert doca_cc_consumer_post_recv_task instance into a generalised task for use with progress engine.
 *
 * @param [in] task
 * Doca_cc_consumer_post_recv_task instance.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_cc_consumer_task_post_recv_as_task(struct doca_cc_consumer_task_post_recv *task);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* DOCA_CC_CONSUMER_H_ */
