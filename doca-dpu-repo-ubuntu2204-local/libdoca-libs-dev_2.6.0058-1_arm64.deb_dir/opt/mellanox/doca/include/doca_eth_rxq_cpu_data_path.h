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
 * @file doca_eth_rxq_cpu_data_path.h
 * @page DOCA ETH RXQ
 * @defgroup DOCAETHTXQ DOCA ETH RXQ CPU DATA PATH
 * DOCA ETH RXQ library.
 *
 * @{
 */
#ifndef DOCA_ETH_RXQ_CPU_DATA_PATH_H_
#define DOCA_ETH_RXQ_CPU_DATA_PATH_H_

#include <doca_buf.h>
#include <doca_ctx.h>
#include <doca_pe.h>
#include <doca_buf_inventory.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque structure representing a DOCA ETH RXQ instance.
 */
struct doca_eth_rxq;

/**
 * Opaque structures representing DOCA ETH RXQ tasks.
 */

struct doca_eth_rxq_task_recv;			/**< DOCA ETH RXQ task for receiving a single packet. Supported
						  *  in DOCA_ETH_RXQ_TYPE_REGULAR mode.
						  */

/**
 * Opaque structures representing DOCA ETH RXQ events.
 */

struct doca_eth_rxq_event_managed_recv;		/**< DOCA ETH RXQ event for receiving multiple packets. Supported
						  *  in DOCA_ETH_RXQ_TYPE_MANAGED_MEMPOOL mode.
						  */

/**
 * @brief Function to execute on task completion.
 *
 * @details This function is called by doca_pe_progress() when related task identified as completed successfully.
 * When this function called the ownership of the task object passed from DOCA back to user.
 * Inside this callback user may decide on the task object:
 * - re-submit task with doca_task_submit(); task object ownership passed to DOCA
 * - release task with doca_task_free(); task object ownership passed to DOCA
 * - keep the task object for future re-use; user keeps the ownership on the task object
 * Inside this callback the user shouldn't call doca_pe_progress().
 * Any failure/error inside this function should be handled internally or differed;
 * due to the mode of nested in doca_pe_progress() execution this callback doesn't return error.
 *
 * NOTE: this callback type utilized successful & failed task completions.
 *
 * @param [in] task_recv
 * The successfully completed task.
 * The implementation can assume this value is not NULL.
 * @param [in] task_user_data
 * user_data attached to the task.
 * @param [in] ctx_user_data
 * user_data attached to the ctx.
 */
typedef void (*doca_eth_rxq_task_recv_completion_cb_t)(struct doca_eth_rxq_task_recv *task_recv,
						       union doca_data task_user_data,
						       union doca_data ctx_user_data);

/**
 * @brief Function to be executed on managed receive event occurrence.
 *
 * @note The packet buffer returned is valid as long as it wasn't freed by the user.
 *	 Holding the buffer for a long period of time might will block receiving incoming packets
 *	 as mentioned above for the DOCA_ETH_RXQ_TYPE_MANAGED_MEMPOOL type.
 *
 * @param [in] event_managed_recv
 * The managed receive event.
 * The implementation can assume this value is not NULL.
 * @param [in] pkt
 * doca_buf containing the received packet (NULL in case of error callback).
 * @param [in] event_user_data
 * user_data attached to the event.
 */
typedef void (*doca_eth_rxq_event_managed_recv_handler_cb_t)(struct doca_eth_rxq_event_managed_recv *event_managed_recv,
							     struct doca_buf *pkt,
							     union doca_data event_user_data);

/**
 * @brief This method sets the doca_eth_rxq_task_recv tasks configuration.
 * can only be called before calling doca_ctx_start().
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *	 Function is relevant only in the case of context of type DOCA_ETH_RXQ_TYPE_REGULAR.
 *
 * @param [in] eth_rxq
 * Pointer to doca_eth_rxq instance.
 * @param [in] task_completion_cb
 * Task completion callback.
 * @param [in] task_error_cb
 * Task error callback.
 * @param [in] task_recv_num
 * Number of doca_eth_rxq_task_recv tasks.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - eth_rxq argument is a NULL pointer.
 * - DOCA_ERROR_NOT_PERMITTED - eth_rxq context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_task_recv_set_conf(struct doca_eth_rxq *eth_rxq,
					     doca_eth_rxq_task_recv_completion_cb_t task_completion_cb,
					     doca_eth_rxq_task_recv_completion_cb_t task_error_cb,
					     uint32_t task_recv_num);

/**
 * @brief This method registers a doca_eth_rxq_event_managed_recv event.
 * can only be called before calling doca_ctx_start().
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] eth_rxq
 * Pointer to doca_eth_rxq instance.
 * @param [in] user_data
 * doca_data to attach to the event.
 * @param [in] success_event_handler
 * Method that is invoked once a successful event is triggered
 * @param [in] error_event_handler
 * Method that is invoked once an error event is triggered
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_BAD_STATE - eth_rxq context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_event_managed_recv_register(struct doca_eth_rxq *eth_rxq,
						      union doca_data user_data,
						      doca_eth_rxq_event_managed_recv_handler_cb_t success_event_handler,
						      doca_eth_rxq_event_managed_recv_handler_cb_t error_event_handler);

/**
 * @brief This method allocates and initializes a doca_eth_rxq_task_recv task.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] eth_rxq
 * Pointer to doca_eth_rxq instance.
 * @param [in] user_data
 * doca_data to attach to the task.
 * @param [in] pkt
 * Buffer to receive packet.
 * @param [out] task_recv
 * doca_eth_rxq_task_recv task that was allocated.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_NO_MMEORY - no more tasks to allocate.
 * - DOCA_ERROR_BAD_STATE - eth_rxq context state is not running.
 * - DOCA_ERROR_NOT_SUPPORTED - in case eth_rxq is not an instance for CPU.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_task_recv_allocate_init(struct doca_eth_rxq *eth_rxq,
						  union doca_data user_data, struct doca_buf *pkt,
						  struct doca_eth_rxq_task_recv **task_recv);

/**
 * @brief This method sets packet buffer to doca_eth_rxq_task_recv task.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] task_recv
 * The task to set to.
 * @param [in] pkt
 * Packet buffer to set.
 */
DOCA_EXPERIMENTAL
void doca_eth_rxq_task_recv_set_pkt(struct doca_eth_rxq_task_recv *task_recv, struct doca_buf *pkt);

/**
 * @brief This method gets packet buffer from doca_eth_rxq_task_recv task.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] task_recv
 * The task to get from.
 * @param [out] pkt
 * Packet buffer to get.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_task_recv_get_pkt(const struct doca_eth_rxq_task_recv *task_recv, struct doca_buf **pkt);

/**
 * @brief This method checks if L3 checksum of finished doca_eth_rxq_task_recv task is ok.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] task_recv
 * The task to get from.
 * @param [out] l3_ok
 * Indicator whether L3 checksum is ok or not.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_task_recv_get_l3_ok(const struct doca_eth_rxq_task_recv *task_recv,
					      uint8_t *l3_ok);

/**
 * @brief This method checks if L3 checksum of finished doca_eth_rxq_event_managed_recv event is ok.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] event_managed_recv
 * The event to get from.
 * @param [out] l3_ok
 * Indicator whether L3 checksum is ok or not.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_event_managed_recv_get_l3_ok(const struct doca_eth_rxq_event_managed_recv *event_managed_recv,
						       uint8_t *l3_ok);

/**
 * @brief This method checks if L4 checksum of finished doca_eth_rxq_task_recv task is ok.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] task_recv
 * The task to get from.
 * @param [out] l4_ok
 * Indicator whether L4 checksum is ok or not.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_task_recv_get_l4_ok(const struct doca_eth_rxq_task_recv *task_recv,
					      uint8_t *l4_ok);

/**
 * @brief This method checks if L4 checksum of finished doca_eth_rxq_event_managed_recv event is ok.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] event_managed_recv
 * The event to get from.
 * @param [out] l4_ok
 * Indicator whether L4 checksum is ok or not.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_event_managed_recv_get_l4_ok(const struct doca_eth_rxq_event_managed_recv *event_managed_recv,
						       uint8_t *l4_ok);

/**
 * @brief This method gets status of finished doca_eth_rxq_event_managed_recv event.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] event_managed_recv
 * The event to get status from.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * Any other doca_error_t indicates that the event failed (event depended)
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_rxq_event_managed_recv_get_status(const struct doca_eth_rxq_event_managed_recv *event_managed_recv);

/**
 * @brief This method converts a doca_eth_rxq_task_recv task to doca_task.
 *
 * @note Supported for DOCA ETH RXQ instance for CPU only.
 *
 * @param [in] task_recv
 * doca_eth_rxq_task_recv task.
 *
 * @return doca_task
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_eth_rxq_task_recv_as_doca_task(struct doca_eth_rxq_task_recv *task_recv);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_ETH_RXQ_CPU_DATA_PATH_H_ */

/** @} */
