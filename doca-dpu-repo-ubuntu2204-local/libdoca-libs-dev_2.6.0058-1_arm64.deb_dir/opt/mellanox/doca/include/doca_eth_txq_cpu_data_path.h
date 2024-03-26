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
 * @file doca_eth_txq_cpu_data_path.h
 * @page DOCA ETH TXQ
 * @defgroup DOCAETHTXQ DOCA ETH TXQ CPU DATA PATH
 * DOCA ETH TXQ library.
 *
 * @{
 */
#ifndef DOCA_ETH_TXQ_CPU_DATA_PATH_H_
#define DOCA_ETH_TXQ_CPU_DATA_PATH_H_

#include <doca_buf.h>
#include <doca_ctx.h>
#include <doca_pe.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque structure representing a DOCA ETH TXQ instance.
 */
struct doca_eth_txq;

/**
 * Opaque structures representing DOCA ETH TXQ tasks.
 */

struct doca_eth_txq_task_send;			/** DOCA ETH TXQ task for transmitting a packet */
struct doca_eth_txq_task_lso_send;		/** DOCA ETH TXQ task for transmitting an LSO packet */

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
 * Please see doca_pe_progress for details.
 *
 * Any failure/error inside this function should be handled internally or differed;
 * due to the mode of nested in doca_pe_progress() execution this callback doesn't return error.
 *
 * NOTE: this callback type utilized successful & failed task completions.
 *
 * @param [in] task_send
 * The successfully completed task.
 * The implementation can assume this value is not NULL.
 * @param [in] task_user_data
 * user_data attached to the task.
 * @param [in] ctx_user_data
 * user_data attached to the ctx.
 */
typedef void (*doca_eth_txq_task_send_completion_cb_t)(struct doca_eth_txq_task_send *task_send,
						       union doca_data task_user_data,
						       union doca_data ctx_user_data);

/**
 * @brief Function to execute on task completion.
 *
 * @param [in] task_lso_send
 * The successfully completed task.
 * The implementation can assume this value is not NULL.
 * @param [in] task_user_data
 * user_data attached to the task.
 * @param [in] ctx_user_data
 * user_data attached to the ctx.
 */
typedef void (*doca_eth_txq_task_lso_send_completion_cb_t)(struct doca_eth_txq_task_lso_send *task_lso_send,
							   union doca_data task_user_data,
							   union doca_data ctx_user_data);

/**
 * @brief This method sets the doca_eth_txq_task_send tasks configuration.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] eth_txq
 * Pointer to doca_eth_txq instance.
 * @param [in] task_completion_cb
 * Task completion callback.
 * @param [in] task_error_cb
 * Task error callback.
 * @param [in] task_send_num
 * Number of doca_eth_txq_task_send tasks.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid parameter was given.
 * - DOCA_ERROR_NOT_PERMITTED - eth_txq context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_txq_task_send_set_conf(struct doca_eth_txq *eth_txq,
					     doca_eth_txq_task_send_completion_cb_t task_completion_cb,
					     doca_eth_txq_task_send_completion_cb_t task_error_cb,
					     uint32_t task_send_num);

/**
 * @brief This method sets the doca_eth_txq_task_lso_send tasks configuration.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *	 This a mandatory setter in case the user is going to use doca_eth_txq_task_lso_send tasks.
 *
 * @param [in] eth_txq
 * Pointer to doca_eth_txq instance.
 * @param [in] task_completion_cb
 * Task completion callback.
 * @param [in] task_error_cb
 * Task error callback.
 * @param [in] mss
 * Maximum Segment Size which is the maximum data size that can be sent in each segment of the LSO packet.
 * @param [in] max_lso_header_size
 * Maximum header size of the LSO packet.
 * @param [in] task_lso_send_num
 * Number of doca_eth_txq_task_lso_send tasks.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid parameter was given.
 * - DOCA_ERROR_NOT_PERMITTED - eth_txq context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_txq_task_lso_send_set_conf(struct doca_eth_txq *eth_txq,
						 doca_eth_txq_task_lso_send_completion_cb_t task_completion_cb,
						 doca_eth_txq_task_lso_send_completion_cb_t task_error_cb,
						 uint16_t mss, uint16_t max_lso_header_size,
						 uint32_t task_lso_send_num);

/**
 * @brief This method allocates and initializes a doca_eth_txq_task_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] eth_txq
 * Pointer to doca_eth_txq instance.
 * @param [in] pkt
 * Buffer that contains the packet to send.
 * @param [in] user_data
 * doca_data to attach to the task
 * @param [out] task_send
 * doca_eth_txq_task_send task that was allocated.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_NO_MMEORY - no more tasks to allocate.
 * - DOCA_ERROR_BAD_STATE - eth_txq context state is not running.
 * - DOCA_ERROR_NOT_SUPPORTED - in case eth_txq is not an instance for CPU.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_txq_task_send_allocate_init(struct doca_eth_txq *eth_txq, struct doca_buf *pkt,
						  union doca_data user_data,
						  struct doca_eth_txq_task_send **task_send);

/**
 * @brief This method allocates and initializes a doca_eth_txq_task_lso_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] eth_txq
 * Pointer to doca_eth_txq instance.
 * @param [in] pkt_payload
 * Buffer that contains the payload of the packet to send.
 * @param [in] headers
 * A gather list of the headers of the packet to send.
 * @param [in] user_data
 * doca_data to attach to the task
 * @param [out] task_lso_send
 * doca_eth_txq_task_lso_send task that was allocated.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_NO_MMEORY - no more tasks to allocate.
 * - DOCA_ERROR_BAD_STATE - eth_txq context state is not running.
 * - DOCA_ERROR_NOT_SUPPORTED - in case eth_txq is not an instance for CPU.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_txq_task_lso_send_allocate_init(struct doca_eth_txq *eth_txq, struct doca_buf *pkt_payload,
						      struct doca_gather_list *headers, union doca_data user_data,
						      struct doca_eth_txq_task_lso_send **task_lso_send);

/**
 * @brief This method sets packet buffer to doca_eth_txq_task_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_send
 * The task to set to.
 * @param [in] pkt
 * Packet buffer to set.
 */
DOCA_EXPERIMENTAL
void doca_eth_txq_task_send_set_pkt(struct doca_eth_txq_task_send *task_send, struct doca_buf *pkt);

/**
 * @brief This method sets packet payload buffer to doca_eth_txq_task_lso_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_lso_send
 * The task to set to.
 * @param [in] pkt_payload
 * Packet payload buffer to set.
 */
DOCA_EXPERIMENTAL
void doca_eth_txq_task_lso_send_set_pkt_payload(struct doca_eth_txq_task_lso_send *task_lso_send,
						struct doca_buf *pkt_payload);

/**
 * @brief This method sets headers to doca_eth_txq_task_lso_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_lso_send
 * The task to set to.
 * @param [in] headers
 * A gather list of the headers of the packet to set.
 */
DOCA_EXPERIMENTAL
void doca_eth_txq_task_lso_send_set_headers(struct doca_eth_txq_task_lso_send *task_lso_send, struct doca_gather_list *headers);

/**
 * @brief This method gets packet buffer from doca_eth_txq_task_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_send
 * The task to get from.
 * @param [out] pkt
 * Packet buffer to get.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - task_send or pkt is a NULL pointer.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_txq_task_send_get_pkt(const struct doca_eth_txq_task_send *task_send, struct doca_buf **pkt);

/**
 * @brief This method gets data buffer from doca_eth_txq_task_lso_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_lso_send
 * The task to get from.
 * @param [out] pkt_payload
 * Packet buffer buffer to get.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - task_lso_send or pkt_payload is a NULL pointer.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_txq_task_lso_send_get_pkt_payload(const struct doca_eth_txq_task_lso_send *task_lso_send,
							struct doca_buf **pkt_payload);

/**
 * @brief This method gets headers from doca_eth_txq_task_lso_send task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_lso_send
 * The task to get from.
 * @param [out] headers
 * A gather list of the headers of the packet to get.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_eth_txq_task_lso_send_get_headers(const struct doca_eth_txq_task_lso_send *task_lso_send,
						    struct doca_gather_list **headers);

/**
 * @brief This method converts a doca_eth_txq_task_send task to doca_task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_send
 * doca_eth_txq_task_send task.
 *
 * @return doca_task
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_eth_txq_task_send_as_doca_task(struct doca_eth_txq_task_send *task_send);

/**
 * @brief This method converts a doca_eth_txq_task_lso_send task to doca_task.
 *
 * @note Supported for DOCA ETH TXQ instance for CPU only.
 *
 * @param [in] task_lso_send
 * doca_eth_txq_task_send task.
 *
 * @return doca_task
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_eth_txq_task_lso_send_as_doca_task(struct doca_eth_txq_task_lso_send *task_lso_send);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_ETH_TXQ_CPU_DATA_PATH_H_ */

/** @} */
