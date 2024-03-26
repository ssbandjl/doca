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
 * @file doca_dpa_dev_rdma.h
 * @page doca dpa rdma
 * @defgroup DPA_RDMA DOCA DPA rdma
 * @ingroup DPA_DEVICE
 * DOCA DPA rdma
 * @{
 */

#ifndef DOCA_DPA_DEV_RDMA_H_
#define DOCA_DPA_DEV_RDMA_H_

#include <doca_dpa_dev.h>
#include <doca_dpa_dev_buf.h>
#include <doca_dpa_dev_sync_event.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DPA RDMA handle type definition
 */
__dpa_global__ typedef uint64_t doca_dpa_dev_rdma_t;

/**
 * @brief DPA RDMA SRQ handle type definition
 */
__dpa_global__ typedef uint64_t doca_dpa_dev_rdma_srq_t;

/**
 * @brief Send an RDMA read operation
 *
 * @param[in] rdma - RDMA DPA handle
 * @param[in] dst_mem - destination buffer DPA handle
 * @param[in] dst_offset - offset on the destination buffer
 * @param[in] src_mem - source buffer DPA handle
 * @param[in] src_offset - offset on the source buffer
 * @param[in] length - length of buffer
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_read(doca_dpa_dev_rdma_t rdma,
			    doca_dpa_dev_buf_t dst_mem,
			    uint64_t dst_offset,
			    doca_dpa_dev_buf_t src_mem,
			    uint64_t src_offset,
			    size_t length);

/**
 * @brief Send an RDMA write operation
 *
 * @param[in] rdma - RDMA DPA handle
 * @param[in] dst_mem - destination buffer DPA handle
 * @param[in] dst_offset - offset on the destination buffer
 * @param[in] src_mem - source buffer DPA handle
 * @param[in] src_offset - offset on the source buffer
 * @param[in] length - length of buffer
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_write(doca_dpa_dev_rdma_t rdma,
			     doca_dpa_dev_buf_t dst_mem,
			     uint64_t dst_offset,
			     doca_dpa_dev_buf_t src_mem,
			     uint64_t src_offset,
			     size_t length);

/**
 * @brief Post an RDMA send operation
 *
 * @param[in] rdma - RDMA DPA handle
 * @param[in] buf - send buffer DPA handle
 * @param[in] offset - offset on the send buffer
 * @param[in] length - length of send buffer
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_post_send(doca_dpa_dev_rdma_t rdma, doca_dpa_dev_buf_t buf, uint64_t offset, size_t length);

/**
 * @brief Post an RDMA receive operation
 *
 * @param[in] rdma - RDMA DPA handle
 * @param[in] buf - received buffer DPA handle
 * @param[in] offset - offset on the received buffer
 * @param[in] length - length of received buffer
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_post_receive(doca_dpa_dev_rdma_t rdma, doca_dpa_dev_buf_t buf, uint64_t offset, size_t length);

/**
* @brief Post an RDMA receive operation on a SRQ
*
* @param[in] rdma_srq – RDMA SRQ DPA handle
* @param[in] buf - received buffer DPA handle
* @param[in] offset - offset on the received buffer
* @param[in] length - length of received buffer
*
* @return
* This function does not return any value
*/
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_srq_post_receive(doca_dpa_dev_rdma_srq_t rdma_srq, doca_dpa_dev_buf_t buf, uint64_t offset,
					size_t length);

/**
 * @brief Send an RDMA atomic fetch and add operation
 *
 * @param[in] rdma - RDMA DPA handle
 * @param[in] dst_mem - destination buffer DPA handle
 * @param[in] dst_offset - offset on the destination buffer
 * @param[in] value - value to add to the destination buffer
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_atomic_fetch_add(doca_dpa_dev_rdma_t rdma,
					doca_dpa_dev_buf_t dst_mem,
					uint64_t dst_offset,
					size_t value);

/**
 * @brief Signal to set a remote sync event count
 *
 * @param[in] rdma - RDMA DPA handle
 * @param[in] remote_sync_event - remote sync event DPA handle
 * @param[in] count - count to set
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_signal_set(doca_dpa_dev_rdma_t rdma,
				  doca_dpa_dev_sync_event_remote_net_t remote_sync_event,
				  uint64_t count);

/**
 * @brief Signal to atomically add to a remote sync event count
 *
 * @param[in] rdma - RDMA DPA handle
 * @param[in] remote_sync_event - remote sync event DPA handle
 * @param[in] count - count to add
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_signal_add(doca_dpa_dev_rdma_t rdma,
				  doca_dpa_dev_sync_event_remote_net_t remote_sync_event,
				  uint64_t count);

/**
 * @brief Synchronize all operations on an RDMA DPA handle
 *
 * @param[in] rdma - RDMA DPA handle
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_rdma_synchronize(doca_dpa_dev_rdma_t rdma);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* DOCA_DPA_DEV_RDMA_H_ */
