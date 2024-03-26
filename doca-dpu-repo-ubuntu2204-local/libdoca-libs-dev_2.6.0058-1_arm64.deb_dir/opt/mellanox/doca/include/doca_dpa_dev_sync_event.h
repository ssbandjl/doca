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
 * @file doca_dpa_dev_sync_event.h
 * @page doca dpa sync event
 * @defgroup DPA_SE DOCA DPA Sync Event
 * @ingroup DPA_DEVICE
 * DOCA DPA Sync Event
 * @{
 */

#ifndef DOCA_DPA_DEV_SYNC_EVENT_H_
#define DOCA_DPA_DEV_SYNC_EVENT_H_

#include <doca_dpa_dev.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DPA sync event handle type definition
 */
__dpa_global__ typedef uint64_t doca_dpa_dev_sync_event_t;

/**
 * @brief DPA remote sync event handle type definition
 */
__dpa_global__ typedef uint64_t doca_dpa_dev_sync_event_remote_net_t;

/**
 * @brief Get the counter's value of a DOCA Sync Event
 *
 * @param[in] dpa_dev_se_handle - DOCA DPA device sync event handle
 * @param[out] value - DOCA sync event counter value
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_sync_event_get(doca_dpa_dev_sync_event_t dpa_dev_se_handle, uint64_t *value);

/**
 * @brief Atomically increase the counter of a DOCA Sync Event by a given value
 *
 * @param[in] dpa_dev_se_handle - DOCA DPA device sync event handle
 * @param[in] value - the value to increment DOCA sync event by
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_sync_event_update_add(doca_dpa_dev_sync_event_t dpa_dev_se_handle, uint64_t value);

/**
 * @brief Set the counter of a DOCA Sync Event to a given value
 *
 * @param[in] dpa_dev_se_handle - DOCA DPA device sync event handle
 * @param[in] value - the value to set the DOCA sync event to
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_sync_event_update_set(doca_dpa_dev_sync_event_t dpa_dev_se_handle, uint64_t value);

/**
 * @brief Wait for the value of a DOCA Sync Event to be greater than a given value.
 *
 * @param[in] dpa_dev_se_handle - DOCA DPA device sync event handle
 * @param[in] value - the value to wait for the DOCA Sync Event to be greater than
 * @param[in] mask - mask to apply (bitwise AND) on the DOCA Sync Event value for comparison with wait threshold.
 *
 * @return
 * This function does not return any value
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_sync_event_wait_gt(doca_dpa_dev_sync_event_t dpa_dev_se_handle, uint64_t value, uint64_t mask);


#ifdef __cplusplus
}
#endif

/** @} */

#endif /* DOCA_DPA_DEV_SYNC_EVENT_H_ */
