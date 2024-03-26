/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2020 Mellanox Technologies, Ltd
 */

#ifndef RTE_SFT_DRIVER_H_
#define RTE_SFT_DRIVER_H_

/**
 * @file
 * RTE generic SFT API (driver side)
 *
 * This file provides implementation helpers for internal use by PMDs, they
 * are not intended to be exposed to applications and are not subject to ABI
 * versioning.
 */

#include <stdint.h>

#include "rte_ethdev.h"
#include "ethdev_driver.h"
#include "rte_sft.h"
#include "rte_flow.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rte_sft_entry;

__rte_internal int
rte_sft_error_set(struct rte_sft_error *error, int code,
		  enum rte_sft_error_type type, const void *cause,
                  const char *message);

#define RTE_SFT_STATE_INVALID (0)
#define RTE_SFT_STATE_FLAG_FID_VALID (1 << 0)
#define RTE_SFT_STATE_FLAG_ZONE_VALID (1 << 1)
#define RTE_SFT_STATE_FLAG_FLOW_MISS (1 << 2)
#define RTE_SFT_STATE_MASK 0x1f

#define RTE_SFT_MISS_TCP_FLAGS (1 << 0)

#define SFT_PATTERNS_NUM 8
#define SFT_ACTIONS_NUM 8

RTE_STD_C11
struct rte_sft_decode_info {
	union {
		uint32_t fid; /**< The fid value. */
		uint32_t zone; /**< The zone value. */
	};
	union {
		uint32_t state;
		struct {
			uint32_t fid_valid:1;
			uint32_t zone_valid:1;
			uint32_t direction:1;
			uint32_t control:1;
			uint32_t reserved:27;
		};
	};
	/**< Flags that mark the packet state. see RTE_SFT_STATE_FLAG_*. */
};

/**
 * @internal
 * Init the SFT pmd
 *
 * @param dev
 *   ethdev handle of port.
 * @param nb_queue
 *   Number of queues.
 * @param data_len
 *   The length of the data in uint32_t increments.
 * @param error[out]
 *   Verbose of the error.
 *
 * @return
 *   Negative errno value on error, 0 on success.
 */
typedef int (*sft_start_t) (struct rte_eth_dev *dev, uint16_t nb_queue,
			   uint16_t data_len, struct rte_sft_error *error);

/**
 * @internal
 * close the SFT pmd
 *
 * @param dev
 *   ethdev handle of port.
 * @param error[out]
 *   Verbose of the error.
 *
 * @return
 *   Negative errno value on error, 0 on success.
 */
typedef int (*sft_stop_t) (struct rte_eth_dev *dev, struct rte_sft_error *error);

/**
 * @internal
 * Insert a flow to the SFT HW component.
 *
 * @param dev
 *   ethdev handle of port.
 * @param fid
 *   Flow ID.
 * @param zone
 *   Flow zone.
 * @param queue
 *   The sft working queue.
 * @param pattern
 *   The matching pattern.
 * @param miss_conditions
 *   The conditions that forces a miss even if the 5 tuple was matched
 *   see RTE_SFT_MISS_*.
 * @param actions
 *   Set pf actions to apply in case the flow was hit. If no terminating action
 *   (queue, rss, drop, port) was given, the terminating action should be taken
 *   from the flow that resulted in the SFT.
 * @param miss_actions
 *   Set pf actions to apply in case the flow was hit. but the miss conditions
 *   were hit. (6 tuple match but tcp flags are on) If no terminating action
 *   (queue, rss, drop, port) was given, the terminating action should be taken
 *   from the flow that resulted in the SFT.
 * @param data
 *   The application data to attached to the flow.
 * @param data_len
 *   The length of the data in uint32_t increments.
 * @param state
 *   The application state to set.
 * @param error[out]
 *   Verbose of the error.
 *
 * @return
 *   Pointer to sft_entry in case of success, null otherwise and rte_sft_error
 *   is set.
 */
typedef struct rte_sft_entry *(*sft_entry_create_t)
	(struct rte_eth_dev *dev, uint32_t fid, uint32_t zone, uint16_t queue,
	 struct rte_flow_item *pattern, uint64_t miss_conditions,
	 struct rte_flow_action *actions,
	 struct rte_flow_action *miss_actions, const uint32_t *data,
	 uint16_t data_len, uint8_t state, bool initiator,
	 struct rte_sft_error *error);

/**
 * @internal
 * Modify the state and the data of SFT flow in HW component.
 *
 * @param dev
 *   ethdev handle of port.
 * @param entry
 *   The entry to modify.
 * @param queue
 *   The sft working queue.
 * @param entry
 *   The entry to modify.
 * @param data
 *   The application data to attached to the flow.
 * @param data_len
 *   The length of the data in uint32_t increments.
 * @param state
 *   The application state to set.
 * @param error[out]
 *   Verbose of the error.
 *
 * @return
 *   Negative errno value on error, 0 on success.
 */
typedef int (*sft_entry_modify_t)(struct rte_eth_dev *dev, uint16_t queue,
				  struct rte_sft_entry *entry,
				  const uint32_t *data, uint16_t data_len,
				  uint8_t state, struct rte_sft_error *error);

/**
 * @internal
 * Destroy SFT flow in HW component.
 *
 * @param dev
 *   ethdev handle of port.
 * @param entry
 *   The entry to destroy.
 * @param queue
 *   The sft working queue.
 * @param error[out]
 *   Verbose of the error.
 *
 * @return
 *   Negative errno value on error, 0 on success.
 */
typedef int (*sft_entry_destroy_t)(struct rte_eth_dev *dev,
				   struct rte_sft_entry *entry, uint16_t queue,
				   struct rte_sft_error *error);

/**
 * @internal
 * Decode sft state and FID from mbuf.
 *
 * @param dev
 *   ethdev handle of port.
 * @param entry
 *   The entry to modify.
 * @param queue
 *   The sft working queue.
 * @param mbuf
 *   The input mbuf.
 * @param info[out]
 *   The decoded sft data.
 * @param error[out]
 *   Verbose of the error.
 *
 * @return
 *   Negative errno value on error, 0 on success.
 */
typedef int (*sft_entry_decode_t)(struct rte_eth_dev *dev, uint16_t queue,
				  const struct rte_mbuf *mbuf,
				  struct rte_sft_decode_info *info,
				  struct rte_sft_error *error);

/**
 *  @internal
 *  Query sft entry hw counters.
 * @param dev
 *   ethdev handle of port.
 * @param queue
 *   The sft working queue.
 * @param entry
 *   The entry to query.
 * @param data[out]
 *   The output statistic.
 * @param error[out]
 *   Verbose of the error.
 *
 * @return
 *   Negative errno value on error, 0 on success.
 */
typedef int
(*sft_query_t)(struct rte_eth_dev *dev, uint16_t queue,
		struct rte_sft_entry *entry, struct rte_flow_query_count *data,
		struct rte_sft_error *error);

/**
 *  @internal
 *  Debug PMD sft
 */
typedef void
(*sft_debug_t)(struct rte_eth_dev *dev, struct rte_sft_entry *entry[2],
	       struct rte_sft_error *error);

/**
 * Generic sft operations structure implemented and returned by PMDs.
 *
 * If successful, this operation must result in a pointer to a PMD-specific.
 *
 * See also rte_sft_ops_get().
 *
 * These callback functions are not supposed to be used by applications
 * directly, which must rely on the API defined in rte_sft.h.
 */
struct rte_sft_ops {
	sft_start_t sft_start;
	sft_stop_t sft_stop;
	sft_entry_create_t sft_create_entry;
	sft_entry_modify_t sft_entry_modify;
	sft_entry_destroy_t sft_entry_destroy;
	sft_entry_decode_t sft_entry_decode;
	sft_query_t sft_query;
	sft_debug_t sft_debug;
};

#ifdef __cplusplus
}
#endif

#endif /* RTE_SFT_DRIVER_H_ */
