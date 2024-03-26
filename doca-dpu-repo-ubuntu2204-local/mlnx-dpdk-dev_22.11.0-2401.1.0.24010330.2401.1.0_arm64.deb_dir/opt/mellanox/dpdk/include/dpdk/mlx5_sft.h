/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2020 Mellanox Technologies, Ltd
 */

#ifndef RTE_PMD_MLX5_MLX5_H
#define RTE_PMD_MLX5_MLX5_H

#include <rte_flow.h>
#include <rte_sft_driver.h>

#define MLX5_SFT_L1_IMPLICIT_ACTIONS_NUM 4
#define MLX5_SFT_L1_ACTIONS_NUM (SFT_ACTIONS_NUM + \
				 MLX5_SFT_L1_IMPLICIT_ACTIONS_NUM)
__extension__
union mlx5_sft_entry_flags {
	uint8_t val;
	struct {
		uint8_t initiator:1;
		uint8_t ipv4:1;
		uint8_t ipv6:1;
		uint8_t udp:1;
		uint8_t tcp:1;
	};
};

struct rte_sft_entry {
	ILIST_ENTRY(uint32_t)next;
	uint32_t idx;
	uint32_t fid;
	uint32_t sft_l0_flow;
	uint32_t sft_l1_flow;
	union mlx5_sft_entry_flags flags;
	uint64_t miss_conditions;
};

#define MLX5_SFT_QUEUE_MAX			(64)
#define MLX5_SFT_FID_ZONE_MASK			(0x00FFFFFF)
#define MLX5_SFT_RSVD_SHIFT			(24)
#define MLX5_SFT_FID_ZONE_STAT_SHIFT		(0)
#define MLX5_SFT_FID_ZONE_STAT_MASK		(0xF)
#define MLX5_SFT_USER_STAT_SHIFT		(16)
#define MLX5_SFT_USER_STAT_MASK			(0xFF)

#define MLX5_SFT_ENCODE_MARK(valid, usr) \
	((((valid) & MLX5_SFT_FID_ZONE_STAT_MASK) << \
	   MLX5_SFT_FID_ZONE_STAT_SHIFT) | \
	 (((usr) & MLX5_SFT_USER_STAT_MASK) << \
	  MLX5_SFT_USER_STAT_SHIFT))

#ifdef RTE_LIBRTE_MLX5_DEBUG

#define MLX5_SFT_ENTRY_FLOW_COUNT_NB		(16)
int mlx5_sft_flow_count_query(void);

#endif

#endif
