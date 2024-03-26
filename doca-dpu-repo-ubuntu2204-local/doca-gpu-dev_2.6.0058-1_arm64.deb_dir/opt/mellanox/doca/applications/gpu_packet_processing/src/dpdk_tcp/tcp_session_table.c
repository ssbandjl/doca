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

#include <rte_jhash.h>

#include "tcp_session_table.h"

struct rte_hash_parameters tcp_session_ht_params = {
	.name = "tcp_session_ht",
	.entries = TCP_SESSION_MAX_ENTRIES,
	.key_len = sizeof(struct tcp_session_key),
	.hash_func = rte_jhash,
	.hash_func_init_val = 0,
	.extra_flag = 0, // remember: if >1 lcore is needed, make this thread-safe and update insert/delete logic
};

struct rte_hash *tcp_session_table;

