/*
 * Copyright © 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef _SNAP_POLL_GROUPS_H
#define _SNAP_POLL_GROUPS_H

#include <pthread.h>
#include <sys/queue.h>

struct snap_pg_q_entry {
	TAILQ_ENTRY(snap_pg_q_entry) entry;
};

struct snap_pg {
	int id;

	TAILQ_HEAD(, snap_pg_q_entry) q_list;
	pthread_spinlock_t lock;
};

struct snap_pg_ctx {
    /* Polling groups */
	struct snap_pg *pgs;
	int npgs;
};

void snap_pgs_free(struct snap_pg_ctx *ctx);
int snap_pgs_alloc(struct snap_pg_ctx *ctx, int nthreads);
void snap_pgs_suspend(struct snap_pg_ctx *ctx);
void snap_pgs_resume(struct snap_pg_ctx *ctx);
struct snap_pg *snap_pg_get_next(struct snap_pg_ctx *ctx);
void snap_pg_usage_decrease(size_t pg_index);

struct snap_pg *snap_pg_get_admin(struct snap_pg_ctx *ctx);

#endif
