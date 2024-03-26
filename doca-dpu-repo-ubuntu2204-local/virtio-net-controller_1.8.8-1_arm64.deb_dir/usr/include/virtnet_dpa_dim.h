/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef VIRTNET_DPA_DIM_H
#define VIRTNET_DPA_DIM_H

/*
 * Number of events between DIM iterations.
 * Causes a moderation of the algorithm run.
 */
#define VIRTNET_DPA_DIM_NEVENTS 64

#define USEC_PER_MSEC   1000L
#define VIRTNET_DPA_DIM_CQ_PERIOD_NUM_MODES 2
#define VIRTNET_DPA_DIM_RX_PROFILE_CNT 6
#define VIRTNET_DPA_DIM_RX_DEFAULT_PROFILE 2
#define VIRTNET_DPA_DIM_RX_FINDING_STABLE_TIMES 2
#define VIRTNET_DPA_DIM_DEFAULT_ON_TOP_TIRED 3

/*
 * Calculate the gap between two values.
 * Take wrap-around and variable size into consideration.
 */
#define BIT_GAP(bits, end, start) ((((end) - (start)) + BIT_ULL(bits)) \
		& (BIT_ULL(bits) - 1))

/*
 * Is a difference between values justifies taking an action.
 * We consider 5% difference as significant.
 */
#define IS_SIGNIFICANT_DIFF(val, ref) \
	((val > ref) ? (((100UL * ((val) - (ref))) / (ref)) > 5) :\
	 (((100UL * ((ref) - (val))) / (ref)) > 5))

/*
 * enum virtnet_dpa_dim_tune_state - DIM algorithm tune states
 *
 * These will determine which action the algorithm should perform.
 *
 * @DIM_PARKING_ON_TOP: Algorithm found a local top point.
 *                      Exit on significant difference.
 * @DIM_PARKING_FINDING_STABLE: Algorithm is trying to find a stable traffic.
 *                              if pps is same for 3 period, will exit.
 * @DIM_GOING_RIGHT: Algorithm is currently trying higher moderation levels
 * @DIM_GOING_LEFT: Algorithm is currently trying lower moderation levels
 */
enum virtnet_dpa_dim_tune_state {
	DIM_PARKING_FINDING_STABLE,
	DIM_PARKING_ON_TOP,
	DIM_GOING_RIGHT,
	DIM_GOING_LEFT,
};

/*
 * enum dim_stats_state - DIM algorithm statistics states
 *
 * These will determine the verdict of current iteration.
 *
 * @DIM_STATS_WORSE: Current iteration shows worse performance than before
 * @DIM_STATS_SAME:  Current iteration shows same performance than before
 * @DIM_STATS_BETTER: Current iteration shows better performance than before
 */
enum virtnet_dpa_dim_stats_state {
	DIM_STATS_WORSE,
	DIM_STATS_SAME,
	DIM_STATS_BETTER,
};

#define VIRTNET_DPA_DIM_RX_CQE_PROFILES { \
	{0, 0},\
	{16, 32},\
	{32, 32},\
	{64, 32},\
	{256, 32},\
	{1023, 32} \
}

struct virtnet_dpa_dim_cq_moder {
	uint16_t usec;
	uint16_t pkts;
};

void *virtnet_dpa_dim_func(void *context);
#endif
