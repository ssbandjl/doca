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

#ifndef VIRTNET_DPA_HOST_DBG_H
#define VIRTNET_DPA_HOST_DBG_H

#include "virtnet_dpa_common.h"

cJSON *
virtnet_dpa_vq_latency_stats_query(struct virtnet_prov_vq *prov_vq);

#endif
