/*
 * Copyright (c) 2013-2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RESOURCE_DUMP_SEGMENT_H
#define RESOURCE_DUMP_SEGMENT_H

#include <common/compatibility.h>
#if __BYTE_ORDER == __BIG_ENDIAN
#include "resource_dump_segments_be.h"
#else
#include "resource_dump_segments_le.h"
#endif

#ifdef __cplusplus
namespace mft
{
namespace resource_dump
{
#endif

#ifdef __cplusplus
constexpr
#else
inline
#endif
  uint32_t
  to_segment_data_size(uint16_t length_dw)
{
    return length_dw * 4 - (sizeof(resource_dump_segment_header_t) + sizeof(resource_segment_sub_header_t));
}

#ifdef __cplusplus
} // namespace resource_dump
} // namespace mft
#endif

#endif // RESOURCE_DUMP_SEGMENT_H
