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

#ifndef __VIRTNET_DPA_UTIL_H__
#define __VIRTNET_DPA_UTIL_H__

#include <stdint.h>
#include <virtnet_dpa_stack.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POW2(x) (1 << (x))
#define POW2MASK(x) (POW2(x) - 1)
typedef uint32_t be32_t;
#define NUM_POW2(x) (x & (x - 1))

#define MINIMUM(a, b) ((a)<(b) ? (a) : (b))

#if defined(E_MODE_LE)
#define le16_to_cpu(val) (val)
#define le32_to_cpu(val) (val)
#define le64_to_cpu(val) (val)
#define be16_to_cpu(val) __builtin_bswap16(val)
#define be32_to_cpu(val) __builtin_bswap32(val)
#define be64_to_cpu(val) __builtin_bswap64(val)

#define cpu_to_le16(val) (val)
#define cpu_to_le32(val) (val)
#define cpu_to_le64(val) (val)
#define cpu_to_be16(val) __builtin_bswap16(val)
#define cpu_to_be32(val) __builtin_bswap32(val)
#define cpu_to_be64(val) __builtin_bswap64(val)

#elif defined(E_MODE_BE)
#define le16_to_cpu(val) __builtin_bswap16(val)
#define le32_to_cpu(val) __builtin_bswap32(val)
#define le64_to_cpu(val) __builtin_bswap64(val)
#define be16_to_cpu(val) (val)
#define be32_to_cpu(val) (val)
#define be64_to_cpu(val) (val)

#define cpu_to_le16(val) __builtin_bswap16(val)
#define cpu_to_le32(val) __builtin_bswap32(val)
#define cpu_to_le64(val) __builtin_bswap64(val)
#define cpu_to_be16(val) (val)
#define cpu_to_be32(val) (val)
#define cpu_to_be64(val) (val)

#endif

/* WRITE_ONCE_SCALAR() - Doesn't work with structure
 * Force writing the field at the supplied memory address. This assurance is
 * needed in below cases.
 * 1. Force writing the memory without tearing(splitting) stores.
 * 2. Force writing the memory even though it may not be read by the same
 *    writer and avoid any compiler optimization to skip this write.
 */
#define WRITE_ONCE_SCALAR(field, data) \
	(*((volatile typeof(field) *)(&(field))) = (data))

/* READ_ONCE_SCALAR() - Doesn't work with structure
 * Force reading the field for the supplied memory address. This assurance
 * is needed in below cases.
 * 1. When a memory is shared between two threads, or written by device and
 *    read by sw.
 * 2. need to read the field without tearing them in multiple loads.
 */
#define READ_ONCE_SCALAR(field) (*((volatile typeof(field) *)(&(field))))

#ifdef __cplusplus
}
#endif

#endif
