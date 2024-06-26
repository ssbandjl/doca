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

#ifndef RESOURCE_DUMP_SEGMENT_BE_H
#define RESOURCE_DUMP_SEGMENT_BE_H

#include <stdint.h>

#ifdef __cplusplus
namespace mft
{
namespace resource_dump
{
#endif

#ifdef __cplusplus
enum struct SegmentType : uint16_t
#else
enum SegmentType
#endif
{
    notice = 0xfff9,
    command,
    terminate,
    error,
    reference,
    info,
    menu
};

typedef struct resource_dump_segment_header
{
    uint16_t length_dw;
    uint16_t segment_type;
} resource_dump_segment_header_t;

typedef struct segment_params_data
{
    uint32_t index1;
    uint32_t index2;
    uint16_t num_of_obj1;
    uint16_t num_of_obj2;
} segment_params_data_t;

typedef struct resource_segment_sub_header
{
    uint32_t reserved;
    uint32_t index1;
    uint32_t index2;
} resource_segment_sub_header_t;

typedef struct notice_segment_data
{
    uint16_t reserved0;
    uint16_t syndrome_id;
    uint32_t reserved1;
    uint32_t reserved2;
    char notice[32];
} notice_segment_data_t;

typedef struct command_segment_data
{
    uint16_t segment_called;
    uint16_t vhca_id;
    segment_params_data_t segment_params;
} command_segment_data_t;

typedef notice_segment_data_t error_segment_data;

struct reference_segment_data
{
    uint16_t reserved;
    uint16_t reference_segment_type;
    segment_params_data_t segment_params;
};

typedef struct info_segment_data
{
    uint8_t reserved[3];
    uint8_t dump_version;
    uint32_t hw_version;
    uint32_t fw_version;
} info_segment_data_t;

typedef struct menu_segment_sub_header
{
    uint16_t reserved;
    uint16_t num_of_records;
} menu_segment_sub_header_t;

typedef struct menu_record_data
{
    uint16_t reserved : 3;
    uint16_t event_supported : 1;
    uint16_t num_of_obj2_supports_active : 1;
    uint16_t num_of_obj2_supports_all : 1;
    uint16_t must_have_num_of_obj2 : 1;
    uint16_t support_num_of_obj2 : 1;
    uint16_t num_of_obj1_supports_active : 1;
    uint16_t num_of_obj1_supports_all : 1;
    uint16_t must_have_num_of_obj1 : 1;
    uint16_t support_num_of_obj1 : 1;
    uint16_t must_have_index2 : 1;
    uint16_t support_index2 : 1;
    uint16_t must_have_index1 : 1;
    uint16_t support_index1 : 1;

    uint16_t segment_type;

    char segment_name[16];
    char index1_name[16];
    char index2_name[16];
} menu_record_data_t;

#ifdef __cplusplus
} // namespace resource_dump
} // namespace mft
#endif

#endif // RESOURCE_DUMP_SEGMENT_BE_H
