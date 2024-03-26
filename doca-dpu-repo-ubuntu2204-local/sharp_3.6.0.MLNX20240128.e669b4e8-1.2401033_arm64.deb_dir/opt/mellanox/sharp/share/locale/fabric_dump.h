/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef AM_DUMP_H_
#define AM_DUMP_H_

#include <stdio.h>

/*
 * AM_DUMP_FILE_VERSION FORMAT: <N1.N2>
 * When removing field/table, increase N1, and reset N2.
 * When Adding new field/table, increase N2.
 */
#define AM_DUMP_FILE_VERSION     "1.0"
#define AM_DUMP_INDEX_TABLE_NAME "INDEX_TABLE"

#define AM_DUMP_MAX_STR_LEN 64
#define SHARP_DUMP_LINE_LEN 1024

typedef unsigned int (*fabric_dump_tbl_pfn_t)(FILE*, void*);

typedef struct fabric_dump_tbl_record
{
    const char* name;
    fabric_dump_tbl_pfn_t fabric_dump_tbl_pfn;
    long int offset;
    long int size;
    long int line;
    long int rows;
} fabric_dump_tbl_record_t;

typedef struct fabric_dump_tbl_context
{
    fabric_dump_tbl_record_t* p_records;
} fabric_dump_tbl_context_t;

void get_date_str(char* str, int len, bool full);
void fabric_dump_tables(FILE* file, void* cxt, const char* name);
uint32_t calc_crc_file_content(const char* file_name);

#endif   // AM_DUMP_H_
