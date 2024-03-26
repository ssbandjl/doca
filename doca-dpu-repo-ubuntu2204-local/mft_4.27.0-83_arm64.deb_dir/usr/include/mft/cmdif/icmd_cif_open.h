/*
 * Copyright (c) 2013-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#ifndef _ICMD_OPEN_LIB /* guard */
#define _ICMD_OPEN_LIB

#ifdef __cplusplus
extern "C"
{
#endif

#include <mtcr.h>
#include <common/compatibility.h>
#include "icmd_cif_common.h"

#ifndef IN
#define IN
#define OUT
#define INOUT
#endif

    enum
    {
        GET_FW_INFO = 0x8007,
        FLASH_REG_ACCESS = 0x9001,
    };

#ifdef MST_UL
    // instead of cib_cif.h in mstflint
    enum
    {
        GET_ICMD_QUERY_CAP = 0x8400,
        SET_ITRACE = 0xf003,
    };
#endif

    struct connectib_icmd_get_fw_info;
    int gcif_get_fw_info(mfile* mf, OUT struct connectib_icmd_get_fw_info* fw_info);
    struct icmd_hca_icmd_query_cap_general;
    int get_icmd_query_cap(mfile* mf, struct icmd_hca_icmd_query_cap_general* icmd_query_caps);
    struct icmd_hca_icmd_mh_sync_in;
    struct icmd_hca_icmd_mh_sync_out;
    int gcif_mh_sync(mfile* mf,
                     struct icmd_hca_icmd_mh_sync_in* mh_sync_in,
                     struct icmd_hca_icmd_mh_sync_out* mh_sync_out);

    int gcif_mh_sync_status(mfile* mf,
                            struct icmd_hca_icmd_mh_sync_in* mh_sync_in,
                            struct icmd_hca_icmd_mh_sync_out* mh_sync_out);

#ifdef __cplusplus
}
#endif

#endif
