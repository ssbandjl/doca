/*
 * Copyright (C) Jan 2013 Mellanox Technologies Ltd. All rights reserved.
 * Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __MTCR_MF__
#define __MTCR_MF__
#include "mtcr_com_defs.h"
#ifdef __FreeBSD__
#include <sys/pciio.h>
#endif

typedef void (*f_mpci_change)(void* mf);

/*  All fields in follow structure are not supposed to be used */
/*  or modified by user programs. Except i2c_secondary that may be */
/*  modified before each access to target I2C secondary address */
struct mfile_t
{
    u_int16_t hw_dev_id;
    u_int16_t rev_id;
    MType tp;     /*  type of driver */
    MType res_tp; /*  Reserved type of driver for HCR usage */
    DType dtype;  /*  target device to access to */
    DType itype;  /*  interface device to access via */
    char real_name[DEV_NAME_SZ];
    cables_info ext_info; /*keeps info for calculate the correct secondary address (0x50 + offset) */
    unsigned char i2c_secondary;
    int gpio_en;
    void* ctx;
    void* ptr;
#ifdef __WIN__
    long mtusb_serial;
    MT_ulong_ptr_t fd;
    MT_ulong_ptr_t res_fd;
#else
    int is_vm;               /*  if the machine is VM    */
    io_region* iorw_regions; /* For LPC devices */
    int regions_num;
    char* dev_name;
    int fd;
    int res_fd;          /*  Will be used with HCR if need*/
    int is_mtserver_req; // request came from mtServer - means came from remote client
    void* bar_virtual_addr;
    unsigned int bar0_gw_offset; // for MST_BAR0_GW_PCI devices, offset from BAR0 - gateway - for R/W operations
    int file_lock_descriptor; // file descriptor to the lock file aka semaphore in order to protect parallel read/write
                              // GW operations
    void* fallback_mf;
    int old_mst; // for mst driver compatibility
    unsigned short mst_version_major;
    unsigned int mst_version_minor;
    unsigned int vsec_addr;
    u_int32_t vsec_cap_mask;
    void* ul_ctx;
    f_mpci_change mpci_change;
#endif
    mtcr_status_e icmd_support;
    unsigned int big_endian;      // NVIDIA devices support BE data, while Mellanox is LE.
    unsigned int cr_space_offset; // Default is 0. for NVIDIA devices starting from BW00 - might change.
    unsigned int map_size;
    int vsec_supp;
    int i2c_smbus;
    unsigned int i2c_RESERVED; /*  Reserved for internal usage (i2c internal) */
    enum Mdevs_t flags;
    u_int32_t connectx_wa_slot; /* apply connectx cr write workaround */
    int connectx_wa_last_op_write;
    u_int32_t connectx_wa_stat;
    u_int64_t connectx_wa_max_retries;
    u_int64_t connectx_wa_num_of_writes;
    u_int64_t connectx_wa_num_of_retry_writes;
    int server_ver_major;
    int server_ver_minor;
    dev_info* dinfo;
    icmd_params icmd;
    int address_space;
    tools_hcr_params hcr_params;      // for tools HCR access
    access_reg_params acc_reg_params; // for sending access registers
    void* dl_context;                 // Dynamic libs Ctx
    void* cable_ctx;
    void* cable_chip_ctx;
    int is_cable;
    unsigned int linkx_chip_devid;
    gearbox_info gb_info;
    retimer_info rt_info; // Retimers information, e.g Arcus-E
    MType remote_type;
    int sock; /*  in not -1 - remote interface */
    int using_ssh;
    int is_remote;
    void* ssh_utility_ctx;
    void* ssh_utility_lib;
#ifdef __FreeBSD__
    struct pcisel sel;
    unsigned int vpd_cap_addr;
    int wo_addr;
    int connectx_flush;
    int fdlock;
    struct page_list_fbsd user_page_list;
#else
    struct page_list user_page_list;
#endif
    void* dma_props; // For dma purpose
    int supports_predefined_tiles;
    addr_bound tile_address_map[MAX_TILE_NUM];
    void* mft_core_device;
};

typedef struct mfile_t mfile;

#endif // __MTCR_MF__
