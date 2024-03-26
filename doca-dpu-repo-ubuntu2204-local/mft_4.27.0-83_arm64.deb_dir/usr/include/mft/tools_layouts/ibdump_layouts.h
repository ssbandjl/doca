
/*                  - Mellanox Confidential and Proprietary -
 *
 *  Copyright (C) 2010-2011, Mellanox Technologies Ltd.  ALL RIGHTS RESERVED.
 *
 *  Except as specifically permitted herein, no portion of the information,
 *  including but not limited to object code and source code, may be reproduced,
 *  modified, distributed, republished or otherwise exploited in any form or by
 *  any means for any purpose without the prior written permission of Mellanox
 *  Technologies Ltd. Use of software subject to the terms and conditions
 *  detailed in the file "LICENSE.txt".
 *
 */
 

/***
         *** This file was generated at "2023-04-16 18:41:04"
         *** by:
         ***    > /auto/mswg/release/tools/a-me/a-me-1.2.2/a-me-1.2.2-24/adabe_plugins/adb2c/adb2pack.py --input adb/connectib/connectib.adb --file-prefix ibdump --prefix ibdump_ --no-adb-utils
         ***/
#ifndef IBDUMP_LAYOUTS_H
#define IBDUMP_LAYOUTS_H


#ifdef __cplusplus
extern "C" {
#endif

#include "adb_to_c_utils.h"
/* Description -   */
/* Size in bytes - 16 */
struct ibdump_icmd_set_port_sniffer {
/*---------------- DWORD[1] (Offset 0x4) ----------------*/
	/* Description -  */
	/* 0x4.0 - 0x4.0 */
	u_int8_t sx_rx_;
/*---------------- DWORD[1] (Offset 0x4) ----------------*/
	/* Description -  */
	/* 0x4.16 - 0x4.16 */
	u_int8_t attach_detach_;
/*---------------- DWORD[2] (Offset 0x8) ----------------*/
	/* Description -  */
	/* 0x8.24 - 0x8.31 */
	u_int8_t port;
/*---------------- DWORD[3] (Offset 0xc) ----------------*/
	/* Description -  */
	/* 0xc.0 - 0xc.23 */
	u_int32_t sniffer_qpn;
};

/* Description -   */
/* Size in bytes - 16 */
union ibdump_ibdump_Nodes {
/*---------------- DWORD[0] (Offset 0x0) ----------------*/
	/* Description -  */
	/* 0x0.0 - 0xc.31 */
	struct ibdump_icmd_set_port_sniffer icmd_set_port_sniffer;
};


/*================= PACK/UNPACK/PRINT FUNCTIONS ======================*/
/* icmd_set_port_sniffer */
void ibdump_icmd_set_port_sniffer_pack(const struct ibdump_icmd_set_port_sniffer *ptr_struct, u_int8_t *ptr_buff);
void ibdump_icmd_set_port_sniffer_unpack(struct ibdump_icmd_set_port_sniffer *ptr_struct, const u_int8_t *ptr_buff);
void ibdump_icmd_set_port_sniffer_print(const struct ibdump_icmd_set_port_sniffer *ptr_struct, FILE *fd, int indent_level);
unsigned int ibdump_icmd_set_port_sniffer_size(void);
#define IBDUMP_ICMD_SET_PORT_SNIFFER_SIZE    (0x10)
void ibdump_icmd_set_port_sniffer_dump(const struct ibdump_icmd_set_port_sniffer *ptr_struct, FILE *fd);
/* ibdump_Nodes */
void ibdump_ibdump_Nodes_pack(const union ibdump_ibdump_Nodes *ptr_struct, u_int8_t *ptr_buff);
void ibdump_ibdump_Nodes_unpack(union ibdump_ibdump_Nodes *ptr_struct, const u_int8_t *ptr_buff);
void ibdump_ibdump_Nodes_print(const union ibdump_ibdump_Nodes *ptr_struct, FILE *fd, int indent_level);
unsigned int ibdump_ibdump_Nodes_size(void);
#define IBDUMP_IBDUMP_NODES_SIZE    (0x10)
void ibdump_ibdump_Nodes_dump(const union ibdump_ibdump_Nodes *ptr_struct, FILE *fd);


#ifdef __cplusplus
}
#endif

#endif // IBDUMP_LAYOUTS_H
