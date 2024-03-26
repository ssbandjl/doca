/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _OSM_TENANT_H_
#define _OSM_TENANT_H_

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <complib/cl_ptr_vector.h>
#include <complib/cl_hashmap.h>
#include <opensm/osm_subnet.h>


#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else				/* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif				/* __cplusplus */

BEGIN_C_DECLS

typedef	struct osm_sm		osm_sm_t;
typedef	struct osm_log		osm_log_t;
typedef	struct osm_opensm	osm_opensm_t;

/* -------------------------------------------------------------------------------------------------
 *
 * Internal typedefs
 */
typedef cl_hashmap_t		osm_tenant_guid_list_t;

typedef struct {
	osm_tenant_guid_list_t	virtual_guids;
	osm_tenant_guid_list_t	physical_guids;
} osm_tenant_t;

typedef struct {
	osm_log_t		*p_log;
	osm_tenant_t		*p_tenant;		/* struct being filled in by parser */
	uint64_t		map_id;			/* id for current map being updated */
	osm_tenant_guid_list_t	*p_list;		/* current list being updated */
} osm_tenant_parser_t;

/* -------------------------------------------------------------------------------------------------
 *
 * API functions and definitions
 */

typedef struct {
	osm_log_t		*p_log;
	osm_tenant_t		*p_tenant;
	char			*filename;
	uint32_t		crc;
} osm_tenant_mgr_t;

int osm_tenant_guidlist_insert_guid(osm_tenant_guid_list_t *p_list, uint64_t guid, uint64_t map_id);

ib_api_status_t osm_tenant_mgr_init(osm_tenant_mgr_t *p_tenant_mgr, osm_sm_t *p_sm);
void osm_tenant_mgr_destroy(osm_tenant_mgr_t *p_tenant_mgr, osm_sm_t *p_sm);
int osm_tenant_mgr_rescan(osm_subn_t *p_subn);
void osm_tenant_mgr_config_change(osm_subn_t *p_subn, void *p_value);
int osm_tenant_mgr_validate(osm_sm_t *p_sm, uint64_t virtual_guid, uint64_t physical_guid, uint16_t vport_index);

END_C_DECLS

#endif				/* _OSM_TENANT_H_ */
