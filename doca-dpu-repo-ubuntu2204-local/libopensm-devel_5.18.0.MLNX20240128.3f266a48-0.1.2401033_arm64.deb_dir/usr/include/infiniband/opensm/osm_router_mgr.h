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

/*
 * Abstract:
 * 	Declaration of osm_router_mgr_t.
 *	This object represents the Router Manager object.
 *	This object is part of the OpenSM family of objects.
 */
#ifndef _OSM_ROUTER_MGR_H_
#define _OSM_ROUTER_MGR_H_

#include <complib/cl_passivelock.h>
#include <complib/cl_qlist.h>
#include <opensm/osm_madw.h>
#include <opensm/osm_subnet.h>
#include <opensm/osm_router.h>
#include <opensm/osm_log.h>
#include <stdlib.h>

#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else				/* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif				/* __cplusplus */

BEGIN_C_DECLS

typedef struct osm_router_mgr {
	osm_subn_t *p_subn;
	osm_log_t * p_log;
	cl_fmap_t prefix_rtr_table;
	cl_fmap_t flid_to_rtrs;
	uint16_t ri_routers_changes_bitmask;
} osm_router_mgr_t;

/*
* FIELDS
*	p_subn
*		Pointer to the Subnet object for this subnet.
*
*	p_log
*		Pointer to the log object.
*
*	prefix_rtr_table
*		table of <subnet prefix, pkey> -> list of routers that have
*		these subnets and pkeys
*
*	flid_to_rtrs
*		a mapping of FLID to the list of routers that have this FLID enabled.
*
*	ri_routers_changes_bitmask
*		A bitmask describing changes that have occurred in the routers of the subnet
*		as reported by router info MAD.
*
* SEE ALSO
*	Router Manager object
*********/

/****s* OpenSM: Router Manager/osm_router_prefix_rtrs_map_t
* NAME
*	osm_router_prefix_rtrs_map_t
*
* DESCRIPTION
* 	Container of a map of pointers to routers that OpenSM should try to use
* 	when given subnet prefix + p_key
* 	as roots when calculating multicast trees.
* SYNOPSIS
*/
typedef struct osm_router_prefix_rtrs_map {
	cl_fmap_item_t map_item;
	cl_map_t rtrs;
	ib_net64_t * rtr_guid_arr;
	uint16_t num_valid_routers;
} osm_router_prefix_rtrs_map_t;
/*
* FIELDS
*	map_item
*		Linkage structure for cl_fmap.  MUST BE FIRST MEMBER!
*
*	rtrs
*		map of router guids.
*
*	rtr_guid_arr
*		Array of router guids for each prefix and pkey combination.
*
*	num_valid_routers
*		Number of valid routers is rtr_guid_arr length.
*
* SEE ALSO
*	Router Manager object
*********/

/****s* OpenSM: Router Manager/osm_subnet_prefix_pkey_t
* NAME
*	osm_subnet_prefix_pkey_t
*
* DESCRIPTION
*	subnet prefix p_key pair
* SYNOPSIS
*/
typedef struct osm_subnet_prefix_pkey {
	ib_net64_t subnet_prefix;
	ib_net16_t pkey;
} osm_subnet_prefix_pkey_t;
/*
* FIELDS
*	subnet_prefix
*		subnet_prefix published in the routing table
*
*	p_key
*		p_key published for that subnet prefix in the routing table
*
* SEE ALSO
*	Router Manager
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_init
* NAME
*	osm_router_mgr_init
*
* DESCRIPTION
*	Initializes Router Manger object.
*
* SYNOPSIS
*/
ib_api_status_t osm_router_mgr_init(IN osm_router_mgr_t * p_mgr, IN struct osm_sm * sm);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
*	sm
*		[in] Pointer to the SM object.
*
* RETURN VALUES
*	IB_SUCCESS if the Router Manager object was initialized successfully.
*
* SEE ALSO
*	Router Manager object
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_destroy
* NAME
*	osm_router_mgr_destroy
*
* DESCRIPTION
*	Cleans resources that were allocated for Router Manger object.
*
*	Further operations should not be attempted on the destroyed object.
*	This function should only be called after a call to osm_router_mgr_init
*
* SYNOPSIS
*/
void osm_router_mgr_destroy(IN osm_router_mgr_t * p_mgr, IN struct osm_sm * sm);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
*	sm
*		[in] Pointer to the SM object.
*
* RETURN VALUES
* 	This function does not return a value.
*
* SEE ALSO
*	Router Manager object
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_destroy_prefix_router_db
* NAME
*	osm_router_mgr_destroy_prefix_router_db
*
* DESCRIPTION
*	The osm_router_mgr_destroy_prefix_router_db function destroys a prefix router db, releasing
*	all resources.
*
* SYNOPSIS
*/
void osm_router_mgr_destroy_prefix_router_db(IN osm_router_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
* RETURN VALUE
*	This function does not return a value.
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_destroy_flid_router_db
* NAME
*	osm_router_mgr_destroy_flid_router_db
*
* DESCRIPTION
*	The osm_router_mgr_destroy_flid_router_db destroys the flid to router db, releasing
*	all resources.
*
* SYNOPSIS
*/
void osm_router_mgr_destroy_flid_router_db(IN osm_router_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
* RETURN VALUE
*	This function does not return a value.
*********/

/****f* OpenSM: Router Manager/osm_get_routers_by_subnet_prefix
* NAME
*	osm_get_routers_by_subnet_prefix
*
* DESCRIPTION
*	This looks for the given subnet prefix in the routers table by subnet prefix
*	NOTE: this code is not thread safe. Need to grab the lock before
*	calling it.
*
* SYNOPSIS
*/
osm_router_prefix_rtrs_map_t *
osm_get_routers_by_subnet_prefix(IN osm_router_mgr_t * p_mgr,
				 osm_subnet_prefix_pkey_t *p_subnet_pkey);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
*	p_subnet_pkey
*		[in] The subnet pkey pair
*
* RETURN VALUES
*	The routers map pointer if found. NULL otherwise.
*********/

/****f* OpenSM: OpenSM/osm_router_mgr_free_rtr_map_item
* NAME
*       osm_router_mgr_free_rtr_map_item
*
* DESCRIPTION
*       free input parameter
*
* SYNOPSIS
*/
void osm_router_mgr_free_rtr_map_item(IN osm_router_prefix_rtrs_map_t * p_item);
/*
* PARAMETERS
*      p_item
*               [in] Pointer to the map item.
*
* RETURN VALUE
*       None
*
* SEE ALSO
*
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_prefix_router_tbl_remove_rtr
* NAME
*	osm_router_mgr_prefix_router_tbl_remove_rtr
*
* DESCRIPTION
*	The osm_router_mgr_prefix_router_tbl_remove_rtr function removes a router
*	from all subnet prefixes.
*
* SYNOPSIS
*/
void osm_router_mgr_prefix_router_tbl_remove_rtr(IN osm_router_mgr_t *p_mgr,
						 IN struct osm_router * p_rtr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
*	p_rtr
*		[in] Pointer to a router object,
*		which router table to remove.
*
* RETURN VALUE
*	This function does not return a value.
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_set_rtr_lid_tables
* NAME
*	osm_router_mgr_set_rtr_lid_tables
*
* DESCRIPTION
*	The osm_router_mgr_set_rtr_lid_tables sets the LID tables on the routers
*
* SYNOPSIS
*/
void osm_router_mgr_set_rtr_lid_tables(IN osm_router_mgr_t *p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
* RETURN VALUE
*	This function does not return a value.
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_process
* NAME
*	osm_router_mgr_process
*
* DESCRIPTION
*	The osm_router_mgr_process processes all the routers
*
* SYNOPSIS
*/
void osm_router_mgr_process(IN osm_router_mgr_t *p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
*
* RETURN VALUE
*	This function does not return a value.
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_is_foreign_flid_enabled
* NAME
*	osm_router_mgr_is_foreign_flid_enabled
*
* DESCRIPTION
*	The osm_router_mgr_is_foreign_flid_enabled checks if the given FLID is enabled
*
* SYNOPSIS
*/
boolean_t osm_router_mgr_is_foreign_flid_enabled(IN osm_router_mgr_t *p_mgr, uint16_t flid);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
* 	flid
* 		[in] Given FLID
*
* RETURN VALUE
*	This function returns true if the given FLID is enabled on any router.
*********/

/****f* OpenSM: Router Manager/osm_router_mgr_process_changes
* NAME
*	osm_router_mgr_process_changes
*
* DESCRIPTION
*	osm_router_mgr_process_changes processes all the routers when only FLIDs were changed.
*
* SYNOPSIS
*/
void osm_router_mgr_process_changes(IN osm_router_mgr_t *p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to Router Manger object.
*
* RETURN VALUE
*	This function does not return a value.
*********/

END_C_DECLS
#endif				/* _OSM_ROUTER_MGR_H_ */
