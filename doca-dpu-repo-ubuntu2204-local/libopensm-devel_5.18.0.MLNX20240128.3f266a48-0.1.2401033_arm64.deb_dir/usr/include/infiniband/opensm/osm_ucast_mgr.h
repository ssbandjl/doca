/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2004-2009 Voltaire, Inc. All rights reserved.
 * Copyright (c) 2002-2009 Mellanox Technologies LTD. All rights reserved.
 * Copyright (c) 1996-2003 Intel Corporation. All rights reserved.
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
 * 	Declaration of osm_ucast_mgr_t.
 *	This object represents the Unicast Manager object.
 *	This object is part of the OpenSM family of objects.
 */

#ifndef _OSM_UCAST_MGR_H_
#define _OSM_UCAST_MGR_H_

#include <complib/cl_passivelock.h>
#include <complib/cl_qlist.h>
#include <opensm/osm_madw.h>
#include <opensm/osm_subnet.h>
#include <opensm/osm_switch.h>
#include <opensm/osm_log.h>
#include <opensm/osm_ucast_cache.h>
#include <complib/cl_mpthreadpool.h>
#include <stdlib.h>

#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else				/* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif				/* __cplusplus */

BEGIN_C_DECLS

/* possible values are 32, 64, 128, 256. Default is 128 */
#define SCATTER_BUF_SIZE 128

#define UCAST_MGR_MAX_GROUP_ID		0xFFFE
#define UCAST_MGR_INVALID_GROUP_ID	0xFFFF
#define UCAST_MGR_MAX_PLFT		3
#define UCAST_MGR_MAX_DFP2_UNIQUE_PLFT	2

/****h* OpenSM/Unicast Manager
* NAME
*	Unicast Manager
*
* DESCRIPTION
*	The Unicast Manager object encapsulates the information
*	needed to control unicast LID forwarding on the subnet.
*
*	The Unicast Manager object is thread safe.
*
*	This object should be treated as opaque and should be
*	manipulated only through the provided functions.
*
* AUTHOR
*	Steve King, Intel
*
*********/
struct osm_sm;
/****s* OpenSM: Unicast Manager/osm_ucast_mgr_t
* NAME
*	osm_ucast_mgr_t
*
* DESCRIPTION
*	Unicast Manager structure.
*
*	This object should be treated as opaque and should
*	be manipulated only through the provided functions.
*
* SYNOPSIS
*/
typedef struct osm_ucast_mgr {
	char *scatter_statebufs;
	struct random_data *scatter_bufs;
	struct osm_sm *sm;
	osm_subn_t *p_subn;
	osm_log_t *p_log;
	cl_plock_t *p_lock;
	cl_mp_thread_pool_t thread_pool;
	uint16_t max_lid;
	cl_qlist_t port_order_list;
	cl_qlist_t *port_order_lists;
	cl_qlist_t *sw_lists;
	boolean_t is_dor;
	boolean_t some_hop_count_set;
	cl_qmap_t cache_sw_tbl;
	cl_qmap_t cache_vlid_tbl;
	boolean_t cache_valid;
	boolean_t vlid_cache_valid;
	boolean_t variables_invalidate_cache;
	boolean_t variables_reinit_thread_pool;
	boolean_t rebalancing_required;
	struct timeval rerouting_timestamp;
	boolean_t ar_configured;
	int guid_routing_order_count;
	/* Adaptive routig group IDs */
	osm_db_domain_t *p_g2groupid;
	uint64_t used_group_ids[UCAST_MGR_MAX_GROUP_ID + 1];
	uint16_t lid_to_group_id[IB_LID_UCAST_END_HO + 1][UCAST_MGR_MAX_PLFT];
	uint16_t free_group_id_idx;
	osm_db_domain_t *p_flid2groupid;
} osm_ucast_mgr_t;
/*
* FIELDS
*	sm
*		Pointer to the SM object.
*
*	p_subn
*		Pointer to the Subnet object for this subnet.
*
*	p_log
*		Pointer to the log object.
*
*	p_lock
*		Pointer to the serializing lock.
*
*	max_lid
*		Max LID of all the switches in the subnet.
*
*	port_order_list
*		List of ports ordered for routing.
*
*	is_dor
*		Dimension Order Routing (DOR) will be done
*
*	some_hop_count_set
*		Initialized to FALSE at the beginning of each the min hop
*		tables calculation iteration cycle, set to TRUE to indicate
*		that some hop count changes were done.
*
*	cache_sw_tbl
*		Cached switches table.
*
*	cache_vlid_tbl
*		Cached vLIDs table
*
*	cache_valid
*		TRUE if the unicast cache is valid.
*
*	vlid_cache_valid
*		TRUE if the vLID cache is valid
*
*	variables_invalidate_cache
*		TRUE if one of the variables from config file, relevant to
*		the routing calculations, has changed.
*
*	variables_reinit_thread_pool
*		TRUE if one of the variables from config file, relevant to
*		the ucast_mgr_t thread pool, has changed.
*
*	rebalancing_required
*		TRUE if current routing is unbalanced
*
*	rerouting_timestamp
*		Last time rerouting happen during heavy sweep
*
*	ar_configured
*		Indicates that Adaptive Routing is configured on at least one
*		switch in the subnet.
*		Note: SHIELD does not affect this indicator.
*
*	guid_routing_order_count
*		Number of nodes in guid_routing_order_file
*
*	p_g2groupid
*		GUID to AR group ID persistent database.
*
* 	used_group_ids
*		 An array of used adaptive routing group ids.
*		 keeps track of used and free groups ids.
*
*	free_group_id_idx
*		Iterator for searching unused adaptive routing group id.
*
*	p_flid2groupid
*		Switch GUID, that was assigned with an FLID and needs a group id.
*
*
* SEE ALSO
*	Unicast Manager object
*********/

/****f* OpenSM: Unicast Manager/osm_ucast_mgr_construct
* NAME
*	osm_ucast_mgr_construct
*
* DESCRIPTION
*	This function constructs a Unicast Manager object.
*
* SYNOPSIS
*/
void osm_ucast_mgr_construct(IN osm_ucast_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to a Unicast Manager object to construct.
*
* RETURN VALUE
*	This function does not return a value.
*
* NOTES
*	Allows osm_ucast_mgr_destroy
*
*	Calling osm_ucast_mgr_construct is a prerequisite to calling any other
*	method except osm_ucast_mgr_init.
*
* SEE ALSO
*	Unicast Manager object, osm_ucast_mgr_init,
*	osm_ucast_mgr_destroy
*********/

/****f* OpenSM: Unicast Manager/osm_ucast_mgr_destroy
* NAME
*	osm_ucast_mgr_destroy
*
* DESCRIPTION
*	The osm_ucast_mgr_destroy function destroys the object, releasing
*	all resources.
*
* SYNOPSIS
*/
void osm_ucast_mgr_destroy(IN osm_ucast_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to the object to destroy.
*
* RETURN VALUE
*	This function does not return a value.
*
* NOTES
*	Performs any necessary cleanup of the specified
*	Unicast Manager object.
*	Further operations should not be attempted on the destroyed object.
*	This function should only be called after a call to
*	osm_ucast_mgr_construct or osm_ucast_mgr_init.
*
* SEE ALSO
*	Unicast Manager object, osm_ucast_mgr_construct,
*	osm_ucast_mgr_init
*********/

/****f* OpenSM: Unicast Manager/osm_ucast_mgr_init
* NAME
*	osm_ucast_mgr_init
*
* DESCRIPTION
*	The osm_ucast_mgr_init function initializes a
*	Unicast Manager object for use.
*
* SYNOPSIS
*/
ib_api_status_t osm_ucast_mgr_init(IN osm_ucast_mgr_t * p_mgr,
				   IN struct osm_sm * sm);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object to initialize.
*
*	sm
*		[in] Pointer to the SM object.
*
* RETURN VALUES
*	IB_SUCCESS if the Unicast Manager object was initialized
*	successfully.
*
* NOTES
*	Allows calling other Unicast Manager methods.
*
* SEE ALSO
*	Unicast Manager object, osm_ucast_mgr_construct,
*	osm_ucast_mgr_destroy
*********/

/****f* OpenSM: Unicast Manager/osm_ucast_mgr_set_fwd_tables
* NAME
*	osm_ucast_mgr_set_fwd_tables
*
* DESCRIPTION
*	Setup forwarding table for the switch (from prepared new_lft).
*
* SYNOPSIS
*/
void osm_ucast_mgr_set_fwd_tables(IN osm_ucast_mgr_t * p_mgr, IN osm_node_filter_t filter);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object.
*
*	filter
*		[in] Filter to select which switches to setup.
*
* SEE ALSO
*	Unicast Manager
*********/

/****f* OpenSM: Unicast Manager/ucast_mgr_setup_all_switches
* NAME
*	ucast_mgr_setup_all_switches
*
* DESCRIPTION
*
* SYNOPSIS
*/
int ucast_mgr_setup_all_switches(osm_subn_t * p_subn);
/*
* PARAMETERS
*	p_subn
*		[in] pointer to subnet object
*
* RETURN VALUE
*	Returns zero on success and negative value on failure.
*
* SEE ALSO
*	Unicast Manager
*********/

/****f* OpenSM: Unicast Manager/alloc_ports_priv
* NAME
*	alloc_ports_priv
*
* DESCRIPTION
*
* SYNOPSIS
*/
int alloc_ports_priv(osm_ucast_mgr_t * mgr, cl_qlist_t *port_list);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object.
*
*	port_list
*		[in] List of ports to alloc priv objects in.
*		     init_ports_priv should be called after
*		     alloc_ports_priv to reset allocated priv objects
*
* RETURN VALUE
*	Returns zero on success and negative value on failure.
*
* SEE ALSO
*	Unicast Manager
*********/

/****f* OpenSM: Unicast Manager/init_ports_priv
* NAME
*	init_ports_priv
*
* DESCRIPTION
*
* SYNOPSIS
*/
void init_ports_priv(osm_ucast_mgr_t * mgr, cl_qlist_t *port_list);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object.
*
*	port_list
*		[in] List of ports to initialize priv objects in
*
* SEE ALSO
*	Unicast Manager
*********/

/****f* OpenSM: Unicast Manager/free_ports_priv
* NAME
*	free_ports_priv
*
* DESCRIPTION
*
* SYNOPSIS
*/
void free_ports_priv(osm_ucast_mgr_t * mgr, uint32_t i);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object.
*	i
*		[in] Index of compute thread. It should be zero
*			for non-parallel routing engines
*
* SEE ALSO
*	Unicast Manager
*********/

/****f* OpenSM: Unicast Manager/osm_ucast_mgr_build_lid_matrices
* NAME
*	osm_ucast_mgr_build_lid_matrices
*
* DESCRIPTION
*	Build switches's lid matrices.
*
* SYNOPSIS
*/
int osm_ucast_mgr_build_lid_matrices(IN osm_ucast_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object.
*
* NOTES
*	This function processes the subnet, configuring switches'
*	min hops tables (aka lid matrices).
*
* SEE ALSO
*	Unicast Manager
*********/

/****f* OpenSM: Unicast Manager/osm_ucast_mgr_build_lft_tables
* NAME
*	osm_ucast_mgr_build_lft_tables
*
* DESCRIPTION
*	Build switches's lft tables.
*
* SYNOPSIS
*/
int osm_ucast_mgr_build_lft_tables(IN osm_ucast_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object.
*
* NOTES
*	This function processes minhop tables, configuring switches
*	lft tables.
*
* SEE ALSO
*	Unicast Manager
*********/

/****f* OpenSM: Unicast Manager/osm_ucast_mgr_process
* NAME
*	osm_ucast_mgr_process
*
* DESCRIPTION
*	Process and configure the subnet's unicast forwarding tables.
*
* SYNOPSIS
*/
int osm_ucast_mgr_process(IN osm_ucast_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_ucast_mgr_t object.
*
* RETURN VALUES
*	Returns zero on success and negative value on failure.
*
* NOTES
*	This function processes the subnet, configuring switch
*	unicast forwarding tables.
*
* SEE ALSO
*	Unicast Manager, Node Info Response Controller
*********/

int osm_ucast_add_guid_to_order_list(void *, uint64_t, char *);
void osm_ucast_clear_prof_ignore_flag(cl_map_item_t * const, void *);
int osm_ucast_mark_ignored_port(void *, uint64_t, char *);
void osm_ucast_add_port_to_order_list(cl_map_item_t * const, void *);
void osm_ucast_process_tbl(IN cl_map_item_t * const p_map_item, IN void *context);
int alloc_random_bufs(osm_ucast_mgr_t * p_mgr, int num_of_bufs);
int ucast_dummy_build_lid_matrices(void *context);
int osm_ucast_mgr_calculate_missing_routes(IN OUT osm_ucast_mgr_t * p_mgr);
int osm_ucast_mgr_init_port_order_lists(IN osm_ucast_mgr_t * p_mgr,
					IN uint32_t number_of_thread);
void osm_ucast_mgr_release_port_order_lists(IN osm_ucast_mgr_t * p_mgr,
					    IN uint32_t number_of_thread);

void osm_ucast_mgr_reset_lfts_all_switches(osm_subn_t * p_subn);
 
void osm_ucast_mgr_update_groupid_maps(osm_ucast_mgr_t *p_mgr);
uint16_t osm_ucast_mgr_get_groupid(osm_ucast_mgr_t * p_mgr, uint16_t lid, uint8_t plft);
uint16_t osm_ucast_mgr_make_groupid(osm_ucast_mgr_t * p_mgr, uint16_t lid, uint8_t plft);
uint16_t osm_ucast_mgr_find_free_groupid(osm_ucast_mgr_t * p_mgr);
int osm_ucast_mgr_set_groupid(osm_ucast_mgr_t * p_mgr, uint16_t lid, uint16_t group_id, uint8_t plft);
int osm_ucast_mgr_remove_groupid(osm_ucast_mgr_t * p_mgr, uint16_t lid);
void osm_ucast_mgr_group_id_db_store(IN osm_ucast_mgr_t * p_mgr);
void osm_ucast_mgr_thread_pool_init(IN osm_ucast_mgr_t * p_mgr);

void osm_ucast_mgr_mp_thread_pool_init(IN osm_ucast_mgr_t * p_mgr,
				       IN cl_mp_thread_pool_t * mp_thread_pool,
				       IN const char * name,
				       OUT uint32_t * number_of_threads);

END_C_DECLS
#endif				/* _OSM_UCAST_MGR_H_ */
