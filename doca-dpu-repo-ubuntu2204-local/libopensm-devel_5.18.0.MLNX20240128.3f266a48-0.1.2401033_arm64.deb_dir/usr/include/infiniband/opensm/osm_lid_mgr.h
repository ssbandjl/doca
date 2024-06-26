/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2004-2009 Voltaire, Inc. All rights reserved.
 * Copyright (c) 2002-2005 Mellanox Technologies LTD. All rights reserved.
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
 * 	Declaration of osm_lid_mgr_t.
 *	This object represents the LID Manager object.
 *	This object is part of the OpenSM family of objects.
 */

#ifndef _OSM_LID_MGR_H_
#define _OSM_LID_MGR_H_

#include <complib/cl_passivelock.h>
#include <opensm/osm_base.h>
#include <opensm/osm_madw.h>
#include <opensm/osm_subnet.h>
#include <opensm/osm_db.h>
#include <opensm/osm_log.h>
#include <opensm/osm_vlid_mgr.h>
#include <opensm/osm_port.h>
#include <opensm/osm_nvlink.h>

#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else				/* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif				/* __cplusplus */

BEGIN_C_DECLS
#define OSM_LID_MGR_LIST_SIZE_MIN 256
/****h* OpenSM/LID Manager
* NAME
*	LID Manager
*
* DESCRIPTION
*	The LID Manager object encapsulates the information
*	needed to control LID assignments on the subnet.
*
*	The LID Manager object is thread safe.
*
*	This object should be treated as opaque and should be
*	manipulated only through the provided functions.
*
* AUTHOR
*	Steve King, Intel
*
*********/
struct osm_sm;

/****d* OpenSM: osm_reserved_lid_type
* NAME
*	osm_reserved_lid_type_t
*
* DESCRIPTION
*	Enumerates classification of reserved LIDs, such as LIDs that loaded
*	from persistent LIDs file (guid2lid) or LIDs that currently assigned
*	to devices in the subnet.
*
* SYNOPSIS
*/
typedef enum _osm_reserved_lid_type {
	OSM_RESERVED_LID_NONE = 0,
	OSM_RESERVED_LID_PERSISTENT = 1,
	OSM_RESERVED_LID_DISCOVERED = 2,
	OSM_RESERVED_GLOBAL_FLID = 4,
	OSM_RESERVED_NVLINK_LID = 8
} osm_reserved_lid_type_t;
/*
* NOTES
*	FLIDs and NVLIDs DO NOT coexist.
*	NVLink LIDs: ALIDs or GPU port LIDs.
*
*********/

#define OSM_RESERVED_SPECIAL_LID_USED_MASK	(OSM_RESERVED_LID_PERSISTENT | \
						 OSM_RESERVED_LID_DISCOVERED)

typedef struct lid_mgr_nvlink_data {
	osm_db_domain_t *p_g2alid;
	osm_db_domain_t *p_g2nvlid;
	cl_qlist_t free_alid_ranges;
	uint16_t alid_range_start;
	uint16_t alid_range_top;
	uint16_t num_alids_per_gpu;
	uint16_t p_start_ranges_per_plane[OSM_NVLINK_MAX_PLANES + 1];
	cl_qlist_t p_free_ranges_per_plane[OSM_NVLINK_MAX_PLANES + 1];
	uint16_t global_lid_range_top;
	uint16_t lid_range_top_per_plane[OSM_NVLINK_MAX_PLANES + 1];
	osm_db_domain_t *p_guid2planes;
} lid_mgr_nvlink_data_t;
/*
* FIELDS
*	p_g2alid
*		Pointer to the database domain storing GPU guid to anycast lid mapping.
*		Data is stored by guid, first alid, second alid.
*		As number of ALIDs is known to SM (currently hardcoded for Single Node),
*		and is planned to be up to 2 ALIDs per GPU.
*
* 	free_alid_ranges
*		A list of available free Anycast LID ranges.
*
* 	alid_range_start
*		Anycast LID range start LID.
*
*	alid_range_top
*		Maximal Anycast LID configured on the subnet. Set to switches by LFTSplit MAD.
*
* 	num_alids_per_gpu
*		Number of Anycast LIDs to be configured for each GPU.
* 
* 	p_start_ranges_per_plane
*		Array of LIDs, each refers to start LID range of a plane.
*
* 	p_free_ranges_per_plane
*		Array of range lists, indexed by plane.
*		Each list holds available free GPU port LID ranges.
*
*	p_guid2planes
*		Pointer to the database domain storing switch GUID to plane mapping.
*
* SEE ALSO
*	osm_lid_mgr_t
*********/

/****s* OpenSM: LID Manager/osm_lid_mgr_t
* NAME
*	osm_lid_mgr_t
*
* DESCRIPTION
*	LID Manager structure.
*
*	This object should be treated as opaque and should
*	be manipulated only through the provided functions.
*
* SYNOPSIS
*/
typedef struct osm_lid_mgr {
	struct osm_sm *sm;
	osm_subn_t *p_subn;
	osm_db_t *p_db;
	osm_log_t *p_log;
	cl_plock_t *p_lock;
	osm_db_domain_t *p_g2l;
	osm_db_domain_t *p_g2fl;
	cl_qlist_t free_ranges;
	cl_qlist_t free_flid_ranges;
	boolean_t dirty;
	uint8_t used_lids[IB_LID_UCAST_END_HO + 1];
	osm_vlid_mgr_t vlid_mgr;
	lid_mgr_nvlink_data_t nvlink_data;
} osm_lid_mgr_t;
/*
* FIELDS
*	sm
*		Pointer to the SM object.
*
*	p_subn
*		Pointer to the Subnet object for this subnet.
*
*	p_db
*		Pointer to the database (persistency) object
*
*	p_log
*		Pointer to the log object.
*
*	p_lock
*		Pointer to the serializing lock.
*
*	p_g2l
*		Pointer to the database domain storing guid to lid mapping.
*
*	p_g2fl
*		Pointer to the database domain storing switch guid to flid mapping.
*
*	free_ranges
*		A list of available free lid ranges. The list is initialized
*		by the code that initializes the lid assignment and is consumed
*		by the procedure that finds a free range. It holds elements of
*		type osm_lid_mgr_range_t
*
*	free_flid_range
*		A free flid range
*
*	dirty
*		 Indicates that lid table was updated
*
*	used_lids
*		 An array of used lids. keeps track of
*		 existing and non existing mapping of guid->lid
*
*	nvlink_data
*		Structure holds all NVLink configuration data that is relevant for LID Manager.
*
* SEE ALSO
*	LID Manager object
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_construct
* NAME
*	osm_lid_mgr_construct
*
* DESCRIPTION
*	This function constructs a LID Manager object.
*
* SYNOPSIS
*/
void osm_lid_mgr_construct(IN osm_lid_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to a LID Manager object to construct.
*
* RETURN VALUE
*	This function does not return a value.
*
* NOTES
*	Allows osm_lid_mgr_destroy
*
*	Calling osm_lid_mgr_construct is a prerequisite to calling any other
*	method except osm_lid_mgr_init.
*
* SEE ALSO
*	LID Manager object, osm_lid_mgr_init,
*	osm_lid_mgr_destroy
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_destroy
* NAME
*	osm_lid_mgr_destroy
*
* DESCRIPTION
*	The osm_lid_mgr_destroy function destroys the object, releasing
*	all resources.
*
* SYNOPSIS
*/
void osm_lid_mgr_destroy(IN osm_lid_mgr_t * p_mgr);
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
*	LID Manager object.
*	Further operations should not be attempted on the destroyed object.
*	This function should only be called after a call to
*	osm_lid_mgr_construct or osm_lid_mgr_init.
*
* SEE ALSO
*	LID Manager object, osm_lid_mgr_construct,
*	osm_lid_mgr_init
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_init
* NAME
*	osm_lid_mgr_init
*
* DESCRIPTION
*	The osm_lid_mgr_init function initializes a
*	LID Manager object for use.
*
* SYNOPSIS
*/
ib_api_status_t
osm_lid_mgr_init(IN osm_lid_mgr_t * p_mgr, IN struct osm_sm * sm);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_lid_mgr_t object to initialize.
*
*	sm
*		[in] Pointer to the SM object for this subnet.
*
* RETURN VALUES
*	CL_SUCCESS if the LID Manager object was initialized
*	successfully.
*
* NOTES
*	Allows calling other LID Manager methods.
*
* SEE ALSO
*	LID Manager object, osm_lid_mgr_construct,
*	osm_lid_mgr_destroy
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_process_sm
* NAME
*	osm_lid_mgr_process_sm
*
* DESCRIPTION
*	Configures the SM's port with its designated LID values.
*
* SYNOPSIS
*/
int osm_lid_mgr_process_sm(IN osm_lid_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_lid_mgr_t object.
*
* RETURN VALUES
*	Returns 0 on success and non-zero value otherwise.
*
* NOTES
*
* SEE ALSO
*	LID Manager
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_process_subnet
* NAME
*	osm_lid_mgr_process_subnet
*
* DESCRIPTION
*	Configures subnet ports (except the SM port itself) with their
*	designated LID values.
*
* SYNOPSIS
*/
int osm_lid_mgr_process_subnet(IN osm_lid_mgr_t * p_mgr);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_lid_mgr_t object.
*
* RETURN VALUES
*	Returns 0 on success and non-zero value otherwise.
*
* NOTES
*
* SEE ALSO
*	LID Manager
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_get_vport_lid
* NAME
*	osm_lid_mgr_get_vport_lid
*
* DESCRIPTION
*	Allocate lid for a given virtual port
*
* SYNOPSIS
*/
ib_net16_t osm_lid_mgr_get_vport_lid(IN osm_lid_mgr_t * p_mgr,
				     IN osm_port_t * p_port,
				     IN osm_vport_t * p_vport);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_lid_mgr_t object.
*
*	p_port
*		[in] Pointer to the port this vport belongs to.
*
*	p_mgr
*		[in] Pointer to the vport which requires the lid.
* RETURN VALUES
*	Assigned lid for the vport
*	Zero if failed to assign lid
*
* NOTES
*
* SEE ALSO
*	LID Manager
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_clear_vport_lid
* NAME
*	osm_lid_mgr_clear_vport_lid
*
* DESCRIPTION
*	Clear the lid for a given virtual port
*	Keeping the guid2lid mapping of this vport is optional
*
* SYNOPSIS
*/
void osm_lid_mgr_clear_vport_lid(IN osm_lid_mgr_t * p_mgr,
				 IN osm_port_t * p_port,
				 IN osm_vport_t * p_vport,
				 IN boolean_t clear_guid2lid);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_lid_mgr_t object.
*
*	p_port
*		[in] Pointer to the port this vport belongs to.
*
*	p_mgr
*		[in] Pointer to the vport.
*
*	clear_guid2lid
*		[in] Indication whether to clear also from giud2lid db
*
* RETURN VALUES
*	None
*
* NOTES
*
* SEE ALSO
*	LID Manager
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_get_switch_flid
* NAME
*	osm_lid_mgr_get_switch_flid
*
* DESCRIPTION
*	Get the flid for a given switch
*
* SYNOPSIS
*/
uint16_t osm_lid_mgr_get_switch_flid(IN osm_lid_mgr_t * p_mgr,
				     IN osm_port_t * p_port);

/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_lid_mgr_t object.
*
*	p_port
*		[in] Pointer to the switch end port object.
*
* RETURN VALUES
*	Acquired flid for the given switch
*
* NOTES
*
* SEE ALSO
*	LID Manager
*********/

/****f* OpenSM: LID Manager/osm_lid_mgr_is_flid
* NAME
*	osm_lid_mgr_is_flid
*
* DESCRIPTION
*	Returns whether the given LID is FLID
*
* SYNOPSIS
*/
boolean_t osm_lid_mgr_is_flid(IN osm_lid_mgr_t * p_mgr, IN uint16_t lid);
/*
* PARAMETERS
*	p_mgr
*		[in] Pointer to an osm_lid_mgr_t object.
*
*	lid
*		[in] A LID that is being checked.
*
* RETURN VALUES
*	None
*
* NOTES
*
* SEE ALSO
*	LID Manager
*********/
END_C_DECLS
#endif				/* _OSM_LID_MGR_H_ */
