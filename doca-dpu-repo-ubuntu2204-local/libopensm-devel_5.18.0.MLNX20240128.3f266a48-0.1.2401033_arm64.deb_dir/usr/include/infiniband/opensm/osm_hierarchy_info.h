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
 * 	Declaration of hierarchy information and planarization related functionality.
 *
 * Author: Julia Levin, Nvidia
 */

#ifndef _OSM_HIERARCHY_INFO_H_
#define _OSM_HIERARCHY_INFO_H_

#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else				/* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif				/* __cplusplus */

BEGIN_C_DECLS
/*
 * Forward references.
 */
struct osm_port;
struct osm_physp;
struct osm_node;
struct osm_sm;

/*
 * Template GUID that describes Hierarchy Info for NDR generation switches and HCA.
 * This template GUID specified in the architecture specification "Hierarchy Information Use-Cases"
 */
#define OSM_HI_TEMPLATE_GUID_NDR	3

typedef enum osm_hierarchical_info_template_guid_ndr {
	HI_TEMPLATE_GUID_NDR_RECORD_SELECTOR_SPLIT = 0,
	HI_TEMPLATE_GUID_NDR_RECORD_SELECTOR_PORT,
	HI_TEMPLATE_GUID_NDR_RECORD_SELECTOR_CAGE,
	HI_TEMPLATE_GUID_NDR_RECORD_SELECTOR_ASIC,
	HI_TEMPLATE_GUID_NDR_RECORD_SELECTOR_SLOT,
	HI_TEMPLATE_GUID_NDR_RECORD_SELECTOR_TYPE,
	HI_TEMPLATE_GUID_NDR_RECORD_SELECTOR_BDF
} osm_hierarchical_info_template_guid_ndr_t;

/*
 * Template GUID that describes Hierarchy Info for XDR generation switches and HCA.
 * This template GUID specified in the architecture specification
 * "Hierarchy Information for planarize topology".
 */
#define OSM_HI_TEMPLATE_GUID_4		4

#define OSM_HI_TEMPLATE_GUID_5		5

typedef enum osm_hierarchical_info_template_guid_4 {
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_PORT_TYPE = 0,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_ASIC_NAME,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_IB_PORT,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_SWITCH_CAGE,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_IPIL,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_SPLIT,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_ASIC,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_TYPE = 8,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_IS_CAGE_MANAGER,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_PLANE,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_NUM_OF_PLANES,
	HI_TEMPLATE_GUID_4_RECORD_SELECTOR_APORT
} osm_hierarchical_info_template_guid_4_t;

typedef enum osm_hierarchical_info_template_guid_5 {
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_PORT_TYPE = 0,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_NUM_ON_BASE_BOARD,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_IB_PORT,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_CAGE,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_IPIL,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_SPLIT,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_BDF = 9,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_PLANE,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_NUM_OF_PLANES,
	HI_TEMPLATE_GUID_5_RECORD_SELECTOR_APORT
} osm_hierarchical_info_template_guid_5_t;

#define OSM_HI_MAX_NUM_OF_ASICS				4

#define OSM_PORT_LABEL_LEN				100

typedef struct osm_hierarchy_info {
	ib_hierarchy_info_t block;
	boolean_t is_planarized;
	char port_label[OSM_PORT_LABEL_LEN];
	int aport;
	int plane_number;
	int num_of_planes;
	int asic;
	int port_location_type;
	osm_aport_t *p_aport;
} osm_hierarchy_info_t;

/****f* OpenSM: Hierarchy Information/osm_get_hierarchy_info
* NAME
*	osm_get_hierarchy_info
*
* DESCRIPTION
*	Get the hierarchy info MAD for specific port.
*
* SYNOPSIS
*/
void osm_get_hierarchy_info(IN struct osm_sm * sm, IN struct osm_node * p_node,
			    IN struct osm_port * p_port,
			    IN struct osm_physp * p_physp,
			    IN uint8_t index);
/*
* PARAMETERS
*	p_sm
*		[in] Pointer to an osm_sm_t object.
*
*	p_node
*		[in] Pointer to the parent Node object of this Physical Port.
*
*	p_port
*		[in] Pointer to a pointer to a Port object of this Physical port.
*
*	p_physp
*		[in] Pointer to an osm_physp_t object.
*
* 	index
* 		[in] hierarchy index of the block to get.
*
* RETURN VALUES
*	This function does not return a value.
*
* NOTES
*
* SEE ALSO
*	Port, Physical Port
*********/

/****f* OpenSM: Hierarchy Information/osm_hierarchy_info_init
* NAME
*	osm_hierarchy_info_init
*
* DESCRIPTION
*	Initialize the hierarchy info structure.
*
* SYNOPSIS
*/
static inline void osm_hierarchy_info_init(IN OUT osm_hierarchy_info_t* p_hi)
{
	p_hi->aport = -1;
	p_hi->plane_number = -1;
	p_hi->num_of_planes = -1;
	p_hi->asic = -1;
	p_hi->port_location_type = -1;
}
/*
* PARAMETERS
*	p_hi
*		[in] Pointer to an osm_hierarchy_info_t object.
*
* RETURN VALUES
*	This function does not return a value.
*
* NOTES
*
* SEE ALSO
*********/

/****f* OpenSM: Hierarchy Information/osm_build_aports
* NAME
*	osm_build_aports
*
* DESCRIPTION
*	This function builds system and their aports based on hierarchy info in the ports.
*
* SYNOPSIS
*/
void osm_build_aports(IN OUT struct osm_sm * sm);
/*
* PARAMETERS
*	p_sm
*		[in] Pointer to an osm_sm_t object.
*
* RETURN VALUES
*	This function does not return a value.
*
* NOTES
*
* SEE ALSO
*********/

/****f* OpenSM: Hierarchy Information/osm_analyze_systems_aports
* NAME
*	osm_analyze_systems_aports
*
* DESCRIPTION
*	This function analyzes the aports for asymmetricity.
*
* SYNOPSIS
*/
void osm_analyze_systems_aports(IN OUT struct osm_sm * sm);
/*
* PARAMETERS
*	p_sm
*		[in] Pointer to an osm_sm_t object.
*
* RETURN VALUES
*	This function does not return a value.
*
* NOTES
*
* SEE ALSO
*********/

/****f* OpenSM: Hierarchy Information/osm_hierarchy_is_planarized_physp
* NAME
*	osm_hierarchy_is_planarized_physp
*
* DESCRIPTION
*	This function returns TRUE if the port has a valid planarization information.
*
* SYNOPSIS
*/
boolean_t osm_hierarchy_is_planarized_physp(IN const struct osm_physp * p_physp);
/*
* PARAMETERS
*	p_physp
*		[in] Pointer to an osm_physp object.
*
* RETURN VALUES
*	TRUE if the physical port has a valid planarized information, FALSE otherwise.
*
* NOTES
*
* SEE ALSO
*********/

/****f* OpenSM: Hierarchy Information/osm_system_delete
* NAME
*	osm_system_delete
*
* DESCRIPTION
*	Destroys and deallocates the object.
*
* SYNOPSIS
*/
void osm_system_delete(IN OUT osm_system_t ** p_system);
/*
* PARAMETERS
*	p_system
*		[in] Pointer to the object to destroy.
*
* RETURN VALUE
*	None.
*
* NOTES
*
* SEE ALSO
*	System object, osm_system_new
*********/

static inline boolean_t osm_aport_is_valid(IN osm_aport_t *p_aport)
{
	return (p_aport->min_found_plane_number != 0);
}

static inline boolean_t osm_hierarchy_is_asymmetric_aport(IN osm_aport_t *p_aport)
{
	return p_aport->asymmetric;
}

boolean_t osm_hierarchy_is_asymmetric_aport_physp(IN struct osm_physp *p_physp);

/****f* OpenSM: Hierarchy Information/osm_build_system_ar_group
* NAME
*	osm_build_system_ar_group
*
* DESCRIPTION
*	Builds routing from prism switch towards non-prism (ex. Black Mamba) Leaf's HCAs.
*	This functionality is applied on routing tables after routing engine calculations.
*
* SYNOPSIS
*/
void osm_build_system_ar_group(IN osm_subn_t *p_subn);
/*
* PARAMETERS
*	p_subn
*		[in] Pointer to osm_subn_t object
*
* RETURN VALUE
*	None.
*
* NOTES
*
* SEE ALSO
*********/

END_C_DECLS

#endif
