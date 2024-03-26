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
 *	Module for NVLink discovery
 *
 * Author:
 *	Or Nechemia, NVIDIA
 */

#ifndef OSM_NVLINK_H
#define OSM_NVLINK_H

#include <iba/ib_types.h>
#include <complib/cl_types_osd.h>
#include <complib/cl_dispatcher.h>
#include <opensm/osm_subnet.h>
#include <opensm/osm_log.h>
#include <opensm/osm_gpu.h>

/* Some hardcoded values for Single Node (SN) case of NVLink */
#define OSM_NVLINK_MIN_PLANES			1
#define OSM_NVLINK_MAX_PLANES			18
#define OSM_NVLINK_NUM_ALIDS_PER_GPU		1
#define OSM_NVLINK_MAX_NUM_ALIDS_PER_GPU	2
/* For SN topologies, LID ranges of all LID types are of predefined length */
#define OSM_NVLINK_GLOBAL_LID_RANGE_LEN		0x400
#define OSM_NVLINK_NVLID_RANGE_LEN		0x800
#define OSM_NVLINK_GPU_LMC			0
#define OSM_NVLINK_SN_NUM_RAILS			9
#define OSM_NVLINK_NO_PLANE			0
#define OSM_NVLINK_NO_RAIL			0xFF

#define OSM_RAIL_FILTER_SUBN			0
#define OSM_RAIL_FILTER_CALC			1

typedef struct osm_nvlink_sw_data {
	uint8_t plane;
	ib_rail_filter_config_t *subn_rail_filter_cfg;
	ib_rail_filter_config_t *calc_rail_filter_cfg;
	uint8_t max_ingress_block;
	uint8_t max_egress_block;
} osm_nvlink_sw_data_t;

/****d* OpenSM: NVLink/osm_nvlink_port_type_t
* NAME
*       osm_nvlink_port_type_t
*
* DESCRIPTION
* 	Enumerates port type for NVLink, in which same port can be referred
* 	in differnt point of view.
*
* SYNOPSIS
*/
typedef enum _osm_nvlink_port_type
{
	IB_PORT_TYPE_INGRESS,
	IB_PORT_TYPE_EGRESS,
	IB_PORT_TYPE_MAX,
} osm_nvlink_port_type_t;
/***********/

void osm_nvlink_sw_data_init(struct osm_switch * p_sw);
void osm_nvlink_sw_data_destroy(osm_nvlink_sw_data_t * nvlink_sw_data);

/****f* OpenSM: NVLink/osm_nvlink_send_set_alid_info
* NAME
*	osm_nvlink_send_set_alid_info
*
* DESCRIPTION
*	Send calculated ALIDInfo to input GPU.
*	Call this function AFTER ALIDs are already configured into GPU's ALIDInfo structure
*	Returns status indicates if MAD was sent successfully.
*
* SYNOPSIS
*/
ib_api_status_t osm_nvlink_send_set_alid_info(struct osm_sm * sm, struct osm_gpu * p_gpu);
/*
* PARAMETERS
*	p_sm
*		Pointer to an osm_sm_t object
*
*	p_gpu
*		Pointer to the GPU.
*
* RETURN VALUES
*	IB_SUCCESS status, if MAD was sent successfully. Otherwise, relevant error status.
*
* SEE ALSO
* 	ib_alid_info_t
*********/


/****f* OpenSM: NVLink/osm_nvlink_set_gpu_extended_node_info
* NAME
*	osm_nvlink_set_gpu_extended_node_info
*
* DESCRIPTION
*	Send calculated ExtendedNodeInfo to input GPU.
*	Returns status indicates if MAD was sent successfully.
*
* SYNOPSIS
*/
ib_api_status_t osm_nvlink_set_gpu_extended_node_info(struct osm_sm * sm, struct osm_gpu * p_gpu);
/*
* PARAMETERS
*	p_sm
*		Pointer to an osm_sm_t object
*
*	p_gpu
*		Pointer to the GPU.
*
* RETURN VALUES
*	IB_SUCCESS status, if MAD was sent successfully. Otherwise, relevant error status.
*
* SEE ALSO
* 	ib_mlnx_ext_node_info_t
*********/

/****f* OpenSM: NVLink/osm_nvlink_init
* NAME
*	osm_nvlink_init
*
* DESCRIPTION
*	Function for initialization of NVLink data parameters relevant for LID Manager.
*	Returns status indicates if was initialized successfully.
*
* SYNOPSIS
*/
ib_api_status_t osm_nvlink_init(osm_subn_t * p_subn);
/*
* PARAMETERS
*	p_subn
*		[in] Pointer to an osm_subn_t object
*
* RETURN VALUES
*	IB_SUCCESS status, if initialized successfully. Otherwise, relevant error status.
*
* SEE ALSO
* 	osm_lid_mgr_t, lid_mgr_nvlink_data_t
*********/

/****f* OpenSM: NVLink/osm_nvlink_discovery
* NAME
*	osm_nvlink_discovery
*
* DESCRIPTION
*	Function for post discovery required for NVLink configuration.
*	Returns status indicates if was successfully done.
*
* SYNOPSIS
*/
ib_api_status_t osm_nvlink_discovery(osm_subn_t * p_subn);
/*
* PARAMETERS
*	p_subn
*		[in] Pointer to an osm_subn_t object
*
* RETURN VALUES
*	Status indicates if NVLink preprocess was done successfully.
*
* SEE ALSO
*********/

/****f* OpenSM: NVLink/osm_nvlink_get_rail_filter_mad
* NAME
*	osm_nvlink_get_rail_filter_mad
*
* DESCRIPTION
*	Returns RailFilterConfig MAD block relevant to input rail, ingress block and egress block
*
* SYNOPSIS
*/
ib_rail_filter_config_t * osm_nvlink_get_rail_filter_mad(struct osm_switch * p_sw,
							 uint8_t rail,
							 uint8_t ingress_block,
							 uint8_t egress_block,
							 boolean_t is_calc);
/*
* PARAMETERS
*	p_sw
*		Pointer to a Switch object
*
*	rail
*		Rail number
*
*	ingress_block
*		Ingress block number, as used by RailFilterConfig attribute modifier
*
*	egress_block
*		Egress block number, as used by RailFilterConfig attribute modifier
*
*	is_calc
*		Boolean indicates whether the get refers to lastly calcuted
*		RailFilterConfig value (TRUE), or to the latest value
*		received from the subnet (FALSE)
*
* RETURN VALUES
*	RailFilterConfig MAD block relevant to input rail, ingress block and egress block
*
* SEE ALSO
* 	ib_rail_filter_config_t
*********/

#endif				/* ifndef OSM_NVLINK_H */
