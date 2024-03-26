/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2020 Mellanox Technologies LTD. All rights reserved.
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
 *    OSM Key Manager for key configuration of multiple classes
 *
 * Author:
 *    Or Nechemia, Mellanox
 */

#ifndef OSM_KEY_MGR_H
#define OSM_KEY_MGR_H

#include <iba/ib_types.h>
#include <complib/cl_types_osd.h>
#include <complib/cl_dispatcher.h>
#include <opensm/osm_subnet.h>
#include <opensm/osm_log.h>
#include <opensm/osm_vendor_specific.h>
#include <opensm/osm_congestion_control.h>
#include <opensm/osm_n2n.h>

#define OSM_KEY_TYPE_MANAGER		0
#define OSM_KEY_TYPE_NEIGHBOR_SW	1

struct osm_opensm;

/****f* OpenSM: KeyManager/osm_key_mgr_generate_key
* NAME
* 	osm_key_mgr_generate_key
*
* DESCRIPTION
* 	Get key for port by its guid
*
* SYNOPSIS
*/
ib_net64_t osm_key_mgr_generate_key(struct osm_opensm * p_osm, osm_port_t * p_port,
				    uint8_t mgmt_class, uint8_t key_type);
/*
* PARAMETERS
*	p_osm
*		Pointer to an osm_opensm_t object.
*
*	p_port
*		Pointer to port
*
*	mgmt_class
*		Management class. Currently supported: CC, VS, N2N
*
*	neighbor_key
*		Boolean indicates of which N2N Class key is referred -
*		manager key or node to node key.
*
* RETURN VALUE
*	Class key for input port.
*
* SEE ALSO
*
*********/

void osm_key_mgr_get_cached_key(struct osm_opensm * p_osm, osm_port_t * p_port,
				uint8_t mgmt_class, uint8_t key_type);

void osm_key_mgr_set_cached_key(struct osm_opensm * p_osm, osm_port_t * p_port,
				uint8_t mgmt_class, uint8_t key_type);

#endif				/* ifndef OSM_KEY_MGR_H */
