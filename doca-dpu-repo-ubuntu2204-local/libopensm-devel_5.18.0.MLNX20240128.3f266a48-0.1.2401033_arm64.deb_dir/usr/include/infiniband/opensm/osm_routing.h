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
 *	Declaration of osm_subn_t.
 *	This object represents an IBA subnet.
 *	This object is part of the OpenSM family of objects.
 */

#ifndef _OSM_ROUTING_H_
#define _OSM_ROUTING_H_

#include <iba/ib_types.h>
#include <complib/cl_qmap.h>
#include <complib/cl_fleximap.h>
#include <complib/cl_map.h>
#include <complib/cl_ptr_vector.h>
#include <complib/cl_u64_vector.h>
#include <complib/cl_list.h>
#include <complib/cl_vector.h>
#include <opensm/osm_base.h>
#include <opensm/osm_prefix_route.h>
#include <opensm/osm_db.h>
#include <opensm/osm_log.h>
#include <stdio.h>

#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else				/* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif				/* __cplusplus */

BEGIN_C_DECLS

/*
 * Forward declarations for compiler warnings.
 */
struct osm_physp;
struct osm_mgrp_box;
typedef struct osm_subn osm_subn_t;
typedef struct osm_opensm osm_opensm_t;
typedef struct _ib_slvl_table ib_slvl_table_t;

/****d* OpenSM: OpenSM/osm_adv_routing_type_enum
 * NAME
 *       osm_routing_engine_type_enum
 *
 * DESCRIPTION
 *       Enumerates the possible advanced routing types that
 *       could be used to route a subnet.
 *
 * SYNOPSIS
 */
typedef enum _osm_routing_engine_type_enum {
	OSM_ROUTING_ENGINE_TYPE_NONE = 0,	/* Entries must be contiguous */
	OSM_ROUTING_ENGINE_TYPE_MINHOP,
	OSM_ROUTING_ENGINE_TYPE_UPDN,
	OSM_ROUTING_ENGINE_TYPE_DNUP,
	OSM_ROUTING_ENGINE_TYPE_FILE,
	OSM_ROUTING_ENGINE_TYPE_FTREE,
	OSM_ROUTING_ENGINE_TYPE_PQFT,
	OSM_ROUTING_ENGINE_TYPE_LASH,
	OSM_ROUTING_ENGINE_TYPE_DOR,
	OSM_ROUTING_ENGINE_TYPE_TORUS_2QOS,
	OSM_ROUTING_ENGINE_TYPE_SSSP,
	OSM_ROUTING_ENGINE_TYPE_DFSSSP,
	OSM_ROUTING_ENGINE_TYPE_CHAIN,
	OSM_ROUTING_ENGINE_TYPE_DFP,
	OSM_ROUTING_ENGINE_TYPE_DFP2,
	OSM_ROUTING_ENGINE_TYPE_AR_DOR,
	OSM_ROUTING_ENGINE_TYPE_AR_UPDN,
	OSM_ROUTING_ENGINE_TYPE_AR_FTREE,
	OSM_ROUTING_ENGINE_TYPE_AR_TORUS,
	OSM_ROUTING_ENGINE_TYPE_KDOR_HC,
	OSM_ROUTING_ENGINE_TYPE_KDOR_GHC,
	OSM_ROUTING_ENGINE_TYPE_AR_MINHOP,
	OSM_ROUTING_ENGINE_TYPE_AUTO,

	OSM_ROUTING_ENGINE_TYPE_END		/* Only used to demark end of list */
} osm_routing_engine_type_t;

/****d* OpenSM: OpenSM/osm_adv_routing_type_enum
 * NAME
 *       osm_adv_routing_type_enum
 *
 * DESCRIPTION
 *       Enumerates the possible advanced routing types that
 *       could be used to route a subnet.
 *
 * SYNOPSIS
 */
typedef enum _osm_adv_routing_type_enum {
	OSM_ADV_ROUTING_TYPE_NONE,		/* Entries must be contiguous */
	OSM_ADV_ROUTING_TYPE_AR_LAG,
	OSM_ADV_ROUTING_TYPE_AR_TREE,
	OSM_ADV_ROUTING_TYPE_AR_MINHOP,

	OSM_ADV_ROUTING_TYPE_END		/* Only used to demark end of list */
} osm_adv_routing_type_enum;
/***********/

/****d* OpenSM: OpenSM/osm_adv_routing_engine_mode_enum
 * NAME
 *       osm_adv_routing_engine_mode_enum
 *
 * DESCRIPTION
 *       Enumerates the advanced routing engine modes.
 *
 * SYNOPSIS
 */
typedef enum _osm_adv_routing_mode_enum {
	OSM_ADV_ROUTING_MODE_DISABLED = 0,
	OSM_ADV_ROUTING_MODE_ENABLED,
	OSM_ADV_ROUTING_MODE_RN,
	OSM_ADV_ROUTING_MODE_AUTO,
	OSM_ADV_ROUTING_MODE_MAX
} osm_adv_routing_mode_enum;
/***********/

/****s* OpenSM: OpenSM/osm_routing_engine_attributes
 * NAME
 *	struct osm_routing_engine_attributes
 *
 * DESCRIPTION
 *	OpenSM routing engine attribute
 * NOTES
 *	routing engine attributes and capabilities
 */
struct osm_routing_engine_attributes {
	osm_adv_routing_mode_enum ar_mode_auto;
	osm_adv_routing_mode_enum shield_mode_auto;
	boolean_t pfrn_mode;
	boolean_t pfrn_policy_required;
	unsigned int hbf_weight_auto[OSM_AR_HBF_WEIGHT_SUBGROUP_NUM];
};
/*
 * FIELDS
 *
 *	ar_mode_auto
 *		The ar_mode configuration in case it is "auto"
 *
 *	shield_mode_auto
 *		The shield_mode configuration in case it is "auto"
 *
 *	pfrn_mode
 *		Boolean indicates whether routing engine supports pFRN.
 *		If true, pFRN is configured when shield_mode is "auto".
 *
 *	hbf_weight_auto
 *		The Hash Based Forwarding weights configuration in case it is "auto"
 *
 */

/****s* OpenSM: OpenSM/osm_routing_engine
 * NAME
 *	struct osm_routing_engine
 *
 * DESCRIPTION
 *	OpenSM routing engine module definition.
 * NOTES
 *	routing engine structure - multicast callbacks may be
 *	added later.
 */
struct osm_routing_engine {
	osm_routing_engine_type_t type;
	char *name;
	void *context;
	struct osm_routing_engine_attributes attributes;
	osm_adv_routing_type_enum adv_routing_engine_auto;
	const char *(*get_name)(struct osm_routing_engine *re);
	struct osm_routing_engine_attributes (*get_attributes)(struct osm_routing_engine *re);
	int (*build_lid_matrices) (void *context);
	int (*ucast_build_fwd_tables) (void *context);
	void (*ucast_dump_tables) (void *context);
	void (*update_sl2vl)(void *context, IN struct osm_physp *port,
			     IN uint8_t in_port_num, IN uint8_t out_port_num,
			     IN OUT ib_slvl_table_t *t);
	void (*update_vlarb)(void *context, IN struct osm_physp *port,
			     IN uint8_t port_num,
			     IN OUT ib_vl_arb_table_t *block,
			     unsigned block_length, unsigned block_num);
	uint8_t (*path_sl)(void *context, IN uint8_t path_sl_hint,
			   IN const ib_net16_t slid, IN const ib_net16_t dlid);
	ib_api_status_t (*mcast_build_stree)(void *context,
					     IN OUT struct osm_mgrp_box *mgb);
	int (*ITR_init)(void *context);
	int (*ITR_get_bs)(void *context, IN ib_net64_t sw_guid,
				OUT ib_net64_t *bs_guid);
	void (*calculate_flids) (void* context);
	void (*destroy) (void *context);
	struct osm_routing_engine *next;
};

/*
 * FIELDS
 *	name
 *		The routing engine name (will be used in logs).
 *
 *	attributes
 *		Routing engine attributes and capabilities.
 *		Data should be accessed using get_attributes callback function.
 *
 *	adv_routing_engine_auto
 *		The adv_routing_engine configuration in case it is "auto"
 *
 *	context
 *		The routing engine context. Will be passed as parameter
 *		to the callback functions.
 *
 * 	get_name
 *		Callback function for retrieving routing engine name.
 *
 *	get_attributes
 *		The callback function for retrieving routing engines attributes.
 *
 *	build_lid_matrices
 *		The callback for lid matrices generation.
 *
 *	ucast_build_fwd_tables
 *		The callback for unicast forwarding table generation.
 *
 *	ucast_dump_tables
 *		The callback for dumping unicast routing tables.
 *
 *	update_sl2vl(void *context, IN osm_physp_t *port,
 *		     IN uint8_t in_port_num, IN uint8_t out_port_num,
 *		     OUT ib_slvl_table_t *t)
 *		The callback to allow routing engine input for SL2VL maps.
 *		*port is the physical port for which the SL2VL map is to be
 *		updated. For switches, in_port_num/out_port_num identify
 *		which part of the SL2VL map to update.  For router/HCA ports,
 *		in_port_num/out_port_num should be ignored.
 *
 *	update_vlarb(void *context, IN osm_physp_t *port,
 *		     IN uint8_t port_num,
 *		     IN OUT ib_vl_arb_table_t *block,
 *		     unsigned block_length, unsigned block_num)
 *		The callback to allow routing engine input for VLArbitration.
 *		*port is the physical port for which the VLArb table is to be
 *		updated.
 *
 *	path_sl
 *		The callback for computing path SL.
 *
 *	mcast_build_stree
 *		The callback for building the spanning tree for multicast
 *		forwarding, called per MLID.
 *
 *	ITR_init
 *		Perform any action required in order to support ITR (inter
 *		Topology Routing).
 *
 *	ITR_get_BS_lid
 *		Get a lid and return the lid of the Border Switch to route
 *		to in ITR in order to get to this lid.
 *
 *	destroy
 *		The destroy method, may be used for routing engine
 *		internals cleanup.
 *
 *	next
 *		Pointer to next routing engine in the list.
 */


/****s* OpenSM: OpenSM/f_re_setup_t
 * NAME
 *      f_re_setup_t
 *
 * DESCRIPTION
 *      Function pointer to a routing engine setup function.
 */
typedef int (*f_re_setup_t) (IN struct osm_routing_engine *, IN osm_opensm_t *);

/****f* OpenSM: OpenSM/osm_routing_engine_init
 * NAME
 *	osm_routing_engine_init
 *
 * DESCRIPTION
 *	Initialize the routing engine subsystem.
 *
 * SYNOPSIS
 */
void osm_routing_engine_init(void);
/*
 * PARAMETERS
 *	None
 *
 * RETURN VALUES
 *	None
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_type_str
 * NAME
 *	osm_routing_engine_name
 *
 * DESCRIPTION
 *	Returns a string for the specified routing engine type.
 *
 * SYNOPSIS
 */
const char *osm_routing_engine_type_str(IN osm_routing_engine_type_t type);
/*
 * PARAMETERS
 *	type
 *		[in] routing engine type.
 *
 * RETURN VALUES
 *	Pointer to routing engine name.
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_type
 * NAME
 *	osm_routing_engine_type
 *
 * DESCRIPTION
 *	Returns a string for the specified routing engine.
 *
 * SYNOPSIS
 */
osm_routing_engine_type_t osm_routing_engine_type(IN struct osm_routing_engine *p_re);
/*
 * PARAMETERS
 *	p_re
 *	        [in] pointer to routing engine
 *
 * RETURN VALUES
 *	Pointer to routing engine type.
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_name
 * NAME
 *	osm_routing_engine_name
 *
 * DESCRIPTION
 *	Returns a string for the specified routing engine.
 *
 * SYNOPSIS
 */
const char *osm_routing_engine_name(IN struct osm_routing_engine *p_re);
/*
 * PARAMETERS
 *	p_re
 *	        [in] pointer to routing engine
 *
 * RETURN VALUES
 *	Pointer to routing engine name.
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_cmp
 * NAME
 *	osm_routing_engine_cmp
 *
 * DESCRIPTION
 *	Returns a routing engine type specified routing engine name string.
 *
 * SYNOPSIS
 */
int osm_routing_engine_cmp(IN struct osm_routing_engine *p_re1, IN struct osm_routing_engine *p_re2);
/*
 * PARAMETERS
 *	p_re1
 *	        [in] pointer to routing engine
 *	p_re2
 *	        [in] pointer to routing engine
 *
 * RETURN VALUES
 *	TRUE    Routing engines are equal
 *	FALSE   Routing engines are different
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_update_routing_engines
 * NAME
 *	osm_update_routing_engines
 *
 * DESCRIPTION
 *	The osm_update_routing_engines function updates routing engines
 *
 * SYNOPSIS
 */
void osm_update_routing_engines(IN osm_subn_t *subn, IN const char *engine_names);
/*
 * PARAMETERS
 *	subn
 *	        [in] Pointer to a subnet object.
 *	engine_names
 *	        [in] Routing engine names
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/setup_routing_engine
 * NAME
 *	setup_routing_engine
 *
 * DESCRIPTION
 *	The setup_routing_engine function set up routing engine according
 *	to input name
 *
 * SYNOPSIS
 */
struct osm_routing_engine *setup_routing_engine(IN osm_subn_t *subn, IN const char *name);
/*
 * PARAMETERS
 *	subn
 *	        [in] Pointer to a subnet object.
 *
 *	engine_names
 *	        [in] Routing engine name
 *
 * RETURN VALUE
 *	Routing engine that was setup according to input name.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/
/****f* OpenSM: OpenSM/osm_setup_routing_engines
 * NAME
 *	osm_setup_routing_engines
 *
 * DESCRIPTION
 *	The osm_setup_routing_engines function set up routing engines
 *
 * SYNOPSIS
 */
void osm_setup_routing_engines(IN osm_subn_t *subn, IN const char *engine_names);
/*
 * PARAMETERS
 *	subn
 *	        [in] Pointer to a subnet object.
 *	engine_names
 *	        [in] Routing engine names
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_free
 * NAME
 *	osm_routing_engine_free
 *
 * DESCRIPTION
 *	The osm_routing_engine_free function set up routing engines
 *
 * SYNOPSIS
 */
void osm_routing_engine_free(IN struct osm_routing_engine *p_re);
/*
 * PARAMETERS
 *	p_re
 *	        [in] Pointer to a routing engine object
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/void destroy_default_routing_engine
 * NAME
 *	destroy_default_routing_engine
 *
 * DESCRIPTION
 *	destroy the default routing engine
 *
 * SYNOPSIS
 */
void osm_destroy_default_routing_engine(IN osm_subn_t *subn);
/*
 * PARAMETERS
 *	subn
 *	        [in] Pointer to a subnet object.
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/osm_destroy_routing_engines
 * NAME
 *	osm_destroy_routing_engines
 *
 * DESCRIPTION
 *	The osm_destroy_routing_engines function destroy routing engines
 *
 * SYNOPSIS
 */
void osm_destroy_routing_engines(IN osm_subn_t *subn);
/*
 * PARAMETERS
 *	subn
 *	        [in] Pointer to a subnet object.
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_create
 * NAME
 *	osm_routing_engine_create
 *
 * DESCRIPTION
 *	The osm_routing_engine_create is used by plugins to create
 *	a new routing engine.
 *
 * SYNOPSIS
 */
int osm_routing_engine_create(IN osm_opensm_t *p_osm, IN const char *name, IN f_re_setup_t f_setup);
/*
 * PARAMETERS
 *	p_osm
 *	        [in] Pointer to a opensm object.
 *	name
 *	        [in] Routing engine name.
 *	f_setup
 *	        [in] Routing engine setup function.
 *
 * RETURN VALUE
 *	Zero    Success
 *	Nonzero Error
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_is_named
 * NAME
 *	osm_routing_engine_is_named
 *
 * DESCRIPTION
 *	Check if the routing engined has the name provided.
 *
 * RETURN VALUES
 *	TRUE    Name matches
 *	FALSE   Name doesn't match
 *
 * SYNOPSIS
 */
static inline boolean_t osm_routing_engine_is_named(IN struct osm_routing_engine *p_re, IN const char *name)
{
	const char	*re_name = osm_routing_engine_name(p_re);

	if (p_re == NULL) {
		return FALSE;
	} else if (name == NULL) {
		return FALSE;
	} else {
		return (strcmp(re_name, name) == 0) ? TRUE : FALSE;
	}
}

/*
 * Inline routing engine checkers.
 */

#define OSM_ROUTING_ENGINE_IS(NAME,STRING)                                              \
static inline int osm_routing_engine_is_ ## NAME (struct osm_routing_engine *p_re)      \
{                                                                                       \
	return osm_routing_engine_is_named(p_re, STRING);                               \
}

OSM_ROUTING_ENGINE_IS(none,        "none");
OSM_ROUTING_ENGINE_IS(minhop,      "minhop");
OSM_ROUTING_ENGINE_IS(updn,        "updn");
OSM_ROUTING_ENGINE_IS(dnup,        "dnup");
OSM_ROUTING_ENGINE_IS(file,        "file");
OSM_ROUTING_ENGINE_IS(ftree,       "ftree");
OSM_ROUTING_ENGINE_IS(pqft,        "pqft");
OSM_ROUTING_ENGINE_IS(lash,        "lash");
OSM_ROUTING_ENGINE_IS(dor,         "dor");
OSM_ROUTING_ENGINE_IS(torus_2QoS,  "torus-2QoS");
OSM_ROUTING_ENGINE_IS(dfsssp,      "dfsssp");
OSM_ROUTING_ENGINE_IS(sssp,        "sssp");
OSM_ROUTING_ENGINE_IS(chain,       "chain");
OSM_ROUTING_ENGINE_IS(dfp,         "dfp");
OSM_ROUTING_ENGINE_IS(dfp2,        "dfp2");
OSM_ROUTING_ENGINE_IS(ar_dor,      "ar_dor");
OSM_ROUTING_ENGINE_IS(ar_updn,     "ar_updn");
OSM_ROUTING_ENGINE_IS(ar_ftree,    "ar_ftree");
OSM_ROUTING_ENGINE_IS(ar_torus,    "ar_torus");
OSM_ROUTING_ENGINE_IS(kdor_hc,     "kdor-hc");
OSM_ROUTING_ENGINE_IS(kdor_ghc,    "kdor-ghc");
OSM_ROUTING_ENGINE_IS(ar_minhop,   "ar_minhop");
OSM_ROUTING_ENGINE_IS(auto,        "auto")

/****f* OpenSM: OpenSM/osm_routing_engine_name
 * NAME
 *	osm_routing_engine_name
 *
 * DESCRIPTION
 *	Returns a string for the specified routing engine.
 *
 * SYNOPSIS
 */
const char *osm_routing_engine_name(IN struct osm_routing_engine *p_re);
/*
 * PARAMETERS
 *	p_re
 *		[in] pointer to routing engine.
 *
 * RETURN VALUES
 *	Pointer to routing engine name.
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_routing_engine_cmp
 * NAME
 *	osm_routing_engine_cmp
 *
 * DESCRIPTION
 *	Returns a routing engine type specified routing engine name string.
 *
 * SYNOPSIS
 */
boolean_t osm_routing_engine_cmp(IN struct osm_routing_engine *p_re1, IN struct osm_routing_engine *p_re2);
/*
 * PARAMETERS
 *	p_re1
 *		[in] pointer to routing engine
 *	p_re2
 *		[in] pointer to routing engine
 *
 * RETURN VALUES
 *	TRUE	if routing engines are the same
 *	FALSE	if routing engines are the different
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_update_routing_engines
 * NAME
 *	osm_update_routing_engines
 *
 * DESCRIPTION
 *	The osm_update_routing_engines function updates routing engines
 *
 * SYNOPSIS
 */
void osm_update_routing_engines(IN osm_subn_t *subn, IN const char *engine_names);
/*
 * PARAMETERS
 *	subn
 *		[in] Pointer to a subnet object.
 *	engine_names
 *		[in] Routing engine names
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/setup_routing_engine
 * NAME
 *	setup_routing_engine
 *
 * DESCRIPTION
 *	The setup_routing_engine function set up routing engine according
 *	to input name
 *
 * SYNOPSIS
 */
struct osm_routing_engine *setup_routing_engine(IN osm_subn_t *subn, IN const char *name);
/*
 * PARAMETERS
 *	subn
 *		[in] Pointer to a subnet object.
 *
 *	engine_names
 *		[in] Routing engine name
 *
 * RETURN VALUE
 * 	Routing engine that was setup according to input name.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/osm_setup_routing_engines
 * NAME
 *	osm_setup_routing_engines
 *
 * DESCRIPTION
 *	The osm_setup_routing_engines function set up routing engines
 *
 * SYNOPSIS
 */
void osm_setup_routing_engines(IN osm_subn_t *subn, IN const char *engine_names);
/*
 * PARAMETERS
 *	subn
 *		[in] Pointer to a subnet object.
 *	engine_names
 *		[in] Routing engine names
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/void destroy_default_routing_engine
 * NAME
 *	destroy_default_routing_engine
 *
 * DESCRIPTION
 *	destroy the default routing engine
 *
 * SYNOPSIS
 */
void osm_destroy_default_routing_engine(IN osm_subn_t *subn);
/*
 * PARAMETERS
 *	subn
 *		[in] Pointer to a subnet object.
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****f* OpenSM: OpenSM/osm_destroy_routing_engines
 * NAME
 *	osm_destroy_routing_engines
 *
 * DESCRIPTION
 *	The osm_destroy_routing_engines function destroy routing engines
 *
 * SYNOPSIS
 */
void osm_destroy_routing_engines(IN osm_subn_t *subn);
/*
 * PARAMETERS
 *	subn
 *		[in] Pointer to a subnet object.
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

/****s* OpenSM: OpenSM/osm_adv_routing_engine
 * NAME
 *	struct osm_adv_routing_engine
 *
 * DESCRIPTION
 *	OpenSM advanced routing engine module definition.
 */
typedef struct osm_adv_routing_engine {
	char *name;
	void *context;
	int (*route) (void *context);
	void (*destroy) (void *context);
}osm_adv_routing_engine_t;
/*
 * FIELDS
 *	name
 *		Advanced routing engine name.
 *
 *	context
 *		Advanced routing engine context. Will be passed as parameter
 *		to the callback functions.
 *
 *	route
 *		Callback for calculating routing tables and switches
 *		configuration stage.
 *
 *	destroy
 *		The destroy method, may be used for advanced routing engine
 *		internals cleanup.
 */

/****f* OpenSM: OpenSM/osm_adv_routing_engine_init
 * NAME
 *      osm_adv_routing_engine_init
 *
 * DESCRIPTION
 *      Initialize the advanced routing engine subsystem.
 *
 * SYNOPSIS
 */
void osm_adv_routing_engine_init(void);
/*
 * PARAMETERS
 *      None
 *
 * RETURN VALUES
 *      None
 *
 * NOTES
 *
 * SEE ALSO
 *********/

/****f* OpenSM: OpenSM/osm_setup_advanced_routing_engine
 * NAME
 *	osm_setup_advanced_routing_engine
 *
 * DESCRIPTION
 * 	Initialize advance routing engine instance
 *
 * SYNOPSIS
 */
void osm_setup_advanced_routing_engine(IN osm_subn_t *p_subn, IN const char *engine_name);
/*
 * PARAMETERS
 *	p_subn
 *		[in] Pointer to a subnet object.
 *
 *	engine_name
 *		[in] Advanced routing engine names
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

void osm_create_advanced_routing_engine(IN osm_subn_t *p_subn, IN const char *engine_name);

/****f* OpenSM: OpenSM/osm_destroy_advanced_routing_engine
 * NAME
 *	osm_destroy_advanced_routing_engine
 *
 * DESCRIPTION
 *	Destroy initialized advanced routing engine.
 *
 * SYNOPSIS
 */
void osm_destroy_advanced_routing_engine(IN osm_subn_t *p_subn);
/*
 * PARAMETERS
 *	p_subn
 *		[in] Pointer to a subnet object.
 *
 * RETURN VALUE
 *	This function does not return a value.
 *
 * NOTES
 *
 * SEE ALSO
 *
 *********/

END_C_DECLS

#endif				/* _OSM_ROUTING_H_ */
