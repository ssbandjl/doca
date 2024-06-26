/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2004-2009 Voltaire, Inc. All rights reserved.
 * Copyright (c) 2002-2013 Mellanox Technologies LTD. All rights reserved.
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
 * 	Declaration of osm_vl15_t.
 *	This object represents a VL15 interface object.
 *	This object is part of the OpenSM family of objects.
 */

#ifndef _OSM_VL15INTF_H_
#define _OSM_VL15INTF_H_

#include <iba/ib_types.h>
#include <complib/cl_spinlock.h>
#include <complib/cl_event.h>
#include <complib/cl_thread.h>
#include <complib/cl_qlist.h>
#include <opensm/osm_stats.h>
#include <opensm/osm_log.h>
#include <opensm/osm_madw.h>
#include <opensm/osm_mad_pool.h>
#include <vendor/osm_vendor_api.h>
#include <opensm/osm_subnet.h>

#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else				/* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif				/* __cplusplus */

BEGIN_C_DECLS
/****h* OpenSM/VL15
* NAME
*	VL15
*
* DESCRIPTION
*	The VL15 object encapsulates the information needed by the
*	OpenSM to instantiate the VL15 interface.  The OpenSM allocates
*	one VL15 object per subnet.
*
*	The VL15 object transmits MADs to the wire at a throttled rate,
*	so as to not overload the VL15 buffering of subnet components.
*	OpenSM modules may post VL15 MADs to the VL15 interface as fast
*	as possible.
*
*	The VL15 object is thread safe.
*
*	This object should be treated as opaque and should
*	be manipulated only through the provided functions.
*
* AUTHOR
*	Steve King, Intel
*
*********/
/****d* OpenSM: SM/osm_vl15_state_t
* NAME
*	osm_vl15_state_t
*
* DESCRIPTION
*	Enumerates the possible states of OpenSM VL15 object.
*
* SYNOPSIS
*/
typedef enum _osm_vl15_state {
	OSM_VL15_STATE_INIT = 0,
	OSM_VL15_STATE_READY
} osm_vl15_state_t;
/***********/

/****s* OpenSM: VL15/osm_vl15_t
* NAME
*	osm_vl15_t
*
* DESCRIPTION
*	VL15 structure.
*
*	This object should be treated as opaque and should
*	be manipulated only through the provided functions.
*
* SYNOPSIS
*/
typedef struct osm_vl15 {
	osm_thread_state_t thread_state;
	osm_vl15_state_t state;
	uint32_t max_wire_smps;
	uint32_t max_wire_smps2;
	uint32_t max_smps_timeout;
	cl_event_t signal;
	cl_thread_t poller;
	cl_qlist_t rfifo;
	cl_qlist_t ufifo;
	cl_qlist_t hp_ufifo;
	cl_spinlock_t lock;
	osm_vendor_t *p_vend;
	osm_log_t *p_log;
	osm_stats_t *p_stats;
	osm_subn_t *p_subn;
	uint8_t port_index;
} osm_vl15_t;
/*
* FIELDS
*	thread_state
*		Tracks the thread state of the poller thread.
*
*	state
*		Tracks the state of the VL15 interface itself.
*
*	max_wire_smps
*		Maximum number of VL15 MADs allowed on the wire at one time.
*
*	max_wire_smps2
*		Maximum number of timeout based SMPs allowed to be outstanding.
*
*	max_smps_timeout
*		Wait time in usec for timeout based SMPs.
*
*	signal
*		Event on which the poller sleeps.
*
*	poller
*		Worker thread pool that services the fifo to transmit VL15 MADs
*
*	rfifo
*		First-in First-out queue for outbound VL15 MADs for which
*		a response is expected, aka the "response fifo"
*
*	ufifo
*		First-in First-out queue for outbound VL15 MADs for which
*		no response is expected, aka the "unicast fifo".
*
*	hp_ufifo
*		First-in First-out high priority queue for outbound VL15 MADs for which
*		no response is expected. SMInfo response MADs are placed in this queue
*
*	lock
*		Spinlock guarding the FIFO.
*
*	p_vend
*		Pointer to the vendor transport object.
*
*	p_log
*		Pointer to the log object.
*
*	p_stats
*		Pointer to the OpenSM statistics block.
*
*	p_subn
*		Pointer to the OpenSM subnet object
*
* 	port_index
* 		OpenSM binding port index
*
* SEE ALSO
*	VL15 object
*********/

/****s* OpenSM: VL15/osm_super_vl15_t
* NAME
*	osm_super_vl15_t
*
* DESCRIPTION
*	Super VL15 structure. It holds vl15 objects
*	per each OpenSM auxiliary port.
*
*	This object should be treated as opaque and should
*	be manipulated only through the provided functions.
*
* SYNOPSIS
*/
typedef struct osm_super_vl15 {
	osm_vl15_t		vl15[OSM_MAX_BINDING_PORTS];
	uint8_t			num_ports;
	struct osm_opensm	*p_osm;
} osm_super_vl15_t;
/*
* FIELDS
*	vl15
*		vl15 objects per each OpenSM auxiliary port.
*	num_ports
*		number of OpenSM binding ports
*	p_osm
*		pointer to osm_opensm
* SEE ALSO
*	VL15 object
*/

/****f* OpenSM: VL15/osm_vl15_construct
* NAME
*	osm_vl15_construct
*
* DESCRIPTION
*	This function constructs an VL15 object.
*
* SYNOPSIS
*/
void osm_vl15_construct(IN osm_vl15_t * p_vl15);
/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to a VL15 object to construct.
*
* RETURN VALUE
*	This function does not return a value.
*
* NOTES
*	Allows calling osm_vl15_destroy.
*
*	Calling osm_vl15_construct is a prerequisite to calling any other
*	method except osm_vl15_init.
*
* SEE ALSO
*	VL15 object, osm_vl15_init, osm_vl15_destroy
*********/

/****f* OpenSM: VL15/osm_super_vl15_construct
* NAME
*	osm_super_vl15_construct
*
* DESCRIPTION
*	This function constructs a Super VL15 object.
*
* SYNOPSIS
*/
void osm_super_vl15_construct(IN OUT osm_super_vl15_t * p_super_vl15,
			      IN osm_subn_opt_t * p_opt);
/*
* PARAMETERS
*	p_super_vl15
*		[in] Pointer to a Super VL15 object to construct.
*
*	p_opt
*		[in] Pointer to the subnet options structure.
*
* RETURN VALUE
*	This function does not return a value.
*
* NOTES
*	Allows calling osm_vl15_destroy.
*
*	Calling osm_super_vl15_construct is a prerequisite to calling any other
*	method except osm_super_vl15_init.
*
* SEE ALSO
*	Super VL15 object, osm_super_vl15_init, osm_super_vl15_destroy
*********/

/****f* OpenSM: VL15/osm_vl15_destroy
* NAME
*	osm_vl15_destroy
*
* DESCRIPTION
*	The osm_vl15_destroy function destroys the object, releasing
*	all resources.
*
* SYNOPSIS
*/
void osm_vl15_destroy(IN osm_vl15_t * p_vl15, IN struct osm_mad_pool *p_pool);
/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to a VL15 object to destroy.
*
*	p_pool
*		[in] The pointer to the mad pool to return outstanding mads to
*
* RETURN VALUE
*	This function does not return a value.
*
* NOTES
*	Performs any necessary cleanup of the specified VL15 object.
*	Further operations should not be attempted on the destroyed object.
*	This function should only be called after a call to osm_vl15_construct or
*	osm_vl15_init.
*
* SEE ALSO
*	VL15 object, osm_vl15_construct, osm_vl15_init
*********/

/****f* OpenSM: VL15/osm_super_vl15_destroy
* NAME
*	osm_super_vl15_destroy
*
* DESCRIPTION
*	The osm_super_vl15_destroy function destroys the object, releasing
*	all resources.
*
* SYNOPSIS
*/
void osm_super_vl15_destroy(IN osm_super_vl15_t * p_super_vl,
			    IN struct osm_mad_pool *p_pool);
/*
* PARAMETERS
*	p_super_vl15
*		[in] Pointer to a Super VL15 object to destroy.
*
*	p_pool
*		[in] The pointer to the mad pool to return outstanding mads to
*
* RETURN VALUE
*	This function does not return a value.
*
* NOTES
*	Performs any necessary cleanup of the specified VL15 objects in Super VL15.
*	Further operations should not be attempted on the destroyed object.
*	This function should only be called after a call to osm_super_vl15_construct or
*	osm_super_vl15_init.
*
* SEE ALSO
*	Super VL15 object, osm_super_vl15_construct, osm_super_vl15_init
*********/

/****f* OpenSM: VL15/osm_vl15_init
* NAME
*	osm_vl15_init
*
* DESCRIPTION
*	The osm_vl15_init function initializes a VL15 object for use.
*
* SYNOPSIS
*/
ib_api_status_t osm_vl15_init(IN osm_vl15_t * p_vl15, IN osm_vendor_t * p_vend,
			      IN osm_log_t * p_log, IN osm_stats_t * p_stats,
			      IN osm_subn_t * p_subn,
			      IN int32_t max_wire_smps,
			      IN int32_t max_wire_smps2,
			      IN uint32_t max_smps_timeout,
			      IN uint8_t port_index);
/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to an osm_vl15_t object to initialize.
*
*	p_vend
*		[in] Pointer to the vendor transport object.
*
*	p_log
*		[in] Pointer to the log object.
*
*	p_stats
*		[in] Pointer to the OpenSM stastics block.
*
*	p_subn
*		[in] Pointer to the Opensm subnet object
*
*	max_wire_smps
*		[in] Maximum number of SMPs allowed on the wire at one time.
*
*	max_wire_smps2
*		[in] Maximum number of timeout based SMPs allowed to be
*		     outstanding.
*
*	max_smps_timeout
*		[in] Wait time in usec for timeout based SMPs.
*
*
* RETURN VALUES
*	IB_SUCCESS if the VL15 object was initialized successfully.
*
* NOTES
*	Allows calling other VL15 methods.
*
* SEE ALSO
*	VL15 object, osm_vl15_construct, osm_vl15_destroy
*********/

/****f* OpenSM: VL15/osm_vl15_init
* NAME
*	osm_super_vl15_init
*
* DESCRIPTION
*	The osm_super_vl15_init function initializes a super VL15 object for use.
*
* SYNOPSIS
*/
ib_api_status_t osm_super_vl15_init(IN OUT osm_super_vl15_t * p_super_vl15,
				    IN osm_vendor_t * p_vend[],
				    IN osm_log_t * p_log,
				    IN osm_stats_t p_stats[],
				    IN osm_subn_t * p_subn,
				    IN const osm_subn_opt_t * p_opt);
/*
* PARAMETERS
*	p_osm
*		[in] Pointer to an osm_opensm_t object to initialize.
*
*	p_opt
*		[in] Pointer to the subnet options structure.
* SEE ALSO
*	super VL15 object, VL15 object
*********/

/****f* OpenSM: VL15/osm_vl15_post_req_batch
* NAME
*       osm_vl15_post_req_batch
*
* DESCRIPTION
*       Posts a list of Set/Get MADs of the same kind
*       to the VL15 interface for transmission.
*       Only Set/Get SMP MADs should be contained in the list.
*       From performance reason no explicit validity check is done.
*
* SYNOPSIS
*/

void osm_vl15_post_req_batch(IN osm_vl15_t * p_vl15, IN cl_qlist_t * p_list);

/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to an osm_vl15_t object.
*	p_list
*		[in] Pointer to a list of MAD wrapper structures containing the MADs
*
* RETURN VALUES
* 	This function does not return a value.
*
* NOTES
* 	The osm_vl15_construct or osm_vl15_init must be called before using
* 	this function.
* SEE ALSO
*       VL15 object, osm_vl15_construct, osm_vl15_init
*********/

/****f* OpenSM: VL15/osm_super_vl15_post_req_batch
* NAME
*       osm_super_vl15_post_req_batch
*
* DESCRIPTION
*       Posts a list of Set/Get MADs of the same kind
*       to the main OpenSM port VL15 interface for transmission.
*       Only Set/Get SMP MADs should be contained in the list.
*       From performance reason no explicit validity check is done.
*
* SYNOPSIS
*/
void osm_super_vl15_post_req_batch(IN osm_super_vl15_t * p_super_vl,
				   IN cl_qlist_t * p_list);
/*
* PARAMETERS
*	p_super_vl15
*		[in] Pointer to an osm_super_vl15_t object.
*	p_list
*		[in] Pointer to a list of MAD wrapper structures containing the MADs
*
* RETURN VALUES
* 	This function does not return a value.
*
* NOTES
* 	The osm_super_vl15_construct or osm_super_vl15_init must be called before using
* 	this function.
* SEE ALSO
*       Super VL15 object, osm_super_vl15_construct, osm_super_vl15_init
*********/

/****f* OpenSM: VL15/osm_vl15_post
* NAME
*	osm_vl15_post
*
* DESCRIPTION
*	Posts a MAD to the VL15 interface for transmission.
*
* SYNOPSIS
*/
void osm_vl15_post(IN osm_vl15_t * p_vl15, IN osm_madw_t * p_madw, IN boolean_t force_flag);
/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to an osm_vl15_t object.
*
*	p_madw
*		[in] Pointer to a MAD wrapper structure containing the MAD.
*
*	force_flag
*		[in] If set to TRUE, high priority ucast queue should be used
*
* RETURN VALUES
*	This function does not return a value.
*
* NOTES
*	The osm_vl15_construct or osm_vl15_init must be called before using
*	this function.
*
* SEE ALSO
*	VL15 object, osm_vl15_construct, osm_vl15_init
*********/

/****f* OpenSM: VL15/osm_super_vl15_post
* NAME
*	osm_super_vl15_post
*
* DESCRIPTION
*	Posts a MAD to the least loaded OpenSM port VL15 interface for transmission.
*
* SYNOPSIS
*/
void osm_super_vl15_post(IN osm_super_vl15_t * p_super_vl,
			 IN osm_madw_t * p_madw,
			 boolean_t force_flag,
			 IN const struct osm_physp *p_physp);
/*
* PARAMETERS
*	p_super_vl15
*		[in] Pointer to an osm_super_vl15_t object.
*
*	p_madw
*		[in] Pointer to a MAD wrapper structure containing the MAD.
*
*	force_flag
*		[in] If set to TRUE, high priority ucast queue should be used
*
*	p_physp
*		[in] Pointer to an osm_physp_t object.
*
* RETURN VALUES
*	This function does not return a value.
*
* NOTES
*	The osm_vl15_construct or osm_vl15_init must be called before using
*	this function.
*
* SEE ALSO
*	VL15 object, osm_vl15_construct, osm_vl15_init
*********/


/****f* OpenSM: VL15/osm_vl15_poll
* NAME
*	osm_vl15_poll
*
* DESCRIPTION
*	Causes the VL15 Interface to consider sending another QP0 MAD.
*
* SYNOPSIS
*/
void osm_vl15_poll(IN osm_vl15_t * p_vl, IN boolean_t force_flag);
/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to an osm_vl15_t object.
*
*	force_flag
*		[in] If set to TRUE, poller thread should be signalled immediately
*
* RETURN VALUES
*	None.
*
* NOTES
*	This function signals the VL15 that it may be possible to send
*	a SMP.  This function checks three criteria before sending a SMP:
*	1) The VL15 worker is IDLE
*	2) There are no QP0 SMPs currently outstanding
*	3) There is something on the VL15 FIFO to send
*
* SEE ALSO
*	VL15 object, osm_vl15_construct, osm_vl15_init
*********/

/****f* OpenSM: VL15/osm_super_vl15_poll
* NAME
*	osm_super_vl15_poll
*
* DESCRIPTION
*	Causes the VL15 Interface at OpenSM port at 'port_index'
*	to consider sending another QP0 MAD.
*
* SYNOPSIS
*/
void osm_super_vl15_poll(IN osm_super_vl15_t * p_super_vl,
			 IN boolean_t force_flag,
			 uint8_t port_index);
/*
* PARAMETERS
*	p_super_vl15
*		[in] Pointer to an osm_super_vl15_t object.
*
*	force_flag
*		[in] If set to TRUE, poller thread should be signalled immediately
*
*	port_index
*		[in] a Port index of OpenSM binding port
*
* RETURN VALUES
*	None.
*/

/****f* OpenSM: VL15/osm_vl15_shutdown
* NAME
*	osm_vl15_shutdown
*
* DESCRIPTION
*	Cleanup all outstanding MADs on both fifo's.
*  This is required to return all outstanding MAD resources.
*
* SYNOPSIS
*/
void osm_vl15_shutdown(IN osm_vl15_t * p_vl, IN osm_mad_pool_t * p_mad_pool);
/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to an osm_vl15_t object.
*
*	p_mad_pool
*		[in] The MAD pool owning the mads.
*
* RETURN VALUES
*	None.
*
* NOTES
*
* SEE ALSO
*	VL15 object, osm_vl15_construct, osm_vl15_init
*********/

void osm_super_vl15_shutdown(IN osm_super_vl15_t * p_super_vl,
			     IN osm_mad_pool_t * p_mad_pool);

/****f* OpenSM: VL15/osm_vl15_cancel_dr_request
* NAME
*	osm_vl15_cancel_dr_request
*
* DESCRIPTION
*	Cancel pending requests to specified direct route.
*
* SYNOPSIS
*/
void osm_vl15_cancel_dr_request(IN osm_vl15_t * p_vl,
				IN osm_mad_pool_t * p_mad_pool,
				IN uint8_t hop_count,
				IN const uint8_t *initial_path);
/*
* PARAMETERS
*	p_vl15
*		[in] Pointer to an osm_vl15_t object.
*
*	p_mad_pool
*		[in] The MAD pool owning the mads.
*
*	hop_count
*		[in] DR route hop count.
*
* 	initial_path
* 		[in] DR initial path.
*
* RETURN VALUES
*	None.
*
* NOTES
*
* SEE ALSO
*	VL15 object, osm_vl15_construct, osm_vl15_init
*********/

/****f* OpenSM: VL15/osm_super_vl15_cancel_dr_request
* NAME
*	osm_super_vl15_cancel_dr_request
*
* DESCRIPTION
*	Cancel pending requests to specified direct route
*	for main OpenSM port only
*
* SYNOPSIS
*/
void osm_super_vl15_cancel_dr_request(IN osm_super_vl15_t * p_super_vl,
				      IN osm_mad_pool_t * p_mad_pool,
				      IN uint8_t hop_count,
				      IN const uint8_t *initial_path,
				      IN uint8_t port_index);
/*
* PARAMETERS
*	p_super_vl15
*		[in] Pointer to an osm_super_vl15_t object.
*
*	p_mad_pool
*		[in] The MAD pool owning the mads.
*
*	hop_count
*		[in] DR route hop count.
*
* 	initial_path
* 		[in] DR initial path.
*
*	port_index
*		[in] a Port index of OpenSM binding port
*
* RETURN VALUES
*	None.
*
* NOTES
*
* SEE ALSO
*	Super VL15 object, osm_super_vl15_construct, osm_super_vl15_init
*********/

END_C_DECLS
#endif				/* _OSM_VL15INTF_H_ */
