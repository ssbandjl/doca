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
 *      Implementation of Asym Algorithm.
 */

#ifndef _OSM_UCAST_ASYM_H_
#define _OSM_UCAST_ASYM_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include <ctype.h>
#include <complib/cl_types_osd.h>
#include <complib/cl_types.h>
#include <complib/cl_byteswap.h>
#include <complib/cl_u64_vector.h>
#include <complib/cl_debug.h>
#include <opensm/osm_switch.h>
#include <opensm/osm_opensm.h>
#include <opensm/osm_ucast_mgr.h>
#include <opensm/osm_mcast_mgr.h>
#include <opensm/osm_max_flow_algorithm.h>

/* ========================================================================================================
 *
 * Set manipulation macros.
 */
typedef	ssize_t				asym_iter_t;
typedef cl_u64_vector_t			asym_set_t;

#define	asym_set_create(P_SET)		cl_u64_vector_construct(P_SET);				\
					cl_u64_vector_init(P_SET, 1, TRUE);			\
					cl_u64_vector_set(P_SET, (uint64_t)NULL, 0x0ull);
#define	asym_set_destroy(P_SET)		cl_u64_vector_destroy(P_SET)
#define	asym_set_size(P_SET)		cl_u64_vector_get_size(P_SET)
#define	asym_set_get(P_SET, INDEX)	cl_u64_vector_get(P_SET, (size_t)INDEX)
#define	asym_set_find(P_SET, VALUE)	cl_u64_vector_find_from_start(P_SET, (uint64_t)VALUE)
#define	asym_set_insert(P_SET, VALUE)	cl_u64_vector_insert(P_SET, (uint64_t)VALUE, NULL)
#define	asym_set_remove(P_SET, INDEX)	cl_u64_vector_remove(P_SET, INDEX)
#define	asym_set_remove_all(P_SET)	cl_u64_vector_remove_all(P_SET);			\
					cl_u64_vector_set(P_SET, (uint64_t)NULL, 0x0ull);

#define	asym_set_is_empty(P_SET)	(asym_set_size(P_SET) == 1)
#define	asym_set_is_not_empty(P_SET)	(!asym_set_is_empty(P_SET))
#define	asym_set_contains(P_SET, VALUE)	(asym_set_find(P_SET, VALUE) != asym_set_size(P_SET))

#define	foreach_set_element(P_SET, P_VOID, TYPE, ITER)						\
	for (ITER = (asym_iter_t)asym_set_size((P_SET)) - 1;					\
	     P_VOID = (TYPE)asym_set_get(P_SET, ITER), ITER >= 1;				\
	     ITER--)

/* ========================================================================================================
 *
 * Looping and mapping macros.
 */

#define index_to_node(INDEX)		&p_asym->nodes[INDEX]
#define index_to_vertex(INDEX)		&p_asym->p_graph->vertices[INDEX]
#define index_to_distance(INDEX)	&p_asym->distances[INDEX * p_asym->num_switches]

#define foreach_asym_node(P_ASYM, P_NODE)							\
	for (P_NODE = P_ASYM->nodes;								\
	     P_NODE < &P_ASYM->nodes[P_ASYM->num_switches];					\
	     P_NODE++)

#define foreach_asym_link(P_NODE, P_LINK)							\
	for (P_LINK = P_NODE->links;								\
	     P_LINK < &P_NODE->links[P_NODE->num_links];					\
	     P_LINK++)

#define foreach_asym_pod(P_ASYM,P_POD)								\
	for (P_POD  = (asym_pod_t *)cl_qmap_head(&((P_ASYM)->pod_map));      			\
	     P_POD != (asym_pod_t *)cl_qmap_end(&((P_ASYM)->pod_map));       			\
	     P_POD  = (asym_pod_t *)cl_qmap_next((cl_map_item_t*)P_POD))

#define	foreach_pod_leaf(P_POD, P_SW, ITER)							\
	foreach_set_element(&P_POD->leaves, P_SW, osm_switch_t *, ITER)
#define	foreach_pod_spine(P_POD, P_SW, ITER)							\
	foreach_set_element(&P_POD->spines, P_SW, osm_switch_t *, ITER)
#define	foreach_set_switch(P_SET, P_SW, ITER)							\
	foreach_set_element(P_SET, P_SW, osm_switch_t *, ITER)

/* ========================================================================================================
 *
 * ID macros.
 */
#define asym_switch_is_core(P_SW)       ((P_SW)->rank == ASYM_RANK_CORE)
#define asym_switch_is_spine(P_SW)      ((P_SW)->rank == ASYM_RANK_SPINE)
#define asym_switch_is_leaf(P_SW)       ((P_SW)->rank == ASYM_RANK_LEAF)

/* ========================================================================================================
 *
 * These data structs are shared between AR_ASYM and AR_ASYM_ALGORITHM.
 */
typedef struct {
	cl_map_item_t		map_item;
	int			index;
	int			fully_connected_leaves;
	int			num_spines;
	int			num_leaves;
	asym_set_t		spines;
	asym_set_t		leaves;
} asym_pod_t;

typedef struct {
	uint16_t		type;
	uint16_t		min_lid;
	uint16_t		max_lid;
} asym_port_t;

typedef struct {
	int			index;

	int			index1;
	uint8_t			port1;
	int			index2;
	uint8_t			port2;
} asym_link_t;

typedef struct {
	int			index;
	int			num_links;
	int			q_max;
	uint8_t			rank;
	uint16_t		group_id;
	boolean_t		is_leaf_switch;
	boolean_t		is_fully_connected;
	int			num_connections;
	uint16_t		ca_count;
	uint16_t		max_group_id;
	uint16_t		lid;
	int			pod;
	osm_switch_t		*p_sw;
	asym_port_t		ports[256];
	asym_link_t		**links;
} asym_node_t;

typedef struct {
	osm_opensm_t		*p_osm;
	osm_subn_t		*p_subn;
	asym_node_t		*nodes;
	asym_link_t		*links;
	uint16_t		*queues;
	uint8_t			*distances;
	osm_log_t		*p_log;
	osm_ucast_mgr_t		*p_mgr;
	flow_graph_t		*p_graph;
	cl_qmap_t		pod_map;
	int			num_pods;
	int			num_switches;
	uint8_t			ar_state;
	int			ar_enable;
	int			ar_mode;
	int			num_links;
	int			max_rank;
	void			*context;
	void			*p_algo;
} asym_t;

void asym_set_new_lft_entry(osm_subn_t *p_subn, osm_switch_t *p_sw, uint16_t lid, int state, int id);
void asym_set_new_static_lft_entry(osm_subn_t *p_subn, osm_switch_t *p_sw, uint16_t lid);
void asym_set_new_free_lft_entry(osm_subn_t *p_subn, osm_switch_t *p_sw, uint16_t lid, uint16_t group_id, osm_ar_subgroup_t *ar_group);
void asym_set_remote_group_ids(asym_t *p_asym, osm_switch_t *p_sw, osm_switch_t *p_dst_sw, osm_ar_subgroup_t *p_group);
void asym_set_local_group_id(asym_t *p_asym, osm_switch_t *p_sw);

/* ========================================================================================================
 *
 * These data structs are private to AR_ASYM_ALGORITHM.
 */

/* ========================================================================================================
 * Macros.
 */
#define sw_name(P_SW)		(P_SW->p_node->print_desc)

/*
 * Mapping macros.
 */
#define	index_to_alink(INDEX)		 (&p_algo->p_alink[INDEX])
#define	index_to_adata(INDEX)		 (&p_algo->p_adata[INDEX])
#define index_to_anode(INDEX)		 (&p_asym->nodes[INDEX])
#define index_to_switch(INDEX)		((&p_asym->nodes[INDEX])->p_sw)

#define switch_to_adata(P_SW)		index_to_adata((P_SW)->re_index)
#define switch_to_anode(P_SW)		index_to_anode((P_SW)->re_index)

#define adata_to_anode(P_ADATA)		index_to_anode((P_ADATA)->index)
#define adata_to_switch(P_ADATA)	index_to_switch((P_ADATA)->index)

#define anode_to_adata(P_ANODE)		index_to_adata((P_ANODE)->index)
#define anode_to_switch(P_ANODE)	index_to_switch((P_ANODE)->index)

#define	alink_index(P_ALINK)		 (P_ALINK - p_algo->p_alink)

#define alink_to_switches(P_ALINK, P_SW1, P_SW2)							\
	P_SW1 = index_to_switch((P_ALINK)->p_edge->index);						\
	P_SW2 = index_to_switch((P_ALINK)->p_edge->adj_index)

#define alink_to_ports(P_ALINK, P_SW1, P_SW2)								\
	P_SW1 = index_to_switch((P_ALINK)->p_edge->port_num);						\
	P_SW2 = index_to_switch((P_ALINK)->p_edge->adj_port_num)

#define alink_to_adata(P_ALINK, P_ADATA1, P_ADATA2)							\
	P_ADATA1 = index_to_adata((P_ALINK)->p_edge->index);						\
	P_ADATA2 = index_to_adata((P_ALINK)->p_edge->adj_index)


/*
 * Floating point comparison macros.
 */
#define	DOUBLE_EPSILON	0.0000000000000001
#define W_eq(f1, f2) (fabs(f1 - f2) < DOUBLE_EPSILON)
#define W_ne(f1, f2) (!W_eq(f1,f2))
#define W_gt(f1, f2) (f1 >  f2)
#define W_ge(f1, f2) (f1 >= f2)
#define W_lt(f1, f2) (f1 <  f2)
#define W_le(f1, f2) (f1 <= f2)

/*
 * Looping macros.
 */
#define foreach_anode(P_ASYM, P_ANODE)									\
	for (P_ANODE = P_ASYM->nodes; P_ANODE < &P_ASYM->nodes[P_ASYM->num_switches]; P_ANODE++)

#define foreach_adata(P_ALGO, P_ADATA)									\
	for (P_ADATA = P_ALGO->p_adata; P_ADATA < &P_ALGO->p_adata[P_ALGO->num_adata]; P_ADATA++)

#define foreach_alink(P_ALGO, P_ALINK)									\
	for (P_ALINK = P_ALGO->p_alink; P_ALINK < &P_ALGO->p_alink[P_ALGO->num_alink]; P_ALINK++)

#define	foreach_layer_switch(P_LAYER, P_SW, ITER)							\
	foreach_set_element(P_LAYER, P_SW, osm_switch_t *, ITER)

#define	foreach_link_removal(P_REMS, P_ALINK, ITER)							\
	foreach_set_element(P_REMS, P_ALINK, asym_alink_t *, ITER)

#define	foreach_switch_removal(P_REMS, P_SW, ITER)							\
	foreach_set_element(P_REMS, P_SW, osm_switch_t *, ITER)

/*
 * Link macros.
 */
#define	LINK_DOWN				0
#define	LINK_UP					1
#define	LINK_FAKED				2
#define	LINK_REMOVED				3
#define	LINK_NA					4

#define link_is_up(P_ALINK)			(P_ALINK->state == LINK_UP)
#define link_is_down(P_ALINK)			(P_ALINK->state == LINK_DOWN)
#define link_is_faked(P_ALINK)			(P_ALINK->state == LINK_FAKED)
#define link_is_removed(P_ALINK)		(P_ALINK->state == LINK_REMOVED)

#define	alink_test(P_ADATA, P_ALINK)									\
	(P_ALINK->p_edge->index == P_ADATA->index)

#define	alink_is_connected_to_adata(P_ADATA, P_ALINK)							\
	((P_ALINK->p_edge->index == P_ADATA->index) || (P_ALINK->p_edge->adj_index == P_ADATA->index))

#define	alink_local_index(P_ADATA, P_ALINK)								\
	((alink_test(P_ADATA, P_ALINK)) ? P_ALINK->p_edge->index : P_ALINK->p_edge->adj_index)

#define	alink_local_port_num(P_ADATA, P_ALINK)								\
	((alink_test(P_ADATA, P_ALINK)) ? P_ALINK->p_edge->port_num : P_ALINK->p_edge->adj_port_num)

#define	alink_local_adata(P_ADATA, P_ALINK)								\
	(index_to_adata(alink_local_index(P_ADATA, P_ALINK)))

#define	alink_local_switch(P_ADATA, P_ALINK)								\
	(index_to_switch(alink_local_index(P_ADATA, P_ALINK)))

#define	alink_remote_index(P_ADATA, P_ALINK)								\
	((alink_test(P_ADATA, P_ALINK)) ? P_ALINK->p_edge->adj_index : P_ALINK->p_edge->index)

#define	alink_remote_port_num(P_ADATA, P_ALINK)								\
	((alink_test(P_ADATA, P_ALINK)) ? P_ALINK->p_edge->adj_port_num : P_ALINK->p_edge->port_num)

#define	alink_remote_adata(P_ADATA, P_ALINK)								\
	(index_to_adata(alink_remote_index(P_ADATA, P_ALINK)))

#define	alink_remote_switch(P_ADATA, P_ALINK)								\
	(index_to_switch(alink_remote_index(P_ADATA, P_ALINK)))

#define link_is_incoming(P_ADATA, P_ALINK)								\
	(P_ADATA->layer == alink_remote_adata(P_ADATA, P_ALINK)->layer+1)

#define link_is_outgoing(P_ADATA, P_ALINK)								\
	(P_ADATA->layer == alink_remote_adata(P_ADATA, P_ALINK)->layer-1)

#define foreach_link(P_ADATA, P_ALINK, ITER)								\
	for (ITER = 0; P_ALINK = P_ADATA->p_links[ITER], ITER < P_ADATA->num_links; ITER++)

#define foreach_incoming_link(P_ADATA, P_ALINK, ITER)							\
	foreach_link(P_ADATA, P_ALINK, ITER)								\
		if (link_is_up(P_ALINK) && link_is_incoming(P_ADATA,P_ALINK))

#define foreach_outgoing_link(P_ADATA, P_ALINK, ITER)							\
	foreach_link(P_ADATA, P_ALINK, ITER)								\
		if (link_is_up(P_ALINK) && link_is_outgoing(P_ADATA,P_ALINK))

/*
 * Rank definitions.
 */
#define	ASYM_RANK_CORE		0
#define	ASYM_RANK_SPINE		1
#define	ASYM_RANK_LEAF		2

/*
 * Layer defines.
 */
#define LAYER_DST		1
#define LAYER_DST_SPINE		2
#define LAYER_CORE		3
#define LAYER_SRC_SPINE		4
#define LAYER_SRC		5

#define	MAX_LAYERS		8

/*
 * Miscellaneous defines.
 */
#define	ASYM_TASK_SWITCHES	0
#define	ASYM_TASK_PODS		1

/* -------------------------------------------------------------------------------------------------------- */

typedef	double			weight_t;

/*
 * Algorithm typdefs.
 */
typedef struct {
	int			index;
	flow_edge_t		*p_edge;

	int			state;
	weight_t		lw;
	weight_t		rlw;
} asym_alink_t;

typedef struct {
	int			index;
	int			num_links;
	asym_alink_t		*p_links[257];

	int			layer;
	weight_t		nw;
	weight_t		rnw;
	int			incoming_count;
	int			outgoing_count;
	boolean_t		balanced;
} asym_adata_t;

typedef struct {
	osm_switch_t		*p_src_sw;
	osm_switch_t		*p_dst_sw;

	asym_t			*p_asym;
	int			layer_number;
	int			retries;
	asym_set_t		layers[MAX_LAYERS];
	int			dst_distance;

	asym_set_t		removals;

	size_t			alink_size;
	size_t			num_alink;
	asym_alink_t		*p_alink;

	size_t			adata_size;
	size_t			num_adata;
	asym_adata_t		*p_adata;

	size_t			distances_size;
	uint8_t			*distances;

	size_t			visited_size;
	uint8_t			*visited;

	size_t			queue_size;
	uint16_t		*queue;
	int			q_max;

	asym_adata_t		*p_last_unbalanced;
} asym_algorithm_t;

void *asym_algorithm_init(asym_t *p_asym);
void asym_algorithm_free(void *p_algo);
int asym_algorithm_pod_to_pod(asym_algorithm_t *p_algo, asym_pod_t *p_src_pod, asym_pod_t *p_dst_pod);

#endif	// _OSM_UCAST_ASYM_H_
