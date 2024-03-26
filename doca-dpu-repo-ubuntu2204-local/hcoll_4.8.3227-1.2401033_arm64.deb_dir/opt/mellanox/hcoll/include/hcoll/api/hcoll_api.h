/*
* Copyright (c) 2001-2011, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* This software product is a proprietary product of Mellanox Technologies Ltd.
* (the "Company") and all right, title, and interest and to the software product,
* including all associated intellectual property rights, are and shall
* remain exclusively with the Company.
*
* This software product is governed by the End User License Agreement
* provided with the software product.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef HCOLL_API_H_
#define HCOLL_API_H_

#include "hcoll_version.h"
#include "hcoll_runtime_api.h"

typedef int (*hcoll_barrier_fn_t)
  (void* hcoll_context);
typedef int (*hcoll_bcast_fn_t)
  (void *buff, int count, dte_data_representation_t datatype, int root,
   void* hcoll_context);
typedef int (*hcoll_gather_fn_t)
  (void *sbuf, int scount, dte_data_representation_t sdtype,
   void *rbuf, int rcount, dte_data_representation_t rdtype,
   int root, void* hcoll_context);
typedef int (*hcoll_allgather_fn_t)
  (void *sbuf, int scount, dte_data_representation_t sdtype,
   void *rbuf, int rcount, dte_data_representation_t rdtype,
   void* hcoll_context);
typedef int (*hcoll_allgatherv_fn_t)
  (void *sbuf, int scount, dte_data_representation_t sdtype,
   void * rbuf, int *rcounts, int *disps,  dte_data_representation_t rdtype,
   void* hcoll_context);
typedef int (*hcoll_allreduce_fn_t)
  (void *sbuf, void *rbuf, int count, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, void* hcoll_context);
typedef int (*hcoll_alltoall_fn_t)
  (void *sbuf, int scount, dte_data_representation_t sdtype,
   void* rbuf, int rcount, dte_data_representation_t rdtype,
   void* hcoll_context);
typedef int (*hcoll_gatherv_fn_t)
    (void* sbuf, int scount, dte_data_representation_t sdtype,
     void* rbuf, int *rcounts, int *displs, dte_data_representation_t rdtype,
     int root, void *hcoll_context);
typedef int (*hcoll_reduce_fn_t)
  (void *sbuf, void* rbuf, int count, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, int root, void* hcoll_context);
typedef int (*hcoll_alltoallv_fn_t)
  (void *sbuf, int *scounts, int *sdisps, dte_data_representation_t sdtype,
   void *rbuf, int *rcounts, int *rdisps, dte_data_representation_t rdtype,
   void* hcoll_context);
typedef int (*hcoll_alltoallw_fn_t)
  (void *sbuf, int *scounts, int *sdisps, dte_data_representation_t *sdtypes,
   void *rbuf, int *rcounts, int *rdisps, dte_data_representation_t *rdtypes,
   void* hcoll_context);
typedef int (*hcoll_exscan_fn_t)
  (void *sbuf, void *rbuf, int count, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, void* hcoll_context);
typedef int (*hcoll_reduce_scatter_fn_t)
  (void *sbuf, void *rbuf, int *rcounts, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, void* hcoll_context);
typedef int (*hcoll_reduce_scatter_block_fn_t)
  (void *sbuf, void *rbuf, int rcount, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, void* hcoll_context);
typedef int (*hcoll_scan_fn_t)
  (void *sbuf, void *rbuf, int count, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, void* hcoll_context);
typedef int (*hcoll_scatter_fn_t)
  (void *sbuf, int scount, dte_data_representation_t sdtype,
   void *rbuf, int rcount, dte_data_representation_t rdtype,
   int root, void* hcoll_context);
typedef int (*hcoll_scatterv_fn_t)
  (void *sbuf, int *scounts, int *disps, dte_data_representation_t sdtype,
   void* rbuf, int rcount, dte_data_representation_t rdtype,
   int root, void* hcoll_context);

typedef int (*hcoll_ibarrier_fn_t)
  (void* hcoll_context, void **runtime_handle);
typedef int (*hcoll_ibcast_fn_t)
  (void *buff, int count, dte_data_representation_t datatype, int root,
   void **runtime_handle, void* hcoll_context);
typedef int (*hcoll_iallgather_fn_t)
  (void *sbuf, int scount, dte_data_representation_t sdtype,
   void *rbuf, int rcount, dte_data_representation_t rdtype,
   void* hcoll_context, void **runtime_handle);
typedef int (*hcoll_iallgatherv_fn_t)
  (void *sbuf, int scount, dte_data_representation_t sdtype,
   void *rbuf, int *rcount, int *disps, 
   dte_data_representation_t rdtype,
   void* hcoll_context, void **runtime_handle);
typedef int (*hcoll_iallreduce_fn_t)
  (void *sbuf, void *rbuf, int count, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, void* hcoll_context, void **runtime_handle);
typedef int (*hcoll_ireduce_fn_t)
  (void *sbuf, void* rbuf, int count, dte_data_representation_t dtype,
   hcoll_dte_op_t *op, int root, void* hcoll_context, void **runtime_handle);
typedef int (*hcoll_igatherv_fn_t)
    (void* sbuf, int scount, dte_data_representation_t sdtype,
     void* rbuf, int *rcounts, int *displs, dte_data_representation_t rdtype,
     int root, void *hcoll_context, void **runtime_handle);
typedef int (*hcoll_ialltoallv_fn_t)
  (void *sbuf, int *scounts, int *sdisps, dte_data_representation_t sdtype,
   void *rbuf, int *rcounts, int *rdisps, dte_data_representation_t rdtype,
   void* hcoll_context, void **runtime_handle);

struct hcoll_collectives_t {
    /* Collective function pointers */
    /* blocking functions */
    hcoll_allgather_fn_t coll_allgather;
    hcoll_allgatherv_fn_t coll_allgatherv;
    hcoll_allreduce_fn_t coll_allreduce;
    hcoll_alltoall_fn_t coll_alltoall;
    hcoll_alltoallv_fn_t coll_alltoallv;
    hcoll_alltoallw_fn_t coll_alltoallw;
    hcoll_barrier_fn_t coll_barrier;
    hcoll_bcast_fn_t coll_bcast;
    hcoll_exscan_fn_t coll_exscan;
    hcoll_gather_fn_t coll_gather;
    hcoll_gatherv_fn_t coll_gatherv;
    hcoll_reduce_fn_t coll_reduce;
    hcoll_reduce_scatter_fn_t coll_reduce_scatter;
    hcoll_reduce_scatter_block_fn_t coll_reduce_scatter_block;
    hcoll_scan_fn_t coll_scan;
    hcoll_scatter_fn_t coll_scatter;
    hcoll_scatterv_fn_t coll_scatterv;

    hcoll_ibarrier_fn_t coll_ibarrier;
    hcoll_ibcast_fn_t coll_ibcast;
    hcoll_iallgather_fn_t coll_iallgather;
    hcoll_iallgatherv_fn_t coll_iallgatherv;
    hcoll_iallreduce_fn_t coll_iallreduce;
    hcoll_ireduce_fn_t coll_ireduce;
    hcoll_igatherv_fn_t coll_igatherv;
    hcoll_ialltoallv_fn_t coll_ialltoallv;
};
typedef struct hcoll_collectives_t hcoll_collectives_t;
extern hcoll_collectives_t hcoll_collectives;

struct hcoll_init_opts_t {
    int base_tag;               /* IN: base tag value */
    int max_tag;                /* IN: max tag value */
    int enable_thread_support;  /* IN: runtime mpi thread level provided */

    int mem_hook_needed;        /* OUT: mem release cllback needed */
};
typedef struct hcoll_init_opts_t hcoll_init_opts_t;

int init_hcoll_collectives(void);
int hcoll_init(void);
int hcoll_init_with_opts(hcoll_init_opts_t **opts);
int hcoll_read_init_opts(hcoll_init_opts_t **opts);
void hcoll_free_init_opts(hcoll_init_opts_t *opts);
void* hcoll_create_context(rte_grp_handle_t group);

/* The following 2 APIs are deprecated since v3.7. Use hcoll_context_free instead. */
int hcoll_destroy_context(void *hcoll_context, rte_grp_handle_t group, int* context_destroyed);
int hcoll_group_destroy_notify(void * hcoll_context);

int hcoll_context_free(void * hcoll_context, rte_grp_handle_t group);
int hcoll_finalize(void);
int hcoll_set_runtime_tag_offset(int value, int max_tag);
void hcoll_get_info(int argc, char **argv);
int hcoll_check_mem_release_cb_needed(void);
void hcoll_mem_unmap(void *buf, size_t length, void *cbdata, int from_alloc);

/* The next API is deprecated since v4.0. The function has no effect. */
int hcoll_rte_p2p_disabled_notify(void);

extern int (*hcoll_progress_fn)(void);
const char *hcoll_get_version(void);

#define HCOLL_IN_PLACE (void*)1


#endif /* HCOLL_API_H_ */
