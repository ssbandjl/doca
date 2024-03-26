#ifndef HCOL_RUNTIME_API_H
#define HCOL_RUNTIME_API_H


#include "hcoll_dte.h"
#include "hcoll_constants.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#define RTE_PUBLIC typedef


/*typedef void* rte_ec_handle_t; */
typedef void* rte_grp_handle_t;

struct rte_ec_handle_t {
#if defined(RTE_DEBUG) && RTE_DEBUG > 0
    /* handle to the group */
    rte_grp_handle_t group;
#endif
    /* my rank in the group */
    int rank;
    /* uniq process handle same across all the communicators */
    void * handle;
};
typedef struct rte_ec_handle_t rte_ec_handle_t;

extern rte_ec_handle_t rte_ec_any_source;


typedef struct iovec rte_iovec_t;

#define RTE_EC_NULL_HANDLE NULL


enum {
    HCOLRTE_REQUEST_DONE,
    HCOLRTE_REQUEST_IDLE,
    HCOLRTE_REQUEST_ACTIVE,
    HCOLRTE_REQUEST_LAST
};

struct rte_request_handle_t{
    void *data;
    union{
        int status;
        int ops_num;
    };
};
typedef struct rte_request_handle_t rte_request_handle_t;

/**
* @brief Start a non blocking receive
*
* @param [in] dte_data_representation_t data type represenation
* @param [in] count number of data types (for the
generalized data type this value is ignored , as a generalized
version is included in it 's description
* @param [in] rte_ec_handle_t source handle - RTE defined
* @param [in] rte_grp_handle_t group
* @param [in] uint32_t tag
*/

RTE_PUBLIC int (* rte_recv_nb ) (struct dte_data_representation_t,
                                 uint32_t count ,
                                 void *buffer,
                                 rte_ec_handle_t ,
                                 rte_grp_handle_t ,
                                 uint32_t ,
                                 rte_request_handle_t *);

/**
* @brief Start a non blocking send
*
* @param [in] struct dte_data_representation_t
* @param [in] count number of data types (for the
generalized data type this value is ignored , as a generalized
version is included in it 's description
* @param [in] rte_ec_handle_t dest handle - RTE defined
* @param [in] rte_grp_handle_t group
* @param [in] uint32_t tag
*/
RTE_PUBLIC int (* rte_send_nb ) ( struct dte_data_representation_t,
                                  uint32_t ,
                                  void *buffer,
                                  rte_ec_handle_t ,
                                  rte_grp_handle_t ,
                                  uint32_t ,
                                  rte_request_handle_t *);

/**
* @brief test a request for completition
*/
RTE_PUBLIC int (* rte_test ) ( rte_request_handle_t * request ,
                               int * completed );



/* compare handles */
RTE_PUBLIC int (* rte_ec_handle_compare ) ( rte_ec_handle_t handle_1 ,
                                            rte_grp_handle_t
                                            group_handle_1 ,
                                            rte_ec_handle_t handle_2 ,
                                            rte_grp_handle_t
                                            group_handle_2 );

/* get handle */
RTE_PUBLIC int (* rte_get_ec_handles ) ( int num_ec ,
                                         int * ec_indexes ,
                                         rte_grp_handle_t group_handle,
                                         rte_ec_handle_t * ec_handles );


/* wait for request completion */
RTE_PUBLIC int (* rte_wait_completion) (rte_request_handle_t *req);

RTE_PUBLIC int (* rte_ec_on_local_node)(rte_ec_handle_t ec, rte_grp_handle_t group);

RTE_PUBLIC uint32_t (* rte_jobid)(void);

RTE_PUBLIC void (* rte_progress)(void);
/**
* @brief Return handle to the calling processes / threads execution
context
*
* @param [in] rte_grp_handle_t group
* @return rte_ec_handle_t
*/
RTE_PUBLIC int (*rte_get_my_ec) ( rte_grp_handle_t grp_h, rte_ec_handle_t *ec_handle);

/* Return group size */
RTE_PUBLIC int (*rte_group_size) ( rte_grp_handle_t group );

RTE_PUBLIC int (*rte_my_rank)( rte_grp_handle_t group);

RTE_PUBLIC rte_grp_handle_t (*rte_world_group)(void);


/* The following 3 functions are needed to implement non-blocking collectives that have to
 * return some handle to runtime. So, let's the runtime define what it's gonna use as a
 * handle
 */
RTE_PUBLIC void* (*rte_get_coll_handle)(void);

/**
 * @brief Tests an opaque handle for completion
 * @param [in] void* handle
 * @return 1 if completed, 0 if not
 */
RTE_PUBLIC int (*rte_coll_handle_test)(void *handle);

RTE_PUBLIC void (*rte_coll_handle_free)(void *handle);
/**
 * @brief Mark a collective request as a completed
 */
RTE_PUBLIC void (*rte_coll_handle_complete)(void *handle);


/**
 * @brief Return a 16bit wide runtime group id indetifier.
 *        NOTE: if a runtime can not implement this routine, e.g. if group
 *              ids of the runtime don't fit into 16 bits, then it should
 *              be left unimplemented and the corresponding function pointer
 *              hcoll_rte_functions.rte_group_id_fn should be set to NULL.
 *              In this case libhcoll would generate group id internally with
 *              the cost of extra collective communication during
 *              hcoll_context_create.
 */
RTE_PUBLIC int (*rte_group_id)(rte_grp_handle_t group);

RTE_PUBLIC int (*rte_world_rank)(rte_grp_handle_t group, rte_ec_handle_t ec_handle);


typedef enum {
    HCOLL_MPI_COMBINER_DUP,
    HCOLL_MPI_COMBINER_CONTIGUOUS,
    HCOLL_MPI_COMBINER_VECTOR,
    HCOLL_MPI_COMBINER_HVECTOR,
    HCOLL_MPI_COMBINER_INDEXED,
    HCOLL_MPI_COMBINER_HINDEXED,
    HCOLL_MPI_COMBINER_INDEXED_BLOCK,
    HCOLL_MPI_COMBINER_HINDEXED_BLOCK,
    HCOLL_MPI_COMBINER_STRUCT,
    HCOLL_MPI_COMBINER_SUBARRAY,
    HCOLL_MPI_COMBINER_DARRAY,
    HCOLL_MPI_COMBINER_F90_REAL,
    HCOLL_MPI_COMBINER_F90_COMPLEX,
    HCOLL_MPI_COMBINER_F90_INTEGER,
    HCOLL_MPI_COMBINER_RESIZED,
    HCOLL_MPI_COMBINER_LAST
} hcoll_mpi_type_combiner_t;

RTE_PUBLIC int (*rte_get_mpi_type_envelope)(void *mpi_type, int *num_integers,
                                            int *num_addresses, int *num_datatypes,
                                            hcoll_mpi_type_combiner_t *combiner);

RTE_PUBLIC int (*rte_get_mpi_constants)(size_t *mpi_datatype_size,
                                        int *mpi_order_c, int *mpi_order_fortran,
                                        int *mpi_distribute_block,
                                        int *mpi_distribute_cyclic,
                                        int *mpi_distribute_none,
                                        int *mpi_distribute_dflt_darg);

RTE_PUBLIC int (*rte_get_mpi_type_contents)(void *mpi_type, int max_integers, int max_addresses, int max_datatypes,
                                            int *array_of_integers, void *array_of_addresses, void *array_of_datatypes);

RTE_PUBLIC int (*rte_get_hcoll_type)(void *mpi_type, hcoll_datatype_t *hcoll_type);
RTE_PUBLIC int (*rte_set_hcoll_type)(void *mpi_type, hcoll_datatype_t hcoll_type);
struct rte_comm_fns {

    /*ALL the following functions are to be implemented by runtime
     * ----------------------------------------------------------*/
    /* nonblocking receive function */
    rte_recv_nb recv_fn ;
    /* nonblocking send function */
    rte_send_nb send_fn ;
    /* test for communication completion */
    rte_test test_fn ;
    /* compare if two handles reference the same underlying end -
       point */
    rte_ec_handle_compare ec_cmp_fn ;
    /* get a list of communication handles */
    rte_get_ec_handles get_ec_handles_fn ;
    /* get the number of procs in a group */
    rte_group_size rte_group_size_fn;
    /* get my rank in the group */
    rte_my_rank rte_my_rank_fn;
    /* check if the endpoint_connection is on the same node with
      the current process */
    rte_ec_on_local_node rte_ec_on_local_node_fn;
    /* get the handle to the group describing the whole job:
       i.e. all the launched processes */
    rte_world_group rte_world_group_fn;
    /* get the job identifier */
    rte_jobid rte_jobid_fn;
    /* progress runtime send/recv apis */
    rte_progress rte_progress_fn;
    rte_get_coll_handle rte_get_coll_handle_fn;
    rte_coll_handle_test rte_coll_handle_test_fn;
    rte_coll_handle_free rte_coll_handle_free_fn;
    rte_coll_handle_complete rte_coll_handle_complete_fn;

    /*This is the only one implemented internally by hcol
     * ---------------------------------------------------*/
    /* wait for rte_send_nb/rte_recv_nb completion */
    rte_wait_completion rte_wait_completion_fn;
    rte_group_id rte_group_id_fn;
    rte_world_rank rte_world_rank_fn;
    /* MPI Type fuctionality
     * The following 5 APIs are optional and need to be implemented
     * only if the full MPI datatypes support is required in the libhcoll.
     */
    rte_get_mpi_constants     rte_get_mpi_constants_fn;
    rte_get_mpi_type_envelope rte_get_mpi_type_envelope_fn;
    rte_get_mpi_type_contents rte_get_mpi_type_contents_fn;
    rte_get_hcoll_type        rte_get_hcoll_type_fn;
    rte_set_hcoll_type        rte_set_hcoll_type_fn;
};
typedef struct rte_comm_fns rte_comm_fns_t;

extern rte_comm_fns_t hcoll_rte_functions;

#define HCOLRTE_DECLSPEC extern

#endif /* HCOL_RUNTIME_API_H */
