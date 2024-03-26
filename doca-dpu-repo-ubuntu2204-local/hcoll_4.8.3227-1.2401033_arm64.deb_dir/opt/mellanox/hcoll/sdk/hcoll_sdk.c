/**
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "hcoll/api/hcoll_api.h"

#define MAX_TAG 10000
#define host_len 100

struct request_t
{
    int busy_place;                 /* indicate if the request is being used at the moment */
    int req_complete;               /* indicate if the request is completed */
    MPI_Request m_req;
};

typedef struct request_t request_t;

struct proc_t
{
    int world_rank;                 /* rank in MPI_COMM_WORLD communicator */
    char host_name[host_len];       /* host name where process is running at */
};

typedef struct proc_t proc_t;

/* intracommunicator representation only */
typedef struct group_t
{
    struct group_t* parent_grp;     /* pointer to parent communicator */
    proc_t** proc_arr;              /* array of processes from local group of communicator */
    int m_rank;                     /* rank of the current process in this group */
    int size;                       /* number of processes in group */
    int cid;                        /* context id of this communicator */
    MPI_Comm ptr;
} group_t;

long req_max_count = 1000000;
request_t** req_array = NULL;
group_t world_group;
static proc_t** proc_array = NULL;

static int request_complete(request_t* request)
{
    request->req_complete = 1;
    return 0;
}

/* chech if the current process is in group */
static int im_in_group(rte_grp_handle_t grp)
{
    if (((group_t*)grp)->m_rank != -1)
        return 1;
    return 0;
}

/* print error messages */
void print_error(const char* msg)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("%s\n", msg);
    }
}

/* free memory */
void global_free()
{
    int i;
    if (req_array != NULL)
    {
        for (i = 0; i < req_max_count; i++)
        {
            if (req_array[i] != NULL)
            {
                free(req_array[i]);
            }
        }
        free(req_array);
    }
    if (proc_array != NULL)
    {
        int num_proc;
        MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
        for (i = 0; i < num_proc; i++)
        {
            if (proc_array[i] != NULL)
            {
                free(proc_array[i]);
            }
        }
        free(proc_array);
    }
}

/* convert dte_general_representation data into regular iovec array which is
  used in rml
  */
static inline int count_total_dte_repeat_entries(struct dte_data_representation_t *data){
    unsigned int i;

    struct dte_generalized_iovec_t * dte_iovec =
            data->rep.general_rep->data_representation.data;
    int total_entries_number = 0;
    for (i=0; i< dte_iovec->repeat_count; i++){
        total_entries_number += dte_iovec->repeat[i].n_elements;
    }
    return total_entries_number;
}

/*********************************************
 *********************************************
 The following functions should be implemented
 in order to use hcoll
 ********************************************/

/* non-blocking send implementation -- This is implemented for HCOLL wireup purposes */
static int send_nb(dte_data_representation_t data,
                   uint32_t count,
                   void *buffer,
                   rte_ec_handle_t ec_h,
                   rte_grp_handle_t grp_h,
                   uint32_t tag, rte_request_handle_t *req)
{
    int tag_temp = (int) tag;
#if RTE_DEBUG
    assert(ec_h.group == grp_h);
#endif
    if (! ec_h.handle)
    {
        fprintf(stderr,"***Error in hcolrte_rml_recv_nb: wrong null argument: "
                "ec_h.handle = %p, ec_h.rank = %d\n",ec_h.handle,ec_h.rank);
        return 1;
    }
    /*do inline nb recv*/
    int rc;
    size_t size;
    request_t* request;
    int err;
    group_t* comm;
    comm = (group_t*)grp_h;

    if (!buffer && !HCOL_DTE_IS_ZERO(data)) {
        fprintf(stderr, "***Error in hcolrte_rml_recv_nb: buffer pointer is NULL"
                " for non DTE_ZERO INLINE data representation\n");
        return 1;
    }
    size = (size_t)data.rep.in_line_rep.data_handle.in_line.packed_size*count/8;
    request = (request_t*)malloc(sizeof(request_t));
    if (request == NULL)
    {
        fprintf(stderr, "***Error in hcolrte_rml_recv_nb: couldn't allocate memory for request data\n");
        return 1;
    }
    request->req_complete = 0;

    /* put here -tag because hcoll uses negative tags for internal stuff mostly because OMPI reserves negative
     * tags for internal collectives, but in this case we are above the PML level and we must use positive tags
     * else MPI_Isend/recv complain about invalid tags.
     */
    if(tag_temp < 0){
        tag_temp = -1*tag_temp;
    }
    err = MPI_Isend(buffer, size, MPI_UNSIGNED_CHAR, ec_h.rank, tag_temp, ((group_t*)grp_h)->ptr, &(request->m_req));
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "***Error in hcolrte_rml_recv_nb: MPI_Isend returned incorrect status\n");
        free(request);
        return 1;
    }
    req->data = (void *)request;
    req->status = HCOLRTE_REQUEST_ACTIVE;


    return HCOLL_SUCCESS;
}

/* non-blocking recv implementation This is implemented for HCOLL wireup services */
static int recv_nb(dte_data_representation_t data ,
        uint32_t count ,
        void *buffer,
        rte_ec_handle_t ec_h,
        rte_grp_handle_t grp_h,
        uint32_t tag,
        rte_request_handle_t * req)
{
    int tag_temp = (int) tag;
#if RTE_DEBUG
    assert(ec_h.group == grp_h);
#endif
    if (! ec_h.handle)
    {
        fprintf(stderr,"***Error in hcolrte_rml_recv_nb: wrong null argument: "
                "ec_h.handle = %p, ec_h.rank = %d\n",ec_h.handle,ec_h.rank);
        return 1;
    }
    /*do inline nb recv*/
    int rc;
    size_t size;
    request_t* request;
    int err;
    group_t* comm;
    comm = (group_t*)grp_h;

    if (!buffer && !HCOL_DTE_IS_ZERO(data)) {
        fprintf(stderr, "***Error in hcolrte_rml_recv_nb: buffer pointer is NULL"
                " for non DTE_ZERO INLINE data representation\n");
        return 1;
    }
    size = (size_t)data.rep.in_line_rep.data_handle.in_line.packed_size*count/8;
    request = (request_t*)malloc(sizeof(request_t));
    if (request == NULL)
    {
        fprintf(stderr, "***Error in hcolrte_rml_recv_nb: couldn't allocate memory for request data\n");
        return 1;
    }
    request->req_complete = 0;

    /* put here -tag because hcoll uses negative tags for internal stuff mostly because OMPI reserves negative
     * tags for internal collectives, but in this case we are above the PML level and we must use positive tags
     * else MPI_Isend/recv complain about invalid tags.
     */
    if(tag_temp < 0 ){
        tag_temp = -tag_temp;
    }
    err = MPI_Irecv(buffer, size, MPI_UNSIGNED_CHAR, ec_h.rank, tag_temp, ((group_t*)grp_h)->ptr, &(request->m_req));
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "***Error in hcolrte_rml_recv_nb: MPI_Isend returned incorrect status\n");
        free(request);
        return 1;
    }
    req->data = (void *)request;
    req->status = HCOLRTE_REQUEST_ACTIVE;


    return HCOLL_SUCCESS;
}

/* check if request completed */
static int test( rte_request_handle_t * request ,
                 int * completed )
{
    request_t * req  = (request_t*)(request->data);
    if (HCOLRTE_REQUEST_ACTIVE != request->status)
    {
        *completed = 1;
        return HCOLL_SUCCESS;
    }

    MPI_Status status;
    MPI_Request_get_status( req->m_req, &(req->req_complete), &status );
    *completed = req->req_complete;
    if (*completed)
    {
        free(req);
        request->status = HCOLRTE_REQUEST_DONE;
    }

    return HCOLL_SUCCESS;
}

/* progress hcoll internal staff */
static void progress(void)
{
    hcoll_progress_fn();
}

/* Each process (connection) can be represented via unique pair: ec_handle + group. */
/* In this implementation ec_handle is represented by structure proc_t
 * which is the same in all groups the process is related to */

/* This function compares handles of two processes (connections) */
static int ec_handle_compare( rte_ec_handle_t handle_1 ,
                              rte_grp_handle_t
                              group_handle_1 ,
                              rte_ec_handle_t handle_2 ,
                              rte_grp_handle_t
                              group_handle_2 )
{
    return handle_1.handle == handle_2.handle;
}

/* return num_ec handles of processes with the specified ec_indexes in the specified group */
/* In this implementation ec_index is matching to the rank in the group. */
static int get_ec_handles( int num_ec ,
                           int * ec_indexes ,
                           rte_grp_handle_t grp_h,
                           rte_ec_handle_t * ec_handles )
{
    int i;
    int w_rank;
    group_t* grp = (group_t*)grp_h;
    for (i = 0; i < num_ec; i++)
    {
        proc_t* proc = grp->proc_arr[ec_indexes[i]];
#ifdef RTE_DEBUG
        ec_handles[i].group = grp_h;
#endif
        ec_handles[i].rank = ec_indexes[i];
        ec_handles[i].handle = (void *)proc;
    }
    return HCOLL_SUCCESS;
}

/* get ec_handle of current process in group */
static int get_my_ec(rte_grp_handle_t grp_h, rte_ec_handle_t *ec_handle)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    proc_t* my_proc = proc_array[my_rank];
    ec_handle->handle = (void *)my_proc;
    ec_handle->rank = my_rank;
#ifdef RTE_DEBUG
    ec_handle->group = grp_h;
#endif
    return HCOLL_SUCCESS;
}

/* check if the process is running at the same node as currect process */
static int ec_on_local_node (rte_ec_handle_t ec, rte_grp_handle_t group)
{
    char my_hostname[host_len];
    gethostname(my_hostname, host_len);
    proc_t *proc = (proc_t *)ec.handle;
    if (strncmp(my_hostname, proc->host_name, host_len))
    {
        return 0;
    }
    return 1;
}

/* return number of processes in group */
static int group_size ( rte_grp_handle_t group )
{
    return ((group_t *)group)->size;
}

/* return rank of process in group */
static int my_rank (rte_grp_handle_t grp_h)
{
    return ((group_t *)grp_h)->m_rank;
}

/* return handle of internal group matching to MPI_COMM_WORLD communicator */
static rte_grp_handle_t get_world_group_handle(void)
{
    return (rte_grp_handle_t)(&world_group);
}

/* return job id */
static uint32_t jobid(void)
{
    return 0;
}

/* get first free request from the request array for collective operation */
static void* get_coll_handle(void)
{
    request_t *req = NULL;
    int i = 0, j = 0;
    while (i < req_max_count)
    {
        if (!req_array[i]->busy_place)
        {
            req = req_array[i];
            /* set busy_place to 1 */
            req->busy_place = 1;
            req->req_complete = 0;
            break;
        }
        i++;
    }
    /* if there is no enough memory for next request, try to realloc it */
    if (i == req_max_count-1)
    {
        req_max_count *= 2;
        request_t** tmp;
        tmp = malloc(sizeof(request_t*) * req_max_count);
        if (tmp == NULL)
        {
            print_error("There is no enough memory for coll handle. Couldn't reallocate.");
            return NULL;
        }
        for (j = 0; j < req_max_count; j++)
        {
            tmp[j] = NULL;
        }
        memcpy(tmp, req_array, sizeof(req_array)/sizeof(request_t*));
        free(req_array);
        req_array = tmp;

        req = req_array[req_max_count/2];
        /* set busy_place to 1 */
        req->busy_place = 1;
        req->req_complete = 0;
    }
    return (void *)req;
}

/* check if collective operation is completed */
static int coll_handle_test(void* handle)
{
    request_t *req = (request_t *)handle;
    return req->req_complete;
}

/* indicate that this request handle can now be used for another collective operation  */
static void coll_handle_free(void *handle)
{
    request_t *req = (request_t *)handle;
    /* set busy_place to 0 */
    req->busy_place = 0;
    req->req_complete = 0;
}

/* indicate that collective operation is finished */
static void coll_handle_complete(void *handle)
{
    request_t *req = (request_t *)handle;
    request_complete(req);
}

/* return commucicator id */
static int group_id(rte_grp_handle_t group)
{
   return ((group_t*)group)->cid;
}

/* return rank of process with known ec in MPI_COMM_WORLD communicator */
static int world_rank(rte_grp_handle_t grp_h, rte_ec_handle_t ec)
{
    group_t* grp = (group_t*)grp_h;
    return grp->proc_arr[ec.rank]->world_rank;
}

/* initialize hcoll function */
static void init_functions(void)
{
    hcoll_rte_functions.send_fn = send_nb;
    hcoll_rte_functions.recv_fn = recv_nb;
    hcoll_rte_functions.ec_cmp_fn = ec_handle_compare;
    hcoll_rte_functions.get_ec_handles_fn = get_ec_handles;
    hcoll_rte_functions.rte_group_size_fn = group_size;
    hcoll_rte_functions.test_fn = test;
    hcoll_rte_functions.rte_my_rank_fn = my_rank;
    hcoll_rte_functions.rte_ec_on_local_node_fn = ec_on_local_node;
    hcoll_rte_functions.rte_world_group_fn = get_world_group_handle;
    hcoll_rte_functions.rte_jobid_fn = jobid;
    hcoll_rte_functions.rte_progress_fn = progress;
    hcoll_rte_functions.rte_get_coll_handle_fn = get_coll_handle;
    hcoll_rte_functions.rte_coll_handle_test_fn = coll_handle_test;
    hcoll_rte_functions.rte_coll_handle_free_fn = coll_handle_free;
    hcoll_rte_functions.rte_coll_handle_complete_fn = coll_handle_complete;
    hcoll_rte_functions.rte_group_id_fn = group_id;
    hcoll_rte_functions.rte_world_rank_fn = world_rank;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_rank;
    int count;
    int root;
    char *buffer = NULL, *correct_buffer = NULL;
    int i,j;
    int new_grp_size;
    group_t* comm = NULL;
    int num_iter = 1000;
    int num_elem = 1500;
    int err = MPI_SUCCESS;
    int rc = HCOLL_SUCCESS;
    int bcast_err = 0;
    int allreduce_err = 0;
    int cmp;
    hcoll_init_opts_t *opts;

    MPI_Comm_size(MPI_COMM_WORLD, &count);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 2)
    {
        if (world_rank == 0)
        {
            print_error("You can specify number of iterations via the first parameter and number of buffer elements via the second. Otherwise default parameters will be used.");
        }
    }
    else if ((argc < 3) || (argc > 3))
    {
        if (world_rank == 0)
        {
            print_error("Parameters are ignored, default will be used. If you want, please set number of iterations via the first parameter and number of buffer elements via the second.");
        }
    }
    else
    {
        /* set number of iterations */
        num_iter = atoi(argv[1]);
        /* set size of buffer */
        num_elem = atoi(argv[2]);
    }

    /* allocate memory for world's process array */
    proc_array = malloc(sizeof(proc_t*) * count);
    if (proc_array == NULL)
    {
        print_error("Couldn't allocate memory process array.");
        return -1;
    }

    /* allocate memory for request array */
    req_array = malloc(sizeof(request_t*) * req_max_count);
    if (req_array == NULL)
    {
        global_free();
        print_error("Couldn't allocate memory for request array.");
        return -1;
    }
    for (i = 0; i < req_max_count; i++)
    {
        req_array[i] = malloc(sizeof(request_t));
        if (req_array[i] == NULL)
        {
            global_free();
            print_error("Couldn't allocate memory for request.");
            return -1;
        }
        req_array[i]->busy_place = 0;
        req_array[i]->req_complete = 0;
    }

    /* gather information about each process's world rank and host name */
    char* send_buf = (char*)malloc(sizeof(MPI_BYTE)*host_len);
    gethostname(send_buf, host_len);
    char* recv_buf = (char*)malloc(sizeof(MPI_BYTE) * host_len * count);
    err = MPI_Allgather(send_buf,
                  host_len,
                  MPI_BYTE,
                  recv_buf,
                  host_len,
                  MPI_BYTE,
                  MPI_COMM_WORLD);

    if (err != MPI_SUCCESS)
    {
        print_error("MPI_Allgather returned incorrect status.");
        return -1;
    }

    char* send_rank = (char*)malloc(sizeof(int));
    *(int*)send_rank = world_rank;
    char* recv_rank = (char*)malloc(sizeof(int) * count);

    /* create processes and fill world's process array */
    for (i = 0; i < count; i++)
    {
        proc_t* proc = malloc(sizeof(proc_t));
        if (proc == NULL)
        {
            global_free();
            free(send_buf);
            free(recv_buf);
            free(send_rank);
            free(recv_rank);
            print_error("Couldn't allocate memory for process.");
            return -1;
        }
        proc->world_rank = i; /* it's just i //((int*)recv_rank)[i];*/
        strncpy(proc->host_name, recv_buf + i * host_len, (size_t)host_len);
        proc_array[i] = proc;
    }

    free(send_buf);
    free(recv_buf);
    free(send_rank);
    free(recv_rank);

    /* initialize world comm */
    world_group.parent_grp = NULL;
    world_group.proc_arr = proc_array;
    world_group.m_rank = world_rank;
    world_group.size = count;
    world_group.ptr = MPI_COMM_WORLD;
    world_group.cid = 0;

    /* create new comm from only even ranks */
    comm = (group_t*)malloc(sizeof(group_t));
    comm->m_rank = -1;
    new_grp_size = count/2;
    if (new_grp_size < 2)
    {
        print_error("The new group size shouldn't be less than 2. Please run more processes.");
        global_free();
        free(comm);
        return -1;
    }
    int* ranks = malloc(sizeof(int)*new_grp_size);
    proc_t** new_grp_procs = malloc(sizeof(proc_t*) * new_grp_size);
    for (i = 0; i < new_grp_size; i++)
    {
        for (j = 0; j < world_group.size; j++)
        {
            if (i*2 == (world_group.proc_arr[j])->world_rank)
            {
                new_grp_procs[i] = world_group.proc_arr[j];
                break;
            }
        }
        if (world_rank == new_grp_procs[i]->world_rank)
        {
            comm->m_rank = i;
        }
        ranks[i] = i*2;
    }
    MPI_Group orig_group, new_group;
    MPI_Comm new_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
    MPI_Group_incl(orig_group, new_grp_size, ranks, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
    MPI_Group_free(&new_group);
    free(ranks);
    if (new_comm == NULL)
    {
        print_error("Couldn't create a new communicator. MPI_Comm_create failed.");
    }
    comm->size = new_grp_size;
    comm->parent_grp = &world_group;
    comm->proc_arr = new_grp_procs;
    comm->cid = 1;
    comm->ptr = new_comm;

    /* init hcoll functions */
    init_functions();
    /* initialize hcoll */
    hcoll_read_init_opts(&opts);
    opts->base_tag = 100;
    opts->max_tag = MAX_TAG;
    opts->enable_thread_support = 0;
    rc = hcoll_init_with_opts(&opts);
    if (rc != HCOLL_SUCCESS)
    {
        global_free();
        free(new_grp_procs);
        free(comm);
        print_error("Hcoll initialization failed.");
        return -1;
    }

    if (hcoll_collectives.coll_barrier == NULL)
    {
        print_error("Barrier collective undefined.");
        global_free();
        free(new_grp_procs);
        free(comm);
        return -1;
    }

    /* allocate buffers */
    buffer = (char*)malloc(sizeof(double)*num_elem);
    if (buffer == NULL)
    {
        global_free();
        free(new_grp_procs);
        free(comm);
        print_error("Couldn't allocate memory for test buffer.");
        return -1;
    }
    /* allocate buffer for correct results to check results */
    correct_buffer = (char*)malloc(sizeof(double)*num_elem);
    if (correct_buffer == NULL)
    {
        global_free();
        free(new_grp_procs);
        free(comm);
        print_error("Couldn't allocate memory for correct buffer.");
        return -1;
    }
    /* fill correct buffer with correct values */
    for (i = 0; i < num_elem; i++)
    {
        ((double*)correct_buffer)[i] = 13.7;
    }

    /* will run test on two different communicators */
    group_t* test_comms[2];
    void* context[2];
    test_comms[0] = &world_group;
    test_comms[1] = comm;
    int c;
    /* run collectives on each communicator */
    for (c = 0; c < 2; c++)
    {
        group_t* test_comm = test_comms[c];
        if (test_comm->ptr == NULL)
        {
            continue;
        }
        if (im_in_group(test_comm))
        {
            /* At first we should create hcoll context for that group handle */
            context[c] = hcoll_create_context((rte_grp_handle_t)test_comm);
            if (context[c] == NULL)
            {
                print_error("Couldn't create context.");
                continue;
            }
            hcoll_collectives.coll_barrier(context[c]);
            bcast_err = -1;
            allreduce_err = -1;
            /* do multiple iterations */
            for (i = 0; i < num_iter; i++)
            {
                /* choose root */
                srand(i);
                root = rand() % test_comm->size;
                if (test_comm->m_rank == root)
                {
                    if (buffer)
                    {
                        for (j = 0; j < num_elem; j++)
                        {
                            ((double*)buffer)[j] = 13.7;
                        }
                    }
                }
                else
                {
                    if (buffer)
                    {
                        for (j = 0; j < num_elem; j++)
                        {
                            ((double*)buffer)[j] = -1.0;
                        }
                    }
                }

                /* call bcast */
                cmp = 0;
                if (hcoll_collectives.coll_bcast != NULL)
                {
                    if (bcast_err < 0 )
                        bcast_err = 0;
                    rc = hcoll_collectives.coll_bcast(buffer, num_elem, float64_dte, root, context[c]);
                    if (rc != HCOLL_SUCCESS)
                    {
                        print_error("Hcoll bcast returned incorrect status.");
                    }

                    /* compare resuts */
                    cmp = strncmp(correct_buffer, buffer, sizeof(correct_buffer));
                }
                /* summarize results using allreduce */
                int res = -1;
                int correct_res = -1;

                /* call other implementation of MPI_Allreduce to get bcast results and check hcoll allreduce */
                err = MPI_Allreduce (&cmp, &correct_res, 1, MPI_INT, MPI_SUM, test_comm->ptr);
                if (err != MPI_SUCCESS)
                {
                    print_error("MPI_Allreduce returned incorrect status. Can't check hcoll collectives correctness.");
                    free(buffer);
                    free(correct_buffer);
                    hcoll_context_free(context[c], (rte_grp_handle_t)test_comms[c]);
                    global_free();
                    free(comm);
                    free(new_grp_procs);
                    hcoll_finalize();
                    return -1;
                }
                else
                {
                    if (test_comm->m_rank == root)
                    {
                        if (correct_res != 0)
                        {
                            bcast_err++;
                        }
                    }

                    /* allreduce algorithm should be set by environment variable */
                    /* otherwise the function pointer is NULL */
                    if (hcoll_collectives.coll_allreduce != NULL)
                    {
                        if (allreduce_err < 0 )
                            allreduce_err = 0;
                        dte_data_representation_t Dtype = DTE_INT32;
                        hcoll_dte_op_t *Op = &hcoll_dte_op_sum;
                        rc = hcoll_collectives.coll_allreduce(&cmp, &res, 1, Dtype, Op, context[c]);
                        if (rc != HCOLL_SUCCESS)
                        {
                            print_error("Hcoll allreduce returned incorrect status.");
                        }
                        if (test_comm->m_rank == root)
                        {
                            if ((correct_res != res))
                            {
                                allreduce_err++;
                            }
                        }
                    }
                }
            }
            if (test_comm->m_rank == root)
            {
                printf("\nTest result on %s comm:\n", (c == 0) ? "MPI_COMM_WORLD" : "NEW");
                if (bcast_err >= 0)
                {
                    printf("Bcast test %s\n", (bcast_err == 0) ? "succeded" : "failed");
                }
                else
                {
                    printf("Hcoll bcast function pointer is NULL. So hcoll bcast test will not run.\n");
                }
                if (allreduce_err >= 0)
                {
                    printf("Allreduce test %s\n", (allreduce_err == 0) ? "succeded" : "failed");
                }
                else
                {
                    printf("Hcoll allreduce function pointer is NULL. So hcoll allreduce test will not run.\nPlease use command line option to specify the allreduce algorithm.\n");
                }
            }
        }
    }

    for(c=1; c>=0; c--) {
        if (im_in_group(test_comms[c])) {
            hcoll_context_free(context[c], (rte_grp_handle_t)test_comms[c]);
        }
    }

    free(buffer);
    free(correct_buffer);
    free(comm);
    free(new_grp_procs);
    global_free();

    hcoll_free_init_opts(opts);
    hcoll_finalize();

    MPI_Finalize();
    if ((bcast_err == 0) && (allreduce_err == 0))
        return 0;
    else
        return -1;
}
