/**
 * Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#if !defined(SHARP_MPI_TEST_H)
#define SHARP_MPI_TEST_H

#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "api/sharp.h"
#include "mpi.h"

#if HAVE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#endif

extern int sharp_world_rank;
extern int enable_sharp_coll;
extern int perf_test_iterations;
extern int perf_test_skips;
extern int perf_with_barrier;
extern int nbc_count;
extern int coll_root_rank;

enum test_mode
{
    TEST_BASIC,
    TEST_COMPLEX,
    TEST_PERF,
    TEST_ALL
};

enum test_coll
{
    TEST_ALLREDUCE = 1 << 0,
    TEST_IALLREDUCE = 1 << 1,
    TEST_REDUCE = 1 << 2,
    TEST_IREDUCE = 1 << 3,
    TEST_BARRIER = 1 << 4,
    TEST_IBARRIER = 1 << 5,
    TEST_BCAST = 1 << 6,
    TEST_REDUCE_SCATTER = 1 << 7,
    TEST_ALLGATHER = 1 << 8,
};

enum
{
    USE_MALLOC,
    USE_HUGETLB,
    USE_HUGETHP
};

enum
{
    TEST_DATA_LAYOUT_CONTIG,
    TEST_DATA_LAYOUT_IOV
};

struct data_template
{
    unsigned int unsigned_int_val;
    int int_val;
    unsigned long unsigned_long_val;
    long long_val;
    float float_val;
    double double_val;
    unsigned short unsigned_short_val;
    short short_val;
    unsigned char uint8_val;
    char int8_val;
    short float_short_val;

    unsigned int unsigned_int_loc_tag_val;
    int int_loc_tag_val;
    unsigned long unsigned_long_loc_tag_val;
    long long_loc_tag_val;
    float float_loc_tag_val;
    double double_loc_tag_val;
    unsigned short unsigned_short_loc_tag_val;
    short short_loc_tag_val;
    short float_short_loc_tag_val;
    int int_min_max_loc_result;
};

struct sharp_conf_t
{
    int rank;
    int size;
    long jobid;
    enum test_mode test_mode;
    int host_allocator_type;
    int sdata_layout;
    int siov_count;
    int rdata_layout;
    int riov_count;
    uint32_t run_colls;
    size_t min_message_size;
    size_t max_message_size;
    enum sharp_data_memory_type s_mem_type;
    enum sharp_data_memory_type r_mem_type;
    int datatype;
    int run_ppn_comm_parallel;
};
typedef struct sharp_conf_t sharp_conf_t;

/*TODO move it dtype.h */
struct sharp_test_data_types_t
{
    char name[64];
    int id;
    MPI_Datatype mpi_id;
    int size;
};

struct sharp_op_type_t
{
    char name[64];
    int id;
    MPI_Op mpi_op;
};

enum
{
    PARSE_ARGS_OK = 0,
    PARSE_ARGS_HELP = 1,
    INVALID_ARG = 2,
    INVALID_MAX_GROUPS = 3,
    INVALID_GROUP_TYPE = 4,
    INVALID_TEST_MODE = 5,
    INVALID_COLL_LIST = 6,
    INVALID_MEM_TYPE = 7,
};

/* comm types */
enum
{
    COMM_TYPE_WORLD = 0,
    COMM_TYPE_WORLD_DUP = 1,
    COMM_TYPE_WORLD_REVERSE = 2,
    COMM_TYPE_N_SPLIT = 3,
    COMM_TYPE_N_SPLIT_REVERSE = 4,
    COMM_TYPE_HALF = 5,
    COMM_TYPE_PPN_COMM = 6,
    COMM_TYPE_RANDOM = 7,
    COMM_TYPE_PPN_JOBS = 8,
    COMM_TYPE_MAX = 9,
};

struct coll_sharp_component_t
{
    struct sharp_coll_context* sharp_coll_context;
    struct sharp_coll_caps sharp_caps;
    int is_leader;
    int ppn;
    int node_local_rank;
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int src_shmid;
    int dst_shmid;
};
typedef struct coll_sharp_component_t coll_sharp_component_t;

struct coll_sharp_module_t
{
    struct sharp_coll_comm* sharp_coll_comm;
    MPI_Comm comm;
};
typedef struct coll_sharp_module_t coll_sharp_module_t;
extern coll_sharp_component_t coll_sharp_component;
extern sharp_conf_t sharp_conf;

void coll_test_sharp_allreduce_complex(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_allreduce_basic(coll_sharp_module_t* sharp_comm, int is_rooted);
void coll_test_sharp_allreduce_perf(coll_sharp_module_t* sharp_comm, int is_rooted);

void coll_test_sharp_iallreduce_complex(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_iallreduce_basic(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_iallreduce_perf(coll_sharp_module_t* sharp_comm);

void coll_test_sharp_barrier_complex(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_barrier_basic(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_ibarrier_basic(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_barrier_perf(coll_sharp_module_t* sharp_comm);

void coll_test_sharp_bcast_basic(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_bcast_perf(coll_sharp_module_t* sharp_comm);

void coll_test_sharp_reduce_scatter_basic(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_reduce_scatter_perf(coll_sharp_module_t* sharp_comm);

void coll_test_sharp_allgather_basic(coll_sharp_module_t* sharp_comm);
void coll_test_sharp_allgather_perf(coll_sharp_module_t* sharp_comm);
/* The i-th bit */
#define __SHARP_TEST_BIT(i) (1ull << (i))

#define __GETTIMEOFDAY  0
#define __CLOCK_GETTIME 1

#define __TIMER __CLOCK_GETTIME

#define SHARP_MSEC_PER_SEC 1000ull      /* Milli */
#define SHARP_USEC_PER_SEC 1000000ul    /* Micro */
#define SHARP_NSEC_PER_SEC 1000000000ul /* Nano */

#if __TIMER == __GETTIMEOFDAY
static inline double sharp_time_sec(void)
{
    double wtime;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    wtime = tv.tv_sec;
    wtime += (double)tv.tv_usec / SHARP_USEC_PER_SEC;
    return wtime;
}
#elif __TIMER == __CLOCK_GETTIME
static inline double sharp_time_sec()
{
    struct timespec tv;
    double wtime;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    wtime = (tv.tv_sec);
    wtime += (double)(tv.tv_nsec) / SHARP_NSEC_PER_SEC;
    return wtime;
}
#else
#error "unknown timer"
#endif

#define sharp_time_msec() (sharp_time_sec() * SHARP_MSEC_PER_SEC)
#define sharp_time_usec() (sharp_time_sec() * SHARP_USEC_PER_SEC)
#define sharp_time_nsec() (sharp_time_sec() * SHARP_NSEC_PER_SEC)
#endif /*SHARP_MPI_TEST_H*/
