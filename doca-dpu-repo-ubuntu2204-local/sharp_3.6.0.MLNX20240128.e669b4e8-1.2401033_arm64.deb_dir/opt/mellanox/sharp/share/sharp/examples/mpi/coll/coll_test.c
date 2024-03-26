/**
 * Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <coll_test.h>
#include <sys/time.h>

#define MAX_IB_DEVICES (16)
#define DEBUG          (0)

int client_id;
int enable_sharp_coll = 0;
int sharp_world_rank;
int sharp_world_size;
int perf_test_iterations = 1000;
int perf_test_skips = 100;
int perf_with_barrier = 1;
int nbc_count = 512;
int coll_root_rank = -1;
int test_flat_ppn = 1;
int sharp_test_group_type = COMM_TYPE_WORLD;
int sharp_test_num_splits = 2; /* default odd even comm split */
coll_sharp_component_t coll_sharp_component;
char ib_dev_list[MAX_IB_DEVICES][16];
int num_ib_devices;

extern MPI_Datatype sharp_test_mpi_min_max_datatype[SHARP_DTYPE_NULL][SHARP_DTYPE_NULL];

const char* comm_type_names[] = {
    [COMM_TYPE_WORLD] = "Comm World",
    [COMM_TYPE_WORLD_DUP] = "Comm World Dup",
    [COMM_TYPE_WORLD_REVERSE] = "Comm World Reverse",
    [COMM_TYPE_N_SPLIT] = "Comm split (default: Odd/even)",
    [COMM_TYPE_N_SPLIT_REVERSE] = "Comm Split Reverse",
    [COMM_TYPE_HALF] = "Comm Split Half",
    [COMM_TYPE_PPN_COMM] = "Comm per PPN Rank",
    [COMM_TYPE_PPN_JOBS] = "Sharp job and Comm per PPN Rank",
};

static struct option long_options[] = {{"help", no_argument, NULL, 'h'},
                                       {"perf_with_barrier", required_argument, NULL, 'B'},
                                       {"collectives", required_argument, NULL, 'c'},
                                       {"iov_count", required_argument, NULL, 'C'},
                                       {"ib-dev", required_argument, NULL, 'd'},
                                       {"data_layout", required_argument, NULL, 'D'},
                                       {"iters", required_argument, NULL, 'i'},
                                       {"jobid", required_argument, NULL, 'j'},
                                       {"host_alloc_type", required_argument, NULL, 'H'},
                                       {"mode", required_argument, NULL, 'm'},
                                       {"mem_type", required_argument, NULL, 'M'},
                                       {"nbc_count", required_argument, NULL, 'N'},
                                       {"flat_ppn", required_argument, NULL, 'p'},
                                       {"size", required_argument, NULL, 's'},
                                       {"root", required_argument, NULL, 'r'},
                                       {"datatype", required_argument, NULL, 'T'},
                                       {"group_type", required_argument, NULL, 't'},
                                       {"skips", required_argument, NULL, 'x'},
                                       {0, 0, 0, 0}};

static const char usage_string[] =
    "Usage:   sharp_mpi_test  [OPTIONS]\n"
    "Options:\n"
    "-h, --help		Show this help message and exit\n"
    "-B, --perf_with_barrier Sync Allreduce with Barrier in Allreduce collective, Default:1 \n"
    "-c, --collectives	Comma separated list of collectives:[allreduce|iallreduce|reduce|ireduce|barrier|ibarrier|bcast|reduce_scatter|allgather|all|none]] to run.\n"
    "			 Default; run all blocking collectives\n"
    "-C, --iov_count	Number of entries in IOV list, if used. Default: 2 Max:SHARP_COLL_DATA_MAX_IOV \n"
    "-d, --ib-dev		Use IB device <dev:port> (default first device found)\n"
    "-D, --data_layout	Data layout (contig, iov) for sender and receiver side. Default: contig\n"
    "-j, --jobid		Explicit Job ID \n"
    "-i, --iters 		Number of iterations to run perf benchmark\n"
    "-H, --host_alloc_type	Host memory allocation method ( hugetlb, hugethp(MADVISE Transparant Hugepage support), malloc)\n"
    "-m, --mode		Test modes: <basic|complex|perf|all> . Default: basic\n"
    "-M, --mem_type		Memory type(host,cuda) used in the communication buffers format: <src memtype>:<recv memtype>\n"
    "-N, --nbc_count 	Number of non-blocking operation posted before waiting for completion (max: 512)\n"
    "-p, --flat_ppn		Number of process per node default:1 \n"
    "-r, --root		Root for bcast, reduce operations \n"
    "-s, --size		Set  the minimum and/or the maximum message size. format:[MIN:]MAX Default:<4:max_ost_payload>\n"
    "-T, --datatype		Datatype of SHARP aggregation request(half_int, half_float, int, float, long, double) \n"
    "-t, --group_type	Set specific group type(world(default), world-dup, world-rev, half, slipt:<n>, split-rev:<n>, split:ppn-comm, split:ppn-jobs) to test\n"
    "-x, --skips		Number of warmup iterations to run perf benchmark\n";

sharp_conf_t sharp_conf = {
    .rank = 0,
    .size = 0,
    .jobid = 0,
    .test_mode = TEST_PERF,
    .sdata_layout = TEST_DATA_LAYOUT_CONTIG,
    .siov_count = 2,
    .rdata_layout = TEST_DATA_LAYOUT_CONTIG,
    .host_allocator_type = USE_MALLOC,
    .riov_count = 2,
    .run_colls = TEST_ALLREDUCE | TEST_BARRIER,
    .min_message_size = 4,
    .max_message_size = 256,
    .s_mem_type = SHARP_MEM_TYPE_HOST,
    .r_mem_type = SHARP_MEM_TYPE_HOST,
    .datatype = SHARP_DTYPE_INT,
    .run_ppn_comm_parallel = 0,

};
int oob_bcast(void* comm_context, void* buf, int size, int root)
{
    MPI_Comm comm = (MPI_Comm)comm_context;
    MPI_Bcast(buf, size, MPI_BYTE, root, comm);
    return 0;
}

int oob_barrier(void* comm_context)
{
    MPI_Comm comm = (MPI_Comm)comm_context;
    MPI_Barrier(comm);
    return 0;
}
int oob_gather(void* comm_context, int root, void* sbuf, void* rbuf, int len)
{
    MPI_Comm comm = (MPI_Comm)comm_context;
    MPI_Gather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root, comm);
    return 0;
}

int oob_progress(void)
{
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    return flag;
}

static void setup_comm_manager_rank(MPI_Comm comm, coll_sharp_component_t* sharp)
{
    int name_length, size, i, rank;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* todo: do not use stack allocation but malloc */
    int name_len[size];
    int offsets[size];

    MPI_Get_processor_name(sharp->node_name, &name_length);

    /* collect hostname len from all ranks */
    MPI_Allgather(&name_length, 1, MPI_INT, &name_len[0], 1, MPI_INT, comm);

    /* calculate receive buffer byte count based on hostname len sum. */
    int bytes = 0;
    for (i = 0; i < size; ++i) {
        offsets[i] = bytes;
        bytes += name_len[i];
    }
    bytes++;

    char receive_buffer[bytes];
    receive_buffer[bytes - 1] = 0;

    // collect hostnames, form comma separated list
    MPI_Allgatherv(&sharp->node_name[0], name_length, MPI_CHAR, &receive_buffer[0], &name_len[0], &offsets[0], MPI_CHAR, comm);

    int node_rank_count = 0;
    int node_rank_min = rank;

    for (i = 0; i < size; i++) {
        int idx = offsets[i];
        int len = name_len[i];

        if (name_length != len) {
            continue;
        }

        if (0 == strncmp(sharp->node_name, &receive_buffer[idx], len)) {
            if (i == rank) {
                sharp->node_local_rank = node_rank_count;
#if DEBUG
                printf(stdout "rank:%d local rank:%d\n", i, sharp->node_local_rank);
#endif
            }
            node_rank_count++;
            if (node_rank_min > i) {
                node_rank_min = i;
            }
        }
    }

    sharp->is_leader = (rank == node_rank_min);
    sharp->ppn = node_rank_count;
}

int coll_component_open(MPI_Comm mpi_comm)
{
    int ret;
    struct sharp_coll_init_spec init_spec = {0};
    struct timeval tval;
    int job_size, job_rank;

    coll_sharp_component.sharp_coll_context = NULL;

    if (!enable_sharp_coll)
        return 0;

    MPI_Comm_size(mpi_comm, &job_size);
    MPI_Comm_rank(mpi_comm, &job_rank);

    gettimeofday(&tval, NULL);
    srand((int)tval.tv_usec);

    init_spec.progress_func = oob_progress;
    /* coverity[dont_call] */
    if (sharp_conf.jobid && sharp_test_group_type != COMM_TYPE_PPN_JOBS) {
        init_spec.job_id = sharp_conf.jobid;
    } else {
        init_spec.job_id = (gethostid() << 32) | (rand() ^ getpid());
    }

    MPI_Bcast(&init_spec.job_id, sizeof(init_spec.job_id), MPI_BYTE, 0, mpi_comm);

    init_spec.world_rank = job_rank;
    init_spec.world_size = job_size;
    init_spec.enable_thread_support = 0;
    if (test_flat_ppn || sharp_test_group_type == COMM_TYPE_PPN_JOBS) {
        init_spec.world_local_rank = 0;
        init_spec.group_channel_idx = 0;
    } else {
        init_spec.world_local_rank = coll_sharp_component.node_local_rank;
        init_spec.group_channel_idx = coll_sharp_component.node_local_rank;
        if (sharp_test_group_type == COMM_TYPE_PPN_COMM) {
            init_spec.group_channel_idx = 0;
        }
    }

    init_spec.oob_colls.barrier = oob_barrier;
    init_spec.oob_colls.bcast = oob_bcast;
    init_spec.oob_colls.gather = oob_gather;
    init_spec.oob_ctx = (void*)mpi_comm;
    init_spec.config = sharp_coll_default_config;
    init_spec.config.ib_dev_list = ib_dev_list[coll_sharp_component.node_local_rank % num_ib_devices];

    if (test_flat_ppn && coll_sharp_component.ppn > num_ib_devices &&
        (sharp_test_group_type != COMM_TYPE_PPN_COMM && sharp_test_group_type != COMM_TYPE_PPN_JOBS))
    {
        fprintf(stderr, "PPN > 1 not supported with this group type \n");
        return -1;
    }

#if DEBUG
    fprintf(stdout,
            "[rank:%d, localrank:%d] Using IB device :%s world_local_rank:%d group_channel_idx:%d \n",
            sharp_world_rank,
            coll_sharp_component.node_local_rank,
            init_spec.config.ib_dev_list,
            init_spec.world_local_rank,
            init_spec.group_channel_idx);
#endif

    ret = sharp_coll_init(&init_spec, &coll_sharp_component.sharp_coll_context);
    if (ret != SHARP_COLL_SUCCESS) {
        return -1;
    }

    ret = sharp_coll_caps_query(coll_sharp_component.sharp_coll_context, &coll_sharp_component.sharp_caps);

    coll_sharp_component.src_shmid = 0;
    coll_sharp_component.dst_shmid = 0;

    return ret;
}

int coll_module_enable(MPI_Comm comm, coll_sharp_module_t* sharp_module)
{
    int size, rank, ret;
    struct sharp_coll_comm_init_spec comm_spec;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    sharp_module->comm = comm;
    sharp_module->sharp_coll_comm = NULL;

    if (!enable_sharp_coll)
        return 0;

    comm_spec.rank = rank;
    comm_spec.size = size;
    comm_spec.oob_ctx = (void*)sharp_module->comm;
    comm_spec.group_world_ranks = NULL;
    ret = sharp_coll_comm_init(coll_sharp_component.sharp_coll_context, &comm_spec, &sharp_module->sharp_coll_comm);
    if (ret < 0) {
        fprintf(stdout, "sharp communicator creation failed: %s\n", sharp_coll_strerror(ret));
        return -1;
    }
    return 0;
}

void coll_module_destroy(coll_sharp_module_t* sharp_module)
{
    if (!enable_sharp_coll || !sharp_module->sharp_coll_comm)
        return;
    sharp_coll_comm_destroy(sharp_module->sharp_coll_comm);
    sharp_module->sharp_coll_comm = NULL;
    sharp_module->comm = NULL;
}
void coll_component_close(void)
{
    if (!enable_sharp_coll || !coll_sharp_component.sharp_coll_context)
        return;
    sharp_coll_finalize(coll_sharp_component.sharp_coll_context);
}

static void sharp_run_basic(coll_sharp_module_t* sharp_comm)
{
    if (sharp_conf.run_colls & TEST_ALLREDUCE)
        coll_test_sharp_allreduce_basic(sharp_comm, 0);
    if (sharp_conf.run_colls & TEST_IALLREDUCE)
        coll_test_sharp_iallreduce_basic(sharp_comm);
    if (sharp_conf.run_colls & TEST_REDUCE)
        coll_test_sharp_allreduce_basic(sharp_comm, 1);
    if (sharp_conf.run_colls & TEST_BARRIER)
        coll_test_sharp_barrier_basic(sharp_comm);
    if (sharp_conf.run_colls & TEST_IBARRIER)
        coll_test_sharp_ibarrier_basic(sharp_comm);
    if (sharp_conf.run_colls & TEST_BCAST)
        coll_test_sharp_bcast_basic(sharp_comm);
    if (sharp_conf.run_colls & TEST_REDUCE_SCATTER)
        coll_test_sharp_reduce_scatter_basic(sharp_comm);
    if (sharp_conf.run_colls & TEST_ALLGATHER)
        coll_test_sharp_allgather_basic(sharp_comm);
}

static void sharp_run_complex(coll_sharp_module_t* sharp_comm)
{
    if (sharp_conf.run_colls & TEST_ALLREDUCE)
        coll_test_sharp_allreduce_complex(sharp_comm);
    if (sharp_conf.run_colls & TEST_IALLREDUCE)
        coll_test_sharp_iallreduce_complex(sharp_comm);
    if (sharp_conf.run_colls & TEST_BARRIER)
        coll_test_sharp_barrier_complex(sharp_comm);
}
static void sharp_run_perf(coll_sharp_module_t* sharp_comm)
{
    if (sharp_conf.run_colls & TEST_ALLREDUCE)
        coll_test_sharp_allreduce_perf(sharp_comm, 0);
    if (sharp_conf.run_colls & TEST_IALLREDUCE)
        coll_test_sharp_iallreduce_perf(sharp_comm);
    if (sharp_conf.run_colls & TEST_REDUCE)
        coll_test_sharp_allreduce_perf(sharp_comm, 1);
    if (sharp_conf.run_colls & TEST_BARRIER)
        coll_test_sharp_barrier_perf(sharp_comm);
    if (sharp_conf.run_colls & TEST_BCAST)
        coll_test_sharp_bcast_perf(sharp_comm);
    if (sharp_conf.run_colls & TEST_REDUCE_SCATTER)
        coll_test_sharp_reduce_scatter_perf(sharp_comm);
    if (sharp_conf.run_colls & TEST_ALLGATHER)
        coll_test_sharp_allgather_perf(sharp_comm);
}

void coll_module_op(coll_sharp_module_t* sharp_comm)
{
    if (sharp_conf.test_mode == TEST_BASIC) {
        sharp_run_basic(sharp_comm);
    } else if (sharp_conf.test_mode == TEST_COMPLEX) {
        sharp_run_complex(sharp_comm);
    } else if (sharp_conf.test_mode == TEST_PERF) {
        sharp_run_perf(sharp_comm);
    } else {
        sharp_run_basic(sharp_comm);
        sharp_run_complex(sharp_comm);
        sharp_run_perf(sharp_comm);
    }
}

void sharp_setenv(char* sharp_var, char* env_var, char* default_val)
{
    char* tmp = getenv(env_var);

    if (NULL == tmp) {
        if (NULL != default_val) {
            tmp = default_val;
        } else {
            return;
        }
    }
    setenv(sharp_var, tmp, 0);
}

void sharp_env2int(char* env_var, int* val, int default_val)
{
    long tmp;
    char *endptr, *str;

    *val = default_val;

    str = getenv(env_var);
    if (NULL == str)
        return;

    tmp = strtol(str, &endptr, 10);
    if (*endptr != '\0') {
        fprintf(stdout, "Invalid  %s environment value\n", env_var);
    } else {
        *val = tmp;
    }
}

void mpi_get_communicator(MPI_Comm* comm)
{
    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    switch (sharp_test_group_type) {
        case COMM_TYPE_WORLD:
            *comm = MPI_COMM_WORLD;
            break;
        case COMM_TYPE_WORLD_DUP:
            /* dup of comm_world */
            MPI_Comm_dup(MPI_COMM_WORLD, comm);
            break;
        case COMM_TYPE_WORLD_REVERSE:
            /*reverse comm */
            MPI_Comm_split(MPI_COMM_WORLD, 0, size - rank, comm);
            break;
        case COMM_TYPE_N_SPLIT:
            /* subset of world. odd, even groups */
            MPI_Comm_split(MPI_COMM_WORLD, (rank % sharp_test_num_splits), rank, comm);
            break;
        case COMM_TYPE_N_SPLIT_REVERSE:
            /* subset of world. odd, even groups in reserse*/
            MPI_Comm_split(MPI_COMM_WORLD, (rank % sharp_test_num_splits), (size - rank), comm);
            break;
        case COMM_TYPE_HALF:
            MPI_Comm_split(MPI_COMM_WORLD, (rank < (size / 2)), rank, comm);
            break;
        case COMM_TYPE_PPN_COMM:
        case COMM_TYPE_PPN_JOBS:
            MPI_Comm_split(MPI_COMM_WORLD, coll_sharp_component.node_local_rank, rank, comm);
            break;
        case COMM_TYPE_RANDOM:
            /* TODO: random comm with random size */
        default:
            *comm = MPI_COMM_WORLD;
    }
}

void mpi_free_comm(MPI_Comm* comm)
{
    if (*comm != MPI_COMM_WORLD && *comm != MPI_COMM_NULL) {
        MPI_Comm_free(comm);
    }
}

void setup_sharp_env(sharp_conf_t* sharp_conf)
{
    MPI_Comm_size(MPI_COMM_WORLD, &sharp_conf->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &sharp_conf->rank);

    sharp_env2int("ENABLE_SHARP_COLL", &enable_sharp_coll, enable_sharp_coll);
    sharp_env2int("SHARP_RUN_PPN_COMM_PARALLEL", &sharp_conf->run_ppn_comm_parallel, 0);
}

static size_t sharp_coll_test_power_floor(size_t x)
{
    size_t power = 1;
    while (x >>= 1)
        power <<= 1;
    return power;
}

static size_t sharp_coll_test_power_ceil(size_t x)
{
    if (x <= 1)
        return 1;
    size_t power = 2;
    x--;
    while (x >>= 1)
        power <<= 1;
    return power;
}

int parse_opts(int argc, char** argv, sharp_conf_t* sharp_conf)
{
    int opt, option_index;
    int res = PARSE_ARGS_OK;
    long tmp_id = 0;
    size_t val1, val2;
    char *endptr, *saveptr, *tmp_ptr = NULL;

    while (res == PARSE_ARGS_OK) {
        option_index = 0;
        endptr = "\0";
        opt = getopt_long(argc, argv, "ht:d:m:c:i:j:H:x:B:N:M:p:s:r:D:T:C", long_options, &option_index);
        if (opt == -1)   // no more args to parse
            break;
        switch (opt) {
            case 'h':
                return PARSE_ARGS_HELP;
            case 'c':
                sharp_conf->run_colls = 0;
                saveptr = optarg;
                for (endptr = strtok_r(optarg, ",", &saveptr); endptr; endptr = strtok_r(NULL, ",", &saveptr)) {
                    if (strcmp(endptr, "allreduce") == 0) {
                        sharp_conf->run_colls |= TEST_ALLREDUCE;
                    } else if (strcmp(endptr, "iallreduce") == 0) {
                        sharp_conf->run_colls |= TEST_IALLREDUCE;
                    } else if (strcmp(endptr, "reduce") == 0) {
                        sharp_conf->run_colls |= TEST_REDUCE;
                    } else if (strcmp(endptr, "ireduce") == 0) {
                        sharp_conf->run_colls |= TEST_IREDUCE;
                    } else if (strcmp(endptr, "barrier") == 0) {
                        sharp_conf->run_colls |= TEST_BARRIER;
                    } else if (strcmp(endptr, "ibarrier") == 0) {
                        sharp_conf->run_colls |= TEST_IBARRIER;
                    } else if (strcmp(endptr, "bcast") == 0) {
                        sharp_conf->run_colls |= TEST_BCAST;
                    } else if (strcmp(endptr, "reduce_scatter") == 0) {
                        sharp_conf->run_colls |= TEST_REDUCE_SCATTER;
                    } else if (strcmp(endptr, "allgather") == 0) {
                        sharp_conf->run_colls |= TEST_ALLGATHER;
                    } else if (strcmp(endptr, "none") == 0) {
                        sharp_conf->run_colls = 0;
                    } else if (strcmp(endptr, "all") == 0) {
                        sharp_conf->run_colls |= TEST_ALLREDUCE | TEST_IALLREDUCE | TEST_BARRIER;
                    } else {
                        fprintf(stderr, "Invalid --collectives option\n");
                        res = INVALID_ARG;
                    }
                }
                break;
            case 'C':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0 || tmp_id > SHARP_COLL_DATA_MAX_IOV) {
                    fprintf(stderr, "Invalid --iov_count value specified\n");
                    res = INVALID_ARG;
                } else {
                    sharp_conf->siov_count = tmp_id;
                    sharp_conf->riov_count = tmp_id;
                }
                break;
            case 'D':
                if (strcmp(optarg, "contig") == 0) {
                    sharp_conf->sdata_layout = TEST_DATA_LAYOUT_CONTIG;
                    sharp_conf->rdata_layout = TEST_DATA_LAYOUT_CONTIG;
                } else if (strcmp(optarg, "iov") == 0) {
                    sharp_conf->sdata_layout = TEST_DATA_LAYOUT_IOV;
                    sharp_conf->rdata_layout = TEST_DATA_LAYOUT_IOV;
                } else {
                    fprintf(stderr, "Invalid --data_layout option\n");
                }
                break;
            case 't':
                saveptr = optarg;
                endptr = strtok_r(optarg, ":", &saveptr);
                if (endptr) {
                    if (strcmp(endptr, "world") == 0) {
                        sharp_test_group_type = COMM_TYPE_WORLD;
                    } else if (strcmp(endptr, "world-dup") == 0) {
                        sharp_test_group_type = COMM_TYPE_WORLD_DUP;
                    } else if (strcmp(endptr, "world-rev") == 0) {
                        sharp_test_group_type = COMM_TYPE_WORLD_REVERSE;
                    } else if (strcmp(endptr, "split") == 0) {
                        endptr = strtok_r(NULL, ",", &saveptr);
                        if (endptr) {
                            if (!strcmp(endptr, "ppn-jobs")) {
                                sharp_test_group_type = COMM_TYPE_PPN_JOBS;
                                test_flat_ppn = 0;
                            } else if (!strcmp(endptr, "ppn-comm")) {
                                sharp_test_group_type = COMM_TYPE_PPN_COMM;
                                test_flat_ppn = 0;
                            } else {
                                sharp_test_group_type = COMM_TYPE_N_SPLIT;
                                sharp_test_num_splits = strtol(endptr, &tmp_ptr, 10);
                                if (*tmp_ptr != '\0') {
                                    fprintf(stderr, "Invalid --group_type option\n");
                                    res = INVALID_ARG;
                                }
                            }
                        }
                    } else if (strcmp(endptr, "split-rev") == 0) {
                        sharp_test_group_type = COMM_TYPE_N_SPLIT_REVERSE;
                        endptr = strtok_r(NULL, ",", &saveptr);
                        if (endptr) {
                            sharp_test_num_splits = strtol(endptr, &tmp_ptr, 10);
                            if (*tmp_ptr != '\0') {
                                fprintf(stderr, "Invalid --group_type option\n");
                                res = INVALID_ARG;
                            }
                        }
                    } else if (strcmp(endptr, "half") == 0) {
                        sharp_test_group_type = COMM_TYPE_HALF;
                    } else {
                        fprintf(stderr, "Invalid --group_type option\n");
                        res = INVALID_ARG;
                    }
                } else {
                    fprintf(stderr, "Invalid --group_type option\n");
                    res = INVALID_ARG;
                }
                break;
            case 'T':
                if (0 == strcmp(optarg, "half_int"))
                    sharp_conf->datatype = SHARP_DTYPE_SHORT;
                else if (0 == strcmp(optarg, "half_float"))
                    sharp_conf->datatype = SHARP_DTYPE_FLOAT_SHORT;
                else if (0 == strcmp(optarg, "int"))
                    sharp_conf->datatype = SHARP_DTYPE_INT;
                else if (0 == strcmp(optarg, "float"))
                    sharp_conf->datatype = SHARP_DTYPE_FLOAT;
                else if (0 == strcmp(optarg, "long"))
                    sharp_conf->datatype = SHARP_DTYPE_LONG;
                else if (0 == strcmp(optarg, "double"))
                    sharp_conf->datatype = SHARP_DTYPE_DOUBLE;
                else if (0 == strcmp(optarg, "byte"))
                    sharp_conf->datatype = SHARP_DTYPE_INT8;
                else {
                    fprintf(stderr, "Invalid --datatype option\n");
                    res = INVALID_ARG;
                }
                break;
            case 'd':
                saveptr = optarg;
                for (num_ib_devices = 0, endptr = strtok_r(optarg, ",", &saveptr); endptr && (num_ib_devices < MAX_IB_DEVICES);
                     endptr = strtok_r(NULL, ",", &saveptr))
                {
                    strcpy(ib_dev_list[num_ib_devices++], endptr);
                }
                break;
            case 'm':
                if (0 == strcmp(optarg, "basic"))
                    sharp_conf->test_mode = TEST_BASIC;
                else if (0 == strcmp(optarg, "complex"))
                    sharp_conf->test_mode = TEST_COMPLEX;
                else if (0 == strcmp(optarg, "perf"))
                    sharp_conf->test_mode = TEST_PERF;
                else if (0 == strcmp(optarg, "all"))
                    sharp_conf->test_mode = TEST_ALL;
                else {
                    fprintf(stderr, "Invalid --mode option\n");
                    res = INVALID_ARG;
                }
                break;
            case 'i':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0)
                    res = INVALID_ARG;
                else
                    perf_test_iterations = tmp_id;
                break;
            case 'j':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0)
                    res = INVALID_ARG;
                else
                    sharp_conf->jobid = tmp_id;
                break;
            case 'H':
                if (0 == strcmp(optarg, "malloc"))
                    sharp_conf->host_allocator_type = USE_MALLOC;
                else if (0 == strcmp(optarg, "hugetlb"))
                    sharp_conf->host_allocator_type = USE_HUGETLB;
                else if (0 == strcmp(optarg, "hugethp"))
                    sharp_conf->host_allocator_type = USE_HUGETHP;
                else {
                    fprintf(stderr, "Invalid --host_alloc_type option\n");
                    res = INVALID_ARG;
                }
                break;
            case 'r':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0)
                    res = INVALID_ARG;
                else
                    coll_root_rank = tmp_id;
                break;
            case 'x':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0)
                    res = INVALID_ARG;
                else
                    perf_test_skips = tmp_id;
                break;
            case 'N':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0 || tmp_id > 512)
                    res = INVALID_ARG;
                else
                    nbc_count = tmp_id;
                break;
            case 'p':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0)
                    res = INVALID_ARG;
                else
                    test_flat_ppn = tmp_id;
                break;
            case 'B':
                tmp_id = strtol(optarg, &endptr, 10);
                if (*endptr != '\0' || tmp_id < 0)
                    res = INVALID_ARG;
                else
                    perf_with_barrier = tmp_id;
                break;
            case 'M':
                saveptr = optarg;
                endptr = strtok_r(optarg, ":", &saveptr);
                if (endptr) {
                    if (0 == strcmp(endptr, "cuda")) {
                        sharp_conf->s_mem_type = SHARP_MEM_TYPE_CUDA;
                    } else if (0 == strcmp(endptr, "host")) {
                        sharp_conf->s_mem_type = SHARP_MEM_TYPE_HOST;
                    } else {
                        fprintf(stderr, "Invalid --mem_type option\n");
                        res = INVALID_ARG;
                        break;
                    }

                    endptr = strtok_r(NULL, ":", &saveptr);
                    if (endptr) {
                        if (0 == strcmp(endptr, "cuda")) {
                            sharp_conf->r_mem_type = SHARP_MEM_TYPE_CUDA;
                        } else if (0 == strcmp(endptr, "host")) {
                            sharp_conf->r_mem_type = SHARP_MEM_TYPE_HOST;
                        } else {
                            fprintf(stderr, "Invalid --mem_type option\n");
                            res = INVALID_ARG;
                            break;
                        }
                    } else {
                        sharp_conf->r_mem_type = sharp_conf->s_mem_type;
                    }
                }
                break;
            case 's':
                val1 = val2 = 0;
                saveptr = optarg;
                endptr = strtok_r(optarg, ":", &saveptr);
                if (endptr) {
                    val1 = atoll(endptr);
                    endptr = strtok_r(NULL, ":", &saveptr);
                    if (endptr) {
                        val2 = atoll(endptr);
                    }
                } else {
                    res = INVALID_ARG;
                }
                if (val2 == 0) {
                    assert(val1 != 0);
                    sharp_conf->max_message_size = sharp_coll_test_power_floor(val1);
                    sharp_conf->min_message_size = 4;
                } else {
                    sharp_conf->max_message_size = sharp_coll_test_power_floor(val2);
                    sharp_conf->min_message_size = sharp_coll_test_power_ceil(val1);
                }
                break;
            default:
                res = INVALID_ARG;
        }
    }

    if (opt == -1 && argv[optind] != NULL)
        res = INVALID_ARG;
    return res;
}

void print_parser_msg(int parse_res)
{
    if (parse_res == PARSE_ARGS_HELP)
        printf("%s\n", usage_string);
    else if (parse_res == INVALID_ARG)
        fprintf(stderr,
                "ERROR. Either a non-existing command was specified, "
                "or an option with a missing argument was given\n\n");
}

const char* sharp_get_host_name(void)
{
    static char hostname[256] = {0};

    if (*hostname == 0) {
        gethostname(hostname, sizeof(hostname));
        strtok(hostname, ".");
    }
    return hostname;
}

int main(int argc, char** argv)
{
    int rank, size, parse_res, mpi_rank, local_rank;
    int ret = 0;
    MPI_Comm mpi_comm;
    int dtype, tag_dtype;
    coll_sharp_module_t sharp_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sharp_world_rank = rank;
    sharp_world_size = size;

    parse_res = parse_opts(argc, argv, &sharp_conf);
    if (parse_res != PARSE_ARGS_OK) {
        if (rank == 0) {
            print_parser_msg(parse_res);
            if (parse_res != PARSE_ARGS_HELP) {
                print_parser_msg(PARSE_ARGS_HELP);
                ret = -1;
            }
        }
        goto out;
    }

    if (!num_ib_devices) {
        fprintf(stderr, "Please specify IB device with -d test option\n");
        goto out;
    }

    if (getenv("SLEEP")) {
        sleep(atoi(getenv("SLEEP")));
    }

    setup_comm_manager_rank(MPI_COMM_WORLD, &coll_sharp_component);

    if ((sharp_conf.s_mem_type == SHARP_MEM_TYPE_CUDA || sharp_conf.r_mem_type == SHARP_MEM_TYPE_CUDA)) {
#if HAVE_CUDA
        int num_gpus, gpu_index, gpu_index_stride;
        cudaError_t cerr;
        cerr = cudaGetDeviceCount(&num_gpus);
        if (cerr != cudaSuccess) {
            fprintf(stdout, "cudaGetDeviceCount failed\n");
            goto out;
        }

        gpu_index_stride = num_gpus / coll_sharp_component.ppn;
        gpu_index = (coll_sharp_component.node_local_rank * gpu_index_stride) % num_gpus;

        cerr = cudaSetDevice(gpu_index);
        if (cerr != cudaSuccess) {
            fprintf(stdout, "cudaSetDevice failed\n");
            goto out;
        }
#if DEBUG
        fprintf(stdout, "[Host:%s Rank:%d, GPU:%d] Testing with cuda buffers\n", coll_sharp_component.node_name, rank, gpu_index);
#endif
#else
        fprintf(stderr, "CUDA support is not configured\n");
        goto out;
#endif
    }

    setup_sharp_env(&sharp_conf);

    for (dtype = 0; dtype < SHARP_DTYPE_NULL; dtype++) {
        for (tag_dtype = 0; tag_dtype < SHARP_DTYPE_NULL; tag_dtype++) {
            sharp_test_mpi_min_max_datatype[dtype][tag_dtype] = MPI_DATATYPE_NULL;
        }
    }
    sharp_test_mpi_min_max_datatype[SHARP_DTYPE_FLOAT][SHARP_DTYPE_INT] = MPI_FLOAT_INT;
    sharp_test_mpi_min_max_datatype[SHARP_DTYPE_INT][SHARP_DTYPE_INT] = MPI_2INT;

    mpi_get_communicator(&mpi_comm);
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    if (0 == coll_component_open((sharp_test_group_type == COMM_TYPE_PPN_JOBS) ? mpi_comm : MPI_COMM_WORLD)) {
        if (0 == coll_module_enable(mpi_comm, &sharp_comm)) {
            if (!sharp_conf.run_ppn_comm_parallel &&
                (sharp_test_group_type == COMM_TYPE_PPN_COMM || sharp_test_group_type == COMM_TYPE_PPN_JOBS))
            {
                for (local_rank = 0; local_rank < coll_sharp_component.ppn; local_rank++) {
                    if (coll_sharp_component.node_local_rank == local_rank) {
                        if (sharp_test_group_type != COMM_TYPE_WORLD && mpi_rank == 0) {
                            fprintf(stdout, "MPI COMM Type : %s\n", comm_type_names[sharp_test_group_type]);
                        }
                        coll_module_op(&sharp_comm);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }

            } else {
                if (sharp_test_group_type != COMM_TYPE_WORLD && mpi_rank == 0) {
                    fprintf(stdout, "MPI COMM Type : %s\n", comm_type_names[sharp_test_group_type]);
                }
                coll_module_op(&sharp_comm);
            }
            coll_module_destroy(&sharp_comm);
        } else {
            fprintf(stdout, "sharp comm create failed for :  %s\n", comm_type_names[sharp_test_group_type]);
            ret = -1;
        }

    } else {
        if (mpi_rank == 0)
            fprintf(stdout, "SHArP coll failed to initialize..\n");
        ret = -1;
    }

    mpi_free_comm(&mpi_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    coll_component_close();
out:
    MPI_Finalize();
    return ret;
}
