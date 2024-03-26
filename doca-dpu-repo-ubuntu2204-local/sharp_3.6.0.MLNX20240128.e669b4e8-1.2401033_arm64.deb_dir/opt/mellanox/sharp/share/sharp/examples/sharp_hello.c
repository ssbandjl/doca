/**
 * Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <api/sharp.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "common/sharp_common.h"

static char* ib_dev_list;
static int verbose_level = 3;

enum
{
    PARSE_ARGS_OK = 0,
    PARSE_ARGS_HELP = 1,
    PARSE_ARGS_VERSION = 2,
    PARSE_ARGS_INVALID = 3,
};

static struct option long_options[] = {{"help", no_argument, NULL, 'h'},
                                       {"ib-dev", required_argument, NULL, 'd'},
                                       {"verbose", required_argument, NULL, 'v'},
                                       {"version", no_argument, NULL, 'V'},
                                       {0, 0, 0, 0}};
static const char usage_string[] = "usage:  sharp_hello [< -d | --ib_dev> <device>] [OPTIONS]\n"
                                   "OPTIONS:\n"
                                   "\t[-d | --ib_dev]      - HCA to use\n"
                                   "\t[-v | --verbose]     - libsharp coll verbosity level(default :2)\n"
                                   "\t\t\t\t  Levels: (0-fatal 1-err 2-warn 3-info 4-debug 5-trace)\n"
                                   "\t[-V | --version]     - print program version\n"
                                   "\t[-h | --help]        - show this usage\n\n";

static int oob_bcast(void* comm_context, void* buf, int size, int root)
{
    return 0;
}

static int oob_barrier(void* comm_context)
{
    return 0;
}

static int oob_gather(void* comm_context, int root, void* sbuf, void* rbuf, int len)
{
    memcpy(rbuf, sbuf, len);
    return 0;
}

int parse_opts(int argc, char** argv)
{
    int opt, option_index;
    int res = PARSE_ARGS_OK;
    char *endptr, *env_val;

    while (res == PARSE_ARGS_OK) {
        option_index = 0;
        endptr = "\0";
        opt = getopt_long(argc, argv, "Vhd:v:", long_options, &option_index);
        if (opt == -1)
            break;
        switch (opt) {
            case 'h':
                return PARSE_ARGS_HELP;
            case 'd':
                free(ib_dev_list);
                ib_dev_list = strdup(optarg);
                break;
            case 'v':
                verbose_level = strtol(optarg, &endptr, 10);
                break;
            case 'V':
                sharp_print_version(stdout, "sharp_hello");
                return PARSE_ARGS_VERSION;
            default:
                res = PARSE_ARGS_INVALID;
        }
    }

    if ((opt == -1 && argv[optind] != NULL))
        return PARSE_ARGS_INVALID;

    if (verbose_level >= 0 && verbose_level <= 5) {
        if (asprintf(&env_val, "%d", verbose_level) < 0) {
            return PARSE_ARGS_INVALID;
        }
        setenv("SHARP_COLL_LOG_LEVEL", env_val, 1);
        free(env_val);
    } else {
        fprintf(stderr, "invalid sharp coll verbose level :%d\n", verbose_level);
        return PARSE_ARGS_HELP;
    }

    return res;
}

int main(int argc, char** argv)
{
    int ret = 0, rc;
    struct sharp_coll_init_spec init_spec = {0};
    struct sharp_coll_context* sharp_coll_context;
    struct sharp_coll_comm* sharp_coll_comm;
    struct sharp_coll_comm_init_spec comm_spec;
    struct timeval tval;

    rc = parse_opts(argc, argv);
    if (rc != PARSE_ARGS_OK) {
        if (rc == PARSE_ARGS_HELP) {
            fprintf(stdout, "%s", usage_string);
            goto exit;
        }
        if (rc == PARSE_ARGS_VERSION) {
            goto exit;
        }
        ret = -rc;
        goto exit;
    }

    gettimeofday(&tval, NULL);
    srand((int)tval.tv_usec);

    init_spec.progress_func = NULL;
    /* coverity[dont_call] */
    init_spec.job_id = (gethostid() << 32) | rand();
    init_spec.world_rank = 0;
    init_spec.world_size = 1;
    init_spec.world_local_rank = 0;
    init_spec.enable_thread_support = 0;
    init_spec.oob_colls.barrier = oob_barrier;
    init_spec.oob_colls.bcast = oob_bcast;
    init_spec.oob_colls.gather = oob_gather;
    init_spec.oob_ctx = NULL;
    init_spec.config = sharp_coll_default_config;
    init_spec.config.ib_dev_list = ib_dev_list;

    /* initialize sharp coll */
    ret = sharp_coll_init(&init_spec, &sharp_coll_context);
    if (ret < 0) {
        fprintf(stderr, "sharp_coll_init failed: %s\n", sharp_coll_strerror(ret));
        goto exit;
    }

    /* create sharp group */
    comm_spec.rank = 0;
    comm_spec.size = 1;
    comm_spec.oob_ctx = NULL;
    comm_spec.group_world_ranks = NULL;
    ret = sharp_coll_comm_init(sharp_coll_context, &comm_spec, &sharp_coll_comm);
    if (ret < 0) {
        fprintf(stderr, "sharp communicator creation failed: %s\n", sharp_coll_strerror(ret));
        goto coll_finalize;
    }

    /* run Barrier */
    ret = sharp_coll_do_barrier(sharp_coll_comm);
    if (ret == SHARP_COLL_SUCCESS) {
        fprintf(stdout, "Test Passed.\n");
    } else {
        fprintf(stderr, "Test Failed: %s\n", sharp_coll_strerror(ret));
    }

    /* destroy group */
    sharp_coll_comm_destroy(sharp_coll_comm);

coll_finalize:
    /* finalize sharp coll */
    sharp_coll_finalize(sharp_coll_context);
exit:
    free(ib_dev_list);
    return ret;
}
