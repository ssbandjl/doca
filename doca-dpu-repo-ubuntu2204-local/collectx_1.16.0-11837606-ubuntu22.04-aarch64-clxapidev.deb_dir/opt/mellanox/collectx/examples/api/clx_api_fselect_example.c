/*
* Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) 2015-2016 Mellanox Technologies Ltd. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:

* 1. Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived from
* this software without specific prior written permission.

* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
* HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "api/clx_api.h"

#define USEC_IN_SEC 1000000UL
#define SOURCES_START_IDX 5

void _print_src_list(const char **srcs) {
    const char **srcs_ptr = srcs;
    while ((srcs) && (*srcs_ptr)) {
        if (srcs != srcs_ptr) {
            printf(", ");
        }
        printf("%s", *srcs_ptr);
        srcs_ptr++;
    }

    if (!srcs) {
        printf("NULL");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("incorrect number of arguments\n");
        printf("Usage: ./clx_api_fselect_example path/to/data_root start_epoch_us end_epoch_us fname_template [src1 src2...]\n");
        return 1;
    }

    const char *data_root       = argv[1];
    const char *start_epoch_str = argv[2];
    const char *end_epoch_str   = argv[3];
    const char *fname_template  = argv[4];
    const char **sources        = NULL;
    char        *endptr;

    /* Allocate enough space for char* of source names + null */
    if (argc > SOURCES_START_IDX) {
        sources = malloc((argc - SOURCES_START_IDX + 1) * sizeof(char*));
        if (!sources) {
            printf("Couldn't malloc memory for sources list\n");
            return 1;
        }
        const char **src_ptr = sources;

        for (int i=5; i < argc; i++) {
            *src_ptr = argv[i];
            src_ptr++;
        }
        *src_ptr = NULL;
    }

    unsigned long long ts_start = strtoull(start_epoch_str, &endptr, 0);
    if (*endptr != 0) {
        printf("Bad start_epoch %s\n", start_epoch_str);
        return 1;
    }

    unsigned long long ts_end = strtoull(end_epoch_str, &endptr, 0);
    if (*endptr != 0) {
        printf("Bad end_epoch %s\n", end_epoch_str);
        return 1;
    }

    printf("Running with\n");
    printf("\tdata_path      = %s\n", data_root);
    printf("\tstart_epoch    = %llu\n", ts_start);
    printf("\tend_epoch      = %llu\n", ts_end);
    printf("\tfname_template = %s\n", fname_template);
    printf("\tsources        = ");
    _print_src_list(sources);

    int res = 1;
    clx_api_fselect_ctx_t *ctx =
        clx_api_fselect_begin_ex(data_root, ts_start, ts_end, fname_template, sources);
    if (!ctx) {
        printf("Cannot begin enumeration\n");
        goto begin_failed;
    }

    const char *f;
    while ((f = clx_api_fselect_next(ctx)) != 0) {
        printf("File found: %s\n", f);
    }

    clx_api_fselect_end(ctx);
    free(sources);
    res = 0;

begin_failed:
    return res;
}
