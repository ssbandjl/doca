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

void dump_counter_names(void* context) {
    int num_counters;
    clx_api_counter_info_t* counters = clx_api_get_all_counters(context, NULL, &num_counters);
    if (!counters) {
        printf("No counters found for that schema\n.");
        return;
    }
    printf("There are %d counters\n", num_counters);
    if (num_counters != 0) {
        printf("List of all counters:\n");

        for (int i = 0; i < num_counters; i++) {
            clx_api_counter_info_t counter = counters[i];
            printf("\tcounter \"%s\"\n", counter.counter_name);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("incorrect number of arguments\n");
        printf("Usage: ./clx_counter_api_read path/to/schema_dir [path/to/file.bin]\n");
        return 0;
    }

    void* context = NULL;
    void* file    = NULL;
    void* cset    = NULL;

    char schema_dir[128] = "";
    snprintf(schema_dir, sizeof(schema_dir), "%s", argv[1]);
    printf("schema_dir = %s\n", schema_dir);

    // Create context
    context = clx_api_read_create_context(schema_dir);
    if (context == NULL) {
        printf("cannot create context\n");
        goto error;
    }

    if (argc == 2) {
        // no binary file specified, just dump contents of schema
        dump_counter_names(context);
        exit(0);
    }

    char file_name[128] = "";
    snprintf(file_name, sizeof(file_name), "%s", argv[2]);
    printf("file_name = %s\n", file_name);

    // Open file
    file = clx_api_open_counter_file(context, file_name);

    if (file == NULL) {
        printf("cannot read bin file\n");
        goto error;
    }

    cset = clx_api_create_counterset(context, file);
    if (cset == NULL) {
        goto error;
    }

    // get and dump all available counter names for current file
    int num_counters;
    clx_api_counter_info_t * all_counters = clx_api_get_all_counters(context, file, &num_counters);
    int i;
    printf("List of all counters:\n");
    for (i = 0; i < num_counters; i++) {
        clx_api_counter_info_t counter = all_counters[i];
        printf("\tcounter '%s'\n", counter.counter_name);
    }

    // option 1: use all the counters
    if (clx_api_add_all_counters(cset) < 0) {
        printf("Cannot add all counters to uninitialized counter set");
    }

    // option 2: select a hard-coded counters
    char exact_name[] = "name_11";
    if (clx_api_add_counter_exact(cset, exact_name) < 0) {
        // -1 if counter not found in schema
        printf("cannot find counter '%s' in counter set\n", exact_name);
    }

    // option 3: select a hard-coded set of counters, matching a token
    if (clx_api_add_counters_matching(cset, "0") < 0) {
        printf("cannot find match for token '0' in counter set\n");
    }

    // get all counters included to counter set
    clx_api_counter_info_t * cset_counters = clx_api_get_counters(cset);
    int num_cset_counters = clx_api_get_num_counters(cset);

    // prepare a buffer to store counters for a single timestamp
    void* data = clx_api_allocate_counters_buffer(cset);

    bool done = 0;
    while (!done) {
        uint64_t timestamp = 0;
        char source[64]="";
        // given a file and counterset, get the next set of data and the appropriate timestamp
        done = clx_api_get_next_data(file, cset, &timestamp, source, data);

        printf("timestamp = %"PRIu64"\n", timestamp);
        printf("source    = %s\n", source);

        // get counter into an appropriately typed variable, based on the 'type'
        // that is stored in the metadata that cset maintains
        size_t idx;
        for (idx = 0; idx < num_cset_counters; idx++) {
            clx_api_data_type_t type = clx_api_get_type(cset, idx);
            switch (type) {
            case CLX_API_DATA_TYPE_FP64: {
                double v_d = clx_api_get_double(cset, idx, data);
                printf("\tgot double value:   %lf of counter %s\n", v_d, cset_counters[idx].counter_name);
                (void) v_d;  // do something with data
                break;
            }
            case CLX_API_DATA_TYPE_INT64: {
                int64_t v_i = clx_api_get_int64(cset, idx, data);
                (void) v_i;  // do something with datas
                printf("\tgot int64_t value:  %"PRId64" of counter %s\n", v_i, cset_counters[idx].counter_name);

                break;
            }
            case CLX_API_DATA_TYPE_UINT64: {
                uint64_t v_u = clx_api_get_uint64(cset, idx, data);
                printf("\tgot uint64_t value: %"PRIu64" of counter %s\n", v_u, cset_counters[idx].counter_name);

                (void) v_u;  // do something with data
                break;
            }
            case CLX_API_DATA_TYPE_STRING: {
                char* v_s = clx_api_get_str(cset, idx, data);
                printf("\tgot string value: '%s' of counter %s\n", v_s, cset_counters[idx].counter_name);
                free(v_s);
                break;
            }
            case CLX_API_DATA_TYPE_BIT64:
            default:
                break;
            }
        }
    }

    // cleanup
    free(data);

    clx_api_destroy_counter_set(cset);
    clx_api_destroy_and_close_file(file);
    clx_api_read_destroy_context(context);

    return 0;
error:
    clx_api_destroy_counter_set(cset);
    clx_api_destroy_and_close_file(file);
    clx_api_read_destroy_context(context);
    return -1;
}
