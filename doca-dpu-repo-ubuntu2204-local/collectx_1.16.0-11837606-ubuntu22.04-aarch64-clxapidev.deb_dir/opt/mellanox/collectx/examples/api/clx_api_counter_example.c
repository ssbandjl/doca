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


#define _GNU_SOURCE         /* See feature_test_macros(7) */

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "api/clx_api.h"

#define MAX(A, B) (((A) > (B)) ? (A) : (B))

typedef struct test_counter_provider_info_t {
    uint32_t v1;
    // add implementation-specific fields here
} test_counter_provider_info_t;

uint64_t round_up(uint64_t number, uint16_t multiple) {
    if (multiple == 0) {
        return number;
    }
    return ((number + multiple - 1) / multiple) * multiple;
}


bool test_counter_provider_initialize(clx_api_context_t* ctx, clx_api_provider_t* provider) {
    test_counter_provider_info_t* info = (test_counter_provider_info_t*) provider->data;
    char* group_name = "my_counters";

#define NUM_COUNTERS 40
    // first half of counters are CLX_API_DATA_TYPE_UINT64
    clx_api_counter_info_t counter = { "cntr_name", "value", "", CLX_API_DATA_TYPE_UINT64, CLX_FIELD_VALUE_ABSOLUTE, 0, 8};
    uint32_t counter_num;
    bool ok;
    for (int i = 0; i < NUM_COUNTERS / 2; i++) {
        // note: asprintf allocates memory that should be freed later
        if (asprintf(&counter.counter_name, "name_%d", i) < 0) {
            printf("cannot print counter name 'name_%d'", i);
        }
        ok = clx_api_add_counter(ctx, &counter, group_name, &counter_num);
        // counter_num * 8 is the offset in buffer at which this counter will be stored.
        free(counter.counter_name);
        if (!ok) {
            printf("[error] Aborting %s", __FUNCTION__);
            return false;
        }
    }

    // second half of counters are CLX_API_DATA_TYPE_STRING
    for (int i = 0; i < NUM_COUNTERS - NUM_COUNTERS / 2; i++)  {
        int counter_len = round_up(MAX(64 / (i / 4 + 1), 2), 2);
        clx_api_counter_info_t str_counter = { "test_str", "str_val", "",
                                               CLX_API_DATA_TYPE_STRING, CLX_FIELD_VALUE_ABSOLUTE,
                                               0, counter_len};
        // note: asprintf allocates memory that should be freed later
        if (asprintf(&str_counter.counter_name, "str_name_%d", i) < 0) {
            printf("cannot print counter name 'str_name_%d'", i);
        }
        ok = clx_api_add_counter(ctx, &str_counter, group_name, &counter_num);
        free(str_counter.counter_name);
        if (!ok) {
            printf("[error] Aborting %s", __FUNCTION__);
            return false;
        }
    }

    info->v1 = 123;
    return true;
}


clx_api_provider_t* get_provider(void) {
    clx_api_provider_t* provider = calloc(1, sizeof(clx_api_provider_t));
    clx_api_version_t ver = { { 1, 0, 0 } };  // compiler bug.
    provider->name = strdup("My_counter_provider");
    provider->version = ver;
    provider->initialize = &test_counter_provider_initialize;

    //  for storing the event type idx returned for each event.
    //  can be used for additional provider-specific info
    provider->data = calloc(1, sizeof(test_counter_provider_info_t));
    return provider;
}

void destroy_provider(clx_api_provider_t* provider) {
    free(provider->name);
    free(provider->data);
    free(provider);
}

#define MULTISOURCE
// enable MULTISOURCE to clone api context to use it for the second source

// #define WITH_IPC
// enable ipc transport
// IPC transport sents the data to collector 'clx' process from this app

int main(void) {
    //  Define params
    clx_api_params_t params = {0};
    // params.max_file_size = 1 * 512 * 1024;  // 512 KB
    params.max_file_size = 1 * 1024 * 1024;    // 1 MB
    params.max_file_age = 60 * 60 * 1000000L;  // 1 hour

    params.buffer_size = 64 * 1024;  // for counters set buffer size to zero to send data immediately
                                           // or make it large (e.g 60000) to reduce num of transactions

    params.schema_path = "tmp/example/schema";
    params.data_root = "tmp/example";


    params.source_id = "A1";  // part of the path convention: data_dir/year/date/hash/source_id/timestamp.bin
    params.source_tag = "";   // for counters, need source_tag = ""
    params.file_write_enabled = true;
    params.ipc_enabled = 0;

#ifdef WITH_IPC
    params.ipc_enabled                = 1;
    params.ipc_sockets_dir            = strdup("/tmp/ipc_sockets");  // should be the same ipc_sockets_dir from clx_config.ini

    params.ipc_max_reattach_time_msec = 5000;  // 5 seconds for overal reattach procedure
    params.ipc_max_reattach_tries     = 10;    // 10 tries during reattach procedure

    params.ipc_socket_timeout_msec    = 3000;  // timeout for UD socket
#endif

    params.data_path_template = CLX_API_DATA_PATH_TEMPLATE;

    // params.fb_enable = true;

    clx_api_provider_t* provider = get_provider();
    if (provider == NULL) {
        printf("Failed to initialize provider\n");
        return -1;
    }

    clx_api_context_t* ctx = clx_api_create_context(&params, provider);

    uint64_t t = 0;
#ifdef MULTISOURCE
    // For testing the 'clx_api_clone_context()' function
    t = clx_api_get_timestamp();
    clx_api_params_t params_2 = params;  // copy element by element
    params_2.source_id = "A2";
    params_2.source_tag = "";
    params_2.schema_path = "tmp/example/schema";
    params_2.data_root = "tmp/example";
    params_2.data_path_template = CLX_API_DATA_PATH_TEMPLATE;
    clx_api_context_t* ctx2 = clx_api_clone_context(ctx, &params_2);
#endif

    for (int i = 1; i < 1000; i++) {
        uint32_t data_size = 0;
        t = clx_api_get_timestamp();

        // get the data buffer to fill. If there is no space, page will be written/exported via IPC,fluent-bit or prometheus
        void* data = clx_api_get_counters_buffer(ctx, t, &data_size);
        if (data) {
            //  fill half of the buffer with numeric counters
            memset(data, 0, data_size);
            uint64_t *p = data;
            *p = getpid(); p+=1;
            *p = t;        p+=1;
            *p = i;        p+=8;
            *p = 10;       p+=10;

            // copy parts of a_string to the CLX_API_DATA_TYPE_STRING counters
            char a_str[] = "0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopq";
            size_t prev_offset = (NUM_COUNTERS / 2) * 8;
            char* p2 = (char*)data + prev_offset;
            for (int j = 0; j < NUM_COUNTERS - NUM_COUNTERS / 2; j++) {
                int counter_len = round_up(MAX(64 / (j / 4 + 1), 2), 2);
                strncpy(p2, a_str, counter_len);
                p2 += counter_len;
            }
        } else {
            printf("Error: failed to get more data: %d\n", i);
        }

#ifdef MULTISOURCE
        data_size = 0;
        data = clx_api_get_counters_buffer(ctx2, t, &data_size);
        if (data) {
            uint64_t *p = data;

            *p = getpid();         p+=1;
            *p = t;                p+=1;
            for (int j = 2; j < NUM_COUNTERS; j++) {
                *p++ = j;
            }

            char b_str[] = "0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopq";
            size_t prev_offset = (NUM_COUNTERS / 2) * 8;
            char* p2 = (char*)data + prev_offset;
            for (int j = 0; j < NUM_COUNTERS - NUM_COUNTERS / 2; j++) {
                int counter_len = round_up(MAX(64 / (j / 4 + 1), 2), 2);
                strncpy(p2, b_str, counter_len);
                p2 += counter_len;
            }
        }
#endif
    }

    destroy_provider(provider);
#ifdef MULTISOURCE
    clx_api_destroy_context(ctx2);
#endif
    clx_api_destroy_context(ctx);
    return 0;
}

