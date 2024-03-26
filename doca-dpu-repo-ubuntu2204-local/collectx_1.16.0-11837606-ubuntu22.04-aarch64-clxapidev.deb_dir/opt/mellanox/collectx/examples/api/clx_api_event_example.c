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


#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "api/clx_api.h"

typedef struct test_event_provider_info_t {
    uint8_t ev1_type_index;
    uint8_t ev2_type_index;
    uint8_t ev3_type_index;
    // add implementation-specific fields here
} test_event_provider_info_t;

// Actual types from which we will serialize

typedef struct __attribute__((packed)) test_event_t {
    clx_api_timestamp_t timestamp;
    uint16_t            source_index;
    uint64_t            value;
    char                name[16];
} test_event_t;

typedef struct __attribute__((packed)) another_test_event_t {
    clx_api_timestamp_t timestamp;
    uint16_t            source_index;
    uint64_t            value;
    uint64_t            another_value;
    char                name[16];
} another_test_event_t;

typedef struct __attribute__((packed)) test_event_3_batch_t {
    clx_api_timestamp_t timestamp;
    uint16_t            source_index;
    char                name[16];
    uint64_t            batch_id;
    uint64_t            batch_record_id;
} test_event_3_batch_t;


bool test_event_provider_initialize(clx_api_context_t* ctx, clx_api_provider_t* provider) {
    test_event_provider_info_t* info = (test_event_provider_info_t*) provider->data;
    //
    // The info below describes data event entries,
    // and must exactly describe the structs defined above.
    //
    clx_api_event_field_info_t fields_ev1[] = {
        { "timestamp",    "Event timestamp", "timestamp", CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "source_index", "Source index",    "uint16_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "value",        "Some value",      "uint64_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "name",         "Event name",      "char",      CLX_FIELD_VALUE_ABSOLUTE, 16 }
    };
    if (CLX_API_OK != clx_api_add_event_type(ctx, "test_event1", fields_ev1, NUM_OF_FIELDS(fields_ev1), &info->ev1_type_index)) {
        return false;
    }

    clx_api_event_field_info_t fields_ev2[] = {
        { "timestamp",     "Event timestamp", "timestamp", CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "source_index",  "Source index",    "uint16_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "value",         "Some value",      "uint64_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "another_value", "Another value",   "uint64_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "name",          "Event name",      "char",      CLX_FIELD_VALUE_ABSOLUTE, 16 }
    };
    if (CLX_API_OK != clx_api_add_event_type(ctx, "test_event2", fields_ev2, NUM_OF_FIELDS(fields_ev2), &info->ev2_type_index)) {
        return false;
    }

    clx_api_event_field_info_t fields_ev3[] = {
        { "timestamp",       "Event timestamp",    "timestamp", CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "source_index",    "Source index",       "uint16_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "name",            "Event name",         "char",      CLX_FIELD_VALUE_ABSOLUTE, 16 },
        { "batch_id",        "Number of batch",    "uint64_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
        { "batch_record_id", "Record ID in batch", "uint64_t",  CLX_FIELD_VALUE_ABSOLUTE, 1 },
    };

    if (CLX_API_OK != clx_api_add_event_type(ctx, "test_event_3_batch", fields_ev3,
                                             NUM_OF_FIELDS(fields_ev3), &info->ev3_type_index)) {
        return false;
    }
    return true;
}

clx_api_provider_t* get_provider(void) {
    clx_api_provider_t* provider = calloc(1, sizeof(clx_api_provider_t));
    clx_api_version_t ver = { { 1, 0, 0 } };  // compiler bug.
    provider->name = strdup("My_provider");
    provider->version = ver;
    provider->initialize = &test_event_provider_initialize;

    // for storing the event type idx returned for each event.
    // can be used for additional provider-specific info
    provider->data = calloc(1, sizeof(test_event_provider_info_t));
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
    int i, j, k;
    int result = 0;

    clx_api_params_t params = {0};
    params.max_file_size = 1 * 1024 * 1024;  // 1 MB
    params.max_file_age = 60 * 60 * 1000000L;  // 1 hour
    params.buffer_size = 0;  // use default

    params.schema_path = "tmp/example/schema";
    params.data_root = "tmp/example";

    params.source_id = "S1";
    params.source_tag = "example_event";
    params.file_write_enabled = true;

    params.ipc_enabled = 0;

#ifdef WITH_IPC
    params.ipc_enabled     = 1;
    params.ipc_sockets_dir = strdup("/tmp/ipc_sockets");  // should be the same ipc_sockets_dir from clx_config.ini

    params.ipc_max_reattach_time_msec = 5000;  // 5 seconds for overal reattach procedure
    params.ipc_max_reattach_tries     = 10;    // 10 tries during reattach procedure

    params.ipc_socket_timeout_msec    = 3000;  // timeout for UD socket
#endif


    // from common/defs.h:
    // #define CLX_DATA_PATH_TEMPLATE          "{{year}}/{{month}}{{day}}/{{source}}/{{tag}}{{id}}.bin"
    // params.data_path_template = strdup(CLX_DATA_PATH_TEMPLATE); // was in ipc branch
    params.data_path_template = CLX_API_DATA_PATH_TEMPLATE;

    // params.fb_enable = true;

    clx_api_provider_t* provider = get_provider();
    if (provider == NULL) {
        printf("Failed to initialize provider\n");
        return -1;
    }

    clx_api_context_t* ctx = clx_api_create_context(&params, provider);


#ifdef MULTISOURCE
    // For testing the 'clx_api_clone_context()' function
    clx_api_params_t params_2 = params;  // copy element by element
    params_2.source_id = "S2";
    params_2.source_tag = "example_event";
    params_2.schema_path = "tmp/example/schema";
    params_2.data_root = "tmp/example";
    params_2.data_path_template = CLX_API_DATA_PATH_TEMPLATE;
    // params_2.fb_enable = true;

    clx_api_context_t* ctx2 = clx_api_clone_context(ctx, &params_2);
#endif


    // generate some example events
    test_event_t e1 = { 11111, 1, 1, "1111" };
    test_event_t e2 = { 22222, 2, 2, "2222" };
    test_event_t e3 = { 33333, 3, 3, "3333" };
    test_event_t* ev[] = { &e1, &e2, &e3 };

    another_test_event_t a1 = { 11111, 1, 1, 11, "4444" };
    another_test_event_t a2 = { 22222, 2, 2, 22, "5555" };
    another_test_event_t a3 = { 33333, 3, 3, 33, "6666" };
    another_test_event_t* aev[] = { &a1, &a2, &a3 };

#ifdef MULTISOURCE
    test_event_t e1_2 = { 77777, 1, 1, "777777" };
    test_event_t e2_2 = { 88888, 2, 2, "888888" };
    test_event_t e3_2 = { 99999, 3, 3, "999999" };
    test_event_t* ev_2[] = { &e1_2, &e2_2, &e3_2 };

    another_test_event_t a1_2 = { 10101, 5, 6, 611, "4444" };
    another_test_event_t a2_2 = { 20222, 5, 6, 622, "5555" };
    another_test_event_t a3_2 = { 33333, 5, 7, 633, "6666" };
    another_test_event_t* aev_2[] = { &a1_2, &a2_2, &a3_2 };
#endif

    test_event_provider_info_t* info = (test_event_provider_info_t*) provider->data;


    for (k = 0; k < 1000; k++) {
        clx_api_timestamp_t ts = clx_api_get_timestamp();

        // some action in the event data
        e1.timestamp = ts;
        e1.value = k;

        e2.timestamp = ts;
        e2.value = k;

        e3.timestamp = ts;
        e3.value = k;

        a1.timestamp = ts;
        a1.value = 2 * k;

        a2.timestamp = ts;
        a2.value = 3 * k;

        a3.timestamp = ts;
        a3.value = 4 * k;

        void* ev1_data = (void*) ev[k % 3];
        bool ok = clx_api_write_event(ctx, ev1_data, info->ev1_type_index, 1);
        if (!ok) {
            printf("Failed to write type-1 event %d\n", k);
            result = 1;
            break;
        }

        (void) aev;
        void* ev2_data = (void*) aev[k % 3];

        ok = clx_api_write_event(ctx, ev2_data, info->ev2_type_index, 1);
        if (!ok) {
            printf("Failed to write type-2 event %d\n", k);
            result = 1;
            break;
        }


        // some action in the event data
#ifdef MULTISOURCE
        e1_2.timestamp = ts;
        e1_2.value = k + 1;

        e2_2.timestamp = ts;
        e2_2.value = k + 1;

        e3_2.timestamp = ts;
        e3_2.value = k + 1;

        a1_2.timestamp = ts;
        a1_2.value = 2 * k + 1;

        a2_2.timestamp = ts;
        a2_2.value = 3 * k + 1;

        a3_2.timestamp = ts;
        a3_2.value = 4 * k + 1;

        ev1_data = (void*) ev_2[k % 3];
        ok = clx_api_write_event(ctx2, ev1_data, info->ev1_type_index, 1);
        if (!ok) {
            printf("Failed to write type-1 event %d\n", k);
            result = 1;
            break;
        }

        ev2_data = (void*) aev_2[k % 3];
        ok = clx_api_write_event(ctx2, ev2_data, info->ev2_type_index, 1);
        if (!ok) {
            printf("Failed to write type-2 event %d\n", k);
            result = 1;
            break;
        }
#endif
    }

    // TEST multiple events write
    clx_api_params_t params_3 = params;  // copy element by element
    params_3.source_id        = "BATCH";

    clx_api_context_t* ctx3 = clx_api_clone_context(ctx, &params_3);

    int event_array_size = 200;
    char* tmp = getenv("BATCH_SIZE");
    if (tmp != NULL) {
        event_array_size = atoi(tmp);
    }
    printf("Max BATCH_SIZE is '%d'\n", event_array_size);

    test_event_3_batch_t* event_batch = (test_event_3_batch_t*) calloc(event_array_size, sizeof(test_event_3_batch_t));
    if (!event_batch) {
        printf("Failed to allocate event_batch");
        return 1;
    }

    for (i = 0; i < event_array_size; i++) {
        // printf("Batch %d\n", i);

        clx_api_timestamp_t ts = clx_api_get_timestamp();
        snprintf(event_batch[i].name, sizeof(event_batch[i].name), "batch_event");
        for (j = 0; j <= i; j++) {
            event_batch[j].timestamp       = ts;
            event_batch[j].source_index    = 0;
            event_batch[j].batch_id        = i;
            event_batch[j].batch_record_id = j;
        }

        // printf("Writing batch %d of %d events\n", i, i + 1);
        bool ok = clx_api_write_event(ctx3, event_batch, info->ev3_type_index, i + 1);
        if (!ok) {
            printf("Failed to write batch %d of %d events\n", i, i + 1);
            result = 1;
            break;
        }
    }
    free(event_batch);

    destroy_provider(provider);
#ifdef MULTISOURCE
    clx_api_destroy_context(ctx2);
#endif
    clx_api_destroy_context(ctx3);
    clx_api_destroy_context(ctx);
    return result;
}

