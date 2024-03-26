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

#define OPAQUE_DATA_SIZE 256

// 0816ddf2-e801-444e-a2be-7bdbac46f704
static clx_guid_t test_app_id = {0x08, 0x16, 0xdd, 0xf2, 0xe8, 0x01, 0x44, 0x4e, 0xa2, 0xbe, 0x7b, 0xdb, 0xac, 0x46, 0xf7, 0x04};

bool test_opaque_event_provider_initialize(clx_api_context_t* ctx, clx_api_provider_t* provider) {
    return true;
}

int main(void) {
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

    params.enable_opaque_events = true;

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

    clx_api_provider_t provider = {
        .version = {{1, 0, 0}},
        .name = "My_provider",
        .initialize = test_opaque_event_provider_initialize,
    };
    clx_api_context_t* ctx = clx_api_create_context(&params, &provider);

    char data[OPAQUE_DATA_SIZE];

    int k;
    for (k = 0; k < 1000; k++) {
        memset(data, (char)k, sizeof(data));

        bool ok = clx_api_opaque_event_write(ctx, test_app_id, k, 0, data, sizeof(data));
        if (!ok) {
            printf("Failed to write opaque event#%d of %zu bytes\n", k, sizeof(data));
            break;
        }
    }

    clx_api_destroy_context(ctx);
}

