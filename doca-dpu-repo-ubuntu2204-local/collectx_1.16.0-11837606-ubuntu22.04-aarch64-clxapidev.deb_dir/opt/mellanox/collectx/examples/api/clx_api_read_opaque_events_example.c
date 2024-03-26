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

#define CLX_GUID_FMT "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x"

#define CLX_GUID_ARGS(guid)                                                                                                 \
    guid[0], guid[1], guid[2], guid[3], guid[4], guid[5], guid[6], guid[7], guid[8], guid[9], guid[10], guid[11], guid[12], \
        guid[13], guid[14], guid[15]


#define CLX_GUID_SCANF_FMT \
    "%2" SCNx8 "%2" SCNx8 "%2" SCNx8 "%2" SCNx8 "-%2" SCNx8 "%2" SCNx8 "-%2" SCNx8 "%2" SCNx8 \
    "-%2" SCNx8 "%2" SCNx8 "-%2" SCNx8 "%2" SCNx8 "%2" SCNx8 "%2" SCNx8 "%2" SCNx8 "%2" SCNx8

#define CLX_GUID_SCANF_ARGS(guid)                                                                                             \
    &guid[0], &guid[1], &guid[2], &guid[3], &guid[4], &guid[5], &guid[6], &guid[7], &guid[8], &guid[9], &guid[10], &guid[11], \
        &guid[12], &guid[13], &guid[14], &guid[15]

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        printf("incorrect number of arguments\n");
        printf("Usage: ./clx_counter_api_read path/to/file.bin path/to/schema_dir [app_id_guid]\n");
        return 0;
    }

    bool any_app_id = (argc == 3);

    printf("Running with\n");
    printf("\tfile_name  = %s\n", argv[1]);
    printf("\tschema_dir = %s\n", argv[2]);
    printf("\tapp_id     = %s\n", any_app_id ? argv[3] : "any");

    clx_guid_t app_id;
    if (!any_app_id) {
        if (sscanf(argv[3], CLX_GUID_SCANF_FMT, CLX_GUID_SCANF_ARGS(app_id)) != CLX_GUID_SIZE) {
            printf(
                "ERROR: bad app_id GUID %s. Please provide a GUID in the canonical form (XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX)\n",
                argv[3]);
            return 1;
        }
    }

    // Create context
    void *context =
        clx_api_read_opaque_events_create_context(argv[1], argv[2], any_app_id ? CLX_API_READ_OPAQUE_EVENT_APP_ID_ANY : app_id);
    if (context == NULL) {
        printf("ERROR: cannot create context\n");
        return 1;
    }

    printf("#app_id,user_defined1,user_defined2,data_size,data[0]\n");
    while (true) {
        clx_api_read_opaque_event_info_t info;
        int                              res = clx_api_read_opaque_events_get_next(context, &info);
        if (res == -1) {
            printf("ERROR: cannot get next event\n");
            break;
        }
        if (res == 0) {
            printf("No more events\n");
            break;
        }

        const uint8_t *data = (const uint8_t *)info.data;
        printf(CLX_GUID_FMT ",%" PRIu64 ",%" PRIu64 ",%" PRIu32 ",%" PRIu8 "\n", CLX_GUID_ARGS(info.app_id), info.user_defined1,
               info.user_defined2, info.data_size, data[0]);
    }

    clx_api_read_opaque_events_destroy_context(context);
    return 0;
}
