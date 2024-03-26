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

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        printf("incorrect number of arguments\n");
        printf("Usage: ./clx_api_read_events_example path/to/file.bin path/to/schema_dir [path/to/fset_file]\n");
        return 0;
    }

    const char *file_name  = argv[1];
    const char *schema_dir = argv[2];
    const char *fset_file  = (argc == 4) ? argv[3] : NULL;

    printf("Running with\n");
    printf("\tfile_name  = %s\n", file_name);
    printf("\tschema_dir = %s\n", schema_dir);
    printf("\tfset_file  = %s\n", fset_file);

    int res = 1;
    clx_api_field_set_enum_t *e;
    clx_api_field_info_t finfo;

    clx_api_file_t *file = clx_api_file_open(file_name, schema_dir);
    if (file == NULL) {
        printf("ERROR: cannot create file with (%s, %s)\n", file_name, schema_dir);
        goto file_open_failed;
    }

    clx_api_field_set_t *fset = clx_api_field_set_create(schema_dir, fset_file);
    if (fset == NULL) {
        printf("ERROR: cannot create field set with (%s, %s)\n", schema_dir, fset_file);
        goto fset_create_failed;
    }

    e = clx_api_field_set_enum_begin(fset, "switch_event", true);
    if (!e) {
        printf("ERROR: cannot begin enumeration of switch_event fields\n");
        goto fset_create_failed;
    }

    printf("switch_event fields:\n");
    while (clx_api_field_set_enum_next(e, &finfo)) {
        printf("\tvariant#%zu: field: %s %d\n", finfo.type_idx, finfo.name, finfo.type);
    }
    clx_api_field_set_enum_end(e);

    if (!clx_api_field_set_add_token(fset, "switch_event", "^pxdd")) {
        printf("ERROR: cannot dynamically add token \n");
        goto add_token_failed;
    }

    clx_api_event_t *evt = NULL;
    while ((evt = clx_api_file_get_next_event(file)) != NULL) {
        const char *evt_name = clx_api_event_get_name(evt);

        printf("Event: %s\n", evt_name);

        size_t num_fields;
        if (!clx_api_field_set_read(fset, evt, &num_fields)) {
            printf("WARNING: cannot read event %s\n", evt_name);
            continue;
        }

        size_t idx;
        for (idx = 0; idx < num_fields; idx++) {
            const char *        field_name = clx_api_field_set_get_name(fset, idx);
            clx_api_data_type_t field_type = clx_api_field_set_get_type(fset, idx);

            switch (field_type) {
            case CLX_API_DATA_TYPE_INT64: {
                int64_t v = clx_api_field_set_get_int64(fset, idx);
                printf("\tgot int64_t value:  %" PRId64 " of field %s\n", v, field_name);
                break;
            }
            case CLX_API_DATA_TYPE_UINT64: {
                uint64_t v = clx_api_field_set_get_uint64(fset, idx);
                printf("\tgot uint64_t value:  %" PRIu64 " of field %s\n", v, field_name);
                break;
            }
            case CLX_API_DATA_TYPE_FP64: {
                double v = clx_api_field_set_get_double(fset, idx);
                printf("\tgot double value:  %lf of field %s\n", v, field_name);
                break;
            }
            case CLX_API_DATA_TYPE_STRING: {
                char *v = clx_api_field_set_get_string(fset, idx);
                printf("\tgot string value:  %s of field %s\n", v, field_name);
                free(v);
                break;
            }
            default:
                printf("WARNING: unknown type of field %s (%d)\n", field_name, field_type);
                break;
            }
        }
    }

    res = 0;

add_token_failed:
    clx_api_field_set_destroy(fset);
fset_create_failed:
    clx_api_file_close(file);
file_open_failed:
    return res;
}
