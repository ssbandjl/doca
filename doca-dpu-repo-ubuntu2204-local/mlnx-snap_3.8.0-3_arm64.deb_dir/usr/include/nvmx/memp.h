/*
 *   Copyright © 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVMX_SRC_MEMP_H_
#define NVMX_SRC_MEMP_H_

typedef void (*memp_callback_t)(void *data, void *user);

typedef void(*memp_poll_t)(void *ctx);

typedef struct memp memp_t;
typedef struct nvme_io_driver_req nvme_io_driver_req_t;
typedef struct nvme_async_req nvme_async_req_t;

enum MEMP_COUNTERS
{
    MEMP_COUNTER_CHUNKS,
    MEMP_COUNTER_CHUNKS_ALLOC,
    MEMP_COUNTER_CHUNKS_FREE,
    MEMP_COUNTER_ALLOC,
    MEMP_COUNTER_ASYNC_COMPLETE,
    MEMP_COUNTER_IMMEDIATE_COMPLETE,
    MEMP_COUNTER_CANCEL,
    MEMP_COUNTER_FREE,
    MEMP_COUNTER_ERROR,
    MEMP_COUNTER_PENDING_NOMEM,
    MEMP_COUNTER_PENDING_QUOTA,
    MEMP_COUNTERS_COUNT
};

static inline const char *memp_counter_name(enum MEMP_COUNTERS counter)
{
    static const char *counter_names[MEMP_COUNTERS_COUNT] = {
        "chunks", "chunks_alloc", "chunks_free", "alloc", "async",
        "immediate", "cancel", "free", "error", "nomem", "quota"
    };

    return counter_names[counter];
};

// Main application thread API

memp_t *memp_init(void *buf, size_t buf_size, size_t thread_count);
void memp_destroy(memp_t *memp_t);
void *memp_get_buffer(memp_t *memp);
size_t memp_get_size(memp_t *memp);
size_t *memp_get_counters(memp_t *memp, int tid);

// Worker thread thread API

void *memp_alloc(memp_t *memp, size_t size, void *user, memp_callback_t cb, void **tag, int tid);
void *memp_alloc_ex(nvme_io_driver_req_t *req, memp_callback_t cb);
void *memp_alloc_req(nvme_async_req_t *req, memp_callback_t cb);
void memp_free(memp_t *memp, void *p);
int memp_cancel(memp_t *memp, void *tag, int thread_id);
void memp_thread_poll(memp_t *memp, int tid);

#endif /* NVMX_SRC_MEMP_H_ */
