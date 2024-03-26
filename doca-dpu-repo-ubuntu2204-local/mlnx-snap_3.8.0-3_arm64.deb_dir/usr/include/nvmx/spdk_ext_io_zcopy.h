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

#ifndef NVMX_SRC_SPDK_ZCOPY_H_
#define NVMX_SRC_SPDK_ZCOPY_H_

#include <infiniband/verbs.h>

typedef struct spdk_ext_io_zcopy_ctx spdk_ext_io_zcopy_ctx_t;

struct spdk_ext_io_zcopy_ctx {
    bool enabled;
    struct snap_device *sdev;
    struct snap_cross_mkey * (*find_key)(spdk_ext_io_zcopy_ctx_t *zcopy_ctx,
                struct ibv_pd *pd);
    void *key_table;

    /* cache a pd-mkey pair */
    struct ibv_pd *pd;
    struct snap_cross_mkey *mkey;
};

void spdk_ext_io_zcopy_init(spdk_ext_io_zcopy_ctx_t *zcopy_ctx,
                            void *key_table,
                            struct snap_device *sdev);
void spdk_ext_io_zcopy_clear(spdk_ext_io_zcopy_ctx_t *zcopy_ctx);
int spdk_ext_io_zcopy_status(struct spdk_bdev *bdev);
int spdk_ext_io_zcopy_memory_domain_translate_memory_cb(
        struct spdk_memory_domain *src_domain,
        spdk_ext_io_zcopy_ctx_t *zcopy, struct spdk_memory_domain *dst_domain,
        struct spdk_memory_domain_translation_ctx *dst_domain_ctx,
        void *addr, size_t len,
        struct spdk_memory_domain_translation_result *result);

#endif /* NVMX_SRC_SPDK_ZCOPY_H_ */
