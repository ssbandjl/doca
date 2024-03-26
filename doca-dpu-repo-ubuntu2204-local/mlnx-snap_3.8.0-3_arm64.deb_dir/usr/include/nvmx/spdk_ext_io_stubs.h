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

#ifndef NVMX_SRC_SPDK_EXT_IO_SPDK_EXT_IO_STUBS_H_
#define NVMX_SRC_SPDK_EXT_IO_SPDK_EXT_IO_STUBS_H_

typedef struct spdk_ext_io_ctx
{
} spdk_ext_io_ctx_t;

static inline int spdk_ext_io_init() { return 0; }
static inline void spdk_ext_io_clear() {}

static inline void spdk_ext_io_context_init(spdk_ext_io_ctx_t *ext_io_ctx, void *sdev,
        void *key_table, struct effdma_domain *domain, memzero_ops_t *memzero_ops) {}
static inline void spdk_ext_io_context_clear(spdk_ext_io_ctx_t *ext_io_ctx) {}

static inline int spdk_ext_io_zcopy_status(struct spdk_bdev *bdev) { return -ENOTSUP; }
static inline int spdk_ext_io_effdma_status(struct spdk_bdev *bdev) { return -ENOTSUP; }

#endif /* NVMX_SRC_SPDK_EXT_IO_SPDK_EXT_IO_STUBS_H_ */
