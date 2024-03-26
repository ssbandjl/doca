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

#ifndef NVMX_SRC_SPDK_DOMAIN_H_
#define NVMX_SRC_SPDK_DOMAIN_H_

#include <spdk/bdev.h>
#include "memzero_core.h"

#include "effdma/effdma_domain.h"
#ifndef HAVE_SPDK_EXT_IO
#include "spdk_ext_io_stubs.h"
#else
#include <spdk/dma.h>
#include "compiler.h"
#include "nvme_emu_log.h"
#include "spdk_ext_io_zcopy.h"
#include "spdk_ext_io_effdma.h"
#include "spdk_ext_io_memzero.h"

typedef struct spdk_ext_io_ctx
{
    spdk_ext_io_zcopy_ctx_t zcopy_ctx;
    spdk_ext_io_effdma_ctx_t effdma_ctx;
    spdk_ext_io_memzero_ctx_t memzero_ctx;
    struct spdk_bdev_ext_io_opts ext_io_opts;
} spdk_ext_io_ctx_t;

int spdk_ext_io_init();
void spdk_ext_io_clear();

void spdk_ext_io_context_init(spdk_ext_io_ctx_t *ext_io_ctx, void *sdev,
                              void *key_table,
                              struct effdma_domain *domain,
                              memzero_ops_t *memzero_ops);
void spdk_ext_io_context_clear(spdk_ext_io_ctx_t *ext_io_ctx);
int spdk_ext_io_get_bdev_status(struct spdk_bdev *bdev, int domain_type);

static inline bool spdk_ext_io_validate_iov(struct spdk_bdev *bdev,
                struct iovec *iov, size_t len, size_t alignment)
{
    size_t i;
    const size_t addr_mask = spdk_bdev_get_buf_align(bdev) - 1;

    alignment--;

    for (i = 0; i < len; i++) {
        const size_t addr = (size_t)iov[i].iov_base;

        if (unlikely(addr & addr_mask)) {
            nvmx_debug("ZCOPY_FAIL: iova 0x%lx is not bdev aligned to %zu\n",
                       addr, addr_mask + 1);
            return false;
        }

        if (unlikely(addr & alignment)) {
            nvmx_debug("ZCOPY_FAIL: iova 0x%lx is not aligned to %zu\n",
                       addr, alignment + 1);
            return false;
        }
    }

    return true;
}

#endif

#endif /* NVMX_SRC_SPDK_DOMAIN_H_ */
