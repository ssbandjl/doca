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

#ifndef NVMX_SRC_SPDK_EXT_IO_EFFDMA_DOMAIN_H_
#define NVMX_SRC_SPDK_EXT_IO_EFFDMA_DOMAIN_H_

#include "nvme.h"
#include "effdma_address_space.h"

struct effdma_domain;

void effdma_domain_clear();
struct effdma_domain *effdma_domain_find(int pf_id, int vf_id);
struct effdma_domain *effdma_domain_find_or_create(int pf_id, int vf_id);

int effdma_domain_translate(struct effdma_domain *domain, uint64_t siova, uint64_t *tiova);
address_space_t *effdma_domain_transaction_begin(struct effdma_domain *domain);
void effdma_domain_transaction_commit(struct effdma_domain *domain);
void effdma_domain_transaction_abort(struct effdma_domain *domain);
uint64_t effdma_normalize_size(uint64_t size, enum nvme_cmd_vs_iova_mgmt_szu szu);

#endif /* NVMX_SRC_SPDK_EXT_IO_EFFDMA_DOMAIN_H_ */
