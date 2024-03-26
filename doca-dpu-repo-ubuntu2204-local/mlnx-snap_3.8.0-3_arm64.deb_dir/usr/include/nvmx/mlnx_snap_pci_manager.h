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

#ifndef _MLNX_SNAP_PCI_MANAGER_H
#define _MLNX_SNAP_PCI_MANAGER_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <infiniband/verbs.h>
#include <snap.h>
#include <snap_nvme.h>
#include <snap_virtio_blk.h>

struct snap_pci **mlnx_snap_get_snap_pci_list(enum snap_emulation_type type,
                                              const char *rdma_dev,
                                              int *spci_list_sz);
struct snap_context *mlnx_snap_get_snap_context(const char *rdma_dev);
struct ibv_context *mlnx_snap_get_ibv_context(const char *rdma_dev);
struct snap_pci *mlnx_snap_get_snap_pci(enum snap_emulation_type type,
                                        const char *rdma_dev,
                                        const char *pci_bdf,
                                        int pf_index, int vf_index);

uint32_t mlnx_snap_acquire_namespace_id(const char *rdma_dev);
void mlnx_snap_release_namespace_id(const char *rdma_dev, uint32_t nsid);

int mlnx_snap_pci_manager_init();
void mlnx_snap_pci_manager_clear();

struct snap_pci *mlnx_snap_get_pf_snap_pci_by_vuid(enum snap_emulation_type type,
                                        const char *rdma_dev, const char *vuid);
#endif
