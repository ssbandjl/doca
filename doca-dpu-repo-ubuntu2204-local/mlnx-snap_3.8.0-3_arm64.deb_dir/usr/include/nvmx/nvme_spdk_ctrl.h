#ifndef _NVME_SPDK_CTRL_H
#define _NVME_SPDK_CTRL_H
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include "nvme_emu.h"
#include "nvme_emu_ep.h"
#include "queue.h"


struct nvme_spdk_namespace;
#define NVME_SPDK_NAMESPACE_NAME_MAXLEN 16

/* TODO: query mpsmax from nvme_reg_cap_mpsmax */
#define NVME_CTRL_PAGE_SHIFT  16
#define NVME_CTRL_PAGE_SIZE   (1 << NVME_CTRL_PAGE_SHIFT)

typedef struct nvme_spdk_ctrl {
    char name[NVME_EMU_NAME_MAXLEN];
    nvme_emu_ep_t *ep;
    int cntlid;

    /* list of private namespaces */
    TAILQ_HEAD(, nvme_spdk_namespace) namespaces;
    pthread_spinlock_t namespaces_lock;
    bool expose_namespaces;

    TAILQ_ENTRY(nvme_spdk_ctrl) entry;

    bool zcopy;
    bool tcp_rx_zcopy;
    struct ibv_pd *pd;  
    
    struct spdk_mem_map *mem_map;

    int pf_id;
    int vf_id;
    struct effdma_domain *effdma_domain;
} nvme_spdk_ctrl_t;

int nvme_spdk_create_ctrl(nvme_emu_ep_t *ep);
void nvme_spdk_delete_ctrl(nvme_spdk_ctrl_t *ctrl);
void nvme_spdk_stop_ctrl(nvme_emu_ep_t *ep);
int nvme_spdk_start_ctrl(nvme_emu_ep_t *ep);
void nvme_spdk_zcopy_dma_done(void *user, int status);

#endif
