/*
 * NVMe emulation posix ctrl
 */
#ifndef _NVME_POSIX_CTRL_H
#define _NVME_POSIX_CTRL_H

#include "nvme_emu.h"

typedef struct nvme_posix_ctrl nvme_posix_ctrl_t;

struct nvme_posix_ctrl {
    int io_fd;
    char *io_bbuf;
    nvme_emu_ep_t *ep;
    uint64_t disk_size_b;
    uint8_t disk_block_order;
};

int nvme_emu_create_posix_ctrl(nvme_emu_ep_t *ep);
void nvme_emu_delete_posix_ctrl(nvme_posix_ctrl_t *ctrl);
int nvme_start_posix_ctrl(nvme_emu_ep_t *ep);
void nvme_stop_posix_ctrl(nvme_emu_ep_t *ep);


#endif
