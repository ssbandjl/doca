/*
 * NVMe emulation memory disk ctrl
 */
#ifndef _NVME_MEM_CTRL_H
#define _NVME_MEM_CTRL_H

#include "nvme_emu.h"
#include "nvme_emu_io_driver.h"

typedef struct nvme_mem_ctrl nvme_mem_ctrl_t;

struct nvme_mem_ctrl {
    char *memdisk_base;
    nvme_emu_ep_t *ep;
    uint64_t disk_size_b;
    uint8_t disk_block_order;
};

int nvme_emu_create_mem_ctrl(nvme_emu_ep_t *ep);
void nvme_emu_delete_mem_ctrl(nvme_mem_ctrl_t *ctrl);
int nvme_start_mem_ctrl(nvme_emu_ep_t *ep);
void nvme_stop_mem_ctrl(nvme_emu_ep_t *ep);


#endif
