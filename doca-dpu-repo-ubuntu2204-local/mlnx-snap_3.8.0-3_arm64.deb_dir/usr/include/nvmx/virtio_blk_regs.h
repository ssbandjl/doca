#ifndef _VIRTIO_BLK_REGS_H
#define _VIRTIO_BLK_REGS_H
#include <stdint.h>
#include <snap.h>

#define VIRTIO_BLK_REGS_DEF_SIZE_MAX 4096
#define VIRTIO_BLK_REGS_DEF_SEG_MAX 1
#define VIRTIO_BLK_REGS_DEF_QUEUE_SIZE 64
#define VIRTIO_BLK_REGS_MAX_QUEUE_SIZE 256


struct virtio_blk_regs_set_attr {
    uint64_t num_blocks;
    uint32_t blk_size;
    uint32_t size_max;
    uint32_t seg_max;
    uint32_t max_queues;
    uint32_t queue_depth;
    bool legacy_mode;
    bool admin_queue;
};

void virtio_blk_regs_set(const struct virtio_blk_regs_set_attr *attr,
                          struct snap_virtio_blk_registers *regs);
#endif
