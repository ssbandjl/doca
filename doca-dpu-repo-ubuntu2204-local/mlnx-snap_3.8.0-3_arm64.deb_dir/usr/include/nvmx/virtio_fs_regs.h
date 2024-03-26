#ifndef _VIRTIO_FS_REGS_H
#define _VIRTIO_FS_REGS_H
#include <stdint.h>
#include <snap.h>

#define VIRTIO_FS_REGS_DEF_QUEUE_SIZE 64
#define VIRTIO_FS_REGS_MAX_QUEUE_SIZE 128

struct virtio_fs_regs_set_attr {
    uint32_t max_queues;
    uint32_t queue_depth;
    const char *tag;
    bool legacy_mode;
};

void virtio_fs_regs_set(const struct virtio_fs_regs_set_attr *attr,
                        struct snap_virtio_fs_registers *regs);
#endif
