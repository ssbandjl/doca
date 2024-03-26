#ifndef _VIRTIO_FS_CTRL_H
#define _VIRTIO_FS_CTRL_H

#include <stdio.h>
#include <stdbool.h>

typedef void (*vf_change_cb_t)(void *ctx, int pf_id, int num_vfs);

enum virtio_fs_ctrl_dev_type {
    VIRTIO_FS_CTRL_FS_DEV_NONE,
};

struct virtio_fs_ctrl {
    size_t nthreads;
    enum virtio_fs_ctrl_dev_type dev_type;
    pthread_mutex_t dev_lock;
    void (*fsdev_close_cb)(struct virtio_fs_ctrl *ctrl);
    int pf_id;
    struct snap_context *sctx;
    struct ibv_pd *pd;
    struct snap_virtio_fs_ctrl *sctrl;
    void *vf_change_cb_arg;
    vf_change_cb_t vf_change_cb;
};

struct virtio_fs_ctrl_init_attr {
    const char *emu_manager_name;
    int pf_id;
    int vf_id;
    const char *dev_type;
    uint32_t nthreads;
    uint32_t num_queues;
    uint32_t queue_depth;
    const char *tag;
    void *vf_change_cb_arg;
    vf_change_cb_t vf_change_cb;
    bool force_in_order;
    bool suspended;
    bool recover;
};

void virtio_fs_ctrl_progress(void *ctrl);
int virtio_fs_ctrl_progress_all_io(void *ctrl);
int virtio_fs_ctrl_progress_io(void *arg, int thread_id);
void virtio_fs_ctrl_suspend(void *ctrl);
bool virtio_fs_ctrl_is_suspended(void *ctrl);
int virtio_fs_ctrl_resume(void *arg);
int virtio_fs_ctrl_state_save(void *ctrl, char *file);
int virtio_fs_ctrl_state_restore(void *ctrl, char *file);
int virtio_fs_ctrl_lm_enable(void *ctrl, char *lm_channel_name);
void virtio_fs_ctrl_lm_disable(void *ctrl);

struct virtio_fs_ctrl *
virtio_fs_ctrl_init(const struct virtio_fs_ctrl_init_attr *attr);
void virtio_fs_ctrl_destroy(void *ctrl);
#endif
