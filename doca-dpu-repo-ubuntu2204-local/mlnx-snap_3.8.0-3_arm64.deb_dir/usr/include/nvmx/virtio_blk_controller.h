#ifndef _VIRTIO_BLK_CTRL_H
#define _VIRTIO_BLK_CTRL_H
#include <stdio.h>
#include <stdbool.h>
#include "memp.h"
#include "spdk_ext_io/spdk_ext_io.h"

#define VBLK_EMU_NAME_PREFIX "VblkEmu"
#define VBLK_EMU_NAME_MAXLEN 32

typedef void (*vf_change_cb_t)(void *ctx, int pf_id, int num_vfs);
typedef void (*hotunplug_device_cb_t)(void *ctx);
typedef void (*hotunplug_timeout_cb_t)(void *ctx);
typedef void (*bdev_detach_cb_t)(void *arg, bool success);

enum virtio_blk_ctrl_bdev_type {
    VIRTIO_BLK_CTRL_BDEV_NONE,
    VIRTIO_BLK_CTRL_BDEV_SPDK,
};

struct virtio_spdk_bdev_io_ctx {
    struct spdk_io_channel *spdk_channel;
    struct spdk_thread *spdk_thread;
    struct spdk_thread *caller_thread;
    spdk_msg_fn caller_cb_fn;
    struct virtio_blk_ctrl *ctrl;
    spdk_ext_io_ctx_t ext_io_ctx;
};

struct virtio_spdk_bdev_ctx {
    struct spdk_bdev *bdev;
    struct spdk_bdev_desc *desc;
    struct virtio_spdk_bdev_io_ctx *io_channels;
    size_t num_io_channels;
    bool zcopy;
};

struct virtio_blk_ctrl {
    char name[VBLK_EMU_NAME_MAXLEN];
    size_t nthreads;
    enum virtio_blk_ctrl_bdev_type bdev_type;
    pthread_mutex_t bdev_lock;
    void (*bdev_close_cb)(struct virtio_blk_ctrl *ctrl);
    int pf_id;
    int vf_id;
    struct snap_context *sctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct snap_virtio_blk_ctrl *sctrl;
    struct virtio_spdk_bdev_ctx bctx;
    char serial[VIRTIO_BLK_ID_BYTES + 1];
    void *vf_change_cb_arg;
    vf_change_cb_t vf_change_cb;
    void (*destroy_done_cb)(void *arg);
    void *destroy_done_cb_arg;
    bool zcopy;
    memp_t *memp;
    void *hotunplug_device_ctx;
    hotunplug_device_cb_t hotunplug_device_cb;
    hotunplug_timeout_cb_t hotunplug_timeout_cb;
    time_t hotunplug_timer_expire;
    bool force_hotunplug_device;
    bool pending_bdev_detach;
    void (*bdev_detach_done)(void *arg, bool success);
    void *bdev_detach_done_arg;
};

struct virtio_blk_ctrl_init_attr {
    const char *emu_manager_name;
    int pf_id;
    int vf_id;
    const char *bdev_type;
    const char *bdev_name;
    uint32_t nthreads;
    uint32_t num_queues;
    uint32_t queue_depth;
    uint32_t size_max;
    uint32_t seg_max;
    const char *serial;
    void *vf_change_cb_arg;
    vf_change_cb_t vf_change_cb;
    bool force_in_order;
    bool suspended;
    bool recover;
    bool use_mem_pool;
    memp_t *memp;
    void *hotunplug_device_ctx;
    hotunplug_device_cb_t hotunplug_device_cb;
    hotunplug_timeout_cb_t hotunplug_timeout_cb;
    bool force_recover;
    bool admin_q;
};

struct virtio_blk_ctrl_io_stat
{
    uint64_t read_ios;
    uint64_t completed_read_ios;
    uint64_t completed_unord_r_ios;
    uint64_t write_ios;
    uint64_t completed_write_ios;
    uint64_t completed_unord_w_ios;
    uint64_t flush_ios;
    uint64_t completed_flush_ios;
    uint64_t completed_unord_f_ios;
    uint64_t err_read_os;
    uint64_t err_write_os;
    uint64_t err_flush_os;
    uint64_t outstand_in_ios;
    uint64_t outstand_in_bdev_ios;
    uint64_t outstand_to_host_ios;
    uint64_t fatal_ios;
};

void virtio_blk_ctrl_progress(void *ctrl);
int virtio_blk_ctrl_progress_all_io(void *ctrl);
int virtio_blk_ctrl_progress_io(void *arg, int thread_id);
void virtio_blk_ctrl_suspend(void *ctrl);
bool virtio_blk_ctrl_is_suspended(void *ctrl);
int virtio_blk_ctrl_resume(void *arg);
int virtio_blk_ctrl_state_save(void *ctrl, char *file);
int virtio_blk_ctrl_state_restore(void *ctrl, char *file);
int virtio_blk_ctrl_lm_enable(void *ctrl, char *lm_channel_name);
void virtio_blk_ctrl_lm_disable(void *ctrl);

void virtio_blk_ctrl_bdev_detach(void *ctrl_arg, void *done_cb_arg,
                                 bdev_detach_cb_t done_cb);
int virtio_blk_ctrl_bdev_attach(void *ctrl_arg, const char *bdev_type,
                                const char *bdev_name, uint32_t size_max,
                                uint32_t seg_max, const char *serial);
int virtio_blk_ctrl_get_debugstat(void *ctrl_arg,
                        struct snap_virtio_ctrl_debugstat *ctrl_debugstat);
struct virtio_blk_ctrl *
virtio_blk_ctrl_init(const struct virtio_blk_ctrl_init_attr *attr);
void virtio_blk_ctrl_destroy(void *arg, void (*done_cb)(void *arg),
                             void *done_cb_arg);

void virtio_blk_ctrl_get_stat(struct virtio_blk_ctrl *ctrl, struct virtio_blk_ctrl_io_stat *stat);
bool virtio_blk_ctrl_is_zcopy(struct virtio_blk_ctrl *ctrl);
bool virtio_blk_ctrl_dma_pool_enabled(struct virtio_blk_ctrl *ctrl);

#endif
