#ifndef _SPDK_EMU_MGR_H
#define _SPDK_EMU_MGR_H
#include <stdio.h>
#include <stdint.h>
#include "queue.h"

#define SPDK_EMU_NAME_MAXLEN 32
#define SPDK_EMU_MANAGER_NAME_MAXLEN 16


struct snap_conf;
struct spdk_emu_device_hotunplug_ctx;

enum spdk_emu_protocol {
    SPDK_EMU_PROTOCOL_NONE,
    SPDK_EMU_PROTOCOL_NVME,
    SPDK_EMU_PROTOCOL_VIRTIO_BLK,
    SPDK_EMU_PROTOCOL_VIRTIO_FS,
};

static inline const char *
spdk_emu_protocol_to_string(enum spdk_emu_protocol protocol)
{
    switch (protocol) {
    case SPDK_EMU_PROTOCOL_NVME:
        return "nvme";
    case SPDK_EMU_PROTOCOL_VIRTIO_BLK:
        return "virtio_blk";
    case SPDK_EMU_PROTOCOL_VIRTIO_FS:
        return "virtio_fs";
    default:
        return "none";
    }
}

typedef void (*spdk_emu_mgr_fini_cb_t)(void *arg);

struct spdk_emu_io_thread {
    int id;
    struct spdk_emu_ctx *ctrl_ctx;
    struct spdk_thread *spdk_thread;
    struct spdk_thread *spdk_thread_creator;
    struct spdk_poller *spdk_poller;
};

struct spdk_emu_ctx {
    void *ctrl;
    const struct spdk_emu_ctx_ctrl_ops *ctrl_ops;
    enum spdk_emu_protocol protocol;
    char emu_manager[SPDK_EMU_MANAGER_NAME_MAXLEN];
    struct snap_pci *spci;
    char emu_name[SPDK_EMU_NAME_MAXLEN];
    struct spdk_poller *admin_poller;
    struct spdk_poller *admin_intensive_poller;
    struct spdk_poller *io_poller;
    size_t num_io_threads;
    struct spdk_emu_io_thread *io_threads;
    LIST_ENTRY(spdk_emu_ctx) entry;
    LIST_ENTRY(spdk_emu_ctx) nvme_subsys_entry;
    struct nvme_subsystem *nvme_subsys;
    struct snap_conf *sconfig;
    uint16_t num_vfs;
    uint16_t num_vfs_next;
    bool vfs_deletion_in_progress;
    bool intensive_polling;
    uint64_t intensive_polling_end;

    /* Callback to be called after ctx is destroyed */
    void (*fini_cb)(void *arg);
    void *fini_cb_arg;

    bool should_stop;
    struct spdk_emu_device_hotunplug_ctx *hotunplug_ctx;
    time_t hotunplug_timer_expire;
};

struct spdk_emu_ctx_create_attr {
    void *priv;
    enum spdk_emu_protocol protocol;
    const char *emu_manager;
    struct snap_pci *spci;
};

LIST_HEAD(spdk_emu_list_head, spdk_emu_ctx);

extern struct spdk_emu_list_head spdk_emu_list;
extern pthread_mutex_t spdk_emu_list_lock;
extern bool spdk_emu_rpc_finished;

struct spdk_emu_ctx *
spdk_emu_ctx_find_by_pci_id(const char *emu_manager,
                            int pf_id, int vf_id,
                            enum spdk_emu_protocol protocol);

struct spdk_emu_ctx *
spdk_emu_ctx_find_by_pci_bdf(const char *emu_manager, const char *pci_bdf);

struct spdk_emu_ctx *spdk_emu_ctx_find_by_emu_name(const char *emu_name);
struct spdk_emu_ctx *
spdk_emu_ctx_find_by_nqn_cntlid(const char *subnqn, int cntlid);
struct spdk_emu_ctx *
spdk_emu_ctx_find_by_vuid(const char *vuid);
struct spdk_emu_ctx *
spdk_emu_ctx_find(const char *subnqn, int cntlid, const char *name);

struct spdk_emu_ctx *
spdk_emu_ctx_create(const struct spdk_emu_ctx_create_attr *attr);

void spdk_emu_ctx_destroy(struct spdk_emu_ctx *ctx, void (*fini_cb)(void *arg),
                          void *fini_cb_arg);

void spdk_emu_ctx_fini_cb_rpc(void *arg);

int spdk_emu_init();
void spdk_emu_clear();

struct spdk_emu_device_hotunplug_ctx {
    struct snap_pci *spci;
    struct spdk_jsonrpc_request *request;
    bool timeout_once;
};

struct spdk_emu_ctx *
spdk_emu_hotunplug_create_controller(void  *arg, struct ibv_device *ibv_dev);
#endif
