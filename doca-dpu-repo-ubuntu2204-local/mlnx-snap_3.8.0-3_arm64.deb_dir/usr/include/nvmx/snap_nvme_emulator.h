/*
 * SNAP NVMe emulator controller
 */
#ifndef _SNAP_NVME_EMULATOR_H
#define _SNAP_NVME_EMULATOR_H

#include <stdint.h>
#include <linux/if_ether.h>
#include <infiniband/verbs.h>

#include "nvme_emu.h"

#if HAVE_SNAP
#include <snap.h>
#include <snap_nvme.h>
#include <snap_dma.h>
#endif

#define SNAP_EMU_MAX_BCOPY 8192

typedef struct snap_nvme_emulator snap_nvme_emulator_t;
typedef struct snap_nvme_emu_sq snap_nvme_emu_sq_t;
typedef struct snap_async_req snap_async_req_t;

#if HAVE_SNAP
struct snap_nvme_emulator {
    nvme_emulator_t              emulator;

    bool                         curr_enabled;
    bool                         prev_enabled;

    struct ibv_context           *ibctx;
    struct ibv_pd                *pd;
    struct ibv_mr                *mr;
    clock_t                      last_bar_cb;
    uint32_t                     num_queues;
    snap_nvme_emu_sq_t           **sqs;
    struct snap_nvme_sq_counters **sqcs;
    struct snap_context          *sctx;
    struct snap_device           *sdev;
    struct snap_nvme_ctrl_counters* ctrlc;
    bool                         is_started;
    bool                         flr_active;
    struct snap_device_attr      attr;
    int                          num_of_vfs;
};

struct snap_nvme_emu_sq {
    nvme_emulator_t              *emulator;
    nvme_queue_ops_t             *ops;

    uint32_t                     db_addr;
    uint32_t                     dma_rkey;
    void                         *bbuf;
    struct ibv_mr                *mr;
    struct snap_nvme_sq          *sq;
    struct snap_nvme_sq_be       *sq_be;
    struct snap_dma_q            *dma_q;

    struct snap_cross_mkey       *mkey;
    struct ibv_mr                *null_mr;
};

struct snap_async_req {
    nvme_async_req_t            base_req;
    struct snap_dma_completion  comp;
    struct ibv_mr               *mr;
};

static inline snap_nvme_emulator_t *to_snap_nvme_emulator(nvme_emulator_t *emulator)
{
    return container_of(emulator, snap_nvme_emulator_t, emulator);
}

#endif

#endif
