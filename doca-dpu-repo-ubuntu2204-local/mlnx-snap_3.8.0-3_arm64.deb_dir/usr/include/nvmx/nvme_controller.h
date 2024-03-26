#ifndef _NVME_CTRL_H
#define _NVME_CTRL_H

#if HAVE_SNAP
#include <snap_json_rpc_client.h>
#endif

#include "nvme.h"
#include "nvme_emu_io_driver.h"
#include "snap_conf.h"
#include "queue.h"

#define NVME_SUPPORTED_MANDATORY_AENS (NVME_AVAILABLE_SPARE_ERR | \
                                       NVME_TEMPERATURE_ERR | \
                                       NVME_DEV_RELIABILITY_ERR | \
                                       NVME_READ_ONLY_ERR | \
                                       NVME_VOLATILE_MEM_BACKUP_ERR)

#define NVME_SUPPORTED_OPTIONAL_AENS  (NVME_NAMESPACE_ATTR_NOTICE)

#define NVME_SUPPORTED_ASYNC_EVENTS (NVME_SUPPORTED_MANDATORY_AENS | \
                                     NVME_SUPPORTED_OPTIONAL_AENS)

#define NVME_DEFAULT_SUBNQN_PREFIX "nqn.2014.08.org.nvmexpress.snap:"

#define NVME_CTRL_MAX_NAME_LEN 256

#define NVME_FW_SLOTS_COUNT 4

typedef struct nvme_ctrl nvme_ctrl_t;
typedef struct memp memp_t;

typedef void (*vf_change_cb_t)(void *ctx, int pf_id, int num_vfs);
typedef void* (*rpc_method_is_supported_cb_t)(nvme_ctrl_t *ctrl);
typedef int (*rpc_method_handle_cb_t)(void *ctx, nvme_ctrl_t *ctrl);
typedef void (*hotunplug_device_cb_t)(void *ctx);
typedef void (*hotunplug_timeout_cb_t)(void *arg);

typedef struct nvme_ctrl_init_param {
    uint32_t        nthreads;
    void           *vf_change_ctx;
    vf_change_cb_t  vf_change_cb;
    rpc_method_is_supported_cb_t rpc_method_is_supported_cb;
    rpc_method_handle_cb_t rpc_method_handle_cb;
    snap_conf_t    *sconfig;
    memp_t         *memp;
    void            *hotunplug_device_ctx;
    hotunplug_device_cb_t hotunplug_device_cb;
    hotunplug_timeout_cb_t hotunplug_timeout_cb;
} nvme_ctrl_init_param_t;

typedef struct nvme_op_stat {
    uint64_t total;
    uint64_t completed;
    uint64_t err;
    uint64_t completed_bytes_count;
    uint64_t completed_units_count;
    uint64_t err_bytes_count;
} nvme_op_stat_t;

typedef struct nvme_stat {
    nvme_op_stat_t read;
    nvme_op_stat_t write;
    nvme_op_stat_t flush;
    bool valid;
} nvme_stat_t;

typedef struct nvme_namespace {
    nvme_id_ns_t id_ns;
    /*
     * From the spec 5.15.5 - A controller shall not return multiple
     * descriptors with the same Namespace Identifier Type (NIDT) -
     * so NVME_NS_ID_TYPES_NUM descriptors is enough.
     */
    nvme_id_ns_descriptor_t id_descs[NVME_NS_ID_TYPES_NUM];
    uint32_t id;
    uint32_t remote_id;
    uint64_t size_b;
    uint8_t  block_order;
    uint16_t backend_metadata;
    void    *backend_ns;
    nvme_emu_dev_ns_t *nvme_emu_ns;
    TAILQ_ENTRY(nvme_namespace) entry;
    nvme_stat_t *sq_stat;
} nvme_namespace_t;

typedef struct nvme_squeue nvme_squeue_t;
typedef struct nvme_cqueue nvme_cqueue_t;

typedef struct nvme_request {
    nvme_squeue_t          *sq;
    uint16_t                status;
    uint8_t                 has_sg;
    nvme_cqe_t              cqe;
    TAILQ_ENTRY(nvme_request) entry;
    TAILQ_ENTRY(nvme_request) rpc_entry;
    bool                    is_async_event;
    nvme_op_stat_t          *ns_stat;
    nvme_op_stat_t          *ctrl_stat;
    void                    *rpc_buf;
    size_t                  fw_size;
    size_t                  fw_offset;
    void                    *fw_buf;
    nvme_io_driver_req_t    io_req;
} nvme_request_t;

struct nvme_ctrl_pg;

/**
 * enum nvme_squeue_state - state of nvme_squeue
 * @NVME_SQ_RUNNING:    Queue receives and operates commands
 * @NVME_SQ_FLUSHING:   Queue stops receiving new commands and operates
 *                      commands already received
 * @NVME_SQ_SUSPENDED:  Queue doesn't receive new commands and has no
 *                      commands to operate
 */
enum nvme_squeue_state {
    NVME_SQ_RUNNING,
    NVME_SQ_FLUSHING,
    NVME_SQ_SUSPENDED,
};

struct nvme_squeue {
    uint16_t    sqid;
    uint16_t    cqid;
    uint32_t    head;
    uint32_t    tail;
    uint32_t    size;
    uint64_t    dma_addr;

    nvme_ctrl_t	     *ctrl;
    nvme_queue_ops_t  ops;

    TAILQ_HEAD(sq_req_list, nvme_request) req_list;
    TAILQ_HEAD(out_req_list, nvme_request) out_req_list;
    TAILQ_ENTRY(nvme_squeue) entry;
    TAILQ_ENTRY(nvme_squeue) pg_entry;
    struct nvme_ctrl_pg *pg;

    void   *reqs_base;
    enum nvme_squeue_state state;
    nvme_request_t *delete_req;

    int thread_id;

    void *io_buf;
    struct ibv_mr *mr;
};

struct nvme_cqueue {
    uint8_t     phase;
    uint16_t    cqid;
    uint32_t    head;
    uint32_t    tail;
    uint32_t    vector;
    uint32_t    size;
    uint64_t    dma_addr;

    nvme_ctrl_t	     *ctrl;
    nvme_queue_ops_t  ops;
    TAILQ_HEAD(sq_list, nvme_squeue) sq_list;
    TAILQ_HEAD(cq_req_list, nvme_request) req_list;
    struct nvme_ctrl_pg *pg;
};

/**
 * enum nvme_ctrl_state - Nvme controller internal state
 *
 *  @STOPPED: All on-demand resources (squeues + cqueues) are cleaned.
 *
 *  @STARTED: Controller is live and ready to handle IO requests. All
 *    requested on-demand resources (squeues + cqueues) are created
 *    successfully.
 *
 *  @SUSPENDED: All squeues + cqueues are flushed and suspended. Internal
 *    controller state will stay constant. DMA access to the host memory is
 *    stopped. The state is equivalent of doing quiesce+freeze in live migration
 *    terms. In order to do a safe shutdown, application should put controller
 *    in the suspended state before controller is stopped.
 *    NOTE: going to the suspened state is an async operation.
 *
 *  @SUSPENDING: indicates that suspend operation is in progress
 */
enum nvme_ctrl_state {
    NVME_CTRL_STOPPED,
    NVME_CTRL_STARTED,
    NVME_CTRL_SUSPENDED,
    NVME_CTRL_SUSPENDING
};

struct nvme_ctrl {
    uint32_t    page_size;
    uint32_t    page_size_min;
    uint16_t    page_bits;
    uint16_t    max_prp_ents;
    uint16_t    cqe_size;
    uint16_t    sqe_size;
    uint32_t    num_namespaces;
    uint32_t    max_namespaces;
    uint32_t    num_queues;
    uint32_t    max_q_ents;
    uint32_t    cmb_size_mb;
    uint32_t    cmbsz;
    uint32_t    cmbloc;
    uint8_t     *cmbuf;
    uint64_t    irq_status;

    /* NVME mandatory features */
    uint32_t    arbitration;
    uint32_t    power_management;
    uint16_t    over_temperature_threshold;
    uint16_t    under_temperature_threshold;
    uint16_t    temperature;

    /* Aggregation settings for IO completion queues */
    int         cq_period;    /* 1 usec resolution */
    int         cq_max_count;

    uint32_t    async_event_config;
    uint32_t    async_event_mask;
    snap_conf_t *sconfig;
    void          *vf_change_ctx;
    vf_change_cb_t vf_change_cb;

    TAILQ_HEAD(, nvme_namespace) namespaces;
    uint32_t                     changed_nsids[1024];
    uint16_t                     num_changed_nsids;
    pthread_spinlock_t           namespaces_lock;

    TAILQ_HEAD(, nvme_request)  async_event_reqs;
    pthread_mutex_t             async_event_lock;
    uint16_t                    pending_aens;
    uint8_t                     masked_logpage_ids;
    size_t                      num_aers;

#if HAVE_SNAP
    const char                           *rpc_server;
    struct snap_json_rpc_client          *rpc_client;
    struct snap_json_rpc_client_response *rpc_rsp;
    pthread_t       rpc_thread;
    sem_t           rpc_thread_kick;
    sem_t           rpc_thread_idle;
    nvme_request_t *rpc_thread_req;
    bool            rpc_busy;
    TAILQ_HEAD(, nvme_request) rpc_pending_list;
#endif
    nvme_emulator_t    *dev;
    nvme_emu_driver_t  *driver;
    nvme_squeue_t      **sq;
    nvme_cqueue_t      **cq;
    nvme_squeue_t      admin_sq;
    nvme_cqueue_t      admin_cq;
    bool               should_stop;
    bool               allow_set_num_qs;
    enum nvme_ctrl_state state;

    nvme_stat_t        *sq_stat;

    nvme_ctrl_id_t     id_ctrl;
    /* Some nvme drivers are not completely compliant with the NVME spec.
     * But we still want to work with them. The field has a bitmap of
     * nvme driver bugs that we should be aware of.
     */
    enum nvme_quirks   quirks;

    /* Polling groups */
    struct nvme_ctrl_pg *pgs;
    int npgs;

    int fw_file;
    bool fw_update_enable;
    char filename[PATH_MAX];
    bool iova_mgmt_enable;

    rpc_method_is_supported_cb_t rpc_method_is_supported_cb;
    rpc_method_handle_cb_t rpc_method_handle_cb;

    memp_t *memp;

    hotunplug_device_cb_t hotunplug_device_cb;
    hotunplug_timeout_cb_t hotunplug_timeout_cb;
    void                *hotunplug_device_ctx;
    time_t hotunplug_timer_expire;
    bool force_hotunplug_device;
    char *bfb_info[NVME_FW_SLOTS_COUNT + 1];
};

typedef char* (*rpc_method_cb_t)(const void *arg, int *status);

struct nvme_snap_method {
    char *name;
    rpc_method_cb_t rpc_method_cb;
    void *priv;
};

/*
 * Progress all io queues sceduled on the thread thread_id
 *
 * params
 * @ctrl - controller instance
 * @thread_id - logic thread id. Must be >= 0 and < nthreads
 */
int nvme_ctrl_progress_io(nvme_ctrl_t *ctrl, int thread_id);

int nvme_ctrl_progress_all_io(nvme_ctrl_t *ctrl);

/*
 * Progress admin queue
 */
int nvme_ctrl_progress_admin(nvme_ctrl_t *ctrl);
/*
 * Progress mmio
 */
void nvme_ctrl_progress_mmio(nvme_ctrl_t *ctrl);

nvme_ctrl_t *nvme_ctrl_init(nvme_ctrl_init_param_t *param);
void nvme_ctrl_destroy(nvme_ctrl_t *ctrl);
void nvme_ctrl_bar_write_event(void *ctrl);
int nvme_ctrl_stop(nvme_ctrl_t *ctrl);
void nvme_ctrl_suspend(nvme_ctrl_t *ctrl);
bool nvme_ctrl_is_suspended(nvme_ctrl_t *ctrl);

void nvme_admin_cmd_dump(const nvme_cmd_t *cmd);
void nvme_io_cmd_dump(const nvme_cmd_t *cmd);

void nvme_ctrl_pgs_suspend(nvme_ctrl_t *ctrl);
void nvme_ctrl_pgs_resume(nvme_ctrl_t *ctrl);

struct nvme_ctrl_caps {
    char version[16];
    uint32_t max_namespaces;
    uint32_t max_nsid;
    bool offload;
    bool mempool;
};

void nvme_ctrl_query_caps(nvme_ctrl_t *ctrl, struct nvme_ctrl_caps *caps);

void nvme_ctrl_mask_unmask_async_event(nvme_ctrl_t *ctrl,
                enum NvmeAsyncEventConfig event, bool mask);
#endif
