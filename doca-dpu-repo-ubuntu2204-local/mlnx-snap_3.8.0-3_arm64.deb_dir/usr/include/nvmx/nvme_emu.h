#ifndef _NVME_EMU_H
#define _NVME_EMU_H

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <infiniband/verbs.h>

#include "nvme_emu_log.h"
#include "nvme_regs.h"
#include "json/json.h"
#include "utils.h"
#include "memzero_core.h"

#define NVME_DISK_SIZE_MB_VAR           "NVME_DISK_SIZE_MB"
#define NVME_LBA_SIZE_VAR               "NVME_LBA_SIZE"
#define NVME_METADATA_SIZE_VAR          "NVME_METADATA_SIZE"

#define NVME_DEFAULT_POLLING_TIME_INTERVAL_MSEC  0
#define NVME_POLLING_TIME_INTERVAL_MSEC          "NVME_POLLING_TIME_INTERVAL"

#define NVME_EMU_NAME_PREFIX "NvmeEmu"
#define NVME_EMU_NAME_MAXLEN 32


#define NVME_EMU_BAR_CB_TRIGGER (100 * 1000 / CLOCKS_PER_SEC)	/* 100 msec */

struct snap_conf;

/* TODO: defined are here because both snap emulator/libsnap and nvmx code
 * use different queue macros. So we have to avoid including snap_nvme_emulator.h
 */
#define SNAP_EMU_NAME      "snap_emulator"

/**
 * enum nvme_emu_status - NVMe emulation status values
 */
enum nvme_emu_status {
    NVMX_OK  =  0,
    /** @NVMX_INPROGRESS: operation is in progress and has not completed yet */
    NVMX_INPROGRESS = 1,
    NVMX_ERR = -1,
    NVMX_ERR_NOTSUPPORTED = -2,
    NVMX_ERR_NOTREADY = -3
};

#define NVME_EMU_DB_STRIDE       8

typedef struct nvme_emulator        nvme_emulator_t;
/* Declaration from driver and ep */
typedef struct nvme_emu_ep nvme_emu_ep_t;
typedef struct nvme_emu_driver nvme_emu_driver_t;
typedef struct nvme_emu_driver_queue nvme_emu_driver_queue_t;
typedef struct nvme_queue_ops       nvme_queue_ops_t;
typedef struct nvme_emu_ev_data     nvme_emu_ev_data_t;
typedef struct nvme_emu_dev_ns      nvme_emu_dev_ns_t;
/**
 * typedef nvme_async_req_t - See &struct nvme_async_req
 */
typedef struct nvme_async_req       nvme_async_req_t;

typedef struct memp memp_t;

/* Called when host has written to the bar config space. 
 * Where addr >= 0x0 and addr < 0x50 */
typedef void (*nvme_emu_mmio_write_cb_t)(void *ctrl);

enum nvme_emu_queue_type {
    NVME_EMU_QUEUE_SQE,  /* queue receives SQ entries */
    NVME_EMU_QUEUE_NVME2NVMF_LOCAL,
    NVME_EMU_QUEUE_NVME2NVMF_REMOTE
};

typedef int attach_namespace_fn_t(void *ctrl, uint32_t nsid, uint64_t size_b,
                                  uint8_t block_order, uint16_t backend_metadata,
                                  nvme_id_ns_descriptor_t id_descs[], void *priv_ns);
typedef int detach_namespace_fn_t(void *ctrl, uint32_t nsid);

struct nvme_emu_dev_ns {
    void *ns;//Low level namespace
    nvme_emulator_t *dev;
};

void nvme_emu_dev_remove_namespace(nvme_emu_dev_ns_t *ns);
nvme_emu_dev_ns_t *nvme_emu_dev_add_namespace(nvme_emulator_t *dev,
                                uint32_t src_nsid, uint32_t dst_nsid,
                                uint16_t md_size, uint8_t lba_size);

struct nvme_queue_ops {
    int  q_type; /* queue type */
    void *q_impl;/* actual queue implementation */
    int  (*process_sq)(nvme_queue_ops_t *ops, uint32_t db_addr, const void *cmd, uint32_t len);
    int  (*process_cq)(nvme_queue_ops_t *ops, uint32_t db_addr, void *cmd, uint32_t len);

    int              qid;
    int              page_bits;
    nvme_emulator_t *dev;

    /* not currently used */
    int  (*nvme_queue_init)();
    void (*nvme_queue_destroy)();
};

typedef struct nvme_emu_counters {
    size_t data_read;
    size_t data_write;
    size_t cmd_read;
    size_t cmd_write;
} nvme_emu_counters_t;

/* add/remove a queue layer implementation 
 * if is_adm is true the layer applies to the admin qps
 * TODO: extend to filter/priority
 */
int nvme_emu_queue_type_add(nvme_emulator_t *dev, int queue_type, bool is_adm,
                            nvme_queue_ops_t *ops);
void nvme_emu_queue_type_del(nvme_emulator_t *dev, int queue_type, bool is_adm);

struct nvme_emu_create_cq_attr {
    int qid;
    uint64_t base_addr;
    uint16_t size;
    int vector;
    int cq_period;
    int cq_max_count;
    bool irq_disable;
};

struct nvme_emu_create_sq_attr {
    int id;
    uint64_t base_addr;
    uint16_t size;
    nvme_queue_ops_t *cq_ops;
    bool fe_only;
};

/* create/destroy SQ/CQ
 * On success the ops structure is filled 
 */
int nvme_emu_create_sq(nvme_emulator_t *dev,
                       const struct nvme_emu_create_sq_attr *attr,
                       nvme_queue_ops_t *ops);
int nvme_emu_create_cq(nvme_emulator_t *dev, nvme_queue_ops_t *ops,
                       const struct nvme_emu_create_cq_attr *qattr);
int nvme_emu_attach_sq_qp(nvme_emulator_t *dev, int qid, struct ibv_qp *qp);
int nvme_emu_detach_sq_qp(nvme_emulator_t *dev, int qid);
/* remote sq or cq */
int nvme_emu_delete_sq(nvme_queue_ops_t *ops, int sqid);
int nvme_emu_delete_cq(nvme_queue_ops_t *ops, int cqid);

static inline int nvme_emu_process_sq(nvme_queue_ops_t *ops, uint32_t db_addr, const void *cmd, uint32_t len)
{
    return ops->process_sq(ops, db_addr, cmd, len);
}

static inline int nvme_emu_process_cq(nvme_queue_ops_t *ops, uint32_t db_addr, void *cmd, uint32_t len)
{
    return ops->process_cq(ops, db_addr, cmd, len);
}

struct ibv_pd *nvme_emu_get_pd(nvme_emulator_t *emulator);

typedef struct nvme_query_attr {
    /* Device Capabilities */
    uint32_t    max_namespaces;         /**< max number of namespaces support per SQ */
    uint32_t    max_nsid;         /**< max NSID value supported by HW */
    uint16_t    max_reg_size;       /**< max size of the bar register area */
    uint32_t    max_emulated_cq_num;    /**< max number of CQs that can be emulated */
    uint32_t    max_emulated_sq_num;    /**< max number of SQs that can be emulated */
    uint32_t    max_emulated_pfs;    /**< max number of NVMe physical functions that can be emulated */
    uint32_t    max_emulated_vfs;    /**< max number of NVMe virtual functions that can be emulated */
} nvme_query_attr_t;

struct nvme_emulator {
    /* Mellanox Emulation SDK */
    void                        *ctrl;   /* points to controller */
    nvme_emu_mmio_write_cb_t     bar_write_host_cb;
    nvme_bar_instance_t          bar;
    nvme_queue_ops_t             *admin_q_ops;
    nvme_queue_ops_t             *io_q_ops;
    nvme_query_attr_t            dev_caps;
    uint16_t                     vid;
    uint16_t                     ssvid;

    struct snap_conf             *config;
    char                         name[16];

    uint8_t                      pci_hotplug_state_cur;
    uint8_t                      pci_hotplug_state_prev;
};

void nvme_emu_qops_clone(nvme_emulator_t *dev,
                                nvme_queue_ops_t *ops, int qid);

/* NVME emulation functions */

/* Opening this device will use simple in memory
 * emulation of the nvme controller.
 */
#define NVME_EMU_TEST_DEV "nvme_test"

/**
 * Setup NVME emulator
 * The function will do following things:
 * - check that the device exists
 * - check device attributes: (dev_emu and cmd_on_behalf)
 * - initialize NVME emulation (init hca on behalf of host)
 * - create QP that to pass doorbells and to read/write from
 *   the host memory
 *
 * return:
 *
 * Emulator object on success, Otherwise NULL.
 */
nvme_emulator_t *nvme_emu_dev_open(const char *hca_manager,
                                   const char *hca_transport,
                                   struct snap_conf *config,
                                   memp_t *memp,
                                   nvme_emu_mmio_write_cb_t bar_write_ev_handler,
                                   void *dev_ctx,
                                   uint32_t *num_queues);

/* close NVME emulation device */
void nvme_emu_dev_close(nvme_emulator_t *dev);

/* start NVMe emulation device */
int nvme_emu_dev_start(nvme_emulator_t *dev);

/* stop NVMe emulation device */
int nvme_emu_dev_stop(nvme_emulator_t *dev);

/* check if NVMe emulation device started */
bool nvme_emu_dev_is_started(nvme_emulator_t *dev);

/* read from host bar space */
int nvme_emu_mmio_read(nvme_emulator_t *dev, void *buf, uint32_t bar_addr,
		               unsigned len);

/* write to host bar space */ 
int nvme_emu_mmio_write(nvme_emulator_t *dev, void *buf, uint32_t bar_addr,
                        unsigned len);


/* do dma transfers to/from hst memory */
/* read from the host memory */
void nvme_emu_dma_read(nvme_queue_ops_t *ops, void *dst_buf, uint64_t srcaddr,
                       size_t len, uint32_t rmkey);
/* write to the host memory */
void nvme_emu_dma_write(nvme_queue_ops_t *ops, void *src_buf, uint64_t dstaddr,
                        size_t len, uint32_t rmkey);
/* send to host memory */
int nvme_emu_send(nvme_queue_ops_t *ops, void *src_buf, size_t len);

/* generate interrupt on host */
void nvme_emu_interrupt(nvme_queue_ops_t *ops);

/**
 * nvme_emu_progress_sq() - Handle in-flight IO requests on SQ
 * @ops:        queue operations
 *
 * This function should be called iteratively by a polling thread.
 * On each call, checks for any new IO requests received, and handle
 * them (move them through all required steps until completion).
 *
 * Return: number of events (send and receive) that were processed
 */
int nvme_emu_progress_sq(nvme_queue_ops_t *ops);

/**
 * nvme_emu_progress_mmio() - Handle admin queue / PCI bar changes
 * @dev:        nvme emulator
 *
 * This function should be called iteratively by a polling thread.
 * On each call, checks for any PCI bar changes or any new admin
 * queue commands received, and handle them until completion.
 */
void nvme_emu_progress_mmio(nvme_emulator_t *dev);

bool nvme_emu_query_ctrl_counters(nvme_emulator_t *dev, nvme_emu_counters_t *counters);
bool nvme_emu_query_sq_counters(nvme_emulator_t *dev, int qid, nvme_emu_counters_t *counters);

void nvme_emu_destroy_zcopy_key_table(void *key_table);
void *nvme_emu_create_zcopy_key_table(char *name);

memzero_ops_t *nvme_emu_dma_create_memzero(nvme_emulator_t *dev, int qid);

bool nvme_emu_is_pf(nvme_emulator_t *dev);

/* Async request API */

/**
 * typedef nvme_req_cb_t - async request completion callback
 * @req:     async request. See &struct nvme_async_req
 * @status:  request completion status as defined in &enum nvme_emu_status
 *
 * The callback is called when the async request has been completed
 */
typedef void (*nvme_req_cb_t)(void *user, enum nvme_emu_status status);

/**
 * enum nvme_async_req_op - DMA operation type
 *
 * Note that the direction is given from the point of view of the
 * NVMe controller.
 *
 * @NVME_REQ_DMA_TO_HOST:    DMA write from the controller to the host memory
 * @NVME_REQ_DMA_FROM_HOST:  DMA read from the host memory to the controller
 * @NVME_REQ_ZCOPY:          No DMA is done, only iovec is filled.
 */
enum nvme_async_req_op {
        NVME_REQ_DMA_TO_HOST,
        NVME_REQ_DMA_FROM_HOST,
        NVME_REQ_IOV,
};

typedef struct {
    /* Base address in host memory */
    uint64_t host_addr;
    /* Total descriptors count in current segment */
    uint32_t nr_total;
    /* Amount of processed descriptors in current segment */
    uint32_t nr_processed;
    /* Amount of descriptors in current fragment */
    uint32_t fragment_len;
    /* Signals that this is a last descriptor */
    bool last_segment;
    /* Current fragment */
    nvme_sgl_desc_t *fragment;
} sgl_segment;

/**
 * struct nvme_async_req - NVMe async DMA request
 *
 * The request is used to copy data to or from the host memory
 * in a non-blocking, asynchronous way.
 *
 * The request is a low level building block which is used by
 * a higher level APIs such as nvme_prp_rw(), nvme_sgl_rw() and
 * by the following io driver functions:
 *  nvme_driver_write_prp_nb()
 *  nvme_driver_read_prp_nb()
 *  nvme_driver_write_sgl_nb()
 *  nvme_driver_read_sgl_nb()
 * to transfer data from or to the host memory.
 *
 * Request lifecycle:
 *
 *  - The request must be allocated. Because request may have private data
 *    use nvme_emu_request_size() to get true request size.
 *  - Use nvme_emu_request_init() to initialize request.
 *  - Set &nvme_async_req.comp_cb and call nvme_prp_rw() or nvme_sgl_rw()
 *  - Or setup &nvme_async_req.dma_cmd and call nvme_emu_request_submit()
 *  - Use nvme_emu_request_reset() and free request's memory.
 */
struct nvme_async_req {
    /** @ops: queue operations */
    nvme_queue_ops_t    *ops;
    /** @memory pool for this request */
    memp_t              *memp;
    /** @thread_id: I/O thread id or -1 for main application thread */
    int                 thread_id;
    /** @data_buf:  memory buffer from/to read/write data */
    uint8_t             *io_buf;
    size_t              data_len;
    size_t              aux_len;
    /** @aux_buf:   memory buffer for aux data: PRP list or SGL segments */
    uint8_t             *aux_buf;
    /** @bdata:     memory buffer synchronous PRP operations */
    uint8_t             *bdata;
    bool                is_prp;
    enum nvme_status_code err_code;
    /**
     * @comp_cb: callback called by nvme_prp_rw() or nvme_sgl_rw()
     *           once the request is completed
     */
    nvme_req_cb_t       comp_cb;
    void                *comp_cb_user;
    /**
     * @len: size of data to be read/written from/to prp/sgl
     */
    size_t              len;

    bool                is_tcp_rx_zcopy;
    /* 
     * @tcp_rx_zcopy: fields related to RX zcopy flow
     * The req.tcp_rx_zcopy.sgl/iov/iov_cnt are for the source SGL and are
     * not the same as the req.iov/iov_cnt
     */
    struct {
        struct ibv_sge       sgl[SGL_MAX_DESCRIPTORS];
        uint32_t             lkey[SGL_MAX_DESCRIPTORS];
        struct iovec         *iov;
        int                  iov_cnt;
        enum nvme_emu_status req_status;
    } tcp_rx_zcopy;

    /**
     * @iov: prp/sg list converted to the dest io vector
     */
    struct iovec        *iov;
    /**
     * @iov_cnt: number of dest io vectors
     */
    size_t               iov_cnt;
    /**
     * @iov_cnt: max number of io vectors
     */
    size_t               iov_max;

    /** @prp: prp state which is used by the nvme_prp_rw() */
    struct {
        /**
         * @prp.prp1: as defined by the NVMe spec
         */
        uint64_t  prp1;
        /**
         * @prp.prp2: as defined by the NVMe spec
         */
        uint64_t  prp2;
        /**
         * @prp.list_size: prp list size
         */
        int       list_size;
        /**
         * @prp.saved_op: used to save original operation when we need to
         * read prp list entries.
         */
        enum nvme_async_req_op  saved_op;
    } prp;

    struct {
        /**
         * @sgl.sgl: as defined by the NVMe spec
         */
        nvme_sgl_desc_t sgl;
        struct {
            uint32_t data_length;
            uint32_t bytes_processed;
            uint8_t eof:1;
            uint8_t iov:1;
            uint8_t data:1;
            uint8_t fragment:1;
            uint8_t write:1;
        } operation_context;

        sgl_segment segment;
    } sgl;

    /**
     * @dma_cmd: describe dma operation that will be done by nvme_request_submit()
     *
     * DMA_TO_HOST: (data_buf + offset, len) write to (raddr, rkey)
     * DMA_FROM_HOST: (data_buf + offset, len)  read from (raddr, rkey)
     */
    struct {
        /**
         * @dma_cmd.op: operation type
         */
        enum nvme_async_req_op op;
        /**
         * @dma_cmd.raddr: address in the host memory
         */
        uint64_t raddr;
        /**
         * @dma_cmd.rkey: host rkey
         */
        uint32_t rkey;
        /**
         * @dma_cmd.laddr: address in local memory
         */
        uint8_t *laddr;
        /**
         * @dma_cmd.len: how many bytes should be transferred
         */
        size_t   len;
        /**
         * @dma_cmd.comp_cb: called once the DMA operation is completed
         */
        nvme_req_cb_t comp_cb;
        void *comp_cb_user;
        bool writev2v;
    } dma_cmd;
};

/**
 * nvme_emu_request_size() - Get async request size
 * @dev: nvme emulator
 *
 * The function should be used to obtain true request size which
 * can be then used to allocate memory with malloc() or calloc()
 *
 * Return: true async request size, including private size:
 *         sizeof(nvme_async_req_t) + private_data_size
 */
int nvme_emu_request_size(nvme_emulator_t *dev);

/**
 * nvme_emu_sq_init() - Initialize sq
 * @ops:    queue operations
 * @io_buf: pointer to data buffer
 * @len:    data buffer length
 *
 * The function registers resources needed by the sq.
 *
 * Rertun: ibv_mr pointer or Null on error
 */
struct ibv_mr *nvme_emu_sq_init(nvme_queue_ops_t *ops, uint8_t *io_buf, size_t len);

/**
 * nvme_emu_request_init() - Initialize async request
 * @ops:    queue operations
 * @req:    request to initialize
 * @mr:     mr for request
 *
 * The function allocates resources needed by the async request.
 *
 * Rertun: NVMX_OK or error status
 */
int nvme_emu_request_init(nvme_queue_ops_t *ops, nvme_async_req_t *req, struct ibv_mr *mr);

/**
 * nvme_emu_sq_reset() - Free sq resources
 * @req: first async request on sq to be freed
 *
 * The function frees resources allocated by nvme_emu_sq_init().
 * It does not free data buffer memory.
 */
void nvme_emu_sq_reset(nvme_async_req_t *req);

/**
 * nvme_emu_request_reset() - Free async request resources
 * @req: async request
 *
 * The function frees resources allocated by nvme_emu_request_init().
 * It does not free data buffer memory.
 */
void nvme_emu_request_reset(nvme_async_req_t *req);

/**
 * nvme_emu_request_submit() - Start DMA operation
 * @req: async request
 *
 * The function starts DMA operation specified by the
 * &struct nvme_async_req->dma_cmd. &struct nvme_async_req->dma_cmd.comp_cb
 * will be called when the operation completes.
 *
 * Return: NVMX_INPROGRESS or error status
 */
int nvme_emu_request_submit(nvme_async_req_t *req);

/**
 * Helper function to set number of queues according to caps
 */
void nvme_emu_set_num_queues(nvme_query_attr_t *dev_caps,
                             struct snap_conf *config,
                             uint32_t *num_queues);

int snap_nvme_ctrl_hotunplug(nvme_emulator_t *dev);
#endif
