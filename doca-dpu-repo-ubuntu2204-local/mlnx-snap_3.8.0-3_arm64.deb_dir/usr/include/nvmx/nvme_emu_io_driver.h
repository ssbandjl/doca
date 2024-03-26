#ifndef _NVME_EMU_IO_DRIVER_H
#define _NVME_EMU_IO_DRIVER_H

#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "nvme.h"
#include "nvme_emu.h"

#include "json/json.h"

 /* 512 MB is default disk size
 * Can be changed via env var NVME_DISK_SIZE
 */
#define NVME_DISK_DEFAULT_SIZE    (512 * 1024 * 1024ULL)
#define NVME_DISK_MAX_STRLEN       9
#define NVME_DISK_DEFAULT_LBA      9
#define NVME_DISK_DEFAULT_METADATA 0
#define NVME_DISK_SIZE_VAR        "NVME_IO_DISK_SIZE"

struct snap_conf;

enum nvme_emu_driver_state {
    NVME_EMU_DRIVER_NEW           = 0x0,
    NVME_EMU_DRIVER_ACTIVE        = 0x1,
};

/*
 * NVMe Emulation driver types
 */
enum nvme_emu_driver_type {
    /* single backend driver */
    NVME_EMU_SINGLE_EP_DRIVER_TYPE   = 0x1,
    /* HA driver */
    NVME_EMU_MULTIPLE_EP_DRIVER_TYPE = 0x2,
};

enum nvme_emu_driver_queue_type {
    NVME_EMU_DRIVER_SQ   = 0x1,
    NVME_EMU_DRIVER_CQ   = 0x2,
};

struct nvme_json_val;

struct nvme_emu_driver_queue {
    nvme_emu_driver_t *driver;
    enum nvme_emu_driver_queue_type type;
    uint16_t qid;
    uint16_t depth;

    /* valid for NVME_EMU_DRIVER_SQ */
    uint16_t cqid;
};

/*
 * Template structure that each underling layer should supply to
 * the Nvme ctrl for processing data received from the host OS.
 */
struct nvme_emu_driver {
    int                           index;
    enum nvme_emu_driver_type     type;
    enum nvme_emu_driver_state    state;
    pthread_mutex_t               state_lock;
    int                           nr_endpoints;
    int                           active_endpoint;
    pthread_mutex_t               ep_lock;
    nvme_emu_ep_t                 *endpoints;
    nvme_emulator_t               *dev;
    uint32_t                      ver;
    uint8_t                       mdts;
    int                           num_queues;
    int                           num_created_io_squeues;
    int                           num_created_io_cqueues;
    nvme_emu_driver_queue_t       *sqs;
    nvme_emu_driver_queue_t       *cqs;
    struct snap_conf              *sconfig;
    pthread_t                     d_thread;
    sem_t                         sem;
    bool                          cross_rdma_dev_support;

    /* Controller callbacks */
    attach_namespace_fn_t            *attach_namespace_cb;
    detach_namespace_fn_t            *detach_namespace_cb;
};

int nvme_driver_start(nvme_emu_driver_t *driver);
void nvme_driver_stop(nvme_emu_driver_t *driver);
void nvme_driver_destroy(nvme_emu_driver_t *driver);
nvme_emu_driver_t *nvme_driver_create(nvme_emulator_t *dev,
                                      struct snap_conf *sconfig,
                                      int num_queues,
                                      attach_namespace_fn_t *attach_namespace_fn,
                                      detach_namespace_fn_t *detach_namespace_fn,
                                      int index, uint32_t ver, uint8_t mdts);

int nvme_driver_create_sq(nvme_emu_driver_t *driver, uint16_t sqid,
                          uint64_t dma_addr, uint16_t size,
                          nvme_queue_ops_t *cq_ops, nvme_queue_ops_t *sq_ops);
int nvme_driver_create_cq(nvme_emu_driver_t *driver, int qid,
                          uint64_t dma_addr,uint16_t size, uint16_t vector,
                          int cq_period, int cq_max_count,
                          uint16_t irq_enabled, nvme_queue_ops_t *cq_ops);
void nvme_driver_delete_sq(nvme_emu_driver_t *driver, nvme_queue_ops_t *sq_ops,
                           int qid);
void nvme_driver_delete_cq(nvme_emu_driver_t *driver, nvme_queue_ops_t *cq_ops,
                           int qid);
void nvme_driver_fail_ep(nvme_emu_driver_t *driver, nvme_emu_ep_t *ep);
int nvme_driver_attach_ep_volume(nvme_emu_driver_t *driver, uint32_t nsid,
                                 uint64_t size_b, uint8_t block_order,
                                 uint16_t backend_metadata,
                                 nvme_id_ns_descriptor_t id_descs[], void *priv_ns);
void nvme_driver_detach_ep_volume(nvme_emu_driver_t *driver, uint32_t nsid);

/**
 * struct nvme_io_driver_req - NVMe async io driver request
 *
 * The request is used to transfer data to/from disk in a non-blocking,
 * asynchronous way.
 *
 * Request lifecycle:
 *
 * - The request must be allocated. Because request may have private data both
 *   in the @async_req and in the io driver controller implementation.
 *   nvme_driver_request_size() must be used to obtain true size of the request.
 *
 * - Use nvme_driver_request_init() to initialize request
 * - Use one of the following functions to start io operation:
 *      nvme_driver_write_prp_nb()
 *      nvme_driver_read_prp_nb()
 *      nvme_driver_write_sgl_nb()
 *      nvme_driver_read_sgl_nb()
 * - Use nvme_driver_request_reset() to release resources. Free request.
 *
 * Request memory layout is:
 *
 *  nvme_io_driver_req_t
 *  nvme_async_req_t
 *  async_req_private_data
 *  nr_backends * backend_private_data
 *
 *  Note: that multiple ep are not supported yet!!!
 */
struct nvme_io_driver_req {
    /** @ep: io driver endpoint */
    nvme_emu_ep_t   *ep;
    /** @disk_offset: disk offset to read/write */
    uint64_t         disk_offset;
    /** @backend: Context for backend retrieval on DMA completions */
    void            *ns;
    /** @comp_cb: callback to call once the request is completed */
    nvme_req_cb_t    comp_cb;
    void             *comp_cb_user;
    /** @id: request id */
    int              id;
    /** @zcopy: indicates that request my be processed by ext_io path */
    bool             ext_io;

    struct spdk_bdev_io  *bdev_io;

    // DANGER!!! The "async_req" field may be casted to snap_async_req
    // So it must stay at the last position, do not add extra fields after it
    // TODO: Subject for future refactoring

    /** @async_req: keep state of the DMA operation */
    nvme_async_req_t async_req;
};

/**
 * typedef nvme_io_driver_req_t - See &struct nvme_io_driver_req
 */
typedef struct nvme_io_driver_req nvme_io_driver_req_t;

/**
 * nvme_driver_request_size() - Get io request size
 * @driver: io driver
 *
 * The function should be used to obtain true request size which
 * can be then used to allocate memory with malloc() or calloc()
 *
 * Return: true async request size, including private size:
 *         sizeof(nvme_io_driver_req_t) + private_async_data_size + private_io_data_size
 */
int nvme_driver_request_size(nvme_emu_driver_t *driver);

/**
 * nvme_driver_sq_init() - Initialize sq
 * @driver:  io driver
 * @memp:    memory pool if exists
 * @ops:     nvme queue operations
 * @q_size:  queue size
 * @req:     request to initialize
 *
 * Return: ibv_mr pointer or Null on error
 */
struct ibv_mr *nvme_driver_sq_init(nvme_emu_driver_t *driver, memp_t *memp,
                                   nvme_queue_ops_t *ops, uint16_t q_size,
                                   nvme_io_driver_req_t *req);

/**
 * nvme_driver_request_init() - Initialize io request
 * @driver:   io driver
 * @ops:      nvme queue operations
 * @req:      request to initialize
 * @comp_cb:  completion callback that will be called on the request completion
 * @reqid:    request ID
 * @memp:     memory pool
 * @data_len: data len for request io_buf
 * @aux_len:  aux len for request io_buf
 * @io_buf:   sq io_buf
 * @mr:       sq mr
 *
 * Return: NVMX_OK or error status
 */
int nvme_driver_request_init(nvme_emu_driver_t *driver, nvme_queue_ops_t *ops,
                             nvme_io_driver_req_t *req, nvme_req_cb_t comp_cb,
                             void *comp_cb_user, int reqid, memp_t *memp,
                             size_t data_len, size_t aux_len, uint8_t *io_buf,
                             struct ibv_mr *mr);
/**
 * nvme_driver_sq_reset() - Free sq resources
 * @req:    first io request on sq to free
 *
 * The function frees resources allocated by the request.
 */
void nvme_driver_sq_reset(nvme_io_driver_req_t *req);

/**
 * nvme_driver_request_reset() - Free io request resources
 * @req:    io request to free
 *
 * The function frees resources allocated by the request.
 */
void nvme_driver_request_reset(nvme_io_driver_req_t *req);

/**
 * nvme_driver_progress() - Progress io requests on the specific queue
 * @ops: nvme queue
 *
 * NOTE: NOT IMPLEMENTED
 * We don't need it now because memdisk has no io progress and spdk bdev progress
 * is done elsewhere.
 */
void nvme_driver_progress(nvme_queue_ops_t *ops);

/**
 * nvme_driver_write_prp_nb() - Write data to disk from the prp list
 * @req:    io request that will be used to keep state
 * @rw:     NVMe io command as defined by spec
 *
 * The function starts non blocking, async write to disk from the memory
 * location described by the prp list in @rw
 *
 * Return: NVMX_INPROGRESS or error status
 */
int nvme_driver_write_prp_nb(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);

/**
 * nvme_driver_read_prp_nb() - Read data from disk to the prp list
 * @req:    io request that will be used to keep state
 * @rw:     NVMe io command as defined by spec
 *
 * The function starts non blocking, async read from disk to the memory
 * location described by the prp list in @rw
 *
 * Return: NVMX_INPROGRESS or error status
 */
int nvme_driver_read_prp_nb(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);

/**
 * nvme_driver_write_sgl_nb() - Write data to disk from the scatter gather list
 * @req:    io request that will be used to keep state
 * @rw:     NVMe io command as defined by spec
 *
 * The function starts non blocking, async write to disk from the memory
 * location described by the scatter gather list in @rw
 *
 * NOTE: only one element scatter gather list is supported.
 *
 * Return: NVMX_INPROGRESS or error status
 */
int nvme_driver_write_sgl_nb(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);

/**
 * nvme_driver_read_sgl_nb() - Read data from disk to the scatter gather list
 * @req:    io request that will be used to keep state
 * @rw:     NVMe io command as defined by spec
 *
 * The function starts non blocking, async read rfom disk to the memory
 * location described by the scatter gather list in @rw
 *
 * NOTE: only one element scatter gather list is supported.
 *
 * Return: NVMX_INPROGRESS or error status
 */
int nvme_driver_read_sgl_nb(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);

#endif
