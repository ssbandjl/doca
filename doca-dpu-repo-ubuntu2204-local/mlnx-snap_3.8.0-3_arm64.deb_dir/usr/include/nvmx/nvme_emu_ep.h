#ifndef _NVME_EMU_EP_H
#define _NVME_EMU_EP_H

#include "nvme_emu.h"
#include "nvme_emu_io_driver.h"
#include "nvme.h"

#define NVME_EP_TYPE_DEFAULT "posix_io"

typedef struct nvme_emu_ep_ops nvme_emu_ep_ops_t;
typedef struct nvme_emu_ep_queue nvme_emu_ep_queue_t;

/*
 * NVMe Emulation endpoint family
 */
enum nvme_emu_ep_fam {
    /* Default */
    NVME_EMU_ADRFAM_NONE = 0x0,
    /* IPv4 (AF_INET) */
    NVME_EMU_ADRFAM_IPV4 = 0x1,
    /* IPv6 (AF_INET6) */
    NVME_EMU_ADRFAM_IPV6 = 0x2,
    /* InfiniBand (AF_IB) */
    NVME_EMU_ADRFAM_IB   = 0x3,
    /* Fibre Channel */
    NVME_EMU_ADRFAM_FC   = 0x4,
};

/*
 * NVMe Emulation endpoint types
 */
enum nvme_emu_ep_type {
    /* NVMe-oF RDMA endpoint */
    NVME_EMU_EP_TYPE_NVMF_RDMA  = 0x1,
    /* POSIX endpoint */
    NVME_EMU_EP_TYPE_POSIX      = 0x2,
    /* Memory endpoint */
    NVME_EMU_EP_TYPE_MEM        = 0x3,
    /* SPDK endpoint */
    NVME_EMU_EP_TYPE_SPDK       = 0x4,
};

/*
 * NVMe Emulation endpoint states:
 *   - NVME_EMU_EP_NEW: All transport resources are cleaned and now we're
 *     ready to start (initial state during endpoint creation or a state
 *     we move after destroying transport).
 *
 *   - NVME_EMU_EP_ADMIN_START: initial low level ctrl resources bring up.
 *     For example, in nvmf_rdma we'll create the admin queue, identify
 *     remote ctrl and it's namespaces and start keep alive mechanism.
 *
 *   - NVME_EMU_EP_START: Successfully connect/reconnect/re-build transport
 *     layer resources. For fabrics, create admin queue and connect it,
 *     identify ctrl and namespaces, connect IO queues (at least 1 or in case
 *     of reconnection return to previous state of IO queues count) in
 *     transport level and start keep alive mechanism. This differs from
 *     NVME_EMU_EP_ADMIN_START state in meaning of the transport level
 *     connectivity to remote ctrl from IO queues perspective. Fabrics
 *     protocol connectivity is not established for IO queues yet.
 *
 *   - NVME_EMU_EP_ACTIVE: low level ctrl activation (first SQ creation and
 *     activation). For fabrics active ctrl, send fabric connect command
 *     and receive successful connect response.
 *
 *   - NVME_EMU_EP_DELETING: get notification from low level transport layer
 *     that something wrong has happened (For example, in case of keep alive
 *     expiration for fabrics ctrl). In this case, just mark the failed
 *     endpoint (deactivate if needed). Later on, the progress thread will stop
 *     the endpoint (and move to NVME_EMU_EP_STOP state).
 */
enum nvme_emu_ep_state {
    NVME_EMU_EP_NEW = 0x0,
    NVME_EMU_EP_ADMIN_START,
    NVME_EMU_EP_START,
    NVME_EMU_EP_ACTIVE,
    NVME_EMU_EP_DELETING,
};

/*
 * NVMe Emulation endpoint queue states:
 *   - NVME_EMU_EP_QUEUE_NEW: initial state during queue creation or a state
 *     we move after destroying transport queue.
 *
 *   - NVME_EMU_EP_QUEUE_START: transport queue created, but not active.
 *
 *   - NVME_EMU_EP_QUEUE_ACTIVE: transport queue activated.
 */
enum nvme_emu_ep_queue_state {
    NVME_EMU_EP_QUEUE_NEW = 0x0,
    NVME_EMU_EP_QUEUE_START,
    NVME_EMU_EP_QUEUE_ACTIVE,
};

enum nvme_emu_ep_queue_type {
    NVME_EMU_EP_SQ = 0x1,
    NVME_EMU_EP_CQ = 0x2,
};

#define NVME_EMU_MAX_EP_TYPE_LEN 16

#define NVME_EMU_ADDR_MAX_LEN 256
#define NVME_EMU_MAX_NAME_LEN 256
#define NVME_EMU_MAX_PORT_LEN 32

struct nvme_emu_ep_ops {
    int (*write_prp)(nvme_emu_ep_t *ep,
                     nvme_io_driver_req_t *req,
                     const nvme_cmd_rw_t *rw);
    int (*read_prp)(nvme_emu_ep_t *ep,
                    nvme_io_driver_req_t *req,
                    const nvme_cmd_rw_t *rw);
    int (*write_sgl)(nvme_emu_ep_t *ep,
                     nvme_io_driver_req_t *req,
                     const nvme_cmd_rw_t *rw);
    int (*read_sgl)(nvme_emu_ep_t *ep,
                    nvme_io_driver_req_t *req,
                    const nvme_cmd_rw_t *rw);
    int (*create_cq)(nvme_emu_ep_t *ep, uint16_t cqid, uint16_t size);
    void (*delete_cq)(nvme_emu_ep_t *ep, uint16_t cqid);
    int (*activate_cq)(nvme_emu_ep_t *ep, uint16_t cqid);
    void (*deactivate_cq)(nvme_emu_ep_t *ep, uint16_t cqid);
    int (*create_sq)(nvme_emu_ep_t *ep, uint16_t sqid, uint16_t size,
                     uint16_t cqid);
    void (*delete_sq)(nvme_emu_ep_t *ep, uint16_t sqid);
    int (*activate_sq)(nvme_emu_ep_t *ep, uint16_t sqid);
    void (*deactivate_sq)(nvme_emu_ep_t *ep, uint16_t sqid);

    /* request api */
    int (*request_init)(nvme_emu_ep_t *ep, nvme_queue_ops_t *ops,
                        nvme_io_driver_req_t *req);
    void (*request_reset)(nvme_io_driver_req_t *req);

    int (*write_prp_nb)(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);
    int (*read_prp_nb)(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);

    int (*write_sgl_nb)(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);
    int (*read_sgl_nb)(nvme_io_driver_req_t *req, const nvme_cmd_rw_t *rw);

    int (*io_buf_create)(size_t len, memp_t *memp, nvme_io_driver_req_t *req);
    void (*sq_reset)(nvme_io_driver_req_t *req);
};

struct nvme_emu_ep_queue {
    nvme_emu_ep_t                *ep;
    enum nvme_emu_ep_queue_type  type;
    enum nvme_emu_ep_queue_state state;
    uint16_t                     depth;
    uint16_t                     qid;

    /* valid only for NVME_EMU_EP_QUEUE_SQ */
    uint16_t                     cqid;
};

/*
 * This identifies a unique endpoint on an NVMe emulation. It will be used
 * by the Emulation driver to create an endpoint. E.g. for NVMe-oF driver it
 * will describe the target subsystem. For Block driver it will describe the
 * block device that will be used as an endpoint.
 */
struct nvme_emu_ep {
    nvme_emu_driver_t       *io_driver;
    enum nvme_emu_ep_type   ep_type;
    enum nvme_emu_ep_fam    ep_fam;
    char                    name[NVME_EMU_MAX_NAME_LEN + 1];
    char                    addr[NVME_EMU_ADDR_MAX_LEN + 1];
    char                    port[NVME_EMU_MAX_PORT_LEN + 1];
    char                    nqn[NVME_EMU_MAX_NQN_LEN + 1];
    void                    *ctrl; //transport specific
    nvme_emu_ep_ops_t       ep_ops;
    enum nvme_emu_ep_state  state; //controlled by the io_driver
    nvme_emu_ep_queue_t     *sqs;
    nvme_emu_ep_queue_t     *cqs;
    int                     num_created_io_squeues;
    int                     num_created_io_cqueues;

    int                     backend_req_size;
};

int nvme_emu_ep_create_backend(nvme_emu_ep_t *ep);
void nvme_emu_ep_destroy_backend(nvme_emu_ep_t *ep);
int nvme_emu_ep_start_backend(nvme_emu_ep_t *ep);
void nvme_emu_ep_stop_backend(nvme_emu_ep_t *ep, bool clear_namespaces);
int nvme_emu_ep_activate_backend(nvme_emu_ep_t *ep, int *activated_cqs,
                                 int *activated_sqs);
void nvme_emu_ep_deactivate_backend(nvme_emu_ep_t *ep);
void nvme_emu_ep_fail_backend(nvme_emu_ep_t *ep);
int nvme_emu_ep_attach_volume(nvme_emu_ep_t *ep, uint32_t nsid,
                              uint64_t size_b, uint8_t block_order,
                              uint16_t backend_metadata,
                              nvme_id_ns_descriptor_t id_descs[], void *priv_ns);
void nvme_emu_ep_detach_volume(nvme_emu_ep_t *ep, uint32_t nsid);
void nvme_emu_ep_stop_sqs(nvme_emu_ep_t *ep);
int nvme_emu_ep_start_sqs(nvme_emu_ep_t *ep);
void nvme_emu_ep_stop_cqs(nvme_emu_ep_t *ep);
int nvme_emu_ep_start_cqs(nvme_emu_ep_t *ep);
int nvme_emu_ep_create_sq(nvme_emu_ep_t *ep, uint16_t sqid, uint16_t depth,
                          uint16_t cqid);
void nvme_emu_ep_delete_sq(nvme_emu_ep_t *ep, uint16_t sqid);
void nvme_emu_ep_deactivate_sq(nvme_emu_ep_t *ep, uint16_t sqid);
int nvme_emu_ep_create_cq(nvme_emu_ep_t *ep, uint16_t cqid, uint16_t depth);
void nvme_emu_ep_delete_cq(nvme_emu_ep_t *ep, uint16_t cqid);
void nvme_emu_ep_deactivate_cq(nvme_emu_ep_t *ep, uint16_t cqid);
#endif
