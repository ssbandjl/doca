#ifndef _NVMF_RDMA_H
#define _NVMF_RDMA_H

#include <stdbool.h>
#include <pthread.h>
#include <semaphore.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include "nvmf_emu_ctrl.h"

#define NVME_RDMA_DEFAULT_PORT 4420
#define NVME_RDMA_TIMEOUT_IN_MS 3000
#define NVMF_RDMA_POOL_BATCH 16
#define NVMF_RDMA_MAX_COMPLETIONS 64
#define NVMF_RDMA_PAYLOAD_BUFFER_SIZE 32768
#define NVMF_MAX_CC_RESPONSE_QUERIES 10000
#define NVMF_SLEEP_BETWEEN_CC_RESPONSE_QUERIES_IN_USECS 500

typedef struct nvmf_rdma_qpair nvmf_rdma_qpair_t;
typedef struct nvmf_rdma_ctrl nvmf_rdma_ctrl_t;

/*
 * RDMA Queue state machine:
 *   DISCONNECTED <==(create/destroy queue) ==> READY
 *   READY <==(activate/deactivate queue)==> CONNECTED
 */
enum nvmf_rdma_queue_state {
    /* RDMA connection is down */
    NVMF_RDMA_QUEUE_DISCONNECTED = 0,

    /* RDMA connection established, NVMf connection is down */
    NVMF_RDMA_QUEUE_READY = 1,

    /* NVMf connection established */
    NVMF_RDMA_QUEUE_CONNECTED = 2,
};

/*
 * RDMA Queue Pair service types
 */
enum nvmf_rdma_qptype {
	/* Reliable connected */
	NVMF_RDMA_QPTYPE_RELIABLE_CONNECTED		= 0x1,
	/* Reliable datagram */
	NVMF_RDMA_QPTYPE_RELIABLE_DATAGRAM		= 0x2,
};

/*
 * RDMA provider types
 */
enum nvmf_rdma_prtype {
	/* No provider specified */
	NVMF_RDMA_PRTYPE_NONE		= 0x1,
	/* InfiniBand */
	NVMF_RDMA_PRTYPE_IB		= 0x2,
	/* RoCE v1 */
	NVMF_RDMA_PRTYPE_ROCE		= 0x3,
	/* RoCE v2 */
	NVMF_RDMA_PRTYPE_ROCE2		= 0x4,
	/* iWARP */
	NVMF_RDMA_PRTYPE_IWARP		= 0x5,
};

/*
 * RDMA connection management service types
 */
enum nvmf_rdma_cms {
        /* Sockets based endpoint addressing */
        NVMF_RDMA_CMS_RDMA_CM      = 0x1,
};

struct nvmf_rdma_request_private_data {
	uint16_t		recfmt; /* record format */
	uint16_t		qid; /* queue id */
	uint16_t		hrqsize; /* host receive queue size */
	uint16_t		hsqsize; /* host send queue size */
	uint16_t		cntlid; /* controller id */
	uint8_t			reserved[22];
};

struct nvmf_rdma_accept_private_data {
	uint16_t		recfmt; /* record format */
	uint16_t		crqsize; /* controller receive queue size */
	uint8_t			reserved[28];
};

struct nvmf_rdma_reject_private_data {
	uint16_t		recfmt; /* record format */
	uint16_t		sts; /* status */
};

enum nvmf_rdma_transport_error {
	NVMF_RDMA_ERROR_INVALID_PRIVATE_DATA_LENGTH	= 0x1,
	NVMF_RDMA_ERROR_INVALID_RECFMT			= 0x2,
	NVMF_RDMA_ERROR_INVALID_QID			= 0x3,
	NVMF_RDMA_ERROR_INVALID_HSQSIZE			= 0x4,
	NVMF_RDMA_ERROR_INVALID_HRQSIZE			= 0x5,
	NVMF_RDMA_ERROR_NO_RESOURCES			= 0x6,
	NVMF_RDMA_ERROR_INVALID_IRD			= 0x7,
	NVMF_RDMA_ERROR_INVALID_ORD			= 0x8,
};

enum nvmf_rdma_offload {
    NVMF_RDMA_OFFLOAD_NONE   = 0x0,
    NVMD_RDMA_OFFLOAD_SNAPv3 = 0x2  /* snap (libsnap) emulator offload on BF1&BF2 */
};

struct nvmf_rdma_rsp {
    nvme_cqe_t                    cqe;
    struct ibv_mr                 *cqe_mr;
    struct ibv_sge                recv_sgl;
    struct ibv_recv_wr            recv_wr;
};

struct nvmf_rdma_req {
    nvmf_req_t                    nvmf_req;
    struct ibv_mr                 *cmd_mr;
    struct ibv_mr                 *payload_mr;
    struct ibv_send_wr            send_wr;
    struct ibv_sge                send_sgl;
    TAILQ_ENTRY(nvmf_rdma_req)    entry;
    nvmf_rdma_qpair_t             *queue;
};

/* NVMe RDMA qpair */
struct nvmf_rdma_qpair {
    nvmf_rdma_ctrl_t               *ctrl;
    struct rdma_event_channel      *cm_channel;
    struct rdma_cm_id              *cm_id;
    struct ibv_comp_channel        *comp_channel;
    struct ibv_cq                  *cq;
    pthread_t                      cq_thread;
    struct ibv_qp                  *qp;
    int                            qid;

    unsigned int                   cmnd_capsule_len;
    uint16_t                       queue_depth;

    pthread_mutex_t                state_lock;
    enum nvmf_rdma_queue_state     state;

    struct nvmf_rdma_req           *rdma_reqs;
    struct ibv_mr                  *rdma_reqs_mr;
    struct nvmf_rdma_rsp           *rdma_rsps;
    struct ibv_mr                  *rdma_rsps_mr;

    char                           *total_payload;
    struct ibv_mr                  *total_payload_mr;

    pthread_spinlock_t             reqs_lock;
    TAILQ_HEAD(, nvmf_rdma_req)    free_reqs;
    pthread_spinlock_t             free_reqs_lock;
    TAILQ_HEAD(, nvmf_rdma_req)    outstanding_reqs;
    pthread_spinlock_t             outstanding_reqs_lock;

    enum nvmf_rdma_offload         offload_type;
};

struct nvmf_rdma_ctrl {
    nvmf_emu_ctrl_t        ctrl;
    nvmf_rdma_qpair_t      *queues;
    struct ibv_pd          *pd;
};

static inline nvmf_rdma_ctrl_t *to_rdma_ctrl(nvmf_emu_ctrl_t *ctrl)
{
    return container_of(ctrl, nvmf_rdma_ctrl_t, ctrl);
}

static inline struct nvmf_rdma_req *to_rdma_req(nvmf_req_t *nvmf_req)
{
    return container_of(nvmf_req, struct nvmf_rdma_req, nvmf_req);
}

nvmf_emu_ctrl_t *nvmf_rdma_alloc_ctrl(nvme_emu_ep_t *ep);
void nvmf_rdma_free_ctrl(nvmf_emu_ctrl_t *ctrl);
#endif

