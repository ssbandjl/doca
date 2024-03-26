/*
 * NVMe emulation NVMe-oF controller
 */
#ifndef _NVMF_EMU_CTRL_H
#define _NVMF_EMU_CTRL_H

#include <pthread.h>
#include <semaphore.h>
#include <uuid/uuid.h>

#include "nvmf.h"
#include "nvme_emu.h"
#include "nvme_emu_ep.h"
#include "queue.h"

#define NVMF_SUPPORTED_ASYNC_EVENTS NVME_NAMESPACE_ATTR_NOTICE

typedef struct nvmf_emu_ctrl nvmf_emu_ctrl_t;

/*
 * State of nvmf_emu_ctrl_t (in particular, during initialization).
 */
enum nvmf_ctrl_state {
    NVMF_CTRL_DISCONNECTED = 0,
    NVMF_CTRL_CONNECTED = 1,
};


typedef struct nvmf_req {
    nvme_cmd_t                    cmd;
    uint8_t                       retries;
    bool                          timed_out;
    char                          *payload;
    uint32_t                      payload_size;
    int                           id;
    sem_t                         sync;
    bool                          is_sync;
    nvmf_completion_status_t      *status;
} nvmf_req_t;

typedef struct nvme_nsid_list {
    uint32_t ns_list[1024];
} nvme_nsid_list_t;

typedef struct nvme_ns {
    nvmf_emu_ctrl_t               *ctrl;
    uint8_t                       block_order;
    uint64_t                      size_b;
    nvme_id_ns_t                  ns_data;
    nvme_id_ns_descriptor_t       ns_descs_data[NVME_NS_ID_TYPES_NUM];

    /*
     * Size of data transferred as part of each block,
     * including metadata if FLBAS indicates the metadata is transferred
     * as part of the data buffer at the end of each LBA.
     */
    uint32_t                      md_size;
    uint32_t                      pi_type;
    uint32_t                      id;
    TAILQ_ENTRY(nvme_ns)          entry;
} nvme_ns_t;

/*
 * Namespaces list declaration.
 * TAILQ_HEAD macro actually creates "struct nvme_ns_list"
 * data structure, to be used as the list head.
 * To be able to use this struct more freely (e.g. move
 * pointers to that struct), nvme_ns_list_t is defined.
 */
typedef TAILQ_HEAD(nvme_ns_list, nvme_ns) nvme_ns_list_t;

struct nvmf_emu_queue_ops {
    int (*create)(nvmf_emu_ctrl_t *ctrl, uint16_t qid, uint16_t size);
    int (*destroy)(nvmf_emu_ctrl_t *ctrl, uint16_t qid);
    int (*activate)(nvmf_emu_ctrl_t *ctrl, uint16_t qid);
    void (*deactivate)(nvmf_emu_ctrl_t *ctrl, uint16_t qid);
};

struct nvmf_emu_cmd_ops {
    int (*send_cmd_sync)(nvmf_emu_ctrl_t *ctrl, uint16_t qid,
                         nvmf_req_t *nvmf_req, void *data, int data_len,
                         nvmf_completion_status_t *status);
    int (*send_cmd_async)(nvmf_emu_ctrl_t *ctrl, uint16_t qid,
                          nvmf_req_t *nvmf_req, void *data, int data_len);
    nvmf_req_t* (*get_request)(nvmf_emu_ctrl_t *ctrl, uint16_t qid, bool sync);
    void (*put_request)(nvmf_emu_ctrl_t *ctrl, uint16_t qid, nvmf_req_t *req);
};

struct nvmf_emu_ctrl {
    enum nvmf_ctrl_state        state;
    pthread_mutex_t             state_lock;
    TAILQ_ENTRY(nvmf_emu_ctrl)  entry;
    nvme_emu_ep_t               *ep;

    /* List of namespaces */
    nvme_ns_list_t              ns_list;
    pthread_mutex_t             ns_list_lock;
    bool                        pi_capable;
    bool                        md_configurable;
    /*
     * checks that need to be verified according to emulated device
     */
    uint16_t                    cntlid;
    int                         nr_queues;
    uint32_t                    max_io_queue_size;
    uint8_t                     mdts;
    union nvme_cc_register      cc;
    uint64_t                    cap;
    uint32_t                    vs;
    int                         page_sz;
    nvme_ctrl_id_t              cdata;

    struct nvmf_transport_id    id;
    char                        src_addr[NVMF_TRADDR_MAX_LEN + 1];
    char                        hostnqn[NVMF_NQN_MAX_LEN + 1];
    uuid_t                      hostid;

    int                         ka_timeout_ms;
    pthread_t                   kato_thread;
    sem_t                       ka_sem;

    sem_t                       fail_sem;

    struct nvmf_emu_queue_ops   q_ops;
    struct nvmf_emu_cmd_ops     cmd_ops;

    pthread_t                   aen_thread;
    sem_t                       aen_sem;
    nvme_aen_completion_t       aen_event;
};

typedef union nvme_log_page_t {
    nvme_error_log_t          error_log;
    nvme_smart_log_t          smart_log;
    nvme_fw_slot_info_log_t   fw_slot_info_log;
    nvme_changed_nslist_log_t changed_nslist_log;
} nvme_log_page_t;

int nvme_emu_ep_to_nvmf_ep(nvme_emu_ep_t *ep, struct nvmf_transport_id *id);
int nvmf_emu_create_ctrl(nvme_emu_ep_t *ep);
void nvmf_emu_delete_ctrl(nvmf_emu_ctrl_t *ctrl);
int nvmf_emu_start_ctrl(nvme_emu_ep_t *ep);
void nvmf_emu_stop_ctrl(nvme_emu_ep_t *ep, bool clear_namespaces);
void nvmf_emu_fail_ctrl(nvmf_emu_ctrl_t *ctrl);

int nvme_fabric_qpair_connect(struct nvmf_emu_ctrl *ctrl, uint16_t qid,
                              uint32_t num_entries);
int nvme_fabric_prop_get_cmd(struct nvmf_emu_ctrl *ctrl, uint32_t offset,
                             uint8_t size, void *value);
int nvme_fabric_prop_set_cmd(struct nvmf_emu_ctrl *ctrl, uint32_t offset,
                             uint8_t size, uint64_t value);
int nvmf_enable_ctrl(nvmf_emu_ctrl_t *ctrl, uint64_t cap);
void nvmf_disable_ctrl(nvmf_emu_ctrl_t *ctrl);

int nvmf_identify_ctrl(nvmf_emu_ctrl_t *ctrl);

int nvme_ctrl_submit_async_event_cmd(struct nvmf_emu_ctrl *ctrl);
int nvme_ctrl_get_log_page_cmd(struct nvmf_emu_ctrl *ctrl, nvme_log_page_t *log,
                               uint32_t nsid, uint8_t page_id, uint8_t lsp);

int nvme_identify_cmd(struct nvmf_emu_ctrl *ctrl, void *payload,
                      int payload_size, uint32_t nsid, uint8_t cns);

int nvme_ctrl_set_feature_cmd(struct nvmf_emu_ctrl *ctrl, uint8_t feature,
                              uint32_t dw11, uint32_t *res);
int nvme_ctrl_get_feature_cmd(struct nvmf_emu_ctrl *ctrl, uint8_t feature,
                              uint32_t *res);

#endif
