#ifndef _NVME_SPDK_SUBSYSTEM_H
#define _NVME_SPDK_SUBSYSTEM_H
#include <spdk/nvme_spec.h>
#include "nvme.h"
#include "nvme_emu.h"
#include "queue.h"
#ifdef HAVE_RTE_HASH_H
#include <rte_hash.h>
#endif

#define NVME_SUBSYS_NS_HASH_SIZE 1023
#define NVME_SUBSYS_MAX_CONTROLLERS 1024
#define NVME_MAX_SUBSYSTEMS 1024

#ifndef SPDK_NVME_NQN_FIELD_SIZE
#define SPDK_NVME_NQN_FIELD_SIZE 256
#endif

typedef void (*nvme_spdk_delete_cb_t)(void *);

typedef struct nvme_spdk_ctrl nvme_spdk_ctrl_t;
struct snap_conf;
struct nvme_spdk_namespace_io_context;

typedef struct nvme_spdk_namespace {
    struct spdk_bdev *bdev;
    const char *bdev_name;
    struct spdk_bdev_desc *bdev_desc;
    struct nvme_spdk_namespace_io_context *io_contexts;
    size_t num_io_contexts;

    nvme_spdk_ctrl_t *ctrl;
    bool zcopy;
    bool tcp_rx_zcopy;
    bool tcp_tx_zcopy;
    bool effdma;
    uint32_t nsid;
    uint32_t block_size;
    uint64_t num_blocks;
    uint32_t md_size;
    nvme_id_ns_descriptor_t ns_desc[4];

    TAILQ_ENTRY(nvme_spdk_namespace) entry;
#ifndef HAVE_RTE_HASH_H
    LIST_ENTRY(nvme_spdk_namespace) nvme_subsys_entry;
#endif

    char qn[SPDK_NVME_NQN_FIELD_SIZE + 1];
    char protocol[32];
    char bdev_type[32];

    bool in_deletion;
    nvme_spdk_delete_cb_t deleter_cb;
    void *deleter_cb_arg;
} nvme_spdk_namespace_t;

struct nvme_subsystem {
    char subnqn[SPDK_NVME_NQN_FIELD_SIZE + 1];
    char sn[SPDK_NVME_CTRLR_SN_LEN + 1];
    char mn[SPDK_NVME_CTRLR_MN_LEN + 1];
    uint16_t subsysid;
    uint32_t nn;
    uint32_t mnan;
    uint32_t num_ns;

    bool taken_cntlids[NVME_SUBSYS_MAX_CONTROLLERS];
#ifdef HAVE_RTE_HASH_H
    struct rte_hash *ns_hash;
#else
    pthread_mutex_t ns_list_lock;
    LIST_HEAD(, nvme_spdk_namespace) ns_list;
#endif
    LIST_HEAD(, spdk_emu_ctx) ctrl_list;

    LIST_ENTRY(nvme_subsystem) entry;
};

struct spdk_emu_controller_nvme_create_attr {
    char *nqn;
    char *emu_manager;
    int   pf_id;
    int   vf_id;
    char *pci_bdf;
    char *conf_file;
    int   nr_io_queues;
    int   mdts;
    int   max_namespaces;
    int   quirks;
    char *rdma_device;
    struct snap_conf *sconfig;
    int subsys_id;
    char *mem;
    int cntlid;
    int iova_mgmt;
    char *vuid;
};

struct spdk_emu_controller_nvme_delete_attr {
    char *name;
    char *subnqn;
    int cntlid;
    char *vuid;
};

struct spdk_emu_controller_nvme_get_iostat_attr {
    char *name;
};

struct spdk_emu_controller_nvme_get_debugstat_attr {
    char *name;
    bool fw_counters;
};

bool nvme_subsys_ns_exists(struct nvme_subsystem *subsys,
                           uint32_t nsid);

int nvme_subsys_ns_list_allocated_nsids(struct nvme_subsystem *subsys,
                                        uint32_t *ns_list, size_t ns_list_sz,
                                        uint32_t min_nsid);

int nvme_subsys_ns_add(struct nvme_subsystem *subsys,
                       nvme_spdk_namespace_t *ns);

void nvme_subsys_ns_del(struct nvme_subsystem *subsys,
                        nvme_spdk_namespace_t *ns);

void nvme_subsys_ctrl_del(struct spdk_emu_ctx *ctx);

void nvme_ctrl_config_destroy(struct snap_conf *sconfig);

struct nvme_subsystem *subsystem_nvme_find(const char *nqn);
#endif
