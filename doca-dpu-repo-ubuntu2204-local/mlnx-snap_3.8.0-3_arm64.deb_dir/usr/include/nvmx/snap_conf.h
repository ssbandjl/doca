#ifndef _SNAP_CONF_H
#define _SNAP_CONF_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <uuid/uuid.h>

#include "utils.h"
#include "nvme_emu.h"
#include "nvme_emu_ep.h"

/* Limits */
#define SNAP_CONFIG_MAX_MDTS 6

enum snap_config_emu_mode {
    SNAP_EMU_SQE_ONLY,  /* queue receives SQ entries */
    SNAP_EMU_SQE_CC,
    SNAP_EMU_CC_REMOTE,
    SNAP_EMU_MAX
};

static const char *emu_modes_str[] = { "sqe_only",
                                       "sqe_cc",
                                       "cc_remote" };

/* Defaults */
#define SNAP_CONFIG_DEFAULT_SN "MNC12"
#define SNAP_CONFIG_DEFAULT_MN "Mellanox BlueField NVMe SNAP Controller"
#define SNAP_CONFIG_DEFAULT_TYPE emu_modes_str[SNAP_EMU_SQE_ONLY]
#define SNAP_CONFIG_DEFAULT_MDTS_OTHER NVME_CTRL_MDTS_DEFAULT_OTHER
#define SNAP_CONFIG_DEFAULT_MDTS_CC_REMOTE NVME_CTRL_MDTS_DEFAULT_CC_REMOTE
#define SNAP_CONFIG_DEFAULT_NUM_QUEUES 32
#define SNAP_CONFIG_DEFAULT_SQES 0x6
#define SNAP_CONFIG_DEFAULT_CQES 0x4
#define SNAP_CONFIG_DEFAULT_QUIRKS 0
#define SNAP_CONFIG_DEFAULT_MAX_NAMESPACES SNAP_CONFIG_DEFAULT_MNAN
#define SNAP_CONFIG_DEFAULT_ONCS 0
#define SNAP_CONFIG_DEFAULT_CQ_PERIOD 3
#define SNAP_CONFIG_DEFAULT_CQ_MAX_COUNT 6
#define SNAP_CONFIG_DEFAULT_PATH_KA_TO_MS 15000
#define SNAP_CONFIG_RPC_SERVER_SPDK "/var/tmp/spdk.sock"
#define SNAP_CONFIG_RPC_SERVER_NONE ""
#define SNAP_CONFIG_DEFAULT_RPC_SERVER SNAP_CONFIG_RPC_SERVER_NONE
#define SNAP_CONFIG_DEFAULT_ZCOPY_ALIGN 512
#define SNAP_CONFIG_DEFAULT_MEM "static"
#define SNAP_CONFIG_DEFAULT_NN 1024
#define SNAP_CONFIG_DEFAULT_MNAN 1024

#define MAX_SN_LEN 20
#define MAX_MN_LEN 40
#define MAX_RDMA_DEV_LEN 32
#define MAX_PCI_FUNC_LEN 16
#define MAX_RPC_SERVER_LEN 128
#define MAX_VER_LEN 8
#define MAX_MEM_LEN 7

#define SNAP_CONFIG_ALLOWED_QUIRKS (NVME_QUIRK_ASYNC_EV_CONFIG | \
                                    NVME_QUIRK_NS_CHANGE_FORCE_UNMASK | \
                                    NVME_QUIRK_FORCE_OACS_NS_MGMT)

typedef struct backend_path {
    char  addr[NVME_EMU_ADDR_MAX_LEN + 1];
    char  nqn[NVME_EMU_MAX_NQN_LEN + 1];
    int   ka_timeout_ms;
    int   port;
} snap_config_be_path_t;

typedef struct backend {
    char id[NVME_EMU_MAX_NAME_LEN + 1];
    char name[NVME_EMU_MAX_NAME_LEN + 1];
    char be_type[NVME_EMU_MAX_EP_TYPE_LEN + 1];
    uuid_t hostid;
    uint64_t size_b;
    uint8_t block_order;
    int  paths_cnt;
    snap_config_be_path_t *paths;
} snap_config_backend_t;

struct nvme_subsystem;

typedef struct snap_conf {
    int pci_index;
    int vf_pci_index;
    char pci_bdf[MAX_PCI_FUNC_LEN + 1];
    uint16_t cntlid;
    uint32_t num_queues;
    uint8_t mdts;
    char sn[MAX_SN_LEN];
    char mn[MAX_MN_LEN];
    uint32_t nn;
    uint32_t mnan;
    enum snap_config_emu_mode emu_mode;
    enum nvme_quirks quirks;
    uint32_t max_namespaces;
    uint16_t oncs;
    uint8_t sqes;
    uint8_t cqes;
    uint32_t zcopy_align;
    int cq_period;
    int cq_max_count;
    int backends_cnt;
    snap_config_backend_t *backends;
    char emu_manager[MAX_RDMA_DEV_LEN];
    char rdma_dev[MAX_RDMA_DEV_LEN];
    uint32_t version;
    bool iova_mgmt;
    char mem[MAX_MEM_LEN];
    char rpc_server[MAX_RPC_SERVER_LEN];
    struct {
        struct nvme_subsystem *subsys;
        /* hack to allow getting nqn in the parts of code
         * that don't depend on spdk
         */
        char *subnqn;
    } spdk;
} snap_conf_t;

snap_conf_t *snap_config_create(const char *json_file);
void snap_config_destroy(snap_conf_t *config);

/*Getters*/
static inline const char *snap_config_get_sn(snap_conf_t *config)
{
    return config->sn;
}

static inline const char *snap_config_get_mn(snap_conf_t *config)
{
    return config->mn;
}

static inline uint32_t snap_config_get_nn(snap_conf_t *config)
{
    return config->nn;
}

static inline uint32_t snap_config_get_mnan(snap_conf_t *config)
{
    return config->mnan;
}

static inline enum
snap_config_emu_mode snap_config_get_emu_mode(snap_conf_t *config)
{
    return config->emu_mode;
}

static inline const char
*snap_config_get_emu_mode_str(snap_conf_t *config)
{
    return emu_modes_str[config->emu_mode];
}

static inline uint8_t
snap_config_get_mdts(snap_conf_t *config)
{
    return config->mdts;
}

static inline uint32_t
snap_config_get_num_queues(snap_conf_t *config)
{
    return config->num_queues;
}

static inline int
snap_config_get_pci_func(snap_conf_t *config)
{
    return config->pci_index;
}

static inline int
snap_config_get_pci_vfunc(snap_conf_t *config)
{
    return config->vf_pci_index;
}

static inline const char *
snap_config_get_pci_bdf(snap_conf_t *config)
{
    return config->pci_bdf;
}

static inline int snap_config_get_cq_period(snap_conf_t *config)
{
    return config->cq_period;
}

static inline int snap_config_get_cq_max_count(snap_conf_t *config)
{
    return config->cq_max_count;
}

static inline uint16_t snap_config_get_oncs(snap_conf_t *config)
{
    return config->oncs;
}

static inline uint32_t snap_config_get_max_namespaces(snap_conf_t *config)
{
    return config->max_namespaces;
}

static inline enum
nvme_quirks snap_config_get_quirks(snap_conf_t *config)
{
    return config->quirks;
}

static inline uint8_t snap_config_get_sqes(snap_conf_t *config)
{
    return config->sqes;
}

static inline uint8_t snap_config_get_cqes(snap_conf_t *config)
{
    return config->cqes;
}

const snap_config_backend_t *snap_config_get_backend(snap_conf_t *config,
                                                     int id);

static inline
int snap_config_get_backends_cnt(snap_conf_t *config)
{
    return config->backends_cnt;
}

static inline
uint64_t snap_config_get_backend_size_b(const snap_config_backend_t *backend)
{
    return backend->size_b;
}

static inline
uint8_t snap_config_get_backend_block_order(const snap_config_backend_t *backend)
{
    return backend->block_order;
}

static inline
void snap_config_get_backend_hostid(const snap_config_backend_t *backend,
                                    uuid_t hostid)
{
    uuid_copy(hostid, backend->hostid);
}

static inline
int snap_config_get_paths_cnt(const snap_config_backend_t *backend)
{
    return backend->paths_cnt;
}

static inline const snap_config_be_path_t
*snap_config_get_be_path(const snap_config_backend_t *backend,
                        int path_id)
{
    return &(backend)->paths[path_id];
}

const snap_config_be_path_t
*snap_config_get_path(snap_conf_t *config, int be_id, int path_id);
bool snap_config_get_offload(snap_conf_t *config);

static inline uint16_t snap_config_get_cntlid(snap_conf_t *config)
{
    return config->cntlid;
}

static inline const char *snap_config_get_emu_manager(snap_conf_t *config)
{
    return config->emu_manager;
}

static inline const char *snap_config_get_rdma_dev(snap_conf_t *config)
{
    return config->rdma_dev;
}

static inline const char *snap_config_get_rpc_server(snap_conf_t *config)
{
    return config->rpc_server;
}

static inline const char *snap_config_get_subnqn(snap_conf_t *config)
{
    if (!config->spdk.subsys)
        return NULL;

    return config->spdk.subnqn;
}

static inline uint32_t snap_config_get_zcopy_align(snap_conf_t *config)
{
    return config->zcopy_align;
}

static inline const char *snap_config_get_mem(snap_conf_t *config)
{
    return config->mem;
}

static inline const bool snap_config_get_iova_mgmt(snap_conf_t *config)
{
    return config->iova_mgmt;
}

/*Setters*/
int snap_config_set_sn(snap_conf_t *config, const char *sn);
int snap_config_set_mn(snap_conf_t *config, const char *mn);
void snap_config_set_nn(snap_conf_t *config, uint32_t nn);
void snap_config_set_mnan(snap_conf_t *config, uint32_t mnan);
int snap_config_set_emu_mode(snap_conf_t *config, const char *type);
void snap_config_set_mdts(snap_conf_t *config, uint32_t mdts);
int snap_config_set_num_queues(snap_conf_t *config, uint32_t num_q);
void snap_config_set_pci_func(snap_conf_t *config, int index);
void snap_config_set_pci_vfunc(snap_conf_t *config, int index);
void snap_config_set_pci_bdf(snap_conf_t *config, const char *bdf);
void snap_config_set_cq_period(snap_conf_t *config, uint32_t cq_period);
void snap_config_set_cq_max_count(snap_conf_t *config, uint32_t cq_max_count);
int snap_config_set_oncs(snap_conf_t *config, uint32_t oncs);
int snap_config_set_max_namespaces(snap_conf_t *config, uint32_t max_ns);
int snap_config_set_quirks(snap_conf_t *config,
                           enum nvme_quirks quirks);
void snap_config_set_sqes(snap_conf_t *config, uint32_t sqes);
void snap_config_set_cqes(snap_conf_t *config, uint32_t cqes);
void snap_config_set_cntlid(snap_conf_t *config, uint16_t cntlid);
int snap_config_set_emu_manager(snap_conf_t *config, const char *emu_manager);
int snap_config_set_rdma_dev(snap_conf_t *config, const char *rdma_dev);
int snap_config_set_mem(snap_conf_t *config, const char *mem);
int snap_config_set_rpc_server(snap_conf_t *config, const char *rpc_server);
void snap_config_set_zcopy_align(snap_conf_t *config, uint32_t zcopy_align);
void snap_config_set_iova_mgmt(snap_conf_t *config, bool iova_mgmt);

#endif
