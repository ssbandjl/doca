#ifndef _NVME_H
#define _NVME_H

#include "compiler.h"
#include <uuid/uuid.h>

#define NVME_VERSION(mjr, mnr, ter) \
	(((uint32_t)(mjr) << 16) | \
	((uint32_t)(mnr) << 8) | \
	(uint32_t)(ter))

#define NVME_NAMESPACE_GLOBAL 0xFFFFFFFF
#define NVME_EFFECTS_LOG_IO_OFF 256
#define NVME_EMU_MAX_NQN_LEN 223

/* Same limit like at Linux kernel pci driver */
#define SGL_MAX_DESCRIPTORS 128

enum nvme_quirks {
    NVME_QUIRK_ASYNC_EV_CONFIG        = 1 << 0,
    NVME_QUIRK_NS_CHANGE_FORCE_UNMASK = 1 << 1,
    NVME_QUIRK_RESERVED               = 1 << 2,
    NVME_QUIRK_FORCE_OACS_NS_MGMT     = 1 << 3,
};

enum NvmeCstsShift {
    CSTS_RDY_SHIFT      = 0,
    CSTS_CFS_SHIFT      = 1,
    CSTS_SHST_SHIFT     = 2,
    CSTS_NSSRO_SHIFT    = 4,
    CSTS_PP_SHIFT       = 5,
};

enum NvmeCstsMask {
    CSTS_RDY_MASK   = 0x1,
    CSTS_CFS_MASK   = 0x1,
    CSTS_SHST_MASK  = 0x3,
    CSTS_NSSRO_MASK = 0x1,
    CSTS_PP_MASK    = 0x1,
};

enum NvmeCsts {
    NVME_CSTS_READY         = 1 << CSTS_RDY_SHIFT,
    NVME_CSTS_FAILED        = 1 << CSTS_CFS_SHIFT,
    NVME_CSTS_SHST_NORMAL   = 0 << CSTS_SHST_SHIFT,
    NVME_CSTS_SHST_PROGRESS = 1 << CSTS_SHST_SHIFT,
    NVME_CSTS_SHST_COMPLETE = 2 << CSTS_SHST_SHIFT,
    NVME_CSTS_NSSRO         = 1 << CSTS_NSSRO_SHIFT,
};

enum nvme_psdt_value {
    NVME_PSDT_PRP              = 0x0,
    NVME_PSDT_SGL_MPTR_CONTIG  = 0x1,
    NVME_PSDT_SGL_MPTR_SGL     = 0x2,
    NVME_PSDT_RESERVED         = 0x3
};

enum nvme_sgls_support_mask {
    NVME_SGLS_NOT_SUPPORTED         = 0x00,
    NVME_SGLS_SUPPORTED_NO_REQS     = 0x01,
    NVME_SGLS_SUPPORTED_WITH_REQS   = 0x02,
    NVME_SGLS_LARGE_SGL             = 0x40000,
};

enum nvme_ocas_bits {
    NVME_OACS_SEC_SEND_RECV         = 0x0001,
    NVME_OACS_FORMAT                = 0x0002,
    NVME_OACS_FW_COMMIT_DOWNLOAD    = 0x0004,
    NVME_OACS_NS_MGMT               = 0x0008,
    NVME_OACS_SELF_TEST             = 0x0010,
    NVME_OACS_DIRECTIVES            = 0x0020,
    NVME_OACS_MI_SEND_RECV          = 0x0040,
    NVME_OACS_VIRT_MGMT             = 0x0080,
    NVME_OACS_DOORBELL_CONFIG       = 0x0100,
    NVME_OACS_GET_LBA_CAP           = 0x0200
};

enum nvme_commit_action {
    COMMIT_ACTION_DOWNLOAD                  = 0x00,
    COMMIT_ACTION_DOWNLOAD_AND_ACTIVATE     = 0x01,
    COMMIT_ACTION_ACTIVATE                  = 0x02,
    COMMIT_ACTION_DOWNLOAD_AND_ACTIVATE_NOW = 0x03,
    COMMIT_ACTION_REPLACE_BOOTP             = 0x06,
    COMMIT_ACTION_ACTIVATE_BOOTP            = 0x07
};

typedef struct NVME_PACKED nvme_lbaf {
    uint16_t    ms;
    uint8_t     ds;
    uint8_t     rp;
} nvme_lbaf_t;

#define NVME_ID_NS_FLBAS_EXTENDED(flbas)    ((flbas >> 4) & 0x1)
#define NVME_ID_NS_FLBAS_INDEX(flbas)       ((flbas & 0xf))
#define NVME_ID_NS_DPS_TYPE_MASK            0x7

typedef struct NVME_PACKED nvme_id_ns_t {
    uint64_t    nsze;
    uint64_t    ncap;
    uint64_t    nuse;
    uint8_t     nsfeat;
    uint8_t     nlbaf;
    uint8_t     flbas;
    uint8_t     mc;
    uint8_t     dpc;
    uint8_t     dps;
    uint8_t     res30[74];
    uint8_t     nguid[16];
    uint8_t     eui64[8];
    nvme_lbaf_t lbaf[16];
    uint8_t     res192[192];
    uint8_t     vs[3712];
} nvme_id_ns_t;

enum nvme_sgl_descriptor_type {
    NVME_SGL_TYPE_DATA_BLOCK           = 0x0,
    NVME_SGL_TYPE_BIT_BUCKET           = 0x1,
    NVME_SGL_TYPE_SEGMENT              = 0x2,
    NVME_SGL_TYPE_LAST_SEGMENT         = 0x3,
    NVME_SGL_TYPE_KEYED_DATA_BLOCK     = 0x4,
    NVME_SGL_TYPE_TRANSPORT_DATA_BLOCK = 0x5,
    /* 0x6 - 0xE reserved */
    NVME_SGL_TYPE_VENDOR_SPECIFIC      = 0xF
};

enum nvme_sgl_descriptor_subtype {
    NVME_SGL_SUBTYPE_ADDRESS           = 0x0,
    NVME_SGL_SUBTYPE_OFFSET            = 0x1,
};

typedef struct NVME_PACKED nvme_sgl_datablock_desc {
    uint64_t    address;
    uint32_t    length;
} nvme_sgl_datablock_desc_t;

typedef struct NVME_PACKED nvme_keyed_sgl_datablock_desc {
    uint64_t    address;
    uint32_t    length:24;
    uint32_t    key;
} nvme_keyed_sgl_datablock_desc_t;

typedef struct NVME_PACKED nvme_sgl_segment_desc {
    uint64_t    address;
    uint32_t    length;
} nvme_sgl_segment_desc_t;

typedef struct NVME_PACKED nvme_sgl_last_segment_desc {
    uint64_t    address;
    uint32_t    length;
} nvme_sgl_last_segment_desc_t;

typedef struct NVME_PACKED nvme_sgl_bit_bucket_desc {
    uint32_t    :32;
    uint32_t    :32;
    uint32_t    length;
} nvme_sgl_bit_bucket_desc_t;

typedef struct NVME_PACKED nvme_sgl_desc {

    union {
        uint8_t type_specific[15];
        nvme_sgl_datablock_desc_t data_block;
        nvme_keyed_sgl_datablock_desc_t keyed_data_block;
        nvme_sgl_segment_desc_t segment;
        nvme_sgl_last_segment_desc_t last_segment;
        nvme_sgl_bit_bucket_desc_t bit_bucket;
    };

    uint8_t subtype : 4;
    uint8_t type : 4;
} nvme_sgl_desc_t;

union nvme_data_ptr {
    struct {
        uint64_t prp1;
        uint64_t prp2;
    };
    nvme_sgl_desc_t sgl;
};

typedef struct NVME_PACKED nvme_psd {
    uint16_t    mp;
    uint16_t    reserved;
    uint32_t    enlat;
    uint32_t    exlat;
    uint8_t     rrt;
    uint8_t     rrl;
    uint8_t     rwt;
    uint8_t     rwl;
    uint8_t     resv[16];
} nvme_psd_t;

typedef struct NVME_PACKED nvme_ctrl_id {
    uint16_t vid;
    uint16_t ssvid;
    uint8_t sn[20];
    uint8_t mn[40];
    uint8_t fr[8];
    uint8_t rab;
    uint8_t ieee[3];
    struct {
        uint8_t multi_port : 1;
        uint8_t multi_ctrl : 1;
        uint8_t sr_iov : 1;
        uint8_t ana_reporting : 1;
        uint8_t reserved : 4;
    } cmic;
    uint8_t mdts;
    uint16_t cntlid;
    union {
        uint32_t raw;
        struct {
            uint32_t ter : 8;
            uint32_t mnr : 8;
            uint32_t mjr : 16;
        } bits;
    } ver;
    uint32_t rtd3r;
    uint32_t rtd3e;
    union {
        uint32_t raw;
        struct {
            uint32_t : 8;
            uint32_t ns_attribute_notices : 1;
            uint32_t fw_activation_notices : 1;
            uint32_t : 1;
            uint32_t ana_change_notices : 1;
            uint32_t : 19;
            uint32_t discovery_log_change_notices : 1;
        } bits;
    } oaes;
    struct {
        uint32_t host_id_exhid_supported: 1;
        uint32_t non_operational_power_state_permissive_mode: 1;
        uint32_t : 30;
    } ctratt;
    uint8_t rsvd100[11];
    uint8_t cntrltype;
    uint8_t fguid[16];
    uint16_t crdt[3];
    uint8_t rsvd112[122];
    struct {
        uint16_t security : 1;
        uint16_t format : 1;
        uint16_t firmware : 1;
        uint16_t ns_manage : 1;
        uint16_t device_self_test : 1;
        uint16_t directives : 1;
        uint16_t nvme_mi : 1;
        uint16_t virtualization_management : 1;
        uint16_t doorbell_buffer_config : 1;
        uint16_t get_lba_status : 1;
        uint16_t : 6;
    } oacs;
    uint8_t acl;
    uint8_t aerl;
    struct {
        uint8_t slot1_ro : 1;
        uint8_t num_slots : 3;
        uint8_t activation_without_reset : 1;
        uint8_t : 3;
    } frmw;
    struct {
        uint8_t ns_smart : 1;
        uint8_t celp : 1;
        uint8_t edlp : 1;
        uint8_t telemetry : 1;
        uint8_t : 4;
    } lpa;
    uint8_t elpe;
    uint8_t npss;
    struct {
        uint8_t spec_format : 1;
        uint8_t : 7;
    } avscc;
    uint8_t apsta;
    uint16_t wctemp;
    uint16_t cctemp;
    uint16_t mtfa;
    uint32_t hmpre;
    uint32_t hmmin;
    uint64_t tnvmcap[2];
    uint64_t unvmcap[2];
    struct {
        uint32_t num_rpmb_units : 3;
        uint32_t auth_method : 3;
        uint32_t : 2;
        uint32_t : 8;
        uint32_t total_size : 8;
        uint32_t access_size : 8;
    } rpmbs;
    uint16_t edstt;
    struct {
        uint8_t one_only : 1;
        uint8_t : 7;
    } dsto;
    uint8_t fwug;
    uint16_t kas;
    uint8_t rsvd511[190];
    struct {
        uint8_t min : 4;
        uint8_t max : 4;
    } sqes;
    struct {
        uint8_t min : 4;
        uint8_t max : 4;
    } cqes;
    uint16_t maxcmd;
    uint32_t nn;
    union {
        uint16_t raw;
        struct {
            uint16_t compare : 1;
            uint16_t write_unc : 1;
            uint16_t dsm: 1;
            uint16_t write_zeroes: 1;
            uint16_t set_features_save: 1;
            uint16_t reservations: 1;
            uint16_t timestamp: 1;
            uint16_t verify: 1;
            uint16_t copy: 1;
            uint16_t : 7;
        } bits;
    } oncs;
    struct {
        uint16_t compare_and_write : 1;
        uint16_t : 15;
    } fuses;
    struct {
        uint8_t format_all_ns: 1;
        uint8_t erase_all_ns: 1;
        uint8_t crypto_erase_supported: 1;
        uint8_t : 5;
    } fna;
    struct {
        uint8_t present : 1;
        uint8_t flush_broadcast : 2;
        uint8_t : 5;
    } vwc;
    uint16_t awun;
    uint16_t awupf;
    uint8_t nvscc;
    uint8_t rsvd704;
    uint16_t acwu;
    uint16_t ocfs;

    uint32_t sgls;
    uint32_t mnan;

    uint8_t rsvd712[224];
    uint8_t subnqn[256];
    uint8_t rsvd2047[1024];
    nvme_psd_t psd[32];
    struct {
#define NVME_CTRL_JSON_RPC_2_0_MJR 2
        uint16_t json_rpc_2_0_mjr;
#define NVME_CTRL_JSON_RPC_2_0_MNR 0
        uint16_t json_rpc_2_0_mnr;
#define NVME_CTRL_JSON_RPC_2_0_TER 0
        uint16_t json_rpc_2_0_ter;
#define NVME_CTRL_IOVA_MAP_UNMAP_RANGE 0x0001U
#define NVME_CTRL_IOVA_UNMAP_ALL 0x0002U
        uint16_t iova_mmts;
        uint8_t  iova_mms;
        uint8_t rsvd[1015];
    } vs;
} nvme_ctrl_id_t;

#define NVME_CTRL_MDTS_DEFAULT_OTHER        1
#define NVME_CTRL_MDTS_DEFAULT_CC_REMOTE    4
#define NVME_CRTL_MDTS_MAX                  6

#define NVME_CQE_STATUS_IS_ERROR(status) (le16_to_cpu(status) >> 1)

typedef struct NVME_PACKED nvme_cqe {
    uint32_t    result;
    uint32_t    rsvd;
    uint16_t    sq_head;
    uint16_t    sq_id;
    uint16_t    cid;
    uint16_t    status;
} nvme_cqe_t;


typedef struct NVME_PACKED nvme_cmd_common {
    uint8_t opc : 8;
    uint8_t fuse : 2;
    uint8_t : 4;
    uint8_t psdt : 2;
    uint16_t cid;
    uint32_t nsid;
    uint64_t rsvd;
    uint64_t mptr;
    union nvme_data_ptr dptr;
} nvme_cmd_common_t;

typedef struct NVME_PACKED nvme_cmd {
    nvme_cmd_common_t common;
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} nvme_cmd_t;

typedef struct NVME_PACKED nvme_vendor_specific_cmd {
    nvme_cmd_common_t common;
    uint32_t    ndt;
    uint32_t    ndm;
    uint32_t    cdw12;
    uint32_t    cdw13;
    uint32_t    cdw14;
    uint32_t    cdw15;
} nvme_vendor_specific_cmd_t;

typedef struct NVME_PACKED nvme_cmd_abort {
    nvme_cmd_common_t common;
    uint16_t sqid;
    uint16_t cid;
    uint32_t rsvd40[5];
} nvme_cmd_abort_t;

enum NvmeAbortResult {
    NVME_ABORT_SUCCEEDED = 0x00,
    NVME_ABORT_FAILED    = 0x01
};

typedef struct NVME_PACKED nvme_cmd_get_feature {
    nvme_cmd_common_t common;
    uint32_t fid : 8;
    uint32_t sel : 3;
    uint32_t : 21;
    uint32_t    dw11;
    uint32_t    dw12;
    uint32_t    dw13;
    uint32_t    dw14;
    uint32_t    dw15;
} nvme_cmd_get_feature_t;

typedef struct NVME_PACKED nvme_cmd_set_feature {
    nvme_cmd_common_t common;
    uint32_t fid : 8;
    uint32_t : 23;
    uint32_t sv : 1;
    uint32_t    dw11;
    uint32_t    dw12;
    uint32_t    dw13;
    uint32_t    dw14;
    uint32_t    dw15;
} nvme_cmd_set_feature_t;

enum NvmeFeatureIds {
    NVME_ARBITRATION                        = 0x1,
    NVME_POWER_MANAGEMENT                   = 0x2,
    NVME_LBA_RANGE_TYPE                     = 0x3,
    NVME_TEMPERATURE_THRESHOLD              = 0x4,
    NVME_ERROR_RECOVERY                     = 0x5,
    NVME_VOLATILE_WRITE_CACHE               = 0x6,
    NVME_NUMBER_OF_QUEUES                   = 0x7,
    NVME_INTERRUPT_COALESCING               = 0x8,
    NVME_INTERRUPT_VECTOR_CONF              = 0x9,
    NVME_WRITE_ATOMICITY                    = 0xa,
    NVME_ASYNCHRONOUS_EVENT_CONF            = 0xb,
    NVME_AUTONOMOUS_POWER_STATE_TRANSIT     = 0xc,
    NVME_HOST_MEM_BUFFER                    = 0xd,
    NVME_TIMESTAMP                          = 0xe,
    NVME_KEEP_ALIVE_TIMER                   = 0xf,
    NVME_HOST_CONTROLLED_THERMAL_MANAGEMENT = 0x10,
    NVME_SOFTWARE_PROGRESS_MARKER           = 0x80,
    NVME_HOST_IDENTIFIER                    = 0x81
};

enum nvme_feature_select {
    NVME_FEATURE_CURRENT = 0x0,
    NVME_FEATURE_DEFAULT = 0x1,
    NVME_FEATURE_SAVED   = 0x2,
    NVME_FEATURE_CAPS    = 0x3,
};

enum NvmeFeatureCaps {
    NVME_FEATURE_CHANGEABLE = (1 << 2)
};

#define NVME_FTR_PM_PSTATE(dw11)    ((dw11) & 0xF)

#define NVME_FTR_TEMP_THRESH(dw11)    ((dw11) & 0xFFFF)

#define NVME_FTR_AGGR_TIME_SHIFT     8
#define NVME_FTR_AGGR_THRESH(dw11)  ((dw11) & 0xFF)
#define NVME_FTR_AGGR_TIME(dw11)    (((dw11) >> NVME_FTR_AGGR_TIME_SHIFT) & 0xFF)

enum nvme_admin_command {
    NVME_ADMIN_DELETE_SQ      = 0x00,
    NVME_ADMIN_CREATE_SQ      = 0x01,
    NVME_ADMIN_GET_LOG_PAGE   = 0x02,
    NVME_ADMIN_DELETE_CQ      = 0x04,
    NVME_ADMIN_CREATE_CQ      = 0x05,
    NVME_ADMIN_IDENTIFY       = 0x06,
    NVME_ADMIN_ABORT          = 0x08,
    NVME_ADMIN_SET_FEATURES   = 0x09,
    NVME_ADMIN_GET_FEATURES   = 0x0a,
    NVME_ADMIN_ASYNC_EV_REQ   = 0x0c,
    NVME_ADMIN_FW_COMMIT      = 0x10,
    NVME_ADMIN_FW_DOWNLOAD    = 0x11,
    NVME_ADMIN_KEEP_ALIVE     = 0x18,
    NVME_ADMIN_FORMAT_NVM     = 0x80,
    NVME_ADMIN_SECURITY_SEND  = 0x81,
    NVME_ADMIN_SECURITY_RECV  = 0x82,

    /* Mellanox vendor specific commands */
    NVME_ADMIN_FW_RECOVER           = 0xc0,
    NVME_ADMIN_VS_JSON_RPC_2_0_REQ  = 0xc1,
    NVME_ADMIN_VS_JSON_RPC_2_0_RSP  = 0xc2,
    NVME_ADMIN_VS_IOVA_MGMT         = 0xc4,
};

enum nvme_io_command {
    NVME_IO_FLUSH              = 0x00,
    NVME_IO_WRITE              = 0x01,
    NVME_IO_READ               = 0x02,
    NVME_IO_WRITE_UNCOR        = 0x04,
    NVME_IO_COMPARE            = 0x05,
    NVME_IO_WRITE_ZEROS        = 0x08,
    NVME_IO_DSM                = 0x09,
};

enum nvme_status_code {
    NVME_SC_SUCCESS                = 0x0000,
    NVME_SC_INVALID_OPCODE         = 0x0001,
    NVME_SC_INVALID_FIELD          = 0x0002,
    NVME_SC_CID_CONFLICT           = 0x0003,
    NVME_SC_DATA_TRAS_ERROR        = 0x0004,
    NVME_SC_POWER_LOSS_ABORT       = 0x0005,
    NVME_SC_INTERNAL_DEV_ERROR     = 0x0006,
    NVME_SC_CMD_ABORT_REQ          = 0x0007,
    NVME_SC_CMD_ABORT_SQ_DEL       = 0x0008,
    NVME_SC_CMD_ABORT_FAILED_FUSE  = 0x0009,
    NVME_SC_CMD_ABORT_MISSING_FUSE = 0x000a,
    NVME_SC_INVALID_NSID           = 0x000b,
    NVME_SC_CMD_SEQ_ERROR          = 0x000c,
    NVME_SC_SGL_INVALID_SEGMENT_DESCRIPTOR = 0x000d,
    NVME_SC_SGL_INVALID_DESCRIPTORS_NUMBER = 0x000e,
    NVME_SC_SGL_INVALID_DATA_LENGTH        = 0x000f,
    NVME_SC_SGL_INVALID_DESCRIPTOR_TYPE    = 0x0011,
    NVME_SC_PRP_OFFSET_INVALID     = 0x0013,
    NVME_SC_LBA_RANGE              = 0x0080,
    NVME_SC_CAP_EXCEEDED           = 0x0081,
    NVME_SC_NS_NOT_READY           = 0x0082,
    NVME_SC_NS_RESV_CONFLICT       = 0x0083,
    NVME_SC_INVALID_CQID           = 0x0100,
    NVME_SC_INVALID_QID            = 0x0101,
    NVME_SC_MAX_QSIZE_EXCEEDED     = 0x0102,
    NVME_SC_ACL_EXCEEDED           = 0x0103,
    NVME_SC_RESERVED               = 0x0104,
    NVME_SC_AER_LIMIT_EXCEEDED     = 0x0105,
    NVME_SC_INVALID_FW_SLOT        = 0x0106,
    NVME_SC_INVALID_FW_IMAGE       = 0x0107,
    NVME_SC_INVALID_IRQ_VECTOR     = 0x0108,
    NVME_SC_INVALID_LOG_ID         = 0x0109,
    NVME_SC_INVALID_FORMAT         = 0x010a,
    NVME_SC_FW_REQ_CONV_RESET      = 0x010b,
    NVME_SC_INVALID_QUEUE_DEL      = 0x010c,
    NVME_SC_FID_NOT_SAVEABLE       = 0x010d,
    NVME_SC_FID_NOT_NSID_SPEC      = 0x010f,
    NVME_SC_FW_REQ_SUSYSTEM_RESET  = 0x0110,
    NVME_SC_FW_REQ_CTRL_RESET      = 0x0111,
    NVME_SC_FW_REQ_MAX_TIME_VIOL   = 0x0112,
    NVME_SC_FW_ACTIVATION_PROHIB   = 0x0113,
    NVME_SC_OVERLAPPING_RANGE      = 0x0114,
    NVME_SC_BOOTP_WRITE_PROHIB     = 0x011E,
    NVME_SC_CONFLICTING_ATTRS      = 0x0180,
    NVME_SC_INVALID_PROT_INFO      = 0x0181,
    NVME_SC_WRITE_TO_RO            = 0x0182,
    NVME_SC_WRITE_FAULT            = 0x0280,
    NVME_SC_UNRECOVERED_READ       = 0x0281,
    NVME_SC_E2E_GUARD_ERROR        = 0x0282,
    NVME_SC_E2E_APP_ERROR          = 0x0283,
    NVME_SC_E2E_REF_ERROR          = 0x0284,
    NVME_SC_CMP_FAILURE            = 0x0285,
    NVME_SC_ACCESS_DENIED          = 0x0286,
    NVME_SC_MORE                   = 0x2000,
    NVME_SC_DNR                    = 0x4000,
    NVME_SC_NO_COMPLETE            = 0xffff,
};

enum NvmeAsyncEventConfig {
    NVME_AVAILABLE_SPARE_ERR     = 1 << 0,
    NVME_TEMPERATURE_ERR         = 1 << 1,
    NVME_DEV_RELIABILITY_ERR     = 1 << 2,
    NVME_READ_ONLY_ERR           = 1 << 3,
    NVME_VOLATILE_MEM_BACKUP_ERR = 1 << 4,
    NVME_NAMESPACE_ATTR_NOTICE   = 1 << 8,
    NVME_FW_ACTIVATION_NOTICE    = 1 << 9,
    NVME_TELEMETRY_LOG_NOTICE    = 1 << 10,
};

enum NvmeLogPage {
    NVME_LOG_ERROR                     = 0x01,
    NVME_LOG_HEALTH_INFORMATION        = 0x02,
    NVME_LOG_FIRMWARE_SLOT             = 0x03,
    NVME_LOG_CHANGED_NS_LIST           = 0x04,
    NVME_LOG_COMMAND_EFFECTS_LOG       = 0x05,
    NVME_LOG_DISCOVERY                 = 0x70,
    NVME_LOG_RESERVATION_NOTIFICATION  = 0x80,
};

enum NvmeAsyncEventType {
    NVME_ASYNC_EVENT_TYPE_ERROR        = 0x0,
    NVME_ASYNC_EVENT_TYPE_SMART        = 0x1,
    NVME_ASYNC_EVENT_TYPE_NOTICE       = 0x2,
    NVME_ASYNC_EVENT_TYPE_IO           = 0x6,
    NVME_ASYNC_EVENT_TYPE_VENDOR       = 0x7,
};

enum NvmeAsyncEventInfoNotice {
    NVME_ASYNC_EVENT_NS_ATTR_CHANGED           = 0x0,
    NVME_ASYNC_EVENT_FW_ACTIVATION_START       = 0x1,
    NVME_ASYNC_EVENT_TELEMETRY_LOG_CHANGED     = 0x2,
};

enum NvmeAsyncEventInfoStatus {
    NVME_ASYNC_EVENT_SUBSYSTEM_RELIABILITY     = 0x0,
    NVME_ASYNC_EVENT_TEMPERATURE_THRESHOLD     = 0x1,
    NVME_ASYNC_EVENT_SPARE_BELOW_THRESHOLD     = 0x2,
};

enum NvmeIdentifyCns {
    NVME_IDENTIFY_CNS_NAMESPACE             = 0x00,
    NVME_IDENTIFY_CNS_CTRL                  = 0x01,
    NVME_IDENTIFY_CNS_ACTIVE_NS_LIST        = 0x02,
    NVME_IDENTIFY_CNS_NS_ID_DESCRIPTOR_LIST = 0x03,
    NVME_IDENTIFY_CNS_ALLOCATED_NS_LIST     = 0x10,
};

typedef struct NVME_PACKED nvme_fw_slot_info_log {
    uint8_t     afi_current:3;
    uint8_t     :1;
    uint8_t     afi_next:3;
    uint8_t     :1;
    uint8_t     reserved1[7];
    uint8_t     frs[7][8];
    uint8_t     reserved2[448];
} nvme_fw_slot_info_log_t;

typedef struct NVME_PACKED nvme_error_log {
    uint64_t err_cnt;
    uint16_t sqid;
    uint16_t cid;
    uint16_t status;
    uint16_t err_location;
    uint64_t lba;
    uint32_t nsid;
    uint8_t vs;
    uint8_t trtype;
    uint8_t rsvd30[2];
    uint64_t command_specific;
    uint16_t trtype_specific;
    uint8_t rsvd42[22];
} nvme_error_log_t;

typedef struct NVME_PACKED nvme_smart_log {
    struct {
        uint8_t available_spare : 1;
        uint8_t temperature : 1;
        uint8_t device_reliability : 1;
        uint8_t read_only : 1;
        uint8_t volatile_memory_backup : 1;
        uint8_t : 3;
    } critical_warning;
    uint16_t temperature;
    uint8_t available_spare;
    uint8_t available_spare_threshold;
    uint8_t percentage_used;
    uint8_t rsvd1[26];
    uint64_t data_units_read[2];
    uint64_t data_units_written[2];
    uint64_t host_read_commands[2];
    uint64_t host_write_commands[2];
    uint64_t controller_busy_time[2];
    uint64_t power_cycles[2];
    uint64_t power_on_hours[2];
    uint64_t unsafe_shutdowns[2];
    uint64_t media_errors[2];
    uint64_t number_of_err_info_log_entries[2];
    uint8_t rsvd2[320];
} nvme_smart_log_t;

typedef struct NVME_PACKED nvme_changed_nslist_log {
    uint32_t     ns_list[1024] __attribute__ ((aligned (32)));
} nvme_changed_nslist_log_t;

typedef struct NVME_PACKED nvme_bf_pkg_info_log {
    uint8_t     nfw[64];
    uint8_t     bnfw[64];
    uint8_t     os[64];
    uint8_t     oos[64];
    uint8_t     krnl[64];
    uint8_t     okrnl[64];
    uint8_t     spdk[64];
    uint8_t     ospdk[64];
    uint8_t     lsnap[64];
    uint8_t     olsnap[64];
    uint8_t     reserved[3456];
} nvme_bf_pkg_info_log_t;

typedef struct NVME_PACKED nvme_cmds_supported_and_effects_entry {
    uint16_t csupp : 1;
    uint16_t lbcc : 1;
    uint16_t ncc : 1;
    uint16_t nic : 1;
    uint16_t ccc : 1;
    uint16_t rsvd1 : 11;

    uint16_t cse : 3;
    uint16_t uuid_sel_sup : 1;
    uint16_t rsvd2 : 12;
} nvme_cmds_supported_and_effects_entry_t;

typedef struct NVME_PACKED nvme_commands_and_effects_log {
    nvme_cmds_supported_and_effects_entry_t acs[256] __attribute__ ((aligned (32)));
    nvme_cmds_supported_and_effects_entry_t iocs[256] __attribute__ ((aligned (32)));
    uint8_t rsvd[2048];
} nvme_commands_and_effects_log_t;

enum NvmeLogIdentifier {
    NVME_LOG_ERROR_INFO             = 0x01,
    NVME_LOG_SMART_INFO             = 0x02,
    NVME_LOG_FW_SLOT_INFO           = 0x03,
    NVME_LOG_CHANGED_NS_LIST_INFO   = 0x04,
    NVME_LOG_COMMANDS_EFFECTS_LOG   = 0x05,
    NVME_LOG_BF_PKG_INFO            = 0xC0,
};

typedef struct NVME_PACKED nvme_cmd_identify {
    nvme_cmd_common_t common;
    uint8_t     cns;
    uint8_t     rsvd;
    uint16_t    cntid;
    uint32_t    rsvd11[5];
} nvme_cmd_identify_t;

enum NvmeNamespaceIdentifierType {
    NVME_NS_IDENTIFIER_EUI64  = 0x1,
    NVME_NS_IDENTIFIER_NGUID  = 0x2,
    NVME_NS_IDENTIFIER_UUID   = 0x3,
    NVME_NS_IDENTIFIER_LAST
};

#define NVME_NS_ID_TYPES_NUM (NVME_NS_IDENTIFIER_LAST - 1)

typedef struct NVME_PACKED nvme_eui64 {
    uint8_t raw[8];
} nvme_eui64_t;

typedef struct NVME_PACKED nvme_nguid {
    uint8_t raw[16];
} nvme_ngiud_t;

typedef struct NVME_PACKED nvme_uuid {
    union {
        uuid_t raw;
        struct {
            uint32_t      time_low;
            uint16_t      time_mid;
            uint16_t      time_hi_and_version;
            uint16_t      clk_seq; /* hi_res + low */
            struct {
                uint32_t  byte5;
                uint16_t  byte1;
            } node;
        } bits;
    };
} nvme_uuid_t;

typedef union NVME_PACKED nvme_id_ns_global {
        nvme_eui64_t  eui64;
        nvme_ngiud_t  nguid;
        nvme_uuid_t   uuid;
} nvme_id_ns_global_t;

typedef struct NVME_PACKED nvme_id_ns_descriptor {
    uint8_t              nidt;
    uint8_t              nidl;
    uint8_t              rsvd[2];
    nvme_id_ns_global_t  nid;
} nvme_id_ns_descriptor_t;

typedef struct NVME_PACKED nvme_cmd_create_cq {
    nvme_cmd_common_t common;
    uint16_t qid;
    uint16_t qsize;
    uint32_t pc : 1;
    uint32_t ien : 1;
    uint32_t : 14;
    uint32_t iv : 16;
    uint32_t rsvd12[4];
} nvme_cmd_create_cq_t;

typedef struct NVME_PACKED nvme_cmd_get_page_log {
    nvme_cmd_common_t common;
    uint16_t lid : 8;
    uint16_t lsp : 4;
    uint16_t : 3;
    uint16_t rae : 1;
    uint16_t numdl;
    uint16_t numdu; // NUMDU + NUMDL combined
    uint16_t rsvd;
    uint32_t rsvd2[4];
} nvme_cmd_get_page_log_t;

typedef struct NVME_PACKED nvme_cmd_create_sq {
    nvme_cmd_common_t common;
    uint16_t qid;
    uint16_t qsize;
    uint16_t pc : 1;
    uint16_t qprio : 2;
    uint16_t : 13;
    uint16_t cqid;
    uint32_t rsvd12[4];
} nvme_cmd_create_sq_t;

enum NvmeQueueFlags {
    NVME_Q_PC           = 1,
    NVME_Q_PRIO_URGENT  = 0,
    NVME_Q_PRIO_HIGH    = 1,
    NVME_Q_PRIO_NORMAL  = 2,
    NVME_Q_PRIO_LOW     = 3,
};

typedef struct NVME_PACKED nvme_cmd_delete_q {
    nvme_cmd_common_t common;
    uint16_t qid;
    uint16_t rsvd;
    uint32_t rsvd11[5];
} nvme_cmd_delete_q_t;

typedef struct NVME_PACKED nvme_cmd_rw {
    nvme_cmd_common_t common;
    uint64_t slba;
    uint16_t nlb;
    uint16_t control;
    uint32_t dsmgmt;
    uint32_t reftag;
    uint16_t apptag;
    uint16_t appmask;
} nvme_cmd_rw_t;

enum {
    NVME_DSM_IDR = 1 << 0,
    NVME_DSM_IDW = 1 << 1,
    NVME_DSM_AD  = 1 << 2,
};

typedef struct NVME_PACKED nvme_dsm_range {
    uint8_t  access;
    uint8_t  flags;
    uint8_t  rsvd3;
    uint8_t  cas;
    uint32_t nlb;
    uint64_t slba;
} nvme_dsm_range_t;

typedef struct NVME_PACKED nvme_cmd_dsm {
    nvme_cmd_common_t common;
    uint32_t nr : 8;
    uint32_t : 24;
    uint32_t attributes : 3;
    uint32_t : 29;
    uint32_t rsvd[4];
} nvme_cmd_dsm_t;

typedef struct NVME_PACKED nvme_aen_completion  {
    uint32_t ev_type : 3;
    uint32_t reserved1 : 5;
    uint32_t ev_info : 8;
    uint32_t log_page_identifier : 8;
    uint32_t reserved2 : 8;
} nvme_aen_completion_t;

typedef struct NVME_PACKED nvme_cmd_fw_download {
    nvme_cmd_common_t common;
    uint32_t numd;
    uint32_t ofst;
    uint32_t rsvd[4];
} nvme_cmd_fw_download_t;

typedef struct NVME_PACKED nvme_cmd_fw_commit {
    nvme_cmd_common_t common;
    uint32_t firmware_slot : 3;
    uint32_t commit_action : 3;
    uint32_t : 25;
    uint32_t bpid : 1;
    uint32_t rsvd[5];
} nvme_cmd_fw_commit_t;

typedef struct NVME_PACKED nvme_cmd_fw_recover {
    nvme_cmd_common_t common;
    uint32_t rsvd[2];
    uint32_t pid;
    uint32_t rsvd2[3];
} nvme_cmd_fw_recover_t;

static inline uint32_t
nvme_acmd_get_log_page_cmd_num_dwords(const nvme_cmd_get_page_log_t *c)
{
    return (le16_to_cpu(c->numdu) << 16) | le16_to_cpu(c->numdl);
}

enum nvme_cmd_vs_iova_mgmt_opm {
    NVME_IOVA_OPM_MAP_RANGE = 0,
    NVME_IOVA_OPM_UNMAP_RANGE = 1,
    NVME_IOVA_OPM_UNMAP_ALL = 2
};

enum nvme_cmd_vs_iova_mgmt_sc {
    NVME_IOVA_MAX_CAPACITY_EXCEEDED = 0x01C0,
    NVME_IOVA_INVALID_FUNCTION_ID   = 0x01C1,
    NVME_IOVA_UNSUPPORTED_MODE_TYPE = 0x01C2
};

enum nvme_cmd_vs_iova_mgmt_szu {
    NVME_IOVA_SZU_4K,
    NVME_IOVA_SZU_64K,
    NVME_IOVA_SZU_1M,
    NVME_IOVA_SZU_16M,
    NVME_IOVA_SZU_256M,
    NVME_IOVA_SZU_1G,
    NVME_IOVA_SZU_4G,
    NVME_IOVA_SZU_RESERVED,
};

typedef struct NVME_PACKED nvme_cmd_vs_iova_mgmt {
    nvme_cmd_common_t common;
    // DW10
    struct {
        uint32_t fid:16;
        uint32_t opm:4;
    };
    // DW11-DW12
    uint64_t    siova;
    // DW13-DW14
    uint64_t    tiova;
    // DW15
    struct {
        uint32_t szu:4;
        uint32_t size:28;
    };

} nvme_cmd_vs_iova_mgmt_t;

#endif
