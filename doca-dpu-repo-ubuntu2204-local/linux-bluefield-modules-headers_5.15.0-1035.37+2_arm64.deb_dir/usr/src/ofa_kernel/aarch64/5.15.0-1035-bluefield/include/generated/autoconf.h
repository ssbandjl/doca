#ifndef __OFED_BUILD__
#include_next <generated/autoconf.h>
#define CONFIG_BACKPORT_LRO 1
#define CONFIG_DEBUG_INFO 1
#define CONFIG_ENABLE_BASECODE_EXTRAS 1
#define CONFIG_ENABLE_VFIO 1
#define CONFIG_GPU_DIRECT_STORAGE 1
#define CONFIG_INFINIBAND 1
#define CONFIG_INFINIBAND_ADDR_TRANS 1
#define CONFIG_INFINIBAND_ADDR_TRANS_CONFIGFS 1
#define CONFIG_INFINIBAND_IPOIB 1
#define CONFIG_INFINIBAND_IPOIB_CM 1
#define CONFIG_INFINIBAND_IPOIB_DEBUG 1
#define CONFIG_INFINIBAND_USER_ACCESS 1
#define CONFIG_INFINIBAND_USER_ACCESS_UCM 1
#define CONFIG_INFINIBAND_USER_MAD 1
#define CONFIG_INFINIBAND_USER_MEM 1
#define CONFIG_MLX5_ACCEL 1
#define CONFIG_MLX5_BRIDGE 1
#define CONFIG_MLX5_CLS_ACT 1
#define CONFIG_MLX5_CORE 1
#define CONFIG_MLX5_CORE_EN 1
#define CONFIG_MLX5_CORE_EN_DCB 1
#define CONFIG_MLX5_CORE_IPOIB 1
#define CONFIG_MLX5_DEBUG 1
#define CONFIG_MLX5_EN_ARFS 1
#define CONFIG_MLX5_EN_IPSEC 1
#define CONFIG_MLX5_EN_RXNFC 1
#define CONFIG_MLX5_EN_SPECIAL_SQ 1
#define CONFIG_MLX5_EN_TLS 1
#define CONFIG_MLX5_ESWITCH 1
#define CONFIG_MLX5_INFINIBAND 1
#define CONFIG_MLX5_IPSEC 1
#define CONFIG_MLX5_MPFS 1
#define CONFIG_MLX5_SF 1
#define CONFIG_MLX5_SF_CFG 1
#define CONFIG_MLX5_SF_MANAGER 1
#define CONFIG_MLX5_SF_SFC 1
#define CONFIG_MLX5_SW_STEERING 1
#define CONFIG_MLX5_TC_CT 1
#define CONFIG_MLX5_TC_SAMPLE 1
#define CONFIG_MLX5_TLS 1
#define CONFIG_MLXDEVM 1
#define CONFIG_MLXFW 1

#else
#undef CONFIG_AUXILIARY_BUS
#undef CONFIG_BACKPORT_LRO
#undef CONFIG_BF_DEVICE_EMULATION
#undef CONFIG_BF_POWER_FAILURE_EVENT
#undef CONFIG_BLK_DEV_NVME
#undef CONFIG_COMPAT_KOBJECT_BACKPORT
#undef CONFIG_COMPAT_VERSION
#undef CONFIG_DEBUG_INFO
#undef CONFIG_ENABLE_BASECODE_EXTRAS
#undef CONFIG_ENABLE_MLX5_FS_DEBUGFS
#undef CONFIG_ENABLE_VFIO
#undef CONFIG_ENABLE_XDP
#undef CONFIG_GPU_DIRECT_STORAGE
#undef CONFIG_INFINIBAND
#undef CONFIG_INFINIBAND_ADDR_TRANS
#undef CONFIG_INFINIBAND_ADDR_TRANS_CONFIGFS
#undef CONFIG_INFINIBAND_CORE_DUMMY
#undef CONFIG_INFINIBAND_IPOIB
#undef CONFIG_INFINIBAND_IPOIB_CM
#undef CONFIG_INFINIBAND_IPOIB_DEBUG
#undef CONFIG_INFINIBAND_IPOIB_DEBUG_DATA
#undef CONFIG_INFINIBAND_ISER
#undef CONFIG_SCSI_ISCSI_ATTRS
#undef CONFIG_ISCSI_TCP
#undef CONFIG_INFINIBAND_ISERT
#undef CONFIG_INFINIBAND_ISERT_DUMMY
#undef CONFIG_INFINIBAND_ISER_DUMMY
#undef CONFIG_INFINIBAND_MADEYE
#undef CONFIG_INFINIBAND_ON_DEMAND_PAGING
#undef CONFIG_INFINIBAND_PA_MR
#undef CONFIG_INFINIBAND_SDP_DEBUG
#undef CONFIG_INFINIBAND_SDP_DEBUG_DATA
#undef CONFIG_INFINIBAND_SDP_RECV_ZCOPY
#undef CONFIG_INFINIBAND_SDP_SEND_ZCOPY
#undef CONFIG_INFINIBAND_SRP
#undef CONFIG_INFINIBAND_SRP_DUMMY
#undef CONFIG_INFINIBAND_USER_ACCESS
#undef CONFIG_INFINIBAND_USER_ACCESS_UCM
#undef CONFIG_INFINIBAND_USER_MAD
#undef CONFIG_INFINIBAND_USER_MEM
#undef CONFIG_INFINIBAND_WQE_FORMAT
#undef CONFIG_IPOIB_ALL_MULTI
#undef CONFIG_IPOIB_VERSION
#undef CONFIG_ISCSI_TCP
#undef CONFIG_MEMTRACK
#undef CONFIG_MLNX_BLOCK_REQUEST_MODULE
#undef CONFIG_MLX5_ACCEL
#undef CONFIG_MLX5_BRIDGE
#undef CONFIG_MLX5_CLS_ACT
#undef CONFIG_MLX5_CORE
#undef CONFIG_MLX5_CORE_EN
#undef CONFIG_MLX5_CORE_EN_DCB
#undef CONFIG_MLX5_CORE_IPOIB
#undef CONFIG_MLX5_DEBUG
#undef CONFIG_MLX5_EN_ARFS
#undef CONFIG_MLX5_EN_IPSEC
#undef CONFIG_MLX5_EN_MACSEC
#undef CONFIG_MLX5_EN_RXNFC
#undef CONFIG_MLX5_EN_SPECIAL_SQ
#undef CONFIG_MLX5_EN_TLS
#undef CONFIG_MLX5_ESWITCH
#undef CONFIG_MLX5_FPGA
#undef CONFIG_MLX5_FPGA_IPSEC
#undef CONFIG_MLX5_FPGA_TLS
#undef CONFIG_MLX5_INFINIBAND
#undef CONFIG_MLX5_IPSEC
#undef CONFIG_MLX5_MPFS
#undef CONFIG_MLX5_SF
#undef CONFIG_MLX5_SF_CFG
#undef CONFIG_MLX5_SF_MANAGER
#undef CONFIG_MLX5_SF_SFC
#undef CONFIG_MLX5_SW_STEERING
#undef CONFIG_MLX5_TC_CT
#undef CONFIG_MLX5_TC_SAMPLE
#undef CONFIG_MLX5_TLS
#undef CONFIG_MLX5_VDPA_NET_DUMMY
#undef CONFIG_MLXDEVM
#undef CONFIG_MLXFW
#undef CONFIG_NVME_COMMON
#undef CONFIG_NVME_CORE
#undef CONFIG_NVME_FABRICS
#undef CONFIG_NVME_FC
#undef CONFIG_NVME_HOST_DUMMY
#undef CONFIG_NVME_HOST_WITHOUT_FC
#undef CONFIG_NVME_MULTIPATH
#undef CONFIG_NVME_POLL
#undef CONFIG_NVME_RDMA
#undef CONFIG_NVME_TARGET
#undef CONFIG_NVME_TARGET_DUMMY
#undef CONFIG_NVME_TARGET_FC
#undef CONFIG_NVME_TARGET_FCLOOP
#undef CONFIG_NVME_TARGET_LOOP
#undef CONFIG_NVME_TARGET_RDMA
#undef CONFIG_NVME_TARGET_TCP
#undef CONFIG_NVME_TCP
#undef CONFIG_RDMA_RXE
#undef CONFIG_RDMA_RXE_DUMMY
#undef CONFIG_SCSI_ISCSI_ATTRS
#undef CONFIG_SCSI_SRP_ATTRS
#undef CONFIG_SUNRPC_XPRT_RDMA
#undef CONFIG_SUNRPC_XPRT_RDMA_DUMMY
#define CONFIG_BACKPORT_LRO 1
#define CONFIG_DEBUG_INFO 1
#define CONFIG_ENABLE_BASECODE_EXTRAS 1
#define CONFIG_ENABLE_VFIO 1
#define CONFIG_GPU_DIRECT_STORAGE 1
#define CONFIG_INFINIBAND 1
#define CONFIG_INFINIBAND_ADDR_TRANS 1
#define CONFIG_INFINIBAND_ADDR_TRANS_CONFIGFS 1
#define CONFIG_INFINIBAND_IPOIB 1
#define CONFIG_INFINIBAND_IPOIB_CM 1
#define CONFIG_INFINIBAND_IPOIB_DEBUG 1
#define CONFIG_INFINIBAND_USER_ACCESS 1
#define CONFIG_INFINIBAND_USER_ACCESS_UCM 1
#define CONFIG_INFINIBAND_USER_MAD 1
#define CONFIG_INFINIBAND_USER_MEM 1
#define CONFIG_MLX5_ACCEL 1
#define CONFIG_MLX5_BRIDGE 1
#define CONFIG_MLX5_CLS_ACT 1
#define CONFIG_MLX5_CORE 1
#define CONFIG_MLX5_CORE_EN 1
#define CONFIG_MLX5_CORE_EN_DCB 1
#define CONFIG_MLX5_CORE_IPOIB 1
#define CONFIG_MLX5_DEBUG 1
#define CONFIG_MLX5_EN_ARFS 1
#define CONFIG_MLX5_EN_IPSEC 1
#define CONFIG_MLX5_EN_RXNFC 1
#define CONFIG_MLX5_EN_SPECIAL_SQ 1
#define CONFIG_MLX5_EN_TLS 1
#define CONFIG_MLX5_ESWITCH 1
#define CONFIG_MLX5_INFINIBAND 1
#define CONFIG_MLX5_IPSEC 1
#define CONFIG_MLX5_MPFS 1
#define CONFIG_MLX5_SF 1
#define CONFIG_MLX5_SF_CFG 1
#define CONFIG_MLX5_SF_MANAGER 1
#define CONFIG_MLX5_SF_SFC 1
#define CONFIG_MLX5_SW_STEERING 1
#define CONFIG_MLX5_TC_CT 1
#define CONFIG_MLX5_TC_SAMPLE 1
#define CONFIG_MLX5_TLS 1
#define CONFIG_MLXDEVM 1
#define CONFIG_MLXFW 1
#endif
#undef CONFIG_MLX5_EN_MACSEC
