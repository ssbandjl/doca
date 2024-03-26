#ifndef SPDK_CONFIG_H
#define SPDK_CONFIG_H
#define SPDK_CONFIG_APPS 1
#define SPDK_CONFIG_ARCH armv8-a
#undef SPDK_CONFIG_ASAN
#undef SPDK_CONFIG_AVAHI
#undef SPDK_CONFIG_CET
#undef SPDK_CONFIG_COVERAGE
#define SPDK_CONFIG_CROSS_PREFIX 
#define SPDK_CONFIG_CRYPTO 1
#define SPDK_CONFIG_CRYPTO_MLX5 1
#undef SPDK_CONFIG_CUSTOMOCF
#undef SPDK_CONFIG_DAOS
#define SPDK_CONFIG_DAOS_DIR 
#undef SPDK_CONFIG_DEBUG
#undef SPDK_CONFIG_DPDK_COMPRESSDEV
#define SPDK_CONFIG_DPDK_DIR /workspace/packages/spdk/src/dpdk/build
#define SPDK_CONFIG_DPDK_INC_DIR 
#define SPDK_CONFIG_DPDK_LIB_DIR 
#undef SPDK_CONFIG_DPDK_PKG_CONFIG
#define SPDK_CONFIG_ENV /workspace/packages/spdk/src/lib/env_dpdk
#define SPDK_CONFIG_EXAMPLES 1
#undef SPDK_CONFIG_FC
#define SPDK_CONFIG_FC_PATH 
#undef SPDK_CONFIG_FIO_PLUGIN
#define SPDK_CONFIG_FIO_SOURCE_DIR /usr/src/fio
#undef SPDK_CONFIG_FUSE
#undef SPDK_CONFIG_FUZZER
#define SPDK_CONFIG_FUZZER_LIB 
#undef SPDK_CONFIG_HAVE_ARC4RANDOM
#undef SPDK_CONFIG_HAVE_LIBARCHIVE
#define SPDK_CONFIG_HAVE_LIBBSD 1
#undef SPDK_CONFIG_IDXD
#undef SPDK_CONFIG_IDXD_KERNEL
#undef SPDK_CONFIG_IPSEC_MB
#define SPDK_CONFIG_IPSEC_MB_DIR 
#define SPDK_CONFIG_ISAL 1
#define SPDK_CONFIG_ISAL_CRYPTO 1
#define SPDK_CONFIG_ISCSI_INITIATOR 1
#define SPDK_CONFIG_LIBDIR 
#undef SPDK_CONFIG_LTO
#undef SPDK_CONFIG_NVME_CUSE
#undef SPDK_CONFIG_OCF
#define SPDK_CONFIG_OCF_PATH 
#define SPDK_CONFIG_OPENSSL_PATH 
#undef SPDK_CONFIG_PGO_CAPTURE
#undef SPDK_CONFIG_PGO_USE
#undef SPDK_CONFIG_PMDK
#define SPDK_CONFIG_PMDK_DIR 
#define SPDK_CONFIG_PREFIX /opt/mellanox/spdk
#define SPDK_CONFIG_RAID5F 1
#undef SPDK_CONFIG_RBD
#define SPDK_CONFIG_RDMA 1
#define SPDK_CONFIG_RDMA_PROV mlx5_dv
#define SPDK_CONFIG_RDMA_SEND_WITH_INVAL 1
#define SPDK_CONFIG_RDMA_SET_ACK_TIMEOUT 1
#define SPDK_CONFIG_RDMA_SET_TOS 1
#define SPDK_CONFIG_SHARED 1
#undef SPDK_CONFIG_SMA
#undef SPDK_CONFIG_TESTS
#undef SPDK_CONFIG_TSAN
#undef SPDK_CONFIG_UBLK
#undef SPDK_CONFIG_UBSAN
#undef SPDK_CONFIG_UNIT_TESTS
#define SPDK_CONFIG_URING 1
#define SPDK_CONFIG_URING_PATH 
#define SPDK_CONFIG_URING_ZNS 1
#undef SPDK_CONFIG_USDT
#undef SPDK_CONFIG_VBDEV_COMPRESS
#undef SPDK_CONFIG_VBDEV_COMPRESS_MLX5
#undef SPDK_CONFIG_VFIO_USER
#define SPDK_CONFIG_VFIO_USER_DIR 
#define SPDK_CONFIG_VHOST 1
#define SPDK_CONFIG_VIRTIO 1
#undef SPDK_CONFIG_VTUNE
#define SPDK_CONFIG_VTUNE_DIR 
#undef SPDK_CONFIG_WERROR
#define SPDK_CONFIG_WPDK_DIR 
#define SPDK_CONFIG_XLIO 1
#define SPDK_CONFIG_XLIO_DIR 
#undef SPDK_CONFIG_XNVME
#endif /* SPDK_CONFIG_H */