/*
 * Copyright © 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef SNAP_VIRTIO_ADM_SPEC_H
#define SNAP_VIRTIO_ADM_SPEC_H
#include <stdint.h>
#include "snap_macros.h"

#define VIRTIO_SPEC_VER_1_2	0x12
#define VIRTIO_SPEC_VER_1_3	0x13

#ifndef VIRTIO_F_ADMIN_VQ
#define VIRTIO_F_ADMIN_VQ 41
#endif

#ifndef VIRTIO_F_ADMIN_MIGRATION
#define VIRTIO_F_ADMIN_MIGRATION 44
#endif

#ifndef VIRTIO_F_ADMIN_MIGRATION_DYNAMIC_INTERNAL_STATE_TRACK
#define VIRTIO_F_ADMIN_MIGRATION_DYNAMIC_INTERNAL_STATE_TRACK 45
#endif

#ifndef VIRTIO_F_ADMIN_DIRTY_PAGE_PUSH_BITMAP_TRACK
#define VIRTIO_F_ADMIN_DIRTY_PAGE_PUSH_BITMAP_TRACK 46
#endif

#ifndef VIRTIO_F_ADMIN_DIRTY_PAGE_PUSH_BYTEMAP_TRACK
#define VIRTIO_F_ADMIN_DIRTY_PAGE_PUSH_BYTEMAP_TRACK 47
#endif

#ifndef VIRTIO_F_ADMIN_DIRTY_PAGE_PULL_BITMAP_TRACK
#define VIRTIO_F_ADMIN_DIRTY_PAGE_PULL_BITMAP_TRACK 48
#endif

#ifndef VIRTIO_F_ADMIN_DIRTY_PAGE_PULL_BYTEMAP_TRACK
#define VIRTIO_F_ADMIN_DIRTY_PAGE_PULL_BYTEMAP_TRACK 49
#endif

#define SNAP_VQ_ADM_MIG_CTRL 64
#define SNAP_VQ_ADM_MIG_IDENTITY 0
#define SNAP_VQ_ADM_MIG_GET_STATUS 1
#define SNAP_VQ_ADM_MIG_MODIFY_STATUS 2
#define SNAP_VQ_ADM_MIG_GET_STATE_PENDING_BYTES 3
#define SNAP_VQ_ADM_MIG_SAVE_STATE 4
#define SNAP_VQ_ADM_MIG_RESTORE_STATE 5

#define SNAP_VQ_ADM_DP_TRACK_CTRL 65
#define SNAP_VQ_ADM_DP_IDENTITY 0
#define SNAP_VQ_ADM_DP_START_TRACK 1
#define SNAP_VQ_ADM_DP_STOP_TRACK 2
#define SNAP_VQ_ADM_DP_GET_MAP_PENDING_BYTES 3
#define SNAP_VQ_ADM_DP_REPORT_MAP 4

/* Status Code (SC) that are transport, device and vendor independent */
#define SNAP_VIRTIO_ADM_STATUS_COMMON_START 0
#define SNAP_VIRTIO_ADM_STATUS_COMMON_END 31

/* Status Code (SC) that are transport specific */
#define SNAP_VIRTIO_ADM_STATUS_TRANSPORT_START 32
#define SNAP_VIRTIO_ADM_STATUS_TRANSPORT_END 63

/* Status Code (SC) that are device specific */
#define SNAP_VIRTIO_ADM_STATUS_DEVICE_START 64
#define SNAP_VIRTIO_ADM_STATUS_DEVICE_END 95

/* Status Code (SC) that are reserved */
#define SNAP_VIRTIO_ADM_STATUS_RESERVED_START 96
#define SNAP_VIRTIO_ADM_STATUS_RESERVED_END 127

#define SNAP_VIRTIO_ADM_CMD_OPCODES_ARR_LEN	64

enum snap_virtio_adm_status {
	SNAP_VIRTIO_ADM_STATUS_OK = 0,
	SNAP_VIRTIO_ADM_STATUS_ERR = 1,
	SNAP_VIRTIO_ADM_STATUS_INVALID_CLASS = 2,
	SNAP_VIRTIO_ADM_STATUS_INVALID_COMMAND = 3,
	SNAP_VIRTIO_ADM_STATUS_DATA_TRANSFER_ERR = 4,
	SNAP_VIRTIO_ADM_STATUS_DEVICE_INTERNAL_ERR = 5,
	SNAP_VIRTIO_ADM_STATUS_DNR = (1<<7),
	/*spec v1.3*/
	SNAP_VIRTIO_ADMIN_STATUS_EAGAIN = 11,
	SNAP_VIRTIO_ADMIN_STATUS_ENOMEM = 12,
	SNAP_VIRTIO_ADMIN_STATUS_EINVAL = 22,
	SNAP_VIRTIO_ADMIN_MAX
};

enum snap_virtio_adm_status_qualifier {
	SNAP_VIRTIO_ADMIN_STATUS_Q_OK = 0x0,
	SNAP_VIRTIO_ADMIN_STATUS_Q_INVALID_COMMAND = 0x1,
	SNAP_VIRTIO_ADMIN_STATUS_Q_INVALID_OPCODE = 0x2,
	SNAP_VIRTIO_ADMIN_STATUS_Q_INVALID_FIELD = 0x3,
	SNAP_VIRTIO_ADMIN_STATUS_Q_INVALID_GROUP = 0x4,
	SNAP_VIRTIO_ADMIN_STATUS_Q_INVALID_MEMBER = 0x5,
	SNAP_VIRTIO_ADMIN_STATUS_Q_NORESOURCE = 0x6,
	SNAP_VIRTIO_ADMIN_STATUS_Q_TRYAGAIN = 0x7,
	SNAP_VIRTIO_ADMIN_STATUS_Q_MAX
};

enum snap_virtio_adm_opcode {
	SNAP_VIRTIO_ADMIN_CMD_LIST_QUERY = 0X0,
	SNAP_VIRTIO_ADMIN_CMD_LIST_USE = 0X1,
	SNAP_VIRTIO_ADMIN_CMD_LEGACY_COMMON_CFG_WRITE = 0x2,
	SNAP_VIRTIO_ADMIN_CMD_LEGACY_COMMON_CFG_READ = 0x3,
	SNAP_VIRTIO_ADMIN_CMD_LEGACY_DEV_CFG_WRITE = 0x4,
	SNAP_VIRTIO_ADMIN_CMD_LEGACY_DEV_CFG_READ = 0x5,
	SNAP_VIRTIO_ADMIN_CMD_LEGACY_NOTIFY_INFO = 0x6,
	SNAP_VIRTIO_ADMIN_CMD_MAX
};

enum snap_virtio_adm_group_type {
	SNAP_VIRTIO_ADMIN_GROUP_TYPE_SRIOV = 0X1,
	SNAP_VIRTIO_ADMIN_GROUP_TYPE_MAX
};

enum snap_virtio_adm_notify_flags {
	SNAP_VIRTIO_ADMIN_NOTIFY_END_OF_LIST = 0, /* End of list */
	SNAP_VIRTIO_ADMIN_NOTIFY_OWNER_DEV = 1, /* owner device */
	SNAP_VIRTIO_ADMIN_NOTIFY_MEMBER_DEV = 2, /* member device */
};

struct snap_virtio_adm_cmd_hdr_v1_2 {
	uint8_t cmd_class;
	uint8_t command;
} SNAP_PACKED;

struct snap_virtio_adm_cmd_ftr_v1_2 {
	/*
	 * Bits (6:0) - Status Code (SC)
	 * Indicate status information for the command
	 *
	 * Bit (7) - Do Not Retry (DNR)
	 * If set to 1, indicates that if the same command is submitted
	 * again - it is expected to fail.
	 * If cleared to 0, indicates that the same command is submitted
	 * again may succeed.
	 */
	uint8_t status;
} SNAP_PACKED;

struct snap_virtio_adm_cmd_hdr_v1_3 {
	/* Device-readable part */
	__le16 opcode;
	/*
	 * 1 - SR-IOV
	 * 2-65535 - reserved
	 */
	__le16 group_type;
	/* unused, reserved for future extensions */
	uint8_t reserved1[12];
	__le64 group_member_id;
} SNAP_PACKED;

struct snap_virtio_adm_cmd_ftr_v1_3 {
	__le16 status;
	__le16 status_qualifier;
	/* unused, reserved for future extensions */
	uint8_t reserved2[4];
} SNAP_PACKED;

union snap_virtio_adm_cmd_hdr {
	struct snap_virtio_adm_cmd_hdr_v1_2 hdr_v1_2;
	struct snap_virtio_adm_cmd_hdr_v1_3 hdr_v1_3;
};

union snap_virtio_adm_cmd_ftr {
	struct snap_virtio_adm_cmd_ftr_v1_2 ftr_v1_2;
	struct snap_virtio_adm_cmd_ftr_v1_3 ftr_v1_3;
};

struct snap_vq_adm_get_pending_bytes_data {
	__le16 vdev_id;
	__le16 reserved;
};

struct snap_vq_adm_get_pending_bytes_result {
	__le64 pending_bytes;
};

struct snap_vq_adm_modify_status_data {
	__le16 vdev_id;
	__le16 internal_status;
};

struct snap_vq_adm_save_state_data {
	__le16 vdev_id;
	__le16 reserved[3];
	__le64 offset;
	__le64 length; /* Num of data bytes to be returned by the device */
};

struct snap_vq_adm_restore_state_data {
	__le16 vdev_id;
	__le16 reserved;
	__le64 offset;
	__le64 length; /* Num of data bytes to be consumed by the device */
};

struct snap_vq_adm_get_status_data {
	__le16 vdev_id;
	__le16 reserved;
};

struct snap_vq_adm_get_status_result {
	__le16 internal_status; /* Value from enum snap_virtio_internal_status */
	__le16 reserved;
};

/* dirty page tracking */
struct virtio_admin_dirty_page_identity_result {
	__le16 log_max_pages_track_pull_bitmap_mode; /* Per managed device (log) */
	__le16 log_max_pages_track_pull_bytemap_mode; /* Per managed device (log) */
	__le32 max_track_ranges; /* Maximum number of ranges a device can track */
};

enum snap_vq_adm_dirty_page_track_mode {
	VIRTIO_M_DIRTY_TRACK_PUSH_BITMAP = 1, /* Use push mode with bit granularity */
	VIRTIO_M_DIRTY_TRACK_PUSH_BYTEMAP = 2, /* Use push mode with byte granularity */
	VIRTIO_M_DIRTY_TRACK_PULL_BITMAP = 3, /* Use pull mode with bit granularity */
	VIRTIO_M_DIRTY_TRACK_PULL_BYTEMAP = 4, /* Use pull mode with byte granularity */

	/* experimental, non standard */
	VIRTIO_M_DIRTY_TRACK_PULL_PAGELIST = 0xF001 /* report pages as a raw pagelist */
};

struct snap_vq_adm_sge {
	__le64 addr;
	__le32 len;
	__le32 reserved;
};

struct snap_vq_adm_dirty_page_track_start {
	__le16 vdev_id;
	__le16 track_mode;
	__le32 vdev_host_page_size;
	__le64 vdev_host_range_addr;
	__le64 range_length;
};

struct snap_vq_adm_dirty_page_track_stop {
	__le16 vdev_id;
	__le16 reserved[3];
	__le64 vdev_host_range_addr;
};


struct snap_virtio_admin_cmd_list {
	/* Indicates which of the below fields were returned */
	__le64 opcodes[SNAP_VIRTIO_ADM_CMD_OPCODES_ARR_LEN];
};

struct snap_virtio_admin_cmd_data_lr_write {
	uint8_t offset; /* Starting offset of the register(s) to write. */
	uint8_t reserved[7];
	uint8_t data[];
};

struct snap_virtio_admin_cmd_data_lr_read {
	uint8_t offset; /* Starting offset of the register(s) to read. */
};

struct snap_virtio_pci_lr_notify_info {
	uint8_t flags; /* 0 = end of list, 1 = owner device, 2 = member device */
	uint8_t bar; /* BAR of the member or the owner device */
	uint8_t reserved[6];
	__le64 offset; /* Offset within bar */
};

struct snap_virtio_admin_cmd_lr_notify_info {
	struct snap_virtio_pci_lr_notify_info entries[4];
};

union snap_virtio_adm_cmd_in {
	struct snap_vq_adm_get_pending_bytes_data pending_bytes_data;
	struct snap_vq_adm_modify_status_data modify_status_data;
	struct snap_vq_adm_save_state_data save_state_data;
	struct snap_vq_adm_get_status_data get_status_data;
	struct snap_vq_adm_restore_state_data restore_state_data;
	struct snap_vq_adm_dirty_page_track_start dp_track_start_data;
	struct snap_vq_adm_dirty_page_track_stop dp_track_stop_data;
	struct snap_virtio_admin_cmd_list admin_cmd_list;
	struct snap_virtio_admin_cmd_data_lr_write lr_write_data;
	struct snap_virtio_admin_cmd_data_lr_read lr_read_data;
	__le16 vdev_id;
};

union snap_virtio_adm_cmd_out {
	struct snap_vq_adm_get_pending_bytes_result pending_bytes_res;
	struct snap_vq_adm_get_status_result get_status_res;
	struct snap_virtio_admin_cmd_list admin_cmd_list;
	struct snap_virtio_admin_cmd_lr_notify_info lr_notify_info;
	uint8_t lr_read_out[16];
};

struct snap_virtio_adm_cmd_layout {
	union snap_virtio_adm_cmd_hdr hdr;
	union snap_virtio_adm_cmd_in in;
	/* Additional data defined by variadic cmd_in structures */
	union snap_virtio_adm_cmd_out out;
	/* Additional data defined by variadic cmd_out structures */
	union snap_virtio_adm_cmd_ftr ftr;
};

#endif
