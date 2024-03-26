/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* access_ok has 3 parameters */
/* #undef HAVE_ACCESS_OK_HAS_3_PARAMS */

/* acpi_storage_d3 exist */
#define HAVE_ACPI_STORAGE_D3 1

/* addrconf_addr_eui48 is defined */
#define HAVE_ADDRCONF_ADDR_EUI48 1

/* addrconf_ifid_eui48 is defined */
#define HAVE_ADDRCONF_IFID_EUI48 1

/* adjfine is defined */
/* #undef HAVE_ADJUST_BY_SCALED_PPM */

/* alloc_netdev_mqs has 5 params */
/* #undef HAVE_ALLOC_NETDEV_MQS_5_PARAMS */

/* alloc_netdev_mq has 4 params */
/* #undef HAVE_ALLOC_NETDEV_MQ_4_PARAMS */

/* array_index_nospec is defined */
#define HAVE_ARRAY_INDEX_NOSPEC 1

/* mm.h has assert_fault_locked */
/* #undef HAVE_ASSERT_FAULT_LOCKED */

/* atomic_fetch_add_unless is defined */
#define HAVE_ATOMIC_FETCH_ADD_UNLESS 1

/* atomic_pinned_vm is defined */
#define HAVE_ATOMIC_PINNED_VM 1

/* linux/blkdev.h has bdev_discard_granularity */
/* #undef HAVE_BDEV_DISCARD_GRANULARITY */

/* bdev_is_partition is defined */
#define HAVE_BDEV_IS_PARTITION 1

/* blkdev.h has bdev_max_zone_append_sectors */
#define HAVE_BDEV_MAX_ZONE_APPEND_SECTORS 1

/* bdev_nr_bytes exist */
/* #undef HAVE_BDEV_NR_BYTES */

/* genhd.h has bdev_nr_sectors */
#define HAVE_BDEV_NR_SECTORS 1

/* blkdev.h has bdev_nr_zones */
/* #undef HAVE_BDEV_NR_ZONES */

/* bdev_start_io_acct is defined */
/* #undef HAVE_BDEV_START_IO_ACCT */

/* bdev_start_io_acct is defined */
/* #undef HAVE_BDEV_START_IO_ACCT_3_PARAM */

/* linux/blkdev.h has bdev_write_cache */
/* #undef HAVE_BDEV_WRITE_CACHE */

/* bdev_write_zeroes_sectors is defined */
#define HAVE_BDEV_WRITE_ZEROES_SECTORS 1

/* genhd.h has bd_set_nr_sectors */
/* #undef HAVE_BD_SET_NR_SECTORS */

/* genhd.h has bd_set_size */
/* #undef HAVE_BD_SET_SIZE */

/* bio.h has bio_add_zone_append_page */
#define HAVE_BIO_ADD_ZONE_APPEND_PAGE 1

/* linux/bio.h has bip_get_seed */
#define HAVE_BIO_BIP_GET_SEED 1

/* struct bio has member bi_bdev */
#define HAVE_BIO_BI_BDEV 1

/* struct bio has member bi_cookie */
/* #undef HAVE_BIO_BI_COOKIE */

/* struct bio has member bi_disk */
/* #undef HAVE_BIO_BI_DISK */

/* linux/bio.h bio_endio has 1 parameter */
#define HAVE_BIO_ENDIO_1_PARAM 1

/* bio_for_each_bvec is defined in bio.h */
#define HAVE_BIO_FOR_EACH_BVEC 1

/* bio.h bio_init has 3 parameters */
#define HAVE_BIO_INIT_3_PARAMS 1

/* bio.h bio_init has 5 parameters */
/* #undef HAVE_BIO_INIT_5_PARAMS */

/* bio_integrity_payload has members bip_iter */
#define HAVE_BIO_INTEGRITY_PYLD_BIP_ITER 1

/* if bio.h has bio_max_segs */
#define HAVE_BIO_MAX_SEGS 1

/* if bio.h has BIO_MAX_VECS */
#define HAVE_BIO_MAX_VECS 1

/* blkdev.h has bio_split_to_limits */
/* #undef HAVE_BIO_SPLIT_TO_LIMITS */

/* bdev_start_io_acct is defined */
/* #undef HAVE_BIO_START_IO_ACCT */

/* bitfield.h exist */
#define HAVE_BITFIELD_H 1

/* bitmap_free is defined */
#define HAVE_BITMAP_FREE 1

/* bitmap_from_arr32 is defined */
#define HAVE_BITMAP_FROM_ARR32 1

/* bitmap_kzalloc is defined */
#define HAVE_BITMAP_KZALLOC 1

/* bitmap_zalloc_node is defined */
/* #undef HAVE_BITMAP_ZALLOC_NODE */

/* include/linux/bits.h exists */
#define HAVE_BITS_H 1

/* blist_flags_t is defined */
#define HAVE_BLIST_FLAGS_T 1

/* blkcg_get_fc_appid is defined */
#define HAVE_BLKCG_GET_FC_APPID 1

/* linux/blkdev.h has bio_integrity_bytes */
#define HAVE_BLKDEV_BIO_INTEGRITY_BYTES 1

/* blkdev_compat_ptr_ioctl is defined */
#define HAVE_BLKDEV_COMPAT_PTR_IOCTL 1

/* dma_map_bvec exist */
#define HAVE_BLKDEV_DMA_MAP_BVEC 1

/* blkdev_issue_flush has 1 params */
#define HAVE_BLKDEV_ISSUE_FLUSH_1_PARAM 1

/* blkdev_issue_flush has 2 params */
/* #undef HAVE_BLKDEV_ISSUE_FLUSH_2_PARAM */

/* __blkdev_issue_zeroout exist */
#define HAVE_BLKDEV_ISSUE_ZEROOUT 1

/* blkdev_put has holder param */
/* #undef HAVE_BLKDEV_PUT_HOLDER */

/* linux/blkdev.h has QUEUE_FLAG_QUIESCED */
#define HAVE_BLKDEV_QUEUE_FLAG_QUIESCED 1

/* linux/blkdev.h has req_bvec */
#define HAVE_BLKDEV_REQ_BVEC 1

/* REQ_TYPE_DRV_PRIV is defined */
/* #undef HAVE_BLKDEV_REQ_TYPE_DRV_PRIV */

/* blkdev.h blk_add_request_payload has 4 parameters */
/* #undef HAVE_BLK_ADD_REQUEST_PAYLOAD_HAS_4_PARAMS */

/* genhd.h has blk_alloc_disk */
#define HAVE_BLK_ALLOC_DISK 1

/* blk_alloc_queue_node has 3 args */
/* #undef HAVE_BLK_ALLOC_QUEUE_NODE_3_ARGS */

/* linux/blkdev.h has blk_alloc_queue_rh */
/* #undef HAVE_BLK_ALLOC_QUEUE_RH */

/* blk_cleanup_disk() is defined */
#define HAVE_BLK_CLEANUP_DISK 1

/* BLK_EH_DONE is defined */
#define HAVE_BLK_EH_DONE 1

/* blk_execute_rq has 2 params */
/* #undef HAVE_BLK_EXECUTE_RQ_2_PARAM */

/* blk_execute_rq has 3 params */
#define HAVE_BLK_EXECUTE_RQ_3_PARAM 1

/* blk_execute_rq has 4 params */
/* #undef HAVE_BLK_EXECUTE_RQ_4_PARAM */

/* blk_execute_rq_nowait has 2 params */
/* #undef HAVE_BLK_EXECUTE_RQ_NOWAIT_2_PARAM */

/* blk_execute_rq_nowait has 3 params */
/* #undef HAVE_BLK_EXECUTE_RQ_NOWAIT_3_PARAM */

/* blk_execute_rq_nowait has 5 params */
/* #undef HAVE_BLK_EXECUTE_RQ_NOWAIT_5_PARAM */

/* blk_freeze_queue_start is defined */
#define HAVE_BLK_FREEZE_QUEUE_START 1

/* BLK_INTEGRITY_DEVICE_CAPABLE is defined */
#define HAVE_BLK_INTEGRITY_DEVICE_CAPABLE 1

/* linux/blk-integrity.h exists */
/* #undef HAVE_BLK_INTEGRITY_H */

/* struct blk_integrity has sector_size */
/* #undef HAVE_BLK_INTEGRITY_SECTOR_SIZE */

/* blk_mark_disk_dead exist */
#define HAVE_BLK_MARK_DISK_DEAD 1

/* BLK_MAX_WRITE_HINTS is defined */
#define HAVE_BLK_MAX_WRITE_HINTS 1

/* blk_mq_alloc_disk is defined */
#define HAVE_BLK_MQ_ALLOC_DISK 1

/* linux/blk-mq.h blk_mq_alloc_request has 3 parameters */
#define HAVE_BLK_MQ_ALLOC_REQUEST_HAS_3_PARAMS 1

/* linux/blk-mq.h has blk_mq_alloc_request_hctx */
#define HAVE_BLK_MQ_ALLOC_REQUEST_HCTX 1

/* blk_mq_all_tag_busy_iter is defined */
/* #undef HAVE_BLK_MQ_ALL_TAG_BUSY_ITER */

/* blk_types.h has BLK_STS_ZONE_ACTIVE_RESOURCE */
#define HAVE_BLK_MQ_BLK_STS_ZONE_ACTIVE_RESOURCE 1

/* linux/blk-mq.h has busy_tag_iter_fn return bool */
/* #undef HAVE_BLK_MQ_BUSY_TAG_ITER_FN_BOOL_2_PARAMS */

/* linux/blk-mq.h has busy_tag_iter_fn return bool */
#define HAVE_BLK_MQ_BUSY_TAG_ITER_FN_BOOL_3_PARAMS 1

/* linux/blk-mq.h blk_mq_complete_request has 2 parameters */
/* #undef HAVE_BLK_MQ_COMPLETE_REQUEST_HAS_2_PARAMS */

/* linux/blk-mq.h has blk_mq_complete_request_remote */
#define HAVE_BLK_MQ_COMPLETE_REQUEST_REMOTE 1

/* blk-mq.h has blk_mq_complete_request_sync */
/* #undef HAVE_BLK_MQ_COMPLETE_REQUEST_SYNC */

/* blk_mq_delay_kick_requeue_list is defined */
#define HAVE_BLK_MQ_DELAY_KICK_REQUEUE_LIST 1

/* blk_mq_destroy_queue is defined */
/* #undef HAVE_BLK_MQ_DESTROY_QUEUE */

/* blk_mq_end_request accepts blk_status_t as second parameter */
#define HAVE_BLK_MQ_END_REQUEST_TAKES_BLK_STATUS_T 1

/* blk_mq_freeze_queue_wait is defined */
#define HAVE_BLK_MQ_FREEZE_QUEUE_WAIT 1

/* blk_mq_freeze_queue_wait_timeout is defined */
#define HAVE_BLK_MQ_FREEZE_QUEUE_WAIT_TIMEOUT 1

/* BLK_MQ_F_NO_SCHED is defined */
#define HAVE_BLK_MQ_F_NO_SCHED 1

/* blk-mq.h has blk_mq_hctx_set_fq_lock_class */
#define HAVE_BLK_MQ_HCTX_SET_FQ_LOCK_CLASS 1

/* blk-mq.h has enum hctx_type */
#define HAVE_BLK_MQ_HCTX_TYPE 1

/* blk_mq_map_queues is defined */
#define HAVE_BLK_MQ_MAP_QUEUES 1

/* struct blk_mq_ops has commit_rqs */
#define HAVE_BLK_MQ_OPS_COMMIT_RQS 1

/* linux/blk-mq.h blk_mq_ops exit_request has 3 parameters */
#define HAVE_BLK_MQ_OPS_EXIT_REQUEST_HAS_3_PARAMS 1

/* linux/blk-mq.h blk_mq_ops init_request has 4 parameters */
#define HAVE_BLK_MQ_OPS_INIT_REQUEST_HAS_4_PARAMS 1

/* struct blk_mq_ops has map_queue */
/* #undef HAVE_BLK_MQ_OPS_MAP_QUEUE */

/* struct blk_mq_ops has map_queues */
#define HAVE_BLK_MQ_OPS_MAP_QUEUES 1

/* function map_queues returns int */
#define HAVE_BLK_MQ_OPS_MAP_QUEUES_RETURN_INT 1

/* struct blk_mq_ops has poll */
#define HAVE_BLK_MQ_OPS_POLL 1

/* struct blk_mq_ops has poll 1 arg */
#define HAVE_BLK_MQ_OPS_POLL_1_ARG 1

/* struct blk_mq_ops has poll 2 args */
/* #undef HAVE_BLK_MQ_OPS_POLL_2_ARG */

/* struct blk_mq_ops has queue_rqs */
/* #undef HAVE_BLK_MQ_OPS_QUEUE_RQS */

/* timeout from struct blk_mq_ops has 1 param */
/* #undef HAVE_BLK_MQ_OPS_TIMEOUT_1_PARAM */

/* include/linux/blk-mq-pci.h exists */
#define HAVE_BLK_MQ_PCI_H 1

/* blk_mq_pci_map_queues is defined */
#define HAVE_BLK_MQ_PCI_MAP_QUEUES_3_ARGS 1

/* blk_mq_quiesce_tagset is defined */
/* #undef HAVE_BLK_MQ_QUEIESCE_TAGSET */

/* linux/blk-mq.h has struct blk_mq_queue_map */
#define HAVE_BLK_MQ_QUEUE_MAP 1

/* blk_mq_quiesce_queue exist */
#define HAVE_BLK_MQ_QUIESCE_QUEUE 1

/* linux/blk-mq.h has blk_mq_request_completed */
#define HAVE_BLK_MQ_REQUEST_COMPLETED 1

/* blk-mq.h blk_mq_requeue_request has 2 parameters */
#define HAVE_BLK_MQ_REQUEUE_REQUEST_2_PARAMS 1

/* blk_mq_req_flags_t is defined */
#define HAVE_BLK_MQ_REQ_FLAGS_T 1

/* blk_mq_rq_state is defined */
#define HAVE_BLK_MQ_RQ_STATE 1

/* linux/blk-mq.h has blk_mq_set_request_complete */
#define HAVE_BLK_MQ_SET_REQUEST_COMPLETE 1

/* blk_mq_tagset_busy_iter is defined */
/* #undef HAVE_BLK_MQ_TAGSET_BUSY_ITER */

/* linux/blk-mq.h has blk_mq_tagset_wait_completed_request */
#define HAVE_BLK_MQ_TAGSET_WAIT_COMPLETED_REQUEST 1

/* blk_mq_tag_set member ops is const */
#define HAVE_BLK_MQ_TAG_SET_HAS_CONST_OPS 1

/* blk_mq_tag_set has member map */
#define HAVE_BLK_MQ_TAG_SET_HAS_MAP 1

/* blk_mq_tag_set has member nr_maps */
#define HAVE_BLK_MQ_TAG_SET_HAS_NR_MAP 1

/* blk_mq_unquiesce_queue is defined */
#define HAVE_BLK_MQ_UNQUIESCE_QUEUE 1

/* blk_mq_update_nr_hw_queues is defined */
#define HAVE_BLK_MQ_UPDATE_NR_HW_QUEUES 1

/* blk_mq_wait_quiesce_done is defined */
/* #undef HAVE_BLK_MQ_WAIT_QUIESCE_DONE */

/* blk_mq_wait_quiesce_done with tagset param is defined */
/* #undef HAVE_BLK_MQ_WAIT_QUIESCE_DONE_TAGSET */

/* bio.h blk_next_bio has 3 parameters */
#define HAVE_BLK_NEXT_BIO_3_PARAMS 1

/* blk_opf_t is defined */
/* #undef HAVE_BLK_OPF_T */

/* blk_path_error is defined */
#define HAVE_BLK_PATH_ERROR 1

/* blk_queue_flag_set is defined */
#define HAVE_BLK_QUEUE_FLAG_SET 1

/* blk_queue_make_request existing */
/* #undef HAVE_BLK_QUEUE_MAKE_REQUEST */

/* blk_queue_max_active_zones exist */
#define HAVE_BLK_QUEUE_MAX_ACTIVE_ZONES 1

/* blk_queue_max_write_zeroes_sectors is defined */
#define HAVE_BLK_QUEUE_MAX_WRITE_ZEROES_SECTORS 1

/* blk_queue_split has 1 param */
#define HAVE_BLK_QUEUE_SPLIT_1_PARAM 1

/* blk_queue_virt_boundary exist */
#define HAVE_BLK_QUEUE_VIRT_BOUNDARY 1

/* blkdev.h has blk_queue_write_cache */
#define HAVE_BLK_QUEUE_WRITE_CACHE 1

/* blkdev.h has blk_queue_zone_sectors */
#define HAVE_BLK_QUEUE_ZONE_SECTORS 1

/* blk_rq_append_bio is defined */
/* #undef HAVE_BLK_RQ_APPEND_BIO */

/* if blk-mq.h has blk_rq_bio_prep */
#define HAVE_BLK_RQ_BIO_PREP 1

/* blk_rq_is_passthrough is defined */
#define HAVE_BLK_RQ_IS_PASSTHROUGH 1

/* blk_rq_map_user_iv is defined */
/* #undef HAVE_BLK_RQ_MAP_USER_IO */

/* blk_rq_nr_discard_segments is defined */
#define HAVE_BLK_RQ_NR_DISCARD_SEGMENTS 1

/* blk_rq_payload_bytes exist */
#define HAVE_BLK_RQ_NR_PAYLOAD_BYTES 1

/* blk_rq_nr_phys_segments exist */
#define HAVE_BLK_RQ_NR_PHYS_SEGMENTS 1

/* linux/blk-mq.h has blk_should_fake_timeout */
#define HAVE_BLK_SHOULD_FAKE_TIMEOUT 1

/* blk_status_t is defined */
#define HAVE_BLK_STATUS_T 1

/* blk_types.h has BLK_STS_RESV_CONFLICT */
/* #undef HAVE_BLK_STS_RESV_CONFLICT */

/* REQ_DRV is defined */
#define HAVE_BLK_TYPES_REQ_DRV 1

/* REQ_HIPRI is defined */
#define HAVE_BLK_TYPES_REQ_HIPRI 1

/* REQ_INTEGRITY is defined */
#define HAVE_BLK_TYPES_REQ_INTEGRITY 1

/* REQ_NOUNMAP is defined */
#define HAVE_BLK_TYPES_REQ_NOUNMAP 1

/* enum req_opf is defined */
#define HAVE_BLK_TYPES_REQ_OPF 1

/* REQ_OP_DISCARD is defined */
#define HAVE_BLK_TYPES_REQ_OP_DISCARD 1

/* REQ_OP_FLUSH is defined */
#define HAVE_BLK_TYPES_REQ_OP_FLUSH 1

/* linux/blk_types.h has op_is_sync */
#define HAVE_BLK_TYPE_OP_IS_SYNC 1

/* linux/blkdev.h has bdev_zone_no */
/* #undef HAVE_BLK_ZONE_NO */

/* struct block_device_operations has submit_bio */
#define HAVE_BLOCK_DEVICE_OPERATIONS_SUBMIT_BIO 1

/* bpf_prog_add\bfs_prog_inc functions return struct */
/* #undef HAVE_BPF_PROG_ADD_RET_STRUCT */

/* bpf_prog_inc is exported by the kernel */
#define HAVE_BPF_PROG_INC_EXPORTED 1

/* bpf_prog_sub is defined */
#define HAVE_BPF_PROG_SUB 1

/* filter.h has bpf_warn_invalid_xdp_action get 3 params */
/* #undef HAVE_BPF_WARN_IVALID_XDP_ACTION_GET_3_PARAMS */

/* include/linux/build_bug.h exists */
#define HAVE_BUILD_BUG_H 1

/* bus_find_device get const */
#define HAVE_BUS_FIND_DEVICE_GET_CONST 1

/* bus_type remove function return void */
#define HAVE_BUS_TYPE_REMOVE_RETURN_VOID 1

/* linux/bvec.h has bvec_set_page */
/* #undef HAVE_BVEC_SET_PAGE */

/* bvec_set_virt is defined */
/* #undef HAVE_BVEC_SET_VIRT */

/* linux/bvec.h has bvec_virt */
#define HAVE_BVEC_VIRT 1

/* call_switchdev_notifiers is defined with 4 params */
#define HAVE_CALL_SWITCHDEV_NOTIFIERS_4_PARAMS 1

/* cancel_work is exported by the kernel */
/* #undef HAVE_CANCEL_WORK_EXPORTED */

/* linux/cdev.h has cdev_set_parent */
#define HAVE_CDEV_SET_PARENT 1

/* __cgroup_bpf_run_filter_sysctl have 7 parameters */
#define HAVE_CGROUP_BPF_RUN_FILTER_SYSCTL_7_PARAMETERS 1

/* linux/cgroup_rdma exists */
#define HAVE_CGROUP_RDMA_H 1

/* __check_old_set_param is defined */
/* #undef HAVE_CHECK_OLD_SET_PARAM */

/* vxlan_build_gbp_hdr is defined */
/* #undef HAVE_CHECK_VXLAN_BUILD_GBP_HDR */

/* VXLAN_GBP_MASK is defined */
#define HAVE_CHECK_VXLAN_GBP_MASK 1

/* class_create get 1 param */
/* #undef HAVE_CLASS_CREATE_GET_1_PARAM */

/* dev_uevent get const struct device */
/* #undef HAVE_CLASS_DEV_UEVENT_CONST_DEV */

/* struct class has class_groups */
#define HAVE_CLASS_GROUPS 1

/* cycle_t is defined in linux/clocksource.h */
/* #undef HAVE_CLOCKSOURCE_CYCLE_T */

/* compat_ptr_ioctl is exported by the kernel */
#define HAVE_COMPAT_PTR_IOCTL_EXPORTED 1

/* linux/compat.h has compat_uptr_t */
#define HAVE_COMPAT_UPTR_T 1

/* default_groups is list_head */
#define HAVE_CONFIGFS_DEFAULT_GROUPS_LIST 1

/* if configfs_item_operations drop_link returns int */
/* #undef HAVE_CONFIGFS_DROP_LINK_RETURNS_INT */

/* configfs.h has configfs_register_group */
#define HAVE_CONFIGFS_REGISTER_GROUP 1

/* const __read_once_size exist */
/* #undef HAVE_CONST_READ_ONCE_SIZE */

/* struct ctl_table have "child" field */
#define HAVE_CTL_TABLE_CHILD 1

/* struct dcbnl_rtnl_ops has dcbnl_get/set buffer */
#define HAVE_DCBNL_GETBUFFER 1

/* debugfs_create_file_unsafe is exported by the kernel */
#define HAVE_DEBUGFS_CREATE_FILE_UNSAFE 1

/* debugfs.h debugfs_create_ulong defined */
#define HAVE_DEBUGFS_CREATE_ULONG 1

/* debugfs.h debugfs_lookup defined */
#define HAVE_DEBUGFS_LOOKUP 1

/* DEFINE_SHOW_ATTRIBUTE is defined */
#define HAVE_DEFINE_SHOW_ATTRIBUTE 1

/* genhd.h has device_add_disk */
/* #undef HAVE_DEVICE_ADD_DISK */

/* genhd.h has device_add_disk 3 args and must_check */
#define HAVE_DEVICE_ADD_DISK_3_ARGS_AND_RETURN 1

/* genhd.h has device_add_disk */
#define HAVE_DEVICE_ADD_DISK_3_ARGS_NO_RETURN 1

/* genhd.h has device_add_disk retrun */
#define HAVE_DEVICE_ADD_DISK_RETURN 1

/* struct device has dma_ops */
#define HAVE_DEVICE_DMA_OPS 1

/* device.h has device_remove_file_self */
#define HAVE_DEVICE_REMOVE_FILE_SELF 1

/* devlink.h has devlink_alloc get 3 params */
#define HAVE_DEVLINK_ALLOC_GET_3_PARAMS 1

/* include/net/devlink.h devlink_alloc_ns defined */
#define HAVE_DEVLINK_ALLOC_NS 1

/* devlink_param_driverinit_value_get exist */
#define HAVE_DEVLINK_DRIVERINIT_VAL 1

/* struct devlink_ops.eswitch_mode_set has extack */
#define HAVE_DEVLINK_ESWITCH_MODE_SET_EXTACK 1

/* devlink_flash_update_end_notify */
/* #undef HAVE_DEVLINK_FLASH_UPDATE_END_NOTIFY */

/* devlink_flash_update_params has struct firmware fw */
#define HAVE_DEVLINK_FLASH_UPDATE_PARAMS_HAS_STRUCT_FW 1

/* devlink_flash_update_status_notify */
#define HAVE_DEVLINK_FLASH_UPDATE_STATUS_NOTIFY 1

/* devlink.h has devlink_fmsg_binary_pair_nest_start is defined */
#define HAVE_DEVLINK_FMSG_BINARY_PAIR_NEST_START 1

/* devlink_fmsg_binary_pair_put exists */
#define HAVE_DEVLINK_FMSG_BINARY_PAIR_PUT_ARG_U32_RETURN_INT 1

/* devlink_fmsg_binary_pair_put exists */
/* #undef HAVE_DEVLINK_FMSG_BINARY_PAIR_PUT_ARG_U32_RETURN_VOID */

/* devlink_fmsg_binary_put exists */
#define HAVE_DEVLINK_FMSG_BINARY_PUT 1

/* include/net/devlink.h exists */
#define HAVE_DEVLINK_H 1

/* eswitch_encap_mode_set/get is defined */
#define HAVE_DEVLINK_HAS_ESWITCH_ENCAP_MODE_SET 1

/* eswitch_encap_mode_set/get is defined with enum */
#define HAVE_DEVLINK_HAS_ESWITCH_ENCAP_MODE_SET_GET_WITH_ENUM 1

/* eswitch_inline_mode_get/set is defined */
#define HAVE_DEVLINK_HAS_ESWITCH_INLINE_MODE_GET_SET 1

/* eswitch_mode_get/set is defined */
#define HAVE_DEVLINK_HAS_ESWITCH_MODE_GET_SET 1

/* flash_update is defined */
#define HAVE_DEVLINK_HAS_FLASH_UPDATE 1

/* info_get is defined */
#define HAVE_DEVLINK_HAS_INFO_GET 1

/* port_function_roce/mig_get/set is defined */
/* #undef HAVE_DEVLINK_HAS_PORT_FN_ROCE_MIG */

/* port_function_hw_addr_get/set is defined */
#define HAVE_DEVLINK_HAS_PORT_FUNCTION_HW_ADDR_GET 1

/* port_function_state_get/set is defined */
#define HAVE_DEVLINK_HAS_PORT_FUNCTION_STATE_GET 1

/* rate functions are defined */
#define HAVE_DEVLINK_HAS_RATE_FUNCTIONS 1

/* reload is defined */
/* #undef HAVE_DEVLINK_HAS_RELOAD */

/* reload_up/down is defined */
#define HAVE_DEVLINK_HAS_RELOAD_UP_DOWN 1

/* devlink_health_reporter_create has 4 args */
#define HAVE_DEVLINK_HEALTH_REPORTER_CREATE_4_ARGS 1

/* devlink_health_reporter_create has 5 args */
/* #undef HAVE_DEVLINK_HEALTH_REPORTER_CREATE_5_ARGS */

/* devlink_health_reporter_state_update exist */
#define HAVE_DEVLINK_HEALTH_REPORTER_STATE_UPDATE 1

/* structs devlink_health_reporter & devlink_fmsg exist */
#define HAVE_DEVLINK_HEALTH_REPORT_BASE_SUPPORT 1

/* devlink.h has devlink_info_driver_name_put */
#define HAVE_DEVLINK_INFO_DRIVER_NAME_PUT 1

/* devlink_info_version_fixed_put exist */
#define HAVE_DEVLINK_INFO_VERSION_FIXED_PUT 1

/* port_fn_ipsec_crypto_get is defined */
/* #undef HAVE_DEVLINK_IPSEC_CRYPTO */

/* port_fn_ipsec_packet_get is defined */
/* #undef HAVE_DEVLINK_IPSEC_PACKET */

/* devlink_net exist */
#define HAVE_DEVLINK_NET 1

/* struct devlink_param exist */
#define HAVE_DEVLINK_PARAM 1

/* devlink_params_publish is exported by the kernel */
/* #undef HAVE_DEVLINK_PARAMS_PUBLISHED */

/* enum devlink_param_cmode exists */
#define HAVE_DEVLINK_PARAM_CMODE 1

/* devlink enum has HAVE_DEVLINK_PARAM_GENERIC_ID_ENABLE_ETH */
#define HAVE_DEVLINK_PARAM_GENERIC_ID_ENABLE_ETH 1

/* enum DEVLINK_PARAM_GENERIC_ID_ENABLE_REMOTE_DEV_RESET exist */
#define HAVE_DEVLINK_PARAM_GENERIC_ID_ENABLE_REMOTE_DEV_RESET 1

/* struct devlink_param exist */
#define HAVE_DEVLINK_PARAM_GENERIC_ID_ENABLE_ROCE 1

/* devlink enum has DEVLINK_PARAM_GENERIC_ID_IO_EQ_SIZE */
#define HAVE_DEVLINK_PARAM_GENERIC_ID_IO_EQ_SIZE 1

/* devlink enum has HAVE_DEVLINK_PARAM_GENERIC_ID_MAX */
#define HAVE_DEVLINK_PARAM_GENERIC_ID_MAX 1

/* devlink_param_publish is exported by the kernel */
/* #undef HAVE_DEVLINK_PARAM_PUBLISH */

/* devlink.h devlink_param_register defined */
#define HAVE_DEVLINK_PARAM_REGISTER 1

/* devlink_port_attrs_set has 2 parameters */
#define HAVE_DEVLINK_PORT_ATRRS_SET_GET_2_PARAMS 1

/* devlink_port_attrs_set has 5 parameters */
/* #undef HAVE_DEVLINK_PORT_ATRRS_SET_GET_5_PARAMS */

/* devlink_port_attrs_set has 7 parameters */
/* #undef HAVE_DEVLINK_PORT_ATRRS_SET_GET_7_PARAMS */

/* devlink_port_attrs_pci_pf_set has 2 params */
/* #undef HAVE_DEVLINK_PORT_ATTRS_PCI_PF_SET_2_PARAMS */

/* devlink_port_attrs_pci_pf_set has 4 params */
/* #undef HAVE_DEVLINK_PORT_ATTRS_PCI_PF_SET_4_PARAMS */

/* devlink_port_attrs_pci_pf_set has 4 params and controller num */
#define HAVE_DEVLINK_PORT_ATTRS_PCI_PF_SET_CONTROLLER_NUM 1

/* devlink.h devlink_port_attrs_pci_pf_set get 2 params */
/* #undef HAVE_DEVLINK_PORT_ATTRS_PCI_PF_SET_GET_2_PARAMS */

/* devlink.h has devlink_port_attrs_pci_sf_set get 4 params */
/* #undef HAVE_DEVLINK_PORT_ATTRS_PCI_SF_SET_GET_4_PARAMS */

/* devlink.h has devlink_port_attrs_pci_sf_set get 5 params */
#define HAVE_DEVLINK_PORT_ATTRS_PCI_SF_SET_GET_5_PARAMS 1

/* devlink.h devlink_port_attrs_pci_vf_set get 3 params */
/* #undef HAVE_DEVLINK_PORT_ATTRS_PCI_VF_SET_GET_3_PARAMS */

/* devlink_port_attrs_pci_vf_set has 5 params */
/* #undef HAVE_DEVLINK_PORT_ATTRS_PCI_VF_SET_GET_5_PARAMS */

/* devlink_port_attrs_pci_vf_set has 5 params and controller num */
#define HAVE_DEVLINK_PORT_ATTRS_PCI_VF_SET_GET_CONTROLLER_NUM 1

/* enum devlink_port_flavour exist */
#define HAVE_DEVLINK_PORT_FLAVOUR 1

/* enum DEVLINK_PORT_FLAVOUR_PCI_SF is defined */
#define HAVE_DEVLINK_PORT_FLAVOUR_PCI_SF 1

/* enum DEVLINK_PORT_FLAVOUR_VIRTUAL is defined */
#define HAVE_DEVLINK_PORT_FLAVOUR_VIRTUAL 1

/* enum devlink_port_fn_opstate exist */
#define HAVE_DEVLINK_PORT_FN_OPSTATE 1

/* enum devlink_port_fn_state exist */
#define HAVE_DEVLINK_PORT_FN_STATE 1

/* devlink_health_reporter_create is defined */
#define HAVE_DEVLINK_PORT_HEALTH_REPORTER_CREATE 1

/* devlink_port_health_reporter_destroy is defined */
#define HAVE_DEVLINK_PORT_HEALTH_REPORTER_DESTROY 1

/* devlink struct devlink_port_new_attrs exist */
#define HAVE_DEVLINK_PORT_NEW_ATTRS_STRUCT 1

/* struct devlink_port_ops exists */
/* #undef HAVE_DEVLINK_PORT_OPS */

/* devlink struct devlink_port exist */
#define HAVE_DEVLINK_PORT_STRUCT 1

/* devlink_port_type_eth_set get 1 param */
/* #undef HAVE_DEVLINK_PORT_TYPE_ETH_SET_GET_1_PARAM */

/* devlink_port_type_eth_set exist */
#define HAVE_DEVLINK_PORT_TYPE_ETH_SET_GET_2_PARAM 1

/* devlink.h has devlink_register get 1 params */
#define HAVE_DEVLINK_REGISTER_GET_1_PARAMS 1

/* devlink_reload_disable exist */
/* #undef HAVE_DEVLINK_RELOAD_DISABLE */

/* reload_down has 3 params */
/* #undef HAVE_DEVLINK_RELOAD_DOWN_HAS_3_PARAMS */

/* reload_down has 5 params */
#define HAVE_DEVLINK_RELOAD_DOWN_SUPPORT_RELOAD_ACTION 1

/* devlink_reload_enable exist */
/* #undef HAVE_DEVLINK_RELOAD_ENABLE */

/* devlink.h has devlink_resources_unregister 1 params */
#define HAVE_DEVLINK_RESOURCES_UNREGISTER_1_PARAMS 1

/* devlink.h has devlink_resources_unregister 2 params */
/* #undef HAVE_DEVLINK_RESOURCES_UNREGISTER_2_PARAMS */

/* devlink.h has devlink_resource_register_6_params */
#define HAVE_DEVLINK_RESOURCE_REGISTER_6_PARAMS 1

/* devlink.h has devlink_resource_register_8_params */
/* #undef HAVE_DEVLINK_RESOURCE_REGISTER_8_PARAMS */

/* devlink.h has devlink_set_features */
#define HAVE_DEVLINK_SET_FEATURES 1

/* devlink.h has devlink_to_dev */
#define HAVE_DEVLINK_TO_DEV 1

/* devlink_ops.trap_action_set has 4 args */
#define HAVE_DEVLINK_TRAP_ACTION_SET_4_ARGS 1

/* devlink has DEVLINK_TRAP_GENERIC_ID_DMAC_FILTER */
#define HAVE_DEVLINK_TRAP_DMAC_FILTER 1

/* devlink has devlink_trap_groups_register */
#define HAVE_DEVLINK_TRAP_GROUPS_REGISTER 1

/* devlink has DEVLINK_TRAP_GROUP_GENERIC with 2 args */
#define HAVE_DEVLINK_TRAP_GROUP_GENERIC_2_ARGS 1

/* devlink_trap_report has 5 args */
#define HAVE_DEVLINK_TRAP_REPORT_5_ARGS 1

/* devlink struct devlink_trap exists */
#define HAVE_DEVLINK_TRAP_SUPPORT 1

/* devlink.h has devl_health_reporter_create */
/* #undef HAVE_DEVL_HEALTH_REPORTER_CREATE */

/* devlink.h has devl_param_driverinit_value_get */
/* #undef HAVE_DEVL_PARAM_DRIVERINIT_VALUE_GET */

/* devlink.h has devl_port_health_reporter_create */
/* #undef HAVE_DEVL_PORT_HEALTH_REPORTER_CREATE */

/* devlink.h devl_port_register defined */
#define HAVE_DEVL_PORT_REGISTER 1

/* devl_rate_leaf_create 3 param */
/* #undef HAVE_DEVL_RATE_LEAF_CREATE_GET_3_PARAMS */

/* devlink.h has devl_register */
/* #undef HAVE_DEVL_REGISTER */

/* devlink.h has devl_resources_unregister */
#define HAVE_DEVL_RESOURCES_UNREGISTER 1

/* devlink.h has devl_resource_register */
#define HAVE_DEVL_RESOURCE_REGISTER 1

/* devlink.h devl_trap_groups_register defined */
#define HAVE_DEVL_TRAP_GROUPS_REGISTER 1

/* devnode get const struct device */
/* #undef HAVE_DEVNODE_GET_CONST_DEVICE */

/* function dev_addr_mod exists */
#define HAVE_DEV_ADDR_MOD 1

/* dev_change_flags has 3 parameters */
#define HAVE_DEV_CHANGE_FLAGS_HAS_3_PARAMS 1

/* dev_page_is_reusable is defined */
#define HAVE_DEV_PAGE_IS_REUSABLE 1

/* pm_domain.h has dev_pm_domain_attach */
#define HAVE_DEV_PM_DOMAIN_ATTACH 1

/* set_latency_tolerance is defined */
#define HAVE_DEV_PM_INFO_SET_LATENCY_TOLERANCE 1

/* DEV_PM_QOS_RESUME_LATENCY is defined */
#define HAVE_DEV_PM_QOS_RESUME_LATENCY 1

/* dev_xdp_prog_id is defined */
#define HAVE_DEV_XDP_PROG_ID 1

/* disk_set_zoned is defined */
/* #undef HAVE_DISK_SET_ZONED */

/* disk_uevent exist */
#define HAVE_DISK_UEVENT 1

/* disk_update_readahead exists */
#define HAVE_DISK_UPDATE_READAHEAD 1

/* dma-attrs.h has struct dma_attrs */
/* #undef HAVE_DMA_ATTRS */

/* DMA_ATTR_NO_WARN is defined */
#define HAVE_DMA_ATTR_NO_WARN 1

/* struct dma_buf_attach_ops has allow_peer2peer */
#define HAVE_DMA_BUF_ATTACH_OPS_ALLOW_PEER2PEER 1

/* dma_buf_dynamic_attach get 4 params */
#define HAVE_DMA_BUF_DYNAMIC_ATTACH_GET_4_PARAMS 1

/* linux/dma-map-ops.h has DMA_F_PCI_P2PDMA_SUPPORTED */
/* #undef HAVE_DMA_F_PCI_P2PDMA_SUPPORTED */

/* dma-mapping.h has dma_map_sgtable */
#define HAVE_DMA_MAP_SGTABLE 1

/* linux/dma-mapping.h has dma_max_mapping_size */
#define HAVE_DMA_MAX_MAPPING_SIZE 1

/* dma_opt_mapping_size is defined */
/* #undef HAVE_DMA_OPT_MAPPING_SIZE */

/* linux/dma-mapping.h has dma_pci_p2pdma_supported */
/* #undef HAVE_DMA_PCI_P2PDMA_SUPPORTED */

/* dma_pool_zalloc is defined */
#define HAVE_DMA_POOL_ZALLOC 1

/* linux/dma-resv.h has dma_resv_excl_fence */
#define HAVE_DMA_RESV_EXCL_FENCE 1

/* linux/dma-resv.h exists */
#define HAVE_DMA_RESV_H 1

/* linux/dma-resv.h has DMA_RESV_USAGE_KERNEL */
/* #undef HAVE_DMA_RESV_USAGE_KERNEL */

/* linux/dma-resv.h has dma_resv_wait_timeout */
#define HAVE_DMA_RESV_WAIT_TIMEOUT 1

/* dma_alloc_attrs takes unsigned long attrs */
#define HAVE_DMA_SET_ATTR_TAKES_UNSIGNED_LONG_ATTRS 1

/* dma_set_min_align_mask is defined in dma-mapping */
#define HAVE_DMA_SET_MIN_ALIGN_MASK 1

/* dma-mapping.h has dma_zalloc_coherent function */
/* #undef HAVE_DMA_ZALLOC_COHERENT */

/* elfcorehdr_addr is exported by the kernel */
#define HAVE_ELFCOREHDR_ADDR_EXPORTED 1

/* struct enum has member BIO_REMAPPED */
#define HAVE_ENUM_BIO_REMAPPED 1

/* enum flow_block_binder_type exists */
#define HAVE_ENUM_FLOW_BLOCK_BINDER_TYPE 1

/* enum scsi_scan_mode is defined */
#define HAVE_ENUM_SCSI_SCAN_MODE 1

/* enum tc_htb_command is defined */
#define HAVE_ENUM_TC_HTB_COMMAND 1

/* esp_output_fill_trailer is defined */
#define HAVE_ESP_OUTPUT_FILL_TRAILER 1

/* ethtool supprts 25G,50G,100G link speeds */
#define HAVE_ETHTOOL_25G_50G_100G_SPEEDS 1

/* ethtool supprts 50G-pre-lane link modes */
#define HAVE_ETHTOOL_50G_PER_LANE_LINK_MODES 1

/* get/set_rxfh_context is defined */
#define HAVE_ETHTOOL_GET_RXFH_CONTEXT 1

/* get/set_settings is defined */
/* #undef HAVE_ETHTOOL_GET_SET_SETTINGS */

/* linux/ethtool_netlink.h exists */
#define HAVE_ETHTOOL_NETLINK_H 1

/* ethtool_pause_stats is defined */
#define HAVE_ETHTOOL_PAUSE_STATS 1

/* ethtool_rmon_hist_range is defined */
#define HAVE_ETHTOOL_RMON_HIST_RANGE 1

/* eth_get_headlen is defined with 2 params */
/* #undef HAVE_ETH_GET_HEADLEN_2_PARAMS */

/* eth_get_headlen is defined with 3 params */
#define HAVE_ETH_GET_HEADLEN_3_PARAMS 1

/* ETH_MAX_MTU exists */
#define HAVE_ETH_MAX_MTU 1

/* ETH_MIN_MTU exists */
#define HAVE_ETH_MIN_MTU 1

/* ext_pi_ref_tag is defined */
/* #undef HAVE_EXT_PI_REF_TAG */

/* FC_APPID_LEN is defined */
#define HAVE_FC_APPID_LEN 1

/* struct fib6_entry_notifier_info exists */
#define HAVE_FIB6_ENTRY_NOTIFIER_INFO 1

/* struct fib6_entry_notifier_info has member struct fib6_info */
#define HAVE_FIB6_INFO_IN_FIB6_ENTRY_NOTIFIER_INFO 1

/* function fib6_info_nh_dev exists */
#define HAVE_FIB6_INFO_NH_DEV 1

/* function fib_info_nh exists */
#define HAVE_FIB_INFO_NH 1

/* fib_lookup has 4 params */
#define HAVE_FIB_LOOKUP_4_PARAMS 1

/* fib_lookup is exported by the kernel */
/* #undef HAVE_FIB_LOOKUP_EXPORTED */

/* fib_nh has fib_nh_dev */
#define HAVE_FIB_NH_DEV 1

/* fib_nh_notifier_info is defined */
#define HAVE_FIB_NH_NOTIFIER_INFO 1

/* has net/fib_notifier.h */
#define HAVE_FIB_NOTIFIER_HEADER_FILE 1

/* struct fib_notifier_info has member family */
#define HAVE_FIB_NOTIFIER_INFO_HAS_FAMILY 1

/* sruct file has f_iocb_flags */
/* #undef HAVE_FILE_F_IOCB_FLAGS */

/* uring_cmd is defined in file_operations */
/* #undef HAVE_FILE_OPERATIONS_URING_CMD */

/* uring_cmd_iopoll is defined in file_operations */
/* #undef HAVE_FILE_OPERATIONS_URING_CMD_IOPOLL */

/* struct devlink_ops flash_update get 3 params */
#define HAVE_FLASH_UPDATE_GET_3_PARAMS 1

/* tcf_queue_work has 2 params per prio */
#define HAVE_FLOWER_MULTI_MASK 1

/* FLOW_ACTION_CONTINUE exists */
/* #undef HAVE_FLOW_ACTION_CONTINUE */

/* FLOW_ACTION_CT exists */
#define HAVE_FLOW_ACTION_CT 1

/* struct flow_action_entry has ct_metadata.orig_dir */
#define HAVE_FLOW_ACTION_CT_METADATA_ORIG_DIR 1

/* net/flow_offload.h struct flow_action_entry has act_cookie */
/* #undef HAVE_FLOW_ACTION_ENTRY_ACT_COOKIE */

/* net/flow_offload.h struct flow_action_entry has act pointer */
#define HAVE_FLOW_ACTION_ENTRY_ACT_POINTER 1

/* net/flow_offload.h struct flow_action_entry has cookie */
/* #undef HAVE_FLOW_ACTION_ENTRY_COOKIE */

/* net/flow_offload.h struct flow_action_entry has hw_index */
/* #undef HAVE_FLOW_ACTION_ENTRY_HW_INDEX */

/* net/flow_offload.h struct flow_action_entry has miss_cookie */
#define HAVE_FLOW_ACTION_ENTRY_MISS_COOKIE 1

/* struct flow_action_entry has hw_index */
/* #undef HAVE_FLOW_ACTION_HW_INDEX */

/* flow_action_hw_stats_check exists */
#define HAVE_FLOW_ACTION_HW_STATS_CHECK 1

/* FLOW_ACTION_JUMP and PIPE exists */
/* #undef HAVE_FLOW_ACTION_JUMP_AND_PIPE */

/* struct flow_action_entry has mpls */
#define HAVE_FLOW_ACTION_MPLS 1

/* FLOW_ACTION_POLICE exists */
#define HAVE_FLOW_ACTION_POLICE 1

/* struct flow_action_entry has police.exceed */
/* #undef HAVE_FLOW_ACTION_POLICE_EXCEED */

/* struct flow_action_entry has police.index */
#define HAVE_FLOW_ACTION_POLICE_INDEX 1

/* struct flow_action_entry has police.rate_pkt_ps */
#define HAVE_FLOW_ACTION_POLICE_RATE_PKT_PS 1

/* FLOW_ACTION_PRIORITY exists */
#define HAVE_FLOW_ACTION_PRIORITY 1

/* struct flow_action_entry has ptype */
#define HAVE_FLOW_ACTION_PTYPE 1

/* FLOW_ACTION_REDIRECT_INGRESS exists */
#define HAVE_FLOW_ACTION_REDIRECT_INGRESS 1

/* FLOW_ACTION_VLAN_PUSH_ETH exists */
/* #undef HAVE_FLOW_ACTION_VLAN_PUSH_ETH */

/* struct flow_block_cb exist */
#define HAVE_FLOW_BLOCK_CB 1

/* flow_block_cb_alloc is defined */
#define HAVE_FLOW_BLOCK_CB_ALLOC 1

/* flow_block_cb_setup_simple is defined */
#define HAVE_FLOW_BLOCK_CB_SETUP_SIMPLE 1

/* struct flow_block_offload exists */
#define HAVE_FLOW_BLOCK_OFFLOAD 1

/* struct flow_cls_offload exists */
#define HAVE_FLOW_CLS_OFFLOAD 1

/* flow_cls_offload_flow_rule is defined */
#define HAVE_FLOW_CLS_OFFLOAD_FLOW_RULE 1

/* FLOW_DISSECTOR_F_STOP_BEFORE_ENCAP is defined */
#define HAVE_FLOW_DISSECTOR_F_STOP_BEFORE_ENCAP 1

/* flow_dissector.h exist */
#define HAVE_FLOW_DISSECTOR_H 1

/* FLOW_DISSECTOR_KEY_CVLAN is defined */
#define HAVE_FLOW_DISSECTOR_KEY_CVLAN 1

/* flow_dissector.h has FLOW_DISSECTOR_KEY_ENC_CONTROL */
#define HAVE_FLOW_DISSECTOR_KEY_ENC_CONTROL 1

/* flow_dissector.h has FLOW_DISSECTOR_KEY_ENC_IP */
#define HAVE_FLOW_DISSECTOR_KEY_ENC_IP 1

/* flow_dissector.h has FLOW_DISSECTOR_KEY_ENC_KEYID */
#define HAVE_FLOW_DISSECTOR_KEY_ENC_KEYID 1

/* FLOW_DISSECTOR_KEY_ENC_OPTS is defined */
#define HAVE_FLOW_DISSECTOR_KEY_ENC_OPTS 1

/* FLOW_DISSECTOR_KEY_IP is defined */
#define HAVE_FLOW_DISSECTOR_KEY_IP 1

/* FLOW_DISSECTOR_KEY_META is defined */
#define HAVE_FLOW_DISSECTOR_KEY_META 1

/* flow_dissector.h has FLOW_DISSECTOR_KEY_MPLS */
#define HAVE_FLOW_DISSECTOR_KEY_MPLS 1

/* FLOW_DISSECTOR_KEY_TCP is defined */
#define HAVE_FLOW_DISSECTOR_KEY_TCP 1

/* FLOW_DISSECTOR_KEY_VLAN is defined */
#define HAVE_FLOW_DISSECTOR_KEY_VLAN 1

/* struct flow_dissector_key_vlan has vlan_eth_type */
#define HAVE_FLOW_DISSECTOR_KEY_VLAN_ETH_TYPE 1

/* flow_dissector.h has struct flow_dissector_mpls_lse */
#define HAVE_FLOW_DISSECTOR_MPLS_LSE 1

/* flow_dissector.h has dissector_uses_key */
#define HAVE_FLOW_DISSECTOR_USES_KEY 1

/* flow_indr_block_bind_cb_t has 4 parameters */
/* #undef HAVE_FLOW_INDR_BLOCK_BIND_CB_T_4_PARAMS */

/* flow_indr_block_bind_cb_t has 7 parameters */
#define HAVE_FLOW_INDR_BLOCK_BIND_CB_T_7_PARAMS 1

/* flow_indr_block_cb_alloc exist */
#define HAVE_FLOW_INDR_BLOCK_CB_ALLOC 1

/* flow_indr_dev_register exists */
#define HAVE_FLOW_INDR_DEV_REGISTER 1

/* flow_indr_dev_unregister receive flow_setup_cb_t parameter */
/* #undef HAVE_FLOW_INDR_DEV_UNREGISTER_FLOW_SETUP_CB_T */

/* HAVE_FLOW_OFFLOAD_ACTION exists */
/* #undef HAVE_FLOW_OFFLOAD_ACTION */

/* flow_offload_has_one_action exists */
#define HAVE_FLOW_OFFLOAD_HAS_ONE_ACTION 1

/* uapi ethtool has FLOW_RSS */
#define HAVE_FLOW_RSS 1

/* flow_rule_match_cvlan is exported by the kernel */
#define HAVE_FLOW_RULE_MATCH_CVLAN 1

/* flow_rule_match_meta exists */
#define HAVE_FLOW_RULE_MATCH_META 1

/* flow_setup_cb_t is defined */
#define HAVE_FLOW_SETUP_CB_T 1

/* flow_stats_update has 5 parameters */
/* #undef HAVE_FLOW_STATS_UPDATE_5_PARAMS */

/* flow_stats_update has 6 parameters */
#define HAVE_FLOW_STATS_UPDATE_6_PARAMS 1

/* fs.h has FMODE_NOWAIT */
#define HAVE_FMODE_NOWAIT 1

/* FOLL_LONGTERM is defined */
#define HAVE_FOLL_LONGTERM 1

/* for_ifa defined */
/* #undef HAVE_FOR_IFA */

/* struct kiocb is defined in linux/fs.h */
#define HAVE_FS_HAS_KIOCB 1

/* linux/fs.h has struct kiocb ki_complete 2 args */
/* #undef HAVE_FS_KIOCB_KI_COMPLETE_2_ARG */

/* net/macsec.c has function macsec_get_real_dev */
/* #undef HAVE_FUNC_MACSEC_GET_REAL_DEV */

/* net/macsec.c has function macsec_netdev_is_offloaded */
/* #undef HAVE_FUNC_MACSEC_NETDEV_IS_OFFLOADED */

/* struct gendisk has conv_zones_bitmap */
/* #undef HAVE_GENDISK_CONV_ZONES_BITMAP */

/* struct gendisk has open_mode */
/* #undef HAVE_GENDISK_OPEN_MODE */

/* genhd.h has GENHD_FL_EXT_DEVT */
#define HAVE_GENHD_FL_EXT_DEVT 1

/* genhd.h has GENHD_FL_UP */
/* #undef HAVE_GENHD_FL_UP */

/* struct genl_family has member policy */
#define HAVE_GENL_FAMILY_POLICY 1

/* struct genl_family has member resv_start_op */
/* #undef HAVE_GENL_FAMILY_RESV_START_OP */

/* struct genl_ops has member validate */
#define HAVE_GENL_OPS_VALIDATE 1

/* gettimex64 is defined */
#define HAVE_GETTIMEX64 1

/* .get_link_ext_state is defined */
#define HAVE_GET_LINK_EXT_STATE 1

/* ethtool_ops has get_module_eeprom_by_page */
#define HAVE_GET_MODULE_EEPROM_BY_PAGE 1

/* get_pause_stats is defined */
#define HAVE_GET_PAUSE_STATS 1

/* get_pid_task is exported by the kernel */
#define HAVE_GET_PID_TASK_EXPORTED 1

/* get_random_u32 defined */
#define HAVE_GET_RANDOM_U32 1

/* get_random_u32_inclusive defined */
/* #undef HAVE_GET_RANDOM_U32_INCLUSIVE */

/* ndo_get_ringparam get 4 parameters */
/* #undef HAVE_GET_RINGPARAM_GET_4_PARAMS */

/* get/set_fecparam is defined */
#define HAVE_GET_SET_FECPARAM 1

/* get/set_link_ksettings is defined */
#define HAVE_GET_SET_LINK_KSETTINGS 1

/* get/set_tunable is defined */
#define HAVE_GET_SET_TUNABLE 1

/* get_task_comm is exported by the kernel */
/* #undef HAVE_GET_TASK_COMM_EXPORTED */

/* get_task_pid is exported by the kernel */
#define HAVE_GET_TASK_PID_EXPORTED 1

/* get_user_pages has 4 params */
/* #undef HAVE_GET_USER_PAGES_4_PARAMS */

/* get_user_pages has 5 params */
#define HAVE_GET_USER_PAGES_5_PARAMS 1

/* get_user_pages has 7 params */
/* #undef HAVE_GET_USER_PAGES_7_PARAMS */

/* get_user_pages has 8 params */
/* #undef HAVE_GET_USER_PAGES_8_PARAMS */

/* get_user_pages_longterm is defined */
/* #undef HAVE_GET_USER_PAGES_LONGTERM */

/* get_user_pages_remote is defined with 7 parameters */
/* #undef HAVE_GET_USER_PAGES_REMOTE_7_PARAMS */

/* get_user_pages_remote is defined with 7 parameters and parameter 2 is
   integer */
#define HAVE_GET_USER_PAGES_REMOTE_7_PARAMS_AND_SECOND_INT 1

/* get_user_pages_remote is defined with 8 parameters */
/* #undef HAVE_GET_USER_PAGES_REMOTE_8_PARAMS */

/* get_user_pages_remote is defined with 8 parameters with locked */
/* #undef HAVE_GET_USER_PAGES_REMOTE_8_PARAMS_W_LOCKED */

/* GRO_LEGACY_MAX_SIZE defined */
/* #undef HAVE_GRO_LEGACY_MAX_SIZE */

/* GRO_MAX_SIZE defined */
/* #undef HAVE_GRO_MAX_SIZE */

/* guid_parse is defined */
#define HAVE_GUID_PARSE 1

/* gfpflags_allow_blocking is defined */
#define HAVE_HAS_GFPFLAGES_ALLOW_BLOCKING 1

/* __GFP_DIRECT_RECLAIM is defined */
#define HAVE_HAS_GFP_DIRECT_RECLAIM 1

/* devlink_health_reporter_ops has diagnose */
#define HAVE_HEALTH_REPORTER_DIAGNOSE 1

/* devlink_health_reporter_ops.recover has extack */
#define HAVE_HEALTH_REPORTER_RECOVER_HAS_EXTACK 1

/* have hmm_pfn_to_map_order */
#define HAVE_HMM_PFN_TO_MAP_ORDER 1

/* hmm_range_fault has one param */
#define HAVE_HMM_RANGE_FAULT_HAS_ONE_PARAM 1

/* hmm_range has hmm_pfns */
#define HAVE_HMM_RANGE_HAS_HMM_PFNS 1

/* hwmon.h hwmon_device_register_with_info exist */
#define HAVE_HWMON_DEVICE_REGISTER_WITH_INFO 1

/* hwmon.h hwmon_ops has read_string */
#define HAVE_HWMON_OPS_READ_STRING 1

/* hwmon.h hwmon_ops read_string get const str */
#define HAVE_HWMON_READ_STRING_CONST_STR 1

/* HWTSTAMP_FILTER_NTP_ALL is defined */
#define HAVE_HWTSTAMP_FILTER_NTP_ALL 1

/* rdma/ib_umem.h ib_umem_dmabuf_get_pinned defined */
/* #undef HAVE_IB_UMEM_DMABUF_GET_PINNED */

/* ida_alloc is defined */
#define HAVE_IDA_ALLOC 1

/* ida_alloc_max is defined */
#define HAVE_IDA_ALLOC_MAX 1

/* idr.h has ida_alloc_range */
#define HAVE_IDA_ALLOC_RANGE 1

/* idr.h has ida_free */
#define HAVE_IDA_FREE 1

/* ida_is_empty is defined */
#define HAVE_IDA_IS_EMPTY 1

/* idr_get_next_ul is exported by the kernel */
#define HAVE_IDR_GET_NEXT_UL_EXPORTED 1

/* idr_is_empty is defined */
/* #undef HAVE_IDR_IS_EMPTY */

/* idr_preload is exported by the kernel */
#define HAVE_IDR_PRELOAD_EXPORTED 1

/* idr_remove return value exists */
#define HAVE_IDR_REMOVE_RETURN_VALUE 1

/* struct idr has idr_rt */
#define HAVE_IDR_RT 1

/* ieee_getqcn is defined */
#define HAVE_IEEE_GETQCN 1

/* struct ifla_vf_guid is defined */
#define HAVE_IFLA_VF_GUID 1

/* trust is defined */
#define HAVE_IFLA_VF_IB_NODE_PORT_GUID 1

/* struct ifla_vf_stats is defined */
#define HAVE_IFLA_VF_STATS 1

/* inet_addr_is_any is defined */
#define HAVE_INET_ADDR_IS_ANY 1

/* inet_confirm_addr has 5 parameters */
#define HAVE_INET_CONFIRM_ADDR_5_PARAMS 1

/* inet_confirm_addr is exported by the kernel */
#define HAVE_INET_CONFIRM_ADDR_EXPORTED 1

/* include/linux/inet_lro.h exists */
/* #undef HAVE_INET_LRO_H */

/* inet_pton_with_scope is defined */
#define HAVE_INET_PTON_WITH_SCOPE 1

/* netdev_lag_upper_info has hash_type */
#define HAVE_INFO_HASH_TYPE 1

/* fs.h has inode_lock */
#define HAVE_INODE_LOCK 1

/* interval_tree functions exported by the kernel */
#define HAVE_INTERVAL_TREE_EXPORTED 1

/* INTERVAL_TREE takes rb_root */
/* #undef HAVE_INTERVAL_TREE_TAKES_RB_ROOT */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* devlink_fmsg_u8_pair_put returns int */
#define HAVE_INT_DEVLINK_FMSG_U8_PAIR 1

/* int_pow defined */
#define HAVE_INT_POW 1

/* invalidate_page defined */
/* #undef HAVE_INVALIDATE_PAGE */

/* linux/compat.h has in_compat_syscall */
#define HAVE_IN_COMPAT_SYSCALL 1

/* fs.h has IOCB_NOWAIT */
#define HAVE_IOCB_NOWAIT 1

/* iov_iter_is_bvec is defined */
#define HAVE_IOV_ITER_IS_BVEC_SET 1

/* can include linux/io-64-nonatomic-hi-lo.h */
#define HAVE_IO_64_NONATOMIC_HI_LO_H 1

/* linux/io-64-nonatomic-lo-hi.h exists */
#define HAVE_IO_64_NONATOMIC_LO_HI_H 1

/* io_uring_cmd exists */
/* #undef HAVE_IO_URING_CMD */

/* struct io_uring_cmd has cookie */
/* #undef HAVE_IO_URING_CMD_COOKIE */

/* io_uring_cmd_done has 4 params */
/* #undef HAVE_IO_URING_CMD_DONE_4_PARAMS */

/* can include linux/io_uring.h */
#define HAVE_IO_URING_H 1

/* io_uring_sqe_cmd is defined */
/* #undef HAVE_IO_URING_SQE_CMD */

/* ip6_dst_hoplimit is exported by the kernel */
#define HAVE_IP6_DST_HOPLIMIT 1

/* ip6_make_flowinfo is defined */
#define HAVE_IP6_MAKE_FLOWINFO 1

/* IP6_ECN_set_ce has 2 parameters */
#define HAVE_IP6_SET_CE_2_PARAMS 1

/* netns_ipv4 tcp_death_row memebr is not pointer */
/* #undef HAVE_IPV4_NOT_POINTER_TCP_DEATH_ROW */

/* if ipv6_stub has ipv6_dst_lookup_flow */
#define HAVE_IPV6_DST_LOOKUP_FLOW 1

/* if ipv6_stub has ipv6_dst_lookup_flow in addrconf.h */
#define HAVE_IPV6_DST_LOOKUP_FLOW_ADDR_CONF 1

/* ipv6_dst_lookup takes net */
/* #undef HAVE_IPV6_DST_LOOKUP_TAKES_NET */

/* ipv6_mod_enabled is defined */
#define HAVE_IPV6_MOD_ENABLED 1

/* net/ipv6_stubs.h exists */
#define HAVE_IPV6_STUBS_H 1

/* uapi ethtool has IPV6_USER_FLOW */
#define HAVE_IPV6_USER_FLOW 1

/* net/ip.h has ip_sock_set_tos */
#define HAVE_IP_SOCK_SET_TOS 1

/* struct ip_tunnel_info is defined */
#define HAVE_IP_TUNNEL_INFO 1

/* ip_tunnel_info_opts_set has 4 params */
#define HAVE_IP_TUNNEL_INFO_OPTS_SET_4_PARAMS 1

/* struct irq_affinity has priv */
#define HAVE_IRQ_AFFINITY_PRIV 1

/* irq_calc_affinity_vectors is defined */
#define HAVE_IRQ_CALC_AFFINITY_VECTORS_3_ARGS 1

/* irq_data member affinity is defined */
/* #undef HAVE_IRQ_DATA_AFFINITY */

/* irq_get_affinity_mask is defined */
#define HAVE_IRQ_GET_AFFINITY_MASK 1

/* irq_get_effective_affinity_mask is defined */
#define HAVE_IRQ_GET_EFFECTIVE_AFFINITY_MASK 1

/* include/linux/irq_poll.h exists */
#define HAVE_IRQ_POLL_H 1

/* irq_set_affinity_and_hint is defined */
/* #undef HAVE_IRQ_UPDATE_AFFINITY_HINT */

/* iscsi_target_core.h has struct iscsit_cmd */
/* #undef HAVE_ISCSIT_CMD */

/* iscsi_target_core.h has struct iscsit_conn */
/* #undef HAVE_ISCSIT_CONN */

/* iscsit_conn has members local_sockaddr */
/* #undef HAVE_ISCSIT_CONN_LOCAL_SOCKADDR */

/* iscsit_conn has member login_sockaddr */
/* #undef HAVE_ISCSIT_CONN_LOGIN_SOCKADDR */

/* iscsit_set_unsolicited_dataout is defined */
#define HAVE_ISCSIT_SET_UNSOLICITED_DATAOUT 1

/* iscsit_get_rx_pdu is defined */
#define HAVE_ISCSIT_TRANSPORT_ISCSIT_GET_RX_PDU 1

/* rdma_shutdown is defined */
#define HAVE_ISCSIT_TRANSPORT_RDMA_SHUTDOWN 1

/* libiscsi.h has struct iscsi_cmd */
/* #undef HAVE_ISCSI_CMD */

/* iscsi_conn has members local_sockaddr */
#define HAVE_ISCSI_CONN_LOCAL_SOCKADDR 1

/* iscsi_conn has member login_sockaddr */
#define HAVE_ISCSI_CONN_LOGIN_SOCKADDR 1

/* iscsi_conn_unbind is defined */
#define HAVE_ISCSI_CONN_UNBIND 1

/* iscsi_eh_cmd_timed_out is defined */
#define HAVE_ISCSI_EH_CMD_TIMED_OUT 1

/* libiscsi.h iscsi_host_remove has 2 parameters */
#define HAVE_ISCSI_HOST_REMOVE_2_PARAMS 1

/* iscsi_put_endpoint is defined */
#define HAVE_ISCSI_PUT_ENDPOINT 1

/* is_cow_mapping is defined */
#define HAVE_IS_COW_MAPPING 1

/* is_pci_p2pdma_page is defined */
/* #undef HAVE_IS_PCI_P2PDMA_PAGE_IN_MEMREMAP_H */

/* is_pci_p2pdma_page is defined */
#define HAVE_IS_PCI_P2PDMA_PAGE_IN_MM_H 1

/* is_tcf_csum is defined */
#define HAVE_IS_TCF_CSUM 1

/* __is_tcf_gact_act is defined with 3 variables */
#define HAVE_IS_TCF_GACT_ACT 1

/* __is_tcf_gact_act is defined with 2 variables */
/* #undef HAVE_IS_TCF_GACT_ACT_OLD */

/* is_tcf_gact_goto_chain is defined */
#define HAVE_IS_TCF_GACT_GOTO_CHAIN 1

/* is_tcf_gact_ok is defined */
#define HAVE_IS_TCF_GACT_OK 1

/* is_tcf_gact_shot is defined */
#define HAVE_IS_TCF_GACT_SHOT 1

/* is_tcf_mirred_egress_mirror is defined */
#define HAVE_IS_TCF_MIRRED_EGRESS_MIRROR 1

/* is_tcf_mirred_egress_redirect is defined */
#define HAVE_IS_TCF_MIRRED_EGRESS_REDIRECT 1

/* is_tcf_mirred_mirror is defined */
/* #undef HAVE_IS_TCF_MIRRED_MIRROR */

/* is_tcf_mirred_redirect is defined */
/* #undef HAVE_IS_TCF_MIRRED_REDIRECT */

/* is_tcf_police is defined */
#define HAVE_IS_TCF_POLICE 1

/* is_tcf_skbedit_mark is defined */
#define HAVE_IS_TCF_SKBEDIT_MARK 1

/* is_tcf_tunnel_set and is_tcf_tunnel_release are defined */
#define HAVE_IS_TCF_TUNNEL 1

/* is_tcf_vlan is defined */
#define HAVE_IS_TCF_VLAN 1

/* is_vlan_dev get const */
#define HAVE_IS_VLAN_DEV_CONST 1

/* ITER_DEST is defined */
/* #undef HAVE_ITER_DEST */

/* kcalloc_node is defined */
#define HAVE_KCALLOC_NODE 1

/* linux/net.h has kernel_getsockname 2 parameters */
#define HAVE_KERNEL_GETSOCKNAME_2_PARAMS 1

/* ethtool.h kernel_ethtool_ringparam has tcp_data_split member */
/* #undef HAVE_KERNEL_RINGPARAM_TCP_DATA_SPLIT */

/* kfree_const is defined */
#define HAVE_KFREE_CONST 1

/* function kfree_rcu_mightsleep is defined */
/* #undef HAVE_KFREE_RCU_MIGHTSLEEP */

/* kmalloc_array_node is defined */
#define HAVE_KMALLOC_ARRAY_NODE 1

/* string.h has kmemdup_nul */
#define HAVE_KMEMDUP_NUL 1

/* kobj_ns_grab_current is exported by the kernel */
#define HAVE_KOBJ_NS_GRAB_CURRENT_EXPORTED 1

/* linux/kobject.h kobj_type has default_groups member */
#define HAVE_KOBJ_TYPE_DEFAULT_GROUPS 1

/* kref_read is defined */
#define HAVE_KREF_READ 1

/* kstrtobool is defined */
#define HAVE_KSTRTOBOOL 1

/* kstrtox.h exist */
#define HAVE_KSTRTOX_H 1

/* ktime_get_ns defined */
#define HAVE_KTIME_GET_NS 1

/* ktime is union and has tv64 */
/* #undef HAVE_KTIME_UNION_TV64 */

/* ktls related structs exists */
#define HAVE_KTLS_STRUCTS 1

/* kvcalloc is defined */
#define HAVE_KVCALLOC 1

/* function kvfree_call_rcu is defined */
#define HAVE_KVFREE_CALL_RCU 1

/* kvmalloc is defined */
#define HAVE_KVMALLOC 1

/* kvmalloc_array is defined */
#define HAVE_KVMALLOC_ARRAY 1

/* kvmalloc_node is defined */
#define HAVE_KVMALLOC_NODE 1

/* kvzalloc is defined */
#define HAVE_KVZALLOC 1

/* kvzalloc_node is defined */
#define HAVE_KVZALLOC_NODE 1

/* enum netdev_lag_tx_type is defined */
#define HAVE_LAG_TX_TYPE 1

/* uapi/bpf.h exists */
#define HAVE_LINUX_BPF_H 1

/* linux/bpf_trace exists */
#define HAVE_LINUX_BPF_TRACE_H 1

/* include/linux/count_zeros.h exists */
#define HAVE_LINUX_COUNT_ZEROS_H 1

/* linux/device/bus.h exists */
#define HAVE_LINUX_DEVICE_BUS_H 1

/* uapi/linux/mei_uuid.h is exists */
/* #undef HAVE_LINUX_MEI_UUID_H */

/* linux/nvme-fc-driver.h exists */
#define HAVE_LINUX_NVME_FC_DRIVER_H 1

/* linux/overflow.h is defined */
#define HAVE_LINUX_OVERFLOW_H 1

/* linux/sed-opal.h exists */
/* #undef HAVE_LINUX_SED_OPAL_H */

/* list_is_first is defined */
#define HAVE_LIST_IS_FIRST 1

/* linux/lockdep.h has lockdep_unregister_key */
#define HAVE_LOCKDEP_UNREGISTER_KEY 1

/* linux/lockdep.h has lockdep_assert_held_exclusive */
/* #undef HAVE_LOCKUP_ASSERT_HELD_EXCLUSIVE */

/* linux/lockdep.h has lockdep_assert_held_write */
#define HAVE_LOCKUP_ASSERT_HELD_WRITE 1

/* linux/kern_levels.h has LOGLEVEL_DEFAULT */
#define HAVE_LOGLEVEL_DEFAULT 1

/* memalloc_noio_save is defined */
#define HAVE_MEMALLOC_NOIO_SAVE 1

/* if memalloc_noreclaim_save exists */
#define HAVE_MEMALLOC_NORECLAIM_SAVE 1

/* memcpy_and_pad is defined */
#define HAVE_MEMCPY_AND_PAD 1

/* memdup_user_nul is defined */
#define HAVE_MEMDUP_USER_NUL 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* map_lock has mmap_read_lock */
#define HAVE_MMAP_READ_LOCK 1

/* mmget is defined */
#define HAVE_MMGET 1

/* mmgrab is defined */
#define HAVE_MMGRAB 1

/* mmput_async is exported by the kernel */
#define HAVE_MMPUT_ASYNC_EXPORTED 1

/* mmu interval notifier defined */
#define HAVE_MMU_INTERVAL_NOTIFIER 1

/* mmu_notifier_call_srcu defined */
/* #undef HAVE_MMU_NOTIFIER_CALL_SRCU */

/* struct mmu_notifier_ops has alloc/free_notifier */
#define HAVE_MMU_NOTIFIER_OPS_HAS_FREE_NOTIFIER 1

/* mmu_notifier_range_blockable defined */
#define HAVE_MMU_NOTIFIER_RANGE_BLOCKABLE 1

/* ib_umem_notifier_invalidate_range_start get struct mmu_notifier_range */
#define HAVE_MMU_NOTIFIER_RANGE_STRUCT 1

/* mmu_notifier_synchronize defined */
#define HAVE_MMU_NOTIFIER_SYNCHRONIZE 1

/* mmu_notifier_unregister_no_release defined */
/* #undef HAVE_MMU_NOTIFIER_UNREGISTER_NO_RELEASE */

/* mm.h has gup_must_unshare get 3 params */
/* #undef HAVE_MM_GUP_MUST_UNSHARE_GET_3_PARAMS */

/* mm_kobj is exported by the kernel */
#define HAVE_MM_KOBJ_EXPORTED 1

/* struct page has _count */
/* #undef HAVE_MM_PAGE__COUNT */

/* mq_rq_state is defined */
#define HAVE_MQ_RQ_STATE 1

/* math64.h has mul_u32_u32 */
#define HAVE_MUL_U32_U32 1

/* linux/skbuff.h napi_build_skb is defined */
#define HAVE_NAPI_BUILD_SKB 1

/* napi_consume_skb is defined */
#define HAVE_NAPI_CONSUME_SKB 1

/* napi_reschedule exists */
#define HAVE_NAPI_RESCHEDULE 1

/* NAPI_STATE_MISSED is defined */
#define HAVE_NAPI_STATE_MISSED 1

/* ndo_add_vxlan_port is defined */
/* #undef HAVE_NDO_ADD_VXLAN_PORT */

/* ndo_bridge_getlink is defined */
/* #undef HAVE_NDO_BRIDGE_GETLINK */

/* ndo_bridge_getlink is defined */
#define HAVE_NDO_BRIDGE_GETLINK_NLFLAGS 1

/* ndo_bridge_setlink is defined */
/* #undef HAVE_NDO_BRIDGE_SETLINK */

/* ndo_bridge_setlink is defined */
#define HAVE_NDO_BRIDGE_SETLINK_EXTACK 1

/* extended ndo_change_mtu is defined */
/* #undef HAVE_NDO_CHANGE_MTU_EXTENDED */

/* extended ndo_change_mtu_rh74 is defined */
/* #undef HAVE_NDO_CHANGE_MTU_RH74 */

/* ndo_dflt_bridge_getlink is defined */
/* #undef HAVE_NDO_DFLT_BRIDGE_GETLINK_FLAG_MASK */

/* ndo_dflt_bridge_getlink is defined */
/* #undef HAVE_NDO_DFLT_BRIDGE_GETLINK_FLAG_MASK_NFLAGS */

/* ndo_dflt_bridge_getlink is defined */
#define HAVE_NDO_DFLT_BRIDGE_GETLINK_FLAG_MASK_NFLAGS_FILTER 1

/* net_device_ops has ndo_eth_ioctl is defined */
#define HAVE_NDO_ETH_IOCTL 1

/* eth_phy_stats is defined */
#define HAVE_NDO_ETH_PHY_STATS 1

/* ndo_get_coalesce get 4 parameters */
#define HAVE_NDO_GET_COALESCE_GET_4_PARAMS 1

/* ndo_get_devlink_port is defined */
#define HAVE_NDO_GET_DEVLINK_PORT 1

/* get_fec_stats is defined */
#define HAVE_NDO_GET_FEC_STATS 1

/* ndo_get_offload_stats is defined */
#define HAVE_NDO_GET_OFFLOAD_STATS 1

/* extended ndo_get_offload_stats is defined */
/* #undef HAVE_NDO_GET_OFFLOAD_STATS_EXTENDED */

/* ndo_get_phys_port_name is defined */
#define HAVE_NDO_GET_PHYS_PORT_NAME 1

/* is defined */
/* #undef HAVE_NDO_GET_PHYS_PORT_NAME_EXTENDED */

/* HAVE_NDO_GET_PORT_PARENT_ID is defined */
#define HAVE_NDO_GET_PORT_PARENT_ID 1

/* ndo_get_stats64 is defined */
/* #undef HAVE_NDO_GET_STATS64 */

/* ndo_get_stats64 is defined and returns void */
#define HAVE_NDO_GET_STATS64_RET_VOID 1

/* ndo_get_vf_guid is defined */
#define HAVE_NDO_GET_VF_GUID 1

/* ndo_get_vf_stats is defined */
#define HAVE_NDO_GET_VF_STATS 1

/* ndo_has_offload_stats gets net_device */
/* #undef HAVE_NDO_HAS_OFFLOAD_STATS_EXTENDED */

/* ndo_has_offload_stats gets net_device */
#define HAVE_NDO_HAS_OFFLOAD_STATS_GETS_NET_DEVICE 1

/* get_link_ext_stats is defined */
/* #undef HAVE_NDO_LINK_EXT_STATS */

/* ndo_select_queue has 3 params with no fallback */
#define HAVE_NDO_SELECT_QUEUE_HAS_3_PARMS_NO_FALLBACK 1

/* ndo_setup_tc is defined */
#define HAVE_NDO_SETUP_TC 1

/* ndo_setup_tc takes 4 parameters */
/* #undef HAVE_NDO_SETUP_TC_4_PARAMS */

/* ndo_setup_tc_rh is defined */
/* #undef HAVE_NDO_SETUP_TC_RH_EXTENDED */

/* ndo_setup_tc takes chain_index */
/* #undef HAVE_NDO_SETUP_TC_TAKES_CHAIN_INDEX */

/* ndo_setup_tc takes tc_setup_type */
#define HAVE_NDO_SETUP_TC_TAKES_TC_SETUP_TYPE 1

/* ndo_set_tx_maxrate is defined */
#define HAVE_NDO_SET_TX_MAXRATE 1

/* extended ndo_set_tx_maxrate is defined */
/* #undef HAVE_NDO_SET_TX_MAXRATE_EXTENDED */

/* ndo_set_vf_guid is defined */
#define HAVE_NDO_SET_VF_GUID 1

/* ndo_set_vf_vlan is defined in net_device_ops */
#define HAVE_NDO_SET_VF_VLAN 1

/* ndo_set_vf_vlan is defined in net_device_ops_extended */
/* #undef HAVE_NDO_SET_VF_VLAN_EXTENDED */

/* ndo_tx_timeout get 2 params */
#define HAVE_NDO_TX_TIMEOUT_GET_2_PARAMS 1

/* ndo_add_vxlan_port is defined */
/* #undef HAVE_NDO_UDP_TUNNEL_ADD */

/* extended ndo_add_vxlan_port is defined */
/* #undef HAVE_NDO_UDP_TUNNEL_ADD_EXTENDED */

/* net_device_ops has ndo_xdp is defined */
#define HAVE_NDO_XDP 1

/* extended ndo_xdp is defined */
/* #undef HAVE_NDO_XDP_EXTENDED */

/* ndo_xdp_flush is defined */
/* #undef HAVE_NDO_XDP_FLUSH */

/* net_device_ops has ndo_xdp_xmit is defined */
#define HAVE_NDO_XDP_XMIT 1

/* ndo_xsk_wakeup is defined */
#define HAVE_NDO_XSK_WAKEUP 1

/* netdev_bpf struct has pool member */
#define HAVE_NETDEV_BPF_XSK_BUFF_POOL 1

/* struct net_device has devlink_port as member */
/* #undef HAVE_NETDEV_DEVLINK_PORT */

/* netdev_master_upper_dev_get_rcu is defined */
/* #undef HAVE_NETDEV_FOR_EACH_ALL_UPPER_DEV_RCU */

/* netdev_for_each_lower_dev is defined */
#define HAVE_NETDEV_FOR_EACH_LOWER_DEV 1

/* function netdev_get_xmit_slave exists */
#define HAVE_NETDEV_GET_XMIT_SLAVE 1

/* netdev_has_upper_dev_all_rcu is defined */
#define HAVE_NETDEV_HAS_UPPER_DEV_ALL_RCU 1

/* linux/netdevice.h has netdev_hold */
/* #undef HAVE_NETDEV_HOLD */

/* IFF_RXFH_CONFIGURED is defined */
#define HAVE_NETDEV_IFF_RXFH_CONFIGURED 1

/* netdev_lag_hash has NETDEV_LAG_HASH_VLAN_SRCMAC */
#define HAVE_NETDEV_LAG_HASH_VLAN_SRCMAC 1

/* netdev_master_upper_dev_link gets 4 parameters */
/* #undef HAVE_NETDEV_MASTER_UPPER_DEV_LINK_4_PARAMS */

/* netdev_master_upper_dev_link gets 5 parameters */
#define HAVE_NETDEV_MASTER_UPPER_DEV_LINK_5_PARAMS 1

/* netdevice.h has struct netdev_nested_priv */
#define HAVE_NETDEV_NESTED_PRIV_STRUCT 1

/* struct netdev_net_notifier is defined */
#define HAVE_NETDEV_NET_NOTIFIER 1

/* netdev_notifier_changeupper_info is defined */
#define HAVE_NETDEV_NOTIFIER_CHANGEUPPER_INFO 1

/* struct netdev_notifier_changeupper_info has upper_info */
#define HAVE_NETDEV_NOTIFIER_CHANGEUPPER_INFO_UPPER_INFO 1

/* struct netdev_notifier_info is defined */
#define HAVE_NETDEV_NOTIFIER_INFO 1

/* struct netdev_notifier_info has extack */
#define HAVE_NETDEV_NOTIFIER_INFO_EXTACK 1

/* netdev_notifier_info_to_dev is defined */
#define HAVE_NETDEV_NOTIFIER_INFO_TO_DEV 1

/* ndo_set_vf_trust is defined in net_device_ops */
#define HAVE_NETDEV_OPS_NDO_SET_VF_TRUST 1

/* extended ndo_set_vf_trust is defined */
/* #undef HAVE_NETDEV_OPS_NDO_SET_VF_TRUST_EXTENDED */

/* netdev_phys_item_id is defined */
#define HAVE_NETDEV_PHYS_ITEM_ID 1

/* netdev_port_same_parent_id is defined */
#define HAVE_NETDEV_PORT_SAME_PARENT_ID 1

/* netdev_reg_state is defined */
#define HAVE_NETDEV_REG_STATE 1

/* netdev_walk_all_lower_dev_rcu is defined */
#define HAVE_NETDEV_WALK_ALL_LOWER_DEV_RCU 1

/* netdev_walk_all_upper_dev_rcu is defined */
#define HAVE_NETDEV_WALK_ALL_UPPER_DEV_RCU 1

/* struct netdev_xdp is defined */
/* #undef HAVE_NETDEV_XDP */

/* netdev_xmit_more is defined */
#define HAVE_NETDEV_XMIT_MORE 1

/* netif_carrier_event exists */
#define HAVE_NETIF_CARRIER_EVENT 1

/* netif_device_present get const */
#define HAVE_NETIF_DEVICE_PRESENT_GET_CONST 1

/* NETIF_F_GRO_HW is defined in netdev_features.h */
#define HAVE_NETIF_F_GRO_HW 1

/* NETIF_F_GSO_IPXIP6 is defined in netdev_features.h */
#define HAVE_NETIF_F_GSO_IPXIP6 1

/* NETIF_F_GSO_PARTIAL is defined in netdev_features.h */
#define HAVE_NETIF_F_GSO_PARTIAL 1

/* HAVE_NETIF_F_GSO_UDP_L4 is defined in netdev_features.h */
#define HAVE_NETIF_F_GSO_UDP_L4 1

/* NETIF_F_HW_TLS_RX is defined in netdev_features.h */
#define HAVE_NETIF_F_HW_TLS_RX 1

/* netif_is_bareudp is defined */
#define HAVE_NETIF_IS_BAREDUDP 1

/* netif_is_bareudp is defined */
#define HAVE_NETIF_IS_BAREUDP 1

/* netif_is_geneve is defined */
#define HAVE_NETIF_IS_GENEVE 1

/* netif_is_gretap is defined */
#define HAVE_NETIF_IS_GRETAP 1

/* NETIF_IS_LAG_MASTER is defined in netdevice.h */
#define HAVE_NETIF_IS_LAG_MASTER 1

/* NETIF_IS_LAG_PORT is defined in netdevice.h */
#define HAVE_NETIF_IS_LAG_PORT 1

/* netif_is_rxfh_configured is defined */
#define HAVE_NETIF_IS_RXFH_CONFIGURED 1

/* netif_is_vxlan is defined */
#define HAVE_NETIF_IS_VXLAN 1

/* netif_napi_add get 3 params */
/* #undef HAVE_NETIF_NAPI_ADD_GET_3_PARAMS */

/* netdevice.h has netif_napi_add_weight */
/* #undef HAVE_NETIF_NAPI_ADD_WEIGHT */

/* netif_trans_update is defined */
#define HAVE_NETIF_TRANS_UPDATE 1

/* struct netlink_callback has member extack */
#define HAVE_NETLINK_CALLBACK_EXTACK 1

/* struct netlink_ext_ack exists */
#define HAVE_NETLINK_EXTACK 1

/* struct netlink_ext_ack is defined */
#define HAVE_NETLINK_EXT_ACK 1

/* struct netns_frags has rhashtable */
/* #undef HAVE_NETNS_FRAGS_RHASHTABLE */

/* netpoll_poll_dev is exported by the kernel */
#define HAVE_NETPOLL_POLL_DEV_EXPORTED 1

/* net/bareudp.h is exists */
#define HAVE_NET_BAREUDP_H 1

/* struct net_device has devlink_port member */
/* #undef HAVE_NET_DEVICE_DEVLINK_PORT */

/* net_device close_list is defined */
#define HAVE_NET_DEVICE_HAS_CLOSE_LIST 1

/* struct net_device has devlink_port */
/* #undef HAVE_NET_DEVICE_HAS_DEVLINK_PORT */

/* struct net_device has lower_level */
#define HAVE_NET_DEVICE_LOWER_LEVEL 1

/* net_device min/max is defined */
#define HAVE_NET_DEVICE_MIN_MAX_MTU 1

/* extended min/max_mtu is defined */
/* #undef HAVE_NET_DEVICE_MIN_MAX_MTU_EXTENDED */

/* net_device needs_free_netdev is defined */
#define HAVE_NET_DEVICE_NEEDS_FREE_NETDEV 1

/* struct net_device_ops_extended is defined */
/* #undef HAVE_NET_DEVICE_OPS_EXTENDED */

/* net/flow_keys.h exists */
/* #undef HAVE_NET_FLOW_KEYS_H */

/* net/gro.h is defined */
#define HAVE_NET_GRO_H 1

/* net/lag.h exists */
#define HAVE_NET_LAG_H 1

/* net/lag.h exists */
#define HAVE_NET_LAG_PORT_DEV_TXABLE 1

/* net/bareudp.h is exists */
#define HAVE_NET_MEMREMAP_H 1

/* net_namespace get const struct device */
/* #undef HAVE_NET_NAMESPACE_GET_CONST_DEVICE */

/* net/nexthop.h is defined */
#define HAVE_NET_NEXTHOP_H 1

/* net/page_pool.h is defined */
#define HAVE_NET_PAGE_POOL_OLD_H 1

/* net/page_pool/types.h is defined */
/* #undef HAVE_NET_PAGE_POOL_TYPES_H */

/* net_prefetch is defined */
#define HAVE_NET_PREFETCH 1

/* net/psample.h exists */
#define HAVE_NET_PSAMPLE_H 1

/* kernel does synchronize_net for us */
#define HAVE_NET_SYNCHRONIZE_IN_SET_REAL_NUM_TX_QUEUES 1

/* net/tc_act/tc_mpls.h exists */
#define HAVE_NET_TC_ACT_TC_MPLS_H 1

/* net/tls.h exists */
#define HAVE_NET_TLS_H 1

/* net/xdp.h is defined */
#define HAVE_NET_XDP_HEADER 1

/* net/xdp.h is defined workaround for 5.4.17-2011.1.2.el8uek.x86_64 */
/* #undef HAVE_NET_XDP_HEADER_UEK_KABI */

/* nla_nest_start_noflag exist */
#define HAVE_NLA_NEST_START_NOFLAG 1

/* nla_parse takes 6 parameters */
#define HAVE_NLA_PARSE_6_PARAMS 1

/* nla_parse_deprecated exist */
#define HAVE_NLA_PARSE_DEPRECATED 1

/* nla_policy has validation_type */
#define HAVE_NLA_POLICY_HAS_VALIDATION_TYPE 1

/* nla_put_u64_64bit is defined */
#define HAVE_NLA_PUT_U64_64BIT 1

/* nla_strscpy exist */
#define HAVE_NLA_STRSCPY 1

/* nlmsg_parse_deprecated exist */
#define HAVE_NLMSG_PARSE_DEPRECATED 1

/* nlmsg_validate_deprecated exist */
#define HAVE_NLMSG_VALIDATE_DEPRECATED 1

/* NL_SET_ERR_MSG_WEAK_MOD exists */
#define HAVE_NL_SET_ERR_MSG_WEAK_MOD 1

/* current_link_speed/width not exposed */
/* #undef HAVE_NO_LINKSTA_SYSFS */

/* nvme_auth_transform_key returns u8 */
/* #undef HAVE_NVME_AUTH_TRANSFORM_KEY_U8 */

/* NVME_IOCTL_IO64_CMD_VEC is defined */
/* #undef HAVE_NVME_IOCTL_IO64_CMD_VEC */

/* include/linux/once.h exists */
#define HAVE_ONCE_H 1

/* op_is_write is defined */
#define HAVE_OP_IS_WRITE 1

/* linux/page_ref.h has page_count */
#define HAVE_PAGE_COUNT 1

/* struct page has dma_addr */
#define HAVE_PAGE_DMA_ADDR 1

/* struct page has dma_addr array member */
/* #undef HAVE_PAGE_DMA_ADDR_ARRAY */

/* page_is_pfmemalloc is defined */
#define HAVE_PAGE_IS_PFMEMALLOC 1

/* pfmemalloc is defined */
/* #undef HAVE_PAGE_PFMEMALLOC */

/* net/page_pool/helpers.h has page_pool_nid_changed */
/* #undef HAVE_PAGE_POLL_NID_CHANGED_HELPERS */

/* net/page_pool.h has page_pool_nid_changed */
#define HAVE_PAGE_POLL_NID_CHANGED_OLD 1

/* net/page_pool/types.h has page_pool_put_defragged_page */
/* #undef HAVE_PAGE_POOL_DEFRAG_PAGE */

/* net/page_pool.h page_pool_get_dma_addr defined */
/* #undef HAVE_PAGE_POOL_GET_DMA_ADDR_HELPER */

/* net/page_pool.h page_pool_get_dma_addr defined */
#define HAVE_PAGE_POOL_GET_DMA_ADDR_OLD 1

/* net/page_pool.h has page_pool_release_page */
#define HAVE_PAGE_POOL_RELEASE_PAGE_IN_PAGE_POOL_H 1

/* net/page_pool/types.h has page_pool_release_page */
/* #undef HAVE_PAGE_POOL_RELEASE_PAGE_IN_TYPES_H */

/* page_ref_count/add/sub/inc defined */
#define HAVE_PAGE_REF_COUNT_ADD_SUB_INC 1

/* param_ops_ullong is defined */
#define HAVE_PARAM_OPS_ULLONG 1

/* linux/moduleparam.h has param_set_uint_minmax */
#define HAVE_PARAM_SET_UINT_MINMAX 1

/* part_stat.h exists */
#define HAVE_PART_STAT_H 1

/* linux/pci.h has pcie_aspm_enabled */
#define HAVE_PCIE_ASPM_ENABLED 1

/* pci.h has pcie_find_root_port */
#define HAVE_PCIE_FIND_ROOT_PORT 1

/* pcie_get_minimum_link is defined */
/* #undef HAVE_PCIE_GET_MINIMUM_LINK */

/* pcie_print_link_status is defined */
#define HAVE_PCIE_PRINT_LINK_STATUS 1

/* pcie_relaxed_ordering_enabled is defined */
#define HAVE_PCIE_RELAXED_ORDERING_ENABLED 1

/* pci_bus_addr_t is defined */
#define HAVE_PCI_BUS_ADDR_T 1

/* PCI_CLASS_STORAGE_EXPRESS is defined */
#define HAVE_PCI_CLASS_STORAGE_EXPRESS 1

/* pci_enable_atomic_ops_to_root is defined */
#define HAVE_PCI_ENABLE_ATOMIC_OPS_TO_ROOT 1

/* linux/aer.h has pci_enable_pcie_error_reporting */
#define HAVE_PCI_ENABLE_PCIE_ERROR_REPORTING 1

/* pci.h struct pci_error_handlers has reset_done */
#define HAVE_PCI_ERROR_HANDLERS_RESET_DONE 1

/* pci.h struct pci_error_handlers has reset_notify */
/* #undef HAVE_PCI_ERROR_HANDLERS_RESET_NOTIFY */

/* pci.h struct pci_error_handlers has reset_prepare */
#define HAVE_PCI_ERROR_HANDLERS_RESET_PREPARE 1

/* linux/pci.h has pci_free_irq */
#define HAVE_PCI_FREE_IRQ 1

/* PCI_VENDOR_ID_AMAZON is defined in pci_ids */
#define HAVE_PCI_IDS_PCI_VENDOR_ID_AMAZON 1

/* pci_iov_get_pf_drvdata is defined */
/* #undef HAVE_PCI_IOV_GET_PF_DRVDATA */

/* pci_iov_vf_id is defined */
/* #undef HAVE_PCI_IOV_VF_ID */

/* linux/pci.h has pci_irq_vector, pci_free_irq_vectors, pci_alloc_irq_vectors
   */
#define HAVE_PCI_IRQ_API 1

/* pci_irq_get_affinity is defined */
#define HAVE_PCI_IRQ_GET_AFFINITY 1

/* pci_irq_get_node is defined */
/* #undef HAVE_PCI_IRQ_GET_NODE */

/* linux/pci-p2pdma.h exists */
#define HAVE_PCI_P2PDMA_H 1

/* pci_p2pdma_map_sg_attrs defined */
#define HAVE_PCI_P2PDMA_MAP_SG_ATTRS 1

/* pci_p2pdma_unmap_sg defined */
#define HAVE_PCI_P2PDMA_UNMAP_SG 1

/* pci_pool_zalloc is defined */
#define HAVE_PCI_POOL_ZALLOC 1

/* pci_release_mem_regions is defined */
#define HAVE_PCI_RELEASE_MEM_REGIONS 1

/* pci_request_mem_regions is defined */
#define HAVE_PCI_REQUEST_MEM_REGIONS 1

/* PCI_VENDOR_ID_REDHAT is defined */
#define HAVE_PCI_VENDOR_ID_REDHAT 1

/* linux/proc_fs.h has pde_data */
/* #undef HAVE_PDE_DATA */

/* pinned_vm is defined */
/* #undef HAVE_PINNED_VM */

/* dev_pm_qos_update_user_latency_tolerance is exported by the kernel */
#define HAVE_PM_QOS_UPDATE_USER_LATENCY_TOLERANCE_EXPORTED 1

/* linux/suspend.h has pm_suspend_via_firmware */
#define HAVE_PM_SUSPEND_VIA_FIRMWARE 1

/* pnv-pci.h has pnv_pci_set_p2p */
/* #undef HAVE_PNV_PCI_SET_P2P */

/* port_function_hw_addr_get has 4 params */
#define HAVE_PORT_FUNCTION_HW_ADDR_GET_GET_4_PARAM 1

/* port_function_state_get has 4 params */
#define HAVE_PORT_FUNCTION_STATE_GET_4_PARAM 1

/* struct proc_ops is defined */
#define HAVE_PROC_OPS_STRUCT 1

/* net.h struct proto_ops has sendpage */
#define HAVE_PROTO_OPS_SENDPAGE 1

/* linux/pr.h exists */
#define HAVE_PR_H 1

/* enum pr_status is defined */
/* #undef HAVE_PR_STATUS */

/* ptp_classify_raw is defined */
#define HAVE_PTP_CLASSIFY_RAW 1

/* adjphase is defined */
#define HAVE_PTP_CLOCK_INFO_ADJPHASE 1

/* gettime 32bit is defined */
/* #undef HAVE_PTP_CLOCK_INFO_GETTIME_32BIT */

/* adjfine is defined */
#define HAVE_PTP_CLOCK_INFO_NDO_ADJFINE 1

/* adjfreq is defined */
#define HAVE_PTP_CLOCK_INFO_NDO_ADJFREQ 1

/* ptp_find_pin_unlocked is defined */
#define HAVE_PTP_FIND_PIN_UNLOCK 1

/* PTP_PEROUT_DUTY_CYCLE is defined */
#define HAVE_PTP_PEROUT_DUTY_CYCLE 1

/* __put_task_struct is exported by the kernel */
#define HAVE_PUT_TASK_STRUCT_EXPORTED 1

/* put_unaligned_le24 existing */
/* #undef HAVE_PUT_UNALIGNED_LE24 */

/* put_unaligned_le24 existing in asm-generic/unaligned.h */
#define HAVE_PUT_UNALIGNED_LE24_ASM_GENERIC 1

/* put_user_pages_dirty_lock has 2 parameters */
/* #undef HAVE_PUT_USER_PAGES_DIRTY_LOCK_2_PARAMS */

/* put_user_pages_dirty_lock has 3 parameters */
/* #undef HAVE_PUT_USER_PAGES_DIRTY_LOCK_3_PARAMS */

/* struct Qdisc_ops has ingress_block_set */
#define HAVE_QDISC_SUPPORTS_BLOCK_SHARING 1

/* QUEUE_FLAG_DISCARD is defined */
#define HAVE_QUEUE_FLAG_DISCARD 1

/* QUEUE_FLAG_PCI_P2PDMA is defined */
#define HAVE_QUEUE_FLAG_PCI_P2PDMA 1

/* QUEUE_FLAG_STABLE_WRITES is defined */
#define HAVE_QUEUE_FLAG_STABLE_WRITES 1

/* QUEUE_FLAG_WC_FUA is defined */
#define HAVE_QUEUE_FLAG_WC_FUA 1

/* radix_tree_is_internal_node is defined */
#define HAVE_RADIX_TREE_IS_INTERNAL 1

/* radix_tree_iter_delete is defined */
#define HAVE_RADIX_TREE_ITER_DELETE 1

/* radix_tree_iter_delete is exported by the kernel */
#define HAVE_RADIX_TREE_ITER_DELETE_EXPORTED 1

/* radix_tree_iter_lookup is defined */
#define HAVE_RADIX_TREE_ITER_LOOKUP 1

/* struct rb_root_cached is defined */
#define HAVE_RB_ROOT_CACHED 1

/* refcount.h exists */
#define HAVE_REFCOUNT 1

/* refcount.h has refcount_dec_if_one */
#define HAVE_REFCOUNT_DEC_IF_ONE 1

/* linux/security.h has register_blocking_lsm_notifier */
#define HAVE_REGISTER_BLOCKING_LSM_NOTIFIER 1

/* register_fib_notifier has 4 params */
#define HAVE_REGISTER_FIB_NOTIFIER_HAS_4_PARAMS 1

/* linux/security.h has register_lsm_notifier */
/* #undef HAVE_REGISTER_LSM_NOTIFIER */

/* register_netdevice_notifier_dev_net is defined */
#define HAVE_REGISTER_NETDEVICE_NOTIFIER_DEV_NET 1

/* register_netdevice_notifier_rh is defined */
/* #undef HAVE_REGISTER_NETDEVICE_NOTIFIER_RH */

/* release_pages is defined */
#define HAVE_RELEASE_PAGES 1

/* mm.h has release_pages */
/* #undef HAVE_RELEASE_PAGES_IN_MM_H */

/* blkdev.h struct request has block_device */
#define HAVE_REQUEST_BDEV 1

/* request_firmware_direct is defined */
#define HAVE_REQUEST_FIRMWARE_DIRECT 1

/* blkdev.h struct request has deadline */
#define HAVE_REQUEST_HAS_DEADLINE 1

/* blkdev.h struct request has mq_hctx */
#define HAVE_REQUEST_MQ_HCTX 1

/* struct request_queue has backing_dev_info */
/* #undef HAVE_REQUEST_QUEUE_BACKING_DEV_INFO */

/* struct request_queue has integrity */
#define HAVE_REQUEST_QUEUE_INTEGRITY 1

/* struct request_queue has q_usage_counter */
#define HAVE_REQUEST_QUEUE_Q_USAGE_COUNTER 1

/* blkdev.h struct request_queue has timeout_work */
#define HAVE_REQUEST_QUEUE_TIMEOUT_WORK 1

/* blkdev.h struct request has rq_flags */
#define HAVE_REQUEST_RQ_FLAGS 1

/* linux/blk-mq.h has request_to_qc_t */
#define HAVE_REQUEST_TO_QC_T 1

/* blk_types.h has REQ_IDLE */
#define HAVE_REQ_IDLE 1

/* req_op exist */
#define HAVE_REQ_OP 1

/* enum req_opf has REQ_OP_DRV_OUT */
#define HAVE_REQ_OPF_REQ_OP_DRV_OUT 1

/* enum req_op has REQ_OP_DRV_OUT */
/* #undef HAVE_REQ_OP_REQ_OP_DRV_OUT */

/* blkdev.h struct request has rq_disk */
#define HAVE_REQ_RQ_DISK 1

/* genhd.h has revalidate_disk_size */
/* #undef HAVE_REVALIDATE_DISK_SIZE */

/* struct bio_aux is defined */
/* #undef HAVE_RH7_STRUCT_BIO_AUX */

/* struct rhashtable_params has insecure_elasticity */
/* #undef HAVE_RHASHTABLE_INSECURE_ELASTICITY */

/* struct rhashtable_params has insecure_max_entries */
/* #undef HAVE_RHASHTABLE_INSECURE_MAX_ENTRIES */

/* rhashtable.h has rhashtable_lookup_get_insert_fast */
#define HAVE_RHASHTABLE_LOOKUP_GET_INSERT_FAST 1

/* struct rhashtable has max_elems */
/* #undef HAVE_RHASHTABLE_MAX_ELEMS */

/* file rhashtable-types exists */
#define HAVE_RHASHTABLE_TYPES 1

/* struct rhltable is defined */
/* #undef HAVE_RHLTABLE */

/* rpc reply expected */
#define HAVE_RPC_REPLY_EXPECTED 1

/* rpc_task_gfp_mask is exported by the kernel */
/* #undef HAVE_RPC_TASK_GPF_MASK_EXPORTED */

/* struct rpc_xprt_ops has 'bc_num_slots' field */
#define HAVE_RPC_XPRT_OPS_BC_NUM_SLOTS 1

/* struct rpc_xprt_ops has 'bc_up' field */
/* #undef HAVE_RPC_XPRT_OPS_BC_UP */

/* struct rpc_xprt_ops *ops' field is const inside 'struct rpc_xprt' */
#define HAVE_RPC_XPRT_OPS_CONST 1

/* struct rpc_xprt_ops has 'free_slot' field */
#define HAVE_RPC_XPRT_OPS_FREE_SLOT 1

/* struct rpc_xprt_ops has 'set_retrans_timeout' field */
/* #undef HAVE_RPC_XPRT_OPS_SET_RETRANS_TIMEOUT */

/* struct rpc_xprt_ops has 'wait_for_reply_request' field */
#define HAVE_RPC_XPRT_OPS_WAIT_FOR_REPLY_REQUEST 1

/* struct rpc_xprt has 'recv_lock' field */
/* #undef HAVE_RPC_XPRT_RECV_LOCK */

/* struct rpc_xprt has 'xprt_class' field */
#define HAVE_RPC_XPRT_XPRT_CLASS 1

/* if file rq_end_io_ret exists */
/* #undef HAVE_RQ_END_IO_RET */

/* rt6_lookup takes 6 params */
#define HAVE_RT6_LOOKUP_TAKES_6_PARAMS 1

/* linux/rtnetlink.h has net_rwsem */
#define HAVE_RTNETLINK_NET_RWSEM 1

/* newlink has 4 paramters */
/* #undef HAVE_RTNL_LINK_OPS_NEWLINK_4_PARAMS */

/* newlink has 5 paramters */
#define HAVE_RTNL_LINK_OPS_NEWLINK_5_PARAMS 1

/* rt_gw_family is defined */
#define HAVE_RT_GW_FAMILY 1

/* rt_uses_gateway is defined */
#define HAVE_RT_USES_GATEWAY 1

/* sched_mmget_not_zero is defined */
/* #undef HAVE_SCHED_MMGET_NOT_ZERO */

/* linux/sched/mm.h exists */
#define HAVE_SCHED_MM_H 1

/* mmget_not_zero is defined */
#define HAVE_SCHED_MM_MMGET_NOT_ZERO 1

/* linux/sched/signal.h exists */
#define HAVE_SCHED_SIGNAL_H 1

/* linux/sched/task.h exists */
#define HAVE_SCHED_TASK_H 1

/* scsi_block_targets is defined */
/* #undef HAVE_SCSI_BLOCK_TARGETS */

/* scsi_change_queue_depth exist */
#define HAVE_SCSI_CHANGE_QUEUE_DEPTH 1

/* scsi_cmd_to_rq is defined */
#define HAVE_SCSI_CMD_TO_RQ 1

/* scsi_cmnd has members prot_flags */
#define HAVE_SCSI_CMND_PROT_FLAGS 1

/* scsi_device.h struct scsi_device has member budget_map */
#define HAVE_SCSI_DEVICE_BUDGET_MAP 1

/* scsi_device.h has function scsi_internal_device_block */
/* #undef HAVE_SCSI_DEVICE_SCSI_INTERNAL_DEVICE_BLOCK */

/* scsi_device.h struct scsi_device has member state_mutex */
#define HAVE_SCSI_DEVICE_STATE_MUTEX 1

/* scsi_done is defined */
#define HAVE_SCSI_DONE 1

/* scsi_get_sector is defined */
#define HAVE_SCSI_GET_SECTOR 1

/* scsi_host.h scsi_host_busy_iter fn has 2 args */
/* #undef HAVE_SCSI_HOST_BUSY_ITER_FN_2_ARGS */

/* Scsi_Host has members max_segment_size */
#define HAVE_SCSI_HOST_MAX_SEGMENT_SIZE 1

/* Scsi_Host has members nr_hw_queues */
#define HAVE_SCSI_HOST_NR_HW_QUEUES 1

/* scsi_host_template has members change_queue_type */
/* #undef HAVE_SCSI_HOST_TEMPLATE_CHANGE_QUEUE_TYPE */

/* scsi_host_template has member init_cmd_priv */
#define HAVE_SCSI_HOST_TEMPLATE_INIT_CMD_PRIV 1

/* scsi_host_template has members shost_groups */
/* #undef HAVE_SCSI_HOST_TEMPLATE_SHOST_GROUPS */

/* scsi_host_template has members track_queue_depth */
#define HAVE_SCSI_HOST_TEMPLATE_TRACK_QUEUE_DEPTH 1

/* scsi_host_template has members use_blk_tags */
/* #undef HAVE_SCSI_HOST_TEMPLATE_USE_BLK_TAGS */

/* scsi_host_template has members use_host_wide_tags */
/* #undef HAVE_SCSI_HOST_TEMPLATE_USE_HOST_WIDE_TAGS */

/* Scsi_Host has members virt_boundary_mask */
#define HAVE_SCSI_HOST_VIRT_BOUNDARY_MASK 1

/* SCSI_MAX_SG_SEGMENTS is defined */
/* #undef HAVE_SCSI_MAX_SG_SEGMENTS */

/* QUEUE_FULL is defined */
/* #undef HAVE_SCSI_QUEUE_FULL */

/* scsi_host.h has enum scsi_timeout_action */
/* #undef HAVE_SCSI_TIMEOUT_ACTION */

/* scsi_transfer_length is defined */
#define HAVE_SCSI_TRANSFER_LENGTH 1

/* scsi/scsi_transport_fc.h has FC_PORT_ROLE_NVME_TARGET */
#define HAVE_SCSI_TRANSPORT_FC_FC_PORT_ROLE_NVME_TARGET 1

/* if secpath_set returns struct sec_path * */
#define HAVE_SECPATH_SET_RETURN_POINTER 1

/* select_queue_fallback_t is defined */
#define HAVE_SELECT_QUEUE_FALLBACK_T 1

/* select_queue_fallback_t has third parameter */
#define HAVE_SELECT_QUEUE_FALLBACK_T_3_PARAMS 1

/* ndo_select_queue has a second net_device parameter */
/* #undef HAVE_SELECT_QUEUE_NET_DEVICE */

/* linux/net.h has sendpage_ok */
#define HAVE_SENDPAGE_OK 1

/* genhd.h has set_capacity_revalidate_and_notify */
/* #undef HAVE_SET_CAPACITY_REVALIDATE_AND_NOTIFY */

/* struct se_cmd has member sense_info */
#define HAVE_SE_CMD_HAS_SENSE_INFO 1

/* target_core_base.h se_cmd transport_complete_callback has three params */
#define HAVE_SE_CMD_TRANSPORT_COMPLETE_CALLBACK_HAS_THREE_PARAM 1

/* sgl_alloc is defined */
#define HAVE_SGL_ALLOC 1

/* sgl_free is defined */
#define HAVE_SGL_FREE 1

/* sg_alloc_table_chained has 3 parameters */
/* #undef HAVE_SG_ALLOC_TABLE_CHAINED_3_PARAMS */

/* sg_alloc_table_chained has 4 parameters */
/* #undef HAVE_SG_ALLOC_TABLE_CHAINED_4_PARAMS */

/* sg_alloc_table_chained has 4 params */
/* #undef HAVE_SG_ALLOC_TABLE_CHAINED_GFP_MASK */

/* sg_alloc_table_chained has nents_first_chunk parameter */
#define HAVE_SG_ALLOC_TABLE_CHAINED_NENTS_FIRST_CHUNK_PARAM 1

/* __sg_alloc_table_from_pages has 9 params */
/* #undef HAVE_SG_ALLOC_TABLE_FROM_PAGES_GET_9_PARAMS */

/* linux/scatterlist.h has sg_append_table */
#define HAVE_SG_APPEND_TABLE 1

/* SG_MAX_SEGMENTS is defined */
#define HAVE_SG_MAX_SEGMENTS 1

/* sg_zero_buffer is defined */
#define HAVE_SG_ZERO_BUFFER 1

/* show_class_attr_string get const */
/* #undef HAVE_SHOW_CLASS_ATTR_STRING_GET_CONST */

/* linux/overflow.h has size_add size_mul size_sub */
#define HAVE_SIZE_MUL_SUB_ADD 1

/* skb_dst_update_pmtu is defined */
#define HAVE_SKB_DST_UPDATE_PMTU 1

/* skb_flow_dissect is defined */
#define HAVE_SKB_FLOW_DISSECT 1

/* skb_flow_dissect_flow_keys has 2 parameters */
/* #undef HAVE_SKB_FLOW_DISSECT_FLOW_KEYS_HAS_2_PARAMS */

/* skb_flow_dissect_flow_keys has 3 parameters */
#define HAVE_SKB_FLOW_DISSECT_FLOW_KEYS_HAS_3_PARAMS 1

/* linux/skbuff.h skb_frag_fill_page_desc is defined */
/* #undef HAVE_SKB_FRAG_FILL_PAGE_DESC */

/* skb_frag_off is defined */
#define HAVE_SKB_FRAG_OFF 1

/* linux/skbuff.h skb_frag_off_add is defined */
#define HAVE_SKB_FRAG_OFF_ADD 1

/* linux/skbuff.h skb_frag_off_set is defined */
#define HAVE_SKB_FRAG_OFF_SET 1

/* skb_inner_transport_offset is defined */
#define HAVE_SKB_INNER_TRANSPORT_OFFSET 1

/* linux/skbuff.h skb_metadata_set defined */
#define HAVE_SKB_METADATA_SET 1

/* skb_put_zero is defined */
#define HAVE_SKB_PUT_ZERO 1

/* linux/skbuff.h has skb_queue_empty_lockless */
#define HAVE_SKB_QUEUE_EMPTY_LOCKLESS 1

/* skb_set_redirected is defined */
#define HAVE_SKB_SET_REDIRECTED 1

/* sk_buff has member sw_hash */
#define HAVE_SKB_SWHASH 1

/* linux/tcp.h has skb_tcp_all_headers */
/* #undef HAVE_SKB_TCP_ALL_HEADERS */

/* skwq_has_sleeper is defined */
#define HAVE_SKWQ_HAS_SLEEPER 1

/* xmit_more is defined */
/* #undef HAVE_SK_BUFF_XMIT_MORE */

/* sk_data_ready has 2 params */
/* #undef HAVE_SK_DATA_READY_2_PARAMS */

/* sk_wait_data has 3 params */
#define HAVE_SK_WAIT_DATA_3_PARAMS 1

/* sock_create_kern has 5 params is defined */
#define HAVE_SOCK_CREATE_KERN_5_PARAMS 1

/* net/sock.h has sock_no_linger */
#define HAVE_SOCK_NO_LINGER 1

/* net/sock.h has sock_setsockopt sockptr_t */
#define HAVE_SOCK_SETOPTVAL_SOCKPTR_T 1

/* net/sock.h has sock_set_priority */
#define HAVE_SOCK_SET_PRIORITY 1

/* net/sock.h has sock_set_reuseaddr */
#define HAVE_SOCK_SET_REUSEADDR 1

/* split_page is exported by the kernel */
#define HAVE_SPLIT_PAGE_EXPORTED 1

/* struct pci_driver has member
   sriov_get_vf_total_msix/sriov_set_msix_vec_count */
#define HAVE_SRIOV_GET_SET_MSIX_VEC_COUNT 1

/* build_bug.h has static_assert */
#define HAVE_STATIC_ASSERT 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* fs.h has stream_open */
#define HAVE_STREAM_OPEN 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* strnicmp is defined */
/* #undef HAVE_STRNICMP */

/* strscpy is defined */
#define HAVE_STRSCPY 1

/* strscpy_pad is defined */
#define HAVE_STRSCPY_PAD 1

/* struct bio has member bi_error */
/* #undef HAVE_STRUCT_BIO_BI_ERROR */

/* struct bio has member bi_iter */
#define HAVE_STRUCT_BIO_BI_ITER 1

/* struct dma_attrs is defined */
/* #undef HAVE_STRUCT_DMA_ATTRS */

/* net/ipv6.h has struct hop_jumbo_hdr */
/* #undef HAVE_STRUCT_HOP_JUMBO_HDR */

/* ieee_qcn is defined */
#define HAVE_STRUCT_IEEE_QCN 1

/* struct ifla_vf_stats has memebers rx_dropped and tx_dropped */
#define HAVE_STRUCT_IFLA_VF_STATS_RX_TX_DROPPED 1

/* ethtool.h has struct kernel_ethtool_ringparam */
/* #undef HAVE_STRUCT_KERNEL_ETHTOOL_RINGPARAM */

/* struct rtnl_link_ops has netns_refund */
#define HAVE_STRUCT_LINK_OPS_IPOIB_LINK_OPS_HAS_NETNS_REFUND 1

/* net/dst_metadata.h has struct macsec_info */
/* #undef HAVE_STRUCT_MACSEC_INFO_METADATA */

/* net/psample.h has struct psample_metadata */
#define HAVE_STRUCT_PSAMPLE_METADATA 1

/* struct switchdev_brport_flags exist */
#define HAVE_STRUCT_SWITCHDEV_BRPORT_FLAGS 1

/* struct switchdev_obj_port_vlan has vid */
#define HAVE_STRUCT_SWITCHDEV_OBJ_PORT_VLAN_VID 1

/* linux/bio.h submit_bio has 1 parameter */
#define HAVE_SUBMIT_BIO_1_PARAM 1

/* submit_bio_noacct exist */
#define HAVE_SUBMIT_BIO_NOACCT 1

/* supported_coalesce_params is defined */
#define HAVE_SUPPORTED_COALESCE_PARAM 1

/* struct svcxprt_rdma has 'sc_pending_recvs' field */
#define HAVE_SVCXPRT_RDMA_SC_PENDING_RECVS 1

/* svc_fill_write_vector getting 2 params */
#define HAVE_SVC_FILL_WRITE_VECTOR_2_PARAMS 1

/* svc_fill_write_vector getting 3 params */
/* #undef HAVE_SVC_FILL_WRITE_VECTOR_3_PARAMS */

/* svc_fill_write_vector getting 4 params */
/* #undef HAVE_SVC_FILL_WRITE_VECTOR_4_PARAMS */

/* svc_pool_wake_idle_thread is exported by the kernel */
/* #undef HAVE_SVC_POOL_WAKE_IDLE_THREAD */

/* struct svc_rdma_pcl exists */
#define HAVE_SVC_RDMA_PCL 1

/* struct svc_rdma_recv_ctxt has 'rc_stream' field */
#define HAVE_SVC_RDMA_RECV_CTXT_RC_STREAM 1

/* svc_rdma_release_rqst has externed */
/* #undef HAVE_SVC_RDMA_RELEASE_RQST */

/* struct svc_rqst has rq_xprt_hlen */
/* #undef HAVE_SVC_RQST_RQ_XPRT_HLEN */

/* struct svc_serv has sv_cb_list */
#define HAVE_SVC_SERV_SV_CB_LIST_LIST_HEAD 1

/* struct svc_serv has sv_cb_list */
/* #undef HAVE_SVC_SERV_SV_CB_LIST_LWQ */

/* 'struct svc_xprt_ops *xcl_ops' field is const inside 'struct
   svc_xprt_class' */
#define HAVE_SVC_XPRT_CLASS_XCL_OPS_CONST 1

/* svc_xprt_close is exported by the sunrpc core */
/* #undef HAVE_SVC_XPRT_CLOSE */

/* svc_xprt_deferred_close is exported by the sunrpc core */
#define HAVE_SVC_XPRT_DEFERRED_CLOSE 1

/* svc_xprt_is_dead has defined */
#define HAVE_SVC_XPRT_IS_DEAD 1

/* svc_xprt_received is exported by the sunrpc core */
#define HAVE_SVC_XPRT_RECEIVED 1

/* struct svc_xprt_ops 'xpo_prep_reply_hdr' field */
/* #undef HAVE_SVC_XPRT_XPO_PREP_REPLY_HDR */

/* struct svc_xprt_ops 'xpo_secure_port' field */
#define HAVE_SVC_XPRT_XPO_SECURE_PORT 1

/* struct svc_xprt has 'xpt_remotebuf' field */
#define HAVE_SVC_XPRT_XPT_REMOTEBUF 1

/* enum switchdev_attr_id has SWITCHDEV_ATTR_ID_BRIDGE_VLAN_PROTOCOL */
#define HAVE_SWITCHDEV_ATTR_ID_BRIDGE_VLAN_PROTOCOL 1

/* include/net/switchdev.h exists */
#define HAVE_SWITCHDEV_H 1

/* HAVE_SWITCHDEV_OPS is defined */
/* #undef HAVE_SWITCHDEV_OPS */

/* SWITCHDEV_PORT_ATTR_SET is defined */
#define HAVE_SWITCHDEV_PORT_ATTR_SET 1

/* switchdev_port_same_parent_id is defined */
/* #undef HAVE_SWITCHDEV_PORT_SAME_PARENT_ID */

/* linux/sysctl.h has SYSCTL_ZERO defined */
#define HAVE_SYSCTL_ZERO_ENABLED 1

/* sysfs_emit is defined */
#define HAVE_SYSFS_EMIT 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* enum t10_dif_type is defined */
#define HAVE_T10_DIF_TYPE 1

/* linux/t10-pi.h exists */
#define HAVE_T10_PI_H 1

/* t10_pi_prepare is defined */
/* #undef HAVE_T10_PI_PREPARE */

/* t10_pi_ref_tag() exists */
#define HAVE_T10_PI_REF_TAG 1

/* target_put_sess_cmd in target_core_fabric.h has 1 parameter */
#define HAVE_TARGET_PUT_SESS_CMD_HAS_1_PARAM 1

/* target_stop_session is defined */
#define HAVE_TARGET_STOP_SESSION 1

/* interrupt.h has tasklet_setup */
#define HAVE_TASKLET_SETUP 1

/* TCA_FLOWER_KEY_FLAGS_FRAG_IS_FIRST is defined */
#define HAVE_TCA_FLOWER_KEY_FLAGS_FRAG_IS_FIRST 1

/* TCA_FLOWER_KEY_FLAGS_IS_FRAGMENT is defined */
#define HAVE_TCA_FLOWER_KEY_FLAGS_IS_FRAGMENT 1

/* TCA_TUNNEL_KEY_ENC_DST_PORT is defined */
#define HAVE_TCA_TUNNEL_KEY_ENC_DST_PORT 1

/* TCA_TUNNEL_KEY_ENC_TOS is defined */
#define HAVE_TCA_TUNNEL_KEY_ENC_TOS 1

/* TCA_VLAN_ACT_MODIFY exists */
#define HAVE_TCA_VLAN_ACT_MODIFY 1

/* tc_action_stats_update is defined */
/* #undef HAVE_TCF_ACTION_STATS_UPDATE */

/* tc_action_stats_update is defined and has 5 params */
/* #undef HAVE_TCF_ACTION_STATS_UPDATE_5_PARAMS */

/* struct tcf_common is defined */
/* #undef HAVE_TCF_COMMON */

/* struct tcf_exts has actions as array */
#define HAVE_TCF_EXTS_HAS_ARRAY_ACTIONS 1

/* tcf_exts_num_actions is exported by the kernel */
#define HAVE_TCF_EXTS_NUM_ACTIONS 1

/* tcf_exts_stats_update is defined */
/* #undef HAVE_TCF_EXTS_STATS_UPDATE */

/* tcf_exts_to_list is defined */
/* #undef HAVE_TCF_EXTS_TO_LIST */

/* tcf_hash helper functions have tcf_hashinfo parameter */
/* #undef HAVE_TCF_HASH_WITH_HASHINFO */

/* tcf_mirred_dev is defined */
#define HAVE_TCF_MIRRED_DEV 1

/* tcf_mirred_ifindex is defined */
/* #undef HAVE_TCF_MIRRED_IFINDEX */

/* tcf_pedit_nkeys is defined */
#define HAVE_TCF_PEDIT_NKEYS 1

/* struct tcf_pedit_parms has member tcfp_keys_ex */
#define HAVE_TCF_PEDIT_PARMS_TCFP_KEYS_EX 1

/* struct tcf_pedit has member tcfp_keys_ex */
/* #undef HAVE_TCF_PEDIT_TCFP_KEYS_EX_FIX */

/* tcf_tunnel_info is defined */
#define HAVE_TCF_TUNNEL_INFO 1

/* tcf_vlan_push_prio is defined */
#define HAVE_TCF_VLAN_PUSH_PRIO 1

/* linux/tcp.h has tcp_sock_set_nodelay */
#define HAVE_TCP_SOCK_SET_NODELAY 1

/* linux/tcp.h has tcp_sock_set_syncnt */
#define HAVE_TCP_SOCK_SET_SYNCNT 1

/* struct tc_action_ops has id */
#define HAVE_TC_ACTION_OPS_HAS_ID 1

/* struct tc_block_offload is defined */
/* #undef HAVE_TC_BLOCK_OFFLOAD */

/* struct tc_block_offload has extack */
/* #undef HAVE_TC_BLOCK_OFFLOAD_EXTACK */

/* pkt_cls.h enum enum tc_fl_command has TC_CLSFLOWER_STATS */
/* #undef HAVE_TC_CLSFLOWER_STATS_FIX */

/* TC_CLSMATCHALL_STATS is defined */
#define HAVE_TC_CLSMATCHALL_STATS 1

/* tc_cls_can_offload_and_chain0 is defined */
#define HAVE_TC_CLS_CAN_OFFLOAD_AND_CHAIN0 1

/* struct tc_cls_flower_offload has common */
/* #undef HAVE_TC_CLS_FLOWER_OFFLOAD_COMMON_FIX */

/* struct tc_cls_flower_offload has egress_dev */
/* #undef HAVE_TC_CLS_FLOWER_OFFLOAD_EGRESS_DEV */

/* struct tc_cls_flower_offload has stats field */
/* #undef HAVE_TC_CLS_FLOWER_OFFLOAD_HAS_STATS_FIELD_FIX */

/* struct tc_cls_common_offload has extack */
/* #undef HAVE_TC_CLS_OFFLOAD_EXTACK_FIX */

/* struct tc_cls_flower_offload is defined */
/* #undef HAVE_TC_FLOWER_OFFLOAD */

/* struct tc_htb_command has moved_qid */
/* #undef HAVE_TC_HTB_COMMAND_HAS_MOVED_QID */

/* tc_mqprio_qopt_offload is defined */
#define HAVE_TC_MQPRIO_QOPT_OFFLOAD 1

/* tc_setup_cb_egdev_register is defined */
/* #undef HAVE_TC_SETUP_CB_EGDEV_REGISTER */

/* tc_setup_flow_action is defined */
#define HAVE_TC_SETUP_FLOW_ACTION_FUNC 1

/* tc_setup_flow_action has rtnl_held */
/* #undef HAVE_TC_SETUP_FLOW_ACTION_WITH_RTNL_HELD */

/* TC_TC_SETUP_FT is defined */
#define HAVE_TC_SETUP_FT 1

/* tc_setup_offload_action is defined */
/* #undef HAVE_TC_SETUP_OFFLOAD_ACTION_FUNC */

/* tc_setup_offload_action is defined and get 3 param */
/* #undef HAVE_TC_SETUP_OFFLOAD_ACTION_FUNC_HAS_3_PARAM */

/* TC_SETUP_QDISC_MQPRIO is defined */
#define HAVE_TC_SETUP_QDISC_MQPRIO 1

/* TC_SETUP_TYPE is defined */
#define HAVE_TC_SETUP_TYPE 1

/* linux/skbuff.h struct tc_skb_ext has act-miss */
#define HAVE_TC_SKB_EXT_ACT_MISS 1

/* tc_skb_ext_alloc is defined */
#define HAVE_TC_SKB_EXT_ALLOC 1

/* struct tc_to_netdev has egress_dev */
/* #undef HAVE_TC_TO_NETDEV_EGRESS_DEV */

/* struct tc_to_netdev has tc */
/* #undef HAVE_TC_TO_NETDEV_TC */

/* struct timerqueue_head has struct rb_root_cached */
#define HAVE_TIMERQUEUE_HEAD_RB_ROOT_CACHED 1

/* timer_setup is defined */
#define HAVE_TIMER_SETUP 1

/* struct tlsdev_ops has tls_dev_resync */
#define HAVE_TLSDEV_OPS_HAS_TLS_DEV_RESYNC 1

/* net/tls.h has tls_driver_ctx */
#define HAVE_TLS_DRIVER_CTX 1

/* net/tls.h has tls_is_skb_tx_device_offloaded */
/* #undef HAVE_TLS_IS_SKB_TX_DEVICE_OFFLOADED */

/* tls_offload_context_tx has destruct_work as member */
#define HAVE_TLS_OFFLOAD_DESTRUCT_WORK 1

/* net/tls.h has struct tls_offload_resync_async is defined */
#define HAVE_TLS_OFFLOAD_RESYNC_ASYNC_STRUCT 1

/* net/tls.h has tls_offload_rx_force_resync_request */
/* #undef HAVE_TLS_OFFLOAD_RX_FORCE_RESYNC_REQUEST */

/* net/tls.h has tls_offload_rx_resync_async_request_start */
#define HAVE_TLS_OFFLOAD_RX_RESYNC_ASYNC_REQUEST_START 1

/* trace_block_bio_complete has 2 param */
#define HAVE_TRACE_BLOCK_BIO_COMPLETE_2_PARAM 1

/* trace_block_bio_remap has 4 param */
/* #undef HAVE_TRACE_BLOCK_BIO_REMAP_4_PARAM */

/* include/trace/trace_events.h exists */
#define HAVE_TRACE_EVENTS_H 1

/* trace/events/rdma_core.h exists */
#define HAVE_TRACE_EVENTS_RDMA_CORE_HEADER 1

/* trace/events/sock.h has trace_sk_data_ready */
/* #undef HAVE_TRACE_EVENTS_TRACE_SK_DATA_READY */

/* rpcrdma.h exists */
/* #undef HAVE_TRACE_RPCRDMA_H */

/* trace_xdp_exception is defined */
#define HAVE_TRACE_XDP_EXCEPTION 1

/* linux/atomic/atomic-instrumented.h has try_cmpxchg */
#define HAVE_TRY_CMPXCHG 1

/* type cycle_t is defined in linux/types.h */
/* #undef HAVE_TYPE_CYCLE_T */

/* type rcu_callback_t is defined */
#define HAVE_TYPE_RCU_CALLBACK_T 1

/* type __poll_t is defined */
#define HAVE_TYPE___POLL_T 1

/* uapi/linux/nvme_ioctl.h exists */
#define HAVE_UAPI_LINUX_NVME_IOCTL_H 1

/* uapi/linux/nvme_ioctl.h has NVME_IOCTL_RESCAN */
#define HAVE_UAPI_LINUX_NVME_IOCTL_RESCAN 1

/* uapi/linux/nvme_ioctl.h has NVME_URING_CMD_ADMIN */
/* #undef HAVE_UAPI_LINUX_NVME_NVME_URING_CMD_ADMIN */

/* uapi/linux/nvme_ioctl.h has struct nvme_passthru_cmd64 */
#define HAVE_UAPI_LINUX_NVME_PASSTHRU_CMD64 1

/* uapi/linux/tls.h exists */
#define HAVE_UAPI_LINUX_TLS_H 1

/* udp4_hwcsum is exported by the kernel */
#define HAVE_UDP4_HWCSUM 1

/* udp_tunnel.h has struct udp_tunnel_nic_info is defined */
#define HAVE_UDP_TUNNEL_NIC_INFO 1

/* udp_tunnel.h has udp_tunnel_drop_rx_port is defined */
#define HAVE_UDP_TUNNEL_RX_INFO 1

/* ib_umem_notifier_invalidate_range_start has parameter blockable */
/* #undef HAVE_UMEM_NOTIFIER_PARAM_BLOCKABLE */

/* net/xdp.h has __xdp_rxq_info_reg */
/* #undef HAVE_UNDERSCORE_XDP_RXQ_INFO_REG */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* include/linux/units.h exists */
#define HAVE_UNITS_H 1

/* struct flow_block_offload has unlocked_driver_cb */
#define HAVE_UNLOCKED_DRIVER_CB 1

/* unpin_user_pages_dirty_lock is exported by the kernel */
#define HAVE_UNPIN_USER_PAGES_DIRTY_LOCK_EXPORTED 1

/* unpin_user_page_range_dirty_lock is exported by the kernel */
#define HAVE_UNPIN_USER_PAGE_RANGE_DIRTY_LOCK_EXPORTED 1

/* unregister_netdevice_notifier_net is defined */
#define HAVE_UNREGISTER_NETDEVICE_NOTIFIER_NET 1

/* update_pmtu has 4 paramters */
/* #undef HAVE_UPDATE_PMTU_4_PARAMS */

/* flow_cls_offload has use_act_stats */
/* #undef HAVE_USE_ACT_STATS */

/* uuid_be_to_bin is defined */
/* #undef HAVE_UUID_BE_TO_BIN */

/* uuid_equal is defined */
#define HAVE_UUID_EQUAL 1

/* uuid_gen is defined */
#define HAVE_UUID_GEN 1

/* uuid_is_null is defined */
#define HAVE_UUID_IS_NULL 1

/* struct vdpa_config_ops has get_vq_dma_dev defined */
/* #undef HAVE_VDPA_CONFIG_OPS_GET_VQ_DMA_DEV */

/* sturct vdpa_dev_set_config has device_features */
/* #undef HAVE_VDPA_SET_CONFIG_HAS_DEVICE_FEATURES */

/* linux/vfio_pci_core.h exists */
#define HAVE_VFIO_PCI_CORE_H 1

/* vfio_pci_core_init_dev exists */
/* #undef HAVE_VFIO_PCI_CORE_INIT */

/* sturct vfio_precopy_info exists */
/* #undef HAVE_VFIO_PRECOPY_INFO */

/* trust is defined */
#define HAVE_VF_INFO_TRUST 1

/* vlan_proto is defined */
#define HAVE_VF_VLAN_PROTO 1

/* struct vlan_ethhdr has addrs member */
/* #undef HAVE_VLAN_ETHHDR_HAS_ADDRS */

/* vlan_get_encap_level is defined */
/* #undef HAVE_VLAN_GET_ENCAP_LEVEL */

/* linux/vmalloc.h has __vmalloc 3 params */
/* #undef HAVE_VMALLOC_3_PARAM */

/* vm_fault_t is defined */
#define HAVE_VM_FAULT_T 1

/* vm_flags_clear exists */
/* #undef HAVE_VM_FLAGS_CLEAR */

/* vm_operations_struct has .fault */
#define HAVE_VM_OPERATIONS_STRUCT_HAS_FAULT 1

/* vxlan_vni_field is defined */
#define HAVE_VXLAN_VNI_FIELD 1

/* want_init_on_alloc is defined */
#define HAVE_WANT_INIT_ON_ALLOC 1

/* WQ_NON_REENTRANT is defined */
/* #undef HAVE_WQ_NON_REENTRANT */

/* xa_array is defined */
#define HAVE_XARRAY 1

/* xa_for_each_range is defined */
#define HAVE_XA_FOR_EACH_RANGE 1

/* struct xfrmdev_ops has member xdo_dev_policy_add */
#define HAVE_XDO_DEV_POLICY_ADD 1

/* struct xfrmdev_ops has member xdo_dev_policy_add get extack */
#define HAVE_XDO_DEV_POLICY_ADD_GET_EXTACK 1

/* struct xfrmdev_ops has member xdo_dev_state_advance_esn */
#define HAVE_XDO_DEV_STATE_ADVANCE_ESN 1

/* struct xfrmdev_ops has member xdo_dev_state_update_curlft */
#define HAVE_XDO_DEV_STATE_UPDATE_CURLFT 1

/* struct xfrmdev_ops has member xdo_dev_state_add get extack */
#define HAVE_XDO_XFRM_ADD_STATE_GET_EXTACK 1

/* filter.h xdp_buff data_hard_start is defined */
#define HAVE_XDP_BUFF_DATA_HARD_START_FILTER_H 1

/* xdp.h xdp_buff data_hard_start is defined */
#define HAVE_XDP_BUFF_DATA_HARD_START_XDP_H 1

/* xdp_buff has daya_meta as member */
#define HAVE_XDP_BUFF_HAS_DATA_META 1

/* xdp_buff has flags as member */
/* #undef HAVE_XDP_BUFF_HAS_FLAGS */

/* xdp_buff has frame_sz as member */
#define HAVE_XDP_BUFF_HAS_FRAME_SZ 1

/* linux/filter.h struct xdp_buff exists */
#define HAVE_XDP_BUFF_ON_FILTER 1

/* net/xdp.h has xdp_convert_buff_to_frame */
#define HAVE_XDP_CONVERT_BUFF_TO_FRAME 1

/* net/xdp.h has convert_to_xdp_frame */
/* #undef HAVE_XDP_CONVERT_TO_XDP_FRAME_IN_NET_XDP */

/* net/xdp.h has convert_to_xdp_frame workaround for
   5.4.17-2011.1.2.el8uek.x86_64 */
/* #undef HAVE_XDP_CONVERT_TO_XDP_FRAME_IN_UEK_KABI */

/* filter.h has xdp_do_flush_map */
#define HAVE_XDP_DO_FLUSH_MAP 1

/* net/xdp.h struct xdp_frame_bulk exists */
#define HAVE_XDP_FRAME_BULK 1

/* struct xdp_frame is defined */
#define HAVE_XDP_FRAME_IN_NET_XDP 1

/* struct xdp_frame is defined in 5.4.17-2011.1.2.el8uek.x86_64 */
/* #undef HAVE_XDP_FRAME_IN_UEK_KABI */

/* xdp_update_skb_shared_info is defined */
#define HAVE_XDP_GET_SHARED_INFO_FROM_BUFF 1

/* struct bpf_prog_aux has xdp_has_frags as member */
/* #undef HAVE_XDP_HAS_FRAGS */

/* net/xdp.h struct xdp_buff exists */
#define HAVE_XDP_H_HAVE_XDP_BUFF 1

/* net/xdp.h has xdp_init_buff */
#define HAVE_XDP_INIT_BUFF 1

/* struct net_device has struct net_device has xdp_metadata_ops member */
/* #undef HAVE_XDP_METADATA_OPS */

/* XDP_REDIRECT is defined */
#define HAVE_XDP_REDIRECT 1

/* net/xdp.h has struct xdp_rxq_info */
#define HAVE_XDP_RXQ_INFO_IN_NET_XDP 1

/* net/xdp.h has struct xdp_rxq_info WA for 5.4.17-2011.1.2.el8uek.x86_64 */
/* #undef HAVE_XDP_RXQ_INFO_IN_UEK_KABI */

/* net/xdp.h has xdp_rxq_info_reg get 4 params */
#define HAVE_XDP_RXQ_INFO_REG_4_PARAMS 1

/* net/xdp.h has xdp_rxq_info_reg_mem_model */
#define HAVE_XDP_RXQ_INFO_REG_MEM_MODEL_IN_NET_XDP 1

/* net/xdp.h has xdp_rxq_info_reg_mem_model workaround for
   5.4.17-2011.1.2.el8uek.x86_64 */
/* #undef HAVE_XDP_RXQ_INFO_REG_MEM_MODEL_IN_UEK_KABI */

/* xdp_set_data_meta_invalid is defined */
#define HAVE_XDP_SET_DATA_META_INVALID_FILTER_H 1

/* xdp_set_data_meta_invalid is defined */
#define HAVE_XDP_SET_DATA_META_INVALID_XDP_H 1

/* xdp_set_features_flag defined */
/* #undef HAVE_XDP_SET_FEATURES_FLAG */

/* net/xdp_sock_drv.h exists */
#define HAVE_XDP_SOCK_DRV_H 1

/* chunk_size is defined */
#define HAVE_XDP_UMEM_CHUNK_SIZE 1

/* flags is defined */
#define HAVE_XDP_UMEM_FLAGS 1

/* xdp_update_skb_shared_info is defined */
/* #undef HAVE_XDP_UPDATE_SKB_SHARED_INFO */

/* XDRBUF_SPARSE_PAGES has defined in linux/sunrpc/xdr.h */
#define HAVE_XDRBUF_SPARSE_PAGES 1

/* xdr_buf_subsegment get const */
#define HAVE_XDR_BUF_SUBSEGMENT_CONST 1

/* xdr_decode_rdma_segment has defined */
#define HAVE_XDR_DECODE_RDMA_SEGMENT 1

/* xdr_encode_rdma_segment has defined */
#define HAVE_XDR_ENCODE_RDMA_SEGMENT 1

/* xdr_init_decode has rqst as a parameter */
#define HAVE_XDR_INIT_DECODE_RQST_ARG 1

/* xdr_init_encode has rqst as a parameter */
#define HAVE_XDR_INIT_ENCODE_RQST_ARG 1

/* xdr_item_is_absent has defined */
#define HAVE_XDR_ITEM_IS_ABSENT 1

/* xdr_stream_encode_item_absent has defined */
#define HAVE_XDR_STREAM_ENCODE_ITEM_ABSENT 1

/* xdr_stream_remaining as defined */
#define HAVE_XDR_STREAM_REMAINING 1

/* xfrm_dev_offload has dir as member */
#define HAVE_XFRM_DEV_DIR 1

/* xfrm_dev_offload has flags */
#define HAVE_XFRM_DEV_OFFLOAD_FLAG_ACQ 1

/* xfrm_dev_offload has real_dev as member */
#define HAVE_XFRM_DEV_REAL_DEV 1

/* xfrm_dev_offload has type as member */
#define HAVE_XFRM_DEV_TYPE 1

/* struct xfrm_offload has inner_ipproto */
#define HAVE_XFRM_OFFLOAD_INNER_IPPROTO 1

/* XFRM_OFFLOAD_PACKET is defined */
#define HAVE_XFRM_OFFLOAD_PACKET 1

/* xfrm_dev_offload has state as member */
/* #undef HAVE_XFRM_STATE_DIR */

/* xfrm_state_offload has real_dev as member */
/* #undef HAVE_XFRM_STATE_REAL_DEV */

/* struct svc_xprt_ops has 'xpo_read_payload' field */
/* #undef HAVE_XPO_READ_PAYLOAD */

/* struct svc_xprt_ops has 'xpo_release_ctxt' field */
#define HAVE_XPO_RELEASE_CTXT 1

/* struct svc_xprt_ops has 'xpo_result_payload' field */
#define HAVE_XPO_RESULT_PAYLOAD 1

/* xpo_secure_port is defined and returns void */
#define HAVE_XPO_SECURE_PORT_NO_RETURN 1

/* xprt_add_backlog is exported by the sunrpc core */
#define HAVE_XPRT_ADD_BACKLOG 1

/* struct xprt_class has 'netid' field */
#define HAVE_XPRT_CLASS_NETID 1

/* xprt_lock_connect is exported by the sunrpc core */
#define HAVE_XPRT_LOCK_CONNECT 1

/* *send_request has 'struct rpc_rqst *req' as a param */
#define HAVE_XPRT_OPS_SEND_REQUEST_RQST_ARG 1

/* xprt_pin_rqst is exported by the sunrpc core */
#define HAVE_XPRT_PIN_RQST 1

/* struct rpc_xprt has 'queue_lock' field */
#define HAVE_XPRT_QUEUE_LOCK 1

/* xprt_reconnect_delay is exported by the kernel */
#define HAVE_XPRT_RECONNECT_DELAY 1

/* get cong request */
#define HAVE_XPRT_REQUEST_GET_CONG 1

/* xprt_wait_for_buffer_space has xprt as a parameter */
#define HAVE_XPRT_WAIT_FOR_BUFFER_SPACE_RQST_ARG 1

/* xsk_buff_alloc is defined */
#define HAVE_XSK_BUFF_ALLOC 1

/* xsk_buff_alloc_batch is defined */
/* #undef HAVE_XSK_BUFF_ALLOC_BATCH */

/* xsk_buff_dma_sync_for_cpu get 2 params */
#define HAVE_XSK_BUFF_DMA_SYNC_FOR_CPU_2_PARAMS 1

/* xsk_buff_xdp_get_frame_dma is defined */
#define HAVE_XSK_BUFF_GET_FRAME_DMA 1

/* xsk_buff_set_size is defined */
/* #undef HAVE_XSK_BUFF_SET_SIZE */

/* xsk_umem_adjust_offset is defined */
/* #undef HAVE_XSK_UMEM_ADJUST_OFFSET */

/* net/xdp_sock.h has xsk_umem_consume_tx get 2 params */
/* #undef HAVE_XSK_UMEM_CONSUME_TX_GET_2_PARAMS_IN_SOCK */

/* net/xdp_soc_drv.h has xsk_umem_consume_tx get 2 params */
/* #undef HAVE_XSK_UMEM_CONSUME_TX_GET_2_PARAMS_IN_SOCK_DRV */

/* xsk_umem_release_addr_rq is defined */
/* #undef HAVE_XSK_UMEM_RELEASE_ADDR_RQ */

/* __atomic_add_unless is defined */
/* #undef HAVE___ATOMIC_ADD_UNLESS */

/* __blkdev_issue_discard is defined */
#define HAVE___BLKDEV_ISSUE_DISCARD 1

/* __blkdev_issue_discard has 5 params */
/* #undef HAVE___BLKDEV_ISSUE_DISCARD_5_PARAM */

/* __cancel_delayed_work is defined */
/* #undef HAVE___CANCEL_DELAYED_WORK */

/* __ethtool_get_link_ksettings is defined */
#define HAVE___ETHTOOL_GET_LINK_KSETTINGS 1

/* __flow_indr_block_cb_register is defined */
/* #undef HAVE___FLOW_INDR_BLOCK_CB_REGISTER */

/* __get_task_comm is exported by the kernel */
#define HAVE___GET_TASK_COMM_EXPORTED 1

/* HAVE___IP_DEV_FIND is exported by the kernel */
#define HAVE___IP_DEV_FIND 1

/* __ip_tun_set_dst has 7 params */
/* #undef HAVE___IP_TUN_SET_DST_7_PARAMS */

/* netdevice.h has __netdev_tx_sent_queue */
#define HAVE___NETDEV_TX_SENT_QUEUE 1

/* __tc_indr_block_cb_register is defined */
/* #undef HAVE___TC_INDR_BLOCK_CB_REGISTER */

/* Name of package */

/* Define to the address where bug reports for this package should be sent. */

/* Define to the full name of this package. */

/* Define to the full name and version of this package. */

/* Define to the one symbol short name of this package. */

/* Define to the home page for this package. */

/* Define to the version of this package. */

/* The size of `unsigned long long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG_LONG 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */

/* Make sure LINUX_BACKPORT macro is defined for all external users */
#ifndef LINUX_BACKPORT
#define LINUX_BACKPORT(__sym) backport_ ##__sym
#endif

/* Defines in this section calculated in ofed_scripts/configure 
 * based on defines prior this section
 *  _________________________________________________________ */
#define REBASE_STAGE_BACKPORTS 1
#define HAVE_BASECODE_EXTRAS 1
#define HAVE_TC_SETUP_FLOW_ACTION 1
#define HAVE_HMM_RANGE_FAULT_SUPPORT 1
#define HAVE_DEVLINK_HEALTH_REPORT_SUPPORT 1
#define HAVE_KTLS_RX_SUPPORT 1
#define HAVE_DEVLINK_PORT_ATRRS_SET_GET_SUPPORT 1
#define HAVE_DEVLINK_PORT_ATTRS_PCI_PF_SET 1
/* #undef HAVE_XSK_UMEM_CONSUME_TX_GET_2_PARAMS */
#define HAVE_SVC_FILL_WRITE_VECTOR 1
#define HAVE_GET_USER_PAGES_GUP_FLAGS 1
#define HAVE_MMGET_NOT_ZERO 1
/* #undef HAVE_XDP_SUPPORT */
/* #undef HAVE_VFIO_SUPPORT */
#define HAVE_XDP_FRAME 1
/* #undef HAVE_XSK_ZERO_COPY_SUPPORT */
#define HAVE_XDP_RXQ_INFO 1
#define HAVE_NET_XDP_H 1
/* #undef HAVE_XDP_CONVERT_TO_XDP_FRAME */
#define HAVE_XDP_RXQ_INFO_REG_MEM_MODEL 1
#define HAVE_KERNEL_WITH_VXLAN_SUPPORT_ON 1
#define HAVE_IS_PCI_P2PDMA_PAGE 1
#define HAVE_BLK_MQ_BUSY_TAG_ITER_FN_BOOL 1
#define HAVE_BLK_TYPES_REQ_OP_DRV_OUT 1
#define HAVE_DEVLINK_PORT_TYPE_ETH_SET 1
/* #undef HAVE_DEVLINK_PER_AUXDEV */
/* #undef HAVE_NO_REFCNT_BIAS */
#define HAVE_TC_CLS_OFFLOAD_EXTACK 1
#define HAVE_TC_CLSFLOWER_STATS 1
#define HAVE_TC_CLS_FLOWER_OFFLOAD_HAS_STATS_FIELD 1
#define HAVE_TC_CLS_FLOWER_OFFLOAD_COMMON 1
#define HAVE_PRIO_CHAIN_SUPPORT 1
#define HAVE_TC_INDR_API 1
#define HAVE_DEVICE_ADD_DISK_3_ARGS 1
/* #undef HAVE_TCF_PEDIT_TCFP_KEYS_EX */
#define HAVE_NIC_TEMPERATURE_SUPPORTED 1
/* #undef HAVE_VDPA_SUPPORT */
#define HAVE_DEVLINK_RESOURCE_SUPPORT 1
#define HAVE_NET_PAGE_POOL_H 1
#define HAVE_SHAMPO_SUPPORT 1
#define HAVE_PAGE_POOL_GET_DMA_ADDR 1
#define HAVE_PAGE_POLL_NID_CHANGED 1
#define HAVE_XDP_SET_DATA_META_INVALID 1
#define HAVE_XDP_BUFF_DATA_HARD_START 1
#define HAVE_FILTER_H_HAVE_XDP_BUFF 1
/* #undef HAVE_GUP_MUST_UNSHARE_GET_3_PARAMS */
#define HAVE_PAGE_POOL_RELEASE_PAGE 1
#define HAVE_DEVLINK_FMSG_BINARY_PAIR_PUT_ARG_U32 1
