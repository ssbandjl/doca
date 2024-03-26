/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2020 Mellanox Technologies, Ltd
 */

#ifndef RTE_PMD_PRIVATE_MLX5_H_
#define RTE_PMD_PRIVATE_MLX5_H_

#include <rte_compat.h>

/**
 * @file
 * MLX5 public header.
 *
 * This interface provides the ability to support private PMD
 * dynamic flags.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define RTE_PMD_MLX5_FINE_GRANULARITY_INLINE "mlx5_fine_granularity_inline"

/**
 * Returns the dynamic flags name, that are supported.
 *
 * @param[out] names
 *   Array that is used to return the supported dynamic flags names.
 * @param[in] n
 *   The number of elements in the names array.
 *
 * @return
 *   The number of dynamic flags that were copied if not negative.
 *   Otherwise:
 *   - ENOMEM - not enough entries in the array
 *   - EINVAL - invalid array entry
 */
__rte_experimental
int rte_pmd_mlx5_get_dyn_flag_names(char *names[], unsigned int n);

#define MLX5_DOMAIN_BIT_NIC_RX	(1 << 0) /**< NIC RX domain bit mask. */
#define MLX5_DOMAIN_BIT_NIC_TX	(1 << 1) /**< NIC TX domain bit mask. */
#define MLX5_DOMAIN_BIT_FDB	(1 << 2) /**< FDB (TX + RX) domain bit mask. */

/**
 * Synchronize the flows to make them take effort on hardware.
 * It only supports DR flows now. For DV and Verbs flows, there is no need to
 * call this function, and a success will return directly in case of Verbs.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] domains
 *   Refer to "/usr/include/infiniband/mlx5dv.h".
 *   Bitmask of domains in which the synchronization will be done.
 *   MLX5_DOMAIN_BIT* macros are used to specify the domains.
 *   An ADD or OR operation could be used to synchronize flows in more than
 *   one domain per call.
 *
 * @return
 *   - (0) if successful.
 *   - Negative value if an error.
 */
__rte_experimental
int rte_pmd_mlx5_sync_flow(uint16_t port_id, uint32_t domains);

/**
 * External Rx queue rte_flow index minimal value.
 */
#define MLX5_EXTERNAL_RX_QUEUE_ID_MIN (UINT16_MAX - 1000 + 1)

/**
 * Tag level to set the linear hash index.
 */
#define MLX5_LINEAR_HASH_TAG_INDEX 255

/**
 * Update mapping between rte_flow queue index (16 bits) and HW queue index (32
 * bits) for RxQs which is created outside the PMD.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] dpdk_idx
 *   Queue index in rte_flow.
 * @param[in] hw_idx
 *   Queue index in hardware.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_errno is set.
 *   Possible values for rte_errno:
 *   - EEXIST - a mapping with the same rte_flow index already exists.
 *   - EINVAL - invalid rte_flow index, out of range.
 *   - ENODEV - there is no Ethernet device for this port id.
 *   - ENOTSUP - the port doesn't support external RxQ.
 */
__rte_experimental
int rte_pmd_mlx5_external_rx_queue_id_map(uint16_t port_id, uint16_t dpdk_idx,
					  uint32_t hw_idx);

/**
 * Remove mapping between rte_flow queue index (16 bits) and HW queue index (32
 * bits) for RxQs which is created outside the PMD.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] dpdk_idx
 *   Queue index in rte_flow.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_errno is set.
 *   Possible values for rte_errno:
 *   - EINVAL - invalid index, out of range, still referenced or doesn't exist.
 *   - ENODEV - there is no Ethernet device for this port id.
 *   - ENOTSUP - the port doesn't support external RxQ.
 */
__rte_experimental
int rte_pmd_mlx5_external_rx_queue_id_unmap(uint16_t port_id,
					    uint16_t dpdk_idx);

/**
 * The rate of the host port shaper will be updated directly at the next
 * available descriptor threshold event to the rate that comes with this flag set;
 * set rate 0 to disable this rate update.
 * Unset this flag to update the rate of the host port shaper directly in
 * the API call; use rate 0 to disable the current shaper.
 */
#define MLX5_HOST_SHAPER_FLAG_AVAIL_THRESH_TRIGGERED 0

/**
 * Configure a HW shaper to limit Tx rate for a host port.
 * The configuration will affect all the ethdev ports belonging to
 * the same rte_device.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] rate
 *   Unit is 100Mbps, setting the rate to 0 disables the shaper.
 * @param[in] flags
 *   Host shaper flags.
 * @return
 *   0 : operation success.
 *   Otherwise:
 *   - ENOENT - no ibdev interface.
 *   - EBUSY  - the register access unit is busy.
 *   - EIO    - the register access command meets IO error.
 */
__rte_experimental
int rte_pmd_mlx5_host_shaper_config(int port_id, uint8_t rate, uint32_t flags);

struct flexio_process;
struct flexio_outbox;
struct rte_pmd_mlx5_dev_process;
struct rte_pmd_mlx5_dev_ctx;
struct rte_pmd_mlx5_dev_table;
struct rte_pmd_mlx5_dev_action;

enum rte_pmd_mlx5_host_device_type {
	RTE_PMD_MLX5_DEVICE_TYPE_LOCAL,
	RTE_PMD_MLX5_DEVICE_TYPE_DPA,
};

struct rte_pmd_mlx5_host_device_info {
	enum rte_pmd_mlx5_host_device_type type;
	union {
		struct {
			struct flexio_process *process;
			struct flexio_outbox *outbox;
		} dpa;
	};
	/* Number of device queue */
	uint16_t queues;
	/* Max number of queued rule create delete operations  */
	uint16_t queue_size;
};

struct rte_pmd_mlx5_host_action {
	enum rte_flow_action_type type;
	/* Table to bind the actions from */
	struct rte_flow_template_table *table;
	union {
		struct {
			/* ID of destination table */
			uint32_t index;
		} jump;

		struct {
			/* Action template index containing modify fields */
			uint8_t template_index;
		} modify_field;

		struct {
			struct mlx5dv_devx_obj *obj;
			uint32_t id;
		} count;
	};
};

/**
 * PMD private items/actions
 */
#define RTE_PMD_MLX5_FLOW_ITEM_TYPE_IPSEC_SYNDROME (INT_MIN + 1000000 + 4)
#define RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_INSERT_TRAILER (INT_MIN + 1000000 + 5)
#define RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_REMOVE_TRAILER (INT_MIN + 1000000 + 6)
#define RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_INSERT_HEADER (INT_MIN + 1000000 + 7)
#define RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_REMOVE_HEADER (INT_MIN + 1000000 + 8)
#define RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_CRYPTO (INT_MIN + 1000000 + 9)

#define RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_MIN RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_INSERT_TRAILER
#define RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_MAX RTE_PMD_MLX5_RTE_FLOW_ACTION_TYPE_CRYPTO

enum rte_pmd_mlx5_trailer_type {
	RTE_PMD_MLX5_TRAILER_TYPE_IPSEC = 1
};

struct mlx5_flow_action_trailer {
	enum rte_pmd_mlx5_trailer_type type;
	size_t size;
};

enum rte_pmd_mlx5_header_anchor {
	RTE_PMD_MLX5_HEADER_ANCHOR_NONE = 0,
	RTE_PMD_MLX5_HEADER_ANCHOR_PACKET,
	RTE_PMD_MLX5_HEADER_ANCHOR_MAC,
	RTE_PMD_MLX5_HEADER_ANCHOR_FIRST_VLAN,
	RTE_PMD_MLX5_HEADER_ANCHOR_IPV6_IPV4,
	RTE_PMD_MLX5_HEADER_ANCHOR_ESP,
	RTE_PMD_MLX5_HEADER_ANCHOR_TCP_UDP,
	RTE_PMD_MLX5_HEADER_ANCHOR_TUNNEL_HEADER,
	RTE_PMD_MLX5_HEADER_ANCHOR_INNER_MAC,
	RTE_PMD_MLX5_HEADER_ANCHOR_INNER_IPV6_IPV4,
	RTE_PMD_MLX5_HEADER_ANCHOR_INNER_TCP_UDP,
	RTE_PMD_MLX5_HEADER_ANCHOR_L4_PAYLOAD,
	RTE_PMD_MLX5_HEADER_ANCHOR_INNER_L4_PAYLOAD,
	RTE_PMD_MLX5_HEADER_ANCHOR_MAX
};

struct mlx5_flow_action_insert_header {
	enum rte_pmd_mlx5_header_anchor start;
	uint8_t *data;
	size_t size;
	uint8_t offset;
	bool encap;
	bool push_esp;
};

struct mlx5_flow_action_remove_header {
	enum rte_pmd_mlx5_header_anchor start;
	enum rte_pmd_mlx5_header_anchor end;
	size_t size;
	bool decap;
};

enum rte_pmd_mlx5_crypto_opcode {
	RTE_PMD_MLX5_CRYPTO_OPCODE_ENCRYPT = 0,
	RTE_PMD_MLX5_CRYPTO_OPCODE_DECRYPT
};

enum rte_pmd_mlx5_crypto_type {
	RTE_PMD_MLX5_CRYPTO_TYPE_IPSEC = 0
};

struct mlx5_flow_action_crypto {
	enum rte_pmd_mlx5_crypto_opcode opcode;
	enum rte_pmd_mlx5_crypto_type type;
	/* DevX object ID. */
	uint32_t crypto_id;
	/* Action offset in bulk array. */
	uint32_t offset;
	/* Jump to group after ASO is taken successfully. */
	uint32_t next_group;
	/* Jump to group to skip ASO for bad decryption syndromes. */
	uint32_t fail_group;
	bool aso;
};

struct mlx5_flow_item_ipsec_syndrome {
	union {
		uint8_t syndrome;
		uintptr_t pad;
	};
};

/**
 * Enable traffic for external SQ.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] sq_num
 *   SQ HW number.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_errno is set.
 *   Possible values for rte_errno:
 *   - EINVAL - invalid sq_number or port type.
 *   - ENODEV - there is no Ethernet device for this port id.
 */
__rte_experimental
int rte_pmd_mlx5_external_sq_enable(uint16_t port_id, uint32_t sq_num);

/**
 * Export Tx UAR created by DPDK for the input port.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[out] uar
 *   UAR object to return
 * @return
 *   0 : operation success.
 *   Otherwise:
 *   - ENODEV - Invalid port_id.
 *   - EINVAL - Invalid argument.
 */
__rte_experimental
int
rte_pmd_mlx5_export_uar(uint16_t port_id, void **uar);

/**
 * Convert UTC timestamp in the corresponding clock queue wci.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] ts
 *   UTC timestamp
 * @return
 *   Clock queue event index.
 */
__rte_experimental
uint32_t
rte_pmd_mlx5_txpp_convert_tx_ts(uint16_t port_id, uint64_t ts);

/**
 * Return clock queue id created on a device.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @return
 *   Clock queue id on success.
 *   Otherwise:
 *   - ENODEV - Invalid port_id.
 *   - ENOTSUP - Clock queue not created.
 */
__rte_experimental
int
rte_pmd_mlx5_txpp_idx(uint16_t port_id);

/**
 * Open new device process, the device can be DPA, ARM or Other.
 * All used  mlx5dr objects should be binded to the device process inorder
 * to be used on the device.
 * NOTE: This function should only be called from the host.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] info
 *   The device info.
 * @return device process on success NULL otherwise.
 */
__rte_experimental
struct rte_pmd_mlx5_dev_process *
rte_pmd_mlx5_host_process_open(uint16_t port_id,
			       struct rte_pmd_mlx5_host_device_info *info);

/**
 * Close device process.
 * NOTE: This function should only be called from the host.
 *
 * @param[in] dev_process
 *   Open device process.
 * @return 0 on success non zero otherwise.
 */
__rte_experimental
int
rte_pmd_mlx5_host_process_close(struct rte_pmd_mlx5_dev_process *dev_process);

/**
 * Get mlx5dr context from process used in mlx5dr_dev.
 * NOTE: This function should only be called from the host.
 *
 * @param[in] dev_process
 *   Open device process.
 * @return device context on success NULL otherwise.
 */
__rte_experimental
struct rte_pmd_mlx5_dev_ctx *
rte_pmd_mlx5_host_get_dev_ctx(struct rte_pmd_mlx5_dev_process *dev_process);

/**
 * Bind existing rte table to device process.
 * NOTE: This function should only be called from the host.
 *
 * @param[in] dev_process
 *   Open device process.
 * @param[in] table
 *   Existing rte table.
 * @return device table on success NULL otherwise.
 */
__rte_experimental
struct rte_pmd_mlx5_dev_table *
rte_pmd_mlx5_host_table_bind(struct rte_pmd_mlx5_dev_process *dev_process,
			     struct rte_flow_template_table *table);

/**
 * Unbind rte table from device process.
 * NOTE: This function should only be called from the host.
 *
 * @param[in] dev_process
 *   Open device process.
 * @param[in] dev_table
 *   Binded device table.
 * @return 0 on success non zero otherwise
 */
__rte_experimental
int
rte_pmd_mlx5_host_table_unbind(struct rte_pmd_mlx5_dev_process *dev_process,
			       struct rte_pmd_mlx5_dev_table *dev_table);

/**
 * Bind existing rte action to device process.
 * NOTE: This function should only be called from the host.
 *
 * @param[in] dev_process
 *   Open device process.
 * @param[in] action
 *   rte flow action type to data for binding.
 * @return device action on success NULL otherwise.
 */
__rte_experimental
struct rte_pmd_mlx5_dev_action *
rte_pmd_mlx5_host_action_bind(struct rte_pmd_mlx5_dev_process *dev_process,
			      struct rte_pmd_mlx5_host_action *action);

/**
 * Unbind dev action from device process.
 * NOTE: This function should only be called from the host.
 *
 * @param[in] dev_process
 *   Open device process.
 * @param[in] dev_action
 *   Binded device action.
 * @return 0 on success non zero otherwise.
 */
__rte_experimental
int
rte_pmd_mlx5_host_action_unbind(struct rte_pmd_mlx5_dev_process *dev_process,
				struct rte_pmd_mlx5_dev_action *dev_action);

/**
 * Get dev rule handle size since the handle is an opaque internal struct.
 * NOTE: This function should only be called from the host.
 *
 * @return size of a single dev_rule handle.
 */
__rte_experimental
size_t
rte_pmd_mlx5_host_get_dev_rule_handle_size(void);

/**
 * TAG related parameter selection.
 */
enum rte_pmd_mlx5_tag_param {
	RTE_PMD_MLX5_IPSEC_ASO_RETURN_REG,
	/**< The value should be used in IPsec offload create PRM call */
	RTE_PMD_MLX5_IPSEC_SYNDROME_TAG,
	/**< RTE Flow TAG resource index for IPsec ASO syndrome */
	RTE_PMD_MLX5_IPSEC_SEQ_NUMBER_TAG,
	/**< RTE Flow TAG resource index for IPsec sequence number */
	RTE_PMD_MLX5_LINEAR_HASH_TAG,
	/**< RTE Flow TAG resource index for linear hash value */
};

/**
 * Query tag related parameters used by PMD for specific port.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] param
 *   The tag parameter identifier.
 * @param[out] value
 *   Pointer to the byte value be filled with result.
 * @return
 *   0 : operation success.
 *   Otherwise:
 *   -ENODEV - Invalid port_id.
 *   -EINVAL - Invalid argument.
 *   -ENOTSUP - Parameter not supported.
 */
__rte_experimental
int
rte_pmd_mlx5_query_tag_param(uint16_t port_id,
			     enum rte_pmd_mlx5_tag_param param,
			     uint8_t *value);

/**
 * User configuration structure using to create parser for single GENEVE TLV option.
 */
struct mlx5_pmd_geneve_tlv {
	/**
	 * The class of the GENEVE TLV option.
	 * Relevant only when 'match_on_class_mode' is 1.
	 */
	rte_be16_t option_class;
	/**
	 * The type of the GENEVE TLV option.
	 * This field is the identifier of the option.
	 */
	uint8_t option_type;
	/**
	 * The length of the GENEVE TLV option data excluding the option header
	 * in DW granularity.
	 */
	uint8_t option_len;
	/**
	 * Indicator about class field role in this option:
	 *  0 - class is ignored.
	 *  1 - class is fixed (the class defines the option along with the type).
	 *  2 - class matching per flow.
	 */
	uint8_t match_on_class_mode;
	/**
	 * The offset of the first sample in DW granularity.
	 * This offset is relative to first of option data.
	 * The 'match_data_mask' corresponds to option data since this offset.
	 */
	uint8_t offset;
	/**
	 * The number of DW to sample.
	 * This field describes the length of 'match_data_mask' in DW
	 * granularity.
	 */
	uint8_t sample_len;
	/**
	 * Array of DWs which each bit marks if this bit should be sampled.
	 * Each nonzero DW consumes one DW from maximum 7 DW in total.
	 */
	rte_be32_t *match_data_mask;
};

/**
 * Creates GENEVE TLV parser for the selected port.
 * This function must be called before first use of GENEVE option.
 *
 * This API is port oriented, but the configuration is done once for all ports
 * under the same physical device. Each port should call this API before using
 * GENEVE OPT item, but it must use the same options in the same order inside
 * the list.
 *
 * Each physical device has 7 DWs for GENEVE TLV options. Each nonzero element
 * in 'match_data_mask' array consumes one DW, and choosing matchable mode for
 * class consumes additional one.
 * Calling this API for second port under same physical device doesn't consume
 * more DW, it uses same configuration.
 *
 * @param[in] port_id
 *   The port identifier of the Ethernet device.
 * @param[in] tlv_list
 *   A list of GENEVE TLV options to create parser for them.
 * @param[in] nb_options
 *   The number of options in TLV list.
 *
 * @return
 *   A pointer to TLV handle on success, NULL otherwise and rte_errno is set.
 *   Possible values for rte_errno:
 *   - ENOMEM - not enough memory to create GENEVE TLV parser.
 *   - EEXIST - this port already has GENEVE TLV parser or another port under
 *              same physical device has already prepared a different parser.
 *   - EINVAL - invalid GENEVE TLV requested.
 *   - ENODEV - there is no Ethernet device for this port id.
 *   - ENOTSUP - the port doesn't support GENEVE TLV parsing.
 */
__rte_experimental
void *
rte_pmd_mlx5_create_geneve_tlv_parser(uint16_t port_id,
				      const struct mlx5_pmd_geneve_tlv tlv_list[],
				      uint8_t nb_options);

/**
 * Destroy GENEVE TLV parser for the selected port.
 * This function must be called after last use of GENEVE option and before port
 * closing.
 *
 * @param[in] handle
 *   Handle for the GENEVE TLV parser object to be destroyed.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_errno is set.
 *   Possible values for rte_errno:
 *   - EINVAL - invalid handle.
 *   - ENOENT - there is no valid GENEVE TLV parser in this handle.
 *   - EBUSY - one of options is in used by template table.
 */
__rte_experimental
int
rte_pmd_mlx5_destroy_geneve_tlv_parser(void *handle);

/* MLX5 flow engine mode definition for live migration. */
enum mlx5_flow_engine_mode {
	MLX5_FLOW_ENGINE_MODE_ACTIVE, /* active means high priority, effective in HW. */
	MLX5_FLOW_ENGINE_MODE_STANDBY, /* standby mode with lower priority flow rules. */
};

/**
 * When set on the flow engine of a standby process, ingress flow rules will be effective
 * in active and standby processes, so the ingress traffic may be duplicated.
 */
#define MLX5_FLOW_ENGINE_FLAG_STANDBY_DUP_INGRESS      RTE_BIT32(0)

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice
 *
 * Set the flow engine mode of the process to active or standby,
 * affecting network traffic handling.
 *
 * If one device does not support this operation or fails,
 * the whole operation is failed and rolled back.
 *
 * It is forbidden to have multiple flow engines with the same mode
 * unless only one of them is configured to handle the traffic.
 *
 * The application's flow engine is active by default.
 * The configuration from the active flow engine is effective immediately
 * while the configuration from the standby flow engine is queued by hardware.
 * When configuring the device from a standby flow engine,
 * it has no effect except for below situations:
 *   - traffic not handled by the active flow engine configuration
 *   - no active flow engine
 *
 * When flow engine of a process is changed from a standby to an active mode,
 * all preceding configurations that are queued by hardware
 * should become effective immediately.
 * Before mode transition, all the traffic handling configurations
 * set by the active flow engine should be flushed first.
 *
 * In summary, the operations are expected to happen in this order
 * in "old" and "new" applications:
 *   device: already configured by the old application
 *   new:    start as active
 *   new:    probe the same device
 *   new:    set as standby
 *   new:    configure the device
 *   device: has configurations from old and new applications
 *   old:    clear its device configuration
 *   device: has only 1 configuration from new application
 *   new:    set as active
 *   device: downtime for connecting all to the new application
 *   old:    shutdown
 *
 * @param mode
 *   The desired mode `mlx5_flow_engine_mode`.
 * @param flags
 *   Mode specific flags.
 * @return
 *   Positive value on success, -rte_errno value on error:
 *   - (> 0) Number of switched devices.
 *   - (-EINVAL) if error happen and rollback internally.
 *   - (-EPERM) if operation failed and can't recover.
 */
__rte_experimental
int rte_pmd_mlx5_flow_engine_set_mode(enum mlx5_flow_engine_mode mode, uint32_t flags);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice
 *
 * Get summary of MODIFY sub fields from the list of actions per table and
 * actions template index. This API can serve applications for fast rules
 * insertions using HWS APIs.
 *
 * @param tbl
 *   Pointer to table under query
 * @param act_template_idx
 *   The table's action template index
 * @param [out] num_modify
 *   Pointer to the number of MODIFY subfields found in the table's actions
 * @param [out] cmd_blob
 *   Pointer to the copied PRM struct related to the MODIFY action. The
 *   application should allocate the array of size: sizeof(uint64_t) *
 *   MLX5_MHDR_MAX_CMD.
 * @param [out] offset
 *   Pointer to array of MODIFY subfields offsests. The application should
 *   allocate the array of size: sizeof(uint8_t) * MLX5_MHDR_MAX_CMD.
 * @param [out] size
 *   Pointer to array of subfields sizes in double words units.
 *   For example: IPv6 has a size 4. The application should allocate the array
 *   of size: sizeof(uint8_t) * MLX5_MHDR_MAX_CMD.
 * @return
 *   0 on success
 */
__rte_experimental
int rte_pmd_mlx5_get_modify_hdr_info(struct rte_flow_template_table *tbl,
		uint8_t act_template_idx, uint8_t *num_modify,
		void *cmd_blob, uint8_t *offset, uint8_t *size);

#ifdef __cplusplus
}
#endif

#endif /* RTE_PMD_PRIVATE_MLX5_H_ */
