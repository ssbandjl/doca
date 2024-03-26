/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#ifndef FLOW_CT_COMMON_H_
#define FLOW_CT_COMMON_H_

#include <doca_dev.h>
#include <doca_argp.h>
#include <doca_flow.h>
#include <doca_flow_ct.h>

struct ct_config {
	char ct_dev_pci_addr[DOCA_DEVINFO_PCI_ADDR_SIZE];	  /* Flow CT DOCA device PCI address */
};

/*
 * Register the command line parameters for the DOCA Flow CT samples
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t flow_ct_register_params(void);

/*
 * Initialize DOCA Flow CT library
 *
 * @ct_dev [in]: Flow CT device
 * @flags [in]: Flow CT flags
 * @nb_arm_queues [in]: Number of threads the sample will use
 * @nb_ctrl_queues [in]: Number of control queues
 * @nb_user_actions [in]: Number of CT user actions
 * @flow_log_cb [in]: Flow log callback
 * @nb_ipv4_sessions [in]: Number of IPv4 sessions
 * @nb_ipv6_sessions [in]: Number of IPv6 sessions
 * @o_match_inner [in]: Origin match inner
 * @o_zone_mask [in]: Origin zone mask
 * @o_modify_mask [in]: Origin modify mask
 * @r_match_inner [in]: Reply match inner
 * @r_zone_mask [in]: Reply zone mask
 * @r_modify_mask [in]: Reply modify mask
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t
init_doca_flow_ct(struct doca_dev *ct_dev, uint32_t flags, uint32_t nb_arm_queues, uint32_t nb_ctrl_queues,
		  uint32_t nb_user_actions, doca_flow_ct_flow_log_cb flow_log_cb, uint32_t nb_ipv4_sessions,
		  uint32_t nb_ipv6_sessions, bool o_match_inner, struct doca_flow_meta *o_zone_mask,
		  struct doca_flow_meta *o_modify_mask, bool r_match_inner, struct doca_flow_meta *r_zone_mask,
		  struct doca_flow_meta *r_modify_mask);

/*
 * Initialize DPDK environment for DOCA Flow CT
 *
 * @argc [in]: Number of program command line arguments
 * @dpdk_argv [in]: DPDK command line arguments create by argp library
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t flow_ct_dpdk_init(int argc, char **dpdk_argv);

/*
 * Verify if DOCA device is ECPF by checking all supported capabilities
 *
 * @dev_info [in]: DOCA device info
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t flow_ct_capable(struct doca_devinfo *dev_info);

#endif /* FLOW_CT_COMMON_H_ */
