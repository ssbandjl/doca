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

#include <doca_log.h>

#include <dpdk_utils.h>

#include "flow_ct_common.h"

#define DPDK_ADDITIONAL_ARG 2

DOCA_LOG_REGISTER(FLOW_CT_COMMON);

/*
 * ARGP Callback - Handle DOCA Flow CT device PCI address parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
pci_addr_callback(void *param, void *config)
{
	struct ct_config *cfg = (struct ct_config *)config;
	const char *dev_pci_addr = (char *)param;
	int len;

	len = strnlen(dev_pci_addr, DOCA_DEVINFO_PCI_ADDR_SIZE);
	/* Check using >= to make static code analysis satisfied */
	if (len >= DOCA_DEVINFO_PCI_ADDR_SIZE) {
		DOCA_LOG_ERR("Entered device PCI address exceeding the maximum size of %d", DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(cfg->ct_dev_pci_addr, dev_pci_addr, len + 1);

	return DOCA_SUCCESS;
}

doca_error_t
flow_ct_register_params(void)
{
	doca_error_t result;

	struct doca_argp_param *dev_pci_addr_param;

	/* Create and register DOCA Flow CT device PCI address */
	result = doca_argp_param_create(&dev_pci_addr_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(dev_pci_addr_param, "p");
	doca_argp_param_set_long_name(dev_pci_addr_param, "pci-addr");
	doca_argp_param_set_description(dev_pci_addr_param, "DOCA Flow CT device PCI address");
	doca_argp_param_set_callback(dev_pci_addr_param, pci_addr_callback);
	doca_argp_param_set_type(dev_pci_addr_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(dev_pci_addr_param);
	result = doca_argp_register_param(dev_pci_addr_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t
init_doca_flow_ct(struct doca_dev *ct_dev, uint32_t flags, uint32_t nb_arm_queues, uint32_t nb_ctrl_queues,
		  uint32_t nb_user_actions, doca_flow_ct_flow_log_cb flow_log_cb, uint32_t nb_ipv4_sessions,
		  uint32_t nb_ipv6_sessions, bool o_match_inner, struct doca_flow_meta *o_zone_mask,
		  struct doca_flow_meta *o_modify_mask, bool r_match_inner, struct doca_flow_meta *r_zone_mask,
		  struct doca_flow_meta *r_modify_mask)
{
	struct doca_flow_ct_cfg ct_cfg;
	doca_error_t result;

	if (o_zone_mask == NULL || o_modify_mask == NULL) {
		DOCA_LOG_ERR("Origin masks can't be null");
		return DOCA_ERROR_INVALID_VALUE;
	} else if (r_zone_mask == NULL || r_modify_mask == NULL) {
		DOCA_LOG_ERR("Reply masks can't be null");
		return DOCA_ERROR_INVALID_VALUE;
	}

	memset(&ct_cfg, 0, sizeof(ct_cfg));

	ct_cfg.flags |= DOCA_FLOW_CT_FLAG_MANAGED;

	ct_cfg.doca_dev = ct_dev;
	ct_cfg.nb_arm_queues = nb_arm_queues;
	ct_cfg.nb_ctrl_queues = nb_ctrl_queues;
	ct_cfg.nb_user_actions = nb_user_actions;
	ct_cfg.aging_core = nb_arm_queues + 1;
	ct_cfg.flow_log_cb = flow_log_cb;
	ct_cfg.nb_arm_sessions[DOCA_FLOW_CT_SESSION_IPV4] = nb_ipv4_sessions;
	ct_cfg.nb_arm_sessions[DOCA_FLOW_CT_SESSION_IPV6] = nb_ipv6_sessions;

	ct_cfg.direction[0].match_inner = o_match_inner;
	ct_cfg.direction[0].zone_match_mask = o_zone_mask;
	ct_cfg.direction[0].meta_modify_mask = o_modify_mask;
	ct_cfg.direction[1].match_inner = r_match_inner;
	ct_cfg.direction[1].zone_match_mask = r_zone_mask;
	ct_cfg.direction[1].meta_modify_mask = r_modify_mask;

	ct_cfg.flags |= flags;

	result = doca_flow_ct_init(&ct_cfg);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to initialize DOCA Flow CT: %s", doca_error_get_name(result));

	return result;
}

doca_error_t
flow_ct_dpdk_init(int argc, char **dpdk_argv)
{
	char *argv[argc + DPDK_ADDITIONAL_ARG];

	memcpy(argv, dpdk_argv, sizeof(argv[0]) * argc);
	argv[argc++] = "-a";
	argv[argc++] = "pci:00:00.0";

	return dpdk_init(argc, argv);
}

doca_error_t
flow_ct_capable(struct doca_devinfo *dev_info)
{
	return doca_flow_ct_cap_is_dev_supported(dev_info);
}
