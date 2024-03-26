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

#include <string.h>
#include <time.h>

#include <doca_argp.h>
#include <doca_cc.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_pe.h>

#include "cc_ctrl_path_common.h"
#include "common.h"

DOCA_LOG_REGISTER(CC_CTRL_PATH_COMMON);

#define CC_REC_QUEUE_SIZE 10 /* Maximum amount of message in queue */
#define CC_SEND_TASK_NUM 1024 /* Number of CC send tasks  */

/**
 * Argument parsing section
 */

/*
 * ARGP Callback - Handle Comm Channel DOCA device PCI address parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
pci_addr_callback(void *param, void *config)
{
	struct cc_config *cfg = (struct cc_config *)config;
	const char *dev_pci_addr = (char *)param;
	int len;

	len = strnlen(dev_pci_addr, DOCA_DEVINFO_PCI_ADDR_SIZE);
	/* Check using >= to make static code analysis satisfied */
	if (len >= DOCA_DEVINFO_PCI_ADDR_SIZE) {
		DOCA_LOG_ERR("Entered device PCI address exceeding the maximum size of %d",
			     DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(cfg->cc_dev_pci_addr, dev_pci_addr, len + 1);

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle Comm Channel DOCA device representor PCI address parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
rep_pci_addr_callback(void *param, void *config)
{
	struct cc_config *cfg = (struct cc_config *)config;
	const char *rep_pci_addr = (char *)param;
	int len;

	len = strnlen(rep_pci_addr, DOCA_DEVINFO_PCI_ADDR_SIZE);
	/* Check using >= to make static code analysis satisfied */
	if (len >= DOCA_DEVINFO_PCI_ADDR_SIZE) {
		DOCA_LOG_ERR("Entered device representor PCI address exceeding the maximum size of %d",
			     DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(cfg->cc_dev_rep_pci_addr, rep_pci_addr, len + 1);

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle text to copy parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
text_callback(void *param, void *config)
{
	struct cc_config *conf = (struct cc_config *)config;
	const char *txt = (char *)param;
	int txt_len = strnlen(txt, MAX_TXT_SIZE);

	/* Check using >= to make static code analysis satisfied */
	if (txt_len >= MAX_TXT_SIZE) {
		DOCA_LOG_ERR("Entered text exceeded buffer size of: %d", MAX_USER_TXT_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(conf->text, txt, txt_len + 1);

	return DOCA_SUCCESS;
}

doca_error_t
register_cc_params(void)
{
	doca_error_t result;

	struct doca_argp_param *dev_pci_addr_param, *text_param, *rep_pci_addr_param;

	/* Create and register Comm Channel DOCA device PCI address */
	result = doca_argp_param_create(&dev_pci_addr_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(dev_pci_addr_param, "p");
	doca_argp_param_set_long_name(dev_pci_addr_param, "pci-addr");
	doca_argp_param_set_description(dev_pci_addr_param, "DOCA Comm Channel device PCI address");
	doca_argp_param_set_callback(dev_pci_addr_param, pci_addr_callback);
	doca_argp_param_set_type(dev_pci_addr_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(dev_pci_addr_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	/* Create and register Comm Channel DOCA device representor PCI address */
	result = doca_argp_param_create(&rep_pci_addr_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(rep_pci_addr_param, "r");
	doca_argp_param_set_long_name(rep_pci_addr_param, "rep-pci");
	doca_argp_param_set_description(rep_pci_addr_param,
					"DOCA Comm Channel device representor PCI address (needed only on DPU)");
	doca_argp_param_set_callback(rep_pci_addr_param, rep_pci_addr_callback);
	doca_argp_param_set_type(rep_pci_addr_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(rep_pci_addr_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	/* Create and register text to send param */
	result = doca_argp_param_create(&text_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(text_param, "t");
	doca_argp_param_set_long_name(text_param, "text");
	doca_argp_param_set_description(text_param, "Text to be sent to the other side of channel");
	doca_argp_param_set_callback(text_param, text_callback);
	doca_argp_param_set_type(text_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(text_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register program param: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

void
clean_cc_ctrl_path_client(struct doca_cc_client *client, struct doca_pe *pe)
{
	doca_error_t result;

	if (client != NULL) {
		result = doca_cc_client_destroy(client);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("Failed to destroy client properly with error = %s", doca_error_get_name(result));
	}

	if (pe != NULL) {
		result = doca_pe_destroy(pe);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("Failed to destroy pe properly with error = %s", doca_error_get_name(result));
	}
}

doca_error_t
init_cc_ctrl_path_client(const char *server_name, struct doca_dev *hw_dev, struct cc_ctrl_path_client_cb_config *cb_cfg,
			 struct doca_cc_client **client, struct doca_pe **pe)
{
	doca_error_t result;
	struct doca_ctx *ctx;
	union doca_data user_data;

	result = doca_pe_create(pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed creating pe with error = %s", doca_error_get_name(result));
		return result;
	}

	result = doca_cc_client_create(hw_dev, server_name, client);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create client with error = %s", doca_error_get_name(result));
		goto destroy_pe;
	}

	ctx = doca_cc_client_as_ctx(*client);

	result = doca_pe_connect_ctx(*pe, ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed adding pe context to client with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	result = doca_ctx_set_state_changed_cb(ctx, cb_cfg->ctx_state_changed_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed setting state change callback with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	result = doca_cc_client_task_send_set_conf(*client, cb_cfg->send_task_comp_cb, cb_cfg->send_task_comp_err_cb,
						   CC_SEND_TASK_NUM);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed setting send task cbs with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	result = doca_cc_client_event_msg_recv_register(*client, cb_cfg->msg_recv_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed adding message recv event cb with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	/* Config the data path related events */
	if (cb_cfg->data_path_mode == true) {
		result = doca_cc_client_event_consumer_register(*client, cb_cfg->new_consumer_cb,
								cb_cfg->expired_consumer_cb);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed adding consumer event cb with error = %s", doca_error_get_name(result));
			goto destroy_client;
		}
	}

	/* Set client properties */
	result = doca_cc_client_set_max_msg_size(*client, CC_MAX_MSG_SIZE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set msg size property with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	result = doca_cc_client_set_recv_queue_size(*client, CC_REC_QUEUE_SIZE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set msg size property with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	user_data.ptr = cb_cfg->ctx_user_data;
	result = doca_ctx_set_user_data(ctx, user_data);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set ctx user data with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	/* Client is not started until connection is finished, so getting connection in progress */
	result = doca_ctx_start(ctx);
	if (result != DOCA_ERROR_IN_PROGRESS) {
		DOCA_LOG_ERR("Failed to start client context with error = %s", doca_error_get_name(result));
		goto destroy_client;
	}

	return DOCA_SUCCESS;

destroy_client:
	doca_cc_client_destroy(*client);
	*client = NULL;
destroy_pe:
	doca_pe_destroy(*pe);
	*pe = NULL;
	return result;
}

void
clean_cc_ctrl_path_server(struct doca_cc_server *server, struct doca_pe *pe)
{
	doca_error_t result;

	if (server != NULL) {
		result = doca_cc_server_destroy(server);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("Failed to destroy server properly with error = %s", doca_error_get_name(result));
	}

	if (pe != NULL) {
		result = doca_pe_destroy(pe);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("Failed to destroy pe properly with error = %s", doca_error_get_name(result));
	}
}

doca_error_t
init_cc_ctrl_path_server(const char *server_name, struct doca_dev *hw_dev, struct doca_dev_rep *rep_dev,
			 struct cc_ctrl_path_server_cb_config *cb_cfg, struct doca_cc_server **server,
			 struct doca_pe **pe)
{
	doca_error_t result;
	union doca_data user_data;
	struct doca_ctx *ctx;

	result = doca_pe_create(pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed creating pe with error = %s", doca_error_get_name(result));
		return result;
	}

	result = doca_cc_server_create(hw_dev, rep_dev, server_name, server);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create server with error = %s", doca_error_get_name(result));
		goto destroy_pe;
	}

	ctx = doca_cc_server_as_ctx(*server);

	result = doca_pe_connect_ctx(*pe, ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed adding pe context to server with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	result = doca_ctx_set_state_changed_cb(ctx, cb_cfg->ctx_state_changed_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed setting state change callback with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	result = doca_cc_server_task_send_set_conf(*server, cb_cfg->send_task_comp_cb, cb_cfg->send_task_comp_err_cb,
						   CC_SEND_TASK_NUM);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed setting send task cbs with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	result = doca_cc_server_event_msg_recv_register(*server, cb_cfg->msg_recv_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed adding message recv event cb with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	result = doca_cc_server_event_connection_register(*server, cb_cfg->server_connection_event_cb,
							  cb_cfg->server_disconnection_event_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed adding connection event cbs with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	/* Config the data_path related events */
	if (cb_cfg->data_path_mode == true) {
		result = doca_cc_server_event_consumer_register(*server, cb_cfg->new_consumer_cb,
								cb_cfg->expired_consumer_cb);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed adding consumer event cb with error = %s", doca_error_get_name(result));
			goto destroy_server;
		}
	}

	/* Set server properties */
	result = doca_cc_server_set_max_msg_size(*server, CC_MAX_MSG_SIZE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set msg size property with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	result = doca_cc_server_set_recv_queue_size(*server, CC_REC_QUEUE_SIZE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set msg size property with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	user_data.ptr = cb_cfg->ctx_user_data;
	result = doca_ctx_set_user_data(ctx, user_data);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set ctx user data with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	result = doca_ctx_start(ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start server context with error = %s", doca_error_get_name(result));
		goto destroy_server;
	}

	return DOCA_SUCCESS;

destroy_server:
	doca_cc_server_destroy(*server);
	*server = NULL;
destroy_pe:
	doca_pe_destroy(*pe);
	*pe = NULL;
	return result;
}
