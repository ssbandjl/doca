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

/**
 * @file doca_cc.h
 * @page comm_channel_v2
 * @defgroup DOCA_CC Comm Channel
 *
 * DOCA Communication Channel library let you set a direct communication channel between the host and the DPU.
 * The channel is run over RoCE/IB protocol and is not part of the TCP/IP stack.
 * Please follow the programmer guide for usage instructions.
 *
 * @{
 */
#ifndef DOCA_CC_H_
#define DOCA_CC_H_

#include <stddef.h>
#include <stdint.h>

#include <doca_compat.h>
#include <doca_error.h>
#include <doca_types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct doca_dev;
struct doca_dev_rep;
struct doca_devinfo;

/*********************************************************************************************************************
 * DOCA CC Connection
 *********************************************************************************************************************/

/* Representantion of a comms channel point to point connection */
struct doca_cc_connection;

/**
 * Set the user data for a given connection.
 *
 * @param [in] connection
 * DOCA CC connection instance.
 * @param [in] user_data
 * User data for the given connection.
 *
 * @return
 * DOCA_SUCCESS on success.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_connection_set_user_data(struct doca_cc_connection *connection,
					      union doca_data user_data);

/**
 * Get the user data from a given connection.
 *
 * @param [in] connection
 * DOCA CC connection instance.
 *
 * @return
 * User data for the given connection.
 */
DOCA_EXPERIMENTAL
union doca_data doca_cc_connection_get_user_data(const struct doca_cc_connection *connection);

/**
 * Get the doca_cc_server context from a given connection.
 *
 * @param [in] connection
 * DOCA CC connection instance.
 *
 * @return
 * doca_cc_server object on success.
 * NULL if the connection is related to a client context.
 */
DOCA_EXPERIMENTAL
struct doca_cc_server *doca_cc_server_get_server_ctx(const struct doca_cc_connection *connection);

/**
 * Get the doca_cc_client context from a given connection.
 *
 * @param [in] connection
 * DOCA CC connection instance.
 *
 * @return
 * doca_cc_client object on success.
 * NULL if the connection is related to a server context.
 */
DOCA_EXPERIMENTAL
struct doca_cc_client *doca_cc_client_get_client_ctx(const struct doca_cc_connection *connection);

/*********************************************************************************************************************
 * DOCA CC General Capabilities
 *********************************************************************************************************************/

/**
 * Get the maximum name length that can be used in a cc instance.
 *
 * @param [in] devinfo
 * devinfo to query the capability for.
 * @param [out] max_name_len
 * The cc max name length, including the terminating null byte ('\0').
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_cap_get_max_name_len(const struct doca_devinfo *devinfo, uint32_t *max_name_len);

/**
 * Get the maximum message size that can be used on any comm channel instance.
 *
 * @param [in] devinfo
 * devinfo to query the capability for.
 * @param [out] size
 * The maximum size of a message available on any cc instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_cap_get_max_msg_size(const struct doca_devinfo *devinfo, uint32_t *size);

/**
 * Get the maximal recv queue size that can be used on any comm channel instance.
 *
 * @param [in] devinfo
 * devinfo to query the capability for.
 * @param [out] size
 * The maximal recv queue size supported on any cc instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_cap_get_max_recv_queue_size(const struct doca_devinfo *devinfo, uint32_t *size);

/**
 * Get the maximal number of clients that can be connected to a single doca_cc server.
 *
 * @param [in] devinfo
 * devinfo to query the capability for.
 * @param [out] num_clients
 * The number of clients that can be connected to a single doca_cc server.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_cap_get_max_clients(const struct doca_devinfo *devinfo, uint32_t *num_clients);

/*********************************************************************************************************************
 * DOCA CC server Context
 *********************************************************************************************************************/

/* Doca CC server end point instance */
struct doca_cc_server;

/**
 * Create a DOCA CC server instance.
 *
 * @param [in] dev
 * Device to use in DOCA CC server instance.
 * @param [in] repr
 * Representor device to use in CC server instance.
 * @param [in] name
 * Identifier for server associated with instance. Must be NULL terminated.
 * Max length, including terminating '\0', is obtained by doca_cc_cap_get_max_name_len().
 * @param [out] cc_server
 * Pointer to pointer to be set to created doca_cc server instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - one or more of the arguments is null.
 * - DOCA_ERROR_NO_MEMORY - failed to alloc doca_cc.
 * - DOCA_ERROR_INITIALIZATION - failed to initialise a mutex.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_create(struct doca_dev *dev, struct doca_dev_rep *repr, const char *name,
				    struct doca_cc_server **cc_server);

/**
 * Destroy a DOCA CC server instance.
 *
 * @param [in] cc_server
 * DOCA CC server instance to be destroyed.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - cc argument is a NULL pointer.
 * - DOCA_ERROR_IN_USE - Unable to gain exclusive access to the cc instance.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_destroy(struct doca_cc_server *cc_server);

/**
 * Check if given device is capable of running a comm channel server.
 *
 * @param [in] devinfo
 * The DOCA device information.
 *
 * @return
 * DOCA_SUCCESS - in case device can run as a server.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - provided devinfo can not implement a cc server.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_cap_server_is_supported(const struct doca_devinfo *devinfo);

/**
 * Set the maximum message size property for the doca_cc instance.
 * If not called, a default value will be used and can be queried using doca_cc_server_get_max_msg_size().
 *
 * @param [in] cc_server
 * DOCA CC server instance.
 * @param [in] size
 * The maximum size of a message to set for the instance. Can be queried with doca_cc_cap_get_max_msg_size().
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_set_max_msg_size(struct doca_cc_server *cc_server, uint32_t size);

/**
 * Get the maximum message size that can be sent on the comm channel instance.
 *
 * @param [in] cc_server
 * DOCA CC server instance.
 * @param [out] size
 * The maximum size of a message for the instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_get_max_msg_size(const struct doca_cc_server *cc_server, uint32_t *size);

/**
 * Set the recv queue size property for the doca_cc instance.
 * If not called, a default value will be used and can be queried using doca_cc_server_get_recv_queue_size().
 *
 * @param [in] cc_server
 * DOCA CC server instance.
 * @param [in] size
 * The recv queue size set for the instance. Can be queried with doca_cc_cap_get_max_recv_queue_size().
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_set_recv_queue_size(struct doca_cc_server *cc_server, uint32_t size);

/**
 * Get the recv queue size property set on the doca_cc instance.
 *
 * @param [in] cc_server
 * DOCA CC server instance.
 * @param [out] size
 * The recv queue size set for the instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_get_recv_queue_size(const struct doca_cc_server *cc_server, uint32_t *size);

/**
 * Get the doca device property of the associated doca_cc instance.
 *
 * @param [in] cc_server
 * DOCA CC server instance.
 * @param [out] dev
 * Current device used in the doca_cc instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_get_device(const struct doca_cc_server *cc_server, struct doca_dev **dev);

/**
 * Get the device representor property of the associated doca_cc server instance.
 *
 * @param [in] cc_server
 * DOCA CC server instance.
 * @param [out] repr
 * Current device representor used in the doca_cc server instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_get_device_repr(const struct doca_cc_server *cc_server, struct doca_dev_rep **repr);

/**
 * Convert doca_cc_server instance into a generalised context for use with doca core objects.
 *
 * @param [in] cc_server
 * DOCA CC server instance. This must remain valid until after the context is no longer required.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_ctx *doca_cc_server_as_ctx(struct doca_cc_server *cc_server);

/*********************************************************************************************************************
 * DOCA CC Client Context
 *********************************************************************************************************************/

/* Doca CC Client end point instance */
struct doca_cc_client;

/**
 * Create a DOCA CC client instance.
 *
 * @param [in] dev
 * Device to use in DOCA CC client instance.
 * @param [in] name
 * Identifier for the server the client will connect to.
 * Max length, including terminating '\0', is obtained by doca_cc_cap_get_max_name_len().
 * @param [out] cc_client
 * Pointer to pointer to be set to created doca_cc client instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - one or more of the arguments is null.
 * - DOCA_ERROR_NO_MEMORY - failed to alloc doca_cc.
 * - DOCA_ERROR_INITIALIZATION - failed to initialise a mutex.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_create(struct doca_dev *dev, const char *name, struct doca_cc_client **cc_client);

/**
 * Destroy a DOCA CC client instance.
 *
 * @param [in] cc_client
 * DOCA CC client instance to be destroyed.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - cc argument is a NULL pointer.
 * - DOCA_ERROR_IN_USE - Unable to gain exclusive access to the cc instance.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_destroy(struct doca_cc_client *cc_client);

/**
 * Check if given device is capable of running a comm channel client.
 *
 * @param [in] devinfo
 * The DOCA device information.
 *
 * @return
 * DOCA_SUCCESS - in case device can run as a client.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - provided devinfo can not implement a cc client
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_cap_client_is_supported(const struct doca_devinfo *devinfo);

/**
 * Set the maximum message size property for the doca_cc instance.
 * If not called, a default value will be used and can be queried using doca_cc_client_get_max_msg_size().
 *
 * @param [in] cc_client
 * DOCA CC client instance.
 * @param [in] size
 * The maximum size of a message to set for the instance. Can be queried with doca_cc_cap_get_max_msg_size().
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_set_max_msg_size(struct doca_cc_client *cc_client, uint32_t size);

/**
 * Get the maximum message size that can be sent on the comm channel instance.
 *
 * @param [in] cc_client
 * DOCA CC client instance.
 * @param [out] size
 * The maximum size of a message for the instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_get_max_msg_size(const struct doca_cc_client *cc_client, uint32_t *size);

/**
 * Set the recv queue size property for the doca_cc instance.
 * If not called, a default value will be used and can be queried using doca_cc_client_get_recv_queue_size().
 *
 * @param [in] cc_client
 * DOCA CC client instance.
 * @param [in] size
 * The recv queue size to set for the instance. Limit can be queried with doca_cc_cap_get_max_recv_queue_size().
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_set_recv_queue_size(struct doca_cc_client *cc_client, uint32_t size);

/**
 * Get the recv queue size property set on the doca_cc instance.
 *
 * @param [in] cc_client
 * DOCA CC client instance.
 * @param [out] size
 * The recv queue size for the instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_get_recv_queue_size(const struct doca_cc_client *cc_client, uint32_t *size);

/**
 * Get the doca device property of the associated doca_cc instance.
 *
 * @param [in] cc_client
 * DOCA CC client instance.
 * @param [out] dev
 * Current device used in the doca_cc instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_get_device(const struct doca_cc_client *cc_client, struct doca_dev **dev);

/**
 * Convert doca_cc instance into a generalised context for use with doca core objects.
 *
 * @param [in] cc_client
 * DOCA CC client instance. This must remain valid until after the context is no longer required.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_ctx *doca_cc_client_as_ctx(struct doca_cc_client *cc_client);

/**
 * Get the connection object associated with the client ctx. Can only be called after starting the ctx.
 *
 * @param [in] cc_client
 * DOCA CC client instance.
 * @param [out] connection
 * The connection object associated with the client.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is not started.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_get_connection(const struct doca_cc_client *cc_client,
					   struct doca_cc_connection **connection);

/*********************************************************************************************************************
 * DOCA CC - Send Task
 *********************************************************************************************************************/

/* Task instance to send a message on the control channel */
struct doca_cc_task_send;

/**
 * Function executed on doca_cc send task completion. Used for both task success and failure.
 *
 * @param [in] task
 * Doca cc send task that has completed.
 * @param [in] task_user_data
 * The task user data.
 * @param [in] ctx_user_data
 * Doca_cc context user data.
 *
 * The implementation can assume this value is not NULL.
 */
typedef void (*doca_cc_task_send_completion_cb_t)(struct doca_cc_task_send *task, union doca_data task_user_data,
						  union doca_data ctx_user_data);

/**
 * Get the maximal send tasks num that can be used on any cc instance.
 *
 * @param [in] devinfo
 * devinfo to query the capability for.
 * @param [out] max_send_tasks
 * The maximal supported number of send tasks for any cc instance.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_cap_get_max_send_tasks(const struct doca_devinfo *devinfo, uint32_t *max_send_tasks);

/**
 * Configure the doca_cc_server send task callback and parameters.
 *
 * @param [in] cc_server
 * The doca_cc_server instance.
 * @param [in] task_completion_cb
 * Send task completion callback.
 * @param [in] task_error_cb
 * Send task error callback.
 * @param [in] num_send_tasks
 * Number of send tasks to create.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_task_send_set_conf(struct doca_cc_server *cc_server,
					       doca_cc_task_send_completion_cb_t task_completion_cb,
					       doca_cc_task_send_completion_cb_t task_error_cb,
					       uint32_t num_send_tasks);

/**
 * Configure the doca_cc_client send task callback and parameters.
 *
 * @param [in] cc_client
 * The doca_cc_client instance.
 * @param [in] task_completion_cb
 * Send task completion callback.
 * @param [in] task_error_cb
 * Send task error callback.
 * @param [in] num_send_tasks
 * Number of send tasks to create.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_BAD_STATE - cc instance is already active.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_task_send_set_conf(struct doca_cc_client *cc_client,
					       doca_cc_task_send_completion_cb_t task_completion_cb,
					       doca_cc_task_send_completion_cb_t task_error_cb,
					       uint32_t num_send_tasks);

/**
 * Allocate and initialise a doca_cc_server send task.
 *
 * @param [in] cc_server
 * The doca_cc_server instance.
 * @param [in] peer
 * Connected endpoint to send the message to.
 * @param [in] msg
 * Message or data to sent to associated peer.
 * @param [in] len
 * Length of the message to send.
 * @param [out] task
 * Pointer to a doca_cc_send_task instance populated with input parameters.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_NO_MEMORY - no available tasks to allocate.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_task_send_alloc_init(struct doca_cc_server *cc_server, struct doca_cc_connection *peer,
						  const void *msg, uint32_t len, struct doca_cc_task_send **task);

/**
 * Allocate and initialise a doca_cc_client send task.
 *
 * @param [in] cc_client
 * The doca_cc_client instance.
 * @param [in] peer
 * Connected endpoint to send the message to.
 * @param [in] msg
 * Message or data to sent to associated peer.
 * @param [in] len
 * Length of the message to send.
 * @param [out] task
 * Pointer to a doca_cc_send_task instance populated with input parameters.
 *
 * @return
 * DOCA_SUCCESS on success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case of invalid input.
 * - DOCA_ERROR_NO_MEMORY - no available tasks to allocate.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_task_send_alloc_init(struct doca_cc_client *cc_client, struct doca_cc_connection *peer,
						 const void *msg, uint32_t len, struct doca_cc_task_send **task);

/**
 * Convert a doca_cc_send_task task to doca_task.
 *
 * @param [in] task
 * Doca_cc_send_task task to convert.
 *
 * @return
 * Non NULL upon success, NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_cc_task_send_as_task(struct doca_cc_task_send *task);

/*********************************************************************************************************************
 * DOCA CC - Receive Message Event Registration
 *********************************************************************************************************************/

/* Async event instance for receiving a message from a connected endpoint */
struct doca_cc_event_msg_recv;

/**
 * Function executed on a doca_cc receive message event.
 *
 * @param [in] event
 * Doca_cc recv message event that has triggered.
 * @param [in] recv_buffer
 * Pointer to the message data associated with the event.
 * @param [in] msg_len
 * Length of the message data associated with the event.
 * @param [in] cc_connection
 * Pointer to the connection instance that generated the message event.
 *
 * The implementation can assume these values are not NULL.
 */
typedef void (*doca_cc_event_msg_recv_cb_t)(struct doca_cc_event_msg_recv *event, uint8_t *recv_buffer,
					    uint32_t msg_len, struct doca_cc_connection *cc_connection);

/**
 * @brief Configure the doca_cc recv event callback for server context.
 *
 * @param [in] cc_server
 * Pointer to doca_cc_server instance.
 * @param [in] recv_event_cb
 * Recv event callback.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_BAD_STATE - doca_cc context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_event_msg_recv_register(struct doca_cc_server *cc_server,
						    doca_cc_event_msg_recv_cb_t recv_event_cb);

/**
 * @brief Configure the doca_cc recv event callback for client context.
 *
 * @param [in] cc_client
 * Pointer to doca_cc_client instance.
 * @param [in] recv_event_cb
 * Recv event callback.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_BAD_STATE - doca_cc context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_event_msg_recv_register(struct doca_cc_client *cc_client,
						    doca_cc_event_msg_recv_cb_t recv_event_cb);

/*********************************************************************************************************************
 * DOCA CC - Connection Event Registration
 *********************************************************************************************************************/

/* Async event instance for a connection status change */
struct doca_cc_event_connection_status_changed;

/**
 * Function executed on a doca_cc connection event.
 *
 * @param [in] event
 * Doca_cc connection event that has triggered.
 * @param [in] cc_connection
 * Pointer to the peer which triggered the connection event.
 * @param [in] change_successful
 * 1 if the action (connect/disconnect) was successful, 0 otherwise.
 *
 * The implementation can assume these values are not NULL.
 */
typedef void (*doca_cc_event_connection_status_changed_cb_t)(struct doca_cc_event_connection_status_changed *event,
							     struct doca_cc_connection *cc_connection,
							     uint8_t change_successful);

/**
 * @brief Configure the doca_cc recv event callback for server context.
 *
 * @param [in] cc_server
 * Pointer to doca_cc_server instance.
 * @param [in] connect_event_cb
 * Callback for connect event.
 * @param [in] disconnect_event_cb
 * Callback for disconnect event.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_BAD_STATE - doca_cc context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_event_connection_register(struct doca_cc_server *cc_server,
						       doca_cc_event_connection_status_changed_cb_t connect_event_cb,
						       doca_cc_event_connection_status_changed_cb_t disconnect_event_cb);

/*********************************************************************************************************************
 * DOCA CC - Connection Statistics
 *********************************************************************************************************************/

/** Available counters for connection statistics query*/
enum doca_cc_counter {
	DOCA_CC_COUNTER_SENT_MESSAGES = 1, /* total number of messages sent from local cc over a given cc_connection. */
	DOCA_CC_COUNTER_SENT_BYTES = 2,	   /* total number of bytes sent from local cc over a given cc_connection. */
	DOCA_CC_COUNTER_RECV_MESSAGES = 3,     /* total number of messages received on local cc over a given
					    * cc_connection.
					    */
	DOCA_CC_COUNTER_RECV_BYTES = 4,	   /* total number of bytes received on local cc over a given cc_connection. */
};

/**
 * @brief update statistics for given cc_connection
 *
 * Should be used before calling to any connection information function to update the saved statistics.
 *
 * @param [in] cc_connection
 * Pointer to cc_connection to update statistics in.
 *
 * @return
 * DOCA_SUCCESS on success.
 * DOCA_ERROR_INVALID_VALUE if cc_connection is NULL.
 * DOCA_ERROR_CONNECTION_INPROGRESS if connection is not yet established.
 * DOCA_ERROR_CONNECTION_ABORTED if the connection failed.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_connection_update_info(struct doca_cc_connection *cc_connection);

/**
 * @brief get statistics counter for a given cc_connection
 *
 * This function will return statistics for a given cc_connection, updated to the last time
 * doca_cc_connection_update_info() was called.
 *
 * @param [in] cc_connection
 * Pointer to cc_connection to query statistics for.
 * @param [in] counter_type
 * Which statistics counter should be queried.
 * @param [out] counter_value
 * Will contain the value for the counter on the given cc_connection.
 *
 * @return
 * DOCA_SUCCESS on success.
 * DOCA_ERROR_INVALID_VALUE if any of the arguments are NULL or if the counter is not valid.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_connection_get_counter(const struct doca_cc_connection *cc_connection,
					    enum doca_cc_counter counter_type, uint64_t *counter_value);

/*********************************************************************************************************************
 * DOCA CC - Consumer Event Registration
 *********************************************************************************************************************/

/* Async event instance for a consumer change*/
struct doca_cc_event_consumer;

/**
 * Function executed on a doca_cc consumer event.
 *
 * @param [in] event
 * Doca_cc consumer event that has triggered.
 * @param [in] cc_connection
 * Pointer to the cc_connection which triggered that has generated the consumer event.
 * @param [in] id
 * The ID of the newly created or destroyed consumer.
 *
 * The implementation can assume these values are not NULL.
 */
typedef void (*doca_cc_event_consumer_cb_t)(struct doca_cc_event_consumer *event,
					    struct doca_cc_connection *cc_connection, uint32_t id);

/**
 * @brief Configure the doca_cc callback for for receiving consumer events on server context.
 *
 * @param [in] cc_server
 * Pointer to doca_cc_server instance.
 * @param [in] new_consumer_event_cb
 * Consumer event callback on creation of a new consumer.
 * @param [in] expired_consumer_event_cb
 * Consumer event callback on when a consumer has expired.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_BAD_STATE - doca_cc_servcie context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_server_event_consumer_register(struct doca_cc_server *cc_server,
						     doca_cc_event_consumer_cb_t new_consumer_event_cb,
						     doca_cc_event_consumer_cb_t expired_consumer_event_cb);

/**
 * @brief Configure the doca_cc callback for for receiving consumer events on client context.
 *
 * @param [in] cc_client
 * Pointer to doca_cc_client instance.
 * @param [in] new_consumer_event_cb
 * Consumer event callback on creation of a new consumer.
 * @param [in] expired_consumer_event_cb
 * Consumer event callback on when a consumer has expired.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_BAD_STATE - doca_cc_servcie context state is not idle.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_cc_client_event_consumer_register(struct doca_cc_client *cc_client,
						    doca_cc_event_consumer_cb_t new_consumer_event_cb,
						    doca_cc_event_consumer_cb_t expired_consumer_event_cb);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* DOCA_CC_H_ */
