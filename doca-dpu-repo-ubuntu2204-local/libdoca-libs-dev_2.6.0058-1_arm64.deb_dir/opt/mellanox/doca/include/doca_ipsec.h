/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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
 * Definition of an abstract implementation of IPSec protocol offload.
 */

/**
 * @file doca_ipsec.h
 * @page ipsec
 * @defgroup DOCA_IPSEC IPsec
 * DOCA IPSEC library. For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */

#ifndef DOCA_IPSEC_H_
#define DOCA_IPSEC_H_

#include <doca_buf.h>
#include <doca_compat.h>
#include <doca_ctx.h>
#include <doca_error.h>
#include <doca_pe.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief IPSec encryption key type */
enum doca_encryption_key_type {
	DOCA_ENCRYPTION_KEY_AESGCM_128, /**< size of 128 bit */
	DOCA_ENCRYPTION_KEY_AESGCM_256, /**< size of 256 bit */
};

/** @brief IPSec encryption key */
struct doca_encryption_key {
	enum doca_encryption_key_type type; /**< size of enc key */
	union {
		struct {
			uint64_t implicit_iv; /**< The IV is inserted into the GCM engine is calculated by */
			uint32_t salt;	      /**< The salt is inserted into the GCM engine is calculated by */
			void *raw_key;	      /**< Raw key buffer. Actual size of this buffer defined by type. */
		} aes_gcm;
	};
};

/** @brief IPSec replay window size */
enum doca_ipsec_replay_win_size {
	DOCA_IPSEC_REPLAY_WIN_SIZE_32 = 32,   /**< size of 32 bit */
	DOCA_IPSEC_REPLAY_WIN_SIZE_64 = 64,   /**< size of 64 bit */
	DOCA_IPSEC_REPLAY_WIN_SIZE_128 = 128, /**< size of 128 bit */
	DOCA_IPSEC_REPLAY_WIN_SIZE_256 = 256, /**< size of 256 bit */
};

/** @brief IPSec icv length */
enum doca_ipsec_icv_length {
	DOCA_IPSEC_ICV_LENGTH_8 = 8,   /**< size of 8 bit */
	DOCA_IPSEC_ICV_LENGTH_12 = 12, /**< size of 12 bit */
	DOCA_IPSEC_ICV_LENGTH_16 = 16, /**< size of 16 bit */
};

/** @brief IPSec direction of the key, incoming packets or outgoing */
enum doca_ipsec_direction {
	DOCA_IPSEC_DIRECTION_INGRESS_DECRYPT = 0, /**< incoming packets, decription */
	DOCA_IPSEC_DIRECTION_EGRESS_ENCRYPT = 1	  /**< outgoing packets, encription */
};

/** @brief IPSec offload mode */
enum doca_ipsec_sa_offload {
	DOCA_IPSEC_SA_OFFLOAD_CRYPTO = 0,	/**< IPSec ipsec offload */
	DOCA_IPSEC_SA_OFFLOAD_FULL = 1, /**< IPSec full offload - to enable SN and AR offload */
};

/** @brief IPSec sa events attributes - when turned on will trigger an event */
struct doca_ipsec_sa_event_attrs {
	uint32_t remove_flow_packet_count;
	/**< Packet counter, Decrements for every packet passing through the SA.
	 * Event are triggered occurs when the counter reaches soft- lifetime and hard-lifetime (0).
	 * When counter reaches hard-lifetime, all passing packets will return a relevant Syndrome.
	 */
	uint32_t remove_flow_soft_lifetime;
	/**< Soft Lifetime threshold value.
	 * When remove_flow_packet_count reaches this value a soft lifetime event is triggered (if armed).
	 * See remove_flow_packet_count field in this struct fro more details.
	 */
	uint32_t soft_lifetime_arm : 1;
	/**< 1 when armed/to arm 0 otherwise. */
	uint32_t hard_lifetime_arm : 1;
	/**< 1 when armed/to arm 0 otherwise. */
	uint32_t remove_flow_enable : 1;
	/**< 1 when remove flow enabled/to enable; 0 otherwise. */
	uint32_t esn_overlap_event_arm : 1;
	/**< 1 when armed/to arm 0 otherwise. */
};

/** @brief IPSec sa sn attributes - attributes for sequence number - only if SN or AR enabled */
struct doca_ipsec_sa_attr_sn {
	uint32_t esn_overlap : 1; /**< new/old indication of the High sequence number MSB - when set is old */
	uint32_t esn_enable : 1;  /**< when set esn is enabled */
	uint64_t sn_initial;	  /**< set the initial sequence number - in antireplay set the lower bound of the window */
};

/** @brief IPSec sa egress attributes - attributes for outgoing data */
struct doca_ipsec_sa_attr_egress {
	uint32_t sn_inc_enable : 1; /**< when set sn increment offloaded */
};

/** @brief IPSec sa egress attributes - attributes for incoming data */
struct doca_ipsec_sa_attr_ingress {
	uint32_t antireplay_enable : 1;
	/**< when enabled activates anti-replay protection window. */
	enum doca_ipsec_replay_win_size replay_win_sz;
	/**< Anti replay window size to enable sequence replay attack handling. */
};

/** @brief IPSec attributes for create */
struct doca_ipsec_sa_attrs {
	struct doca_encryption_key key;					/**< IPSec encryption key */
	enum doca_ipsec_icv_length icv_length;			/**< Authentication Tag length */
	struct doca_ipsec_sa_attr_sn sn_attr;			/**< sn attributes */
	enum doca_ipsec_direction direction;			/**< egress/ingress */
	union {							/**< egress/ingress attr */
		struct doca_ipsec_sa_attr_egress egress;	/**< egress attr */
		struct doca_ipsec_sa_attr_ingress ingress;	/**< ingress attr */
	};
	struct doca_ipsec_sa_event_attrs event;			/**< Reserve future use - ipsec events flags */
};

/**
 * @brief Opaque structure representing a doca ipsec instance.
 */
struct doca_ipsec;

/**
 * @brief Create a DOCA ipsec instance.
 *
 * @param [in] dev
 * The device to attach to the ipsec instance.
 * @param [out] ipsec
 * Pointer to pointer to be set to point to the created doca_ipsec instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - ipsec argument is a NULL pointer.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate sufficient memory for doca_ipsec.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_create(struct doca_dev *dev, struct doca_ipsec **ipsec);

/**
 * @brief Destroy DOCA IPSEC instance.
 *
 * @param [in] ctx
 * Instance to be destroyed, MUST NOT BE NULL.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_IN_USE - the ctx still in use.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_destroy(struct doca_ipsec *ctx);

/**
 * @brief Convert IPSec instance into doca context
 *
 * @param [in] ctx
 * IPSEC instance. This must remain valid until after the context is no longer required.
 *
 * @return
 * Non NULL - doca_ctx object on success.
 * Error:
 * - NULL.
 */
DOCA_EXPERIMENTAL
struct doca_ctx *doca_ipsec_as_ctx(struct doca_ipsec *ctx);

/**
 * @brief set the sa pool size for sa objects that are return by create
 *
 * @note The range of valid values for this property depend upon the device in use. This means that acceptance of a
 * value through this API does not ensure the value is acceptable, this will be validated as part of starting the
 * context
 *
 * @param [in] ctx
 * IPSEC instance.
 *
 * @param [in] pool_size
 * Number of items to have available. default is 16,384
 *
 * @return
 * DOCA_SUCCESS - Property was successfully set
 * Error code - in case of failure:
 * DOCA_ERROR_INVALID_VALUE - received invalid input.
 * DOCA_ERROR_NO_LOCK - Unable to gain exclusive control of ipsec instance.
 * DOCA_ERROR_IN_USE - ipsec instance is currently started.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_set_sa_pool_size(struct doca_ipsec *ctx, uint32_t pool_size);


/**
 * @brief set the the offload type object, with SN and AR offload
 *
 * Can enable anti-replay protection window and sn increment offloaded.
 * Without this, SN / AR are not enabled
 *
 * @param [in] ctx
 * IPSEC instance.
 *
 * @param [in] offload
 * see enum doca_ipsec_sa_offload - default is DOCA_IPSEC_SA_OFFLOAD_FULL
 *
 * @return
 * DOCA_SUCCESS - Property was successfully set
 * Error code - in case of failure:
 * DOCA_ERROR_INVALID_VALUE - received invalid input.
 * DOCA_ERROR_NO_LOCK - Unable to gain exclusive control of ipsec instance.
 * DOCA_ERROR_IN_USE - ipsec instance is currently started.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_set_offload_type(struct doca_ipsec *ctx, enum doca_ipsec_sa_offload offload);

/**
 * @brief Get is device support sn_enabled capabilities
 *
 * @param [in] devinfo
 * The DOCA device information
 *
 * @return
 * DOCA_SUCCESS - in case of success - capability supported.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - failed to query device capabilities
 *                              or provided devinfo does not support the given capabilitie.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_sequence_number_get_supported(const struct doca_devinfo *devinfo);

/**
 * @brief Get is device support antireplay_enable capabilities
 *
 * @param [in] devinfo
 * The DOCA device information
 *
 * @return
 * DOCA_SUCCESS - in case of success - capability supported.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - failed to query device capabilities
 *                              or provided devinfo does not support the given capabilitie.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_antireplay_get_supported(const struct doca_devinfo *devinfo);

/**
 * @brief IPSEC events handler, update relevnt data according to HW events
 * ESN overlap handler function
 * Update the msb of the sn when overlap event occurse, and arm the event again for next time
 *
 * @param [in] ctx
 * IPSEC instance.
 * @param [out] next_update_time
 * Should call again in the next microseconds - i.e. if using sleep between calls then - usleep(next_update_time)
 * Only valid on return value DOCA_SUCCESS
 *
 * @return
 * Non NULL - sa object of ipsec.
 * Error:
 * DOCA_SUCCESS - Property was successfully set
 * Error code - in case of failure:
 * DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_event_handler(struct doca_ipsec *ctx, uint64_t *next_update_time);

/**
 * @brief DOCA IPSec SA opaque handle.
 * This object should be passed to DOCA Flow to create enc/dec action
 */
struct doca_ipsec_sa;

/**
 * @brief This task preforms a sa creation.
 */
struct doca_ipsec_task_sa_create;

/**
 * @brief Function to execute on completion of a sa create task.
 *
 * @details This function is called by doca_pe_progress() when a sa create task is successfully identified
 * as completed.
 * When this function is called the ownership of the task object passes from DOCA back to user.
 * Inside this callback the user may decide on the task object:
 * - re-submit task with doca_task_submit(); task object ownership passed to DOCA
 * - release task with doca_task_free(); task object ownership passed to DOCA
 * - keep the task object for future re-use; user keeps the ownership on the task object
 * Inside this callback the user shouldn't call doca_pe_progress().
 * Please see doca_pe_progress() for details.
 *
 * Any failure/error inside this function should be handled internally or deferred;
 * Since this function is nested in the execution of doca_pe_progress(), this callback doesn't return an error.
 *
 * @note This callback type is utilized for both successful & failed task completions.
 *
 * @param [in] task
 * The completed sa create task.
 * @note The implementation can assume this value is not NULL.
 * @param [in] task_user_data
 * The doca_data supplied to the task by the application (during task allocation or by a setter).
 * @param [in] ctx_user_data
 * The doca_data supplied to the doca_ctx by the application (using a setter).
 */
typedef void (*doca_ipsec_task_sa_create_completion_cb_t)(struct doca_ipsec_task_sa_create *task,
							union doca_data task_user_data,
							union doca_data ctx_user_data);

/**
 * Check if a given device supports executing a sa create task.
 *
 * @param [in] devinfo
 * The DOCA device information that should be queried.
 *
 * @return
 * DOCA_SUCCESS - in case device supports the task.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - provided devinfo does not support the task.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_cap_task_sa_create_is_supported(const struct doca_devinfo *devinfo);

/**
 * @brief This method sets the sa create tasks configuration.
 *
 * @param [in] ipsec
 * The ipsec instance to config.
 * @param [in] successful_task_completion_cb
 * A callback function for sa create tasks that were completed successfully.
 * @param [in] error_task_completion_cb
 * A callback function for sa create tasks that were completed with an error.
 * @param [in] num_tasks
 * Number of sa create tasks.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_task_sa_create_set_conf(struct doca_ipsec *ipsec,
					      doca_ipsec_task_sa_create_completion_cb_t successful_task_completion_cb,
					      doca_ipsec_task_sa_create_completion_cb_t error_task_completion_cb,
					      uint8_t num_tasks);

/**
 * @brief This method allocates and initializes a sa create task.
 *
 * @param [in] ipsec
 * The ipsec instance to allocate the task for.
 * @param [in] sa_attrs
 * ipsec sa attr
 * @param [in] user_data
 * doca_data to attach to the task.
 * @param [out] task
 * On success, an allocated and initialized sa create task.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - no more tasks to allocate.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_task_sa_create_allocate_init(struct doca_ipsec *ipsec,
						   const struct doca_ipsec_sa_attrs *sa_attrs,
						   union doca_data user_data,
						   struct doca_ipsec_task_sa_create **task);

/**
 * @brief This method converts an ipsec sa create task to a doca_task.
 *
 * @param [in] task
 * The task that should be converted.
 *
 * @return
 * doca_task
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_ipsec_task_sa_create_as_task(struct doca_ipsec_task_sa_create *task);

/**
 * @brief This method sets the sa_attrs of a sa create task.
 *
 * @param [in] task
 * The task to set.
 * @param [in] sa_attrs
 * ipsec sa attr
 *
 */
DOCA_EXPERIMENTAL
void doca_ipsec_task_sa_create_set_sa_attrs(struct doca_ipsec_task_sa_create *task,
					       const struct doca_ipsec_sa_attrs *sa_attrs);

/**
 * @brief This method gets the sa_attrs of a sa create task.
 *
 * @param [in] task
 * The task that should be queried.
 *
 * @return
 * The task's sa_attrs.
 */
DOCA_EXPERIMENTAL
const struct doca_ipsec_sa_attrs *doca_ipsec_task_sa_create_get_sa_attrs(const struct doca_ipsec_task_sa_create *task);

/**
 * @brief This method sets the sa of a sa create task.
 *
 * @param [in] task
 * The task to set.
 * @param [in] sa
 * ipsec sa
 *
 */
DOCA_EXPERIMENTAL
void doca_ipsec_task_sa_create_set_sa(struct doca_ipsec_task_sa_create *task,
					       const struct doca_ipsec_sa *sa);

/**
 * @brief This method gets the sa of a sa create task.
 *
 * @param [in] task
 * The task that should be queried.
 *
 * @return
 * The task's sa.
 */
DOCA_EXPERIMENTAL
const struct doca_ipsec_sa *doca_ipsec_task_sa_create_get_sa(const struct doca_ipsec_task_sa_create *task);

/**
 * @brief This task preforms a sa destroy.
 */
struct doca_ipsec_task_sa_destroy;

/**
 * @brief Function to execute on completion of a sa destroy task.
 *
 * @details This function is called by doca_pe_progress() when a sa destroy task is successfully identified
 * as completed.
 * When this function is called the ownership of the task object passes from DOCA back to user.
 * Inside this callback the user may decide on the task object:
 * - re-submit task with doca_task_submit(); task object ownership passed to DOCA
 * - release task with doca_task_free(); task object ownership passed to DOCA
 * - keep the task object for future re-use; user keeps the ownership on the task object
 * Inside this callback the user shouldn't call doca_pe_progress().
 * Please see doca_pe_progress() for details.
 *
 * Any failure/error inside this function should be handled internally or deferred;
 * Since this function is nested in the execution of doca_pe_progress(), this callback doesn't return an error.
 *
 * @note This callback type is utilized for both successful & failed task completions.
 *
 * @param [in] task
 * The completed sa destroy task.
 * @note The implementation can assume this value is not NULL.
 * @param [in] task_user_data
 * The doca_data supplied to the task by the application (during task allocation or by a setter).
 * @param [in] ctx_user_data
 * The doca_data supplied to the doca_ctx by the application (using a setter).
 */
typedef void (*doca_ipsec_task_sa_destroy_completion_cb_t)(struct doca_ipsec_task_sa_destroy *task,
							union doca_data task_user_data,
							union doca_data ctx_user_data);

/**
 * Check if a given device supports executing a sa destroy task.
 *
 * @param [in] devinfo
 * The DOCA device information that should be queried.
 *
 * @return
 * DOCA_SUCCESS - in case device supports the task.
 * Error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - provided devinfo does not support the task.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_cap_task_sa_destroy_is_supported(const struct doca_devinfo *devinfo);

/**
 * @brief This method sets the sa destroy tasks configuration.
 *
 * @param [in] ipsec
 * The ipsec instance to config.
 * @param [in] successful_task_completion_cb
 * A callback function for sa destroy tasks that were completed successfully.
 * @param [in] error_task_completion_cb
 * A callback function for sa destroy tasks that were completed with an error.
 * @param [in] num_tasks
 * Number of sa destroy tasks.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_task_sa_destroy_set_conf(struct doca_ipsec *ipsec,
					      doca_ipsec_task_sa_destroy_completion_cb_t successful_task_completion_cb,
					      doca_ipsec_task_sa_destroy_completion_cb_t error_task_completion_cb,
					      uint8_t num_tasks);

/**
 * @brief This method allocates and initializes a sa destroy task.
 *
 * @param [in] ipsec
 * The ipsec instance to allocate the task for.
 * @param [in] sa
 * ipsec sa
 * @param [in] user_data
 * doca_data to attach to the task.
 * @param [out] task
 * On success, an allocated and initialized sa destroy task.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - no more tasks to allocate.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_ipsec_task_sa_destroy_allocate_init(struct doca_ipsec *ipsec,
						   const struct doca_ipsec_sa *sa,
						   union doca_data user_data,
						   struct doca_ipsec_task_sa_destroy **task);

/**
 * @brief This method converts an ipsec sa destroy task to a doca_task.
 *
 * @param [in] task
 * The task that should be converted.
 *
 * @return
 * doca_task
 */
DOCA_EXPERIMENTAL
struct doca_task *doca_ipsec_task_sa_destroy_as_task(struct doca_ipsec_task_sa_destroy *task);

/**
 * @brief This method sets the sa of a sa destroy task.
 *
 * @param [in] task
 * The task to set.
 * @param [in] sa
 * ipsec sa
 *
 */
DOCA_EXPERIMENTAL
void doca_ipsec_task_sa_destroy_set_sa(struct doca_ipsec_task_sa_destroy *task,
					       const struct doca_ipsec_sa *sa);

/**
 * @brief This method gets the sa of a sa destroy task.
 *
 * @param [in] task
 * The task that should be queried.
 *
 * @return
 * The task's sa.
 */
DOCA_EXPERIMENTAL
const struct doca_ipsec_sa *doca_ipsec_task_sa_destroy_get_sa(const struct doca_ipsec_task_sa_destroy *task);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_IPSEC_H_ */
