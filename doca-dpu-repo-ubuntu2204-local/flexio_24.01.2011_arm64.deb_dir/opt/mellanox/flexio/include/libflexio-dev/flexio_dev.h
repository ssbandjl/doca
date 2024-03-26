/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file flexio_dev.h
 * @page Flex IO SDK dev
 * @defgroup FlexioSDKDev Dev
 * @ingroup FlexioSDK
 * Flex IO SDK device API for DPA programs.
 * Includes services for DPA programs.
 *
 * @{
 */

#ifndef _FLEXIO_DEV_H_
#define _FLEXIO_DEV_H_

#include <stdint.h>

typedef uint64_t flexio_uintptr_t;

#define      __unused        __attribute__((__unused__))

#define __FLEXIO_ENTRY_POINT_START _Pragma("clang section text=\".entry_point\"")
#define __FLEXIO_ENTRY_POINT_END   _Pragma("clang section text=\"\"")

/* #define SPINLOCK_DEBUG */

/** Return status of Flex IO dev API functions. */
typedef enum {
	FLEXIO_DEV_STATUS_SUCCESS = 0,
	FLEXIO_DEV_STATUS_FAILED  = 1,
} flexio_dev_status_t;

/** Flex IO UAR extension ID prototype. */
typedef uint32_t flexio_uar_device_id;

/** Flex IO dev CQ CQE creation modes. */
enum cq_ce_mode {
	MLX5_CTRL_SEG_CE_CQE_ON_CQE_ERROR       = 0x0,
	MLX5_CTRL_SEG_CE_CQE_ON_FIRST_CQE_ERROR = 0x1,
	MLX5_CTRL_SEG_CE_CQE_ALWAYS             = 0x2,
	MLX5_CTRL_SEG_CE_CQE_AND_EQE            = 0x3,
};

enum flexio_dev_nic_counter_ids {
	FLEXIO_DEV_NIC_COUNTER_PORT0_RX_BYTES = 0x10,
	FLEXIO_DEV_NIC_COUNTER_PORT1_RX_BYTES = 0x11,
	FLEXIO_DEV_NIC_COUNTER_PORT2_RX_BYTES = 0x12,
	FLEXIO_DEV_NIC_COUNTER_PORT3_RX_BYTES = 0x13,

	FLEXIO_DEV_NIC_COUNTER_PORT0_TX_BYTES = 0x20,
	FLEXIO_DEV_NIC_COUNTER_PORT1_TX_BYTES = 0x21,
	FLEXIO_DEV_NIC_COUNTER_PORT2_TX_BYTES = 0x22,
	FLEXIO_DEV_NIC_COUNTER_PORT3_TX_BYTES = 0x23,
};

struct flexio_dev_thread_ctx;

/* Function types for NET and RPC handlers */

/**
 * RPC handler callback function type.
 *
 * Defines an RPC handler for most useful callback function.
 *
 * arg - argument of the RPC function.
 *
 * return uint64_t - result of the RPC function.
 */
typedef uint64_t (flexio_dev_rpc_handler_t)(uint64_t arg);

/**
 * Unpack the arguments and call the user function.
 *
 * This callback function is used at runtime to unpack the
 * arguments from the call on Host and then call the function on DPA.
 * This function is called internally from flexio dev.
 *
 * argbuf - Argument buffer that was written by Host.
 * func - Function pointer to user function.
 *
 * return uint64_t - result of the RPC function.
 */
typedef uint64_t (flexio_dev_arg_unpack_func_t)(void *argbuf, void *func);

/**
 * Asynchronous RPC handler callback function type.
 *
 * Defines an RPC handler callback function.
 *
 * arg - argument of the RPC function.
 *
 * return void.
 */
typedef void (flexio_dev_async_rpc_handler_t)(uint64_t arg);

/**
 * Event handler callback function type.
 *
 * Defines an event handler callback function.
 * On handler function end, need to call flexio_dev_process_finish() instead of a regular
 * return statement, in order to properly release resources back to the OS.
 *
 * thread_arg - an argument for the executing thread.
 *
 * return void.
 */
typedef void (flexio_dev_event_handler_t)(uint64_t thread_arg);

/**
 * Describes Flex IO dev spinlock.
 */
struct spinlock_s {
	uint32_t locked;        /**< Indication for spinlock lock state. */
#ifdef SPINLOCK_DEBUG
	uint32_t locker_tid;    /**< Locker thread ID. */
#endif
};

/**
 * @brief Initialize a spinlock mechanism.
 *
 * Initialize a spinlock mechanism, must be called before use.
 *
 * @param[in] lock - A pointer to spinlock_s structure.
 *
 * @return void.
 */
#ifdef SPINLOCK_DEBUG

#define spin_init(lock) \
	do { \
		__atomic_store_n(&((lock)->locked), 0, __ATOMIC_SEQ_CST); \
		__atomic_store_n(&((lock)->locker_tid), 0xffffffff, __ATOMIC_SEQ_CST); \
	} while (0)

#else

#define spin_init(lock) __atomic_store_n(&((lock)->locked), 0, __ATOMIC_SEQ_CST)

#endif

/**
 * @brief Lock a spinlock mechanism.
 *
 * Lock a spinlock mechanism.
 *
 * @param[in] lock - A pointer to spinlock_s structure.
 *
 * @return void.
 */
#ifdef SPINLOCK_DEBUG

#define spin_lock(lock) \
	do { \
		struct flexio_dev_thread_ctx *dtctx; \
		uint32_t tid; \
		flexio_dev_get_thread_ctx(&dtctx); \
		tid = flexio_dev_get_thread_id(dtctx); \
		while (__atomic_exchange_n(&((lock)->locked), 1, __ATOMIC_SEQ_CST)) {;} \
		__atomic_store_n(&((lock)->locker_tid), tid, __ATOMIC_SEQ_CST); \
	} while (0)

#else
#define spin_lock(lock) \
	do { \
		while (__atomic_exchange_n(&((lock)->locked), 1, __ATOMIC_SEQ_CST)) {;} \
	} while (0)
#endif

/**
 * @brief Unlock a spinlock mechanism.
 *
 * Unlock a spinlock mechanism.
 *
 * @param[in] lock - A pointer to spinlock_s structure.
 *
 * @return void.
 */
#ifdef SPINLOCK_DEBUG

#define spin_unlock(lock) \
	do { \
		__atomic_store_n(&((lock)->locker_tid), 0xffffffff, __ATOMIC_SEQ_CST); \
		__atomic_store_n(&((lock)->locked), 0, __ATOMIC_SEQ_CST); \
	} while (0)

#else
#define spin_unlock(lock) __atomic_store_n(&((lock)->locked), 0, __ATOMIC_SEQ_CST)
#endif

/**
 * @brief Atomic try to catch lock.
 *
 * makes attempt to take lock. Returns immediately.
 *
 * @param[in] lock - A pointer to spinlock_s structure.
 *
 * @return zero on success. Nonzero otherwise.
 */
#define spin_trylock(lock) __atomic_exchange_n(&((lock)->locked), 1, __ATOMIC_SEQ_CST)

/**
 * @brief Request thread context.
 *
 * This function requests the thread context. Should be called
 * for every start of thread.
 *
 * @param[in] dtctx - A pointer to a pointer of flexio_dev_thread_ctx structure.
 *
 * @return 0 on success negative value on failure.
 */
int flexio_dev_get_thread_ctx(struct flexio_dev_thread_ctx **dtctx);

/**
 * @brief Exit from a thread, leave process active.
 *
 * This function releases resources back to OS.
 * For the next DUAR the thread will restart from the beginning.
 *
 * @param[in] void.
 *
 * @return Function does not return.
 */
void flexio_dev_thread_reschedule(void);
void flexio_dev_reschedule(void) __attribute__ ((deprecated));

/**
 * @brief Exit from a thread, mark it as finished.
 *
 * This function releases resources back to OS.
 * The thread will be marked as finished so next DUAR will not trigger it.
 *
 * @param[in] void.
 *
 * @return Function does not return.
 */
void flexio_dev_thread_finish(void);

/**
 * @brief Exit from a thread, and retrigger it.
 *
 * This function asks the OS to retrigger the thread.
 * The thread will not wait for the next DUAR to be triggered but will be triggered
 * immediately.
 *
 * @param[in] void.
 *
 * @return Function does not return.
 */
void flexio_dev_thread_retrigger(void);

/**
 * @brief Exit flexio process (no errors).
 *
 * This function releases resources back to OS and returns '0x40' in dpa_process_status.
 * All threads for the current process will stop executing and no new threads will be able
 * to trigger for this process.
 * Threads state will NOT be changes to 'finished' (will remain as is).
 *
 * @param[in] void.
 *
 * @return Function does not return.
 */
void flexio_dev_process_finish(void);
void flexio_dev_finish(void) __attribute__ ((deprecated));

/**
 * @brief Put a string to messaging queue.
 *
 * This function puts a string to host's default stream messaging queue.
 * This queue has been serviced by host application.
 * Would have no effect, if the host application didn't configure device messaging stream
 * environment.
 * In order to initialize/configure device messaging environment -
 * On HOST side - after flexio_process_create, a stream should be created, therefore
 * flexio_msg_stream_create should be called, and the default stream should be created.
 * On DEV side - before using flexio_dev_puts, the thread context is needed, therefore
 * flexio_dev_get_thread_ctx should be called before.
 *
 * @param[in] dtctx - A pointer to a pointer of flexio_dev_thread_ctx structure.
 * @param[in] str - A pointer to string.
 *
 * @return length of messaged string.
 */
int flexio_dev_puts(struct flexio_dev_thread_ctx *dtctx, char *str);

/**
 * @brief Config thread outbox object without any checks.
 *
 * This function updates the thread outbox object of the given thread
 * context, but it doesn't check for correctness or redundancy (same ID as current configured).
 *
 * @param[in] dtctx - A pointer to flexio_dev_thread_ctx structure.
 * @param[in] outbox_config_id - The outbox object config id.
 *
 * @return void.
 */
void flexio_dev_outbox_config_fast(struct flexio_dev_thread_ctx *dtctx, uint16_t outbox_config_id);

/**
 * @brief Config thread outbox object.
 *
 * This function updates the thread outbox object of the given thread
 * context.
 *
 * @param[in] dtctx - A pointer to flexio_dev_thread_ctx structure.
 * @param[in] outbox_config_id - The outbox object config id.
 *
 * @return flexio_dev_status_t.
 */
flexio_dev_status_t flexio_dev_outbox_config(struct flexio_dev_thread_ctx *dtctx,
					     uint16_t outbox_config_id);

/**
 * @brief Config thread window object.
 *
 * This function updates the thread window object of the given thread
 * context.
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 * @param[in] window_config_id - The window object id.
 * @param[in] mkey - mkey object.
 *
 * @return flexio_dev_status_t.
 */
flexio_dev_status_t flexio_dev_window_config(struct flexio_dev_thread_ctx *dtctx,
					     uint16_t window_config_id, uint32_t mkey);

/**
 * @brief Config thread window mkey object.
 *
 * This function updates the thread window mkey object of the given thread
 * context.
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 * @param[in] mkey - mkey object.
 *
 * @return flexio_dev_status_t.
 */
flexio_dev_status_t flexio_dev_window_mkey_config(struct flexio_dev_thread_ctx *dtctx,
						  uint32_t mkey);

/**
 * @brief Generate device address from host allocated memory.
 *
 * This function generates a memory address to be used by device to access host side memory,
 * according to already create window object. from a host allocated address.
 *
 * @param[in]  dtctx - A pointer to a flexio_dev_thread_ctx structure.
 * @param[in]  haddr - Host allocated address.
 * @param[out] daddr - A pointer to write the device generated matching address.
 *
 * @return flexio_dev_status_t.
 */
flexio_dev_status_t flexio_dev_window_ptr_acquire(struct flexio_dev_thread_ctx *dtctx,
						  uint64_t haddr, flexio_uintptr_t *daddr);

/**
 * @brief Copy a buffer from host memory to device memory.
 *
 * This function copies specified number of bytes from host memory to device memory.
 * UNSUPPORTED at this time.
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 * @param[in] daddr - A pointer to the device memory buffer.
 * @param[in] haddr - A pointer to the host memory allocated buffer.
 * @param[in] size - Number of bytes to copy.
 *
 * @return flexio_dev_status_t.
 */
flexio_dev_status_t flexio_dev_window_copy_from_host(struct flexio_dev_thread_ctx *dtctx,
						     void *daddr, uint64_t haddr, uint32_t size);

/**
 * @brief Copy a buffer from device memory to host memory.
 *
 * This function copies specified number of bytes from device memory to host memory.
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 * @param[in] haddr - A pointer to the host memory allocated buffer.
 * @param[in] daddr - A pointer to the device memory buffer.
 * @param[in] size - Number of bytes to copy.
 *
 * @return flexio_dev_status_t.
 */
flexio_dev_status_t flexio_dev_window_copy_to_host(struct flexio_dev_thread_ctx *dtctx,
						   uint64_t haddr, const void *daddr,
						   uint32_t size);

/**
 * @brief Get thread ID from thread context
 *
 * This function queries a thread context for its thread ID (from thread metadata).
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 *
 * @return thread ID value.
 */
uint32_t flexio_dev_get_thread_id(struct flexio_dev_thread_ctx *dtctx);

/**
 * @brief Get thread local storage address from thread context
 *
 * This function queries a thread context for its thread local storage (from thread metadata).
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 *
 * @return thread local storage value.
 */
flexio_uintptr_t flexio_dev_get_thread_local_storage(struct flexio_dev_thread_ctx *dtctx);

/* Flex IO device messaging. */
#define FLEXIO_MSG_DEV_BROADCAST_STREAM 0xff /* The 256'th stream - outputs to all streams */
#define FLEXIO_MSG_DEV_MAX_STREAMS_AMOUNT 255
#define FLEXIO_MSG_DEV_DEFAULT_STREAM_ID 0
/* Flex IO device messaging levels. */
/* A usage of FLEXIO_MSG_DEV_NO_PRINT in flexio_dev_msg will terminate with no log. */
#define FLEXIO_MSG_DEV_NO_PRINT 0
#define FLEXIO_MSG_DEV_ALWAYS_PRINT 1
#define FLEXIO_MSG_DEV_ERROR 2
#define FLEXIO_MSG_DEV_WARN 3
#define FLEXIO_MSG_DEV_INFO 4
#define FLEXIO_MSG_DEV_DEBUG 5

typedef uint8_t flexio_msg_dev_level;

/**
 * @brief Creates message entry and outputs from the device to the host side.
 * Same as a regular printf but with protection from simultaneous print from different threads.
 *
 * @params[in] stream_id - the relevant msg stream, created and passed from the host.
 * @params[in] level - messaging level.
 * @params[in] format, ... - same as for regular printf.
 *
 * @return - same as from regular printf.
 */
int flexio_dev_msg(int stream_id, flexio_msg_dev_level level, const char *format, ...)
__attribute__ ((format(printf, 3, 4)));

/* Device messaging streams concise MACROs */
#define flexio_dev_msg_err(strm_id, ...) flexio_dev_msg(strm_id, FLEXIO_MSG_DEV_ERROR, __VA_ARGS__)
#define flexio_dev_msg_warn(strm_id, ...) flexio_dev_msg(strm_id, FLEXIO_MSG_DEV_WARN, __VA_ARGS__)
#define flexio_dev_msg_info(strm_id, ...) flexio_dev_msg(strm_id, FLEXIO_MSG_DEV_INFO, __VA_ARGS__)
#define flexio_dev_msg_dbg(strm_id, ...) flexio_dev_msg(strm_id, FLEXIO_MSG_DEV_DEBUG, __VA_ARGS__)

/**
 * @brief Create message entry and outputs from the device to host's default stream.
 * Same as a regular printf but with protection from simultaneous print from different threads.
 *
 * @params[in] level - messaging level.
 * @params[in] ... - format and the parameters. Same as for regular printf.
 *
 * @return - same as from regular printf.
 */
#define flexio_dev_msg_dflt(lvl, ...) \
	flexio_dev_msg(FLEXIO_MSG_DEV_DEFAULT_STREAM_ID, lvl, __VA_ARGS__)

/**
 * @brief Create message entry and outputs from the device to all of the host's open streams.
 * Same as a regular printf but with protection from simultaneous print from different threads.
 *
 * @params[in] level - messaging level.
 * @params[in] ... - format and the parameters. Same as for regular printf.
 *
 * @return - same as from regular printf.
 */
#define flexio_dev_msg_broadcast(lvl, ...) \
	flexio_dev_msg(FLEXIO_MSG_DEV_BROADCAST_STREAM, lvl, __VA_ARGS__)

/**
 * @brief Create message entry and outputs from the device to host's default stream,
 * with FLEXIO_MSG_DEV_INFO message level.
 * Same as a regular printf but with protection from simultaneous print from different threads.
 *
 * @params[in] ... - format and the parameters. Same as for regular printf.
 *
 * @return - same as from regular printf.
 */
#define flexio_dev_print(...) \
	flexio_dev_msg(FLEXIO_MSG_DEV_DEFAULT_STREAM_ID, FLEXIO_MSG_DEV_INFO, __VA_ARGS__)

/**
 * @brief exit point for continuable event handler routine
 *
 * This function is used to mark the exit point on continuable event handler where
 * user wishes to continue execution on next event. In order to use this API the event handler must
 * be created with continuable flag enabled, otherwise call will have no effect.
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 *
 * @return Function does not return.
 */
void flexio_dev_yield(struct flexio_dev_thread_ctx *dtctx);

/**
 * @brief get programable congestion control table base address
 *
 * This function gets the programable congestion control table base address.
 *
 * @param[in] gvmi - PCC table GVMI.
 *
 * @return PCC table base address for the given GVMI.
 */
uint64_t flexio_dev_get_pcc_table_base(uint16_t gvmi);

/**
 * @brief set extension ID for outbox
 *
 * This function sets the GVMI for the outbox to operate on.
 *
 * @param[in] dtctx - A pointer to a flexio_dev_thread_ctx structure.
 * @param[in] device_id - The device ID.
 *
 * @return flexio_dev_status_t.
 */
flexio_dev_status_t flexio_dev_outbox_config_uar_extension(struct flexio_dev_thread_ctx *dtctx,
							   flexio_uar_device_id device_id);

/**
 * @brief Prepare a list of counters to read
 *
 * The list is stored in kernel memory. A single counters config per process is supported.
 * Note that arrays memory must be defined in global or heap memory only.
 *
 * @param[out] values - buffer to store counters values (32b) read by
 *                      flexio_dev_nic_counters_sample().
 * @param[in]  counter_ids - An array of counter ids.
 * @param[in]  num_counters - number of counters in the counter_ids array
 * @return void
 *       process crashes in case of:
 *       counters_ids too large
 *       bad pointers of values, counter_ids
 *       unknown counter
 */
void flexio_dev_nic_counters_config(uint32_t *values, uint32_t *counter_ids, uint32_t num_coutners);

/**
 * @brief Sample counters according to the prior configuration call
 *
 * Sample counter_ids, num_counters and values buffer provided in the last successful call to
 * flexio_dev_config_nic_counters().
 * This call ensures fastest sampling on a pre-checked counter ids and buffers.
 *
 * @return void.
 *    process crashes in case of: flexio_dev_config_nic_counters() never called
 */
void flexio_dev_nic_counters_sample(void);

/** @} */

#endif /* _FLEXIO_DEV_H_ */
