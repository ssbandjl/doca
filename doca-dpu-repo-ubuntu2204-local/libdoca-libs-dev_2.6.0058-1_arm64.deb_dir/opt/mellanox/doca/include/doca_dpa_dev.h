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
 * @defgroup DPA_DEVICE DPA Device
 * DOCA DPA Device library. For more details please refer to the user guide on DOCA devzone.
 *
 * @ingroup DPA
 *
 * @{
 */

#ifndef DOCA_DPA_DEV_H_
#define DOCA_DPA_DEV_H_

/**
 * @brief declares that we are compiling for the DPA Device
 *
 * @note Must be defined before the first API use/include of DOCA
 */
#define DOCA_DPA_DEVICE

/** Include to define compatibility with current version, define experimental Symbols */
#include <doca_compat.h>
#include <doca_dpa_common.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DPA pointer type definition
 */
__dpa_global__ typedef uint64_t doca_dpa_dev_uintptr_t;

/**
 * @brief static inline wrapper
 */
#ifndef __forceinline
#define __forceinline static inline __attribute__((always_inline))
#endif /* __forceinline */

/**
 * \brief Obtains the thread rank
 *
 * Retrieves the thread rank for a given kernel on the DPA.
 * The function returns a number in {0..N-1}, where N is the number of threads requested for launch during a kernel
 * submission
 *
 * @return
 * Returns the thread rank.
 */
DOCA_EXPERIMENTAL
unsigned int doca_dpa_dev_thread_rank(void);

/**
 * \brief Obtains the number of threads running the kernel
 *
 * Retrieves the number of threads assigned to a given kernel. This is the value `nthreads` that was passed in to
 * 'doca_dpa_kernel_launch_update_set/doca_dpa_kernel_launch_update_add'
 *
 * @return
 * Returns the number of threads running the kernel
 */
DOCA_EXPERIMENTAL
unsigned int doca_dpa_dev_num_threads(void);

/**
 * \brief Yield a DPA thread
 *
 * This function yields a DPA thread that is running a kernel
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_yield(void);

/**
 * \brief Print logs to Host
 *
 * This function prints from device to host's standard output stream or the user defined file
 * set by doca_dpa_log_file_set_path().
 * The log level will determine the print according to the verbosity set by doca_dpa_set_log_level().
 * It is recommended to use the bellow defined MACROs for device logging for better readability.
 * Multiple threads may call these MACROs simultaneously. Printing is a convenience service, and due to limited
 * buffering on the host, not all print statements may appear on the host.
 *
 * @param[in] log_level - level for device log
 * @param[in] format - format string that contains the text to be written to host (same as from regular printf)
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_log(doca_dpa_dev_log_level_t log_level, const char *format, ...)
	__attribute__ ((format (printf, 2, 3)));

/**
 * @brief Generate a DOCA DPA device CRITICAL log message
 *
 * Will generate critical application log. This call affects the performance.
 *
 */
#define DOCA_DPA_DEV_LOG_CRIT(...) doca_dpa_dev_log(DOCA_DPA_DEV_LOG_LEVEL_CRIT, __VA_ARGS__)

/**
 * @brief Generate a DOCA DPA device ERROR log message
 *
 * Will generate error application log. This call affects the performance.
 *
 */
#define DOCA_DPA_DEV_LOG_ERR(...) doca_dpa_dev_log(DOCA_DPA_DEV_LOG_LEVEL_ERROR, __VA_ARGS__)

/**
 * @brief Generate a DOCA DPA device WARNING log message
 *
 * Will generate warning application log. This call affects the performance.
 *
 */
#define DOCA_DPA_DEV_LOG_WARN(...) doca_dpa_dev_log(DOCA_DPA_DEV_LOG_LEVEL_WARNING, __VA_ARGS__)

/**
 * @brief Generate a DOCA DPA device INFO log message
 *
 * Will generate info application log. This call affects the performance.
 *
 */
#define DOCA_DPA_DEV_LOG_INFO(...) doca_dpa_dev_log(DOCA_DPA_DEV_LOG_LEVEL_INFO, __VA_ARGS__)

/**
 * @brief Generate a DOCA DPA device DEBUG log message
 *
 * Will generate debug application log. This call affects the performance.
 *
 */
#define DOCA_DPA_DEV_LOG_DBG(...) doca_dpa_dev_log(DOCA_DPA_DEV_LOG_LEVEL_DEBUG, __VA_ARGS__)

/**
 * @brief Creates trace message entry with arguments
 *
 * This function prints traces arguments from device to host's standard output stream,
 * or to the user's defined outfile set by doca_dpa_trace_file_set_path().
 * It is recommended to use trace for enhanced performance in logging
 *
 * @param[in] arg1 - argument #1 to format into the template
 * @param[in] arg2 - argument #2 to format into the template
 * @param[in] arg3 - argument #3 to format into the template
 * @param[in] arg4 - argument #4 to format into the template
 * @param[in] arg5 - argument #5 to format into the template
 *
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_trace(uint64_t arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5);

/**
 * @brief Flush the trace message buffer to Host
 *
 * As soon as a buffer is fully occupied it is internally sent to host, however
 * user can ask partially occupied buffer to be sent to host.
 * Its intended use is at end of run to flush whatever messages left
 *
 * @note: Frequent call to this API might cause performance issues.
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_trace_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_DPA_DEV_H_ */

/** @} */
