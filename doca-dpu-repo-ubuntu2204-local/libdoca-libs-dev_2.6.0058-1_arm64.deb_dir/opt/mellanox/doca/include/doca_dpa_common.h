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
 * @defgroup DPA_COMMON DPA Common
 * DOCA DPA Common library. For more details please refer to the user guide on DOCA devzone.
 *
 * @ingroup DPA
 *
 * @{
 */
#ifndef DOCA_DPA_COMMON_H_
#define DOCA_DPA_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DOCA DPA device log levels, sorted by verbosity from high to low
 */
typedef enum doca_dpa_dev_log_level {
	DOCA_DPA_DEV_LOG_LEVEL_DISABLE	= 10,	/**< Disable log messages */
	DOCA_DPA_DEV_LOG_LEVEL_CRIT	= 20,	/**< Critical log level */
	DOCA_DPA_DEV_LOG_LEVEL_ERROR	= 30,	/**< Error log level */
	DOCA_DPA_DEV_LOG_LEVEL_WARNING	= 40,	/**< Warning log level */
	DOCA_DPA_DEV_LOG_LEVEL_INFO	= 50,	/**< Info log level */
	DOCA_DPA_DEV_LOG_LEVEL_DEBUG	= 60,	/**< Debug log level */
} doca_dpa_dev_log_level_t;

#ifdef __cplusplus
}
#endif

#endif /* DOCA_DPA_COMMON_H_ */

/** @} */
