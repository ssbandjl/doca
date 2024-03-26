/*
 * Copyright (c) 2012-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef _AM_LOG_H
#define _AM_LOG_H

#include "am/am_log_utils.h"
#include "common/log.h"
#include "ibis.h"

#ifndef SHARP_DEFAULT_LOG_CAT
#define SHARP_DEFAULT_LOG_CAT SHARP_LOG_CAT_GENERAL
#endif

#define VERBOSITY_ERROR   0x01
#define VERBOSITY_WARNING 0x02
#define VERBOSITY_INFO    0x04
#define VERBOSITY_VERBOSE 0x08
#define VERBOSITY_DEBUG   0x10
#define VERBOSITY         0xFF

#define WRITE_LOG(fmt, arg...)           log_out(SHARP_DEFAULT_LOG_CAT, fmt, ##arg)
#define ERROR(fmt, arg...)               log_error(SHARP_DEFAULT_LOG_CAT, fmt, ##arg)
#define WARNING(fmt, arg...)             log_warn(SHARP_DEFAULT_LOG_CAT, fmt, ##arg)
#define INFO(fmt, arg...)                log_info(SHARP_DEFAULT_LOG_CAT, fmt, ##arg)
#define DEBUG(fmt, arg...)               log_debug(SHARP_DEFAULT_LOG_CAT, fmt, ##arg)
#define VERBOSE(fmt, arg...)             log_trace(SHARP_DEFAULT_LOG_CAT, fmt, ##arg)
#define LOG_BY_LEVEL(level, fmt, arg...) log_by_level(SHARP_DEFAULT_LOG_CAT, level, fmt, ##arg)

// The default LOG TIMED is in minutes
#define LOG_ONCE(level, fmt, arg...)                                                              \
    {                                                                                             \
        static LogOnce log_once{};                                                                \
        log_once.log(SHARP_DEFAULT_LOG_CAT, level, __FILE__, __LINE__, __FUNCTION__, fmt, ##arg); \
    }

#define LOG_TIMED(interval_minutes, level, fmt, arg...)                                            \
    {                                                                                              \
        static LogTimedMinutes<interval_minutes> log_timed{};                                      \
        log_timed.log(SHARP_DEFAULT_LOG_CAT, level, __FILE__, __LINE__, __FUNCTION__, fmt, ##arg); \
    }

#define LOG_TIMED_SECONDS(interval_seconds, level, fmt, arg...)                                    \
    {                                                                                              \
        static LogTimedSeconds<interval_seconds> log_timed{};                                      \
        log_timed.log(SHARP_DEFAULT_LOG_CAT, level, __FILE__, __LINE__, __FUNCTION__, fmt, ##arg); \
    }

// indicates that this line is used by regression - do not remove.
#define DEBUG_VERIFICATION(fmt, arg...) log_debug(SHARP_DEFAULT_LOG_CAT, fmt, ##arg)

void SetLogOptions();
void ibis_log_msg_function(const char* file_name, unsigned line_num, const char* function_name, int level, const char* format, ...);

#endif   // _AM_LOG_H
