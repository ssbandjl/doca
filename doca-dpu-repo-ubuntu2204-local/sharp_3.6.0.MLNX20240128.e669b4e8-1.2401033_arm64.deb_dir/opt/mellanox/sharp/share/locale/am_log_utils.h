
/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <map>
#include <string>
#include <type_traits>
#include "common/log.h"

class LogOnce
{
   public:
    template <typename... LogParameters>
    void log(char const* const category,
             const int level,
             char const* const file_name,
             const int line_number,
             char const* const function_name,
             char const* const format,
             LogParameters&&... log_parameters)
    {
        if (!m_should_log_) {
            return;
        }
        if (!log_check_level(category, level)) {
            return;
        }

        std::string log_timed_format = "O: ";
        log_timed_format.append(format);
        log_send(category,
                 level,
                 file_name,
                 line_number,
                 function_name,
                 log_timed_format.c_str(),
                 std::forward<LogParameters>(log_parameters)...);
        m_should_log_ = false;
    }

   private:
    bool m_should_log_ = true;
};

template <typename ChronoDurationType, uint32_t Interval>
class LogTimed
{
   public:
    template <typename... LogParameters>
    void log(char const* const category,
             const int level,
             char const* const file_name,
             const int line_number,
             char const* const function_name,
             char const* const format,
             LogParameters&&... log_parameters)
    {
        ++m_number_of_requests_since_last_log_;
        if (!log_check_level(category, level)) {
            return;
        }
        if ((0 != m_last_log_.time_since_epoch().count()) && (!ShouldLog())) {
            return;
        }

        m_last_log_ = std::chrono::steady_clock::now();
        std::string log_timed_format = "T[%zu]: ";
        log_timed_format.append(format);
        log_send(category,
                 level,
                 file_name,
                 line_number,
                 function_name,
                 log_timed_format.c_str(),
                 m_number_of_requests_since_last_log_,
                 std::forward<LogParameters>(log_parameters)...);
        m_number_of_requests_since_last_log_ = 0;
    }

   private:
    bool ShouldLog() const
    {
        const auto time_since_last_log =
            std::chrono::duration_cast<ChronoDurationType>(std::chrono::steady_clock::now() - m_last_log_).count();
        return (Interval <= time_since_last_log);
    }

    std::chrono::steady_clock::time_point m_last_log_;
    std::size_t m_number_of_requests_since_last_log_ = 0;
};

template <uint32_t Interval>
using LogTimedMinutes = LogTimed<std::chrono::minutes, Interval>;

template <uint32_t Interval>
using LogTimedSeconds = LogTimed<std::chrono::seconds, Interval>;
