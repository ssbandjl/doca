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
#pragma once
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

namespace file_utils
{

constexpr std::size_t g_default_maximum_printf_msg_size = 150;

class DumpFile
{
   public:
    DumpFile(std::string file_name,
             std::string file_path,
             const std::size_t maximum_rotation_files,
             const std::size_t maximum_file_size_bytes,
             const bool enable_rotation);
    DumpFile(const DumpFile&) = delete;
    DumpFile& operator=(const DumpFile&) = delete;

    bool IsValid() const;
    bool IsRotationEnabled() const;
    bool Print(const std::string& print_message, const bool add_time_prefix = true);

    template <typename... PrintfParameters>
    bool Printf(const bool add_time_prefix, char const* const msg_format, const PrintfParameters&... printf_parameters)
    {
        return Printf<PrintfParameters...>(g_default_maximum_printf_msg_size, add_time_prefix, msg_format, printf_parameters...);
    }

    template <typename... PrintfParameters>
    bool Printf(const std::size_t buffer_size,
                const bool add_time_prefix,
                char const* const msg_format,
                const PrintfParameters&... printf_parameters)
    {
        std::string msg{};
        msg.resize(buffer_size);
        const auto number_of_written_chars = std::snprintf(&msg[0], buffer_size, msg_format, printf_parameters...);
        if ((number_of_written_chars < 0) || (number_of_written_chars > (int)buffer_size)) {
            WARNING("Could not format message for dump file %s, failed with error: %d buffer size: %zu",
                    m_dump_file_name_.c_str(),
                    number_of_written_chars,
                    buffer_size);
            return false;
        }
        msg.resize(static_cast<std::size_t>(number_of_written_chars));
        return Print(msg, add_time_prefix);
    }

   private:
    bool RotateFilesIfMaximumSizeWasReached();

    std::string m_dump_file_name_;
    std::string m_dump_file_path_;
    std::size_t m_maximum_rotation_files_;
    std::size_t m_maximum_file_size_bytes_;
    bool m_is_rotation_enabled_;
    std::ofstream m_dump_file_;
};

std::unique_ptr<file_utils::DumpFile> GetDumpFileIfEnabled(char const* const dump_name, char const* const dump_file_name);
}   // namespace file_utils
