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

#include <string>
#include <utility>

class ReservedString
{
   public:
    ReservedString(const std::size_t reserve_size) { m_str_.reserve(reserve_size); }
    inline void Reserve(const std::size_t size) { m_str_.reserve(size); }

    inline ReservedString& operator<<(char const* const msg_buffer)
    {
        m_str_ += msg_buffer;
        return *this;
    }

    inline ReservedString& operator<<(const std::string& msg)
    {
        m_str_ += msg;
        return *this;
    }

    template <typename Type, typename = void>
    inline ReservedString& operator<<(const Type num)
    {
        m_str_ += std::to_string(num);
        return *this;
    }

    inline const std::string& Get() const { return m_str_; }
    inline std::string& Get() { return m_str_; }
    inline void Clear() { m_str_.clear(); }

   private:
    std::string m_str_;
};
