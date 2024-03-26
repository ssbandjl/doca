/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/time.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "am_log.h"

#define AM_EXE_NAME "sharp_am"

class OptionManager;
struct OptionInfo;
class EventManager;
class Fabric;

extern bool g_should_update_options;
extern bool g_should_terminate;
extern OptionManager g_option_manager;
extern OptionInfo g_option_info;
extern EventManager g_event_manager;
extern Fabric g_fabric;
extern int64_t g_cache_line_size;

// Common timestamp, use this shorter name
using SharpTimestamp = std::chrono::time_point<std::chrono::system_clock>;

// Linux error codes range from 1 to 133
constexpr int g_retry_parse_file = 134;

class SharpAggMgr
{
    static int ms_exit_code_;
    static bool ms_stop_;

   public:
    static void ExitAggMgr(int exit_code);
    static bool IsStopped() { return ms_stop_; }
    static void SetStop() { ms_stop_ = true; }   // stop without sending signal
    static int GetExitCode() { return ms_exit_code_; }
};

enum class FileMode : uint8_t
{
    WRITE,
    READ
};

namespace AMCommon
{
std::string GetDumpFilePath(char const* const dump_file_name);
void FlushAndCloseFile(FILE* f, const char* file_name);
std::unique_ptr<FILE, std::function<void(FILE*)>> OpenFile(const std::string& file_path, const FileMode file_mode);
void RetrieveCacheLineSize();   // Gets the value from the system
int64_t GetCacheLineSize();     // Gets the saves value, need to be used after retrieving from the system
int ParseGuidFile(std::set<uint64_t>& guids, std::string& guid_file);
bool FailoverEnabled();
};   // namespace AMCommon

struct AggregatedLogMessage
{
    AggregatedLogMessage() = default;
    AggregatedLogMessage(const std::size_t reserve_size) { m_log_message_.reserve(reserve_size); }

    void Clear()
    {
        m_log_message_.clear();
        m_number_of_elements = 0;
    }

    void AddElement(char const* const new_element)
    {
        if (!m_log_message_.empty()) {
            m_log_message_ += ", ";
        }
        m_log_message_ += new_element;
        ++m_number_of_elements;
    }
    void AddElement(const std::string& new_element) { AddElement(new_element.c_str()); }

    void AddCounterElement(char const* const counter_msg, const std::size_t counter)
    {
        if (0 == counter) {
            return;
        }
        if (!m_log_message_.empty()) {
            m_log_message_ += ", ";
        }
        m_log_message_ += counter_msg + std::to_string(counter);
        ++m_number_of_elements;
    }

    char const* GetLogMessage(char const* const default_msg = "") const
    {
        if (0 == m_number_of_elements) {
            return default_msg;
        }
        return m_log_message_.c_str();
    }

    std::size_t GetSize() const { return m_number_of_elements; }
    bool IsEmpty() const { return 0 == m_number_of_elements; }

    std::string m_log_message_;
    std::size_t m_number_of_elements = 0;
};

template <typename EnumType, typename VectorType = std::vector<std::pair<std::string, EnumType>>>
static bool ConvertStringToEnum(const std::string& field_str, const VectorType& str_to_enum_vec, EnumType& field)
{
    const auto vec_it =
        std::find_if(str_to_enum_vec.begin(),
                     str_to_enum_vec.end(),
                     [&field_str](const typename VectorType::value_type& str_to_enum_pair) { return str_to_enum_pair.first == field_str; });
    if (vec_it == str_to_enum_vec.end()) {
        return false;
    }
    field = vec_it->second;
    return true;
};

template <typename EnumType, typename VectorType = std::vector<std::pair<std::string, EnumType>>>
static std::string ConvertEnumToString(const EnumType enum_value,
                                       const std::vector<std::pair<std::string, EnumType>>& str_to_enum_vec,
                                       char const* const default_str = "UNKNOWN")
{
    const auto vec_it = std::find_if(str_to_enum_vec.begin(),
                                     str_to_enum_vec.end(),
                                     [enum_value](const typename VectorType::value_type& str_to_enum_pair)
                                     { return str_to_enum_pair.second == enum_value; });
    if (vec_it != str_to_enum_vec.end()) {
        return vec_it->first;
    }
    return default_str;
};

void RemoveWhitespaceCharacters(std::string& str);

namespace hash_utils
{
template <typename ContainerType>
uint64_t XorHashCombine(const ContainerType& elements,
                        std::function<uint64_t(const typename ContainerType::value_type&)> get_id_from_element_cb)
{
    if (elements.empty()) {
        return 0;
    }

    uint64_t hash_result{1};
    for (const auto& current_element : elements) {
        auto hash2 = get_id_from_element_cb(current_element);
        auto tmp = hash_result * hash2;
        hash_result = (hash_result ^ hash2) + (hash_result << (sizeof(std::size_t) - 8)) + (hash_result >> 2) + (hash2 << 8) +
                      (hash2 >> 5) + (tmp << 48) + ((tmp * tmp) << 32) + (tmp >> 32);
    }
    return hash_result;
}
}   // namespace hash_utils

template <typename NumberType, std::size_t FillSize = 16>
std::string NumberToHexStr(const NumberType num, const bool should_fill_zeroes = true)
{
    static_assert(!std::is_same<uint8_t, NumberType>::value,
                  "Do not call this method with uint8_t because stringstream will treat it like a char, use uint16_t instead!");

    std::stringstream ss{};
    if (should_fill_zeroes) {
        ss << "0x" << std::setfill('0') << std::setw(FillSize) << std::hex << num;
    } else {
        ss << "0x" << std::setfill('0') << std::hex << num;
    }
    return ss.str();
}

std::string TimeStampToText(const SharpTimestamp& timestamp);
uint64_t HexStrToNumber(const std::string& hex_str);

template <typename NumberType>
NumberType MegaBytesToBytes(const NumberType mb)
{
    return mb * 1024 * 1024;
}

template <typename DeviceType, typename ContainerType, typename KeyType = uint64_t>
DeviceType* FindAndGetUniquePointerInMap(const ContainerType& map_container, const KeyType& key)
{
    const auto find_result = map_container.find(key);
    if (find_result == map_container.end()) {
        return nullptr;
    }
    return find_result->second.get();
}

template <typename MapType, typename KeyType, typename ValueType, typename... ValueParameters>
ValueType* AddOrReplaceUniquePtrToMap(char const* const type_str,
                                      MapType& map,
                                      const KeyType key,
                                      const ValueParameters&... value_parameters)
{
    try {
        auto insertion_pair = map.insert(make_pair<KeyType, std::unique_ptr<ValueType>>(KeyType{key}, nullptr));
        if (!insertion_pair.second) {
            // Key already exists in map
            VERBOSE("Replacing %s with key: %s", type_str, std::to_string(key).c_str());
        }
        insertion_pair.first->second = std::unique_ptr<ValueType>(new ValueType(value_parameters...));
        auto* p_node = insertion_pair.first->second.get();
        return p_node;
    } catch (const std::exception&) {
        ERROR("Failed to create %s with key: %s", type_str, std::to_string(key).c_str());
        throw;
    }
}

template <class Callback>
std::unique_ptr<void, typename std::decay<Callback>::type> ScopeGuard(Callback&& callback)
{
    return std::unique_ptr<void, typename std::decay<Callback>::type>{(void*)1, std::forward<Callback>(callback)};
}

// For hash (unordered set/map) of a pair
// Use this as the hashing function when using pairs for unordered containers
struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const
    {
        std::size_t hash1 = std::hash<T1>()(pair.first);
        std::size_t hash2 = std::hash<T2>()(pair.second);
        std::size_t tmp = hash1 * hash2;
        std::size_t hash_result = (hash1 ^ hash2) + (hash1 << (sizeof(std::size_t) - 8)) + (hash1 >> 2) + (hash2 << 8) + (hash2 >> 5) +
                                  (tmp << 48) + ((tmp * tmp) << 32) + (tmp >> 32);
        return hash_result;
    }
};
