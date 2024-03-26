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

#include "agg_types.h"
#include "buffer_calc.h"
#include "common/sharp_common.h"
#include "smx/smx_types.h"
#include "sub_tree_info.h"

extern const std::size_t g_max_log_message_size;

struct JobSubTreeScore
{
    uint8_t m_max_radix;
    uint8_t m_min_radix;
    uint8_t m_height;
    uint16_t m_num_ans;
    uint16_t m_max_load;               // max load on AN
    uint32_t m_total_load;             // load sum of all ANs
    uint32_t m_path_total_sat_load;    // SAT load sum of all AggPaths
    uint32_t m_total_sat_load;         // SAT load sum of all ANs
    uint16_t m_max_sat_load;           // max SAT load on AN
    uint16_t m_max_tree_id_sat_load;   // max sat load on tree ID
    uint16_t m_root_sat_load;          // SAT load of the root (not just of the tree, but absolute root)
    uint16_t m_mutual_ans;             // number of mutual ans in multi tree job
    uint16_t m_data_path_version;

    sharp_quota m_qota;
    sharp_quota m_abs_qota;   // use requested absolute values

    sharp_percent_t m_quota_percent;
    sharp_percent_t m_qps_percent;
    uint8_t m_job_priority;

    uint8_t m_child_index_per_port;
    bool m_multicast_enabled;
    bool m_allocate_sat;
    bool m_check_sat;
    bool m_check_rmc;
    bool m_reproducibility_enabled;
    bool m_exclusive_lock_available;
    bool m_fp19_supported;
    bool m_bfloat19_supported;
    bool m_rmc_supported;
    bool m_extended_data_types_supported;
    uint16_t m_agg_rate;
    SharpMtu m_mtu;
    sharp_resource_priority m_priority;
    uint64_t m_priority_epoch;
    uint8_t m_number_of_possible_links_to_use_in_job;   // used in Quasi fat tree algorithm in dynamic trees

    JobSubTreeScore();

    template <class quota_t>
    void UpdateQuotaVal(quota_t& quota_val,
                        quota_t abs_val,
                        quota_t available_quota,
                        quota_t total_quota,
                        sharp_percent_t default_percent,
                        sharp_percent_t max_percent);

    void Reset(const JobResource& job_resource, sharp_resource_priority priority);

    string ToString() const;

    bool IsWorseThan(const JobSubTreeScore& job_sub_tree_score, JobSubTreeInfo& job_sub_tree_info) const;

    int Update(TreeNode& tree_node, bool is_root, sharp_trees_t sat_tree_id, JobSubTreeInfo& sub_tree_info);

   private:
    int CalculateQuota(uint16_t radix,
                       uint8_t ports,
                       uint16_t max_allowed_radix,
                       uint32_t available_buffers,
                       bool enable_reproducibility,
                       const AggNodeInfo& agg_node_info);

    int SetOstsByBuffers(uint16_t radix,
                         uint16_t max_allowed_radix,
                         unsigned int user_data_length,
                         bool enable_reproducibility,
                         const BufferCalculationParameters& buff_calc_param,
                         uint32_t available_buffers);
};
