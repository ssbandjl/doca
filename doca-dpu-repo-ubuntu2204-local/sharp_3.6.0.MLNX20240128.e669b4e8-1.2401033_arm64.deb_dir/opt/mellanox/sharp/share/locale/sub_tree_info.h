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
#include "reserved_string.h"
#include "smx/smx_types.h"

using SharpTreeConnVec = std::vector<sharp_tree_conn>;

extern const std::size_t g_tree_dump_message_size;

struct JobSubTreeInfo
{
    ListTreeNodePtr m_tree_nodes;
    SharpTreeConnVec m_tree_connections;
    sharp_tree m_tree_info;
    AggNodeFabricInfo* m_tree_root;
    string m_log_message;
    sharp_job_id_t m_sharp_job_id;

    void Clear();

    void UpdateLogMessage(ReservedString& message, int log_level);
    void SetExclusiveLock(bool exclusive_lock);
    void SetDataPathVersion(uint16_t data_path_version_bit_mask);
    void SetExtendedDataTypes(bool extended_data_types_supported);

    int SetPkey(uint16_t pkey) const;
    void GetSubTreeAggNodes(SetAggNodeFabricPtr& agg_nodes) const;
    uint16_t GetMutualAnsNumber(const SetAggNodeFabricPtr& prev_agg_nodes) const;

    void SetSrcDstPr(ibv_sa_path_rec& path_rec, const PortInfo& src, const PortInfo& dst);

    void SetConnectionPathRec(sharp_tree_conn& connection, const PortInfo& an, const PortInfo& sd, uint16_t pkey, SharpMtu mtu);

    int UpdateTreeConnections(uint32_t connections_number, uint8_t child_index_per_port, uint16_t pkey);

    void ModifyANsAvailableTreeIDs(bool is_available);
    void ModifyANsAvailableTreeIDsLowPriority(bool is_available);
};
