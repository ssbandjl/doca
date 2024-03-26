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

#ifndef AN_CONFIG_MANAGER_H_
#define AN_CONFIG_MANAGER_H_

#include <algorithm>
#include <functional>
#include "fabric_provider.h"
#include "port_data.h"

class FabricGraph;

class AnPortResourcesRecord
{
   public:
    u_int64_t node_guid;
    phys_port_t port_num;
    u_int16_t requester_packets;
    u_int16_t requester_buffer_cap;
    u_int16_t responder_packets;
    u_int16_t responder_buffer_cap;
    u_int8_t num_flows;

    AnPortResourcesRecord()
        : node_guid(0),
          port_num(0),
          requester_packets(0),
          requester_buffer_cap(0),
          responder_packets(0),
          responder_buffer_cap(0),
          num_flows(0)
    {}

    AnPortResourcesRecord(u_int64_t node_guid)
        : node_guid(node_guid),
          port_num(0),
          requester_packets(0),
          requester_buffer_cap(0),
          responder_packets(0),
          responder_buffer_cap(0),
          num_flows(0)
    {}

    static int Init(vector<ParseFieldInfo<class AnPortResourcesRecord>>& parse_section_info);

    bool SetNodeGuid(const char* field_str) { return CsvParser::Parse(field_str, node_guid, 0); }

    bool SetPortNum(const char* field_str) { return CsvParser::Parse(field_str, port_num, 0); }

    bool SetRequesterPackets(const char* field_str) { return CsvParser::Parse(field_str, requester_packets, 0); }

    bool SetRequesterBufferCap(const char* field_str) { return CsvParser::Parse(field_str, requester_buffer_cap, 0); }

    bool SetResponderPackets(const char* field_str) { return CsvParser::Parse(field_str, responder_packets, 0); }

    bool SetResponderBufferCap(const char* field_str) { return CsvParser::Parse(field_str, responder_buffer_cap, 0); }

    bool SetNumFlows(const char* field_str) { return CsvParser::Parse(field_str, num_flows, 0); }

    bool operator<(const AnPortResourcesRecord& rhs) const
    {
        if (node_guid < rhs.node_guid)
            return true;

        if (node_guid > rhs.node_guid)
            return false;

        if (port_num == 0xff && rhs.port_num != 0xff)
            return true;

        if (port_num != 0xff && rhs.port_num == 0xff)
            return false;

        if (port_num < rhs.port_num)
            return true;

        if (port_num > rhs.port_num)
            return false;

        return false;
    }

    static bool GuidCompare(const AnPortResourcesRecord& lhs, const AnPortResourcesRecord& rhs) { return lhs.node_guid < rhs.node_guid; }
};

class AnConfigManager
{
    FabricGraph* m_fabric_graph_;
    FabricProvider* m_fabric_provider_;
    AnPortResourcesRecord* m_default_port_resource_record_;
    vector<AnPortResourcesRecord> m_port_resource_records_;

   public:
    AnConfigManager() : m_fabric_graph_(NULL), m_fabric_provider_(NULL), m_default_port_resource_record_(NULL){};

    int Init(FabricGraph* fabric_graph, FabricProvider* fabric_provider);

    int ConfigureDevice(Port* p_port);
    void CompareDeviceConfigurationWithConf(Port* p_port, int* p_operation_status);

    void ConfigureDeviceCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);
    void GetAnPortsConfigurationCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    // Provide a way to execute a callback on all port_resource_records entities
    inline void ExecCallbackOnPortResourceRecords(const std::function<void(const AnPortResourcesRecord&)>& callback)
    {
        std::for_each(m_port_resource_records_.begin(), m_port_resource_records_.end(), callback);
    }

   private:
    int Load(string file_name);
};

#endif   // AN_CONFIG_MANAGER_H_
