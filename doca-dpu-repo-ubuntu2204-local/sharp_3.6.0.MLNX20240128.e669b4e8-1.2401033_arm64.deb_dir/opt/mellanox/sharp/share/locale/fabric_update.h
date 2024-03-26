/*
 * Copyright (c) 2004-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under the terms of the
 * OpenIB.org BSD license included below:
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

#ifndef FABRIC_UPDATE_H_
#define FABRIC_UPDATE_H_

#include "agg_types.h"
#include "port_data.h"

struct PortDataUpdate;
struct PortUpdateInfo;
struct PathUpdate;
class Fabric;

struct PortDataUpdateSort
{
    inline bool operator()(const PortDataUpdate& lhs, const PortDataUpdate& rhs) const;
};

typedef std::set<PortDataUpdate, PortDataUpdateSort> SetPortDataUpdate;
typedef std::list<PortDataUpdate> ListPortDataUpdate;
typedef std::list<PortUpdateInfo> ListPortUpdateInfo;
typedef std::list<PathUpdate> ListPathUpdate;

enum FabricUpdateType
{
    FABRIC_UPDATE_TYPE_NEW = 0,
    FABRIC_UPDATE_TYPE_UPDATE = 1,
    FABRIC_UPDATE_TYPE_BECOME_ACTIVE = 2,
    FABRIC_UPDATE_TYPE_BECOME_INACTIVE = 3,
    FABRIC_UPDATE_TYPE_REROUTE = 4
};

static inline const char* FabricUpdateType2Char(const FabricUpdateType update_type)
{
    switch (update_type) {
        case FABRIC_UPDATE_TYPE_NEW:
            return ("created");
        case FABRIC_UPDATE_TYPE_UPDATE:
            return ("updated");
        case FABRIC_UPDATE_TYPE_BECOME_ACTIVE:
            return ("become active");
        case FABRIC_UPDATE_TYPE_BECOME_INACTIVE:
            return ("become inactive");
        case FABRIC_UPDATE_TYPE_REROUTE:
            return ("reroute");
        default:
            return ("invalid");
    }
};

// PortDataUpdate is calculated on FabricGraph side and converted to
// PortUpdateInfo to be accessed on Fabric side
struct PortDataUpdate
{
    PortData* m_port_data;
    const string m_host_name;
    FabricUpdateType m_update_type;
    const AnMinHopsTable* m_an_min_hops_table;   // for DFP
    const string m_msg;                          // Used to pass 'reason' for port update INACTIVE

    PortDataUpdate(PortData* port_data, FabricUpdateType update_type, const AnMinHopsTable* an_min_hops_table = NULL, const string msg = "")
        : m_port_data(port_data), m_host_name(""), m_update_type(update_type), m_an_min_hops_table(an_min_hops_table), m_msg(msg)
    {}

    PortDataUpdate(PortData* port_data, const string& host_name, FabricUpdateType update_type, const string msg = "")
        : m_port_data(port_data), m_host_name(host_name), m_update_type(update_type), m_an_min_hops_table(NULL), m_msg(msg)
    {}
};

struct PortUpdateInfo
{
    FabricUpdateType m_update_type;
    PortInfo m_port_info;
    PortInfo m_phy_port_info;
    PortType m_port_type;
    // Compute
    const string m_host_name;
    // AggNode
    AggNodeInfo m_agg_node_info;
    sharp_an_id_t m_an_id;
    AnMinHopsTable m_an_min_hops_table;   // for DFP
    uint64_t m_am_key;
    string m_switch_desc;
    const string m_msg;   // Used to pass 'reason' for port update INACTIVE

    // PortUpdateInfo() : m_port_type(CA_PORT_TYPE_UNKNOWN), m_an_id(0) {}

    PortUpdateInfo(const PortDataUpdate& port_update_data);
};

struct PathUpdate
{
    FabricUpdateType m_update_type;
    sharp_path_id_t m_path_id;
    sharp_an_id_t m_down_an_id;
    sharp_an_id_t m_up_an_id;
    uint8_t m_sw_hops;   // hops between AN switches
    uint8_t m_num_semaphores;
    uint16_t m_agg_rate;

    // AggPath QP allocation will be called from both sides(2 ports that are connected by the AggPath).
    // Therefore the amount of QPs should be 2 * min(max_port1_qps, max_port2_qps)
    uint32_t m_sat_qps;
    uint32_t m_llt_qps;
    AnToAnInfo m_an_to_an_info;

    PathUpdate(FabricUpdateType update_type,
               sharp_path_id_t path_id,
               sharp_an_id_t down_an_id,
               sharp_an_id_t up_an_id,
               uint8_t sw_hops,
               uint8_t num_semaphores = 0,
               uint16_t agg_rate = 0,
               uint32_t sat_qps = 0,
               uint32_t llt_qps = 0,
               const AnToAnInfo* an_to_an_info = NULL)
        : m_update_type(update_type),
          m_path_id(path_id),
          m_down_an_id(down_an_id),
          m_up_an_id(up_an_id),
          m_sw_hops(sw_hops),
          m_num_semaphores(num_semaphores),
          m_agg_rate(agg_rate),
          m_sat_qps(sat_qps),
          m_llt_qps(llt_qps)
    {
        if (an_to_an_info) {
            m_an_to_an_info = *an_to_an_info;
        }
    }
};

class FabricUpdateList
{
    uint64_t m_epoch_;
    port_key_t m_local_port_key_update_;
    FabricTopologyInfo m_topology_info_;
    ListPortUpdateInfo m_ports_update_;
    ListPathUpdate m_paths_update_;
    pthread_mutex_t m_list_lock_;

   public:
    explicit FabricUpdateList() : m_epoch_(0), m_local_port_key_update_(0), m_topology_info_(TOPOLOGY_TYPE_NONE)
    {
        pthread_mutex_init(&m_list_lock_, NULL);
    }

    ~FabricUpdateList(){};

    void AddUpdates(const FabricTopologyInfo& topology_info,
                    ListPortDataUpdate& ports_update,
                    ListPathUpdate& paths_update,
                    port_key_t port_key,
                    uint64_t epoch);
    void LogPortUpdates(const ListPortDataUpdate& ports_update);
    // void AddUpdates(const SetPortDataUpdate &updates, uint64_t epoch);
    // void AddUpdates(const ListPathUpdate &updates, uint64_t epoch);
    void HandleUpdates();
};

bool PortDataUpdateSort::operator()(const PortDataUpdate& lhs, const PortDataUpdate& rhs) const
{
    return (lhs.m_port_data->GetPortInfo().m_port_key < rhs.m_port_data->GetPortInfo().m_port_key);
}

#endif   // FABRIC_UPDATE_H_
