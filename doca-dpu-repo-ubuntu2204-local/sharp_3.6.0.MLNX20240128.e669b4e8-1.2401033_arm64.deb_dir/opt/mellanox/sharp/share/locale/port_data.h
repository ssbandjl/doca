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

#ifndef PORT_DATA_H_
#define PORT_DATA_H_

#include <memory>
#include "agg_node.h"
#include "agg_types.h"

class PortData;
class HostInfo;

struct PortDataSort
{
    inline bool operator()(const PortData* p_lhs, const PortData* p_rhs) const;
};

using SetPortDataConstPtr = std::set<PortData const*, PortDataSort>;
using VecPortDataConstPtr = std::vector<PortData const*>;
using VecPortDataPtr = std::vector<std::unique_ptr<PortData>>;
using MapPortDataUniquePtr = std::map<port_key_t, std::unique_ptr<PortData>>;
using MapPortDataPtr = std::map<port_key_t, PortData*>;

enum PortType
{
    CA_PORT_TYPE_UNKNOWN = 0,
    CA_PORT_TYPE_COMPUTE = 1,
    CA_PORT_TYPE_AGG_NODE = 2,
    CA_PORT_TYPE_UNSUPPORTED_AGG_NODE = 3,
    CA_VPORT_TYPE = 4,
    CA_PORT_TYPE_OTHER = 5
};

static inline const char* CaPortType2Char(const PortType port_type)
{
    switch (port_type) {
        case CA_PORT_TYPE_UNKNOWN:
            return ("UNKNOWN");
        case CA_PORT_TYPE_COMPUTE:
            return ("COMPUTE");
        case CA_PORT_TYPE_AGG_NODE:
            return ("AN");
        case CA_PORT_TYPE_UNSUPPORTED_AGG_NODE:
            return ("UNSUPPORTED_AN");
        case CA_VPORT_TYPE:
            return ("VIRTUAL");
        default:
            return ("INVALID");
    }
};

class PortData
{
   protected:
    PortInfo m_port_info_;
    uint64_t m_epoch_;
    PortType m_port_type_;
    std::unique_ptr<AggNode> m_agg_node_;
    HostInfo* m_host_info_;
    bool m_is_enabled_ = true;

   public:
    PortData(const PortInfo& port_info, uint64_t epoch, const PortType port_type = CA_PORT_TYPE_UNKNOWN)
        : m_port_info_(port_info),
          m_epoch_(epoch),
          m_port_type_(port_type),
          m_agg_node_(nullptr),
          m_host_info_(nullptr),
          m_is_enabled_{true}
    {}

    virtual ~PortData() = default;
    PortData(const PortData&) = delete;
    PortData& operator=(const PortData&) = delete;
    PortData(PortData&&) = default;
    PortData& operator=(PortData&&) = default;

    void SetType(PortType port_type) { m_port_type_ = port_type; }
    PortType GetType() const { return m_port_type_; }

    inline void Enable() { m_is_enabled_ = true; }
    inline void Disable() { m_is_enabled_ = false; }
    inline bool IsEnabled() const { return m_is_enabled_; }
    inline bool IsDisabled() const { return !m_is_enabled_; }

    const PortInfo& GetPortInfo() const { return m_port_info_; }
    virtual const PortInfo& GetPhysPortInfo() const;
    void SetPortInfo(const PortInfo& port_info) { m_port_info_ = port_info; }

    void SetTimestamp(const PortTimestamp& timestamp) { m_port_info_.m_timestamp = timestamp; }

    void SetPortInfoPeerKey(const port_key_t& m_peer_key) { m_port_info_.m_peer_key = m_peer_key; }

    void SetPortInfoLid(const lid_t& lid) { m_port_info_.m_lid = lid; }

    void SetAggNode(AggNode* p_agg_node)
    {
        m_agg_node_.reset(p_agg_node);
        m_port_type_ = CA_PORT_TYPE_AGG_NODE;
    }
    AggNode* GetAggNode() { return m_agg_node_.get(); }

    HostInfo const* GetHostInfo() const { return m_host_info_; }
    HostInfo* GetHostInfo() { return m_host_info_; }
    // sharpd_id_t GetSharpdId() const { return m_host_info_->GetSharpdId(); }

    void SetHostInfo(HostInfo* p_host_info) { m_host_info_ = p_host_info; }

    string ToString() const;
    virtual void Update(const PortInfo& port_info);
};

class VportData : public PortData
{
    PortInfo m_phys_port_info_;
    std::string m_host_name_;

   public:
    VportData(const PortInfo& vport_info, const std::string& host_name, const uint64_t epoch, const PortInfo& phys_port_info)
        : PortData(vport_info, epoch, CA_VPORT_TYPE), m_phys_port_info_(phys_port_info), m_host_name_{host_name} {};
    virtual ~VportData() = default;

    virtual const PortInfo& GetPhysPortInfo() const;
    std::string GetHostName() const;
    virtual void Update(const PortInfo& vport_info);
};

bool PortDataSort::operator()(const PortData* p_lhs, const PortData* p_rhs) const
{
    return (p_lhs->GetPortInfo().m_port_key < p_rhs->GetPortInfo().m_port_key);
}

#endif   // PORT_DATA_H_
