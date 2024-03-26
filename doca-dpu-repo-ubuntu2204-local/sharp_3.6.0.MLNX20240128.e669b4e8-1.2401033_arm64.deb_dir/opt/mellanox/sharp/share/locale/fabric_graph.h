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

#ifndef AGG_FABRIC_GRAPH_H_
#define AGG_FABRIC_GRAPH_H_

#include <functional>
#include <memory>
#include <unordered_map>

#include "agg_ib_types.h"
#include "agg_types.h"
#include "am_common.h"
#include "amkey_manager.h"
#include "an_config_manager.h"
#include "fabric_graph_update.h"
#include "fabric_update.h"
#include "option_manager.h"
#include "port_data.h"

class Vnode;
class Vport;

using ListOfNodes = std::list<class Node*>;
using ListPortPtr = std::list<class Port*>;

using VectorOfPorts = std::vector<Port*>;
using VectorOfUniquePorts = std::vector<std::unique_ptr<Port>>;
using VectorsOfListOfNodes = std::vector<ListOfNodes>;
using SetNodesPtr = std::set<class Node*>;
using SetPortsPtr = std::set<class Port*>;
using SetOfGuids = std::set<uint64_t>;

using MapGuidToNodePtr = std::map<uint64_t, std::unique_ptr<Node>>;
using MapGuidToVnodePtr = std::map<uint64_t, std::unique_ptr<Vnode>>;
using MapPortKeyToAnPortPtr = std::map<port_key_t, class Port*>;
using MapStrToListOfNodes = std::map<string, ListOfNodes>;
using MapGuidToHCCoordinates = std::map<uint64_t, uint16_t>;
using MapPortGuidToNodeGuidAndPortIndex = std::map<uint64_t, std::pair<uint64_t, std::size_t>>;
using MapGuidToVportPtr = std::map<uint64_t, std::unique_ptr<Vport>>;
using VectorOfNodes = std::vector<class Node*>;
using VectorOfVports = std::vector<class Vport*>;

// Used for hash set of invalid ports
using PairNodeGuidPortNum = std::pair<u_int64_t, phys_port_t>;
using HashMapNodeGuidPortNumToStr = std::unordered_map<PairNodeGuidPortNum, char const* const, pair_hash>;

enum class PortStatus : uint8_t
{
    ISOLATED,
    VALID
};
extern std::vector<std::pair<std::string, PortStatus>> g_port_status_str_to_enum;
enum class SmdbPortState : uint8_t
{
    NOC,
    DOWN,
    INIT,
    ARM,
    ACTIVE
};
extern std::vector<std::pair<std::string, SmdbPortState>> g_port_state_str_to_enum;
enum class DiscThroughState : uint8_t
{
    NORMAL,
    IGNORED
};
extern std::vector<std::pair<std::string, DiscThroughState>> g_disc_through_state_str_to_enum;

enum SpecialPortType
{
    SPECIAL_PORT_TYPE_AN = 1,
    SPECIAL_PORT_TYPE_ROUTER = 2,
    SPECIAL_PORT_TYPE_ETH_GW = 3,
    SPECIAL_PORT_TYPE_UNKNOWN = 0xFF,
};

class FabricDbException : public std::exception
{
};

class SmRecord
{
   public:
    string version;
    u_int64_t pid;
    string host_name;
    u_int64_t port_guid;
    u_int64_t subnet_prefix;
    u_int64_t lid;
    string routing_engine;

    SmRecord() : version(""), pid(0), host_name(""), port_guid(0), subnet_prefix(0), lid(0), routing_engine("") {}

    static int Init(vector<ParseFieldInfo<class SmRecord>>& parse_section_info);

    bool SetSmVersion(const char* field_str) { return CsvParser::Parse(field_str, version); }

    bool SetSmPid(const char* field_str) { return CsvParser::Parse(field_str, pid, 10); }

    bool SetSmHostName(const char* field_str) { return CsvParser::Parse(field_str, host_name); }

    bool SetSmPortGuid(const char* field_str) { return CsvParser::Parse(field_str, port_guid, 16); }

    bool SetSmSubnetPrefix(const char* field_str) { return CsvParser::Parse(field_str, subnet_prefix, 0); }

    bool SetSmlid(const char* field_str) { return CsvParser::Parse(field_str, lid, 10); }

    bool SetSmRoutingEngine(const char* field_str) { return CsvParser::Parse(field_str, routing_engine); }
};

class NodeRecord
{
   public:
    u_int64_t node_guid;
    string node_description;
    u_int16_t num_ports;
    u_int8_t node_type;
    //    u_int8_t            class_version
    //    u_int8_t            base_version
    //    u_int16_t           device_id
    //    u_int8_t            local_port_num

    NodeRecord() : node_guid(0), node_description(""), num_ports(0), node_type(0) {}

    static int Init(vector<ParseFieldInfo<class NodeRecord>>& parse_section_info);

    bool SetNodeGuid(const char* field_str) { return CsvParser::Parse(field_str, node_guid, 16); }

    bool SetNodeDescription(const char* field_str)
    {
        const auto parser_result = CsvParser::Parse(field_str, node_description);
        RemoveWhitespaceCharacters(node_description);
        return parser_result;
    }

    bool SetNumPorts(const char* field_str) { return CsvParser::Parse(field_str, num_ports, 10); }

    bool SetNodeType(const char* field_str) { return CsvParser::Parse(field_str, node_type, 10); }
};

class PortRecord
{
   public:
    u_int64_t node_guid;
    phys_port_t port_num;
    u_int64_t port_guid;
    u_int16_t lid;
    SmdbPortState state;
    u_int16_t link_width;
    u_int16_t link_speed;
    PortStatus status;
    DiscThroughState disc_through_state;
    string timestamp;
    u_int8_t special_port_type;

    PortRecord()
        : node_guid(0),
          port_num(0),
          port_guid(0),
          lid(0),
          state{SmdbPortState::ACTIVE},
          link_width(0),
          link_speed(0),
          status{PortStatus::VALID},
          disc_through_state{DiscThroughState::NORMAL},
          timestamp{},
          special_port_type(SPECIAL_PORT_TYPE_UNKNOWN)
    {}

    static int Init(vector<ParseFieldInfo<class PortRecord>>& parse_section_info);

    bool SetNodeGuid(const char* field_str) { return CsvParser::Parse(field_str, node_guid, 16); }

    bool SetPortNum(const char* field_str) { return CsvParser::Parse(field_str, port_num, 10); }

    bool SetPortGuid(const char* field_str) { return CsvParser::Parse(field_str, port_guid, 16); }

    bool SetLID(const char* field_str) { return CsvParser::Parse(field_str, lid, 10); }

    bool SetLinkWidth(const char* field_str)
    {
        string str;
        int num = 0;
        CsvParser::Parse(field_str, str);
        int ret = std::sscanf(str.c_str(), "%dx", &num);
        link_width = uint16_t(num);
        return ret ? true : false;
    }

    bool SetLinkSpeed(const char* field_str) { return CsvParser::Parse(field_str, link_speed, 10); }

    bool SetPortState(const char* field_str);

    bool SetStatus(const char* field_str);

    bool SetDiscThroughState(const char* field_str);

    bool SetTimestamp(const char* field_str) { return CsvParser::Parse(field_str, timestamp); }

    bool SetSpecialPortType(const char* field_str) { return CsvParser::Parse(field_str, special_port_type, 0); }
};

class LinkRecord
{
   public:
    u_int64_t node_guid1;
    u_int64_t node_guid2;
    phys_port_t port_num1;
    phys_port_t port_num2;

    LinkRecord() : node_guid1(0), node_guid2(0), port_num1(0), port_num2(0) {}

    static int Init(vector<ParseFieldInfo<class LinkRecord>>& parse_section_info);

    bool SetNodeGuid1(const char* field_str) { return CsvParser::Parse(field_str, node_guid1, 16); }

    bool SetNodeGuid2(const char* field_str) { return CsvParser::Parse(field_str, node_guid2, 16); }

    bool SetPortNum1(const char* field_str) { return CsvParser::Parse(field_str, port_num1, 10); }

    bool SetPortNum2(const char* field_str) { return CsvParser::Parse(field_str, port_num2, 10); }
};

class SwitchRecord
{
   public:
    u_int64_t node_guid;
    u_int16_t num_ports;
    u_int8_t status;
    u_int8_t rank;
    u_int16_t coordinate;

    SwitchRecord() : node_guid(0), num_ports(0), status(0), rank(0), coordinate(0) {}

    static int Init(vector<ParseFieldInfo<class SwitchRecord>>& parse_section_info);

    bool SetNodeGuid(const char* field_str) { return CsvParser::Parse(field_str, node_guid, 16); }

    bool SetNumPorts(const char* field_str) { return CsvParser::Parse(field_str, num_ports, 10); }

    bool SetStatus(const char* field_str) { return CsvParser::Parse(field_str, status, 10); }

    bool SetRank(const char* field_str) { return CsvParser::Parse(field_str, rank, 10); }

    bool SetCoordinate(const char* field_str) { return CsvParser::Parse(field_str, coordinate, 16); }
};

class AnToAnRecord
{
   public:
    u_int64_t node_guid1;
    u_int64_t node_guid2;
    phys_port_t port_num1;
    phys_port_t port_num2;
    string timestamp;

    AnToAnRecord() : node_guid1(0), node_guid2(0), port_num1(0), port_num2(0) {}

    static int Init(vector<ParseFieldInfo<class AnToAnRecord>>& parse_section_info);

    bool SetNodeGuid1(const char* field_str) { return CsvParser::Parse(field_str, node_guid1, 16); }

    bool SetNodeGuid2(const char* field_str) { return CsvParser::Parse(field_str, node_guid2, 16); }

    bool SetPortNum1(const char* field_str) { return CsvParser::Parse(field_str, port_num1, 10); }

    bool SetPortNum2(const char* field_str) { return CsvParser::Parse(field_str, port_num2, 10); }

    bool SetTimestamp(const char* field_str) { return CsvParser::Parse(field_str, timestamp); }
};

class VnodeRecord
{
   public:
    u_int64_t vnode_guid;
    string vnode_description;
    u_int16_t num_ports;

    VnodeRecord() : vnode_guid(0), vnode_description(""), num_ports(0) {}

    static int Init(vector<ParseFieldInfo<class VnodeRecord>>& parse_section_info);

    bool SetVnodeGuid(const char* field_str) { return CsvParser::Parse(field_str, vnode_guid, 16); }

    bool SetVnodeDescription(const char* field_str)
    {
        const auto parser_result = CsvParser::Parse(field_str, vnode_description);
        RemoveWhitespaceCharacters(vnode_description);
        return parser_result;
    }

    bool SetNumPorts(const char* field_str) { return CsvParser::Parse(field_str, num_ports, 10); }
};

class VportRecord
{
   public:
    u_int64_t vport_guid;
    u_int16_t index;
    u_int64_t vnode_guid;
    u_int64_t port_guid;
    phys_port_t vport_num;
    u_int8_t state;
    u_int8_t is_lid_required;
    u_int16_t vport_lid_index;
    u_int16_t vport_lid;
    u_int16_t active_lid;

    VportRecord()
        : vport_guid(0),
          index(0),
          vnode_guid(0),
          port_guid(0),
          vport_num(0),
          state(0),
          is_lid_required(0),
          vport_lid_index(0),
          vport_lid(0),
          active_lid(0)
    {}

    static int Init(vector<ParseFieldInfo<class VportRecord>>& parse_section_info);

    bool SetVportGuid(const char* field_str) { return CsvParser::Parse(field_str, vport_guid, 16); }

    bool SetIndex(const char* field_str) { return CsvParser::Parse(field_str, index, 10); }

    bool SetVnodeGuid(const char* field_str) { return CsvParser::Parse(field_str, vnode_guid, 16); }

    bool SetPortGuid(const char* field_str) { return CsvParser::Parse(field_str, port_guid, 16); }

    bool SetVportNum(const char* field_str) { return CsvParser::Parse(field_str, vport_num, 10); }

    bool SetState(const char* field_str) { return CsvParser::Parse(field_str, state, 10); }

    bool SetIsLidRequired(const char* field_str) { return CsvParser::Parse(field_str, is_lid_required, 10); }

    bool SetVportLidIndex(const char* field_str) { return CsvParser::Parse(field_str, vport_lid_index, 10); }

    bool SetVportLid(const char* field_str) { return CsvParser::Parse(field_str, vport_lid, 10); }

    bool SetActiveLid(const char* field_str) { return CsvParser::Parse(field_str, active_lid, 10); }
};

class SmPortsRecord
{
   public:
    u_int64_t port_guid;
    u_int8_t status;

    SmPortsRecord() : port_guid(0), status(0) {}

    static int Init(vector<ParseFieldInfo<class SmPortsRecord>>& parse_section_info);

    bool SetPortGuid(const char* field_str) { return CsvParser::Parse(field_str, port_guid, 16); }

    bool SetStatus(const char* field_str) { return CsvParser::Parse(field_str, status, 10); }
};

class SmsRecord
{
   public:
    u_int64_t port_guid;
    u_int16_t lid;
    u_int8_t priority;
    u_int8_t state;

    SmsRecord() : port_guid(0), lid(0), priority(0), state(0) {}

    static int Init(vector<ParseFieldInfo<class SmsRecord>>& parse_section_info);

    bool SetPortGuid(const char* field_str) { return CsvParser::Parse(field_str, port_guid, 16); }

    bool SetLID(const char* field_str) { return CsvParser::Parse(field_str, lid, 10); }

    bool SetPriority(const char* field_str) { return CsvParser::Parse(field_str, priority, 10); }

    bool SetState(const char* field_str) { return CsvParser::Parse(field_str, state, 10); }
};

typedef std::map<AnToAnKey, AnToAnInfo> MapAnToAnInfo;
typedef MapAnToAnInfo::iterator MapAnToAnInfoIter;
typedef std::pair<MapAnToAnInfoIter, bool> MapAnToAnInfoInsertRes;

class CommandManager;

enum NodeState
{
    AGG_NODE_NEW = 0,
    AGG_NODE_DISCOVERY,
    AGG_NODE_REDISCOVERY,
    AGG_NODE_AMKEY_SET,
    AGG_NODE_AMKEY_RECOVERY,
    AGG_NODE_CLEAN,
    AGG_NODE_AN_INFO_CONFIGURE,
    AGG_NODE_PORT_CREDITS_CONFIGURE,
    AGG_NODE_READY_TO_BECOME_ACTIVE,
    AGG_NODE_ACTIVE,
    AGG_NODE_ERROR,
};

static inline const char* NodeStateToChar(NodeState state)
{
    switch (state) {
        case AGG_NODE_NEW:
            return ("New");
        case AGG_NODE_DISCOVERY:
            return ("Discovery");
        case AGG_NODE_REDISCOVERY:
            return ("Rediscovery");
        case AGG_NODE_AMKEY_SET:
            return ("AMKey set");
        case AGG_NODE_AMKEY_RECOVERY:
            return ("AMKey recovery");
        case AGG_NODE_CLEAN:
            return ("Clean");
        case AGG_NODE_AN_INFO_CONFIGURE:
            return ("ANInfo configure");
        case AGG_NODE_PORT_CREDITS_CONFIGURE:
            return ("Port credits configure");
        case AGG_NODE_READY_TO_BECOME_ACTIVE:
            return ("Ready to become active");
        case AGG_NODE_ACTIVE:
            return ("Active");
        case AGG_NODE_ERROR:
            return ("Error");
        default:
            return ("Unknown Aggregation Node state");
    }
}

///////////////////////////////////////////////////////////////////////////////
//
// Port class.
//

class Port
{
    uint64_t m_guid_;                       // The port GUID (on SW only on Port0)
    class Port* m_remote_port_;             // Port connected on the other side of link
    class Node* m_node_;                    // The node the port is part of.
    phys_port_t m_number_;                  // Physical ports are identified by number.
    lid_t m_base_lid_;                      // The base lid assigned to the port.
    PortTimestamp m_timestamp_;             // Port update timestamp reported by subnet manager
    SpecialPortType m_special_port_type_;   // Port type reported by subnet manager.
    uint16_t m_port_rate_;

    Epoch m_epoch_;   // last update number

    std::unique_ptr<PortData> m_port_data_;

    uint8_t m_error_count;
    bool m_use_grh_;
    bool m_is_active_;

   public:
    // constructor
    Port(Node* p_node_ptr,
         phys_port_t number,
         uint64_t port_guid,
         lid_t lid,
         const PortTimestamp& timestamp,
         SpecialPortType special_port_type,
         uint16_t port_rate)
        : m_guid_(port_guid),
          m_remote_port_(NULL),
          m_node_(p_node_ptr),
          m_number_(number),
          m_base_lid_(lid),
          m_timestamp_(timestamp),
          m_special_port_type_(special_port_type),
          m_port_rate_(port_rate),
          m_epoch_(),
          m_port_data_(nullptr),
          m_error_count(0),
          m_use_grh_(false),
          m_is_active_{true}
    {}

    Port(const Port&) = delete;
    Port& operator=(const Port&) = delete;
    Port(Port&&) = default;
    Port& operator=(Port&&) = default;

    void SetEpoch(uint64_t epoch) { m_epoch_.SetEpoch(epoch); }

    void RevertEpoch(const uint64_t curr_epoch) { m_epoch_.Revert(curr_epoch); }

    uint64_t GetCurrEpoch() const { return m_epoch_.m_curr_epoch_; }
    uint64_t GetPrevEpoch() const { return m_epoch_.m_prev_epoch_; }
    bool GetUseGRH() const { return m_use_grh_; }
    bool IsActive() const { return m_is_active_; }
    void Activate() { m_is_active_ = true; }
    void Disable() { m_is_active_ = false; }

    // get the port name
    string GetName() const;

    // connect the port to another node port
    // call separately on each port of the connection
    void Connect(Port* p_other_port, uint64_t epoch);

    void UpdatePortInfo(const PortInfo& port_info);

    uint64_t GetGuid() const { return m_guid_; }

    Port* GetRemotePort() const { return m_remote_port_; }

    Node* GetNode() { return m_node_; }

    phys_port_t GetNum() const { return m_number_; }

    void SetBaseLid(lid_t lid) { m_base_lid_ = lid; }
    lid_t GetBaseLid() { return m_base_lid_; }

    SpecialPortType GetSpecialPortType() const { return m_special_port_type_; }
    uint16_t GetPortRate() const { return m_port_rate_; }

    PortData* CreatePortData(const PortInfo& port_info);
    PortData* GetPortData() const { return m_port_data_.get(); }

    const PortTimestamp& GetTimestamp() const { return m_timestamp_; }
    void SetTimestamp(const PortTimestamp& timestamp) { m_timestamp_ = timestamp; }

    uint8_t GetErrorCount() { return m_error_count; }
    void IncMadErrorCount() { m_error_count++; }
    void ResetMadErrorCount() { m_error_count = 0; }

   private:
    // disconnect the port and remote port.
    void Disconnect();
    void DisconnectRemote();
};

///////////////////////////////////////////////////////////////////////////////
//
// Vport class.
//

class Vport
{
    uint64_t m_guid_;        // The port VGUID (on CA only)
    class Vnode* m_vnode_;   // The vnode the vport is part of.
    class Port* m_port_;     // The physical port the vport is part of.
    phys_port_t m_number_;   // vports are identified by number.
    lid_t m_vlid_;           // The base vlid assigned to the vport.
    Epoch m_epoch_;          // last update number
    std::unique_ptr<VportData> m_vport_data_;
    bool m_use_grh_;   // indicates whether use GRH or not.

   public:
    // constructor
    Vport(const phys_port_t number,
          const uint64_t vport_guid,
          const lid_t vlid,
          Vnode* const p_vnode,
          Port* const p_port,
          const bool use_grh = false)
    {
        Reset(number, vport_guid, vlid, p_vnode, p_port, use_grh);
    }

    void Reset(const phys_port_t number,
               const uint64_t vport_guid,
               const lid_t vlid,
               Vnode* const p_vnode,
               Port* const p_port,
               const bool use_grh = false);

    void SetEpoch(uint64_t epoch) { m_epoch_.SetEpoch(epoch); }

    void RevertEpoch(uint64_t curr_epoch) { m_epoch_.Revert(curr_epoch); }

    uint64_t GetCurrEpoch() const { return m_epoch_.m_curr_epoch_; }
    uint64_t GetPrevEpoch() const { return m_epoch_.m_prev_epoch_; }
    bool GetUseGRH() const { return m_use_grh_; }

    // get the port name
    string GetName() const;

    uint64_t GetGuid() const { return m_guid_; }

    lid_t GetVlid() { return m_vlid_; }

    VportData* GetVportData() { return m_vport_data_.get(); }

    Vnode* GetVnode() { return m_vnode_; }
    Vnode const* GetVnode() const { return m_vnode_; }
    Port* GetPhysPort() { return m_port_; }

    phys_port_t GetNum() { return m_number_; }

    VportData* CreateVportData(const PortInfo& vport_info, const std::string& host_name);

    void Dump(ostream& sout) const;
};

///////////////////////////////////////////////////////////////////////////////
//
// Vnode class.
//

class Vnode
{
    uint64_t m_guid_;        // The virtual node GUID
    string m_description_;   // Description of the vnode
    uint64_t m_epoch_;       // last update number

   public:
    // constructor
    Vnode(const uint64_t vport_guid, const string& vnode_description) : m_guid_(vport_guid), m_description_(vnode_description), m_epoch_()
    {}

    inline void SetEpoch(uint64_t epoch) { m_epoch_ = epoch; }
    inline uint64_t GetCurrEpoch() { return m_epoch_; }
    inline const string& GetDescription() const { return m_description_; }
    inline void UpdateDescription(const std::string& new_description) { m_description_ = new_description; }
    inline uint64_t GetGuid() const { return m_guid_; }
};

enum UpDownMinhopDirection
{
    UPDN_MINHOP_DIRECTION_UP = 0,
    UPDN_MINHOP_DIRECTION_DOWN = 1
};

//
// Node class
//
class Node
{
    VectorOfUniquePorts m_ports_;   // Vector of all the ports (in index 0 we will put port0 if exist)
    uint64_t m_guid_;               // Node Guild
    lid_t m_switch_lid_;            // Lid of management port (valid only for IB_SW_NODE)
    NodeType m_type_;               // Either a CA or SW
    string m_description_;          // Description of the node
    bool m_is_active_ = true;

    NodeTopologyInfo m_topology_info_;   // Topology information for this node.

    Port* m_agg_peer_;   // port connected to agg_node;

    MinHopsTable m_min_hops_table_;   // minimal hops number to sw node, key=LID value=hops [0xFF, 0xFF, 1, 0, 0xFF, 2]
    MinHopsIndexTable
        m_min_hops_index_table_;   // contain all keys(LIDs) of min hops table [2,3,5] that have a valid range (not MAX_NUM_HOPS)

    PrivateAppData m_app_data_1;

    uint64_t m_epoch_;   // last update number

    // temp protocol variables
    mutable SetNodesPtr m_adjacent_switch_nodes_;
    mutable uint16_t m_cycle_number_;   // used to check if visited

    NodeState m_state_;   // Indicates state of Node. Used to determine which action is needed next on this node.

   public:
    // Constructor
    Node(const NodeType node_type,
         const phys_port_t number_of_ports,
         const uint64_t node_guid,
         const string& description,
         const TopologyType topology_type);
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) = default;
    Node& operator=(Node&&) = default;

    inline NodeState GetState() const { return m_state_; }
    inline void SetEpoch(uint64_t epoch) { m_epoch_ = epoch; }
    inline uint64_t GetCurrEpoch() const { return m_epoch_; }
    // get the node name
    inline const string& GetDescription() const { return m_description_; }
    inline NodeType GetType() const { return m_type_; }
    void SetSwitchLid(lid_t lid) { m_switch_lid_ = lid; }
    lid_t GetSwitchLid() const { return m_switch_lid_; }
    NodeTopologyInfo& GetTopologyInfo() { return m_topology_info_; }
    inline phys_port_t GetTotalNumberOfPorts() const { return m_ports_.size() - 1; }
    inline phys_port_t GetNumberOfAllocatedPorts() const
    {
        return std::count_if(m_ports_.begin(), m_ports_.end(), [](const std::unique_ptr<Port>& port) { return (nullptr != port); });
    }
    inline uint64_t GetGuid() const { return m_guid_; }
    inline void SetAggPeer(Port* p_agg_peer) { m_agg_peer_ = p_agg_peer; }
    inline void SetAppData1(const uint64_t val) { m_app_data_1.val = val; }
    inline Port* GetAggPeer(Port* p_agg_peer) { return m_agg_peer_; }
    inline const MinHopsTable* GetMinHopsTable() const { return &m_min_hops_table_; }
    inline const PrivateAppData& GetAppData1() const { return m_app_data_1; }
    inline bool IsVisited(uint16_t cycle_number) const { return (cycle_number == m_cycle_number_); }
    inline bool SetVisited(uint16_t cycle_number) const { return (m_cycle_number_ = cycle_number); }
    bool IsActive() const { return m_is_active_; }
    void Activate() { m_is_active_ = true; }
    void Disable() { m_is_active_ = false; }

    void SetState(NodeState state);
    string GetName() const;

    // get or create port, add it to ca_port_by_guid if the node is IB_CA_NODE
    Port* GetOrCreatePort(const phys_port_t port_index,
                          const uint64_t port_guid,
                          const lid_t lid,
                          const PortTimestamp& timestamp,
                          const SpecialPortType special_port_type,
                          const uint16_t port_rate,
                          MapPortGuidToNodeGuidAndPortIndex& ca_port_by_guid);
    // get a port by number num = 1..N:
    Port* GetPort(const phys_port_t num);

    bool IsAnyPort(const std::function<bool(const std::unique_ptr<Port>&)>& condition_callback);
    bool HasAtLeastOneActivePort(const uint64_t epoch);
    bool IsAggregationNode();
    void ExecuteOnPorts(const std::function<void(std::unique_ptr<Port>&)>& execute_callback);

    Port* GetAggValidPeer();
    // Return pointer to aggregation node object connected to this node.
    AggNode* GetValidAggNode();

    void UpdateDownNodesRank(ListOfNodes& nodes_queue);

    void CalculateUpDownNodesMinHopsTable(UpDownMinhopDirection direction);
    bool UpdateAdjacentNodesMinhopsTable();
    void SetNumHops(uint64_t sw_lid, uint8_t hops);
    void ClearMinhopsTable();
    uint8_t GetNumHops(uint64_t to_sw_guid);

    void Dump(ostream& sout, const MapGuidToNodePtr& node_by_guid) const;
    void DumpMinHopsTable(ostream& sout, const MapGuidToNodePtr& node_by_guid) const;

    // temp protocol variables
    void ResetProtocolVariables();

    const SetNodesPtr& GetAdjacentSwitchNodes() const;

   private:
    void DumpDescLine(ostream& sout) const;
    bool UpdatesMinHopsTable(MinHopsTable& adjacent_min_hops_table, MinHopsIndexTable& adjacent_min_hops_index_table);

};   // Class Node

class FabricGraph
{
   protected:
    FabricProvider m_fabric_provider_;
    AMKeyManager m_amkey_manager_;
    CommandManager* m_command_manager_ptr_;

    // Data structures used during smdb/virt file parsing
    MapPortGuidToNodeGuidAndPortIndex m_ca_port_guid_to_node_guid_and_index_;   // map of all fabric ca ports
    MapGuidToNodePtr m_node_by_guid_;                                           // Provides the node by guid
    MapGuidToVnodePtr m_vnode_by_guid_;                                         // Provides the vnode by guid
    VectorsOfListOfNodes m_node_by_rank_;                                       // Provides the node by node rank (rank 0 is root)
    MapGuidToVportPtr m_vport_by_guid_;                                         // set of all fabric vports
    VectorOfPorts m_port_by_lid_;                                               // Pointer to the Port by its lid
    uint8_t m_max_rank_;
    lid_t m_max_lid_;   // Track max lid used.
    SetNodesPtr m_root_nodes_;
    SetNodesPtr m_sw_nodes_;   // Switch nodes.
    std::set<uint16_t> m_coordinates_set;
    uint16_t m_max_dfp_group_;
    uint64_t m_subnet_prefix_;
    SetOfGuids m_ignore_host_guids_;

    uint64_t m_epoch_;   // Serial number of smdb updates
    uint64_t m_vport_epoch_;
    bool m_are_vports_inconsistent_with_physical_ports_;

    // In case AM recovered(AMKey), need to rediscover the fabric regardless of smdb file change.
    bool m_rediscover_required_;

    MapPortKeyToAnPortPtr m_an_port_by_key_;   // AggNode ptr node by node key
    ListAggPathPtr m_paths_;                   // list of all paths in the fabric

    FabricTopologyData m_topology_data_;
    port_key_t m_sm_port_guid_;

    MapAnToAnInfo m_map_an_to_an_info_;   // Routing information of single
                                          // hop path

    // temp update DB
    ListPortDataUpdate m_ports_data_update_;
    ListPathUpdate m_paths_update_;
    port_key_t m_sm_port_guid_update_;
    // SetPortDataUpdate       m_delayed_port_data_updates_;
    MapPortKeyToAnPortPtr m_mad_send_retry_;
    bool m_startup_update_fabric_state_;
    HashMapNodeGuidPortNumToStr m_invalid_ports_hash_;

    bool m_job_handling_started_;
    // control_path_version (IB: active am class version) to be set as  on all ANs
    uint8_t m_control_path_version_;
    uint16_t m_min_tree_table_size_;
    u_int16_t m_data_path_version_;

    sharp_job_id_t m_max_jobs_number_;

    // device configuration manager
    AnConfigManager m_an_config_manager_;

   public:
    // Constructor
    FabricGraph(CommandManager* command_manager_ptr);

    // FabricGraph is a created and owned by FilesBasedFabricDB, it should never be copied
    FabricGraph(const FabricGraph&) = delete;
    FabricGraph& operator=(const FabricGraph&) = delete;

    // Processes states of AggNodes
    void ProcessAggNodesStates(const MapPortKeyToAnPortPtr& p_ports);

    // Get Topology type from one location in order to support
    // setting type different from type configured, if required.
    const FabricTopologyInfo& GetFabricTopologyInfo() const { return m_topology_data_.GetTopologyInfo(); }

    TopologyType GetTopologyType() const { return m_topology_data_.GetTopologyInfo().m_topology_type; }

    CommandManager& GetCommandManager() const { return *m_command_manager_ptr_; }

    void ResizeTopologyData();

    // Return adjacent switch nodes of a switch
    // const SetNodesPtr &(Node *p_node) const;

    int Init();

    inline bool IsRediscoverRequired() { return m_rediscover_required_; }
    inline void clearRediscoverRequired() { m_rediscover_required_ = false; }

    void StartCommandHandling();
    bool HandleAggNodesInitState(bool seamless_restart);
    void CompareAggNodePortConfigWithConf();

    inline void DumpAMKeysToFile() { m_amkey_manager_.DumpAMKeysToFile(); }

    // Validates that AM is running in SM port
    void ValidateLocalPort();
    int UpdateFabricStart();
    int UpdateFabricEnd();
    int UpdateFabricFailed();
    void RevertFabricEpoch();

    int UpdateVportStart();
    int UpdateVportEnd();
    int UpdateVportFailed();
    void RevertVportEpoch();

    // Retry sending mad that received temporary error on the current epoch
    void MadSendRetry();

    // return MAX_NUM_HOPS if no path found
    uint8_t GetNumHops(uint64_t from_sw_guid, uint64_t to_sw_guid);

    // Add a link into the fabric - this will create nodes / ports and link between them
    // by calling the forward methods MakeNode + MakeLinkBetweenPorts
    int AddLink(const string& type1,
                phys_port_t num_ports_1,
                uint64_t node_guid_1,
                uint64_t port_guid_1,
                string& desc1,
                lid_t lid1, /*uint8_t lmc1, */
                phys_port_t port_num_1,
                const string& type2,
                phys_port_t num_ports_2,
                uint64_t node_guid_2,
                uint64_t port_guid_2,
                string& desc2,
                lid_t lid2, /*uint8_t lmc2, */
                phys_port_t port_num_2);

    uint32_t GetNodesNumber() const { return (uint32_t)m_node_by_guid_.size(); }

    uint32_t GetCaPortsNumber() const { return (uint32_t)m_ca_port_guid_to_node_guid_and_index_.size(); }
    std::size_t GetNumberOfPhysicalPorts() const
    {
        std::size_t number_of_ports = 0;
        for (const auto& current_node_pair : m_node_by_guid_) {
            number_of_ports += current_node_pair.second->GetNumberOfAllocatedPorts();
        }
        return number_of_ports;
    }

    int AssignNodesRank(SetOfGuids& root_guids);
    int AssignNodesHyperCubeCoordinates(MapGuidToHCCoordinates& coordinates_map);

    bool SetRetryOnMadFailure(int rec_status, Port* p_port, NodeState state);

    void SetAggNodeActiveState(Port* p_port);
    void HandleFabricUpdates();

    ////////////////////////////////////////////
    /// Fabric CSV Parser Call Backs Function
    ////////////////////////////////////////////
    int CreateNode(const NodeRecord& node_record);
    void UpdateEpochForNodes();
    int CreatePort(const PortRecord& port_record);
    void MarkPortForDeletion(const PortRecord& port_record, Node* p_node, char const* const reason);
    void DisablePort(const u_int64_t node_guid, const phys_port_t port_num);
    int CreateLink(const LinkRecord& link_record);
    int CreateSwitchTopoTree(const SwitchRecord& switch_record);
    int AssignNodeHyperCubeCoordinate(const SwitchRecord& switch_record);
    int AssignNodeDfpGroupInfo(const SwitchRecord& switch_record);
    int UpdateAnToAnRouting(const AnToAnRecord& an_to_an_record);
    int ParseSmRecord(const SmRecord& sm_record);
    int ParseSmPortsRecord(const SmPortsRecord& sm_ports_record);
    int ParseSmsRecord(const SmsRecord& sms_record);
    TopologyType TopologyStrToType(const std::string& topology_str);

    //////////////////////////////////////////////////
    /// Fabric virtual CSV Parser Call Backs Function
    //////////////////////////////////////////////////
    int CreateVnode(const VnodeRecord& vnode_record);
    int CreateVport(const VportRecord& vport_record);

    ////////////////////////////////////////////
    /// Fabric Provider Call Backs Function
    ////////////////////////////////////////////

    void SetAMKeyCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    void RecoverAMKeyCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    void DiscoverAggNodeCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    void RediscoverAggNodeCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    void CleanAggNodeCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    void ConfigureAggNodeCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

   private:
    int CalculateNodesRank();

    void CalculateMinHopsTables();

    Node* GetNodeByGuid(const uint64_t node_guid);
    Vnode* GetVnodeByGuid(const uint64_t vnode_guid);
    Port* GetCaPortByGuid(const uint64_t port_guid);
    Vport* GetVportByGuid(const uint64_t vport_guid);

    // create a new node in fabric (don't check if exists already)
    Node* MakeNode(const NodeType type, const phys_port_t num_ports, const uint64_t node_guid, const string& node_description);

    Vnode* MakeVnode(const uint64_t vnode_guid, const string& vnode_description);

    Vport* MakeVport(const uint64_t vport_guid,
                     const lid_t vlid,
                     const phys_port_t vport_num,
                     Vnode* p_vnode,
                     Port* p_port,
                     const uint8_t is_lid_required);

    // set the node's port given data (create one of does not exist).
    Port* SetNodePort(Node* p_node,
                      const uint64_t port_guid,
                      const lid_t lid,
                      const phys_port_t port_number,
                      const PortTimestamp& timestamp,
                      const SpecialPortType special_port_type,
                      const uint16_t port_rate);

    // Add a link between the given ports.
    // not creating sys ports for now.
    int MakeLinkBetweenPorts(Port* p_port1, Port* p_port2);

    // set a lid port
    // void SetLidPort(lid_t lid, Port *p_port);

    // get a port by lid
    Port* GetPortByLid(lid_t lid)
    {
        if (m_port_by_lid_.empty() || (m_port_by_lid_.size() < (unsigned)lid + 1))
            return NULL;
        return (m_port_by_lid_[lid]);
    };

    // dump out the contents of the entire fabric
    int Dump(ostream& sout) const;
    int Dump() const;

    int AddRootNode(const uint64_t root_guid);

    void RemoveLidPort(Port* p_port);
    void SetLidPort(lid_t lid, Port* p_port);

    // Clean all required AggNodes.
    void CleanAggNodes();
    void CleanAggNode(Port* p_port);

    void SetAMKeys();
    void SetAMKey(Port* p_port);

    // Configure all required AggNodes.
    void ConfigureAggNode(Port* p_port);
    void ConfigureAggNodePorts(Port* p_port);

    void UpdateTopologyData();

    void CalculateAggNodeGraph();
    void CalculateMinhopAggNodeGraph();
    void CalculateTreeAggNodeGraph();
    void CalculateAnMinHopsTables();
    void CalculateDfpAnMinHopsTables();

    void BuildCaUpdateList();
    void BuildVportsUpdateList();
    void BuildPathsUpdateList();

    void CreateCaPortData(Port* p_port, const PortInfo& port_info);
    void CreateVPortData(Vport* p_vport, const PortInfo& vport_info);
    int DiscoverAggNode(Port* p_port);
    int RecoverAMKey(Port* p_port);
    void AddUpdateAn(PortData* port_data, FabricUpdateType update_type);
    int AddComputePort(PortData* p_port_data, const string& host_name);

    void CheckIsCaPortUpdate(Port* p_port, const PortInfo& port_info);
    int RediscoverAggNode(Port* p_port);
    void CreateVportUpdateIfNeeded(Vport* p_vport,
                                   const PortInfo& new_vport_info,
                                   std::vector<uint64_t>& old_vports_guids,
                                   AggregatedLogMessage& disabled_vports_guids,
                                   AggregatedLogMessage& enabled_vports_guids);

    static void GetHostName(const PortInfo& port_info, string& hca_host_name);

    bool IsValidLid(lid_t lid) const { return (lid && lid <= FABRIC_MAX_VALID_LID); };

    void SetDataPathVersion();

    static node_min_hop_key_t GetNodeMinHopKey(Port* port);
};

#endif   // AGG_FABRIC_GRAPH_H_
