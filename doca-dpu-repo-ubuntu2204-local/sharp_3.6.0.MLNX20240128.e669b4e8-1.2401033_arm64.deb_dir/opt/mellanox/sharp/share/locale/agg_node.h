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

#ifndef AGG_NODE_H_
#define AGG_NODE_H_

#include <chrono>
#include "agg_types.h"
#include "smx/smx_types.h"

class AggNode;
class AggPath;
class AggNodeFabricInfo;
class Port;
class CommandManager;
class AnToAnInfo;

struct SortByAggNodeAppData1
{
    inline bool operator()(const AggNode* p_lhs, const AggNode* p_rhs) const;
};

struct AggPathSort
{
    inline bool operator()(const AggPath* p_lhs, const AggPath* p_rhs) const;
};

typedef std::list<class AggNode*> ListAggNodePtr;
typedef std::list<class AggPath*> ListAggPathPtr;
typedef std::vector<class TreeEdge> VecTreeEdge;
typedef std::vector<class TreeEdge*> VecTreeEdgePtr;

typedef std::vector<ListAggNodePtr> VecListAggNodePtr;
typedef std::vector<AggNode*> VecAggNodePtr;
typedef std::vector<AggPath*> VecAggPathPtr;

typedef std::map<port_key_t, AggNode*> MapPortKeyToAggNodePtr;
typedef pair<MapPortKeyToAggNodePtr::iterator, bool> MapPortKeyToAggNodePtrInsertRes;

typedef std::set<AggPath*, AggPathSort> SetAggPathPtr;
typedef std::map<port_key_t, SetAggPathPtr> MapAggNodeKeyToPath;

// Containers that hold AggNodeFabricInfo
typedef std::list<class AggNodeFabricInfo*> ListAggNodeFabricPtr;
typedef std::vector<AggNodeFabricInfo*> VecAggNodeFabricPtr;
typedef std::map<port_key_t, AggNodeFabricInfo*> MapPortKeyToAggNodeFabricPtr;
typedef pair<MapPortKeyToAggNodeFabricPtr::iterator, bool> MapPortKeyToAggNodeFabricPtrInsertRes;

enum AggPathDirection
{
    AGG_PATH_DIRECTION_FIRST = 0,
    AGG_PATH_DIRECTION_UP = AGG_PATH_DIRECTION_FIRST,
    AGG_PATH_DIRECTION_DOWN = AGG_PATH_DIRECTION_UP + 1,
    AGG_PATH_DIRECTION_SIZE = AGG_PATH_DIRECTION_DOWN + 1
};

enum AggPathActionNeeded
{
    AGG_PATH_ACTION_NEEDED_NONE = 0,
    AGG_PATH_ACTION_NEEDED_RECOVER,        // need to recover a path that becomes active
    AGG_PATH_ACTION_NEEDED_RECOVER_EDGE,   // On Qp ERROR
    AGG_PATH_ACTION_NEEDED_REROUTE,        // need to wait for job to end and recover
};

class AggPath
{
    AggNode* m_node_[AGG_PATH_DIRECTION_SIZE];
    bool m_is_valid_;
    uint8_t m_sw_hops_;   // hops between AN switches
    Epoch m_epoch_;       // last update number
    AnToAnInfo m_an_to_an_info_;

    sharp_path_id_t m_path_id_;
    static sharp_path_id_t m_next_id_;

    AggPathActionNeeded m_action_needed_;
    uint8_t m_active_semaphores_;
    uint8_t m_sat_load_;
    uint8_t m_exclusive_lock_load_;
    uint16_t m_agg_rate_;
    set<sharp_trees_t> m_tree_load_;
    uint32_t m_max_llt_qps_;
    uint32_t m_max_sat_qps_;
    uint32_t m_llt_qps_;
    uint32_t m_sat_qps_;
    sharp_resource_priority m_priority_;

   public:
    AggPath(AggNode* p_node_a,
            AggPathDirection node_a_direction,
            AggNode* p_node_b,
            AggPathDirection node_b_direction,
            uint64_t epoch,
            const AnToAnInfo& an_to_an_info)
        : m_is_valid_(true),
          m_sw_hops_(0),
          m_epoch_(epoch),
          m_an_to_an_info_(an_to_an_info),
          m_action_needed_(AGG_PATH_ACTION_NEEDED_NONE),
          m_active_semaphores_(0),
          m_sat_load_(0),
          m_exclusive_lock_load_(0),
          m_agg_rate_(0),
          m_max_llt_qps_(0),
          m_max_sat_qps_(0),
          m_llt_qps_(0),
          m_sat_qps_(0),
          m_priority_(SHARP_RESOURCE_PRIORITY_NORMAL)
    {
        m_node_[node_a_direction % 2] = p_node_a;
        m_node_[node_b_direction % 2] = p_node_b;
        m_path_id_ = m_next_id_++;
    }

    AggPath(AggNode* p_down_node,
            AggNode* p_up_node,
            sharp_path_id_t path_id,
            uint8_t sw_hops,
            uint64_t epoch,
            const AnToAnInfo& an_to_an_info)
        : m_is_valid_(true),
          m_sw_hops_(sw_hops),
          m_epoch_(epoch),
          m_an_to_an_info_(an_to_an_info),
          m_path_id_(path_id),
          m_action_needed_(AGG_PATH_ACTION_NEEDED_NONE),
          m_active_semaphores_(0),
          m_sat_load_(0),
          m_exclusive_lock_load_(0),
          m_agg_rate_(0),
          m_max_llt_qps_(0),
          m_max_sat_qps_(0),
          m_llt_qps_(0),
          m_sat_qps_(0),
          m_priority_(SHARP_RESOURCE_PRIORITY_NORMAL)
    {
        m_node_[AGG_PATH_DIRECTION_DOWN] = p_down_node;
        m_node_[AGG_PATH_DIRECTION_UP] = p_up_node;
    }

    sharp_resource_priority GetPriority() const { return m_priority_; }
    void SetPriority(sharp_resource_priority priority) { m_priority_ = priority; }
    void Update(uint8_t num_semaphores);

    sharp_path_id_t GetId() const { return m_path_id_; }

    void SetValid(bool is_valid, uint64_t epoch)
    {
        m_is_valid_ = is_valid;
        SetEpoch(epoch);
    }

    bool IsExclusiveLockAvailable() { return m_sat_load_ < m_active_semaphores_ ? true : false; }

    void IncSatLoad() { m_sat_load_++; }

    void DecSatLoad();

    uint32_t GetSatLoad() { return m_sat_load_; }

    void IncExclusiveLockLoad() { m_exclusive_lock_load_++; }

    void DecExclusiveLockLoad();

    bool IsLockAvailable() { return m_exclusive_lock_load_ < m_active_semaphores_ ? true : false; }

    bool HasQPsQuota(uint16_t qps_number, bool is_sat = false)
    {
        if (is_sat) {
            return ((m_max_sat_qps_) >= (m_sat_qps_ + qps_number));
        } else {
            return ((m_max_llt_qps_) >= (m_llt_qps_ + qps_number));
        }
    }

    void SetMaxQPs(uint32_t sat_qps, uint32_t llt_qps)
    {
        m_max_sat_qps_ = sat_qps;
        m_max_llt_qps_ = llt_qps;
    }

    void SetActiveSemaphores(uint8_t num_semaphores) { m_active_semaphores_ = num_semaphores; }

    void IncQPCount(bool is_sat)
    {
        if (is_sat) {
            m_sat_qps_++;
        } else {
            m_llt_qps_++;
        }
    }

    void DecQPCount(bool is_sat);

    void InsertTreeId(sharp_trees_t tree_id);
    bool TreeIdExist(sharp_trees_t tree_id);
    void EraseTreeId(sharp_trees_t tree_id);
    uint8_t GetTreeLoadSize();

    bool IsValid() const { return m_is_valid_; }

    void SetAggRate(uint16_t rate) { m_agg_rate_ = rate; }

    uint16_t GetAggRate() const { return m_agg_rate_; }

    void SetPathActionNeeded(AggPathActionNeeded action)
    {
        if (action == AGG_PATH_ACTION_NEEDED_RECOVER_EDGE && m_action_needed_ != AGG_PATH_ACTION_NEEDED_NONE) {
            return;
        }
        m_action_needed_ = action;
    }
    AggPathActionNeeded GetPathActionNeeded() const { return m_action_needed_; }

    AggNode* GetNode(AggPathDirection direction) { return m_node_[direction % 2]; }

    uint8_t GetSwHops() const { return m_sw_hops_; }
    void SetSwHops(uint8_t sw_hops) { m_sw_hops_ = sw_hops; }

    void SetEpoch(uint64_t epoch) { m_epoch_.SetEpoch(epoch); }

    uint64_t GetCurrEpoch() { return m_epoch_.m_curr_epoch_; }
    uint64_t GetPrevEpoch() { return m_epoch_.m_prev_epoch_; }

    AggNode* GetRemoteNode(const AggNode* local_agg_node) const
    {
        return (local_agg_node == m_node_[AGG_PATH_DIRECTION_FIRST] ? m_node_[AGG_PATH_DIRECTION_FIRST + 1]
                                                                    : m_node_[AGG_PATH_DIRECTION_FIRST]);
    }

    const AnToAnInfo& GetAnToAnInfo() const { return m_an_to_an_info_; }

    void AnToAnUpdate(const AnToAnInfo& an_to_an_info) { m_an_to_an_info_.Update(an_to_an_info); }

    void AnToAnUpdatePrev() { m_an_to_an_info_.UpdatePrev(); }

    void Recover(CommandManager* p_command_manager);

    // Compare AggPath object by comparing keys of their AggNode objects
    bool operator<(const AggPath& rhs) const;

    sharp_path_id_t GetId() { return m_path_id_; }

    string ToString() const;
};

class AggNode
{
    AggPortInfo m_agg_port_info_;

   protected:
    AggNodeInfo m_agg_node_info_;

    MapAggNodeKeyToPath m_path_[AGG_PATH_DIRECTION_SIZE];
    SetAggPathPtr m_agg_paths_;
    // MapAggNodeKeyToPath m_down_path_;
   private:
    sharp_an_id_t m_an_id_;
    static sharp_an_id_t m_next_id_;

    const Port* m_port_;   // containing port in fabric_graph

    // Temporary members
    // This information is volatile and can be used by different
    // methods and algorithms to save different data, temporarily
    PrivateAppData m_app_data_1;

    // Indicator that port resources configuration is required for this AggNode
    bool m_configure_port_resources_;

    AnMinHopsTable m_an_min_hops_table_;
    string m_switch_desc_;
    sharp_resource_priority m_priority_;

   public:
    AggNode(const PortInfo& port_info, const AggNodeInfo* p_agg_node_info, Port* p_port, uint64_t am_key, string switch_desc);

    AggNode(const PortInfo& port_info,
            const AggNodeInfo& agg_node_info,
            sharp_an_id_t an_id,
            uint64_t am_key,
            string switch_desc,
            const AnMinHopsTable an_min_hops_table)
        : m_agg_port_info_(agg_node_info.m_active_control_path_version, port_info, am_key),
          m_agg_node_info_(agg_node_info),
          m_an_id_(an_id),
          m_port_(NULL),
          m_configure_port_resources_(false),
          m_an_min_hops_table_(an_min_hops_table),
          m_switch_desc_(switch_desc),
          m_priority_(SHARP_RESOURCE_PRIORITY_NORMAL)
    {}

    virtual ~AggNode();   // Keep the destructor virtual, so that child classes also get their destructor

    bool operator==(const AggNode& rhs) const;

    sharp_resource_priority GetPriority() const { return m_priority_; }
    void SetPriority(sharp_resource_priority priority) { m_priority_ = priority; }
    const string GetSwitchDesc() const { return m_switch_desc_; }
    AggPortInfo& GetAggPortInfo() { return m_agg_port_info_; }
    const AggPortInfo& GetAggPortInfo() const { return m_agg_port_info_; }
    const PortInfo& GetPortInfo() const { return m_agg_port_info_.m_port_info; }
    uint8_t GetPortRank() const { return m_agg_port_info_.m_port_info.m_peer_topology_info.GetRank(); }
    inline uint8_t GetPortRankIfAvailable() const { return m_agg_port_info_.m_port_info.m_peer_topology_info.GetRankIfAvailable(); }
    const string& GetDescription() const { return GetPortInfo().m_node_desc; }
    sharp_an_id_t GetId() const { return m_an_id_; }
    static sharp_an_id_t GetAnNodesNumber() { return m_next_id_; }
    port_key_t GetKey() const { return GetPortInfo().m_port_key; }

    const AggNodeInfo& GetAggNodeInfo() const { return m_agg_node_info_; }
    void SetAggNodeInfo(const AggNodeInfo* p_agg_node_info)
    {
        m_agg_node_info_ = *p_agg_node_info;
        m_agg_port_info_.m_active_control_path_version = m_agg_node_info_.m_active_control_path_version;
    }

    void UpdateAggNodeInfo(const AggNodeInfo* p_agg_node_info)
    {
        m_agg_node_info_.m_num_of_jobs = p_agg_node_info->m_num_of_jobs;
        m_agg_node_info_.m_radix = p_agg_node_info->m_radix;
        m_agg_node_info_.m_endianness = p_agg_node_info->m_endianness;
        m_agg_node_info_.m_enable_reproducibility = p_agg_node_info->m_enable_reproducibility;
        m_agg_node_info_.m_reproducibility_disable_supported = p_agg_node_info->m_reproducibility_disable_supported;

        // m_max_control_path_version_supported cannot be changed
        m_agg_node_info_.m_active_control_path_version =
            (p_agg_node_info->m_active_control_path_version == 0 ? 1 : p_agg_node_info->m_active_control_path_version);
        m_agg_port_info_.m_active_control_path_version = m_agg_node_info_.m_active_control_path_version;

        m_agg_node_info_.m_active_data_path_version = p_agg_node_info->m_active_data_path_version;
        m_agg_node_info_.m_reproducibility_per_job_supported = p_agg_node_info->m_reproducibility_per_job_supported;
        m_agg_node_info_.m_enable_reproducibility_per_job = p_agg_node_info->m_enable_reproducibility_per_job;
        m_agg_node_info_.m_tree_job_default_binding = p_agg_node_info->m_tree_job_default_binding;
        m_agg_node_info_.m_am_key_supported = p_agg_node_info->m_am_key_supported;
        m_agg_node_info_.m_qp_to_port_select_supported = p_agg_node_info->m_qp_to_port_select_supported;
    }

    void SetCleanRequired(const bool clean_required)
    {
        m_agg_node_info_.m_clean_required = clean_required;
        if (clean_required) {
            // if clean required was set to True, we need to reset m_clean_required_reset_timepoint
            // that way we'll return 'false' in GetTimeSinceLastCleanRequiredResetInMinutes
            m_agg_node_info_.m_clean_required_reset_timepoint = std::chrono::steady_clock::time_point{};
            m_agg_port_info_.ClearInactiveReasonMessage();
        } else {
            // clean_required is set to false when fabric asks fabric_graph to clean it.
            // if 10 minutes after clean_required was set to false fabric could not clean it, we treat it's status as 'AUTO RECOVER FAILED'
            m_agg_node_info_.m_clean_required_reset_timepoint = std::chrono::steady_clock::now();
        }
    }

    inline std::pair<uint32_t, bool> GetTimeSinceLastCleanRequiredResetInMinutes() const
    {
        if (0 == m_agg_node_info_.m_clean_required_reset_timepoint.time_since_epoch().count()) {
            return {0, false};
        }
        const auto time_since_clean_required_reset_minutes =
            std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() -
                                                             m_agg_node_info_.m_clean_required_reset_timepoint)
                .count();
        return {time_since_clean_required_reset_minutes, true};
    }

    bool IsCleanRequired() const { return m_agg_node_info_.m_clean_required; }

    void UpdateAggNodeInfoSemaphores(uint8_t active_semaphores) { m_agg_node_info_.m_num_active_semaphores = active_semaphores; }

    void UpdateAggNodeInfoSemaphores(uint8_t port_num, uint8_t active_semaphores)
    {
        if (m_agg_node_info_.m_semaphores_per_port) {
            m_agg_node_info_.m_ports_to_semaphores[port_num] = active_semaphores;
        } else {
            m_agg_node_info_.m_num_active_semaphores = active_semaphores;
        }
    }

    uint8_t GetPortNumSemaphores(uint8_t port_num)
    {
        MapPortSemaphores::iterator iter = m_agg_node_info_.m_ports_to_semaphores.find(port_num);
        if (iter != m_agg_node_info_.m_ports_to_semaphores.end()) {
            return iter->second;
        } else {
            return m_agg_node_info_.m_num_active_semaphores;
        }
    }

    uint32_t GetPortSATQPs(uint8_t port_num)
    {
        MapPortSATQps::iterator iter = m_agg_node_info_.m_ports_to_num_sat_qps.find(port_num);
        if (iter != m_agg_node_info_.m_ports_to_num_sat_qps.end()) {
            return iter->second;
        } else {
            return m_agg_node_info_.m_max_sat_qps_per_port;
        }
    }

    uint8_t GetNumActiveSemaphores() const
    {
        // for NDR m_agg_node_info_.m_semaphores_per_port = true
        // (semaphores located at port). MAX flows per NDR switch is 64
        if (m_agg_node_info_.m_semaphores_per_port) {
            return MAX_SEMAPHORES;
        } else {
            return m_agg_node_info_.m_num_active_semaphores;
        }
    }

    string ToString() const { return GetPortInfo().GetName(); }

    void DumpPaths(FILE* f, const char* ident) const;

    // if new path created and paths != NULL add the new path to paths
    void Connect(AggNode* p_agg_node, AggPathDirection direction, uint64_t epoch, ListAggPathPtr* p_paths, AnToAnInfo* an2an_info);

    void Createpath(AggNode* p_agg_node,
                    AggPathDirection direction,
                    uint64_t epoch,
                    ListAggPathPtr* p_paths,
                    const AnToAnInfo& an_to_an_info);

    const SetAggPathPtr& GetAggPaths() const { return m_agg_paths_; }

    const MapAggNodeKeyToPath& GetAggPathsByDirection(AggPathDirection direction) const;

    AggPath* GetPathMinTreeLoad(AggNode* p_remote_agg_node, sharp_trees_t tree_id);

    AggPath* GetPathByPort(AggNode* p_remote_agg_node, sharp_trees_t tree_id, uint32_t port);

    void SetPort(Port* p_port) { m_port_ = p_port; }
    const Port* GetPort() const { return m_port_; }

    // insert newly created path to connected agg_nodes containers
    void InsertNewPath(AggPath* p_path,
                       AggPathDirection direction,
                       AggPathDirection opposite_direction,
                       AggNode* p_remote_node,
                       uint64_t epoch);

    void SetAppData1(uint64_t val) { m_app_data_1.val = val; }

    const PrivateAppData& GetAppData1() const { return m_app_data_1; }

    // Set port resources configuration required indicator
    void SetConfigurePortResources(bool configure_port_resources) { m_configure_port_resources_ = configure_port_resources; }

    // Return port resources configuration required indicator
    bool GetConfigurePortResources() const { return m_configure_port_resources_; }

    AnMinHopsTable& GetAnMinHops() { return m_an_min_hops_table_; };

    void UpdateAnMinhopTable(const AnMinHopsTable& an_min_hops_table)
    {
        if (an_min_hops_table.size())
            m_an_min_hops_table_ = an_min_hops_table;
    }

    uint8_t GetNumFlows(phys_port_t port_num);
    void FillAggPathVec(vector<AggPath*>& vec_agg_path, uint32_t port_num = 0);

   private:
    void DumpPaths(FILE* f, const char* ident, AggPathDirection direction) const;

    // In the future we might implement several agg nodes graphs
    // PGFT Torus etc.
    // The following method should be implemented by AggGraphInfo
    // and derived classes PgftGraphInfo and TorusGraphInfo
    uint8_t GetDirectionNumber() const { return AGG_PATH_DIRECTION_SIZE; }
    const char* GetDirectionString(AggPathDirection direction) const;
    AggPathDirection GetOppositeDirection(AggPathDirection direction) const;
};

struct AggNodePtrSort
{
    bool operator()(const AggNode* lhs, const AggNode* rhs) const { return (lhs->GetKey() < rhs->GetKey()); }
};

bool SortByAggNodeAppData1::operator()(const AggNode* p_lhs, const AggNode* p_rhs) const
{
    return p_lhs->GetAppData1().val < p_rhs->GetAppData1().val;
}

bool AggPathSort::operator()(const AggPath* p_lhs, const AggPath* p_rhs) const
{
    return (*p_lhs < *p_rhs);
}

//---------------------
// Iterator for Map of base AggNode
//---------------------
class AggNodeIterator : std::iterator<std::input_iterator_tag, AggNode*>
{
    MapPortKeyToAggNodePtr::iterator m_iter_;

   public:
    explicit AggNodeIterator(const MapPortKeyToAggNodePtr::iterator& iter) : m_iter_(iter) {}

    AggNodeIterator& operator++()
    {
        ++m_iter_;
        return *this;
    }

    bool operator==(const AggNodeIterator& agg_node_iter) { return m_iter_ == agg_node_iter.m_iter_; }

    bool operator!=(const AggNodeIterator& agg_node_iter) { return m_iter_ != agg_node_iter.m_iter_; }

    AggNode* operator*() { return m_iter_->second; }
};

//---------------------
// Iterator for Map of Fabric AggNode
//---------------------
class AggNodeFabricIterator : std::iterator<std::input_iterator_tag, AggNodeFabricInfo*>
{
    MapPortKeyToAggNodeFabricPtr::iterator m_iter_;

   public:
    explicit AggNodeFabricIterator(const MapPortKeyToAggNodeFabricPtr::iterator& iter) : m_iter_(iter) {}

    AggNodeFabricIterator& operator++()
    {
        ++m_iter_;
        return *this;
    }

    bool operator==(const AggNodeFabricIterator& agg_node_iter) { return m_iter_ == agg_node_iter.m_iter_; }

    bool operator!=(const AggNodeFabricIterator& agg_node_iter) { return m_iter_ != agg_node_iter.m_iter_; }

    AggNodeFabricInfo* operator*() { return m_iter_->second; }
};

struct topology_minhop_tables
{
    std::map<uint64_t, std::list<uint64_t>> adjacent_table;
    std::map<uint64_t, MinHopsTable> minhop_tables;
    // guid to min_hop_key(LID) table
    // should be removed once node ID is implemented
    std::map<uint64_t, node_min_hop_key_t> minhop_keys_table;

    void Clear()
    {
        adjacent_table.clear();
        minhop_tables.clear();
        minhop_keys_table.clear();
    };
};

struct FabricTopologyInfo
{
    union
    {
        uint8_t m_max_sw_rank;
        uint8_t m_dimensions;
        uint8_t m_group_number;
    } m_info;

    TopologyType m_topology_type;

    struct topology_minhop_tables m_topology_minhop_tables;

    FabricTopologyInfo(TopologyType topology_type) : m_topology_type(topology_type) { memset(&m_info, 0, sizeof(m_info)); }

    uint8_t GetSize() const;
    string ToString() const;
};

class FabricTopologyData
{
    FabricTopologyInfo m_topology_info_;

    // For HYPER_CUBE, AggNodes are kept in a vector by their coordinate value.
    VecAggNodePtr m_agg_node_by_sw_coord_;

   public:
    // For TREE & DFP topologies, AggNodes are kept in a vector of ranks
    // At each rank, there is a list of AggNodes
    VecListAggNodePtr m_agg_node_by_sw_rank_;

    const FabricTopologyInfo& GetTopologyInfo() const { return m_topology_info_; }

    FabricTopologyData(TopologyType topology_type) : m_topology_info_(topology_type) {}

    void SetTopologyType(TopologyType topology_type);
    void UpdateTopologyInfo(const FabricTopologyInfo& topology_info);
    void Resize(uint8_t size);

    void Clear()
    {
        m_agg_node_by_sw_rank_.clear();
        m_agg_node_by_sw_coord_.clear();
    }

    static void GetCoordDimension(uint16_t coordinates, uint8_t& dimensions)
    {
        while (coordinates >= (1 << dimensions)) {
            dimensions++;
        }
    }

    struct topology_minhop_tables& GetTopologyMinHopTables() { return m_topology_info_.m_topology_minhop_tables; }
    std::map<uint64_t, std::list<uint64_t>>& GetAdjacentTable() { return m_topology_info_.m_topology_minhop_tables.adjacent_table; }
    std::map<uint64_t, MinHopsTable>& GetMinHopTables() { return m_topology_info_.m_topology_minhop_tables.minhop_tables; }
    std::map<uint64_t, node_min_hop_key_t>& GetMinHopKeysTable() { return m_topology_info_.m_topology_minhop_tables.minhop_keys_table; }

    void InsertAggNode(AggNode* p_agg_node);

    string ToString() const { return m_topology_info_.ToString(); }
};

bool compare_agg_node_ptr(AggNode* p_lhs, AggNode* p_rhs);

#endif   // AGG_NODE_H_
