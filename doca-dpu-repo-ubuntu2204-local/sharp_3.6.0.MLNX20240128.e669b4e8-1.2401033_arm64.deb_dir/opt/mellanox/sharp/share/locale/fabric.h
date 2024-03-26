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

#ifndef AGG_FABRIC_H_
#define AGG_FABRIC_H_

#include <unordered_map>

#include "agg_node.h"
#include "agg_types.h"
#include "am_common.h"
#include "am_log.h"
#include "dump_file.h"
#include "fabric_db.h"
#include "fabric_dump_parser.h"
#include "fabric_provider.h"
#include "option_manager.h"
#include "port_data.h"
#include "smx/smx_types.h"
#include "thread_pool.h"
#include "tree_manager.h"

class FabricProvider;
class TreeManager;
class CommandManager;
class TreeNode;
class TreeEdge;
class FabricDb;
class HostInfo;
class PortData;
class SharpJob;
class AggNode;
class AggNodeFabricInfo;
struct SpanningInfo;
struct QPData;

struct HostInfoSort
{
    inline bool operator()(const HostInfo* p_lhs, const HostInfo* p_rhs) const;
};

struct TreeNodePtrSort
{
    bool operator()(const TreeNode* lhs, const TreeNode* rhs) const;
};

typedef std::set<SharpJob*> SetJobPtr;
typedef std::unordered_set<SharpJob*> UnorderedSetJobPtr;
typedef std::map<sharp_job_id_t, SetTreeIds> MapSharpJobIdToTreeIds;
typedef std::set<class TreeNode*> SetTreeNodePtr;
typedef std::set<class TreeNode*, TreeNodePtrSort> SetTreeNodePtrSortedByANGuid;
typedef std::set<const HostInfo*, HostInfoSort> SetHostInfoPtr;
typedef std::set<port_key_t> SetPortKey;
typedef std::list<class PortData*> ListPortDataPtr;
typedef std::list<struct SpanningInfo> ListSpanningInfo;
typedef std::vector<class AggNodeFabricInfo*> VecAnFabricInfoPtr;
typedef std::vector<class TreeNode*> VecTreeNodePtr;
typedef std::unordered_map<sharp_trees_t, class TreeNode*> MapTreeIdToTreeNode;
typedef std::unordered_map<uint64_t, std::string> MapTreeHashToDumpMessage;
typedef std::vector<class TreeEdge> VecTreeEdge;
typedef std::vector<class AggTree*> VecAggTreePtr;
typedef std::map<sharp_trees_t, class AggTree*> MapTreeIdToAggTree;
typedef std::map<sharp_job_id_t, MapTreeIdToAggTree> MapJobIdToMapTreeIdToTree;
typedef VecAggTreePtr::iterator AggTreesIter;
typedef VecAggTreePtr::const_iterator AggTreesConstIter;

typedef std::map<string, ListAggNodeFabricPtr> MapStrListAggNodeFabricPtr;
typedef std::map<string, class AggNodeFabricInfo*> MapStrAggNodeFabricPtr;
typedef std::multimap<port_key_t, class AggNodeFabricInfo*> MapPortKeyToAggNodeFabrics;
typedef std::map<string, class HostInfo*> MapStrToHostInfo;

typedef std::map<port_key_t, port_key_t> MapPortKeyToPortKey;
typedef std::map<uint32_t, QPData> MapQPNToQPData;
typedef MapQPNToQPData::iterator MapQPNToQPDataIter;
typedef MapPortKeyToPortKey::iterator MapPortKeyToPortKeyIter;
typedef std::multimap<port_key_t, port_key_t> MapPortKeyToPortKeys;
typedef MapPortKeyToPortKeys::iterator MapPortKeyToPortKeysIter;

typedef std::map<std::pair<port_key_t, port_key_t>, uint16_t> MapTreeTurns;
typedef std::map<port_key_t, AggNodeRecord> MapPortKeyToAggNodeRecord;
typedef std::map<port_key_t, AggNodeInfoRecord> MapPortKeyToAggNodeInfoRecord;
typedef std::map<string, uint32_t> MapFileNameToCRC;
typedef std::vector<struct AggPathRecord> VecAggPathRecord;

enum class DynamicTreeAlgoEnum
{
    SUPER_POD_ORIENTED_ALGORITHM = 0,
    QUASI_FAT_TREE_ORIENTED_ALGORITHM = 1
};

class HostInfo
{
    string m_host_name_;
    sharp_sd_info m_sharpd_info_;
    MapPortDataPtr m_ports_data_;

    static sharpd_id_t m_next_id_;

   public:
    explicit HostInfo(const string& host_name);

    void AddPortData(PortData* p_port_data);
    void ChangePortDataState(const port_key_t port_key, const bool should_enable);
    void EnablePortData(const port_key_t port_key);
    void DisablePortData(const port_key_t port_key);
    void DeletePortFromPortsData(const port_key_t port_key);

    inline const string& GetHostName() const { return m_host_name_; }
    inline const sharp_sd_info& GetSharpdInfo() const { return m_sharpd_info_; }
    inline sharpd_id_t GetSharpdId() const { return m_sharpd_info_.sharpd_id; }
    inline const MapPortDataPtr& GetPortsData() const { return m_ports_data_; }
};

struct SpanningInfo
{
    AggNodeFabricInfo* m_agg_node;
    uint16_t m_trees_load;
    uint32_t m_leafs_spanning_load;
    SetAnFabricInfoPtr m_spanning_set;
    uint16_t m_tree_turns;
    uint16_t m_tree_child_idx;   // Number of children

    SpanningInfo() : m_agg_node(NULL), m_trees_load(0), m_leafs_spanning_load(0), m_tree_turns(0), m_tree_child_idx(0) {}

    // Prefer build tree using node that spans over maximal leafs, with minimal
    // load and less used turns (pair of two consecutive paths)
    int32_t GetScore() const { return (int32_t)m_spanning_set.size() * 5 - m_trees_load - m_tree_turns; }

    bool operator<(SpanningInfo const& rhs) const;
    bool operator>(SpanningInfo const& rhs) const { return (rhs < *this); }
};

// use bits values to enable selecting several states (e.g. configured or error)
enum QpStateEnum
{
    QP_STATE_UNALLOCATED = 1,
    QP_STATE_ALLOCATED = 2,
    QP_STATE_CONFIGURED = 4,
    QP_STATE_ERROR = 8,
    QP_STATE_UNALLOCATE_REQUIRED = 16,
    QP_STATE_UNCONFIGURE_REQUIRED = 32,
    QP_STATE_UNCONFIGURE = (QP_STATE_UNCONFIGURE_REQUIRED | QP_STATE_CONFIGURED)
};

static inline const char* QpStateToStr(QpStateEnum qp_state)
{
    switch (qp_state) {
        case QP_STATE_UNALLOCATED:
            return ("unallocated");
        case QP_STATE_ALLOCATED:
            return ("allocated");
        case QP_STATE_CONFIGURED:
            return ("configured");
        case QP_STATE_ERROR:
            return ("error");
        case QP_STATE_UNALLOCATE_REQUIRED:
            return ("unallocate required");
        case QP_STATE_UNCONFIGURE_REQUIRED:
            return ("unconfigure required");
        case QP_STATE_UNCONFIGURE:
            return ("unconfigure required or configured");
        default:
            return ("unknown");
    }
};

struct QPData
{
    uint32_t qpn;
    uint32_t index;
    QpStateEnum state;
    SharpMtu mtu;
    lid_t rlid;
    uint8_t ts;
    uint32_t rqpn;
    bool is_parent;
    sharp_trees_t tree_id;
    uint32_t port;
    bool is_root_qp;
    bool is_multicast_qp;
};

class Qp
{
    uint32_t m_qp_num_;
    QpStateEnum m_qp_state_;

   public:
    Qp() : m_qp_num_(0), m_qp_state_(QP_STATE_UNALLOCATED) {}

    void SetNumber(uint32_t qp_number) { m_qp_num_ = qp_number; }

    uint32_t GetNumber() const { return m_qp_num_; }

    QpStateEnum GetState() const { return m_qp_state_; }

    void Set(uint32_t qp_num, QpStateEnum qp_state, AggNodeFabricInfo* p_an_fabric_info);

    void SetState(QpStateEnum qp_state) { m_qp_state_ = qp_state; }

    void Clear()
    {
        m_qp_num_ = 0;
        m_qp_state_ = QP_STATE_UNALLOCATED;
    }
};

class Fabric
{
    MapPortDataUniquePtr m_port_data_by_key_;
    VecPortDataPtr m_old_ports_;
    AggregatedLogMessage m_disabled_ports_msg_;
    MapStrListAggNodeFabricPtr m_agg_node_by_desc_;         // Nodes list ptr by node desc
    MapPortKeyToAggNodeFabricPtr m_agg_node_by_port_key_;   // AggNode ptr node by node key

    FabricTopologyData m_topology_data_;

    MapStrToHostInfo m_hosts_;
    VecAggTreePtr m_agg_trees_;   // Vector of all the trees, computed at start time
    // Maps between job and id a map of tree ids to agg trees. computed at each begin job. discarded at job end
    MapJobIdToMapTreeIdToTree m_job_agg_trees_;
    VecAggNodeFabricPtr m_agg_nodes_;
    VecAggPathPtr m_agg_paths_;

    ////Optimization
    MapPortKeyToPortKey m_compute_to_sw_port_;
    // set of closest AggNode to compute port
    // we might have several AggNodes to compute ports if not directly connected
    MapPortKeyToAggNodeFabrics m_compute_port_to_agg_nodes_;
    MapPortKeyToAggNodeFabricPtr m_sw_to_agg_node_;

    // set of all computes PortData that are connected to ANs not spans by trees
    SetPortDataConstPtr m_compute_not_on_tree_;
    MapPortKeyToPortKeys m_sw_with_no_an_to_compute_ports_;

    ////collection for the current epoch
    MapPortKeyToPortKeys m_sw_to_update_compute_ports_;

    //  accumulates updates until the next build_trees is executed
    MapPortKeyToAggNodeFabrics m_update_compute_to_agg_nodes_;

    uint64_t m_epoch_;   // Current update index
    bool m_trees_built_;
    uint16_t m_llt_trees_to_build_;
    uint16_t m_max_tree_id_;
    uint16_t m_max_llt_tree_id_;
    bool m_dynamic_tree_allocation;
    DynamicTreeAlgoEnum m_dynamic_tree_algo;
    port_key_t m_sm_port_guid_;

    FabricProvider* m_fabric_provider_;
    TreeManager m_tree_manager_;
    CommandManager* m_command_manager_;
    FabricDb* m_fabric_db_;

    AggTreesIter m_llt_end_itr_;
    uint8_t m_control_path_version_;
    bool m_agg_nodes_file_update_req_;
    bool m_is_seamless_restart_in_progress_;
    MapPortKeyToAggNodeRecord m_agg_node_records_by_port_key_;
    MapPortKeyToAggNodeInfoRecord m_agg_node_info_records_by_port_key_;
    VecAggPathRecord m_agg_path_records_with_action_;
    MapFileNameToCRC m_file_crc_by_filename_;
    AMGeneralInfoRecord m_general_info_record_;
    bool m_rebuild_trees_required_;
    MapTreeHashToDumpMessage m_trees_history_by_hash_;
    std::unique_ptr<file_utils::DumpFile> m_trees_history_file_ptr_;

    void CheckUpdateTreesPerPortSet(const SetPortDataConstPtr& compute_ports);
    int CreateDumpDirectory(const string& dir_name) const;

   public:
    explicit Fabric()
        : m_disabled_ports_msg_(4 * 1024),
          m_topology_data_(TOPOLOGY_TYPE_NONE),
          // m_topology_type_(TOPOLOGY_TYPE_NONE),
          // m_max_sw_rank_(0),
          // m_dimensions_(0),
          m_epoch_(0),
          m_trees_built_(false),
          m_llt_trees_to_build_(0),
          m_max_tree_id_(0),
          m_max_llt_tree_id_(0),
          m_dynamic_tree_allocation(false),
          m_dynamic_tree_algo(DynamicTreeAlgoEnum::SUPER_POD_ORIENTED_ALGORITHM),
          m_sm_port_guid_(0),
          m_fabric_provider_(NULL),
          m_tree_manager_(),
          m_command_manager_(NULL),
          m_fabric_db_(NULL),
          m_control_path_version_(0),
          m_agg_nodes_file_update_req_(false),
          m_is_seamless_restart_in_progress_(false),
          m_rebuild_trees_required_(false)
    {}

    ~Fabric();

    int Init(int& seamless_restart);
    void ConfigureLocalPort(port_key_t local_port);
    void Clean();

    uint16_t GetMaxTreeId() { return m_max_tree_id_; }

    uint16_t GetMaxLltTreeId() { return m_max_llt_tree_id_; }

    bool IsDynamicTreeAllocation() const { return m_dynamic_tree_allocation; }

    void SetDynamicTreeAllocation(bool val) { m_dynamic_tree_allocation = val; }

    DynamicTreeAlgoEnum GetDynamicTreeAlgo() { return m_dynamic_tree_algo; }

    void FabricDumpTables();
    void SetCommandManager(CommandManager* p_command_manager);
    void SetFabricDb(FabricDb* p_fabric_db) { m_fabric_db_ = p_fabric_db; }

    const MapStrListAggNodeFabricPtr& GetAggNodesByDesc() const { return m_agg_node_by_desc_; }
    const VecAggPathPtr& GetAggPaths() const { return m_agg_paths_; }

    int BuildTreesSeamlessRestart();
    int RestoreFromNetworkExistingTreeNodes(const SetTreeIds tree_ids);
    int RestoreFromNetworkAllQpsOnAllAggNodes();
    int RestoreFromNetworkTreeNodesConfigurations(const SetTreeIds tree_ids);
    int AssignRestoredQPs();
    int RestoreAggPaths();
    bool IsSeamlessRestartInProgress() { return m_is_seamless_restart_in_progress_; }
    void BuildTrees();
    void RebuildSkeletonTrees();
    uint16_t GetLltTreesCount() { return m_llt_trees_to_build_; }

    uint16_t GetTotalTreesNum() const { return m_agg_trees_.size(); }

    uint16_t GetAggNodesNum() const { return m_agg_node_by_port_key_.size(); }

    sharp_trees_t GetSatTreeId(sharp_trees_t llt_tree_id);
    sharp_trees_t GetLltTreeId(sharp_trees_t sat_tree_id);

    bool IsSatTreeId(sharp_trees_t tree_id) const;

    FabricProvider* GetFabricProvider() const { return m_fabric_provider_; }

    TopologyType GetTopologyType() const { return m_topology_data_.GetTopologyInfo().m_topology_type; }

    FabricTopologyData& GetTopologyData() { return m_topology_data_; }

    VecListAggNodePtr& GetAggNodeBySwRank() { return m_topology_data_.m_agg_node_by_sw_rank_; }

    const MapPortKeyToAggNodeFabricPtr& GetPeerKeyToAggNodePtr() const { return m_sw_to_agg_node_; }

    inline uint8_t GetMaxSwRank() const { return m_topology_data_.GetTopologyInfo().m_info.m_max_sw_rank; }

    inline uint8_t GetMaxSwRankIfAvailable() const
    {
        if (GetTopologyType() != TOPOLOGY_TYPE_TREE) {
            return INVALID_TREE_RANK;
        }
        return m_topology_data_.GetTopologyInfo().m_info.m_max_sw_rank;
    }

    // Return number of dimension on Hyper-Cube
    uint8_t GetNumDimensions() const { return m_topology_data_.GetTopologyInfo().m_info.m_dimensions; }

    // Trees
    int GetAvailableLltTreeId(int start_index = 0) const { return GetAvailableTreeId(start_index, m_llt_trees_to_build_ - 1); }

    int GetAvailableSatTreeId(int start_index) const
    {
        return GetAvailableTreeId(((start_index > (m_llt_trees_to_build_ - 1)) ? start_index : m_llt_trees_to_build_),
                                  (m_llt_trees_to_build_ * 2) - 1);
    }

    AggTree* CreateTree(sharp_trees_t tree_id);
    AggTree* GetTree(sharp_trees_t tree_id) const;
    AggTreesIter GetTreesIter() { return m_agg_trees_.begin(); }
    bool IsTreesEnd(AggTreesIter& iter) { return (iter == m_agg_trees_.end()); }
    bool IsLltTreesEnd(AggTreesIter& iter) { return (iter == m_llt_end_itr_); }
    AggTreesConstIter GetTreesIter() const { return m_agg_trees_.begin(); }
    bool IsTreesEnd(AggTreesConstIter& iter) const { return (iter == m_agg_trees_.end()); }
    bool IsLltTreesEnd(AggTreesConstIter& iter) const { return (iter == m_llt_end_itr_); }

    void DeleteTree(sharp_trees_t tree_id);
    void DeleteAllTrees();

    AggTree* CreateJobTree(sharp_job_id_t job_id, sharp_trees_t tree_id);
    void DeleteJobTrees(sharp_job_id_t job_id, bool force = false, bool clean_flow = false);
    void DeleteAllJobTrees(bool force = false);
    AggTree* GetJobTree(sharp_job_id_t job_id, sharp_trees_t tree_id);
    void GetJobTreeIds(sharp_job_id_t job_id, SetTreeIds& set_tree_id);

    void GetAggNodesResourceData(sharp_resource_agg_node*& agg_node_recources, uint32_t& agg_node_num);
    void FillANSwitchPortsResourceData(sharp_resource_agg_node*& agg_node_recources, uint32_t& agg_node_num);
    void FillANStatusDetails(sharp_resource_agg_node& smx_agg_node, const AggNodeFabricInfo& fabric_agg_node);
    void GetAggTreesResourceData(sharp_resource_agg_tree*& tree_recources, uint32_t& agg_trees_num);
    void GetAggPathResourceData(sharp_resource_link*& link_recources,
                                uint32_t& link_recources_links_num,
                                sharp_resource_link* links = NULL,
                                uint32_t links_num = 0);
    uint16_t GetAggPathNum() const;

    int SetTreePriority(sharp_resource_priority priority, sharp_trees_t tree_id);

    // UpdateFabric
    void DeleteOldPorts();
    void HandlePortsUpdates(const FabricTopologyInfo& topology_info, ListPortUpdateInfo& updates, uint64_t epoch);
    void LogPortsUpdates(uint32_t new_cn_ports_counter,
                         uint32_t new_an_ports_counter,
                         uint32_t new_vports_counter,
                         uint32_t active_vports_counter);
    void UpdatesPathsState(ListPathUpdate& updates, uint64_t epoch);
    void HandlePathsUpdates(ListPathUpdate& updates, uint64_t epoch);
    void HandleAggNodePortUpdateFabricDB(ListPortUpdateInfo& updates);

    void CheckUpdateTrees(const SharpJob* p_job);

    int FabricUpdateEnd();
    void Start(uint32_t agg_nodes_number,
               uint32_t paths_number,
               sharp_job_id_t max_jobs_number,
               uint8_t control_path_version,
               uint16_t tree_table_size);

    // get node by desc or node key
    AggNodeFabricInfo* GetAggNode(const string& node_desc, uint64_t port_key);
    AggNodeFabricInfo* GetAggNode(uint64_t port_key);
    AggNodeFabricInfo* GetAggNode(const string& node_desc);
    AggNodeFabricInfo* GetAggNode(lid_t lid);

    AggNodeFabricInfo* GetAggNodeById(sharp_an_id_t an_id) const;
    void InsertAggNodeById(AggNodeFabricInfo* p_agg_node);

    AggPath* GetAggPathById(sharp_path_id_t path_id) const;
    void InsertAggPathById(AggPath* p_agg_path);

    // Aggregation nodes iterator
    AggNodeFabricIterator GetAggNodeIteratorBegin() { return AggNodeFabricIterator(m_agg_node_by_port_key_.begin()); }

    AggNodeFabricIterator GetAggNodeIteratorEnd() { return AggNodeFabricIterator(m_agg_node_by_port_key_.end()); }

    const HostInfo* GetHostInfo(const string& host_name) const;
    uint8_t GetNumHops(AggNodeFabricInfo* p_agg_node1, AggNodeFabricInfo* p_agg_node2);

    const PortData* FindConstPortData(port_key_t port_key) const;

    uint8_t GetControlPathVersion() const { return m_control_path_version_; }

    int GetMinimalShortestPathGraph(uint64_t* guids,
                                    uint32_t num_guids,
                                    std::map<uint64_t, std::set<uint64_t>>& sub_graph,
                                    std::map<uint64_t, std::list<uint64_t>>& leaf_to_hosts);

    void SubDivide(uint64_t root,
                   uint64_t src,
                   std::list<uint64_t>& destinations,
                   std::map<uint64_t, std::set<uint64_t>>& known_dst_map,
                   std::map<uint64_t, std::set<uint64_t>>& sub_graph);

    void UpdateAggNodeInfo(AggNodeFabricInfo* p_agg_node, const AggNodeInfo& agg_node_info, const PortInfo& port_info);

    void TryRecoverTrees();
    int ReadAggNodeStateFromDump();
    int ValidateAggNodeStateFromDump();
    void UpdateAggNodeStateFromDump();
    int ValidateRequiredFWCapabilitiesForSeamlessRestart();
    int CompareAggNodeInfoWithDump();
    int UpdateAggPathActionFromDump();
    void ReleaseQpsLeftFromSeamlessRestart();
    void CleanDataRecoveredDuringSeamlessRestart();
    int ValidateConfigFileCRC();
    void MarkInvalidSwitchesForRediscovery();

    // This method returns the leaf switch that is directly connected to 'port_key, NULL if not found
    AggNodeFabricInfo* GetSwitchThatIsConnectedToHost(port_key_t port_guid);

    int BuildQuasiFatTreeForJob(const SetPortDataConstPtr& compute_ports,
                                JobSubTreeInfo& job_sub_tree_info,
                                const JobResource& job_resource,
                                JobSubTreeScore& sub_tree_score,
                                TreeNodesVecUniquePtr& tmp_tree_nodes_vec,
                                sharp_job_id_t sharp_job_id);

    ////////////////////////////////////////////
    /// Fabric CSV Parser Call Backs Function
    ////////////////////////////////////////////
    int ParseAggNode(const AggNodeRecord& agg_node_record);
    int ParseAggNodeInfo(const AggNodeInfoRecord& agg_node_info_record);
    int ParseAggPath(const AggPathRecord& agg_path_record);
    int ParseConfigFileCRC(const FileCRCRecord& file_crc_record);
    int ParseAMGeneralInfo(const AMGeneralInfoRecord& am_general_info_record);
    AMGeneralInfoRecord& GetAMGeneralInfoRecord() { return m_general_info_record_; }
    uint64_t GetEpoch() { return m_epoch_; }
    void AddFabricGraphUpdate(FabricGraphPortDataUpdate& port_update);

   private:
    // find existing PortData
    // Returns NULL if not found
    PortData* FindPortData(port_key_t port_key, const bool should_ignore_invalid_ports = true);
    PortData* CreatePortData(const PortInfo& port_info);
    void CreateVportData(const PortInfo& vport_info, const std::string& host_name, const PortInfo& phys_port_info);

    void CreatePath(AggNodeFabricInfo* p_down_node,
                    AggNodeFabricInfo* p_up_node,
                    sharp_path_id_t path_id,
                    uint8_t sw_hops,
                    uint64_t epoch,
                    uint8_t num_semaphores,
                    uint16_t agg_rate,
                    uint32_t sat_qps,
                    uint32_t llt_qps,
                    const AnToAnInfo& p_an_to_an_info);

    // Allocate and configure multicast QPs for AggNodes
    void ConfigureMulticast();

    int DiscoverAggNode(PortData* p_port_data);
    int AddComputePort(PortData* p_port_data, const string& host_name);
    void AddAggNode(PortData* p_port_data,
                    const AggNodeInfo& agg_node_info,
                    sharp_an_id_t an_id,
                    const AnMinHopsTable& an_min_hops_table,
                    uint64_t am_key,
                    string switch_desc);
    int AddHostInfoToVport(PortData* p_vport_data, const string& host_name);

    int UpdateCaPort(PortInfo& port_info);

    void DeleteAllPortData();
    void DeleteAllHosts();

    void HandlemComutesOnSwitchWithNewAggNode(port_key_t sw_key);
    void EraseSwMultimapElement(MapPortKeyToPortKeys& sw_multimap, port_key_t sw_key, port_key_t element);

    // Associate compute ports to aggregation nodes
    // This method needs to be invoked before invoking UpdateComputeToTreeNode
    int UpdateComputeToAggNode();
    void UpdateComputeSwDb(const PortData* p_port_data, bool is_new);

    // Associate compute ports to tree nodes
    // UpdateComputeToAggNode needs to be invoked before this method
    int UpdateComputeToTreeNode(bool use_all_computes = false);
    // clear mapping between compute to tree nodes on all valid trees
    void ClearComputeToTreesNode(port_key_t compute_port_key);

    // Use the trees information to prepare information that will be used during
    // resource limitation per reservation
    void PrepareReservationResourceLimit();

    // old virtual port is a port that was deleted from m_port_data_by_key_ because fabric received INACTIVE update for it
    void ActivateVport(const PortUpdateInfo& port_update_info);
    void HandleVportUpdate(const PortUpdateInfo& port_update_info, PortData* p_port_data);

    void HandlePortUpdate(const PortUpdateInfo& port_update_info);

    bool IsTreeUpdateSupported()
    {
        if (GetTopologyType() == TOPOLOGY_TYPE_TREE || GetTopologyType() == TOPOLOGY_TYPE_DFP) {
            return true;
        }
        return false;
    }
    int ConfigureTreesOnFabric();
    int GetAvailableTreeId(int start_index, int max_tree_id) const;

    void DumpTrees() const;
    void DumpTreesState() const;
    void DumpTreesStructure() const;
    void CreateTreesHistoricData();
    void ExcludeExistingTreesFromHistoricData();
    void DumpTreesStructureHistory();   // dumps historic trees structure that are no longer exist
    int DumpFabric() const;
    void DumpFabric(FILE* f) const;
    void SetHostInfo(PortData* p_port_data, const string& host_name);

    template <typename PortType, typename... AdditionalPortDataParameters>
    PortData* CreatePortDataImpl(char const* const port_type_str,
                                 const PortInfo& port_info,
                                 const AdditionalPortDataParameters&... port_data_parameters)
    {
        try {
            std::unique_ptr<PortData> p_port_data = std::unique_ptr<PortData>(new PortType(port_info, port_data_parameters...));
            auto insert_pair =
                m_port_data_by_key_.insert(make_pair<port_key_t, std::unique_ptr<PortData>>(port_key_t{port_info.m_port_key}, nullptr));
            if (!insert_pair.second) {   // the key already exists
                INFO("Replacing existing %s in m_port_data_by_key_ with key: " U64H_FMT, port_type_str, port_info.m_port_key);
                m_old_ports_.push_back(std::move(insert_pair.first->second));
                m_old_ports_.back()->Disable();
            }
            // replace nullptr or the old port in the map to the new port
            insert_pair.first->second = std::move(p_port_data);
            return insert_pair.first->second.get();
        } catch (const std::bad_alloc&) {
            ERROR("Failed to create %s for %s\n", port_type_str, port_info.GetName().c_str());
            throw;
        }
    }
};

// Recognize the global variables
extern Fabric g_fabric;

bool HostInfoSort::operator()(const HostInfo* p_lhs, const HostInfo* p_rhs) const
{
    return (p_lhs->GetSharpdId() < p_rhs->GetSharpdId());
}

#endif   // AGG_FABRIC_H_
