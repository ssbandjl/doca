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

#ifndef AGG_NODE_FABRIC_INFO_H_
#define AGG_NODE_FABRIC_INFO_H_

#include "agg_node.h"
#include "am_bit_array.h"
#include "am_log.h"
#include "fabric_dump.h"
#include "job_manager.h"

struct JobData
{
    sharp_job_id_t sharp_job_id;
    uint64_t job_id;
    uint64_t job_key;
    uint64_t reservation_id;
    std::string reservation_key;   // Reservation ID from reservation
    ResourceLimitValueByRank resource_limit_allocated_by_rank;
    uint8_t addr_type;
    char addr[SHARP_SMX_ADDRESS_MAX_LEN];
    sharp_job_state job_state;
    std::vector<sharp_trees_t> tree_ids;
    std::vector<uint64_t> tree_feature_masks;
    uint16_t max_ost;
    uint32_t max_buffer;
    uint16_t max_groups;
    uint32_t ud_qpn;
    uint16_t max_qps;
    uint8_t priority;
    uint32_t num_host_guids;
    std::vector<uint64_t> host_guids;
    uint8_t num_channels_per_conn;
    uint8_t num_rails;
    uint8_t num_trees;
    string job_file_path;
};

typedef std::map<sharp_job_id_t, JobData> MapRunningJobData;
typedef MapRunningJobData::iterator MapRunningJobDataIter;

class CommandManager;

class AggNodeFabricInfo : public AggNode
{
    MapTreeIdToTreeNode m_tree_nodes_;   // TreeNodes belongs to this node
    uint32_t m_leafs_spanning_load_;     // Number of leafs spans from this AN
    uint32_t m_allocated_system_qps_;    // Number of qps allocated for system trees
                                         // and not for a specific job
    uint32_t m_allocated_sat_qps_;
    uint32_t m_allocated_llt_qps_;
    uint16_t m_trees_load_;                // Number of TreeNodes
    uint16_t m_job_load_;                  // Number of jobs using the AggNode
    uint16_t m_job_sat_load_;              // Number of jobs using SAT on the AggNode
    uint16_t m_job_exclusive_lock_load_;   // Number of jobs using ExclusiveLock on the AggNode

    uint16_t m_root_count_;   // number of trees this node is the
                              // root node. for tree calculation

    // available_quota
    AggNodeInfo m_max_available_quota_0_;   // For jobs with priority == 0
    AggNodeInfo m_available_quota_0_;       // For jobs with priority == 0
    AggNodeInfo m_all_available_quota_;     // For all jobs

    // DS for build trees
    SetAnFabricInfoPtr m_descendant_leafs_;   // descendant leaf ANs
    MapTreeTurns m_map_tree_turns_;           // count the number of identical tree
                                              // turn in order to create unique trees
    bool m_is_pure_ftree_;                    // true if only one path to
                                              // every descendant.

    bool m_is_valid_;
    char m_date_str_[AM_DUMP_MAX_STR_LEN];
    Qp m_multicast_qp_;
    SetJobPtr m_pending_jobs_;   // Pending jobs that should be
                                 // ended before starting AN recover.

    SetTreeIds m_tree_ids_set_;                               // tree ids that were restored during seamless restart
    MapSharpJobIdToTreeIds m_sharp_job_id_to_tree_ids_set_;   // tree ids that were bound to a job. used for seamless restart
    MapQPNToQPData m_qpn_to_qp_data_map_;                     // qp number to qp data map. used for seamless restart
    MapRunningJobData m_running_job_data_;                    // data of running jobs during startup. used for seamless restart

    AMBitArray m_available_tree_ids;                // Bit array of available tree ids
    AMBitArray m_available_tree_ids_low_priority;   // Bit array of available tree ids, with previous errors on those tree ids.
                                                    // Those tree ids will be used only if no other tree id is available.
                                                    // This array is cleared upon successful job (configuration  + cleanup)
    uint64_t
        m_priority_epoch_;   // This epoch specifies when last error happened. While picking trees for job, prefer AggNode with oldest error

   public:
    AggNodeFabricInfo(const PortInfo& port_info,
                      const AggNodeInfo& agg_node_info,
                      sharp_an_id_t an_id,
                      uint64_t am_key,
                      string switch_desc,
                      const AnMinHopsTable an_min_hops_table,
                      uint16_t num_llt_trees);

    const MapTreeIdToTreeNode& GetTreeNodes() const { return m_tree_nodes_; }
    AMBitArray& GetAvailableTreeIds() { return m_available_tree_ids; }
    AMBitArray& GetAvailableTreeIdsLowPriority() { return m_available_tree_ids_low_priority; }
    void GetAvailableQuota(uint8_t priority, AggNodeInfo& available_quota) const;

    const SetAnFabricInfoPtr& GetdescendantLeafs() const { return m_descendant_leafs_; }

    uint16_t GetMaxAllowedRadix() { return std::min(g_option_info.m_max_tree_radix, m_agg_node_info_.m_radix); }

    bool IsPureFatTree() const { return m_is_pure_ftree_; }

    void UpdateTimeStamp() { get_date_str(m_date_str_, AM_DUMP_MAX_STR_LEN, true); }
    const char* GetDateStr() const { return m_date_str_; }

    void SetValid(bool is_valid)
    {
        if (m_is_valid_ != is_valid) {
            UpdateTimeStamp();
        }
        m_is_valid_ = is_valid;
    }
    bool IsValid() const;

    uint16_t GetTreesLoad() const { return m_trees_load_; }

    bool IsRecoveryPending();

    string TreesString() const
    {
        std::stringstream stream;
        for (int i = 0; i < g_fabric.GetMaxTreeId(); i++) {
            MapTreeIdToTreeNode::const_iterator iter = m_tree_nodes_.find(i);
            if (iter != m_tree_nodes_.end()) {
                stream << i << ",";
            }
        }
        return stream.str();
    }

    void ClearInternalConfig();
    void UnconfigureTreeNodes(CommandManager* p_command_manager);

    void RecoverAfterRestart();

    TreeNode* GetTreeNode(sharp_trees_t tree_id) const
    {
        TreeNode* p_tree_node = NULL;
        MapTreeIdToTreeNode::const_iterator iter = m_tree_nodes_.find(tree_id);
        if (iter != m_tree_nodes_.end()) {
            p_tree_node = iter->second;
        }

        return p_tree_node;
    }

    void AddTreeNode(sharp_trees_t tree_id, TreeNode* p_tree_node);

    int RemoveTreeNode(sharp_trees_t tree_id, bool force = false, bool clean_flow = false);

    bool HasQPsQuota(uint16_t qps_number, bool sat_qps = false) const
    {
        bool sat_or_llt_qps_check;
        if (sat_qps) {
            sat_or_llt_qps_check = (m_all_available_quota_.m_max_sat_qps >= (m_allocated_sat_qps_ + qps_number));
        } else {
            sat_or_llt_qps_check = (m_all_available_quota_.m_max_llt_qps >= (m_allocated_llt_qps_ + qps_number));
        }
        return (m_all_available_quota_.m_qps >= qps_number) && sat_or_llt_qps_check;
    }

    // check if there is enough quota and return true if there is
    // available_quota (OUT) the available_quota for the given priority
    bool HasQuota(uint16_t min_qps, uint8_t priority, AggNodeInfo& available_quota) const;
    bool HasQuota(const sharp_quota& quota, uint8_t priority, AggNodeInfo& available_quota) const;

    void AllocateSystemQps(uint32_t qps_number);
    void DeallocateSystemQps(uint32_t qps_number);
    uint32_t GetAllocatedSystemQpsNumber() { return m_allocated_system_qps_; }

    int AddJobToTreeNode(SharpJob* p_job, sharp_trees_t tree_id);
    int RemoveJobFromTreeNode(SharpJob* p_job, sharp_trees_t tree_id);
    void RemoveJobFromNode(SharpJob* p_job);

    int AllocQuota(const sharp_quota& quota, uint8_t priority);
    int FreeQuota(const Quota& quota, uint8_t priority);

    void ClearDescendantLeafs() { m_descendant_leafs_.clear(); }

    void BuildDescendantLeafs();
    void DumpDescendants(FILE* f, const char* ident) const;

    // fill the spanning_info_list for all descendant that are
    // also in the given descendant_leafs
    void GetSpanningInfo(const SetAnFabricInfoPtr& descendant_leafs,
                         ListSpanningInfo& spanning_info_list,
                         port_key_t parent_key,
                         sharp_trees_t tree_id);

    uint16_t GetTreeTurns(port_key_t parent_key, port_key_t descendant_key) const;
    void AddTreeTurn(port_key_t parent_key, port_key_t descendant_key);

    uint16_t GetJobLoad() const { return m_job_load_; }
    uint16_t GetJobSatLoad() const { return m_job_sat_load_; }
    uint16_t GetJobExclusiveLockLoad() const { return m_job_exclusive_lock_load_; }
    uint16_t GetJobSatLoad(sharp_trees_t tree_id) const;
    bool IsTreeNodeRoot(SharpJob* p_job, TreeNode* p_tree_node);
    void RestoreFromNetworkExistingTreeNodes(int* p_operation_status, SetTreeIds tree_ids);
    void RestoreFromNetworkTreeToJobBindings(int* p_operation_status, const sharp_job_id_t sharp_job_id, const SetTreeIds tree_ids);
    void RestoreFromNetworkAllQps(int* p_operation_status);
    void RestoreFromNetworkExistingTreeNodesConfigurations(int* p_operation_status);
    int AssignRestoredQPsToTreeEdges();
    void RestoreFromNetworkAllJobIdsOfRunningJobs(int* p_operation_status);
    void RestoreFromNetworkAllJobQuotasOfRunningJobs(int* p_operation_status);
    void ReleaseQpsLeftFromSeamlessRestart();
    void CleanDataRecoveredDuringSeamlessRestart();
    void RestoreTreeNodeAggPaths(int& p_operation_status);

    MapRunningJobDataIter GetRunningJobsDataStartIter() { return m_running_job_data_.begin(); }

    MapRunningJobDataIter GetRunningJobsDataEndIter() { return m_running_job_data_.end(); }

    bool IsTreeIdExist(sharp_trees_t tree_id) { return (m_tree_ids_set_.find(tree_id) != m_tree_ids_set_.end()); }

    bool IsTreeIdBoundToJob(sharp_job_id_t sharp_job_id, sharp_trees_t tree_id)
    {
        auto tree_ids_set_iter = m_sharp_job_id_to_tree_ids_set_.find(sharp_job_id);
        if (tree_ids_set_iter != m_sharp_job_id_to_tree_ids_set_.end()) {
            SetTreeIds& tree_ids = tree_ids_set_iter->second;
            if (tree_ids.find(tree_id) != tree_ids.end()) {
                return true;
            }
        }
        return false;
    }

    QPData* GetQPData(uint32_t qpn)
    {
        if (m_qpn_to_qp_data_map_.find(qpn) == m_qpn_to_qp_data_map_.end()) {
            return NULL;
        } else {
            return &m_qpn_to_qp_data_map_[qpn];
        }
    }

    int RemoveQPDataFromMap(uint32_t qpn)
    {
        if (m_qpn_to_qp_data_map_.find(qpn) == m_qpn_to_qp_data_map_.end()) {
            return -1;
        } else {
            m_qpn_to_qp_data_map_.erase(qpn);
        }
        return 0;
    }

    MapQPNToQPDataIter GetQPNToQPDataStartIter() { return m_qpn_to_qp_data_map_.begin(); }

    MapQPNToQPDataIter GetQPNToQPDataEndIter() { return m_qpn_to_qp_data_map_.end(); }

    MapQPNToQPData GetMapQpData() { return m_qpn_to_qp_data_map_; }

    JobData* GetRunningJobData(sharp_job_id_t sharp_job_id);

    void SetPathActionNeededOnRecover();
    void FillTreeEdgesVec(uint32_t port_num, VecTreeEdgePtr& tree_edges);

    void SetPriorityEpoch(uint64_t epoch);
    uint64_t GetPriorityEpoch() { return m_priority_epoch_; }

    bool IsLockAvailable() const;
    bool IsExclusiveLockAvailable() const;

    // Multicast
    uint32_t GetMulticastQpn();

    int AllocateMulticastQps(int* p_operation_status);

    // int ReleaseMulticastQps(int *p_operation_status);

    int ConfigureMulticastQps(int* p_operation_status);

    void UpdateLeafsSpanningLoad(uint16_t num_descendant_leafs) { m_leafs_spanning_load_ += num_descendant_leafs; }

    ////////////////////////////////////////////
    /// DFP
    ////////////////////////////////////////////

    uint8_t GetRootCount() { return m_root_count_; }
    void IncreaseRootCount() { m_root_count_++; }

    uint32_t UpdateDescendantLeafs(const AnMinHopsTable& parent_an_min_hops_table, const SetAnFabricInfoPtr& parent_descendant_leafs);

    uint8_t GetDfpTreeHeight();
    uint16_t GetGroupNaiborsNumber() const;

   private:
    QpStateEnum QPCConfigStateToQpStateEnum(int state)
    {
        switch (state) {
            case 0:
                return QP_STATE_ALLOCATED;
            case 1:
                return QP_STATE_CONFIGURED;
            case 2:
                return QP_STATE_ERROR;
            default:
                return QP_STATE_UNCONFIGURE_REQUIRED;
        }
    }

    void RecoverMulticastQps();

    void AddDescendantLeafs(AggNodeFabricInfo* p_descendant_node);

    // Get SetAnFabricInfoPtr of descendant that may span leafs
    // On DFP generates DescendantLeafs on all descendants
    // input parameter descendant_leafs is relevant only for DFP.
    //       get leafs that spans given descendant_leafs
    void GetSpanningLeafs(SetAnFabricInfoPtr& spanning_leafs, const SetAnFabricInfoPtr& descendant_leafs);

    // call only after calling HasQuota
    void AllocQuota(AggNodeInfo& available_quota, const sharp_quota& quota);

    int FreeQuota(AggNodeInfo& available_quota,
                  const AggNodeInfo& max_quota,   // for sanity
                  const Quota& quota);

    ////////////////////////////////////////////
    /// AggNode Provider Call Backs Function
    ////////////////////////////////////////////

    void AllocateMulticastQpsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void ReleaseMulticastQpsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void ConfigureMulticastQpCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RestoreFromNetworkExistingTreeNodeCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RestoreFromNetworkTreeToJobBindingsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RestoreFromNetworkAllQpsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RestoreFromNetworkExistingTreeNodesConfigurationsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RestoreFromNetworkQpConfigurationCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RestoreFromNetworkAllJobIdsOfRunningJobsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RestoreFromNetworkAllJobQuotasOfRunningJobsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void QueryTreeConfigCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void QueryQpDatabaseCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void DisableQpCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RemoveQpFromTreeCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void ReleaseQpsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);
};

struct AnFabricInfoPtrSort
{
    bool operator()(const AggNodeFabricInfo* lhs, const AggNodeFabricInfo* rhs) const { return (lhs->GetKey() < rhs->GetKey()); }
};

#endif   // AGG_NODE_FABRIC_INFO_H_
