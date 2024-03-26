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

#ifndef TREE_NODE_H_
#define TREE_NODE_H_

#include "agg_node.h"
#include "agg_node_fabric_info.h"
#include "agg_types.h"
#include "fabric_provider.h"
#include "smx/smx_types.h"

#define TREE_UNASSIGNED_RANK     0xFF
#define CHECK_PATH_IS_VALID      true    // used in create\allocate cases
#define DONT_CHECK_PATH_IS_VALID false   // used in delete\remove cases, UnConfigureEdge

class SharpJob;
class TreeNode;

typedef std::set<uint8_t> SetChildIndex;
typedef std::list<const PortData*> ListPortDataConstPtr;

class ResourceAllocInfo
{
    uint8_t m_children_number_;
    // SetHostInfoPtr          m_hosts_;
    ListPortDataConstPtr m_compute_ports_;
    SetChildIndex m_used_child_indexes_;

   public:
    ResourceAllocInfo(const ResourceAllocInfo& resource_alloc_info)
    {
        m_children_number_ = resource_alloc_info.m_children_number_;
        m_compute_ports_ = ListPortDataConstPtr(resource_alloc_info.m_compute_ports_);
        m_used_child_indexes_ = SetChildIndex(resource_alloc_info.m_used_child_indexes_);
    }

    void Reset()
    {
        m_children_number_ = 0;
        // m_hosts_.clear();
        m_compute_ports_.clear();
        m_used_child_indexes_.clear();
    }

    ResourceAllocInfo() : m_children_number_(0) {}

    // Update ChildrenNumber if using child_index_per_port > 1
    void SetChildrenNumber(uint8_t children_number) { m_children_number_ = children_number; }

    uint16_t GetChildrenNumber() const { return m_children_number_; }
    uint16_t GetComputePortsNumber() const { return (uint16_t)m_compute_ports_.size(); }
    // uint8_t get_hosts_number() const { return (uint8_t)m_hosts_.size(); }

    void AddComputePort(const PortData* p_compute_port)
    {
        m_children_number_++;
        m_compute_ports_.push_back(p_compute_port);
        // m_hosts_.insert(p_compute_port->GetHostInfo());
    }

    void AddUsedChildIndex(uint16_t child_index)
    {
        m_children_number_++;
        m_used_child_indexes_.insert(child_index);
        // TODO: Convert m_used_child_index_ to bitset (performance optimization)
    }

    const ListPortDataConstPtr& GetComputePorts() const { return m_compute_ports_; }

    const SetChildIndex& GetUsedChildIndexes() const { return m_used_child_indexes_; }
};

class TreeEdge
{
    TreeNode* m_local_tree_node_;
    TreeNode* m_remote_tree_node_;
    AggPath* m_path_;
    TreeEdge* m_remote_tree_edge_;
    SetJobPtr m_pending_jobs_;   // wait for these jobs to end before repairing
    Qp m_qp_;
    uint16_t m_pkey_;
    bool m_configured_;   // true if configured on tree
    uint8_t m_sat_load_;
    uint8_t m_exclusive_lock_load_;
    SharpMtu m_mtu_;

   public:
    TreeEdge()
        : m_local_tree_node_(NULL),
          m_remote_tree_node_(NULL),
          m_path_(NULL),
          m_remote_tree_edge_(NULL),
          m_qp_(),
          m_pkey_(0),
          m_configured_(false),
          m_sat_load_(0),
          m_exclusive_lock_load_(0),
          m_mtu_(SHARP_MTU_UNKNOWN)
    {}

    void Clear()
    {
        m_local_tree_node_ = NULL;
        m_remote_tree_node_ = NULL;
        m_path_ = NULL;
        m_remote_tree_edge_ = NULL;
        m_qp_.Clear();
        m_configured_ = false;
        m_sat_load_ = 0;
        m_exclusive_lock_load_ = 0;
    }

    void IncSatLoad() { m_sat_load_++; }

    void DecSatLoad();

    uint8_t GetSatLoad() { return m_sat_load_; }

    void IncExclusiveLockLoad() { m_exclusive_lock_load_++; }

    void DecExclusiveLockLoad();

    uint8_t GetExclusiveLockLoad() { return m_exclusive_lock_load_; }

    void ClearInternalConfig()
    {
        m_qp_.Clear();
        m_configured_ = false;
        m_pending_jobs_.clear();
    }

    TreeNode* GetLocalTreeNode() const { return m_local_tree_node_; }

    TreeNode* GetRemoteTreeNode() const { return m_remote_tree_node_; }

    bool IsConfigured() const { return m_configured_; }

    bool IsRecoverPending() const { return !m_pending_jobs_.empty(); }

    AggPath* GetPath() const { return m_path_; }

    void SetAggPath(AggPath* agg_path) { m_path_ = agg_path; }

    bool IsPathValid() const { return (m_path_ && m_path_->IsValid()); }

    bool IsSatPathValid() const { return (m_path_ && m_path_->IsValid() && m_path_->GetSwHops() == 1); }

    bool IsAllocateQp() const { return (IsConfigOnFabricRequiredNoLogs(CHECK_PATH_IS_VALID) && m_qp_.GetState() == QP_STATE_UNALLOCATED); }

    inline bool IsConfigOnFabricRequiredNoLogs(bool check_path_is_valid) const;

    bool IsConfigOnFabricRequired(bool check_path_is_valid) const;

    string ToString() const;
    const char* QpsToChar() const;

    TreeEdge* GetRemoteTreeEdge() const { return m_remote_tree_edge_; }

    int SetTreeNode(TreeNode* local_tree_node, TreeNode* remote_tree_node);

    void SetRemoteTreeEdge(TreeEdge* remote_tree_edge)
    {
        m_remote_tree_edge_ = remote_tree_edge;
        remote_tree_edge->m_remote_tree_edge_ = this;
    }

    Qp& GetQp() { return m_qp_; }

    // Updates the state of the edge and the QPs according to information from the subnet
    void RestoreEdgeInfo(FabricProvider* p_fabric_provider);

    int RestoreAggPath();

    uint16_t GetPkey() const { return m_pkey_; }

    void SetPkey(uint16_t pkey) { m_pkey_ = pkey; }

    const Qp& GetQp() const { return m_qp_; }

    bool SetQpsState(QpStateEnum qp_state)
    {
        return (m_remote_tree_edge_ ? (SetQpState(qp_state) && m_remote_tree_edge_->SetQpState(qp_state)) : SetQpState(qp_state));
    }

    bool SetQpState(QpStateEnum qp_state);

    void SetConfigured(bool is_configured_on_tree) { m_configured_ = is_configured_on_tree; }

    void AddPendingJob(SharpJob* p_job) { m_pending_jobs_.insert(p_job); }

    void SetPendingJobs(SetJobPtr& pending_jobs);
    int RemovePendingJob(SharpJob* p_job);

    // configure local and remote QPs on the fabric
    // Call this only if RemoteTreeEdge is not NULL and config is required
    void ConfigureQps(FabricProvider* p_fabric_provider,
                      FabricProviderCallbackContext& callback_context,
                      QpStateEnum required_curr_state,
                      bool is_disable,
                      bool is_config_local,
                      bool is_config_remote);

    // Call this only if RemoteTreeEdge is not NULL and config is required
    void RemoveQpsFromTree(FabricProvider* p_fabric_provider, int* p_operation_status, bool is_config_local, bool is_config_remote);

    // Call this only if RemoteTreeEdge is not NULL and config is required
    void AllocateQp(FabricProvider* p_fabric_provider, int* p_operation_status);

    // Call this only if RemoteTreeEdge is not NULL and config is required
    void AddQpToTree(FabricProvider* p_fabric_provider, int* p_operation_status);

    // Call this only if RemoteTreeEdge is not NULL and config is required
    void DeallocateQps(FabricProvider* p_fabric_provider, int* p_operation_status, bool is_config_local, bool is_config_remote);

    // Calc TreeEdge MTU according to TreeNodes on both sides
    void CalculateMtu();

    TreeEdge* GetTwinTreeEdge();

   private:
    // configure local QP on the fabric
    void ConfigureQp(FabricProvider* p_fabric_provider,
                     FabricProviderCallbackContext& callback_context,
                     QpStateEnum required_curr_state,
                     bool is_disable);

    void RemoveQpFromTree(FabricProvider* p_fabric_provider, int* p_operation_status, bool is_parent);

    void DeallocateQp(FabricProvider* p_fabric_provider, int* p_operation_status);

    ////////////////////////////////////////////
    /// TreeEdge Provider Call Backs Function
    ////////////////////////////////////////////

    void ConfigureQpCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void RemoveQpFromTreeCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void QueryTreeConfigCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void QueryQpDatabaseCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void ReleaseQpsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);
};

/* For SAT jobs cleanup flow as follows:
 1. TREE_NODE_JOB_STATE_CLEAN_JOB_TREE_QP_DISCONNECT
 2. TREE_NODE_JOB_STATE_CLEAN_QPS
 3. TREE_NODE_JOB_STATE_CLEAN_JOB_TREE
 4. TREE_NODE_JOB_STATE_CLEAN_JOB
 5. TREE_NODE_JOB_STATE_DELETE_TREE_NODE

 For LLT jobs cleanup flows as follows:
 1. TREE_NODE_JOB_STATE_CLEAN_QPS
 2. TREE_NODE_JOB_STATE_CLEAN_JOB
 3. TREE_NODE_JOB_STATE_DELETE_TREE_NODE
*/
enum TreeNodeJobStateEnum
{
    TREE_NODE_JOB_STATE_NULL,
    TREE_NODE_JOB_STATE_CLEAN_JOB_TREE_QP_DISCONNECT,   // ResourceCleanup opcode 7
    TREE_NODE_JOB_STATE_CLEAN_JOB_TREE,                 // ResourceCleanup opcode 3
    TREE_NODE_JOB_STATE_CLEAN_JOB,                      // ResourceCleanup opcode 1
    TREE_NODE_JOB_STATE_CLEAN_QPS,                      // Disable QPs, remove from tree, unallocate QPs
    TREE_NODE_JOB_STATE_DELETE_TREE_NODE,               // TreeConfig TreeState 0, opcode 2
};

enum TreeNodeStateEnum
{
    TREE_NODE_STATE_NOT_CONFIGURED,         // Default value, not configured
    TREE_NODE_STATE_NOT_CONFIGURED_ERROR,   // ERROR state. Error occurred before TreeConfig MAD with WRITE_TREE opcode sent
    TREE_NODE_STATE_CONFIGURED,             // Configured successfully
    TREE_NODE_STATE_CONFIG_ERROR            // ERROR state. Error occurred after TreeConfig MAD with WRITE_TREE opcode sent
};

class TreeNode
{
    sharp_trees_t m_tree_id_;
    uint8_t m_rank_;
    uint16_t m_children_number_;
    uint16_t m_child_index_;
    uint16_t m_exclusive_lock_load_;
    TreeNodeJobStateEnum m_job_state_;   // Used only in dynamic tree allocation mode.
    AggNodeFabricInfo* m_agg_node_;      // The fabric Agg Node
    TreeEdge m_parent_edge_;             // The Edge to Parent TreeNode
    VecTreeEdge m_descendants_;          // Descendants tree nodes
    ResourceAllocInfo m_alloc_info_;
    TreeNodeStateEnum m_tree_node_state_;
    SetJobPtr m_jobs;
    TreeNode* m_sat_node;   // SAT node related to this LLT node
    TreeNode* m_llt_node;   // LLT node related to this SAT node

   public:
    TreeNode(sharp_trees_t tree_id, AggNodeFabricInfo* p_agg_node)
        : m_tree_id_(tree_id),
          m_rank_(TREE_UNASSIGNED_RANK),
          m_children_number_(0),
          m_child_index_(0),
          m_exclusive_lock_load_(0),
          m_job_state_(TREE_NODE_JOB_STATE_NULL),
          m_agg_node_(p_agg_node),
          m_parent_edge_(),
          m_tree_node_state_(TREE_NODE_STATE_NOT_CONFIGURED),
          m_sat_node(NULL),
          m_llt_node(NULL)
    {
        m_descendants_.resize(MAX_TREE_RADIX);
        m_agg_node_->AddTreeNode(m_tree_id_, this);
    }

    ~TreeNode() { m_agg_node_->RemoveTreeNode(m_tree_id_, true); }

    string ToString() const
    {
        std::stringstream stream;
        stream << "TreeNode tree id:" << m_tree_id_ << " number of children: " << (int)m_children_number_
               << " on AggNode: " << m_agg_node_->ToString();
        return stream.str();
    }

    void ClearInternalConfig();

    sharp_trees_t GetTreeID() { return m_tree_id_; }

    AggNodeFabricInfo* GetNode() { return m_agg_node_; }

    AggNodeFabricInfo const* GetNode() const { return m_agg_node_; }

    TreeNode* GetParent() { return m_parent_edge_.GetRemoteTreeNode(); }

    VecTreeEdge& GetDescendants() { return m_descendants_; }

    TreeNode* GetChild(uint8_t child_index)
    {
        if (child_index < m_children_number_) {
            return m_descendants_[child_index].GetRemoteTreeNode();
        }
        return NULL;
    }
    const TreeNode* GetChild(uint8_t child_index) const
    {
        if (child_index < m_children_number_) {
            return m_descendants_[child_index].GetRemoteTreeNode();
        }
        return NULL;
    }

    TreeEdge& GetParentEdge() { return m_parent_edge_; }

    // Returns 0 if connection is OK
    int CheckValidParentConnection();

    TreeEdge* GetChildLocalEdge(uint8_t child_index)
    {
        if (child_index < m_children_number_) {
            return &m_descendants_[child_index];
        }
        return NULL;
    }

    TreeEdge* GetChildRemoteEdge(uint8_t child_index)
    {
        if (child_index < m_children_number_) {
            return m_descendants_[child_index].GetRemoteTreeEdge();
        }
        return NULL;
    }

    uint8_t GetChildIndex() const { return m_child_index_; }

    uint8_t GetRank() const { return m_rank_; }
    void SetRootRank() { m_rank_ = 0; }
    void SetRank(uint8_t rank) { m_rank_ = rank; }

    void SetSatNode(TreeNode* p_sat_node) { m_sat_node = p_sat_node; }
    TreeNode* GetSatNode() const { return m_sat_node; }

    void SetLltNode(TreeNode* p_llt_node) { m_llt_node = p_llt_node; }
    TreeNode* GetLltNode() const { return m_llt_node; }

    ResourceAllocInfo& GetAllocInfo() { return m_alloc_info_; }

    bool CanAddDescendant() { return (m_children_number_ < m_agg_node_->GetMaxAllowedRadix()); }

    TreeNodeStateEnum GetTreeNodeState() const { return m_tree_node_state_; }
    void SetTreeNodeState(TreeNodeStateEnum state) { m_tree_node_state_ = state; }

    int AddDescendant(TreeNode* p_tree_node)
    {
        if (!CanAddDescendant()) {
            return -1;
        }

        if (m_descendants_[m_children_number_].SetTreeNode(this, p_tree_node)) {
            return -1;
        }
        m_descendants_[m_children_number_].SetRemoteTreeEdge(&p_tree_node->GetParentEdge());

        // update child
        if (p_tree_node->SetParent(this)) {
            return -1;
        }
        p_tree_node->SetChildIndexValue(m_children_number_);

        m_children_number_++;
        return 0;
    }

    uint8_t GetChildrenNumber() const { return m_children_number_; }

    int RotateTreeNode();

    void UpdateDescendantsRank(ListTreeNodePtr& nodes_queue);
    void PushDescendantsToQueue(ListTreeNodePtr& nodes_queue);

    void DumpTreeNode(std::string& dump_message) const;
    void RecursiveDump(FILE* f) const;

    int AllocateQps(int* p_operation_status, bool is_config_remote);

    void UnallocateQps(int* p_operation_status, TreeEdge* p_tree_edge, bool is_config_remote, bool check_qp_state);

    // if p_tree_edge != NULL configure the given edge QP and remote QP
    void ConfigureQps(int* p_operation_status,
                      bool is_disable,
                      bool is_config_local,
                      bool is_config_remote,
                      TreeEdge* p_tree_edge,
                      bool is_config_descendants,
                      bool is_config_parent);

    // Unconfigure local and remote QPs on the fabric with Unconfigure Required state
    void UnconfigureQps(int* p_operation_status, TreeEdge* p_tree_edge);

    int ConfigureTreeNode(int* p_operation_status);
    void RemoveRemoteQpsFromTree(int* p_operation_status);
    void RemoveLocalQpsFromTree(int* p_operation_status);

    int AddJob(SharpJob* p_job);
    int RemoveJob(SharpJob* p_job);

    const SetJobPtr& GetJobs() { return m_jobs; }

    uint16_t GetJobsLoad() const { return m_jobs.size(); }
    void IncExclusiveLockLoad() { m_exclusive_lock_load_++; }
    void DecExclusiveLockLoad() { m_exclusive_lock_load_--; }
    uint16_t GetExclusiveLockLoad() { return m_exclusive_lock_load_; }

    void SetJobState(TreeNodeJobStateEnum job_state) { m_job_state_ = job_state; }

    TreeNodeJobStateEnum GetJobState() { return m_job_state_; }

    void RecoverEdge(TreeEdge* p_tree_edge);

    int UnConfigureEdge(TreeEdge* p_tree_edge);

    int ConfigureEdge(TreeEdge* p_tree_edge);

    void Recover();

    // return true if lock available, false otherwise.
    bool CheckExclusiveLockAvailability(bool is_sat_job, bool is_root = false);
    bool IsLockAvailable(bool is_sat_job, bool is_root = false);

    /**
     * UnConfigure - unconfigure QPs on tree node and it neighbors.
     * Disable QP, remove QP from tree and deallocate QP.
     * Function is called when a node state changed to the INACTIVE state.
     * @return
     */
    int UnConfigure(bool is_config_local, bool is_config_remote, bool is_config_parent, bool is_config_descendants);

    /**
     * Configure - configure QPs on tree node and it neighbors.
     * Allocate QPs, enable QP and add to tree.
     * Function is called when a node state changed to the ACTIVE state and if no pending jobs
     * or after restart and last pending job trigger the function.
     * @return
     */
    int Configure();

    /**
        Disables TreeNode: TreeConfig MAD, WRITE_TREE opcode, tree_state DISABLED
    */
    void TreeConfigDisable(int* p_operation_status);

    void TryRecoverTree();

    int AssignRestoredQPsToTreeEdges();

    void RestoreAggPaths(int& operation_status);

    bool IsAnyTreeConfigExistOnTreeNode(bool check_qps_only = false);

    void AllocateQpsCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void AddQpToTreeCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void ConfigureTreeNodeCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void TreeConfigDisableCallback(FabricProviderCallbackContext* callback_context, int res, void* p_data);

    void SetChildIndexValue(uint8_t child_index) { m_child_index_ = child_index; }

   private:
    int SetParent(TreeNode* p_tree_node) { return m_parent_edge_.SetTreeNode(this, p_tree_node); }

    inline void DumpTreeNode(std::string& dump_message, const char* prefix, const AggNodeFabricInfo* p_agg_node) const;

    inline void DumpTreeNode(FILE* f, const char* prefix, const AggNodeFabricInfo* p_agg_node) const;

    void RemoveQp(int* p_operation_status, TreeEdge* tree_edge, bool is_parent);

    bool IsConfigOnFabricRequired(TreeEdge* p_tree_edge, bool check_path_is_valid);
};

bool TreeEdge::IsConfigOnFabricRequiredNoLogs(bool check_path_is_valid) const
{
    return ((GetRemoteTreeEdge() != NULL) && (m_path_->IsValid() || !check_path_is_valid) && GetRemoteTreeNode()->GetNode()->IsValid());
}

#endif   // TREE_NODE_H_
