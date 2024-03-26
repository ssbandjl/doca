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

#pragma once

#include "am_common.h"
#include "dump_file.h"
#include "sub_tree_score.h"

struct FabricProviderCallbackContext;
struct AggNodeQuotaAllocInfo;
struct JobData;
class TreeEdge;
class AggNode;
class ReservationManager;
class ParallelTreeFinder;

using SetTreeEdgePtr = std::set<TreeEdge*>;
using ListTreeIds = std::list<sharp_trees_t>;
using JobSubTreeInfoVec = std::vector<JobSubTreeInfo>;
using MapAnToQuotaAllocInfo = std::map<AggNodeFabricInfo*, AggNodeQuotaAllocInfo>;
using MapAnToQuotaAllocInfoInsertRes = std::pair<MapAnToQuotaAllocInfo::iterator, bool>;
using VectorANFabricInfoAndANQuotaInfoPair = std::vector<std::pair<AggNodeFabricInfo const*, AggNodeQuotaAllocInfo const*>>;
using MapPortKeyToQuota = std::map<string, sharp_quota>;

// ports set indexed by rail id
using VecSetPortData = std::vector<SetPortDataConstPtr>;
using VecVecPortData = std::vector<VecPortDataConstPtr>;

struct AggNodeQuotaAllocInfo
{
    Quota m_quota;
    bool m_is_configured;
    ListTreeIds m_tree_ids;        // list of tree_ids on this node
    ListTreeIds m_root_tree_ids;   // list of tree_ids that are root on this node
    bool m_prevent_lock;           // prevent lock on node (unless used by SAT)

    AggNodeQuotaAllocInfo() : m_is_configured(false), m_prevent_lock(true) { m_quota = Quota(); }
};

struct SharpExtJobId
{
    uint64_t job_id;
    uint64_t reservation_id;
    std::string reservation_key;

    bool operator<(const SharpExtJobId& rhs) const
    {
        return job_id < rhs.job_id || (job_id == rhs.job_id && reservation_id < rhs.reservation_id);
    }
};

class SharpJob
{
    ReservationManager& m_reservation_manager_;
    ParallelTreeFinder& m_tree_finder_;

    SharpExtJobId m_external_job_id_;   // external job Id
    sharp_job_id_t m_sharp_job_id_;

    uint8_t m_priority_;

    bool m_multicast_enabled_;
    bool m_job_deleted_at_reservation_removal;
    string m_job_file_path;

    // Reservation resources that are used by this job
    ResourceLimitValueByRank m_resource_limit_allocated_by_rank;

    // Job may contain hosts list or ports list
    SetHostInfoPtr m_hosts_;
    VecSetPortData m_ports_by_rail_;

    JobSubTreeInfoVec m_sub_tree_info_vec_;
    MapAnToQuotaAllocInfo m_an_to_quota_alloc_info_;

    uint8_t m_trees_number_;
    SetTreeIds m_tree_ids_;
    SetAggNodeFabricPtr m_agg_nodes_;

    // DOTO remove m_num_channels_per_conn_ from SharpJob
    uint8_t m_num_channels_per_conn_;

    // tree edges waiting for this job to end before it can be repaired
    SetTreeEdgePtr m_pending_tree_edges_;

    const smx_ep m_sharpd_ep_;   // sharpd_0 address

    // sharpd_0 connection id, used for keep-alive
    // when keep-alive is enabled we open connection to sharpd_0 until the job ends
    // During the job keep-alive checks if the connection is still alive
    int m_sharpd_conn_id_;
    SharpTimestamp m_job_info_send_time;
    bool m_job_info_reply_received;

    bool m_reproducible;

    uint64_t m_job_key_;

    sharp_job_state m_job_state_;
    bool m_configure_fp19_;
    bool m_configure_bfloat19_;

    // True when the client request that rmc will be supported by every tree in this Job
    bool m_rmc_supported_on_all_trees_ = false;

    uint8_t m_req_num_trees_;

    file_utils::DumpFile* m_resource_alloc_dump_file_ptr_;

    int FindCoalescingRails(const sharp_begin_job& begin_job_data,
                            vector<uint8_t>& coalescing_rails,
                            VecVecPortData& vec_rail_to_vec_port_data,
                            uint32_t& num_guids_per_rail);

    sharp_am_status CoalesceRails(VecVecPortData& vec_rail_to_vec_port_data, uint32_t first_index, uint32_t second_index);

    sharp_am_status CoalesceValidRails(VecVecPortData& vec_rail_to_vec_port_data, vector<uint8_t>& coalescing_rails);

    void SetJobPriority(uint8_t priority);

   public:
    SharpJob(ReservationManager& reservation_manager,
             ParallelTreeFinder& tree_finder,
             SharpExtJobId external_job_id,
             sharp_job_id_t sharp_job_id,
             bool enable_mcast,
             const smx_ep* ep,
             uint64_t job_key,
             file_utils::DumpFile* resource_alloc_dump_file);

    void SetJobFilePath(string job_file_path) { m_job_file_path = job_file_path; }
    int AddHostInfo(const char* hostname);

    sharp_am_status AnalizeRailsInfo(const sharp_begin_job& begin_job_data);

    int SetTreesSpanningEndpoints(VecPortDataConstPtr& vec_port_data, uint8_t rail);

    static int ParseHostlistCallback(const char* hostname, void* arg);

    sharp_am_status Init(const sharp_begin_job& begin_job_data);

    string GetName() const;
    const SharpExtJobId& GetExternalJobId() const { return m_external_job_id_; }
    sharp_job_id_t GetSharpJobId() const { return m_sharp_job_id_; }

    const string& GetReservationKey() const { return m_external_job_id_.reservation_key; }
    const char* GetReservationKeyCharPtr() const { return m_external_job_id_.reservation_key.c_str(); }

    // Increase count at rank and return the new value
    unsigned int IncreaseReservationResourcesCountAtRank(unsigned int rank)
    {
        m_resource_limit_allocated_by_rank[rank]++;
        return m_resource_limit_allocated_by_rank[rank];
    }

    // Get value
    unsigned int GetReservationResourcesCountAtRank(unsigned int rank) const { return m_resource_limit_allocated_by_rank[rank]; }

    // Reset all values to zero
    void ResetReservationResourcesCount()
    {
        std::fill(m_resource_limit_allocated_by_rank.begin(), m_resource_limit_allocated_by_rank.end(), 0);
    }

    uint8_t GetJobPriority() { return m_priority_; }
    const smx_ep* GetSharpdZeroAddress() const { return &m_sharpd_ep_; }
    int GetSharpdConnId() const { return m_sharpd_conn_id_; }
    void SetSharpdConnId(int sharpd_conn_id) { m_sharpd_conn_id_ = sharpd_conn_id; }

    bool IsMulticastEnabled() { return m_multicast_enabled_; }

    bool IsDeletedAtReservationRemoval() { return m_job_deleted_at_reservation_removal; }

    void MarkDeletedAtReservationRemoval() { m_job_deleted_at_reservation_removal = true; }

    void UnmarkDeletedAtReservationRemoval() { m_job_deleted_at_reservation_removal = false; }

    void DisableMulticast() { m_multicast_enabled_ = false; }

    JobSubTreeInfo* GetJobSubTreeInfoForTreeId(uint16_t tree_id);

    void ResizeSubTreeInfoVec(uint8_t num_trees) { m_sub_tree_info_vec_.resize(num_trees); }

    JobSubTreeInfo& GetSubTreeInfo(uint8_t tree_number) { return m_sub_tree_info_vec_[tree_number]; }
    const JobSubTreeInfo& GetSubTreeInfo(uint8_t tree_number) const { return m_sub_tree_info_vec_[tree_number]; }
    // get the next unused SubTreeInfo
    JobSubTreeInfo& GetSubTreeInfo() { return m_sub_tree_info_vec_[m_trees_number_]; }

    int ClearJobSubTreeInfo(sharp_trees_t tree_id);

    bool IsReproducible() const { return m_reproducible; }

    void SetJobState(sharp_job_state state);

    sharp_job_state GetJobState() const { return m_job_state_; }

    void CommitSubTreeInfo(const JobSubTreeInfo& job_sub_tree_info, const SetAggNodeFabricPtr& agg_nodes);

    uint8_t GetTreesNumber() const { return m_trees_number_; }
    const SetTreeIds& GetTreeIds() const { return m_tree_ids_; }
    const SetAggNodeFabricPtr& GetAggNodes() const { return m_agg_nodes_; }

    u_int32_t GetHostNumber() const { return (u_int32_t)m_hosts_.size(); }
    const SetHostInfoPtr& GetHosts() { return m_hosts_; }
    const SetPortDataConstPtr& GetPorts(uint8_t rail) const { return m_ports_by_rail_[rail]; }
    uint8_t GetNumberOfRails() const { return m_ports_by_rail_.size(); }
    uint32_t GetPortsNumberForRail(uint8_t rail);
    uint32_t GetPortsNumberForAllRails();
    u_int32_t GetConnectionsNumber() const;
    bool IsJobInfoSent() const { return m_job_info_send_time != SharpTimestamp::min(); }
    const SharpTimestamp& GetJobInfoTimeStamp() const { return m_job_info_send_time; }
    bool IsJobInfoReplyReceived() const { return m_job_info_reply_received; }
    void SetJobInfoSendTimeStamp(bool job_info_send);
    void SetJobInfoReplyReceived(bool job_info_reply_received) { m_job_info_reply_received = job_info_reply_received; }

    sharp_am_status FabricQuotaConfig();
    void FabricJobResourceCleanup();
    void FabricJobResourceCleanupV2();
    void DisconnectSatJobFromTrees(bool cleanup_v2 = false);
    void CleanJobSatTrees(bool cleanup_v2 = false);
    void CleanSharpJob(bool cleanup_v2 = false);
    void CleanJobTreeConfigOnFabric();
    void DisableJobTreeNodes();

    void AddPendingTreeEdge(TreeEdge* p_tree_edge);

    void HandlePendingTreeEdges();

    void GetSubTreesBFS(ListTreeNodePtr* sub_tree_nodes_list);
    void GetSubTreeNodesBFS(ListTreeNodePtr* sub_tree_nodes_list, struct JobSubTreeInfo& sub_tree_info);
    void CreateJobInfoFile(const smx_ep* ep, const string& persistent_path);
    void DeleteJobInfoFile();

    uint8_t GetNumChannelsPerConn() const { return m_num_channels_per_conn_; }
    void AddTreeId(sharp_trees_t tree_id);
    void AddAggNode(AggNodeFabricInfo* p_agg_node);
    int SetTreeRootForJob();
    int SetQuotaAllocForJob();
    bool IsExclusiveLockUsedByTree(sharp_trees_t tree_id);
    int CopyInfoFromJobData(const JobData* job_data);

    int AllocateQuota(JobSubTreeInfo& job_sub_tree_info, uint8_t child_index_per_port);

    void FreeQuota();
    void FreeJobResource();
    int AllocateSat(JobSubTreeScore& result, JobSubTreeInfo& job_sub_tree_info_llt);

    void ModifyAvailableTreeIDsInAllSubTrees(bool is_available);

    sharp_am_status CalcAndValidateReservationResources(const JobSubTreeInfo& job_sub_tree_info);
    void ApplyJobReservationResourcesChange(bool is_added_job);
    bool IsAnyTreeNodeOnJob();

    sharp_am_status AllocateJobResource(const JobResource& job_resource, const sharp_begin_job& begin_Job_data, uint8_t rail_number);

    void UpdateJobDataQpcOpts(sharp_job_data& job_data) const;
    void UpdateJobData(sharp_job_data& job_data);
    static void FreeJobData(sharp_job_data& job_data);
    void PrepareJobTreesInfoMessage(sharp_job_trees_info& job_info);
    int ReconstructTrees();
    int UpdateSubTreesInfo();
    void RestorePeerTreeIds(bool is_sat);

    void PrintResourceAllocationSummary();
    void CleanJobsDataRestoredDuringSeamlessRestart();

   private:
    VectorANFabricInfoAndANQuotaInfoPair GetANFabricInfoAndANQuotaInfoPairsSortedByPortRankAndGuid(
        const MapPortKeyToQuota& port_key_to_quota = {});
    void WriteResourcesAllocDetailsToFile(char const* const resource_msg_format,
                                          AggNodeFabricInfo const* const an_fabric_info,
                                          const uint32_t qps1,
                                          const uint32_t qps2,
                                          const uint32_t qps3,
                                          const uint32_t buffers1,
                                          const uint32_t buffers2,
                                          const uint32_t buffers3,
                                          const uint32_t osts1,
                                          const uint32_t osts2,
                                          const uint32_t osts3,
                                          const uint32_t groups1,
                                          const uint32_t groups2,
                                          const uint32_t groups3);
    int DumpJobResourceFree();
    int DumpJobResourceAllocation(const uint16_t tree_id, const MapPortKeyToQuota& port_key_to_quota);

    void SetComputePorts();

    ////////                 Callback          ///////////
    // m_data1 = int operation_status;
    // m_data2 = *AggNode
    // m_data3 = *AggNodeQuotaAllocInfo
    void JobResourceCleanupCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);
    // m_data1 = int operation_status;
    // m_data2 = *AggNode
    void DisconnectJobTreeQpCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);
    // m_data1 = int operation_status;
    // m_data2 = *AggNode
    void CleanJobTreeCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    // m_data1 = int operation_status;
    // m_data2 = *AggNode
    void SatCleanupCallback(FabricProviderCallbackContext* p_context,
                            int rec_status,
                            void* p_data,
                            const char* caller_func_name,
                            int next_state);

    // m_data1 = int operation_status;
    // m_data2 = *AggNode
    // m_data3 = *AggNodeQuotaAllocInfo
    void FabricQuotaConfigCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);

    // m_data1 = int operation_status;
    // m_data2 = *AggNode
    // m_data3 = *AggNodeQuotaAllocInfo
    void FabricTreeToJobBindCallback(FabricProviderCallbackContext* p_context, int rec_status, void* p_data);
    //////////////////////////////////////////////////////
};
