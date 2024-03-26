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

#ifndef JOB_MANAGER_H_
#define JOB_MANAGER_H_

#include "agg_tree.h"
#include "agg_types.h"
#include "parallel_tree_finder.h"
#include "reservation_manager.h"
#include "sharp_job.h"

#define MAX_SHARP_RAILS_NUM 4

typedef std::set<SharpJob*> SetJobPtr;
typedef std::pair<SetJobPtr::iterator, bool> SetJobPtrInsertRes;
typedef std::map<SharpExtJobId, SharpJob*> ExtJobIdToJob;
typedef std::pair<SharpExtJobId, SharpJob*> ExtJobIdToJobPair;
typedef std::map<sharp_job_id_t, SharpJob*> SharpJobIdToJob;
typedef std::list<SharpJob*> ListJobs;
typedef std::map<string, ListJobs> ListOfJobsPerReservationKey;   // Each reservation_key can have multiple jobs
typedef std::map<int, SharpJob*> ConnectionIdToJob;
typedef std::list<sharp_job_id_t> ListJobIds;

class JobManager
{
    friend class ReservationManager;
    friend class JobManagerStartJobTests_InitializeJobManager_MakeSureThatMembersWereInitializedCorrectly_Test;

    ParallelTreeFinder m_tree_finder_;
    ReservationManager& m_reservation_manager_;

    // Containers of current jobs
    ExtJobIdToJob m_jobs_;
    SharpJobIdToJob m_sharp_job_id_to_job_;
    ConnectionIdToJob m_connection_id_to_job_;
    ListOfJobsPerReservationKey m_reservation_key_list_of_jobs;   // Hash table. Key is reservation_key, value is a list of jobs operating
                                                                  // under that reservation_key

    // List of available ids (that can still be used)
    ListJobIds m_available_job_ids_;
    CommandManager* m_command_manager_;

    sharp_job_id_t m_max_jobs_number_;
    bool m_is_init_ = false;

    std::unique_ptr<file_utils::DumpFile> m_resource_alloc_dump_file_;

   public:
    JobManager(CommandManager* p_command_manager, ReservationManager& reservation_manager)
        : m_reservation_manager_(reservation_manager),
          m_command_manager_(p_command_manager),
          m_max_jobs_number_(0),
          m_resource_alloc_dump_file_{file_utils::GetDumpFileIfEnabled("Resource Alloc", "sharp_am_resource_alloc.dump")}
    {}

    ~JobManager();

    void Init(const sharp_job_id_t max_jobs_number);
    void InitTreeFinder() { m_tree_finder_.Init(); };
    void ReCreateTreeFinderTasks() { m_tree_finder_.ReCreateTasks(); };
    inline std::size_t GetJobsCount() const { return m_jobs_.size(); };

    void BeginJob(const sharp_begin_job& begin_job_data, const smx_ep* ep, uint64_t tid);
    void EndJob(const sharp_end_job& end_job_data);
    void ConnectionDisconnected(int conn_id);
    void RefreshJobsConnection();
    void StartJobReconnection(const bool should_reconnect_to_clients = false);
    void CheckJobsActiveness(const bool should_reconnect_to_clients = false);
    void JobInfoReplyReceived(const sharp_mgmt_job_info& job_info);

    SharpJob* FindJob(uint64_t client_job_id, uint64_t reservation_id);
    const SharpJob* GetSharpJob(sharp_job_id_t sharp_job_id) const;
    SharpJob* GetSharpJobMod(sharp_job_id_t sharp_job_id);

    void HandleJobError(sharp_job_id_t sharp_job_id, const smx_ep* ep);

    void GetJobsOnTree(SetJobPtr& job_ids, sharp_trees_t tree_id);
    void GetJobsOnSwitch(SetJobPtr& job_ids, uint64_t agg_node_guid);
    void GetJobsOnLink(SetJobPtr& job_ids, uint64_t agg_node_guid, uint32_t port_num);
    // const ListJobs* GetJobsByReservationId(const char* reservation_key) const;
    void GetJobsByReservationKey(SetJobPtr& job_ids, const char* reservation_key) const;

    int RestoreFromNetworkAllJobIdsOfRunningJobs();
    bool IsJobFileSupportSeamlessRestart(persistent_job_info* data);
    int UpdateRunningJobsData(persistent_job_info& data, string job_file_path);
    int RestoreFromNetworkAllJobQuotasOfRunningJobs();
    int RestoreFromNetworkTreeNodesConfigurations();
    int UpdateJobsWithRestoredJobsData();
    int UpdateRunningJobsWithTreesInfo();
    int UpdateRunningJobsQuota();
    int CreateJobTreeNodes();
    int ReconstructJobTrees();
    int AddRestoredJobsToTreeNodes();
    int UpdatePendingTreeEdgesOnRunningJobs();
    void CleanRestoredJobs();
    void CleanJobsDataRestoredDuringSeamlessRestart();
    void CleanEndedJobs();
    SharpJob* CreateJobFromPersistentJobInfo(persistent_job_info* job_info);

    int PublishAmAddress();
    void FreeJobId(const sharp_job_id_t sharp_job_id);

    void RemoveJobIdFromPool(const sharp_job_id_t sharp_job_id);

    void HandleJobInfoRequest(const sharp_jobs_request& job_info_request, const smx_ep* ep, uint64_t tid);

    uint32_t GetInternalJobIdForJob(const uint64_t external_job_id) const;

    // Provide a way to execute a callback on all jobs
    inline void ExecCallbackOnJobs(const std::function<void(const ExtJobIdToJobPair&)>& callback)
    {
        std::for_each(m_jobs_.begin(), m_jobs_.end(), callback);
    }

    void CleanJob(SharpJob* job, bool gen_job_end_event = true);

   private:
    SharpJob const* GetSharpJobFromExternalJobId(const uint64_t external_job_id) const;

    sharp_am_status BeginJobInternal(SharpJob* p_job, const sharp_begin_job& begin_job_data, sharp_job_data& job_data);

    SharpJob* AllocateJob(const sharp_begin_job& begin_job_data,
                          const smx_ep* ep,
                          sharp_am_status& status,
                          const uint64_t job_key,
                          sharp_job_data& job_data);

    void ReCleanJob(SharpJob* job);
    void FreeJob(SharpJob*& p_job);

    sharp_job_id_t AllocateJobId();

    sharp_am_status GetJobResource(JobResource& job_resource, const sharp_begin_job& begin_job_data, SharpJob* p_job);
    bool CreateJobConnection(SharpJob* const job, const bool should_disconnect_before_creating_connection = false);
    void EndJobConnection(SharpJob* const job);
    bool IsPeriodicRefreshDisabled();
};
#endif   // JOB_MANAGER_H_
