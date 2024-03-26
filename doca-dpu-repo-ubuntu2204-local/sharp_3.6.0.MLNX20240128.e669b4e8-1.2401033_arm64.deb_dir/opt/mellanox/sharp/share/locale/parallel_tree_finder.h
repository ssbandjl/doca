/*
 * Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef PARALLEL_TREE_FINDER_H
#define PARALLEL_TREE_FINDER_H

#include "agg_tree.h"
#include "dump_file.h"
#include "sub_tree_info.h"
#include "sub_tree_score.h"
#include "thread_pool.h"

class ParallelTreeFinder;

// A single task of tree finding, checking whether a single tree can match
class TreeFinderTask : public ThreadPoolTask
{
    AggTree& m_agg_tree_;
    ParallelTreeFinder& m_tree_finder_;

    JobSubTreeInfo m_job_sub_tree_info_;
    JobSubTreeScore m_sub_tree_result_;
    sharp_job_id_t m_sharp_job_id_;

   public:
    TreeFinderTask(AggTree& agg_tree, ParallelTreeFinder& tree_finder)
        : m_agg_tree_(agg_tree), m_tree_finder_(tree_finder), m_sharp_job_id_(0)
    {}

    virtual ~TreeFinderTask() {}

    inline JobSubTreeInfo& GetTaskJobSubTreeInfo() { return m_job_sub_tree_info_; }
    inline JobSubTreeScore& GetTaskJobSubTreeScore() { return m_sub_tree_result_; }

    sharp_trees_t GetTreeId() const { return m_agg_tree_.GetId(); }
    void SetSharpJobId(sharp_job_id_t sharp_job_id) { m_sharp_job_id_ = sharp_job_id; }

    virtual void Run();
};

using TreeFinderTasksVector = std::vector<TreeFinderTask>;

// Invoke method on all trees In parallel using multi threading
class ParallelTreeFinder : public ThreadPoolTasksCollection
{
    ThreadPool<default_task_queues_size> m_thread_pool_;
    TreeFinderTasksVector m_tasks_;

    // search data
    SetPortDataConstPtr const* m_compute_ports_;
    JobResource const* m_job_resource_;

    std::unique_ptr<file_utils::DumpFile> m_job_failures_dump_file_ptr_;

    void CreateTasks();

   public:
    ParallelTreeFinder()
        : ThreadPoolTasksCollection(),
          m_thread_pool_(),
          m_compute_ports_(nullptr),
          m_job_resource_(nullptr),
          m_job_failures_dump_file_ptr_{
              file_utils::GetDumpFileIfEnabled("Failed Job Requests", "sharp_am_failed_job_requests_details.dump")}
    {}

    ~ParallelTreeFinder();

    void Init();
    bool FindBestTree(const SetPortDataConstPtr& compute_ports,
                      const JobResource& job_resource,
                      const SharpJob& p_job,
                      const sharp_job_id_t job_id,
                      JobSubTreeInfo& best_tree_info,
                      JobSubTreeScore& best_tree_result);

    void DumpJobRequestFailure(const uint64_t job_id,
                               const uint64_t external_job_id,
                               JobSubTreeInfo const* const best_tree_info = nullptr,
                               char const* const failure_reason = nullptr);

    inline const SetPortDataConstPtr* GetComputePorts() { return m_compute_ports_; }
    inline const JobResource* GetJobResource() { return m_job_resource_; }

    int CreateLltTreeForJob(const SharpJob& p_job, JobSubTreeInfo& job_sub_tree_info);

    int CreateSatTreeForJob(const SharpJob& p_job,
                            const JobResource& job_resource,
                            JobSubTreeInfo& job_sub_tree_info,
                            JobSubTreeScore& sub_tree_result);

    void ReCreateTasks();
};

#endif   // PARALLEL_TREE_FINDER_H
