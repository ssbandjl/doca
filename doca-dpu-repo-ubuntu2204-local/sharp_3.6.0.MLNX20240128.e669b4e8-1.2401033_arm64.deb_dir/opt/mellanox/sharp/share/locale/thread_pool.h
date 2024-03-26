/*
 * Copyright (c) 2017-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <pthread.h>
#include <sys/sysinfo.h>   // get_nprocs()
#include <algorithm>
#include <array>
#include <cstddef>
#include <list>
#include <queue>

#include "agg_types.h"
#include "am_log.h"

#define DEFAULT_THREADPOOL_THREADS_NUMBER 5

typedef std::queue<class ThreadPoolTask*> TaskQueue;
typedef std::list<pthread_t> ListThreads;

static constexpr std::size_t default_task_queues_size = 1;

class ThreadPoolTask
{
    bool m_check_result_;

   public:
    // Run should not throw for error handling.
    // It should change the tasks state instead.
    virtual void Run() = 0;
    virtual ~ThreadPoolTask() {}

    void SetCheckResult(bool is_check_result) { m_check_result_ = is_check_result; }

    bool IsCheckResult() { return m_check_result_; }

    virtual void Finalize(){};
};

template <size_t ThreadPoolQueueSize = 1>
class ThreadPool
{
    using TaskQueue = std::queue<class ThreadPoolTask*>;

    size_t m_task_queues_index_;
    std::array<TaskQueue, ThreadPoolQueueSize> m_task_queues_;
    ListThreads m_threads_;
    bool m_stop_;
    bool m_init_;

    pthread_mutex_t m_queue_lock_;
    pthread_cond_t m_queue_cond_;

   public:
    ThreadPool() : m_task_queues_index_(0), m_stop_(false), m_init_(false) {}

    void Stop() { m_stop_ = true; }

    void AddTask(ThreadPoolTask* p_task)
    {
        static_assert(ThreadPoolQueueSize == 1, "This member method is available only if the size of the queue is 1");
        AddTaskImp(p_task, 0);
    }

    void AddTask(ThreadPoolTask* p_task, const size_t index)
    {
        static_assert(ThreadPoolQueueSize != 1, "This member method is available only if the size of the queue is more than 1");
        AddTaskImp(p_task, index);
    }

    ~ThreadPool()
    {
        if (!m_init_) {
            return;
        }

        pthread_mutex_lock(&m_queue_lock_);
        m_stop_ = true;
        pthread_mutex_unlock(&m_queue_lock_);

        // signal waiting threads so they could exit
        pthread_cond_broadcast(&m_queue_cond_);

        for (ListThreads::iterator iter = m_threads_.begin(); iter != m_threads_.end(); ++iter) {
            void* result;
            int rc = pthread_join(*iter, &result);
            if (rc) {
                ERROR("ThreadPool pthread_join() failed: %23"
                      "s",
                      strerror(errno));
            }
        }

        ThreadPoolTask* p_task;
        for (auto& queue : m_task_queues_) {
            while (!queue.empty()) {
                p_task = queue.front();
                queue.pop();
                p_task->Finalize();
            }
        }

        pthread_mutex_destroy(&m_queue_lock_);
        pthread_cond_destroy(&m_queue_cond_);
        INFO("ThreadPool destroyed\n");
    }

    int Init(uint16_t num_threads)
    {
        if (m_init_) {
            return 0;
        }

        int rc;

        rc = pthread_mutex_init(&m_queue_lock_, NULL);
        if (rc) {
            ERROR("ThreadPool failed to init mutex: %s\n", strerror(errno));
            return rc;
        }

        rc = pthread_cond_init(&m_queue_cond_, NULL);
        if (rc) {
            pthread_mutex_destroy(&m_queue_lock_);
            ERROR("ThreadPool failed to init condition variable: %s\n", strerror(errno));
            return rc;
        }

        m_init_ = true;

        if (num_threads == 0) {
            num_threads = get_nprocs();

            if (num_threads == 0) {
                WARNING("Failed to get number of available processors. "
                        "Using %u threads.\n",
                        DEFAULT_THREADPOOL_THREADS_NUMBER);
                num_threads = DEFAULT_THREADPOOL_THREADS_NUMBER;
            }
        }
        auto thread_run = [](void* arg) -> void*
        {
            static_cast<ThreadPool<ThreadPoolQueueSize>*>(arg)->ThreadRun();
            return nullptr;
        };
        for (uint16_t i = 0; i < num_threads; ++i) {
            pthread_t worker_thread;
            rc = pthread_create(&worker_thread, NULL, thread_run, this);

            if (rc != 0) {
                ERROR("Failed to create thread rc: %d\n", rc);
                return -1;
            }

            m_threads_.push_back(worker_thread);
        }

        INFO("ThreadPool init with %u threads\n", (uint16_t)m_threads_.size());

        return 0;
    }

    // the method executed by each thread
    void ThreadRun()
    {
        ThreadPoolTask* p_task = nullptr;

        DEBUG("Start handle ThreadPool tasks\n");

        while (true) {
            pthread_mutex_lock(&m_queue_lock_);
            while (!m_stop_) {
                const bool all_queues_empty =
                    std::all_of(m_task_queues_.begin(), m_task_queues_.end(), [](const TaskQueue& queue) { return queue.empty(); });
                if (!all_queues_empty) {
                    break;
                }
                pthread_cond_wait(&m_queue_cond_, &m_queue_lock_);
            }

            if (m_stop_) {
                pthread_mutex_unlock(&m_queue_lock_);
                break;
            }

            // choose next queue to handle
            // round-robin between indexes
            for (size_t i = 0; i < ThreadPoolQueueSize; ++i) {
                m_task_queues_index_ = (m_task_queues_index_ + 1) % ThreadPoolQueueSize;
                if (!m_task_queues_[m_task_queues_index_].empty()) {
                    break;
                }
            }

            p_task = m_task_queues_[m_task_queues_index_].front();
            m_task_queues_[m_task_queues_index_].pop();

            pthread_mutex_unlock(&m_queue_lock_);

            p_task->Run();
            p_task->Finalize();
        }

        DEBUG("Stop handle ThreadPool tasks\n");
    }

   private:
    void AddTaskImp(ThreadPoolTask* p_task, size_t task_queue_index)
    {
        pthread_mutex_lock(&m_queue_lock_);
        m_task_queues_[task_queue_index].push(p_task);
        pthread_cond_signal(&m_queue_cond_);

        pthread_mutex_unlock(&m_queue_lock_);
    }
};

class ThreadPoolTasksCollection
{
   private:
    uint16_t m_num_tasks_in_progress_;
    pthread_mutex_t m_tasks_lock_;
    pthread_cond_t m_tasks_cond_;

   protected:
    bool m_is_init_;

   public:
    ThreadPoolTasksCollection() : m_num_tasks_in_progress_(0), m_is_init_(false) {}

    ~ThreadPoolTasksCollection();

    void Init();

    void AddTaskToThreadPool(ThreadPool<default_task_queues_size>& thread_pool, ThreadPoolTask* p_task);

    void WaitForTasks();

    // this should be called before each tasks ends
    void OnTaskEnd();
};

#endif   // THREAD_POOL_H
