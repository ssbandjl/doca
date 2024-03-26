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

#ifndef COMMAND_MANAGER_H_
#define COMMAND_MANAGER_H_

#include <list>
#include <map>
#include <memory>

#include "agg_node_fabric_info.h"
#include "agg_types.h"
#include "fabric_update.h"
#include "fd_event_listener.h"
#include "job_manager.h"
#include "reservation_manager.h"
#include "smx/smx_api.h"
#include "smx/smx_types.h"
#include "thread_pool.h"
#include "traps.h"

class MsgContext;
using ContextList = std::list<MsgContext*>;

class MsgContext
{
    sharp_msg_type m_msg_type;
    sharp_job_id_t m_job_id;
    ContextList::iterator context_list_it_;

   public:
    MsgContext(sharp_msg_type msg_type, sharp_job_id_t job_id) : m_msg_type(msg_type), m_job_id(job_id) {}

    virtual ~MsgContext(){};

    sharp_msg_type GetMsgType() const { return m_msg_type; }

    sharp_job_id_t GetJobId() const { return m_job_id; }

    void SetContextListIterator(ContextList::iterator context_list_it) { context_list_it_ = context_list_it; }
    ContextList::iterator GetContextListIterator() const { return context_list_it_; }
};

class MsgTask : public ThreadPoolTask
{
    const smx_connection_info m_conn_info;
    const smx_ep m_ep;
    sharp_smx_msg* m_msg;
    sharp_msg_type m_type;
    CommandManager* m_manager;

   public:
    MsgTask(CommandManager* manager, const smx_connection_info& conn_info, const smx_ep& ep, sharp_smx_msg* msg, sharp_msg_type type)
        : m_conn_info(conn_info), m_ep(ep), m_msg(msg), m_type(type), m_manager(manager)
    {}

    virtual ~MsgTask(){};

    virtual void Run();

    virtual void Finalize();
};

class ControlMsgTask : public ThreadPoolTask
{
    int m_conn_id;
    sharp_control_type m_type;
    MsgContext* m_context;
    CommandManager* m_manager;

   public:
    ControlMsgTask(CommandManager* manager, int conn_id, sharp_control_type type, MsgContext* context)
        : m_conn_id(conn_id), m_type(type), m_context(context), m_manager(manager)
    {}

    virtual ~ControlMsgTask(){};

    virtual void Run();

    virtual void Finalize();
};

class CommandManager
{
   public:
    enum State
    {
        STARTUP_STATE = 0,
        INIT_STATE,
        PENDING_STATE,
        BUILD_TREES_STATE,
        READY_STATE
    };

    enum TaskQueueIndex
    {
        CLIENT_SIDE_TASK_QUEUE,
        UNIX_DOMAIN_SIDE_TASK_QUEUE
    };

    static constexpr std::size_t am_task_queues_size = 2;

   protected:
    typedef std::set<MsgContext*> MsgContexts;

    JobManager m_job_manager_;
    TrapsQueue m_traps_queue_;
    FabricUpdateList m_fabric_updates_;
    FabricGraphUpdateList m_fabric_graph_updates_;
    ReservationManager m_reservation_manager_;

    // we store contexts simply to delete them during teardown in case the callback (HandleControlTaskMessage) did not receive them
    ContextList m_context_list_;

    bool m_smx_started_;
    bool m_suppress_errors_;
    enum State m_state_;

    bool m_started_;
    TimerEvent* m_job_info_polling_timer_;
    TimerEvent* m_pending_mode_timeout_;
    TimerEvent* m_try_recover_trees_timer_;
    TimerEvent* m_job_reconnection_timer_;
    TimerEvent* m_refresh_keepalive_connections_timer_;
    FdEventListener* m_pending_mode_listener_;
    FdEventListener* m_try_recover_trees_listener_;
    FdEventListener* m_job_reconnection_listener_;
    FdEventListener* m_refresh_keepalive_connections_listener_;

    ThreadPool<am_task_queues_size>* m_thread_pool_;
    int m_is_seamless_restart_;   // 1 - error, 0 - OK

    void TurnOnSuppressErrors() { m_suppress_errors_ = true; }
    void TurnOffSuppressErrors() { m_suppress_errors_ = false; }

   public:
    CommandManager()
        : m_job_manager_(this, m_reservation_manager_),
          m_traps_queue_(*this),
          m_fabric_updates_(),
          m_fabric_graph_updates_(),
          m_reservation_manager_(this),
          m_context_list_(),
          m_smx_started_(false),
          m_suppress_errors_(false),
          m_state_(STARTUP_STATE),
          m_started_(false),
          m_job_info_polling_timer_(NULL),
          m_pending_mode_timeout_(NULL),
          m_try_recover_trees_timer_(NULL),
          m_job_reconnection_timer_(NULL),
          m_refresh_keepalive_connections_timer_(NULL),
          m_pending_mode_listener_(NULL),
          m_try_recover_trees_listener_(NULL),
          m_job_reconnection_listener_(NULL),
          m_refresh_keepalive_connections_listener_(NULL),
          m_thread_pool_(NULL),
          m_is_seamless_restart_(0)
    {
        g_fabric.SetCommandManager(this);
    }

    ~CommandManager();

    static inline const char* StateToChar(State state)
    {
        switch (state) {
            case STARTUP_STATE:
                return ("STARTUP state");
            case INIT_STATE:
                return ("INIT state");
            case PENDING_STATE:
                return ("PENDING state");
            case BUILD_TREES_STATE:
                return ("BUILD TREES state");
            case READY_STATE:
                return ("READY state");
            default:
                return ("UNKNOWN state");
        }
    }

    void Start(sharp_job_id_t max_jobs_number);

    bool IsStarted() const { return m_started_; }

    int CreatePersistentDirectory() const;

    JobManager& GetJobManager() { return m_job_manager_; }

    void HandleCommand(const smx_connection_info* conn_info, const smx_ep* ep, sharp_smx_msg* msg, sharp_msg_type type);

    void HandleCommandTaskMessage(const smx_connection_info* conn_info, const smx_ep* ep, sharp_smx_msg* msg, sharp_msg_type type);

    void HandleControlMessage(int conn_id, enum sharp_control_type type, MsgContext* context);

    void HandleControlTaskMessage(int conn_id, enum sharp_control_type type, std::unique_ptr<MsgContext> context);

    int SmxChangePort(uint64_t port_guid);

    void ProtectContextMessage(MsgContext* context);

    int SendBeginJobReply(sharp_job_data& job_data, sharp_am_status status, const smx_ep* ep, uint64_t tid);

    void SendEventStList(const smx_ep* p_ep, struct sharp_event_list& event_list, uint64_t tid) const;

    void SendResourceMessage(const smx_ep* p_ep, struct sharp_resource_message& resource_data, uint64_t tid) const;

    int SendJobError(uint64_t external_job_id,
                     uint32_t sharp_job_id,
                     const char* reservation_key,
                     enum sharp_error_value error,
                     enum sharp_error_type type,
                     bool suppress_error,
                     const smx_ep* ep,
                     MsgContext* msg_context = NULL);

    int SendJobError(sharp_job_error& job_error_message, const smx_ep* ep, MsgContext* msg_context = NULL);

    void HandleJobError(uint64_t external_job_id,
                        uint32_t sharp_job_id,
                        const char* reservation_key,
                        enum sharp_error_value error,
                        enum sharp_error_type type,
                        bool suppress_error,
                        const smx_ep* ep,
                        MsgContext* context = NULL);

    void SendSignal(smx_ep* p_ep, uint64_t flags) const;

    void SendSignal(smx_ep* p_ep, struct sharp_am_signal* p_am_signal_message = NULL) const;

    int SendJobInfoRequest(const smx_ep* ep, sharp_mgmt_job_info_list_request* p_request_message = NULL) const;

    void SendEndJob(smx_ep* p_ep, const SharpJob* sharp_job);

    void SendReservationInfoList(const smx_ep* p_ep, struct sharp_reservation_info_list& reservation_info_message, uint64_t tid) const;

    void SendTopologyInfo(const smx_ep* p_ep, struct sharp_topology_info_list& topology_info_message, uint64_t tid) const;

    void SendJobInfoList(const smx_ep* p_ep, struct sharp_jobs_list& job_info_message, uint64_t tid) const;

    void AddFabricUpdates(const FabricTopologyInfo& topology_info,
                          ListPortDataUpdate& ports_update,
                          ListPathUpdate& paths_update,
                          port_key_t port_key,
                          uint64_t epoch)
    {
        // AddUpdates clears ports_update and paths_update
        m_fabric_updates_.AddUpdates(topology_info, ports_update, paths_update, port_key, epoch);
    }

    void AddFabricGraphUpdate(FabricGraphPortDataUpdate& port_update) { m_fabric_graph_updates_.AddUpdate(port_update); }

    void GetFabricGraphUpdates(ListFabricGraphPortDataUpdate& port_data_updates) { m_fabric_graph_updates_.GetUpdates(port_data_updates); }

    inline void HandleAggNodeUpdate(const PortInfo& port_info, bool update_val)
    {
        m_traps_queue_.HandleAggNodeUpdateArray(port_info, update_val);
    }

    void CheckAndHandleUnfinishedJobsData();

    void IsPendingModeEnabled();
    int ExecuteSeamlessRestartFlow(bool& seamless_restart_failed_before_jobs_creation);

    bool IsAmInStartupState() const;

    bool IsAmInInitState() const;

    bool IsAmInPendingState() const;

    bool IsAmInReadyState() const;

    bool IsAmFinishedInitState() const;

    int OpenCommandsConnection();

    static void JobInfoIntervalTimer(const void* delegate, void* context);

    static void PendingModeTimeOutTimer(const void* delegate, void* context);

    static void TryRecoverTreesTimer(const void* delegate, void* context);

    static void JobsReconnectionTimer(const void* delegate, void* context);

    static void RefreshKeepaliveConnectionsTimer(const void* delegate, void* context);

    void JobReconnectionTimerStop() const;

    bool IsValidStateChange(State state);

    int SetAmState(State state);

    void SmxStart();

    inline int GetIsSeamlessRestart() { return m_is_seamless_restart_; }

    inline void SetIsSeamlessRestart(const int new_val) { m_is_seamless_restart_ = new_val; }

    void StartJobReconnection();

    int SetJobReconnectionTimer();

    void DestroyJobReconnectionTimer();

    bool IsJobReconnectionTimerStarted();

    void PrintSeamlessRestartStatusToLog();
    std::string GetSeamlessRestartFailure();
    void PrintSeamlessRestartFailureToLog();

    // We need these methods only in order to enable Fabric object to invoke the methods
    // on reservation manager object
    void PrepareReservationResourceLimit(const AggTree* agg_tree) { m_reservation_manager_.PrepareReservationResourceLimit(agg_tree); }
    void PrepareReservationResourceWithoutLimit(uint8_t max_sw_rank)
    {
        m_reservation_manager_.PrepareReservationResourceWithoutLimit(max_sw_rank);
    }
    void ReadPersistentReservationInfoFiles() { m_reservation_manager_.ReadPersistentReservationInfoFiles(); }

   private:
    void PrintAndLogClientErrorDetails(sharp_smx_msg* msg) const;

    void SmxStop();

    int SmxSend(const smx_ep* ep, sharp_smx_msg* p_msg, sharp_msg_type msg_type) const;

    int SmxAsyncSend(const smx_ep* ep, sharp_smx_msg* p_msg, sharp_msg_type msg_type, const MsgContext* msg_contex = NULL) const;

    int SmxGetLocalEp(smx_ep* p_ep) const;

    int ChechIfDirExists(const string& dir_name) const;

    int BuildDataFromJobFile(const char* job_file, persistent_job_info*& data) const;

    void CreateJobInfoFile(string& file_path, const sharp_job_data& job_data, const smx_ep* ep) const;

    void SetPendingModeTimers();

    void SetTryRecoverTreesTimer();

    void SetRefreshKeepaliveConnectionsTimer();

    int CreatePendingJobs(bool seamless_restart);

    bool IsUnfinishedJobsFilesExist();

    void EndPendingMode();

    void DestroyPendingModeTimers();

    void PendingModeTimeOut();

    void SendJobErrorToJobs();

    bool ValidatePrivilegedConnection(const smx_connection_info* conn_info, sharp_msg_type type);

    void HandleMessageInStartupState(int conn_id, const smx_ep* ep, sharp_smx_msg* msg, sharp_msg_type type);
    void HandleMessageInInitState(int conn_id, const smx_ep* ep, sharp_smx_msg* msg, sharp_msg_type type);
    void HandleMessageInPendingState(int conn_id, const smx_ep* ep, sharp_smx_msg* msg, sharp_msg_type type);
    void HandleMessageInReadyState(int conn_id, const smx_ep* ep, sharp_smx_msg* msg, sharp_msg_type type);

    void DeleteJobFile(char* p_file_full_path) const;

    void HandleTopologyRequest(const sharp_topology_info_request& topology_info_request, const smx_ep* ep, uint64_t tid);

    void HandleResourceAggTrees(sharp_resource_message& resource_data, SetJobPtr& jobs, sharp_resource_message& req_resource_data);
    void HandleResourceAggNodes(sharp_resource_message& resource_data, SetJobPtr& jobs, sharp_resource_message& req_resource_data);
    void HandleResourceLinks(sharp_resource_message& resource_data, SetJobPtr& jobs, sharp_resource_message& req_resource_data);

    void HandleResourceRequest(sharp_resource_message& resource_priority_data, const smx_ep* ep, uint64_t tid);

    sharp_topology_info_list GetTopologyInfo(const sharp_topology_info_request& topology_info_request);
};
#endif   // COMMAND_MANAGER_H_
