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

#ifndef _FABRIC_PROVIDER_H
#define _FABRIC_PROVIDER_H

#include <infiniband/ibis/memory_pool.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <memory>

#include "agg_types.h"
#include "event_manager.h"
#include "fd_event_listener.h"
#include "ibis.h"

// Status for CPI.Get which indicates that port does not support AM MADs.
// Value defined according to IB specification chapter 13.4.7 table 122.
#define SHARP_IB_STATUS_UNSUP_CLASS 0xC
#define SHARP_IB_STATUS_TIMEOUT     0x00FE

#define QPC_QP_STATE_DISABLED 0
#define QPC_QP_STATE_ACTIVE   1
#define QPC_QP_STATE_ERROR    2

#define TREE_CONFIG_INVALID_TREE_ID 0xFFFF
#define AN_ACTIVE_JOBS_MAX_INDEX    48   // Max index in ANActiveJob MAD

#define QPC_QP_TRANSPORT_TYPE_UD 3   // UD - Unreliable Datagram

template <typename _Type, void (_Type::*_Method)(const clbck_data_t&, int, void*)>
void ibis_forwardClbck(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data)
{
    (reinterpret_cast<_Type*>(clbck_data.m_p_obj)->*_Method)(clbck_data, rec_status, p_attribute_data);
}

struct configure_quota_flags
{
    bool prevent_lock;
    bool job_is_reproducible;
    bool fp19_en;
    bool bfloat19_en;
};

struct qp_on_tree
{
    bool is_parent;
    uint32_t qpn;
    uint32_t index;
    sharp_trees_t tree_id;
    bool is_root_qp;
};

struct sharp_quota;
struct PortInfo;
struct FabricProviderCallbackContext;
struct qp_record;

typedef std::list<uint32_t> QpList;
typedef std::pair<uint8_t, uint32_t> ChildQp;
typedef std::list<ChildQp> ChildQpList;
typedef std::vector<class AnAdapter*> VectorAnAdapterPtr;
typedef std::vector<qp_record> VectorQpRecord;
typedef std::vector<qp_on_tree> VectorQpOnTree;

typedef void (*FabricProviderCallbackDlg)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_data);

typedef void (*TrapHandlingCallback)(void* callback_context,
                                     SharpTrapNumberEnum trap_number,
                                     void* p_data,
                                     ib_address_t* p_ib_address,
                                     MAD_AggregationManagement* p_am_mad,
                                     Notice* p_notice);

// ENUMS
enum FabricProviderOpMode
{
    FP_OP_MODE_CLIENT_ONLY = 0,    // Send MADs only
    FP_OP_MODE_CLIENT_SERVER = 1   // Send MADs, listen on traps and IB events
                                   // and register SR
};

struct FabricProviderCallbackContext
{
    void* m_handler_obj;
    void* m_data1;
    void* m_data2;
    void* m_data3;
    void* m_data4;

    FabricProviderCallbackContext() : m_handler_obj(NULL), m_data1(NULL), m_data2(NULL), m_data3(NULL), m_data4(NULL) {}

    void Init()
    {
        m_handler_obj = NULL;
        m_data1 = NULL;
        m_data2 = NULL;
        m_data3 = NULL;
        m_data4 = NULL;
    }
};

typedef struct FabricProviderCallbackContext fabric_provider_callback_context_t;
template <typename _Type, void (_Type::*_Method)(fabric_provider_callback_context_t*, int, void*)>
void fabric_provider_forwardClbck(fabric_provider_callback_context_t* callback_context, int rec_status, void* p_attribute_data)
{
    (reinterpret_cast<_Type*>(callback_context->m_handler_obj)->*_Method)(callback_context, rec_status, p_attribute_data);
}

typedef void (*fabric_provider_handle_data_func_t)(fabric_provider_callback_context_t* callback_context,
                                                   int rec_status,
                                                   void* p_attribute_data);

class IbEvent : public FdEvent
{
   public:
    IbEvent(handle_cb* cb, const void* delegate, void* context, uint64_t port_guid, uint64_t events)
        : FdEvent(cb, delegate, context), m_port_guid_(port_guid), m_events_(events), m_port_(0), m_device_(NULL)
    {}

    ~IbEvent();

    int Init();
    void Handle() const;

   private:
    uint64_t m_port_guid_;
    uint64_t m_events_;

    int m_port_;
    ibv_context* m_device_;
};

class FabricProvider;

class AnAdapter
{
    enum sharp_error_syndrom
    {
        TRAP_SHARP_ERROR_LOCK_SEMAPHORE_TIMEOUT = 0,
        TRAP_SHARP_ERROR_OST_NO_PROGRESS_TIMEOUT = 1,
        TRAP_SHARP_ERROR_BAD_SAT_REQUEST_CLASSIFICATION = 2,
        TRAP_SHARP_ERROR_SAT_ANDR_GOT_MULTIPLE_TARGETS = 3,
        TRAP_SHARP_ERROR_SAT_WITH_TARGET_HEADER = 4,
        TRAP_SHARP_ERROR_SAT_NO_DESTINATIONS_ON_ROOT = 5,
        TRAP_SHARP_ERROR_SAT_EXCEEDS_NUMBER_OF_OUTSTANDING_OPERATIONS = 6,
        TRAP_SHARP_ERROR_SAT_UNBALANCED_FIFO_DATA = 7,
        TRAP_SHARP_ERROR_SAT_UNBALANCED_OST_ADDR = 8,
        TRAP_SHARP_ERROR_SAT_DATA_CORRUPTION = 9,
        TRAP_SHARP_ERROR_SAT_BAD_DATA_GRANULARITY = 10,
    };

    uint8_t m_version;

   public:
    AnAdapter(int version) : m_version(version) {}
    virtual ~AnAdapter() {}

    uint8_t GetVersion() { return m_version; }

    virtual int SendResourceCleanupRequest(FabricProvider* p_fabric_provider,
                                           clbck_data_t& ibis_clbck_data,
                                           const PortInfo* p_port_info,
                                           uint8_t opcode,
                                           sharp_job_id_t job_id,
                                           sharp_trees_t tree_id,
                                           uint64_t am_key,
                                           uint8_t& mad_cnt) = 0;

    virtual void HandleTrapQpError(TrapHandlingCallback traps_handling_callback,
                                   void* p_traps_callback_context,
                                   u_int8_t* p_notice_data_details,
                                   ib_address_t* p_ib_address,
                                   MAD_AggregationManagement* p_am_mad,
                                   Notice* p_notice) = 0;

    virtual void HandleTrapSharpInvalidRequest(TrapHandlingCallback traps_handling_callback,
                                               void* p_traps_callback_context,
                                               u_int8_t* p_notice_data_details,
                                               ib_address_t* p_ib_address,
                                               MAD_AggregationManagement* p_am_mad,
                                               Notice* p_notice) = 0;

    virtual void HandleTrapSharpError(TrapHandlingCallback traps_handling_callback,
                                      void* p_traps_callback_context,
                                      u_int8_t* p_notice_data_details,
                                      ib_address_t* p_ib_address,
                                      MAD_AggregationManagement* p_am_mad,
                                      Notice* p_notice);

    virtual void HandleTrap(TrapHandlingCallback traps_handling_callback,
                            void* p_traps_callback_context,
                            SharpTrapNumberEnum trap_number,
                            ib_address_t* p_ib_address,
                            MAD_AggregationManagement* p_am_mad,
                            Notice* p_notice);
};

class AnAdapterV1 : public AnAdapter
{
   public:
    AnAdapterV1() : AnAdapter(1) {}
    virtual ~AnAdapterV1() {}

    virtual int SendResourceCleanupRequest(FabricProvider* p_fabric_provider,
                                           clbck_data_t& ibis_clbck_data,
                                           const PortInfo* p_port_info,
                                           uint8_t opcode,
                                           sharp_job_id_t job_id,
                                           uint16_t,
                                           uint64_t am_key,
                                           uint8_t& mad_cnt);

    virtual void HandleTrapQpError(TrapHandlingCallback traps_handling_callback,
                                   void* p_traps_callback_context,
                                   u_int8_t* p_notice_data_details,
                                   ib_address_t* p_ib_address,
                                   MAD_AggregationManagement* p_am_mad,
                                   Notice* p_notice);

    virtual void HandleTrapSharpInvalidRequest(TrapHandlingCallback traps_handling_callback,
                                               void* p_traps_callback_context,
                                               u_int8_t* p_notice_data_details,
                                               ib_address_t* p_ib_address,
                                               MAD_AggregationManagement* p_am_mad,
                                               Notice* p_notice);
};

class AnAdapterV2 : public AnAdapter
{
   public:
    AnAdapterV2() : AnAdapter(2) {}
    virtual ~AnAdapterV2() {}

    virtual int SendResourceCleanupRequest(FabricProvider* p_fabric_provider,
                                           clbck_data_t& ibis_clbck_data,
                                           const PortInfo* p_port_info,
                                           uint8_t opcode,
                                           sharp_job_id_t job_id,
                                           sharp_trees_t tree_id,
                                           uint64_t am_key,
                                           uint8_t& mad_cnt);

    virtual void HandleTrapQpError(TrapHandlingCallback traps_handling_callback,
                                   void* p_traps_callback_context,
                                   u_int8_t* p_notice_data_details,
                                   ib_address_t* p_ib_address,
                                   MAD_AggregationManagement* p_am_mad,
                                   Notice* p_notice);

    virtual void HandleTrapSharpInvalidRequest(TrapHandlingCallback traps_handling_callback,
                                               void* p_traps_callback_context,
                                               u_int8_t* p_notice_data_details,
                                               ib_address_t* p_ib_address,
                                               MAD_AggregationManagement* p_am_mad,
                                               Notice* p_notice);
};

struct SrDeleter
{
    void operator()(struct sr_ctx* sr_context);
};

// This class provides the MAD to query/modify items on the fabric
// It is also refreshing SRs and handles traps from the fabric
class FabricProvider
{
    static const int sr_lease_time = 300; /* in seconds (minimum 10 secs) */

    VectorAnAdapterPtr m_vec_an_adapter_ptr_;

   public:
    FabricProvider(FabricProviderOpMode op_mode)
        : m_listener_thread_(),
          m_stop_listener_flag_(true),
          m_sr_lease_time_(sr_lease_time),
          m_sr_context_(nullptr),
          m_port_invalid_time{.tv_sec = 0, .tv_usec = 0},
          m_event_listener_started(false),
          m_traps_handling_callback_(NULL),
          m_traps_callback_context_(NULL),
          m_operation_mode_(op_mode){};

    ~FabricProvider();

    int Init();

    int ConfigureLocalPort(uint64_t port_guid);

    // Validate local port state
    // Return 0 if local port is valid. Non-zero if not valid.
    // Throws runtime exception if fails to retrieve local port information.
    void ValidateLocalPort();
    // Port validatiuon timeout
    // 1 - timeout accoured
    // 0 - not accoured
    int CheckPortValidationTimeout();
    // Reset timeout
    void ResetPortValidationTimeout();

    // Traps
    void RegisterTrapsHandler(TrapHandlingCallback callback, void* callback_context)
    {
        m_traps_handling_callback_ = callback;
        m_traps_callback_context_ = callback_context;
    }

    void StopTrapsHandling()
    {
        StopListener();
        m_traps_handling_callback_ = NULL;
        m_traps_callback_context_ = NULL;
    }

    static inline const char* AmTrapNumberToStr(uint16_t trap_number);

    void SendRepressMsg(ib_address_t* p_ib_address, MAD_AggregationManagement* p_am_mad, Notice* p_noice);

    int EventListenerStart(int (*get_address_cb)(uint8_t*, struct sr_addr_info*));
    int EventListenerStop();

    static void RegisterAddress(const void* delegate, void* context);

    void WaitForPendingTransactions();

    int SetAMKey(const AggPortInfo& agg_port_info,
                 void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                 FabricProviderCallbackContext* callback_context,
                 uint64_t am_key);

    // Try to recover configured AMKey
    int RecoverAMKey(const PortInfo& port_info,
                     void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                     FabricProviderCallbackContext* callback_context,
                     uint64_t am_key,
                     bool disable_overwrite_zero_am_key);   // will not replace am_key if current am_key is 0.

    // Discover aggregation node properties
    int DiscoverAn(const PortInfo& port_info,
                   void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                   FabricProviderCallbackContext* callback_context,
                   uint64_t am_key,
                   bool disable_overwrite_zero_am_key);   // will not replace am_key if current am_key is 0.

    // Bind TreeIds to JobId
    int BindTreeIdsToJob(const AggPortInfo& agg_port_info,
                         vector<uint16_t> tree_ids_to_bind,
                         int offset,
                         sharp_job_id_t job_id,
                         void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                         FabricProviderCallbackContext* callback_context);

    // Get binded TreeIds to JobId
    int SendTreeToJobBindingGet(const AggPortInfo& agg_port_info,
                                sharp_job_id_t job_id,
                                VecTreeIds tree_ids_to_check,
                                void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                                FabricProviderCallbackContext* callback_context);

    // Configure quota on an aggregation node
    int ConfigureQuota(const AggPortInfo& agg_port_info,
                       sharp_trees_t tree_id,
                       bool is_root,
                       sharp_job_id_t job_id,
                       uint32_t ud_qpn,
                       const Quota& quota,
                       uint64_t job_key,
                       configure_quota_flags flags,
                       void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                       FabricProviderCallbackContext* callback_context);

    // Get quota on an aggregation node
    int GetQuota(const AggPortInfo& agg_port_info,
                 sharp_job_id_t job_id,
                 void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                 FabricProviderCallbackContext* callback_context);

    // Release all allocated resources on an aggregation node
    int CleanAn(const AggPortInfo& agg_port_info,
                void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                FabricProviderCallbackContext* callback_context);

    // Release all resources of specific job on an aggregation node
    int CleanJob(const AggPortInfo& agg_port_info,
                 sharp_job_id_t job_id,
                 void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                 FabricProviderCallbackContext* callback_context);

    // Release resources of specific (job, tree) on an aggregation node
    // CLEAN_JOB_TREE - SUPPORTED for SAT TREES only -
    // Closes and disconnects all jobs QPs and clean tree QPs (by sending dummy last).
    // Sends dummy-last, closes groups, releases OSTs & ORTs and releases the lock.
    int CleanJobTree(const AggPortInfo& agg_port_info,
                     sharp_job_id_t job_id,
                     sharp_trees_t tree_id,
                     void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                     FabricProviderCallbackContext* callback_context);

    // Release list of allocated QPs on an aggregation node
    int ReleaseQp(const AggPortInfo& agg_port_info,
                  QpList* qps,
                  void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                  FabricProviderCallbackContext* callback_context);

    // Set aggregation node configuration settings
    // Support setting aggregation node radix and endianness
    int ConfigureAn(const AggPortInfo& agg_port_info,
                    uint8_t version,
                    uint8_t max_control_path_version_supported,
                    uint16_t data_path_version_supported,
                    uint8_t radix,
                    uint8_t endianness,
                    uint8_t disable_reproducibility,
                    uint8_t enable_reproducibility_per_job,
                    uint8_t tree_job_default_binding,
                    void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                    FabricProviderCallbackContext* callback_context);

    // Allocate QPs on an aggregation node
    int AllocateQp(const AggPortInfo& agg_port_info,
                   int num_qps,
                   void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                   FabricProviderCallbackContext* callback_context);

    // Configure QP properties on an aggregation node
    int ConfigureQp(const AggPortInfo& agg_port_info,
                    uint32_t qpn,
                    uint8_t port,
                    const PortInfo* p_rem_port_info,
                    uint32_t rem_qpn,
                    bool is_multicast,
                    bool is_disable,
                    bool is_sat,
                    uint16_t pkey,
                    SharpMtu mtu,
                    void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                    FabricProviderCallbackContext* callback_context);

    // Send QPCConfig Get request
    int GetQpConfiguration(const AggPortInfo& agg_port_info,
                           uint32_t qpn,
                           void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                           FabricProviderCallbackContext* callback_context);

    // Disconnect job_tree connection(QP) (job_id, tree_id) on an aggregation node
    // SUPPORTED for SAT TREES only - Close and disconnect the all children QPs
    int DisconnectJobTreeQp(const AggPortInfo& agg_port_info,
                            sharp_job_id_t job_id,
                            sharp_trees_t tree_id,
                            void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                            FabricProviderCallbackContext* callback_context);

    // Configure tree structure on an aggregation node
    // is_update_tree == true indicates update of existing tree
    // parent_qpn == 0 and is_update_tree == false indicates this node is root
    // if is_sat: configure streaming aggregation tree using llt_id for lock
    int ConfigureTree(const AggPortInfo& agg_port_info,
                      sharp_trees_t tree_id,
                      uint32_t parent_qpn,
                      const ChildQpList* child_qp_list,
                      bool is_update_tree,
                      bool is_sat,
                      sharp_trees_t llt_id,   // llt used for lock
                      void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                      FabricProviderCallbackContext* callback_context);

    // TreeConfig Disable
    int TreeConfigDisable(const AggPortInfo& agg_port_info,
                          sharp_trees_t tree_id,
                          void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                          FabricProviderCallbackContext* callback_context);

    // Query tree structure on an aggregation node
    // qpn=0 means query all QPs. Otherweise look for specific QP
    int QueryTreeConfig(const AggPortInfo& agg_port_info,
                        sharp_trees_t tree_id,
                        uint32_t qpn,
                        FabricProviderCallbackDlg callback_function,
                        FabricProviderCallbackContext* callback_context);

    // Send GET TreeConfig to obtain state of tree node
    int GetTreeConfig(const AggPortInfo& agg_port_info,
                      const sharp_trees_t tree_id,
                      fabric_provider_handle_data_func_t callback_dlg,
                      FabricProviderCallbackContext* callback_context);

    // Configure AN switch port credits
    int ConfigureAnPorts(const AggPortInfo& agg_port_info,
                         uint8_t port_num,
                         uint16_t requester_packets,
                         uint16_t requester_buffer_cap,
                         uint16_t responder_packets,
                         uint16_t responder_buffer_cap,
                         uint8_t num_flows,
                         void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                         FabricProviderCallbackContext* callback_context);

    // Get Configurations of  AN switch port credits
    int GetAnPortsConfiguration(const AggPortInfo& agg_port_info,
                                uint8_t port_num,
                                void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                                FabricProviderCallbackContext* callback_context);

    // Send ANActiveJobs Get request
    int GetANActiveJobs(const AggPortInfo& agg_port_info,
                        void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                        FabricProviderCallbackContext* callback_context);

    // Query AN QP
    // qpn=0 means query all QPs. Otherweise look for specific QP
    int QueryQPDatabase(const AggPortInfo& agg_port_info,
                        uint32_t qpn,
                        FabricProviderCallbackDlg callback_function,
                        FabricProviderCallbackContext* callback_context);

    inline int AddParentQpToTree(const AggPortInfo& agg_port_info,
                                 u_int16_t tree_id,
                                 uint32_t parent_qpn,
                                 void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                                 FabricProviderCallbackContext* callback_context)
    {
        return UpdateTreeConfig(agg_port_info, 0, 0, tree_id, parent_qpn, false, f, callback_context);
    }

    inline int AddChildQpToTree(const AggPortInfo& agg_port_info,
                                uint32_t child_qpn,
                                u_int8_t child_index,
                                u_int16_t tree_id,
                                void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                                FabricProviderCallbackContext* callback_context)
    {
        return UpdateTreeConfig(agg_port_info, child_qpn, child_index, tree_id, 0, false, f, callback_context);
    }

    inline int RemoveChildQpFromTree(const AggPortInfo& agg_port_info,
                                     uint32_t child_qpn,
                                     u_int8_t child_index,
                                     u_int16_t tree_id,
                                     void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                                     FabricProviderCallbackContext* callback_context)
    {
        return UpdateTreeConfig(agg_port_info, child_qpn, child_index, tree_id, 0, true, f, callback_context);
    }

    inline int RemoveParentQpFromTree(const AggPortInfo& agg_port_info,
                                      u_int16_t tree_id,
                                      uint32_t parent_qpn,
                                      void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                                      FabricProviderCallbackContext* callback_context)
    {
        return UpdateTreeConfig(agg_port_info, 0, 0, tree_id, parent_qpn, true, f, callback_context);
    }

    uint64_t GetSubnetPrefix() { return m_ibis_port_properties_.subnet_prefix; }

    uint64_t GetPortGuid() { return m_ibis_port_properties_.port_guid; }

    bool IsPortUp() { return (m_ibis_port_properties_.state > IBIS_IB_PORT_STATE_DOWN); }

    void UpdatePortProperties();

   private:
    Ibis m_ibis_mads_;
    Ibis m_ibis_traps_;

    pthread_t m_listener_thread_;
    bool m_stop_listener_flag_;

    port_properties_t m_ibis_port_properties_;

    int m_sr_lease_time_;
    std::unique_ptr<struct sr_ctx, SrDeleter> m_sr_context_;
    struct timeval m_port_invalid_time;

    FdEventListener m_fd_event_listener_;
    bool m_event_listener_started;

    // Traps
    TrapHandlingCallback m_traps_handling_callback_;
    void* m_traps_callback_context_;

    FabricProviderOpMode m_operation_mode_;

    // Disable compiler generated copy-assignment operators
    FabricProvider& operator=(const FabricProvider&);
    FabricProvider(const FabricProvider&);

    struct FpCallbackData
    {
        FabricProviderCallbackDlg callback_function;
        FabricProviderCallbackContext context;
        const PortInfo* port_info;
        uint64_t am_key;
        uint8_t version;
        uint8_t pending_mad_count;
        int res;

        // TODO: create different class for each operation
        struct
        {
            bool clean_required;
            bool disable_overwrite_zero_am_key;
            AggNodeDiscoveryInfo discover_an_info;
            IB_ClassPortInfo class_port_info;

            void Init()
            {
                clean_required = false;
                discover_an_info = AggNodeDiscoveryInfo();
            };
        } discover_an;

        struct
        {
            int num_qps;
            QpList allocated_qps;

            void Init()
            {
                num_qps = 0;
                allocated_qps.clear();
            };
        } allocate_qp;

        struct
        {
            QpList requested_qps;
            QpList released_qps;

            void Init()
            {
                requested_qps.clear();
                released_qps.clear();
            };
        } release_qp;

        struct
        {
            sharp_trees_t tree_id;
            ChildQpList requested_child_qps;
            uint8_t opcode;

            void Init()
            {
                tree_id = 0;
                requested_child_qps.clear();
                opcode = 0;
            };
        } tree_config;

        struct
        {
            int offset;
            sharp_job_id_t job_id;
            vector<uint16_t> tree_ids_to_bind;

            void Init()
            {
                offset = 0;
                job_id = 0;
                tree_ids_to_bind.clear();
            };
        } tree_bind;

        struct
        {
            int offset;
            sharp_job_id_t job_id;
            SetTreeIds tree_ids_binded;
            VecTreeIds tree_ids_to_check;

            void Init()
            {
                offset = 0;
                job_id = 0;
                tree_ids_binded.clear();
                tree_ids_to_check.clear();
            };
        } tree_bind_get;

        struct
        {
            uint8_t opcode;
            sharp_job_id_t job_id;
            sharp_trees_t tree_id;

            void Init()
            {
                opcode = 0;
                job_id = 0;
                tree_id = 0;
            };
        } resource_cleanup;

        struct
        {
            uint32_t qpn;
            VectorQpRecord qp_records;
            void Init()
            {
                qpn = 0;
                qp_records.clear();
            };
        } qp_database;

        struct
        {
            uint32_t qpn;
            sharp_trees_t tree_id;
            VectorQpOnTree qps_on_tree;
            void Init()
            {
                qpn = 0;
                tree_id = TREE_CONFIG_INVALID_TREE_ID;
                qps_on_tree.clear();
            };
        } query_tree_config;

        int init()
        {
            pending_mad_count = 0;
            res = 0;
            callback_function = NULL;
            port_info = NULL;
            context.Init();

            discover_an.Init();
            allocate_qp.Init();
            release_qp.Init();
            tree_config.Init();
            resource_cleanup.Init();
            qp_database.Init();
            query_tree_config.Init();

            return 0;
        };
    };

    MemoryPool<FpCallbackData> fp_callback_data_pool;

    // Aggregation nodes discovery process functions

    struct sr_ctx* CreateSrContext(struct sr_config& conf);

    void DiscoverAnGetClassPortInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void DiscoverAnGetAMKeyInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    int SendSetAMKeyRequest(FpCallbackData* fcd,
                            void (*f)(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data),
                            uint64_t am_key);

    int SendGetANSATQPInfo(FpCallbackData* fcd);

    int SendGetANSemaphoreInfo(FpCallbackData* fcd);

    void RecoverAMKeyInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void DiscoverAnSetAMKeyInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void DiscoverAnSetClassPortInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    int SendANInfoGetRequest(FpCallbackData* fcd);

    void DiscoverAnANInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void GetANSATQPInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void GetANSemaphoreInfoCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void TreeToJobBindGetCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void TreeToJobBindSetCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void SetAMKeyCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    // Quota configuration process functions
    void ConfigureQuotaCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void GetQuotaCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    // Aggregation nodes resources cleanup functions
    int SendResourceCleanupRequest(FpCallbackData* fcd,
                                   void (*f)(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data),
                                   uint8_t opcode,
                                   sharp_job_id_t job_id,
                                   sharp_trees_t tree_id);

    void ResourceCleanupCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    // QP allocation and releasing process functions
    int AllocateQpAlloc(FpCallbackData* fcd);

    void AllocateQpAllocCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    int AllocateQpConfirm(FpCallbackData* fcd, AM_QPAllocation* am_qp_allocation);

    void AllocateQpConfirmCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    int ReleaseQpRelease(FpCallbackData* fcd);

    void ReleaseQpReleaseCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    // Aggregation node configuration process functions
    void ConfigureAnCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    // Tree configuration process functions
    int ConfigureTreeAddChildren(FpCallbackData* fcd, uint32_t parent_qpn, uint8_t opcode, bool is_sat, sharp_trees_t llt_id);

    // Tree To Job Bind get process functions
    int TreeToJobBindGet(FpCallbackData* fcd);

    // Tree To Job Bind set process functions
    int TreeToJobBindSet(FpCallbackData* fcd);

    void ConfigureTreeAddChildrenCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void TreeConfigDisableCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    int UpdateTreeConfig(const AggPortInfo& agg_port_info,
                         uint32_t child_qpn,
                         u_int8_t child_index,
                         u_int16_t tree_id,
                         uint32_t parent_qpn,
                         bool is_remove,
                         void (*f)(FabricProviderCallbackContext* callback_context, int rec_status, void* p_attribute_data),
                         FabricProviderCallbackContext* callback_context);

    void UpdateTreeConfigCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void ConfigureQpCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void GetQpConfigrationCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void QueryTreeConfigCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void GetTreeConfigCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void ConfigureAnPortsCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void GetAnPortsConfigurationCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void QueryQPDatabaseCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    void GetANActiveJobsCallback(const clbck_data_t& clbck_data, int rec_status, void* p_attribute_data);

    static inline const char* ResourceCleanupOpToStr(uint8_t opcode);

    static inline const char* TreeConfigOpToStr(uint8_t opcode);

    static inline const char* QpAllocationOpToStr(uint8_t opcode);

    // Traps listener functions
    int StartListener();

    void StopListener();

    static void* ListenerThreadMain(void* _fabric_provider);

    // Traps handling functions
    static void HandleAmTrapDlg(ib_address_t* p_ib_address, void* p_class_data, void* p_attribute_data, void* context);

    void HandleAmTrap(ib_address_t* p_ib_address, MAD_AggregationManagement* p_am_mad, Notice* p_noice);

    void SetBufferCalculationParameters(BufferCalculationParameters& buffer_calculation_parameters,
                                        uint16_t line_size,
                                        uint8_t worst_case_num_lines,
                                        uint8_t num_lines_chunk_mode,
                                        uint8_t half_buffer_line_optimization_supported);

    void PrintCallbackError(const clbck_data_t& clbck_data,
                            int rec_status,
                            const char* callback_name,
                            const char* format_suffix = NULL,
                            ...);

    void HandleCallbackEnd(FpCallbackData* fcd, void* p_data, int res);

    int QueryTreeConfigNextRecord(FpCallbackData* fcd, uint32_t record_locator);
    int QueryQPDatabaseNextRecord(FpCallbackData* fcd, uint32_t record_locator);

    friend class AnAdapterV1;
    friend class AnAdapterV2;
};

#endif /* _FABRIC_PROVIDER_H */
