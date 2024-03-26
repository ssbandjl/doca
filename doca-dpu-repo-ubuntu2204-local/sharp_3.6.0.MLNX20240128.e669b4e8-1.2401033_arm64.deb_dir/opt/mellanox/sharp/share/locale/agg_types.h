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

#ifndef AGG_TYPES_H_
#define AGG_TYPES_H_

#include <list>
#include <map>
#include <set>
#include <vector>
// #include <fstream>
#include <inttypes.h>
#include <string.h>
#include <sys/types.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "reserved_string.h"

#include "buffer_calc.h"

using std::binary_function;
using std::dec;
using std::endl;
using std::hex;
using std::list;
using std::make_pair;
using std::map;
using std::ostream;
using std::pair;
using std::set;
using std::string;
using std::vector;

// MAX_LLT_TREE_ID & MAX_SAT_TREE_ID are not used. Only for information purposes
#define MAX_LLT_TREE_ID 510   // tree id 64 is saved for FW usage (Eagle)
#define MAX_SAT_TREE_ID 1022

#define MAX_TREE_TABLE_SIZE    1022
#define MIN_TREE_TABLE_SIZE    63
#define MIN_LLT_TREES_TO_BUILD 63

#define MAX_TREE_ID     MAX_SAT_TREE_ID
#define INVALID_TREE_ID 0xFFFF   // invalid tree identifier

#define INVALID_NUM_SEMAPHORES 0xFF
#define DEFAULT_NUM_SEMAPHORES 1

// Maximal tree radix
// Value must be between 16 and 252
#define MAX_TREE_RADIX 252

#define MAX_NUM_HOPS 0xFF
#define MAX_PORT_NUM 254

#define INVALID_TREE_RANK  0xFF
#define INVALID_COORDINATE 0xFFFF

#define DEFAULT_AM_KEY 0

#define MAX_TREES_PER_JOB     4
#define MAX_USER_DATA_PER_OST 1024
#define MAX_DFP_RANK          3

#define MAX_CONTROL_PATH_VERSION_SUPPORTED 2

#ifndef PRIx64
#if __WORDSIZE == 64
#define PRIx64 "lx"
#else
#define PRIx64 "llx"
#endif
#endif

#ifndef U64H_FMT
// #define U64H_FMT    "0x%"PRIx64
// #define U64H_FMT    "0x%016llx"
#define U64H_FMT "0x%016lx"
#endif   // U64H_FMT

#ifndef U64D_FMT
#define U64D_FMT "%lu"
#endif   // U64D_FMT

#define QPN_FMT "0x%08x"

#define PERCENT_FMT "%.2f"

#define JOB_FILE_PATH_SIZE PATH_MAX   // 4096B (defined in linux/limits.h)

#define JOB_FILE_VERSION 1

#define JOBS_NUMBER_V1 256

#define IS_TEMP_ERR(rec_status) \
    ((rec_status == IBIS_MAD_STATUS_TIMEOUT) || (rec_status == IBIS_MAD_STATUS_GENERAL_ERR) || (rec_status == IBIS_MAD_STATUS_SEND_FAILED))

#define MAX_SEMAPHORES 0x40

///////////////////////////////////////////////////////////////////////////////
//
// TYPEDEFS
//

typedef uint8_t phys_port_t;
typedef uint8_t rank_t;
typedef uint16_t lid_t;
typedef uint16_t device_id_t;
typedef uint64_t port_key_t;
typedef uint32_t sharp_job_id_t;
typedef uint64_t sharpd_id_t;
typedef uint16_t sharp_trees_t;
typedef uint32_t sharp_osts_t;
typedef uint16_t sharp_user_data_size_t;
typedef uint32_t sharp_buffers_t;
typedef uint32_t sharp_groups_t;
typedef uint32_t sharp_qps_t;
typedef uint16_t sharp_an_id_t;
typedef uint32_t sharp_path_id_t;
typedef float sharp_percent_t;

typedef std::vector<uint8_t> MinHopsTable;         // min hops table, key=LID, value=hops [0xFF, 0xFF, 1, 0, 0xFF, 2]
typedef std::vector<uint16_t> MinHopsIndexTable;   // contain all keys(LID) of min hops table [2,3,5]
typedef lid_t node_min_hop_key_t;

typedef std::set<class AggNodeFabricInfo*> SetAnFabricInfoPtr;
typedef std::set<AggNodeFabricInfo*> SetAggNodeFabricPtr;   // Important to keep as ordered set, since we compare sets
typedef std::list<class TreeNode*> ListTreeNodePtr;
typedef std::vector<uint8_t> AnMinHopsTable;
typedef std::map<uint8_t, uint8_t> MapPortSemaphores;
typedef std::map<uint8_t, uint32_t> MapPortSATQps;
typedef std::set<sharp_trees_t> SetTreeIds;
typedef std::vector<sharp_trees_t> VecTreeIds;

// messages of agg node states
extern char const* const g_removed_from_network_msg;
extern char const* const g_ignored_by_configuration_msg;
extern char const* const g_invalid_lid_msg;
extern char const* const g_invalid_port_state_msg;
extern char const* const g_unreachable_port_msg;
extern char const* const g_invalid_port_status_msg;

// ENUMS
enum AM_RETURN_VALUE
{
    AM_RETURN_OK = 0,
    AM_RETURN_GENERAL_ERROR = -1,
    AM_RETURN_MEMORY_ERROR = -2
};

enum SharpTrapNumberEnum
{
    SHARP_TRAP_NUMBER_QP_ERROR = 0,
    SHARP_TRAP_NUMBER_INVALID_REQ,
    SHARP_TRAP_NUMBER_SHARP_ERROR,
    SHARP_TRAP_NUMBER_QP_ALLOC_TIMEOUT,
    SHARP_TRAP_NUMBER_AMKEY_VIOLATION,
    SHARP_TRAP_NUMBER_UNSUPPORTED_TRAP
};

enum SharpMtu
{
    SHARP_MTU_UNKNOWN = 0,
    SHARP_MTU_256 = 1,
    SHARP_MTU_512,
    SHARP_MTU_1K,
    SHARP_MTU_2K,
    SHARP_MTU_4K,
    SHARP_MTU_MAX = SHARP_MTU_4K,
};

enum SeamlessRestartStatus
{
    // Seamless restart activated
    SEAMLESS_RESTART_SUCCESS = 0,

    // Seamless restart disabled
    DISABLED_BY_CONFIGURATION,
    NO_PENDING_JOBS,
    PARSE_AGG_NODES_DUMP_FAILED,
    JOB_FILES_OLD_OR_CORRUPTED,
    NUMBER_OF_RETRIES_EXCEEDED_MAX_RETRIES,
    AM_VERSION_DIFFERENT_FROM_DUMP_VERSION,

    // Seamless restart failed
    AGG_NODE_DUMP_NOT_FOUND,
    PORT_CONFIGURATION_COMPARE_FAILED,
    OPTION_COMPARE_FAILED,
    DUMP_FILES_CRC_CHECK_FAILED,
    BUILD_TREES_FAILED,
    QUERY_RUNNING_JOBS_FAILED,
    HANDLE_PERSISTENT_JOBS_FILES_FAILED,
    RESTORE_JOB_QUOTAS_FAILED,
    JOB_OBJECTS_CREATION_FAILED,
    TREES_CONFIGURATION_RESTORE_FAILED,
    RESTORE_AGG_NODE_STATES_FAILED,
    AGG_NODES_CONFIGURATON_COMPARE_FAILED,
    RESTORE_AGG_PATH_ACTION_FAILED,
    HANDLE_PENDING_TREE_EDGES_FAILED,
    RESTORE_AGG_PATHS_FAILED
};

static inline const char* SharpTrapNumberToChar(const SharpTrapNumberEnum trap_number)
{
    switch (trap_number) {
        case SHARP_TRAP_NUMBER_QP_ERROR:
            return ("QP Error");
        case SHARP_TRAP_NUMBER_INVALID_REQ:
            return ("Invalid request");
        // SharpError syndromes: 0 - Lock Semaphore timeout, 1 - OST no progress timeout ...
        case SHARP_TRAP_NUMBER_SHARP_ERROR:
            return ("SHARP error");
        default:
            return ("UNKNOWN");
    }
};

union PrivateAppData
{
    void* ptr;
    uint64_t val;

    PrivateAppData() { val = 0; }
};

struct Epoch
{
    uint64_t m_curr_epoch_;   // last update number
    uint64_t m_prev_epoch_;

    Epoch() : m_curr_epoch_(0), m_prev_epoch_(0) {}
    explicit Epoch(uint64_t epoch) : m_curr_epoch_(epoch), m_prev_epoch_(0) {}

    void SetEpoch(const uint64_t epoch)
    {
        if (m_curr_epoch_ != epoch) {
            m_prev_epoch_ = m_curr_epoch_;
            m_curr_epoch_ = epoch;
        }
    }

    void Revert(const uint64_t curr_epoch)
    {
        if (curr_epoch == m_curr_epoch_) {
            m_curr_epoch_ = m_prev_epoch_;
        }
    }
};

// when received from SD
//  m_osts - per tree
//  m_buffer - unused
//  m_groups - per tree
//  m_qps - per tree per port
struct Quota
{
    sharp_trees_t m_treesPerJob;
    sharp_osts_t m_osts;
    sharp_user_data_size_t m_user_data_per_ost;
    sharp_buffers_t m_buffers;
    sharp_groups_t m_groups;
    sharp_qps_t m_qps;

    Quota(sharp_trees_t trees_per_job,
          sharp_osts_t osts_per_tree,
          sharp_user_data_size_t user_data_per_ost,
          sharp_groups_t groups_per_tree,
          sharp_qps_t qps_per_tree_per_host)
        : m_treesPerJob(trees_per_job),
          m_osts(osts_per_tree),
          m_user_data_per_ost(user_data_per_ost),
          m_buffers(0),
          m_groups(groups_per_tree),
          m_qps(qps_per_tree_per_host)
    {}

    Quota() : m_treesPerJob(0), m_osts(0), m_user_data_per_ost(0), m_buffers(0), m_groups(0), m_qps(0) {}

    string ToString() const
    {
        std::stringstream stream;
        stream << "{t:" << m_treesPerJob << ", q:" << m_qps << ", ud:" << m_user_data_per_ost << ", b:" << m_buffers << ", o:" << m_osts
               << ", g:" << m_groups << " }";

        return stream.str();
    }

    friend bool operator==(const Quota& c1, const Quota& c2)
    {
        return (c1.m_treesPerJob == c2.m_treesPerJob && c1.m_osts == c2.m_osts && c1.m_user_data_per_ost == c2.m_user_data_per_ost &&
                c1.m_buffers == c2.m_buffers && c1.m_groups == c2.m_groups && c1.m_qps == c2.m_qps);
    }

    friend bool operator!=(const Quota& c1, const Quota& c2) { return !(c1 == c2); }
};

enum SharpExclusiveLock
{
    NO_EXCLUSIVE_LOCK = 0,
    EXCLUSIVE_LOCK_BEST_EFFORT,
    EXCLUSIVE_LOCK
};

struct JobResource
{
    Quota m_quota;
    sharp_percent_t m_quota_percent;
    sharp_percent_t m_qps_percent;
    uint8_t m_job_priority;
    uint8_t m_child_index_per_port;
    bool m_multicast_enabled;
    bool m_reproducibility_enabled;
    uint64_t m_req_feature_mask;
    uint16_t m_pkey;
    SharpExclusiveLock m_exclusive_lock;
    bool m_request_rmc;

    JobResource()
        : m_quota(),
          m_quota_percent(0),
          m_qps_percent(0),
          m_job_priority(0),
          m_child_index_per_port(1),
          m_multicast_enabled(false),
          m_reproducibility_enabled(false),
          m_req_feature_mask(0),
          m_pkey(0),
          m_exclusive_lock(NO_EXCLUSIVE_LOCK),
          m_request_rmc(false)
    {}
};

enum TopologyType
{
    TOPOLOGY_TYPE_NONE,
    TOPOLOGY_TYPE_TREE,
    TOPOLOGY_TYPE_HYPER_CUBE,
    TOPOLOGY_TYPE_DFP
};

static inline const char* TopologyTypeToStr(TopologyType topology_type)
{
    switch (topology_type) {
        case TOPOLOGY_TYPE_NONE:
            return ("none");
        case TOPOLOGY_TYPE_TREE:
            return ("tree");
        case TOPOLOGY_TYPE_HYPER_CUBE:
            return ("hypercube");
        case TOPOLOGY_TYPE_DFP:
            return ("dragonfly+");
        default:
            return ("unknown");
    }
};

struct NodeTopologyInfo
{
    uint8_t m_tree_rank;
    uint16_t m_coordinates;
    TopologyType m_topology_type;

    // NodeTopologyInfo() : m_topology_type(TOPOLOGY_TYPE_NONE) {}

    // Constructor
    explicit NodeTopologyInfo(TopologyType topology_type)
        : m_tree_rank(INVALID_TREE_RANK), m_coordinates(INVALID_COORDINATE), m_topology_type(topology_type)
    {}

    uint16_t GetCoordinates() const
    {
        if ((m_topology_type != TOPOLOGY_TYPE_HYPER_CUBE) && (m_topology_type != TOPOLOGY_TYPE_DFP)) {
            throw std::logic_error("not hyper cube or DFP topology");
        }
        return m_coordinates;
    }

    void SetCoordinates(uint16_t coordinates)
    {
        if ((m_topology_type != TOPOLOGY_TYPE_HYPER_CUBE) && (m_topology_type != TOPOLOGY_TYPE_DFP)) {
            throw std::logic_error("not hyper cube or DFP topology");
        }
        m_coordinates = coordinates;
    }

    inline bool IsRankInvalid() const
    {
        if ((m_topology_type == TOPOLOGY_TYPE_NONE) || (m_topology_type == TOPOLOGY_TYPE_HYPER_CUBE)) {
            return false;
        }
        return (INVALID_TREE_RANK == m_tree_rank);
    }

    uint8_t GetRankIfAvailable() const
    {
        if ((m_topology_type != TOPOLOGY_TYPE_TREE) && (m_topology_type != TOPOLOGY_TYPE_DFP)) {
            return INVALID_TREE_RANK;
        }
        return m_tree_rank;
    }

    uint8_t GetRank() const
    {
        if ((m_topology_type != TOPOLOGY_TYPE_TREE) && (m_topology_type != TOPOLOGY_TYPE_DFP)) {
            throw std::logic_error("not tree or DFP topology");
        }
        return m_tree_rank;
    }

    void SetRank(uint8_t rank)
    {
        if ((m_topology_type != TOPOLOGY_TYPE_TREE) && (m_topology_type != TOPOLOGY_TYPE_DFP)) {
            throw std::logic_error("not tree or DFP topology");
        }
        m_tree_rank = rank;
    }

    string ToString() const
    {
        std::stringstream stream;
        switch (m_topology_type) {
            case TOPOLOGY_TYPE_TREE:
                stream << "rank:" << dec << (int)m_tree_rank;
                break;
            case TOPOLOGY_TYPE_HYPER_CUBE:
                stream << "coordinates:0x" << std::setfill('0') << std::setw(4) << hex << (int)m_coordinates;
                break;
            case TOPOLOGY_TYPE_DFP:
                stream << "rank:" << dec << (int)m_tree_rank << "coordinates:0x" << std::setfill('0') << std::setw(4) << hex
                       << (int)m_coordinates;
                break;
            default:
                break;
        }
        return stream.str();
    }

    bool operator==(const NodeTopologyInfo& rhs) const
    {
        if (m_topology_type != rhs.m_topology_type) {
            throw std::invalid_argument("not matching topology types");
        }

        switch (m_topology_type) {
            case TOPOLOGY_TYPE_TREE:
                return (m_tree_rank == rhs.m_tree_rank);
                break;
            case TOPOLOGY_TYPE_HYPER_CUBE:
                return (m_coordinates == rhs.m_coordinates);
                break;
            case TOPOLOGY_TYPE_DFP:
                return ((m_coordinates == rhs.m_coordinates) && (m_tree_rank == rhs.m_tree_rank));
                break;
            default:
                throw std::invalid_argument("invalid topology type");
        }
    }

    bool operator!=(const NodeTopologyInfo& rhs) const { return !(*this == rhs); }

    bool operator<(const NodeTopologyInfo& rhs) const
    {
        if (m_topology_type != rhs.m_topology_type) {
            throw std::invalid_argument("not matching topology types");
        }
        switch (m_topology_type) {
            case TOPOLOGY_TYPE_TREE:
                return (m_tree_rank < rhs.m_tree_rank);
                break;
            case TOPOLOGY_TYPE_HYPER_CUBE:
                return (m_coordinates < rhs.m_coordinates);
                break;
            case TOPOLOGY_TYPE_DFP:
                if (m_tree_rank != rhs.m_tree_rank) {
                    return (m_tree_rank < rhs.m_tree_rank);
                }
                return (m_coordinates < rhs.m_coordinates);
                break;
            default:
                throw std::invalid_argument("invalid topology type");
        }
    }

    bool operator>(const NodeTopologyInfo& rhs) const { return (rhs < *this); }

    bool operator<=(const NodeTopologyInfo& rhs) const { return !(*this > rhs); }

    bool operator>=(const NodeTopologyInfo& rhs) const { return !(*this < rhs); }
};

struct PortTimestamp
{
    string m_timestamp_str;

    PortTimestamp() : m_timestamp_str("") {}

    explicit PortTimestamp(const string& timestamp_str) : m_timestamp_str(timestamp_str) {}

    explicit PortTimestamp(const PortTimestamp& port_timestamp) { m_timestamp_str = port_timestamp.m_timestamp_str; }

    void SetTimestamp(const string& timestamp_str) { m_timestamp_str = timestamp_str; }

    const string& ToString() const { return m_timestamp_str; }

    bool operator==(const PortTimestamp& rhs) const { return m_timestamp_str.compare(rhs.m_timestamp_str) == 0; }

    bool operator!=(const PortTimestamp& rhs) const { return !(*this == rhs); }
};

struct PortInfo
{
    port_key_t m_port_key;
    uint64_t m_subnet_prefix;
    string m_node_desc;
    lid_t m_lid;
    // peer (remote port) information
    port_key_t m_peer_key;
    NodeTopologyInfo m_peer_topology_info;
    PortTimestamp m_timestamp;
    bool m_use_grh;

    // PortInfo() : m_port_key(0), m_lid(0), m_peer_key(0) {}

    PortInfo(port_key_t port_key,
             uint64_t subnet_prefix,
             const string& node_desc,
             lid_t lid,
             port_key_t peer_key,
             NodeTopologyInfo peer_topology_info,
             const PortTimestamp& timestamp,
             bool use_grh = false)
        : m_port_key(port_key),
          m_subnet_prefix(subnet_prefix),
          m_node_desc(node_desc),
          m_lid(lid),
          m_peer_key(peer_key),
          m_peer_topology_info(peer_topology_info),
          m_timestamp(timestamp),
          m_use_grh(use_grh)
    {}

    string GetName() const
    {
        std::stringstream stream;
        stream << m_node_desc << " GUID:0x" << hex << m_port_key;
        return stream.str();
    }

    string GetSwString() const
    {
        std::stringstream stream;
        stream << "switch GUID:0x" << hex << m_peer_key;
        return stream.str();
    }

    string ToString() const
    {
        std::stringstream stream;
        stream << m_node_desc << " GUID:0x" << hex << m_port_key << " PEER GUID:0x" << m_peer_key << " lid:" << dec << m_lid << " "
               << m_peer_topology_info.ToString();
        return stream.str();
    }
};

struct AggPortInfo
{
    uint8_t m_active_control_path_version;
    const PortInfo& m_port_info;
    uint64_t m_am_key;
    std::string m_inactive_reason_msg_;

    AggPortInfo(uint8_t version, const PortInfo& port_info, uint64_t am_key)
        : m_active_control_path_version(version), m_port_info(port_info), m_am_key(am_key)
    {}

    inline void SetInactiveReasonMessage(const std::string& inactive_reason_msg) { m_inactive_reason_msg_ = inactive_reason_msg; }
    inline std::string GetInactiveReasonMessage() const { return m_inactive_reason_msg_; }
    inline void ClearInactiveReasonMessage() { m_inactive_reason_msg_.clear(); }
};

// AggNodeInfo is a struct that holds only data that was received via MADs.
// Data from SMDB (like GUID) can be found in Node class (in fabric_graph.h)
struct AggNodeInfo
{
    MapPortSemaphores m_ports_to_semaphores;
    MapPortSemaphores m_ports_to_max_semaphores;
    MapPortSATQps m_ports_to_num_sat_qps;

    // control_path_version - IB: AM class version
    uint8_t m_max_control_path_version_supported;
    uint8_t m_active_control_path_version;   // 1 = Switch-IB, 2 = Quantum

    // data_path_version - IB: AnInfo sharp version
    bool m_multiple_sver_active_supported;
    uint16_t m_data_path_version_supported;   // bit mask
    uint16_t m_active_data_path_version;      // bit mask

    sharp_job_id_t m_num_of_jobs;
    uint16_t m_radix;
    uint16_t m_max_radix;
    uint32_t m_osts;
    uint32_t m_buffers;
    uint32_t m_groups;
    uint32_t m_qps;
    uint32_t m_max_sat_qps;
    uint32_t m_max_llt_qps;
    uint32_t m_max_sat_qps_per_port;
    uint32_t m_max_llt_qps_per_port;
    uint32_t m_max_user_data_per_ost;
    uint8_t m_endianness;
    uint8_t m_reproducibility_disable_supported;
    bool m_enable_reproducibility;
    bool m_streaming_aggregation_supported;
    bool m_configure_port_credit_supported;
    bool m_enable_reproducibility_per_job;
    bool m_semaphores_per_port;
    u_int8_t m_num_semaphores;
    uint8_t m_num_active_semaphores;
    uint8_t m_reproducibility_per_job_supported;
    uint8_t m_am_key_supported;
    uint8_t m_job_key_supported;
    uint8_t m_tree_job_binding_supported;
    uint8_t m_tree_job_default_binding;
    uint16_t m_tree_table_size;
    SharpMtu m_mtu;
    bool m_fp19_supported;
    bool m_bfloat19_supported;
    bool m_extended_data_types_supported;
    bool m_qp_to_port_select_supported;
    bool m_rmc_supported;
    int m_clean_required;
    std::chrono::steady_clock::time_point m_clean_required_reset_timepoint;

    BufferCalculationParameters m_buffer_calculation_parameters;

    AggNodeInfo()
        : m_max_control_path_version_supported(0),
          m_active_control_path_version(0),
          m_multiple_sver_active_supported(false),
          m_data_path_version_supported(0),
          m_active_data_path_version(0),
          m_num_of_jobs(0),
          m_radix(0),
          m_max_radix(0),
          m_osts(0),
          m_buffers(0),
          m_groups(0),
          m_qps(0),
          m_max_sat_qps(0),
          m_max_llt_qps(0),
          m_max_sat_qps_per_port(0),
          m_max_llt_qps_per_port(0),
          m_max_user_data_per_ost(0),
          m_endianness(0),
          m_reproducibility_disable_supported(0),
          m_enable_reproducibility(false),
          m_streaming_aggregation_supported(false),
          m_configure_port_credit_supported(false),
          m_enable_reproducibility_per_job(false),
          m_semaphores_per_port(false),
          m_num_semaphores(0),
          m_num_active_semaphores(0),
          m_reproducibility_per_job_supported(0),
          m_am_key_supported(0),
          m_job_key_supported(0),
          m_tree_job_binding_supported(0),
          m_tree_job_default_binding(0),
          m_tree_table_size(0),
          m_mtu(SHARP_MTU_UNKNOWN),
          m_fp19_supported(false),
          m_bfloat19_supported(false),
          m_extended_data_types_supported(false),
          m_qp_to_port_select_supported(false),
          m_rmc_supported(false),
          m_clean_required(0),
          m_clean_required_reset_timepoint{},
          m_buffer_calculation_parameters()
    {}

    string ToString() const
    {
        std::stringstream stream;
        stream << "{q:" << m_qps << ", b:" << m_buffers << ", o:" << m_osts << ", g:" << m_groups << " }";

        return stream.str();
    }
};

struct AggNodeDiscoveryInfo
{
    AggNodeInfo m_agg_node_info;
    bool m_resources_clean_required;
    bool m_am_key_set;

    AggNodeDiscoveryInfo() : m_agg_node_info(), m_resources_clean_required(false), m_am_key_set(false) {}
};

class AnToAnKey
{
    u_int64_t m_node_guid1_;
    u_int64_t m_node_guid2_;

   public:
    AnToAnKey(u_int64_t node_guid_a, u_int64_t node_guid_b)
    {
        // In order to preserve a constant key between two guids,
        // we always keep the smaller value at guid1
        if (node_guid_a < node_guid_b) {
            m_node_guid1_ = node_guid_a;
            m_node_guid2_ = node_guid_b;
        } else {
            m_node_guid2_ = node_guid_a;
            m_node_guid1_ = node_guid_b;
        }
    }

    u_int64_t GetNodeGuid1() const { return m_node_guid1_; }
    u_int64_t GetNodeGuid2() const { return m_node_guid2_; }

    bool operator<(const AnToAnKey& an_to_an_key) const
    {
        if (m_node_guid1_ != an_to_an_key.m_node_guid1_) {
            return (m_node_guid1_ < an_to_an_key.m_node_guid1_);
        }
        return (m_node_guid2_ < an_to_an_key.m_node_guid2_);
    }

    bool operator==(const AnToAnKey& an_to_an_key) const
    {
        return (m_node_guid1_ == an_to_an_key.m_node_guid1_ && m_node_guid2_ == an_to_an_key.m_node_guid2_);
    }
};
class AnToAnInfo
{
   public:
    AnToAnKey m_key_;

    phys_port_t m_port_num1_;
    phys_port_t m_port_num2_;

    phys_port_t m_prev_port_num1_;
    phys_port_t m_prev_port_num2_;

    string m_timestamp_;
    string m_prev_timestamp_;

   public:
    AnToAnInfo() : m_key_(0, 0), m_port_num1_(0), m_port_num2_(0), m_prev_port_num1_(0), m_prev_port_num2_(0){};

    AnToAnInfo(u_int64_t node_guid1, u_int64_t node_guid2, phys_port_t port_num1, phys_port_t port_num2, const string& timestamp)
        : m_key_(node_guid1, node_guid2), m_prev_port_num1_(0), m_prev_port_num2_(0), m_timestamp_(timestamp)
    {
        if (m_key_.GetNodeGuid1() == node_guid1) {
            m_port_num1_ = port_num1;
            m_port_num2_ = port_num2;
        } else {
            m_port_num2_ = port_num1;
            m_port_num1_ = port_num2;
        }
    }

    const AnToAnKey& GetKey() const { return m_key_; }
    const string& GetTimestamp() const { return m_timestamp_; }
    const string& GetPrevTimestamp() const { return m_timestamp_; }

    void Update(const AnToAnInfo& an_to_an_info)
    {
        m_port_num1_ = an_to_an_info.m_port_num1_;
        m_port_num2_ = an_to_an_info.m_port_num2_;
        m_timestamp_ = an_to_an_info.m_timestamp_;
    }

    u_int64_t GetNodeGuid1() const { return m_key_.GetNodeGuid1(); }

    u_int64_t GetNodeGuid2() const { return m_key_.GetNodeGuid2(); }

    phys_port_t GetPortNum1() const { return m_port_num1_; }

    phys_port_t GetPortNum2() const { return m_port_num2_; }

    void UpdatePrev()
    {
        m_prev_port_num1_ = m_port_num1_;
        m_prev_port_num2_ = m_port_num2_;
        m_prev_timestamp_ = m_timestamp_;
    }

    bool IsRouteChanged() const { return (m_prev_port_num1_ != m_port_num1_ || m_prev_port_num2_ != m_port_num2_); }

    bool IsTimestampChanged() const { return (m_prev_timestamp_ != m_timestamp_); }
};
struct TrapQpError
{
    sharp_trees_t m_tree_id;
    sharp_qps_t m_local_qp;
    sharp_qps_t m_remote_qp;
    lid_t m_local_an_port_lid;
    lid_t m_remote_port_lid;
    sharp_job_id_t m_sharp_job_id;

    TrapQpError(sharp_trees_t tree_id,
                sharp_qps_t local_qp,
                sharp_qps_t remote_qp,
                lid_t local_an_port_lid,
                lid_t remote_port_lid,
                sharp_job_id_t sharp_job_id)
        : m_tree_id(tree_id),
          m_local_qp(local_qp),
          m_remote_qp(remote_qp),
          m_local_an_port_lid(local_an_port_lid),
          m_remote_port_lid(remote_port_lid),
          m_sharp_job_id(sharp_job_id)
    {}
};

struct TrapSharpError
{
    sharp_job_id_t m_sharp_job_id;

    TrapSharpError(sharp_job_id_t sharp_job_id) : m_sharp_job_id(sharp_job_id) {}
};
#endif   // AGG_TYPES_H_
