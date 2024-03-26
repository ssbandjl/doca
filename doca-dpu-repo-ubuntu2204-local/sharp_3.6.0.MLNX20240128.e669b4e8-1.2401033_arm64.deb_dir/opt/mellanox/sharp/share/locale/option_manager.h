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

#ifndef OPTION_MANAGER_H_
#define OPTION_MANAGER_H_

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#include "agg_types.h"
#include "common/sharp_common.h"
#include "sharp_opt_parser.h"

struct Quota;
extern char const* const g_default_smdb_path;
extern char const* const g_default_ufm_smdb_path;

typedef std::map<uint8_t, Quota> MapPriorityToQuota;

typedef int (*validate_quota_func_t)(int, int, int, int, int, char*, size_t);

enum option_trimming_mode
{
    OPTION_DATA_PATH_TRIMMING_MODE = 0x1
};

struct OptionInfo
{
    //[QUOTA]
    // MapPriorityToQuota  m_priorityToQuota;
    sharp_percent_t m_per_prio_max_quota[MAX_JOB_PRIORITY + 1];
    sharp_percent_t m_per_prio_default_llt_quota[MAX_JOB_PRIORITY + 1];
    sharp_percent_t m_per_prio_default_sat_quota[MAX_JOB_PRIORITY + 1];

    uint8_t m_low_priority_quota;
    uint32_t m_sat_jobs_default_absolute_osts;

    uint16_t m_max_trees_to_build;
    uint8_t m_max_trees;
    uint8_t m_default_trees;
    bool m_dynamic_tree_allocation;
    uint8_t m_dynamic_tree_algorithm;

    uint8_t m_max_hosts_per_an;
    bool m_enable_reproducibility;
    bool m_default_reproducibility;
    bool m_enable_job_pkey_on_tree;
    bool m_enable_exclusive_lock;

    //[FABRIC_CONNECTION]
    // Local port GUID
    uint64_t m_ib_port_guid;
    uint16_t m_max_mads_on_the_wire;

    uint16_t m_timeout;
    uint8_t m_retries;

    uint64_t m_am_key;
    bool m_am_key_protect_bit;
    uint16_t m_am_key_lease_period;
    uint64_t m_sa_key;
    uint8_t m_service_key[16];
    uint8_t m_sharp_sl;

    bool m_support_multicast;
    uint16_t m_multicast_signature;

    uint8_t m_trimming_mode_mask;
    string m_ignore_host_guids_file;
    bool m_ignore_sm_guids;

    //[SMX_OPTIONS]
    uint64_t m_smx_enabled_protocols;
    unsigned m_smx_protocol;

    string m_smx_sock_interface;
    string m_smx_sock_addr_family;
    unsigned m_smx_sock_port;
    uint8_t m_smx_sock_backlog;
    string m_smx_unix_sock_name;
    string m_smx_ucx_interface;
    uint32_t m_ucx_wait_before_connection_close;
    uint32_t m_smx_keepalive_min_time_before_connection_refresh;
    uint16_t m_smx_keepalive_interval;
    uint16_t m_smx_incoming_conn_keepalive_interval;
    uint16_t m_smx_keepalive_refresh_interval;
    uint16_t m_smx_keepalive_min_percentage_of_connections_to_refresh_at_iteration;

    //[SMX_DEBUG]
    string m_smx_send_file;
    string m_smx_recv_file;

    //[LOG]
    string m_log_file;
    uint8_t m_log_verbosity;
    bool m_full_verbosity;
    uint8_t m_syslog_verbosity;
    uint32_t m_max_log_backups;
    uint32_t m_max_log_size;
    bool m_accumulate_log;

    //[GENERAL]
    string m_persistent_path;
    string m_dump_path;
    bool m_generate_dump_files;
    bool m_daemonize;
    string m_pid_file;
    string m_create_config_file;
    uint8_t m_thread_pool_threads;
    //[SHARP_TREE]
    // if not empty generate trees from file
    TopologyType m_topology_type;
    string m_hc_coordinates_file;
    string m_trees_file;
    uint16_t m_max_tree_radix;
    bool m_span_all_agg_nodes;
    uint8_t m_control_path_version;

    uint8_t m_endianness;
    bool m_clean_an_on_discovery;
    bool m_clean_and_exit;
    string m_fabric_lst_file;
    uint8_t m_lst_file_timeout;
    uint8_t m_lst_file_retries;
    string m_fabric_smdb_file;
    string m_fabric_virt_file;
    uint8_t m_fabric_update_interval;
    uint32_t m_local_port_validation_timeout;
    string m_root_guids_file;
    uint32_t m_recovery_retry_interval;
    bool m_enable_parallel_links;
    uint32_t m_job_reconnection_timeout;
    string m_log_categories_file;

    //[AGG_MANAGER] Seamless restart
    bool m_enable_seamless_restart;
    string m_seamless_restart_trees_file;
    uint32_t m_seamless_restart_max_retries;

    //[AGG_MANAGER]
    long m_main_timeout;   // env var: SHARP_AM_TIMEOUT
    bool m_disable_cleanup_on_am_restart;
    bool m_allow_remote_sm;
    uint8_t m_am_retries;
    bool m_reservation_mode;
    bool m_load_reservations;
    bool m_reservation_force_guid_assignment;
    bool m_reservation_stop_jobs_upon_scale_in;
    bool m_force_app_id_match;
    bool m_enable_topology_api;
    bool m_enable_async_send;
    int32_t m_app_resources_default_limit;
    bool m_disable_agg_nodes_upon_error;

    //[AGG_MANAGER] pending mode configuration
    int32_t m_pending_mode_timeout_min;
    uint16_t m_job_info_polling_interval;

    // IBIS log file options
    string m_ibis_log_file;
    unsigned m_ibis_log_size;
    bool m_ibis_accum_log;

    // IB QP Context configuration options
    bool m_ib_qpc_use_grh;
    uint16_t m_ib_qpc_pkey;
    uint8_t m_ib_qpc_ts;
    uint8_t m_ib_qpc_sl;
    uint8_t m_ib_sat_qpc_sl;
    uint8_t m_ib_qpc_traffic_class;
    uint32_t m_ib_qpc_rq_psn;
    uint32_t m_ib_sat_qpc_rq_psn;
    uint32_t m_ib_qpc_sq_psn;
    uint32_t m_ib_sat_qpc_sq_psn;
    uint8_t m_ib_qpc_rnr_mode;
    uint8_t m_ib_sat_qpc_rnr_mode;
    uint8_t m_ib_qpc_rnr_retry_limit;
    uint8_t m_ib_sat_qpc_rnr_retry_limit;
    uint8_t m_ib_qpc_local_ack_timeout;
    uint8_t m_ib_sat_qpc_local_ack_timeout;
    uint8_t m_ib_qpc_timeout_retry_limit;
    uint8_t m_ib_sat_qpc_timeout_retry_limit;
    uint8_t m_ib_sat_max_mtu;

    // Device configuration DB file
    string m_device_configuration_file;

    // Events Configurations
    uint32_t m_event_manager_backlog;

    // --help and --version
    bool m_show_help;
    bool m_show_version;

   public:
    OptionInfo()
        : m_low_priority_quota(0),
          m_sat_jobs_default_absolute_osts(0),
          m_max_trees_to_build(0),
          m_max_trees(0),
          m_default_trees(0),
          m_dynamic_tree_allocation(false),
          m_dynamic_tree_algorithm(0),
          m_max_hosts_per_an(0),
          m_enable_reproducibility(false),
          m_default_reproducibility(false),
          m_enable_job_pkey_on_tree(false),
          m_enable_exclusive_lock(true),
          m_ib_port_guid(0),
          m_max_mads_on_the_wire(0),
          m_timeout(0),
          m_retries(0),
          m_am_key(0),
          m_am_key_protect_bit(false),
          m_am_key_lease_period(0),
          m_sa_key(0),
          m_sharp_sl(0),
          m_support_multicast(false),
          m_multicast_signature(0),
          m_trimming_mode_mask(0),
          m_ignore_sm_guids(false),
          m_smx_enabled_protocols(0),
          m_smx_protocol(0),
          m_smx_sock_port(0),
          m_smx_sock_backlog(0),
          m_ucx_wait_before_connection_close(0),
          m_smx_keepalive_min_time_before_connection_refresh(0),
          m_smx_keepalive_interval(0),
          m_smx_incoming_conn_keepalive_interval(0),
          m_smx_keepalive_refresh_interval(0),
          m_smx_keepalive_min_percentage_of_connections_to_refresh_at_iteration(0),
          m_log_verbosity(0),
          m_full_verbosity(false),
          m_syslog_verbosity(0),
          m_max_log_backups(0),
          m_max_log_size(0),
          m_accumulate_log(true),
          m_generate_dump_files(false),
          m_daemonize(false),
          m_thread_pool_threads(0),
          m_topology_type(TOPOLOGY_TYPE_NONE),
          m_max_tree_radix(0),
          m_span_all_agg_nodes(false),
          m_control_path_version(0),
          m_endianness(0),
          m_clean_an_on_discovery(false),
          m_clean_and_exit(false),
          m_lst_file_timeout(0),
          m_lst_file_retries(0),
          m_fabric_update_interval(0),
          m_local_port_validation_timeout(0),
          m_recovery_retry_interval(0),
          m_enable_parallel_links(false),
          m_job_reconnection_timeout(0),
          m_enable_seamless_restart(false),
          m_seamless_restart_max_retries(0),
          m_main_timeout(-1),
          m_disable_cleanup_on_am_restart(false),
          m_allow_remote_sm(false),
          m_am_retries(0),
          m_reservation_mode(false),
          m_load_reservations(false),
          m_reservation_force_guid_assignment(false),
          m_reservation_stop_jobs_upon_scale_in(true),
          m_force_app_id_match(false),
          m_enable_topology_api(false),
          m_enable_async_send(false),
          m_app_resources_default_limit(-1),
          m_disable_agg_nodes_upon_error(false),
          m_pending_mode_timeout_min(0),
          m_job_info_polling_interval(0),
          m_ibis_log_size(0),
          m_ibis_accum_log(false),
          m_ib_qpc_use_grh(false),
          m_ib_qpc_pkey(0),
          m_ib_qpc_ts(0),
          m_ib_qpc_sl(0),
          m_ib_sat_qpc_sl(0),
          m_ib_qpc_traffic_class(0),
          m_ib_qpc_rq_psn(0),
          m_ib_sat_qpc_rq_psn(0),
          m_ib_qpc_sq_psn(0),
          m_ib_sat_qpc_sq_psn(0),
          m_ib_qpc_rnr_mode(0),
          m_ib_sat_qpc_rnr_mode(0),
          m_ib_qpc_rnr_retry_limit(0),
          m_ib_sat_qpc_rnr_retry_limit(0),
          m_ib_qpc_local_ack_timeout(0),
          m_ib_sat_qpc_local_ack_timeout(0),
          m_ib_qpc_timeout_retry_limit(0),
          m_ib_sat_qpc_timeout_retry_limit(0),
          m_ib_sat_max_mtu(0),
          m_event_manager_backlog(0),
          m_show_help(false),
          m_show_version(false)
    {
        memset(m_per_prio_max_quota, 0, sizeof(m_per_prio_max_quota));
        memset(m_per_prio_default_llt_quota, 0, sizeof(m_per_prio_default_llt_quota));
        memset(m_per_prio_default_sat_quota, 0, sizeof(m_per_prio_default_sat_quota));
    }

    ~OptionInfo() {}

   private:
};

class OptionManager
{
    string m_option_file_;
    sharp_opt_parser m_parser_;

   public:
    explicit OptionManager(const string& option_file) : m_option_file_(option_file), m_parser_() {}

    ~OptionManager();

    int ReadOptions(int argc, char** argv);

    int WriteOptions(const string& config_file);

    void UpdateOptions();

    int OptionsDiff();

    void ValidateOptions();

    void ShowUsage();

    static int ValidateMaxQuota(int trees_per_job,
                                int osts_per_tree,
                                int user_data_per_ost,
                                int groups_per_tree,
                                int qps_per_tree_per_host,
                                char* err_str,
                                size_t err_str_len);

    static int ValidateDefaultQuota(int trees_per_job,
                                    int osts_per_tree,
                                    int user_data_per_ost,
                                    int groups_per_tree,
                                    int qps_per_tree_per_host,
                                    char* err_str,
                                    size_t err_str_len);

    static int ParsePerPrioQuotaString(const char* str, void* val, const void* func, const void*, char* err_str, size_t err_str_len);

    static int ParseQuotaString(const char* str, void* val, const void* func, const void*, char* err_str, size_t err_str_len);

    static int ParseCppString(const char* str, void* val, const void*, const void*, char* err_str, size_t err_str_len);

    static int ParseFolderPathCppString(const char* str, void* val, const void*, const void*, char* err_str, size_t err_str_len);

    static int ParseTopologyType(const char* str, void* val, const void*, const void*, char* err_str, size_t err_str_len);

    static int OptUpdateCb(const char* opt_name, const char* value_str, void* value, void* context);

    static int LogVerbosityUpdateCb(const char* opt_name, const char* value_str, void* value, void* context);

    static int SyslogVerbosityUpdateCb(const char* opt_name, const char* value_str, void* value, void* context);

    static int UpdateCppString(const char* opt_name, const char* value_str, void* value, void* context);

   private:
    int ReadOptionsEnv();
    static int ValidateQuota(int osts_per_tree,
                             int user_data_per_ost,
                             int groups_per_tree,
                             int qps_per_tree_per_host,
                             char* err_str,
                             size_t err_str_len);
};

// Recognize the global variables
extern OptionManager g_option_manager;
extern OptionInfo g_option_info;

#endif   // OPTION_MANAGER_H_
