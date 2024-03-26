/*
 * Copyright (c) 2022-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
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

#ifndef IBDIAGNET_EXPORT_API_H
#define IBDIAGNET_EXPORT_API_H

#include <infiniband/ibdm/Fabric.h>
#include "ibis/ibis_types.h"
#include "infiniband/ibdm/cable/CableRecordData.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EXPORT_API_VERSION 0x01000000

typedef void *export_session_handle_t;

typedef struct {
    u_int64_t node_guid;
    u_int64_t port_guid;
    u_int8_t port_num;
    u_int64_t remote_node_guid;
    u_int64_t remote_port_guid;
    u_int8_t remote_port_num;
    struct SMP_PortInfo *p_smp_port_info;
    struct SMP_MlnxExtPortInfo *p_smp_mlnx_ext_port_info;
    struct PM_PortCounters *p_pm_port_counters;
    struct PM_PortCountersExtended *p_pm_port_counters_extended;
    struct PM_PortExtendedSpeedsCounters *p_pm_port_extended_speeds_counters;
    struct PM_PortExtendedSpeedsRSFECCounters *p_pm_port_extended_speeds_rsfec_counters;
    struct PM_PortCalcCounters *p_pm_port_calc_counters;
    struct VendorSpec_PortLLRStatistics *p_vendor_spec_port_llr_statistics;
    struct PM_PortRcvErrorDetails *p_pm_port_rcv_error_details;
    struct PM_PortXmitDiscardDetails *p_pm_port_xmit_discard_details;
    struct PM_PortSamplesControl *p_pm_port_samples_control;
    cable_record_data_t *p_cable_record_data;

    // Congestion Control
    CC_CongestionPortProfileSettings *curr_port_profile_settings[IB_NUM_VL]; // per VL
    CC_CongestionSLMappingSettings   *curr_sl_mapping_settings;
    CC_CongestionHCAGeneralSettings  *curr_hca_general_settings;
    CC_CongestionHCARPParameters     *curr_hca_rp_parameters;
    CC_CongestionHCANPParameters     *curr_hca_np_parameters;
    CC_CongestionHCAStatisticsQuery  *curr_hca_statistics_query;
    CC_CongestionHCAAlgoConfig       *cc_hca_algo_config_sup;
    CC_CongestionHCAAlgoConfig       *cc_hca_algo_config[MAX_CC_ALGO_SLOT]; // per algo
    CC_CongestionHCAAlgoConfigParams *cc_hca_algo_config_params[MAX_CC_ALGO_SLOT]; // per algo
    CC_CongestionHCAAlgoCounters     *cc_hca_algo_counters[MAX_CC_ALGO_SLOT]; // per algo

} export_data_port_t;

typedef struct {
    u_int64_t node_guid;
    struct SMP_NodeInfo *p_smp_node_info;
    struct SMP_SwitchInfo *p_smp_switch_info;
    struct VendorSpec_GeneralInfo *p_vendor_spec_general_info;
    struct SMP_TempSensing *p_smp_temp_sensing;
    struct VS_SwitchNetworkInfo *p_switch_network_info;

    // Congestion Control
    CC_EnhancedCongestionInfo           *cc_congestion_info;
    CC_CongestionSwitchGeneralSettings  *curr_switch_general_settings;

} export_data_node_t;

/*
 * Define EXPORT_IMPL to get function declarations instead of function pointers typedefs.
 * Export library shall define EXPORT_IMPL before including this file:
 *
 * #define EXPORT_IMPL
 * #include "ibdiagnet_export_api.h"
 */
#ifdef EXPORT_IMPL
#define EXPORT_SYMBOL __attribute__((visibility("default")))

EXPORT_SYMBOL u_int32_t export_get_api_version();

EXPORT_SYMBOL export_session_handle_t export_open_session(u_int64_t);
EXPORT_SYMBOL void export_close_session(export_session_handle_t, int);

EXPORT_SYMBOL int export_data_port(export_session_handle_t, export_data_port_t *);
EXPORT_SYMBOL int export_data_node(export_session_handle_t, export_data_node_t *);
#else
typedef u_int32_t (*PF_export_get_api_version)();

typedef export_session_handle_t (*PF_export_open_session)(u_int64_t);
typedef void (*PF_export_close_session)(export_session_handle_t, int);

typedef int (*PF_export_data_node)(export_session_handle_t, export_data_node_t *);
typedef int (*PF_export_data_port)(export_session_handle_t, export_data_port_t *);
#endif

#ifdef __cplusplus
}
#endif

#endif /* IBDIAGNET_EXPORT_API_H */
