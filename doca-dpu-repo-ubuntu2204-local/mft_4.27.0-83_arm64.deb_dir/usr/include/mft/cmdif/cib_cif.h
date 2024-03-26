/*
 * Copyright © 2013-2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef _ICMD_LIB /* guard */
#define _ICMD_LIB

#ifdef __cplusplus
extern "C"
{
#endif

#include <mtcr.h>
#include <common/compatibility.h>
#include "icmd_cif_common.h"
#include "icmd_cif_open.h"

    typedef enum gcif_context_type_t
    {
        GCIF_CTYPE_MTT = 0x00,
        GCIF_CTYPE_SQ_QP_LIST_REQ = 0x01,
        GCIF_CTYPE_SQ_QP_LIST_RES = 0x02,
        GCIF_CTYPE_BSF = 0x03,
        GCIF_CTYPE_RDB = 0x04,
        GCIF_CTYPE_EXT_RDB = 0x05,
        GCIF_CTYPE_ICM_CRC = 0x06,
        GCIF_CTYPE_QP_COMMON = 0x07,
        GCIF_CTYPE_REQUESTOR_QPC = 0x08,
        GCIF_CTYPE_RESPONDER_QPC = 0x09,
        GCIF_CTYPE_MKEY = 0x0a,
        GCIF_CTYPE_SRQ = 0x0b,
        GCIF_CTYPE_CQ = 0x0c,
        GCIF_CTYPE_TRANS_STATUS_BSF = 0x0d,
        GCIF_CTYPE_TRANS_STATUS_KLM = 0x0e,
        GCIF_CTYPE_PSV = 0x0f,
        GCIF_CTYPE_EXT_PSV = 0x10,
        GCIF_CTYPE_SXDC = 0x11,
        GCIF_CTYPE_PIPA = 0x12,
        GCIF_CTYPE_TIMER = 0x13,
        GCIF_CTYPE_COUNTERS_QP = 0x14,
        GCIF_CTYPE_EQ = 0x15,
        GCIF_CTYPE_MSIX = 0x16,
        GCIF_CTYPE_SXD_GVMI_RATE_LIMITER = 0x17,
        GCIF_CTYPE_PORT0_PKEY = 0x18,
        GCIF_CTYPE_PORT0_GUID = 0x19,
        GCIF_CTYPE_PORT0_INFO = 0x1a,
        GCIF_CTYPE_PORT0_COUNTERS_GVMI = 0x1b,
        GCIF_CTYPE_PORT1_PKEY = 0x1c,
        GCIF_CTYPE_PORT1_GUID = 0x1d,
        GCIF_CTYPE_PORT1_INFO = 0x1e,
        GCIF_CTYPE_PORT1_COUNTERS_GVMI = 0x1f,
        GCIF_CTYPE_STEERING = 0x20,
        GCIF_CTYPE_LDB_CACHE = 0x21,
        GCIF_CTYPE_REQ_SL_CACHE = 0x22,
        GCIF_CTYPE_IRISC = 0x23,
        GCIF_CTYPE_SCRATCHPAD = 0x24,
        GCIF_CTYPE_SQ_POINTERS = 0x25,
        GCIF_CTYPE_SQ_TOKENS = 0x26,
        GCIF_CTYPE_TOC = 0x27,
        GCIF_CTYPE_FW_GVMI_CTX = 0x28,
        GCIF_CTYPE_FW_QPC = 0x29,
        GCIF_CTYPE_FW_MALLOC = 0x2a,
        GCIF_CTYPE_FW_PD = 0x2b,
        GCIF_CTYPE_FW_UAR = 0x2c,
        GCIF_CTYPE_FW_EQ = 0x2d,
        GCIF_CTYPE_FW_CQ = 0x2e,
        GCIF_CTYPE_FW_MKEY = 0x2f,
        GCIF_CTYPE_FW_SRQ = 0x30,
        GCIF_CTYPE_FW_GLOBAL = 0x31,
        GCIF_CTYPE_FW_SQ = 0x32,
        GCIF_CTYPE_GLOBAL_FW_GVMI_CTX = 0x33,

        GCIF_CTYPE_CMAS_QP_RWQ = 0xc9,
        GCIF_CTYPE_CMAS_QP_SWQ,
        GCIF_CTYPE_CMAS_SRQ_WQE,
        GCIF_CTYPE_CMAS_CQE_BUFF,
        GCIF_CTYPE_CMAS_QP_RDB,
        GCIF_CTYPE_CMAS_QP_SDB,
        GCIF_CTYPE_CMAS_SRQ_DB,
        GCIF_CTYPE_CMAS_CQE_RDB,
        GCIF_CTYPE_CMAS_CQE_ARM,
        GCIF_CTYPE_CMAS_EQE_BUFF,
        GCIF_CTYPE_CMAS_TAG_BUFF
    } gcif_context_type_t;

    typedef enum gcif_q_type_t
    {
        GCIF_QTYPE_SQ = 0,
        GCIF_QTYPE_RQ = 1,
        GCIF_QTYPE_CQ = 2,
        GCIF_QTYPE_EQ = 4,
        GCIF_QTYPE_RDB = 5,
        GCIF_QTYPE_SRQ = 6
    } gcif_q_type_t;

    typedef enum gcif_desc_type_t
    {
        GCIF_DESC_TYPE_STEERING_RES = 0,
        GCIF_DESC_TYPE_PKT_DESC = 1,
        GCIF_DESC_TYPE_RXT_RXS = 2,
        GCIF_DESC_TYPE_CTX_FETCH_QP = 3,
    } gcif_desc_type_t;

    typedef enum
    {
        ACCESS_HOST_MEM_WRITE = 0x0,
        ACCESS_HOST_MEM_READ = 0x1,
    } access_host_mem_rw_t;

    typedef enum
    {
        ACCESS_HOST_MEM_MTT = 0x0,
        ACCESS_HOST_MEM_VA,
        ACCESS_HOST_MEM_OFFSET,
        ACCESS_HOST_MEM_CMAS,
    } access_host_mem_addr_t;

    typedef enum
    {
        RQP_PI = 1,
        RQP_CI = 2,
        SQP_PI = 3,
        SQP_CI = 4,
        SRQ_PI = 5,
        SRQ_CI = 6,
        CQ_PI = 7,
        CQ_CI = 8,
        EQ_PI = 9,
        EQ_CI = 10
    } get_pi_ci_t;

    typedef u_int8_t gcif_bool;

    struct gcif_translation_t
    {
        u_int64_t pa;
        u_int64_t len;
    };

    /* --------- Functional API ---------------------------------------- */

    /**
     * A. Wated - Please fill in description.
     * @param[in]  dev           A pointer to a device context.
     * @param[in]  type          The type of context to read.
     * @param[in]  gvmi          GVMI number.
     * @param[in]  context_index The context index.
     * @param[out] context       The read context, returned
     *                           as-is.This buffer must be at least
     *                           <tt>context_size</tt> bytes in
     *                           size.
     * @return     One of the GCIF_STATUS_* values, or a raw
     *             status value (as indicated in cr-space).
     **/

    int gcif_read_context(mfile* mf,
                          gcif_context_type_t type,
                          u_int16_t gvmi,
                          u_int64_t context_index,
                          u_int32_t context_size,
                          u_int8_t* context);

    /**
     * A. Wated - Please fill in description.
     * @param[in]  dev           A pointer to a device context.
     * @param[in]  type          The type of context to write.
     * @param[in]  gvmi          GVMI number.
     * @param[in]  context_index The context index.
     * @param[out] context       The raw context to write - NULL
     *                           terminated.
     * @return     One of the GCIF_STATUS_* values, or a raw
     *             status value (as indicated in cr-space).
     **/
    int gcif_write_context(mfile* mf,
                           gcif_context_type_t type,
                           u_int16_t gvmi,
                           u_int64_t context_index,
                           u_int32_t context_size,
                           u_int8_t* context);
    struct wq_dump_icmd_read_q_entry;
    int gcif_read_q_entry(mfile* mf, OUT struct wq_dump_icmd_read_q_entry* icmd_read_q_entry);

    int gcif_read_icm(mfile* mf, IN u_int64_t address, IN u_int64_t length, OUT u_int8_t* data);

    int gcif_read_memory_by_mkey(mfile* mf, IN u_int32_t mkey, IN u_int64_t va, IN u_int64_t len, OUT u_int8_t* data);

    int gcif_write_memory_by_mkey(mfile* mf, IN u_int32_t mkey, IN u_int64_t va, IN u_int64_t len, IN u_int8_t* data);
    struct gcif_translation_t;
    int gcif_translate_memory_by_mkey(mfile* mf,
                                      IN u_int32_t mkey,
                                      IN u_int64_t va,
                                      IN u_int64_t len,
                                      OUT struct gcif_translation_t* translation_table);

    int gcif_read_memory_by_mtt_ptr(mfile* mf,
                                    IN u_int64_t mtt_ptr,
                                    IN u_int64_t offset,
                                    IN u_int32_t mtt_size,
                                    IN u_int32_t len,
                                    OUT u_int8_t* data);

    int gcif_write_memory_by_mtt_ptr(mfile* mf,
                                     IN u_int64_t mtt_ptr,
                                     IN u_int64_t offset,
                                     IN u_int32_t mtt_size,
                                     IN u_int32_t len,
                                     IN u_int8_t* data);

    int gcif_translate_memory_by_mtt_ptr(mfile* mf,
                                         IN u_int64_t mtt_ptr,
                                         IN u_int64_t offset,
                                         IN u_int32_t mtt_size,
                                         IN u_int32_t len,
                                         OUT struct gcif_translation_t* translation_table);

    int gcif_get_context_max_index(mfile* mf, IN gcif_context_type_t type, OUT u_int8_t* max_index);
    struct devmon_icmd_get_irisc_heart_beat;
    int gcif_get_irisc_heartbeats(mfile* mf, OUT struct devmon_icmd_get_irisc_heart_beat* irisc_heartbeats);
    struct devmon_icmd_get_boot_stage;
    int gcif_get_boot_stage(mfile* mf, OUT struct devmon_icmd_get_boot_stage* boot_stage);
    struct devmon_icmd_get_link_leds;
    int gcif_get_link_leds(mfile* mf, int port_num, OUT struct devmon_icmd_get_link_leds* link_leds);
    struct rx_sx_dump_icmd_read_g_rse_slice_desc;
    int gcif_read_rx_slice_packet(mfile* mf, u_int32_t slice_id, u_int32_t* packet_size, u_int8_t* data);
    int gcif_read_rx_slice_desc(mfile* mf, struct rx_sx_dump_icmd_read_g_rse_slice_desc* icmd_read_rx_slice_desc);

    int
      gcif_read_host_mem(mfile* mf, u_int16_t gvmi, u_int64_t pa, u_int64_t length, u_int16_t prcss_id, u_int8_t* data);

    int gcif_read_host_mem_adv(mfile* mf,
                               u_int16_t gvmi,
                               u_int64_t addr,
                               u_int64_t key_or_ptr,
                               u_int64_t length,
                               u_int8_t addr_type,
                               u_int16_t prcss_id,
                               u_int8_t* data);
    struct wq_dump_icmd_access_host_mem;
    int gcif_access_host_mem(mfile* mf,
                             struct wq_dump_icmd_access_host_mem* host_access,
                             u_int8_t* data,
                             access_host_mem_rw_t read_write,
                             access_host_mem_addr_t addr_type);

    int gcif_access_host_mem_full(mfile* mf,
                                  struct wq_dump_icmd_access_host_mem* host_access,
                                  u_int8_t* data,
                                  access_host_mem_rw_t read_write,
                                  access_host_mem_addr_t addr_type);

    int gcif_write_host_mem(mfile* mf,
                            u_int16_t gvmi,
                            u_int64_t addr,
                            u_int64_t key_or_ptr,
                            u_int64_t length,
                            u_int8_t addr_type,
                            u_int16_t prcss_id,
                            u_int32_t* data);
    struct rx_sx_dump_icmd_read_sx_wq_buffer;
    int gcif_read_wq_buffer(mfile* mf, struct rx_sx_dump_icmd_read_sx_wq_buffer* icmd_read_sx_wq_buffer);
    struct uc_gw_hdr_icmd_phy_uc_set_get_data;
    int gcif_phy_uc_set_get_data(mfile* mf, struct uc_gw_hdr_icmd_phy_uc_set_get_data* phy_uc_data);
    struct gearbox_reg_phy_uc_data_get_request;
    struct gearbox_reg_phy_uc_data_get_response;
    int gcif_phy_uc_get_data_gearbox(mfile* mf,
                                     struct gearbox_reg_phy_uc_data_get_request* request_phy_uc_get_data,
                                     struct gearbox_reg_phy_uc_data_get_response* response_phy_uc_get_data);
    struct gearbox_reg_phy_uc_data_set_request;
    struct gearbox_reg_phy_uc_data_set_response;
    int gcif_phy_uc_set_data_gearbox(mfile* mf,
                                     struct gearbox_reg_phy_uc_data_set_request* request_phy_uc_set_data,
                                     struct gearbox_reg_phy_uc_data_set_response* response_phy_uc_set_data);
    struct gearbox_reg_phy_uc_get_array_prop_get_request;
    struct gearbox_reg_phy_uc_get_array_prop_get_response;
    int gcif_phy_uc_get_array_prop_gearbox(
      mfile* mf,
      struct gearbox_reg_phy_uc_get_array_prop_get_request* request_phy_uc_arr_prop,
      struct gearbox_reg_phy_uc_get_array_prop_get_response* response_phy_uc_arr_prop);
    union cx4_fsdump_icmd_get_ft_list;
    int gcif_get_ft_list(mfile* mf, union cx4_fsdump_icmd_get_ft_list* ft_list);
    union cx4_fsdump_icmd_get_ft_info;
    int gcif_get_ft_info(mfile* mf, union cx4_fsdump_icmd_get_ft_info* ft_info);
    union cx4_fsdump_icmd_get_fg_list;
    int gcif_get_fg_list(mfile* mf, union cx4_fsdump_icmd_get_fg_list* fg_list);
    union cx4_fsdump_icmd_get_fg;
    int gcif_get_fg(mfile* mf, union cx4_fsdump_icmd_get_fg* fg);
    union cx4_fsdump_icmd_get_fte_list;
    int gcif_get_fte_list(mfile* mf, union cx4_fsdump_icmd_get_fte_list* fte_list);
    union cx4_fsdump_icmd_get_fte;
    int gcif_get_fte(mfile* mf, union cx4_fsdump_icmd_get_fte* fte);
    union cx4_fsdump_icmd_get_ste_resources_list;
    int gcif_get_ste_resources_list(mfile* mf, union cx4_fsdump_icmd_get_ste_resources_list* ste_resources_list);
    union cx4_fsdump_icmd_get_ste_open_resources;
    int gcif_get_ste_open_resources(mfile* mf, union cx4_fsdump_icmd_get_ste_open_resources* ste_open_resources);
    union cx6dx_fsdump_icmd_get_ste_resources_list;
    int gcif_get_ste_resources_list_cx6dx(mfile* mf,
                                          union cx6dx_fsdump_icmd_get_ste_resources_list* ste_resources_list);
    union cx6dx_fsdump_icmd_get_ste_open_resources;
    int gcif_get_ste_open_resources_cx6dx(mfile* mf,
                                          union cx6dx_fsdump_icmd_get_ste_open_resources* ste_open_resources);
    struct cx4_fsdump_icmd_access_steering_root;
    int gcif_access_steering_root(mfile* mf, struct cx4_fsdump_icmd_access_steering_root* access_steering_root);
    struct cx6dx_fsdump_icmd_access_steering_root;
    int gcif_access_steering_root_cx6dx(mfile* mf, struct cx6dx_fsdump_icmd_access_steering_root* access_steering_root);
    struct cx4_fsdump_icmd_access_ste;
    int gcif_read_ste(mfile* mf, struct cx4_fsdump_icmd_access_ste* read_ste);
    struct cx6dx_fsdump_icmd_access_ste;
    int gcif_read_ste_cx6dx(mfile* mf, struct cx6dx_fsdump_icmd_access_ste* read_ste);
    struct uc_gw_edr_icmd_phy_uc_get_array_prop;
    int gcif_phy_uc_get_array_prop_EDR(mfile* mf, struct uc_gw_edr_icmd_phy_uc_get_array_prop* phy_uc_arr_prop);
    struct uc_gw_hdr_icmd_phy_uc_get_array_prop;
    int gcif_phy_uc_get_array_prop_HDR(mfile* mf, struct uc_gw_hdr_icmd_phy_uc_get_array_prop* phy_uc_arr_prop);
    struct uc_gw_hdr_icmd_phy_uc_get_array_prop_px;
    int gcif_phy_uc_get_array_prop_px(mfile* mf, struct uc_gw_hdr_icmd_phy_uc_get_array_prop_px* phy_uc_arr_prop_px);
    struct uc_gw_hdr_icmd_phy_uc_get_array_prop_px;
    int gcif_phy_uc_get_array_prop_px_connectx6(mfile* mf,
                                                struct uc_gw_hdr_icmd_phy_uc_get_array_prop_px* phy_uc_arr_prop_px);
    struct uc_gw_edr_icmd_thermal_prot_en;
    int gcif_thermal_prot_en(mfile* mf, struct uc_gw_edr_icmd_thermal_prot_en* therm_prot);
    struct wq_dump_icmd_get_pi_ci;
    int gcif_qp_get_pi_ci(mfile* mf, struct wq_dump_icmd_get_pi_ci* get_pi_ci);

    int gcif_set_toolpf_tracer(mfile* mf, u_int64_t p_addr);
    struct spectrum_icmd_mdio_test;
    int gcif_gearbox_mdio_test(mfile* mf, struct spectrum_icmd_mdio_test* mdio_test);

#ifdef __cplusplus
}
#endif

#endif /* _ICMD_LIB guard */
