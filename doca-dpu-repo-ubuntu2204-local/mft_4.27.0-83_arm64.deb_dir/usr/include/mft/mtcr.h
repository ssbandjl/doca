/*
 * Copyright (c) 2013-2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

/*
 *
 *  mtcr.h - Mellanox Software tools (mst) driver definitions
 *
 */

#ifndef _MST_H
#define _MST_H

#include "mtcr_com_defs.h"
#include "mtcr_mf.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int set_i2c_freq(mfile* mf, u_int8_t freq);
    int get_i2c_freq(mfile* mf, u_int8_t* freq);
    int get_mtusb_version(mfile* mf, unsigned int* major_number, unsigned int* minor_number);
    int get_mtusb_serial_number(mfile* mf, unsigned int* serial_number);
    int is_pci_device(mfile* mf);
    void get_pci_dev_name(mfile* mf, char* buf);
    void get_pci_dev_rdma(mfile* mf, char* buf);

    /*
     * Get list of MST (Mellanox Software Tools) devices.
     * Put all device names as null-terminated strings to buf.
     *
     * Return number of devices found or -1 if buf overflow
     */
    MTCR_API int mdevices(char* buf, int len, int mask);

    /*
     * Get list of MST (Mellanox Software Tools) devices.
     * Put all device names as null-terminated strings to buf.
     * verbosity if you want to see all pci ports devices or not.
     *
     * Return number of devices found or -1 if buf overflow
     */
    MTCR_API int mdevices_v(char* buf, int len, int mask, int verbosity);

    /*
     * Get list of MST (Mellanox Software Tools) devices info records.
     * Return a dynamic allocated array of dev_info records.
     * len will be updated to hold the array length
     *
     */
    MTCR_API dev_info* mdevices_info(int mask, int* len);

    /*
     *  * Get list of MST (Mellanox Software Tools) devices info records.
     *  * Return a dynamic allocated array of dev_info records.
     *  * len will be updated to hold the array length
     *  * Verbosity will decide whether to get all the Physical functions or not.
     */

    MTCR_API dev_info* mdevices_info_v(int mask, int* len, int verbosity);

    /*
     * Destroy the array of dev_info recored obtained by mdevices_info\() function
     *
     */
    MTCR_API void mdevices_info_destroy(dev_info* dev_info, int len);

    /*
     * Open Mellanox Software tools (mst) driver.
     * Return valid mfile ptr or 0 on failure
     */
    MTCR_API mfile* mopend(const char* input_name, DType dtype);

    /*
     * Open Mellanox Software tools (mst) driver.
     * Return valid mfile ptr or 0 on failure
     */
    MTCR_API mfile* mopen_adv(const char* name, MType mtype);

    /*
     * Open Mellanox Software tools (mst) driver. Device type=InfiniHost MType=MST_DEFAULT
     * Return valid mfile ptr or 0 on failure
     */
    MTCR_API mfile* mopen(const char* name);

    /*
     * Close Mellanox driver
     * req. descriptor
     */
    MTCR_API int mclose(mfile* mf);

    /*
     * Read 4 bytes, return number of succ. read bytes or -1 on failure
     */
    MTCR_API int mread4(mfile* mf, unsigned int offset, u_int32_t* value);

    /*
     * Write 4 bytes, return number of succ. written bytes or -1 on failure
     */
    MTCR_API int mwrite4(mfile* mf, unsigned int offset, u_int32_t value);

    /*
     * Read a block of dwords, return number of succ. read bytes or -1 on failure
     * Works for any interface, but can be faster for interfaces where bursts
     * are supported (MTUSB, IB).
     * Data retrns in the same endianess of mread4/mwrite4
     */
    MTCR_API int mread4_block(mfile* mf, unsigned int offset, u_int32_t* data, int byte_len);
    MTCR_API int mwrite4_block(mfile* mf, unsigned int offset, u_int32_t* data, int byte_len);

    /* read buffer as is without changing endians */
    MTCR_API int mread_buffer(mfile* mf, unsigned int offset, u_int8_t* data, int byte_len);

    /* Write buffer as is without changing endians */

    MTCR_API int mwrite_buffer(mfile* mf, unsigned int offset, u_int8_t* data, int byte_len);

    /*
     * Read up to 64 bytes, return number of succ. read bytes or -1 on failure
     */
    MTCR_API int mread64(mfile* mf, unsigned int offset, void* data, int length);

    /*
     * Write up to 64 bytes, return number of succ. written bytes or -1 on failure
     */
    MTCR_API int mwrite64(mfile* mf, unsigned int offset, void* data, int length);

    /*
     * Read up to 64 bytes, return number of succ. read bytes or -1 on failure
     */
    MTCR_API int mread_i2cblock(mfile* mf,
                                unsigned char i2c_secondary,
                                u_int8_t addr_width,
                                unsigned int offset,
                                void* data,
                                int length);

    /*
     * Write up to 64 bytes, return number of succ. written bytes or -1 on failure
     */
    MTCR_API int mwrite_i2cblock(mfile* mf,
                                 unsigned char i2c_secondary,
                                 u_int8_t addr_width,
                                 unsigned int offset,
                                 void* data,
                                 int length);

    /*
     * Set a new value for i2c_secondary
     * Return previous value
     */
    MTCR_API unsigned char mset_i2c_secondary(mfile* mf, unsigned char new_i2c_secondary);
    MTCR_API int mget_i2c_secondary(mfile* mf, unsigned char* new_i2c_secondary_p);

    MTCR_API int mset_i2c_addr_width(mfile* mf, u_int8_t addr_width);
    MTCR_API int mget_i2c_addr_width(mfile* mf, u_int8_t* addr_width);

    MTCR_API int mget_mdevs_flags(mfile* mf, u_int32_t* devs_flags);
    MTCR_API int mget_mdevs_type(mfile* mf, u_int32_t* mtype);
    /*
     * Software reset the device.
     * Return 0 on success, <0 on failure.
     * Currently supported for IB device only.
     * Mellanox switch devices support this feature.
     * HCAs may not support this feature.
     */
    MTCR_API int msw_reset(mfile* mf);

    /*
     * reset the device.
     * Return 0 on success, <0 on failure.
     * Curently supported on 5th Generation HCAs.
     */
    MTCR_API int mhca_reset(mfile* mf);

    MTCR_API int mi2c_detect(mfile* mf, u_int8_t slv_arr[SLV_ADDRS_NUM]);

    MTCR_API int maccess_reg(mfile* mf,
                             u_int16_t reg_id,
                             maccess_reg_method_t reg_method,
                             void* reg_data,
                             u_int32_t reg_size,
                             u_int32_t r_size_reg, // used when sending via icmd interface (how much data should be read
                                                   // back to the user)
                             u_int32_t w_size_reg, // used when sending via icmd interface (how much data should be
                                                   // written to the scratchpad) if you dont know what you are doing
                                                   // then r_size_reg = w_size_reg = your_register_size
                             int* reg_status);

    /**
     * Handles the send command procedure.
     * for completeness, but calling it is strongly advised against.
     * @param[in] dev   A pointer to a device context, previously
     *                  obtained by a call to <tt>gcif_open</tt>.
     * @return          One of the GCIF_STATUS_* values, or a raw
     *                  status value (as indicated in cr-space).
     * NOTE: when calling this function the caller needs to make
     *      sure device supports icmd.
     **/
    MTCR_API int icmd_send_command(mfile* mf, int opcode, void* data, int data_size, int skip_write);

    /**
     * Clear the Tools-HCR semaphore. Use this when an application
     * that uses this library is not terminated cleanly, leaving the
     * semaphore in a locked state.
     * @param[in] dev   A pointer to a device context, previously
     *                  obtained by a call to <tt>gcif_open</tt>.
     * @return          One of the GCIF_STATUS_* values, or a raw
     *                  status value (as indicated in cr-space).
     * NOTE: when calling this function the caller needs to make
     *      sure device supports icmd.
     **/
    MTCR_API int icmd_clear_semaphore(mfile* mf);

    /*
     * send an inline command to the tools HCR
     * limitations:
     * command should not use mailbox
     * NOTE: when calling this function caller needs to make
     *       sure device support tools HCR
     */
    MTCR_API int tools_cmdif_send_inline_cmd(mfile* mf,
                                             u_int64_t in_param,
                                             u_int64_t* out_param,
                                             u_int32_t input_modifier,
                                             u_int16_t opcode,
                                             u_int8_t opcode_modifier);

    /*
     * send a mailbox command to the tools HCR
     * limitations:
     * i.e write data to mailbox execute command (op = opcode op_modifier= opcode_modifier) and read data back from
     * mailbox data_offs_in_mbox: offset(in bytes) to read and write data to and from mailbox should be quad word
     * alligned.
     *  * NOTE: when calling this function caller needs to make
     *       sure device support tools HCR
     */
    MTCR_API int tools_cmdif_send_mbox_command(mfile* mf,
                                               u_int32_t input_modifier,
                                               u_int16_t opcode,
                                               u_int8_t opcode_modifier,
                                               int data_offs_in_mbox,
                                               void* data,
                                               int data_size,
                                               int skip_write);

    MTCR_API int tools_cmdif_unlock_semaphore(mfile* mf);

    /*
     * returns the maximal allowed register size (in bytes)
     * according to the FW access method and access register method
     * or -1 if no restriction applicable
     *
     */
    MTCR_API unsigned int mget_max_reg_size(mfile* mf, maccess_reg_method_t reg_method);

    MTCR_API const char* m_err2str(MError status);

    MTCR_API int mvpd_read4(mfile* mf, unsigned int offset, u_int8_t value[4]);

    MTCR_API int mvpd_write4(mfile* mf, unsigned int offset, u_int8_t value[4]);

    MTCR_API int mget_vsec_supp(mfile* mf);
    MTCR_API int supports_reg_access_gmp(mfile* mf, maccess_reg_method_t reg_method);

    MTCR_API int mget_addr_space(mfile* mf);
    MTCR_API int mset_addr_space(mfile* mf, int space);

    MTCR_API int mclear_pci_semaphore(const char* name);

    MTCR_API int get_dma_pages(mfile* mf, struct mtcr_page_info* page_info, int page_amount);

    MTCR_API int release_dma_pages(mfile* mf, int page_amount);

    MTCR_API int read_dword_from_conf_space(mfile* mf, u_int32_t offset, u_int32_t* data);

    MTCR_API int MWRITE4_SEMAPHORE(mfile* mf, int offset, int value);

    MTCR_API int MREAD4_SEMAPHORE(mfile* mf, int offset, u_int32_t* ptr);

    MTCR_API int is_livefish_device(mfile* mf);

    MTCR_API void set_increase_poll_time(int new_value);

    MTCR_API int mcables_remote_operation_server_side(mfile* mf,
                                                      u_int32_t address,
                                                      u_int32_t length,
                                                      u_int8_t* data,
                                                      int remote_op);

    MTCR_API int mcables_remote_operation_client_side(mfile* mf,
                                                      u_int32_t address,
                                                      u_int32_t length,
                                                      u_int8_t* data,
                                                      int remote_op);

    MTCR_API int mlxcables_remote_operation_client_side(mfile* mf,
                                                        const char* device_name,
                                                        char op,
                                                        char flags,
                                                        const char* reg_name);
    MTCR_API int mcables_send_smp(mfile* mf,
                                  unsigned char* data,
                                  const unsigned int attribute_id,
                                  const unsigned int attribute_modifier,
                                  maccess_reg_method_t reg_method);

    MTCR_API int send_smp_set(mfile* mf,
                              unsigned char* data,
                              const unsigned int attribute_id,
                              const unsigned int attribute_modifier);

    MTCR_API int send_smp_get(mfile* mf,
                              unsigned char* data,
                              const unsigned int attribute_id,
                              const unsigned int attribute_modifier);

    MTCR_API int send_semaphore_lock_smp(mfile* mf, u_int8_t* data, sem_lock_method_t method);

    MTCR_API void set_force_i2c_address(int i2c_address);

    MTCR_API int read_device_id(mfile* mf, u_int32_t* device_id);

    MTCR_API int get_device_flags(const char* name);

    MTCR_API int is_pcie_switch_device(mfile* mf);

#ifdef __cplusplus
}
#endif

#endif
