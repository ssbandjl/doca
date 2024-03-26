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

#ifndef MLXREG_SDK_H_
#define MLXREG_SDK_H_

#include <stdint.h>

/* An environment variable can be used to provide an external adb file.
 * By default if the MFT is installed, then no need to set this variable, but if there is no MFT installed, then a path
 * to the external adb file (NIC or SWITCH) should be exported using this environment variable, e.g:
 * export EXTERNAL_ADB_PATH=/path/to/register_access_table.adb.
 */
#define EXTERNAL_ADB_PATH "EXTERNAL_ADB_PATH"

typedef enum
{
    GET = 1,
    SET,
    SET_READ_MODIFY_WRITE
} access_type;

typedef enum
{
    ERR_CODE_FAILD_TO_OPEN_MST_DEV = -1,
    ERR_CODE_FAILD_TO_SEND_ACCESS_REG = -2,
    ERR_CODE_FAILD_TO_PARSE_PARAMS = -3,
    ERR_CODE_FAILD_TO_INIT_REG_LIB = -4,
    ERR_CODE_FAILD_TO_FIND_REG_NODE = -5,
    ERR_CODE_FAILD_TO_PARSE_FIELD = -6,
    ERR_CODE_INVALID_FIELD_ARG = -7,
    ERR_CODE_INVALID_METHOD = -8,
} ERR_CODE;

typedef struct Field
{
    char name[128];
    uint32_t value;
} Field;

typedef struct RegisterMap
{
    uint32_t number_of_fields;
    Field* fields;
} RegisterMap;

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    /*
     * Function: send_prm_access_reg
     * ----------------------------
     * Send an access register request
     *
     * mst_dev:         mst device name.
     * reg_name:        access register name, e.g PDDR
     * method:          access register request method:
     *                      access_type::GET to read access register state
     *                      access_type::SET to write access register (other fields will be zero)
     *                      access_type::SET_READ_MODIFY_WRITE to write access register with the current fields values
     * params:          Access register parameters to send with the request, e.g "local_port=1,pnat=0"
     *
     * response_outbox: Access register response, returned in a RegisterMap struct
     * returns:         0 if access register request succeed, ERR_CODE if access register failed
     */
    int32_t send_prm_access_reg(const char* mst_dev,
                                const char* reg_name,
                                const access_type method,
                                const char* params,
                                RegisterMap* response_outbox);

    /*
     * Function: free_response_outbox
     * ------------------------------
     * De-allocate the fields inside the response_outbox map struct
     *
     * response_outbox:  Pointer to the response map of type RegisterMap
     */
    void free_response_outbox(RegisterMap* response_outbox);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* MLXREG_SDK_H_ */
