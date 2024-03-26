/*
 * Copyright © 2021-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef API_CLX_PLUGIN_RUNNER_H_
#define API_CLX_PLUGIN_RUNNER_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
    #define BEGIN_C_DECLS  extern "C" {
    #define END_C_DECLS    }
#else
    #define BEGIN_C_DECLS
    #define END_C_DECLS
#endif


typedef struct clx_plugin_runner_params_t {
    char* so_lib_name;                  ///< Path to plugin library to be loaded
    char* cust_opts_fname;              ///< Path to file with additional options should be used
    bool use_ipc;                       ///< Flag if IPC mechanism should be used, disabled by default
    char* ipc_dir;                      ///< Path to directory for storing fules used for IPC
    unsigned long sample_time_ms;       ///< Plugin sampling interval in milliseconds
    unsigned long num_iters;            ///< Amount of iteration for collecting data
    int log_level;                      ///< Debug log level in range [1 , 7]
    bool use_file_write;                ///< Flag if writing to binary files, enabled by default
    char* data_root;                    ///< Path to folder for storing binary files
    bool exit_on_plugin_load_error;     ///< Flag if to immediate exit on failing to load clx plugin
} clx_plugin_runner_params_t;

typedef struct clx_api_param_t {
    const char* key;
    const char* value;
} clx_api_param_t;

struct clx_plugin_runner_context_t;

/**
 * @brief Creates and returns context for clx_plugin_runner
 *
 * @param runner_params pointer to a clx_plugin_runner_params_t structure with parameters to be applied
 * @param custom_params pointer to a clx_api_param_t structure with parameters to be applied
 * 
 * @return Pointer to clx_plugin_runner_context_t context.
 */
struct clx_plugin_runner_context_t* clx_plugin_runner_init_context(const clx_plugin_runner_params_t* runner_params,
                                                                   const clx_api_param_t* custom_params);

/**
 * @brief Initializes loaded clx providers
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 *
 * @return 1 on failure
 * @return 0 on success
 */
int clx_plugin_runner_start(struct clx_plugin_runner_context_t* tester_ctx);

/**
 * @brief Stops loaded clx providers
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 *
 */
void clx_plugin_runner_stop(struct clx_plugin_runner_context_t* tester_ctx);

/**
 * @brief Destroys clx_plugin_runner
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 */
void clx_plugin_runner_destroy_context(struct clx_plugin_runner_context_t* ctx);

/**
 * @brief Returns number of loaded schemas in clx_plugin_runner_context_t
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 *
 * @return number of loaded clx provider's schemas
 */
size_t clx_plugin_runner_get_num_of_schemas(struct clx_plugin_runner_context_t* ctx);


typedef void(*on_data_callback_t)(int, size_t, const char*);

/**
 * @brief Collect a portion of data for passed schema index
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param schema_index index of registerd schema
 * @param on_data_callback delegate to handle collected data buffer
 *
 * @return true on failure
 * @return false on success
 */
bool clx_plugin_runner_do_loop_iteration(struct clx_plugin_runner_context_t* ctx,
                                         int schema_index,
                                         on_data_callback_t on_data_callback);

/**
 * @brief Collects a portions of data for all registered schemas for a configured amount of iterations
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param on_data_callback delegate to handle collected data buffer
 *
 * @return true on failure
 * @return false on success
 */
bool clx_plugin_runner_do_loop(struct clx_plugin_runner_context_t* ctx, on_data_callback_t on_data_callback);

/**
 * @brief Returns null-terminated string with schema description
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param schema_index index of registerd schema
 * @param pretty flag is schema should has pretty indention or not
 *
 * @return NULL on failure
 * @return string with schema description, returned string should be deallocated with free()
 */
char* clx_plugin_runner_get_schema(struct clx_plugin_runner_context_t* ctx, int schema_index, bool pretty);

/**
 * @brief Returns null-terminated string with schema description
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param schema_index index of registerd schema
 * @param pretty flag is schema should has pretty indention or not
 *
 * @return NULL on failure
 * @return string with schema description, returned string should be deallocated with free()
 */
char* clx_plugin_runner_get_schema_id(struct clx_plugin_runner_context_t* ctx, int schema_index);

/**
 * @brief Sets plugin library to be loaded
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param so_lib_name path to clx plugin library
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_so_lib_name(struct clx_plugin_runner_context_t* ctx, const char* so_lib_name);

/**
 * @brief Sets file with additional options should be used
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param cust_opts_fname path to file with options
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_opts_fname(struct clx_plugin_runner_context_t* ctx, const char* cust_opts_fname);

/**
 * @brief Points if IPC mechanism should be used, disabled by default
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param use_ipc bool flag
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_use_ipc(struct clx_plugin_runner_context_t* ctx, bool use_ipc);

/**
 * @brief Sets directory for storing fules used for IPC
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param ipc_dir path to file with options
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_ipc_dir(struct clx_plugin_runner_context_t* ctx, const char* ipc_dir);

/**
 * @brief Sets clx plugin sampling interval in milliseconds, default is 1 millisecond
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param sample_time_ms sampling interval in milliseconds
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_sample_time_ms(struct clx_plugin_runner_context_t* ctx, unsigned long sample_time_ms);

/**
 * @brief Sets amount of iteration for collecting data, default value is 1000
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param num_iters number of iterations, 0 means to collect indefinitely
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_num_iters(struct clx_plugin_runner_context_t* ctx, unsigned long num_iters);

/**
 * @brief Sets debug log level, ERROR log level is set by default
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param log_level debug log level, should be from 1 to 7
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_log_level(struct clx_plugin_runner_context_t* ctx, int log_level);

/**
 * @brief Enables writing to binary files, enabled by default
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param use_file_write, flag if file writing should be enabled or not
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_use_file_write(struct clx_plugin_runner_context_t* ctx, bool use_file_write);

/**
 * @brief Sets data root, by default set to "clx_plugin_runner_data_root"
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param data_root path to data root
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_data_root(struct clx_plugin_runner_context_t* ctx, const char* data_root);

/**
 * @brief Enables immediate exit on failing to load clx plugin, disabled by default
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param exit_on_plugin_load_error flag if immediate exit or not
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_exit_on_plugin_load_error(struct clx_plugin_runner_context_t* ctx,
                                                     bool exit_on_plugin_load_error);

/**
 * @brief Set enabled providers list
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param enabled_providers comma separated string with enabled providers names
 * @param len length of enabled_providers
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_enabled_providers(struct clx_plugin_runner_context_t* ctx, const char* enabled_providers, size_t len);

/**
 * @brief Set disabled providers list
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param disabled_providers comma separated string with disabled providers names
 * @param len length of disabled_providers
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_set_disabled_providers(struct clx_plugin_runner_context_t* ctx, const char* disabled_providers, size_t len);

/**
 * @brief Add a key-value entry to plugin options
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 * @param key option key
 * @param value option value
 *
 * @return true on success
 * @return false on failure
 */
bool clx_plugin_runner_add_plugin_option(struct clx_plugin_runner_context_t* ctx, const char* key, const char* value);

/**
 * @brief Initiates plugin runner's main loop termination
 *
 * @param ctx pointer to a clx_plugin_runner_context_t object
 *
 * @return true on failure
 * @return false on success
 */
bool clx_plugin_runner_initiate_loop_termination(struct clx_plugin_runner_context_t* ctx);


#endif  // API_CLX_PLUGIN_RUNNER_H_
