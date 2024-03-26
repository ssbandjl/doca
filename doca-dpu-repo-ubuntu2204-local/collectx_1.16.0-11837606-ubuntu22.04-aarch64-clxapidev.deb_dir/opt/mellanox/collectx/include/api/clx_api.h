/*
* Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) 2015-2016 Mellanox Technologies Ltd. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:

* 1. Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived from
* this software without specific prior written permission.

* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
* HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef API_CLX_API_H_
#define API_CLX_API_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
    #define BEGIN_C_DECLS  extern "C" {
    #define END_C_DECLS    }
#else
    #define BEGIN_C_DECLS
    #define END_C_DECLS
#endif

typedef void*    clx_api_schema_t;
typedef uint64_t clx_api_timestamp_t;
typedef uint8_t  clx_api_type_index_t;


typedef union clx_api_version_t {
    struct {
        uint8_t    major;
        uint8_t    minor;
        uint16_t   patch;
    };
    uint32_t hex;
} clx_api_version_t;


typedef enum clx_api_ipc_status_t {
    CLX_API_IPC_DISABLED = -1,
    CLX_API_IPC_SUCCESS = 0,
    CLX_API_IPC_FAILED,
} clx_api_ipc_status_t;

#define CLX_GUID_SIZE 16
typedef uint8_t clx_guid_t[CLX_GUID_SIZE];

/**
 * @brief Return status of IPC transport.
 *
 * @param vctx Pointer to clx_api context.
 *
 * @return  -1 if IPC is disabled from config.
 * @return  0 if IPC is connected.
 * @return  1 if latest IPC message is not delivered due to client disconnection.
 */
int clx_api_get_ipc_status(void* vctx);


/** @struct clx_api_provider_t
 * @brief Provider that is used by clx_api context to
 */
typedef struct clx_api_provider_t {
    clx_api_version_t   version;       ///< Version of provider. Will be saved as schema version.
    char*               name;          ///< Provider or counters group identifier.
    char*               description;   ///< Detailed description of the provider.
    void*               data;          ///< Implementation-specific provider data

    bool (*initialize)(void* ctx,  struct clx_api_provider_t* provider);  ///< Initializer callback, should be implemented by user
    ///< Add counters / event types in this callback.
    void (*finalize)(void* ctx, struct clx_api_provider_t* provider);
} clx_api_provider_t;


#define CLX_API_DATA_PATH_TEMPLATE          "{{year}}/{{month}}{{day}}/{{source}}/{{tag}}{{id}}.bin"


/** @struct clx_api_params_t
 *  @brief Parameters for clx_api containing configuration of data destination.
 *
 * Data will be collected into api buffer and will be written to binary files according to data_path_template or
 * will be exported if ipc/fluent bit/prometheus are enabled.
 * The recommended default for data_path_template is:
 *             CLX_DATA_PATH_TEMPLATE="{{year}}/{{month}}{{day}}/{{source}}/{{tag}}{{id}}.bin"
 * Using the convention guarantees not having unmanageable number of files in a single directory
 *
 * Data source_id and source_tag will be used as a part of data_path_template for writing data.
 */
typedef struct clx_api_params_t {
    char* source_id;           /**< {source} in data_path_template. Set to node name, guid, etc */
    char* source_tag;          /**< {tag} in data_path_template. Use empty-string tag for counters and non-empty for events. */
    bool  enable_opaque_events;  /**< set to true in order to allow opaque events sending, to false otherwise */

    char* exporters_dir;       /**< Path where exporters SO files are located */

    uint64_t buffer_size;      /**< Size of the internal buffer in bytes. Set to 0 to use default 60 * 1024 or set manually */

    // File writer parameters.
    bool  file_write_enabled;  /**< set to 1 to enable, to 0 to disable. */
    char* data_root;           /**< data_root in which data will be stored according to the naming convention*/
    char* schema_path;         /**< path to schema folder. Schema contains the meta-data which allows access binary data. */

    char* data_path_template;  /**< Defines the naming convention to be used when writing files */

    /// Once one of the limits is reached, the current file is closed and new a file is generated.
    size_t max_file_size;      /**< File size limit in bytes*/
    clx_api_timestamp_t max_file_age; /**< Time limit for using current binary file in microseconds*/

    // IPC parameters
    bool     ipc_enabled;                 /**< Flag for enabling IPC. set to 1 to enable IPC or to 0 to disable. */
    char*    ipc_sockets_dir;
    /**< Path to ipc sockets dir. Should contain clx ipc socket which is set in clx_config.ini.
     * Recommended default is "/tmp/ipc_sockets".
     */

    uint32_t ipc_max_reattach_time_msec;
    /**< Time limit for reattach tries. If limit is reached client is considered as not connected. Default is 10000=10sec*/
    uint8_t  ipc_max_reattach_tries;
    /**< Number of reattach tries during reattach period. Default is 10 tries */

    uint32_t ipc_socket_timeout_msec;
    /**< Timeout for IPC messaging socket. If timeout is reached during send_receive, client is considered as not connected.
     * Default is 3000=3sec
     */

    // TBD(romanpr): create parameters for prometheus and fluent bit. Now exporters can be enabled only via env variables
    // uint16_t prometheus_port;
    // char* prometheus_host;
    char* prometheus_endpoint;
    char* cset_dir_path;

    char*    netflow_collector_addr;
    uint16_t netflow_collector_port;
    uint16_t netflow_max_packet_size;
} clx_api_params_t;



/** @struct clx_api_value_type_t
 * @brief clx_api_value_type_t is a mode that will be accounted while querying data with clxcli.
 */
typedef enum clx_api_value_type_t {
    CLX_FIELD_VALUE_ABSOLUTE,    /**< Data will be queried as it was collected.
                                      Recommended to use with counters that can increase and decrease.*/
    CLX_FIELD_VALUE_RELATIVE,    ///< Data will be normalized according to first value. Can be used for non-decreasing counters.
    CLX_FIELD_VALUE_DERIVATIVE,  ///< Queried data will be displayed as differences between adjacent values.
} clx_api_value_type_t;

/** @struct clx_api_data_type_t
 * @brief List of data types for counters.
 *
 */
typedef enum clx_api_data_type_t {
    CLX_API_DATA_TYPE_UNDEFINED = 0,

    // Scalar types
    CLX_API_DATA_TYPE_UINT64,
    CLX_API_DATA_TYPE_FP64,
    CLX_API_DATA_TYPE_BIT64,
    CLX_API_DATA_TYPE_INT64,
    CLX_API_DATA_TYPE_STRING,
} clx_api_data_type_t;


/** @struct clx_api_counter_info_t
 * @brief structure that describes counter meta information.
 *
 */
typedef struct clx_api_counter_info_t {
    char*                  counter_name;   ///< Name for querying data
    char*                  description;    ///< Detailed description of counter.
    char*                  units;          ///< String of type "Mb/sec", "Kb/sec", etc. Set to empty string if not used.
    clx_api_data_type_t    value_type;     ///< Collectx data type for counter value.
    clx_api_value_type_t   counting_mode;  ///< Mode for querying collected data by clxcli.
    uint32_t               offset;         ///< Offset for counter value in counters buffer. Computed automatically by clx_api.
    uint32_t               length;         ///< Length of counter in bytes. Set for string type.
} clx_api_counter_info_t;


/** @struct clx_api_event_field_info_t
 * @brief clx_api_event_field_info_t is a structure that describes event field meta information.
 *
 * Event types are based on CollectX built-in types. Consider the following table:
 *   |  type name           |   CollectX alias                     |
 *   |----------------------|--------------------------------------|
 *   |  bool                |   BOOL                               |
 *   |  char                |   CHAR                               |
 *   |  short               |   SHORT                              |
 *   |  int                 |   INT                                |
 *   |  long                |   LONG                               |
 *   |  long long           |   LONGLONG                           |
 *   |  unsigned char       |   UCHAR                              |
 *   |  unsigned short      |   USHORT                             |
 *   |  unsigned int        |   UINT                               |
 *   |  unsigned long       |   ULONG                              |
 *   |  unsigned long long  |   ULONGLONG                          |
 *   |  float               |   FLOAT                              |
 *   |  double              |   DOUBLE                             |
 *   |  int8_t              |   INT8                               |
 *   |  int16_t             |   INT16                              |
 *   |  int32_t             |   INT32                              |
 *   |  int64_t             |   INT64                              |
 *   |  uint8_t             |   UINT8                              |
 *   |  uint16_t            |   UINT16                             |
 *   |  uint32_t            |   UINT32                             |
 *   |  uint64_t            |   UINT64                             |
 *   |  uint64_t            |   CLX_TYPE_TIMESTAMP or "timestamp"  |
 *   |  NULL                |   CLX_TYPE_NONE                      |
 */
typedef struct clx_api_event_field_info_t {
    const char*          field_name;       ///< Name of event field
    const char*          description;      ///< Event field description
    const char*          type_name;        ///< Name of Collectx built in type.
    clx_api_value_type_t counting_mode;    ///< Mode for querying collected data by clxcli.
    uint16_t             array_length;     ///< Array length for this event type. If set to 1 will be used as single value.
} clx_api_event_field_info_t;

/** @struct clx_api_read_opaque_event_info_t
 * @brief structure that describes an opaque event
 *
 */
typedef struct clx_api_read_opaque_event_info_t {
    clx_guid_t       app_id;         ///< user-defined application id to be sent with the \p data
    uint64_t         user_defined1;  ///< user-defined data to be sent with the \p data
    uint64_t         user_defined2;  ///< user-defined data to be sent with the \p data
    uint32_t         data_size;      ///< opaque event's data size
    const void*      data;           ///< opaque event's data
} clx_api_read_opaque_event_info_t;

extern const clx_guid_t CLX_API_READ_OPAQUE_EVENT_APP_ID_ANY;

#define NUM_OF_FIELDS(type) (sizeof(type)/sizeof(clx_api_event_field_info_t))

typedef enum clx_api_ts_error_t {
    CLX_API_OK = 0,
    CLX_API_NOMEMORY,
    CLX_API_DUPLICATE,
    CLX_API_UNDEFINED,
    CLX_API_NOT_IMPLEMENTED,
    CLX_API_SCHEMA_MAX_TYPE_NUM_REACHED,
} clx_api_ts_error_t;

/** @struct clx_api_opaque_event_meta_info_t
 * @brief Opaque event's metadata
 */
typedef struct {
    const char* key;  ///< Key
    const char* val;  ///< Value
} clx_api_opaque_event_meta_info_t;

#ifndef HAVE_EVENT_CONTEXT
    typedef void clx_api_context_t;
#endif


BEGIN_C_DECLS
// ======================= INITIALIZE / DESTROY FUNCTIONS =====================
/**
 * @brief Creates and returns API context based on parameters and provider
 *
 * @param p        Parameters with the configured data destinations.
 * @param provider Defines data to be collected. Implement .initialize callback with creation of counters / events.
 * @return         Pointer to clx_api context.
 */
void* clx_api_create_context(const clx_api_params_t* p, clx_api_provider_t* provider);

/**
 * @brief Clones and returns initial context vctx according to new parameters.
 *
 * Cloned contexts use the same provider, e.i. it will share the same schema.
 * Use clones when the same data should be collected from different sources.
 * Each clone will write the data to its own file, while all clones will share the same exporters.
 *
 * @param vctx Pointer to clx_api context.
 * @param p    parameters of original clx_api context with new source_id / source_tag.
 * @return     Pointer to clx_api context clone.
 * @return     NULL on fail.
 */
void* clx_api_clone_context(void* vctx, const clx_api_params_t* p);

/**
 * @brief Destroys clx_api context. Should be applied to all clones and original context.
 *
 * @param vctx Pointer to clx_api context.
 * @return     true
 * @return     false
 */
bool clx_api_destroy_context(void* vctx);


// ========================= CONFIGURATION FUNCTIONS ===========================
const clx_api_params_t* clx_api_get_params(void* vctx);
/// Get the parameters of context / clone `vctx`.

/**
 * @brief Check if file write is enabled for the clx_api context / clone.
 *
 * @param vctx Pointer to clx_api context
 * @return     true
 * @return     false
 */
bool clx_api_is_file_write_enabled(void* vctx);


/**
 * @brief Get binary file timestamp for the clx_api context / clone.
 *
 * @param vctx  Pointer to clx_api context
 * @return      timestamp of the current bin file.
 */
uint64_t clx_api_get_file_timestamp(void* vctx);

/**
 * @brief Override the CLX API context schema with one loaded from file
 *
 * @param vctx  Pointer to clx_api context
 * @param fname Counters or events schema file name
 * @return      true upon success, false otherwise
 */
bool clx_api_override_schema(void* vctx, const char* fname);

/**
 * @brief Get the counters schema size for the clx_api context / clone.
 *
 * @param vctx  Pointer to clx_api context
 * @return      counters schema size of the context.
 */
uint64_t clx_api_get_counters_schema_size(void* vctx);


/**
 * @brief Get current data root of the clx_api context / clone.
 *
 * @param vctx Pointer to clx_api context.
 * @return Path to data root.
 */
char* clx_api_get_data_root(void* vctx);


// ============================ DATA FUNCTIONS =================================
void clx_api_force_write(void* vctx);

/**
 * @brief Creates event type and stores it into clx_api schema.
 *
 * Fills the index of the event in the schema. Call this only in the provider's .initialize-callback.
 *
 * @param vctx       Pointer to clx_api context.
 * @param event_name Type name which will be used in schema.
 * @param fields     List of event fields.
 * @param num_fields Number of event fields
 * @param index      This field will be filled with the event type index in the schema.
 * @return     true
 * @return     false
 */
int clx_api_add_event_type(void* vctx, char* event_name, clx_api_event_field_info_t* fields, int num_fields, uint8_t* index);


/**
 * @brief Returns max number of types allowed in single clx_api schema.
 *
 * @return max number of types
 */
int clx_api_get_max_types_num(void);


/**
 * @brief Creates event type and stores it into clx_api schema.
 *
 * Fills the index of the event in the schema. Call this only in the provider's .initialize-callback.
 *
 * @param vctx       Pointer to clx_api context.
 * @param event_data Buffer with consecutive events of the same type.
 * @param ti         Event type in to write.
 * @param num_events Number of events in event_data buffer.
 * @return     true
 * @return     false
 */
bool clx_api_write_event(void* vctx, void* event_data, clx_api_type_index_t ti, int num_events);


/**
 * @brief Retrieve data stored at the server.
 *
 * @param vctx       Pointer to clx_api context.
 * @param key        Key that matches the target data
 * @param timestamp  retrieve only data that is newer than the specified time.
 * @return           string value
*/
char* clx_api_get_key_value_from_server(void* vctx, const char* key, uint64_t timestamp);


/**
 * @brief Retrieve data stored at the server.
 *
 * @param vctx       Pointer to clx_api context.
 * @param key_prefix Prefix that matches the key prefix of the target data
 * @param timestamp  retrieve only data that is newer than the specified time.
 * @return           stl value
*/
void* clx_api_get_key_prefix_value_from_server(void* vctx, const char* key_prefix, uint64_t timestamp);


/**
 * @brief Creates new counter and adds it to schema group.
 *
 * Fills the index of the counter in the schema group. Call this only in the provider's .initialize-callback.
 *
 * @param vctx        Pointer to clx_api context.
 * @param counter     Counter info to add.
 * @param group_name  Name of schema counter group where counter will be placed.
 * @param counter_num This field will be filled with the counter index in the group.
 * @return     true
 * @return     false
 */
bool clx_api_add_counter(void* vctx, clx_api_counter_info_t* counter, char* group_name, uint32_t* counter_num);


/**
 * @brief Get counters buffer to write counters manually.
 *
 * Counters buffer will be created as a free memory segment of clx_api internal buffer.
 * If there is not enough space, it will flush previously written data. Returns NULL on error.
 *
 * @param vctx        Pointer to clx_api context.
 * @param counter     Counter info to add.
 * @param group_name  Name of schema counter group where counter will be placed.
 * @param counter_num This field will be filled with the counter index in the group.
 * @return            Pointer to the clx_api buffer.
 * @return            NULL on the failure.
 */
void* clx_api_get_counters_buffer(void* vctx, uint64_t timestamp, uint32_t* data_size);


// =============================== IPC FUNCTIONS ===============================
void clx_api_ipc_connect_if_needed(void* vctx);
/// Use to make sure that IPC connected. If clx process restarted for some reason it will reestablish the connection.


// ================================== UTILS ====================================
/**
 * @brief Get the current time in CollectX format.
 *
 * @return Current timestamp in microseconds.
 */
uint64_t clx_api_get_timestamp(void);

/**
 * @brief Get the current time in CollectX format.
 *
 * @return Current string timestamp in microseconds - need to be freed
 */
char* clx_api_get_str_timestamp(void);

/**
 * @brief Get the time that corresponds the time_t in CollectX format.
 *
 * @return Timestamp in microseconds.
 */
uint64_t clx_api_timestamp_from_time(time_t t);

/**
 * @brief Get the time that corresponds the struct tm in CollectX format.
 *
 * @return Timestamp in microseconds.
 */
uint64_t clx_api_timestamp_from_tm(struct tm* tm);

/**
 * @brief Set the first timestamp for current data_page.
 *
 * Use to initialize first binary file start timestamp to non-default value if needed.
 * If the app that uses clx_api will restart, it will write to the new file.
 * To continue to write to the previous file reset timestamp according to the timestamp in the previous file.
 *
 * @param vctx      Pointer to clx_api context.
 * @param timestamp Timestamp to set.
 */
void clx_api_set_first_timestamp(void* vctx, uint64_t timestamp);


// ============================================================================ //
// =========================      CLX API READ      =========================== //
// ============================================================================ //


// ========================= INITIALIZATION FUNCTIONS ==========================

/**
 * @brief Creates and returns application context.
 *
 * Reads all counter schemas from directory "schema_dir".
 * @param schema_dir
 * @return void* pointer to created context.
 * @return NULL on error
 */
void* clx_api_read_create_context(char* schema_dir);


/**
 * @brief  Read and reatunt counter binary file meta information.
 *
 * Opens counter binary file \p filename, and returns file meta info. If schema for that file
 * was not found in \p schema_dir previously, tries to find schema from the binary
 * file folder.
 *
 * @param context  pointer to CLX API READ context
 * @param filename absolute or relative path to counters binary file.
 * @return void* pointer to file meta data.
 * @return NULL on error
 */
void* clx_api_open_counter_file(void* context, char* filename);


/**
 * @brief Creates and returns empty counter set for current \p file meta info.
 *
 * @param context pointer to CLX API READ context
 * @param file    pointer to file meta info
 * @return void* pointer to an empty counterset.
 * @return NULL on error
 */
void* clx_api_create_counterset(void* context, void* file);


// ============================ DATA READ FUNCTIONS ============================
/**
 * @brief Allocates and returns pointer to a buffer for the counterset.
 *
 * Buffer size depends on number and types of counters included in cset.
 * @param cset pointer to counterset.
 * @return void* pointer to buffer.
 * @return NULL on error
 */
void* clx_api_allocate_counters_buffer(void* cset);


/**
 * @brief Reads single block of counters data from \p file into the \p data buffer.
 * Only counters from \p cset are written to \p data buffer. The timestamp of the
 * data block will be written to \p timestamp.
 *
 * @param file       pointer to counters binary file meta information.
 * @param cset       pointer to counterset.
 * @param[out] timestamp  timestamp of the data block in usec.
 * @param source
 * @param data       counters data buffer that fits \p cset.
 * @return 1 if no more data to read OR cannot read the next data block
 * @return 0 if not finished.
 */
int clx_api_get_next_data(void* file, void* cset, uint64_t* timestamp, char* source, void* data);


// ========================== COUNTERS INFO FUNCTIONS ==========================
/**
 * @brief Returns all the counters of schema corresponding to input \p file and updates \p num_counters.
 *
 * @param context pointer to CLX API READ context
 * @param file    pointer to counters binary file meta information.
 * @param[out] num_counters number of all available counters.
 * @return clx_api_counter_info_t* array of all available counters.
 */
clx_api_counter_info_t* clx_api_get_all_counters(void* context, void* file, int* num_counters);


/**
 * @brief Returns all the counters included to the counter set \p cset.
 *
 * @param cset pointer to counterset.
 * @return clx_api_counter_info_t* array of counters from \p cset.
 */
clx_api_counter_info_t* clx_api_get_counters(void* cset);


/**
 * @brief Returns number of counters contained by counter set \p cset.
 *
 * @param cset pointer to counterset.
 * @return int number of counters from \p cset.
 */
int clx_api_get_num_counters(void* cset);


// ========================== ADD COUNTERS FUNCTIONS ===========================
/**
 * @brief Adds a counter matching exact \p name to counter set \p cset, if it is not already included.
 *
 * @param cset pointer to counterset.
 * @param name exact counter name.
 * @return  1 counter already present
 * @return  0 counter was added
 * @return -1 if counter not found
 */
int clx_api_add_counter_exact(void* cset, char* name);


/**
 * @brief Adds all the counters matching token \p tok to counter set \p cset, that were not included previously.
 *
 * \p tok is a name fragment or a string of format "tok1+tok2-tok".
 *In the last example, considers only counter names that
 * include both "tok1" and "tok2", and not include "tok3".
 * To match counter name from the beginning use "^tok", for name end use "tok$",
 * otherwise it will look for matches everywhere.
 * @param cset pointer to counterset.
 * @param tok a name fragment or a string of several fragments separated with + or -.
 * @return 1 on success
 * @return -1 on failure
 */
int clx_api_add_counters_matching(void* cset, char* tok);


/**
 * @brief Adds all available counters to counter set \p cset.
 *
 * @param cset pointer to counterset.
 * @return 1 on success
 * @return -1 on failure
 */
int clx_api_add_all_counters(void* cset);


// =========================== DATA ACCESS FUNCTIONS ===========================
/**
 * @brief Returns type of the \p idx -th counter of the counter set \p cset.
 * Available types are:
 *                      CLX_API_DATA_TYPE_UINT64,
 *                      CLX_API_DATA_TYPE_FP64,
 *                      CLX_API_DATA_TYPE_BIT64,
 *                      CLX_API_DATA_TYPE_INT64.
 * @param cset pointer to counter set.
 * @param idx index of counter in the \p cset.
 * @return clx_api_data_type_t counter value type.
 */
clx_api_data_type_t clx_api_get_type(void* cset, unsigned idx);


/**
 * @brief Read double value of \p idx counter from counter set \p cset from buffer \p data.
 *
 * @param cset pointer to counter set.
 * @param idx  index of counter in the \p cset.
 * @param data data buffer that is consistent with \p cset.
 * @return double counter value
 */
double   clx_api_get_double(void* cset, unsigned idx, void* data);


/**
 * @brief Read int64_t value of \p idx counter from counter set \p cset from buffer \p data.
 *
 * @param cset pointer to counter set.
 * @param idx  index of counter in the \p cset.
 * @param data data buffer that is consistent with \p cset.
 * @return int64_t counter value
 */
int64_t  clx_api_get_int64(void* cset, unsigned idx, void* data);


/**
 * @brief Read uint64_t value of \p idx counter from counter set \p cset from buffer \p data.
 *
 * @param cset pointer to counter set.
 * @param idx  index of counter in the \p cset.
 * @param data data buffer that is consistent with \p cset.
 * @return uint64_t counter value
 */
uint64_t clx_api_get_uint64(void* cset, unsigned idx, void* data);


/**
 * @brief Read string value of \p idx counter from counter set \p cset from buffer \p data.
 *
 * @param cset pointer to counter set.
 * @param idx  index of counter in the \p cset.
 * @param data data buffer that is consistent with \p cset.
 * @return string counter value
 */
char*    clx_api_get_str(void* cset, unsigned idx, void* data);


// ============================= CLEANUP FUNCTIONS =============================
/**
 * @brief Application context destructor.
 *
 * @param context pointer to CLX API READ context
 */
void clx_api_read_destroy_context(void* context);


/**
 * @brief Counter set destructor. Apply on each counter set.
 *
 * @param cset pointer to counter set.
 */
void clx_api_destroy_counter_set(void* cset);
//

/**
 * @brief File meta info destructor. Apply on each file.
 *
 * @param file    pointer to counters binary file meta information.
 */
void clx_api_destroy_and_close_file(void* file);

/**
 * @brief Creates and returns opaque events read context.
 *
 * Reads all counter schemas from directory \p schema_dir
 *
 * @param filename opaque events binary file name
 * @param schema_dir folder to read the schemas from
 * @param app_id application id of the opaque events to be extracted
 * @return void* pointer to created context.
 * @return NULL on error
 *
 * NOTE: use @ref CLX_API_READ_OPAQUE_EVENT_APP_ID_ANY to read events regardless their app_id.
 */
void* clx_api_read_opaque_events_create_context(const char* filename, const char* schema_path, const clx_guid_t app_id);

/**
 * @brief  Get next opaque event.
 *
 * @param roe_ctx pointer to a context received from \ref clx_api_read_opaque_events_context
 * @param info pointer to a structure with opaque event info to be filled by the function
 *
 * @return 1 if an event extracted
 * @return 0 if there are no more events
 * @return -1 if an error occurred
 */
int clx_api_read_opaque_events_get_next(void* roe_ctx, clx_api_read_opaque_event_info_t* info);

/**
 * @brief Close context destructor.
 *
 * @param enum_ctx pointer to an enumeration context data received from \ref clx_api_read_opaque_events_context
 */
void clx_api_read_opaque_events_destroy_context(void* roe_ctx);

// ============================= OPAQUE EVENTS API =============================

// ====================== DOCA API INITIALIZERS ============================

// for DOCA api:
// 1. clx_api_init_context_with_schema - init global context (create ts, create empty schema)
// 2. user fills schema types
// 3. clx_api_context_apply_schema     - update the schema and initialize client

/**
 * @brief Alternative initializer for DOCA TELEMETRY without provider.
 * Initializes type system with empty schema and returns context. User should call
 * it before adding types to schema.
 *
 * @param name name of schema
 * @param version
 * @return void* clx_api context with empty schema
 */
void* clx_api_init_context_with_schema(const char* name, clx_api_version_t version);


/**
 * @brief Alternative initializer for DOCA TELEMETRY without provider.
 * Prepares the schema to be ready to use. User should call this function
 * after adding types to schema to finalize initialization of internal services.
 *
 * @param vctx clx_api context containing a schema with user types
 * @param p    clx_api parameters
 * @return true  on no errors.
 * @return false on any errors.
 */
bool clx_api_context_apply_schema(void* vctx, const clx_api_params_t* p);

/**
 * @brief  write data as an opaque event
 *
 * @param vctx Pointer to clx_api context.
 * @param app_id user-defined application id to be sent with the \p data
 * @param user_defined1 user-defined data to be sent with the \p data
 * @param user_defined2 user-defined data to be sent with the \p data
 * @param data data to write as an opaque event
 * @param data_size size of \p data
 * @return true if the event write succeeds, false otherwise
 */
bool clx_api_opaque_event_write(void* vctx, const clx_guid_t app_id, uint64_t user_defined1, uint64_t user_defined2,
                                const void* data, uint32_t data_size);

/**
 * @brief  write data as an extended opaque event
 *
 * @param vctx Pointer to clx_api context.
 * @param meta Meta data. Array of @ref clx_api_opaque_event_meta_info_t where the last element has NULL key
 * @param app_id user-defined application id to be sent with the \p data
 * @param user_defined1 user-defined data to be sent with the \p data
 * @param user_defined2 user-defined data to be sent with the \p data
 * @param data data to write as an opaque event
 * @param data_size size of \p data
 * @return true if the event write succeeds, false otherwise
 */
bool clx_api_opaque_event_write_ex(void* vctx, const clx_api_opaque_event_meta_info_t* meta, const clx_guid_t app_id,
                                   uint64_t user_defined1, uint64_t user_defined2, const void* data, uint32_t data_size);

/**
 * @brief  get max data size for opaque events
 *
 * @return max data size
 */
uint32_t clx_api_opaque_event_max_data_size(void* vctx);

/** @struct clx_api_file_t
 * @brief Pointer to an object representing an events file
 */
typedef struct clx_api_file      clx_api_file_t;

/** @struct clx_api_field_set_t
 * @brief Pointer to an object representing an event field set
 */
typedef struct clx_api_field_set clx_api_field_set_t;

/** @struct clx_api_event_t
 * @brief Pointer to an object representing an event
 */
typedef struct clx_api_event clx_api_event_t;

/** @struct clx_api_counters_t
 * @brief Pointer to an object representing an counter
 */
typedef struct clx_api_counters clx_api_counters_t;

/** @struct clx_api_event_t
 * @brief Pointer to an object representing field set enumerator
 */
typedef struct clx_api_field_set_enum clx_api_field_set_enum_t;

/** @struct clx_api_field_info_t
 * @brief Pointer to an object representing a field info
 */
typedef struct clx_api_field_info {
    const char*         name;      ///< Field name
    clx_api_data_type_t type;      ///< Field type
    size_t              type_idx;  ///< Type index (if there are multiple types with the same name)
} clx_api_field_info_t;

/**
 * @brief  open an event file for reading
 *
 * @param filename event file name
 * @param schema_path path to schema folder. Schema contains the meta-data which allows access binary data.
 * @return non-NULL file pointer if succeeds, NULL otherwise
 */
clx_api_file_t* clx_api_file_open(const char* filename, const char* schema_path);

/**
 * @brief  get next event
 *
 * @param file pointer to a file object
 * @return non-NULL event pointer if succeeds, NULL otherwise
 *
 * NOTE: the event pointer is only valid until the next @clx_api_file_get_next_event call
 */
clx_api_event_t* clx_api_file_get_next_event(clx_api_file_t* file);

/**
 * @brief  get next counters
 *
 * @param file pointer to a file object
 * @return non-NULL event pointer if succeeds, NULL otherwise
 *
 * NOTE: the counters pointer is only valid until the next @clx_api_file_get_next_counter call
 */
clx_api_counters_t* clx_api_file_get_next_counters(clx_api_file_t* file);

/**
 * @brief  get name of the event
 *
 * @param event pointer to event object
 * @return event name string
 */
const char* clx_api_event_get_name(const clx_api_event_t* event);

/**
 * @brief  get event's timestamp
 *
 * @param event pointer to event object
 * @return event's timestamp
 */
clx_api_timestamp_t clx_api_event_get_ts(const clx_api_event_t* event);

/**
 * @brief  get counter's timestamp
 *
 * @param event pointer to counter object
 * @return counter's timestamp
 */
clx_api_timestamp_t clx_api_counters_get_ts(const clx_api_counters_t* event);

/**
 * @brief  close event file and destroy the file object
 *
 * @param file pointer to a file object
 */
void clx_api_file_close(clx_api_file_t* file);


/**
 * @brief  returns error state of file object
 *
 * @param file pointer to a file object
 */
bool clx_api_file_is_valid(const clx_api_file_t* file);

/**
 * @brief  create field set to be used for the event data access and interpretation
 *
 * @param schema_path path to schema folder. Schema contains the meta-data which allows access binary data.
 * @param fset_file field set file name
 *
 * @return non-NULL field set pointer if succeeds, NULL otherwise
 */
clx_api_field_set_t* clx_api_field_set_create(const char* schema_path, const char* fset_file);

/**
 * @brief  add a field to a fields set
 *
 * @param field_set pointer to a field set object
 * @param type_name desired type name
 * @param token desired token (see fset file documentation for more info)
 *
 * @return @true if succeeds, @false otherwise
 */
bool clx_api_field_set_add_token(clx_api_field_set_t* field_set, const char* type_name, const char* token);

/**
 * @brief  begin type fields enumeration
 *
 * @param field_set pointer to a field set object
 * @param type_name desired type name
 * @param all_fields @true if all fields should be enumerated, @false if only already selected fields should be enumerated
 *
 * @return non-NULL enumerator pointer if succeeds, NULL otherwise
 */
clx_api_field_set_enum_t* clx_api_field_set_enum_begin(clx_api_field_set_t* field_set, const char* type_name, bool all_fields);

/**
 * @brief  get next type field info
 *
 * @param enumerator pointer to ab enumerator object
 * @param info pointer to a structure that will be filled with the field info
 *
 * @return @true if succeeds, @false otherwise (no more fields)
 */
bool clx_api_field_set_enum_next(clx_api_field_set_enum_t* enumerator, clx_api_field_info_t* info);

/**
 * @brief  end type fields enumeration
 *
 * @param enumerator pointer to ab enumerator object
 */
void clx_api_field_set_enum_end(clx_api_field_set_enum_t* enumerator);

/**
 * @brief  read event data in accordance with the field set requirements
 *
 * @param field_set pointer to a field set object
 * @param event event object to read
 * @param num_fields a variable that will be set into number of event fields read
 *
 * @return true if succeeds, false otherwise
 *
 * NOTE: all the clx_api_field_set_get_...() functions work with the current event (the last read one) and the field index should be
 * in the range [0..@num_fields-1]
 */
bool clx_api_field_set_read(clx_api_field_set_t* field_set, const clx_api_event_t* event, size_t* num_fields);

/**
 * @brief  get event field name by index
 *
 * @param field_set pointer to a field set object
 * @param idx event field index
 *
 * @return event field name
 */
const char* clx_api_field_set_get_name(clx_api_field_set_t* field_set, size_t idx);

/**
 * @brief  get event field type by index
 *
 * @param field_set pointer to a field set object
 * @param idx event field index
 *
 * @return event field type
 */
clx_api_data_type_t clx_api_field_set_get_type(clx_api_field_set_t* field_set, size_t idx);

/**
 * @brief  get value of u64 event field
 *
 * @param field_set pointer to a field set object
 * @param idx event field index
 *
 * @return event field value
 *
 * NOTE: valid only for the fields with type CLX_API_DATA_TYPE_UINT64
 */
uint64_t clx_api_field_set_get_uint64(clx_api_field_set_t* field_set, size_t idx);

/**
 * @brief  get value of i64 event field
 *
 * @param field_set pointer to a field set object
 * @param idx event field index
 *
 * @return event field value
 *
 * NOTE: valid only for the fields with type CLX_API_DATA_TYPE_INT64
 */
int64_t clx_api_field_set_get_int64(clx_api_field_set_t* field_set, size_t idx);

/**
 * @brief  get value of double event field
 *
 * @param field_set pointer to a field set object
 * @param idx event field index
 *
 * @return event field value
 *
 * NOTE: valid only for the fields with type CLX_API_DATA_TYPE_FP64
 */
double clx_api_field_set_get_double(clx_api_field_set_t* field_set, size_t idx);

/**
 * @brief  get value of string event field
 *
 * @param field_set pointer to a field set object
 * @param idx event field index
 *
 * @return string event field value. To be freed by the caller.
 *
 * NOTE: valid only for the fields with type CLX_API_DATA_TYPE_STRING
 */
char* clx_api_field_set_get_string(clx_api_field_set_t* field_set, size_t idx);

/**
 * @brief  destroy the field set object
 *
 * @param field_set pointer to a field set object
 */
void clx_api_field_set_destroy(clx_api_field_set_t* field_set);

/**
 * @brief Create default-initialized API parameters object
 *
 * @return clx_api_params_t* pointer to the created parameter object
 */
clx_api_params_t* clx_api_params_create(void);

/**
 * @brief Initialize API parameters object with the default values
 *
 * @param params pointer to the parameters object
 * @return true if initialization succeeds
 * @return false otherwise
 */
bool clx_api_params_init(clx_api_params_t* params);

/**
 * @brief Copy content of the source API parameters object to the destination one.
 *        The destination has to be properly initialised, as its content will be attempted to erase.
 *
 * @param dst pointer to the destination API parameters object.
 * @param src pointer to the source API parameters object.
 * @return true if copying succeeds
 * @return false if memory allocation fails; the destination object might remain partially copied.
 */
bool clx_api_params_copy(clx_api_params_t* dst, const clx_api_params_t* src);

/**
 * @brief Destroy properly initialized API parameters object, releasing internal contents
 *
 * @param params pointer to the parameters object
 */
void clx_api_params_destroy(clx_api_params_t* params);

/**
 * @brief Destroy internal contents of the API parameters object, freeing the memory pointer
 *
 * @param params pointer to the parameters object
 */
void clx_api_params_delete(clx_api_params_t* params);

/**
 * @brief  destroy the parameter structure (deprecated)
 *
 * @param clx_api_params_t pointer to a parameter object
 */
void clx_api_destroy_params(clx_api_params_t* params);

/**
 * @brief  creates or updates internal statistics gauge with name counter_name and labels label_pairs
 *
 * @param vctx Pointer to clx_api context.
 * @param counter_name    Counter name
 * @param counter_value   Counter value
 * @param ts              Pointer to timestamp, if NULL no timestamp will be shown in statistics
 * @param label_pairs     Pointer to null-terminated pairs of metadata, which will be shown, if NULL no metadata will be shown

 * @return     true in case of success
 * @return     false otherwise
 */
bool clx_api_add_stat_uint64(void* vctx, const char* counter_name, uint64_t counter_value,
                             clx_api_timestamp_t* ts, const char* label_pairs[][2]);

/**
 * @brief  creates or updates internal statistics gauge with name counter_name and labels label_pairs
 *
 * @param vctx Pointer to clx_api context.
 * @param counter_name    Counter name
 * @param counter_value   Counter value
 * @param ts              Pointer to timestamp, if NULL no timestamp will be shown in statistics
 * @param label_pairs     Pointer to null-terminated pairs of metadata, which will be shown, if NULL no metadata will be shown

 * @return     true in case of success
 * @return     false otherwise
 */
bool clx_api_add_stat_double(void* vctx, const char* counter_name, double counter_value,
                             clx_api_timestamp_t* ts, const char* label_pairs[][2]);

/** @struct clx_api_fselect_ctx_t
 * @brief file selection context
 */
typedef struct clx_api_fselect_ctx clx_api_fselect_ctx_t;

/*!
  \def CLX_API_SELECT_ALL
  Special start/end timestamp value that means "any time"
*/
#define CLX_API_SELECT_ALL ((uint64_t)0)

/**
 * @brief  begin the file selection proccess according to the time limits
 *
 * @param data_root name of directory in which the data is stored according to the naming convention
 * @param ts_start lower limit to select
 * @param fname_template naming convention used for writing files
 * @param source desired {source} in data_path_template. Any, if NULL.
 *
 * @return file selection context, NULL if failed
 */
clx_api_fselect_ctx_t* clx_api_fselect_begin(const char* data_root, clx_api_timestamp_t ts_start, clx_api_timestamp_t ts_end,
                                             const char* fname_template, const char* source);

/**
 * @brief  begin the file selection proccess according to the time limits
 *
 * @param data_root name of directory in which the data is stored according to the naming convention
 * @param ts_start lower limit to select
 * @param fname_template naming convention used for writing files
 * @param sources desired {NULL terminated sources list} in data_path_template. Any, if NULL.
 *
 * @return file selection context, NULL if failed
 */
clx_api_fselect_ctx_t* clx_api_fselect_begin_ex(const char* data_root, clx_api_timestamp_t ts_start, clx_api_timestamp_t ts_end,
                                                const char* fname_template, const char** sources);

/**
 * @brief  get next selected file name
 *
 * @param f file selection context
 *
 * @return file name, NULL if done
 */
const char* clx_api_fselect_next(clx_api_fselect_ctx_t* f);

/**
 * @brief  end the file selection proccess
 *
 * @param f file selection context
 */
void clx_api_fselect_end(clx_api_fselect_ctx_t* f);

/** @struct clx_api_eselect_ctx_t
 * @brief event selection context
 */
typedef struct clx_api_eselect_ctx clx_api_eselect_ctx_t;

/**
 * @brief  begin the event selection proccess according to the time limits
 *
 * @param data_root name of directory in which the data is stored according to the naming convention
 * @param ts_start lower limit to select
 * @param fname_template naming convention used for writing files
 * @param source desired {source} in data_path_template. Any, if NULL.
 *
 * @return event selection context, NULL if failed
 */
clx_api_eselect_ctx_t* clx_api_eselect_begin(const char* data_root, clx_api_timestamp_t ts_start, clx_api_timestamp_t ts_end,
                                             const char* fname_template, const char* source);

/**
 * @brief  begin the event selection proccess according to the time limits
 *
 * @param data_root name of directory in which the data is stored according to the naming convention
 * @param ts_start lower limit to select
 * @param fname_template naming convention used for writing files
 * @param sources desired {NULL terminated sources list} in data_path_template. Any, if NULL.
 *
 * @return event selection context, NULL if failed
 */
clx_api_eselect_ctx_t* clx_api_eselect_begin_ex(const char* data_root, clx_api_timestamp_t ts_start, clx_api_timestamp_t ts_end,
                                                const char* fname_template, const char** sources);

/**
 * @brief  get next selected event
 *
 * @param e event selection context
 *
 * @return event, NULL if done
 */
clx_api_event_t* clx_api_eselect_next(clx_api_eselect_ctx_t* e);

/**
 * @brief  end the event selection proccess
 *
 * @param e event selection context
 */
void clx_api_eselect_end(clx_api_eselect_ctx_t* e);


typedef struct clx_api_cselect_ctx clx_api_cselect_ctx_t;

/**
 * @brief  begin the counters selection proccess according to the time limits
 *
 * @param data_root name of directory in which the data is stored according to the naming convention
 * @param ts_start lower limit to select
 * @param fname_template naming convention used for writing files
 * @param source desired {source} in data_path_template. Any, if NULL.
 *
 * @return counters selection context, NULL if failed
 */
clx_api_cselect_ctx_t* clx_api_cselect_begin(const char* data_root, clx_api_timestamp_t ts_start, clx_api_timestamp_t ts_end,
                                             const char* fname_template, const char* source);

/**
 * @brief  begin the counters selection proccess according to the time limits
 *
 * @param data_root name of directory in which the data is stored according to the naming convention
 * @param ts_start lower limit to select
 * @param fname_template naming convention used for writing files
 * @param sources desired {NULL terminated sources list} in data_path_template. Any, if NULL.
 *
 * @return counters selection context, NULL if failed
 */
clx_api_cselect_ctx_t* clx_api_cselect_begin_ex(const char* data_root, clx_api_timestamp_t ts_start, clx_api_timestamp_t ts_end,
                                                const char* fname_template, const char** sources);

/**
 * @brief  get next selected counters
 *
 * @param e counters selection context
 *
 * @return counters, NULL if done
 */
clx_api_counters_t* clx_api_cselect_next(clx_api_cselect_ctx_t* e);

/**
 * @brief  end the counters selection proccess
 *
 * @param e counters selection context
 */
void clx_api_cselect_end(clx_api_cselect_ctx_t* e);


END_C_DECLS

#endif  // API_CLX_API_H_
