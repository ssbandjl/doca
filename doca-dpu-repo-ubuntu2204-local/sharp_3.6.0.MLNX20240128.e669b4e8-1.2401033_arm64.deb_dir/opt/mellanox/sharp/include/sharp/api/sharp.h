/**
 * Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef _SHARP_COLL_API_H
#define _SHARP_COLL_API_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief SHARP coll supported data types
 *
 * The enumeration list describes the data types supported by SHARP coll
 */
#ifdef __cplusplus
extern "C" {
#endif

enum sharp_datatype
{
    SHARP_DTYPE_UNSIGNED,       /**< 32-bit unsigned integer. */
    SHARP_DTYPE_INT,            /**< 32-bit integer. */
    SHARP_DTYPE_UNSIGNED_LONG,  /**< 64-bit unsigned integer. */
    SHARP_DTYPE_LONG,           /**< 64-bit integer. */
    SHARP_DTYPE_FLOAT,          /**< 32-bit long floating point number */
    SHARP_DTYPE_DOUBLE,         /**< 64-bit long floating point number. */
    SHARP_DTYPE_UNSIGNED_SHORT, /**< 16-bit unsigned short integer. */
    SHARP_DTYPE_SHORT,          /**< 16-bit short integer. */
    SHARP_DTYPE_FLOAT_SHORT,    /**< 16-bit floating point number. */
    SHARP_DTYPE_BFLOAT16,       /**< 16-bit Bfloat. */
    SHARP_DTYPE_UINT8,          /**< 8-bit  unsigned integer. */
    SHARP_DTYPE_INT8,           /**< 8-bit  integer. */
    SHARP_DTYPE_NULL            /**< NULL data type */
};

/**
 * @brief SHARP coll supported aggregation operations
 *
 * The enumeration list describes the aggregation operations supported by SHARP coll
 */
enum sharp_reduce_op
{
    SHARP_OP_MAX,    /**< maximum. */
    SHARP_OP_MIN,    /**< minimum. */
    SHARP_OP_SUM,    /**< sum. */
    SHARP_OP_PROD,   /**< product. */
    SHARP_OP_LAND,   /**< logical and.*/
    SHARP_OP_BAND,   /**< bit-wise and. */
    SHARP_OP_LOR,    /**< logical or. */
    SHARP_OP_BOR,    /**< bit-wise or. */
    SHARP_OP_LXOR,   /**< logical xor. */
    SHARP_OP_BXOR,   /**< bit-wise xor. */
    SHARP_OP_MAXLOC, /**< max value and location. */
    SHARP_OP_MINLOC, /**< min value and location. */
    SHARP_OP_NULL
};

/**
 * @brief SHARP coll status code
 *
 * The enumeration list describes the error codes returned by SHARP coll
 */
enum sharp_error_no
{
    SHARP_COLL_SUCCESS = 0,        /**< Success. */
    SHARP_COLL_ERROR = -1,         /**< Error. */
    SHARP_COLL_ENOT_SUPP = -2,     /**< Collective operation not supported. */
    SHARP_COLL_ENOMEM = -3,        /**< No memory. */
    SHARP_COLL_EGROUP_ALLOC = -4,  /**< SHARP Group alloc error. */
    SHARP_COLL_ECONN_TREE = -5,    /**< No connection to sharp tree. */
    SHARP_COLL_EGROUP_JOIN = -6,   /**< Not able to join sharp grou.*/
    SHARP_COLL_EQUOTA = -7,        /**< SHARP resource quota error. */
    SHARP_COLL_ESESS_INIT = -8,    /**< Cannot connect to SHARPD. */
    SHARP_COLL_EDEV = -9,          /**< SHARP device error. */
    SHARP_COLL_EINVAL = -10,       /**< Invalid value. */
    SHARP_COLL_EJOB_CREATE = -11,  /**< Cannot create SHARP job. */
    SHARP_COLL_ETREE_INFO = -12,   /**< SHARP tree info not found. */
    SHARP_COLL_ENOTREE = -13,      /**< No available SHARP trees. */
    SHARP_COLL_EGROUP_ID = -14,    /**< Wrong SHARP group ID. */
    SHARP_COLL_EOOB = -15,         /**< Out-Of-Band collective error. */
    SHARP_COLL_EGROUP_MCAST = -16, /**< Multicast target error. */
    SHARP_COLL_EGROUP_TRIM = -17,  /**< Group trim failed. */
    SHARP_COLL_ELOCK_FAILED = -18, /**< SAT lock failed (can retry) */
    SHARP_COLL_ELOCK_DENIED = -19, /**< SAT lock operation not permitted */
    SHARP_COLL_ENO_RESOURCE = -20, /**< Resource not available */
};

/**
 * @brief SHARP feature mask
 *
 * The enumeration list of sharp job features
 */
enum sharp_job_features
{
    SHARP_FEATURE_LLT = 1 << 0,
    SHARP_FEATURE_REPRODUCIBLE = 1 << 1,
    SHARP_FEATURE_SAT = 1 << 2,
    SHARP_FEATURE_SAT_EXCLUSIVE_LOCK = 1 << 3,
};

enum sharp_aggregation_mode
{
    SHARP_AGGREGATION_NONE = 0,  /**< Optimal mode is determined internally */
    SHARP_AGGREGATION_DATAGRAM,  /**< Force datagram aggregation for relevant collectives. */
    SHARP_AGGREGATION_STREAMING, /**< Force streaming aggregation for relevant collectives. */
};

/* Forward declarations */
struct sharp_coll_context;
struct sharp_coll_comm;

/*
 * @brief SHARP coll configuration descriptor
 *
 * This descriptor defines the configuration for SHARP coll initialization
 */
struct sharp_coll_config
{
    const char* ib_dev_list;     /**< IB device name, port list.  */
    int user_progress_num_polls; /**< Number of polls to do before calling user progress. */
    int coll_timeout;            /**< Timeout (msec) for collective operation, -1 - infinite */
    uint32_t flags;              /**< flags */
    int reserved[3];             /**< Reserved */
};

/**
 * Default SHARP COLL configuration.
 */
extern const struct sharp_coll_config sharp_coll_default_config;

/*
 * @brief SHARP coll Out-Of-Band collectives descriptor
 *
 * This descriptor defines list of OOB collectives application must implement
 * and provide for SHARP coll initialization
 */
struct sharp_coll_out_of_band_colls
{
    /**
     * @brief out-of-band broadcast
     *
     * The pointer refers to application defined out-of-band bcast collective.
     *
     * @param [in] context	User-defined context or NULL.
     * @param [in] buffer	Buffer to send/recv.
     * @param [in] len	Size of the buffer.
     * @param [in] root	Root of the broadcast.
     */
    int (*bcast)(void* context, void* buffer, int len, int root);

    /**
     * @brief out-of-band barrier
     *
     * The pointer refers to application defined out-of-band barrier collective.
     *
     * @param [in] context	User-defined context or NULL.
     */
    int (*barrier)(void* context);

    /**
     * @brief out-of-band gather
     *
     * The pointer refers to application defined out-of-band gather collective.
     *
     * @param [in] context	User-defined context or NULL.
     * @param [in] root	Root of the broadcast.
     * @param [in] sbuf	Buffer to send.
     * @param [in] rbuf	Buffer to recv.
     * @param [in] len	Size of the buffer.
     */
    int (*gather)(void* context, int root, void* sbuf, void* rbuf, int len);
};

/*
 * @brief SHARP coll group initialization spec descriptor
 *
 * This descriptor defines the list of application specification to create SHARP group.
 *
 */
struct sharp_coll_comm_init_spec
{
    int rank;                          /**< Unique process id in the group. */
    int size;                          /**< Size of the SHARP group. */
    void* oob_ctx;                     /**< External group context for OOB functions. */
    const uint32_t* group_world_ranks; /**< List of Global unique process ids of group members. */
    int reserved[2];                   /**< Reserved */
};

/**
 * #brief sharp coll init flags
 */
enum
{
    SHARP_COLL_HIDE_ERRORS = 1 << 0,                       /** Hide errors in job create flow. */
    SHARP_COLL_DISABLE_LAZY_GROUP_RESOURCE_ALLOC = 1 << 1, /** Disable lazy group resource allocation */
};

/*
 * @brief SHARP coll initialization spec descriptor
 *
 * This descriptor defines the list of application specification to initialize SHARP coll.
 *
 */
struct sharp_coll_init_spec
{
    uint64_t job_id;                               /**< Job unique ID */
    int world_rank;                                /**< Global unique process id. */
    int world_size;                                /**< Num of processes in the job. */
    int (*progress_func)(void);                    /**< External progress function. */
    int group_channel_idx;                         /**< local group channel index(0 .. (max - 1))*/
    struct sharp_coll_config config;               /**< @ref sharp_coll_config "SHARP COLL Configuration". */
    struct sharp_coll_out_of_band_colls oob_colls; /**< @ref sharp_coll_out_of_band_colls "List of OOB collectives". */
    int world_local_rank;                          /**< relative rank of this process on this node within its job. */
    int enable_thread_support;                     /**< enable multi threaded support. */
    void* oob_ctx;                                 /**< context for OOB functions in sharp_coll_init */
    int reserved[4];                               /**< Reserved */
};

/**
 * @brief SHARP coll context capabilities.
 *
 * This descriptor defines the list of capabilities supported by a given SHARP group.
 */
struct sharp_coll_caps
{
    int sharp_pkt_version; /**< Sharp packet version */
    uint64_t reserved[2];  /**< Reserved */
    struct
    {
        uint64_t dtypes;       /**< Flags supported from @ref sharp_datatype */
        uint64_t tag_dtypes;   /**< Flags supported from @ref sharp_datatype for MIN/MAX_LOC tag */
        uint64_t reduce_ops;   /**< Flags supported from @ref sharp_reduce_op */
        uint64_t feature_mask; /**< Supported feature mask */
        uint64_t reserved[4];  /**< Reserved */
    } support_mask;

    struct
    {
        int max_osts;           /**< OSTs per tree */
        int user_data_per_ost;  /**< Payload per OST */
        int max_groups;         /**< Groups per tree */
        int max_group_channels; /**< Group channels per */
        int osts_per_group;     /**< OSTs per group */
    } resources;
};

/**
 * @brief SHARP coll buffer types
 *
 * The enumeration list describes the buffer types supported in collective calls.
 * @note Only SHARP_DATA_BUFFER is implemented.
 */
enum sharp_data_buffer_type
{
    SHARP_DATA_BUFFER, /**< Contiguous buffer. */
    SHARP_DATA_IOV     /**< Vector input. */
};

/**
 * Maximal number of IOV entries in a vector
 */
#define SHARP_COLL_DATA_MAX_IOV 15

/**
 * @brief SHARP coll memory types
 *
 * The enumeration list describes the memory types based on its location
 */
enum sharp_data_memory_type
{
    SHARP_MEM_TYPE_HOST, /**< Default system memory */
    SHARP_MEM_TYPE_CUDA, /**< NVIDIA CUDA memory */
    SHARP_MEM_TYPE_LAST
};

/**
 * @brief SHARP coll structure for scatter-gather I/O.
 */
struct sharp_data_iov
{
    void* ptr;        /**< Pointer to a data buffer */
    size_t length;    /**< Length of the buffer in bytes */
    void* mem_handle; /**< memory handle returned from @ref sharp_coll_reg_mr */
};

/**
 * #brief sharp coll registration operation flags
 */
enum
{
    SHARP_COLL_REG_FIELD_DMABUF_FD = 1 << 0,
    SHARP_COLL_REG_FIELD_DMABUF_OFFSET = 1 << 1,
};

/**
 * #brief Operation parameters passed to @ref sharp_coll_reg_mr_v2
 */
struct sharp_coll_reg_params
{
    /* Mask of valid fields in this structure */
    uint64_t field_mask;

    /**
     * dmabuf file descriptor of the memory region to register.
     *
     * If is set along with its corresponding bit in the field_mask -
     * @ref SHARP_COLL_MEM_REG_FIELD_DMABUF_FD, the memory region will be
     * registered using dmabuf mechanism.
     */
    int dmabuf_fd;

    /**
     * When @ref sharp_coll_reg_params_t.dmabuf_fd is provided, this field
     * specifies the offset of the region to register relative to the start of
     * the underlying dmabuf region.
     *
     * If not set (along with its corresponding bit in the field_mask -
     * @ref SHARP_COLL_MEM_REG_FIELD_DMABUF_OFFSET it's assumed to be 0.
     */
    size_t dmabuf_offset;
};

/*
 * @brief SHARP coll input buffer description
 *
 * This descriptor defines the buffer description for SHARP coll operations
 *
 */
struct sharp_coll_data_desc
{
    enum sharp_data_buffer_type type;
    enum sharp_data_memory_type mem_type;
    int reserved[2];
    union
    {
        /* contiguous buffer */
        struct
        {
            void* ptr;        /**< contiguous data buffer. */
            size_t length;    /**< Buffer len. */
            void* mem_handle; /**< memory handle returned from @ref sharp_coll_reg_mr */
        } buffer;

        /* Scatter/gather list */
        struct
        {
            unsigned count;                /**< Number of IOV entries. */
            struct sharp_data_iov* vector; /**< IOV entries. */
        } iov;
    };
};

/*
 * @brief SHARP coll reduce collective specification
 *
 * This descriptor defines the input parameters for SHARP coll reduce/all-reduce/reduce-scatter operations
 *
 */
struct sharp_coll_reduce_spec
{
    int root;                              /**< [in] root process id (ignored for allreduce) */
    struct sharp_coll_data_desc sbuf_desc; /**< [in] source data buffer desc */
    struct sharp_coll_data_desc rbuf_desc; /**< [out] destination data buffer desc */
    enum sharp_datatype dtype;             /**< [in] data type @ref sharp_datatype */
    size_t length;                         /**< [in] reduce operation size */
    enum sharp_reduce_op op;               /**< [in] reduce operator @ref sharp_reduce_op */
    enum sharp_datatype tag_dtype;         /**< [in] Tag datatype for MIN-LOC/MAX-LOC op */
    enum sharp_aggregation_mode aggr_mode; /**< [in] Requested Aggregation mode @ref sharp_aggregation_mode */
    int stream_lock_batch_size;            /**< [in] Acquire the lock and retain for next #ops. lock/#ops/unlock optimization*/
    size_t offset;                         /**< [in] Offset of reduce scatter input. Valid only in reduce-scatter operation */
    int reserved[2];                       /**< Reserved */
};

/*
 * @brief SHARP coll broadcast collective specification
 *
 * This descriptor defines the input parameters for SHARP coll broadcast operation
 *
 */
struct sharp_coll_bcast_spec
{
    int root;                             /**< [in] root process id */
    struct sharp_coll_data_desc buf_desc; /**< [in,out] buffer desc to send/recv bcast data */
    size_t size;                          /**< [in] bcast size */
    int reserved[4];                      /**< Reserved */
};

/*
 * @brief SHARP coll broadcast collective specification v2
 *
 * This descriptor defines the input parameters for SHARP coll broadcast operation
 *
 */
struct sharp_coll_bcast_spec_v2
{
    int root;                              /**< [in] root process id */
    struct sharp_coll_data_desc sbuf_desc; /**< [in] buffer desc to send bcast data */
    struct sharp_coll_data_desc rbuf_desc; /**< [in] buffer desc to recv bcast data */
    size_t size;                           /**< [in] bcast size */
    int reserved[4];                       /**< Reserved */
};

/*
 * @brief SHARP coll allgather collective specification
 *
 * This descriptor defines the input parameters for SHARP coll allgather operation
 *
 */
struct sharp_coll_gather_spec
{
    struct sharp_coll_data_desc sbuf_desc; /**< [in] source data buffer desc */
    struct sharp_coll_data_desc rbuf_desc; /**< [out] destination data buffer desc */
    enum sharp_datatype dtype;             /**< [in] data type @ref sharp_datatype */
    size_t size;                           /**< [in] gather size */
    size_t offset;                         /**< [in] offset of gather receive*/
    int reserved[4];                       /**< Reserved */
};
/**
 * @brief SHARP coll context initialization
 *
 * This routine is initialize SHARP coll library and create @ref sharp_coll_context "SHARP coll context".
 * This is a collective, called from all processes of the job.
 *
 * @warning An application cannot call any SHARP coll routine before sharp_coll_init
 *
 * @param [in]	sharp_coll_spec		SHARP coll specification descriptor.
 * @param [out]	sharp_coll_context	Initialized @ref sharp_coll_context "SHARP coll context".
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_init(struct sharp_coll_init_spec* sharp_coll_spec, struct sharp_coll_context** sharp_coll_context);

/**
 * @brief SHARP coll context finalize
 *
 * This routine finalizes and releases the resources associated with
 * @ref sharp_coll_context "SHARP coll context". typically done once, just before the process ends.
 *
 * @warning An application cannot call any SHARP coll routine after sharp_coll_finalize
 *
 * @param [in] context	SHARP coll context to cleanup.
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_finalize(struct sharp_coll_context* context);

/**
 * @brief SHARP caps initialization
 *
 * This routine is initialize SHARP capabilities description.
 *
 * @param [in]  context     SHARP coll context to query.
 * @param [out] sharp_caps  Initialized @ref sharp_caps "SHARP capabilities".
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_caps_query(struct sharp_coll_context* context, struct sharp_coll_caps* sharp_caps);

/**
 * @brief Progress SHARP coll communication operations.
 *
 * This routine explicitly progresses all SHARP communication operation.
 * For example, this typically called from MPI progress context for MPI case.
 *
 * @param [in] context	SHARP coll context to progress.
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_progress(struct sharp_coll_context* context);

/**
 * @brief SHARP coll communicator(group) initialization
 *
 * This routine creates @ref sharp_coll_comm "SHARP coll group".
 * This is a collective, called from all processes of the SHARP group.
 *
 * @param [in]	context		Handle to SHARP coll context.
 * @param [in]	spec		Input @ref sharp_coll_comm_init_spec "SHARP coll group specification".
 * @param [out] sharp_coll_comm Handle to SHARP coll communicator(group)
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_comm_init(struct sharp_coll_context* context,
                         struct sharp_coll_comm_init_spec* spec,
                         struct sharp_coll_comm** sharp_coll_comm);

/**
 * @brief SHARP coll communicator cleanup
 *
 * This routine cleanup SHARP coll communicator handle returned from @ref sharp_coll_comm_init.
 *
 * @param [in] comm   SHARP coll communicator to destroy.
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_comm_destroy(struct sharp_coll_comm* comm);

/**
 * @brief SHARP coll barrier collective
 *
 * This routine is collective operation blocks until all processes call this routine .
 *
 * @param [in]	comm	SHARP coll communicator to run the barrier.
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_do_barrier(struct sharp_coll_comm* comm);

/**
 * @brief SHARP coll non-blocking barrier collective
 *
 * This routine is non blocking version of @ref sharp_coll_do_barrier "SHARP coll barrier".
 * The progress of this operation is tracked with return request handle
 *
 * @param [in]	comm	SHARP coll communicator to run the barrier.
 * @param [out] handle  Handle representing the communication operation.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_barrier_nb(struct sharp_coll_comm* comm, void** handle);

/**
 * @brief SHARP coll allreduce collective
 *
 * This routine aggregates the data from all processes of the group and
 * distributes the result back to all processes.
 *
 * @param [in]	comm	SHARP coll communicator to run the allreduce collective.
 * @param [in]	spec	Allreduce operation specification.
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_do_allreduce(struct sharp_coll_comm* comm, struct sharp_coll_reduce_spec* spec);

/**
 * @brief SHARP coll non-blocking allreduce collective
 *
 * This routine is non blocking version of @ref sharp_coll_do_allreduce "SHARP coll allreduce".
 * The progress of this operation is tracked with return request handle
 *
 * @param [in]	comm	SHARP coll communicator to run the allreduce collective.
 * @param [in]	spec	Allreduce operation specification.
 * @param [out] handle  Handle representing the communication operation.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_allreduce_nb(struct sharp_coll_comm* comm, struct sharp_coll_reduce_spec* spec, void** handle);

/**
 * @brief SHARP coll reduce collective
 *
 * This routine aggregates the data from all processes of the group to a specific process
 *
 * @param [in]	comm	SHARP coll communicator to run the reduce collective.
 * @param [in]	spec	Reduce operation specification.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_reduce(struct sharp_coll_comm* comm, struct sharp_coll_reduce_spec* spec);

/**
 * @brief SHARP coll non-blocking reduce collective
 *
 * This routine is non blocking version of @ref sharp_coll_do_reduce "SHARP coll Reduce".
 * The progress of this operation is tracked with return request handle
 *
 * @param [in]	comm	SHARP coll communicator to run the reduce collective.
 * @param [in]	spec	Allreduce operation specification.
 * @param [out] handle  Handle representing the communication operation.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_reduce_nb(struct sharp_coll_comm* comm, struct sharp_coll_reduce_spec* spec, void** handle);

/**
//  * @brief SHARP coll reduce-scatter collective
 *
 * This routine aggregates the data from all processes of the group and
 * leaved the reduced result scattered over the processes so that recvbuff on rank i
 * will contain the i-th block of the result. This also does partial reduce-scatter if
 * source buffer is not contain the full input vector.
 *
 * @param [in]	comm	SHARP coll communicator to run the reduce-scatter collective.
 * @param [in]	spec	Reduce-scatter operation specification.
 * Note: spec.length specifies the receive length. sendbuffer length is equal to nranks* spec.length,
 * if not then it perform partial reduce-scatter from spec.offset
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_reduce_scatter(struct sharp_coll_comm* comm, struct sharp_coll_reduce_spec* spec);

/**
 * @brief SHARP coll non-blocking reduce-scatter collective
 *
 * This routine aggregates the data from all processes of the group and
 * leaved the reduced result scattered over the processes so that recvbuff on rank i
 * will contain the i-th block of the result. This also does partial reduce-scatter if
 * source buffer is not contain the full input vector.
 *
 * @param [in]	comm	SHARP coll communicator to run the reduce-scatter collective.
 * @param [in]	spec	Reduce-scatter operation specification.
 * Note: spec.length specifies the receive length. sendbuffer length is equal to nranks* spec.length
 * if not then it perform partial reduce-scatter from spec.offset
 * @param [out] handle  Handle representing the communication operation.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_reduce_scatter_nb(struct sharp_coll_comm* comm, struct sharp_coll_reduce_spec* spec, void** handle);

/**
 * @brief SHARP coll allgather collective
 *
 * This routine gathers the data from all processes of the group at each process.
 * each process receives data from rank i at offset i*size in receive buffer.
 * It assumes receive size at each rank i equal to group_size * size.
 * In case of partial allgather (receive size < group_size * size) it gathers
 * from specific set of ranks based on the offset
 *
 * @param [in]	comm	SHARP coll communicator to run the allgather collective.
 * @param [in]	spec	Allgather operation specification.
 * Note:
 * sendbuf.length specifies the send length.
 * recvbuffer length is equal to nranks* spec.length in case of full allgather
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_allgather(struct sharp_coll_comm* comm, struct sharp_coll_gather_spec* spec);

/**
 * @brief SHARP coll non-blocking allgather collective
 *

 * This routine gathers the data from all processes of the group at each process.
 * each process receives data from rank i at offset i*size in receive buffer.
 * It assumes receive size at each rank i equal to group_size * size.
 * In case of partial allgather (receive size < group_size * size) it gathers
 * from specific set of ranks based on the offset
 *
 * @param [in]	comm	SHARP coll communicator to run the allgather collective.
 * @param [in]	spec	Allgather operation specification.
 * Note:
 * sendbuf.length specifies the send length.
 * recvbuffer length is equal to nranks* spec.length in case of full allgather
 * @param [out] handle  Handle representing the communication operation.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_allgather_nb(struct sharp_coll_comm* comm, struct sharp_coll_gather_spec* spec, void** handle);

/**
 * @brief SHARP coll broadcast collective
 *
 * This routine broadcast data from single process to all processes of the group
 *
 * @param [in]	comm	SHARP coll communicator to run the bcast collective.
 * @param [in]	spec	Bcast operation specification.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_bcast(struct sharp_coll_comm* comm, struct sharp_coll_bcast_spec* spec);

/**
 * @brief SHARP coll non blocking broadcast collective
 *
 * This routine is non-blocking version of @ref sharp_coll_do_bcast "SHARP coll Bcast".
 * The progress of this operation is tracked with return request handle
 *
 * @param [in]	comm	SHARP coll communicator to run the bcast collective.
 * @param [in]	spec	Bcast operation specification.
 * @param [out] handle  Handle representing the communication operation.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 */
int sharp_coll_do_bcast_nb(struct sharp_coll_comm* comm, struct sharp_coll_bcast_spec* spec, void** handle);

/**
 * @brief SHARP coll request test
 *
 * This routine tests for the completion of a specific non-blocking coll operation
 *
 * @param [in]	req	SHARP coll request handle
 *
 * @return Non-zero if request is complete, 0 otherwise
 */
int sharp_coll_req_test(void* handle);

/**
 * @brief SHARP coll request wait
 *
 * This routine returns when the operation identified by non-blocking collective request is complete.
 * The request object is deallocated by the call to sharp_coll_req_wait and the request handle is set to NULL.
 *
 * @param [in]	req	SHARP coll request handle
 *
 * @return Error code as defined by @ref sharp_error_no
 */

int sharp_coll_req_wait(void* handle);

/**
 * @brief SHARP coll request deallocate
 *
 * This routine deallocates request handle
 *
 * @param [in]	req	SHARP coll request handle
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_req_free(void* handle);

/**
 * @brief SHARP coll memory registration.
 *
 * This routine registers external mem buffer
 *
 * @param [in] context	SHARP coll context to progress.
 * @param [in] buf	Buffer to register
 * @param [in] size	length of the buffer in bytes
 * @param [out] mr	memory registration handle.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 * @note Only one outstanding registration supported. no registration cache.
 *
 */
int sharp_coll_reg_mr(struct sharp_coll_context* context, void* buf, size_t size, void** mr);

/**
 * @brief SHARP coll memory registration.
 *
 * This routine registers external mem buffer
 *
 * @param [in] context	SHARP coll context to progress.
 * @param [in] buf	Buffer to register
 * @param [in] size	length of the buffer in bytes
 * @param [out] mr	memory registration handle.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 * @note Only one outstanding registration supported. no registration cache.
 *
 */
int sharp_coll_reg_mr_v2(struct sharp_coll_context* context, void* buf, size_t size, const struct sharp_coll_reg_params* params, void** mr);

/**
 * @brief SHARP coll memory de-registration.
 *
 * This routine de-registers the MR.
 *
 * @param [in] context	SHARP coll context to progress.
 * @param [in] mr	memory registration handle.
 *
 * @return Error code as defined by @ref sharp_error_no
 */
int sharp_coll_dereg_mr(struct sharp_coll_context* context, void* mr);

/**
 * @brief SHARP coll print config flags
 *
 * The enumeration list describes bit masks for different options to print config flags
 */
enum config_print_flags
{
    SHARP_COLL_CONFIG_PRINT_CONFIG = 1, /**< basic configuration. */
    SHARP_COLL_CONFIG_PRINT_HEADER = 2, /**< Print header. */
    SHARP_COLL_CONFIG_PRINT_DOC = 4,    /**< full description. */
    SHARP_COLL_CONFIG_PRINT_HIDDEN = 8  /**< hidden options. */
};

/**
 * @brief SHARP coll print configuration
 *
 * This routine prints SHARP coll configuration to a stream.
 *
 * @param [in] stream		Output stream to print to.
 * @param [in] print_flags	Controls how the configuration is printed.
 *
 * @return Error code as defined by @ref sharp_error_no
 */

int sharp_coll_print_config(FILE* stream, enum config_print_flags print_flags, const char* exec_name);

/**
 * @brief SHARP coll print error string
 *
 * This routine returns error string for a given @ref sharp_error_no "SHARP coll error code".
 *
 * @param [in] error	SHARP coll error code.
 *
 * @return Error string
 */
const char* sharp_coll_strerror(int error);

/**
 * @brief SHARP coll print statistics
 *
 * This routine dumps SHARP coll usage statistics
 *
 * @param [in] context	SHARP coll context to progress.
 *
 * @return Error code as defined by @ref sharp_error_no
 *
 * @note It is expected to Out-Of_Band collectives are operational valid to get
 *	 accumulated stats dumps (SHARP_COLL_STATS_DUMP_MODE=2) during the finalize process.
 *
 */
int sharp_coll_dump_stats(struct sharp_coll_context* context);

#ifdef __cplusplus
}
#endif

#endif
