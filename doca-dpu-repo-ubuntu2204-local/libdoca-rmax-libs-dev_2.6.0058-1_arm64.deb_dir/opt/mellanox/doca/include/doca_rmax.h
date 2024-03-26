/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

/**
 * @file doca_rmax.h
 * @page DOCA RMAX
 * @defgroup DOCARMAX DOCA RMAX engine
 * DOCA RMAX library. For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_RMAX_H_
#define DOCA_RMAX_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <netinet/ip.h>

#include <doca_buf.h>
#include <doca_compat.h>
#include <doca_ctx.h>

/** CPU bitmask container */
typedef uint64_t doca_rmax_cpu_mask_t;

struct doca_rmax_in_stream;
struct doca_rmax_flow;

/** maximum CPU set size */
#define DOCA_RMAX_CPU_SETSIZE 1024
/** number of CPU bits per one cpu mask element */
#define DOCA_RMAX_NCPUBITS (8 * sizeof(doca_rmax_cpu_mask_t))

/**
 * @brief Data structure to describe CPU mask for doca_rmax internal thread
 */
struct doca_rmax_cpu_affinity_mask {
	/** CPU is included in affinity mask if the corresponding bit is set */
	doca_rmax_cpu_mask_t cpu_bits[DOCA_RMAX_CPU_SETSIZE / DOCA_RMAX_NCPUBITS];
};

/** @brief Type of input stream. */
enum doca_rmax_in_stream_type {
	DOCA_RMAX_IN_STREAM_TYPE_GENERIC = 0,
	/**< Generic stream */
	DOCA_RMAX_IN_STREAM_TYPE_RTP_2110,
	/**< SMPTE ST 2110 stream */
};

/**
 * @brief Input packet timestamp format (timestamp, when packet was received).
 */
enum doca_rmax_in_stream_ts_fmt_type {
	DOCA_RMAX_IN_STREAM_TS_FMT_TYPE_RAW_COUNTER = 0,
	/**< Raw number written by HW, representing the HW clock */
	DOCA_RMAX_IN_STREAM_TS_FMT_TYPE_RAW_NANO,
	/**< Time in nanoseconds */
	DOCA_RMAX_IN_STREAM_TS_FMT_TYPE_SYNCED,
	/**< Time in nanoseconds, synced with PTP grandmaster */
};

/**
 * @brief Incoming packet scatter mode, used by input stream
 */
enum doca_rmax_in_stream_scatter_type {
	DOCA_RMAX_IN_STREAM_SCATTER_TYPE_RAW = 0,
	/**< Store raw packet data including network headers */
	DOCA_RMAX_IN_STREAM_SCATTER_TYPE_ULP,
	/**< Store User-Level Protocol only data (discard network header up to L4) */
	DOCA_RMAX_IN_STREAM_SCATTER_TYPE_PAYLOAD,
	/**< Store payload data only (all headers will be discarded) */
};

/**
 * @brief Completion returned by input stream describing the incoming packets.
 *
 * @details Input stream starts to receive packets right after start and
 * attaching any flow.
 */
struct doca_rmax_in_stream_completion {
	uint32_t elements_count;
	/**< Number of packets received */
	uint64_t ts_first;
	/**< Time of arrival of the first packet */
	uint64_t ts_last;
	/**< Time of arrival of the last packet */
	uint32_t seqn_first;
	/**< Sequnce number of the first packet */
	uint32_t memblk_ptr_arr_len;
	/**< Number of memory blocks placed in memblk_ptr_arr. See @ref
	 * doca_rmax_in_stream_get_memblks_count.
	 */
	void **memblk_ptr_arr;
	/**< Array of pointers to the beginning of the memory block as
	 * configured by input stream create step. The offset between packets
	 * inside memory block can be queried by @ref
	 * doca_rmax_in_stream_get_memblk_stride_size
	 */
};

/**
 * @brief Detailed completion error information.
 */
struct doca_rmax_stream_error {
	int code;
	/**< Raw Rivermax error code */
	const char *message;
	/**< Human-readable error */
};

/*********************************************************************************************************************
 * DOCA RMAX - GLOBAL (SINGLETON)
 *********************************************************************************************************************/

/**
 * @brief Set affinity mask for the internal Rivermax thread
 *
 * @details
 * Must be called before doca_rmax_init().
 * @param [in] mask
 * Affinity mask. CPU is included in affinity mask if the corresponding bit is set.
 * By default affinity mask is not set, so internal thread can run on any CPU core.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - invalid affinity mask provided.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_set_cpu_affinity_mask(const struct doca_rmax_cpu_affinity_mask *mask);

/**
 * @brief Get affinity mask for the internal Rivermax thread
 *
 * @param [out] mask
 * Affinity mask. CPU is included in affinity mask if the corresponding bit is set.
 * If CPU affinity mask is unset return value is zeroed.
 *
 * @return
 * DOCA_SUCCESS - always.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_get_cpu_affinity_mask(struct doca_rmax_cpu_affinity_mask *mask);

/**
 * @brief DOCA RMAX library initalization.
 *
 * @details
 * This function initializes the DOCA RMAX global resources. This function must
 * be called after @ref doca_rmax_set_cpu_affinity_mask and before any other
 * DOCA RMAX library call.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_UNSUPPORTED_VERSION - unsupported Rivermax library version.
 * - DOCA_ERROR_INITIALIZATION - Rivermax initialization failed.
 * - DOCA_ERROR_NO_MEMORY - unable to allocate memory.
 * - DOCA_ERROR_NOT_FOUND - there are no supported devices.
 * - DOCA_ERROR_NOT_SUPPORTED - invalid or missing Rivermax license.
 * - DOCA_ERROR_UNEXPECTED - unexpected issue.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_init(void);

/**
 * @brief Uninitialize DOCA RMAX library.
 *
 * @details This function cleans up the DOCA RMAX resources. No DOCA RMAX
 * function may be called after calling this function.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INITIALIZATION - Rivermax is not initialized.
 * - DOCA_ERROR_IN_USE - library is in use.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_release(void);

/**
 * @brief Interrupt the currently executing DOCA RMAX function, if any.
 */
DOCA_EXPERIMENTAL
void doca_rmax_interrupt(void);

/**
 * @brief Query PTP clock capability for device.
 *
 * @param [in] devinfo
 * The device to query
 *
 * @return
 * DOCA_SUCCESS - PTP clock is supported.
 * DOCA_ERROR_NOT_SUPPORTED - PTP clock is not supported.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_INITIALIZATION - Rivermax is not initialized.
 * - DOCA_ERROR_OPERATING_SYSTEM - failed to acquire the IPv4 address from the OS
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_get_ptp_clock_supported(const struct doca_devinfo *devinfo);

/**
 * @brief Set the device to use for obtaining PTP time.
 *
 * @details The device must have PTP clock capability, see @ref
 * doca_rmax_get_ptp_clock_supported.
 *
 * @param [in] dev
 * Device to use for obtaining the PTP time.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_FOUND - there is no IPv4 address associated with this device.
 * - DOCA_ERROR_NOT_SUPPORTED - PTP clock is not supported by device.
 * - DOCA_ERROR_OPERATING_SYSTEM - failed to acquire the IPv4 address from the OS
 * - DOCA_ERROR_UNEXPECTED - unexpected issue.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_set_clock(struct doca_dev *dev);

/*********************************************************************************************************************
 * DOCA RMAX - INPUT STREAM
 *********************************************************************************************************************/

/**
 * @brief Create a DOCA RMAX input stream context.
 *
 * @details
 * Create input stream.
 *
 * @param [in] dev
 * Device where the stream will be created. Must have a valid IPv4 address.
 * @param [out] stream
 * The input stream context created for the DOCA RMAX. Non NULL upon success,
 * NULL otherwise.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - stream argument is a NULL pointer.
 * - DOCA_ERROR_NO_MEMORY - failed to alloc doca_rmax_in_stream.
 * - DOCA_ERROR_INITIALIZATION - failed to initialise DOCA context.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_create(struct doca_dev *dev, struct doca_rmax_in_stream **stream);

/**
 * @brief Destroy a DOCA input stream context.
 *
 * @details
 * Free all allocated resources associated with a DOCA RMAX input stream.
 *
 * @param [in] stream
 * The context to be destroyed
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - stream argument is a NULL pointer.
 * - DOCA_ERROR_BAD_STATE - context is not in idle state
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_destroy(struct doca_rmax_in_stream *stream);

/**
 * @brief Convert a DOCA RMAX input stream to DOCA context.
 *
 * @details
 * DOCA RMAX stream supports all stream operations: create/start/stop/destroy.
 * Only one device and one progress engine must be attached to a stream.
 *
 * @param [in] stream
 * The context to be converted
 *
 * @return
 * The matching doca_ctx instance in case of success,
 * NULL otherwise.
 */
DOCA_EXPERIMENTAL
struct doca_ctx *doca_rmax_in_stream_as_ctx(struct doca_rmax_in_stream *stream);

/**
 * @brief Get input stream type.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_type(const struct doca_rmax_in_stream *stream,
		enum doca_rmax_in_stream_type *value);

/**
 * @brief Get the type of packet's data scatter.
 *
 * @details See enum @ref doca_rmax_in_stream_scatter_type.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_scatter_type(const struct doca_rmax_in_stream *stream,
		enum doca_rmax_in_stream_scatter_type *value);

/**
 * @brief Get stream timestamp format.
 *
 * @brief See enum @ref doca_rmax_in_stream_ts_fmt_type
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_timestamp_format(const struct doca_rmax_in_stream *stream,
		enum doca_rmax_in_stream_ts_fmt_type *value);

/**
 * @brief Get number of elements in the stream buffer.
 *
 * @details This value can differ from value set by @ref
 * doca_rmax_in_stream_set_elements_count if the argument is not a power of
 * two.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_elements_count(const struct doca_rmax_in_stream *stream,
		uint32_t *value);

/**
 * @brief Get number of configured memory blocks.
 *
 * @details Amount of memblks is equal to the number of segments into which the
 * incoming packet will be divided.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_memblks_count(const struct doca_rmax_in_stream *stream,
		uint32_t *value);

/**
 * @brief Get minimal packet segment sizes.
 *
 * @details Array of minimal packet segment sizes that will be received by input
 * stream. Array length equals to the number of memory blocks in the stream buffer.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_UNEXPECTED - unexpected program flow.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_memblk_desc_get_min_size(const struct doca_rmax_in_stream *stream,
		uint16_t *value);

/**
 * @brief Get maximal packet segment sizes.
 *
 * @details Array of maximal packet segment sizes that will be received by input
 * stream. Array length equals to the number of memory blocks in the stream buffer.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_UNEXPECTED - unexpected program flow.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_memblk_desc_get_max_size(const struct doca_rmax_in_stream *stream,
		uint16_t *value);

/**
 * @brief Get size of memory block(s).
 *
 * @details Size of memory block (array of sizes for multiple memblks,
 * the number of memory blocks in stream is more than one).
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_INITIALIZATION - Rivermax is not initialized.
 * - DOCA_ERROR_UNEXPECTED - unexpected program flow.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_memblk_size(const struct doca_rmax_in_stream *stream,
		size_t *value);

/**
 * @brief Get stride size(s).
 *
 * @details Stride size of memory block (array of stride sizes for multiple memory blocks).
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_INITIALIZATION - Rivermax is not initialized.
 * - DOCA_ERROR_UNEXPECTED - unexpected program flow.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_memblk_stride_size(const struct doca_rmax_in_stream *stream,
		uint16_t *value);

/**
 * @brief Get minimal number of packets that input stream must return in read
 * event.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_min_packets(const struct doca_rmax_in_stream *stream,
		uint32_t *value);

/**
 * @brief Get maximal number of packets that input stream must return in read
 * event.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_max_packets(const struct doca_rmax_in_stream *stream,
		uint32_t *value);

/**
 * @brief Get receive timeout.
 *
 * @details The number of usecs that library would do busy wait (polling) for
 * reception of at least `min_packets` number of packets.
 *
 * @param [in] stream
 * The input stream to query.
 * @param [out] value
 * Where to write the current property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * Error code - in case of failure.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_get_timeout_us(const struct doca_rmax_in_stream *stream,
		int *value);

/**
 * @brief Set input stream type.
 *
 * @details Default: DOCA_RMAX_IN_STREAM_TYPE_GENERIC.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_type(struct doca_rmax_in_stream *stream,
		enum doca_rmax_in_stream_type value);

/**
 * @brief Set the type of packet's data scatter.
 *
 * @details See enum @ref doca_rmax_in_stream_scatter_type.
 * Default: DOCA_RMAX_IN_STREAM_SCATTER_TYPE_RAW.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_scatter_type(struct doca_rmax_in_stream *stream,
		enum doca_rmax_in_stream_scatter_type value);

/**
 * @brief Set stream timestamp format.
 *
 * @brief See enum @ref doca_rmax_in_stream_ts_fmt_type
 * Default: DOCA_RMAX_IN_STREAM_TS_FMT_TYPE_RAW_COUNTER.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_timestamp_format(struct doca_rmax_in_stream *stream,
		enum doca_rmax_in_stream_ts_fmt_type value);

/**
 * @brief Set number of elements in the stream buffer.
 *
 * @details Must be set before starting the stream context.
 * See also @ref doca_rmax_in_stream_get_elements_count.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_elements_count(struct doca_rmax_in_stream *stream,
		uint32_t value);

/**
 * @brief Set number of configured memory blocks.
 *
 * @details Amount of memblks is equal to the number of segments into which the
 * incoming packet will be divided.
 * Default: 1.
 * Valid values: 1 and 2.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_memblks_count(struct doca_rmax_in_stream *stream,
		uint32_t value);

/**
 * @brief Set minimal packet segment sizes.
 *
 * @details Array of minimal packet segment sizes that will be received by input
 * stream. Array length equals to the number of memory blocks in the stream buffer.
 * Default: 0.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_UNEXPECTED - unexpected program flow.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_memblk_desc_set_min_size(struct doca_rmax_in_stream *stream,
		const uint16_t *value);

/**
 * @brief Set maximal packet segment sizes.
 *
 * @details Array of maximal packet segment sizes that will be received by input
 * stream. Array length equals to the number of memory blocks in the stream buffer.
 * Must be set before starting the stream context.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_UNEXPECTED - unexpected program flow.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_memblk_desc_set_max_size(struct doca_rmax_in_stream *stream,
		const uint16_t *value);

/**
 * @brief Set memory buffer(s).
 *
 * @details Memory buffer (or head of linked list of memory buffers) for
 * storing received data. The length of linked list must be the same as number
 * of memory blocks configured. Must be set before starting the stream context.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] buf
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_UNEXPECTED - unexpected program flow.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_memblk(struct doca_rmax_in_stream *stream,
		struct doca_buf *buf);

/**
 * @brief Set minimal number of packets that input stream must return in read
 * event.
 *
 * @details Default: 0.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_min_packets(struct doca_rmax_in_stream *stream,
		uint32_t value);

/**
 * @brief Set maximal number of packets that input stream must return in read
 * event.
 *
 * @details Default: 1024.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_max_packets(struct doca_rmax_in_stream *stream,
		uint32_t value);

/**
 * @brief Set receive timeout.
 *
 * @details The number of usecs that library would do busy wait (polling) for
 * reception of at least `min_packets` number of packets.
 *
 * @details Default: 0.
 *
 * @param [in] stream
 * The input stream to write property.
 * @param [in] value
 * Property value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_set_timeout_us(struct doca_rmax_in_stream *stream,
		int value);

/*********************************************************************************************************************
 * DOCA RMAX - INPUT STREAM EVENTS AND TASKS
 *********************************************************************************************************************/

/**
 * Event triggered everytime data is received.
 * To register to this event see doca_rmax_in_stream_event_rx_data_register().
 * In case event success handler is called then doca_rmax_in_stream_event_rx_data_get_completion() can be used.
 * In case event error handler is called then doca_rmax_in_stream_event_rx_data_get_error() can be used.
 */
struct doca_rmax_in_stream_event_rx_data;

/**
 * @brief Function to be executed once Rx data event occures
 *
 * @param [in] event_rx_data
 * The Rx data event. Only valid inside the handler.
 * The implementation can assume this value is not NULL.
 * @param [in] event_user_data
 * user data that was provided during register.
 */
typedef void (*doca_rmax_in_stream_event_rx_data_handler_cb_t)(struct doca_rmax_in_stream_event_rx_data *event_rx_data,
							       union doca_data event_user_data);

/**
 * @brief This method can be used to register to Rx data event
 *
 * Can only be called before calling doca_ctx_start().
 * Once registration is done, then everytime the event occurs during doca_pe_progress(), the handler will be called.
 *
 * @param [in] stream
 * The in stream instance.
 * @param [in] user_data
 * user defined data that will be provided to the handler. Can be used to store the program state.
 * @param [in] success_handler
 * Method that is invoked once a successful event is triggered.
 * @param [in] error_handler
 * Method that is invoked once an error event is triggered.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - in case one of the arguments is NULL.
 * - DOCA_ERROR_BAD_STATE - in stream context state is not idle.
 * - DOCA_ERROR_IN_USE - already registered to event.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_in_stream_event_rx_data_register(struct doca_rmax_in_stream *stream,
							union doca_data user_data,
							doca_rmax_in_stream_event_rx_data_handler_cb_t success_handler,
							doca_rmax_in_stream_event_rx_data_handler_cb_t error_handler);

/**
 * @brief This method gets the completion data from the event
 *
 * @note can only be used inside the event handler.
 *
 * @param [in] rx_event
 * The event to get from. must not be NULL.
 *
 * @return
 * The completion data. Only valid inside the event handler.
 */
DOCA_EXPERIMENTAL
struct doca_rmax_in_stream_completion *
doca_rmax_in_stream_event_rx_data_get_completion(const struct doca_rmax_in_stream_event_rx_data *rx_event);

/**
 * @brief This method gets the error data from the event
 *
 * @note can only be used inside the event handler.
 *
 * @param [in] rx_event
 * The event to get from. must not be NULL.
 *
 * @return
 * The error data. Only valid inside the event handler.
 */
DOCA_EXPERIMENTAL
struct doca_rmax_stream_error *
doca_rmax_in_stream_event_rx_data_get_error(const struct doca_rmax_in_stream_event_rx_data *rx_event);

/*********************************************************************************************************************
 * DOCA RMAX - FLOW
 *********************************************************************************************************************/

/**
 * @brief Create a steering flow for input stream to filter incoming data flow
 * by match criteria.
 *
 * @param [out] flow
 * The flow created for input stream. Non NULL upon success, NULL otherwise.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - unable to allocate memory.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_create(struct doca_rmax_flow **flow);

/**
 * @brief Destroy a steering flow.
 *
 * @param [in] flow
 * Flow to destroy.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_destroy(struct doca_rmax_flow *flow);

/**
 * @brief Attach a flow to a stream
 *
 * @param [in] stream
 * The context for attaching a flow
 * @param [in] flow
 * Flow to operate on
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_INITIALIZATION - Rivermax is not initialized.
 * - DOCA_ERROR_SHUTDOWN - library shutdown in a process.
 * - DOCA_ERROR_UNEXPECTED - unexpected issue.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_attach(const struct doca_rmax_flow *flow,
		const struct doca_rmax_in_stream *stream);

/**
 * @brief Detach a flow from a stream
 *
 * @param [in] stream
 * The context for detaching a flow
 * @param [in] flow
 * Flow to operate on
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_INITIALIZATION - Rivermax is not initialized.
 * - DOCA_ERROR_SHUTDOWN - library shutdown in a process.
 * - DOCA_ERROR_UNEXPECTED - unexpected issue.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_detach(const struct doca_rmax_flow *flow,
		const struct doca_rmax_in_stream *stream);

/* Flow helper functions */

/**
 * @brief Set the source IP filter for the flow
 *
 * @param [in] flow
 * Flow to operate on
 *
 * @param [in] ip
 * Source IPv4 address
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_set_src_ip(struct doca_rmax_flow *flow, const struct in_addr *ip);

/**
 * @brief Set the destination IP filter for the flow
 *
 * @param [in] flow
 * Flow to operate on
 *
 * @param [in] ip
 * Destination IPv4 address
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_set_dst_ip(struct doca_rmax_flow *flow, const struct in_addr *ip);

/**
 * @brief Set the source port filter for the flow
 *
 * @param [in] flow
 * Flow to operate on
 *
 * @param [in] port
 * Source port number. If zero then any source port is accepted.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_set_src_port(struct doca_rmax_flow *flow, uint16_t port);

/**
 * @brief Set the destination port filter for the flow
 *
 * @param [in] flow
 * Flow to operate on
 *
 * @param [in] port
 * Destination port number, non-zero
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_set_dst_port(struct doca_rmax_flow *flow, uint16_t port);

/**
 * @brief Set the tag for the flow
 *
 * @param [in] flow
 * Flow to operate on
 *
 * @param [in] tag
 * Non-zero tag
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_rmax_flow_set_tag(struct doca_rmax_flow *flow, uint32_t tag);

/* End of flow helper functions */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_RMAX_H_ */

/** @} */
