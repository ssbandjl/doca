/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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
 * @file doca_dpa_dev_buf.h
 * @page doca dpa buf
 * @defgroup DPA_BUF DOCA DPA buf
 * @ingroup DPA_DEVICE
 * DOCA DPA buffer
 * @{
 */

#ifndef DOCA_DPA_DEV_BUF_H_
#define DOCA_DPA_DEV_BUF_H_

#include <doca_dpa_dev.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DPA buffer handle type definition
 */
__dpa_global__ typedef uint64_t doca_dpa_dev_buf_t;

/**
 * @brief DPA buffer array handle type definition
 */
__dpa_global__ typedef uint64_t doca_dpa_dev_buf_arr_t;

/**
 * @brief doca dpa device buf declaration
 *
 * User of this struct should relate to it as an opaque and not access its fields, but rather use relevant API for it
 */
struct doca_dpa_dev_buf {
	uintptr_t addr; 		/**< address held by doca dpa device buf */
	uint64_t len;			/**< length of doca dpa device buf */
	unsigned char reserved[16];	/**< reserved field */
} __attribute__((__packed__));		/**< packed */

/**
 * @brief doca dpa device buf array declaration
 *
 * User of this struct should relate to it as an opaque and not access its fields, but rather use relevant API for it
 */
struct doca_dpa_dev_buf_arr {
	struct doca_dpa_dev_buf *bufs; 		/**< doca dpa device bufs */
	unsigned char reserved[20];		/**< reserved field */
} __attribute__((__packed__, aligned(64)));	/**< packed and aligned */

/**
 * @brief Get DPA buffer handle from a DPA buffer array handle
 *
 * @param[in] buf_arr - DOCA DPA device buf array handle
 * @param[in] buf_idx - DOCA DPA buffer index
 *
 * @return
 * Handle to DPA buffer
 */
DOCA_EXPERIMENTAL
__forceinline doca_dpa_dev_buf_t doca_dpa_dev_buf_array_get_buf(doca_dpa_dev_buf_arr_t buf_arr, const uint64_t buf_idx)
{
	struct doca_dpa_dev_buf_arr *dev_buf_arr = (struct doca_dpa_dev_buf_arr *)(buf_arr);
	doca_dpa_dev_buf_t dev_buf = (doca_dpa_dev_buf_t)&(dev_buf_arr->bufs[buf_idx]);

	return dev_buf;
}

/**
 * @brief Get address from a DPA buffer handle
 *
 * @param[in] buf - DOCA DPA device buf handle
 *
 * @return
 * Address held by DPA buffer
 */
DOCA_EXPERIMENTAL
__forceinline uintptr_t doca_dpa_dev_buf_get_addr(doca_dpa_dev_buf_t buf)
{
	struct doca_dpa_dev_buf *dev_buf = (struct doca_dpa_dev_buf *)(buf);

	return dev_buf->addr;
}

/**
 * @brief Get length from a DPA buffer handle
 *
 * @param[in] buf - DOCA DPA device buf handle
 *
 * @return
 * Length of DPA buffer
 */
DOCA_EXPERIMENTAL
__forceinline uint64_t doca_dpa_dev_buf_get_len(doca_dpa_dev_buf_t buf)
{
	struct doca_dpa_dev_buf *dev_buf = (struct doca_dpa_dev_buf *)(buf);

	return dev_buf->len;
}

/**
 * \brief Obtain a pointer to externally allocated memory
 *
 * This function allows the DPA process to obtain a pointer to external memory that is held by a DPA handle.
 * The obtained pointer can be used to load/store data directly from the DPA kernel.
 * The memory being accessed through the returned device pointer is subject to 64B alignment restriction
 *
 * @param[in] buf - DOCA DPA device buf handle
 * @param[in] buf_offset - offset of external address being accessed
 *
 * @return
 * Device address pointing to external address
 */
DOCA_EXPERIMENTAL
doca_dpa_dev_uintptr_t doca_dpa_dev_buf_get_external_ptr(doca_dpa_dev_buf_t buf, uint64_t buf_offset);

/**
 * \brief Initiate a copy data locally from Host
 *
 * This function copies data between two memory regions. The destination buffer, specified by `dest_addr` and `length`
 * will contain the copied data after the memory copy is complete. This is a non-blocking routine
 *
 * @param[in] dst_mem - destination memory buffer to copy into
 * @param[in] dst_offset - offset from start address of destination buffer
 * @param[in] src_mem - source memory buffer to read from
 * @param[in] src_offset - offset from start address of source buffer
 * @param[in] length - size of buffer
 *
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_memcpy_nb(doca_dpa_dev_buf_t dst_mem,
			    uint64_t dst_offset,
			    doca_dpa_dev_buf_t src_mem,
			    uint64_t src_offset,
			    size_t length);

/**
 * \brief Initiate a transpose locally from Host
 *
 * This function transposes a 2D array. The destination buffer, specified by `dest_addr` and `length` will contain the
 * copied data after the operation is complete. This is a non-blocking routine
 *
 * @param[in] dst_mem -destination memory buffer to transpose into
 * @param[in] dst_offset - offset from start address of destination buffer
 * @param[in] src_mem - source memory buffer to transpose from
 * @param[in] src_offset - offset from start address of source buffer
 * @param[in] length - size of buffer
 * @param[in] element_size - size of datatype of one element
 * @param[in] num_columns - number of columns in 2D array
 * @param[in] num_rows - number of rows in 2D array
 *
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_memcpy_transpose2D_nb(doca_dpa_dev_buf_t dst_mem,
					uint64_t dst_offset,
					doca_dpa_dev_buf_t src_mem,
					uint64_t src_offset,
					size_t length,
					size_t element_size,
					size_t num_columns,
					size_t num_rows);

/**
 * \brief Wait for all memory copy operations issued previously to complete
 *
 * This function returns when memory copy operations issued on this thread have been completed.
 * After this call returns, the buffers they referenced by the copy operations can be reused. This call is blocking
 *
 */
DOCA_EXPERIMENTAL
void doca_dpa_dev_memcpy_synchronize(void);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* DOCA_DPA_DEV_BUF_H_ */
