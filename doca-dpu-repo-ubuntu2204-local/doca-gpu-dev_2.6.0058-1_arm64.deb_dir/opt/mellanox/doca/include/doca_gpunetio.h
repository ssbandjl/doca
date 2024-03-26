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
 * @file doca_gpunetio.h
 * @page DOCA GPUNETIO
 * @defgroup DOCAGPUNETIO DOCA GPUNETIO
 * DOCA GPUNETIO library.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_H_
#define DOCA_GPUNETIO_H_

#include <inttypes.h>
#include <stddef.h>

#include <doca_compat.h>
#include <doca_ctx.h>
#include <doca_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Macro to temporarily cast a variable to volatile.
 */
#define DOCA_GPUNETIO_VOLATILE(x) (*(volatile typeof(x) *)&(x))

/**
 * Semaphore list of possible statuses.
 */
enum doca_gpu_semaphore_status {
	/* Semaphore is free and can be (re)used. */
	DOCA_GPU_SEMAPHORE_STATUS_FREE	= 0,
	/* Semaphore is ready with new packets. */
	DOCA_GPU_SEMAPHORE_STATUS_READY	= 1,
	/* Sempahore packets' have been processed. */
	DOCA_GPU_SEMAPHORE_STATUS_DONE	= 2,
	/* Still processing info, don't overwrite. */
	DOCA_GPU_SEMAPHORE_STATUS_HOLD	= 3,
	/* Some error occurred to the processing related to that semaphore. */
	DOCA_GPU_SEMAPHORE_STATUS_ERROR	= 4,
	/* CUDA kernel should just exit from execution. */
	DOCA_GPU_SEMAPHORE_STATUS_EXIT	= 5,
};

/*********************************************************************************************************************
 * DOCA GPUNetIO init
 *********************************************************************************************************************/

/**
 * Opaque structure representing a DOCA GPUNetIO handler.
 */
struct doca_gpu;

/**
 * Opaque structure representing a DOCA GPUNetIO semaphore.
 */
struct doca_gpu_semaphore;
struct doca_gpu_semaphore_gpu;

/**
 * @brief Create a DOCA GPUNETIO handler.
 *
 * @param [in] gpu_bus_id
 * GPU PCIe address.
 * @param [out] gpu_dev
 * Pointer to the newly created gpu device handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - gpu_dev argument is a NULL pointer.
 * - DOCA_ERROR_NOT_FOUND - GPU not found at the input PCIe address
 * - DOCA_ERROR_NO_MEMORY - failed to alloc doca_gpu.
 *
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_create(const char *gpu_bus_id, struct doca_gpu **gpu_dev);

/**
 * @brief Destroy a DOCA GPUNETIO handler.
 *
 * @param [in] gpu_dev
 * Pointer to handler to be destroyed.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_destroy(struct doca_gpu *gpu_dev);

/*********************************************************************************************************************
 * DOCA GPUNetIO memory
 *********************************************************************************************************************/

/**
 * Allocate a GPU accessible memory buffer. Assumes DPDK has been already attached with doca_gpu_to_dpdk().
 * According to the memory type specified, the buffer can be allocated in:
 * - DOCA_GPU_MEM_TYPE_GPU memptr_gpu is not NULL while memptr_cpu is NULL.
 * - DOCA_GPU_MEM_TYPE_GPU_CPU both memptr_gpu and memptr_cpu are not NULL.
 * - DOCA_GPU_MEM_TYPE_CPU_GPU both memptr_gpu and memptr_cpu are not NULL.
 *
 * @param [in] gpu_dev
 * DOCA GPUNetIO handler.
 * @param [in] size
 * Buffer size in bytes.
 * @param [in] alignment
 * Buffer memory alignment.
 * If 0, the return is a pointer that is suitably aligned
 * for any kind of variable (in the same manner as malloc()).
 * Otherwise, the return is a pointer that is a multiple of *align*.
 * Alignment value must be a power of two.
 * @param [in] mtype
 * Type of memory buffer. See enum doca_gpu_memtype for reference.
 * @param [out] memptr_gpu
 * GPU memory pointer. Must be used with CUDA API and within CUDA kernels.
 * @param [out] memptr_cpu
 * CPU memory pointer. Must be used for CPU direct access to the memory.
 *
 * @return
 * Non NULL memptr_gpu pointer on success, NULL otherwise.
 * Non NULL memptr_cpu pointer on success in case of DOCA_GPU_MEM_TYPE_CPU_GPU and DOCA_GPU_MEM_TYPE_GPU_CPU, NULL otherwise.
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NO_MEMORY - if an error occurred dealing with GPU memory.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_mem_alloc(struct doca_gpu *gpu_dev, size_t size, size_t alignment, enum doca_gpu_mem_type mtype,
				void **memptr_gpu, void **memptr_cpu);

/**
 * Free a GPU memory buffer.
 * Only memory allocated with doca_gpu_mem_alloc() can be freed with this function.
 *
 * @param [in] gpu
 * DOCA GPUNetIO handler.
 * @param [in] memptr_gpu
 * GPU memory pointer to be freed.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_mem_free(struct doca_gpu *gpu, void *memptr_gpu);

/**
 * Return a DMABuf file descriptor from a GPU memory address if the GPU device and CUDA installation supports DMABuf.
 *
 * @param [in] gpu_dev
 * DOCA GPUNetIO handler.
 * @param [in] memptr_gpu
 * GPU memory pointer to be freed.
 * @param [in] size
 * Size in bytes to map.
 * @param [out] dmabuf_fd
 * DMABuf file descriptor
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NOT_SUPPORTED - DMABuf not supported
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_dmabuf_fd(struct doca_gpu *gpu_dev, void *memptr_gpu, size_t size, int *dmabuf_fd);

/*********************************************************************************************************************
 * DOCA GPUNetIO semaphore
 *********************************************************************************************************************/

/**
 * Create a DOCA GPUNetIO semaphore.
 *
 * @param [in] gpu_dev
 * DOCA GPUNetIO handler.
 * @param [in] semaphore
 * Pointer to the newly created semaphore handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_create(struct doca_gpu *gpu_dev, struct doca_gpu_semaphore **semaphore);

/**
 * Stop a DOCA GPUNetIO semaphore.
 *
 * @param [in] semaphore
 * DOCA GPUNetIO semaphore handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_destroy(struct doca_gpu_semaphore *semaphore);

/**
 * Type of memory to be used to allocate DOCA GPUNetIO semaphore items.
 * It determines semaphore visibility: GPU only or GPU and CPU.
 *
 * @param [in] semaphore
 * DOCA GPUNetIO semaphore handler.
 * @param [in] mtype
 * DOCA GPUNetIO semaphore items type of memory.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_set_memory_type(struct doca_gpu_semaphore *semaphore, enum doca_gpu_mem_type mtype);

/**
 * Allocate DOCA GPUNetIO semaphore number of items.
 *
 * @param [in] semaphore
 * DOCA GPUNetIO semaphore handler.
 * @param [in] num_items
 * DOCA GPUNetIO semaphore number of items.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_set_items_num(struct doca_gpu_semaphore *semaphore, uint32_t num_items);

/**
 * Attach to each item in the DOCA GPUNetIO semaphore a custom user-defined structure defined
 * through a number of bytes reserved for each item.
 * Specify also which memory to use for the custom info items.
 *
 * @param [in] semaphore
 * DOCA GPUNetIO semaphore handler.
 * @param [in] nbytes
 * DOCA GPUNetIO semaphore number of items.
 * @param [in] mtype
 * DOCA GPUNetIO semaphore memory type.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_set_custom_info(struct doca_gpu_semaphore *semaphore, uint32_t nbytes, enum doca_gpu_mem_type mtype);

/**
 * Start a DOCA GPUNetIO semaphore.
 * Attributes can't be set after this point.
 *
 * @param [in] semaphore
 * DOCA GPUNetIO semaphore handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_start(struct doca_gpu_semaphore *semaphore);

/**
 * Stop a DOCA GPUNetIO semaphore.
 *
 * @param [in] semaphore
 * DOCA GPUNetIO semaphore handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_stop(struct doca_gpu_semaphore *semaphore);

/**
 * Get a DOCA GPUNetIO semaphore handle for GPU.
 * This can be used within a CUDA kernel.
 *
 * @param [in] semaphore_cpu
 * DOCA GPUNetIO semaphore CPU handler.
 * @param [out] semaphore_gpu
 * DOCA GPUNetIO semaphore GPU handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_get_gpu_handle(struct doca_gpu_semaphore *semaphore_cpu, struct doca_gpu_semaphore_gpu **semaphore_gpu);

/**
 * Set DOCA GPUNetIO semaphore status from CPU.
 * This can be done only if item memory was set to DOCA_GPU_MEM_TYPE_GPU_CPU or DOCA_GPU_MEM_TYPE_CPU_GPU.
 *
 * @param [in] semaphore_cpu
 * DOCA GPUNetIO semaphore CPU handler.
 * @param [in] idx
 * DOCA GPUNetIO semaphore item index.
 * @param [in] status
 * DOCA GPUNetIO semaphore status.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_set_status(struct doca_gpu_semaphore *semaphore_cpu, uint32_t idx, enum doca_gpu_semaphore_status status);

/**
 * Get DOCA GPUNetIO semaphore status from CPU.
 * This can be done only if item memory was set to DOCA_GPU_MEM_TYPE_GPU_CPU or DOCA_GPU_MEM_TYPE_CPU_GPU.
 *
 * @param [in] semaphore_cpu
 * DOCA GPUNetIO semaphore CPU handler.
 * @param [in] idx
 * DOCA GPUNetIO semaphore item index.
 * @param [out] status
 * DOCA GPUNetIO semaphore status.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_get_status(struct doca_gpu_semaphore *semaphore_cpu, uint32_t idx, enum doca_gpu_semaphore_status *status);

/**
 * Get pointer to DOCA GPUNetIO semaphore item associated custom info.
 * This can be done only if custom info memory was set to DOCA_GPU_MEM_TYPE_GPU_CPU or DOCA_GPU_MEM_TYPE_CPU_GPU.
 *
 * @param [in] semaphore_cpu
 * DOCA GPUNetIO semaphore CPU handler.
 * @param [in] idx
 * DOCA GPUNetIO semaphore item index.
 * @param [out] custom_info
 * DOCA GPUNetIO semaphore custom info pointer.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
DOCA_EXPERIMENTAL
doca_error_t doca_gpu_semaphore_get_custom_info_addr(struct doca_gpu_semaphore *semaphore_cpu, uint32_t idx, void **custom_info);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_GPUNETIO_H_ */

/** @} */