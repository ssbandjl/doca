/*===--------- dpaintrin.h - Header file for all DPA intrinsics -----------===//
 *
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#ifndef __DPAINTRIN_H
#define __DPAINTRIN_H

/*
 * Users need to define following macro before including this header file to
 * use a specific version of DPA intrinsics
 */
#ifndef DPA_INTRIN_VERSION_USED
#define DPA_INTRIN_VERSION_USED (DPA_INTRIN_VERSION(1, 3))
#endif

#if (DPA_INTRIN_VERSION_USED == (DPA_INTRIN_VERSION(1, 3)))

#if defined(__riscv_xfenceheap)

/// DPA 'Heap' memory space
#define __DPA_HEAP __MSPACE_HEAP
/// DPA 'Memory' memory space
#define __DPA_MEMORY __MSPACE_MEMORY
/// DPA 'MMIO' memory space
#define __DPA_MMIO __MSPACE_MMIO
/// DPA 'System' memory space
#define __DPA_SYSTEM __MSPACE_SYSTEM

/// Read memory operation
#define __DPA_R __MOP_R
/// Write memory operation
#define __DPA_W __MOP_W
/// Read and Write memory operation
#define __DPA_RW __MOP_RW

/// Ensures that all operations (PRED_OP) performed by the calling thread,
/// before the call to __dpa_thread_fence(), are performed and made visible to
/// all threads in the DPA, host, NIC engines, and peer devices as occurring
/// before all operations (SUCC_OP) to the memory space after the call to
/// __dpa_thread_fence()
/// \param MEMORY_SPACE The DPA memory space to apply fence operation. valid
/// memory spaces are __DPA_HEAP, __DPA_MEMORY, __DPA_MMIO, __DPA_SYSTEM.
/// \param PRED_OP Predecessor operation. Valid operations are  __DPA_R,
/// __DPA_W, __DPA_RW
/// \param SUCC_OP Successor operation. Valid operations are  __DPA_R,
/// __DPA_W, __DPA_RW
#define __dpa_thread_fence(MEMORY_SPACE, PRED_OP, SUCC_OP)                     \
  __dpa_thread_fence_internal_1_3(MEMORY_SPACE, PRED_OP, SUCC_OP);

/// Equivalent to calling __dpa_thread_fence(__DPA_MEMORY, OP1, OP2)
#define __dpa_thread_memory_fence(OP1, OP2)                                    \
  __dpa_thread_fence(__DPA_MEMORY, OP1, OP2)

/// Equivalent to calling __dpa_thread_fence(__DPA_MMIO, OP1, OP2)
#define __dpa_thread_outbox_fence(OP1, OP2)                                    \
  __dpa_thread_fence(__DPA_MMIO, OP1, OP2)

/// Equivalent to calling __dpa_thread_fence(__DPA_MMIO, OP1, OP2)
#define __dpa_thread_window_fence(OP1, OP2)                                    \
  __dpa_thread_fence(__DPA_MMIO, OP1, OP2)

/// Equivalent to calling __dpa_thread_fence(__DPA_SYSTEM, __DPA_RW, __DPA_RW)
#define __dpa_thread_system_fence()                                            \
  __dpa_thread_fence(__DPA_SYSTEM, __DPA_RW, __DPA_RW)

/// Ensures that contents in the window memory space of the thread before the
/// call to __dpa_thread_window_read_inv() are invalidated before read
/// operations made by the calling thread after the call to
/// __dpa_thread_window_read_inv().
#define __dpa_thread_window_read_inv()                                         \
  __dpa_thread_fence(__DPA_MMIO, __DPA_R, __DPA_R)

/// Ensures that contents in the window memory space of the thread before the
/// call to __dpa_thread_window_writeback() are performed and made visible to
/// all threads in the DPA, host, NIC engines, and peer devices as occurring
/// before any write operations after the call to
/// __dpa_thread_window_writeback().
#define __dpa_thread_window_writeback()                                        \
  __dpa_thread_fence(__DPA_MMIO, __DPA_W, __DPA_W)

/// Ensures that the contents in the Memory address space of the thread before
/// the call to __dpa_thread_writeback_memory() are performed and made visible
/// to all threads in the DPA, host, NIC engines, and peer devices as occurring
/// before any write operations after the call to
/// __dpa_thread_writeback_memory().
#define __dpa_thread_memory_writeback()                                        \
  __dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W)

#endif // __riscv_xfenceheap

#if defined(__riscv_xrpfxp)
/// Evaluate fixed point Q16.16 reciprocal (1/x) of N.
/// \param N int
#define __dpa_fxp_rcp(N) __dpa_fxp_rcp_internal_1_3(N)
/// Evaluate fixed point Q16.16 power of 2 of N.
/// \param N int
#define __dpa_fxp_pow2(N) __dpa_fxp_pow2_internal_1_3(N)
/// Evaluate fixed point Q16.16 base 2 logarithm of N.
/// \param N unsigned int
#define __dpa_fxp_log2(N) __dpa_fxp_log2_internal_1_3(N)
#endif // __riscv_xrpfxp

#if defined(__riscv_xnvcc)
#define __dpa_data_ignore(ADDR) __dpa_data_ignore_internal_1_3(ADDR)
#endif // __riscv_xnvcc

/// Returns a counter containing the number of cycles from an arbitrary start
/// point in the past on the execution unit the thread is currently scheduled
/// on. Note that the value returned by this function in the thread is
/// meaningful only for the duration of when the thread remains associated with
/// this execution unit.
#define __dpa_thread_cycles() __dpa_thread_cycles_internal_1_3()
/// Returns a counter containing the number of instructions retired from an
/// arbitrary start point in the past by the execution unit the thread is
/// currently scheduled on. Note that the value returned by this function in the
/// software thread is meaningful only for the duration of when the thread
/// remains associated with this execution unit.
#define __dpa_thread_inst_ret() __dpa_thread_inst_ret_internal_1_3()
/// Returns the number of timer ticks from an arbitrary start point in the past
/// on the execution unit the thread is currently scheduled on. Note that the
/// value returned by this function in the thread is meaningful only for the
/// duration of when the thread remains associated with this execution unit.
#define __dpa_thread_time() __dpa_thread_time_internal_1_3()

#else

#error Bad value for DPA_INTRIN_VERSION_USED

#endif // DPA_INTRIN_VERSION_USED

#endif // __DPAINTRIN_H
