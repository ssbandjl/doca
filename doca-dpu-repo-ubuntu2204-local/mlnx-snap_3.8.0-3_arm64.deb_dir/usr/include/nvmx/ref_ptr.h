/*
 *   Copyright © 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * The referenced pointer (ref_ptr_t) container combines a pointer and a reference
 * counter in a single void* type.
 * The actual pointer should be 256 bytes aligned so that lower 8 bits are zero.
 * In this case the container uses the lower 8 bits for a reference counter.
 * The reason to do this is that we have a light-weight inline 8-bytes compare-and-swap
 * operation. Instead, the 16-bytes CAS makes a library function call.
 *
 * The usage model:
 *
 * 1. Admin thread calls ref_ptr_init
 * 2. I/O threads are calling ref_ptr_get and ref_ptr_put when they need to work
 *    with the underlying object.
 * 3. Then admin thread needs to update the object, it calls ref_ptr_replace
 *    The idea here is that admin thread doesn't modify existing object.
 *    Instead, it makes a clone, performs the changes and safely replaces the pointer
 *    with the new one, when no I/O threads are working with it.
 */

#ifndef NVMX_SRC_REF_PTR_H_
#define NVMX_SRC_REF_PTR_H_

#include "nvme_emu_log.h"

typedef union {
    void *ptr;
    uint8_t ref_count;
} __attribute__((aligned(8))) __attribute__((packed)) ref_ptr_t;

static inline void ref_ptr_init(ref_ptr_t *ref_ptr, void *ptr)
{
    ref_ptr->ptr = ptr;
    nvmx_assertv_always(ref_ptr->ref_count == 0, "Improperly aligned pointer");

    // Resolved at compile time, doesn't slow down production if always lock free
    if (!__atomic_always_lock_free(sizeof(*ref_ptr), ref_ptr))
        nvmx_warn("Referenced pointer is not always lock-free");

    // Resolved at compile time, doesn't slow down production if ref_ptr_t struct is ok
    nvmx_assertv_always(sizeof(ref_ptr_t) == sizeof(void *), "Invalid ref_ptr_t size");
    nvmx_assertv_always(offsetof(ref_ptr_t, ptr) == offsetof(ref_ptr_t, ref_count), "Invalid ref_count offset");
}

static inline void *ref_ptr(ref_ptr_t *ref_ptr)
{
    ref_ptr_t ptr = *ref_ptr;

    ptr.ref_count = 0;

    return ptr.ptr;
}

static inline void *ref_ptr_get(ref_ptr_t *ref_ptr)
{
    ref_ptr_t ptr;

    ptr.ptr = __atomic_add_fetch(&ref_ptr->ptr, 1U, __ATOMIC_SEQ_CST);

    nvmx_assertv_always(ptr.ref_count != 0, "Too many references");

    ptr.ref_count = 0;

    return ptr.ptr;
}

static inline void ref_ptr_put(ref_ptr_t *ref_ptr)
{
    (void)__atomic_fetch_sub(&ref_ptr->ptr, 1U, __ATOMIC_SEQ_CST);
}

static inline void ref_ptr_replace(ref_ptr_t *ref_ptr, void *new_ptr)
{
    ref_ptr_t expected = *ref_ptr;
    ref_ptr_t desired = {
            .ptr = new_ptr
    };

    nvmx_assertv_always(desired.ref_count == 0, "Improperly aligned pointer");

    do {
        expected.ref_count = 0;
    } while (!__atomic_compare_exchange(ref_ptr, &expected, &desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
}

#endif /* NVMX_SRC_REF_PTR_H_ */
