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

#ifndef NVMX_SRC_LOCK_FREE_STACK_H_
#define NVMX_SRC_LOCK_FREE_STACK_H_

#include <stddef.h>
#include <stdint.h>
#include "compiler.h"

#define LFS_MAX_TEST_POINTS 4

#ifndef TEST_POINT_CALLBACK
#define TEST_POINT_CALLBACK(n, z)
#endif

typedef struct lfs_entry {
    struct lfs_entry *next;
} lfs_entry_t;

typedef struct lfs_head {
    lfs_entry_t *first;             // Points to actual stack head
    size_t counter;                 // Common approach to address ABA problem
} lfs_head_t __attribute__ ((aligned (16)));

typedef union lfs_stack {

    lfs_head_t head;
    lfs_entry_t *first;             // _Atomic doesn't allow access to its
                                    // members without atomic read of whole
                                    // structure. So we use union to get fast
                                    // non-atomic access to stack head
} lfs_stack_t;

#ifdef __aarch64__

// Lock-free push to stack
static inline void lfs_push_entry(lfs_stack_t *stack, lfs_entry_t *entry)
{
    int ret;

    do {
        struct lfs_head old_head;

        asm volatile(
            "ldaxp" " %0, %1, %2"
            : "=&r" (((uint64_t *)&old_head)[0]),
              "=&r" (((uint64_t *)&old_head)[1])
            : "Q" (*(&stack->head))
            : "memory");

        TEST_POINT_CALLBACK(0, stack);

        struct lfs_head new_head = {
            .first = entry,
            .counter = old_head.counter + 1
        };

        TEST_POINT_CALLBACK(1, stack);

        entry->next = old_head.first;

        TEST_POINT_CALLBACK(2, stack);

        asm volatile(
            "stlxp" " %w0, %1, %2, %3"
            : "=&r" (ret)
            : "r" (((uint64_t *)&new_head)[0]),
              "r" (((uint64_t *)&new_head)[1]),
              "Q" (*(&stack->head))
            : "memory");

        TEST_POINT_CALLBACK(3, stack);
    } while (unlikely(ret));
}

// Lock-free pop from stack
static inline lfs_entry_t *lfs_pop_entry(lfs_stack_t *stack)
{
    for(;;) {
        int ret;
        struct lfs_head old_head;

        asm volatile(
            "ldaxp" " %0, %1, %2"
            : "=&r" (((uint64_t *)&old_head)[0]),
              "=&r" (((uint64_t *)&old_head)[1])
            : "Q" (*(&stack->head))
            : "memory");

        TEST_POINT_CALLBACK(0, stack);

        if (likely(old_head.first)) {

            struct lfs_head new_head = {
                    .first = old_head.first->next,
                    .counter = old_head.counter + 1
            };

            TEST_POINT_CALLBACK(1, stack);

            asm volatile(
                "stlxp" " %w0, %1, %2, %3"
                : "=&r" (ret)
                : "r" (((uint64_t *)&new_head)[0]),
                  "r" (((uint64_t *)&new_head)[1]),
                  "Q" (*(&stack->head))
                : "memory");

            TEST_POINT_CALLBACK(2, stack);

            if (ret)
                continue;

            return old_head.first;
        }

        asm volatile("clrex");
        return NULL;
    }
}
#else

// Lock-free push to stack
static inline void lfs_push_entry(lfs_stack_t *stack, lfs_entry_t *entry)
{
    uint8_t ret;
    struct lfs_head old_head = stack->head;

    TEST_POINT_CALLBACK(0, stack);

    do {
        struct lfs_head new_head = {
            .first = entry,
            .counter = old_head.counter + 1
        };

        TEST_POINT_CALLBACK(1, stack);

        entry->next = old_head.first;

        TEST_POINT_CALLBACK(2, stack);

        asm volatile(
            "lock ; "
            "cmpxchg16b %[dst];"
            " sete %[ret]"
            : [dst] "=m" (*(&stack->head)),
              "=a" (((uint64_t *)&old_head)[0]),
              "=d" (((uint64_t *)&old_head)[1]),
              [ret] "=r" (ret)
            : "b" (((uint64_t *)&new_head)[0]),
              "c" (((uint64_t *)&new_head)[1]),
              "a" (((uint64_t *)&old_head)[0]),
              "d" (((uint64_t *)&old_head)[1]),
              "m" (*(&stack->head))
            : "memory");

        TEST_POINT_CALLBACK(3, stack);

    } while (unlikely(!ret));
}

// Lock-free pop from stack
static inline lfs_entry_t *lfs_pop_entry(lfs_stack_t *stack)
{
    struct lfs_head old_head = stack->head;

    TEST_POINT_CALLBACK(0, stack);

    while (likely(old_head.first)) {
        uint8_t ret;
        struct lfs_head new_head = {
                .first = old_head.first->next,
                .counter = old_head.counter + 1
        };

        TEST_POINT_CALLBACK(1, stack);

        asm volatile(
            "lock ; "
            "cmpxchg16b %[dst];"
            " sete %[ret]"
            : [dst] "=m" (*(&stack->head)),
              "=a" (((uint64_t *)&old_head)[0]),
              "=d" (((uint64_t *)&old_head)[1]),
            [ret] "=r" (ret)
            : "b" (((uint64_t *)&new_head)[0]),
              "c" (((uint64_t *)&new_head)[1]),
              "a" (((uint64_t *)&old_head)[0]),
              "d" (((uint64_t *)&old_head)[1]),
              "m" (*(&stack->head))
            : "memory");

        TEST_POINT_CALLBACK(2, stack);

        if (likely(ret))
            break;
    }

    return old_head.first;
}


#endif

#endif
