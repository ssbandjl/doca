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

#ifndef MEMP_BIT_ARRAY_H_
#define MEMP_BIT_ARRAY_H_

#include <stdbool.h>
#include <limits.h>

#define memp_bit_array unsigned long long
#define MEM_BITS_PER_ELEMENT 64ULL
#define MEM_BITS_ARRAY_SIZE(n) ((n / MEM_BITS_PER_ELEMENT) ? : 1ULL)
#define MEM_BITS_INVALID_INDEX ULONG_MAX

static inline void clr_bit(memp_bit_array *bits, size_t index)
{
    const size_t element_no = index / MEM_BITS_PER_ELEMENT;
    const size_t bit_no = index % MEM_BITS_PER_ELEMENT;
    const memp_bit_array bit_mask = ~(1ULL << bit_no);
    bits[element_no] &= bit_mask;
}

static inline void set_bit(memp_bit_array *bits, size_t index)
{
    const size_t element_no = index / MEM_BITS_PER_ELEMENT;
    const size_t bit_no = index % MEM_BITS_PER_ELEMENT;
    const memp_bit_array bit_mask = 1ULL << bit_no;
    bits[element_no] |= bit_mask;
}

static inline void set_bits(memp_bit_array *bits, size_t len,
                                        size_t bits_to_set)
{
    size_t element_no;

    for (element_no = 0; element_no < len; element_no++) {
        if (bits_to_set >= MEM_BITS_PER_ELEMENT) {
            bits[element_no] = ~0ULL;
            bits_to_set -= MEM_BITS_PER_ELEMENT;
        }
        else
        if (bits_to_set) {
            bits[element_no] = ~(~0ULL << bits_to_set);
            bits_to_set = 0;
        }
        else
            bits[element_no] = 0;
    }
}

static inline size_t find_bit(memp_bit_array *bits, size_t len)
{
    size_t element_no;

    for (element_no = 0; element_no < len; element_no++) {
        const memp_bit_array element = bits[element_no];
        if (element)
            return element_no * MEM_BITS_PER_ELEMENT +
            		__builtin_ctzll(element);
    }

    return MEM_BITS_INVALID_INDEX;
}


#endif /* MEM_BITS_H_ */
