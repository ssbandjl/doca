/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef AM_BIT_ARRAY_H_
#define AM_BIT_ARRAY_H_

// This utility class is used for storing a large amount of bits (more than a single long long variable can hold)
// It provides easy access to each bit and efficient bitwise operations such as AND / OR between two bit arrays.
// It is similar to std:bitset, but provides some functionality that is not available in std::bitset, such as
// efficient bit count and efficient find of first true/false bit location.
// This class can be used for fast detection of mutual resources among many objects, assuming these resources
// can be represented by a bitmap
//
// Note about performance: This class can be improved by using 128 bits operations via SSE2 or even AVX2
// for 256 bits operations, but it will force platform specific code, to support also ARM & PowerPC, so we avoid it
// at the moment.
//
// This class logic is kept in the header file, to enable inlines

#include "am_common.h"
#include "am_log.h"
#include "common/sharp_common.h"

// For easy porting of the code in the future to support different platforms with larger bitwise operations
typedef uint64_t word_t;   // Notice the builtin methods ffsll/ctzll/clzll, they need to be modified if changing the typedef
#define WORD_SIZE_BYTES sizeof(word_t)
#define WORD_SIZE_BITS  (WORD_SIZE_BYTES * 8)

class AMBitArray
{
   private:
    size_t m_size_in_bits_;
    size_t m_size_in_words_;
    uint32_t m_unused_bits_;   // Number of bits in the last word that we should ignore
    word_t* data = NULL;       // We prefer to maintain the memory rather than use std::vector, so we can perform faster operations

   public:
    //-----------------
    // Constructor, requires to know the desired size to use
    AMBitArray(size_t size_in_bits)
    {
        m_size_in_bits_ = size_in_bits;

        // Treat the case of number of bits not equally divided to full words
        m_size_in_words_ = ((m_size_in_bits_ - 1) / WORD_SIZE_BITS) + 1;
        m_unused_bits_ = m_size_in_bits_ % WORD_SIZE_BITS;

        // Allocate the bit array
        // For improved performance, we prefer to allocate on cache line alignment, so we use as fewer cachelines as possible.
        size_t size_to_alloc_bytes = WORD_SIZE_BYTES * m_size_in_words_;
        if (AMCommon::GetCacheLineSize() > 0) {
            // Use posix aligned memory allocation, notice that in case of an error we get RC -1 and &data might not
            // be modified
            int rc = posix_memalign((void**)&data, AMCommon::GetCacheLineSize(), size_to_alloc_bytes);
            if (rc == -1) {
                data = NULL;
            }
        } else {
            // Use regular malloc method
            data = (word_t*)malloc(size_to_alloc_bytes);
        }

        // Check for an error in either allocation method
        if (data == NULL) {
            // Nothing much we can do about it, mark size as 0, so other methods wont fail
            ERROR("Could not allocate memory for bit array, size: %lu", size_to_alloc_bytes);
            m_size_in_bits_ = 0;
            m_size_in_words_ = 0;
        }
    }

    //-----------------
    // Destructor
    ~AMBitArray()
    {
        if (data) {
            free(data);
        }
    }

    //-----------------
    // Get size in bits
    inline size_t GetSizeBits() { return m_size_in_bits_; }

    //-----------------
    // Set all bits to desired value
    // Use memset for best performance
    inline void SetAll(bool value)
    {
        int value_to_set = -1 * value;   // Either all bits are zero or all one. use -1 to set all bits to true, regardless of word size
        memset(data, value_to_set, m_size_in_words_ * WORD_SIZE_BYTES);
    }

    //-----------------
    // Set a specific bit to true
    inline void SetBit2True(size_t location)
    {
        // Make sure we are not asked to set a bit outside the allocated memory area)
        if (unlikely(location >= m_size_in_bits_)) {
            ERROR("Got a request to set a bit at location: %lu, but bit array size is: %lu bits", location, m_size_in_bits_);
            return;
        }
        size_t word_location = location / WORD_SIZE_BITS;
        uint32_t bit_location_in_word = location % WORD_SIZE_BITS;
        word_t bit_to_add = 1LL << bit_location_in_word;
        data[word_location] |= bit_to_add;
    }

    //-----------------
    // Set a specific bit to false
    inline void SetBit2False(size_t location)
    {
        // Make sure we are not asked to set a bit outside the allocated memory area)
        if (unlikely(location >= m_size_in_bits_)) {
            ERROR("Got a request to set a bit at location: %lu, but bit array size is: %lu bits", location, m_size_in_bits_);
            return;
        }
        size_t word_location = location / WORD_SIZE_BITS;
        uint32_t bit_location_in_word = location % WORD_SIZE_BITS;
        word_t bit_to_reset = ~(1LL << bit_location_in_word);   // Sets all bits to 1, beside the desired bit which will be 0
        data[word_location] &= bit_to_reset;
    }

    //-----------------
    // Get the location of the first TRUE bit
    // Notice that 0 is a valid location, returns -1 if there is no true bit
    inline int32_t GetFirstTrue()
    {
        uint32_t bit_location = 0;
        for (uint32_t word_location = 0; word_location < m_size_in_words_; word_location++) {
            // Get the first lsb bit location, notice that ffs returns an index starting at 1, a value of 0 means no true bit
            // Notice that the following builtin method should be modified in case word_t size is modified
            int first_bit_index = __builtin_ffsll(data[word_location]);
            if (first_bit_index > 0) {
                bit_location += first_bit_index - 1;   // -1 since ffs indexing starts from 1, not 0

                // Need to handle the case of last word, it could be that SetAll was used to set all bits and so the true bit is not one
                // that we care about, in that case, it means no true bit was found in the used area
                if (unlikely(bit_location >= m_size_in_bits_)) {
                    return -1;
                }
                return bit_location;
            }

            // Advance the bits location by an entire word before moving to the next word
            bit_location += WORD_SIZE_BITS;
        }

        // If we got here, it means no true bit was found
        return -1;
    }

    //-----------------
    // Get the location of the first FALSE bit
    // Notice that 0 is a valid location, returns -1 if there is no false bit
    // This method is very similar to GetFirstTrue, there is no builtin equivalent to ffs that looks for
    // 0 bits, but we can perform a "not" operation on all bits and than use ffs
    inline int32_t GetFirstFalse()
    {
        uint32_t bit_location = 0;
        for (uint32_t word_location = 0; word_location < m_size_in_words_; word_location++) {
            // Get the first lsb bit location after perform bitwise not on all bits, notice that ffs returns an index starting at 1, a value
            // of 0 means no matching bit Notice that the following builtin method should be modified in case word_t size is modified
            int first_bit_index = __builtin_ffsll(~data[word_location]);
            if (first_bit_index > 0) {
                bit_location += first_bit_index - 1;   // -1 since ffs indexing starts from 1, not 0

                // Need to handle the case of last word, it could be that SetAll was used to set all bits and so the bit is not one
                // that we care about, in that case, it means no bit was found in the used area
                if (unlikely(bit_location >= m_size_in_bits_)) {
                    return -1;
                }
                return bit_location;
            }

            // Advance the bits location by an entire word before moving to the next word
            bit_location += WORD_SIZE_BITS;
        }

        // If we got here, it means no  bit was found
        return -1;
    }

    //-----------------
    // AND operator
    inline void operator&=(const AMBitArray& rhs)
    {
        // We don't mind if rhs size is larger this this object
        // But we do mind if it is smaller, since AND operation with
        // objects that don't exist, we will treat as AND with 0, which is the same as
        // setting zeros.
        size_t cnt = 0;
        while ((cnt < rhs.m_size_in_words_) && (cnt < m_size_in_words_)) {
            data[cnt] &= rhs.data[cnt];
            cnt++;
        }

        // In case rhs is shorter than this bitarray, zero all the remaining words
        while (cnt < m_size_in_words_) {
            data[cnt] = 0;
            cnt++;
        }
    }

    //-----------------
    // OR operator
    inline void operator|=(const AMBitArray& rhs)
    {
        // We only care about the common part of both bitarrays
        size_t cnt = 0;
        while ((cnt < rhs.m_size_in_words_) && (cnt < m_size_in_words_)) {
            data[cnt] |= rhs.data[cnt];
            cnt++;
        }
    }

    //-----------------
    // Equal operator
    inline bool operator==(const AMBitArray& rhs)
    {
        // if size differ, not equal
        if (m_size_in_words_ != rhs.m_size_in_words_) {
            return false;
        }

        size_t cnt = 0;
        while (cnt < m_size_in_words_) {
            if (data[cnt] != rhs.data[cnt]) {
                return false;
            }
            cnt++;
        }

        return true;
    }
};

#endif   // AM_BIT_ARRAY_H_
