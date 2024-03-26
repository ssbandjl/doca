/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _FLEXIO_DEV_DEBUG_H_
#define _FLEXIO_DEV_DEBUG_H_

#include <stdint.h>

/**
 * @brief Print a string to SimX log.
 *
 * This function prints a given string to SimX log according to user
 * apecified length.
 *
 * @param[in] str - A pointer to string buffer.
 * @param[in] len - The length of the string.
 *
 * @return void.
 */
void print_sim_str(const char *str, int len);

/**
 * @brief Print a value in hexadecimal presentation to SimX log.
 *
 * This function prints a given unsigned 64 bit integer value to SimX log
 * in hexadecimal presentation up to a length specified by user.
 * Note that MSB part is dropped if exceeds length.
 *
 * @param[in] val - A value to be printed.
 * @param[in] len - The maximal hexadecimal string length to print.
 *
 * @return void.
 */
void print_sim_hex(uint64_t val, int len);

/**
 * @brief Print a value in decimal presentation to SimX log.
 *
 * This function prints a signed given integer value to SimX log
 * in decimal presentation up to a length specified by user.
 * Note that MSB part is dropped if exceeds length.
 *
 * @param[in] val - A value to be printed.
 * @param[in] len - The maximal decimal string length to print.
 *
 * @return void.
 */
void print_sim_int(int val, int len);

/**
 * @brief Print a a character to SimX log.
 *
 * This function prints a given integer value as a character
 * to SimX log.
 *
 * @param[in] val - A value to be printed.
 *
 * @return void.
 */
void print_sim_putc(int val);

/**
 * @brief Trace a value to SimX log with a specific text.
 *
 * This function prints a given text and integer value with function name and line number
 * to SimX log.
 *
 * @param[in] text - Text to be added to trace.
 * @param[in] val - A value to be traced.
 * @param[in] func - Function name to include in the trace format.
 * @param[in] line - Line number to include in the trace format.
 *
 * @return void.
 */
void sim_trace(const char *text, uint64_t val, const char *func, int line);

#define SIM_TRACE(text, val) sim_trace(text, (uint64_t)(val), __func__, __LINE__)
#define SIM_TRACEVAL(val) SIM_TRACE(#val, val)

#endif
