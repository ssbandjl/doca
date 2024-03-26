/*
 * Copyright (c) 2012-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _BUFFER_CALC_H_
#define _BUFFER_CALC_H_

#ifdef __cplusplus
extern "C" {
#endif

struct BufferCalculationParameters
{
    uint16_t m_line_size;
    uint8_t m_worst_case_num_lines;
    uint8_t m_num_lines_chunk_mode;
    bool m_half_buffer_line_optimization_supported;
    bool m_is_eagle_calculation;
};

static inline unsigned int DivRoundUp(unsigned int dividend, unsigned int divisor)
{
    return (dividend + divisor - 1) / divisor;
}

/*
 * number_of_children_in_operation - Number of children of the AN
 * max_number_of_members - The maximum number of children configured on all ANs
 * user_date_size - The user data size in bytes
 */
static inline unsigned int CalcAggBufferSize(unsigned int number_of_children_in_operation,
                                             unsigned int max_number_of_members,
                                             unsigned int user_data_length,
                                             bool is_reproducible,
                                             const BufferCalculationParameters& buff_calc_param)
{
    if (buff_calc_param.m_is_eagle_calculation) {
        if (is_reproducible) {
            return DivRoundUp(DivRoundUp(number_of_children_in_operation, 4) * 2 * DivRoundUp(user_data_length, 32),
                              (max_number_of_members / 2)) *
                   max_number_of_members * 16;
        } else {
            return max_number_of_members * 16;
        }
    } else {
        if (buff_calc_param.m_num_lines_chunk_mode != 1) {
            return (unsigned int)-1;
        }
        uint8_t num_of_lines_in_chunk = 2;

        if (is_reproducible) {
            uint8_t optimization_factor = (buff_calc_param.m_half_buffer_line_optimization_supported
                                               ? ((user_data_length <= (buff_calc_param.m_line_size / 2)) ? 2 : 1)
                                               : 1);

            return DivRoundUp(DivRoundUp(number_of_children_in_operation, 4 * optimization_factor) * 2 *
                                  DivRoundUp(user_data_length, buff_calc_param.m_line_size),
                              num_of_lines_in_chunk) *
                   num_of_lines_in_chunk * buff_calc_param.m_line_size;
        } else {
            uint8_t additional_lines_for_non_aligned = 0;
            if ((user_data_length > buff_calc_param.m_line_size) && buff_calc_param.m_worst_case_num_lines &&
                (number_of_children_in_operation > 1))
            {
                additional_lines_for_non_aligned = buff_calc_param.m_worst_case_num_lines;
            }

            return DivRoundUp(DivRoundUp(user_data_length, buff_calc_param.m_line_size) + additional_lines_for_non_aligned,
                              num_of_lines_in_chunk) *
                   num_of_lines_in_chunk * buff_calc_param.m_line_size;
        }
    }
}

// Eagle calculation only
static inline unsigned int CalcUserDataLength(unsigned int number_of_children_in_operation,
                                              unsigned int max_number_of_members,
                                              unsigned int agg_buffer_size)
{
    return agg_buffer_size / (max_number_of_members * 16) * (max_number_of_members / 2) /
           (DivRoundUp(number_of_children_in_operation, 4) * 2) * 32;
}

// Eagle calculation only
static inline unsigned int CalcRadix(unsigned int user_data_length, unsigned int max_number_of_members, unsigned int agg_buffer_size)
{
    return agg_buffer_size / (max_number_of_members * 16) * (max_number_of_members / 2) / (DivRoundUp(user_data_length, 32) * 2) * 4;
}

#ifdef __cplusplus
}
#endif

#endif /* _BUFFER_CALC_H_ */
