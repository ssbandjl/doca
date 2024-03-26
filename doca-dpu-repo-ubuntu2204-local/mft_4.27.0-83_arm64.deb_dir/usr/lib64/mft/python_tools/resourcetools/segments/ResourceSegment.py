# Copyright (C) Jan 2020 Mellanox Technologies Ltd. All rights reserved.
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the
# OpenIB.org BSD license below:
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#      - Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#
#      - Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials
#        provided with the distribution.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --


#######################################################
#
# ResourceSegment.py
# Python implementation of the Class ResourceSegment
# Generated by Enterprise Architect
# Created on:      14-Aug-2019 10:11:57 AM
# Original author: talve
#
#######################################################
from segments.Segment import Segment
from segments.SegmentFactory import SegmentFactory
from resourcedump_lib.utils import constants

import struct
import sys


class ResourceSegment(Segment):
    """this class is responsible for holding Resource segment data.
    """

    AGGREGATE_BIT_MASK = 0x1000000

    _segment_type_id = constants.RESOURCE_DUMP_SEGMENT_TYPE_RESOURCE

    aggregate_dword_struct = struct.Struct('I')
    indices_struct = struct.Struct('II')

    def __init__(self, data):
        """initialize the class by setting the class data.
        """
        super().__init__(data)
        self.resource_type = constants.RESOURCE_DUMP_SEGMENT_TYPE_RESOURCE
        self.index1, self.index2 = self.unpack_indices()  # TODO: for optimization move this line to later stage, before parsing

    def get_type(self):
        """get the general segment type.
        """
        return self.resource_type

    def unpack_aggregate_bit(self):
        aggregate_dword, = self.aggregate_dword_struct.unpack_from(self.raw_data, self.segment_header_struct.size)  # todo: test aggregate big/little
        return aggregate_dword & self.AGGREGATE_BIT_MASK

    def aggregate(self, other):
        data_start = self.segment_header_struct.size + self.aggregate_dword_struct.size + self.indices_struct.size
        self.raw_data += other.raw_data[data_start:]

    def unpack_indices(self):
        index1, index2 = self.indices_struct.unpack_from(self.raw_data, self.segment_header_struct.size + self.aggregate_dword_struct.size)
        return index1, index2

    def additional_title_info(self):
        """return index1 and index2 if exists in the segment.
        """
        return " ; index1 = {0}, index2 = {1}".format(hex(self.index1), hex(self.index2))


SegmentFactory.register(constants.RESOURCE_DUMP_SEGMENT_TYPE_RESOURCE, ResourceSegment)
