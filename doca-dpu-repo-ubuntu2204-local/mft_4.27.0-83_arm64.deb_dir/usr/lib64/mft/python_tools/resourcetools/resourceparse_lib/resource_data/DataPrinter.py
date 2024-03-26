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
# DataPrinter.py
# Python implementation of the Class DataPrinter
# Generated by Enterprise Architect
# Created on:      19-Dec-2019 3:18:37 PM
# Original author: talve
#
#######################################################
from resourceparse_lib.utils import constants as cs


class DataPrinter:
    """This class is responsible for set and manage the parser output.
    """
    def __init__(self, verbosity, out_file):
        self._verbosity = verbosity
        self._out_file = out_file
        self._top_notice_db = []

    def print_notice_before_parse(self, notice_msg):
        """This method print notice message according the output type and the.
        """
        if self._verbosity > 0:
            if self._out_file:
                self._top_notice_db.append(notice_msg)
            else:
                print(notice_msg)

    def print_parsed_segment(self, parsed_segment_db, title, segment_separator):
        """This method print the parsed segments after check if we need to print to a file or to screen.
        """
        if self._out_file:
            self._print_to_file(parsed_segment_db, title, segment_separator)
        else:
            self._print_to_screen(parsed_segment_db, title, segment_separator)

    def _print_to_screen(self, parsed_segment_db, title, segment_separator):
        """This method print the parsed segments to the screen.
        """
        if title:
            print(title)
        for seg in parsed_segment_db:
            if segment_separator:
                print(segment_separator)
            parsed_seg = seg.get_parsed_data()
            for field in parsed_seg:
                print(field)
        if segment_separator:
            print(segment_separator)

    def _print_to_file(self, parsed_segment_db, title, segment_separator):
        """This method print the parsed segments to a file.
        """
        with open(self._out_file, "w") as out_file:
            for notice_section in self._top_notice_db:
                out_file.write(notice_section + "\n")
            out_file.write(title + "\n")
            for seg in parsed_segment_db:
                if segment_separator:
                    out_file.write(segment_separator + "\n")
                parsed_seg = seg.get_parsed_data()
                for field in parsed_seg:
                    out_file.write(field + "\n")
            if segment_separator:
                out_file.write(segment_separator + "\n")
        print("write to file: ", self._out_file)

    @classmethod
    def _get_fixed_field(cls, field):
        if str(field).find("Warning[") != -1:
            fixed_field = "Warning"
        else:
            fixed_field = field
        return fixed_field

    @classmethod
    def _build_body_msg(cls, field):
        """This method the body string of a line base on the field content.
        """
        if field == "                    Segment":
            body = " - "
        elif field == "RAW DATA":
            body = ":"
        elif str(field).find("Warning[") != -1:
            body = ":"
        elif field.find("DWORD") != cs.PARSER_STRING_NOT_FOUND:
            body = ":"
        else:
            body = " = "
        return body
