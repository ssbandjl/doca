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

import subprocess
import logging
import os
import platform
from re import match as regex_match
from re import sub as regex_subs
from re import finditer as re_finditer
from xml.etree import ElementTree
from tempfile import NamedTemporaryFile

from mlxburn_constants import *


def query(query_target, is_device=True, exit_on_error=True, do_print=False, quick=False, must_query_flags=[], subcommand='q', *additional_args):
    target_flag = "-d"
    target_type_str = "device"
    if not is_device:
        target_flag = "-i"
        target_type_str = "file"

    log_func = error_exit
    warning_suffix = ""
    if not exit_on_error:
        log_func = logging.warning
        warning_suffix = " Ignoring."

    query_cmd = [BURN_PROGRAM, target_flag, query_target]

    if quick:
        query_cmd.append("-qq")

    query_cmd.extend(must_query_flags)

    query_cmd.append(subcommand)
    query_cmd.extend(additional_args)

    logging.debug("running {}".format(query_cmd))
    proc_ret = subprocess.run(query_cmd, stdout=subprocess.PIPE, universal_newlines=True)
    if proc_ret.returncode != 0:
        log_func("Failed to query {}: {}. Error: {}.{}".format((target_type_str), (query_target), (proc_ret.stdout.strip()), (warning_suffix)))
        return None

    lines = proc_ret.stdout.strip().split("\n")
    query_map = parse_query(lines)

    if do_print:
        for line in lines:
            logging.info(line)

    return query_map


def parse_query(lines):
    query_map = {}
    key = None
    for line in lines:
        line = line.strip()
        m = regex_match(r"(.+):\s* (.+)", line)
        if m:
            key = regex_subs(r"\s+", "_", m.group(1))
            value = m.group(2).strip()
        elif not key or not line:
            if line:
                logging.debug("query line:\n{} could not be parsed".format(line))
            continue
        else:
            # key is previous key
            value = line

        field_map = {}
        for n in re_finditer(r"(\w+)=(\S+)", value):
            field_map[n.group(1)] = n.group(2)
        if field_map:  # check if value is a dictionary
            value = field_map
        else:  # check if value is a list
            value_list = [val.strip() for val in value.split(",")]
            if len(value_list) > 1:
                value = value_list

        existing_value = query_map.get(key)
        if existing_value:
            if not isinstance(existing_value, list):
                existing_value = [existing_value]
            existing_value.append(value)
            value = existing_value
        query_map[key] = value
    return query_map


def check_python_version():
    py_major, py_minor = [int(e) for e in platform.python_version_tuple()[:2]]
    enc_py_major, enc_py_minor, _ = ENCRYPTION_MIN_PYTHON_VERSION
    if py_major < enc_py_major or py_minor < enc_py_minor:
        error_exit("please install python {} or newer for encryption. Tool can't encrypt the image for non encrypted image please use -noencrypt flag".format(('.'.join([str(a) for a in ENCRYPTION_MIN_PYTHON_VERSION]))))


def normalize_XML(xml_str: str) -> str:
    return regex_subs(r'\&(?!(?:lt;|apos;|amp;|gt;|quot;))', r'&amp;', xml_str)


def error_exit(message: str, error_code=1):
    logging.error(message)
    raise Exception(message, error_code)


def execute_check(cmd: list, error_message=DEFAULT_ERROR_MESSAGE, stdout=subprocess.PIPE, shell=False):
    if shell:
        cmd = '"{}" '.format(cmd[0]) + ' '.join(cmd[1:])

    proc_ret = subprocess.run(cmd, stdout=stdout, universal_newlines=True, shell=shell)
    rc = proc_ret.returncode
    if rc != 0:
        if error_message == DEFAULT_ERROR_MESSAGE:
            for stream in ("stderr", "stdout"):
                if getattr(proc_ret, stream, None):
                    error_message += "\n" + PROCESS_OUTPUT.format(prog=r"{prog}", stream=stream, content=r"{" + stream + r"}")
        error_message = error_message.format(prog=cmd[0], rc=rc, cmd=" ".join(cmd), stdout=proc_ret.stdout, stderr=proc_ret.stderr)
        error_exit(error_message)

    return proc_ret


def bin_sect_to_tempfile(element: ElementTree.Element):
    bin_str = element.text
    temp_file = NamedTemporaryFile(delete=False)

    for byte in bin_str.split():
        try:
            bytesWritten = temp_file.write(bytes([int(byte, 16)]))
            if bytesWritten != 1:
                raise Exception()
        except BaseException:
            error_exit("Failed writing to temporary file")
    temp_file.close()
    return temp_file


def bin_sect_to_string(element: ElementTree.Element) -> str:
    bin_str = element.text
    if element.get('name') == UUID_SECTION:
        return bin_str.strip()
    else:
        return regex_subs(r"\s", "", bin_str)


def is_valid_file(path):
    if not os.path.exists(path):
        msg = "file {} not found".format(path)
        raise argparse.ArgumentTypeError(msg)
