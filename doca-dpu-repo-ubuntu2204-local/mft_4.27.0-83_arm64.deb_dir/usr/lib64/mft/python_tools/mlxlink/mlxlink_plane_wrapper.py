
# Copyright (c) 2004-2010 Mellanox Technologies LTD. All rights reserved.
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
import sys
import json


class MlxlinkWrapper:
    def __init__(self):
        self.stdout = None
        self.stderr = None

    def run_mlxlink(self, mlxlink_arguments: list[str]):
        result = subprocess.run(
            mlxlink_arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        self.stdout = result.stdout
        self.stderr = result.stderr


def find_device_arg_index(argv):
    # Find the index of the device given as argument to mlxlink command.
    for index, arg in enumerate(argv):
        if arg == '-d' or arg == '--device':
            return index + 1


def get_system_image_guid(aggregated_port_fd):
    # remove /dev/mst prefix from the file name.
    fd_base_name = aggregated_port_fd.split('/')[-1]
    # remove "_aggregated_port" suffix from the file name.
    system_image_guid = fd_base_name[:fd_base_name.find("_aggregated_port")]
    return system_image_guid


def get_lid_list() -> list[str]:
    aggregated_port_fd = sys.argv[find_device_arg_index(sys.argv)]
    system_image_guid = get_system_image_guid(aggregated_port_fd)
    with open(aggregated_port_fd) as f:
        aggregated_port_info = json.load(f)
    # The LIDs in each plane are the JSON keys found under the System Image GUID.
    lids = aggregated_port_info[system_image_guid].keys()
    return lids


def main(mlxlink_arguments):
    lids = get_lid_list()
    # Read ASIC mapping from the aggregated port device
    # Run the same commands but replace the device according to the asic map
    # Add the port for the command
    for lid in lids:
        mlxlink_wrapper = MlxlinkWrapper()
        # replace aggregated port device with actual lid device.
        mlxlink_arguments[find_device_arg_index(mlxlink_arguments)] = lid
        try:
            mlxlink_wrapper.run_mlxlink(mlxlink_arguments)
        except subprocess.CalledProcessError as e:
            print(f"mlxlink returned a non-zero exit code: {e.returncode}")
            print(f"{e.stderr}")
            return 1
        except FileNotFoundError:
            print("mlxlink executable not found. Please provide the correct path.")
            return 1

        lines = mlxlink_wrapper.stdout.strip().split('\n')
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py mlxlink_arguments")
        sys.exit(1)
    main(sys.argv[1:])
