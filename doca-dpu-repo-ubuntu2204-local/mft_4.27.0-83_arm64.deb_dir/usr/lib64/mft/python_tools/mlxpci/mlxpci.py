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
# Author:  Ahmed Awwad     ahmadaw@mellanox.com    Created: 2019-Jan

# Python Imports
import subprocess
import argparse
import platform
import sys
import re

# Common Imports
import tools_version
from mlxpci_lib import PCIDeviceFactory
from mlxpci_lib import NotSupportedDeviceException

# Constants
DEVICE_HELP_MESSAGE = " 1) in case the user didn't supply a device we'll execute for all the PCI devices with MLNX vendor ID.\n" \
                      "2) if the user won't supply the full dbdf we'll execute on all the device under the ... " \
                      "Example: if the user supply only dbd then we'll find the function under this dbd."

# Helper methods


def exec_cmd(cmd):
    """
    A function for executing commands
    """
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=False,
                         shell=True)
    output = p.communicate()
    stat = p.wait()
    return stat, output[0].decode('utf-8'), output[1].decode('utf-8')  # RC, Stdout, Stderr


def parse_cmd():
    """
    A function to parse cmd line for mlxpci
    """
    parser = argparse.ArgumentParser(description='Mellanox Pci Operations')
    parser.add_argument('--version', '-v',
                        help="Print tool's version",
                        action="version",
                        version=tools_version.GetVersionString("mlxpci"))
    parser.add_argument('--device', '-d',
                        required=False,
                        type=validate_input_device,
                        help=DEVICE_HELP_MESSAGE,
                        default=[])
    parser.add_argument('command',
                        nargs=1,
                        choices=['save', 'load'],
                        help='save/load pci configuration')
    parser.add_argument('--log',
                        choices=['critical', 'error', 'warning', 'info', 'debug'],
                        help=argparse.SUPPRESS,
                        default="info")
    args = parser.parse_args()
    return args


def validate_input_device(device_input):
    """
    This is to validate the device is given in dbdf format
    """
    machine_platform = platform.platform()
    if "Linux" in machine_platform:
        cmd = "lspci -s {0} -D".format(device_input)
        (rc, out, _) = exec_cmd(cmd)
        if rc != 0 or not out:
            raise argparse.ArgumentTypeError("{0} is not valid PCI device".format(device_input))
        devices = []
        for line in out.split('\n')[:-1]:
            devices.append(line[:12])
        return devices
    elif "FreeBSD" in machine_platform:
        cmd = "pciconf -l {0}".format(device_input)
        (rc, out, _) = exec_cmd(cmd)
        if rc != 0 or not out:
            err_msg = "{0} is not valid PCI device".format(device_input)
            raise argparse.ArgumentTypeError(err_msg)
        return [device_input]
    else:
        raise RuntimeError("OS [%s] is not supported yet" % machine_platform)


def get_mlnx_devices():
    """
    Get mellanox devices and returns a list as dbdf
    """
    devices = []
    machine_platform = platform.platform()
    if "Linux" in machine_platform:
        cmd = 'lspci -d 15b3: -D'
    elif "FreeBSD" in machine_platform:
        cmd = 'pciconf -l | grep 15b3'
    (rc, out, _) = exec_cmd(cmd)
    if rc != 0:
        raise RuntimeError("Failed to execute '{0}'".format(cmd))
    for line in out.splitlines():
        if "FreeBSD" in machine_platform:
            dbdf_fbsd_ptrn = re.compile("\\S+@(pci\\S+):\\s+")
            dbdf_match = dbdf_fbsd_ptrn.search(out)
            if out is not None:
                dbdf = dbdf_match.groups()[0]
        else:
            dbdf = line.split()[0]
        devices.append(dbdf)
    return devices


def main():
    args = parse_cmd()
    debug_level = args.log
    devices = args.device
    if devices == []:
        devices = get_mlnx_devices()

    command = args.command[0]
    for device in devices:
        try:
            pci_device = PCIDeviceFactory().get(device, debug_level)
        except NotSupportedDeviceException:
            continue
        if command == "save":
            pci_device.save_configuration_space(to_file=True)
        elif command == "load":
            pci_device.restore_configuration_space()


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except Exception as e:
        sys.stderr.write("-E- {0}\n".format(str(e)))
        rc = 1
    sys.exit(rc)
