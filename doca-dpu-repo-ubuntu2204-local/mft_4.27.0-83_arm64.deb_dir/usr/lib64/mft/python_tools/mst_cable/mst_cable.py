#
# Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of Nvidia Corporation and its affiliates
# (the "Company") and all right, title, and interest in and to the software
# product, including all associated intellectual property rights, are and
# shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

from __future__ import print_function
import sys
import os
import subprocess
import re
from collections import namedtuple

linux_devices_directory = "/dev/mst/"


class Command(object):
    def __init__(self, cmd_str):
        self.cmd_str = cmd_str

    def execute(self):
        p = subprocess.Popen(self.cmd_str,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=False,
                             shell=True)
        stdout, stderr = p.communicate()
        stat = p.wait()
        return stat, stdout, stderr


class I2cError(Exception):
    pass


class I2cprimary(object):

    def __init__(self, mtusb_device_name):
        self.mtusb_device_name = mtusb_device_name

    def read(self, i2c_secondary_address, address_width, address, size):
        cmd = "i2c -a {0} -x {1} {2} read {3:#x} {4:#x}".format(address_width, size, self.mtusb_device_name, i2c_secondary_address, address)
        # print(cmd)
        rc, out, err = Command(cmd).execute()

        if rc == 0:
            out = out.strip()
            result = []
            for index in range(len(out) - 2, -1, -2):
                byte_value = int(out[index:index + 2], 16)
                result.append(byte_value)
            return result
        else:
            raise I2cError("Failed to read from i2c secondary address {0} ({1})".format(i2c_secondary_address, err))

    def scan(self, devices):
        result = []
        for device in devices:
            try:
                devid = self.read(device.i2c_secondary_address, device.address_width, device.devid_address, 4)[0]
                if devid == device.devid:
                    result.append(device)
            except I2cError:
                pass
        return result


class Linkx(object):

    @classmethod
    def discover(cls, mst_devices):
        cable_devices, mtusb_devices = [], []  # Note that cable_device can contain mtusb device (TestBoard)
        for mst_device in mst_devices:
            if 'cable' in mst_device:
                cable_devices.append(mst_device)
            elif 'mtusb' in mst_device:
                mtusb_devices.append(mst_device)

        linkx_devices1 = cls._discover_chips_in_cables(cable_devices)
        linkx_devices2 = cls._discover_chip_on_EVB(mtusb_devices)
        return linkx_devices1 + linkx_devices2

    @staticmethod
    def convert(in_str):
        result = []
        if in_str is not None and len(in_str):
            for token in in_str.split("\n"):
                if len(token):
                    try:
                        result.append((int(token.split(",")[0]), int(token.split(",")[1])))
                    except BaseException:
                        pass
        return result

    @classmethod
    def _discover_chips_in_cables(cls, cable_devices):
        """
        Discover chips in a trasceiver via micro-controller (FW GateWay)
        The transcevicer can be connected in a system (NIC/Switch) or on a board (TestBoard)
        The function returns a list with the name of discovered linkx devices
        """
        linkx_devices = []
        for cable_device in cable_devices:
            cmd = "mlxcables -d {0} --discover".format(cable_device)
            rc, out, _ = Command(cmd).execute()
            if rc == 0:  # TODO need to
                result = Linkx.convert(out.decode("utf-8"))
                for device_id, num_of_devices in result:
                    if num_of_devices == 1:
                        linkx_device_name = "{0}_lx{1}".format(cable_device, device_id)
                        linkx_devices.append(linkx_device_name)
                    elif num_of_devices > 1:
                        for device_num in range(num_of_devices):
                            linkx_device_name = "{0}_lx{1}_{2}".format(cable_device, device_id, device_num)
                            linkx_devices.append(linkx_device_name)
                    else:
                        raise Exception('Failed to discover chips in cables')
        return linkx_devices

    @classmethod
    def _discover_chip_on_EVB(cls, mtusb_devices):
        """
        Check if the device connected to MTUSB is a board with a linkx chip
        The function returns a list with the name of the discovered linkx devices
        """
        linkx_devices = []
        for mtusb_device in mtusb_devices:

            # Create a list of known i2c secondary (Linkx) devices
            LinkxDevice = namedtuple('LinkxDevice', 'devid i2c_secondary_address address_width devid_address')

            ardbeg_rev0 = LinkxDevice(devid=0x6e, i2c_secondary_address=0x66, address_width=2, devid_address=0x2c >> 2)  # Defective Ardbeg
            ardbeg_rev1 = LinkxDevice(devid=0x7e, i2c_secondary_address=0x66, address_width=2, devid_address=0x2c >> 2)
            ardbeg_mirrored = LinkxDevice(devid=0x70, i2c_secondary_address=0x67, address_width=2, devid_address=0x2c >> 2)
            baritone = LinkxDevice(devid=0x6b, i2c_secondary_address=0x5e, address_width=2, devid_address=0x2c >> 2)
            baritone_mirrored = LinkxDevice(devid=0x71, i2c_secondary_address=0x5f, address_width=2, devid_address=0x2c >> 2)
            menhit_ver0 = LinkxDevice(devid=0x6f, i2c_secondary_address=0x12, address_width=2, devid_address=0x18014 >> 2)
            menhit_ver1 = LinkxDevice(devid=0x72, i2c_secondary_address=0x12, address_width=2, devid_address=0x18014 >> 2)
            menhit_ver2 = LinkxDevice(devid=0x73, i2c_secondary_address=0x12, address_width=2, devid_address=0x18014 >> 2)
            arcus_p_tc = LinkxDevice(devid=0x7f, i2c_secondary_address=0x51, address_width=2, devid_address=0x2c >> 2)
            arcus_p_rev0 = LinkxDevice(devid=0x80, i2c_secondary_address=0x50, address_width=2, devid_address=0xc2 >> 2)
            arcuse_sddv = LinkxDevice(devid=0x82, i2c_secondary_address=0x48, address_width=4, devid_address=0xf0014)
            linkx_devices_to_scan = [ardbeg_rev0, ardbeg_rev1, ardbeg_mirrored, baritone, baritone_mirrored,
                                     menhit_ver0, menhit_ver1, menhit_ver2, arcus_p_tc, arcus_p_rev0, arcuse_sddv]

            # Scan i2c secondarys
            i2c_primary = I2cprimary(mtusb_device)

            i2c_secondarys = i2c_primary.scan(linkx_devices_to_scan)

            if len(i2c_secondarys) == 1:
                linkx_device_name = "{0}_lx{1}".format(mtusb_device.replace(linux_devices_directory, ""), i2c_secondarys[0].devid)
                linkx_devices.append(linkx_device_name)
            elif len(i2c_secondarys) > 1:
                raise Exception('i2c primary detected more than one i2c secondarys devices')

        return linkx_devices


if __name__ == '__main__':  # Implementation for Linux (Windows import the python module)

    try:

        mst_devices = []
        for file_name in os.listdir(linux_devices_directory):
            if 'mtusb' in file_name and 'cable' not in file_name:
                mst_devices.append(linux_devices_directory + file_name)
            else:
                mst_devices.append(file_name)

        linkx_devices = Linkx.discover(mst_devices)
        chipset_cnt = 0
        # Create a file for each link device
        for linkx_device in linkx_devices:
            linkx_device_file = linux_devices_directory + linkx_device
            if not os.path.exists(linkx_device_file):
                open(linkx_device_file, 'w').close()
                chipset_cnt += 1

        print("-I- Added {0} chipset devices ..".format(chipset_cnt))
    except Exception as err:
        print(err, file=sys.stderr)
        sys.exit(1)
