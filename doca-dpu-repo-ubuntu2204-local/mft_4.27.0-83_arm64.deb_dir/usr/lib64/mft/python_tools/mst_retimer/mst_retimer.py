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
import argparse
import shutil

MFT_DEVS_DIR = "/dev/mst/"
RETIMER_DEVS_DIR = "/dev/mst/retimer"
DEV_ID_CRSPACE_ADDRESS = 0xf0014

CMDS = ["add",
        "del",
        "status"]

# Append new retimers to this dictionary
RETIMERS_DIC = {
    "ArcusE": {
        "devid": 0x282,
        "i2c_secondary_address": 0x48,
        "address_width": 4,
        "data_len": 4,
        "devid_address": 0xf0014
    },
    # "ArcusE": {
    #     "devid": 110,
    #     "i2c_secondary_address": 0x48,
    #     "address_width": 4,
    #     "data_len": 4,
    #     "devid_address": 0xf0014
    # }
}


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


class Retimer(object):

    @classmethod
    def discover_external_chips(cls):
        cable_devices, devices_list = [], []
        for mst_device in os.listdir(MFT_DEVS_DIR):
            if 'cable' in mst_device:
                cable_devices.append(mst_device)
            # if 'mtusb' in mst_device and 'cable' not in mst_device:
            #     mtusb_devices.append(MFT_DEVS_DIR + mst_device)

        devices_list = cls._discover_chips_in_cables(cable_devices)
        # retimers_in_evb = cls._discover_chip_on_EVB(mtusb_devices)
        # devices_list.extend(retimers_in_evb)
        retimers_num = cls._create_files(devices_list)
        return retimers_num

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
    def _retimers_ids(cls):
        ids_list = []
        for dev_info in RETIMERS_DIC.values():
            ids_list.append(dev_info["devid"])
        return ids_list

    @classmethod
    def _create_files(cls, devices_list):
        retimer_devs = 0
        if len(devices_list) > 0:
            # crate folder if not exist
            cls._create_dir(RETIMER_DEVS_DIR)
            for device in devices_list:
                # create new file
                dev_file = os.path.join(RETIMER_DEVS_DIR, device)
                if not os.path.exists(dev_file):
                    open(dev_file, 'w').close()
                    retimer_devs += 1
        return retimer_devs

    @classmethod
    def _discover_chips_in_cables(cls, cable_devices):
        """
        Discover retimers in a module via MCU (FW Gateway)
        The function returns a list with the name of discovered retimer devices
        """
        linkx_devices = []
        for cable_device in cable_devices:
            cmd = "mlxcables -d {0} --discover".format(cable_device)
            rc, out, _ = Command(cmd).execute()
            if not rc:
                result = Retimer.convert(out.decode("utf-8"))
                for device_id, num_of_devices in result:
                    if device_id in cls._retimers_ids():
                        if num_of_devices == 1:
                            linkx_device_name = "{0}_rt{1}".format(cable_device, 642)
                            linkx_devices.append(linkx_device_name)
                        else:  # num_of_devices > 1:
                            for device_num in range(num_of_devices):
                                linkx_device_name = "{0}_rt{1}_{2}".format(cable_device, device_id, device_num)
                                linkx_devices.append(linkx_device_name)
        return linkx_devices

    @classmethod
    def _discover_chip_on_EVB(cls, mtusb_devices):
        retimer_devs = []
        retimer_folder_exist = False
        for mtusb_dev in mtusb_devices:  # for each mtusb device found in mst status
            for dev_info in RETIMERS_DIC.values():  # check if retimer by reading i2c dev id
                ret = cls._i2c_cmd_read(mtusb_dev, dev_info["i2c_secondary_address"], dev_info["address_width"],
                                        DEV_ID_CRSPACE_ADDRESS, dev_info["data_len"])
                if ret != -1 and int(ret, 16) == dev_info["devid"]:
                    retimer_dev_name = "{}_rt{}".format(mtusb_dev.replace("/dev/mst/", ""), dev_info["devid"])
                    # # create retimer dirif not exist
                    # if not retimer_folder_exist:
                    #     cls._create_dir(RETIMER_DEVS_DIR)
                    #     retimer_folder_exist = True
                    # #create new file+link
                    # dev_file = os.path.join(RETIMER_DEVS_DIR, retimer_dev_name)
                    # if not os.path.exists(dev_file):
                    #     os.symlink(mtusb_dev, dev_file)
                    #     retimer_devs += 1
                    # break
                    dev_file = os.path.join(RETIMER_DEVS_DIR, retimer_dev_name)
                    retimer_devs.append(dev_file)
        return retimer_devs

    @classmethod
    def _i2c_cmd_read(cls, mtusb_device, i2c_secondary_address, address_width, address, size):
        cmd = "i2c -a {0} -x {1} {2} read {3:#x} {4:#x}".format(
            address_width, size, mtusb_device, i2c_secondary_address, address)
        rc, out, err = Command(cmd).execute()
        if (rc == 0):
            return out
        else:
            return -1

    @classmethod
    def _create_dir(cls, dir_name):
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError:
                print("-E- Creation of the directory %s failed" % dir_name)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(prog="mst_retimer",
                                         description="Retimers discovery script")
        parser.add_argument("command", choices=CMDS)
        args = parser.parse_args()

        if args.command == "del":
            if os.path.exists(RETIMER_DEVS_DIR):
                shutil.rmtree(RETIMER_DEVS_DIR,
                              ignore_errors=False, onerror=None)

        elif args.command == "add":
            chipset_cnt = Retimer.discover_external_chips()
            if chipset_cnt:
                print("-I- Added {} retimer device{} ..".format(chipset_cnt, "s" if chipset_cnt > 1 else ""))
            else:
                print("-I- No Retimer devices found")

        elif args.command == "status":
            if os.path.exists(RETIMER_DEVS_DIR):
                print("Retimer devices:")
                print("----------------")
                for file in os.listdir(RETIMER_DEVS_DIR):
                    print(file)

    except Exception as err:
        print(err, file=sys.stderr)
        sys.exit(1)
