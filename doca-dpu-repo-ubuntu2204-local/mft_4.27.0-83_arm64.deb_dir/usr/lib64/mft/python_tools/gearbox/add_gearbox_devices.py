#
# Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of Nvidia Corporation and its affiliates
# (the "Company") and all right, title, and interest in and to the software
# product, including all associated intellectual property rights, are and
# shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#


import sys
if sys.version_info.major < 3:
    import commands as sub
else:
    import subprocess as sub
import os
import glob
import re
from mtcr import MstDevice
import math
import regaccess

DEV_TYPE_REG_ADDR = 0xf0014
DEVICE_ID_MASK = 0xFFFF
GB_MNGR_secondary_ADDR = 0x33
DEV_PATH = "/dev/"
GBOX_DIR = '/dev/mst/gbox'
SYSTEM_BOARD_TYPE = "/sys/devices/virtual/dmi/id/board_name"
LC_PRESENT = "/var/run/hw-management/system/lc{0}_present"
DEV_SW_PTRN = re.compile(r'.+(mt5\d{4}_pciconf\d)')
LC_PREFIX = "/dev/lc"
GEARBOX_PREFIX = "gearbox0"
MANAGER_PREFIX = "fpga"

QUERY_LINECARD = 1
QUERY_GEARBOX = 2
GB_MNGR_INDEX = 4
MAX_SLOTS = 8

class Gearbox:
    def __init__(self, device_id,
                 gbox_str, spectrum_device_id,
                 board_type):
        self.device_id = device_id
        self.gbox_str = gbox_str
        self.spectrum_device_id = spectrum_device_id
        self.board_type = board_type
        self.mst_devices_path = '/dev/mst/'

    def check_if_device_gbox(self, device_name,
                             device_id, set_i2c_secondary):
        dev = MstDevice(device_name, 0x48)
        could_read = True
        try:
            if set_i2c_secondary:
                dev.msetI2cscondary(GB_MNGR_secondary_ADDR)
            output = dev.read4(DEV_TYPE_REG_ADDR)
            output = output & DEVICE_ID_MASK
        except BaseException:
            could_read = False
        dev.close()
        if could_read and output == device_id:
            return True
        return False

    def check_if_device_spectrum(self, mst_device):
        could_read = True
        try:
            output = mst_device.read4(DEV_TYPE_REG_ADDR)
            output = output & DEVICE_ID_MASK
        except BaseException:
            could_read = False
        mst_device.close()

        if could_read and output == self.spectrum_device_id:
            return True
        return False

    def create_dir(self):
        if not os.path.isdir(GBOX_DIR):
            try:
                os.makedirs(GBOX_DIR)
            except OSError:
                print("-E- Creation of the directory %s failed" % GBOX_DIR)

    def create_gbox_dir(self):
        self.create_dir()

    def delete_gbox_empty_dir(self):
        if os.path.isdir(GBOX_DIR):
            try:
                os.rmdir(GBOX_DIR)
            except OSError:
                print("-E- Deletion of the directory %s failed" % GBOX_DIR)

    def is_gbox_dir_empty(self):
        if os.path.exists(GBOX_DIR):
            return (len(os.listdir(GBOX_DIR)) == 0)

    def create_file(self, file_name,
                    parent_device):
        if not os.path.exists(file_name):
            parent_file = self.mst_devices_path + parent_device
            if os.path.islink(parent_file):
                link_name = os.readlink(parent_file)
                os.symlink(link_name, file_name)

    def create_files_for_gbox(self, parent_device):
        # check first file with same parent was not created yet
        if len(glob.glob(GBOX_DIR + "/" + parent_device + "*")) == 0:
            for i in range(0, 4):
                file_name = GBOX_DIR + "/" + parent_device + self.gbox_str + '_ln0_' + str(i)
                self.create_file(file_name, parent_device)
            return 4
        return 0

    def check_board_type(self):
        cmd = "cat " + SYSTEM_BOARD_TYPE
        rc, output = sub.getstatusoutput(cmd)
        if not rc:
            return (output == self.board_type)
        return False

    def check_LC_present(self, index):
        cmd = "cat " + LC_PRESENT.format(index)
        rc, output = sub.getstatusoutput(cmd)
        if not rc:
            return (output)
        return False

    def create_i2c_gb_devices(self):
        # if not the board type - return 0 devices
        if not self.check_board_type():
            return 0, 0

        # /dev/lc1/gearbox01
        created_devices = 0
        created_mngr_devices = 0
        dirs = glob.glob(DEV_PATH + "*")
        detected_lcs = []

        self.create_gbox_dir()

        # get lcs dettected
        for dir_name in dirs:
            if LC_PREFIX in dir_name:
                lc_index = dir_name.replace(LC_PREFIX, '')
                gb_files = glob.glob(dir_name + "/*")

                for gb_file in gb_files:
                    add_mngr = 0
                    add_gb = 0
                    if GEARBOX_PREFIX in gb_file:
                        file_name = os.path.basename(gb_file)
                        gb_index = file_name.replace(GEARBOX_PREFIX, '')

                        new_file = GBOX_DIR + "/i2c" + self.gbox_str + "_ln" + str(lc_index) + "_" + str(gb_index)
                        add_gb = 1
                    if MANAGER_PREFIX in gb_file:
                        # if mng
                        new_file = GBOX_DIR + "/i2c" + self.gbox_str + "_ln" + str(lc_index) + "_mngr"
                        add_mngr = 1
                    if (add_gb + add_mngr) > 0 and not os.path.exists(new_file):
                        os.symlink(gb_file, new_file)
                        created_devices += add_gb
                        created_mngr_devices += add_mngr

        return created_devices, created_mngr_devices

    def get_devices_on_slot(self, reg_access,
                            slot_index):
        devices_on_slot = []
        request_message_sequence = 0
        while True:
            try:
                mddq_get = reg_access.sendMddq(QUERY_GEARBOX, request_message_sequence, slot_index)
                if mddq_get[0]["data_valid"]:
                    if mddq_get[1]["device_type"] == 0:  # 0 for GB
                        devices_on_slot.append(mddq_get[1]["device_index"])
                elif mddq_get[0]["response_message_sequence"] == 0:
                    break
            except regaccess.RegAccException:
                break
            request_message_sequence += 1
        return devices_on_slot

    def query_switch_gbs(self, mst_str):
        reg_access = regaccess.RegAccess(pci_device=mst_str)
        # query_type: (1: slot information, 2: devices on slot))
        # request_message_sequence: (slots or devices indicator, if set, we have a device)
        # slot_index: line card index
        # query_index: not related to GBs
        slot_index = 1
        slots = {}
        switch_code = DEV_SW_PTRN.match(mst_str).group(1)
        # query slot information
        try:
            while slot_index <= MAX_SLOTS:
                # send MDDQ only for present LCs
                if self.check_LC_present(slot_index) == "1":
                    mddq_get = reg_access.sendMddq(QUERY_LINECARD, 0, slot_index)
                    if (mddq_get[1]["lc_ready"] == 1) and (mddq_get[1]["card_type"] != 3):
                        slots.update({slot_index: self.get_devices_on_slot(reg_access, slot_index)})
                slot_index += 1
        except regaccess.RegAccException:
            pass

        num_of_gboxs = 0
        num_of_mgboxs = 0
        for lc in slots.keys():
            for gbox in slots[lc]:
                sfx = str(gbox)
                if gbox == GB_MNGR_INDEX:
                    sfx = "mngr"
                    num_of_mgboxs += 1
                else:
                    num_of_gboxs += 1
                file_name = GBOX_DIR + "/" + "switch_" + switch_code + self.gbox_str + "_ln" + str(lc) + "_" + sfx
                # if file do not exist - create it as link
                if file_name and not os.path.exists(file_name):
                    os.symlink(mst_str, file_name)

        return num_of_gboxs, num_of_mgboxs

    def create_switch_gb_devices(self):
        num_of_switch_gb_devs = 0
        for dev in glob.glob("/dev/mst/gbox/*"):
            if 'switch_' in dev:
                num_of_switch_gb_devs += 1

        switches_list = []
        for dev in glob.glob("/dev/mst/*"):
            if DEV_SW_PTRN.match(dev):
                switches_list.append(dev)

        self.create_gbox_dir()

        num_of_gboxs = 0
        num_of_mgboxs = 0
        for switch in switches_list:
            mst_dev = MstDevice(switch)
            if not (self.check_if_device_spectrum(mst_dev)):
                mst_dev.close()
                continue
            mst_dev.close()
            rc = self.query_switch_gbs(switch)
            num_of_gboxs += rc[0]
            num_of_mgboxs += rc[1]

        num_of_new_gbs = num_of_gboxs - num_of_switch_gb_devs

        return num_of_new_gbs, num_of_mgboxs
