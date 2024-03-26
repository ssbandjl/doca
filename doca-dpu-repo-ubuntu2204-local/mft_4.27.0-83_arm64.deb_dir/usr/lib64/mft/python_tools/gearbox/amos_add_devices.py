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
import os
import glob
import traceback

if sys.version_info < (3, 6):
    print("-E- This tool supports python version >= 3.6. Exiting...")
    sys.exit(1)

try:
    from add_gearbox_devices import Gearbox
except Exception as err:
    print("-E- {0} could not import : {1}".format(os.path.basename(__file__), str(err)))
    traceback.print_exc()
    sys.exit(1)

AMOS_GBOX_STR = "_gbox53108"
BOARD_TYPE = "VMOD0011"
SPECTRUM_3_DEV_ID = 0x250
AMOS_DEVICE_ID = 0x252
AMOS_GB_MNGR_DEV_ID = 0x253


class Amos(Gearbox):
    def __init__(self):
        super(Amos, self).__init__(AMOS_DEVICE_ID, AMOS_GBOX_STR,
                                   SPECTRUM_3_DEV_ID, BOARD_TYPE)

    def create_files_for_gbox_manager(self, parent_device):
        file_name = self.gbox_dir + "/" + parent_device + self.gbox_str + '_ln0_' + 'mngr'
        # check first manager file do not exist
        if not os.path.exists(file_name):
            self.create_file(file_name, parent_device)
            return 1

        return 0

    def get_mtusb_devices(self):
        gbox_list = []
        gbox_mngr_list = []

        files_list = glob.glob(self.mst_devices_path + "*")
        for dev in files_list:
            # if mtusb device - is link
            if os.path.islink(dev):
                # check if gbox or mngr
                if self.check_if_device_gbox(dev, self.device_id, False) or self.check_if_device_gbox(dev, AMOS_GB_MNGR_DEV_ID, True):
                    dev_name = dev.replace(self.mst_devices_path, '')
                    gbox_list.append(dev_name)
                    gbox_mngr_list.append(dev_name)

        return gbox_list, gbox_mngr_list

    def create_mtusb_gb_devices(self):
        gb_devs_created = 0
        gbm_devs_created = 0

        gbox_list, gbox_mngr_list = self.get_mtusb_devices()

        if (len(gbox_list) + len(gbox_mngr_list)) > 0:
            self.create_gbox_dir()
            # create gbox 0-3 for each mtusb
            for dev in gbox_list:
                gb_devs_created += self.create_files_for_gbox(dev)
            # # create mngr for mtusb
            for dev in gbox_mngr_list:
                gbm_devs_created += self.create_files_for_gbox_manager(dev)

        return gb_devs_created, gbm_devs_created


if __name__ == '__main__':

    amos_gearbox = Amos()

    gb_mtusb_devs_count, gbm_mtusb_devs_count = amos_gearbox.create_mtusb_gb_devices()

    gb_i2c_devs_count, gbm_i2c_devs_count = amos_gearbox.create_i2c_gb_devices()

    gb_switch_devs_count, gbm_switch_devs_count = amos_gearbox.create_switch_gb_devices()

    gb_devs_count = gb_mtusb_devs_count + gb_i2c_devs_count + gb_switch_devs_count
    gbm_devs_count = gbm_mtusb_devs_count + gbm_i2c_devs_count + gbm_switch_devs_count

    if amos_gearbox.is_gbox_dir_empty():
        amos_gearbox.delete_gbox_empty_dir()
    if gb_devs_count > 0:
        print("-I- {0} Amos Gearbox devices added".format(gb_devs_count))
    else:
        print("-I- No new Amos Gearbox devices found")
    if gbm_devs_count > 0:
        print("-I- {0} Amos Gearbox Manager devices added".format(gbm_devs_count))
