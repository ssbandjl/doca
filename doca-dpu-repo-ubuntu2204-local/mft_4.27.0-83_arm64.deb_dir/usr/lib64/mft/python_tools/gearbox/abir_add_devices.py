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


ABIR_GBOX_STR = "_abir_gbox53XXX"
BOARD_TYPE = "VMOD0011"
SPECTRUM_4_DEV_ID = 0x254
ABIR_DEVICE_ID = 0x256


class Abir(Gearbox):
    def __init__(self):
        super(Abir, self).__init__(ABIR_DEVICE_ID, ABIR_GBOX_STR,
                                   SPECTRUM_4_DEV_ID, BOARD_TYPE)

    def get_mtusb_devices(self):
        gbox_list = []

        files_list = glob.glob(self.mst_devices_path + "*")
        for dev in files_list:
            # if mtusb device - is link
            if os.path.islink(dev):
                # check if gbox
                if self.check_if_device_gbox(dev, self.device_id, False):
                    dev_name = dev.replace(self.mst_devices_path, '')
                    gbox_list.append(dev_name)

        return gbox_list

    def create_mtusb_gb_devices(self):
        gb_devs_created = 0

        gbox_list = self.get_mtusb_devices()

        if len(gbox_list) > 0:
            self.create_gbox_dir()
            # create gbox 0-3 for each mtusb
            for dev in gbox_list:
                gb_devs_created += self.create_files_for_gbox(dev)

        return gb_devs_created


if __name__ == '__main__':

    abir_gearbox = Abir()

    abir_mtusb_devs_count = abir_gearbox.create_mtusb_gb_devices()

    if abir_gearbox.is_gbox_dir_empty():
        abir_gearbox.delete_gbox_empty_dir()
    if abir_mtusb_devs_count > 0:
        print("-I- {0} Abir Gearbox devices added".format(abir_mtusb_devs_count))
    else:
        print("-I- No new Abir Gearbox devices found")
