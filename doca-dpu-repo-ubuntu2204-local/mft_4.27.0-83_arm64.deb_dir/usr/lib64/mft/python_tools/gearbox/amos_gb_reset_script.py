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
import argparse
#from mtcr import MstDevice

MUX_secondary_ADDR = 0x32
MUX_CONFIG_ADDR = 0x25DC
MUX_ADDR_WIDTH = 2

gearbox_ids_list = [
    {'id': '4', 'description': 'AGBM', 'set_value': 0xEF},
    {'id': '0', 'description': 'AGB0', 'set_value': 0xFE},
    {'id': '1', 'description': 'AGB1', 'set_value': 0xFD},
    {'id': '2', 'description': 'AGB2', 'set_value': 0xFB},
    {'id': '3', 'description': 'AGB3', 'set_value': 0xF7},
    {'id': 'all', 'description': 'AGB3', 'set_value': 0xBF}
]


def get_value_by_gb_id(gb_id):
    for gb in gearbox_ids_list:
        if gb['id'] == gb_id:
            return gb['set_value']


def gearbox_ids_options():
    'Return a list with all the ids options'
    return [gb_id['id'] for gb_id in gearbox_ids_list]


def run_reset_command(dev_name, gearbox_id):
    value = get_value_by_gb_id(gearbox_id)

    reset_cmd = "i2c -a 2 -d 1 {0} w 0x32 0x2519 {1}".format(dev_name, hex(value))
    print("-I- Command -  {0}  - executed".format(reset_cmd))
    rc, output = sub.getstatusoutput(reset_cmd)

    if rc:
        print("-E- Reset failed. {0}".format(reset_cmd))
    else:
        reset_cmd = "i2c -a 2 -d 1 {0} w 0x32 0x2519 0xFF".format(dev_name)
        print("-I- Command -  {0}  - executed".format(reset_cmd))
        rc, output = sub.getstatusoutput(reset_cmd)

        if rc:
            print("-E- Reset failed. {0}".format(reset_cmd))
        else:
            print("-I- Reset Successede.")


def parse():
    parser = argparse.ArgumentParser(description='parse input')
    parser.add_argument('-d',
                        '--device',
                        action="store",
                        required=True,
                        dest="dev_name",
                        help='Store the device namee')
    parser.add_argument('-g',
                        '--gearbox_id',
                        action="store",
                        required=True,
                        dest="gearbox_id",
                        choices=gearbox_ids_options(),
                        help='gearbox id to set: 0: AGB0, 1:AGB1, 2:AGB2, 3:AGB3, 4:AGBM')
    args = parser.parse_args()
    return args.dev_name, args.gearbox_id


if __name__ == '__main__':
    dev_name, gearbox_id = parse()

    run_reset_command(dev_name, gearbox_id)
