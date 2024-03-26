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
    {'id': '4', 'description': 'AGBM', 'set_value': 0x10},
    {'id': '0', 'description': 'AGB0', 'set_value': 0x20},
    {'id': '1', 'description': 'AGB1', 'set_value': 0x21},
    {'id': '2', 'description': 'AGB2', 'set_value': 0x22},
    {'id': '3', 'description': 'AGB3', 'set_value': 0x23}
]


def get_value_by_gb_id(gb_id):
    for gb in gearbox_ids_list:
        if gb['id'] == gb_id:
            return gb['set_value']


def gearbox_ids_options():
    'Return a list with all the ids options'
    return [gb_id['id'] for gb_id in gearbox_ids_list]


def get_command(dev_name, gearbox_id, cmd, value=0):
    if cmd != 'r' and cmd != 'w':
        print("-E- Unknown command.")
        exit

    return_cmd = "i2c -a {0} {1} {2} {3} {4}".format(MUX_ADDR_WIDTH, dev_name, cmd, hex(MUX_secondary_ADDR), hex(MUX_CONFIG_ADDR))

    if cmd == 'w':
        return_cmd = return_cmd + " {0}".format(hex(value))

    return return_cmd


def run_config_command(dev_name, gearbox_id):
    value = get_value_by_gb_id(gearbox_id)

    #qeury_cmd = "i2c -a 2 {0} w 0x32 0x25dc".format(dev_name)
    config_cmd = get_command(dev_name, gearbox_id, 'w', value)
    rc, output = sub.getstatusoutput(config_cmd)

    if rc:
        print("-E- Mux configuration failed. {0}".format(output))
    else:
        runquery_command(dev_name, gearbox_id)


def runquery_command(dev_name, gearbox_id):
    #commnad_str = "i2c -a 2 {0} r 0x32 0x25dc".format(dev_name)
    query_cmd = get_command(dev_name, gearbox_id, 'r')
    rc, output = sub.getstatusoutput(query_cmd)
    if not rc:
        print(output)


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

    run_config_command(dev_name, gearbox_id)
