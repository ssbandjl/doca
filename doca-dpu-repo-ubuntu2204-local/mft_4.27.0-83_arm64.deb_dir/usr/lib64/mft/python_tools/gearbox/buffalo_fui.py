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

line_card_ids = [
    {'id': '9', 'description': 'ALL', 'set_value': 0xFF},
    {'id': '1', 'description': 'LC1', 'set_value': 0x04},
    {'id': '2', 'description': 'LC2', 'set_value': 0x05},
    {'id': '3', 'description': 'LC3', 'set_value': 0x06},
    {'id': '4', 'description': 'LC4', 'set_value': 0x07},
    {'id': '5', 'description': 'LC5', 'set_value': 0x08},
    {'id': '6', 'description': 'LC6', 'set_value': 0x09},
    {'id': '7', 'description': 'LC7', 'set_value': 0x0A},
    {'id': '8', 'description': 'LC8', 'set_value': 0x0B}
]

device_type = [
    {'type': 'MNG', 'valididy_str': ['CPLD000232', 'CPLD060232', 'CPLD000292', 'CPLD000227'], 'method': 'run_mng_command'},
    {'type': 'MAIN', 'valididy_str': ['CPLD000245', 'CPLD000246'], 'method': 'run_main_command'},
    {'type': 'LCi_CPLD', 'valididy_str': ['CPLD000217'], 'method': 'run_cpld_command'},
    {'type': 'LCi_FPGA', 'valididy_str': ['FPGA000220'], 'method': 'run_fpga_command'}
]


def get_value_by_lc_id(lc_id):
    return [lc['set_value'] for lc in line_card_ids if lc['id'] == lc_id][0]


def line_card_ids_options():
    'Return a list with all the ids options'
    return [lc_id['id'] for lc_id in line_card_ids]


def get_validity_str(dev_type):
    return [dev['valididy_str']for dev in device_type if dev['type'] == dev_type][0]


def get_method(dev_type):
    return [dev['method']for dev in device_type if dev['type'] == dev_type][0]


def device_type_options():
    'Return a list with all the types'
    return [dev['type'] for dev in device_type]


def run_command(cmd):
    rc, output = sub.getstatusoutput(cmd)
    if rc:
        print("-E- Failed runnning command {0}. \n{1}".format(cmd, output))
    return rc


def set_jtag(value):
    run_command("echo {0} > /var/run/hw-management/system/jtag_enable". format(value))


def cpld_update(image_path):
    print("Starting CPLD update, please wait...\n")
    return run_command("cpldupdate --gpio --no_jtag_enable {0}".format(image_path))

# burning CPLDs on management board:


def run_mng_command(image_path, dummy):
    set_jtag(1)
    rc = cpld_update(image_path)
    set_jtag(0)
    if not rc:
        print("+----------------+")
        print("| passed for MNG |")
        print("+----------------+")


# burning CPLDs on switch board board
def run_main_command(image_path, dummy):
    set_jtag(2)
    rc = cpld_update(image_path)
    set_jtag(0)
    if not rc:
        print("+-----------------+")
        print("| passed for MAIN |")
        print("+-----------------+")


def set_upgrade_en(value, lc_id, lc_type):
    if lc_type == 'fpga' and value == 1:
        run_command("echo 0 >  /var/run/hw-management/lc{0}/system/cpld_upgrade_en".format(lc_id))
    run_command("echo {0} > /var/run/hw-management/lc{1}/system/{2}_upgrade_en".format(value, lc_id, lc_type))


def check_image_validity(dev_type):
    valid_str_list = get_validity_str(dev_type)
    valid_image = False
    for str in valid_str_list:
        if str in image_path:
            valid_image = True
            break
    if not valid_image:
        print("-E- image file do not match for device type {0}".format(dev_type))
        exit()


def run_lc_command_inner(image_path, lc_id, lc_type):
    #value = get_value_by_lc_id(lc_id)
    set_jtag(int(lc_id) + 3)
    set_upgrade_en(1, lc_id, lc_type)
    rc = cpld_update(image_path)
    set_upgrade_en(0, lc_id, lc_type)
    set_jtag(0)
    if rc:
        exit()
    else:
        print("+-----------------+")
        print("| passed for LC={0} |".format(lc_id))
        print("+-----------------+")


def run_lc_command(image_path, lc_id, lc_type):
    if int(lc_id) not in range(1, len(line_card_ids) + 1):
        print("-E- {0} is not a valid LC index. Please provide valid LC index [1..9]".format(lc_id))
        exit()
    # run for all LCs
    if (lc_id == 9):
        for id in range(1, 9):
            run_lc_command_inner(image_path, id, lc_type)
    else:
        run_lc_command_inner(image_path, lc_id, lc_type)


def run_cpld_command(image_path, lc_id):
    run_lc_command(image_path, lc_id, "cpld")


def run_fpga_command(image_path, lc_id):
    run_lc_command(image_path, lc_id, "fpga")


def parse():
    parser = argparse.ArgumentParser(description='parse input')
    parser.add_argument('-t',
                        '--type',
                        action="store",
                        required=True,
                        dest="dev_type",
                        choices=device_type_options(),
                        help='MNG / MAIN / LCi_CPLD / LCi_FPGA')
    parser.add_argument('-l',
                        '--lc',
                        action="store",
                        required=False,
                        dest="lc_id",
                        choices=line_card_ids_options(),
                        help='Line card index in hex 1-8 or9 for all LCs')
    parser.add_argument('-i',
                        '--image',
                        action="store",
                        required=True,
                        dest="image_path",
                        help='image to burn full path')
    args = parser.parse_args()
    return args.dev_type, args.lc_id, args.image_path


if __name__ == '__main__':
    dev_type, lc_id, image_path = parse()

    check_image_validity(dev_type)

    method = get_method(dev_type)
    globals()[method](image_path, lc_id)
