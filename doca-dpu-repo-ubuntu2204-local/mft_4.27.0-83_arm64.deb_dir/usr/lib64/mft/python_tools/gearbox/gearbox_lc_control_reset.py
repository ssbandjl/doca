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


def run_command_lc_reset_gbs(dev_name):
    # write 1 to bit 13 in DWORD - address 0x3180
    commnad_str = "mcra {0} 0x2180.13:1 1".format(dev_name)
    print("Running command: {}".format(commnad_str))
    rc, output = sub.getstatusoutput(commnad_str)
    return rc, output


def run_command_lc_reset_all(dev_name):
    commnad_str = "mcra {0} 0xF01C.0 1".format(dev_name)
    print("Running command: {}".format(commnad_str))
    rc, output = sub.getstatusoutput(commnad_str)
    return rc, output


def parse():
    parser = argparse.ArgumentParser(description='parse input')
    parser.add_argument('-d',
                        '--device',
                        action="store",
                        required=True,
                        dest="dev_name",
                        help='Store the device namee')
    parser.add_argument('--gbs',
                        action="store_true",
                        help='reset all agbs')
    parser.add_argument('--all',
                        action="store_true",
                        help='reset all agbs and agbm')
    args = parser.parse_args()

    if (not args.gbs and not args.all):
        print("Error: Mast select --gbs/--all")
        quit()
    return args.dev_name, args.gbs, args.all


if __name__ == '__main__':
    dev_name, is_gbs, is_all = parse()
    if is_all:
        rc, output = run_command_lc_reset_gbs(dev_name)
        if rc:
            print("output = {}".format(output))
    if is_gbs:
        rc, output = run_command_lc_reset_all(dev_name)
        if rc:
            print("output = {}".format(output))
