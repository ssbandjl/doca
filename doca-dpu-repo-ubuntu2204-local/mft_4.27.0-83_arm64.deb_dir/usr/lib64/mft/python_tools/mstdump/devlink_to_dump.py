#
# Copyright (c) 2013-2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
import argparse

tmp_dump_2 = "/tmp/devmon_2.dump"
csvs_path = "/usr/share/mft/mstdump_dbs/"
scratchpad2_devices = ["ConnectX6", "ConnectX6DX", "ConnectX6LX", "ConnectX7", "Bluefield2"]


def get_dword_val(lst):
    return ("0x" + "".join(lst[::-1]))


def dword_pad(num):
    res = num
    num_len = len(hex(num)) - 2

    if(num_len < 8):
        res = str(hex(res)).replace('0x', '')
        res = '0x' + (8 - num_len) * '0' + res
        return res

    return str(res)


def get_valid_address_csv(csv_output):
    file_c = open(csv_output, 'r')
    valid_addresses = []
    lines = file_c.readlines()[1:]

    for line in lines:
        ls = line.strip().split(',')
        start_add = int(ls[0], 0)
        dword_to_read = int(ls[1])
        end_add = start_add + dword_to_read * 4 - 4
        valid_addresses.extend([dword_pad(start_add)])
        runner = start_add + 4
        while(runner <= end_add):
            valid_addresses.extend([dword_pad(runner)])
            runner += 4

    file_c.close()

    return valid_addresses


def get_valid_udmp(file_input, csv_output, file_output):
    file_i = open(file_input, 'r')
    file_o = open(file_output, 'w')
    step = 0
    address = int(0)

    csv_out = set(get_valid_address_csv(csv_output))
    lines = file_i.readlines()[1:]
    count = 0

    for line in lines:
        count += 1
        list = line.strip().split()
        for i in range(0, 4):
            val = get_dword_val(list[i + step:i + step + 4])
            if(val == "0x"):
                continue
            add = dword_pad(address)
            if(add in csv_out):
                new_line = add + " " + val
                file_o.write(new_line + '\n')

            step += 3
            address += 4
        step = 0

    file_i.close()
    file_o.close()


def main():

    parser = argparse.ArgumentParser(description="Convet the devlink output to dump format")
    parser.add_argument('devlink_output', type=str, help="devlink output file")
    parser.add_argument('csv', type=str, help="device csv file")
    parser.add_argument('dump_output', type=str, help="converted devlink to dump file. script output")
    args = parser.parse_args()

    csv = args.csv if '.' in args.csv else csvs_path + args.csv + ".csv"
    dev_name = args.csv.split("/")[-1].split(".")[0] if '.' in args.csv else args.csv

    if os.path.exists(args.dump_output):
        os.remove(args.dump_output)
    if os.path.exists(tmp_dump_2):
        os.remove(tmp_dump_2)

    get_valid_udmp(args.devlink_output, csv, args.dump_output)

    if(dev_name in scratchpad2_devices):
        new_csv = csv.replace(".csv", ".csv2")
        get_valid_udmp(args.devlink_output, new_csv, tmp_dump_2)

        fin = open(args.dump_output, "r")
        data1 = fin.read()
        fin.close()

        fin = open(tmp_dump_2, "r")
        data2 = fin.read()
        fin.close()

        combined_data = data1 + data2

        fout = open(args.dump_output, "w")
        fout.write(combined_data)
        fout.close()


if __name__ == "__main__":
    main()
