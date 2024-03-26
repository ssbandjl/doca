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
import argparse


def dword_pad(num):
    res = num
    num_len = len(hex(num)) - 2

    if(num_len < 8):
        res = str(hex(res)).replace('0x', '')
        res = '0x' + (8 - num_len) * '0' + res
        return res

    return str(res)


def get_last_address_csv1(csv1):
    last_line = csv1.readlines()[-1].strip().split(',')
    start_add = int(last_line[0], 0)
    dword_to_read = int(last_line[1])
    end_add = start_add + dword_to_read * 4 - 4

    return(end_add)


def calc_dwords(add_arr):

    if(not bool(add_arr)):
        return []

    dword_cnt = 1
    add_arr_len = len(add_arr)
    start_add = add_arr[0]
    valid = []

    for i, addr in enumerate(add_arr):
        if(i < add_arr_len - 1):
            if(add_arr[i + 1] - addr == 4):
                dword_cnt += 1
            else:
                valid.extend([dword_pad(start_add) + ',' + str(dword_cnt) + ','])
                dword_cnt = 1
                start_add = add_arr[i + 1]

    if(dword_cnt):
        valid.extend([dword_pad(start_add) + ',' + str(dword_cnt) + ','])

    return valid


def get_scratchpad2_addresses(mstdump, last_add):
    file_c = open(mstdump, 'r')
    csv2_start = False
    add = []
    lines = file_c.readlines()
    i = 0

    for line in lines:
        ls = line.strip().split(' ')
        start_add = int(ls[0], 0)

        if(csv2_start == False):
            if(start_add != last_add):
                continue
            else:
                # print("After this address {} will start scratchpad2".format(dword_pad(start_add)))
                # sys.exit()
                csv2_start = True
                continue
        # print(i)
        # i += 1
        add.extend([start_add])
        # print("add-",dword_pad(add[0]))
        # sys.exit()
    file_c.close()

    return(calc_dwords(add))


def create_csv2(csv1, mstdump, output_dest):
    file_c = open(csv1, 'r')
    file_o = open(output_dest, 'w')
    file_o.write("#Addr, Size, Enable addr\n")
    valid_addresses = []

    last_add = get_last_address_csv1(file_c)
    # print(dword_pad(last_add))
    valid_addresses = set(get_scratchpad2_addresses(mstdump, last_add))
    for add_val in valid_addresses:
        # print(add_val)
        file_o.write(add_val + '\n')

    file_c.close()
    file_o.close()


def main():

    csv1 = str(sys.argv[1])
    mstdump = str(sys.argv[2])
    output_dest = str(sys.argv[3])

    create_csv2(csv1, mstdump, output_dest)


if __name__ == "__main__":

    main()
