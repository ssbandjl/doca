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

import glob

GBOX_DIR = '/dev/mst/gbox/*'

if __name__ == '__main__':
    first = True
    files_list = glob.glob(GBOX_DIR)
    for file_name in files_list:
        if first:
            print("Gearbox devices:")
            print("-------------------")
            first = False
        print(file_name)

    if len(files_list) > 0:
        print("\n")
