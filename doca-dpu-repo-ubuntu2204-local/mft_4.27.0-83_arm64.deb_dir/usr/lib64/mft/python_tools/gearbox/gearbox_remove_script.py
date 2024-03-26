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

GBOX_DIR = '/dev/mst/gbox'

if __name__ == '__main__':

    commnad_str = "rm -rf {0}".format(GBOX_DIR)
    sub.getstatusoutput(commnad_str)

    print("-I- Removed all gearbox devices")
