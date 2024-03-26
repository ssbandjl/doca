#!/usr/bin/env python3
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of Nvidia Corporation
# and its affiliates (the "Company") and all right, title, and interest
# in and to the software product, including all associated intellectual
# property rights, are and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
# All Rights reserved.


import argparse
import os
import subprocess
import sys


def wait_systemd(unit: str):
    """
    unit: Systemd unit name.
    """
    # This command may be run from systemd Exec* directives which do not spawn
    # the MainPID (ex. ExecReload). Therefore, we do not check the value from
    # the following command. Getting an answer from pid 1 is a sufficient
    # synchronization barrier to know that pid 1 has finished spawning this
    # process and done its related postprocessing.
    subprocess.run(
        ["systemctl", "show", f"--property=MainPID", unit],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Wait for unit to be running and exec program.")
    parser.add_argument(
        "unit",
        help="Systemd unit name.",
    )
    parser.add_argument(
        "program",
        help="Executable path.",
    )
    parser.add_argument(
        "prog_args",
        nargs="*",
        help="Program arguments. If they include options, insert a single '--' argument before them.",
    )
    args = parser.parse_args()

    wait_systemd(args.unit)
    os.execv(args.program, [args.program] + args.prog_args)


if __name__ == "__main__":
    main()
