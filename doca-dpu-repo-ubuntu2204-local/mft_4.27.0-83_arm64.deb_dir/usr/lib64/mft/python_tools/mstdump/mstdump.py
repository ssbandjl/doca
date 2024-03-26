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

# Author: Mahmoud Hasan 11.6.2017

# Python Imports ######################


try:
    import sys
    import os
    import platform
    import subprocess
    import tools_version
    import argparse
except Exception as e:
    print("-E- could not import : %s" % str(e))
    sys.exit(1)


# Constants ###########################
PROG = "mstdump"

# reminder of the old help message, maybe return this format in the future
HELP_MESSAGE = '''   Mellanox %s utility, dumps device internal configuration data\n\
   Usage: %s [-full] <device> [i2c-secondary] [-v[ersion] [-h[elp]]]\n\n\
   -full                        :  Dump more expanded list of addresses\n\
        Note: be careful when using this flag, None safe addresses might be read.\n\
   -v | --version               :  Display version info\n\
   -h | --help                  :  Print this help message\n\
   -c | --csv                   :  Database path\n\
        --cause address.offset  :  Specify address and offset
   Example :\n\
            %s %s\n
'''

######################################################################
# Description:  Execute command and get (rc, stdout-output, stderr-output)
######################################################################


def cmd_exec(cmd):
    # print("Executing: %s" % cmd)
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=True,
                         shell=True)
    res = p.communicate()
    stat = p.wait()
    return stat, res[0], res[1]  # RC, Stdout, Stderr


######################################################################
# Description:  Parse arguments
######################################################################


def parse_args():
    class CauseAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            cause_addr, cause_offset = values
            setattr(namespace, self.dest, values)
            setattr(namespace, "cause_addr", cause_addr)
            setattr(namespace, "cause_offset", cause_offset)

        def check_format(value):
            error_msg = ""
            try:
                cause_addr, cause_offset = value.split(".")
                if not cause_addr or not cause_offset:
                    raise ValueError
            except ValueError:
                error_msg = 'cause format "address.offset", both address and offset must be provided'
            else:
                try:
                    if int(cause_addr, 0) < 0 or int(cause_offset, 0) < 0:
                        raise ValueError
                except ValueError:
                    error_msg = 'cause_address and cause_offset must be non-negative numerical values'
            if error_msg:
                raise ValueError(error_msg)

            return (cause_addr, cause_offset)

    arg_parser = argparse.ArgumentParser(prog=PROG)
    arg_parser.add_argument("device")
    arg_parser.add_argument("-v", "-version", "--version", action="version", version=tools_version.GetVersionString(PROG))
    arg_parser.add_argument("-full", "--full", action="store_true", help="Dump more expanded list of addresses")
    arg_parser.add_argument("-ignore_fail", "--ignore_fail", action="store_true", help="Continue dumipng, even if some addresses fails")
    arg_parser.add_argument("-c", "-csv", "--csv", type=lambda s: '"{}"'.format(s), help="Database path")
    arg_parser.add_argument("--cause", type=CauseAction.check_format, action=CauseAction, metavar="address.offset", help="Specify address and offset")
    arg_parser.add_argument("--cause_addr", help=argparse.SUPPRESS)
    arg_parser.add_argument("--cause_offset", help=argparse.SUPPRESS)
    arg_parser.add_argument("--i2c_secondary", type=lambda s: int(s, 0), help="I2C secondary [0-127]")

    return arg_parser.parse_args()


######################################################################
# Description:  Build and run the mlxdump command
######################################################################


def run_mlxdump(mstdump_args):
    try:
        MFT_BIN_DIR = os.environ['MFT_BIN_DIR'] + os.sep
    except BaseException:
        MFT_BIN_DIR = ""

    executable_path = MFT_BIN_DIR + "mlxdump"
    sub_command_str = "mstdump"
    device_str = "-d %s" % mstdump_args.device
    full_str = "--full" if mstdump_args.full else ""
    ignore_fail_str = "--ignore_fail" if mstdump_args.ignore_fail else ""
    cause_str = ""
    if mstdump_args.cause:
        cause_str = "--cause_addr %s --cause_offset %s" % (mstdump_args.cause_addr,
                                                           mstdump_args.cause_offset)
    i2c_secondary_str = "--i2c_secondary %s" % str(mstdump_args.i2c_secondary) if mstdump_args.i2c_secondary else ""

    csv_str = "--csv %s" % mstdump_args.csv if mstdump_args.csv else ""  # will be None if not specified by user

    mlxdump_cmd = "%s %s %s %s %s %s %s %s" % (executable_path, device_str, sub_command_str, full_str, ignore_fail_str, cause_str,
                                               i2c_secondary_str, csv_str)
    return cmd_exec(mlxdump_cmd)

######################################################################
# Description:  Modify the output of mlxdump and get the needed part
######################################################################


def modify_output(mlxdump_output):
    data = mlxdump_output.splitlines(True)
    for line in data[3:]:
        print(line.strip())

######################################################################
# Description:  Modify the output of mlxdump and get the needed part
######################################################################


def modify_error_message(err_msg, parsed_arguments):
    if "Failed to open device:" in err_msg:
        return "Unable to open device %s. Exiting." % parsed_arguments.device
    return err_msg

######################################################################
# Description:  Main
######################################################################


if __name__ == "__main__":
    if platform.system() != "Windows" and os.geteuid() != 0:
        print("-E- Permission denied: User is not root")
        sys.exit(1)
    parsed_arguments = parse_args()
    rc, output, err = run_mlxdump(parsed_arguments)
    if rc:
        print(modify_error_message(output, parsed_arguments))
        sys.exit(1)
    modify_output(output)
