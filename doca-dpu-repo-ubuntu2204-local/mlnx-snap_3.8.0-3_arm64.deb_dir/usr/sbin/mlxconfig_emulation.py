#!/usr/bin/python2

from __future__ import print_function

__author__ = "Max Gurtovoy"
__version__ = '1.0'

import argparse
import sys
import os
import commands

COMPONENTS = {"INTERNAL_CPU_MODEL" : 1,
              "ECPF_ESWITCH_MANAGER" : 1,
              "ECPF_PAGE_SUPPLIER" : 1,
              "NVME_EMULATION_ENABLE": 1,
              "NVME_EMULATION_NUM_VF" : 0,
              "NVME_EMULATION_NUM_PF" : 2,
              "NVME_EMULATION_VENDOR_ID" : 5555,
              "NVME_EMULATION_DEVICE_ID" : 24577}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--check',
                        help="check the mlxconfig configuration status, without setting",
                        default=False,
                        action="store_true",
                        dest="check")
    parser.add_argument('-n', '--components',
                        help="comma seperated list of components and values to check/set",
                        type=str,
                        dest="components")

    args = parser.parse_args()
    components = {}
    if args.components:
        for key_val in args.components.split(","):
            components[key_val.split("=")[0]] = int(key_val.split("=")[1])
    else:
        components = COMPONENTS

    count = 0
    g_rc = 0
    rc, out = commands.getstatusoutput('lspci -nd 15b3: | grep a2d[26]')
    if rc:
        sys.exit(rc)

    devs = []
    for pci_dev in out.split("\n"):
        devs.append(pci_dev.split(" ")[0])

    for mlxdev in devs:
        print("checking %s configuration..." % mlxdev)
        for comp, val in components.items():
            print("checking %s is %d" % (comp, val))
            rc, out = commands.getstatusoutput("mlxconfig -d %s -e q %s" % (mlxdev, comp))
            if rc == 0:
                for l in out.splitlines():
                    if comp in l:
                        tmp = [c for c in l.split() if c != " " and c != "*"]
                        if "(" in tmp[-1]:
                            curr_boot = int(tmp[-2].split("(")[1].strip(")"))
                            next_boot = int(tmp[-1].split("(")[1].strip(")"))
                        else:
                            curr_boot = int(tmp[-2])
                            next_boot = int(tmp[-1])
                        if args.check:
                            print("%s: curr_boot=%d next_boot=%d" % (comp, curr_boot, next_boot))
                            rc = 1 if curr_boot != val or next_boot != val else 0
                        elif curr_boot != val or next_boot != val:
                            print("setting %s to %d" % (comp, val))
                            rc, out = commands.getstatusoutput("mlxconfig -d %s --yes set %s=%d" % (mlxdev, comp, val))
                            if rc == 0:
                                count += 1
                        else:
                            print("%s already configured to %d. Nothing to do." % (comp, val))
            g_rc |= rc

    if count:
        print("num configured devices: %d. please reset the Bluefield" % count)

    sys.exit(g_rc)


if __name__ == "__main__":
    main()
