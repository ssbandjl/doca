#!/usr/bin/env python3

#
# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

import subprocess
import json
import sys
import struct
import hashlib
import itertools
import os
import os.path
import shutil
import argparse
import psutil, shutil, stat
from ctypes import *
import importlib
import binascii
import datetime
import json
import logging
import os
from typing import Dict, Union, Optional, Any, Set
from urllib import request
from pathlib import Path
import pdbparse
import pdbparse.undecorate
import http.client as httplib
import pefile
import binascii
import ast
from pefile import PE

class LinuxApsh():

    def shell_cmd(self, cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (out, err) = map(lambda x: (x or b'').decode('utf-8'), proc.communicate())
        return (out, err)

    # hash_build run
    def run_hash_build(self, args):
        if not args.pid:
            return

        if os.path.exists("hash.zip"):
            os.remove("hash.zip")

        if os.path.exists("apsh_client_build_hash"):
            shutil.rmtree("apsh_client_build_hash")

        os.mkdir("apsh_client_build_hash")
        irregular_files = open("apsh_client_build_hash/doca_apsh_irregular_files.txt", "w")


        with open("/proc/" + str(args.pid) + "/maps", 'r') as map_file:
            files = list(set([line.strip().split()[-1] for line in map_file.readlines() if len(line.strip().split()) == 6]))
            files = [file for file in files if not file.startswith('[')]
            for file in files:
                if os.path.isfile(file):
                    shutil.copy(file, "apsh_client_build_hash")
                else:
                    # Add "#" to the beginning of the line to distinuish it.
                    irregular_files.write("#" + file[file.rfind("/") + 1 :] + "\n")
        irregular_files.close()

        # extract vdso memory from memory
        with open("/proc/" + str(args.pid) + "/maps", 'r') as map_file:
            [start, end] = [int(num, 16) for num in [line.strip().split()[0].split('-') for line in map_file.readlines() if '[vdso]' in line][0]]
            with open("/proc/" + str(args.pid) + "/mem", 'rb') as mem_file:
                mem_file.seek(start)
                vdso_data = mem_file.read(end - start)
                with open("apsh_client_build_hash/vdso", 'wb') as vdso_file:
                    vdso_file.write(vdso_data)

        shutil.make_archive("hash", 'zip', "apsh_client_build_hash")

        if os.path.exists("apsh_client_build_hash"):
            shutil.rmtree("apsh_client_build_hash")

    #   Symbols
    def create_symbols(self, args):
        # get linux distribution
        (out, err) = self.shell_cmd("cat /etc/os-release | grep ^ID=")
        dist = out.strip().split("=")[1].strip("\"")
        print("OS distribution = " + dist)

        # get kernel release
        kernel_release = os.uname()[2]

        if not os.path.isfile(args.path):
            print("Dwarf2Json not found. Please specify path to Dwarf2Json executable using --path")
            print("For download and compilation guide, please visit https://github.com/volatilityfoundation/dwarf2json")
            exit(-1)

        if dist.lower() == 'ubuntu':
            print("Recognized UbuntuOS - Starting....")
            ubuntu_vmlinux_file="/usr/lib/debug/boot/vmlinux-" + kernel_release
            ubuntu_system_map_file="/boot/System.map-" + kernel_release
            if os.path.isfile(ubuntu_vmlinux_file) and os.path.isfile(ubuntu_system_map_file):
                print("found current kernel " + kernel_release + " debug files...")
                print("executing dwarf2json...")
                cmd = "sudo " + args.path + " linux --elf " +ubuntu_vmlinux_file+ " --system-map "+ubuntu_system_map_file+" > ./symbols.json"
                (out, err) = self.shell_cmd(cmd)
            else:
                do_download = False
                valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
                question = 'debug symbols were not found, do you want to install debug symbols? (make sure to have enough space, or it might cause a problem with the package manager!)'
                prompt = " [y/N] "
                default = 'n'
                while True:
                    sys.stdout.write(question + prompt)
                    choice = input().strip().lower()
                    if choice == "":
                        choice = default
                    if choice in valid:
                        do_download = valid[choice]
                        break
                    else:
                        sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
                if do_download == False:
                    return
                (out, err) = self.shell_cmd("lsb_release -cs")
                lsb_release = out[:-1]
                should_backup = False
                backup_file = "/tmp/_backup_ddebs.list"
                original_file = "/etc/apt/sources.list.d/ddebs.list"
                if os.path.isfile(original_file):
                    should_backup = True
                    shutil.copyfile(original_file, backup_file)
                with open(original_file, 'w') as file:
                    file.write("deb http://ddebs.ubuntu.com " + lsb_release + " main restricted universe multiverse\n" +
                                "deb http://ddebs.ubuntu.com " + lsb_release + "-updates main restricted universe multiverse\n" +
                                "deb http://ddebs.ubuntu.com " + lsb_release + "-proposed main restricted universe multiverse\n")
                self.shell_cmd("sudo apt update && apt -y install ubuntu-dbgsym-keyring linux-image-"+kernel_release+"-dbgsym")
                os.remove(original_file)
                if should_backup == True:
                    os.rename(backup_file, original_file)
                if os.path.isfile(ubuntu_vmlinux_file) and os.path.isfile(ubuntu_system_map_file):
                    print("executing dwarf2json...")
                    cmd = "sudo " + args.path + " linux --elf " +ubuntu_vmlinux_file+ " --system-map "+ubuntu_system_map_file+" > ./symbols.json"
                    (out, err) = self.shell_cmd(cmd)
                else:
                    print("vmlinux and system map file not found. Please install ubuntu-dbgsym-keyring linux-image-"+kernel_release+"-dbgsym manually")

        if dist.lower() == 'centos':
            print("Recognized CentOS - Starting....")
            centos_vmlinux_file="/usr/lib/debug/lib/modules/"+kernel_release+"/vmlinux"
            centos_system_map_file="/boot/System.map-"+ kernel_release
            if os.path.isfile(centos_vmlinux_file) and os.path.isfile(centos_system_map_file):
                print("found current kernel " + kernel_release + " debug files...")
                print("executing dwarf2json...")
                cmd = "sudo " + args.path + " linux --elf " +centos_vmlinux_file+ " --system-map "+centos_system_map_file+" > ./symbols.json"
                (out, err) = self.shell_cmd(cmd)
            else:
                print("identified Centos system, installing debug symbols...")
                self.shell_cmd("sed -i 's/enabled=0/enabled=1/g' /etc/yum.repos.d/CentOS-Linux-Debuginfo.repo && yum -y install kernel-debuginfo")
                if os.path.isfile(centos_vmlinux_file) and os.path.isfile(centos_system_map_file):
                    print("executing dwarf2json...")
                    cmd = "sudo " + args.path + " linux --elf " +centos_vmlinux_file+ " --system-map "+centos_system_map_file+" > ./symbols.json"
                    (out, err) = self.shell_cmd(cmd)
                else:
                    print("vmlinux and system map file not found")

        print("cleaning and exit.")


    def create_mem_regions(self, args):
        (out, err) = self.shell_cmd("sudo less /proc/iomem | grep \"System RAM\"")
        dict1 = {}
        output_string = "{\n   \"allowed_regions\" : {"

        for i, line in enumerate(out.splitlines()):
            addresses = line.strip().split()[0]
            start = int(addresses.split('-')[0], 16)
            end = int(addresses.split('-')[1], 16)
            line =  "\n      \""+str(i)+"\" : {" \
                    +"\n         \"length\" : "+str(end - start)+"," \
                    +"\n         \"start\" : "+str(start) \
                    +"\n      },"
            output_string += line

        output_string += "\n   }\n}\n"
        if os.path.exists("mem_regions.json"):
            os.remove("mem_regions.json")
        with open("mem_regions.json", 'w') as file:
            file.write(output_string)


    def create_kpgd_file(self, args):
        if args.find_kpgd == '0':
            return

        if os.path.exists("kpgd_file.conf"):
            os.remove("kpgd_file.conf")
        va_length = 16
        kpgd_va = 0
        kpgd_pa = 0
        kpgd_pa_str = ""
        start_va = 0
        start_pa = 0
        init_task_va_str = ""
        cmd = 'cat /proc/kallsyms | grep "init_top_pgt"'
        (out, err) = self.shell_cmd(cmd)
        if out.find("init_top_pgt") != -1:
            kpgd_va_str = "0x" + out[0:va_length]
            kpgd_va = ast.literal_eval(kpgd_va_str)
        if kpgd_va == 0:
            cmd = 'cat /proc/kallsyms | grep "init_level4_pgt"'
            (out, err) = self.shell_cmd(cmd)
            if out.find("init_level4_pgt") != -1:
                kpgd_va_str = "0x" + out[0:va_length]
                kpgd_va = ast.literal_eval(kpgd_va_str)
        if kpgd_va == 0:
            cmd = 'cat /proc/kallsyms | grep "level4_kernel_pgt"'
            (out, err) = self.shell_cmd(cmd)
            if out.find("level4_kernel_pgt") != -1:
                kpgd_va_str = "0x" + out[0:va_length]
                kpgd_va = ast.literal_eval(kpgd_va_str)
        if kpgd_va == 0:
            cmd = 'cat /proc/kallsyms | grep "swapper_pg_dir"'
            (out, err) = self.shell_cmd(cmd)
            if out.find("swapper_pg_dir") != -1:
                kpgd_va_str = "0x" + out[0:va_length]
                kpgd_va = ast.literal_eval(kpgd_va_str)
        cmd = 'cat /proc/kallsyms | grep "_stext"'
        (out, err) = self.shell_cmd(cmd)
        if out.find("_stext") != -1:
            start_va_str = "0x" + out[0:va_length]
            start_va = ast.literal_eval(start_va_str)
        cmd = 'cat /proc/kallsyms | grep " init_task"'
        (out, err) = self.shell_cmd(cmd)
        if out.find("init_task") != -1:
            init_task_va_str = out[0:va_length]
        if init_task_va_str == "" or kpgd_va == 0 or start_va == 0:
            print("creating kpgd_failed: couldn't find all needed symbols")
            return
        cmd = 'cat /proc/iomem | grep "Kernel code"'
        (out, err) = self.shell_cmd(cmd)
        if out.find("Kernel code") != -1:
            out = out.strip()
            start_pa_str = "0x" + out[0:out.find("-")]
            start_pa = ast.literal_eval(start_pa_str)
        if start_pa == 0:
            print("creating kpgd_failed: couldn't find PA of _stext")
            return
        kpgd_pa = kpgd_va - (start_va - start_pa)
        kpgd_pa_str = hex(kpgd_pa)
        kpgd_pa_str = kpgd_pa_str[2:]
        kpgd_f = open("kpgd_file.conf", "w")
        kpgd_f.write(kpgd_pa_str)
        kpgd_f.write("\n")
        kpgd_f.write(init_task_va_str)
        kpgd_f.write("\n")
        kpgd_f.close()


    def __init__(self, args):
        for func in args.files:
            getattr(self, func_map_linux[func])(args)

class WindowsApsh():

    def rmtree(self,top):
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                filename = os.path.join(root, name)
                os.chmod(filename, stat.S_IWUSR)
                os.remove(filename)
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(top)

    # hash_build run
    def run_hash_build(self, args):
        if not args.pid:
            return

        if os.path.exists("hash.zip"):
            os.remove("hash.zip")

        if os.path.exists("apsh_client_build_hash"):
            self.rmtree("apsh_client_build_hash")

        os.mkdir("apsh_client_build_hash")
        irregular_files = open("apsh_client_build_hash/doca_apsh_irregular_files.txt", "w")

        for map_file in psutil.Process(args.pid).memory_maps():
            new_name = list(map_file.path)[3:]
            for i, c_str in enumerate(new_name):
                if c_str in ['\\', ':']:
                    new_name[i] = '_'
            new_name = 'apsh_client_build_hash/'+ "".join(new_name)
            if (os.path.isfile(map_file.path)):
                shutil.copy(map_file.path, new_name)
            else:
                # Add "#" to the beginning of the line to distinuish it.
                irregular_files.write("#" + map_file.path[map_file.path.rfind("/") + 1 :] + "\n")
        irregular_files.close()

        shutil.make_archive("hash", 'zip', "apsh_client_build_hash")

        if os.path.exists("apsh_client_build_hash"):
            self.rmtree("apsh_client_build_hash")

    MAX_MEM_REGIONS = 20

    CM_RESOURCE_MEMORY_LARGE = 0x0E00
    CM_RESOURCE_MEMORY_LARGE_40 = 0x0200
    CM_RESOURCE_MEMORY_LARGE_48 = 0x0400
    CM_RESOURCE_MEMORY_LARGE_64 =  0x0800

    CmResourceTypeNull = 0 # ResType_All or ResType_None (0x0000)
    CmResourceTypePort = 1 # ResType_IO (0x0002)
    CmResourceTypeInterrupt = 2 # ResType_IRQ (0x0004)
    CmResourceTypeMemory = 3 # ResType_Mem (0x0001)
    CmResourceTypeDma = 4 # ResType_DMA (0x0003)
    CmResourceTypeDeviceSpecific = 5 # ResType_ClassSpecific (0xFFFF)
    CmResourceTypeBusNumber = 6 # ResType_BusNumber (0x0006)
    CmResourceTypeMemoryLarge = 7 # ResType_MemLarge (0x0007)
    CmResourceTypeNonArbitrated = 128 # Not arbitrated if 0x80 bit set
    CmResourceTypeConfigData = 128 # ResType_Reserved (0x8000)
    CmResourceTypeDevicePrivate = 129 # ResType_DevicePrivate (0x8001)
    CmResourceTypePcCardConfig = 130 # ResType_PcCardConfig (0x8002)
    CmResourceTypeMfCardConfig = 131 # ResType_MfCardConfig (0x8003)
    CmResourceTypeConnection = 132 # ResType_Connection (0x8004)

    def parse_memory_map():
        winreg = importlib.import_module('winreg')
        pszSubKey = "Hardware\\ResourceMap\\System Resources\\Physical Memory"
        pszValueName = ".Translated"
        key = winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, pszSubKey)
        dw = winreg.QueryValueEx(key, pszValueName)
        regions = []
        for x in range(20, len(dw[0]), 20):
            start = int.from_bytes(dw[0][x+4:x+12], byteorder='little')
            length = int.from_bytes(dw[0][x+12:x+20], byteorder='little')
            flag = int.from_bytes(dw[0][x+2:x+4], byteorder='big')
            if flag==2: length*=256
            if flag==4: length*=256**2
            if flag==8: length*=256**4
            regions.append({
                "start": start,
                "length": length
            })
        winreg.CloseKey(key)
        return regions


    def create_mem_regions(self, args):
        regions = WindowsApsh.parse_memory_map()
        json = {"allowed_regions": {}}
        for i,r in enumerate(regions):
            json["allowed_regions"][str(i)] = r
        if os.path.exists("mem_regions.json"):
            os.remove("mem_regions.json")
        with open("mem_regions.json", 'w') as file:
            file.write(str(json))

    def get_guid():
        pe = PE("C:\\Windows\\System32\\ntoskrnl.exe")
        debug = pe.DIRECTORY_ENTRY_DEBUG[0].entry
        guid = "{0:08X}{1:04X}{2:04X}{3}{4}".format(debug.Signature_Data1,
        debug.Signature_Data2,
        debug.Signature_Data3,
        binascii.hexlify(debug.Signature_Data4).decode("utf-8"),
        debug.Age).lower()
        return guid

    logger = logging.getLogger(__name__)
    logger.setLevel(1)

    def load_pdbparse(self, args):
        if os.path.isfile(args.path):
            file_directory, file_name = os.path.split(args.path)
            sys.path.append(file_directory)
            return importlib.import_module(file_name.split('.')[0])
        else:
            do_download = False
            valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
            question = 'pdbparse-to-json.py not found.\ndo you want to download?'
            prompt = " [y/N] "
            default = 'n'
            while True:
                sys.stdout.write(question + prompt)
                choice = input().lower()
                if choice == "":
                    do_download = valid[default]
                    break
                elif choice in valid:
                    do_download = valid[choice]
                    break
                else:
                    sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
            if do_download:
                import urllib.request
                pdbparse_url = 'https://raw.githubusercontent.com/volatilityfoundation/volatility3/stable/development/pdbparse-to-json.py'
                urllib.request.urlretrieve(pdbparse_url, "pdbparse-to-json.py")
                sys.path.append(os. getcwd())
                return importlib.import_module('pdbparse-to-json')

    def create_symbols(self, args):
        pdbparse = self.load_pdbparse(args)
        filename = pdbparse.PDBRetreiver().retreive_pdb(WindowsApsh.get_guid(), "ntkrnlmp.pdb")

        delfile = True
        if not filename:
            parser.error("No suitable filename provided or retrieved")
        convertor = pdbparse.PDBConvertor(filename)
        newfilename = "symbols.json"
        p = Path(newfilename)
        Path(p.parent).mkdir(parents=True, exist_ok=True)
        with open(newfilename, "w") as f:
            json.dump(convertor.read_pdb(), f, indent = 2, sort_keys = True)
        print("Temporary PDB file: {}".format(newfilename))

    def __init__(self, args):
        for func in args.files:
            if func != "kpgd_file":
                getattr(self, func_map_windows[func])(args)

def is_pefile_valid():
    latest_valid_version = "2022.5.30"
    current_version = pefile.__version__
    date_valid = datetime.date(*map(int, latest_valid_version.split('.')))
    date_current = datetime.date(*map(int, current_version.split('.')))
    return (date_current <= date_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--os", type=str, help='os to run on - windows or linux', choices=['windows', 'linux'], required=True)
    func_map_linux = {'hash': "run_hash_build",
            'symbols': "create_symbols",
            'memregions': "create_mem_regions",
            'kpgd_file': "create_kpgd_file"
            }
    func_map_windows = {'hash': "run_hash_build",
            'symbols': "create_symbols",
            'memregions': "create_mem_regions",
            }
    func_map_default = {'hash': "run_hash_build",
            'symbols': "create_symbols",
            'memregions': "create_mem_regions",
            }
    default_dwarf2json_path = './dwarf2json'
    default_pdbparse_path = './pdbparse-to-json.py'

    parser.add_argument("--pid", type=int, help='pid of the process to hash (not mandatory, unless file "hash" is needed)', default=0)
    parser.add_argument('--files', nargs='+', help='list of files to create.\n' +
        'Available files for Linux: ' + ', '.join(list(func_map_linux.keys())) + ' (defualt is all).\n' +
        'Available files for Windows: ' + ', '.join(list(func_map_windows.keys())) + ' (defualt is all).\n' +
        '* Important Note: declaring files to create without supplying the appropriate flag will not create the files.\n' +
        '(for example "python3 doca_apsh_config.py --os linux --files kpgd_file" will not create kpgd_file unless "--find_kpgd=1" is added)\n\n', default=list(func_map_default.keys()))
    parser.add_argument('--path', help="path to Dwarf2Json executble (for linux os) or pdbparse-to-json.py (for windows).\n\n" +
        "Dwarf2Json - download and compilation guide can be found at:\nhttps://github.com/volatilityfoundation/dwarf2json\n\n" +
        "pdbparse-to-json.py - can be found at:\nhttps://raw.githubusercontent.com/volatilityfoundation/volatility3/stable/development/pdbparse-to-json.py")
    parser.add_argument("--find_kpgd", type=str, choices=['0', '1'], help='enable finding KPGD (creating "kpgd_file") on the host (relevant only to linux)', default='0')
    args = parser.parse_args()
    if ((not ('kpgd_file' in args.files)) and args.find_kpgd == '1'):
        args.files += ['kpgd_file']
    if (('kpgd_file' in args.files) and args.find_kpgd != '1'):
        print("can't create kpgd_file when find_kpgd=0")
        args.files.remove('kpgd_file')
    if (('hash' in args.files) and (not args.pid)):
        print("can't create hash file when no pid was provided or pid=0")
        args.files.remove('hash')
    print('creating:' + str(args.files))
    args = parser.parse_args()
    if args.os == "linux":
        if not args.path:
            args.path = default_dwarf2json_path
        LinuxApsh(args)
    elif args.os == "windows":
        if not is_pefile_valid():
            print("pefile module has a version higher than 2022.5.30. Try downgrading it to 2022.5.30")
            sys.exit(0)
        if not args.path:
            args.path = default_pdbparse_path
        if args.find_kpgd == '1':
            print("finding kpgd isn't supported on windows, ignoring it")
        WindowsApsh(args)
    else:
        print("unsupported os")
