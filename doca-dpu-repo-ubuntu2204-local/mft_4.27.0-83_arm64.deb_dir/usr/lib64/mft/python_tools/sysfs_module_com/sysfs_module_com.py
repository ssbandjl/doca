#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of Nvidia Corporation and its affiliates
# (the "Company") and all right, title, and interest in and to the software
# product, including all associated intellectual property rights, are and
# shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

import os
import sys
import argparse
import binascii

if sys.version_info[0] < 3:
    print("Error - please install python 3 or higher!\n\
          Please make sure python3 is configured correctly.")
    sys.exit(1)

SYSFS_MAP = {
    "IEI": {
        "MODULE_ACCESS_INFO":
            {
                "SYSFS_ROOT": "/sys_switch/transceiver/",
                "MODULE_NAME_PREFIX": "eth"
            }
    }
}

SYSFS_FILE_NAME = "eeprom"
PAGE_LENGTH = 0x80
MIN_NON_ZERO_PAGE_OFFSET = 128
MAX_PAGE_OFFSET = 255


class SysfsError(Exception):
    pass


class Sysfs:
    args = None

    def __init__(self):
        self.args = self.get_args()

    @classmethod
    def get_args(cls):
        parser = argparse.ArgumentParser(description="Discover & Communicate with sysfs files",
                                         prog="sysfs_module_com.py",
                                         add_help=True)
        parser.add_argument('--discover', action='store_true',
                            help='Present the modules per customer.')
        parser.add_argument('--read', nargs='*', metavar='<page><offset><length>',
                            help='Read a value or a range of values from a specific page.')
        parser.add_argument('--write', nargs='*', metavar='<page><offset><space_separated_bytes>',
                            help='Write a value or a range of values from a specific page.')
        parser.add_argument('--dump', action='store_true',
                            help='Present all pages and values of a specific module.')
        parser.add_argument('--module', nargs=1, metavar='<sysfs_module_com_format>', help='Specify the module '
                                                                                           'you want to read/write from/to.')
        return parser.parse_args()

    @classmethod
    def discover_modules(cls):
        """
        This function finds all the available modules in the current setup and prints them to the user.
        Each line will be as follows: <costumer_idd>_module_<name_of_the_module>
        """
        for customer_id, item in SYSFS_MAP.items():
            customer_module_prefix = customer_id.lower() + '_module_'
            sysfs_root = str(item.get("MODULE_ACCESS_INFO").get("SYSFS_ROOT"))
            module_name_prefix = item.get("MODULE_ACCESS_INFO").get("MODULE_NAME_PREFIX")
            if not os.path.isdir(sysfs_root):
                raise SysfsError(
                    "Error - sysroot {} for customer_id {} does not exist!".format(sysfs_root, customer_id))

            dir_lst = [directory for directory in os.listdir(sysfs_root) if directory.startswith(module_name_prefix)]
            for file_name in dir_lst:
                print(customer_module_prefix + file_name)

    @staticmethod
    def load_eeprom(file_path: str):
        """
        This function loads the pages' file to a db.
        :param file_path: the path to the sysfs file compatible for the customer_id.
        :return: a formatted db.
        """
        if not os.path.isfile(file_path):
            raise SysfsError("Error - the file {} does not exist!".format(file_path))

        with open(file_path, 'rb') as module_file:
            db = []
            for chunk in iter(lambda: module_file.read(1), b''):
                x = binascii.hexlify(chunk)
                db.append(x.decode(encoding='UTF-8', errors='ignore'))
            return db

    @staticmethod
    def calculate_seek_address(page: int, offset: int):
        if page == 0:
            return offset
        if page > 0:
            return 256 + ((page - 1) * MIN_NON_ZERO_PAGE_OFFSET) + offset - MIN_NON_ZERO_PAGE_OFFSET
        raise SysfsError("Error - page cannot be negative value!")

    @classmethod
    def read_from_module(cls, sysfs_path: str, args_lst: list):
        """
        This function reads a number of addresses from the pages' file and prints their values.
        :param sysfs_path: the path to the relevant sysfs file for the required module.
        :param args_lst: a list holding the page, offset and length of which we want to read.
        """
        if len(args_lst) < 3:
            raise SysfsError("Error - '--read' command expects 3 arguments, {} provided!".format(len(args_lst)))
        required_page = int(args_lst[0], 0)
        required_offset = int(args_lst[1])
        required_length = int(args_lst[2])

        if required_offset + required_length > MAX_PAGE_OFFSET or (
                required_page != 0 and not (MIN_NON_ZERO_PAGE_OFFSET <= required_offset <= MAX_PAGE_OFFSET)):
            raise SysfsError("Error - out of page read attempt! page: {} offset: {} length: {}"
                             .format(required_page, required_offset, required_length))

        seek_address = cls.calculate_seek_address(required_page, required_offset)

        db = cls.load_eeprom(sysfs_path)
        if seek_address >= len(db):
            raise SysfsError(
                "Error - page \"{}\" does not appear in the file \"{}\"".format(hex(required_page), sysfs_path))

        output_lst = []
        for i in range(required_length):
            output_lst.append(db[seek_address + i])
        print(' '.join(output_lst))

    @classmethod
    def write_to_module(cls, sysfs_path: str, args_lst: list):
        """
        This function writes a number of addresses to the pages' file.
        :param sysfs_path: the path to the relevant sysfs file for the required module.
        :param args_lst: a list holding the page, offset and the values of which we want to write.
        """
        if len(args_lst) < 3:
            raise SysfsError(
                "Error - '--write' command expects at least 3 arguments, {} provided!".format(len(args_lst)))
        required_page = int(args_lst[0], 0)
        required_offset = int(args_lst[1])
        values_to_write = args_lst[2:]

        if required_offset + len(values_to_write) > MAX_PAGE_OFFSET or (
                required_page != 0 and not (MIN_NON_ZERO_PAGE_OFFSET <= required_offset <= MAX_PAGE_OFFSET)):
            raise SysfsError("Error - out of page write attempt! page: {} offset: {} length: {}"
                             .format(required_page, required_offset, len(values_to_write)))

        seek_address = cls.calculate_seek_address(required_page, required_offset)

        db = cls.load_eeprom(sysfs_path)
        if seek_address >= len(db):
            raise SysfsError(
                "Error - page \"{}\" does not appear in the file \"{}\"".format(hex(required_page), sysfs_path))

        for addr in values_to_write:
            db[seek_address] = '{0:02x}'.format(int(addr, 0))
            seek_address += 1

        db_to_write = binascii.unhexlify(''.join(db))
        with open(sysfs_path, 'wb') as module_file:
            module_file.write(db_to_write)

    @classmethod
    def dump_pages(cls, sysfs_path: str):
        """
        This function prints the entire pages' file according to the format.
        :param sysfs_path: the path to the relevant sysfs file for the required module.
        """
        db = cls.load_eeprom(sysfs_path)
        current_page = 0
        current_byte = 0
        current_offset = 0
        print('Page: {}, Offset: {:03}, Length: 0x80'.format(hex(current_page), current_offset))
        for i in range(0, len(db), 4):
            print('{:03}: {} {} {} {}'.format(current_byte, db[i], db[i + 1], db[i + 2], db[i + 3]))
            if i == (len(db) - 4):
                break
            current_byte += 4
            if current_page != 0:
                if (i + 4) % MIN_NON_ZERO_PAGE_OFFSET == 0:
                    current_page += 1
                    print('\nPage: {}, Offset: {:03}, Length: 0x80'.format(hex(current_page), current_offset))
                    current_byte = MIN_NON_ZERO_PAGE_OFFSET
            else:
                if (i + 4) % 256 == 0:
                    current_page += 1
                    print('\nPage: {}, Offset: {:03}, Length: 0x80'.format(hex(current_page), current_offset))
                    current_byte = MIN_NON_ZERO_PAGE_OFFSET
                elif (i + 4) % MIN_NON_ZERO_PAGE_OFFSET == 0:
                    current_offset = MIN_NON_ZERO_PAGE_OFFSET
                    print('\nPage: {}, Offset: {:03}, Length: 0x80'.format(hex(current_page), current_offset))


if __name__ == "__main__":
    try:
        args = Sysfs.get_args()
        if args.discover:
            Sysfs.discover_modules()
            exit(0)
        elif not args.module:
            raise SysfsError("Error - module must be provided when using read/write/dump command!")

        module = ''.join(args.module)
        module = module.split('_')
        if len(module) != 3:
            raise SysfsError("Error - invalid module!")
        customer_id = module[0].upper()
        module_name = module[2]
        sysfs_path = SYSFS_MAP.get(customer_id).get("MODULE_ACCESS_INFO").get(
            "SYSFS_ROOT") + os.sep + module_name + os.sep + SYSFS_FILE_NAME

        if args.read:
            Sysfs.read_from_module(sysfs_path, args.read)
        elif args.write:
            Sysfs.write_to_module(sysfs_path, args.write)
        elif args.dump:
            Sysfs.dump_pages(sysfs_path)
        else:
            raise SysfsError("Error - invalid parameter provided!")

    except Exception as e:
        print(e)
        exit(-1)
