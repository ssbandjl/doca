#
# Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of Nvidia Corporation and its affiliates
# (the "Company") and all right, title, and interest in and to the software
# product, including all associated intellectual property rights, are and
# shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

#
# mst_ib_add :
# Parse the given lst file and print the device names for
# mst inband access
#
from __future__ import print_function
import sys
import argparse
import re
import collections
import traceback
import csv


class _MstFileParser(object):

    MELLANOX_VEND_ID = "0x2c9"
    UNSUPPORTED_DEV_IDS = ['0xbd34', '0xbd35', '0xbd36', '0xfa66', '0xfa7a', '0x1003', '0x1007',
                           '48436', '48437', '48438', '64102', '64122', '4099', '4103']

    def __init__(self,
                 lst_file,
                 all_or_only_mlnx,
                 file_type,
                 ibdr,
                 hca_id,
                 exit_port_num,
                 print_with_guids=False,
                 aggregated=False):
        self._lst_file = lst_file
        self._all_or_only_mlnx = all_or_only_mlnx
        self._file_type = file_type
        self._ibdr = ibdr
        self._hca_id = hca_id
        self._exit_port_num = exit_port_num
        self._print_with_guids = print_with_guids
        self._aggregated = aggregated
        self._ib_node = collections.namedtuple("ib_node",
                                               ['node_guid', 'port_guid', 'lid', 'desc', 'type', 'dev_id', 'vend_id', 'dr_id'])
        self._ib_db = []

    def _add_nodes(self, node_data, dr_pattern=''):
        """
        Parse the node data and insert it to the IB DB
        """
        raise NotImplementedError("add_nodes() is not implemented")

    def parse_file(self):
        """
        Parse the file in order to extract the nodes data and send it for
        node parsing and db inserting
        """
        raise NotImplementedError("parse_file() is not implemented")

    def _is_ibdr_unsupported(self, dev_id):
        return str(dev_id) in self.UNSUPPORTED_DEV_IDS

    def _is_node_exist_in_db(self, node_guid):
        node_exist = False

        # check if given node guid is already inside the IB db
        # if found, stop searching and return the right indication
        for data in self._ib_db:
            if node_guid == data.node_guid:
                node_exist = True
                break

        return node_exist

    def _is_node_ignored(self, vend_id):
        # flag for the returned value
        is_ignored = False

        # if the given argument is only mlnx and the vend id is defined for ignore
        # need to return indication for ignoring
        if self._all_or_only_mlnx != "all" and self.MELLANOX_VEND_ID != vend_id:
            is_ignored = True

        return is_ignored

    def retrieve_all_mst_ib(self):
        """
        retrieve all the parsed ib data that added to the ib db
        :return: list of strings
        """
        # suffix to the end of the output
        suffix = ""
        # returned list of all the ib's parsed and added to the db
        output_list = []

        # set the suffix only if the user insert the hca and exit port arguments
        if self._hca_id != "":
            suffix += "," + self._hca_id
            if self._exit_port_num != "0":
                suffix += "," + self._exit_port_num

        # for each line in db, set the needed values at the output string in the right format
        for line in self._ib_db:
            # ded id will be converter to decimal
            output_str = line.type + "_MT" + str(int(line.dev_id, 16)) + "_"

            # adjust the description
            desc = str(line.desc)
            desc = re.sub(' ', '_', desc)           # remove all spaces
            if re.findall('^MF0;', desc):
                # remove the 'MFO;' from string
                desc = re.sub('^MF0;', '', desc)
                # remove the string from ':' to '/'
                desc = re.sub(':[^/]*', '', desc)
                desc = re.sub('/', '_', desc)       # change '/' to '_'
                desc = re.sub('_U1', '', desc)      # remove '_UI'
            output_str += desc + "_"

            # add the lid
            if self._ibdr and line.dr_id and not self._is_ibdr_unsupported(line.dr_id):
                output_str += "ibdr-" + line.dr_id
            else:
                output_str += "lid-" + line.lid

            # make sure we don't have bad chars in the dev name
            output_str = re.sub('[:;/]', '_', output_str)

            # if the 'print with guid' is true, need to add the port guid
            output_str += suffix
            if self._print_with_guids:
                output_str += "#" + line.port_guid
            if output_str not in output_list:
                output_list.append(output_str)

        return output_list


class _DiagnetParser(_MstFileParser):

    def _add_nodes(self, node_data, dr_pattern=''):
        # split for all lines in node data:
        lines = node_data.split(" }")

        # re return a list (in that case this method will refer to [0] index of each parsed parameter)
        for ib_data in lines:
            # parse the Node Guid
            node_guid = re.compile("NodeGUID:(\\S*)\\s").findall(ib_data)
            # parse the port Guid
            port_guid = re.compile("PortGUID:(\\S*)\\s").findall(ib_data)
            # parse the LID
            lid = re.compile("LID:([0-9a-f]*)\\s").findall(ib_data)
            # parse the description
            desc = re.compile("{(\\S[^}]*)}").findall(ib_data)
            # parse the Type
            type_ib = re.compile("{\\s(\\S*)").findall(ib_data)
            # parse the Dev Id
            dev_id = re.compile("DevID:(\\S*)\\s").findall(ib_data)
            # parse the Ven Id
            vend_id = re.compile("VenID:(\\S*)\\s").findall(ib_data)

            if node_guid != [] or \
                    port_guid != [] or \
                    lid != [] or \
                    desc != [] or \
                    type_ib != [] or \
                    dev_id != [] or \
                    vend_id != []:

                # adjust port data
                vend_id[0] = "0x" + vend_id[0].lstrip("0")
                dev_id[0] = "0x" + dev_id[0].lstrip("0")
                lid[0] = "{0:#0{1}x}".format(int(lid[0], 16), 6)

                if type_ib[0] == "CA-SM":
                    type_ib[0] = "CA"
                elif type_ib[0] == "SW-SM":
                    type_ib[0] = "SW"

                if self._is_node_ignored(vend_id[0]) or self._is_node_exist_in_db(node_guid):
                    continue

                if dev_id[0] != "43132":
                    if not desc:
                        desc.append('')
                    ib_node = self._ib_node(node_guid[0], port_guid[0], lid[0], desc[0],
                                            type_ib[0], dev_id[0], vend_id[0], "")
                    self._ib_db.append(ib_node)

    def parse_file(self):
        if self._aggregated:
            self.parse_aggregated_file()
        else:
            # flag for the returned value
            active_links_found = False
            # check if line has an active link and send (each line) for parsing and inserting to DB
            try:
                with open(self._lst_file, 'r') as f:
                    for line in f:
                        if "LOG=ACT" in line:
                            active_links_found = True
                            self._add_nodes(line)
            except Exception as e:
                raise RuntimeError(e)

            return active_links_found

    def create_mapping(csv_file_path, string_start_section, string_end_section, column_a, column_b, columns_mapping):
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)

            # Skip the header row if it exists
            next(csv_reader, None)

            found_string = False
            found_columns = False
            for row in csv_reader:
                if found_string:
                    if column_a in row and column_b in row:
                        found_columns = True
                        column_a_offset = row.index(column_a)
                        column_b_offset = row.index(column_b)
                        print(column_a_offset)
                    else:
                        if string_end_section not in row and found_columns:
                            value_a = row[column_a_offset]
                            value_b = row[column_b_offset]
                            if value_a not in columns_mapping:
                                columns_mapping[value_a] = []
                            columns_mapping[value_a].append(value_b)
                            print(value_a, ' ', column_b, ' ', value_b)
                        elif string_end_section in row:
                            found_string = False

                elif string_start_section in row:
                    found_string = True
        return columns_mapping

    def parse_aggregated_file(self):

        mapping = {}
        mapping = self.create_mapping(self._lst_file, 'START_NODES', 'END_NODES', 'PortGUID', 'SystemImageGUID', mapping)
        mapping = self.create_mapping(self._lst_file, 'START_PORTS', 'END_PORTS', 'PortGuid', 'LID', mapping)
        mapping = self.create_mapping(self._lst_file, 'START_PORT_HIERARCHY_INFO', 'END_PORT_HIERARCHY_INFO', 'PortGUID', 'Cage', mapping)
        mapping = self.create_mapping(self._lst_file, 'START_PORT_HIERARCHY_INFO', 'END_PORT_HIERARCHY_INFO', 'PortGUID', 'Port', mapping)
        mapping = self.create_mapping(self._lst_file, 'START_PORT_HIERARCHY_INFO', 'END_PORT_HIERARCHY_INFO', 'PortGUID', 'Split', mapping)
        mapping = self.create_mapping(self._lst_file, 'START_PORT_HIERARCHY_INFO', 'END_PORT_HIERARCHY_INFO', 'PortGUID', 'IsCageManager', mapping)
        mapping = self.create_mapping(self._lst_file, 'START_PORT_HIERARCHY_INFO', 'END_PORT_HIERARCHY_INFO', 'PortGUID', 'IBPort', mapping)

        ib_node_set = collections.namedtuple("port",
                                             ['SystemImageGUID', 'PortGUID', 'LID', 'Cage', 'Port', 'Split', 'IsCageManager', 'IBPort'])
        ib_db = []
        for port in mapping.items():
            if len(port[1]) > 5:
                ib_node = ib_node_set(port[1][0], port[0], port[1][1], port[1][2],
                                      port[1][3], port[1][4], port[1][5], "")

                ib_db.append(ib_node)

        return ib_db


class _NetParser(_MstFileParser):

    def _add_nodes(self, node_data, dr_pattern=''):
        # split to lines
        lines = node_data.split("\n")
        line_index = 0

        # re return a list (in that case this method will refer to [0] index of each parsed parameter)
        # Get vendor ID
        vend_id = re.compile("vendid=(\\S*)").findall(lines[line_index])
        line_index += 1
        if vend_id is None:
            raise Exception(self._lst_file + ":" + str(line_index) +
                            "Bad node format - Missing VendorId: " + lines[line_index])

        # Get Device ID
        dev_id = re.compile("devid=(\\S*)").findall(lines[line_index])
        line_index += 1
        if dev_id is None:
            raise Exception(self._lst_file + ":" + str(line_index) +
                            "Bad node format - Missing Device ID: " + lines[line_index])

        # Get node guid
        node_guid = re.compile(
            "(?:switch|ca)guid=([a-z0-9]*)").findall(lines[line_index])
        line_index += 1
        if not node_guid:
            node_guid = re.compile(
                "(?:switch|ca)guid=([a-z0-9]*)").findall(lines[line_index])
            line_index += 1

        if node_guid is None:
            raise Exception(self._lst_file + ":" + str(line_index) +
                            "Bad node format - Missing NodeGuid: " + lines[line_index])

        # Get type
        type_ib = re.compile("(Ca|Switch)").findall(lines[line_index])
        if type_ib is None:
            raise Exception(self._lst_file + ":" + str(line_index) +
                            "Bad node format - Missing type: " + lines[line_index])

        # description - will be parsed according ib type
        desc = []
        # lid - will be parsed according ib type
        lid = []

        # get description an lid according ib type
        if type_ib is not None:
            if type_ib[0] == "Switch":
                type_ib[0] = "SW"
                desc = re.compile("\\# \"([^\\s]+)").findall(lines[line_index])
                lid = re.compile(
                    "Switch.*port 0 lid ([0-9]+)").findall(lines[line_index])
                if lid == []:
                    raise Exception(self._lst_file + ":" + str(line_index) +
                                    ": Bad node format - Missing lid: " + lines[line_index])
                line_index += 1
            else:  # it's CA
                type_ib[0] = "CA"
                desc = re.compile(
                    "(?:\\#\\s\")([^\"]*)").findall(lines[line_index])
                line_index += 1
                lid = re.compile(
                    "(?:\\#\\slid\\s)([0-9a-f]*)").findall(lines[line_index])
                if lid == []:
                    raise Exception(self._lst_file + ":" + str(line_index + 1) +
                                    ": Bad ibnetdiscover output format (Last node): Failed to get first lid: Maybe the port you scan is down")
            # set the lid string to be as hex number of 4 digits format
            if lid[0] == "0":
                raise Exception("-W- Subnet Manager is not active")
            lid[0] = "{0:#0{1}x}".format(int(lid[0]), 6)

        # if parse succeed and pass the validation insert to ib db
        if vend_id != [] or \
                dev_id != [] or \
                node_guid != [] or \
                type_ib != [] or \
                desc != [] or \
                lid != []:

            # check if some of the node parameters valid for db inserting
            if self._is_node_ignored(vend_id[0]) or \
                    dev_id == "43132" or \
                    self._is_ibdr_unsupported(dev_id) or \
                    self._is_node_exist_in_db(node_guid):
                return
            else:
                # create s db row and insert it to the ib db
                ib_node = self._ib_node(
                    node_guid[0], node_guid[0], lid[0], desc[0], type_ib[0], dev_id[0], vend_id[0], dr_pattern)
                self._ib_db.append(ib_node)

    def parse_file(self):
        # flag for the returned value
        active_links_found = False
        dr_pattern = ''
        # check if line has an active link (something to parse) and send (each section) for parsing and inserting to DB
        try:
            with open(self._lst_file, 'r') as f:
                data = f.read()
                dr_list = []
                # split to lines
                lines = data.split("\n")
                for line in lines:
                    pattern = "DR path slid 0; dlid 0; "
                    if pattern in line:
                        dr = line.index(pattern) + len(pattern)
                        dr_right = dr
                        while (line[dr_right] != ' '):
                            dr_right = dr_right + 1

                        dr_pattern = line[dr:dr_right]

                        if ',' in dr_pattern:
                            dr_pattern = dr_pattern.replace(",", ".")

                        dr_list.append(dr_pattern)

                sections = re.split('(?m)^\\n', data)
                dr_counter = 0
                for s in sections:
                    if "vendid" in s:
                        if len(dr_list) > 0:
                            self._add_nodes(
                                s, dr_list[min(dr_counter, len(dr_list) - 1)])
                            dr_counter = dr_counter + 1
                        else:
                            self._add_nodes(s)
                        active_links_found = True

        except Exception as ex:
            print(ex.args[0])
            return active_links_found

        return active_links_found


#######################################################################################################################


class MstIbAdd(object):

    def __init__(self, lst_file,
                 all_or_only_mlnx,
                 file_type,
                 ibdr,
                 hca_id,
                 exit_port_num,
                 print_with_guids=False,
                 aggregated=False):

        if file_type == "diagnet":
            self._parser = _DiagnetParser(lst_file, all_or_only_mlnx, file_type, ibdr, hca_id, exit_port_num,
                                          print_with_guids)#, aggregated)
        else:
            self._parser = _NetParser(lst_file, all_or_only_mlnx, file_type, ibdr, hca_id, exit_port_num,
                                      print_with_guids)

    def ib_add(self):
        """
        Parse the given lst file and return the device names for
        mst inband access as a list
        """
        # parse file (add each ib to db)
        is_active_links = self._parser.parse_file()

        # error message in case file content don't specify an active links
        # if not is_active_links:
        #    sys.stderr.write(
        #        "-W- Non active links found in the IB fabric. These links are not added to the device list. "
        #        "Make sure that a subnet-manager is running.\n")

        return self._parser.retrieve_all_mst_ib()
#######################################################################################################################


def main():
    parser = argparse.ArgumentParser(description='MST IB ADD')

    parser.add_argument('lst_file',
                        help='<lst_file>')
    parser.add_argument('all_or_only_mlnx',
                        choices=['only_mlnx', 'all'],
                        help='only mellanox or all')
    parser.add_argument('file_type',
                        help='file type')
    parser.add_argument('ibdr',
                        choices=['0', '1'],
                        help='ibdr 0|1')
    parser.add_argument('hca_id',
                        nargs="?", type=str, default="",
                        help='hca id')
    parser.add_argument('exit_port_num',
                        nargs="?", type=str, default="",
                        help='exit port number]')
    parser.add_argument('--with-guids', action='store_true',
                        default=False, dest='print_with_guids',
                        help='print with guids')
    parser.add_argument('aggregated',
                        default=False,
                        help='discovery for aggregated port')

    args = parser.parse_args()

    mst_ib_add = MstIbAdd(lst_file=args.lst_file,
                          all_or_only_mlnx=args.all_or_only_mlnx,
                          file_type=args.file_type,
                          ibdr=args.ibdr,
                          hca_id=args.hca_id,
                          exit_port_num=args.exit_port_num,
                          print_with_guids=args.print_with_guids,
                          aggregated=args.aggregated)

    ib_list = mst_ib_add.ib_add()

    for ib in ib_list:
        print(ib)


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        sys.stderr.write("-E- {0}\n".format(str(e)))
        sys.exit(1)
