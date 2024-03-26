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


# built by Oren duer
# modifed by Alaa Barari abarari@asaltech.com

import sys
import os
import subprocess
import re
from optparse import OptionParser

try:
    import tools_version
except Exception as e:
    print("-E- could not import : %s" % str(e))
    sys.exit(1)

try:

    def signalHandler():
        print('\nYou pressed Ctrl+C!')
        sys.exit(1)

    def isLinux():
        linux = 0
        if 'uname' in dir(os):
            uname = os.uname()
            linux = uname[0] == 'Linux'
        return linux

    def isRoot():
        if isLinux() and os.geteuid() != 0:
            print("-E- Permission denied, you need to run this tool as root")
            sys.exit(1)

    isRoot()

    try:
        MFT_BIN_DIR = os.environ['MFT_BIN_DIR'] + os.sep
    except BaseException:
        MFT_BIN_DIR = ""

    MCRA_BIN = MFT_BIN_DIR + "mcra"
    WQDUMP_BIN = MFT_BIN_DIR + "wqdump"

    if sys.version_info < (2, 4):
        sys.stderr.write("-E- This tool requires python 2.4 or greater\n")
        sys.exit(0)

    parser = OptionParser(description="Displays the current multicast groups and flow steering rules configured in the device")
    parser.add_option("-d", "--dev", dest="dev", default='',
                      help="MST device to use, required")

    parser.add_option("-f", "--file", dest="file", default='',
                      help="MCG dump file to use (for debug), used as input and there is no need for device")

    parser.add_option("-p", "--params", dest="params", default='(64, 32768, 65536)',
                      help="Mcg params, \"(MCG_ENTRY_SIZE, HASH_TABLE_SIZE, MCG_TABLE_SIZE)\", default is (64, 32768, 65536)")

    parser.add_option("-q", "--quiet",
                      dest="quiet", default=False, action='store_true',
                      help="Do not print progress messages to stderr")

    parser.add_option("-v", "--version",
                      dest="printver", default=False, action='store_true',
                      help="Print tool version")

    parser.add_option("-c", "--hopcount",
                      dest="hopcount", default=False, action='store_true',
                      help="add hopCount column")

    parser.add_option("-a", "--advanced",
                      dest="advanced", default=False, action='store_true',
                      help="show all rules")

    (options, args) = parser.parse_args()

    if options.printver:
        tools_version.PrintVersionString("mlxmcg", "1.0.18")
        sys.exit(0)

    if options.dev == '' and options.file == '':
        print('-E- --dev argument required')
        sys.exit(1)

    if options.file != '' and options.dev != '':
        print('-E- Cant use device a long with -f, this is a debug feature that requires no device.')
        sys.exit(1)

    regs = {
        'p0_b0_mcg_size': {'addr': '0x44240.0:5'},
        'p0_b0_mc_hash_size_mc': {'addr': '0x44240.8:5'},
        'p0_b0_mc_hash_size_uc': {'addr': '0x44240.16:5'},
        'p0_b0_mcg_size_64B': {'addr': '0x44240.5:1'},
        'p1_b0_mcg_size': {'addr': '0x44260.0:5'},
        'p1_b0_mc_hash_size_mc': {'addr': '0x44260.8:5'},
        'p1_b0_mc_hash_size_uc': {'addr': '0x44260.16:5'},
        'p1_b0_mcg_size_64B': {'addr': '0x44260.5:1'},
        'log_mc_table_sz': {'addr': '0x1f388.17:5'},
        'cx3_original_hash_shift': {'addr': '0x44254.24:5'},
    }

    tests = [
        ['p0_b0_mcg_size', 'p1_b0_mcg_size'],
        ['p0_b0_mc_hash_size_mc', 'p1_b0_mc_hash_size_mc'],
        ['p0_b0_mc_hash_size_uc', 'p1_b0_mc_hash_size_uc'],
        ['p0_b0_mc_hash_size_mc', 'p0_b0_mc_hash_size_uc']]

    mreadcmd = MCRA_BIN + " " + options.dev
    def mread(addr):
        cmd = mreadcmd + ' ' + addr
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if process.wait() != 0:
            #print ("-E- Could not access device : %s, cmd : %s,  Error : %s" % (options.dev, cmd, process.stdout.read().strip()))
            print("-E- Could not access the device: \"%s\" by the command: \"%s\"" % (options.dev, cmd))
            sys.exit(1)
        str = process.stdout.read()
        try:
            return int(str, 16)
        except Exception as e:
            print("-E- Could not access the device: \"%s\" by the command: \"%s\"" % (options.dev, cmd))
            sys.exit(1)

    def calchashsize():
        hash_size = 1 << regs['p0_b0_mc_hash_size_mc']['val']
        if (hash_size < (1 << (regs['cx3_original_hash_shift']['val'] + 1))):
            hash_size = 1 << (regs['cx3_original_hash_shift']['val'] + 1)
        return hash_size

    def dumpDevice():
        global MCG_ENTRY_SIZE, HASH_TABLE_SIZE, MCG_TABLE_SIZE
        for reg in regs:
            regs[reg]['val'] = mread(regs[reg]['addr'])

        for test in tests:
            if regs[test[0]]['val'] != regs[test[1]]['val']:
                print("Unsupported MCG configuration. %s (%d) should be equal to %s (%d)" %
                      (test[0], regs[test[0]]['val'],
                       test[1], regs[test[1]]['val']))

        if regs['p0_b0_mcg_size_64B']['val']:
            MCG_ENTRY_SIZE = 64
        else:
            MCG_ENTRY_SIZE = 64 * (1 << regs['p0_b0_mcg_size']['val'])
        HASH_TABLE_SIZE = calchashsize()
        MCG_TABLE_SIZE = (1 << regs['log_mc_table_sz']['val'])

    def isSupported(device):
        supported = [0x190, 0x1f5, 0x1f7]
        if "ibdr-" in device or "lid-" in device:
            print("-E- mlxmcg over inband device is not supported, device : '%s'" % (device))
            sys.exit(1)
        cmd = MCRA_BIN + " " + device + " 0xf0014"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.wait() != 0:
            print("-E- Could not access the device: \"%s\" by the command: \"%s\"" % (device, cmd))
            sys.exit(1)
        output = process.communicate()[0]
        try:
            devId = int(output[6:10], 16)
        except Exception as e:
            print("-E- Unexpected output of cmd : %s, output : %s" % (cmd, output))
            sys.exit(1)
        if devId not in supported:
            print("-E- The given device (%s) is not supported, only ConnectX3/ConnectX3-Pro devices are supported." % (device))
            sys.exit(1)

    wqdump_cmd = WQDUMP_BIN + ' -d ' + options.dev + ' --source mcg '
    if options.file == '':
        isSupported(options.dev)
        wqdump_all_cmd = wqdump_cmd + '--dump ALL_QPC '
        dumpDevice()
    else:
        if "win" in sys.platform:
            print("-E- Dump file option is not supported for Windows")
            sys.exit(1)
        wqdump_all_cmd = 'cat ' + options.file
        (MCG_ENTRY_SIZE, HASH_TABLE_SIZE, MCG_TABLE_SIZE) = [int(item.strip(" ()"), 0) for item in options.params.split(",")]

    print("MCG table size:  %d K entries, Hash size:  %d K entries, Entry size:  %d B" % ((MCG_TABLE_SIZE / 1024), (HASH_TABLE_SIZE / 1024), MCG_ENTRY_SIZE))
    l4_prot_map = {0: "--", 3: "Other", 5: "UDP", 6: "TCP"}
    progress_resolution = 1024
    factor = int(MCG_ENTRY_SIZE / 64)

    # replaces (val if cond else other_val) syntax which isn't supported in py < 2.5
    def select_val(condition, true_val, false_val):
        if condition:
            return true_val
        return false_val

    class Mcg:
        fmt = "%-8s %-4s %-7s %-6s %-4s %-5s %-17s %-15s %-15s  %-17s %-7s %-4s %-5s %-5s %-5s "
        fmt_gid = "%-8s %-4s %-7s %-6s %-61s %-17s %-7s %-4s %-5s %-5s %-5s "

        def get_hdr_str(self):
            if options.hopcount or options.advanced:
                return self.fmt % ('ID', 'Prio', 'Proto', 'DQP', 'Port', 'VLAN', 'MAC', 'SIP', 'DIP', 'I-MAC', 'I-VLAN', 'VNI', 'L4', 'SPort', 'DPort') + "HCount  Next   QPs"
            else:
                return self.fmt % ('ID', 'Prio', 'Proto', 'DQP', 'Port', 'VLAN', 'MAC', 'SIP', 'DIP', 'I-MAC', 'I-VLAN', 'VNI', 'L4', 'SPort', 'DPort') + "   Next  QPs"
        def __str__(self):
            str = ''
            self.uniqueStr = ''
            try:
                str = self.fmt_gid % (
                    "%-8x" % self.reg_id,
                    "%-4x" % self.prio,
                    "%-4s" % self.protocol,
                    select_val(self.dqp_check, "%06x" % self.dqp, '--'),
                    "%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x" % (
                        (self.gid[3] >> 16) & 0xffff,
                        (self.gid[3]) & 0xffff,
                        (self.gid[2] >> 16) & 0xffff,
                        (self.gid[2]) & 0xffff,
                        (self.gid[1] >> 16) & 0xffff,
                        (self.gid[1]) & 0xffff,
                        (self.gid[0] >> 16) & 0xffff,
                        (self.gid[0] & 0xffff)),
                    "%-17s" % "--",  # I mac
                    "%-5s" % "--",  # I vlan
                    "%-4s" % "--",  # VNI
                    l4_prot_map[self.l4_proto],
                    select_val(self.sport_check, "%-5d" % self.sport, '--'),
                    select_val(self.dport_check, "%-5d" % self.dport, '--'))

            except BaseException:
                vlan = '--'
                inner_vlan = '--'
                if self.vlan_check:
                    if self.vlan_present:
                        vlan = "%-5d" % self.vlan
                    else:
                        vlan = "none"
                if self.inner_vlan_check:
                    if self.inner_vlan_present:
                        inner_vlan = "%-5d" % self.inner_vlan

                str = self.fmt % (
                    "%-8x" % self.reg_id,
                    "%-4x" % self.prio,
                    "%-4s" % self.protocol,
                    select_val(self.dqp_check, "%06x" % self.dqp, '--'),
                    "%1d" % (2 - self.port),
                    "%-5s" % vlan,
                    select_val(self.mac_check, "%02x:%02x:%02x:%02x:%02x:%02x" % (
                        (self.mac & 0xff0000000000) >> 40,
                        (self.mac & 0xff00000000) >> 32,
                        (self.mac & 0xff000000) >> 24,
                        (self.mac & 0xff0000) >> 16,
                        (self.mac & 0xff00) >> 8,
                        (self.mac & 0xff)), '--'),
                    select_val(self.sip_check, "%d.%d.%d.%d" % ((self.sip & 0xff000000) >> 24,
                                                                (self.sip & 0xff0000) >> 16,
                                                                (self.sip & 0xff00) >> 8,
                                                                (self.sip & 0xff)), '--'),

                    select_val(self.dip_check, "%d.%d.%d.%d" % ((self.dip & 0xff000000) >> 24,
                                                                (self.dip & 0xff0000) >> 16,
                                                                (self.dip & 0xff00) >> 8,
                                                                (self.dip & 0xff)), '--'),
                    select_val(self.inner_mac_check, "%02x:%02x:%02x:%02x:%02x:%02x" % (
                        (self.inner_mac & 0xff0000000000) >> 40,
                        (self.inner_mac & 0xff00000000) >> 32,
                        (self.inner_mac & 0xff000000) >> 24,
                        (self.inner_mac & 0xff0000) >> 16,
                        (self.inner_mac & 0xff00) >> 8,
                        (self.inner_mac & 0xff)), '--'),
                    "%-5s" % inner_vlan,
                    select_val(self.inner_vni_check, "%-5s" % self.vni, '--'),
                    l4_prot_map[self.l4_proto],
                    select_val(self.sport_check, "%-5d" % self.sport, '--'),
                    select_val(self.dport_check, "%-5d" % self.dport, '--'))

            self.uniqueStr = str
            for qp in self.qps:
                self.uniqueStr = self.uniqueStr + "%x " % qp

            return str

        def _parse_ipv6(self, mapping):
            self.gid = []
            self.gid.append(
                mapping['mgi.mac_31_0'] & 0xffffffff)
            self.gid.append(
                (mapping['mgi.vlan_prio'] & 0x7) |
                (mapping['mgi.vlan_cfi'] & 0x1) << 3 |
                (mapping['mgi.vlan_id'] & 0xfff) << 4 |
                (mapping['mgi.mac_47_32'] & 0xffff) << 16)
            self.gid.append(
                (mapping['mgi.et_filter_bits'] & 0xffff) |
                (mapping['mgi.mgi_dip_31_24'] & 0xff) << 16 |
                (mapping['mgi.vep'] & 0x7) << 24 |
                (mapping['mgi.port'] & 0x1) << 27 |
                (mapping['mgi.force_loopback'] & 0x1) << 28 |
                (mapping['mgi.et_other_filter'] & 0x1) << 29 |
                (mapping['mgi.multicast'] & 0x1) << 30 |
                (mapping['mgi.vlan_present'] & 0x1) << 31)
            self.gid.append(
                mapping['mgi.mgi_sip'] & 0xffffffff)

            self.l4_proto = mapping['ctl.l4_protocol']
            self.sport_check = mapping['ctl.sport_check']
            self.sport = mapping['ctl.mgi_sport']
            self.dport_check = mapping['ctl.dport_check']
            self.dport = mapping['ctl.mgi_dport']
            self.dqp_check = mapping['ctl.dqp_check']
            self.dqp = mapping['ctl.mgi_dqp_dip_23_0']

        def _parse_rules(self, mapping):

            self.sip_check = mapping['ctl.sip_check']
            self.sip = mapping['mgi.mgi_sip']
            self.dip_check = mapping['ctl.dip_check']
            self.dip = (mapping['mgi.mgi_dip_31_24'] << 24) + mapping['ctl.mgi_dqp_dip_23_0']
            self.l4_proto = mapping['ctl.l4_protocol']
            self.sport_check = mapping['ctl.sport_check']
            self.sport = mapping['ctl.mgi_sport']
            self.dport_check = mapping['ctl.dport_check']
            self.dport = mapping['ctl.mgi_dport']
            self.dqp_check = mapping['ctl.dqp_check']
            self.dqp = mapping['ctl.mgi_dqp_dip_23_0']

        def _parse_tunnel(self, mapping):
            try:
                self.inner_vni_check = mapping['ctl.vni_check']
                self.vni = mapping['ctl.vni']
                self.inner_vlan_check = mapping['ctl.inner_vlan_check']
                self.inner_vlan_present = mapping['ctl.inner_vlan_present']
                self.inner_vlan = mapping['ctl.inner_vlan_id']
                self.inner_mac_check = mapping['ctl.inner_mac_check']
                self.inner_mac = (mapping['ctl.inner_dmac_47_32'] << 32) + mapping['mgi.inner_dmac_31_0']
            except BaseException:
                self._parse_rules(mapping)

        def _parse_mapping(self, mapping):
            self.index = mapping['index']
            self.protocol = mapping['ctl.protocol']

            protocol = {0: "IPv6", 1: "L2_TUNNEL", 2: "L2", 4: "IPv4", 6: "FCoETH", 7: "all"}
            newName = protocol.get(self.protocol)
            if newName is not None:
                self.protocol = newName

            if self.protocol == 'IPv6':  # ipv6
                self._parse_ipv6(mapping)
            else:
                self.inner_mac_check = 0
                self.inner_vlan_check = 0
                self.inner_vlan_present = 0
                self.inner_vni_check = 0
                self.vni = "--"
                self.inner_mac = 0
                self.dqp_check = 0
                self.dqp = 0
                self.sip_check = 0
                self.sip = 0
                self.dip_check = 0
                self.dip = 0
                self.sport_check = 0
                self.sport = 0
                self.dport_check = 0
                self.dport = 0
                self.dqp_check = 0
                self.dqp = 0
                self.l4_proto = 0

                if self.protocol == 'L2_TUNNEL':  # L2_TUNNEL
                    self._parse_tunnel(mapping)
                else:  # l2 ....
                    self._parse_rules(mapping)

                self.port = mapping['mgi.port']
                self.mac_check = mapping['ctl.mac_check']
                self.mac = (mapping['mgi.mac_47_32'] << 32) + mapping['mgi.mac_31_0']
                self.vlan_check = mapping['ctl.vlan_check']
                self.vlan = mapping['mgi.vlan_id']
                self.vlan_present = mapping['mgi.vlan_present']

            if l4_prot_map.get(self.l4_proto) is None:
                self.l4_proto = 3
            self.next = int(mapping['ctl.next_mcg'] / factor)
            try:
                self.prio = ((mapping['member[0].fw_cxt']) +
                             (mapping['member[1].fw_cxt'] << 4) +
                             (mapping['member[2].fw_cxt'] << 8) +
                             (mapping['member[3].fw_cxt'] << 12)
                             )
            except BaseException:
                try:
                    self.prio = ((mapping['member[0]'] & 0x0f000000) >> 24) + \
                        ((mapping['member[1]'] & 0x0f000000) >> 20) + \
                        ((mapping['member[2]'] & 0x0f000000) >> 16) + \
                        ((mapping['member[3]'] & 0x0f000000) >> 12)
                except BaseException:
                    self.prio = 0

            m = 0
            self.qps = []
            self.member_cnt = mapping['ctl.member_cnt']
            while m < self.member_cnt:
                if mapping.get('member_' + str(m) + '_.qpn') is not None:
                    self.qps.append(mapping['member_' + str(m) + '_.qpn'])
                elif mapping.get('member[' + str(m) + '].qpn') is not None:
                    self.qps.append(mapping['member[' + str(m) + '].qpn'])
                m = m + 1
            try:
                self.reg_id = mapping['member_7_.qpn']
            except BaseException:
                self.reg_id = mapping['member[7].qpn']

        def parse_wqdump_single_mcg(self, output):
            in_mcg = False
            mapping = {}
            for line in output:
                if line.startswith('----------------------------------Index') or line.startswith('MCG MCG Index'):
                    #m = re.search(r"Index 0x([0-9a-fA-F]*) ", line)
                    m = re.search(r"Index 0x?\s*([0-9a-fA-F]+).*?", line)
                    mapping['index'] = int(int(m.group(1), 16) / factor)
                    in_mcg = True
                    continue
                if line.startswith('--------------------------------------------') or line.startswith('--------------------------'):
                    break
                if in_mcg:
                    words = line.split()
                    key = words[0]
                    try:
                        val = int(words[2], 16)
                    except BaseException:
                        val = words[2]
                    mapping[key] = val

            if in_mcg:
                self._parse_mapping(mapping)
                return True
            return False

        def read_wqdump_single_mcg(self, index):
            index = index * factor
            output = os.popen(wqdump_cmd + '--dump QP --qp ' + str(index))
            return output

        def getkey(self):
            m = self
            m.reg_id = 0
            str(m)
            return m.uniqueStr

    progress_cnt = 0
    def progress(x):
        if options.quiet:
            return
        global progress_cnt
        if progress_cnt == 0:
            print('Progress: ')
        if (progress_cnt % progress_resolution) == 0:
            print(x)
        progress_cnt += 1

    output = os.popen(wqdump_all_cmd)

    def fill_mcgtable(output, idx, bucket=None):
        global mcgtable
        while True:
            m = Mcg()
            m.hopCount = idx
            if not m.parse_wqdump_single_mcg(output):
                break
            if bucket is None:
                m.bucket = m.index
            else:
                m.bucket = bucket

            if m.index < HASH_TABLE_SIZE:
                progress('H')
            else:
                progress('T')
            mcgtable[m.index] = m

    mcgtable = dict()
    mcglist = []
    mapMcgList = dict()
    mcgListIndex = 0
    rulescount = dict()
    total = 0
    idx = 0

    fill_mcgtable(output, idx)
    while idx < HASH_TABLE_SIZE:
        try:
            m = mcgtable[idx]
        except BaseException:
            idx += 1
            continue
        bucket = m.bucket
        while True:
            total += 1
            if m.getkey() not in rulescount:
                rulescount[m.getkey()] = 0
                m.minHopCount = m.hopCount
                m.maxHopCount = m.hopCount
                mapMcgList[m.getkey()] = mcgListIndex
                if not options.advanced:
                    mcglist += [m]
                    mcgListIndex += 1
            if options.advanced:
                mcglist += [m]
                mcgListIndex += 1
                m.minHopCount = m.hopCount
                m.maxHopCount = m.hopCount
            else:
                if mcglist[mapMcgList[m.getkey()]].minHopCount > m.hopCount:
                    mcglist[mapMcgList[m.getkey()]].minHopCount = m.hopCount
                if mcglist[mapMcgList[m.getkey()]].maxHopCount < m.hopCount:
                    mcglist[mapMcgList[m.getkey()]].maxHopCount = m.hopCount
            rulescount[m.getkey()] += 1
            next = m.next
            if next >= MCG_TABLE_SIZE:
                break
            try:
                if next not in mcgtable.keys():
                    output = m.read_wqdump_single_mcg(next)
                    fill_mcgtable(output, m.hopCount + 1, m.bucket)
                #m.bucket = bucket
                if next == m.index:
                    break
                m = mcgtable[next]
            except Exception as e:
                sys.stderr.write('')
                sys.stderr.write('mcg	[0x%x].next	points	to	non	existing	mcg	index	0x%x\n' % (m.index, next))
                break
            m.bucket = bucket
            progress('L')
        idx += 1

    if not options.quiet:
        sys.stderr.write('\n')

    m = Mcg()
    print("%-6s %-6s  %s" % ('Bucket', 'Index', m.get_hdr_str()))

    dups = []
    uniqueRules = len(mcglist)
    qpsTable = {}
    for i, m in enumerate(mcglist):
        string = ""
        if len(m.qps) < 2:
            for qp in m.qps:
                string += "%x " % qp
        else:
            string = "SB"
            qpsTable[m.index] = m.qps

        rulecount = rulescount[m.getkey()]
        if rulecount == 1 or options.advanced:
            if options.advanced or (options.hopcount and m.member_cnt):
                print("%6x %6x  %s   %d  %6x  %5s" % (int(m.bucket), int(m.index), m, m.hopCount, int(m.next), string))
            elif m.member_cnt:
                print("%6x %6x  %s %6x  %s" % (int(m.bucket), int(m.index), m, int(m.next), string))
            else:
                total -= rulecount
                uniqueRules -= 1
        else:
            if options.hopcount and m.member_cnt:
                dups.append("%6x %6x  %s   %d-%d %6x %5s  %5d" % (int(m.bucket), int(m.index), m, m.minHopCount, m.maxHopCount, int(m.next), string, rulecount))
            elif m.member_cnt:
                dups.append("%6x %6x  %s %6x %5s  %5d" % (int(m.bucket), int(m.index), m, int(m.next), string, rulecount))
            else:
                total -= rulecount
                uniqueRules -= 1

    if len(dups):
        spaces = "                                                                                                                                                                   "
        print("Duplicated MCGS:" + spaces[0:len(dups[0]) - 20] + "Count")
        for dup in dups:
            print(dup)

    if not options.advanced:
        print("%d Unique rules, %d Total" % (uniqueRules, total))
    else:
        print("%d Total" % total)

    if not len(qpsTable):
        sys.exit(0)

    print("\n\nIndex          QPs")
    print("=" * 117)
    start = "\n            "
    for (index, allQps) in qpsTable.iteritems():
        qps = ""
        allQps.sort()
        for i, qp in enumerate(allQps):
            if i % 15 == 0 and i != 0:
                qps += start
            qps += ("%6x " % qp)
        print("%6x      %s" % (index, qps))
        print("=" * 117)

except SystemExit as e:
    sys.exit(e)
except(KeyboardInterrupt, SystemExit):
    signalHandler()
except Exception as e:
    print("-E- Oops UnExpected Error, Please report this to tool Owner !")
    raise e
    sys.exit(1)
