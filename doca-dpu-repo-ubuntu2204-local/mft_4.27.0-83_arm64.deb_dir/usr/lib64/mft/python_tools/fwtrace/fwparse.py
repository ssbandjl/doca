#!/usr/bin/env python
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

from __future__ import print_function
from mft_core_device import MftCoreDevice
import fw_trace_utilities
import re
import struct
import regaccess
import fw_trace_utilities

class FwTraceParser(object):

    MISSING_STRING = \
        "*** MISSING STRING POINTER, EVENT DATA SEQUENSE 0 IS MISSING ***"

    #############################
    def __init__(self,mstDevice, devInfo, fwStrDBContents):
        self.mstDevice = mstDevice
        self.devInfo = devInfo
        #self.deviceType = deviceType
        self.dwsnMsbSupported = self.isDwsnMsbSupported(mstDevice)
        if (devInfo._mft_core_device._is_switch_ib() or devInfo._mft_core_device._is_switch_ib2()):
        #elif deviceType in self.DEVICES_WITH_IRISC_ID:
            self.eventRE = re.compile(
                br"(\S+)\s+ITRACE(\d*)\s+IriscId:\s*(\S+)\s+Msn:\s*(\S+)\s+Dsn:\s*(\S+)\s+Data:\s*(\S+)\s*")
        elif devInfo.is_dynamic_device_without_irisc_id:
        #if deviceType in self.DYNAMIC_DEVICES_WITHOUT_IRISC_ID:
            self.eventRE = re.compile(
                br"(\S+)\s+(\S+)\s+Msn:\s*(\S+)\s+Dwsn_msb:\s*(\S+)\s+Dsn:\s*(\S+)\s+Data:\s*(\S+)\s*")
        elif not devInfo.is_dynamic_device_without_irisc_id:
        #elif deviceType in self.DEVICES_WITHOUT_IRISC_ID:
            self.eventRE = re.compile(
                br"(\S+)\s+(\S+)\s+Msn:\s*(\S+)\s+Dsn:\s*(\S+)\s+Data:\s*(\S+)\s*")
        else:
            raise RuntimeError(
                "-E- in Failed to initialize the FwParser, "
                "unknown device type=%d" % (devInfo._mft_core_device.get_device_name()))

        # list of tuples (start addr, end addr, data contents)
        self.dataSections = []
        # key is irisc, value is dictionary:
        # {'ts': ts, 'msn' : msn, 'data' : {dsn : data}}
        self.lastMsg = {}
        self.specifierRE = re.compile(
            br"%([-+ #0])?(\d+)?(\.\d*)?(l|ll)?([duxs])")

        # init data sections db
        self.readDataSectionTlvs(fwStrDBContents)

    #############################
    def readDataSectionTlvs(self, fwStrDBContents):
        data = fwStrDBContents

        # read tlv by tlv
        while len(data):
            tlvType, = struct.unpack(">I", data[0:4])
            # print "type = 0x%x" % tlvType
            if tlvType == 1:  # data section
                dataSize, = struct.unpack(">I", data[4:8])
                sectionInfo = data[8: 24]
                sectionDataSize = dataSize - 16
                sectionData = data[24: dataSize + 8]
                data = data[8 + dataSize:]  # prepare data for next TLV
                # print "size = %d" % dataSize
                # print "meta data = %s" % repr(sectionInfo)
                # print "data section raw = %s" % repr(sectionData)
                # print "data section size = %d" % len(sectionData)

                startVirtualAddr, = struct.unpack(">I", sectionInfo[0:4])
                self.dataSections.append(
                    (startVirtualAddr, startVirtualAddr + sectionDataSize,
                     sectionData))

    #############################
    def getFmtString(self, ptr):
        for section in self.dataSections:
            if section[0] <= ptr <= section[1]:
                fmtData = section[2][ptr - section[0]:]
                fmtSize = fmtData.find(b'\0')
                if fmtSize + ptr >= section[1]:
                    # string crosses data section address range
                    return None
                fmt, = struct.unpack("%ds" % fmtSize, fmtData[:fmtSize])
                return fmt.replace(b"%llx", b"%08x%08x")

        return None

    #############################
    def printDataSection(self):
        for idx, section in enumerate(self.dataSections):
            try:
                print("=============== Section: %d ===================" % idx)
                for addr in range(section[0], section[1], 8):
                    print("0x%08x) " % addr,)
                    for i in range(8):
                        byte, = struct.unpack(
                            "B", section[2][addr + i - section[0]])
                        print("%02x " % byte,)

                    for i in range(8):
                        byte, = struct.unpack(
                            "B", section[2][addr + i - section[0]])
                        print("%2s " % chr(byte),)
                    print("")
            except BaseException:
                pass

    #############################
    def calcArgLen(self, fmt):
        return len(self.specifierRE.findall(fmt))

    #############################
    def pushLine(self, line):
        line = line.strip()
        matchPhrase = self.eventRE.search(line)
        if matchPhrase:
            # print "ts = %s, irisc = %s, msn = %s, dsn = %s, \
            # data = %s" % matchPhrase.groups()
            dwsn_msb = "0"

            if (self.devInfo._mft_core_device._is_switch_ib() or self.devInfo._mft_core_device._is_switch_ib2()):
                if(len(matchPhrase.groups()) == 6):
                    ts, irisc, iriscId, msn, dsn, data = matchPhrase.groups()
                    irisc = "ITRACE{}".format(int(irisc,0) * 2 + int(iriscId,0))
                else:
                    raise RuntimeError("parsing {}\nFailed, found {}, expected 6 groups".format(line, len(matchPhrase.groups())) )

            elif self.devInfo.is_dynamic_device_without_irisc_id:
                if(len(matchPhrase.groups()) == 6):
                    ts, irisc, msn, dwsn_msb, dsn, data = matchPhrase.groups()
                else:
                    raise RuntimeError("parsing {}\nFailed, found {}, expected 6 groups".format(line, len(matchPhrase.groups())) )

            elif not self.devInfo.is_dynamic_device_without_irisc_id:
                if(len(matchPhrase.groups()) == 5):
                    ts, irisc, msn, dsn, data = matchPhrase.groups()
                else:
                    raise RuntimeError("parsing {}\nFailed, found {}, expected 5 groups".format(line, len(matchPhrase.groups())) )


            if irisc.strip() == "":
                irisc = "0xffff"
            msn = int(msn,0)
            dsn = int(dsn,0)
            if self.dwsnMsbSupported:
                dsn += (8 * int(dwsn_msb,0))
            data = int(data,0)
            if not (self.devInfo._mft_core_device._is_switch_ib() or self.devInfo._mft_core_device._is_switch_ib2()):
                irisc = irisc.decode(encoding='UTF-8', errors='ignore')
            irisc = irisc.replace("ITRACE","I")
            irisc = irisc.replace("TILE","T")
            ts = ts.decode(encoding='UTF-8', errors='ignore')
            ts = str(ts).replace("U","")
            # print "ts = %s, irisc = 0x%x, msn = %d, dsn = 0x%x, \
            # data = 0x%x" % (ts, irisc, msn, dsn, data)
            self.pushEvent(line, ts, str(irisc), msn, dsn, data)

        else:
            # check if we get to the end of mlxtrace output,
            # then we should flush all buffered events
            if line.find(b"Parsed all events") != -1:
                for irisc in list(self.lastMsg.keys()):
                    self.flushIriscMsg(irisc)

            # Print as is, this is not fwtrace line
            # (maybe hw trace event or some informative message)
            print(line.decode(encoding='UTF-8', errors='ignore'))

    #############################
    def pushEvent(self, line, ts, irisc, msn, dsn, data):
        lastMsg = self.lastMsg.get(irisc)
        if not lastMsg:
            lastMsg = {'ts': ts, 'msn': msn, 'data': {dsn: data}}
            self.lastMsg[irisc] = lastMsg
        else:
            lastMsg = self.lastMsg[irisc]
            if lastMsg['msn'] != msn:
                # print "last msg msn = 0x%x, msg=0x%x, current = 0x%x" % \
                # (lastMsg['msn'], lastMsg['data'].get(0), msn)
                self.flushIriscMsg(irisc)
                lastMsg = {'ts': ts, 'msn': msn, 'data': {dsn: data}}
                self.lastMsg[irisc] = lastMsg
            else:
                lastMsg['data'][dsn] = data

        # Check if we have all required argument, then we can flush
        strPtr = self.lastMsg[irisc]['data'].get(0)
        if strPtr:
            fmtStr = self.getFmtString(strPtr)
            if fmtStr is None:
                print("*** Can't find string with pointer: 0x%x" % strPtr)
            else:
                if self.calcArgLen(fmtStr) < len(self.lastMsg[irisc]['data']):
                    self.flushIriscMsg(irisc)

    #############################
    def flushIriscMsg(self, irisc):
        lastMsg = self.lastMsg[irisc]

        if lastMsg is None:
            return

        strPtr = lastMsg['data'].get(0)
        if strPtr is None:
            print(self.MISSING_STRING)
            del self.lastMsg[irisc]
            return

        ts = lastMsg['ts']
        fmtStr = self.getFmtString(strPtr)
        if fmtStr is None:
            print("*** Can't find string with pointer: 0x%x" % strPtr)
            del self.lastMsg[irisc]
            return
        fmtStr = fmtStr.strip()

        argLen = self.calcArgLen(fmtStr)
        args = [None] * argLen

        if argLen:
            for dsn, data in lastMsg['data'].items():
                if dsn != 0:
                    args[dsn - 1] = data

        for i in range(len(args)):
            if args[i] is None:
                args[i] = "<!!!MISSING-ARGUMENT!!!>"
                fmtStr = self.replaceNthSpecifier(fmtStr, b"%s", i + 1)

        print("{:<16}".format(ts), end="")
        if irisc == "0xffff":
            print("",)
        else:
            print(" {:<16}".format(irisc), end="")

        if len(args):
            try:
                print(fmtStr.decode(encoding='UTF-8', errors='ignore') % tuple(args))
            except Exception as exp:
                raise Exception(
                    "-E- in StrPtr=0x%x, fmt(%s): %s" % (strPtr, fmtStr, exp))

        else:
            print(fmtStr.decode(encoding='UTF-8', errors='ignore'))
        del self.lastMsg[irisc]

    #############################
    def flushAll(self):
        iriscs = self.lastMsg.keys()
        for irisc in iriscs:
            self.flushIriscMsg(irisc)

    #############################
    def replaceNthSpecifier(self, fmt, rep, n):

        def replaceNthWith(n, replacement):

            def replace(match, c=[0]):
                c[0] += 1
                if c[0] == n:
                    return replacement
                else:
                    return match.group(0)

            return replace

        return self.specifierRE.sub(replaceNthWith(n, rep), fmt)

    def isDwsnMsbSupported(self,mstDev):
        if not mstDev:
            return False
        regAccessObj = regaccess.RegAccess(mstDev)
        mteimData = None
        if regAccessObj:
            mteimData = regAccessObj.sendMTEIM()
            if mteimData:
                return mteimData["is_dwsn_msb_supported"]
        return False

#####################################################
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("-E- missing fw strings db file path and string pointer")
        sys.exit(1)

    data = open(sys.argv[1], "r+b").read()
    fwParser = FwTraceParser(None, fw_trace_utilities.DeviceType.DEVICE_CONNECTIB, data)
    ptr = int(sys.argv[2],0)
    # fwParser.printDataSection()
    fmt = fwParser.getFmtString(ptr)
    if fmt is None:
        print("Coudn't find string ptr: 0x%x" % ptr)
    else:
        print("ptr 0x%x == > %s" % (ptr, fmt))

    sys.exit(0)
