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
import sys
import os


# Clear LD_LIBRARY_PATH to prevent pyinstaller compatibility issues
library_path_var = "LD_LIBRARY_PATH"
is_pyinstaller = getattr(sys, 'frozen', False)
if is_pyinstaller and library_path_var in os.environ:
    os.environ[library_path_var] = ""

import string
import getopt
import subprocess
import signal
import re
import shutil
import struct
import tempfile
import getpass
import glob

# mft imports
sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..", "..", "common"))
sys.path.append(os.path.join("..", "..", "mtcr_py"))
sys.path.append(os.path.join("..", "..", "cmdif"))
sys.path.append(os.path.join("..", "..", "reg_access"))
sys.path.append(os.path.join("..", "..", "mft_core", "device", "mst_device", "wrapper"))

import mtcr
import cmdif
import fwparse
import regaccess
from secure_fw_trace import SecureFwTrace
import fw_trace_utilities
from mft_core_device import MftCoreDevice


# In case python version is higher/equal to 2.5 use hashlib
if sys.version_info >= (2, 5):
    from hashlib import md5
else:
    from md5 import md5


class UnbufferedStream(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = UnbufferedStream(sys.stdout)
EXEC_NAME = "fwtrace"
MLXTRACE_EXE = "mlxtrace"
NUM_BYTES_TO_READ_MTRC_STDB = 704

proc = None


def signal_handler(signal, frame):
    print("\nInterrupted, exiting ...")
    global MST_DEVICE
    global proc
    if not KEEP_RUNNING:
        try:
            if not MST_DEVICE:
                MST_DEVICE = mtcr.MstDevice(DEV_NAME)
            regAccessObj = regaccess.RegAccess(MST_DEVICE)
            rc = regAccessObj.sendMtrcCapReleaseOwnership()
        except Exception as e:
            print(e)

    if proc is not None:
        proc.terminate()
        proc.wait()
        proc = None

    sys.exit(0)


#######################################################

def IsWindows():
    return os.name == "nt"


if IsWindows():  # windows
    if getattr(sys, 'frozen', False):
        appPath = os.path.dirname(sys.executable)
    elif __file__:
        appPath = os.path.dirname(__file__)


#######################################################
RUN_HELP = False
MST_DEVICE = None
CMDIFDEV = None
DEV_NAME = None
DUMP_FILE = None
FW_STR_DB_FILE = None
IRISC_NAME_LIST = []
TILE_IRISC_NAME_LIST = []
APU_ENABLE = False
INCLUDE_PHY_UC = False
MASK = None
LEVEL = None
LOG_DELAY = None
STRAMING_MODE = False
SNAPSHOT_MODE = False
TRACER_MODE = None
REAL_TS = False
MLXTRACE_CFG = None
BUF_SIZE = None
FLINT_OCR = ""
GVMI = 0
IGNORE_OLD_EVENTS = False
KEEP_CFG = False
KEEP_RUNNING = False
CONFIG_ONLY = False
FW_CFG_ONLY = False
MEMACCESS_MODE = None
MAX_BUFFER_SIZE_MB = 0xFFFFFFFFFFFFFFFF


#######################################################


class TracerException(Exception):
    pass


#######################################################
def ParseCmdLineArgs():
    global RUN_HELP
    global DEV_NAME
    global DUMP_FILE
    global FW_STR_DB_FILE
    global IRISC_NAME_LIST  # cap_core_tile  #for all cap_num_of_tile
    global TILE_IRISC_NAME_LIST  # cap_core_main
    global APU_ENABLE  # cap_core_dpa
    global INCLUDE_PHY_UC
    global MASK
    global LEVEL
    global LOG_DELAY
    global STRAMING_MODE
    global SNAPSHOT_MODE
    global TRACER_MODE
    global REAL_TS
    global MLXTRACE_CFG
    global BUF_SIZE
    global FLINT_OCR
    global GVMI
    global IGNORE_OLD_EVENTS
    global MEMACCESS_MODE
    global KEEP_CFG
    global KEEP_RUNNING
    global CONFIG_ONLY
    global FW_CFG_ONLY
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "hd:f:i:t:am:l:snc:vS:G:",
            ["help", "device=", "fw_strings=", "irisc=", "tile=", "apu",
             "mask=", "level=", "log_delay=", "stream", "snapshot", "tracer_mode=",
             "mem_access=", "real_ts", "include_phy_uc",
             "cfg=", "dump=", "version", "buf_size=", "ocr", "gvmi=",
             "ignore_old_events", "keep_cfg", "keep_running", "config_only",
             "fw_cfg_only"])
        for o, a in opts:
            if o in ["-h", "--help"]:
                RUN_HELP = True
            elif o in ["-d", "--device"]:
                DEV_NAME = a
            elif o in ["--dump"]:
                DUMP_FILE = a
            elif o in ["-f", "--fw_strings"]:
                FW_STR_DB_FILE = a
            elif o in ["-i", "--irisc"]:
                IRISC_NAME_LIST.append(a)
            elif o in ["-t", "--tile"]:
                TILE_IRISC_NAME_LIST.append(a)
            elif o in ["-a", "--apu"]:
                APU_ENABLE = True
            elif o in ["--include_phy_uc"]:
                INCLUDE_PHY_UC = True
            elif o in ["-m", "--mask"]:
                MASK = a
            elif o in ["-l", "--level"]:
                LEVEL = a
            elif o in ["--log_delay"]:
                LOG_DELAY = a
            elif o in ["-s", "--stream"]:
                STRAMING_MODE = True
            elif o in ["-n", "--snapshot"]:
                SNAPSHOT_MODE = True
            elif o in ["--tracer_mode"]:
                TRACER_MODE = a
            elif o in ["--real_ts"]:
                REAL_TS = True
            elif o in ["-c", "--cfg"]:
                MLXTRACE_CFG = a
            elif o in ["-S", "--buf_size"]:
                BUF_SIZE = a
            elif o in ["--ocr"]:
                FLINT_OCR = "-ocr"
            elif o in ["-v", "--version"]:
                import tools_version
                tools_version.PrintVersionString(EXEC_NAME, None)
                sys.exit(0)
            elif o in ["-G", "--gvmi"]:
                GVMI = a
            elif o in ["--mem_access"]:
                MEMACCESS_MODE = a
            elif o in ["--ignore_old_events"]:
                IGNORE_OLD_EVENTS = True
            elif o in ["--keep_cfg"]:
                KEEP_CFG = True
            elif o in ["--keep_running"]:
                KEEP_RUNNING = True
            elif o in ["--config_only"]:
                CONFIG_ONLY = True
            elif o in ["--fw_cfg_only"]:
                FW_CFG_ONLY = True
            else:
                Usage()
                raise TracerException("Unhandled option: %s" % o)
    except getopt.GetoptError as exp:
        print(exp)
        Usage()
        sys.exit(1)


#######################################################
def GetStatusOutput(cmd):
    """Return (status, output) of executing cmd in a shell.
    This new implementation should work on all platforms.
    """
    pipe = subprocess.Popen(
        cmd, shell=True, universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = str.join("", pipe.stdout.readlines())
    rc = pipe.wait()
    if rc is None:
        rc = 0
    return rc, output


#######################################################
def IsExternal():
    return GetStatusOutput("mlxtrace_int -h")[0] != 0


def GetDevInfoFromDumpFile(dumpFile):
    fd = open(dumpFile, "rb")
    fd.read()
    fd.seek(0x1c)
    deviceType, = struct.unpack(">I", fd.read(4))
    # NOTE: this is according to mlxtrace enum and not dev_mgt enum

    try:
        return deviceType
    except KeyError:
        print("error in GetDeviceInfoFromDumpFile", KeyError)
        raise TracerException(
            "Unknown/Unsupported device type: 0x%x" % deviceType)


#######################################################
def GetDeviceInfoFromDumpFile(dumpFile):
    print(dumpFile)
    deviceID = GetDevInfoFromDumpFile(dumpFile)
    chipRev = -1
    try:
        devInfo = fw_trace_utilities.TracersDeviceInfo(deviceID)
        return devInfo
    except Exception as e:
        print("error in GetDeviceInfoFromDumpFile", e)
        raise TracerException(
            "Unknown/Unsupported device with DevId: 0x%x and ChipRev: 0x%x" %
            (deviceID, chipRev))


#######################################################
def GetDeviceInfo(dev):
    devIdChipRev = dev.read_device_id()
    devId = devIdChipRev & 0xffff
    chipRev = (devIdChipRev >> 16) & 0xf
    try:
        devInfo = fw_trace_utilities.TracersDeviceInfo(devId)
        return devInfo
    except Exception as e:
        print("error in GetDeviceInfo", e)
        raise TracerException(
            "Unknown/Unsupported device with DevId: 0x%x and ChipRev: 0x%x" %
            (devId, chipRev))


#######################################################
def CheckSecureFwArgs(devInfo):

    if GVMI:
        raise TracerException("gvmi is not compatible with secure fw")

    if FW_STR_DB_FILE:
        raise TracerException("Fw strings db file is not compatible with secure fw")

    if len(IRISC_NAME_LIST) > 0:
        if "all" not in IRISC_NAME_LIST:
            raise TracerException("only \"-i all\" is compatible with secure fw")

    if SNAPSHOT_MODE:
        raise TracerException("snapshot is not compatible with secure fw")

    if KEEP_RUNNING:
        raise TracerException("keep_running is not compatible with secure fw")

    if CONFIG_ONLY:
        raise TracerException("config_only is not compatible with secure fw")

    if FW_CFG_ONLY:
        raise TracerException("fw_cfg_only is not compatible with secure fw")

    if KEEP_CFG:
        raise TracerException("keep_cfg is not compatible with secure fw")

    if TRACER_MODE:
        if TRACER_MODE not in ["FIFO", "MEM"]:
            raise TracerException("Unknown tracer mode: %s" % TRACER_MODE)
        elif TRACER_MODE == "FIFO":
            raise TracerException("FIFO mode is not compatible with secure fw")

    if MEMACCESS_MODE:
        raise TracerException("memaccess is not compatible with secure fw")

    if DUMP_FILE:
        raise TracerException("dump file is not compatible with secure fw")

    if MASK and not IsNumeric(MASK):
        classes = MASK.split("+")
        for klass in classes:
            valid_class = False
            for mask_cls in devInfo.mask_classes:
                if mask_cls[0] == klass:
                    valid_class = True
            if not valid_class:
                raise TracerException("Unknown trace class: %s" % klass)

    if (MASK and LEVEL is None) or (LEVEL and MASK is None):
        raise TracerException("both --mask and --level must be provided")

    if LOG_DELAY:
        if (not IsNumeric(LOG_DELAY)) or (int(LOG_DELAY) < 0):
            raise TracerException("--log_delay must be a positive integer")

    if (LOG_DELAY and LEVEL is None):
        raise TracerException("When setting --log_delay, --level must be provided")

    if BUF_SIZE:
        raise TracerException("buf_size is not compatible with secure fw")

    if MLXTRACE_CFG:
        raise TracerException("cfg is not compatible with secure fw")


def isEventsExist():
    if (len(IRISC_NAME_LIST) == 0) and (len(TILE_IRISC_NAME_LIST) == 0) and (APU_ENABLE == False):
        return False
    return True

# "1.all" will enable risc1 in all tiles.
# "1" will be treated like "1.all"
# "all" will be treated like all.all
# "all.1" will enable all iriscs in tile 1
# .1 is invalid input.


def parseTileEvent(devInfo, event):
    first = None
    second = None
    matchPhrase = re.search("^t*(\\w+)\\.*(\\w*)", event)
    try:
        first = matchPhrase.group(1)
        if first != "all":
            risc = int(first)
            if (risc < 0) or (risc > devInfo.max_irisc_per_tile):
                return None
        if(matchPhrase.group(2) != ""):
            second = matchPhrase.group(2)
            if second != "all":
                risc = int(second)
                if (risc < 0) or (risc > devInfo.max_num_of_tiles):
                    return None
        else:
            second = "all"
    except BaseException:
        return None
    return (first, second)


def isValidEvents(devInfo):
    def checkEvent(regPatern, event, eventType, maxSize):
        if (event == "all"):
            return
        matchPhrase = re.search(regPatern, event)
        if(matchPhrase):
            risc = int(matchPhrase.group(1))
            if (risc < 0) or (risc > maxSize):
                raise TracerException("Unknown %s: %s" % (eventType, event))
        else:
            raise TracerException("Unknown %s: %s" % (eventType, event))
    # end of inner function logic

    for event in IRISC_NAME_LIST:
        if (event == "iron"):
            continue
        else:
            checkEvent("^i*(\\d+)", event, "irisc", devInfo.max_main_irisc)  # will match %d or i%d

    # no need to check APU input

    for event in TILE_IRISC_NAME_LIST:
        valid = parseTileEvent(devInfo, event)
        if not valid:
            raise TracerException("Unknown Tile argument format: %s" % event)


def CheckArgs(devInfo):
    if FW_STR_DB_FILE:
        if not os.path.exists(FW_STR_DB_FILE):
            raise TracerException(
                "Fw strings db file doesn't exist: %s" % FW_STR_DB_FILE)

    if (not isEventsExist()) and (SNAPSHOT_MODE is False):
        raise TracerException("Missing FW event name")

    if STRAMING_MODE:
        if SNAPSHOT_MODE:
            raise TracerException(
                "Snapshot and Streaming mode cannot be both enabled")
        if KEEP_RUNNING:
            raise TracerException(
                "keep_running and Streaming mode cannot be both enabled")
        if CONFIG_ONLY:
            raise TracerException(
                "config_only and Streaming mode cannot be both enabled")

    isValidEvents(devInfo)  # check all IRISC / TILES / APU event arguments.

    if DUMP_FILE:
        if FW_STR_DB_FILE is None:
            raise TracerException(
                "In dump file mode you must specify the fw strings db file")
        if STRAMING_MODE:
            raise TracerException(
                "Streaming mode is invalid option in dump file mode")
        if SNAPSHOT_MODE:
            raise TracerException(
                "Snapshot mode is invalid option in dump file mode")
        if KEEP_RUNNING:
            raise TracerException(
                "keep_running is invalid option in dump file mode")
        if CONFIG_ONLY:
            raise TracerException(
                "config_only is invalid option in dump file mode")
        if FW_CFG_ONLY:
            raise TracerException(
                "fw_cfg_only is invalid option in dump file mode")

    if SNAPSHOT_MODE and (TRACER_MODE != "FIFO"):
        raise TracerException("snapshot is only valid in FIFO MODE")

    if MASK and not IsNumeric(MASK):
        classes = MASK.split("+")
        for klass in classes:
            valid_class = False
            for mask_cls in devInfo.mask_classes:
                if mask_cls[0] == klass:
                    valid_class = True
            if not valid_class:
                raise TracerException("Unknown trace class: %s" % klass)

    if (MASK and LEVEL is None) or (LEVEL and MASK is None):
        raise TracerException("both --mask and --level must be provided")

    if LOG_DELAY:
        if (not IsNumeric(LOG_DELAY)) or (int(LOG_DELAY) < 0):
            raise TracerException("--log_delay must be a positive integer")

    if (LOG_DELAY and LEVEL is None):
        raise TracerException("When setting --log_delay, --level must be provided")

    if TRACER_MODE:
        if TRACER_MODE not in ["FIFO", "MEM"]:
            raise TracerException("Unknown tracer mode: %s" % TRACER_MODE)

    if MEMACCESS_MODE:
        if MEMACCESS_MODE not in \
                ("OB_GW", "VMEM", "UDRIVER"):
            raise TracerException(
                "Unknown memaccess mode: %s" % MEMACCESS_MODE)
    if BUF_SIZE:
        temp_size = BUF_SIZE
        try:
            if temp_size.startswith("0x"):
                temp_size = int(temp_size, 16)
            else:
                temp_size = int(temp_size)
            if temp_size > MAX_BUFFER_SIZE_MB:
                raise TracerException("buffer size exceed the limit of 64 bit")
        except ValueError:
            raise TracerException("invalid buffer size: '{}'".format(BUF_SIZE))

#######################################################


def Usage():
    print("Usage:")
    print(" {} -d|--device <device name>".format(EXEC_NAME))
    print("\t -f|--fw_strings <fw strings db file>")
    print("\t --tracer_mode <FIFO | MEM>")
    print("\t --real_ts")
    print("\t -i|--irisc <irisc name>")
    print("\t -t|--tile <risc>.<tile>")
    print("\t -a|--apu <apu event name>")
    print("\t --include_phy_uc")
    print("\t -s|--stream")
    print("\t -n|--snapshot")
    print("\t -c|--cfg <mlxtrace cfg file>")
    print("\t -S|--buf_size <buffer size>")
    print("\t--dump <.trc dump file>")
    print("\t -m|--mask <class1+class2+...classN>")
    print("\t -l|--level <trace level>")
    print("\t --log_delay <delay in uSec>")
    print("\t --ignore_old_events")
    print("\t --keep_cfg")
    print("\t --keep_running")
    print("\t --config_only")
    print("\t --fw_cfg_only")
    print("")

    print("Run with \"-h\" to see the full list of iriscs and trace classes")
    print("Run with \"-d <device name> -h\" to see the specific device iriscs and trace classes")


#######################################################
HELP_DESC = """\
    -h|--help                    Print this help message and exit
    -d|--device                  Mst device name
    -f|--fw_strings              Fw strings db file containing the FW strings
      |--tracer_mode             Tracer mode [FIFO | MEM]
      |--real_ts                 Print real timestamps in [hh:mm:ss.nsec]
      |--gvmi                    Global virtual machine interface
      |--ignore_old_events       Ignore collecting old events
      |--keep_cfg                Does not remove the cfg file at the end of the run.
      |--mem_access              Memory access method: OB_GW, VMEM, UDRIVER
      |--keep_running            Keep the HW tracer unit running after exit
      |--config_only             Configure tracer and exit
      |--fw_cfg_only             Skip HW config and only configure FW events (default=off)\

format
    -i|--irisc                   Irisc name \
(See below for full list of irisc names)
    -t|--tile                    <risc>.<tile> name\
(Run -d <device name> -h to specific device support)
    -a|--apu                     Enable APU"
      |--include_phy_uc          enable phy_uc events with the risc events (main/tiles)"
    -s|--stream                  Run in streaming mode
    -c|--cfg                     HW tracer events cfg file
    -n|--snapshot                Take events snapshot - \
this assumes previous FW configurations
    -S|--buf_size                HW tracer MEM buffer size in [MB]
       --dump                    mlxtrace generated .trc file name
    -m|--mask                    Trace class mask, use \"+\" to enable \
multiple classes or use integer format, e.g: -m class1+class2+... or 0xff00ff00
    -l|--level                   Trace level
      |--log_delay               Fw tracer log delay in uSec
    -v|--version                 Print tool's version and exit
"""


def UpdateDevInfoMTEIM(devInfo, mstDevice):
    regAccessObj = regaccess.RegAccess(mstDevice)
    mteimData = None
    if regAccessObj:
        mteimData = regAccessObj.sendMTEIM()

    if mteimData:
        devInfo.max_main_irisc = mteimData["cap_core_main"]
        devInfo.max_apu = mteimData["cap_core_dpa"]
        devInfo.max_num_of_tiles = mteimData["cap_num_of_tile"]
        devInfo.max_irisc_per_tile = mteimData["cap_core_tile"]


def printAvailableEvents(devInfo):
    # main IRISC
    irisc_names = []
    for i in range(0, devInfo.max_main_irisc):
        if(i == 1):
            irisc_names.append("iron")
        else:
            irisc_names.append("i{}".format(i))
    if(len(irisc_names) > 0):
        irisc_names.append("all")
        print("        Irisc names: [%s]\n" % ", ".join(irisc_names))

    # APU events
    if(devInfo.max_apu > 0):  # device supports seperate APU event
        print("        APU events identifiers [a0...a{}]".format(devInfo.max_apu))

    # Tiles IRISC
    if(devInfo.max_irisc_per_tile > 0):  # device support event seperation between main and tiles
        print("        Number of Tiles: {}, Irisc per Tile: {}".format(devInfo.max_num_of_tiles, devInfo.max_irisc_per_tile))
        print("        Tile events format <risc>.<tile>, accept number in range or \"all\"\n")


def printDevHelp(devInfo):
    print("\nDevice Specific Info:")
    print("====================")
    print("    %s:" % devInfo.name)
    # Print irisc names
    printAvailableEvents(devInfo)

    # Print itrace classes
    trace_levels = []
    for mask in devInfo.mask_classes:
        trace_levels.append(mask[0])

    print("    Trace classes:")
    for i in range(0, len(trace_levels), 5):
        print("            " + ", ".join(trace_levels[i: i + 5]))


def Help():
    print(HELP_DESC)

    if DEV_NAME:
        mstDev = getMstDeviceFromName(DEV_NAME)
        devInfo = GetDeviceInfo(mstDev)
        UpdateDevInfoMTEIM(devInfo, mstDev)
        printDevHelp(devInfo)

#######################################################


def CheckFwStringsDBSignature(devInfo, dev, fwStrDBContents):
    fw_str_db_signature_exists = devInfo.fw_str_db_signature_exists
    if not fw_str_db_signature_exists:
        return

    signAddr = devInfo.get_fw_str_db_signature_addr()
    fwSign = dev.readField(signAddr[0], signAddr[1], signAddr[2])
    m = md5(fwStrDBContents)
    fileSign = int(m.hexdigest(), 16) & 0xffff
    if fwSign != fileSign:
        TracerException(
            "Fw strings db file signature: 0x%04x doesn't not match image "
            "signature: 0x%04x" % (fileSign, fwSign))


def createCfgHeader(devInfo, isExternal):
    if isExternal:
        return "OP1 {}\n".format(devInfo.name)
    else:
        return """# DEVICE TYPE:
DEVICE  {}

# EVENTS:
##################################################
""".format(devInfo.name)

# this function extract the numeric list of main IRISC events,
# it does not check the numbers as we already did it in CheckArgs


def extractMainIRISCEvents(devInfo):
    irisc_events = []
    if "all" in IRISC_NAME_LIST:
        for i in range(0, devInfo.max_main_irisc):
            irisc_events.append(i)
    else:  # user only want specific events
        for event in IRISC_NAME_LIST:
            if event == "iron":
                irisc_events.append(1)
            else:
                matchPhrase = re.search("^i*(\\d+)", event)
                if(matchPhrase):
                    irisc_events.append(int(matchPhrase.group(1)))
                else:
                    raise TracerException("Unknown irisc: %s" % (event))

    return irisc_events


def extractTileEvents(devInfo):
    tileEvents = {}  # key = tile number, value = list of enabled IRISC for that tile.
    for event in TILE_IRISC_NAME_LIST:
        tileUserEvent = parseTileEvent(devInfo, event)
        if(tileUserEvent):
            trace, tile = tileUserEvent
            if "all" in tile:
                for t in range(0, devInfo.max_num_of_tiles):
                    tileEvents[t] = []
                    if "all" in trace:
                        for i in range(0, devInfo.max_irisc_per_tile):
                            tileEvents[t].append(i)
                    else:
                        tileEvents[t].append(int(trace))
            else:
                tileEvents[int(tile)] = []
                if "all" in trace:
                    for i in range(0, devInfo.max_irisc_per_tile):
                        tileEvents[int(tile)].append("{}".format(i))
                else:
                    tileEvents[int(tile)].append(int(trace))
    return tileEvents


def extractAPUIRISCEvents(devInfo):
    apu_events = []
    if APU_ENABLE:
        for i in range(0, devInfo.max_apu):
            apu_events.append(i)
# APU is now all or nothing , we can't enable/disable some of the apu.
    # if "all" in APU_NAME_LIST:
    #     for i in range(0,devInfo["maxAPU"]):
    #         apu_events.append(i)
    # else: #user only want specific events
    #     for event in APU_NAME_LIST:
    #         matchPhrase = re.search("^a*(\d+)",event)
    #         if(matchPhrase):
    #             apu_events.append(int(matchPhrase.group(1)))

    return apu_events


def createCfgEvents(devInfo, isExternal):
    events = ""
    count = 0
    # create MAIN IRISC events
    eventStringFormat = "OP4 {} {}\n" if isExternal else "EVENT   {:<36}{}\n"
    userEvents = extractMainIRISCEvents(devInfo)
    for i in range(0, devInfo.max_main_irisc):
        enable = 1 if i in userEvents else 0
        count += enable
        events += eventStringFormat.format("ITRACE{}".format(i), enable)
    if(devInfo.supports_phy_uc):
        enable = 1 if INCLUDE_PHY_UC else 0
        count += enable
        events += eventStringFormat.format("Main_PhyUC", enable)
    # create TILES IRISC events
    tileEvents = extractTileEvents(devInfo)
    for t in range(0, devInfo.max_num_of_tiles):
        for i in range(0, devInfo.max_irisc_per_tile):
            enable = 0
            if (t in tileEvents) and (i in tileEvents[t]):
                enable = 1
            count += enable
            events += eventStringFormat.format("TILE{}_ITRACE{}".format(t, i), enable)
        if(devInfo.supports_phy_uc):
            enable = 1 if INCLUDE_PHY_UC else 0
            count += enable
            events += eventStringFormat.format("TILE{}_PhyUC".format(t), enable)
    # add APU
    apuEvents = extractAPUIRISCEvents(devInfo)
    for i in range(0, devInfo.max_apu):
        enable = 1 if i in apuEvents else 0
        count += enable
        events += eventStringFormat.format("APUTRACE{}".format(i), enable)
    return events, count

#######################################################


def GetCfgFile(devInfo):
    effCfgFile = GetTmpDir() + os.sep + "itrace_%d.cfg" % os.getpid()
    f = open(effCfgFile, "w+")
    isExt = IsExternal()
    f.write(createCfgHeader(devInfo, isExt))
    fwEvents, enabledEventsCount = createCfgEvents(devInfo, isExt)
    f.write(fwEvents)
    if (enabledEventsCount == 0):
        raise TracerException("no supported FW events has been enabled")

    if MLXTRACE_CFG:
        try:
            hwCfgLines = open(MLXTRACE_CFG, "r").readlines()
            mergedCfg = ""
            for line in hwCfgLines:
                if ("ITRACE" in line) or ("APUTRACE" in line) or ("_PhyUC" in line) or ("DEVICE" in line):
                    continue
                else:
                    mergedCfg += line
            f.write(mergedCfg)
        except Exception as exp:
            f.close()
            raise TracerException(str(exp))

    f.close()
    return effCfgFile


def IsNumeric(str):
    try:
        int(str, 0)
    except BaseException:
        return False  # also dont accept float inputs (by design)

    return True


#######################################################
def ApplyMask(devInfo, cmdifdev):
    if not MASK:
        return 0

    level = int(LEVEL)
    log_delay = 0
    if(LOG_DELAY):
        log_delay = int(LOG_DELAY)
    if IsNumeric(MASK):
        mask = int(MASK, 0)
    else:
        maskClasses = devInfo.mask_classes
        reqClasses = MASK.split("+")
        mask = 0
        for reqClass in reqClasses:
            found = False
            for c in maskClasses:
                if c[0] == reqClass:
                    mask += 1 << c[1]
                    found = True
                    break
            if not found:
                raise TracerException("Unknown trace class: %s" % reqClass)

    if cmdifdev:
        cmdifdev.setItrace(mask, level, log_delay)


#######################################################
def GetTracerMode(devInfo):
    if TRACER_MODE:
        return TRACER_MODE
    else:
        if CONFIG_ONLY:  # config_only is only working in FIFO no point in setting different default mode
            return "FIFO"
        else:
            return devInfo.default_tracer_mode

#######################################################


def GetMemAccessModeFlag():
    return "-a {}".format(MEMACCESS_MODE) if MEMACCESS_MODE else ""


def GetFwConfigOnlyFlag():
    return "--fw_cfg_only" if FW_CFG_ONLY else ""


def GetRealTimestampFlag():
    return "--real_ts" if REAL_TS else ""


def GetGvmiFlag():
    return "--gvmi={}".format(GVMI) if GVMI != 0 else ""


def GetIgnoreOldEventsFlag():
    return "--ignore_old_events" if IGNORE_OLD_EVENTS else ""

#######################################################


def GetConfigOnlyFlag():
    return "--config_only" if CONFIG_ONLY else ""


def GetKeepRunningFlag():
    return "--keep_running" if KEEP_RUNNING else ""


def GetSnapShotFlag():
    return "-n" if SNAPSHOT_MODE else ""

#######################################################


def CreateDataTlv(startAddr, sectionData):
    return struct.pack(">IIIIII", 1, len(sectionData) + 16,
                       startAddr, 0, 0, 0) + sectionData


ITOC_RE = re.compile(
    "======== .*?itoc_entry ========.*?param0\\s+:\\s+(\\S+)"
    ".*?cache_line_crc\\s+:\\s+(\\S+).*?itoc_entry_crc\\s+:\\s+(\\S+)"
    ".*?/(\\S+)-(\\S+)\\s+\\((\\S+)\\)/\\s+\\((\\S+)\\).*?CRC IGNORED$",
    re.DOTALL | re.MULTILINE)


#######################################################
def GetFwImageCrc():
    cmd = "flint -d %s %s v showitoc" % (DEV_NAME, FLINT_OCR)
    rc, out = GetStatusOutput(cmd)
    if rc:
        return "secured"

    totalCrc = ""
    for m in ITOC_RE.finditer(out):
        loadAddr, cacheLineCrc, crc, startAddr, endAddr, size, name = \
            m.groups()
        if name.endswith("_CODE"):
            totalCrc += "%x{}".format(int(crc, 16))
    return totalCrc


#######################################################
def RemoveCacheLineCRC(data):
    # For each 64 data bytes we have 4 bytes crc
    # (actually crc16 and 2 bytes are reserved)
    newData = b""
    # if (len(data) % 68):
    #    print "-E- Fatal error section size isn't multiple of 68 bytes"
    for i in range(0, len(data), 68):
        if i + 68 >= len(data):
            break
        newData = b"".join([newData, data[i:i + 64]])

    return newData


#######################################################
def GetTmpDir():
    if IsWindows():
        return "c:\\tmp"
    else:
        return "/tmp"


def close_mst_dev():
    global MST_DEVICE
    global DUMP_FILE
    if DUMP_FILE:
        return
    if MST_DEVICE is not None:
        MST_DEVICE.close()
        MST_DEVICE = None


def open_mst_dev():
    global MST_DEVICE
    global CMDIFDEV
    global DUMP_FILE
    if DUMP_FILE:
        return
    if MST_DEVICE is None:
        MST_DEVICE = mtcr.MstDevice(DEV_NAME)
        if CMDIFDEV is not None:
            CMDIFDEV = cmdif.CmdIf(MST_DEVICE)


#######################################################
def GetStringsDBwithFlint(cmdifdev, mst_device, cacheFName):
    cmd = "flint -d %s %s v showitoc" % (DEV_NAME, FLINT_OCR)
    rc, out = GetStatusOutput(cmd)
    if rc:
        print(cmd)
        close_mst_dev()
        raise TracerException(
            "Failed to extract fw strings db file from image: %s" % out)

    tlvData = b""
    for m in ITOC_RE.finditer(out):
        sys.stdout.write(".")
        sys.stdout.flush()
        loadAddr, cacheLineCrc, crc, startAddr, endAddr, size, name = \
            m.groups()
        loadAddr = int(loadAddr, 16)
        cacheLineCrc = int(cacheLineCrc, 16)
        startAddr = int(startAddr, 16)
        endAddr = int(endAddr, 16)
        size = int(size, 16)

        if name.endswith("_CODE") and name != "ROM_CODE":
            tmpFile = tempfile.NamedTemporaryFile()
            tmpFileName = tmpFile.name
            tmpFile.close()

            cmd = "flint -d %s %s rb 0x%x 0x%x %s" % \
                (DEV_NAME, FLINT_OCR, startAddr, size, tmpFileName)
            rc, out = GetStatusOutput(cmd)
            if rc:
                print(cmd)
                tmpFile.close()
                close_mst_dev()
                raise TracerException(
                    "Failed to read from flash: %s" % out)
            tmpFile = open(tmpFileName, "rb")
            data = tmpFile.read()
            tmpFile.close()

            if cacheLineCrc == 1:
                data = RemoveCacheLineCRC(data)

            tmp = CreateDataTlv(loadAddr, data)
            tlvData = b"".join([tlvData, tmp])

    print(".")
    # save cache file
    if tlvData == "":
        close_mst_dev()
        raise TracerException(
            "Failed to read fw strings db section from flash: %s" % out)

    fd = open(cacheFName, "w+b")
    fd.write(tlvData)
    fd.close()
    return tlvData


def GetStringsDBwithMtrc(cacheFName, mtrc_cap, regAccessObj):
    sys.stdout.write(".")
    sys.stdout.flush()
    tlvData = b""
    for i, string_db_parameters in enumerate(mtrc_cap["string_db_param"]):
        sys.stdout.write(".")
        sys.stdout.flush()
        data = b""
        if string_db_parameters["string_db_size"] != 0:
            data = regAccessObj.getMtrcStdbStringDbData(string_db_index=i,
                                                        read_size=string_db_parameters["string_db_size"])
            tmp = CreateDataTlv(string_db_parameters["string_db_base_address"], data)
            tlvData = b"".join([tlvData, tmp])

    # save cache file
    if tlvData == "":
        close_mst_dev()
        raise TracerException(
            "Failed to read fw strings db section from flash: %s" % out)

    fd = open(cacheFName, "w+b")
    fd.write(tlvData)
    fd.close()
    return tlvData


def GetFwStringsDBContents(devInfo, cmdifdev, mst_device):
    if FW_STR_DB_FILE:
        fd = open(FW_STR_DB_FILE, "rb")
        out = fd.read()
        fd.close()
        return out

    else:
        # create reaccess object
        regAccessObj = regaccess.RegAccess(mst_device)
        fwVer = ""
        # if regaccess obj exist, get the fw version through regaccess mgir (support dev branch)
        if regAccessObj:
            fwVer = regAccessObj.getFwVersion()

        # in case that the regacces failed or has no version information
        if fwVer == "":
            if cmdifdev:
                fwInfo = cmdifdev.getFwInfo()
                fwVer = "%d.%d.%d" % (fwInfo.MAJOR, fwInfo.MINOR, fwInfo.SUBMINOR)
            else:
                close_mst_dev()
                cmd = "flint -d %s %s -qq q | grep \"FW Version\"" % \
                    (DEV_NAME, FLINT_OCR)
                rc, out = GetStatusOutput(cmd)
                if rc:
                    print(cmd)
                    close_mst_dev()
                    raise TracerException("Failed to read FW version: %s" % out)
                fwVer = out.split(":")[-1].strip()

        # Try to find the fw string db in cache file
        user = getpass.getuser()
        imageCrc = GetFwImageCrc()
        cacheFName = GetTmpDir() + os.sep + \
            "fwtrace_str_db_cache_%s_%s_%s" % \
            (user, fwVer.replace(".", "_"), imageCrc)
        userCacheFiles = glob.glob(GetTmpDir() + os.sep +
                                   "fwtrace_str_db_cache_%s_*" % user)

        if len(userCacheFiles) > 0:
            # fix cache files, only one should be available for each user
            if len(userCacheFiles) > 1:
                for f in userCacheFiles:
                    try:
                        os.remove(f)
                    except BaseException:
                        pass

            if userCacheFiles[0] == cacheFName:
                data = open(cacheFName, "rb").read()
                if data != "":
                    print("-I- Found FW string db cache file, going to use it")
                    return data
            else:
                os.remove(userCacheFiles[0])

        # cache file wasn't found
        sys.stdout.write("Reading FW strings.")
        sys.stdout.flush()

        mtrc_cap = regAccessObj.getMtrcCap()
        mtrc_stdb = regAccessObj.isMtrcStdbSupported()
        mft_core_device = MftCoreDevice(mst_device.read_device_id())
        if mtrc_cap == False or mtrc_stdb == False or mft_core_device._is_switch():
            return GetStringsDBwithFlint(cmdifdev, mst_device, cacheFName)
        else:
            try:
                return GetStringsDBwithMtrc(cacheFName, mtrc_cap, regAccessObj)
            except Exception as e:
                print("Failed to read DB with register, defaulting back to read from flash. reason {}.\n".format(str(e)))
                return GetStringsDBwithFlint(cmdifdev, mst_device, cacheFName)


#######################################################
def Clean():
    close_mst_dev()
    fwStrDBFile = GetTmpDir() + os.sep + "fw_str_db_%d.csv" % os.getpid()
    if os.path.exists(fwStrDBFile):
        os.remove(fwStrDBFile)

    itraceTxtFile = GetTmpDir() + os.sep + "itrace_%d.txt" % os.getpid()
    if os.path.exists(itraceTxtFile):
        os.remove(itraceTxtFile)

    itraceBinFile = GetTmpDir() + os.sep + "itrace_%d.trc" % os.getpid()
    if os.path.exists(itraceBinFile):
        os.remove(itraceBinFile)

    if not KEEP_CFG:
        cfgFile = GetTmpDir() + os.sep + "itrace_%d.cfg" % os.getpid()
        if os.path.exists(cfgFile):
            os.remove(cfgFile)


def GetSkipOwnershipFlag():
    global MST_DEVICE
    skipOwnershipFlag = ""
    if MST_DEVICE is None:
        return skipOwnershipFlag

    regAccessObj = regaccess.RegAccess(MST_DEVICE)
    rc = regAccessObj.sendMtrcCapTakeOwnership()

    if rc == regaccess.ownershipEnum.REG_ACCESS_FAILED_TO_AQUIRE_OWNERSHIP:
        print("Failed to acquire ownership.")
    elif rc == regaccess.ownershipEnum.REG_ACCESS_NO_OWNERSHIP_REQUIRED:
        print("No ownership taking is required.")
    else:
        print("Got ownership successfully!")
        skipOwnershipFlag = "--skip_ownership"
    return skipOwnershipFlag


def getMstDeviceFromName(devName):
    MST_DEVICE = mtcr.MstDevice(devName)
    if MST_DEVICE.is_cable() or MST_DEVICE.is_linkx():
        raise TracerException("Device is not supported")
    return MST_DEVICE


#######################################################
def StartTracer():
    try:
        global CMDIFDEV
        global MST_DEVICE
        ParseCmdLineArgs()
        if RUN_HELP:
            Help()
            return 0
        if DEV_NAME is None and DUMP_FILE is None:
            Usage()
            return 1

        if DEV_NAME:
            MST_DEVICE = getMstDeviceFromName(DEV_NAME)  # DEV_NAME is the mst device name Exmple /mst/dev/mt4123_pciconf0
            devInfo = GetDeviceInfo(MST_DEVICE)  # MST_DEVICE is the opened mfile , devInfo = info from utils DB.
            CMDIFDEV = cmdif.CmdIf(MST_DEVICE)
            UpdateDevInfoMTEIM(devInfo, MST_DEVICE)
        else:
            devInfo = GetDeviceInfoFromDumpFile(DUMP_FILE)

        tokenApplied = False
        if(MST_DEVICE):
            regAccessObj = regaccess.RegAccess(MST_DEVICE)
            tokenApplied = regAccessObj.isCsTokenApplied()
        if fw_trace_utilities.FwTraceUtilities.is_secure_fw(MST_DEVICE) and devInfo._mft_core_device._is_nic() and not tokenApplied:
            # go to secure flow only when secured and no token
            # switchs always skip this due to lack of proper secure driver support (no mlx5 on switchs)
            success_indication = 0
            if fw_trace_utilities.FwTraceUtilities.is_driver_mem_mode_supported():
                try:
                    CheckSecureFwArgs(devInfo)
                    secure_fw_tracer = SecureFwTrace(MST_DEVICE, DEV_NAME, IGNORE_OLD_EVENTS, REAL_TS)
                    open_mst_dev()
                    ApplyMask(devInfo, CMDIFDEV)
                    secure_fw_tracer.parse_driver_mem()
                except Exception as exp:
                    print("-E- %s" % exp)
                    success_indication = 1
            else:
                raise TracerException("Driver mem mode is not supported")
                success_indication = 1

            return success_indication

        CheckArgs(devInfo)
        fwStrDBContents = GetFwStringsDBContents(devInfo, CMDIFDEV, MST_DEVICE)
        open_mst_dev()

        # must get ownership before apply mask
        generalFlags = GetMemAccessModeFlag() + " " + GetSkipOwnershipFlag() + " " + GetFwConfigOnlyFlag() + " " + \
            GetRealTimestampFlag() + " " + GetGvmiFlag() + " " + GetIgnoreOldEventsFlag()

        if DEV_NAME:
            CheckFwStringsDBSignature(devInfo, MST_DEVICE, fwStrDBContents)
            ApplyMask(devInfo, CMDIFDEV)
        cfgFile = GetCfgFile(devInfo)
        tracerMode = GetTracerMode(devInfo)
        fwParser = fwparse.FwTraceParser(MST_DEVICE, devInfo, fwStrDBContents)
        close_mst_dev()

        if STRAMING_MODE:
            cmdArgList = [MLXTRACE_EXE, "-d", DEV_NAME, "-m", tracerMode, generalFlags, "-c", cfgFile, "-S"]
            cmd = " ".join(cmdArgList).split()
            print(" ".join(cmd))  # remove multiple spaces from missing optional arguments for a cleaner print.
            global proc
            if IsWindows():
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        shell=True)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            try:
                while True:
                    if proc.poll():
                        for line in proc.stdout.readlines():
                            fwParser.pushLine(line)
                        print("Stopping ...")
                        break
                    else:
                        line = proc.stdout.readline()
                        fwParser.pushLine(line)
            except KeyboardInterrupt:
                try:
                    proc.terminate()
                    proc.wait()
                except BaseException:
                    pass
                print("\nStopping & Flushing... "
                      "(messages below may have missing arguments)")
                fwParser.flushAll()
        else:
            if not DUMP_FILE:
                nonStreamFlags = GetConfigOnlyFlag() + " " + GetKeepRunningFlag() + " " + GetSnapShotFlag()
                outputfilepathFlag = "-o {}{}itrace_{}.trc".format(GetTmpDir(), os.sep, os.getpid())
                cmdArgList = [MLXTRACE_EXE, "-d", DEV_NAME, "-m", tracerMode, generalFlags,
                              nonStreamFlags, "-c", cfgFile, outputfilepathFlag]
                cmd = " ".join(cmdArgList)

                if BUF_SIZE:
                    cmd += " --buf_size=%s" % BUF_SIZE

                cmd = " ".join(cmd.split())  # remove multiple spaces from missing optional arguments for a cleaner print.
                print(cmd)
                rc = os.system(cmd)
                if os.name != "nt":
                    if rc & 0xff:
                        print("Command was interrupted with signal %d" % (rc & 0xff))
                    rc = rc >> 8
                if rc:
                    raise TracerException("Failed while running mlxtrace (rc = %d)" % rc)

            if DUMP_FILE:
                withDev = "-d {}".format(DEV_NAME) if DEV_NAME else ""
                cmd = "%s %s -p %s -i %s > %s%sitrace_%d.txt" % \
                    (MLXTRACE_EXE, withDev, GetRealTimestampFlag(), DUMP_FILE, GetTmpDir(), os.sep,
                     os.getpid())
            else:
                withDev = "-d {}".format(DEV_NAME) if DEV_NAME else ""
                dmp_file = "%s%sitrace_%d.trc" % \
                    (GetTmpDir(), os.sep, os.getpid())
                if not os.path.exists(dmp_file):
                    print("\n-I- No dump file were generated, exiting ...")
                    return 0
                cmd = "%s %s -p %s -i %s > %s%sitrace_%d.txt" % \
                    (MLXTRACE_EXE, withDev, GetRealTimestampFlag(), dmp_file, GetTmpDir(), os.sep,
                     os.getpid())

            print(cmd)
            if os.system(cmd):
                raise TracerException("Failed to parse mlxtrace dump file")

            if os.path.exists("%s%sitrace_%d.txt" %
                              (GetTmpDir(), os.sep, os.getpid())):
                fd = open("%s%sitrace_%d.txt" %
                          (GetTmpDir(), os.sep, os.getpid()), "rb")
                try:
                    for line in fd:
                        fwParser.pushLine(line)
                except KeyboardInterrupt:
                    print("\nStopping & Flushing... "
                          "(messages below may have missing arguments)")
                    fwParser.flushAll()

                fwParser.flushAll()
                fd.close()

        Clean()
    except Exception as exp:
        print("-E- %s" % exp)
        Clean()
        return 1

    return 0


#######################################################
if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal_handler)
        rc = StartTracer()
    except Exception as exp:
        try:
            print("-E- %s" % str(exp))
        except BaseException:
            pass
        rc = 1
    sys.exit(rc)
