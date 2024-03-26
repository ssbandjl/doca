# Copyright (c) 2004-2010 Mellanox Technologies LTD. All rights reserved.
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the
# OpenIB.org BSD license below:
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#      - Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#
#      - Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials
#        provided with the distribution.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --
# sgude 1_22_2021

# Performed modifications for NVDV5
from enum import IntEnum
import ctypes
import struct
import jtag
import re
import math
import time
import os
import sys
import argparse
import subprocess

FTDI_ERROR_CODES = ["FT_OK",
                    "FT_INVALID_HANDLE",
                    "FT_DEVICE_NOT_FOUND",
                    "FT_DEVICE_NOT_OPENED",
                    "FT_IO_ERROR",
                    "FT_INSUFFICIENT_RESOURCES",
                    "FT_INVALID_PARAMETER",
                    "FT_INVALID_BAUD_RATE",
                    "FT_DEVICE_NOT_OPENED_FOR_ERASE",
                    "FT_DEVICE_NOT_OPENED_FOR_WRITE",
                    "FT_FAILED_TO_WRITE_DEVICE",
                    "FT_EEPROM_READ_FAILED",
                    "FT_EEPROM_WRITE_FAILED",
                    "FT_EEPROM_ERASE_FAILED",
                    "FT_EEPROM_NOT_PRESENT",
                    "FT_EEPROM_NOT_PROGRAMMED",
                    "FT_INVALID_ARGS",
                    "FT_NOT_SUPPORTED",
                    "FT_OTHER_ERROR",
                    "FT_DEVICE_LIST_NOT_READY"]


class BitMode(IntEnum):
    """Function selection."""
    RESET = 0x00    # switch off altnerative mode (default to UART)
    BITBANG = 0x01  # classical asynchronous bitbang mode
    MPSSE = 0x02    # MPSSE mode, available on 2232x chips
    SYNCBB = 0x04   # synchronous bitbang mode
    MCU = 0x08      # MCU Host Bus Emulation mode,
    OPTO = 0x10     # Fast Opto-Isolated Serial Interface Mode
    CBUS = 0x20     # Bitbang on CBUS pins of R-type chips
    SYNCFF = 0x40   # Single Channel Synchronous FIFO mode



class MpsseCmd(IntEnum):
    """MPSSE (Multi-Protocol Synchronous Serial Engine)"""
    WRITE_BYTES_PVE_MSB = 0x10
    WRITE_BYTES_NVE_MSB = 0x11
    WRITE_BITS_PVE_MSB = 0x12
    WRITE_BITS_NVE_MSB = 0x13
    WRITE_BYTES_PVE_LSB = 0x18
    WRITE_BYTES_NVE_LSB = 0x19
    WRITE_BITS_PVE_LSB = 0x1a
    WRITE_BITS_NVE_LSB = 0x1b
    READ_BYTES_PVE_MSB = 0x20
    READ_BYTES_NVE_MSB = 0x24
    READ_BITS_PVE_MSB = 0x22
    READ_BITS_NVE_MSB = 0x26
    READ_BYTES_PVE_LSB = 0x28
    READ_BYTES_NVE_LSB = 0x2c
    READ_BITS_PVE_LSB = 0x2a
    READ_BITS_NVE_LSB = 0x2e
    RW_BYTES_PVE_NVE_MSB = 0x31
    RW_BYTES_NVE_PVE_MSB = 0x34
    RW_BITS_PVE_PVE_MSB = 0x32
    RW_BITS_PVE_NVE_MSB = 0x33
    RW_BITS_NVE_PVE_MSB = 0x36
    RW_BITS_NVE_NVE_MSB = 0x37
    RW_BYTES_PVE_NVE_LSB = 0x39
    RW_BYTES_NVE_PVE_LSB = 0x3c
    RW_BITS_PVE_PVE_LSB = 0x3a
    RW_BITS_PVE_NVE_LSB = 0x3b
    RW_BITS_NVE_PVE_LSB = 0x3e
    RW_BITS_NVE_NVE_LSB = 0x3f
    WRITE_BITS_TMS_PVE = 0x4a
    WRITE_BITS_TMS_NVE = 0x4b
    RW_BITS_TMS_PVE_PVE = 0x6a
    RW_BITS_TMS_PVE_NVE = 0x6b
    RW_BITS_TMS_NVE_PVE = 0x6e
    RW_BITS_TMS_NVE_NVE = 0x6f
    SEND_IMMEDIATE = 0x87
    WAIT_ON_HIGH = 0x88
    WAIT_ON_LOW = 0x89
    READ_SHORT = 0x90
    READ_EXTENDED = 0x91
    WRITE_SHORT = 0x92
    WRITE_EXTENDED = 0x93

class FtdiMpsseCmd(IntEnum):
    """FTDI MPSSE (Multi-Protocol Synchronous Serial Engine) commands"""
    SET_BITS_LOW = 0x80     # Change LSB GPIO output
    SET_BITS_HIGH = 0x82    # Change MSB GPIO output
    GET_BITS_LOW = 0x81     # Get LSB GPIO output
    GET_BITS_HIGH = 0x83    # Get MSB GPIO output
    LOOPBACK_START = 0x84   # Enable loopback
    LOOPBACK_END = 0x85     # Disable loopback
    SET_TCK_DIVISOR = 0x86  # Set clock

class MpsseFlowControl(IntEnum):
    # Flow control arguments
    SIO_DISABLE_FLOW_CTRL = 0x0
    SIO_RTS_CTS_HS = (0x1 << 8)
    SIO_DTR_DSR_HS = (0x2 << 8)
    SIO_XON_XOFF_HS = (0x4 << 8)
    SIO_SET_DTR_MASK = 0x1
    SIO_SET_DTR_HIGH = (SIO_SET_DTR_MASK | (SIO_SET_DTR_MASK << 8))
    SIO_SET_DTR_LOW = (0x0 | (SIO_SET_DTR_MASK << 8))
    SIO_SET_RTS_MASK = 0x2
    SIO_SET_RTS_HIGH = (SIO_SET_RTS_MASK | (SIO_SET_RTS_MASK << 8))
    SIO_SET_RTS_LOW = (0x0 | (SIO_SET_RTS_MASK << 8))

class SDDV_Jtag:
    """Class for interfacing with FTDI 2322x modules via USB/JTAG"""
    state = jtag.states.unknown
    instr = []

    def __init__(self, device_index=0, toggle_TRST=False, cycle_port = False):

        self.device_handle = ctypes.c_void_p()

        self._libraries = None

        if os.name == 'posix':
            try:
                subprocess.run(["modprobe", "-r", "ftdi_sio"], stdout=subprocess.DEVNULL)
                # subprocess.Popen(["modprobe", "-r", "usbserial"], stdout=subprocess.DEVNULL)
            except BaseException:
                pass

            try:
                self._libraries = ctypes.CDLL('libftd2xx.so')
            except BaseException:
                self._libraries = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'ext_libs', 'libftd2xx.so'))
        else:
            self._libraries = ctypes.WinDLL('ftd2xx')

        # Open Device.
        USBftStatus = self._libraries.FT_Open(device_index, ctypes.byref(self.device_handle))
        if USBftStatus:
            raise Exception("Unable to Open device: " + FTDI_ERROR_CODES[USBftStatus])

        dev = ctypes.c_void_p()
        id = ctypes.c_void_p()
        sn = ctypes.create_string_buffer(16)
        desc = ctypes.create_string_buffer(64)

        # Cycle port to overcome that it is stuck - on demand from init. Device must be reinitialized afterwards.
        if cycle_port:
            USBftStatus = self._libraries.FT_CyclePort(self.device_handle)
            if USBftStatus:
                raise Exception("Unable to Open device: " + FTDI_ERROR_CODES[USBftStatus])
            self._libraries.FT_Close(self.device_handle)
            return

        # Get Device Info.
        USBftStatus = self._libraries.FT_GetDeviceInfo(self.device_handle, ctypes.byref(dev), ctypes.byref(id), sn,
                                                       desc, None)
        if USBftStatus:
            raise Exception("Unable to query device: " + FTDI_ERROR_CODES[USBftStatus])

        self.device = dev.value
        self.id = id.value
        self.serial = sn.value
        self.description = desc.value

        # Reset Device.
        USBftStatus = self._libraries.FT_ResetDevice(self.device_handle)
        if USBftStatus:

            raise Exception("Reset device failed: " + FTDI_ERROR_CODES[USBftStatus])

        FT_PURGE_RX = 1
        FT_PURGE_TX = 2

        # Purge RX & TX Buffers.
        USBftStatus = self._libraries.FT_Purge(self.device_handle, FT_PURGE_RX | FT_PURGE_TX)
        if USBftStatus:
            raise Exception("Purge device failed: " + FTDI_ERROR_CODES[USBftStatus])

        # Set Transfer size for USB IN & OUT request to 0x10000.
        USBftStatus = self._libraries.FT_SetUSBParameters(self.device_handle, 65536, 65536)
        if USBftStatus:
            raise Exception("SetUSBParameters failed: " + FTDI_ERROR_CODES[USBftStatus])

        USBftStatus = self._libraries.FT_SetChars(self.device_handle, False, 0, False, 0)
        if USBftStatus:
            raise Exception("SetChars failed: " + FTDI_ERROR_CODES[USBftStatus])
        # Set read & write T/Os to 500 ms.
        USBftStatus = self._libraries.FT_SetTimeouts(self.device_handle, 500, 500)
        if USBftStatus:
            raise Exception("SetTimeout failed: " + FTDI_ERROR_CODES[USBftStatus])

        USBftStatus = self._libraries.FT_SetLatencyTimer(self.device_handle, 1)
        if USBftStatus:
            raise Exception("SetLatencyTimer failed: " + FTDI_ERROR_CODES[USBftStatus])

        # Set flow control to DTR/DSR hardware flow
        USBftStatus = self._libraries.FT_SetFlowControl(self.device_handle, MpsseFlowControl.SIO_DTR_DSR_HS, 0,
                                                        0)  # 500ms read/write timeout
        if USBftStatus:
            raise Exception("SetFlowControl failed: " + FTDI_ERROR_CODES[USBftStatus])

        mask = 0x00
        mode = 0  # Reset the MPSSE controller. Perform a general reset on the MPSSE, not the port itself
        USBftStatus = self._libraries.FT_SetBitMode(self.device_handle, mask, mode)

        mask = 0x00
        mode = 2  # enable MPSSE mode
        USBftStatus = self._libraries.FT_SetBitMode(self.device_handle, mask, mode)

        if USBftStatus:
            raise Exception("SetBitMode failed: " + FTDI_ERROR_CODES[USBftStatus])

        # Set clock frequency to 1 Mhz
        outstr = struct.pack('BBB',  # 0x8B,  # 0x8B set 2232H fast/slow mode
                             FtdiMpsseCmd.SET_TCK_DIVISOR,  # setup tck divide
                             0x05,  # div L 12.5Hz
                             0x00)
        self._ft_write(outstr)

        # outstr = struct.pack('B', 0x8A) # uncomment to disable Clk Divide by 5, allowing 60 MHz clock.
        # self._ft_write(outstr)
        if toggle_TRST:
            self.toggle_TRST()
        outstr = struct.pack('BBB', FtdiMpsseCmd.SET_BITS_LOW, 0x20, 0xFB)  # TRST up
        self._ft_write(outstr)
        # Disable Loopback.
        self._ft_write(struct.pack('B', FtdiMpsseCmd.LOOPBACK_END))
        # set to 0x84 to enable LoopBack (This will connect the TDI/DO output to the TDO/DI input for loopback testing, without external device)

        outstr = self._write_tms('0', self.state.reset)
        self._ft_write(outstr)
        self.state = jtag.states.reset

        for t in self.instr:
            setattr(self, 'w_' + t[0], self.make_write_instr(t))
            setattr(self, 'r_' + t[0], self.make_read_instr(t))

                # Write data
        self.raw_ft_cmds = { 'write_reg_cmd_from_idle': { 'cmd' : b'K\x04\x063\x06\x04k\x00\x81' , 'num_bytes' : 0, 'num_bits' : 7,  'next_state' : jtag.states.exit1_ir },
                 'mv_idle_to_idle': { 'cmd' : b'K\x03\x00' , 'num_bytes' : 0, 'num_bits' : 0,  'next_state' : jtag.states.idle },
                 'mv_exit1_dr_to_idle': { 'cmd' : b'K\x04\x01' , 'num_bytes' : 0, 'num_bits' : 0,  'next_state' : jtag.states.idle },
                 'rd_data_and_mv_shift_dr_to_exit1_dr': { 'cmd' : b'K\x02\x011\x08\x00\x00\x91@\x07\x80\x00\x00\x00\x003\x04\x00k\x00\\x0<1' , 'num_bytes' : 9, 'num_bits' : 5,  'next_state' : jtag.states.exit1_dr },
               }


    def make_write_instr(self, instr_tuple):
        def instr():
            return self.write_ir(instr_tuple[3])
        return instr

    def make_read_instr(self, instr_tuple):
        m = re.search(r'com_([0-9]*)', instr_tuple[1])
        if m and m.group(1):
            num_nibbles = int(int(m.group(1)) / 4)

        def instr():
            return self.write_dr('0x' + num_nibbles * '0')
        return instr

    def purge_rx(self):
        FT_PURGE_RX = 1
        self._purge(buffer = FT_PURGE_RX)

    def purge_tx(self):
        FT_PURGE_TX = 2
        self._purge(buffer = FT_PURGE_TX)

    def _purge(self, buffer):
        # Purge RX & TX Buffers.
        USBftStatus = self._libraries.FT_Purge(self.device_handle, buffer)
        if USBftStatus:
            raise Exception("Purge device failed: " + FTDI_ERROR_CODES[USBftStatus])

    def toggle_TRST(self, times=5):
        for _ in range(times):
            outstr = struct.pack('BBB', FtdiMpsseCmd.SET_BITS_LOW, 0x09, 0xFB)  # TRST
            self._ft_write(outstr)
            outstr = struct.pack('BBB', FtdiMpsseCmd.SET_BITS_LOW, 0x08, 0xFB)  # TRST
            self._ft_write(outstr)

    def _write_tdi_bytes(self, byte_list, read=True):
        """
        Return the command packet needed to write a list of bytes over TDI.
        Bytes are written as-is, so should be LSB first for JTAG.
        """
        if not byte_list:
            return

        if len(byte_list) > 0xFFFF:
            raise Exception("Byte input list is too long!")

        for b in byte_list:
            if b > 0xFF:
                raise Exception("Element in list larger than 8-bit:", hex(b))

        length = len(byte_list) - 1
        length_upper = (length & 0xFF00) >> 8
        length_lower = length & 0x00FF
        byte_str = bytes((b for b in byte_list))

        if read:
           opcode = MpsseCmd.RW_BYTES_PVE_NVE_MSB
        else:
            opcode = MpsseCmd.WRITE_BYTES_NVE_MSB

        # outstr = chr(opcode) + struct.pack('BB', length_lower, length_upper) + byte_str #msb first
        outstr = bytes((opcode, length_lower, length_upper)) + byte_str  # msb first
        return outstr

    def _write_tdi_bits(self, bit_list, read=True):
        """
        Return the command packet needed to write a list of binary values
        over TDI.
        """
        if not bit_list:
            return

        bit_str = ''.join([str(b) for b in bit_list])
        bit_str = bit_str + '0' * (8 - len(bit_str))
        bit_int = int(bit_str, 2)
        length = len(bit_list)
        if length > 8:
            raise ValueError("Input string longer than 8 bits")

        if read:
            opcode = MpsseCmd.RW_BITS_PVE_NVE_MSB
        else:
            opcode = MpsseCmd.WRITE_BITS_NVE_MSB

        # outstr = chr(opcode) + struct.pack('BB', length - 1, bit_int) #msb first
        outstr = bytes((opcode, length - 1, bit_int))  # msb first
        return outstr

    def _write_tms(self, tdi_val, tms, read=False):
        """
        Return the command packet needed to write a single value to TDI and
        a single or multiple values to TMS.
        tdi_val: string or int: '1', '0', 1, 0
        tms: list, string or jtag.TMSPath: '10010', [1, 0, 0, 1, 0],
        ['1', '0', '0', '1', '0']
        """

        # TODO, make this work for lists longer than 7, maybe recurse?
        # also, formatting could be more consistant
        # Check tdi_val formatting

        # print(f'Current state = {self.state} TDI = {tdi_val}')
        # print(f'1 - tms = {tms}')

        tdi_int = int(tdi_val)
        if str(tdi_val) != '0' and str(tdi_val) != '1':
            raise Exception("tdi_val doesn't meet expected format:", tdi_val)

        # Convert tms to list to parse state transitions
        if isinstance(tms, str):
            tms = [int(s) for s in tms]

        tms_len = len(tms)

        if tms_len > 7:
            raise Exception("tms is too long", tms)

        if self.state is not jtag.states.unknown:
            for t in tms:
                # print(t, '{:<12}'.format(self.state), '->', self.state[t])
                self.state = self.state[t]
        # print(f'2 - tms = {tms}')

        if isinstance(tms, list) or isinstance(tms, jtag.TMSPath):
            tms = ''.join([str(x) for x in tms])

        tms = tms[::-1]  # LSb first
        # print(f'3 - tms = {tms}')
        tms_int = int(tms, 2)
        byte1 = (tdi_int << 7) | tms_int

        if read:
            opcode = MpsseCmd.RW_BITS_TMS_PVE_NVE
        else:
            opcode = MpsseCmd.WRITE_BITS_TMS_NVE

        # outstr = struct.pack('BBB',[opcode, tms_len - 1, byte1])
        # print((opcode, tms_len - 1, byte1))
        outstr = bytes((opcode, tms_len - 1, byte1))
        return outstr

    def _write(self, data, write_state=None, next_state=None, read=True, rebuild=True,context=''):
        """Default next state is shift-ir/dr"""

        byte_list, bit_list, last_bit, orig = self.to_bytes_bits(data)
        # print(f'byte_list = {byte_list} - {len(byte_list)}')
        # print(f'bit_list = {bit_list} - {len(bit_list)}')

        outstr = b''

        if write_state:
            outstr += self._write_tms(0, self.state[write_state])

        tmp = self._write_tdi_bytes(byte_list, read=read)

        if tmp:
            outstr += tmp
        tmp = self._write_tdi_bits(bit_list, read=read)

        if tmp:
            outstr += tmp
        tmp = b''

        if next_state:
            tmp = (self._write_tms(last_bit, self.state[next_state], read=read))
        else:
            tmp = (self._write_tms(last_bit, self.state[self.state], read=read))

        # for i in range(1):
        outstr += tmp

        # print(f'{context}: {outstr}')
        self._ft_write(outstr)

        if read:
            # time.sleep(1 / 1000)
            a = self._read(len(byte_list) + (1 if len(bit_list) else 0) + 1)
            if rebuild:
                rebuilt_data = self._rebuild_read_data(a[0], len(byte_list), len(bit_list))
                # print(f"rebuilt data: {rebuilt_data}")
                return self.bin2hex(rebuilt_data)

    def _write_cmd(self, cmd, read=True, rebuild=True, context=''):
        # print(f"{context}: {cmd['cmd']}")
        self._ft_write(cmd['cmd'])
        self.state = cmd['next_state']

        if read:
            # time.sleep(1 / 1000)
            a = self._read(cmd['num_bytes'] + (1 if cmd['num_bits'] else 0) + 1)
            if rebuild:
                rebuilt_data = self._rebuild_read_data(a[0], cmd['num_bytes'], cmd['num_bits'])
                # print(f"rebuilt data: {rebuilt_data}")
                return self.bin2hex(rebuilt_data)


    def _read(self, expected_num):
        """
        Read expected_num bytes from the FTDI chip, once there are that many
        availible in the buffer. Return the raw bytes as a tuple of binary and
        hex strings."""
        bytes_avail = ctypes.c_int()
        while bytes_avail.value != expected_num:
            if bytes_avail.value > expected_num:
                break
                # raise Exception(f"More bytes in buffer than expected {bytes_avail.value} > {expected_num}!")

            USBftStatus = self._libraries.FT_GetQueueStatus(self.device_handle, ctypes.byref(bytes_avail))
            if USBftStatus:
                raise Exception("GetQueueStatus failed: " + FTDI_ERROR_CODES[USBftStatus])

        readback = self._ft_read(bytes_avail.value)
        if readback:
            byte_tuple = struct.unpack('B' * len(readback), readback)
            bin_read_str = ''
            hex_read_str = ''
            if byte_tuple:
                bin_read_str = ''.join([bin(byte)[2:].zfill(8) for byte in byte_tuple])
                # print(f"bin raw data: {bin_read_str}")
                hex_read_str = ''.join([hex(byte)[2:].zfill(2) for byte in byte_tuple])
                # print(f"hex raw data: {hex_read_str}")

        return (bin_read_str, hex_read_str)

    def _flush(self):
        """
        Flush USB receive buffer
        """
        bytes_written = self._ft_write(struct.pack('B', MpsseCmd.SEND_IMMEDIATE))
        return bytes_written

    def _ft_write(self, outstr):
        """
        Low level call to ftdi dll
        """
        bytes_written = ctypes.c_int()
        USBftStatus = self._libraries.FT_Write(self.device_handle, ctypes.c_char_p(outstr), len(outstr),
                                               ctypes.byref(bytes_written))
        if USBftStatus:
            raise Exception("FT_Write failed: " + FTDI_ERROR_CODES[USBftStatus])
        return bytes_written.value

    def _ft_read(self, numbytes):
        """
        Low level call to ftdi dll
        """
        bytes_read = ctypes.c_int()
        inbuf = ctypes.create_string_buffer(numbytes)
        USBftStatus = self._libraries.FT_Read(self.device_handle, inbuf, numbytes, ctypes.byref(bytes_read))
        if USBftStatus:
            raise Exception("FT_Read failed: " + FTDI_ERROR_CODES[USBftStatus])
        return inbuf.raw

    def get_id(self):
        """Return the device ID of the part
        State machine transitions current -> reset -> shift_dr -> exit1_dr
        """
        outstr = self._write_tms('0', self.state.reset)
        self._ft_write(outstr)
        # device ID instruction opcode is 0x02
        outstr = self.write_ir('0x02')
        data = self.write_dr('0x00000000')
        # print(data)
        return data

    def reset(self):
        """
        Return to the reset state
        """
        outstr = self._write_tms('0', self.state.reset)
        # print(f'reset : {outstr}')
        self._ft_write(outstr)
        return

    def idle(self):
        """
        Go to idle state and clock once
        """
        # outstr = self._write_tms('0', j.state.idle.pad(minpause=2))
        outstr = self._write_tms('0', self.state.idle.pad(minpause=3))
        # print(f'idle : {outstr}')
        self._ft_write(outstr)
        return

    def write_ir(self, cmd, next_state=jtag.states.idle, read=True):
        """
        Write cmd while in the shift_ir state, return read back cmd
        """
        cmd_readback = self._write(cmd, write_state=jtag.states.shift_ir,
                                   next_state=next_state, read=read, rebuild=False, context='write_ir')
        return (cmd, cmd_readback)
    
    def write_generic_cmd(self, cmd, read=True):
        """"""
        cmd_readback = self._write_cmd(cmd, read=read, rebuild=False, context='write_generic')
        return (cmd, cmd_readback)

    def write_dr(self, data, next_state=jtag.states.idle, read=True):
        """Write data while in the shift_dr state, return read back data"""

        data_readback = self._write(data, write_state=jtag.states.shift_dr,

                                    next_state=next_state, read=read, context='write_dr')

        return (data, data_readback)

    def _sddv_frame_write(self, addr, data):
        """Write data to the fabric using the JTAG2HOST command."""
        return self._sddv_create_frame(addr, data, write=True)

    def _sddv_frame_read(self, addr):
        """Read data to the fabric using the JTAG2HOST command."""
        return self._sddv_create_frame(addr)
    def sddv_close(self):
        self._libraries.FT_Close(self.device_handle)

    def _sddv_create_frame(self, addr, data=0x0000, write=False):
        # JTAG2UNIT interface opcode is 0xA0
        reg_cmd = '0xA0'
        cya = '0'
        priv_level = '00'
        source_id = '00000'
        # Reset bit should always be set to 1, except for when resetting JTAG2Unit interface
        reset = '1'
        access_mode = '0'
        ack = '0'
        # Valid bit should always be set to 1.
        valid = '1'
        if write:
            write_bit = '1'
        else:
            write_bit = '0'
        addr = '0x' + hex(addr)[2:].zfill(8)
        # For read operations, the second access to DR needs to contain a valid address, but the original address should
        # not be used, as it implies dual access to the register. Device ID addr used here: 0xf0014
        addr_dummy = '0x' + '0xf0014'[2:].zfill(8)
        data = '0x' + hex(data)[2:].zfill(8)
        error = '0'

        pack = '0b' + error + self.hex2bin(data)[2:] + self.hex2bin(addr)[2:] + write_bit + valid + ack + access_mode + reset + source_id + priv_level + cya
        pack_dummy = '0b' + error + self.hex2bin(data)[2:] + self.hex2bin(addr_dummy)[2:] + write_bit + valid + ack + access_mode + reset + source_id + priv_level + cya
        # print(f'pack = {pack}')
        # print(f'pack_dummy = {pack_dummy}')

        # Write data
        cmds = { 'write_reg_cmd_from_idle': { 'cmd' : b'K\x04\x063\x06\x04k\x00\x81' , 'num_bytes' : 0, 'num_bits' : 7,  'next_state' : jtag.states.exit1_ir },
                 'mv_idle_to_idle': { 'cmd' : b'K\x03\x00' , 'num_bytes' : 0, 'num_bits' : 0,  'next_state' : jtag.states.idle },
                 'mv_exit1_dr_to_idle': { 'cmd' : b'K\x04\x01' , 'num_bytes' : 0, 'num_bits' : 0,  'next_state' : jtag.states.idle },
                 'rd_data_and_mv_shift_dr_to_exit1_dr': { 'cmd' : b'K\x02\x011\x08\x00\x00\x91@\x07\x80\x00\x00\x00\x003\x04\x00k\x00\\x0<1' , 'num_bytes' : 9, 'num_bits' : 5,  'next_state' : jtag.states.exit1_dr },
               }
        # res_ir = self.write_ir(reg_cmd, next_state=jtag.states.exit1_ir)
        res_ir = self._write_cmd(cmds['write_reg_cmd_from_idle'])
        res_dr = self.write_dr(pack, jtag.states.exit1_dr)

        self._write_cmd(cmds['mv_exit1_dr_to_idle'], read=False)
        self._write_cmd(cmds['mv_idle_to_idle'], read=False)
       
        # Read back - no need to update IR regsiter - instruction is retained
        if not write:
            res_dr = self.write_dr(pack_dummy, next_state=jtag.states.exit1_dr)
            # self._write_cmd(cmds['rd_data_and_mv_shift_dr_to_exit1_dr'])
            self._write_cmd(cmds['mv_exit1_dr_to_idle'], read=False)

        # self.reset()
        # Reset state machine
        # outstr = self._write_tms('0', self.state.reset)
        # self._ft_write(outstr)
        # self.state = jtag.states.reset

        return (res_ir, res_dr)

    def sddv_write(self, addr, data):
        '''
        sddv register write using JTAG2HOST_INTFC
        This is something specific to Nvidia Test Chip
        :param addr: address in int
        :param data: data in int
        :return: returns data written and observed on tdo
        '''
        res = self._sddv_frame_write(addr, data)
        self.purge_rx()
        # get readback frame
        res_hex = res[1][1]
        error_bit = ((int(res_hex, 16)) >> 77) & 1
        ack_bit = ((int(res_hex, 16)) >> 10) & 1
        # check if ack bit (10) is set
        if ack_bit == 1:
            # get readback data (bits 45-76 in frame) for validation
            #res_int = (int(res_hex, 16) >> 45) & 0xFFFFFFFF

            #if data != res_int:
                # check if error bit (77) is set
            if error_bit != 0:
                #     # if error bit is set, read data will be in the format 0xbadfXXXX
                #     # the last 4 nibbles can be translated to error code
                #     #self.print_frame(int(res_hex, 16))
                raise Exception("Error writing to SDDV:" + self.translate_error(res_hex))
            #    raise Exception(f"Data Written to device: {hex(data)} is not matching with readback data:{hex(res_int)}")

            return data

        else:
            # Ack bit wasn't set by HW. host transaction unsuccessful!
            # self.print_frame(int(res_hex, 16))
            raise Exception(f"Host acknowledgement not received. read_back {res_hex}")

    def sddv_jtag_reset(self):
        cmd = '0xA0'
        cya = '0'
        priv_level = '00'
        source_id = '00000'
        # Reset bit should always be set to 0 when resetting JTAG2Unit interface.
        reset = '0'
        access_mode = '0'
        ack = '0'
        valid = '0'
        write_bit = '0'
        addr = '0x' + hex(0)[2:].zfill(8)
        data = '0x' + hex(0)[2:].zfill(8)
        error = '0'
        pack = '0b' + error + self.hex2bin(data)[2:] + self.hex2bin(addr)[2:] + write_bit + valid + ack + access_mode + reset + source_id + priv_level + cya
        _ = self._write_cmd(self.raw_ft_cmds['write_reg_cmd_from_idle'])
        self.write_ir(cmd, next_state=jtag.states.exit1_ir, read=False)
        _ = self.write_dr(pack, jtag.states.exit1_dr, read=False)
        # self.write_dr(pack, next_state=jtag.states.exit1_dr, read=False)
        self._write_cmd(self.raw_ft_cmds['mv_exit1_dr_to_idle'], read=False)
        self._write_cmd(self.raw_ft_cmds['mv_idle_to_idle'], read=False)
                
        # Toggle reset back to 1 to finalize the reset operation.
        reset = '1'
        pack = '0b' + error + self.hex2bin(data)[2:] + self.hex2bin(addr)[2:] + write_bit + valid + ack + access_mode + reset + source_id + priv_level + cya

        _ = self.write_dr(pack, next_state=jtag.states.exit1_dr)
        self._write_cmd(self.raw_ft_cmds['mv_exit1_dr_to_idle'], read=False)

    def sddv_read(self, addr):
        '''
        sddv register read using JTAG2HOST_INTFC
        This is something specific to Nvidia Test Chip
        :param addr: address in int
        :param data: data in int
        :return: returns data written and observed on tdo, this is just for sanity check comparision
        '''
        res = self._sddv_frame_read(addr)
        # get readback frame
        res_hex = res[1][1]
        frame_int = (int(res_hex, 16))
        ack_bit = (frame_int >> 10) & 1
        # check if ack bit (10) is set
        error_bit = (frame_int >> 77) & 1
        if ack_bit == 1:
            # get readback data (bits 45-76 in frame) and return it
            res_int = (frame_int >> 45) & 0xFFFFFFFF
            # check if error bit (77) is set
            if error_bit != 0:
            # if error bit is set, read data will be in the format badfXXXX
            # the last 4 nibbles can be translated to error code
            # self.print_frame(int(res_hex, 16))
                raise Exception("Error Reading from SDDV:" + self.translate_error(res_hex))
            return res_int

        else:
            # Ack bit wasn't set by HW. host transaction unsuccessful!
            # self.print_frame(int(res_hex, 16))
            raise Exception(f"Host acknowledgement not received.  read_back {res_hex}")

    def print_frame(self, hex_data):
        if type(hex_data) == str:
            hex_data = int(hex_data, 2)
        else:
            hex_data = hex_data
        print(f"CYA [0:0] {hex(hex_data & 0x1)}")
        print(f"PRIV_LEVEL [1:2] {hex((hex_data >> 1) & 0x3)}")
        print(f"SOURCE_ID [3:7] {hex((hex_data >> 3) & 0x1F)}")
        print(f"RESET [8:8] {hex((hex_data >> 8) & 0x1)}")
        print(f"ACCESS MODE [9:9] {hex((hex_data >> 9) & 0x1)}")
        print(f"ACK [10:10] {hex((hex_data >> 10) & 0x1)}")
        print(f"VALID [11:11] {hex((hex_data >> 11) & 0x1)}")
        print(f"WRITE [12:12] {hex((hex_data >> 12) & 0x1)}")
        print(f"ADDRESS [13:44] {hex((hex_data >> 13) & 0xFFFFFFFF)}")
        print(f"DATA [45:76] {hex((hex_data >> 45) & 0xFFFFFFFF)}")
        print(f"ERROR [77:77] {hex((hex_data >> 77) & 0x1)}")

    def translate_error(self, hex_data):
        if hex_data[:6].lower() == "0xbadf":
            error_code = hex_data[6:]
            if error_code == "5040":
                return "Invalid Address"
        return hex_data

    def hex2bin(self, string):
        return '0b' + bin(int(string, 16))[2:].zfill(len(string[2:] * 4))

    def bin2hex(self, string):
        return '0x' + hex(int(string, 2))[2:].zfill(math.ceil((len(string[2:]) - 1) / 4) + 1)

    def to_bytes_bits(self, data):
        """
        Return a tuple containing a list of bytes, remainder bits, the value of
        the last bit and a reconstruction of the original string.
        data: string representation of a binary or hex number '0xabc', '0b101010'
        e.g. to_bytes_bits('0xabc')        -> ([0xab], [1, 1, 0], 0)
             to_bytes_bits('0b010100011')  -> ([0x51], [], 1)
        """
        byte_list = []
        bit_list = []
        if data[-1] == 'L':
            data = data[:len(data) - 1]
        if data[:2] == '0b':
            base = 2
            data = data[2:]
        elif data[:2] == '0x':
            temp_data = ''
            base = 16
            data = data[2:]
            for i in range(len(data)):
                temp_data = temp_data + bin(int(data[i:(i + 1)], 16))[2:].zfill(4)
            data = temp_data

        else:
            raise ValueError("Data does not match expected format", data)
        length = len(data)
        data = data[::-1]  # data needs to be LSb first
        for i in range(int(length / 8)):
            byte_list.append(int(data[(8 * i):(8 * (i + 1))], 2))
        if -(length % 8):
            bit_list = [int(b, 2) for b in data[-(length % 8):]]
        bit_list_last = None
        if bit_list:
            bit_list_last = bit_list[-1]
            bit_list = bit_list[:-1]
        else:
            bit_list = [int(b, 2) for b in bin(int(byte_list[-1]))[2:].zfill(8)]
            byte_list = byte_list[:-1]
            bit_list_last = bit_list[-1]
            bit_list = bit_list[:-1]
        if base == 2:
            original = '0b' + (''.join([bin(b)[2:].zfill(8) for b in byte_list]) +
                               ''.join([bin(b)[2:] for b in bit_list]) +
                               bin(bit_list_last)[2:])[::-1]
        elif base == 16:
            original = hex(int((''.join([bin(b)[2:].zfill(8) for b in byte_list]) +
                                ''.join([bin(b)[2:] for b in bit_list]) +
                                bin(bit_list_last)[2:])[::-1], 2))
        if original[len(original) - 1] == 'L':
            original = original[:len(original) - 1]
        return (byte_list, bit_list, bit_list_last, original)

    def _rebuild_read_data(self, data, num_bytes, num_bits, tms_bit=True):
        """
        Rebuild a binary string received from the FTDI chip
        Return the reconstructed string MSb first.
        String will be formatted as follows:
            num_bytes of valid bytes
            8 - num_bits of garbage
            num_bits of valid bits
            if tms_bit 1 else 0 bit
            8 - (tms_bit 1 else 0 bit) of garbage
        """
        temp_data = ''

        if num_bytes > 0:
            temp_data = data[0:8 * num_bytes]
            data = data[8 * num_bytes:]
        if num_bits > 0:
            # bits are shifted out msb first, and shifted in from the lsb
            temp_data += data[8 - num_bits: 8]
            if len(data) > 8:
                data = data[8:]

        if tms_bit:
            temp_data += data[0]  # bit position 0 contains the value of TDO when TDI is clocked in
            data = data[1:]
        return '0b' + temp_data[::-1]

    def shift(self, bits):
        for i in range(bits):
            mask = int('0x' + 'f' * math.ceil(bits / 4), 16)
            yield "0x%s" % ('0' * math.ceil(bits / 4) + "%x" % (1 << i & mask))[-math.ceil(bits / 4):]

    def test_to_bytes_bits(self):

        tests = ['0b0101010101',
                 '0b101010101111111',
                 '0x11223344',
                 '0b00010001001000100011001101000100',
                 '0xf102030f102030f102030f1020304f1020304f1020304f1020304444f1020304f',
                 '0xabc',
                 '0xabcd',
                 '0xabcde',
                 '0xabcdef',
                 '0b010100011']

        for t in tests:
            byte_list, bit_list, bit_list_last, original = self.to_bytes_bits(t)
            if t[:2] == '0b':
                if original == t:
                    pass
                else:
                    raise Exception("Test failed")
            elif t[:2] == '0x':
                if original == t:
                    pass
                else:
                    raise Exception("Test failed")
        return

    def read_mpsse_settings(self):
        """Read expected_num bytes from the FTDI chip, once there are that many

        availible in the buffer. Return the raw bytes as a tuple of binary and

        hex strings."""

        for i in range(0xFF):
            self._ft_write(struct.pack('BB', 0x85, i))  # set to 84 for LoopBack
            bytes_avail = ctypes.c_int()
            USBftStatus = self._libraries.FT_GetQueueStatus(self.device_handle, ctypes.byref(bytes_avail))
            if USBftStatus:
                raise Exception("GetQueueStatus failed: " + FTDI_ERROR_CODES[USBftStatus])
            readback = self._ft_read(bytes_avail.value)
            print(i, readback)

def parse():
    parser = argparse.ArgumentParser()
    parser_required_args = parser.add_argument_group('Read/Write from JTAG')

    parser_required_args.add_argument("-d", "--device-index",
                                      help="index of NVJTAG device",
                                      required=False,
                                      type=int)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-w', metavar="DATA", help="write operation", type=lambda x: int(x, 0))
    group.add_argument('-r', help="read operation", action='store_true')
    parser_required_args.add_argument("-a", "--address",
                                      help="address to write/read from",
                                      required=False,
                                      type=lambda x: int(x, 0))
    parser_optional_args = parser.add_argument_group('Reset')
    parser_optional_args.add_argument("--reset", help="Reset JTAG interface",
                                      required=False, action='store_true')
    return parser.parse_args()


# Allow for retries for read/write if a transaction fails.
MAX_RETRIES = 20

def open_sddv(device_index=0):
    opened = False
    # The JTAG Device handle cannot be opened simultaneously.
    # Perform retries on device open to allow different tools
    # to run in parallel on one JTAG device.
    while not opened:
        try:
            j = SDDV_Jtag(device_index)
            j.sddv_jtag_reset()
            opened = True
        except BaseException:
            time.sleep(1 / 10)
            continue
    return j


def read_sddv(address, device_index):
    try:
        SDDV = open_sddv(int(device_index, 0))
        retries = 0
        success = False
        while (retries < MAX_RETRIES) and not success:
            try:
                res = str(hex(SDDV.sddv_read(int(address, 0))))
                success = True
                SDDV.sddv_close()
                return res
            except Exception as e:
                if retries == (MAX_RETRIES - 1) and not success:
                    print(e)
                    return str(-1)
                else:
                    retries += 1
                    time.sleep(1 / 10)
                    continue

    except Exception as e:
        print(e)
        return str(-1)


def write_sddv(address, value, device_index):
    try:
        SDDV = open_sddv(int(device_index, 0))
        retries = 0
        success = False
        while (retries < MAX_RETRIES) and not success:
            try:
                SDDV.sddv_write(int(address, 0), int(value, 0))
                success = True
                SDDV.sddv_close()
                return str(0)
            except Exception as e:
                if retries == (MAX_RETRIES - 1) and not success:
                    print(e)
                    return str(-1)
                else:
                    retries += 1
                    time.sleep(1 / 10)
                    continue
    except Exception as e:
        print(e)
        return str(-1)


if __name__ == "__main__":
    args = parse()
    opened = False
    while not opened:
        try:
            j = SDDV_Jtag(args.device_index)
            j.sddv_jtag_reset()
            opened = True

        except BaseException as e:
            time.sleep(1 / 10)
            continue
    retries = 0
    success = False

    while (retries < MAX_RETRIES) and not success:
        try:
            if args.r:
                res = hex(j.sddv_read(args.address))
                success = True
                print(res, end='\0')
            else:
                j.sddv_write(args.address, args.w)
                success = True
        except Exception as e:
            if retries == (MAX_RETRIES - 1) and not success:
                print(e)
                break
            else:
                retries += 1
                time.sleep(1 / 10)
                continue

