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

import ctypes
import os
import platform


class MftCoreDeviceException(Exception):
    pass


MFT_CORE_DEVICE = None
MFT_CORE_DEVICE_SO = "mst_device.so"
MFT_CORE_DEVICE_DLL = "mst_device.dll"

try:
    from ctypes import *
    if platform.system() == "Windows" or os.name == "nt":
        try:
            MFT_CORE_DEVICE = CDLL(MFT_CORE_DEVICE_DLL)
        except BaseException:
            MFT_CORE_DEVICE = CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), MFT_CORE_DEVICE_DLL))
    else:
        try:
            MFT_CORE_DEVICE = CDLL(MFT_CORE_DEVICE_SO)
        except BaseException:
            MFT_CORE_DEVICE = CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), MFT_CORE_DEVICE_SO))
except Exception as exp:
    raise MftCoreDeviceException("Failed to load shared library mst_device: %s" % exp)


def get_num_of_devices():
    MFT_CORE_DEVICE.get_num_of_devices.restype = c_uint
    return MFT_CORE_DEVICE.get_num_of_devices()


def get_num_of_nics_and_switches():
    MFT_CORE_DEVICE.get_num_of_nics_and_switches.restype = c_uint
    return MFT_CORE_DEVICE.get_num_of_nics_and_switches()


def get_device_name_by_index(i):
    dev_name = ctypes.create_string_buffer(64)
    MFT_CORE_DEVICE.get_device_name_by_index.restype = c_void_p
    MFT_CORE_DEVICE.get_device_name_by_index(i, dev_name)
    string = dev_name.value.decode("utf-8")
    return string


def get_device_id_from_str(dev_name):
    MFT_CORE_DEVICE.get_device_id_from_str.restype = c_int
    MFT_CORE_DEVICE.get_device_id_from_str.argtypes = [c_char_p]
    return MFT_CORE_DEVICE.get_device_id_from_str(dev_name.encode("utf-8"))


if MFT_CORE_DEVICE:
    class MftCoreDevice:
        def __init__(self, hwID, jsonDir="default"):
            self.hwID = hwID
            self._initReturnTypes()
            self._LoadDynamicFunction()
            if jsonDir == "default":
                self._createMftCoreDeviceInstance(hwID)
            else:
                self._createMftCoreDeviceInstanceWithJsonDir(hwID, jsonDir.encode("utf-8"))

        def _LoadDynamicFunction(self):
            self._createMftCoreDeviceInstance = MFT_CORE_DEVICE.create_instance
            self._createMftCoreDeviceInstanceWithJsonDir = MFT_CORE_DEVICE.create_instance_with_json_dir
            self._deleteMftCoreDevice = MFT_CORE_DEVICE.delete_instance
            self._is_nic = MFT_CORE_DEVICE.is_nic
            self._is_switch = MFT_CORE_DEVICE.is_switch
            self._isGearbox = MFT_CORE_DEVICE.is_gearbox
            self._is4thGenNIC = MFT_CORE_DEVICE.is_4th_gen_nic
            self._is5thGenNIC = MFT_CORE_DEVICE.is_5th_gen_nic
            self._isFs5 = MFT_CORE_DEVICE.is_fs5
            self._isFs4 = MFT_CORE_DEVICE.is_fs4
            self._isFs3 = MFT_CORE_DEVICE.is_fs3
            self._isFs2 = MFT_CORE_DEVICE.is_fs2
            self._get_device_hw_id = MFT_CORE_DEVICE.get_device_hw_id
            self._getPciDeviceID = MFT_CORE_DEVICE.get_pci_device_id
            self._getDeviceFwMajor = MFT_CORE_DEVICE.get_device_fw_major
            self._is_switch_ib = MFT_CORE_DEVICE.is_switch_ib
            self._is_switch_ib2 = MFT_CORE_DEVICE.is_switch_ib2
            self._FwStrDbSignatureExists = MFT_CORE_DEVICE.fw_str_db_signature_exists
            self._GetMaxMainIrisc = MFT_CORE_DEVICE.get_max_main_irisc
            self._GetMaxAPU = MFT_CORE_DEVICE.get_max_apu
            self._GetMaxNumOfTiles = MFT_CORE_DEVICE.get_max_num_of_tiles
            self._GetMaxIriscPerTile = MFT_CORE_DEVICE.get_max_irisc_per_tile
            self._GetIriscStartAddr = MFT_CORE_DEVICE.get_iris_start_addr
            self._GetIriscStep = MFT_CORE_DEVICE.get_iris_step
            self._GetTileStart = MFT_CORE_DEVICE.get_tile_start
            self._GetTileStep = MFT_CORE_DEVICE.get_tile_step
            self._GetApuStartAddrStep = MFT_CORE_DEVICE.get_apu_start_addr_step
            self._GetApuStep = MFT_CORE_DEVICE.get_apu_step
            self._GetApuMaxNumOfSteps = MFT_CORE_DEVICE.get_apu_max_num_of_steps
            self._SupportPhyUc = MFT_CORE_DEVICE.supports_phy_uc
            self._IsDynamicDeviceWithoutIriscId = MFT_CORE_DEVICE.is_dynamic_device_without_irisc_id
            self._is_nvRisc = MFT_CORE_DEVICE.is_nvRisc
            self._GetFieldAsInt = MFT_CORE_DEVICE.get_field_as_int
            self._GetFieldAsBool = MFT_CORE_DEVICE.get_field_as_bool

        def _initReturnTypes(self):
            MFT_CORE_DEVICE.get_device_name.restype = c_void_p
            MFT_CORE_DEVICE.get_device_short_name.restype = c_void_p
            MFT_CORE_DEVICE.get_device_fw_name.restype = c_void_p
            MFT_CORE_DEVICE.get_field_as_string.restype = c_void_p
            MFT_CORE_DEVICE.get_field_as_int.restype = c_uint
            MFT_CORE_DEVICE.get_field_as_bool.restype = c_bool
            MFT_CORE_DEVICE.is_nic.restype = c_bool
            MFT_CORE_DEVICE.is_switch.restype = c_bool
            MFT_CORE_DEVICE.is_gearbox.restype = c_bool
            MFT_CORE_DEVICE.is_4th_gen_nic.restype = c_bool
            MFT_CORE_DEVICE.is_5th_gen_nic.restype = c_bool
            MFT_CORE_DEVICE.is_fs5.restype = c_bool
            MFT_CORE_DEVICE.is_fs4.restype = c_bool
            MFT_CORE_DEVICE.is_fs3.restype = c_bool
            MFT_CORE_DEVICE.is_fs2.restype = c_bool
            MFT_CORE_DEVICE.get_device_hw_id.restype = c_uint
            MFT_CORE_DEVICE.get_pci_device_id.restype = c_uint
            MFT_CORE_DEVICE.get_device_fw_major.restype = c_uint
            MFT_CORE_DEVICE.is_switch_ib.restype = c_bool
            MFT_CORE_DEVICE.is_switch_ib2.restype = c_bool
            MFT_CORE_DEVICE.fw_str_db_signature_exists.restype = c_bool
            MFT_CORE_DEVICE.get_max_main_irisc.restype = c_uint
            MFT_CORE_DEVICE.get_max_apu.restype = c_uint
            MFT_CORE_DEVICE.get_max_num_of_tiles.restype = c_uint
            MFT_CORE_DEVICE.get_max_irisc_per_tile.restype = c_uint
            MFT_CORE_DEVICE.get_iris_start_addr.restype = c_uint
            MFT_CORE_DEVICE.get_iris_step.restype = c_uint
            MFT_CORE_DEVICE.get_tile_start.restype = c_uint
            MFT_CORE_DEVICE.get_tile_step.restype = c_uint
            MFT_CORE_DEVICE.get_apu_start_addr_step.restype = c_uint
            MFT_CORE_DEVICE.get_apu_step.restype = c_uint
            MFT_CORE_DEVICE.get_apu_max_num_of_steps.restype = c_uint
            MFT_CORE_DEVICE.supports_phy_uc.restype = c_bool
            MFT_CORE_DEVICE.is_dynamic_device_without_irisc_id.restype = c_bool
            MFT_CORE_DEVICE.is_nvRisc.restype = c_bool

        def _getDeviceFWName(self):
            dev_name = ctypes.create_string_buffer(64)
            MFT_CORE_DEVICE.get_device_fw_name(dev_name)
            string = dev_name.value.decode("utf-8")
            return string

        def _getDeviceName(self):
            dev_name = ctypes.create_string_buffer(64)
            MFT_CORE_DEVICE.get_device_name(dev_name)
            string = dev_name.value.decode("utf-8")
            return string

        def _getDeviceShortName(self):
            dev_name = ctypes.create_string_buffer(64)
            MFT_CORE_DEVICE.get_device_short_name(dev_name)
            string = dev_name.value.decode("utf-8")
            return string

        def _GetDefaultTracerMode(self):
            return MFT_CORE_DEVICE.get_default_tracer_mode().decode("utf-8")

        def _getFieldAsString(self, field_name, json_attribute):
            data = ctypes.create_string_buffer(64)
            MFT_CORE_DEVICE.get_field_as_string(data, field_name, json_attribute)
            string = data.value.decode("utf-8")
            return string

else:
    raise MftCoreDeviceException("Failed to load shared library mst_device")
