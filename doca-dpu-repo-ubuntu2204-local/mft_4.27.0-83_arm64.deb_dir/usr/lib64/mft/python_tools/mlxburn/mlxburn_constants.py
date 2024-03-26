# Copyright (C) Jan 2020 Mellanox Technologies Ltd. All rights reserved.
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

import logging
import argparse
import shutil
from enum import Enum

DESC = """Burn or generate FW image for Mellanox devices."""
LOGGING_LEVELS = {"INFORM": logging.INFO, "WARNING": logging.WARNING, "DEBUG": logging.DEBUG}
LOGGING_LEVELS = {"INFO": logging.INFO, "WARNING": logging.WARNING, "DEBUG": logging.DEBUG, "ERROR": logging.ERROR}
DEFAULT_LOGGING_LEVEL = "INFO"

MAIN_MODE = Enum("MAIN_MODE", "SHOW_FW_VER QUERY IMAGE SHOW_VPD")
MAIN_MODE_FLAGS = {MAIN_MODE.SHOW_FW_VER: "fwver", MAIN_MODE.QUERY: "query", MAIN_MODE.SHOW_VPD: "vpd", MAIN_MODE.IMAGE: "image"}

MLNXSW_DEVICE_PREFIX = "mlnxsw-"
MLX_FILE_EXTENSION = ".mlx"
BIN_FILE_EXTENSION = ".BIN"
XML_HEADER_PATTERN = r"<!--\s*MT[\d]{3,5}\s+Firmware\s+image"
RELEVANT_MLX_PATTERNS = (r"*rel.mlx", r"*IS4.mlx ", r"*sx.mlx", r"*IB.mlx", r"*SwitchIB.mlx", r"*ConnectX4.mlx", r"*ConnectX4Lx.mlx", r"*ConnectX5.mlx", r"*ConnectX6.mlx", r"*ConnectX6Dx.mlx", r"*ConnectX6Lx.mlx", r"*ConnectX7.mlx", r"*BlueField.mlx", r"*BlueField-2.mlx", r"*BlueField-3.mlx ", r"*SwitchEN.mlx", r"*SwitchIB-2.mlx", r"*Quantum.mlx", r"*Quantum-2.mlx", r"*Spectrum-2.mlx", r"*Spectrum-3.mlx", r"*Spectrum-4.mlx", r"*LinkXGearboxRetimer.mlx", r"*BW00.mlx")
ENCRYPTION_MIN_PYTHON_VERSION = (3, 6, 0)


DEFAULT_ERROR_MESSAGE = "{prog} execution failed, returncode:{rc}\ncmd: {cmd}"
PROCESS_OUTPUT = "{prog} {stream}:\n{content}"

# bring these lines back after deprecation of py3.4
# SECURITY_MODE = Enum("SECURITY_MODE", "UNKNOWN NONE SHA_DIGEST RSA", start=-1)
# ENCRYPTED_MODE = Enum("ENCRYPTED_MODE", "NONE BEFORE_SIGN AFTER_SIGN", start=0)


class SECURITY_GEN(Enum):
    LEGACY_SECURITY = 0
    FOURTH_GEN_SECURITY = 1


class SECURITY_MODE(Enum):
    UNKNOWN = -1
    NONE = 0
    SHA_DIGEST = 1
    FW_UPDATE = 2
    SECURE_BOOT = 3


class ENCRYPTED_MODE(Enum):
    NONE = 0
    BEFORE_SIGN = 1
    AFTER_SIGN = 2


PUBLIC_KEY_SECTION = "DEV_KEY_PUBLIC.data"
PRIVATE_KEY_SECTION = "DEV_KEY_PEM.data"
UUID_SECTION = "DEV_KEY_UUID.data"
GCM_IV_SECTION = "GCM_IV.data"
ENCRYPTION_KEY_SECTION = "ENCRYPTION_KEY.data"
PSC_BL1_SECTION = "psc_bl1.bin"
PSC_BCT_SECTION = "psc_bct.bin"
PSC_FW_SECTION = "psc_fw.bin"
NCORE_STAGE1_KEY_SECTION = "NCORE_STAGE1_SIGN_KEY.data"
NCORE_STAGE2_KEY_SECTION = "NCORE_STAGE2_SIGN_KEY.data"
NCORE_ENCRYPTION_KEY_SECTION = "NCORE_ENC_KEY.data"
NCORE_JSON_SECTION = "ncore_nvsign_parameters.json"

BDF_PATTERN = r"(\w{4}):(\w{2}:\w{2}.\w)"

BURN_PROGRAM = shutil.which("flint")
T2A_PROGRAM = shutil.which("t2a")
MIC_PROGRAM = shutil.which("mic")
MLXFWENC_PROGRAM = shutil.which("mlxfwencryption")
MLXVPD_PROGRAM = shutil.which("mlxvpd")

MIC_LEGACY_4TH_GEN_DEVICE_ID = 25408
CX3PRO_DEVICE_ID = 503

VSD_MAX_LEN = 208

MLX_FILE_2_DEVICE_NAME = {
    "LinkXGearboxRetimer": "AmosGearBox",
    "LinkXAbirGearboxRetimer": "AbirGearBox",
    "SwitchEN": "Spectrum",
}

DEVICE_NAME_2_PATTERN = {
    "ConnectX3": r'fw-(ConnectX3)-rel\.mlx',
    "ConnectX3Pro": r'fw-(ConnectX3Pro)-rel\.mlx'
}

ADDITIONAL_BURN_ARGS = (
    ("byte_mode", argparse.SUPPRESS),
    ("allow_psid_change", ""),
    ("skip_is", ""),
    ("no", ""),
    # ("qq", argparse.SUPPRESS),
    ("use_image_ps", ""),
    ("use_image_rom", ""),
    ("no_flash_verify", ""),
    ("use_image_guids", ""),
    ("uid", ""),
    ("uids", argparse.SUPPRESS),
    ("log", ""),
    ("banks", ""),
    ("guid", argparse.SUPPRESS),
    ("guids", argparse.SUPPRESS),
    ("mac", ""),
    ("macs", argparse.SUPPRESS),
    ("sysguid", ""),
    ("ndesc", ""),
    ("bsn", ""),
    ("pe_i2c", ""),
    ("pe", argparse.SUPPRESS),
    ("se_i2c", ""),
    ("se", argparse.SUPPRESS),
    ("flash_params", ""),
    ("is3_i2c", "")
)

ADDITIONAL_BURN_QUERY_ARGS = (
    ("ocr", ""),
    ("ignore_dev_data", ""),
    ("use_dev_rom", ""),
    ("no_fw_ctrl", ""),
    ("override_cache_replacement", "")
)

ADDITIONAL_IMGEN_ARGS = (
    ("no_vsd_swap", argparse.SUPPRESS),
    ("gb_bin_file", "Integrate the given gearbox binary file to the FW image."),
    ("user_data", argparse.SUPPRESS),
    ("prof_file", argparse.SUPPRESS)
)

ADDITIONAL_COMMON_ARGS = (
    ("striped_image", ""),
    ("blank_guids", ""),
)

ADDITIONAL_ARGS_HEADER = ("option", "help")
