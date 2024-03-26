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

import argparse
import glob
import logging
import configparser
import shutil
import json

from cli_wrapping_utils import *
from mlxburn_constants import *
from modes.mlxburn_mode import I_MlxburnMode
from mlxburn_utils import *

import dev_mgt


class ImageGenerateMode(I_MlxburnMode):
    def __init__(self, args, mode=MAIN_MODE.IMAGE):
        super().__init__(args)

        self._device_handler = None

        self._device_query_map = {}

        self._temp_files = []
        self._bin_file_h = None
        self._converted_fw_file_h = None
        self._enc_bin_h = None

        self.init_device_properties()

        if mode == MAIN_MODE.IMAGE:
            self._generateImage = False
            if self._args.input_img:
                self._bin_file = args.input_img
            elif args.img_dir:
                self._bin_file = self.select_binary_file()
                logging.info("Using auto detected image file : {}".format(self._bin_file))
            else:
                self._generateImage = True
                self._bin_file = self._args.bin_file
            self._non_encrypted_bin_file = self._args.non_encrypted_bin_file

    def __del__(self):
        for temp_file_obj in [self._bin_file_h, self._converted_fw_file_h, self._enc_bin_h]:
            if temp_file_obj:
                temp_file_obj.close()
        for temp_file_obj in self._temp_files:
            Path(temp_file_obj.name).unlink()

    def execute(self):
        if self._generateImage:
            self.validate_dev_type()
            logging.debug("fw_file = {}".format((self._args.fw_file)))

            if not self._args.conf_file or str(self._args.exp_rom) == "AUTO":
                self._device_query_map = query(self._args.mst_dev, True, True, False, self._args.quick_query, self._args.additional_burn_query_args + self._args.additional_common_args, "q", "full")

            mic_cmd = self.prepare_image_gen_mic_command()
            logging.info("Generating image ...")
            logging.debug("running {}".format((mic_cmd)))
            execute_check(mic_cmd, "Image generation failed: {stdout}")

            if dev_mgt.DevMgt.is_fs3(self._args.dm_dev_id) or dev_mgt.DevMgt.is_fs4(self._args.dm_dev_id):
                self.handle_security()
            elif dev_mgt.DevMgt.is_fs5(self._args.dm_dev_id):
                self.handle_4thgen_security()
            self._bin_file.chmod(0o644)
            logging.info("Image generation completed successfully.")
        return self._bin_file

    def add_temp_file(self, *args, **kwargs):
        is_windows = platform.system() == "Windows"
        if is_windows:
            kwargs["delete"] = False

        tmpFile = NamedTemporaryFile(*args, **kwargs)
        if is_windows:
            self._temp_files.append(tmpFile)

        return tmpFile

    def init_device_properties(self):
        args = self._args

        if self._args.mst_dev:
            self._device_handler = dev_mgt.DevMgt(args.mst_dev, args.i2c_secondary if args.i2c_secondary else -1)

        if not args.dev_type:
            if args.fw_file and not args.mst_dev:  # i.e. args.fw_file and args.wrimage
                m = regex_match(r"^fw-(.+)\.mlx", args.fw_file.name)
                device_name = None
                if m:
                    device_name = m.group(1)
                    if '-' in device_name:
                        device_name_split = device_name.split('-')
                        if device_name_split[1].isalpha():
                            logging.debug("Removing suffix -{} from device name".format(device_name_split[1]))
                            device_name = device_name_split[0]  # Removing "-rel" suffix for CX3PRO/"-ethlt" for fw-ConnectX5-ethlt.mlx/etc..
                        else:
                            device_name = device_name_split[0] + device_name_split[1]
                            if dev_mgt.DevMgt.str2hwId(device_name)["return_code"] != 0:
                                device_name = device_name_split[0]
                    device_name = MLX_FILE_2_DEVICE_NAME.get(device_name, device_name)  # note, this is special case for few devices and not recommended to add more devices like this
                    logging.debug("device_name from mlx file: {}".format(device_name))
                    ret = dev_mgt.DevMgt.str2hwId(device_name)
                else:
                    ret = {"return_code": 1}
            else:
                ret = self._device_handler.getHwId()
            rc = ret["return_code"]
            if rc != 0:
                error_exit("Can not auto detect device type: {}. Please check the given device.".format((rc)))
            else:
                self._args.dev_type = ret["hw_device_id"]
                self._args.dm_dev_id = ret["dm_device_id"]

        if self._device_handler:
            self._device_handler._mstdev.close()

        logging.debug("dev_type: {}".format((args.dev_type)))
        logging.debug("dm_device_id: {}".format((args.dm_dev_id)))

        self.get_mic_device_type()

    # workaround function for MIC compatibility
    def get_mic_device_type(self):
        if dev_mgt.DevMgt.is_4th_gen(self._args.dm_dev_id):
            self._args.dev_type = MIC_LEGACY_4TH_GEN_DEVICE_ID

    def validate_dev_type(self):
        args = self._args
        dm_dev_id = args.dm_dev_id

        if args.base_guid or args.base_mac or args.vpd_r_file:
            if not dev_mgt.DevMgt.is_fs3(dm_dev_id) and not dev_mgt.DevMgt.is_fs4(dm_dev_id) and not dev_mgt.DevMgt.is_fs5(dm_dev_id):
                error_exit("vpd_r_file, base_guid and base_mac options are applicable only for FS3/FS4 images.")

        if args.base_mac:
            if dev_mgt.DevMgt.is_connectib(dm_dev_id) or dev_mgt.DevMgt.is_ib_switch(dm_dev_id):
                error_exit("base_mac is not applicable for the provided device type")

        if (args.base_mac and not args.base_guid) or (not args.base_mac and args.base_guid):
            if (dev_mgt.DevMgt.is_fs3(dm_dev_id) or dev_mgt.DevMgt.is_fs4(dm_dev_id)) and not (dev_mgt.DevMgt.is_connectib(dm_dev_id) or dev_mgt.DevMgt.is_ib_switch(dm_dev_id)):
                error_exit("Both base_guid and base_mac must be specified.")

    def get_fwver_fw_file(self):
        cmd = self.prepare_fwver_cmd()
        proc_ret = execute_check(cmd, "Image version query failed: {rc}")
        return proc_ret.stdout.strip()

    def prepare_base_mic_cmd_generator(self):
        if MIC_PROGRAM is None:
            error_exit("Image generation tool is missing")
        cmd_gen = CmdGenerator(MIC_PROGRAM)
        I_OptArgGenerator.set_option_prefix("-")

        cmd_gen.add_argument(inputArgPipeOAG("format", self._args.format))
        if getattr(logging, self._args.verbosity) > logging.WARNING:
            cmd_gen.add_argument(addFlagOAG("nowarn"))
        if self._args.nofs_img:
            cmd_gen.add_argument(addFlagOAG("nofs"))
        fw_file_OAG = conversionFuncOAG("fw", self.get_fw_file)
        cmd_gen.add_argument(fw_file_OAG)

        self._args.fw_file = Path(fw_file_OAG.optArgs[-1])

        return cmd_gen

    def prepare_base_mic_command(self):
        return self.prepare_base_mic_cmd_generator.get_cmd()

    def prepare_image_gen_mic_command(self):
        cmd_gen = self.prepare_base_mic_cmd_generator()

        cmd_gen.add_argument(conversionFuncOAG("conf", self.find_conf_file))
        cmd_gen.add_argument(conversionFuncOAG("wrimage", self.convert_temp_bin_file))
        cmd_gen.add_argument(conversionFuncOAG("exp_rom", self.find_exp_rom))
        cmd_gen.add_argument(inputArgPipeOAG("vpd_r", self._args.vpd_r_file))
        cmd_gen.add_argument(extensionFuncOAG(self.get_uids_macs))
        cmd_gen.add_argument(extensionFuncOAG(self.get_vsd_strings))

        return cmd_gen.get_cmd() + self._args.additional_imgen_args + self._args.additional_imgen_burn_args + self._args.additional_common_args

    def prepare_fwver_cmd(self):
        cmd_gen = self.prepare_base_mic_cmd_generator()
        cmd_gen.add_argument(addFlagOAG("fwver"))
        return cmd_gen.get_cmd()

    def select_binary_file(self):
        device = self._args.mst_dev
        search_dir = self._args.img_dir

        query_map = query(device, quick=self._args.quick_query, must_query_flags=self._args.additional_burn_query_args + self._args.additional_common_args)

        dev_psid = query_map.get("PSID")
        if not dev_psid:
            error_exit("Can not auto detect FW file. Device {} PSID may not be configured.".format((device)))
        logging.debug("Device PSID: {}".format((dev_psid)))

        bin_files = search_dir.glob(r"*.[bB][iI][nN]")

        for bin_file in bin_files:
            query_map = query(bin_file, is_device=False, exit_on_error=False, quick=self._args.quick_query, must_query_flags=self._args.additional_burn_query_args + self._args.additional_common_args)
            if not query_map:
                continue

            bin_file_psid = query_map.get("PSID")
            logging.debug("Binary file - {} PSID: {}".format((str(bin_file)), (bin_file_psid)))
            if (bin_file_psid == dev_psid):
                return bin_file
        error_exit("Directory {} does not contain an image that matches the device FW (No matching PSID)".format((search_dir)))

    def get_fw_file(self):
        return self.convert_fw_file(self.find_fw_file())

    def convert_fw_file(self, fw_file):
        if fw_file.suffix == MLX_FILE_EXTENSION:
            try:
                with fw_file.open() as fw_file_h:
                    first_line = fw_file_h.readline()
            except BaseException:
                error_exit("Failed to open fw file in order to convert to new format")
            if not regex_match(XML_HEADER_PATTERN, first_line):
                self._converted_fw_file_h = self.add_temp_file()
                converted_fw_file = Path(self._converted_fw_file_h.name)
                t2a_cmd = [str(Path(T2A_PROGRAM)), "MT{}".format((self._args.dev_type)), str(fw_file), str(converted_fw_file)]
                logging.debug("Running {}".format((t2a_cmd)))
                execute_check(t2a_cmd, "errorCode = {rc}.", shell=True)

                return converted_fw_file
        return fw_file

    def get_fw_filename_pattern(self, dev_type_str):
        pattern = r'fw-(.+)\.mlx'
        if dev_type_str in DEVICE_NAME_2_PATTERN:  # Only for old devices backward comp. DO NOT ADD DEVICES TO THIS DICT
            pattern = DEVICE_NAME_2_PATTERN[dev_type_str]
        return pattern

    def find_fw_file(self):
        dm_dev_id = vars(self._args).get("dm_dev_id")
        fw_file = self._args.fw_file
        fw_dir = self._args.fw_dir

        if not fw_file and fw_dir and dm_dev_id:
            logging.debug("dm_dev_id: {}".format((dm_dev_id)))

            dev_type_str = dev_mgt.DevMgt.type2str(dm_dev_id)
            pattern = self.get_fw_filename_pattern(dev_type_str)
            dev_type_str = dev_type_str.replace("-", "").lower()

            for file in fw_dir.iterdir():
                m = regex_match(pattern, file.name)
                if not m:
                    logging.debug("file: {file}, does not match fw-file pattern")
                    continue

                file_type_str = m.group(1).replace("-", "").lower()

                if file_type_str.startswith(dev_type_str):
                    fw_file = file
                    logging.debug("using auto-detected .mlx file: {}".format(fw_file))
                    break
        else:
            logging.debug("using provided .mlx file: {}".format(fw_file))
        if not fw_file:
            error_exit("Can't auto detect fw file: Failed to find fw file in directory {} for device. Please specify fw (mlx) file using -fw flag .".format((fw_dir)))
        return fw_file

    def find_conf_file(self):
        conf_file = self._args.conf_file
        conf_dir = self._args.conf_dir
        conf_dir_list = self._args.conf_dir_list
        device_psid = self._device_query_map.get("PSID")

        if not conf_file:
            for conf_dir_pattern in conf_dir_list:
                if conf_file:
                    break
                for conf_dir_str in glob.glob(conf_dir_pattern):
                    conf_dir = Path(conf_dir_str)
                    if not device_psid:
                        error_exit("Can't auto detect fw configuration file. Device PSID may not be configured.".format())
                    conf_file = self.get_matching_conf_file(device_psid, conf_dir)

                    if conf_file:
                        break
            if not conf_file:
                error_exit("Can't auto detect fw configuration file: $err_msg . Please specify configuration (ini) file using -conf flag .")

            logging.info("Using auto detected configuration file: {} (PSID = {})".format((conf_file), (device_psid)))
        else:
            logging.debug("Using provided configuration file: {} (PSID = {})".format((conf_file), (device_psid)))
        return conf_file

    def get_matching_conf_file(self, required_psid, search_dir: Path):
        found_file = None

        parser = configparser.ConfigParser(strict=False)
        for ini_file in search_dir.glob(r"*.[iI][nN][iI]"):
            parser.clear()

            try:
                parser.read(str(ini_file))
            except configparser.Error as e:
                logging.warning("Failed parsing config-file: {}\n".format(e.message))
                continue

            sec_opt = (("PSID", "PSID"), ("ADAPTER", "PSID"), ("image_info", "psid"))
            psid = None
            for sec, opt in sec_opt:
                psid = parser.get(sec, opt, fallback=None)
                if psid:
                    break

            if psid == required_psid:
                if found_file:
                    return None
                else:
                    found_file = ini_file
        return found_file

    def convert_temp_bin_file(self):
        if self._bin_file:
            return self._bin_file
        bin_file_suffix = ""
        fw_file_ext = self._args.fw_file.suffix
        if fw_file_ext == BIN_FILE_EXTENSION:
            bin_file_suffix = ".img"
        elif fw_file_ext == MLX_FILE_EXTENSION:
            bin_file_suffix = ".bin"

        self._bin_file_h = self.add_temp_file(suffix=bin_file_suffix)
        self._bin_file = Path(self._bin_file_h.name)
        return self._bin_file

    def find_exp_rom(self):
        fw_dir = self._args.fw_dir
        exp_rom = self._args.exp_rom
        exp_rom_dir = self._args.exp_rom_dir
        rom_info = self._device_query_map.get("Rom_Info")

        if exp_rom and str(exp_rom) != "AUTO":
            return exp_rom

        if str(exp_rom) == "AUTO":
            if not exp_rom_dir:
                exp_rom_dir = fw_dir / "exp_rom"

            if not rom_info:
                exp_rom = None
            elif rom_info == "N/A":
                error_exit("Can not auto detect expansion ROM to use. Please specify expansion ROM using -exp_rom flag.")
            else:
                exp_rom = self.select_exp_rom()
        return exp_rom

    def select_exp_rom(self):
        device_rom_info = self._device_query_map.get("Rom_Info")
        exp_rom_dir = self._args.exp_rom_dir

        if not isinstance(device_rom_info, list):
            device_rom_info = [device_rom_info]

        for file in exp_rom_dir.glob("*"):
            query_map = query(file, is_device=False, exit_on_error=False, subcommand='qrom', must_query_flags=self._args.additional_burn_query_args + self._args.additional_common_args)
            if not query_map:
                continue

            # Make sure that we don't take bin with ROM but only ROM files
            if query(file, is_device=False, exit_on_error=False, must_query_flags=self._args.additional_burn_query_args + self._args.additional_common_args):
                continue

            logging.debug("Checking matching of Exp ROM file: {}".format((file)))

            bin_file_rom_info = query_map.get("Rom_Info")
            if not isinstance(bin_file_rom_info, list):
                bin_file_rom_info = [bin_file_rom_info]

            for c_device_rom_info in device_rom_info:
                c_bin_file_rom_info = [ri for ri in bin_file_rom_info if ri.get("type") == c_device_rom_info.get("type")]
                if c_bin_file_rom_info:
                    c_bin_file_rom_info = c_bin_file_rom_info[0]

                if c_bin_file_rom_info.get("type") and \
                    c_bin_file_rom_info.get("devid") == c_device_rom_info.get("devid") and \
                    (c_bin_file_rom_info.get("port") == c_device_rom_info.get("port") or
                     c_bin_file_rom_info.get("port", 0) == 0) and \
                    (c_bin_file_rom_info.get("proto") == c_device_rom_info.get("proto") or
                     (c_bin_file_rom_info.get("proto") == "VPI" and c_device_rom_info.get("proto") == "IB") or
                     (c_bin_file_rom_info.get("proto") == "VPI" and c_device_rom_info.get("proto") == "ETH")):
                    return file

        error_exit("Directory {ex_rom_dir} does not contain an expansion ROM image that matches the device. You can specify expansion ROM manually using the -exp_rom flag")

    def get_uids_macs(self):
        base_guid = self._args.base_guid
        base_mac = self._args.base_mac

        if not base_guid:
            return {}
        if base_mac:
            return self.get_connectx4_uids()
        else:
            return self.get_connectib_guids()

    def get_connectx4_uids(self):
        base_guid = self._args.base_guid
        base_mac = self._args.base_mac

        sections = ("mfg_info", "device_info")
        raw_vals = (("guids", base_guid), ("macs", base_mac))
        limits = (("hi", lambda n: n >> 32), ("lo", lambda n: n & 0xffffffff))

        return {"{}.guids.{}.uid.{}={:#08x}".format((section), (name), (limit), (func(base_num))): []
                for section in sections
                for name, base_num in raw_vals
                for limit, func in limits}

    def get_connectib_guids(self):
        base_guid = self._args.base_guid

        port0_guid = base_guid
        port1_guid = base_guid + 8
        port0_mac = self.guid2mac(port0_guid)
        port1_mac = self.guid2mac(port1_guid)

        sections = ("mfg_info", "device_info")
        raw_vals = ((r"guids[0]", port0_guid), (r"guids[1]", port1_guid), (r"macs[0]", port0_mac), (r"macs[0]", port1_mac))
        limits = (("hi", lambda n: n >> 32), ("lo", lambda n: n & 0xffffffff))

        return {"{}.guids.{}.uid.{}={:#08x}".format((section), (name), (limit), (func(base_num))): []
                for section in sections
                for name, base_num in raw_vals
                for limit, func in limits}

    def guid2mac(self, guid: int):
        return guid & 0xffffff | ((guid >> 16) & 0xffffff000000)

    def get_vsd_strings(self):
        vsd_string = self._args.vsd
        dm_dev_id = self._args.dm_dev_id

        ret = {}
        if vsd_string:
            if dev_mgt.DevMgt.is_fs3(dm_dev_id) or dev_mgt.DevMgt.is_fs4(dm_dev_id):
                ret["image_info.vsd"] = ["{}".format((vsd_string))]
            else:
                ret["ADAPTER.adapter_vsd"] = ["{}".format((vsd_string))]
        return ret

    def handle_4thgen_security(self):
        full_query_map = query(str(self._bin_file), False, True, False, self._args.quick_query, self._args.additional_burn_query_args + self._args.additional_common_args, "q", "full")

        security_mode = self.parse_security_mode(full_query_map)
        if security_mode == SECURITY_MODE.NONE:
            return
        if security_mode == SECURITY_MODE.UNKNOWN:
            error_exit("Failed to handle security mode - Unknown mode")

        encrypted_mode = self.parse_encryption_mode(security_mode, full_query_map)
        security_artifacts = self.get_security_artifacts(security_mode, encrypted_mode, SECURITY_GEN.FOURTH_GEN_SECURITY)
        nvsign_path = "/usr/lib64/mft/python_tools/nvsign/nvsign.py"
        ncore_component_path = self.add_temp_file()
        ncore_component_signed_path = self.add_temp_file()

        if encrypted_mode is ENCRYPTED_MODE.NONE:
            self._args.noencrypt = True
        else:
            security_mode = SECURITY_MODE.SECURE_BOOT

        if security_mode == SECURITY_MODE.FW_UPDATE or security_mode == SECURITY_MODE.SECURE_BOOT:
            flint_cmd = [BURN_PROGRAM, "-i", str(self._bin_file), "set_public_keys", security_artifacts[PUBLIC_KEY_SECTION].name]
            logging.debug(f"running {flint_cmd}")
            execute_check(flint_cmd, "Failed to set the public key section into the image: {rc}")

        extract_ncore_cmd = [BURN_PROGRAM, "-i", str(self._bin_file), "read_ncore_component", ncore_component_path.name]
        logging.info("Extracting NCore section...")
        logging.debug(f"running {extract_ncore_cmd}")
        execute_check(extract_ncore_cmd, "Failed to extract ncore from the image: {stdout}", shell=True)

        with open(security_artifacts[NCORE_JSON_SECTION].name, 'r') as file:
            parameters = json.load(file)
        nvsign_cmd = ["python3.8", nvsign_path, ncore_component_path.name, "--ver_major", parameters["ver_major"], "--ver_minor", parameters["ver_minor"],
                      "--ratchet", parameters["ratchet"], "--magic", parameters["magic"], "--load_dest", parameters["load_dest"], "--entry_point", parameters["entry_point"],
                      "--encrypt_key", security_artifacts[NCORE_ENCRYPTION_KEY_SECTION].name, "--sign_key", security_artifacts[NCORE_STAGE1_KEY_SECTION].name,
                      "--oem_sign_key", security_artifacts[NCORE_STAGE2_KEY_SECTION].name, "--iv_lsb", parameters["iv_lsb"], "--hash_offset", parameters["hash_offset"],
                      "--data_encryption_key_version", parameters["data_encryption_key_version"], "--encryption_derivation_string", parameters["encryption_derivation_string"],
                      "-o", ncore_component_signed_path.name]
        if self._args.noencrypt:
            nvsign_cmd.append("--noencrypt")
        logging.info("Signing NCore section...")
        logging.debug(f"running {nvsign_cmd}")
        execute_check(nvsign_cmd, "Failed to sign ncore component: {stdout}", shell=True)

        set_ncore_cmd = [BURN_PROGRAM, "-i", str(self._bin_file), "--psc_bl1", security_artifacts[PSC_BL1_SECTION].name, "--psc_bct", security_artifacts[PSC_BCT_SECTION].name,
                         "--psc_fw", security_artifacts[PSC_FW_SECTION].name, "--ncore", ncore_component_signed_path.name, "set_signed_fw_components"]
        logging.info("Inserting NCore section...")
        logging.debug(f"running {set_ncore_cmd}")
        execute_check(set_ncore_cmd, "Failed to insert ncore to the image: {stdout}", shell=True)

        if not self._args.noencrypt:
            if MLXFWENC_PROGRAM is None:
                error_exit("Image encryption tool is missing")
            check_python_version()
            self._enc_bin_h = self.add_temp_file()
            enc_bin_file = self._enc_bin_h.name
            mlxfwenc_cmd = [MLXFWENC_PROGRAM, "-i", str(self._bin_file), "--output-file", str(enc_bin_file), "encrypt", "--key", "835fa5cdfde6dc34fe9a61efb60919d3d432b5ef4a0a5f6e49eb4f298be0e263"]
            logging.info("Encrypting the image...")
            logging.debug(f"running {mlxfwenc_cmd}")
            execute_check(mlxfwenc_cmd, "Failed to encrypt the image: {stdout}")
            shutil.copy2(enc_bin_file, self._bin_file)

        logging.info("Signing the image...")
        encapsulation_header_data = self.add_temp_file()
        get_encapsulation_header_cmd = [BURN_PROGRAM, "-i", str(self._bin_file), "get_encapsulation_header_data", str(encapsulation_header_data.name)]
        logging.debug(f"running {get_encapsulation_header_cmd}")
        execute_check(get_encapsulation_header_cmd, "Failed to extract fw image: {stdout}", shell=True)

        extracted_fw_image = self.add_temp_file()
        extract_fw_image_cmd = [BURN_PROGRAM, "-i", str(self._bin_file), "extract_fw_data", str(extracted_fw_image.name)]
        logging.debug(f"running {extract_fw_image_cmd}")
        execute_check(extract_fw_image_cmd, "Failed to extract fw image: {stdout}", shell=True)

        with open(str(encapsulation_header_data.name), 'r') as file:
            parameters = json.load(file)
        encapsulation_header = self.add_temp_file()
        nvsign_cmd = ["python3.8", nvsign_path, extracted_fw_image.name, "--magic UPDT --noencrypt",
                      "--PSC_BL1_ver_major", parameters["PSCB"]["u8_ver_major"], "--PSC_BL1_ver_minor", parameters["PSCB"]["u8_ver_minor"], "--PSC_BL1_ratchet", parameters["PSCB"]["u8_ratchet_level"],
                      "--PSC_FW_ver_major", parameters["PFWM"]["u8_ver_major"], "--PSC_FW_ver_minor", parameters["PFWM"]["u8_ver_minor"], "--PSC_FW_ratchet", parameters["PFWM"]["u8_ratchet_level"],
                      "--NCORE_ver_major", parameters["NCOR"]["u8_ver_major"], "--NCORE_ver_minor", parameters["NCOR"]["u8_ver_minor"], "--NCORE_ratchet", parameters["NCOR"]["u8_ratchet_level"],
                      "--sign_key", security_artifacts[NCORE_STAGE1_KEY_SECTION].name, "-o", str(encapsulation_header.name)]
        logging.debug(f"running {nvsign_cmd}")
        execute_check(nvsign_cmd, "Failed to sign for fw update: {stdout}", shell=True)

        with open(encapsulation_header.name, 'rb') as signed_fw_data_file, open(str(self._bin_file), 'ab') as image_file:
            image_file.write(signed_fw_data_file.read(8192))

    def handle_security(self):
        full_query_map = query(str(self._bin_file), False, True, False, self._args.quick_query, self._args.additional_burn_query_args + self._args.additional_common_args, "q", "full")

        security_mode = self.parse_security_mode(full_query_map)
        if security_mode == SECURITY_MODE.NONE:
            return
        if security_mode == SECURITY_MODE.UNKNOWN:
            error_exit("Failed to handle security mode - Unknown mode")

        encrypted_mode = self.parse_encryption_mode(security_mode, full_query_map)

        security_artifacts = {}
        enc_bin_file = None
        mlxfwenc_cmd = None
        set_auth_tag_needed = False

        security_artifacts = self.get_security_artifacts(security_mode, encrypted_mode, SECURITY_GEN.LEGACY_SECURITY)

        if security_mode == SECURITY_MODE.FW_UPDATE or security_mode == SECURITY_MODE.SECURE_BOOT:
            flint_cmd = [BURN_PROGRAM, "-i", str(self._bin_file), "set_public_keys", security_artifacts[PUBLIC_KEY_SECTION].name]
            execute_check(flint_cmd, "Failed to set the public key section into the image: {rc}")

        if encrypted_mode != ENCRYPTED_MODE.NONE:
            if MLXFWENC_PROGRAM is None:
                error_exit("Image encryption tool is missing")
            check_python_version()

            self._enc_bin_h = self.add_temp_file()
            enc_bin_file = self._enc_bin_h.name
            mlxfwenc_cmd = [MLXFWENC_PROGRAM, "-i", str(self._bin_file), "--gcm-iv", security_artifacts[GCM_IV_SECTION], "--output-file", str(enc_bin_file), "encrypt", "--key", security_artifacts[ENCRYPTION_KEY_SECTION]]
            set_auth_tag_needed = dev_mgt.DevMgt.is_cx7(self._args.dm_dev_id) and full_query_map.get("FW_Version").startswith("28")

        if self._non_encrypted_bin_file and encrypted_mode != ENCRYPTED_MODE.NONE:
            shutil.copy2(self._bin_file, self._non_encrypted_bin_file)
            sign_cmd = [BURN_PROGRAM, "-i", str(self._non_encrypted_bin_file)]
            sign_cmd += ["--private_key", security_artifacts[PRIVATE_KEY_SECTION].name, '--key_uuid "{}"'.format((security_artifacts[UUID_SECTION]))]
            sign_cmd += ["--public_key", security_artifacts[PUBLIC_KEY_SECTION].name]
            if encrypted_mode == ENCRYPTED_MODE.BEFORE_SIGN:
                sign_cmd += ["--nonencrypted_image", str(self._non_encrypted_bin_file)]
            sign_cmd.append("rsa_sign")
            logging.debug("Signing non-encrypted image given by flag --wrimage_non_encrypted {}".format(self._non_encrypted_bin_file))
            logging.debug("running {}".format((sign_cmd)))
            execute_check(sign_cmd, "Failed to sign the image: {stdout}", shell=True)

        if encrypted_mode == ENCRYPTED_MODE.BEFORE_SIGN and not self._args.noencrypt:
            logging.info("Encrypting the image...")
            logging.debug("running {}".format((mlxfwenc_cmd)))
            execute_check(mlxfwenc_cmd, "Failed to encrypt the image: {stdout}")
        elif set_auth_tag_needed and not self._args.noencrypt:
            logging.info("Encrypting the image...")
            logging.debug("running {}".format((mlxfwenc_cmd)))
            execute_check(mlxfwenc_cmd, "Failed to encrypt the image: {stdout}")
            logging.info("Set auth-tag in boot-record...")
            self.set_auth_tag()

        sign_cmd = self.generate_sign_cmd(security_mode, encrypted_mode, security_artifacts, enc_bin_file)
        logging.info("Signing the image...")
        logging.debug("running {}".format((sign_cmd)))
        execute_check(sign_cmd, "Failed to sign the image: {stdout}", shell=True)

        if encrypted_mode == ENCRYPTED_MODE.AFTER_SIGN and not self._args.noencrypt:
            logging.info("Encrypting the image...")
            logging.debug("running {}".format((mlxfwenc_cmd)))
            execute_check(mlxfwenc_cmd, "Failed to encrypt the image: {stdout}")

        if encrypted_mode != ENCRYPTED_MODE.NONE and not self._args.noencrypt:
            shutil.copy2(enc_bin_file, self._bin_file)

    def parse_security_mode(self, query_map: dict) -> SECURITY_MODE:
        security_mode = SECURITY_MODE.NONE
        is_legacy = False

        secAttrs = query_map.get("Security_Attributes", [])

        if query_map.get("Default_Update_Method") == "Legacy":
            is_legacy = True

        if query_map.get("Dev_Secure_Boot_Cap") == "Enabled":
            security_mode = SECURITY_MODE.SECURE_BOOT
        elif "signed-fw" in secAttrs or "secure-fw" in secAttrs:
            security_mode = SECURITY_MODE.FW_UPDATE

        if is_legacy and security_mode != SECURITY_MODE.NONE:
            security_mode = SECURITY_MODE.UNKNOWN

        if security_mode == SECURITY_MODE.NONE:
            if query_map.get("Default_Update_Method") == "fw_ctrl":
                security_mode = SECURITY_MODE.SHA_DIGEST

            else:
                flint_cmd = [BURN_PROGRAM, "-i", str(self._bin_file)] + self._args.additional_common_args + ["v", "showitoc"]
                logging.debug("running {}".format(flint_cmd))
                proc_ret = execute_check(flint_cmd, "Failed to run 'flint verify showitoc' on the image: {rc}")

                for line in proc_ret.stdout.split(os.linesep):
                    m = regex_match(r"type\s*:\s*(.+)", line)
                    if m and m.group(1) == "0xa0":
                        security_mode = SECURITY_MODE.SHA_DIGEST
                        break

        logging.debug("Security-Mode: {}".format((security_mode.name)))
        return security_mode

    def parse_encryption_mode(self, security_mode: SECURITY_MODE, query_map: dict) -> ENCRYPTED_MODE:
        encrypted_mode = ENCRYPTED_MODE.NONE

        if security_mode.value > SECURITY_MODE.SHA_DIGEST.value:
            encrypted_mode = ENCRYPTED_MODE(int(query_map.get("Encrypted_level", ENCRYPTED_MODE.NONE.value)))

        logging.debug("Encryption-Mode: {}".format((encrypted_mode.name)))
        return encrypted_mode

    def get_security_artifacts(self, security_mode, encrypted_mode, security: SECURITY_GEN):
        security_artifacts = {}

        # with tempfile.Te
        fw_str = normalize_XML(self._args.fw_file.read_text(encoding='utf-8'))

        xml_p = ElementTree.fromstringlist(("<root>", fw_str, "</root>"))  # .getroot()

        relevant_bin_sections = []
        relevant_str_sections = []

        if security is SECURITY_GEN.FOURTH_GEN_SECURITY:
            relevant_bin_sections.extend([PSC_BL1_SECTION, PSC_BCT_SECTION, PSC_FW_SECTION])
            relevant_bin_sections.extend([NCORE_STAGE1_KEY_SECTION, NCORE_STAGE2_KEY_SECTION, NCORE_ENCRYPTION_KEY_SECTION, NCORE_JSON_SECTION])

        if security_mode == SECURITY_MODE.FW_UPDATE or security_mode == SECURITY_MODE.SECURE_BOOT:
            relevant_bin_sections.extend([PUBLIC_KEY_SECTION])
            if security is SECURITY_GEN.LEGACY_SECURITY:
                relevant_bin_sections.extend([PRIVATE_KEY_SECTION])
                relevant_str_sections.extend([UUID_SECTION])
        if encrypted_mode != ENCRYPTED_MODE.NONE:
            relevant_str_sections.extend([GCM_IV_SECTION, ENCRYPTION_KEY_SECTION])

        if relevant_bin_sections:
            for element in xml_p.iterfind("bsection"):
                name = element.get('name')
                if name in relevant_bin_sections:
                    element_bin_file = bin_sect_to_tempfile(element)
                    self._temp_files.append(element_bin_file)
                    security_artifacts[name] = element_bin_file
        if relevant_str_sections:
            for element in xml_p.iterfind("section"):
                name = element.get('name')
                if name in relevant_str_sections:
                    element_string = bin_sect_to_string(element)
                    security_artifacts[name] = element_string
        return security_artifacts

    def generate_sign_cmd(self, security_level: SECURITY_MODE, encrypted_mode: ENCRYPTED_MODE, security_artifacts: dict, enc_bin_file: Path):
        if security_level == SECURITY_MODE.NONE:
            return None

        input_file = self._bin_file
        if encrypted_mode == ENCRYPTED_MODE.BEFORE_SIGN and not self._args.noencrypt:
            input_file = enc_bin_file

        command = "rsa_sign"
        if security_level == SECURITY_MODE.SHA_DIGEST or (security_level == SECURITY_MODE.FW_UPDATE and encrypted_mode == ENCRYPTED_MODE.NONE):
            command = "sign"

        flint_cmd = [BURN_PROGRAM, "-i", str(input_file)]

        if security_level != SECURITY_MODE.SHA_DIGEST:
            flint_cmd += ["--private_key", security_artifacts[PRIVATE_KEY_SECTION].name, '--key_uuid "{}"'.format((security_artifacts[UUID_SECTION]))]
            if security_level == SECURITY_MODE.SECURE_BOOT or encrypted_mode != ENCRYPTED_MODE.NONE:
                flint_cmd += ["--public_key", security_artifacts[PUBLIC_KEY_SECTION].name]
                if encrypted_mode == ENCRYPTED_MODE.BEFORE_SIGN:
                    flint_cmd += ["--nonencrypted_image", str(self._bin_file)]
        flint_cmd.append(command)

        return flint_cmd

    def set_auth_tag(self):
        # Read auth-tag from the encrypted image from address 0x4f0 (last DWORD in boot-record for CX7)
        enc_bin_file = self._enc_bin_h.name
        read_tag_cmd = [BURN_PROGRAM, "-i", enc_bin_file, "rb", "0x4f0", "0x4"]
        ret = execute_check(read_tag_cmd, "Failed to read address 0x4f0 from the image: {rc}")

        # Set auth-tag in the original (non-encrypted) image
        auth_tag = ret.stdout.strip(" \t\n\\{\\}")
        logging.debug("Extracted auth-tag {}".format(auth_tag))
        set_tag_cmd = [BURN_PROGRAM, "-i", str(self._bin_file), "ww", "0x4f0", auth_tag]
        ret = execute_check(set_tag_cmd, "Failed to write auth_tag {} to the image in address 0x4f0:".format(auth_tag))
