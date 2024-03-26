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
from pathlib import Path
from re import match as regex_match

import dev_mgt
import tools_version

from mlxburn_constants import *
from mlxburn_utils import is_valid_file


def additional_args_dict_view(args_list):
    for arg_properties in args_list:
        yield {prop: arg for prop, arg in zip(ADDITIONAL_ARGS_HEADER, arg_properties)}


class ParseDeviceAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string):
        match_obj = regex_match(BDF_PATTERN, value)
        if match_obj:
            if match_obj.group(1) == '0000':
                value = match_obj.group(2)
            setattr(namespace, "{}_orig".format((self.dest)), match_obj.group(0))
        setattr(namespace, self.dest, value)


class ParseDeviceTypeAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string):
        if int(value, 0) == MIC_LEGACY_4TH_GEN_DEVICE_ID:
            value = str(CX3PRO_DEVICE_ID)
        dev_props = dev_mgt.DevMgt.getDevIdOffline(int(value, 0))
        if dev_props["return_code"] != 0:
            dev_props = dev_mgt.DevMgt.sw_id2hw_id(int(value, 0))
            if dev_props["return_code"] != 0:
                raise ValueError("Invalid device-type: {}".format((value)))
        setattr(namespace, self.dest, dev_props["hw_device_id"])
        setattr(namespace, "dm_dev_id", dev_props["dm_device_id"])


class SplitAppendMicOptionAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string):
        dest_value = getattr(namespace, self.dest, []) + list(map(lambda x: str.strip(x, "\""), value.split()))
        setattr(namespace, self.dest, dest_value)


class AppendAdditionalAction(argparse._AppendAction):
    _prefix = "--"
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs='*', **kwargs)
    def __call__(self, parser, namespace, values, option_string):
        values = [self._prefix + option_string.lstrip("-")] + values
        for val in values:
            super().__call__(parser, namespace, val, option_string)


class AppendAdditionalOptionMICAction(AppendAdditionalAction):
    _prefix = "-"


def str_check_len(s: str, l: int) -> str:
    if len(s) > l:
        raise ValueError
    return s


class MlxBurn_ArgumentParser():
    def __init__(self):
        # Main Parser of all modes
        self._general_ArgumentParser = argparse.ArgumentParser(allow_abbrev=False)
        self.prog = self._general_ArgumentParser.prog.split(".")[0]
        self._general_ArgumentParser.add_argument("-v", "--version", action="version", version=tools_version.GetVersionString(self.prog))

        # Common to all modes, parent of all
        self._common_ArgumentParser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
        self._common_ArgumentParser.add_argument("-V", "--verbose", dest="verbosity", choices=LOGGING_LEVELS.keys(), default=DEFAULT_LOGGING_LEVEL)

        # Special parsers initializations
        self._mic_ArgumentParser = self._init_mic_argumentParser()
        self._burner_ArgumentParser = self._init_burner_argumentParser()

        # Operation modes subparsers initializations
        self._subparsers = self._general_ArgumentParser.add_subparsers(dest="main_mode", description='{} main operation modes.run: "{} <subcommand> -h" for additional help for the specific mode.'.format((self.prog), (self.prog)))
        self._image_argumentParser = self._init_image_mode_argumentParser()
        self._query_argumentParser = self._init_query_mode_argumentParser()
        self._show_fw_version_argumentParser = self._init_show_fw_ver_ArgumentParser()
        self._show_vpd_argumentParser = self._init_show_vpd_argumentParser()
        self._subparsers.required = True

    def _init_image_mode_argumentParser(self):
        image_mode_argParser = self._subparsers.add_parser("image", allow_abbrev=False, parents=[self._common_ArgumentParser, self._mic_ArgumentParser, self._burner_ArgumentParser], help="Burn or generate FW image for Mellanox devices.")

        source_group = image_mode_argParser.add_mutually_exclusive_group(required=True)
        source_group.add_argument("-i", "-image", "--image", dest="input_img", metavar="fw-image-file", type=Path, help="Do not generate image. Use the given fw image instead.")
        source_group.add_argument("-imd", "-img_dir", "--img_dir", metavar="image directory", type=Path, help="Do not generate image. Select the image to burn from the *.bin in the given directory.")
        source_group.add_argument("-f", "-fw", "--fw", dest="fw_file", metavar="mellanox-fw-file", type=Path, help="Specify Mellanox FW released Firmware File to use (file extension is .mlx)")
        source_group.add_argument("-fd", "-fw_dir", "--fw_dir", metavar="dir", type=Path, help="When specified, the auto detected fw files will be looked for in the given directory. Applicable for burn operation.")

        target_group = image_mode_argParser.add_mutually_exclusive_group(required=True)
        target_group.add_argument("-d", "-dev", "--dev", dest="mst_dev", metavar="mst-dev", action=ParseDeviceAction, help="Burn the image using the given MST device.")
        target_group.add_argument("-o", "-wrimage", "--wrimage", dest="bin_file", metavar="fw-image-file", type=Path, help="Write the image to the given file.")

        image_mode_argParser.add_argument("-c", "-conf", "--conf", dest="conf_file", metavar="fw-conf-file", type=Path, help="FW configuration file (.ini). Needed when generating image (not using -dev flag) or if configuration auto detection fails.")

        image_mode_argParser.add_argument("-ex", "-exp_rom", "--exp_rom", metavar="exp-rom-file", type=Path, help='Integrate the given expansion rom file to the FW image. If the exp-rom-file is set to "AUTO", expansion rom file is auto detected from the files rom in the exp_rom_dir (see below). NOTE: Exp rom auto detection is done for devices that are already burned with an exp-rom image. If "-exp_rom AUTO" is specified for a device with no exp-rom, it would be burnt with no exp rom. To add exp-rom to a device, manually supply the exp rom file to use.')
        image_mode_argParser.add_argument("-exd", "-exp_rom_dir", "--exp_rom_dir", metavar="exp_rom_dir", type=Path, help='The directory in which to look for expansion rom file when "-exp_rom AUTO" is specified. By default, exp-rom files are searched in <fw file directory>/exp_rom/*')
        image_mode_argParser.add_argument("-vpr", "-vpd_r_file", "--vpd_r_file", type=Path, help="Embed the given VPD Read-Only section in the generated image. The vpd_r_file should contain the vpd read only section and the first dword of the vpd write-able section. The file is in binary format, and its size must be a multiple of 4 bytes. Please refer to PCI base spec for VPD structure info.")
        image_mode_argParser.add_argument("-bg", "-base_guid", "--base_guid", type=lambda s: int(s, 16), metavar="GUID", help="Set the given GUID as the image base GUID. The base GUID is used to derive GUIDs and MACs for the HCA ports. It is assumes that 16 GUIDs (base_guid to base_guid + 15) are reserved for the card. *On ConnectX4: only GUIDs will be derrived according to the HCA's configuration.")
        image_mode_argParser.add_argument("-bm", "-base_mac", "--base_mac", type=lambda s: int(s, 16), metavar="MAC", help="Set the given MAC as the image base MAC. the base MAC is used to derrvie MACs for the HCA ports according to the device configuration (Connect-IB and ConnectX-4 and above Adapter Cards only).")
        image_mode_argParser.add_argument("-i2c", "-i2c_secondary", "--i2c_secondary", type=lambda s: int(s, 16), help="")

        # Advanced
        image_mode_argParser.add_argument("-dt", "-dev_type", "--dev_type", action=ParseDeviceTypeAction, help="mlxburn must know the device type in order to work properly.  Use this flag if device type auto-detection fails.")

        conf_dir_group = image_mode_argParser.add_mutually_exclusive_group()
        conf_dir_group.add_argument("-cd", "-conf_dir", "--conf_dir", type=Path, metavar="dir", help="When specified, the auto detected configuration files will be looked for in the given directory, instead of in the firmware file directory. Applicable for burn operation.")
        conf_dir_group.add_argument("-cdl", "-conf_dir_list", "--conf_dir_list", nargs="+", default=[], help="When specified, the auto detected configuration files will be looked for in the given directories, instead of in the firmware file directory. Applicable for burn operation.")

        image_mode_argParser.add_argument("-ne", "-noencrypt", "--noencrypt", action="store_true", help="When specified, the tool will not encrypt bin.")
        image_mode_argParser.add_argument("--wrimage_non_encrypted", default=None, dest="non_encrypted_bin_file", metavar="fw-image-file", type=Path, help=argparse.SUPPRESS)
        image_mode_argParser.add_argument("-qq", "--quick_query", action="store_true", help="")

        image_mode_argParser.add_argument("-img_args", "--img_args", dest="additional_imgen_args", action=SplitAppendMicOptionAction, help=argparse.SUPPRESS)

        for arg_props in additional_args_dict_view(ADDITIONAL_IMGEN_ARGS):
            image_mode_argParser.add_argument('-{}'.format((arg_props["option"])), '--{}'.format((arg_props["option"])), dest="additional_imgen_args", action=AppendAdditionalOptionMICAction, help=arg_props["help"])

        for arg_props in additional_args_dict_view(ADDITIONAL_COMMON_ARGS):
            image_mode_argParser.add_argument('-{}'.format((arg_props["option"])), '--{}'.format((arg_props["option"])), dest="additional_common_args", action=AppendAdditionalOptionMICAction, help=arg_props["help"])

        image_mode_argParser.add_argument("-vsd", "--vsd", metavar="string", type=lambda s: str_check_len(s, VSD_MAX_LEN), help="")
        image_mode_argParser.set_defaults(additional_imgen_args=[])
        image_mode_argParser.set_defaults(additional_imgen_burn_args=[])
        image_mode_argParser.set_defaults(additional_common_args=[])

        image_mode_argParser.set_defaults(main_mode=MAIN_MODE.IMAGE)
        return image_mode_argParser

    def _init_query_mode_argumentParser(self):
        parser = self._subparsers.add_parser("query", parents=[self._common_ArgumentParser], help="Query the HCA or Switch device FW image.")
        parser.add_argument("-d", "-dev", "--dev", dest="mst_dev", action=ParseDeviceAction, required=True, metavar="mst-dev", help="Burn the image using the given MST device.")
        parser.set_defaults(main_mode=MAIN_MODE.QUERY)
        return parser

    def _init_show_fw_ver_ArgumentParser(self):
        parser = self._subparsers.add_parser("fwver", allow_abbrev=False, parents=[self._common_ArgumentParser, self._mic_ArgumentParser], help="When a device is given: Display current loaded firmware version (Deprecated).When a FW file is given (-fw flag): Display the file FW version.")

        target_group = parser.add_mutually_exclusive_group(required=True)
        target_group.add_argument("-d", "-dev", "--dev", dest="mst_dev", action=ParseDeviceAction, metavar="mst-dev", help="Burn the image using the given MST device.")
        target_group.add_argument("-f", "-fw", "--fw", dest="fw_file", metavar="mellanox-fw-file", type=Path, help="Specify Mellanox FW released Firmware File to use (file extension is .mlx)")
        target_group.add_argument("-fd", "-fw_dir", "--fw_dir", metavar="dir", type=Path, help="When specified, the auto detected fw files will be looked for in the given directory. Applicable for burn operation.")

        target_group.add_argument("-dt", "-dev_type", "--dev_type", action=ParseDeviceTypeAction, help="mlxburn must know the device type in order to work properly.  Use this flag if device type auto-detection fails.")

        parser.set_defaults(main_mode=MAIN_MODE.SHOW_FW_VER)
        return parser

    def _init_show_vpd_argumentParser(self):
        parser = self._subparsers.add_parser("vpd", parents=[self._common_ArgumentParser], help="Display the read only section of the PCI VPD (Vital Product Data) of the given device. NOTE: VPD feature may not be supported on certain board types.")

        parser.add_argument("-d", "-dev", "--dev", dest="mst_dev", action=ParseDeviceAction, required=True, metavar="mst-dev", help="Burn the image using the given MST device.")
        parser.add_argument("-vpd_rw", "--vpd_rw", dest="vpd_rw", action="store_true", help="Display also the read/write section of the PCI VPD of the given device.")

        parser.set_defaults(main_mode=MAIN_MODE.SHOW_VPD)
        return parser

    def _init_mic_argumentParser(self):
        parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)

        parser.add_argument("-ft", "-format", "--format", choices=("BINARY", "IMAGE"), default="BINARY", help="Specify which image format to use. Can be specified only with the -wrimage flag. Default is BINARY.")
        parser.add_argument("-nfi", "-nofs_img", "--nofs_img", action="store_true", help="When specified, generated image will not be fail-safe, and burn process will not be failsafe.")

        return parser

    def _init_burner_argumentParser(self):
        parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)

        parser.add_argument("-nf", "-nofs", "--nofs", action="store_true", help="When specified, burn process will not be failsafe.")
        parser.add_argument("-y", "-force", "--force", action="store_true", help='None interactive mode. Assume "yes" for all user questions.')
        parser.add_argument("-burn_args", "--burn_args", dest="additional_burn_args", action=SplitAppendMicOptionAction, help=argparse.SUPPRESS)

        for arg_props in additional_args_dict_view(ADDITIONAL_BURN_ARGS):
            opt_to_add = (arg_props["option"])
            parser.add_argument('-{}'.format(opt_to_add), '--{}'.format(opt_to_add), dest="additional_burn_args", action=AppendAdditionalAction, help=arg_props["help"])

        for arg_props in additional_args_dict_view(ADDITIONAL_BURN_QUERY_ARGS):
            opt_to_add = (arg_props["option"])
            parser.add_argument('-{}'.format(opt_to_add), '--{}'.format(opt_to_add), dest="additional_burn_query_args", action=AppendAdditionalAction, help=arg_props["help"])

        parser.set_defaults(additional_burn_args=[])
        parser.set_defaults(additional_burn_query_args=[])

        return parser

    def _additional_image_mode_validations(self, args):
        if args.bin_file and (args.input_img or args.img_dir):
            raise argparse.ArgumentError("-o/--wrimage argument not allowed with -i/--image or -imd/--img_dir arguments")
        if args.bin_file and not args.fw_file:
            self._image_argumentParser.error("the following arguments must be provided together: --wrimage, --fw")
        if args.bin_file and not args.conf_file:
            self._image_argumentParser.error("the following arguments must be provided together: --wrimage, --conf")
        if args.non_encrypted_bin_file and args.noencrypt:
            self._image_argumentParser.error("the following arguments must not be provided together: --noencrypt, --wrimage_non_encrypted")
        if args.fw_file:
            is_valid_file(args.fw_file)
        if args.conf_file:
            is_valid_file(args.conf_file)

    def _additional_image_mode_defaults(self, args):
        if args.fw_file:
            args.fw_dir = args.fw_file.parent

        if not args.conf_file:
            if not args.conf_dir_list:
                if not args.conf_dir and args.fw_dir:
                    args.conf_dir = args.fw_dir
                args.conf_dir_list.append(str(args.conf_dir))
            else:
                nested_list = [str(comma_sep).split(",") for comma_sep in args.conf_dir_list]  # split comma seperated
                args.conf_dir_list = [conf_dir.strip() for sublist in nested_list for conf_dir in sublist]  # flatten back

        if args.nofs_img:
            args.nofs = True

    def _back_comp_workaround(self, argv):
        argv = [arg.lstrip("-") if arg in map(lambda x: "-" + x, MAIN_MODE_FLAGS.values()) else arg for arg in argv]
        argv = ["query" if arg == "-q" else arg for arg in argv]

        foundMode = -1
        foundImageUniqueArg = -1
        foundVPDUniqueArg = -1
        foundGeneralArg = -1
        foundImgArgs = -1
        for i, arg in enumerate(argv):
            if foundMode == -1 and arg in MAIN_MODE_FLAGS.values():
                foundMode = i
            if arg in ("-d", "-dev", "-wrimage"):
                foundImageUniqueArg = i
            if foundGeneralArg == -1 and arg in ("-v", "--version", "-h", "--help"):
                foundGeneralArg = i
            if arg == "-img_args":
                foundImgArgs = i
            if arg == "-vpd_rw":
                foundVPDUniqueArg = i

        if foundImgArgs != -1:
            argv[foundImgArgs + 1] = '"{}"'.format(argv[foundImgArgs + 1].strip("\"\'"))

        deleteIndL, deleteIndH = 0, 0
        insertArgs = []
        insertInd = 0
        if foundGeneralArg == -1:
            if foundMode != -1:
                deleteIndL, deleteIndH = foundMode, foundMode + 1
                insertArgs = [argv[foundMode]]
            elif foundVPDUniqueArg != -1:
                insertArgs = [MAIN_MODE_FLAGS[MAIN_MODE.SHOW_VPD]]
            elif foundImageUniqueArg != -1:
                insertArgs = [MAIN_MODE_FLAGS[MAIN_MODE.IMAGE]]
        elif foundMode != -1:
            deleteIndL, deleteIndH = foundMode, foundMode + 1
            insertInd = 0 if foundMode < foundGeneralArg else foundGeneralArg + 1
            insertArgs = [argv[foundMode]]
        del(argv[deleteIndL:deleteIndH])
        argv[insertInd:insertInd] = insertArgs

        return argv

    def parse_args(self, argv) -> argparse.Namespace:
        # workaround for backward compatibility of main-modes
        argv = self._back_comp_workaround(argv)
        args = self._general_ArgumentParser.parse_args(argv)

        if args.main_mode in (MAIN_MODE.SHOW_FW_VER, MAIN_MODE.QUERY, MAIN_MODE.SHOW_VPD):
            return args

        self._additional_image_mode_validations(args)
        self._additional_image_mode_defaults(args)

        return args
