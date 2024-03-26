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
import logging
from cli_wrapping_utils import *
from mlxburn_constants import BURN_PROGRAM
from modes.mlxburn_mode import I_MlxburnMode
from mlxburn_utils import execute_check


class BurnMode(I_MlxburnMode):
    def __init__(self, args: argparse.Namespace, bin_file: Path):
        super().__init__(args)
        self._bin_file = bin_file

    def execute(self):
        burn_cmd = self.prepare_burn_command()
        logging.debug("running {}".format((burn_cmd)))
        execute_check(burn_cmd, "Image burn failed: {rc}", None)

        logging.info("Image burn completed successfully.")

    def prepare_burn_command(self):
        cmd_gen = CmdGenerator(BURN_PROGRAM)
        I_OptArgGenerator.set_option_prefix("--")

        cmd_gen.add_argument(inputArgPipeOAG("device", self._args.mst_dev))
        if self._args.nofs:
            cmd_gen.add_argument(addFlagOAG("nofs"))
        cmd_gen.add_argument(inputArgPipeOAG("image", self._bin_file))
        if self._args.force:
            cmd_gen.add_argument(addFlagOAG("yes"))
        if self._args.quick_query:
            cmd_gen.add_argument(addFlagOAG("qq"))
        cmd_gen.add_argument(inputArgPipeOAG("i2c_secondary", self._args.i2c_secondary))
        additional = self._args.additional_burn_query_args + self._args.additional_burn_args + self._args.additional_common_args
        cmd_gen.add_argument(extensionPositionalOAG(*additional))
        cmd_gen.add_argument(addPositionalFlagOAG("burn"))
        cmd_gen.add_argument(inputArgPipeOAG("vsd", self._args.vsd))

        return cmd_gen.get_cmd()
