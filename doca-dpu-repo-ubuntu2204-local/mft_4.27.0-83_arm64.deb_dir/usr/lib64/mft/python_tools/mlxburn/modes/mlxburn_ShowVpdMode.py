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
from cli_wrapping_utils import *
from mlxburn_constants import MLXVPD_PROGRAM
from modes.mlxburn_mode import I_MlxburnMode
from mlxburn_utils import execute_check


class ShowVpdMode(I_MlxburnMode):
    def execute(self):
        cmd = self.prepare_vpd_command()
        logging.debug("Running {}".format((cmd)))
        execute_check(cmd, "Failed to Read VPD: {rc}", None)

    def prepare_vpd_command(self):
        cmd_gen = CmdGenerator(MLXVPD_PROGRAM)
        I_OptArgGenerator.set_option_prefix("-")

        cmd_gen.add_argument(inputArgPipeOAG("d", self._args.mst_dev))
        if self._args.vpd_rw:
            cmd_gen.add_argument(addFlagOAG("vpd_rw"))

        return cmd_gen.get_cmd()
