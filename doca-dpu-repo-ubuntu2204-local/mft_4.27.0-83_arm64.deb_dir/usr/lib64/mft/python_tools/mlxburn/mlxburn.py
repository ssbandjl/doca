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

import platform
import sys
# autopep8: off
TOOL_MIN_PYTHON_VERSION = (3, 4, 0)
py_major, py_minor = [int(e) for e in platform.python_version_tuple()[:2]]
min_py_major, min_py_minor, _ = TOOL_MIN_PYTHON_VERSION
if py_major < min_py_major or py_minor < min_py_minor:
    sys.exit("Error: this tool requires python version >= {}".format('.'.join([str(a) for a in TOOL_MIN_PYTHON_VERSION])))
# autopep8: on
import logging
from modes.mlxburn_BurnMode import BurnMode
from cli_wrapping_utils import *
from mlxburn_constants import *
from mlxburn_argumentParser import MlxBurn_ArgumentParser
from modes.mlxburn_ImageGenerateMode import ImageGenerateMode
from modes.mlxburn_QueryMode import QueryMode
from modes.mlxburn_ShowVpdMode import ShowVpdMode
from modes.mlxburn_ViewVersionMode import ViewVersionMode


def init_logging(args):
    logging.basicConfig(style="{", format="-{levelname[0]}- {message}", level=logging.getLevelName(args.verbosity), stream=sys.stdout)
    logging.debug("Passed arguments:\n{}".format((str(args))))


MAIN_MODE_OBJECTS = {
    MAIN_MODE.SHOW_FW_VER: ViewVersionMode,
    MAIN_MODE.QUERY: QueryMode,
    MAIN_MODE.SHOW_VPD: ShowVpdMode,
    MAIN_MODE.IMAGE: ImageGenerateMode
}


def main(*argv):
    mode = None
    try:
        argParser = MlxBurn_ArgumentParser()
        args = argParser.parse_args(argv)

        init_logging(args)

        mode = MAIN_MODE_OBJECTS[args.main_mode](args)
        ret = mode.execute()
        if args.main_mode == MAIN_MODE.IMAGE and args.mst_dev:
            burn_mode = BurnMode(args, ret)
            burn_mode.execute()
    except KeyboardInterrupt as k:
        exit(k)
    except BaseException:
        raise
    finally:
        del mode
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(*sys.argv[1:]))
    except Exception as e:
        print("Error: {0}. Exiting...".format(e))
        sys.exit(1)
