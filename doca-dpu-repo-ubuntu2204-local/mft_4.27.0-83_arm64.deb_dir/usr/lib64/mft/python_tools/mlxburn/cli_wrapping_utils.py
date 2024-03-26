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

from pathlib import Path
from itertools import chain


class I_OptArgGenerator():
    _option_prefix = "--"
    def __init__(self, option: str, *args):
        self.option = option
        self._args = args
        self.optArgs = ["{}{}".format((self._option_prefix), (option))]
        self._generateArgs()

    @classmethod
    def set_option_prefix(cls, option_prefix):
        cls._option_prefix = option_prefix

    def _generateArgs(self):
        raise NotImplementedError


class addFlagOAG(I_OptArgGenerator):
    def _generateArgs(self): pass


class addPositionalFlagOAG(I_OptArgGenerator):
    def __init__(self, option):
        super().__init__(option)
        self.optArgs = [option]
    def _generateArgs(self): pass


class inputArgPipeOAG(I_OptArgGenerator):
    def _generateArgs(self):
        if self._args[0]:
            self.optArgs.append(str(self._args[0]))
        else:
            self.optArgs = []


# applies given function on given args, sets the resaults as output args
# if result is None, erases the whole option
class conversionFuncOAG(I_OptArgGenerator):
    def __init__(self, option, func, *args):
        self._func = func
        super().__init__(option, *args)

    def _generateArgs(self):
        func_ret = self._func(*self._args)
        if func_ret:
            self.optArgs.append(str(func_ret))
        else:
            self.optArgs = []


class extensionFuncOAG(I_OptArgGenerator):
    def __init__(self, func, *args):
        self._func = func
        super().__init__(None, *args)

    def _generateArgs(self):
        func_ret = self._func(*self._args)
        if func_ret:
            self.optArgs = list(chain(*[["{}{}".format((self._option_prefix), (str(opt)))] + [str(arg) for arg in filter(lambda x:x, args)] for opt, args in func_ret.items()]))
            # self.optArgs = [f"{self._option_prefix}{str(opt)}" for opt in func_ret]
        else:
            self.optArgs = []


class extensionPositionalOAG(I_OptArgGenerator):
    def __init__(self, *args):
        super().__init__(None, *args)

    def _generateArgs(self):
        self.optArgs = self._args


class CmdGenerator():
    def __init__(self, prog: Path) -> None:
        self._prog = prog
        self._argList = []
        self._cmd = [self._prog]

    def add_argument(self, arg: I_OptArgGenerator):
        self._argList.append(arg)

    def get_cmd(self) -> "list[str]":
        for arg in self._argList:
            self._cmd.extend(arg.optArgs)
        return self._cmd
