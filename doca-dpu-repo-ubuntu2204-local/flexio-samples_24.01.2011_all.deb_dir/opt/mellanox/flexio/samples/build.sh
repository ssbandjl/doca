#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# The script used for build FlexIO SDK samples through meson and ninja packages

set -e +x

export_alias=''
CHECK_COMPAT=0
CLEAN=''
JOBS=''
VERBOSE=''
MESONEXTARGS=''
VERBOSE_ARG=0
CHECK_DEV_INPUT_ARG="false"
EN_SIMX_LOG_ARG="false"
DWERROR='true'
OPTIMIZE_LEVEL="1"
BUILD_DIR="build"
WORKDIRS="${BUILD_DIR}"

while [[ $# -gt 0 ]]; do
	key="$1"
	shift # past argument
	case $key in
		--check-compatibility)
			CHECK_COMPAT=1
			;;
		--clean)
			CLEAN="$key"
			;;
		--rebuild)
			CLEAN="$key"
			;;
		--allow-warnings)
			DWERROR='false'
			;;
		--build-dir)
			shift
			BUILD_DIR="$1"
			;;
		-j)
			re='^[0-9]+$'
			JOBSNMB="$1"
			if [[ "${JOBSNMB}" == "" || ${JOBSNMB} == -* || ! ${JOBSNMB} =~ $re ]]; then
				echo "Invalid jobs number"
				exit 1
			fi
			JOBS="-j "${JOBSNMB}
			shift
			;;
		-O)
			OPTIMIZE_LEVEL="$1"
			shift
			case $OPTIMIZE_LEVEL in
				0|1|2)
					;;
				*)
					echo "Invalid optimization level $OPTIMIZE_LEVEL- could be from 0 to 2"
					exit 1
					;;
			esac
			;;
		-v)
			VERBOSE="-v"
			;;
		-h|--help)
			echo "usage: build.sh [--clean] [--rebuild] [--allow-warnings] [-j JOBS] "
			echo "                [-v] [-O level] [--check-compatibility]"
			echo "                [--build-dir BUILD]"
			echo "  -O level - level of optimization and debug from 0 to 2."
			echo "     0 - optimization 0 with debug"
			echo "     1 - optimization 2 with debug and lto - default"
			echo "     2 - optimization 3 with no debug and with lto"
			echo "  -v - print commands that ninja run."
			exit 1
			;;
		*)    # unknown option
			echo "ERROR: unknown parameter: $key"
			exit 1
			;;
	esac
done

if [ $CLEAN ]; then
	rm -rf ${WORKDIRS}
	if [ "$CLEAN" == "--clean" ]; then
		exit 0
	fi
fi

case $OPTIMIZE_LEVEL in
	0)
		MESONBUILD="--buildtype=debug"
		;;
	1)
		MESONBUILD="--buildtype=debugoptimized"
		;;
	2)
		MESONBUILD="--buildtype=release"
		;;
esac

if [ $CHECK_COMPAT -eq 1 ]; then
	./check_compat.sh
	if [ $? -ne 0 ]; then
		exit 1
	fi
fi


MESONARGS="-Dwarning_level=3 -Ddefault_library=static -Dwerror=$DWERROR"
MESONARGUMENTS="$MESONBUILD $MESONEXTARGS $MESONARGS"
$export_alias
CONFIG=$BUILD_DIR
if [ -d $CONFIG ]; then
	meson setup $CONFIG --reconfigure $MESONARGUMENTS
else
	meson setup $CONFIG $MESONARGUMENTS
fi
ninja -C $CONFIG $JOBS $VERBOSE

echo
echo " *** THE BUILD IS SUCCESSFULLY FINISHED ***"
echo