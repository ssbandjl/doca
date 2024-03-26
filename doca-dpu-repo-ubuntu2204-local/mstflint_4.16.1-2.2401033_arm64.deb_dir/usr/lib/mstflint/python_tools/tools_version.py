# Copyright (c) 2004-2010 Mellanox Technologies LTD. All rights reserved.
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
#--

# To be replaced by an external script:
TOOLS_GIT_SHA="N/A"
MSTFLINT_VERSION_STR="mstflint 4.16.1"
TOOLS_BUILD_TIME="Apr 25 2022, 21:22:33"


####################################################################
def GetVersionString(execName, toolVersion = None):
    res = ""
    if (toolVersion == None or toolVersion == ""):
        res = execName + ", "
    else:
        res = "%s %s, " % (execName,toolVersion)

    res += "%s, built on %s. Git SHA Hash: %s" % (MSTFLINT_VERSION_STR, TOOLS_BUILD_TIME, TOOLS_GIT_SHA)
    return res
####################################################################


def PrintVersionString(execName, toolVersion = None):
    print (GetVersionString(execName, toolVersion))


get_version_string = GetVersionString
print_version_string = PrintVersionString
  
