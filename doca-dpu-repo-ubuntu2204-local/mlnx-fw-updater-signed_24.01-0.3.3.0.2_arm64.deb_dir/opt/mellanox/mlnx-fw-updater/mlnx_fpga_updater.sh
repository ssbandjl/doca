#!/bin/bash
#
# Copyright (c) 2017 Mellanox Technologies. All rights reserved.
#
# This Software is licensed under one of the following licenses:
#
# 1) under the terms of the "Common Public License 1.0" a copy of which is
#      available from the Open Source Initiative, see
#      http://www.opensource.org/licenses/cpl.php.
#
# 2) under the terms of the "The BSD License" a copy of which is
#      available from the Open Source Initiative, see
#      http://www.opensource.org/licenses/bsd-license.php.
#
# 3) under the terms of the "GNU General Public License (GPL) Version 2" a
#      copy of which is available from the Open Source Initiative, see
#      http://www.opensource.org/licenses/gpl-license.php.
#
# Licensee has the right to choose one of the above licenses.
#
# Redistributions of source code must retain the above copyright
# notice and one of the license notices.
#
# Redistributions in binary form must reproduce both the above copyright
# notice, one of the license notices in the documentation
# and/or other materials provided with the distribution.

#INSTALL_TYPE="BUNDLE"
INSTALL_TYPE="OFED"

VERSION_NUM="2"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
bold=$(tput bold)
normal=$(tput sgr0)


#========================================================================================
# fHeader
# -------
#========================================================================================
fHeader(){
    header=$1
    echo "******************************************************************"
    echo -n "                      "
    echo ${header}
    echo "******************************************************************"
}

#========================================================================================
# fExit
# -----
# Called when exiting the script, in order to go back to the original Dir
#========================================================================================
fExit(){
    rm -rf /tmp/MFT_install
    exit 1
}

#========================================================================================
# fCheckIsOperationOK
# -------------------
# If the last command failed exit the script
#
# Input :
# $1 = the last operation that was performed (in use for user message)
#========================================================================================
fCheckIsOperationOK(){
    local RES=$?
    local OPER=$1
    if [ $RES == 0 ]; then
        echo -e "${GREEN} ${OPER} done${NC}"
    else
        echo -e "${RED} ${OPER} failed${NC}"
        fExit
    fi
}

#========================================================================================
# fCheckIfVarIsEmpty
# ------------------
#========================================================================================
fCheckIfVarIsEmpty(){
    local var=$1
    local description=$2
    if [ -z "$var" ]; then
        echo -e "${RED}${description} has no value${NC}"
        fExit
    fi
}

#========================================================================================
# fShowMenu
# ---------
# pring the array values to user as menu
# exit the script if array is empty
# recieve user choice
# return value choosen by user
# Input : array
# Output: uChoice
#========================================================================================
fshowMenu(){
    local array=("$@")
    local i=0
    local flag=0
    local arraySize=${#array[@]}

    if [ $arraySize == 0 ]; then
        echo "There are no options"
        fExit
    fi

    while [ $flag == 0 ]
    do
        i=0
        for val in ${array[@]}; do
            echo "${bold}$i) ${val}${normal}"
            let i+=1
        done
        echo -n "${bold}:${normal}"
        read uIndex

        #check if index is valid
        if [ "$uIndex" -ge  0 ]; then
            if [ "$uIndex" -lt  "$i" ]; then
                #uIndex is valid
                flag=1
            else
                echo "Not a valid option"
            fi
        else
            echo "Not a valid option"
        fi
    done
    uChoice=${array[$uIndex]}
}
#========================================================================================
# fFwReset
# --------
# Reset the FW
#========================================================================================
fFwReset(){
    local currDevice=$1
    fCheckIfVarIsEmpty $currDevice Device

    fHeader "FW reset"
#    mlxfwreset -d ${currDevice} reset_fsm_register
    mlxfwreset -d ${currDevice} --level 3 r -y --mst_flags="--with_fpga"
    fCheckIsOperationOK "FW reset"
}

#========================================================================================
# fLoadFpgaTools
# --------------
#========================================================================================
fLoadFpgaTools(){
    fHeader "Load FPGA tools"
    modprobe mlx5_fpga_tools
    fCheckIsOperationOK "Load mlx fpga tools"
}

#========================================================================================
# fMstRestart
# -----------
#========================================================================================
fMstRestart(){
    fHeader "MST restart"
    mst restart --with_fpga
    fCheckIsOperationOK "mst restart"
}

#========================================================================================
# fUpdateFPGA
# -----------
#========================================================================================
fUpdateFPGA(){
    local regDevice=$1

    fHeader "Update FPGA - Burn"

    local deviceRDMA=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  rdma | grep ${regDevice})
    fCheckIfVarIsEmpty $deviceRDMA "RDMA device was not found"

    #if not newer version

    local __fileName=$(basename $PATH_FPGA)
    local __newFpgaVersion=$(echo "$__fileName" | cut -d'.' -f1 | cut -d'_' -f4 | cut -d'v' -f2)
    local __oldFpgaVersion=$(mlx_fpga -d ${deviceRDMA} q |  grep version | cut -d':' -f2 | sed 's/[[:space:]]//g')
    echo "Current FPGA image version: $__oldFpgaVersion"
    echo "New FPGA image version    : $__newFpgaVersion"
    echo ""
    if [ $IS_FORCE_MODE == "Y" ]; then
        mlx_fpga -d ${deviceRDMA} b ${PATH_FPGA}
        fCheckIsOperationOK "Burn FPGA"
        WAS_FPGA_IMAGE_UPDATED="Y"
    else
        if (( __oldFpgaVersion < __newFpgaVersion  )); then
            mlx_fpga -d ${deviceRDMA} b ${PATH_FPGA}
            fCheckIsOperationOK "Burn FPGA"
            WAS_FPGA_IMAGE_UPDATED="Y"
        else
            if (( __oldFpgaVersion == __newFpgaVersion  )); then
                echo "Note: the new version is already installed"
            else
                echo "Note: The new image version is older than the current image version"
            fi

            echo ""
            echo -n "Do you want to install the new FPGA image ? (y/n) :"
            read __ansYN
            if [ $__ansYN == "y" ]; then
                mlx_fpga -d ${deviceRDMA} b ${PATH_FPGA}
                fCheckIsOperationOK "Burn FPGA"
                WAS_FPGA_IMAGE_UPDATED="Y"
            else
                echo "Skip installtion of FPGA image"
            fi
        fi
    fi
    #removed because : this operation will be part of the FW reset
    #fHeader "Update FPGA - Load "
    #mlx_fpga -d ${deviceRDMA} load --user
}

#========================================================================================
# fUpdateFW
# ---------
#========================================================================================
fUpdateFW(){
    local currDevice=$1
    fCheckIfVarIsEmpty $currDevice Device

    fHeader "Update FW"
    if [ $IS_FORCE_MODE == "N" ]; then
        if [ $WAS_FPGA_IMAGE_UPDATED == "N" ]; then
            mlxburn -d ${currDevice} -img_dir  ${PATH_FW}
        else
            mlxburn -d ${currDevice} -img_dir  ${PATH_FW} -force
        fi
    else
        mlxburn -d ${currDevice} -img_dir  ${PATH_FW} -force
    fi
    fCheckIsOperationOK "FW burn"
}

#========================================================================================
# fChooseDevices
# --------------
#========================================================================================
fChooseDevices(){
    local devicesArray=($(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep -v fpga))
    local arrayForMenu=()
    arrayForMenu+=("All")

    gDevicesWithRdma=()
    for device in ${devicesArray[@]}; do
        isRdmaDeviceExist=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  rdma | grep ${device} | wc -l)
        if [ $isRdmaDeviceExist ==  1 ]; then
            gDevicesWithRdma+=($device)
            arrayForMenu+=($device)
        fi
    done
    if [ "${#gDevicesWithRdma[@]}" == 1 ]; then
        uDevice=${gDevicesWithRdma[0]}
    else
        echo "${bold}choose device for installation: ${normal}"
        fshowMenu "${arrayForMenu[@]}"
        uDevice=$uChoice
    fi

    if [ "$uDevice" != "All" ]; then
        DEVICES=()
        DEVICES+=($uDevice)
    else
        DEVICES=${gDevicesWithRdma[@]}
    fi

}

#========================================================================================
# flookforRdmadevices
# -------------------
#========================================================================================
fCheckRdmaDevices(){
    local numOfRdmaDevices=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  rdma |wc -l)
    if [ $numOfRdmaDevices ==  0 ]; then
        fLoadFpgaTools
        fMstRestart
        numOfRdmaDevices=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  rdma |wc -l)
        if [ $numOfRdmaDevices ==  0 ]; then
            echo -e "${RED} cannot find RDMA devices ${OPER}: Aborting"
            fExit
        fi
    fi
}

#========================================================================================
# fInstallMFT
# -----------
#========================================================================================
fInstallMFT(){
    local MftTarFile=${1}
    fCheckIfVarIsEmpty ${1} "Path"

    fHeader "install MFT"

    mkdir /tmp/MFT_install
    tar -xzvf ${MftTarFile} -C /tmp/MFT_install
    fCheckIsOperationOK "Extract MFT"

    installFilePath="/tmp/MFT_install/$(ls  /tmp/MFT_install)/install.sh"

    ${installFilePath}
    fCheckIsOperationOK "Install MFT"

    rm -rf /tmp/MFT_install
    mst start --with_fpga
}

#========================================================================================
# fCheckKernel
# ------------
#========================================================================================
fCheckKernel(){
    local __dirKernel=$1
    local __currKernelVersion=$2
    local __currKernelLastCommit=$3
    fHeader "Check kernel version"

    local __bundleKernelVersion=$( ls ${__dirKernel} | grep -v devel | cut -d"-" -f2 | cut -d"_" -f1)
    local __bundleKernelLastCommit=$( ls ${__dirKernel} | grep -v devel | cut -d"-" -f2 | cut -d"_" -f2)

    if [ "$__currKernelVersion-$__currKernelLastCommit" != "$__bundleKernelVersion-$__bundleKernelLastCommit" ]; then
        echo -e "${RED}The current kernel version is: $__currKernelVersion-$__currKernelLastCommit ${NC}"
        echo -e "${RED}The bundle kernel version is: $__bundleKernelVersion-$__bundleKernelLastCommit ${NC}"
        echo -e "${RED}Please install the kernel from the bundle before running this installation again${NC}"
        fExit
    else
        echo -e "${GREEN}Kernel version is OK ${NC}"
    fi

    #check if develop kernel is installed (needed for MFT installation)
    local __kernelDevel=$(rpm -qa | grep kernel| grep $__bundleKernelVersion | grep $__bundleKernelLastCommit| grep devel)
    if [ -z "$__kernelDevel" ]; then
        echo -e "${RED}The kernel devel version: $__bundleKernelVersion-$__bundleKernelLastCommit is not installed${NC}"
        fExit
    fi
}

#========================================================================================
# fCheckMFT
# ---------
#========================================================================================
fCheckMFT(){
    fHeader "Check MFT"

    local mstVersion=$(mst version)
    if [ "$mstVersion" == "-E- You must be root to use mst tool" ]; then
        echo -e "${RED}You must be root in order to run this installtion script${NC}"
        fExit
    fi
    if [ "${PATH_MFT}" != "NONE" ]; then #when bundle
        currentMftVersion=$(echo "${mstVersion}" | cut -d" " -f3 | cut -d"," -f1)
        bundleMftVersion=$(ls ${PATH_MFT} | cut -d"-" -f2)-$(ls ${PATH_MFT} | cut -d"-" -f3)
        if [ -z "$currentMftVersion" ]; then
            echo -e "${bold}MFT is not installed =>  installing MFT${normal}"
            fInstallMFT ${PATH_MFT}
            currentMftVersion=$(mst version | cut -d" " -f3 | cut -d"," -f1)
        fi
        if [ "$currentMftVersion" != "$bundleMftVersion" ]; then
            echo "${bold}The MFT version is different from the bundle version => install MFT${normal}"
            fInstallMFT ${PATH_MFT}
            currentMftVersion=$(mst version | cut -d" " -f3 | cut -d"," -f1)
        fi
        if [ $IS_FORCE_MODE == "N" ]; then
            echo -e "${GREEN}MFT version is OK ${NC}"
            echo "But after installing a kernel the MFT needs to be installed again"
            echo -n "Do you want to install MFT? (y/n) :"
            read __MFTansYN
            if [ $__MFTansYN == "y" ]; then
                fInstallMFT ${PATH_MFT}
                currentMftVersion=$(mst version | cut -d" " -f3 | cut -d"," -f1)
            fi
        else
            fInstallMFT ${PATH_MFT}
            currentMftVersion=$(mst version | cut -d" " -f3 | cut -d"," -f1)
        fi
    else
        if [ -z "$mstVersion" ]; then
            echo -e "${RED}You need to install MFT before running this script${NC}"
            fExit
        else
            echo -e "${GREEN}MFT is installed${NC}"
        fi
    fi
}

#========================================================================================
# fCheckDevice
# --------------
#========================================================================================
fCheckDevice(){
    local currDevice=$1
    fCheckIfVarIsEmpty $currDevice "Device"

    #check : if device exist
    numRegularDeviceFound=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  -v fpga | grep ${currDevice} |wc -l)
    numRdmaDeviceFound=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  rdma | grep ${currDevice} |wc -l)
    if [ "$numRegularDeviceFound" != 1 ]; then
        echo -e "${RED}Device-${currDevice} is not found${NC}"
        fExit
    fi
    if [ "$numRdmaDeviceFound" != 1 ]; then
        echo -e "${RED}RDMA device is not found${NC}"
        fExit
    fi
    echo -e "${GREEN}The device: ${currDevice} is OK${NC}"
}

#========================================================================================
# fInstallDevice
# --------------
#========================================================================================
fInstallDevice(){
    local currentDevice=$1

    fCheckIfVarIsEmpty $currentDevice "Device"

    fFwReset $currentDevice
    fLoadFpgaTools
    fMstRestart
    fUpdateFPGA $currentDevice
    fUpdateFW $currentDevice
    fFwReset $currentDevice

    local __fileName=$(basename $PATH_FPGA)
    local __newFpgaVersion=$(echo "$__fileName" | cut -d'.' -f1 | cut -d'_' -f4 | cut -d'v' -f2)
    local deviceRDMA=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  rdma | grep ${currentDevice})
    local __currentFpgaVersion=$(mlx_fpga -d ${deviceRDMA} q |  grep version | cut -d':' -f2 | sed 's/[[:space:]]//g')
    if [ "$__newFpgaVersion" != "$__currentFpgaVersion" ]; then
        fHeader "Load image FPGA"
        local __deviceRDMA=$(mst status | grep "/dev/mst/" | cut -d" " -f1 | grep  rdma | grep ${currentDevice})
        mlx_fpga -d ${__deviceRDMA} load --user
    fi
}

#========================================================================================
# fVersion
# --------
#========================================================================================
fVersion(){
    local __versionType="0"
    if [ $INSTALL_TYPE == "BUNDLE" ]; then
        echo "Type: Bundle"
        __versionType="1"
    else
        echo "Type: OFED"
        __versionType="2"
    fi
    echo "Version: ${__versionType}.${VERSION_NUM}"
}

#========================================================================================
# fHelpBundle
# -----------
#========================================================================================
fHelpBundle(){
    echo "$0 [OPTIONS]"
    echo "-h|--help     : Print this message"
    echo "-v|--version  : Print script version"
    echo "-p <path>     : Path of bundle directory (default: script directory)"
    echo "-d <device>   : Device to update (default: the only existing device)"
    echo "-f <40g/10g>  : FPGA image type to burn (default: 40g)"
    echo "-s|--skip     : Skip kernel version check"
    echo "--force       : Install with no questions for user"
}

#========================================================================================
# fHelpOfed
# -----------
#========================================================================================
fHelpOfed(){
    echo "$0 [OPTIONS]"
    echo "-h|--help               : Print this message"
    echo "-v|--version            : Print script version"
    echo "-u <url>                : URL of the bundle tgz (if not provided, bundle tgz file location will be used)"
    echo "-t <bundle tgz file>    : Bundle tgz file (if not provided, path of bundle dir will be used)"
    echo "-p <path of bundle dir> : Path of bundle directory (default: script directory)"
    echo "-d <device>             : Device to update (default: the only existing device)"
    echo "-f <40g/10g>            : FPGA image type to burn (default: 40g)"
    echo "--force                 : Install with no questions for user"
}

#========================================================================================
# fParseUserInputOfed
# -------------------
#========================================================================================
fParseUserInputOfed(){
    local __counterPath=0
    local __foundF="N"
    #echo "# of args = $#"
    while [[ $# -gt 0 ]]; do
        local __key=$1
        case $__key in
        -h|--help)
            fHelpOfed
            exit
        ;;
        -v|--version)
            fVersion
            exit
        ;;
        --force)
            IS_FORCE_MODE="Y"
        ;;
        -d)
            shift
            if [ $# -gt 0 ]; then
                DEVICES+=($1)
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        -p)
            shift
            if [ $# -gt 0 ]; then
                __uDir=$1
                let __counterPath=__counterPath+1
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        -u)
            shift
            if [ $# -gt 0 ]; then
                __uUrl=$1
                let __counterPath=__counterPath+1
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        -t)
            shift
            if [ $# -gt 0 ]; then
                __uTgz=$1
                let __counterPath=__counterPath+1
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        -f)
            shift
            if [ $# -gt 0 ]; then
                if [ "$__foundF" == "N" ]; then
                    __foundF="Y"
                    __uFpgaOption=$1
                    if [ "$__uFpgaOption" != "10g" ]; then
                        if [ "$__uFpgaOption" != "40g" ]; then
                            echo -e "${RED}The fpga option is not valid - should be 10g or 40g${NC}"
                            fExit
                        fi
                    fi
                else
                    echo -e "${RED}You can chose FPGA option only once${NC}"
                    fExit
                fi
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        *)
            echo -e "${RED}Unknown option : $1${NC}"
            fExit
        ;;
        esac
        shift
    done

    if [ "$__counterPath" != "1" ]; then
        echo -e "${RED} You must choose exactly 1 of the following : -u , -t, -p (and you have chosen $__counterPath of them)${NC}"
        fExit
    fi
}
#========================================================================================
# fParseUserInputBundle
# ---------------------
#========================================================================================
fParseUserInputBundle(){
    #echo "# of args = $#"
    local __foundF="N"
    while [[ $# -gt 0 ]]; do
        local __key=$1
        case $__key in
        -h|--help)
            fHelpBundle
            exit
        ;;
        -v|--version)
            fVersion
            exit
        ;;
        --force)
            IS_FORCE_MODE="Y"
        ;;
        -s|--skip)
            SKIP_KERNEL_CHECK="Y"
        ;;
        -d)
            shift
            if [ $# -gt 0 ]; then
                DEVICES+=($1)
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        -p)
            shift
            if [ $# -gt 0 ]; then
                __uDir=$1
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        -f)
            shift
            if [ $# -gt 0 ]; then
                if [ "$__foundF" == "N" ]; then
                    __uFpgaOption=$1
                    __foundF="Y"
                    if [ "$__uFpgaOption" != "10g" ]; then
                        if [ "$__uFpgaOption" != "40g" ]; then
                            echo -e "${RED}The fpga option is not valid - should be 10g or 40g${NC}"
                            fExit
                        fi
                    fi
                else
                    echo -e "${RED}You can chose FPGA option only once${NC}"
                    fExit
                fi
            else
                echo -e "${RED}Unknown option : $1${NC}"
                fExit
            fi
        ;;
        *)
            echo -e "${RED}Unknown option : $1${NC}"
            fExit
        ;;
        esac
        shift
    done
}

#========================================================================================
# fCheckDir
# ---------
#========================================================================================
fCheckDir(){
    local __dirName=$1
    #Check if directory exists
    if [ ! -d "${__dirName}" ]; then
        # Control will enter here if $DIRECTORY doesn't exist.
        echo -e "${RED}Directory: ${__dirName}  doesn't exist${NC}"
        fExit
    fi

    #check if directory is empty
    if [ `ls -1A ${__dirName} | wc -l` -eq 0 ]; then
           echo -e "${RED}Directory: ${__dirName} is empty${NC}"
           fExit
    fi
}

#========================================================================================
# fSetVarsBundle
# --------------
#========================================================================================
fSetVarsBundle(){
    local __currPath=$(pwd)
    local __filePath="$(readlink -f "$0")"
    local __DIR=$(dirname "${__filePath}")
    local __uDevice="N"
    local __uDir="N"
    local __uTgzFile="N"
    local __uFpgaOption="40g"

    fParseUserInputBundle "$@"

    # Directory of bundle
    if [ $__uDir == "N" ]; then
        __uDir=${__DIR}/../
    fi

    fCheckDir ${__uDir}

    local __DIR_MFT=${__uDir}/MFT
    fCheckDir ${__DIR_MFT}

    local __DIR_FW=${__uDir}/FW
    fCheckDir ${__DIR_FW}

    #check FPGA dir
    local __DIR_FPGA=${__uDir}/Images
    fCheckDir ${__DIR_FPGA}

    if [ `ls -1A ${__DIR_FPGA}| grep ${__uFpgaOption} | wc -l` -eq 0 ]; then
           echo -e "${RED}Directory: ${__DIR_FPGA} doesn't contain FPGA 10g option${NC}"
           fExit
    fi

    #checl kernel dir
    local __DIR_KERNEL=${__uDir}/Kernel
    if [ $SKIP_KERNEL_CHECK == "N" ]; then
        fCheckDir ${__DIR_KERNEL}
    fi

    #init global variables
    PATH_MFT=${__DIR_MFT}/$(ls -t ${__DIR_MFT} | head -1)
    PATH_FW=${__DIR_FW}
    PATH_FPGA=${__DIR_FPGA}/$(ls -t ${__DIR_FPGA}| grep ${__uFpgaOption} | head -1)
    PATH_KERNEL=${__DIR_KERNEL}
}
#========================================================================================
# fSetVarsOfed
# ------------
#========================================================================================
fSetVarsOfed(){
    local __uDir="N"
    local __uUrl="N"
    local __uTgz="N"
    local __filePath="$(readlink -f "$0")"
    local __DIR=$(dirname "${__filePath}")
    local __uFpgaOption="40g"

    fParseUserInputOfed "$@"

#echo "__uDir=$__uDir, __uUrl=$__uUrl , __uTgz=$__uTgz"

    if [ "$__uDir" == "N" ]; then
        if [ "$__uUrl" != "N" ]; then
            fHeader "$__uUrl==> download"
            wget $__uUrl
            fCheckIsOperationOK "wget"
            __uTgz=$(basename $__uUrl)
        fi
        if [ "$__uTgz" != "N" ]; then
            fHeader "$__uTgz ==> extract"
            if [[ ${__uTgz: -4} == ".tgz" ]]; then
                __uDir=$(tar -xzvf $__uTgz | head -1)
                tar -xzvf $__uTgz
                fCheckIsOperationOK "tar"
            else
                echo -e "${RED}Not a *.tgz file${NC}"
                fExit
            fi
        fi
    fi
    if [ "$__uDir" == "N" ]; then
        __uDir=${__DIR}/../
    fi

    fCheckDir ${__uDir}

    #Check sub directories
    local __DIR_FW=${__uDir}/FW
    fCheckDir ${__DIR_FW}

    #check FPGA dir
    local __DIR_FPGA=${__uDir}/Images
    fCheckDir ${__DIR_FPGA}

    if [ `ls -1A ${__DIR_FPGA}| grep ${__uFpgaOption} | wc -l` -eq 0 ]; then
           echo -e "${RED}Directory: ${__DIR_FPGA} doesn't contain FPGA 10g option${NC}"
           fExit
    fi

    #init global variables
    PATH_FW=${__DIR_FW}
    PATH_FPGA=${__DIR_FPGA}/$(ls -t ${__DIR_FPGA}| grep ${__uFpgaOption} | head -1)
}
#========================================================================================
# Main
# ----
#========================================================================================
SKIP_KERNEL_CHECK="N"
IS_FORCE_MODE="N"
WAS_FPGA_IMAGE_UPDATED="N"

PATH_MFT="NONE"
PATH_FW="NONE"
PATH_FPGA="NONE"
PATH_KERNEL="NONE"
KERNEL_VERSION=$(uname -r | cut -d"-" -f1)
KERNEL_LAST_COMMIT=$(uname -r | cut -d"-" -f2 | cut -d"_" -f1)
DEVICES=()

if [ $INSTALL_TYPE == "BUNDLE" ]; then
    fSetVarsBundle "$@"
else
    fSetVarsOfed "$@"
fi

echo "PATH_MFT=$PATH_MFT"
echo "PATH_FW=$PATH_FW"
echo "PATH_FPGA=$PATH_FPGA"
echo "PATH_KERNEL=$PATH_KERNEL"
echo "KERNEL_VERSION=$KERNEL_VERSION"
echo "KERNEL_LAST_COMMIT=$KERNEL_LAST_COMMIT"

if [ $INSTALL_TYPE == "BUNDLE" -a $SKIP_KERNEL_CHECK == "N" ]; then
    fCheckKernel $PATH_KERNEL $KERNEL_VERSION $KERNEL_LAST_COMMIT
fi

fCheckMFT $PATH_MFT
fHeader "Check Devices"
# In case that when doing mst status rdma devices are not visible => restart mst with fpga
fCheckRdmaDevices

# Device - get devices to burn
if [ "${#DEVICES[@]}" == "0" ]; then
    fChooseDevices
fi

#check that the devices are valid - with RDMA device
for device in ${DEVICES[@]}; do
    fCheckDevice $device
done

#install devices
for device in ${DEVICES[@]}; do
    echo "Install Device : Device=$device"
    fInstallDevice $device
done

