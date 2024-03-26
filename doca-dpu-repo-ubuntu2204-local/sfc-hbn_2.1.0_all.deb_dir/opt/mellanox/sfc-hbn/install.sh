#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
#
# This Software is licensed under one of the following licenses:
#
# 1) under the terms of the "Common Public License 1.0" a copy of which is
#    available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/cpl.php.
#
# 2) under the terms of the "The BSD License" a copy of which is
#    available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/bsd-license.php.
#
# 3) under the terms of the "GNU General Public License (GPL) Version 2" a
#    copy of which is available from the Open Source Initiative, see
#    http://www.opensource.org/licenses/gpl-license.php.
#
# Licensee has the right to choose one of the above licenses.
#
# Redistributions of source code must retain the above copyright
# notice and one of the license notices.
#
# Redistributions in binary form must reproduce both the above copyright
# notice, one of the license notices in the documentation
# and/or other materials provided with the distribution.
###############################################################################

set -Eeu
trap cleanup SIGHUP SIGINT SIGTERM ERR
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
LOG_FILE="/var/log/sfc-install.log"

cleanup() {
  die "Installation failed. Cleaning up"
}


## Global variables
BR_NAME=${BR_NAME:-br-hbn}
MLNX_SF_FILE="/etc/mellanox/mlnx-sf.conf"
MLNX_OVS_FILE="/etc/mellanox/mlnx-ovs.conf"
MLNX_BF_FILE="/etc/mellanox/mlnx-bf.conf"
HBN_CONF_FILE="/etc/mellanox/hbn.conf"
SFC_CONF_FILE="/etc/mellanox/sfc.conf"
UDEV_SCRIPT_PATH="/usr/lib/udev"
SF_RENAME_SCRIPT="auxdev-sf-netdev-rename"
SF_REP_RENAME_SCRIPT="sf-rep-netdev-rename"
OVS_MAPPING_TEMPLATE_FILE="sfc.conf.tmpl"
IFUP_HOOK_50="50-ifup-hooks"
IFUP_HOOKS_PATH="/etc/networkd-dispatcher/routable.d"
CNI_CONF="10-containerd-net-br-mgmt.conflist"
CNI_CONF_PATH="/etc/cni/net.d/"
SYSCTL_CONF="50-sfc-hbn.conf"
SYSCTL_PATH="/lib/sysctl.d"
REBOOT_OPTION="false"
USER_PCI_ADDR=""
ECPF0=""
ECPF1=""
MIN_VF=0
MAX_VF=127
CPU_AFFINITY_LIST="0-2"
CONFIG_OPTIONS=""
DEBUG_OPTION=""
CLOUD_OPTION=${CLOUD_OPTION:-""}

# Host/Network facing SFs base IDs
BASE_NET_FACING_SF_NUM=2   # 0x1 - 0xE38 (2 - 1000)
BASE_HOST_FACING_SF_NUM=1001   # 0x3E9 - 0x7D0 (1001 - 2000)
BASE_DPU_FACING_SF_NUM=500
# Host facing SFs distributed in the following way:
# 0x3E9..0x438 - SFs mapped to ECPF0 VFs (pf0vf*_sf)  sfnum range: 1001 - 1256
ECPF0_VF_SF_BASE_ID=$BASE_HOST_FACING_SF_NUM  # sfnum: 1001
# 0x439..0x5E8 - SFs mapped to ECPF1 VFs (pf1vf*_sf)  sfnum range: 1257 - 1512
ECPF1_VF_SF_BASE_ID=$((BASE_HOST_FACING_SF_NUM + 0x100)) # sfnum: 1257
#DPU SFs - starting from 500
# 0x5EA,0x5EB - SFs mapped to Host PF representors (pf0hpf_sf, pf1hpf_sf)
PF0HPF_SF_ID=$((BASE_HOST_FACING_SF_NUM + 0x201)) # sfnum: 1514
PF1HPF_SF_ID=$((BASE_HOST_FACING_SF_NUM + 0x202)) # sfnum: 1515

# NVCONFIG variables corresponding to the default number of 16 SFs
PF_BAR2_ENABLE=${PF_BAR2_ENABLE:-0}
PER_PF_NUM_SF=${PER_PF_NUM_SF:-1}
PF_TOTAL_SF=${PF_TOTAL_SF:-40}
PF_SF_BAR_SIZE=${PF_SF_BAR_SIZE:-10}
NUM_PF_MSIX_VALID=${NUM_PF_MSIX_VALID:-0}
PF_NUM_PF_MSIX_VALID=${PF_NUM_PF_MSIX_VALID:-1}
PF_NUM_PF_MSIX=${PF_NUM_PF_MSIX:-228}

distro=`lsb_release -i -s``lsb_release -r -s | tr -d '.'`
distro=${distro,,}

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

usage() {
  cat << EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-p0 | --ecpf0 <num>]  [-p1 | --ecpf1 <num>] [-h] [-v] [-r] [-d] [-hs | --hugepagesz] [-hc | --hugepages]
Generate SF configs according to the number of VFs created on both physical ports
and install OVS SFC configuration for HBN.

Available options:

-p0, --ecpf0         [Optional] Number of VFs created on ECPF0 in range $MIN_VF..$MAX_VF (default: 0)
-p1, --ecpf1         [Optional] Number of VFs created on ECPF1 in range $MIN_VF..$MAX_VF (default: 0)
-r,  --reboot        [Optional] Reboot after install is completed
-d,  --debug         [Optional] Set vswitchd common options in DBG mode
-hs,  --hugepagesz   [Optional] Set hugepagesz size assumed to be in kB (default: 2048)
-hc,  --hugepages    [Optional] Set hugepages number (default: 3072)
-h,  --help          [Optional] Print this help and exit
-v,  --verbose       [Optional] Print script debug info
-c,  --config-file   [Optional] Use config ini file

EOF
  exit
}

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
msg() {
  echo 2>&1 -e "[$TIMESTAMP]   ${1-}" | tee -ia ${LOG_FILE}
}

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "Error: $msg"
  exit "$code"
}

is_number(){
  local string=$1
  case $string in
    ''|*[!0-9]*) die "$string : Not a valid number" ;;
    *) return ;;
  esac
}

validate_address(){
  pci_address_regex="^[a-fA-F0-9]{4}:[a-fA-F0-9]{2}\:[a-fA-F0-9]{2}\.[a-fA-F0-9]{1}$"
  [[ "$USER_PCI_ADDR" =~  $pci_address_regex ]] && msg "matched" || die "Bad pci address format"
}

validate_vf_number(){
  local ecpf=$1
  local num=$2
  is_number $num
  if (($num < $MIN_VF || $num > $MAX_VF)); then
    die "Invalid number of $ecpf VFs: $num. Please enter number in range $MIN_VF..$MAX_VF"
  fi
}

parse_params() {
  # Local variables, if any

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -v | --verbose) set -x ;;
    -p0 | --ecpf0)
      ECPF0="${2-}"
      validate_vf_number ECPF0 $ECPF0
      shift
      ;;
    -p1 | --ecpf1)
      ECPF1="${2-}"
      validate_vf_number ECPF1 $ECPF1
      shift
      ;;
    -r | --reboot)  REBOOT_OPTION="true" ;;
    -d | --debug)  DEBUG_OPTION="true" ;;
    -c | --config-file)  CONFIG_OPTIONS="true";;
    -hs | --hugepagesz)
      HUGEPAGESZ="${2-}"
      shift
      ;;
    -hc | --hugepages)
      HUGEPAGES="${2-}"
      shift
      ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  return 0
}

parse_params "$@"

function sf_create() {
  sfaddcmd="/sbin/mlnx-sf --action create $@ -t --cpu-list $CPU_AFFINITY_LIST"
  msg "Adding sf conf: ${sfaddcmd}"
  echo ${sfaddcmd}  >> ${MLNX_SF_FILE}
}

add_sf_p0(){
  local sf_num=0
  local pci_addr_base="0000:03:00."
  local mac_p0=$(cat /sys/class/net/p0/address)
  if [ -z "${mac_p0}" ]; then die "P0 does not have a mac address.";  fi
  base_mac_ecpf0=$(printf '00:04:4b:%02x:%02x\n' $[RANDOM%256] $[RANDOM%256])
  sf_create --device ${pci_addr_base}0 --sfnum $BASE_NET_FACING_SF_NUM --hwaddr $mac_p0
  sf_create --device ${pci_addr_base}0 --sfnum $PF0HPF_SF_ID --hwaddr $base_mac_ecpf0:f0
  # Host facing SFs mapped to ECPF0 VFs
  if ! [ -z "${ECPF0}" ]; then
    for (( i=0; i < $ECPF0; i++))
    do
      mac_idx=$(printf '%02x' $i)
      sf_num=$((ECPF0_VF_SF_BASE_ID + i))
      sf_create --device ${pci_addr_base}0 --sfnum ${sf_num} --hwaddr $base_mac_ecpf0:$mac_idx
    done
  else
    for (( i=0; i < ${#ARR_REPS_P0[@]}; i++))
    do
      if [ "${ARR_REPS_P0[${i}]}" = "pf0hpf" ]; then
        continue
      fi
      mac_idx=$(printf '%02x' $i)
      suffix=`echo "${ARR_REPS_P0[${i}]}" | cut -d"f" -f 3`
      sf_num=$((ECPF0_VF_SF_BASE_ID + suffix))
      sf_create --device ${pci_addr_base}0 --sfnum ${sf_num} --hwaddr $base_mac_ecpf0:$mac_idx
    done
  fi
}

add_sf_p1(){
  local sf_num=0
  local pci_addr_base="0000:03:00."
  local mac_p1=$(cat /sys/class/net/p1/address)
  if [ -z "${mac_p1}" ]; then die "P1 does not have a mac address.";  fi
  base_mac_ecpf1=$(printf '00:04:4b:%02x:%02x\n' $[RANDOM%256] $[RANDOM%256])
  sf_create --device ${pci_addr_base}0 --sfnum $((BASE_NET_FACING_SF_NUM + 1)) --hwaddr $mac_p1
  sf_create --device ${pci_addr_base}0 --sfnum $PF1HPF_SF_ID --hwaddr $base_mac_ecpf0:f1
  # Host facing SFs mapped to ECPF1 VFs
  if ! [ -z "${ECPF1}" ]; then
    for (( i=0; i < $ECPF1; i++))
    do
      mac_idx=$(printf '%02x' $i)
      sf_num=$((ECPF1_VF_SF_BASE_ID + i))
      sf_create --device ${pci_addr_base}0 --sfnum ${sf_num} --hwaddr $base_mac_ecpf1:$mac_idx
    done
  else
    for (( i=0; i < ${#ARR_REPS_P1[@]}; i++))
    do
      if [ "${ARR_REPS_P1[${i}]}" = "pf1hpf" ]; then
        continue
      fi
      mac_idx=$(printf '%02x' $i)
      suffix=`echo "${ARR_REPS_P1[${i}]}" | cut -d"f" -f 3`
      sf_num=$((ECPF1_VF_SF_BASE_ID + suffix))
      sf_create --device ${pci_addr_base}0 --sfnum ${sf_num} --hwaddr $base_mac_ecpf1:$mac_idx
    done
  fi
}

add_sfs(){
  msg "Clearing existing config in ${MLNX_SF_FILE}";
  > ${MLNX_SF_FILE}
  if [ "${COUNT_UPLINKS}" = "2" ] || [ ${ARR_UPLINKS[0]} = "p0" ]
  then
    add_sf_p0
  fi
  if [ "${COUNT_UPLINKS}" = "2" ] || [ ${ARR_UPLINKS[0]} = "p1" ]
  then
    add_sf_p1
  fi
  local pci_addr_base="0000:03:00."
  base_mac_dpu=$(printf '00:04:4b:%02x:%02x\n' $[RANDOM%256] $[RANDOM%256])
  for (( i=0; i < ${#ARR_DPU_SFS[@]}; i++))
  do
    mac_idx=$(printf '%02x' $i)
    suffix=`echo "${ARR_DPU_SFS[${i}]}" | cut -d"u" -f 2`
    sf_num=$((BASE_DPU_FACING_SF_NUM + suffix))
    sf_create --device ${pci_addr_base}0 --sfnum ${sf_num} --hwaddr $base_mac_dpu:$mac_idx
  done
  msg "==> SF configs saved in $MLNX_SF_FILE"
}

remove_auto_ovsbridge_creation(){
    if [ ! -f $MLNX_OVS_FILE ]; then
      die "File: $MLNX_OVS_FILE is missing. Cannot proceed. Aborting!"
    fi
    sed -i 's/CREATE_OVS_BRIDGES=\"yes\"/CREATE_OVS_BRIDGES=\"no\"/g' ${MLNX_OVS_FILE}
    # Try to delete the default bridge and ignore errors
    /usr/bin/ovs-vsctl --timeout=30 --no-wait --if-exists del-br ${BR_NAME} || true
}

update_mnlx_bf_conf(){
    if [ ! -f $MLNX_BF_FILE ]; then
      die "File: $MLNX_BF_FILE is missing. Cannot proceed. Aborting!"
    fi
    # Set ALLOW_SHARED_RQ  to "no"
    sed -i 's/ALLOW_SHARED_RQ=\"yes\"/ALLOW_SHARED_RQ=\"no\"/g' ${MLNX_BF_FILE}
    # Add ESWITCH_MULTIPORT to "yes" if COUNT_UPLINKS==2
    if [ "${COUNT_UPLINKS}" = "2" ]; then
      grep -q "ENABLE_ESWITCH_MULTIPORT=" $MLNX_BF_FILE && sed "s/^ENABLE_ESWITCH_MULTIPORT=.*/ENABLE_ESWITCH_MULTIPORT=\"yes\"/" -i $MLNX_BF_FILE ||
      sed "$ a\ENABLE_ESWITCH_MULTIPORT=\"yes\"" -i $MLNX_BF_FILE
    else
      grep -q "ENABLE_ESWITCH_MULTIPORT=" $MLNX_BF_FILE && sed "s/^ENABLE_ESWITCH_MULTIPORT=.*/ENABLE_ESWITCH_MULTIPORT=\"no\"/" -i $MLNX_BF_FILE ||
      sed "$ a\ENABLE_ESWITCH_MULTIPORT=\"no\"" -i $MLNX_BF_FILE
    fi
}

confirm_reboot(){
    read -p "==> Reboot option specified. Please confirm (y/n)? " answer
case ${answer:0:1} in
    y|Y )
        msg " Reboot confirmed. Proceeding..."
        reboot
        exit
    ;;
    * )
        msg "Skipping automatic reboot. Please reboot as early as possible for settings to take effect."
    ;;
esac
}

set_vswitchd_debug_options(){
    msg " Setting debug options for vswitchd..."
    sed -i 's/set ovs_ctl ${1-start} --system-id=random/set ovs_ctl ${1-start} --ovs-vswitchd-options=\"-vdpif_netdev:dbg -vnetdev_offload_dpdk:dbg -vdpdk_offload_doca:dbg -vdpdk:dbg\" --system-id=random/g' /etc/init.d/openvswitch-switch
}

remove_vswitchd_debug_options(){
    msg " Removing debug options for vswitchd..."
    sed -i 's/set ovs_ctl ${1-start} --ovs-vswitchd-options=\"-vdpif_netdev:dbg -vnetdev_offload_dpdk:dbg -vdpdk_offload_doca:dbg -vdpdk:dbg\" --system-id=random/set ovs_ctl ${1-start} --system-id=random/g' /etc/init.d/openvswitch-switch
}

generate_ovs_sf_mapping(){
    tmpfile=$(mktemp)
    cat $SCRIPT_DIR/$OVS_MAPPING_TEMPLATE_FILE > ${tmpfile}

    # the sfc.sh script will look into the generated mapping file
    # and add ports based on order.
    # in case of OVS DPDK/DOCA physical uplinks should be first
    # otherwise it will cause slowness on the OVS side since it
    # has to retry all the invalid ports.
    if [ "${COUNT_UPLINKS}" = "2" ] || [ ${ARR_UPLINKS[0]} = "p0" ]
    then
        echo "\"p0~p0_sf_r~p0_sf~p0_sf_r\"" >> $tmpfile
    fi
    if [ "${COUNT_UPLINKS}" = "2" ] || [ ${ARR_UPLINKS[0]} = "p1" ]
    then
        echo "\"p1~p1_sf_r~p1_sf~p1_sf_r\"" >> $tmpfile
    fi

    # Generating mappings for SFs to ECPF0 VFs
    if ! [ -z "${ECPF0}" ]; then
      echo "\"pf0hpf~pf0hpf_sf_r~pf0hpf_sf~pf0hpf_sf_r\"" >> $tmpfile
      for (( i=0; i < $ECPF0; i++ ))
      do
          echo "\"pf0vf${i}~pf0vf${i}_sf_r~pf0vf${i}_sf~pf0vf${i}_sf_r\"" >> $tmpfile
      done
    else
      for (( i=0; i < ${#ARR_REPS_P0[@]}; i++))
      do
          val=${ARR_REPS_P0[${i}]}
          if [ "${val}" = "pf0hpf" ]; then
            echo "\"pf0hpf~pf0hpf_sf_r~pf0hpf_sf~pf0hpf_sf_r\"" >> $tmpfile
          else
            echo "\"${val}~${val}_sf_r~${val}_sf~${val}_sf_r\"" >> $tmpfile
          fi
      done
    fi

    # Generating mappings for SFs to ECPF1 VFs
    if ! [ -z "${ECPF1}" ]; then
      echo "\"pf1hpf~pf1hpf_sf_r~pf1hpf_sf~pf1hpf_sf_r\"" >> $tmpfile
      for (( i=0; i < $ECPF1; i++ ))
      do
          echo "\"pf1vf${i}~pf1vf${i}_sf_r~pf1vf${i}_sf~pf1vf${i}_sf_r\"" >> $tmpfile
      done
    else
      for (( i=0; i < ${#ARR_REPS_P1[@]}; i++))
      do
          val=${ARR_REPS_P1[${i}]}
          if [ "${val}" = "pf1hpf" ]
          then
            echo "\"pf1hpf~pf1hpf_sf_r~pf1hpf_sf~pf1hpf_sf_r\"" >> $tmpfile
          else
            echo "\"${val}~${val}_sf_r~${val}_sf~${val}_sf_r\"" >> $tmpfile
          fi
      done
    fi

    for (( i=0; i < ${#ARR_DPU_SFS[@]}; i+=2))
    do
        val=${ARR_DPU_SFS[${i}]}
        temp=$((i + 1))
        val1=${ARR_DPU_SFS[${temp}]}
        echo '"'${val}'_sf_r~'${val1}'_sf_r~'${val1}'_sf~'${val1}'_sf_r~\"\"~\"\""' >> $tmpfile
    done

    # Add closing paranthesis.
    echo ")" >> $tmpfile

    msg "==> Generated Mapping file"
    cat $tmpfile

    cp $tmpfile $SFC_CONF_FILE
    rm -rf $tmpfile

}

generate_hbn_configuration(){
  msg "==> Generating hbn configuration..."
  ARR_DPU_SFS_ORIG="${ARR_DPU_SFS_ORIG[@]}" ARR_UPLINKS="${ARR_UPLINKS[@]}" ARR_REPS_P0="${ARR_REPS_P0[@]}" ARR_REPS_P1="${ARR_REPS_P1[@]}" python3 $SCRIPT_DIR/generate_hbn_conf.py
}

install_sfc_service(){
    msg "==> Installing sfc systemd service..."
    cp $SCRIPT_DIR/sfc.service /etc/systemd/system/sfc.service
    msg "==> Finished installing sfc systemd service."
    msg "==> Enabling sfc systemd service"
    systemctl daemon-reload
    systemctl enable sfc
    systemctl enable sfc-state-propagation.service
}

use_hbn_config_file(){
  msg "==> Fetching input from configuration file"
  python3 $SCRIPT_DIR/use_hbn_conf.py
  HBN_UPLINKS=`cat /tmp/.HBN_UPLINKS`
  HBN_REPS=`cat /tmp/.HBN_REPS`
  HBN_DPU_SFS=`cat /tmp/.HBN_DPU_SFS`
}

copy_files(){
    msg "==> Start copying udev SF rename scripts"
    # Backup original files first
    cp $UDEV_SCRIPT_PATH/$SF_RENAME_SCRIPT{,.original}
    cp $SCRIPT_DIR/$SF_RENAME_SCRIPT $UDEV_SCRIPT_PATH/
    cp $UDEV_SCRIPT_PATH/$SF_REP_RENAME_SCRIPT{,.original}
    cp $SCRIPT_DIR/$SF_REP_RENAME_SCRIPT $UDEV_SCRIPT_PATH/
    msg "==> Finished copying udev SF rename scripts"

    msg "==> Start copying sysctl conf"
    mkdir -p $SYSCTL_PATH
    cp $SCRIPT_DIR/$SYSCTL_CONF $SYSCTL_PATH/
    msg "==> Finished copying sysctl conf file"

    msg "==> Start copying ovs sfc files"
    # We do not need to copy sfc.conf since it is done by the generate_ovs_sf_mapping function.
    mkdir -p /opt/mellanox/sfc
    cp $SCRIPT_DIR/sfc.sh  /opt/mellanox/sfc/
    chmod +x /opt/mellanox/sfc/sfc.sh
    msg "==> Finished copying ovs sfc files"

    cp $SCRIPT_DIR/ovs-vswitchd-watchdog.py  /opt/mellanox/sfc/
    msg "==> Finished copying ovs-vswitchd-watchdog.py"
}

install_sfc_hbn_software() {
	msg "==> Install SFC HBN packages"

  # The below change was added to force packages to install from local repo
  # instead of online repo.
  # See https://gitlab-master.nvidia.com/hareeshp/sfc-hbn/-/merge_requests/16

  # apt install -y hbn-runtime
  if [ "X$distro" = "Xubuntu2004" ]; then
    apt install -y --allow-downgrades /var/hbn-repo-aarch64-ubuntu2004-local/*.deb
  fi

  if [ "X$distro" = "Xubuntu2204" ]; then
    apt install -y --allow-downgrades /var/hbn-repo-aarch64-ubuntu2204-local/*.deb
  fi

	msg "==> Finished install SFC HBN packages"
}

configure_mgmt_vrf() {
    msg "==> Start copying ifup hooks"
    # IFUP_HOOK_50 used to create mgmt VRF
    # mgmt VRF is supported by NetworkManager in Ubuntu22.04
    mkdir -p $IFUP_HOOKS_PATH
    cp $SCRIPT_DIR/$IFUP_HOOK_50 $IFUP_HOOKS_PATH/
    chmod +x $IFUP_HOOKS_PATH/$IFUP_HOOK_50
    msg "==> Finished copying ifup hooks files"
    msg "==> Start mgmt VRF configuration"
    # Remove cloud-init configuration to prevent netplan files creation
    rm -f /var/lib/cloud/seed/nocloud-net/network-config
    if [ -e /etc/netplan/50-cloud-init.yaml ]; then
      rm -f /etc/netplan/50-cloud-init.yaml
      netplan apply > /dev/null 2>&1 || true
    fi
    # Change oob_net0 configuration using networkd
    cat >> /etc/systemd/network/25-oob-net0.network << EOF
[Match]
Name=oob_net0

[Network]
VRF=mgmt
DHCP=ipv4

[DHCP]
UseDomains=true
EOF
    # Set tmfifo_net0 configuration using networkd
    cat >> /etc/systemd/network/25-tmfifo-net0.network << EOF
[Match]
Name=tmfifo_net0

[Network]
VRF=mgmt
Address=192.168.100.2/30

[Route]
Gateway=192.168.100.1
Metric=1025

[DHCP]
UseDomains=true
EOF
    # Create management VRF
    cat >> /etc/systemd/network/25-mgmt.network << EOF
[Match]
Name=mgmt

[Network]
Address=127.0.0.1/8
Address=::1/128
EOF

  cat >> /etc/systemd/network/25-mgmt.netdev << EOF
[NetDev]
Name=mgmt
Kind=vrf

[VRF]
TableId=1001
EOF
    # Disable containerd and kubelet services
    systemctl disable kubelet
    systemctl disable containerd
    systemctl disable docker
    systemctl disable ssh
    systemctl disable ntp
    # Disable NetworkManager services
    systemctl disable NetworkManager.service
    systemctl disable NetworkManager-wait-online.service
    # Enable containerd and kubelet services for management VRF
    # The actual service files created by /lib/systemd/system-generators/systemd-vrf-generator
    mkdir -p /etc/systemd/system/vrf@mgmt.target.wants
    ln -snf /etc/systemd/system/containerd@.service /etc/systemd/system/vrf@mgmt.target.wants/containerd@mgmt.service
    ln -snf /etc/systemd/system/kubelet@.service /etc/systemd/system/vrf@mgmt.target.wants/kubelet@mgmt.service
    ln -snf /etc/systemd/system/docker@.service /etc/systemd/system/vrf@mgmt.target.wants/docker@mgmt.service
    ln -snf /etc/systemd/system/ntp@.service /etc/systemd/system/vrf@mgmt.target.wants/ntp@mgmt.service
    sed -i -e 's@ -H fd://@@' -e 's/Wants=containerd.service/Wants=containerd@mgmt.service/' /lib/systemd/system/docker.service
    # Generate services files for management VRF
    /lib/systemd/system-generators/systemd-vrf-generator /run/systemd/generator /run/systemd/generator.early /run/systemd/generator.late
    # Start containerd, docker and kubelet services onthe first boot after the deployment
    cat > $SCRIPT_DIR/services_helper << EOF
#!/bin/bash

systemctl daemon-reload
systemctl enable ssh@mgmt.service
systemctl start ssh@mgmt.service
systemctl start ntp@mgmt.service
systemctl start containerd@mgmt.service
systemctl start kubelet@mgmt.service
systemctl start docker@mgmt.service
EOF
    chmod +x $SCRIPT_DIR/services_helper
    sed -i -e "/runcmd:/a\ \ - [ $SCRIPT_DIR/services_helper ]" /var/lib/cloud/seed/nocloud-net/user-data
    sed -i "s|ExecStart=|ExecStartPre=$SCRIPT_DIR/services_helper\nExecStart=|g" /lib/systemd/system/cloud-config.service
    # Prevent from import_doca_telemetry.sh enabling kubelet and containerd services
    perl -ni -e 'print unless /systemctl/' /opt/mellanox/doca/services/telemetry/import_doca_telemetry.sh
    # Use systemd-resolved to keep /etc/resolv.conf up-to-date
    rm -f /etc/resolv.conf
    ln -snf /run/systemd/resolve/resolv.conf /etc/resolv.conf
    # Create .bash_aliases to include VRF in the shell prompt
    cat >> /root/.bash_aliases << 'EOF'
NS=$(ip netns identify)
[ -n "$NS" ] && NS=":${NS}"

VRF=$(ip vrf identify)
[ -n "$VRF" ] && VRF=":${VRF}"

PS1='${debian_chroot:+($debian_chroot)}\u@\h${NS}${VRF}:\w\$ '
EOF
    msg "==> Finished mgmt VRF configuration"
}

disable_netplan() {
  msg "==> Disable netplan configuration by cloud-init"
  echo "network: {config: disabled}" > /etc/cloud/cloud.cfg.d/99-custom-networking.cfg
  /bin/rm -f /etc/netplan/*
}

grub_cmd_update() {
  if ! grep -wq "cgroup_no_v1=net_prio,net_cls," /etc/default/grub; then
    sed -i -r -e 's/^(GRUB_CMDLINE_LINUX=.*)\"/\1 cgroup_no_v1=net_prio,net_cls,"/' /etc/default/grub
  fi

  if grep -Pwq "hugepagesz=.*[0-9]kB" /etc/default/grub; then
    sed -i -r -e 's/hugepagesz=.*[0-9]kB/hugepagesz='"${HUGEPAGESZ}"'kB/g' /etc/default/grub
  else
    sed -i -r -e 's/^(GRUB_CMDLINE_LINUX=.*)\"/\1 hugepagesz='"${HUGEPAGESZ}"'kB"/' /etc/default/grub
  fi

  if grep -Pwq "hugepages=.*[0-9]" /etc/default/grub; then
    sed -i -r -e 's/hugepages=.*[0-9]/hugepages='"${HUGEPAGES}"'/g' /etc/default/grub
  else
    sed -i -r -e 's/^(GRUB_CMDLINE_LINUX=.*)\"/\1 hugepages='"${HUGEPAGES}"'"/' /etc/default/grub
  fi

  update-grub > /dev/null 2>&1
}

cx_nvconfig(){
    msg "==> Start CX device configuration"
    RC=0

    for dev in `lspci -nD -d 15b3: | grep 'a2d[26c]' | cut -d ' ' -f 1`
    do
        msg "==> Running mlxconfig for $dev"
        output=`mlxconfig -y -d $dev set \
            PF_BAR2_ENABLE=$PF_BAR2_ENABLE \
            PER_PF_NUM_SF=$PER_PF_NUM_SF \
            PF_TOTAL_SF=$PF_TOTAL_SF \
            PF_SF_BAR_SIZE=$PF_SF_BAR_SIZE \
            NUM_PF_MSIX_VALID=$NUM_PF_MSIX_VALID \
            PF_NUM_PF_MSIX_VALID=$PF_NUM_PF_MSIX_VALID \
            PF_NUM_PF_MSIX=$PF_NUM_PF_MSIX \
            LAG_RESOURCE_ALLOCATION=1; \
            echo Status: $?`
        rc=${output:0-1}
        RC=$((RC+rc))
        msg "==> MLXCONFIG:"
        msg "$output"
        if [ $rc -ne 0 ]; then
            msg "==> Failed to configure CX device: $dev"
        fi
    done

    msg "==> Finished CX device configuration. Status: $RC"
    msg "==> mlxfwreset or power cycle is required"
    return $RC
}

add_cni_config(){

  # Copying CNI config
  msg "==> Copying CNI configuration"
  cp $SCRIPT_DIR/cni/$CNI_CONF  $CNI_CONF_PATH

}

validate_uplinks(){
  HBN_UPLINKS=""
  for (( i=0; i< 2; i++ )); do
    if ! $(ls /sys/class/net | grep -wq "p${i}"); then
      msg "Interface p${i} not found"
      continue
    fi
    HBN_UPLINKS=$HBN_UPLINKS"p"${i}
    if [ "${i}" = "0" ]; then
      ARR_REPS_P0=("pf0hpf")
      HBN_UPLINKS=$HBN_UPLINKS","
    else
      ARR_REPS_P1=("pf1hpf")
    fi
  done
  # overwrite default
  ARR_UPLINKS=(${HBN_UPLINKS//,/ })
  COUNT_UPLINKS=${#ARR_UPLINKS[@]}
}

detect_uplinks(){
  # Set global configuration variables after validating the uplinks
  HBN_UPLINKS=${HBN_UPLINKS:-"p0,p1"}
  ARR_UPLINKS=(${HBN_UPLINKS//,/ })
  COUNT_UPLINKS=${#ARR_UPLINKS[@]}
  HBN_REPS=${HBN_REPS:-"pf0hpf,pf1hpf,pf0vf0-pf0vf13"}
  ARR_REPS_P0=()
  ARR_REPS_P1=()
  HBN_DPU_SFS=${HBN_DPU_SFS:-"pf0dpu1,pf0dpu3"}
  ARR_DPU_SFS=()
  ARR_DPU_SFS_ORIG=()
  HUGEPAGESZ=${HUGEPAGE_SIZE:-2048}
  HUGEPAGES=${HUGEPAGE_COUNT:-3072}
  validate_uplinks
}

validate_hbn_configuration(){
  python3 $SCRIPT_DIR/validate_hbn_conf.py
}

override_hbn_reps(){
  # Generating mappings for SFs to ECPF1 VFs
  if ! [ -z "${ECPF0}" ] || ! [ -z "${ECPF1}" ]; then
    HBN_REPS=""
  fi
  if ! [ -z "${ECPF0}" ] && [ ${ECPF0} -gt 0 ]; then
    vf_p0_num=$(($ECPF0-1))
    HBN_REPS="${HBN_REPS}pf0vf0-pf0vf${vf_p0_num},"
  fi
  if ! [ -z "${ECPF1}" ] && [ ${ECPF1} -gt 0 ]; then
    vf_p1_num=$(($ECPF1-1))
    HBN_REPS="${HBN_REPS}pf1vf0-pf1vf${vf_p1_num},"
  fi
}

expand_sf_vf_variables(){
  # Expand and group VFs
  override_hbn_reps
  local arr_reps_temp=(${HBN_REPS//,/ })
  local count_arr_reps_temp=${#arr_reps_temp[@]}
  for (( i=0; i< ${count_arr_reps_temp}; i++ )); do
    local val=${arr_reps_temp[${i}]}
    local chk_prefix=`echo "${val}" | cut -d"v" -f 1`
    if  [[ ${arr_reps_temp[${i}]} =~ "-" ]]; then
      local prefix=`echo "${val}" | cut -d"-" -f 1 | cut -d"f" -f 1`
      prefix+="f"
      prefix+=`echo "${val}" | cut -d"-" -f 1 | cut -d"f" -f 2`
      prefix+="f"
      min=`echo "$val" | cut -d"-" -f 1 | cut -d"f" -f 3`
      max=`echo "$val" | cut -d"-" -f 2 | cut -d"f" -f 3`
      for (( j=${min}; j<=${max}; j++)); do
        if [ "${chk_prefix}" = "pf0" ]; then
          ARR_REPS_P0+=(${prefix}${j})
        elif [ "${chk_prefix}" = "pf1" ] && [ "${COUNT_UPLINKS}" = "2" ]; then
          ARR_REPS_P1+=(${prefix}${j})
        fi
      done
    else
      if [ "${chk_prefix}" = "pf0" ]; then
        ARR_REPS_P0+=(${val})
      elif [ "${chk_prefix}" = "pf1" ] && [ "${COUNT_UPLINKS}" = "2" ]; then
        ARR_REPS_P1+=(${val})
      fi
    fi
  done
  # Expand and group SFs
  local arr_dpu_sfs_temp=(${HBN_DPU_SFS//,/ })
  for (( i=0; i< ${#arr_dpu_sfs_temp[@]}; i++ )); do
    local val=${arr_dpu_sfs_temp[${i}]}
    local suffix=`echo "${val}" | cut -d"u" -f 2`
    local prefix=`echo "${val}" | cut -d"u" -f 1`
    prefix+="u"
    ARR_DPU_SFS+=(${prefix}$((suffix - 1)))
    ARR_DPU_SFS+=(${val})
    ARR_DPU_SFS_ORIG+=(${val})
  done
}

###########################################
###             Main                    ###
###########################################
ARGS="$@"
msg ""
msg "#####################################################################################"
msg "==> Starting OVS SFC Installation"
msg "==> Script called as:$0 $ARGS"

install_sfc_hbn_software

if [ "${CONFIG_OPTIONS}" = "true" ]; then
  use_hbn_config_file
fi

detect_uplinks
validate_hbn_configuration

expand_sf_vf_variables
remove_auto_ovsbridge_creation
update_mnlx_bf_conf

add_sfs
generate_ovs_sf_mapping
generate_hbn_configuration

copy_files

add_cni_config
install_sfc_service
if [[ -z "$CLOUD_OPTION" ]]; then
  if [[ "X$distro" =~ Xubuntu2[02]04 ]]; then
    # Configure management VRF for Ubuntu 20.04 and 22.04
    configure_mgmt_vrf
    disable_netplan
  fi
fi
grub_cmd_update
cx_nvconfig

if [ "${DEBUG_OPTION}" = "true" ]; then
  set_vswitchd_debug_options
else
  remove_vswitchd_debug_options
fi

if [ "${REBOOT_OPTION}" = "true" ]; then
  confirm_reboot
fi
msg "Success. OVS SFC Installation completed"
msg ""
msg "#####################################################################################"
msg "                            NOTICE                                                   "
msg "OVS SFC install completed. Please reboot at the earliest for settings to take effect."
msg "#####################################################################################"

