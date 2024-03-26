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

# We do not set safe bash options intentionally since some ovs commands do return non-zero exit code.
# set -euo pipefail

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "Error: $msg"
  exit "$code"
}

msg() {
  echo >&2 -e "${1-}"
}

function retry {
  local retries=$1
  shift

  local count=0
  until "$@"; do
    exit=$?
    wait=$((2 ** $count))
    count=$(($count + 1))
    if [ $count -lt $retries ]; then
      msg "Retry $count/$retries exited $exit, retrying in $wait seconds..."
      sleep $wait
    else
      msg "Retry $count/$retries exited $exit, no more retries left."
      return $exit
    fi
  done
  return 0
}


###  OVS functions
OVS_TIMEOUT=60   # seconds to timeout ovs operation, doca port addition is sometime slow changing from 30 to 60
OFCTL_TIMEOUT=60  # Openflow timeout, doca operations sometime block the main thread for 25 seconds increasing the number from 30 to 60
VSCTL="/usr/bin/ovs-vsctl --timeout=${OVS_TIMEOUT}"
OFCTL="/usr/bin/ovs-ofctl --timeout=${OFCTL_TIMEOUT}"
BR_NAME=${BR_NAME:-br-hbn}
OF_PORT_INDEX=1
MTU_UPLINK=${MTU_UPLINK:-9702}
MTU_INTERFACES=${MTU_INTERFACES:-9216}

check_br(){
    ${VSCTL} br-exists $1
    retval=$?
    if [ ${retval} -ne 0 ]; then
        msg "Bridge $1 not present"
        return ${retval}
    fi
}

create_bridge() {
    local bridgename=$1
    ${VSCTL} --may-exist add-br ${bridgename} -- set bridge ${bridgename} datapath_type=netdev -- set-fail-mode ${bridgename} secure
    local index=$(${VSCTL} list interface vxlan0 2>/dev/null | grep ofport_request | cut -d ":" -f2 | xargs)
    if ! [ -z "${index}" ]; then
        msg "Port vxlan0 is already added to bridge ${bridgename} and configured with ofport_request=${index}"
        return
    fi
    ${VSCTL} --may-exist add-port ${bridgename} vxlan0 -- set interface vxlan0 type=vxlan ofport_request=${OF_PORT_INDEX} options:explicit=true options:tos=inherit options:remote_ip=flow -- set bridge ${bridgename} external_ids:ofport_index=$((${OF_PORT_INDEX}+1))
    OF_PORT_INDEX=$((${OF_PORT_INDEX}+1))
}

ensure_bridge() {
  local bridgename=$1
  msg "Ensuring bridge ${bridgename} is present"
  create_bridge ${bridgename}
  retry 5 check_br ${bridgename}
  retval=$?
    if [ ${retval} -ne 0 ]; then
        msg "Bridge $1 is not present. See Openvswitch logs. Exiting"
        exit -1
    fi
}

delete_bridge() {
    local bridgename=$1
    ${VSCTL} --if-exists del-br ${bridgename}
}

del_flows(){
    local bridge=${1:-br-hbn}
    ${OFCTL} del-flows ${bridge}
}

set_ofport_index(){
  local index=$(${VSCTL} --data=bare list bridge ${BR_NAME} | grep ofport_index | cut -d "=" -f2)
  if ! [ -z "${index}" ]; then
    msg "ofport_index on bridge ${BR_NAME} is :${index}"
    OF_PORT_INDEX=${index}
  fi
}

initialize(){
    delete_bridge ovsbr1
    delete_bridge ovsbr2
    local restart=""
    msg "==> Ensuring OVS Hardware offload is enabled"
    ovs_hw_offload=$(${VSCTL} get Open_vSwitch . other_config:hw-offload 2> /dev/null | tr -d '\"')
    if [ "X${ovs_hw_offload}" != "Xtrue" ]; then
        ${VSCTL} set Open_vSwitch . other_config:hw-offload=true
        msg "Marking OVS for restart after setting hw-offload to true "
        restart="yes"
    fi
    msg "==> Completed OVS Hardware offload configuration"
    msg "==> Ensuring DPDK is enabled"
    ovs_dpdk=$(${VSCTL} get Open_vSwitch . other_config:dpdk-init 2> /dev/null | tr -d '\"')
    if [ "X${ovs_dpdk}" != "Xtrue" ]; then
        ${VSCTL} set Open_vSwitch . other_config:dpdk-init=true
        msg "Marking OVS for restart after setting dpdk-init to true "
        restart="yes"
    fi
    msg "==> Completed DPDK configuration"
    msg "==> Ensuring DOCA is enabled"
    ovs_doca=$(${VSCTL} get Open_vSwitch . other_config:doca-init 2> /dev/null | tr -d '\"')
    if [ "X${ovs_doca}" != "Xtrue" ]; then
        ${VSCTL} set Open_vSwitch . other_config:doca-init=true
        msg "Marking OVS for restart after setting doca-init to true "
        restart="yes"
    fi
    msg "==> Completed DOCA configuration"
    msg "==> Ensuring DPDK-EXTRA is configured"
    ovs_doca_extra=$(${VSCTL} get Open_vSwitch . other_config:dpdk-extra 2> /dev/null | tr -d '\"')
    if [ "X${ovs_doca_extra}" != "X-a 0000:00:00.0" ]; then
        ${VSCTL} set Open_vSwitch . other_config:dpdk-extra="-a 0000:00:00.0"
        msg "Marking OVS for restart after setting dpdk-extra to \"-a 0000:00:00.0\" "
        restart="yes"
    fi
    msg "==> Completed DPDK-EXTRA configuration"
    msg "==> Ensuring CTL PIPE size is configured"
    ovs_ctl_pipe_size=$(${VSCTL} get Open_vSwitch . other_config:ctl-pipe-size 2> /dev/null | tr -d '\"')
    if [ "X${ovs_ctl_pipe_size}" != "X64" ]; then
        ${VSCTL} set Open_vSwitch . other_config:ctl-pipe-size=64
    fi
    msg "==> Completed CTL PIPE size configuration"
    msg "==> Ensuring HW OFFLOAD CT size is configured"
    ovs_hw_offload_ct_size=$(${VSCTL} get Open_vSwitch . other_config:hw-offload-ct-size 2> /dev/null | tr -d '\"')
    if [ "X${ovs_hw_offload_ct_size}" != "X0" ]; then
        ${VSCTL} set Open_vSwitch . other_config:hw-offload-ct-size=0
        msg "Marking OVS for restart after setting hw-offload-ct-size to 0"
        restart="yes"
    fi
    msg "==> Completed HW OFFLOAD CT size configuration"
    msg "==> Ensuring PMD-quiet-idle is configured"
    ovs_pmd_quiet_idle=$(${VSCTL} get Open_vSwitch . other_config:pmd-quiet-idle 2> /dev/null | tr -d '\"')
    if [ "X${ovs_pmd_quiet_idle}" != "Xtrue" ]; then
        ${VSCTL} set Open_vSwitch . other_config:pmd-quiet-idle=true
        msg "Marking OVS for restart after setting pmd-quiet-idle to true "
        restart="yes"
    fi
    msg "==> Completed PMD-quiet-idle size configuration"
    msg "==> Ensuring DPDK max memory zone is configured"
    ovs_dpdk_memzone=$(${VSCTL} get Open_vSwitch . other_config:dpdk-max-memzones 2> /dev/null | tr -d '\"')
    if [ "X${ovs_dpdk_memzone}" != "X50000" ]; then
        ${VSCTL} set Open_vSwitch . other_config:dpdk-max-memzones=50000
        msg "Marking OVS for restart after setting DPDK max memory zone"
        restart="yes"
    fi
    msg "==> Completed DPDK max memory zone configuration"
    if [ "X${restart}" == "Xyes" ]; then
        systemctl restart openvswitch-switch
        msg "Exiting after setting the required other_config"
        exit 1 # Bail out and wait for the script to be called again
    fi
    ${VSCTL} set Open_vSwitch . other_config:max-idle=60000

    set_ofport_index
    ensure_bridge ${BR_NAME}
}

# Add_port() adds given port to the given bridge and sets openflow `ofport_request`
# index based on global OF_PORT_INDEX
add_port(){
    local port=$1
    local bridge=${2:-br-hbn}
    local pci_addr=$(ls -l /sys/class/net/${port}/device | cut -d ":" -f 3,4 | cut -d "/" -f 1)
    if [ ${#pci_addr} -ge 2 ]; then
        pci_addr=${pci_addr::-2}.0
    else
        msg "Failed to find pci address for port ${port}"
            if [ "X$(uname -m)" == "Xx86_64" ]; then
                msg "Defaulting to 08:00.0"
                pci_addr="08:00.0"
            else
                msg "Defaulting to 03:00.0"
                pci_addr="03:00.0"
            fi
    fi
    local rep_index=$(cat /sys/class/net/${port}/phys_port_name | grep pf | cut -d "f" -f 3)
    local phys_port_name=$(cat /sys/class/net/${port}/phys_port_name)
    local index=$(${VSCTL} list interface $port 2>/dev/null | grep ofport_request | cut -d ":" -f2 | xargs)
    if ! [ -z "${index}" ]; then
        msg "Port is already added to bridge ${BR_NAME} and configured with ofport_request=${index}"
        return
    fi
    msg "Adding port: ${port} to bridge: ${bridge} with index ${OF_PORT_INDEX}"
    if [ -z "${rep_index}" ]; then
        if [ "X${phys_port_name}" == "Xp1" ]; then
            ${VSCTL} --may-exist add-port ${bridge} ${port}  -- set interface ${port} type=dpdk ofport_request=${OF_PORT_INDEX} options:dpdk-devargs="${pci_addr},dv_xmeta_en=4,dv_flow_en=2,representor=pf1" -- set interface ${port} mtu_request=${MTU_UPLINK} -- set bridge ${BR_NAME} external_ids:ofport_index=$((${OF_PORT_INDEX}+1)) -- set interface ${port} options:iface-name=${port}
        elif [ "X${port}" == "Xpf0hpf" ]; then
            ${VSCTL} --may-exist add-port ${bridge} ${port}  -- set interface ${port} type=dpdk ofport_request=${OF_PORT_INDEX} options:dpdk-devargs="03:00.0,dv_xmeta_en=4,dv_flow_en=2,representor=pf0vf65535" -- set interface ${port} mtu_request=${MTU_INTERFACES} -- set bridge ${BR_NAME} external_ids:ofport_index=$((${OF_PORT_INDEX}+1)) -- set interface ${port} options:iface-name=${port}
        elif [ "X${port}" == "Xpf1hpf" ]; then
            ${VSCTL} --may-exist add-port ${bridge} ${port}  -- set interface ${port} type=dpdk ofport_request=${OF_PORT_INDEX} options:dpdk-devargs="03:00.0,dv_xmeta_en=4,dv_flow_en=2,representor=pf1vf65535" -- set interface ${port} mtu_request=${MTU_INTERFACES} -- set bridge ${BR_NAME} external_ids:ofport_index=$((${OF_PORT_INDEX}+1)) -- set interface ${port} options:iface-name=${port}
        elif [ "X${phys_port_name}" == "Xp0" ]; then
            ${VSCTL} --may-exist add-port ${bridge} ${port}  -- set interface ${port} type=dpdk ofport_request=${OF_PORT_INDEX} options:dpdk-devargs="${pci_addr},dv_xmeta_en=4,dv_flow_en=2" -- set interface ${port} mtu_request=${MTU_UPLINK} -- set bridge ${BR_NAME} external_ids:ofport_index=$((${OF_PORT_INDEX}+1)) -- set interface ${port} options:iface-name=${port}
        else
            ${VSCTL} --may-exist add-port ${bridge} ${port}  -- set interface ${port} type=dpdk ofport_request=${OF_PORT_INDEX} options:dpdk-devargs="${pci_addr},dv_xmeta_en=4,dv_flow_en=2,representor=${port}" -- set interface ${port} mtu_request=${MTU_INTERFACES} -- set bridge ${BR_NAME} external_ids:ofport_index=$((${OF_PORT_INDEX}+1)) -- set interface ${port} options:iface-name=${port}
        fi
    else
        rep_index=$(cat /sys/class/net/${port}/phys_port_name | grep vf)
        if [ "X${rep_index}" != "X" ]; then
            ${VSCTL} --may-exist add-port ${bridge} ${port}  -- set interface ${port} type=dpdk ofport_request=${OF_PORT_INDEX} options:dpdk-devargs="${pci_addr},dv_xmeta_en=4,dv_flow_en=2,representor=${port}" -- set interface ${port} mtu_request=${MTU_INTERFACES} -- set bridge ${BR_NAME} external_ids:ofport_index=$((${OF_PORT_INDEX}+1)) -- set interface ${port} options:iface-name=${port}
        else
            ${VSCTL} --may-exist add-port ${bridge} ${port}  -- set interface ${port} type=dpdk ofport_request=${OF_PORT_INDEX} options:dpdk-devargs="${pci_addr},dv_xmeta_en=4,dv_flow_en=2,representor=${phys_port_name}" -- set interface ${port} mtu_request=${MTU_INTERFACES} -- set bridge ${BR_NAME} external_ids:ofport_index=$((${OF_PORT_INDEX}+1)) -- set interface ${port} options:iface-name=${port}
        fi
    fi
    OF_PORT_INDEX=$((${OF_PORT_INDEX}+1))
}

# Add_patch_flows() adds openflow rules to set up bidirectional flows between the given ports and bridge.
# The flows uses `ofport_request` index numbers instead of port names.
# These are obtained from the interface table.

add_patch_flow(){
    local port1=$1
    local port2=$2
    local bridge=${3:-br-hbn}

    check_br ${bridge}

    msg "Setting flows for bridge: ${bridge} with ports: ${port1} and ${port2}"
    local port1_idx
    local port2_idx

    port1_idx=$(${VSCTL} list interface ${port1} 2>/dev/null | grep ofport_request | cut -d ":" -f2 | xargs)
    port2_idx=$(${VSCTL} list interface ${port2} 2>/dev/null | grep ofport_request | cut -d ":" -f2 | xargs)
    port1_idx_actual=$(${VSCTL} list interface ${port1} 2>/dev/null | grep 'ofport ' | cut -d ":" -f2 | xargs)
    port2_idx_actual=$(${VSCTL} list interface ${port2} 2>/dev/null | grep 'ofport ' | cut -d ":" -f2 | xargs)

    if [ "${port1_idx_actual}" != "-1" ] && [ "${port1_idx}"  != "${port1_idx_actual}" ]; then
      die "Port: ${port1} does not have ofport:${port1_idx} the same as ofport_request:${port1_idx_actual}"
    fi
    if [ "${port2_idx_actual}" != "-1" ] && [ "${port2_idx}"  != "${port2_idx_actual}" ]; then
      die "Port: ${port2} does not have ofport:${port2_idx} the same as ofport_request:${port2_idx_actual}"
    fi

    ${OFCTL} add-flow ${bridge} table=0,priority=0,in_port=${port1_idx},actions=output:${port2_idx}
    ${OFCTL} add-flow ${bridge} table=0,priority=0,in_port=${port2_idx},actions=output:${port1_idx}
}

set_patch(){

    local port1=$1
    local port2=$2
    local bridge=${3:-br-hbn}
    local port1_hbn_dev=$4
    local port1_of_port=$5
    local port2_hbn_dev=$6
    local port3_of_port=$7

    add_port ${port1} ${bridge}
    add_port ${port2} ${bridge}

    # Set metadata needed by dal2ovs
    if [ ! -z "${port1_hbn_dev}" ] && test -z "$(echo -n "${port1_hbn_dev}" | tr -d 'a-zA-Z0-9_')"; then
      ${VSCTL} set interface ${port1} external_ids:hbn_netdev="${port1_hbn_dev}"
    fi
    if [ ! -z "${port1_of_port}" ] && test -z "$(echo -n "${port1_of_port}" | tr -d 'a-zA-Z0-9_')"; then
      ${VSCTL} set interface ${port1} external_ids:hbn_rep_ofport="${port1_of_port}"
    fi

    if [ ! -z "${port2_hbn_dev}" ] && test -z "$(echo -n "${port2_hbn_dev}" | tr -d 'a-zA-Z0-9_')"; then
      ${VSCTL} set interface ${port2} external_ids:hbn_netdev="${port2_hbn_dev}"
    fi
    if [ ! -z "${port2_of_port}" ] && test -z "$(echo -n "${port2_of_port}" | tr -d 'a-zA-Z0-9_')"; then
      ${VSCTL} set interface ${port2} external_ids:hbn_rep_ofport="${port2_of_port}"
    fi

    add_patch_flow ${port1} ${port2} ${bridge}
}


ensure_cni_bridge_iptables(){
  # Mgmt bridge configuration
  msg "Ensuring br-mgmt bridge"
  if ! (ip l | grep -q br-mgmt) ; then
        msg "Adding br-mgmt"
        ip link add br-mgmt type bridge
  fi
  msg "Ensuring br-mgmt to mgmt vrf"
  ip link set dev br-mgmt master mgmt
  ip link set dev br-mgmt up

    # Add CNI iptables rules to work with mgmt VRF
  msg "Ensuring IPtables rules"
  iptables -P FORWARD ACCEPT
  if ! iptables-save -t nat | grep -q ":CNI-HOSTPORT-DNAT"; then
    msg "Adding DNAT IPtables rules"
    iptables -t nat -N CNI-HOSTPORT-DNAT
    iptables -t nat -A PREROUTING -p tcp -m tcp --dport 8765 -j CNI-HOSTPORT-DNAT
  fi
}


#################################
###         Main              ###
#################################

source /etc/mellanox/sfc.conf  # Get mappings

initialize   # Initialize bridges, setup indexes, cleanup flows etc.

msg "Bridge name: ${BR_NAME}"
for MAP in "${MAPPINGS[@]}"; do
    string=($(echo "$MAP" | tr '~' '\n'))
    port1=${string[0]}
    port2=${string[1]}
    port1_hbn_dev=${string[2]}
    port1_of_port=${string[3]}
    port2_hbn_dev=${string[4]}
    port2_of_port=${string[5]}
    echo "Port mapping : ${port1} <---> ${port2} <---> ${port1_hbn_dev} <---> ${port1_of_port} <---> ${port2_hbn_dev} <---> ${port2_of_port}"
    set_patch ${port1} ${port2} ${BR_NAME} ${port1_hbn_dev} ${port1_of_port} ${port2_hbn_dev} ${port2_of_port}
done

# Store index for future.
${VSCTL} set bridge ${BR_NAME} external_ids:ofport_index=${OF_PORT_INDEX}

ensure_cni_bridge_iptables

msg "SFC Completed"
date -u >> /tmp/sfc-activated
