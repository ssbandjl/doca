#!/bin/bash
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

# This script takes 2 arguments:
# arg1: set or query - always "set" first then "query"
# arg2: device such as /dev/mst/mt41692_pciconf0
# example usage:
# sudo ./pcc_counters.sh set /dev/mst/mt41692_pciconf0
# sudo ./pcc_counters.sh query /dev/mst/mt41692_pciconf0


pci="0"
IFS=""
while read -r line ; do
    IFS=" "
    read -ra LINE <<< "$line"
    if [[ ${LINE[1]} != "" ]] && [[ ${LINE[1]} == $2 ]] ; then
        pci="${LINE[2]}"
    fi
done <<< $(sudo mst status -v)

if [[ $pci == "0" ]] ; then
    echo "ERROR: Bad Device"
    exit
fi
pci=`echo "$(lspci -D | grep $pci | awk {'print $1'})"`

if [ "$1" == "set" ] ; then
    sudo echo 0x0a07, 0x2c11,0x2c12,0x2c13,0x2c14,0x2c15,0x2c16,0x2c17,0x2c18,0x100b,0x100c,0x200d,0x200e,0x200f,0x2010,0x2011,0x2012,0x2013,0x2014 > /sys/kernel/debug/mlx5/"$pci"/diag_cnt/counter_id
    sudo echo 1,100,8c,1,0 > /sys/kernel/debug/mlx5/"$pci"/diag_cnt/params
    sudo echo set > /sys/kernel/debug/mlx5/"$pci"/diag_cnt/dump
elif [ "$1" == "query" ]  ; then
    pcc_counters="/sys/kernel/debug/mlx5/"$pci"/diag_cnt/dump"
    echo "-----------------PCC Counters-----------------"
    IFS=""
    while read -r line ; do
        IFS=","
        read -ra LINE <<< "$line"
        case ${LINE[0]} in
        0a07) echo "Counter: PCC_CNP_COUNT          Value: ${LINE[3]}";;
        100b) echo "Counter: MAD_RTT_PERF_CONT_REQ  Value: ${LINE[3]}";;
        100c) echo "Counter: MAD_RTT_PERF_CONT_RES  Value: ${LINE[3]}";;
        200d) echo "Counter: SX_EVENT_WRED_DROP     Value: ${LINE[3]}";;
        200e) echo "Counter: SX_RTT_EVENT_WRED_DROP Value: ${LINE[3]}";;
        200f) echo "Counter: ACK_EVENT_WRED_DROP    Value: ${LINE[3]}";;
        2010) echo "Counter: NACK_EVENT_WRED_DROP   Value: ${LINE[3]}";;
        2011) echo "Counter: CNP_EVENT_WRED_DROP    Value: ${LINE[3]}";;
        2012) echo "Counter: RTT_EVENT_WRED_DROP    Value: ${LINE[3]}";;
        2013) echo "Counter: HANDLED_SXW_EVENTS     Value: ${LINE[3]}";;
        2014) echo "Counter: HANDLED_RXT_EVENTS     Value: ${LINE[3]}";;
        2c11) echo "Counter: DROP_RTT_PORT0_REQ     Value: ${LINE[3]}";;
        2c12) echo "Counter: DROP_RTT_PORT1_REQ     Value: ${LINE[3]}";;
        2c13) echo "Counter: DROP_RTT_PORT0_RES     Value: ${LINE[3]}";;
        2c14) echo "Counter: DROP_RTT_PORT1_RES     Value: ${LINE[3]}";;
        2c15) echo "Counter: RTT_GEN_PORT0_REQ      Value: ${LINE[3]}";;
        2c16) echo "Counter: RTT_GEN_PORT1_REQ      Value: ${LINE[3]}";;
        2c17) echo "Counter: RTT_GEN_PORT0_RES      Value: ${LINE[3]}";;
        2c18) echo "Counter: RTT_GEN_PORT1_RES      Value: ${LINE[3]}";;
        esac
    done <<< $(sudo more $pcc_counters)
else
    echo "Bad Request: choose 'set' or 'query'"
fi
