#!/bin/bash -eE
# Copyright (c) 2019 Mellanox Technologies. All rights reserved.

function is_number() {
    re='^[0-9]+$'
    if [[ $1 =~ $re ]] ; then
        echo "True"
    else
        echo "False"
    fi
}

export NVME_SF_ECPF_DEV=0000:03:00.0

internal_cpu_enabled=$(mlxconfig -d "$NVME_SF_ECPF_DEV" -e q INTERNAL_CPU_MODEL | sed 's/*//g' | awk '/INTERNAL_CPU_MODEL/ {print $3}')
if [ "$internal_cpu_enabled" != "EMBEDDED_CPU(1)" ] ; then
    logger -p local0.notice -t mlnx_snap "$prog: INTERNAL_CPU_MODEL is disabled for $NVME_SF_ECPF_DEV"
    echo "INTERNAL_CPU_MODEL is disabled for $NVME_SF_ECPF_DEV"
    exit 69 # EX_UNAVAILABLE
fi

nvme_emu_enabled=$(mlxconfig -d "$NVME_SF_ECPF_DEV" -e q NVME_EMULATION_ENABLE | sed 's/*//g' | awk '/NVME_EMULATION_ENABLE/ {print $3}')
vblk_emu_enabled=$(mlxconfig -d "$NVME_SF_ECPF_DEV" -e q VIRTIO_BLK_EMULATION_ENABLE | sed 's/*//g' | awk '/VIRTIO_BLK_EMULATION_ENABLE/ {print $3}')
vfs_emu_enabled=$(mlxconfig -d "$NVME_SF_ECPF_DEV" -e q VIRTIO_FS_EMULATION_ENABLE | sed 's/*//g' | awk '/VIRTIO_FS_EMULATION_ENABLE/ {print $3}')
if [[ ("$nvme_emu_enabled" != "True(1)") && ("$vblk_emu_enabled" != "True(1)") && ("$vfs_emu_enabled" != "True(1)") ]] ; then
    logger -p local0.notice -t mlnx_snap "$prog: all NVME/VIRTO_BLK/VIRTIO_FS emulations are disabled for $NVME_SF_ECPF_DEV"
    echo "all NVME/VIRTO_BLK/VIRTIO_FS emulations are disabled for $NVME_SF_ECPF_DEV"
    exit 69 # EX_UNAVAILABLE
fi

exit 0
