#!/bin/bash -eE
# Copyright © 2022 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.

# $1 = spdk app args
DEFAULT_RPC_ADDR="/var/tmp/spdk.sock"
RPC_SERVER_ADDR=$DEFAULT_RPC_ADDR


# extract the RPC socket path, if non-default
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--rpc-socket)
      RPC_SERVER_ADDR="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      shift # past argument
      ;;
  esac
done

# Wait for 20 seconds until SPDK app is ready to receive RPC commands
for ((i = 100; i != 0; i--)); do
	if /usr/bin/spdk_rpc.py -t 1 -s "$RPC_SERVER_ADDR" rpc_get_methods &> /dev/null; then
		break
	fi

	sleep 0.2
done

if ((i == 0)); then
	echo "Cannot locate SPDK application running"
	exit 1
fi

exit 0
