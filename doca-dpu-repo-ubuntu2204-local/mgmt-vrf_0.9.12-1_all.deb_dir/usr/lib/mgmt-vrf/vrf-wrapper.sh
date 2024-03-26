#!/usr/bin/env bash

# Wrapper to run a command in default VRF if various conditions
# are not met. Assumes original commands are diverted to
# /usr/share/mgmt-vrf.

progname=$(basename "$0")
realname=$(basename $(realpath "$0"))

case "$progname" in
	ping|ping6|traceroute|traceroute6)
		:
		;;
	*)
		echo "$realname: Invalid command \"$progname\"" >&2
		exit 1
		;;
esac

DIVERT_DIR=/usr/share/mgmt-vrf
which=$(which which)
if ! progexec=$(PATH=$DIVERT_DIR/usr/bin:$DIVERT_DIR/usr/sbin:$DIVERT_DIR/bin
	"$which" $progname); then
	echo "$realname: $progname: command not found" >&2
	exit 1
fi

opt_no_vrf=0
params=("$@")
i=0
nb=${#params[@]}
while ((i < nb)); do
	if [ "${params[i]}" = "--no-vrf-switch" ]; then
		opt_no_vrf=1
		unset params[$i]
	fi
	((i++))
done

set -- "${params[@]}"

if [ $opt_no_vrf -eq 0 ]; then
	# get name of management vrf
	mgmt=$(mgmt-vrf name)

	# get vrf we are running in
	vrf=$(ip vrf id)

	# Want all commands to default to front panel ports using default
	# VRF and user specifies argument for other VRF context. So if we
	# are running in mgmt VRF switch to default VRF and run command.
	if [ "$vrf" = "$mgmt" ]; then
		echo "$realname: switching to vrf \"default\"; use '--no-vrf-switch' to disable"
		sudo vrf task set default $$
	fi
fi

exec "$progexec" "$@"
