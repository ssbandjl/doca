#!/bin/bash -e
#
# Copyright (c) 2016-2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
# Testing script for SHARP, to run OSU benchmark.
#

BASE_DIR=$(dirname $0)
SCRIPT_NAME=$(basename $0)

. $BASE_DIR/sharp_funcs.sh

hpcx_module=${hpcx_module:="hpcx-gcc"}

#
# Defaults
#
SHARP_INI_FILE=${SHARP_INI_FILE:-$USER_INI_FILE}
SHARP_PPN="1"
SHARP_TEST_ITERS="10000"
SHARP_TEST_SKIP_ITERS="1000"
SHARP_TEST_MAX_DATA="4096"
SHARP_MAX_PAYLOAD_SIZE=256
SHARP_SR="SHArP.AggregationManager"
SHARP_TMP_DIR_DEFAULT="/tmp"
SHARP_TMP_DIR=${SHARP_TMP_DIR:-$SHARP_TMP_DIR_DEFAULT}
LOGS_DIR=$SHARP_TMP_DIR/sharp_benchmark_logs.$$
MPI_LOG=$LOGS_DIR/sharp_log_mpirun.log
MPI_PT2PT_DEFAULT="ucx"
MPI_PT2PT=${MPI_PT2PT:-$MPI_PT2PT_DEFAULT}
HOSTFILE=$LOGS_DIR/hostfile

function finish()
{
	local rc=$?

	\rm -rf $LOGS_DIR &>/dev/null

	return $((rc))
}

trap finish EXIT

usage()
{
	echo "This script includes OSU benchmarks for MPI_Allreduce and MPI_Barrier blocking collective operations."
	echo "Both benchmarks run with and without using SHARP technology."
	echo ""
	echo "Usage: $SCRIPT_NAME [-t] [-d] [-h] [-f]"
	echo "	-t - tests list (e.g. sharp:barrier)"
	echo "	-d - dry run"
	echo "	-h - display this help and exit"
	echo "  -f - supress error in prerequsites checking"
	echo ""
	echo "Configuration:"
	echo " Runtime:"
	echo "  sharp_ppn - number of processes per compute node (default $SHARP_PPN)"
	echo "  sharp_ib_dev - Infiniband device used for communication. Format <device_name>:<port_number>."
	echo "                 For example: sharp_ib_dev=\"mlx5_0:1\""
	echo "                 This is a mandatory parameter. If it's absent, $SCRIPT_NAME tries to use the first active device on local machine"
	echo "  sharp_groups_num - number of groups per communicator. (default is the number of devices in sharp_ib_dev)"
	echo "  sharp_num_trees - number of trees to request. (default num tress based on the #rails and #channels)"
	echo "  sharp_job_members_type - type of sharp job members list. (default is SHARP_MEMBER_LIST_PROCESSES_DATA)"
	echo "  sharp_hostlist - hostnames of compute nodes used in the benchmark. The list may include normal host names,"
	echo "                   a range of hosts in hostlist format. Under SLURM allocation, SLURM_NODELIST is used as a default"
	echo "  sharp_test_iters - number of test iterations (default $SHARP_TEST_ITERS)"
	echo "  sharp_test_skip_iters - number of test iterations (default $SHARP_TEST_SKIP_ITERS)"
	echo "  sharp_test_max_data - max data size used for testing (default and maximum $SHARP_TEST_MAX_DATA)"
	echo " Environment:"
	echo "  SHARP_INI_FILE - takes configuration from given file instead of $USER_INI_FILE"
	echo "  SHARP_TMP_DIR - store temporary files here instead of $SHARP_TMP_DIR_DEFAULT"
	echo "  HCOLL_INSTALL - use specified hcoll install instead from hpcx"
	echo "  MPI_PT2PT - use specified MPI pt2pt library UCX(default) or MXM"
	echo ""
	echo "Examples:"
	echo "  sharp_ib_dev=\"mlx5_0:1\" $SCRIPT_NAME  # run using \"mlx5_0:1\" IB port. Rest parameters are loaded from $USER_INI_FILE or default"
	echo "  SHARP_INI_FILE=~/benchmark.ini  $SCRIPT_NAME # Override default configuration file"
	echo "  SHARP_INI_FILE=~/benchmark.ini  sharp_hostlist=ajna0[2-3]  sharp_ib_dev=\"mlx5_0:1\" $SCRIPT_NAME # Use specific host list"
	echo "  sharp_ppn=1 sharp_hostlist=ajna0[1-8] sharp_ib_dev=\"mlx5_0:1\" $SCRIPT_NAME  -d # Print commands without actual run"
	echo ""
	echo "Dependencies:"
	echo "  $(print_hostlist_dependency_message)"
	echo "  $SCRIPT_NAME runs \"ibstat\" and \"saquery\" tools on compute nodes. It needs access permmission to umad device."
	echo "  “See \"--umad-dev-rw\" in Mellanox OFED install script."
	exit 2
}

print_configuration()
{
	echo "IB device (sharp_ib_dev): $sharp_ib_dev"
	echo "Number of groups per communicator (sharp_groups_num): ${sharp_groups_num:-$groups_num}"
	echo "Number of trees to request (sharp_num_trees): ${sharp_num_trees:-0}"
	echo "Sharp job members type (sharp_job_members_type): ${sharp_job_members_type:-2}"
	echo "PPN (sharp_ppn): $sharp_ppn"
	echo "Max data size (sharp_test_max_data): $sharp_test_max_data"
	echo "# test iteration (sharp_test_iters): $sharp_test_iters"
	echo "# skip iterations (sharp_test_skip_iters): $sharp_test_skip_iters"
	echo ""
	echo "Configuration file (SHARP_INI_FILE) : $SHARP_INI_FILE"
	echo "Temporary logs (SHARP_TMP_DIR) : $LOGS_DIR"
}

if [ -f $SHARP_INSTALL/../modulefiles/hpcx ]; then
	module load $SHARP_INSTALL/../modulefiles/hpcx
elif module whatis $hpcx_module 2>&1 | grep -q HPC-X; then
	module load $hpcx_module
fi

mpirun="$OMPI_HOME/bin/mpirun"

#
# Command wrapper
# Input parameters: <host name> <command name> <command's prm1 > ... <command's prmN>
# If the host is local host, run the command
# If the host is remote host, run in remote using mpirun
#
command_wrapper()
{
	local host=$1
	local cmd=$2

	if [[ "$host" != "$HOSTNAME" ]]; then
		cmd="$mpirun -H $1 -np 1 $cmd"
	fi

	shift 2
	cmd=$cmd" $@"

	eval $cmd 2>/dev/null
}

#
# Get first active device
# Input parameters: <hostname>
#
get_active_ib_device()
{
	local port
	local ibdev
	local port_info

	for ibdev in $(command_wrapper $1 ibstat -l)
	do
		port=1
		port_info=$( command_wrapper $1 ibstat $ibdev $port )
		( echo "$port_info" | grep -q Active ) && ( echo "$port_info" | grep -q InfiniBand ) && eval echo "$ibdev:$port" && break ||:
	done
}

#
# Check IB device
#
check_ib_device()
{
	local host=$1
	local ib_dev=$2
	local ca_name
	local port
	local port_info
	local sr_info

	if ! check_ib_dev_name "$ib_dev"; then
		return 1
	fi

	ca_name=${ib_dev%:*}
	port=${ib_dev#*:}

	port_info=$( command_wrapper "$host" ibstat "$ca_name" "$port" )

	if [[ $? != "0" ]]; then
		report_check_error "Failed to check $ib_dev $host"
	fi

	if ! [[ "$port_info" =~ Active && "$port_info" =~ InfiniBand ]]; then
		report_check_error "$ib_dev is not active Infiniband device on $host"
	fi

	sr_info=$( command_wrapper "$host" saquery -C "$ca_name" -P "$port" SR )

	if [[ $? != "0" ]]; then
		report_check_error "Failed to fetch SR on $host"
	fi

	if ! [[ "$sr_info" =~ "$SHARP_SR" ]]; then
		report_check_error  "There is no sharp_am running in fabric connected to $ib_dev ($host)"
	fi
}

#
# Report error/warning
#
function report_check_error()
{
	local msg=$1

	if [[ $((force_mode)) == 1 ]]; then
		echo "WARNNING: ${msg}"
	else
		echo "ERROR: ${msg}"
		echo "You can supress the error using \"-f\" force mode"
		exit 1
	fi
}

#
# Parse ib dev provided by a user.
#
parse_sharp_ib_dev_list()
{
	local var=$1
	local ng=0

	if [ -n "${var}" ]; then
		IFS=',' read -ra ADDR <<< "$var"
		for i in "${ADDR[@]}"; do
			if [ -n first_sharp_ib_dev ]; then
				if [ -z $first_sharp_ib_dev ]; then
					first_sharp_ib_dev=$i
				fi
			fi
		(( ng = ng + 1 ))
		done
	fi

	groups_num=$ng
}

#
# Check ib dev provided by a user. If it's empty run auto detection
#
check_and_update_ib_dev()
{
	local var=$1
	local hostlist=$2
	local first_compute_node

	first_compute_node=$( $hostlist_tool $hostlist "-l 1" )
	if [ -z "${!var}" ]; then
		echo "$var is empty. Try to use the first active device"
		if $($hostlist_tool $hostlist -i "$HOSTNAME" -q -0 ); then
			eval $var=$(get_active_ib_device "$HOSTNAME")
		else
			eval $var=$(get_active_ib_device "$first_compute_node" )
		fi

		if [ -z "${!var}" ]; then
			echo "$var can't be empty. Exit."
			exit 1
		fi
	elif ! check_ib_device "$first_compute_node" "${!var}"; then
		if [[ $((force_mode)) == 0 ]]; then
			echo "Use \"-f\" (force mode) for error supression"
			exit 1
		fi
	fi
}

allreduce=1
barrier=1
sharp_allreduce=1
sharp_barrier=1
force_mode=0

while getopts "dt:hf" flag_arg; do
	case $flag_arg in
		d)  dryrun=1   ;;
		t)	allreduce=
			barrier=
			sharp_allreduce=
			sharp_barrier=

			for x in $OPTARG; do
				case $x in
					allreduce)       allreduce=1       ;;
					barrier)         barrier=1         ;;
					sharp:allreduce) sharp_allreduce=1 ;;
					sharp:barrier)   sharp_barrier=1   ;;
					*)               echo "Unknown parameter: $x"
									 usage ;;
				esac
			done ;;
		f) force_mode=1;;
		h) usage ;;
		*) usage ;;

	esac
done


if [ -w $SHARP_INSTALL ]; then
	_local_dir=$SHARP_INSTALL
else
	_local_dir=$HOME/sharp_conf
fi

load_params $_local_dir

OUT_AM=/tmp/d_sharp_am.log
AM_CFG=$DCONF_DIR/sharp_am.cfg

#
# Check dependency
#
check_hostlist_tool

#
# Check/Update mandatory parameters
#
check_and_update_hostlist sharp_hostlist
parse_sharp_ib_dev_list $sharp_ib_dev
check_and_update_ib_dev first_sharp_ib_dev $sharp_hostlist
if [ -z $sharp_ib_dev ]; then
	sharp_ib_dev=$first_sharp_ib_dev
fi

#
# Run OMPI tests
#
sharp_test_iters=${sharp_test_iters:-$SHARP_TEST_ITERS}
sharp_test_skip_iters=${sharp_test_skip_iters:-$SHARP_TEST_SKIP_ITERS}
sharp_test_max_data=${sharp_test_max_data:-$SHARP_TEST_MAX_DATA}
sharp_ppn=${sharp_ppn:-$SHARP_PPN}

bind_to_core=" taskset -c 1 numactl --membind=0 "
nnodes=$($hostlist_tool -e $sharp_hostlist | wc -l)

np=$((sharp_ppn * nnodes))

if (( "$sharp_test_max_data" > "$SHARP_TEST_MAX_DATA" )); then
       echo "For data packets bigger than $SHARP_TEST_MAX_DATA SHARP is not used"
fi

#
# Prepare logs
#
mkdir -p $LOGS_DIR &>/dev/null

print_configuration

#
# If local host is in the hosts list, place it as the first host
# it is performed by concatenating two hosts lists:
# 1. Intersecting local host with hostlist
# 2. Removing local host from hostlist
#
$hostlist_tool -e -i $(hostname) $sharp_hostlist > $HOSTFILE
$hostlist_tool -e -d $sharp_hostlist $(hostname) >> $HOSTFILE


#
# Prepare common options for both all runs: with/without SHARP
#
opt="--bind-to core --map-by node -hostfile $HOSTFILE -np $np "
#opt+="-mca oob ^ud " # Skip using UD for OOB. This workaround prevents open unconnected port in head node
#opt+="--debug-daemons "
#opt+=" --display-map "
opts+=" --report-bindigs "
opt+=" -mca btl_openib_warn_default_gid_prefix 0 "
opt+=" -mca rmaps_dist_device $first_sharp_ib_dev -mca rmaps_base_mapping_policy dist:span "
opt+=" -mca btl_openib_if_include $first_sharp_ib_dev "
opt+=" -x HCOLL_MAIN_IB=$sharp_ib_dev "
if [ "$MPI_PT2PT" = "ucx" ]; then
    opt+=" -mca pml ucx "
    opt+=" -x UCX_NET_DEVICES=$first_sharp_ib_dev "
    # HCOLL_BCOL and HCOLL_SBGP enable using multi-channel feature
    #opt+=" -x HCOLL_SBGP=basesmsocket,p2p -x HCOLL_BCOL=basesmuma,ucx_p2p "
elif [ "$MPI_PT2PT" = "mxm" ]; then
    opt+=" -mca pml yalla -mca osc ^ucx "
    opt+=" -x MXM_RDMA_PORTS=$first_sharp_ib_dev "
    opt+=" -x MXM_LOG_LEVEL=ERROR "
    # HCOLL_BCOL and HCOLL_SBGP enable using multi-channel feature
    #opt+=" -x HCOLL_SBGP=basesmsocket,p2p -x HCOLL_BCOL=basesmuma,mlnx_p2p "
fi
#   opt+=" -x HCOLL_ENABLE_MCAST_ALL=1 "  # default: Enabled
#   opt+=" -x HCOLL_MCAST_NP=1 "
#   opt+=" -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$SHARP_INSTALL/lib "

#  temporary workaround for race condition between libmlx5/libibverbs/librdmacm destructors
ld_preload=""
if [ -f /usr/lib64/libmlx5.so ]; then
    ld_preload="/usr/lib64/libmlx5.so:"
fi

ld_preload+="$SHARP_INSTALL/lib/libsharp.so:$SHARP_INSTALL/lib/libsharp_coll.so"
if [ -n "${HCOLL_INSTALL:-}" ]; then
    ld_preload+=":$HCOLL_INSTALL/lib/libocoms.so:$HCOLL_INSTALL/lib/libhcoll.so"
fi
opt+="-x LD_LIBRARY_PATH "
opt+="-x LD_PRELOAD=$ld_preload "

#
# Find executables
#
osu_allreduce_exe=$(find $OMPI_HOME/tests/osu* -name osu_allreduce | grep -v cuda | tail -1)
osu_barrier_exe=$(find $OMPI_HOME/tests/osu* -name osu_barrier | grep -v cuda | tail -1)

if [ "$sharp_ppn" -eq "1" ]; then
	osu_allreduce_exe="$bind_to_core $osu_allreduce_exe"
	osu_barrier_exe="$bind_to_core $osu_barrier_exe"
fi


#
# Prepare OSU benchmark parameters
#
osu_opt="-i $sharp_test_iters -x $sharp_test_skip_iters -f -m :${sharp_test_max_data}"


#
# Prepare SHARP specific parameters
#
hcoll_sharp_opt=" -x HCOLL_ENABLE_SHARP=3 " # Enables sharp without fallback. Force to use SHArP for all groups.
hcoll_sharp_opt+=" -x SHARP_COLL_LOG_LEVEL=3  " # Enables libsharp logging
#   hcoll_sharp_opt+=" -x SHARP_COLL_GROUP_RESOURCE_POLICY=1 " # Equal distribution of OSTs among the groups. By default : 1
#	hcoll_sharp_opt+=" -x SHARP_COLL_ENABLE_MCAST_TARGET=1 " # UD MCAST. By default : enabled
#	hcoll_sharp_opt+=" -x SHARP_COLL_GROUP_IS_TARGET=1 " # Enable RC distribution. By default : enabled
#	hcoll_sharp_opt+=" -x SHARP_COLL_ENABLE_GROUP_TRIM=1 " # SHARP group trim. By default : enabled
#   hcoll_sharp_opt+=" -x HCOLL_SHARP_UPROGRESS_NUM_POLLS=999 "
hcoll_sharp_opt+=" -x HCOLL_BCOL_P2P_ALLREDUCE_SHARP_MAX=$sharp_test_max_data "
#   hcoll_sharp_opt+=" -x SHARP_COLL_PIPELINE_DEPTH=32 " # Size of fragmentation pipeline for larger collective payload
#	hcoll_sharp_opt+=" -x SHARP_COLL_POLL_BATCH=1 " # How many CQ completions to poll on at once. Maximum:16
# Job quota
hcoll_sharp_opt+=" -x SHARP_COLL_JOB_QUOTA_OSTS=32 " # OST quota request. value 0 mean allocate default quota.
hcoll_sharp_opt+=" -x SHARP_COLL_JOB_QUOTA_MAX_GROUPS=4 "
hcoll_sharp_opt+=" -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=$SHARP_MAX_PAYLOAD_SIZE " # Maximum payload per OST quota request. value 256 is max

num_trees=${sharp_num_trees:-0}
if [ "$num_trees" -gt 0 ]; then
	hcoll_sharp_opt+=" -x SHARP_COLL_JOB_NUM_TREES=$num_trees "
fi

groups_num=${sharp_groups_num:-$groups_num}
if [ "$groups_num" -gt 1 ]; then
	hcoll_sharp_opt+=" -x SHARP_COLL_GROUPS_PER_COMM=$groups_num "
fi

sharp_job_members_type=${sharp_job_members_type:-2}
hcoll_sharp_opt+=" -x SHARP_COLL_JOB_MEMBER_LIST_TYPE=$sharp_job_members_type "

run_cmd()
{
	local _test=$1
	local _cmd=$2
	local _var=$3

	if [ -n "$_test" ]; then
		echo -e "$_cmd"
		if [ -z "$dryrun" ]; then
			$_cmd 2>&1 | tee -a $MPI_LOG
			eval $_var=${PIPESTATUS[0]}
		fi
		echo ""
	fi
}

var1=0
var2=0
var3=0
var4=0

if [ -n "$sharp_allreduce" ] || [ -n "$sharp_barrier" ]; then
	printf "\n\n....................................................\n"
	printf "   w/ SHARP\n"
	printf "....................................................\n"
fi
run_cmd "$sharp_allreduce" "$mpirun $opt $hcoll_sharp_opt $osu_allreduce_exe $osu_opt" "var1"
run_cmd "$sharp_barrier"   "$mpirun $opt $hcoll_sharp_opt $osu_barrier_exe $osu_opt" "var2"

if [ -n "$allreduce" ] || [ -n "$barrier" ]; then
	printf "\n\n....................................................\n"
	printf "   w/o SHARP\n"
	printf "....................................................\n"
fi
run_cmd "$allreduce" "$mpirun $opt $osu_allreduce_exe $osu_opt" "var3"
run_cmd "$barrier"   "$mpirun $opt $osu_barrier_exe $osu_opt" "var4"


if [ $var1 -ne 0 ] || [ $var2 -ne 0 ] || [ $var3 -ne 0 ] || [ $var4 -ne 0 ]; then
	exit 1
else
	exit 0
fi
