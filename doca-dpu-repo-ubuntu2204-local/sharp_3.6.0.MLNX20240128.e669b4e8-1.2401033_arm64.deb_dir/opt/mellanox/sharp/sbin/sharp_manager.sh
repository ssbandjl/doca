#!/bin/bash
#
# Copyright (c) 2016-2018 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

#
# If you want that env. var. will be persistently saved
# please build it the following way: [sharp_am|sharp]_[var_name]
#

umask 002

. $(dirname $0)/sharp_funcs.sh


BASE_NAME=$(basename $0)

usage()
{
	echo "$BASE_NAME helps to system administrator to manage SHARP daemons."
	echo "Most of management operations require root access to nodes."
	echo ""
	echo "Usage: $BASE_NAME <operation> [-p process] [-l hostlist] [-s am_server] [-w write_logs_to] [-x var_name] [-j] [-m] [-h]"
	echo "	Operations:"
	echo "	        'start/restart/stop/status' - apply to SHARP daemons"
	echo "	        'set'       - set SHARP daemons runtime parameters"
	echo "	        'install'   - install SHARP daemons as local service"
	echo "	        'uninstall' - uninstall SHARP daemons as local service"
	echo "	        'cleanup'   - cleanup all daemons logs and system files"
	echo "	        'logs'      - collect log from SHARP daemons"
	echo "	        'ver'       - show version of SHARP daemons"
	echo "  Parameters:"
	echo "	        -p - process name sharp_am"
	echo "	        -l - compute nodes host list"
	echo "	        -s - sharp_am server hostname"
	echo "	        -j - just dry run"
	echo "	        -w - write logs tarball file to dir. Relevant only with 'logs' option"
	echo "	        -m - monitoring of SHARP daemons. Relevant only with 'install' option"
	echo "	        -x - name of configuration parameter to be removed. Relevant only with 'set' option"
	echo "	        -h - display this help and exit"
	echo " Configuration:"
	echo "  There are two groups of parameters:"
	echo "       - daemon specific parameters. They go to configuration files."
	echo "         You can configure them directly in configuration files, located"
	echo "         in \"../conf\" folder or in \"\$SHARP_CONF/conf\" folder."
	echo "         Or you can set them using environment variables and the script"
	echo "         will store them into configuration files."
	echo "         Please refer to help message of  \"sharp_am\" for"
	echo "         for the list of parameters"
	echo "       - script's parameters. They control script execution"
	echo "  Naming convention: <sharp_am|sharp>_<parameter name>"
	echo "       \"sharp_am\" - sharp_am parameter"
	echo "       \"sharp\" - script parameter"
	echo "  sharp_am configuration:"
	echo "       If you use UFM 5.8 or later, please refer to UFM user guide."
	echo "       For Opensm 4.9 or later you don't need any special configuration."
	echo "       For Opensm 4.7-4.8 (Mellanox OFED 3.3-x.x.x - 4.0-x.x.x), you have to configure"
	echo "       root guids (tree like topologies) using \"fabric_load_file\" (\"sharp_am_fabric_load_file\" env. variable)."
	echo " Environment:"
	echo "  SHARP_INI_FILE - takes configuration from given file instead of $USER_INI_FILE"
	echo "  SHARP_CONF - destination folder for SHARP daemons configuration. It should be shared network location."
	echo " Examples:"
	echo "  $BASE_NAME start -l ajna0[1-8]   # run SHARP with defaults/recently used settings"
	echo "  SHARP_CONF=/hpc/local/work/user  $BASE_NAME start -l ajna0[1-8] # store generated configuration in non-standard place"
	echo "  SHARP_CONF=/hpc/local/work/user  sharp_am_root_guids_file=\"/var/log/opensm-root-guids.cfg\" $BASE_NAME start -l ajna0[1-8]  # run SHARP with old Opensm version"
	echo " Dependencies:"
	echo "  $(print_hostlist_dependency_message)"
	echo "  $(print_pdsh_dependency_message)"

	exit 2
}

conn_wrapper()
{
	local cmd="$1 $2"

	if [ -n "$dry_run" ]; then
		printf "Dry run mode: "
		echo \"$cmd\"
	else
		eval "$cmd"
		conn_ret=$?
	fi
}

ssh_wrapper()
{
	if [ x$(hostname -s) == x"$1" -o x$(hostname) == x"$1" ]; then
		conn_wrapper "sudo $2 $3 $4";
	else
		conn_wrapper "ssh -o StrictHostKeyChecking=No -n " "$1 sudo $2 $3 $4";
	fi
}

pdsh_wrapper()
{
	conn_wrapper "pdsh -w " "$1 sudo $2 $3 $4"
}

collect_logs()
{
	LOG_DIR=${write_log_to_dir:-$log_save_dir}/sharp_logs.$$
	TARBALL=${LOG_DIR}.tgz

	mkdir -p $LOG_DIR

	cp -fR $log_save_dir/sharp_benchmark_logs.* $LOG_DIR &> /dev/null
	cp -fR $DCONF_DIR $LOG_DIR &> /dev/null

	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		scp $sharp_AM_server:/var/log/sharp_am.log $LOG_DIR &> /dev/null
		scp $sharp_AM_server:$OUT_AM $LOG_DIR &> /dev/null
	fi

	if [ -z "$dry_run" ]; then
		ibnetdiscover &> $LOG_DIR/ibnetdiscover.out
		ibdiagnet --sharp &> $LOG_DIR/ibdiagnet.out
		cp -fR /var/tmp/ibdiagnet2 $LOG_DIR

		tar -C $LOG_DIR -czf $TARBALL .

		echo "Logs location is $HOSTNAME:$TARBALL"
	fi

	\rm -rf $LOG_DIR $log_save_dir/sharp_benchmark_logs.* &> /dev/null
}

all_run()
{
	AM_COMM="$1 --config-file $SHARP_AM_CONFIG_FILE"
	SD_COMM="$2 --config-file $SHARP_SD_CONFIG_FILE"

	local sharp_sleep_after_cmd=${sharp_sleep_after_cmd:-3}
	local ssh_ret="0"
	local pdsh_ret="0"

	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		ssh_wrapper "$sharp_AM_server" "SHARP_CONF=$SHARP_CONF $AM_COMM | sudo tee $OUT_AM" "&>/dev/null" && sleep $sharp_sleep_after_cmd
		ssh_ret=$conn_ret
	fi

	ret=0

	if (( $ssh_ret != 0 )); then
		ret=$ssh_ret
	elif (( $pdsh_ret != 0 )); then
		ret=$pdsh_ret
	fi
}

start()
{
	generate_conf_files $DCONF_DIR

	echo "SHARP config dir: $(readlink -f $DCONF_DIR)"
	echo "Startup SHARP daemons..."

	all_run	"$SHARP_INSTALL/etc/sharp_am start"
}

stop()
{
	echo "Stopping SHARP daemons..."
	all_run	"$SHARP_INSTALL/etc/sharp_am stop"

	[ -n "$ret" ] && [ $ret -ne 0 ] && echo "Some problem during daemons stop. Force killing of processes."

	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		ssh_wrapper "$sharp_AM_server" "pkill -9 sharp_am &>/dev/null; rm -f $OUT_AM /var/run/sharp_am.pid" "&>/dev/null"
	fi
}

check_on_boot()
{
	TMF_FILE=/tmp/procs.$$
	SHARP_INIT=${SHARP_INSTALL}/sbin/sharp.init

	echo -e "\nBoot state:"

	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		ssh_wrapper "$sharp_AM_server" "readlink -e /etc/init.d/sharp_am" ">$TMF_FILE" "2>/dev/null"
		_list=$((grep $SHARP_INIT $TMF_FILE | cut -d: -f1) 2>/dev/null)
		[ -n "$_list" ] && on_boot="sharp_am" || on_boot=""

		if [ -n "$on_boot" ]; then
			echo " sharp_am is \"service\" on     $sharp_AM_server"
		else
			echo " sharp_am is NOT \"service\" on $sharp_AM_server"
		fi
	fi

	\rm -f $TMP_FILE &>/dev/null
}

status()
{
	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		cnt1=$((ssh_wrapper "$sharp_AM_server" "$SHARP_INSTALL/etc/sharp_am status" | grep "is running" | wc -l) 2>/dev/null)
		cnt2=1

		if [ -n "$dry_run" ]; then
			ssh_wrapper "$sharp_AM_server" "$SHARP_INSTALL/etc/sharp_am status" "2>/dev/null"
			ret_am=0
		elif [ $cnt1 -eq $cnt2 ]; then
			echo "sharp_am is UP"
			ret_am=0
		else
			echo "sharp_am is DOWN."
			ret_am=1
		fi
	else
		ret_am=0
	fi


	[ -z "$dry_run" ] && check_on_boot

	if [ $# -eq 1 ] && [ "$1" = "stop" ]; then
		if [ $ret_am -eq 1 ] ; then
			ret=0
		else
			ret=1
		fi
	else
		if [ $ret_am -eq 1 ] ; then
			ret=1
		else
			ret=0
		fi
	fi
}

show_vers()
{
	TMF_FILE=/tmp/vers.$$

	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		ssh_wrapper "$sharp_AM_server" "$SHARP_INSTALL/bin/sharp_am -v 2>/dev/null | grep '(sharp)'" ">$TMF_FILE" "2>/dev/null"
		[ -z "$dry_run" ] && echo "   $(cat $TMF_FILE | tr -d '\r') - $sharp_AM_server"
			#			str=$(cat $TMF_FILE | tr -d '\r')
			#echo "   $str - $sharp_AM_server"
		#fi
	fi
}

install()
{
	TMP_FILE=/tmp/hosts.$$

	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		ssh_wrapper "$sharp_AM_server" "$SHARP_INSTALL/sbin/sharp_daemons_setup.sh -s -d sharp_am $monit" ">$TMP_FILE" "2>/dev/null"
		_list=$(grep "Service sharp_am is installed" $TMP_FILE | cut -d: -f1) 2>/dev/null

		if [ -n "$_list" ]; then
			echo "Service sharp_am is installed on $sharp_AM_server"
			ret_am=0
		else
			echo "Service sharp_am is missing on $sharp_AM_server"
			ret_am=1
		fi
	fi

	if [ $ret_am -eq 1 ] || [ $retd -eq 1 ]; then
		ret=1
	else
		ret=0
	fi

	\rm -f $TMP_FILE &>/dev/null
}

uninstall()
{
	TMP_FILE=/tmp/hosts.$$

	if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
		ssh_wrapper "$sharp_AM_server" "$SHARP_INSTALL/sbin/sharp_daemons_setup.sh -r -d sharp_am" ">$TMP_FILE" "2>/dev/null"
		_list=$(grep "Service sharp_am is removed" $TMP_FILE | cut -d: -f1) 2>/dev/null

		if [ -n "$_list" ]; then
			echo "Service sharp_am is removed on $sharp_AM_server"
			ret_am=0
		else
			echo "Service sharp_am can't be removed on $sharp_AM_server"
			ret_am=1
		fi
	fi

	if [ $ret_am -eq 1 ] || [ $retd -eq 1 ]; then
		ret=1
	else
		ret=0
	fi

	\rm -f $TMP_FILE &>/dev/null
}

daemon_clean()
{
	dname="$1"
	cmd="$2"

	toclean="/var/lock/subsys/$dname /var/run/$dname.pid /var/log/$dname.log"

	for one_file in $toclean; do
		if [ -z "$process" ] || [ "$process" == "$dname" ]; then
			$cmd "\rm -f $one_file" "&>/dev/null"
		fi
	done
}

cleanup()
{
	daemon_clean "sharp_am" "ssh_wrapper \"$sharp_AM_server\""
}

#
# Main
#
OUT_AM=/tmp/d_sharp_am.log

[ $# -eq 0 ] && usage
if [ $# -ge 1 ]; then
	operation=$1
	shift
fi
[ "$operation" = "-h" ] && usage

#
# Check dependency
#

check_hostlist_tool

if ! check_pdsh_tool ; then
	echo "'pdsh' not found. Dry mode will be invoked."
	dry_run=1
fi

#
# Load paramters
#

load_params

while getopts "p:l:s:w:x:jmhb" flag_arg; do
	case $flag_arg in
		p) process="$OPTARG"                       ;;
		l) sharp_hostlist="$OPTARG"        ;;
		s) sharp_AM_server="$OPTARG"       ;;
		w) write_log_to_dir="$OPTARG"              ;;
		j) dry_run=1                               ;;
		m) monit='-m'                              ;;
		b) socket_based="-b"			   ;;
		x) var2remove="$OPTARG"                    ;;
		h) usage                                   ;;
		*) usage                                   ;;
	esac
done

#
# Check params
#

if [ -z "$sharp_AM_server" ]; then
	if pgrep opensm &> /dev/null; then
		sharp_AM_server=$(hostname -s)
	else
		echo "Please provide AM server. Exit."
		exit 1
	fi
fi

check_and_update_hostlist sharp_hostlist

if [ -z "$process" ] || [ "$process" == "sharp_am" ]; then
	echo -e "sharp_am host: $sharp_AM_server"
fi
echo ""


mkdir -p $DCONF_DIR &> /dev/null
if [ $? -ne 0 ]; then
	echo "Unable to create $DCONF_DIR dir. Exit."
	exit 1
fi


[ -n "$var2remove" ] && eval $var2remove=""
set_local_params "sharp_am"
set_local_params "sharp"

case $operation in
	"start")		start
					if [ -z "$dry_run" ]; then
						sleep 2
						status
					fi
					;;
	"restart")		stop
					start
					if [ -z "$dry_run" ]; then
						sleep 2
						status
					fi
					;;
	"stop")			stop
					status "stop"
					;;
	"status")		status
					;;
	"set")			if [ -n "$dry_run" ]; then
						echo "Nothing to do in dry run mode."
					else
						generate_conf_files $DCONF_DIR
					fi
					;;
	"install")		install
					;;
	"uninstall")	uninstall
					;;
	"cleanup")		cleanup
					;;
	"logs")			collect_logs
					;;
	"ver")			show_vers
					;;
	*)				usage
					;;
esac

echo -e "\nAll Done"
[ -n "$ret" ] && exit $ret
