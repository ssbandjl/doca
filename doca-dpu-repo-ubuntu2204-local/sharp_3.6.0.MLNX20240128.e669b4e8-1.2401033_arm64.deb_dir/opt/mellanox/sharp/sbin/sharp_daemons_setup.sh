#!/bin/bash
#
# Copyright (c) 2016-2018 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

BASE_NAME=$(basename $0)
BASE_DIR=$(dirname $0)
SYSTEM_UNIT_FILES_DIR=${SYSTEM_UNIT_FILES_DIR:-"/etc/systemd/system"}

. $BASE_DIR/sharp_funcs.sh


# Get the linux disto name (it returns systemd in case systemd is supported)
distro_name=$(get_linux_distro)

start_serv()
{
	local serv=$1

	case $distro_name in
	'systemd')
		#if this is not socket based activation, start the service.
		if [ -z ${socket_based} ]; then
			systemctl --quiet start $serv > /dev/null
		#if it is socket based activation, just start the socket
		else
			systemctl --quiet start ${serv}.socket
		fi
	;;
	*)         service $serv start  &> /dev/null      ;;
	esac
}

#This function creates a conf file that overrides settings in
#the original service file. Multiple settings can be send to the function
#in the following format:
#update_unit Section key value key value ....
function update_unit()
{
	local serv=$1
	local section=$2
	shift
	shift
	local drop_folder=${SYSTEM_UNIT_FILES_DIR}/${serv}.service.d
	local drop_file=${drop_folder}/${section}.conf


	mkdir -p ${drop_folder}
	touch ${drop_file}
	#Create section title
	cat > ${drop_file} << EOF
[$section]
EOF
	#Loop over Key Value arguments
	while [ $# -gt 0 ]; do
		key=$1
		shift
		val=$1
		shift
		cat >> ${drop_file} << EOF
$key=$val
EOF
	done

	systemctl daemon-reload > /dev/null
}

add_serv_systemd()
{
	local serv=$1
	local location_dir=$2

	local unit_base_name=${serv}.service
	local unit_source_path=${location_dir}/systemd/system/${unit_base_name}
	local unit_path=${SYSTEM_UNIT_FILES_DIR}/${unit_base_name}

	if ! [[ -f ${unit_source_path} ]]; then
		echo "Error: $unit_source_path doesn't exist."
		echo "Cannot install ${serv} as systemd service. Exit."
		exit 3
	fi

	if [ ! -z ${socket_based} ]; then
		local socket_base_name=${serv}.socket
		local socket_source_path=${location_dir}/systemd/system/${socket_base_name}
		local socket_path=${SYSTEM_UNIT_FILES_DIR}/${socket_base_name}

		if ! [[ -f ${socket_source_path} ]]; then
			echo "Error: $socket_source_path doesn't exist."
			echo "Cannot install ${serv} with socket based activation support. Exit."
			exit 3
		fi
	fi
	local conf_dir="${SYSTEM_UNIT_FILES_DIR}/"$unit_base_name".d"

	# Copy service/socket files to systemd files directory

	echo "Copying ${unit_source_path} to ${unit_path}"
	\cp -rf ${unit_source_path} ${unit_path}
	if ! [[ -f ${unit_path} ]]; then
		echo "Error: $unit_path doesn't exist."
		echo "Cannot install ${serv} as systemd service. Exit."
		exit 3
	fi

	if [ ! -z ${socket_based} ]; then
		\cp -rf ${socket_source_path} ${socket_path}
		echo "Copying ${socket_source_path} to ${socket_path}"
		if ! [[ -f ${socket_path} ]]; then
			echo "Error: $socket_path doesn't exist."
			echo "Cannot install ${serv} with socket based activation support. Exit."
			exit 3
		fi
	fi

	# Enable service/socket

	if [[ ! -z ${socket_based} ]]; then
		systemctl --quiet enable ${socket_path} > /dev/null 2>&1
	else
		systemctl --quiet enable ${unit_path} > /dev/null 2>&1
	fi

	if [[ ${serv} = "sharp_am" ]]; then
		update_unit $serv Service Environment "CONF=-${location_dir}/conf/${serv}.cfg"  ExecStart "" ExecStart "${location_dir}/bin/${serv} -O \$CONF"
	fi
}

add_serv()
{
	local serv=$1
	local location_dir=$2
	local init=$3

	case $distro_name in
	'systemd')
		add_serv_systemd ${serv} ${location_dir}
		;;
	'ubuntu')
		chmod 755 $init
		ln -sf $init /etc/init.d/$serv &>/dev/null
		update-rc.d $serv defaults &> /dev/null
		;;
	*)
		chmod 755 $init
		ln -sf $init /etc/init.d/$serv &>/dev/null
		chkconfig --add $serv &> /dev/null
		;;
	esac
}

rm_serv()
{
	serv=$1

	case $distro_name in
	'systemd')
		local unit_base_name=${serv}.service
                local conf_dir="${SYSTEM_UNIT_FILES_DIR}/"$unit_base_name".d"

		systemctl disable ${serv}.service &> /dev/null
		#if the service was setup with socket based activation, disable the socket as well.
		#the socket needs to be stopped first
		systemctl stop ${serv}.socket &> /dev/null
		systemctl disable ${serv}.socket &> /dev/null

		[[ -f "${SYSTEM_UNIT_FILES_DIR}/${unit_base_name}" ]] && \rm -f ${SYSTEM_UNIT_FILES_DIR}/${unit_base_name} &> /dev/null

		#if the service.d directory is empty, remove it.
		if [[ -d ${conf_dir} ]]; then
			if ! find "$conf_dir" -mindepth 1 -print -quit | grep -q .; then
				\rm -df $conf_dir &> /dev/null
				echo "removed $conf_dir"
			fi
		fi

		systemctl daemon-reload
	 ;;
	'ubuntu') (ls /etc/rc*.d | grep -q $serv) &> /dev/null && update-rc.d -f $serv remove &> /dev/null ;;
	*)        (chkconfig --list | grep $serv) &> /dev/null && chkconfig --del $serv &> /dev/null       ;;
	esac
}

unset_level()
{
	serv=$1

	case $distro_name in
	'systemd') ;;
	'ubuntu') update-rc.d $serv disable 0123456 &> /dev/null   ;;
	*)        chkconfig --level 0123456 $serv off &> /dev/null ;;
	esac
}

is_added()
{
	serv=$1
	ok=1

	case $distro_name in
	'systemd') systemctl list-unit-files --type=service | grep -q $serv && ok=0 ;;
	'ubuntu') ls /etc/rc*.d | grep -q $serv && ok=0                ;;
	*)        (chkconfig --list | grep $serv) &> /dev/null && ok=0 ;;
	esac

	if [ $ok -eq 0 ]; then
		echo "Service $serv is installed"
		return 0
	else
		echo "Error: failed to install service $serv"
		return 1
	fi
}

is_removed()
{
	serv=$1
	ok=1

	case $distro_name in
		'systemd') ! (systemctl list-unit-files --type=service | grep -q $serv) && ok=0 ;;
		'ubuntu') ! (ls /etc/rc*.d | grep -q $serv) && ok=0              ;;
	*)        ! (chkconfig --list | grep $serv) &> /dev/null && ok=0 ;;
	esac

	if [ $ok -eq 0 ]; then
		echo "Service $serv is removed"
		return 0
	else
		echo "Error: failed to remove service $serv"
		return 1
	fi
}

# $1 - service name
monit_add()
{
	SERVICE=$1
	MONIT_DIR=/etc/monit.d
	MONIT_FILE=$MONIT_DIR/${SERVICE}.conf
	TMP_F=/tmp/${SERVICE}_$$.conf

	if [ ! -d $MONIT_DIR ]; then
		echo "Warning: Seems like monit package is not installed. ${SERVICE} will not be monitored."
		return 1
	fi

	cat > $TMP_F << EOF
check process $SERVICE with pidfile /var/run/${SERVICE}.pid
	start program = "/etc/init.d/$SERVICE start"
	stop program = "/etc/init.d/$SERVICE stop"
	if 50 restarts within 50 cycles then timeout
EOF

	cp $TMP_F $MONIT_FILE &> /dev/null
	pkill -1 monit &> /dev/null
}

# $1 - service name
monit_remove()
{
	SERVICE=$1
	MONIT_DIR=/etc/monit.d
	MONIT_FILE=$MONIT_DIR/${SERVICE}.conf

	[ ! -d $MONIT_DIR ] && return 1

	rm -f $MONIT_FILE &> /dev/null
	pkill -1 monit &> /dev/null
}

# $1 - SHARP location dir
# $2 - list of daemons
# $3 - monit flag
setup()
{
	local location_dir=$1
	local init=${location_dir}/sbin/sharp.init

	if [ ! -f "$init" ]; then
		echo "Error: $init doesn't exist. Exit."
		exit 3
	fi

	for daemon in $2; do
	add_serv $daemon ${location_dir} ${init}

	if [[ ${daemon} = "sharp_am" ]]; then
		unset_level sharp_am
	else
		start_serv ${daemon}
	fi

	is_added $daemon
	#monit is deprecated with systemd
	[ $? -eq 0 ] && [ -n "$3" ] && [ -z "$socket_based" ] && monit_add $daemon
	done
}

# $1 - list of daemons
unsetup()
{
	for daemon in $1; do
	[ -x /etc/init.d/$daemon ] && /etc/init.d/$daemon stop
	pkill $daemon &>/dev/null
	rm -f /tmp/d_${daemon}.log /var/run/{$daemon}.pid &> /dev/null

	rm -f /etc/init.d/$daemon &> /dev/null
	rm_serv $daemon

	is_removed $daemon

	[ $? -eq 0 ] && monit_remove $daemon
	done
}

usage()
{
	echo "Usage: `basename $0` (-s | -r) [-p SHARP location dir] -d sharp_am [-m]"
	echo "$BASE_NAME helps to system administrator to manage SHARP daemons."
	echo "All management operations require root access."
	echo ""
	echo "	-s - Setup SHARP daemon"
	echo "	-r - Remove SHARP daemon"
	echo "	-p - Path to alternative SHARP location dir"
	echo "	-d - Daemon name (sharp_am)"
	echo "	-m - Add monit capability for daemon control"
	echo "	-h - Print this help and exit"
	echo ""
	echo "This script uses systemd init system in modern distros like RHEL 7.X . Use systemctl for manage SHARP daemons in such systems."
	echo "In older releases (RHEL 6.x) Upstart is used. Use chkconfig for management."
	echo ""
	echo "Notes:"
	echo "sharp_am.service is NOT started automatically and does NOT support by default auto-restart. Starting sharp_am.service"
	echo "resets SHARP trees and cleans allocated resources. It could affect running applications."
	echo "sharp_am.service have to run on the same IB port used by opensm and SHARP capabilities should be enabled."
	echo ""
	echo "Examples:"
	echo "  RHEL 7.0 and later:"
	echo "	systemctl status sharp_am.service"

	exit 2
}

dlist=""
location_dir=""
to_setup=""
to_remove=""
to_monit=""
while getopts "sp:rd:mhb" flag_arg; do
	case $flag_arg in
	s) to_setup="yes"         ;;
	r) to_remove="yes"        ;;
	p) location_dir="$OPTARG" ;;
	d) dlist="$OPTARG"        ;;
	m) to_monit="yes"         ;;
	b) socket_based="yes"	  ;;
	p) usage                  ;;
	*) usage                  ;;
	esac
done

if [ $(whoami) != "root" ]; then
	echo "Error: root permission required. Exit."
	echo ""
	usage
	exit 1
fi

([ $OPTIND -eq 1 ] || [ -z "$dlist" ]) && usage

[[ -z ${to_setup} && -z ${to_remove} ]] && usage 
[[ -n ${to_setup} && -n ${to_remove} ]] && usage 

[ -z "$location_dir" ] && _dir=$(readlink -f $(dirname $0)/..)

for daemon in $dlist; do
	([ $daemon != "sharp_am" ]) && usage

	[ -z "$location_dir" ] && [ -f "$_dir/bin/$daemon" ] && location_dir=$_dir
done

[ -n "$to_setup" ] && setup "$location_dir" "$dlist" "$to_monit"
[ -n "$to_remove" ] && unsetup "$dlist"

exit 0
