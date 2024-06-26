#!/bin/bash
#
# Copyright (c) 2016-2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

REAL_USER=${SUDO_USER:-$USER}
BASE_NAME=${INI_NAME:-$(basename $0 .sh)}
USER_INI_FILE=${SHARP_INI_FILE:-$(eval echo "~${REAL_USER}/.${BASE_NAME}.ini")}

#
# Validate IB device name.
# It should be in format: <dev_name>:<port num>
# For example: "mlx5_0:1"
#
check_ib_dev_name()
{
	local ib_dev=$1

	if [[ $ib_dev =~ mlx[0-9]+_[0-9]+:[0-9]+ ]]; then
		return 0
	else
		echo "Invalid IB interface \"$ib_dev\". It should be in format: <dev name>:<port num>. For example: \"mlx5_0:1\""
		return 1
	fi
}

set_local_params()
{
	local _prefix=$1
	local _tmp_file=/tmp/params.$$
	local var
	local val

	gen_conf_file_header /tmp/params.$$

	for x in $(set | grep ^$_prefix); do
		var=$(echo "$x" | cut -f1 -d'=')
		val="${!var}"
		[ -n "$val" ] && echo "$var=\"$val\"" >> $_tmp_file
	done

	if [[ -f $USER_INI_FILE ]]; then
		grep -v ^$_prefix $USER_INI_FILE | grep -v ^# >> $_tmp_file ||:
	fi

	if [ "$USER" = "$REAL_USER" ]; then
		mv -f $_tmp_file $USER_INI_FILE
	else
		if ! su $REAL_USER -c "\cp -f $_tmp_file $USER_INI_FILE 2>/dev/null"; then
			\cp -f $_tmp_file $USER_INI_FILE
			chown $REAL_USER $USER_INI_FILE
		fi
		\rm -f $_tmp_file &> /dev/null
	fi
}

gen_conf_file_header()
{
	_gen_tmp_file=$1

	cat > $_gen_tmp_file << EOF
#
# Copyright (C) $(date | awk '{print $6}'). NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Generated by $(basename $0) on host $(hostname)
# $(date)
# from SHARP ver.
EOF
	sed 's/^/#    /' $SHARP_INSTALL/share/doc/sharp/SHARP_VERSION >> $_gen_tmp_file
	echo "#" >> $_gen_tmp_file
}

read_ini_params()
{
	_file=$1

	[ -n "$_file" ] && [ ! -f "$_file" ] && return 0

	for x in $(grep -v '^#' $_file); do
		var=$(echo "$x" | cut -f1 -d'=')
		val=$(echo "$x" | cut -f2 -d'=')
		[ -n "$val" ] && eval ${var}=$val
	done

	return 0
}

get_slurm_nodes()
{
	local result

	if [ -z "$SLURM_NODELIST" ] && [ $(squeue 2>/dev/null | grep $REAL_USER | wc -l) -eq 1 ]; then
		SLURM_NODELIST=$(squeue 2>/dev/null | grep $REAL_USER | awk '{print $NF}')
	fi

	[ -n "$SLURM_NODELIST" ] && result=$SLURM_NODELIST || eval result=""

	echo $result
}

check_and_update_hostlist()
{
	local _var=$1

	if [ -z "${!_var}" ]; then
		echo "List of compute nodes is empty . Try to use SLURM allocation"
		eval $_var=$(get_slurm_nodes)

		if [ -z "${!_var}" ]; then
			echo "Set $_var. Exit"
			exit 1
		fi
	fi
}

read_cfg_file()
{
	_cfg_file=$1
	_suffix=$(basename $_cfg_file .cfg)

	while read param val; do
		if ! echo $param | grep '^#' &>/dev/null; then
			var=${_suffix}_${param}
			(grep -v '^#' $USER_INI_FILE | cut -d= -f1 | grep $var) &>/dev/null && eval $var=$val
		fi
	done < $_cfg_file

	return 0
}

read_cfg_params()
{
	[ -s $1 ] && read_cfg_file $1

	return 0
}

load_params()
{
	read_ini_params $USER_INI_FILE

	if [ $# -eq 1 ]; then
		SHARP_CONF=$1
	else
		SHARP_CONF=${SHARP_CONF:-$sharp_manager_general_conf}
		SHARP_CONF=${SHARP_CONF:-$SHARP_INSTALL}
		sharp_manager_general_conf=$SHARP_CONF
	fi
	DCONF_DIR=$SHARP_CONF/conf

	SHARP_AM_CONFIG_FILE=$DCONF_DIR/sharp_am.cfg
	SHARP_SD_CONFIG_FILE=$DCONF_DIR/sharpd.cfg

	read_cfg_params $SHARP_AM_CONFIG_FILE
	read_cfg_params $SHARP_SD_CONFIG_FILE

	reestablish_env_vars "sharpd"
	reestablish_env_vars "sharp_am"
	reestablish_env_vars "sharp"
}

merge_conf_files()
{
	_from_file=$1
	_to_file=$2
	_tmp_file=/tmp/merge.$$

	if [ -n "$var2remove" ]; then
		del_var=$(echo $var2remove | sed "s/$(basename $_to_file .cfg)_//g")
		sed -i "/$del_var/d" $_to_file
	fi

	while read param val; do
		if grep -qw $param $_to_file &>/dev/null; then
			sed -i "/$param/d" $_to_file
		fi
	done < $_from_file

	cp -f $_from_file $_tmp_file
	[ -s $_to_file ] && cat $_to_file | grep -v ^# >> $_tmp_file

	gen_conf_file_header $_to_file
	cat $_tmp_file >> $_to_file
	\rm -f $_tmp_file &>/dev/null
}

make_conf()
{
	_tmp_cfg=$1
	_name=$2
	_cfg_dir=$3
	_cfg_file=$_cfg_dir/${_name}.cfg

	if ! diff -I '#.*' $_tmp_cfg $_cfg_file &> /dev/null; then
		if [ -s $_cfg_file ]; then
			save_to=$_cfg_dir/${_name}_${_time}.cfg
			echo "Configuration file $_cfg_file will be saved to $save_to"
			[ -w "$_cfg_dir" ] && cp -f $_cfg_file $save_to
		fi

		merge_conf_files $_tmp_cfg $_cfg_file
	fi
}

var4conf_file()
{
	_prefix=$1
	_file=$2

	\rm -f $_file &>/dev/null
	touch $_file

	for x in $(set | grep ^${_prefix}); do
		var=$(echo "$x" | cut -f1 -d'=')
		val="${!var}"
		name=$(echo $var | sed "s/${_prefix}_//g")

		[ -n "$val" ] && echo $name $val >> $_file
	done
}

generate_conf_files()
{
	_cfg_dir=$1

	if [ ! -w $_cfg_dir ]; then
		echo "Unable to write to $_cfg_dir. Exit"
		exit 1
	fi

	TMP_AM=/tmp/sharp_am_$$.cfg
	TMP_SD=/tmp/sharpd_$$.cfg

	var4conf_file "sharp_am" $TMP_AM
	var4conf_file "sharpd"   $TMP_SD

	_time=$(date +%d%m%y_%H:%M)
	make_conf $TMP_AM "sharp_am" $_cfg_dir
	make_conf $TMP_SD "sharpd" $_cfg_dir

	\rm -f $TMP_AM $TMP_SD &> /dev/null
}

store_env_vars()
{
	_prefix=$1
	_save_pref="stored"

	for x in $(set | grep ^$_prefix); do
		var=$(echo "$x" | cut -f1 -d'=')
		[ -n "${!var}" ] && eval ${_save_pref}_${var}=${!var}
	done

	return 0
}

reestablish_env_vars()
{
	_prefix=$1

	for x in $(set | grep ^$_prefix); do
		var=$(echo "$x" | cut -f1 -d'=')
		saved_name=${_save_pref}_${var}
		[ -n "${!saved_name}" ] && eval ${var}=${!saved_name}
	done

	return 0
}

print_hostlist_dependency_message()
{
	echo "This script uses \"python-hostlist\" package. Visit https://www.nsc.liu.se/~kent/python-hostlist/ for details"
}

check_hostlist_tool()
{
	if ! which hostlist &>/dev/null; then
		echo "Unable to find 'hostlist' tool"
		print_hostlist_dependency_message
		echo "Exiting ..."
		exit 1
	else
		hostlist_tool=hostlist
	fi
}

print_pdsh_dependency_message()
{
	echo "This script uses \"pdsh\" tool. Visit https://linux.die.net/man/1/pdsh for details"
}

check_pdsh_tool()
{
	if [ -z "$(which pdsh 2>/dev/null)" ]; then
		print_pdsh_dependency_message
		return 1
	fi

	return 0
}

#
# Get the linux distro name
# Note: This function should be called with $(get_linux_distro), since it is echoing the distro name
# Also, notice this function is set with () instead of {}, since it has an internal function
#

get_linux_distro()
(
	get_linux_distro_from_etc()
	{
		if [ -f /etc/os-release ]; then
			# freedesktop.org and systemd
			. /etc/os-release
			distro=$NAME
		elif type lsb_release >/dev/null 2>&1; then
			# linuxbase.org
			distro=$(lsb_release -si)
		elif [ -f /etc/lsb-release ]; then
			# For some versions of Debian/Ubuntu without lsb_release command
			. /etc/lsb-release
			distro=$DISTRIB_ID
		elif [ -f /etc/debian_version ]; then
			# Older Debian/Ubuntu/etc.
			distro=Ubuntu
		elif [ -f /etc/SuSe-release ]; then
			# Older SuSE/etc.
			distro=Suse
		elif [ -f /etc/redhat-release ]; then
			# Older Red Hat, CentOS, etc.
			distro=RedHat
		else
			# Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
			distro=$(uname -s)
		fi

		# Make sure we return a lower case distro (so we dont have errors when matching)
		distro=$(echo $distro | tr '[:upper:]' '[:lower:]')
	}

	# First check for systemd support
	init_process=$(ps --no-headers -o comm 1)
	if [[ "$init_process" == "systemd" ]]; then
		distro="systemd"
	else
		get_linux_distro_from_etc
	fi

	echo $distro;
)


log_save_dir=${log_save_dir:-/var/tmp}


SHARP_INSTALL=${SHARP_INSTALL:-$(readlink -f $(dirname $0)/..)}
if [ ! -d "$SHARP_INSTALL" ]; then
    echo "Unable to find SHARP dir. Exit."
    exit 1
fi

store_env_vars "sharp_am"
store_env_vars "sharpd"
store_env_vars "sharp"
