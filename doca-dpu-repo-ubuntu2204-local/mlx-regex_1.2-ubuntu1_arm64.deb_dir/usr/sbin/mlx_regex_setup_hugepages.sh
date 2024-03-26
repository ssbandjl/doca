#!/bin/bash -e
# Dependency on package:
#   libhugetlbfs-utils @ CentOS
#   hugepages @ Ubuntu

# Amount of hugepage memory needed by mlx-regex daemon
min_hugemem=${MIN_HUGEMEM:-500M}

# Units of memory for mlx-regex daemon
case $(echo ${min_hugemem: -1}) in
    M)
        unit=m
        ;;
    G)
        unit=g
        ;;
    K)
        unit=k
        ;;
    *)
        echo "[ERROR]: Unsupported unit format for hugepages!"
        exit 1
        ;;
esac

# Check if there is a mount point for 2M hugepages, create if it does not exist.
list_all_mounts=$(/usr/bin/hugeadm --list-all-mounts)
if [[ $list_all_mounts != *"pagesize=2M"* ]]; then
    /usr/bin/hugeadm --create-mounts
    # Check if mount point created successfully.
    list_all_mounts=$(/usr/bin/hugeadm --list-all-mounts)
    if [[ $list_all_mounts != *"pagesize=2M"* ]]; then
        echo "[ERROR]: Unable to create mount point for 2M hugepages!"
        exit 1
    fi
fi

# Have any 2M hugepages been configured yet
num_2m_hugepages=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)
num_2m_hugepages_free=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages)

# Check if existing 2M hugepages can be used or if pool needs adjusted
if [ $num_2m_hugepages -gt 0 ]; then
    if [ $unit = "k" ]; then
        required_size=${min_hugemem%?}
    elif [ $unit = "m" ]; then
	required_size=$((${min_hugemem%?} * 1024))
    elif [ $unit = "g" ]; then
	required_size=$((${min_hugemem%?} * 1024 * 1024))
    fi

    huge_free_size=$((2048 * num_2m_hugepages_free))

    if [ $huge_free_size -ge $required_size ]; then
        exit 0
    fi
fi

# Adjust the 2M pool size
exec /usr/bin/hugeadm --pool-pages-min 2M:+${min_hugemem}
