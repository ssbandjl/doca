#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
import json
import glob
import re
from time import sleep

NVME_CLI = 'nvme'
NVME_SUBSYS_CONFIGFS = '/sys/class/nvme-subsystem/'
SPDK_RPC = 'spdk_rpc.py'


def run_command(command):
    '''
    Run shell command
    @param command: command to run
    @return: result
    '''
    cmd_list = command.split()
    result = subprocess.run(cmd_list, shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return result


def load_module(module):
    '''
    Load kernel module
    @param module: module name to load
    '''
    cmd = f"modprobe {module}"
    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stderr.decode('utf-8'))


def load_nvme_transport_module(transport):
    '''
    Load kernel driver based on transport
    type
    @param transport: transport type
    '''
    if transport == 'rdma':
        load_module('nvme-rdma')
    elif transport == 'tcp':
        load_module('nvme-tcp')


def set_nvme_io_policy(policy, subsys_nqn):
    '''
    Set nvme I/O policy
    @param policy: policy to set
    @param subsys_nqn: subsystem nqn name
    '''
    subsystems = glob.glob(NVME_SUBSYS_CONFIGFS + '*/')
    for subsystem in subsystems:
        with open(subsystem + 'subsysnqn') as s:
            if s.read().splitlines()[0] == subsys_nqn:
                io_policy_path = '{}iopolicy'.format(subsystem)
                if os.path.exists(io_policy_path):
                    with open(io_policy_path, 'w') as p:
                        p.write(policy)
                break


def convert_path_to_dict(path, parser):
    '''
    Convert connection path to dictionary
    @param path: path to convert
    @return: dict with parsed path
    '''
    result = dict()

    try:
        for item in (path.split('/')):
            k, v = item.split('=')
            result[k] = v
    except ValueError:
        parser.error('wrong PATH format! ({})'.format(path))

    return result


def parse_paths(parser, paths):
    '''
    Parse paths to remote target
    @param parser: parser object
    @param paths: list of paths
    @return: list with formatted paths
    '''
    formatted_paths = []

    path_keys = ['transport', 'adrfam',
                 'traddr', 'trsvcid']

    for path in paths:
        formatted_path = convert_path_to_dict(path, parser)
        for path_key in path_keys:
            if path_key not in formatted_path:
                parser.error('wrong PATH format! {} is missing)'.format(path_key))
        if formatted_path['transport'] not in ['rdma', 'tcp']:
            parser.error('wrong PATH format! (transport should be one of rdma/tcp)')
        formatted_paths.append(formatted_path)

    return formatted_paths


def nvme_list():
    '''
    List nvme devices exposed on host
    @return: devices list
    '''
    cmd = NVME_CLI + ' list -o json'
    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stderr.decode('utf-8'))

    if result.stdout:
        result_json = json.loads(result.stdout)

        devices = [
                {
                    'type': 'kernel',
                    'dev': dev["DevicePath"],
                    'block_size': dev["SectorSize"]
                }
                for dev in result_json["Devices"]
        ]

        return devices

    return []


def nvme_list_formatted(nvme_list):
    '''
    Format nvme list
    @return: list of formatted
    nvme devices
    '''
    return ','.join(
        [
            ':'.join(
                    [f"{key}={value}" for key, value in dev.items()]
                    ) for dev in nvme_list
            ]
            )


def connect_nvme(transport, nqn, traddr, trsvcid, hostnqn):
    '''
    Connect nvme device from remote target
    @param transport: transport type
    @param nqn: nqn name
    @param traddr: transport address
    @param trsvcid: transport service
    @param hostnqn: user-defined hostnqn
    '''
    cmd = NVME_CLI + f" connect --transport={transport}" + \
                     f" --traddr={traddr}" + \
                     f" --trsvcid={trsvcid}" + \
                     f" --nqn={nqn} --hostnqn={hostnqn}"

    result = run_command(cmd)

    rc = result.returncode
    if rc:
        out = result.stderr.decode('utf-8')
    else:
        out = result.stdout.decode('utf-8')

    return rc, out


def discover_nvme(transport, traddr, trsvcid, hostnqn=None):
    '''
    Run nvme discover command
    @param transport: transport type
    @param traddr: transport address
    @param trsvcid: transport service id
    @param hostnqn: user-defined hostnqn
    @return: raw output command result
    '''
    cmd = NVME_CLI + f" discover --transport={transport}" + \
                     f" --traddr={traddr} --trsvcid={trsvcid}" + \
                     f" --raw=/dev/stdout"

    if hostnqn:
        cmd += f" --hostnqn={hostnqn}"

    result = run_command(cmd)
    if result.returncode:
        sys.exit(result.stderr.decode('utf-8'))

    output_tail = "Discovery log is saved to /dev/stdout"
    print(result.stdout[:-(len(output_tail) + 1)].hex())


def disconnect_nvme_subsys(nqn):
    '''
    Disconnect from NVMeoF subsystem
    @param nqn: nqn name
    '''
    cmd = NVME_CLI + f" disconnect --nqn {nqn}"

    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stderr.decode('utf-8'))

    print(result.stdout.decode('utf-8'))


def nvme_list_subsys(dev=None):
    '''
    List nvme subsystems
    @param dev: device
    @return: json formatted
    nvme subsystems
    '''
    cmd = NVME_CLI + ' list-subsys -o json'

    if dev:
        cmd += f" {dev}"

    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stderr.decode('utf-8'))

    if result.stdout:
        ret = json.loads(result.stdout.decode('utf-8'))
        if dev:
            # nvme-cli <= 1.8.1 bug: Paths added as new subsystem
            # https://github.com/linux-nvme/nvme-cli/pull/504
            if len(ret['Subsystems']) > 1 and 'Paths' in ret['Subsystems'][1]:
                paths = ret['Subsystems'][1]['Paths']
                del ret['Subsystems'][1]
                ret['Subsystems'][0]['Paths'] = paths
        return ret

    return {}


def nvme_get_bdev_info_formatted(bdev):
    '''
    Get nvme device information in
    formatted output
    @param bdev: device
    @return: formatted list with
    device's paths
    '''

    paths = nvme_list_subsys(bdev)['Subsystems'][0]['Paths']
    result = ','.join(
        [
            'transport={} adrfam=ipv4 {}'.format(path['Transport'], path['Address']).replace(' ', '/')
            for path in paths
        ]
    )

    return result


def spdk_nvme_get_controllers():
    '''
    SPDK NVMe controllers list
    @return: json object with
    spdk nvme controllers
    '''
    cmd = SPDK_RPC + ' bdev_nvme_get_controllers'

    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stdout.decode('utf-8'))

    return json.loads(result.stdout.decode('utf-8'))


def spdk_nvme_attach_controller(name, trtype, traddr,
                                adrfam, trsvcid, subnqn,
                                hostnqn=None, multipath=None,
                                ctrlr_loss_timeout_sec=None,
                                reconnect_delay_sec=None):
    '''
    SPDK attach nvme controller
    @param name: name of NVMe controller (prefix)
    @param trtype: NVMe-oF target trtype
    @param adrfam: NVMe-oF target adrfam
    @param trsvcid: NVMe-oF target trsvcid
    @param subnqn: NVMe-oF target subnqn
    @param hostnqn: NVMe-oF host subnqn

    @param multipath: The behavior when multiple
    paths are created ("disable","failover", or
    "multipath";failover if not specified)

    @param ctrlr_loss_timeout_sec: Time to wait until
    ctrlr is reconnected before deleting ctrlr.

    @param reconnect_delay_sec: Time to delay a reconnect
    trial. (optional)
    '''
    cmd = SPDK_RPC + f" bdev_nvme_attach_controller --name {name}" + \
                     f" --trtype {trtype} --traddr {traddr}" + \
                     f" --adrfam {adrfam} --trsvcid {trsvcid}" + \
                     f" --subnqn {subnqn}"

    if hostnqn:
        cmd += f" --hostnqn {hostnqn}"
    if multipath:
        cmd += f" --multipath {multipath}"
    if ctrlr_loss_timeout_sec:
        cmd += f" --ctrlr-loss-timeout-sec {ctrlr_loss_timeout_sec}"
    if reconnect_delay_sec:
        cmd += f" --reconnect-delay-sec {reconnect_delay_sec}"

    result = run_command(cmd)

    if result.returncode:
        rc = result.returncode
        out = result.stdout.decode('utf-8').replace('\n', ' ')
        return rc, out

    spdk_nvme_bdevs = result.stdout.decode('utf-8').split()

    spdk_nvme_bdevs_detail = [
        spdk_get_bdevs(bdev)[0] for bdev in spdk_nvme_bdevs
    ]

    form_templ = 'dev={}/block_size={}'
    form_list = []

    for bdev in spdk_nvme_bdevs_detail:
        name = bdev['name']
        bs = bdev['block_size']
        form_list.append(form_templ.format(name, bs))

    out = ','.join(form_list)
    rc = result.returncode

    return rc, out


def spdk_nvme_get_controller_name():
    '''
    Calculate next available NVMe
    controller name to be created
    @return: NVMe controller name
    string
    '''
    spdk_nvme_controllers = [
        nvme_controller["name"] for nvme_controller in spdk_nvme_get_controllers()
        ]

    nvme_pattern = re.compile("^Nvme(\d+)$")
    nvme_indexes = [
        int(nvme_pattern.search(dev).group(1)) for dev in spdk_nvme_controllers if nvme_pattern.match(dev)
        ]

    if len(nvme_indexes) > 0:
        nvme_index = sorted(nvme_indexes)[-1] + 1
        return f"Nvme{nvme_index}"

    return "Nvme0"


def spdk_bdev_nvme_detach_controller(ctrl):
    '''
    Detach an NVMe controller and
    delete any associated bdevs
    @param ctrl: Name of the controller
    '''
    cmd = SPDK_RPC + ' bdev_nvme_detach_controller {}'.format(ctrl)

    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stdout.decode('utf-8'))


def spdk_bdev_uring_create(filename, name, block_size):
    '''
    Create a bdev with io_uring backend
    @param filename: Path to device or file (ex: /dev/nvme0n1)
    @param name: bdev name
    @param block_size: Block size for this bdev
    '''
    cmd = SPDK_RPC + f" bdev_uring_create {filename}" + \
                     f" {name} {block_size}"

    result = run_command(cmd)

    return result


def spdk_bdev_uring_delete(name):
    '''
    Delete a uring bdev
    @param name: uring bdev name
    '''
    cmd = SPDK_RPC + f" bdev_uring_delete {name}"

    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stdout.decode('utf-8'))


def spdk_bdev_aio_create(filename, name, block_size):
    '''
    Add a bdev with aio backend
    @param filename: Path to device or file (ex: /dev/sda)
    @param name: Block device name
    @param block_size: Block size for this bdev
    '''
    cmd = SPDK_RPC + f" bdev_aio_create {filename}" + \
                     f" {name} {block_size}"

    result = run_command(cmd)

    return result


def spdk_bdev_aio_delete(name):
    '''
    Delete an aio disk
    @param name: aio bdev name
    '''
    cmd = SPDK_RPC + f" bdev_aio_delete {name}"

    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stdout.decode('utf-8'))


def spdk_get_bdevs(name=None):
    '''
    Show SPDK block devices
    @return: json object with
    block devices
    '''
    cmd = SPDK_RPC + ' bdev_get_bdevs'
    if name:
        cmd += f" --name {name}"

    result = run_command(cmd)

    if result.returncode:
        sys.exit(result.stdout.decode('utf-8'))

    return json.loads(result.stdout.decode('utf-8'))


def spdk_get_bdev_info_formatted(bdev):
    '''
    Show SPDK block device path information
    in formatted output
    @return: formatted string with device path
    '''
    bdev_info = spdk_get_bdevs(bdev)

    if 'aio' in bdev_info[0]['driver_specific']:
        bdev_filename = bdev_info[0]['driver_specific']['aio']['filename']
        return nvme_get_bdev_info_formatted(bdev_filename)
    elif 'uring' in bdev_info[0]['driver_specific']:
        bdev_filename = bdev_info[0]['driver_specific']['uring']['filename']
        return nvme_get_bdev_info_formatted(bdev_filename)
    elif 'nvme' in bdev_info[0]['driver_specific']:
        nvme_path = bdev_info[0]['driver_specific']['nvme']
        nvme_paths = list()

        if isinstance(nvme_path, list):
            nvme_paths = nvme_path
        elif isinstance(nvme_path, dict):
            nvme_paths.append(nvme_path)
        else:
            sys.exit('Unsupported data type {} for SPDK nvme driver!'.format(type(nvme_path)))

        nvme_paths_form = list()
        form_tmpl = 'transport={}/adrfam={}/traddr={}/trsvcid={}'
        for nvme_path in nvme_paths:
            nvme_paths_form.append(form_tmpl.format(nvme_path['trid']['trtype'].lower(),
                                                    nvme_path['trid']['adrfam'].lower(),
                                                    nvme_path['trid']['traddr'],
                                                    nvme_path['trid']['trsvcid']))

        return ','.join(nvme_paths_form)

    else:
        sys.exit(f"not supported driver type for device {bdev}")


def spdk_bdev_nvme_set_options(bdev_retry_count=None):
    '''
    @param bdev_retry_count: the number of attempts
    per I/O in the bdev layer when an I/O fails.
    -1 means infinite retries.
    '''

    cmd = SPDK_RPC + ' bdev_nvme_set_options'

    if bdev_retry_count:
        cmd += f" --bdev-retry-count {bdev_retry_count}"

    result = run_command(cmd)

    rc = result.returncode
    out = result.stdout.decode('utf-8')

    return rc, out


def main():
    parser = argparse.ArgumentParser(description='Configure multipath device')
    parser.add_argument('--op',
                        type=str,
                        choices=[
                            'connect',
                            'disconnect',
                            'discover',
                            'get_bdev_info'
                        ],
                        required=True,
                        help='action (required)')
    parser.add_argument('--protocol',
                        type=str,
                        choices=['nvme'],
                        required=True,
                        help='protocol to use (required)')
    parser.add_argument('--policy',
                        type=str,
                        choices=['numa', 'round-robin'],
                        default='numa',
                        help='i/o policy (default: numa)')
    parser.add_argument('--qn',
                        type=str,
                        help='qn name')
    parser.add_argument('--hostqn',
                        type=str,
                        help='user-defined hostqn')
    parser.add_argument('--path',
                        type=str,
                        action='append',
                        dest='paths',
                        metavar='PATH',
                        help='''
                        connection path
                        (ex. --path=transport=rdma/adrfam=ipv4/traddr=1.1.1.1/trsvcid=4420
                        --path=transport=rdma/adrfam=ipv4/traddr=2.2.2.2/trsvcid=4420)
                        '''
                        )
    parser.add_argument('--dev',
                        type=str,
                        help='namespace name (ex. Nvme0n1)')
    parser.add_argument('--type',
                        type=str,
                        choices=['kernel', 'spdk'],
                        help='namespace type')

    args = parser.parse_args()

    if os.geteuid() != 0:
        sys.exit('root privileges are required to run this script!')

    if args.protocol == 'nvme' and args.op == 'connect':
        if (
            args.paths is None or args.qn is None
           ):
            parser.error('--op=connect and --protocol=nvme require '
                         '--path --qn options.'
                         )
        parsed_paths = parse_paths(parser, args.paths)
        nvme_controller_name = spdk_nvme_get_controller_name()

        path_num = 0
        spdk_fail = False

        for path in parsed_paths:
            path_num = path_num + 1

            multipath = None
            ctrlr_loss_timeout_sec = None
            reconnect_delay_sec = None

            if len(parsed_paths) > 1:
                multipath = "multipath"
                ctrlr_loss_timeout_sec = "-1"
                reconnect_delay_sec = "20"

                if path_num == 1:
                    rc, out = spdk_bdev_nvme_set_options(bdev_retry_count="-1")
                    if rc and not "Operation not permitted" in out:
                        sys.exit(out)

            rc, out = spdk_nvme_attach_controller(
                nvme_controller_name, path['transport'],
                path['traddr'], path['adrfam'],
                path['trsvcid'], args.qn,
                args.hostqn, multipath=multipath,
                ctrlr_loss_timeout_sec=ctrlr_loss_timeout_sec,
                reconnect_delay_sec=reconnect_delay_sec
            )
            if rc:
                if "File exists" in out:
                    sys.exit(out)
                # If some paths were configured -
                # we need to clear them
                if path_num > 1:
                    spdk_bdev_nvme_detach_controller(nvme_controller_name)
                spdk_fail = True
                break

        if not spdk_fail:
            print(out)
        else:
            path_num = 0
            for path in parsed_paths:
                path_num = path_num + 1

                load_nvme_transport_module(path['transport'])
                rc, out = connect_nvme(path['transport'], args.qn,
                                       path['traddr'], path['trsvcid'],
                                       args.hostqn)
                if rc:
                    if path_num > 1:
                        disconnect_nvme_subsys(args.qn)
                    sys.exit(out)

                if path_num == 1:
                    set_nvme_io_policy(args.policy, args.qn)

            # Wait when system will initialize all nvme devices
            # that belong to the current subnqn
            n = 0
            num_of_ret = 5

            while True:

                nvme_devices = []

                for dev in nvme_list():
                    dev_subnqn = nvme_list_subsys(dev['dev'])['Subsystems'][0]['NQN']
                    if dev_subnqn == args.qn:
                        nvme_devices.append(dev)

                if len(nvme_devices) == 0 and num_of_ret != 0:
                    num_of_ret -= 1
                    sleep(0.2)
                    continue

                if len(nvme_devices) == n:
                    break

                n = len(nvme_devices)
                sleep(1)

            spdk_nvme_bdevs = []
            for nvme_dev in nvme_devices:
                fn, name, bs = nvme_dev['dev'], nvme_dev['dev'].split('/')[-1], nvme_dev['block_size']
                # Try to create SPDK URING device on top of kernel nvme
                result = spdk_bdev_uring_create(fn, name, bs)
                if result.returncode and 'Method not found' in result.stdout.decode('utf-8'):
                    # If URING isn't supported, try AIO
                    result = spdk_bdev_aio_create(fn, name, bs)

                if result.returncode:
                    sys.exit(result.stdout.decode('utf-8'))

                spdk_nvme_bdevs.append('dev={}/block_size={}'.format(name, bs))
            print(','.join(spdk_nvme_bdevs))

    elif args.protocol == 'nvme' and args.op == 'disconnect':
        if args.qn is None:
            parser.error('--protocol=nvme and --op=disconnect require '
                         '--qn option'
                         )
        nvme_deleted = False
        spdk_nvme_controllers = spdk_nvme_get_controllers()
        for nvme_ctrl in spdk_nvme_controllers:
            if 'ctrlrs' in nvme_ctrl:
                ctrl = nvme_ctrl['ctrlrs'][0]
            else:
                ctrl = nvme_ctrl

            if args.qn == ctrl['trid']['subnqn']:
                spdk_bdev_nvme_detach_controller(nvme_ctrl['name'])
                nvme_deleted = True
                break
        if not nvme_deleted:
            nvme_dev_list = nvme_list()
            if nvme_dev_list:
                for nvme in nvme_dev_list:
                    nvme_sybsys = nvme_list_subsys(nvme['dev'])
                    if nvme_sybsys and args.qn == nvme_sybsys['Subsystems'][0]['NQN']:
                        nvme_dev_prefix = nvme_sybsys['Subsystems'][0]['Paths'][0]['Name']
                        nvme_pattern = re.compile('^/dev/{}.*'.format(nvme_dev_prefix))
                        for bdev_type in ['uring', 'aio']:
                            for spdk_dev in spdk_get_bdevs():
                                if bdev_type in spdk_dev['driver_specific']:
                                    bdev_filename = spdk_dev['driver_specific'][bdev_type]['filename']
                                    if nvme_pattern.match(bdev_filename):
                                        if bdev_type == 'uring':
                                            spdk_bdev_uring_delete(spdk_dev['name'])
                                        elif bdev_type == 'aio':
                                            spdk_bdev_aio_delete(spdk_dev['name'])

                disconnect_nvme_subsys(args.qn)

    elif args.protocol == 'nvme' and args.op == 'discover':
        if (
            args.paths is None
           ):
            parser.error('--op=discover and --protocol=nvme require '
                         '--path option.'
                         )
        if len(args.paths) != 1:
            parser.error('only one path is allowable with --op=discover')

        parsed_paths = parse_paths(parser, args.paths)
        path = parsed_paths[0]
        load_nvme_transport_module(path['transport'])

        discover_nvme(path['transport'], path['traddr'],
                      path['trsvcid'], args.hostqn
                      )

    elif args.protocol == 'nvme' and args.op == 'get_bdev_info':
        if (
            args.dev is None or args.type is None
           ):
            parser.error(
                            '--op=get_bdev_info and --protocol=nvme require '
                            '--dev --type options.'
                        )
        if args.type == 'spdk':
            print(spdk_get_bdev_info_formatted(args.dev))

        elif args.type == 'kernel':
            print(nvme_get_bdev_info_formatted(args.dev))


if __name__ == '__main__':
    main()
