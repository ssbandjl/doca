#!/usr/bin/env python3

# Copyright (C) 2020-2021 NVIDIA Corporation. ALL RIGHTS RESERVED.
# Copyright 2016 Cumulus Networks, Inc. All rights reserved.
# Author: Julien Fortin, julien@cumulusnetworks.com
#
# DNSWatcher --
#   tool to watch /etc/resolv.conf and configure rules with iproute2
#

import os
import sys
import time
import shlex
import signal
import socket
import argparse
import pyinotify
import subprocess

import logging
import logging.handlers

from threading import Lock

log = None
_nameservers = []
mutex = Lock()


class DNSWatcherActions():
    @staticmethod
    def signal_handler(signum, frame):
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    @staticmethod
    def inotify_callback(notifier):
        if notifier.proc_fun().should_stop():
            return True

    @staticmethod
    def get_rules():
        mutex.acquire(1)
        global _nameservers
        _nameservers = []
        mutex.release()
        try:
            output = subprocess.check_output("ip rule show | grep to", shell=True)
            lines = output.decode('utf-8').split('\n')
            for line in lines:
                if not line: continue
                l = line.split()
                mutex.acquire(1)
                _nameservers.append({'ip': l[4], 'prio': l[0][:-1], 'vrf': l[8]})
                mutex.release()
        except subprocess.CalledProcessError as e:
            log.warning(str(e))
        return True

    @staticmethod
    def get_raw_rules_from_file():
        i = 0
        rules = []
        try:
            with open(DNSWatcher.DNS_FILE_PATH, 'r') as f:
                for line in f:
                    if line.strip().startswith('nameserver'):
                        parsed_rule = line.strip()[10:].strip().split()
                        if parsed_rule:
                            rules.append(parsed_rule)
                        else:
                            log.warning('warning: %s: line %d: "%s" is not valid' % (DNSWatcher.DNS_FILE_PATH, i, line))
                    i += 1
        except IOError as e:
            log.warning(str(e))
        return rules

    @staticmethod
    def parse_raw_rules(raw, default_prio, vrf_enable=False, prio_syntax_enable=False, ):
        new_rules = []
        for rule in raw:
            size = len(rule)

            try:
                socket.inet_aton(rule[0])
            except socket.error as e:
                log.warning('warning: skipping "nameserver %s": %s' % (' '.join(rule), str(e)))
                continue

            if prio_syntax_enable:
                try:
                    index = rule.index('prio')
                    if index + 1 < size:
                        try:
                            prio = int(rule[index + 1])
                        except ValueError as e:
                            log.warning('warning: Skipping "nameserver %s": %s' % (' '.join(rule), str(e)))
                            continue
                    else:
                        log.warning('warning: skipping "nameserver %s": please specify a `prio` value' % ' '.join(rule))
                        continue
                except:
                    prio = default_prio
            else:
                prio = default_prio

            if vrf_enable:
                try:
                    index = rule.index('vrf')
                    if index + 1 < size:
                        vrf = rule[index + 1]
                    else:
                        vrf = 'mgmt'
                except:
                    vrf = None
            else:
                vrf = None
            new_rules.append({'ip': rule[0], 'prio': prio, 'vrf': vrf})
        return new_rules

    @staticmethod
    def get_new_rules(rules):
        mutex.acquire(1)
        rules = [r for r in rules if r['ip'] not in [n['ip'] for n in _nameservers]]
        mutex.release()
        return rules

    @staticmethod
    def get_deprecated_rules(rules):
        mutex.acquire(1)
        rules = [n for n in _nameservers if n['ip'] not in [r['ip'] for r in rules]]
        mutex.release()
        return rules

    @staticmethod
    def modify_rules(action, rules, callback):
        for r in rules:
            try:
                cmd = 'ip rule %s prio %s to %s iif lo' % (action, r['prio'], r['ip'])
                if r['vrf']:
                    cmd = '%s table %s' % (cmd, r['vrf'])
                log.info('executing: %s' % cmd)
                output = subprocess.check_output(shlex.split(cmd))
                callback(r)
            except subprocess.CalledProcessError as e:
                log.warning(str(e) + (output if output else ''))

    @staticmethod
    def add_to_nameserver(rule):
        mutex.acquire(1)
        _nameservers.append(rule)
        mutex.release()

    @staticmethod
    def del_from_nameserver(rule):
        mutex.acquire(1)
        _nameservers.remove(rule)
        mutex.release()

    @staticmethod
    def file_update(default_prio, vrf):
        raw_rules = DNSWatcherActions.get_raw_rules_from_file()
        rules = DNSWatcherActions.parse_raw_rules(raw_rules, default_prio, vrf_enable=vrf)

        to_add = DNSWatcherActions.get_new_rules(rules)
        to_del = DNSWatcherActions.get_deprecated_rules(rules)

        DNSWatcherActions.modify_rules('add', to_add, DNSWatcherActions.add_to_nameserver)
        DNSWatcherActions.modify_rules('del', to_del, DNSWatcherActions.del_from_nameserver)

    @staticmethod
    def file_delete():
        mutex.acquire(1)
        tmp = list(_nameservers)
        mutex.release()
        DNSWatcherActions.modify_rules('del', tmp, DNSWatcherActions.del_from_nameserver)
        time.sleep(42 / 1000000.0)


class DNSWatcherEventHandler(pyinotify.ProcessEvent):
    def __init__(self, wm, descriptor, prio, vrf):
        self.watch_manager = wm
        self.descriptor = descriptor
        self.prio = prio
        self.vrf = vrf
        self._should_stop = False

    def should_stop(self):
        return self._should_stop

    def process_IN_MOVE_SELF(self, event):
        log.info('inotify: IN_MOVE_SELF event')
        self._should_stop = True
        DNSWatcherActions.file_delete()

    def process_IN_DELETE_SELF(self, event):
        log.info('inotify: IN_DELETE_SELF event')
        self._should_stop = True
        DNSWatcherActions.file_delete()

    def process_IN_DELETE(self, event):
        log.info('inotify: IN_DELETE event')
        self._should_stop = True
        DNSWatcherActions.file_delete()

    def process_IN_CLOSE_WRITE(self, event):
        log.info('inotify: IN_CLOSE_WRITE event')
        DNSWatcherActions.file_update(self.prio, self.vrf)

    def process_IN_MODIFY(self, event):
        log.info('inotify: IN_MODIFY event')
        DNSWatcherActions.file_update(self.prio, self.vrf)


class DNSWatcher:
    DNS_FILE_PATH = '/etc/resolv.conf'

    def __init__(self, prio):
        self.prio = prio
        self.vrf = self.get_vrf_status()
        self.ready = DNSWatcherActions.get_rules()

    def get_vrf_status(self):
        try:
            return 'NOT' not in subprocess.check_output("mgmt-vrf status")
        except:
            return False

    def run(self):
        while self.ready:
            DNSWatcherActions.file_update(self.prio, self.vrf)
            while not os.path.isfile(DNSWatcher.DNS_FILE_PATH):
                time.sleep(1)
                if os.path.isfile(DNSWatcher.DNS_FILE_PATH):
                    DNSWatcherActions.file_update(self.prio, self.vrf)
            if not self.inotify_loop():
                return

    def inotify_loop(self):
        mask = pyinotify.IN_DELETE_SELF | pyinotify.IN_DELETE | pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MODIFY | pyinotify.IN_MOVE_SELF
        watch_manager = pyinotify.WatchManager()
        descriptor = watch_manager.add_watch(DNSWatcher.DNS_FILE_PATH, mask, rec=False)
        notifier = pyinotify.Notifier(watch_manager,
                                      DNSWatcherEventHandler(watch_manager, descriptor, self.prio, self.vrf))
        notifier.loop(callback=DNSWatcherActions.inotify_callback)
        return notifier.proc_fun().should_stop()


def parse_arg(argv):
    argparser = argparse.ArgumentParser(description='watch /etc/resolv.conf and configure rules with iproute2')

    argparser.add_argument('-p', '--prio', dest='prio', default='99', help='prio value (default value 99)')

    argparser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, help='verbose')
    argparser.add_argument('-d', '--debug', dest='debug', action='store_true', help='output debug info')
    argparser.add_argument('-l', '--syslog', dest='syslog', action='store_true', help=argparse.SUPPRESS)
    return argparser.parse_args(argv)


def init(args):
    global log

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    try:
        if hasattr(args, 'syslog') and args.syslog:
            root_logger = logging.getLogger()
            syslog_handler = logging.handlers.SysLogHandler(address='/dev/log',
                                                            facility=logging.handlers.SysLogHandler.LOG_DAEMON)
            logging.addLevelName(logging.ERROR, 'error')
            logging.addLevelName(logging.WARNING, 'warning')
            logging.addLevelName(logging.DEBUG, 'debug')
            logging.addLevelName(logging.INFO, 'info')
            root_logger.setLevel(log_level)
            syslog_handler.setFormatter(logging.Formatter(
                '%(name)s: %(levelname)s: %(message)s'))
            root_logger.addHandler(syslog_handler)
            log = logging.getLogger('dsnwatcher')
        else:
            logging.basicConfig(level=log_level,
                                format='%(levelname)s: %(message)s')
            logging.addLevelName(logging.ERROR, 'error')
            logging.addLevelName(logging.WARNING, 'warning')
            logging.addLevelName(logging.DEBUG, 'debug')
            logging.addLevelName(logging.INFO, 'info')
            log = logging.getLogger('dnswatcher')
    except:
        raise


def main(argv):
    args = parse_arg(argv[1:])
    init(args)
    DNSWatcher(args.prio).run()


if __name__ == '__main__':
    main(sys.argv)
