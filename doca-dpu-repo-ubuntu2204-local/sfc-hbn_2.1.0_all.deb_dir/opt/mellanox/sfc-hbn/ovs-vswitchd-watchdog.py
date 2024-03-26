import pyinotify
import subprocess
import sys

class ProcessTransientFile(pyinotify.ProcessEvent):

    def process_IN_MODIFY(self, event):
        sys.stdout.write(f'\t {event.maskname} -> written')

    def process_IN_DELETE(self, event):
        sys.stdout.write(f'\t {event.maskname} -> delete')

    def process_IN_CREATE(self, event):
        sys.stdout.write(f'\t {event.maskname} -> create')

    def process_IN_CLOSE_WRITE(self, event):
        sys.stdout.write(f'\t {event.maskname} -> in_close_write')
        exit(1)

    def process_IN_MOVED_TO(self, event):
        sys.stdout.write(f'\t {event.maskname} -> in_moved_to')

    def process_default(self, event):
        sys.stdout.write(f'default: {event.maskname}')


wm = pyinotify.WatchManager()
notifier = pyinotify.Notifier(wm)
# Monitor the file /var/run/openvswitch/ovs-vswitchd.pid.
# We assume that the file exists (ovs-vswitchd is running)
# and we exit with an return code of 1 in the case
# IN_CLOSE_WRITE (ovs-vswitchd restarted) was signaled
wm.watch_transient_file('/var/run/openvswitch/ovs-vswitchd.pid', pyinotify.ALL_EVENTS, ProcessTransientFile)
notifier.loop()
