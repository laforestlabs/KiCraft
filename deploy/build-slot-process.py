#!/usr/bin/env python3
"""Inspect host flock ownership and signal a validated process via pidfd."""
import errno
import fcntl
import os
from pathlib import Path
import re
import signal
import sys


BUILD_COMMAND = re.compile(
    r"kicraft\.design\.cli_app (?:build|place)|"
    r"kicraft/cli/solve_subcircuits\.py|py_router/route\.py"
)


def lock_owners(path):
    """Return kernel-reported flock owners; None means ownership is unknown."""
    before = os.stat(path)
    key = (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino)
    owners = set()
    for line in Path('/proc/locks').read_text().splitlines():
        fields = line.split()
        # Waiters are not owners; only FLOCK entries correspond to build_slot().
        if len(fields) < 8 or fields[1] != 'FLOCK':
            continue
        device = fields[5].split(':')
        if len(device) != 3:
            raise ValueError('malformed lock device')
        lock_key = (int(device[0], 16), int(device[1], 16), int(device[2]))
        if lock_key == key:
            owner = int(fields[4])
            if owner <= 0:
                return None
            owners.add(owner)
    after = os.stat(path)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        return None
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            # A held flock without a visible kernel owner is not safe to reap.
            if not owners:
                return None
        else:
            if owners:  # Changed between /proc/locks and the flock probe.
                return None
    finally:
        os.close(fd)
    return owners


def send(pid, start, uid, signum, lock_path):
    # The pidfd binds the signal to this process even if its numeric PID is
    # reused after validation. Missing kernel/Python support fails closed.
    fd = os.pidfd_open(pid, 0)
    try:
        proc = Path('/proc') / str(pid)
        fields = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
        if int(fields[19]) != start or os.stat(proc).st_uid != uid or int(fields[1]) != 1:
            raise ValueError('process identity or parent changed')
        command = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if not BUILD_COMMAND.search(command):
            raise ValueError('process is no longer a build command')
        if lock_path:
            owners = lock_owners(lock_path)
            if owners is None or pid not in owners:
                raise ValueError('process no longer owns the slot flock')
        signal.pidfd_send_signal(fd, signum)
    finally:
        os.close(fd)


def main():
    try:
        if sys.argv[1] == 'owners' and len(sys.argv) == 3:
            owners = lock_owners(sys.argv[2])
            if owners is None:
                raise ValueError('flock owner unavailable')
            print('\n'.join(map(str, sorted(owners))))
        elif sys.argv[1] == 'signal' and len(sys.argv) == 7:
            send(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                 int(sys.argv[5]), sys.argv[6])
        else:
            raise ValueError('invalid arguments')
    except (AttributeError, IndexError, OSError, ValueError) as exc:
        print(f'build-slot identity/lock check failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
