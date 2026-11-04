#!/usr/bin/env bash
# Restart the KiCraft web app (the detached setsid instance on this box).
#
# Stops any running `python -m kicraft.server.web`, starts a fresh one from
# the repo root (so it picks up .env and the current working tree), and waits
# until it serves before reporting success. Safe to run when nothing is up:
# it just starts the server. Logs append to logs/kicraft_web.log.
#
# Usage:  deploy/restart-web.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${KICRAFT_WEB_LOG:-$REPO/logs/kicraft_web.log}"
# This canonical Caddy-backed restart defaults to loopback even when an older
# .env says 0.0.0.0; an explicit invoking-environment override still wins.
export KICRAFT_WEB_HOST="${KICRAFT_WEB_HOST:-127.0.0.1}"
PATTERN='^([^ ]*/)?python([0-9.]*)? -m kicraft\.server\.web( |$)'

workers() {
    local pid
    while read -r pid; do
        [ -n "$pid" ] || continue
        [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)" = "$REPO" ] && echo "$pid"
    done < <(pgrep -f "$PATTERN" || true)
}

# Resolve the port exactly as the application does, including environment overrides.
PORT="$(cd "$REPO" && .venv/bin/python -c 'import os; from kicraft.server.config import load_dotenv; load_dotenv(); print(os.environ.get("KICRAFT_WEB_PORT", "8080"))')"

# HTTP alone may be answered by an unrelated process on the configured port.
owns_listener() {
    python3 - "$pid" "$PORT" <<'PY'
import os
from pathlib import Path
import sys

pid, port = sys.argv[1], int(sys.argv[2])
inodes = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    except OSError:
        pass
try:
    fds = Path("/proc", pid, "fd").iterdir()
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in inodes:
            sys.exit(0)
except OSError:
    pass
sys.exit(1)
PY
}

mkdir -p "$REPO/logs"

pids="$(workers)"
if [ -n "$pids" ]; then
    echo "stopping kicraft-web: $pids"
    # shellcheck disable=SC2086  # word-splitting the pid list is intended
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do          # up to 10s of graceful shutdown
        [ -z "$(workers)" ] && break
        sleep 0.5
    done
    pids="$(workers)"
    if [ -n "$pids" ]; then
        echo "still running; sending SIGKILL"
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
else
    echo "no running instance found; starting fresh"
fi

echo "==== $(date -Is) restart-web.sh ====" >> "$LOG"
cd "$REPO"
setsid nohup .venv/bin/python -m kicraft.server.web >> "$LOG" 2>&1 < /dev/null &
pid=$!

for _ in $(seq 1 30); do              # up to 15s for startup
    if ! kill -0 "$pid" 2>/dev/null; then
        break
    fi
    if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" && kill -0 "$pid" 2>/dev/null && owns_listener; then
        echo "kicraft-web is up on :$PORT (pid $pid); log: $LOG"
        exit 0
    fi
    sleep 0.5
done

echo "kicraft-web did not respond on :$PORT after 15s; recent log:" >&2
tail -20 "$LOG" >&2
exit 1
