#!/usr/bin/env bash
# Restart the KiCraft build worker (detached setsid instance, like restart-web.sh).
#
# Stops any running `python -m kicraft.server.build_worker` (SIGTERM first: the
# worker aborts + requeues its in-flight builds before exiting), starts a fresh
# one from the repo root, and waits for its ready line. Safe when nothing is
# up. Logs append to logs/kicraft_build_worker.log.
#
# The worker is optional: without it, the web app runs builds in-process. With
# it, queued builds execute in this separate process and survive web restarts.
#
# Usage:  deploy/restart-build-worker.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${KICRAFT_BUILD_WORKER_LOG:-$REPO/logs/kicraft_build_worker.log}"
PATTERN='python -m kicraft\.server\.build_worker'

mkdir -p "$REPO/logs"

pids="$(pgrep -f "$PATTERN" || true)"
if [ -n "$pids" ]; then
    echo "stopping build worker: $pids (SIGTERM requeues its running builds)"
    # shellcheck disable=SC2086  # word-splitting the pid list is intended
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 40); do          # up to 20s: it may be killing build trees
        pgrep -f "$PATTERN" >/dev/null || break
        sleep 0.5
    done
    if pgrep -f "$PATTERN" >/dev/null; then
        echo "still running; sending SIGKILL"
        pkill -9 -f "$PATTERN" || true
        sleep 1
    fi
else
    echo "no running worker found; starting fresh"
fi

echo "==== $(date -Is) restart-build-worker.sh ====" >> "$LOG"
cd "$REPO"
setsid nohup .venv/bin/python -m kicraft.server.build_worker >> "$LOG" 2>&1 < /dev/null &

for _ in $(seq 1 20); do              # up to 10s for the ready line
    if tail -5 "$LOG" 2>/dev/null | grep -q "\[build-worker\] ready"; then
        echo "build worker is up (pid $(pgrep -f "$PATTERN" | head -1)); log: $LOG"
        exit 0
    fi
    sleep 0.5
done

echo "build worker did not report ready after 10s; recent log:" >&2
tail -20 "$LOG" >&2
exit 1
