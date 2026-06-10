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
PATTERN='python -m kicraft\.server\.web'

PORT="$(grep -E '^KICRAFT_WEB_PORT=' "$REPO/.env" 2>/dev/null | cut -d= -f2- || true)"
PORT="${PORT:-8080}"

mkdir -p "$REPO/logs"

pids="$(pgrep -f "$PATTERN" || true)"
if [ -n "$pids" ]; then
    echo "stopping kicraft-web: $pids"
    # shellcheck disable=SC2086  # word-splitting the pid list is intended
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do          # up to 10s of graceful shutdown
        pgrep -f "$PATTERN" >/dev/null || break
        sleep 0.5
    done
    if pgrep -f "$PATTERN" >/dev/null; then
        echo "still running; sending SIGKILL"
        pkill -9 -f "$PATTERN" || true
        sleep 1
    fi
else
    echo "no running instance found; starting fresh"
fi

echo "==== $(date -Is) restart-web.sh ====" >> "$LOG"
cd "$REPO"
setsid nohup .venv/bin/python -m kicraft.server.web >> "$LOG" 2>&1 < /dev/null &

for _ in $(seq 1 30); do              # up to 15s for startup
    if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/"; then
        echo "kicraft-web is up on :$PORT (pid $(pgrep -f "$PATTERN" | head -1)); log: $LOG"
        exit 0
    fi
    sleep 0.5
done

echo "kicraft-web did not respond on :$PORT after 15s; recent log:" >&2
tail -20 "$LOG" >&2
exit 1
