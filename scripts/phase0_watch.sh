#!/usr/bin/env bash
# Phase 0: live disk/load watch that ABORTS a prod push-to-failure before it does
# real damage. The architecture self-recovers (systemd restart, flock auto-release,
# orphan-reaper), so the only non-recoverable risks are disk-fill and a wedged box.
# This watcher trips on either and (1) writes the harness abort file (graceful) and
# (2) engages KICRAFT_KILL_SWITCH (stops any LLM spend) and (3) optionally stops the
# build worker (hard drain).
#
#   scripts/phase0_watch.sh --abort-file /path/abort
#   scripts/phase0_watch.sh --min-disk-gib 3 --max-load-per-core 10 --interval 2 \
#       --abort-file /path/abort --stop-worker
set -euo pipefail
cd "$(dirname "$0")/.."

MIN_DISK_GIB=2
MAX_LOAD_PER_CORE=8
INTERVAL=2
ABORT_FILE=""
STOP_WORKER=0
WATCH_PATH="$HOME/.kicraft"
while [ $# -gt 0 ]; do
  case "$1" in
    --min-disk-gib) MIN_DISK_GIB="$2"; shift 2;;
    --max-load-per-core) MAX_LOAD_PER_CORE="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --abort-file) ABORT_FILE="$2"; shift 2;;
    --watch-path) WATCH_PATH="$2"; shift 2;;
    --stop-worker) STOP_WORKER=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

CORES="$(nproc 2>/dev/null || echo 1)"
MAX_LOAD="$(awk "BEGIN{print $MAX_LOAD_PER_CORE * $CORES}")"
echo "[watch] min_disk=${MIN_DISK_GIB}GiB max_load=${MAX_LOAD} (=${MAX_LOAD_PER_CORE}/core x ${CORES}) every ${INTERVAL}s on ${WATCH_PATH}"

trip() {
  echo "[watch] !!! TRIPPED: $1 -- aborting the load run"
  [ -n "$ABORT_FILE" ] && echo "abort" > "$ABORT_FILE" && echo "[watch] wrote abort file: $ABORT_FILE"
  # Engage the LLM kill switch in .env (idempotent): stops any model spend.
  if [ -f .env ] && ! grep -q '^KICRAFT_KILL_SWITCH=1' .env; then
    sed -i.bak '/^KICRAFT_KILL_SWITCH=/d' .env 2>/dev/null || true
    echo "KICRAFT_KILL_SWITCH=1" >> .env
    echo "[watch] set KICRAFT_KILL_SWITCH=1 in .env"
  fi
  if [ "$STOP_WORKER" = "1" ]; then
    systemctl stop kicraft-build-worker 2>/dev/null \
      && echo "[watch] stopped kicraft-build-worker" \
      || echo "[watch] could not stop kicraft-build-worker (no systemctl perms?)"
  fi
  exit 1
}

while true; do
  FREE_GIB="$(df -BG --output=avail "$WATCH_PATH" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 999)"
  LOAD1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"
  printf '[watch] free=%sGiB load1=%s\n' "${FREE_GIB:-?}" "$LOAD1"
  [ -n "${FREE_GIB:-}" ] && [ "$FREE_GIB" -lt "$MIN_DISK_GIB" ] 2>/dev/null \
    && trip "disk free ${FREE_GIB}GiB < ${MIN_DISK_GIB}GiB"
  awk "BEGIN{exit !($LOAD1 > $MAX_LOAD)}" && trip "loadavg $LOAD1 > $MAX_LOAD"
  sleep "$INTERVAL"
done
