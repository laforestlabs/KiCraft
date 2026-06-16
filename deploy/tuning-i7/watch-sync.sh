#!/usr/bin/env bash
# Auto-sync: re-run sync-to-admin.sh on an interval so the kicraft.io admin page
# (/admin/tuning) tracks a LIVE tuning run hands-off -- it keeps pushing the
# tiny progress.json every cycle, skipping cleanly until gen 0 produces one.
#
# Inherits the SAME env as sync-to-admin.sh. Export once, then launch detached so
# it survives logout (nohup ignores SIGHUP):
#
#   export CLOUD=kicraft@HOST RUN_ID=i9 \
#          RUN_DIR=/mnt/user/appdata/kicraft-tune/runs/i9 \
#          SSH_KEY=/root/.ssh/kicraft_sync
#   cd /mnt/user/appdata/kicraft-tune/KiCraft/deploy/tuning-i7
#   nohup bash watch-sync.sh &
#
# Logs to /tmp/watch-sync-<RUN_ID>.log. Override the cadence with SYNC_INTERVAL
# (seconds, default 300). Stop it with:  pkill -f watch-sync.sh
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
INTERVAL="${SYNC_INTERVAL:-300}"
LOG="/tmp/watch-sync-${RUN_ID:-run}.log"
exec >>"$LOG" 2>&1
echo "[watch-sync $(date '+%F %T')] start: RUN_ID=${RUN_ID:-?} interval=${INTERVAL}s"
while true; do
    bash "$DIR/sync-to-admin.sh" || echo "[watch-sync] sync exited $?"
    sleep "$INTERVAL"
done
