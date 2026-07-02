#!/usr/bin/env bash
# Push the i7 tuning run's progress to the kicraft.io admin page.
#
# Run this ON THE UNRAID HOST (e.g. as a User Scripts cron, every 5 min). Only
# progress.json travels — a tiny, self-contained chart payload the tuner writes
# each generation — so there is NO live sqlite to sync and nothing can tear. The
# admin page scans ~/.kicraft/tuning on the cloud box, so the file lands at
# ~/.kicraft/tuning/<RUN_ID>/progress.json and shows up automatically.
#
# One-time setup (passwordless SSH from unraid -> cloud, as the web-app user):
#   ssh-keygen -t ed25519 -f /boot/config/ssh/kicraft_tune_id -N ''
#   ssh-copy-id -i /boot/config/ssh/kicraft_tune_id.pub CLOUD_USER@CLOUD_HOST
set -euo pipefail

# --- configure these (or pass as env) ---------------------------------------
RUN_DIR="${RUN_DIR:-/mnt/user/appdata/kicraft-tune/runs/i11}"
CLOUD="${CLOUD:-kicraft@YOUR_CLOUD_HOST}"        # ssh user@host of the kicraft.io box
RUN_ID="${RUN_ID:-i11}"                           # admin will list this as the run name
CLOUD_DIR="${CLOUD_DIR:-.kicraft/tuning/$RUN_ID}"  # relative to the cloud user's $HOME
SSH_KEY="${SSH_KEY:-/boot/config/ssh/kicraft_tune_id}"
# ----------------------------------------------------------------------------

SRC="$RUN_DIR/progress.json"
if [ ! -f "$SRC" ]; then
    echo "no progress.json yet at $SRC (tuner hasn't finished generation 0)"
    exit 0
fi

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new"
$SSH "$CLOUD" "mkdir -p '$CLOUD_DIR'"
rsync -az -e "$SSH" "$SRC" "$CLOUD:$CLOUD_DIR/progress.json"
echo "[$(date '+%F %T')] synced $RUN_ID progress -> $CLOUD:$CLOUD_DIR/"
