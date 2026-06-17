#!/usr/bin/env bash
# Phase 0: restore the live DBs from a snapshot and clear the kill switch, after a
# prod push-to-failure. Stops the services while the DB files are swapped, then
# restarts them via the existing deploy scripts.
#
#   scripts/phase0_restore.sh ~/.kicraft/loadtest_snapshots/<ts>
set -euo pipefail
cd "$(dirname "$0")/.."

SNAP_DIR="${1:?usage: phase0_restore.sh <snapshot-dir>}"
[ -d "$SNAP_DIR" ] || { echo "no such snapshot dir: $SNAP_DIR" >&2; exit 2; }

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
eval "$("$PY" - <<'EOF'
import os
from kicraft.server.config import load_dotenv
load_dotenv()
home = os.path.expanduser("~/.kicraft")
print(f"ACCT_DB={os.environ.get('KICRAFT_USERS_DB', f'{home}/accounts.db')!r}")
print(f"LEDGER_DB={os.environ.get('KICRAFT_SPEND_LEDGER', f'{home}/spend_ledger.db')!r}")
EOF
)"

echo "[restore] stopping services (best effort)"
systemctl stop kicraft-build-worker 2>/dev/null || true
systemctl stop kicraft-web 2>/dev/null || true

for db in "$ACCT_DB" "$LEDGER_DB"; do
  snap="$SNAP_DIR/$(basename "$db")"
  [ -f "$snap" ] || { echo "[restore] no snapshot for $(basename "$db"); skipping"; continue; }
  # Drop any stale WAL/SHM so the restored file is authoritative.
  rm -f "$db" "$db-wal" "$db-shm"
  cp "$snap" "$db"
  echo "[restore] $snap -> $db"
done

# Clear the kill switch the watch may have set.
if [ -f .env ]; then
  sed -i.bak '/^KICRAFT_KILL_SWITCH=1$/d' .env 2>/dev/null || true
  grep -q '^KICRAFT_KILL_SWITCH=' .env || echo "KICRAFT_KILL_SWITCH=0" >> .env
  echo "[restore] cleared KICRAFT_KILL_SWITCH"
fi

echo "[restore] restarting services"
[ -x deploy/restart-web.sh ] && deploy/restart-web.sh || echo "[restore] run deploy/restart-web.sh manually"
[ -x deploy/restart-build-worker.sh ] && deploy/restart-build-worker.sh || true
echo "[restore] done"
