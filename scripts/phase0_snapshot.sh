#!/usr/bin/env bash
# Phase 0: snapshot the live SQLite DBs before a prod push-to-failure.
#
# There are NO backups on the box, so this is the safety net: an online (WAL-safe)
# sqlite .backup of accounts.db + spend_ledger.db into a timestamped dir, plus a
# meta record (git HEAD, disk/mem/cpu, build slots, total spend) so the run is
# reproducible and reversible. Restore with scripts/phase0_restore.sh <dir>.
#
#   scripts/phase0_snapshot.sh                 # -> ~/.kicraft/loadtest_snapshots/<ts>
#   scripts/phase0_snapshot.sh /path/to/dir
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
# Resolve DB paths the way the app does (honor .env, fall back to ~/.kicraft).
eval "$("$PY" - <<'EOF'
import os
from kicraft.server.config import load_dotenv
load_dotenv()
home = os.path.expanduser("~/.kicraft")
acct = os.environ.get("KICRAFT_USERS_DB", f"{home}/accounts.db")
ledger = os.environ.get("KICRAFT_SPEND_LEDGER", f"{home}/spend_ledger.db")
print(f"ACCT_DB={acct!r}")
print(f"LEDGER_DB={ledger!r}")
EOF
)"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP_DIR="${1:-$HOME/.kicraft/loadtest_snapshots/$TS}"
mkdir -p "$SNAP_DIR"

for src in "$ACCT_DB" "$LEDGER_DB"; do
  [ -f "$src" ] || { echo "[snapshot] skip (absent): $src"; continue; }
  dst="$SNAP_DIR/$(basename "$src")"
  "$PY" - "$src" "$dst" <<'EOF'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
s = sqlite3.connect(src); d = sqlite3.connect(dst)
with d:
    s.backup(d)        # online, WAL-safe snapshot
s.close(); d.close()
print(f"[snapshot] {src} -> {dst}")
EOF
done

{
  echo "snapshot_at=$TS"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "nproc=$(nproc 2>/dev/null || echo '?')"
  echo "build_slots=${KICRAFT_BUILD_SLOTS:-default}"
  echo "acct_db=$ACCT_DB"
  echo "ledger_db=$LEDGER_DB"
  echo "--- df ---"; df -h "$HOME/.kicraft" 2>/dev/null || df -h "$HOME"
  echo "--- free ---"; free -m 2>/dev/null || true
  echo "--- spend ---"
  "$PY" - <<EOF || true
import sqlite3
try:
    c = sqlite3.connect("$LEDGER_DB")
    print("spent_total_usd=%.4f" % (c.execute("SELECT COALESCE(SUM(cost_usd),0) FROM spend").fetchone()[0] or 0))
except Exception as e:
    print("spend read failed:", e)
EOF
} > "$SNAP_DIR/meta.txt"

echo "[snapshot] complete -> $SNAP_DIR"
echo "[snapshot] restore with: scripts/phase0_restore.sh $SNAP_DIR"
