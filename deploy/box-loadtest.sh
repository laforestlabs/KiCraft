#!/usr/bin/env bash
# Reversible prod push-to-failure orchestrator.
#
# Snapshots the DBs, starts the disk/load watch (which aborts the run before any
# non-recoverable damage), runs a load scenario at $0 (build-storm replay / mock
# pipeline), then ALWAYS prints the restore command. The box self-recovers from a
# crash (systemd restart + flock auto-release + orphan-reaper); this wrapper makes
# the push diagnostic and reversible.
#
#   deploy/box-loadtest.sh build-storm --n 24 --slots 4 --route
#   deploy/box-loadtest.sh pipeline --n 16 --parallel 4 --build-slots 2
#   MIN_DISK_GIB=3 MAX_LOAD_PER_CORE=10 deploy/box-loadtest.sh build-storm --n 12 --slots 2
set -euo pipefail
cd "$(dirname "$0")/.."

SCENARIO="${1:?usage: box-loadtest.sh <build-storm|pipeline> [scenario args...]}"
shift || true

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"

echo "==> Phase 0: snapshot"
SNAP_OUT="$(scripts/phase0_snapshot.sh)"
echo "$SNAP_OUT"
SNAP_DIR="$(echo "$SNAP_OUT" | sed -n 's/.*complete -> //p' | tail -1)"

ABORT_FILE="$(mktemp -u "${TMPDIR:-/tmp}/kicraft_loadtest_abort.XXXX")"
echo "==> Phase 0: starting watch (abort file: $ABORT_FILE)"
scripts/phase0_watch.sh \
  --min-disk-gib "${MIN_DISK_GIB:-2}" \
  --max-load-per-core "${MAX_LOAD_PER_CORE:-8}" \
  --abort-file "$ABORT_FILE" &
WATCH_PID=$!

restore_hint() {
  kill "$WATCH_PID" 2>/dev/null || true
  echo
  echo "================================================================"
  echo "load run finished. To restore the DBs + clear the kill switch:"
  echo "  scripts/phase0_restore.sh ${SNAP_DIR:-<snapshot-dir>}"
  echo "================================================================"
}
trap restore_hint EXIT

echo "==> Running scenario: $SCENARIO $*"
EXTRA=()
[ "$SCENARIO" = "build-storm" ] && EXTRA=(--abort-file "$ABORT_FILE")
"$PY" -m kicraft.loadtest "$SCENARIO" "$@" "${EXTRA[@]}" || true
