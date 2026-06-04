#!/usr/bin/env bash
# Read-only diagnostics from the live kicraft.io box (no sudo, no writes).
#
# Pulls service state, KiCad library presence, and the newest saved design run
# (brief + state.json + events.jsonl tail) so a wiring/BOM/synth failure can be
# diagnosed from the dev machine without pasting logs by hand. The saved run
# comes from the web app's per-user input-capture feature, so it needs no special
# permissions. The systemd journal is best-effort: the service user cannot read
# it by default, so to unlock it run once on the box:
#     sudo usermod -aG systemd-journal kicraft   # then re-login
#
# Strictly read-only: it never restarts, edits, or runs sudo. Restarts and other
# privileged actions stay gated on purpose.
#
# Usage: deploy/box-diag.sh [user@host]        (default: kicraft@5.78.233.146)
set -uo pipefail
BOX="${1:-${KICRAFT_BOX:-kicraft@5.78.233.146}}"

ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "$BOX" bash -s <<'REMOTE'
echo "=== host ==="; hostname; whoami; uptime | tr -s ' '
echo; echo "=== service: kicraft-web ==="
systemctl is-active kicraft-web 2>/dev/null
systemctl show kicraft-web -p ActiveState,SubState,NRestarts,ExecMainStartTimestamp 2>/dev/null
echo; echo "=== KiCad libraries ==="
echo "symbol libs:    $(ls /usr/share/kicad/symbols/*.kicad_sym 2>/dev/null | wc -l) .kicad_sym files"
echo "footprint libs: $(ls -d /usr/share/kicad/footprints/*.pretty 2>/dev/null | wc -l) .pretty dirs"
echo "KICAD9_SYMBOL_DIR=${KICAD9_SYMBOL_DIR:-(unset)}   KICAD_SYMBOL_DIR=${KICAD_SYMBOL_DIR:-(unset)}"
echo; echo "=== newest saved run ==="
P=$(ls -dt "$HOME"/.kicraft/projects/*/*/ 2>/dev/null | head -1)
if [ -z "${P:-}" ]; then
  echo "(no saved runs under ~/.kicraft/projects)"
else
  echo "DIR: $P"
  echo "--- brief ---"; head -c 500 "$P/brief.txt" 2>/dev/null; echo
  echo "--- state.json (first 3KB) ---"; head -c 3000 "$P/state.json" 2>/dev/null; echo
  echo "--- events: error / stage signals ---"
  grep -aiE 'error|fail|reject|invalid|traceback|exception|"kind": *"stage_|commit' \
       "$P/events.jsonl" 2>/dev/null | tail -25
  echo "--- events: last 15 lines ---"; tail -15 "$P/events.jsonl" 2>/dev/null
fi
echo; echo "=== journal (best-effort; needs systemd-journal group) ==="
journalctl -u kicraft-web -n 50 --no-pager 2>&1 | tail -30
REMOTE
