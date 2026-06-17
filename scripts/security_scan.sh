#!/usr/bin/env bash
# Run the static security scans (bandit SAST, pip-audit dep CVEs, gitleaks secrets)
# into the SecurityResultStore, then print a summary. Findings surface on
# /admin/security. Each scanner degrades to "not_installed" when absent, so this
# never hard-fails on a box missing a tool.
#
#   scripts/security_scan.sh                 # all scanners into the default store
#   scripts/security_scan.sh --tool bandit   # just one
#   KICRAFT_SECURITY_DIR=/tmp/sec scripts/security_scan.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "[security] scanning with: $("$PY" - <<'EOF'
from kicraft.security import scans
print(", ".join(f"{t}={'yes' if scans.tool_available(t) else 'MISSING'}"
                 for t in ("bandit", "pip-audit", "gitleaks")))
EOF
)"

exec "$PY" -m kicraft.security.scans "$@"
