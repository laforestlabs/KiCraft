#!/usr/bin/env bash
# Canonical production deployment: refuse to restart unless all 34 fresh
# real-provider designs commit all five stages, then verify production health.
set -euo pipefail
cd "$(dirname "$0")/.."

./deploy/verify-design-canary.sh
./deploy/restart-web.sh
./deploy/restart-build-worker.sh

curl -sf http://127.0.0.1:8080/ >/dev/null
.venv/bin/python - <<'PY'
import time
from pathlib import Path

log = Path("logs/kicraft_build_worker.log")
deadline = time.monotonic() + 60
last = "<empty>"
while time.monotonic() < deadline:
    lines = [line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    last = lines[-1] if lines else "<empty>"
    if last.startswith("[build-worker] ready"):
        print("production health passed: web HTTP 200; build worker ready")
        break
    time.sleep(1)
else:
    raise SystemExit(f"build worker is not ready after 60s; last log line: {last}")
PY
