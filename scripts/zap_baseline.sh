#!/usr/bin/env bash
# OWASP ZAP baseline DAST against a LOCALLY-running KiCraft instance.
#
# Run the target with the mock LLM + a throwaway DB so the scan never spends or
# touches real data, e.g.:
#   KICRAFT_LLM_MODE=mock KICRAFT_MOCK_TRANSCRIPT=kicraft/loadtest/fixtures/transcript_usb_pd_trigger.json \
#   KICRAFT_USERS_DB=/tmp/zap_accounts.db KICRAFT_SPEND_LEDGER=/tmp/zap_ledger.db \
#   KICRAFT_WEB_PORT=8080 .venv/bin/python -m kicraft.server.web &
#
# Then:
#   scripts/zap_baseline.sh                       # scans http://127.0.0.1:8080
#   scripts/zap_baseline.sh https://staging.host  # scans a remote target
#
# Uses the zaproxy docker image if `zap-baseline.py` is not on PATH. The baseline
# scan is passive + a short active spider; it is safe to point at a staging box.
set -euo pipefail
TARGET="${1:-http://127.0.0.1:8080}"
REPORT="${ZAP_REPORT:-zap-baseline-report.json}"

if command -v zap-baseline.py >/dev/null 2>&1; then
  exec zap-baseline.py -t "$TARGET" -J "$REPORT" -m 2
elif command -v docker >/dev/null 2>&1; then
  echo "[zap] zap-baseline.py not found; using the ghcr.io/zaproxy/zaproxy image"
  exec docker run --rm --network=host -v "$(pwd):/zap/wrk/:rw" \
    ghcr.io/zaproxy/zaproxy:stable zap-baseline.py -t "$TARGET" -J "$REPORT" -m 2
else
  echo "[zap] neither zap-baseline.py nor docker is available; install OWASP ZAP" >&2
  echo "      (https://www.zaproxy.org/) or run the dockerized image." >&2
  exit 2
fi
