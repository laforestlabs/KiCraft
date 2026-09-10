#!/usr/bin/env bash
# Deploy gate: drive the complete 34-brief design corpus through the REAL
# provider, spend guard, and stage pipeline — the same code, .env profile, and
# per-run budget the web app uses. Exits nonzero unless every selected brief
# commits all five LLM stages (intent, functional_spec, architecture, bom,
# wiring). Passing a slug list is only for local diagnosis; production deploys
# invoke this script without arguments and therefore require all 34.
#
# Usage:
#   deploy/verify-design-canary.sh                 # all 34 briefs
#   deploy/verify-design-canary.sh rp2040-min      # diagnostic subset
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
    SLUGS=()
    SELECTION=()
    EXPECTED=34
    LABEL="all 34 briefs"
else
    SLUGS=("$@")
    JOINED=$(IFS=,; echo "${SLUGS[*]}")
    SELECTION=(--only "$JOINED")
    EXPECTED=${#SLUGS[@]}
    LABEL="${SLUGS[*]}"
fi
OUT="logs/self_eval/canary_$(date -u +%Y%m%dT%H%M%SZ)"

echo "design canary: $LABEL -> $OUT"

.venv/bin/python -m kicraft.eval.self_eval \
    --out "$OUT" \
    --design-only --no-judge --lean-events \
    "${SELECTION[@]}"

.venv/bin/python - "$OUT" "$EXPECTED" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1], "summary.json").read_text(encoding="utf-8"))
expected = int(sys.argv[2])
runs = summary.get("runs") or []
if len(runs) != expected:
    print(f"canary failed: expected {expected} brief runs, recorded {len(runs)}")
    sys.exit(1)
failed = []
for run in runs:
    ok = run.get("design_committed") is True
    print(f"[{'OK  ' if ok else 'FAIL'}] {run['slug']}: "
          f"{'all five stages committed' if ok else run.get('design_error') or 'stage failure'}")
    if not ok:
        failed.append(run["slug"])
if failed:
    print(f"canary FAILED: {len(failed)}/{len(runs)} briefs did not commit every stage")
    sys.exit(1)
print(f"canary passed: {len(runs)}/{len(runs)} briefs committed all five LLM stages")
PY
