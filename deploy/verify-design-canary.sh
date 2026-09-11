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
    LABEL="all 34 briefs"
else
    SLUGS=("$@")
    JOINED=$(IFS=,; echo "${SLUGS[*]}")
    SELECTION=(--only "$JOINED")
    LABEL="${SLUGS[*]}"
fi
OUT="logs/self_eval/canary_$(date -u +%Y%m%dT%H%M%SZ)"

echo "design canary: $LABEL -> $OUT"

.venv/bin/python -m kicraft.eval.self_eval \
    --out "$OUT" \
    --design-only --no-judge --lean-events \
    "${SELECTION[@]}"

.venv/bin/python -m kicraft.eval.design_acceptance "$OUT" "${SELECTION[@]}"
