#!/usr/bin/env bash
# On-demand design gate: drive the complete 34-brief design corpus through the
# REAL provider, spend guard, and stage pipeline — the same code, .env profile,
# and per-run budget the web app uses. Exits nonzero unless every selected brief
# commits all five LLM stages (intent, functional_spec, architecture, bom,
# wiring). NOT part of deploy-production.sh; run it manually before/after a
# deploy when you want the real-provider check.
#
# Usage:
#   deploy/verify-design-canary.sh                 # all 34 briefs
#   deploy/verify-design-canary.sh rp2040-min      # diagnostic subset
#
# Judging a change needs repeats: one sample per brief cannot tell a fix from
# run-to-run noise. The four 2026-09-17 canaries spread 0-1/34 across identical
# code, while six individually-correct tested fixes moved the aggregate not at
# all. Set KICRAFT_CANARY_REPEATS=3 (and limit the set with slug args) before
# attributing a change to a delta; the summary then aggregates per-brief medians.
set -euo pipefail
cd "$(dirname "$0")/.."

REPEATS="${KICRAFT_CANARY_REPEATS:-1}"
REPEAT_ARGS=()
if [ "$REPEATS" != "1" ]; then
    REPEAT_ARGS=(--repeats "$REPEATS")
fi

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

echo "design canary: $LABEL -> $OUT (repeats: $REPEATS)"

.venv/bin/python -m kicraft.eval.self_eval \
    --out "$OUT" \
    --design-only --no-judge --lean-events \
    ${REPEAT_ARGS[@]+"${REPEAT_ARGS[@]}"} \
    "${SELECTION[@]}"

.venv/bin/python -m kicraft.eval.design_acceptance "$OUT" "${SELECTION[@]}"
