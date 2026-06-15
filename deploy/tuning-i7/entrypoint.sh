#!/usr/bin/env bash
# Clone/update the repo (code + committed corpus) into the mounted /data volume,
# build a venv that can see the system pcbnew, sanity-check the toolchain, then
# run the tuning loop (default) or any command passed instead.
set -euo pipefail

DATA="${KICRAFT_DATA:-/data}"
REPO="$DATA/KiCraft"
VENV="$DATA/venv"
URL="${KICRAFT_REPO:-https://codeberg.org/LaForestLabs/KiCraft.git}"
BRANCH="${KICRAFT_BRANCH:-main}"

mkdir -p "$DATA"

echo "[entrypoint] repo: $URL @ $BRANCH"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO" reset --hard "origin/$BRANCH"
else
    git clone --depth 1 --branch "$BRANCH" "$URL" "$REPO"
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "[entrypoint] creating venv (with system pcbnew visible) ..."
    python3 -m venv --system-site-packages "$VENV"
fi
export PATH="$VENV/bin:$PATH"
pip install -q --upgrade pip
pip install -q -e "$REPO[design,tuning]"

# Fail fast if the toolchain is broken, before burning time on a run.
python - <<'PY'
import pcbnew, cma
print(f"[entrypoint] toolchain OK: pcbnew {pcbnew.GetBuildVersion()} | cma {cma.__version__}")
PY
test -s /root/.local/lib/freerouting-1.9.0.jar || { echo "[entrypoint] FreeRouting jar missing"; exit 1; }

cores="$(nproc)"
# One build slot per ~4 hardware threads: each eval is itself multi-threaded
# (leaf solvers + a FreeRouting JVM), so this runs a few evals concurrently
# without oversubscribing. Override with KICRAFT_BUILD_SLOTS.
if [ -z "${KICRAFT_BUILD_SLOTS:-}" ]; then
    KICRAFT_BUILD_SLOTS=$(( cores / 4 ))
    [ "$KICRAFT_BUILD_SLOTS" -lt 2 ] && KICRAFT_BUILD_SLOTS=2
fi
export KICRAFT_BUILD_SLOTS
echo "[entrypoint] cores=$cores  build_slots=$KICRAFT_BUILD_SLOTS"

cd "$REPO"
if [ "${1:-tune}" = "tune" ]; then
    OUT="${OUT:-$DATA/runs/${RUN_ID:-i7}}"
    mkdir -p "$OUT"
    ACTIVE="${ACTIVE:-placement_clearance_mm,courtyard_padding_mm,connector_gap_mm,connector_edge_inset_mm,subcircuit_margin_mm,parent_spacing_mm,sa_refine_move_radius_mm,edge_margin_mm}"
    echo "[entrypoint] tuning -> $OUT (gens=${GENS:-40} seeds=${SEEDS:-0,1} popsize=${POPSIZE:-8} timeout=${TIMEOUT:-600})"
    exec python -m kicraft.tuning.cli run \
        --corpus "$REPO/tuning_corpus" --out "$OUT" \
        --mode replay --seeds "${SEEDS:-0,1}" \
        --scalarization "${SCAL:-balanced}" \
        --gens "${GENS:-40}" --popsize "${POPSIZE:-8}" \
        --timeout "${TIMEOUT:-600}" --active "$ACTIVE" \
        --run-id "${RUN_ID:-i7}"
fi

# Any other command (e.g. `bash`, or `python -m kicraft.tuning.cli report ...`)
exec "$@"
