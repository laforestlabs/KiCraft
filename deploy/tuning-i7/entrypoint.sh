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

# Fail fast if pcbnew or the pinned router is unavailable.
python - <<'PY'
import pcbnew, cma
from kicraft.autoplacer.kicad_routing_tools import preflight_kicad_routing_tools

runtime = preflight_kicad_routing_tools()
print(
    f"[entrypoint] toolchain OK: pcbnew {pcbnew.GetBuildVersion()} | "
    f"cma {cma.__version__} | KRT {runtime['version']} "
    f"native {runtime['native_version']}"
)
PY

cores="$(nproc)"
# Eval concurrency. Placement and routing are CPU-bound subprocess workloads.
# This container is dedicated to tuning, so use roughly three quarters of the
# host threads and leave headroom for the OS and orchestration.
if [ -z "${KICRAFT_BUILD_SLOTS:-}" ]; then
    KICRAFT_BUILD_SLOTS=$(( cores * 3 / 4 ))
    [ "$KICRAFT_BUILD_SLOTS" -lt 2 ] && KICRAFT_BUILD_SLOTS=2
fi
export KICRAFT_BUILD_SLOTS
echo "[entrypoint] cores=$cores  build_slots=$KICRAFT_BUILD_SLOTS"

cd "$REPO"
if [ "${1:-tune}" = "tune" ]; then
    OUT="${OUT:-$DATA/runs/${RUN_ID:-i11}}"
    mkdir -p "$OUT"
    # Param-selection mode (mutually exclusive, ACTIVE wins):
    #   ACTIVE=<csv>  -> tune EXACTLY these, skip screening (legacy behavior)
    #   PIN=<csv>     -> always tune these, screening fills the rest up to TOPK
    # Pin the highest-leverage scorer/escape controls; screening fills the rest.
    PIN="${PIN:-signal_escape_length_mm,psw_bbox_packing,psw_aspect_ratio,psw_topology_structure}"
    sel=()
    if [ -n "${ACTIVE:-}" ]; then
        sel=(--active "$ACTIVE")
    elif [ -n "$PIN" ]; then
        sel=(--pin "$PIN")
    fi
    echo "[entrypoint] tuning -> $OUT (gens=${GENS:-40} seeds=${SEEDS:-0,1} popsize=${POPSIZE:-8} timeout=${TIMEOUT:-600} scal=${SCAL:-all_four} top_k=${TOPK:-12})"
    echo "[entrypoint] param selection: ${sel[*]:-screening only}"
    exec python -m kicraft.tuning.cli run \
        --corpus "$REPO/tuning_corpus" --out "$OUT" \
        --mode replay --seeds "${SEEDS:-0,1}" \
        --scalarization "${SCAL:-all_four}" \
        --gens "${GENS:-40}" --popsize "${POPSIZE:-8}" \
        --timeout "${TIMEOUT:-600}" --top-k "${TOPK:-12}" \
        "${sel[@]}" \
        --run-id "${RUN_ID:-i11}"
fi

# Any other command (e.g. `bash`, or `python -m kicraft.tuning.cli report ...`)
exec "$@"
