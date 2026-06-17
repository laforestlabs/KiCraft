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
# FreeRouting is a Swing/AWT app run under xvfb; the headless JRE can't launch it
# (no libawt_xawt) and boards come back unrouted. Fail loudly rather than silently.
find /usr/lib/jvm -name 'libawt_xawt.so' 2>/dev/null | grep -q . \
    || { echo "[entrypoint] ERROR: GUI-capable JRE missing (libawt_xawt.so) — FreeRouting can't run; rebuild the image (needs full default-jre)"; exit 1; }
echo "[entrypoint] FreeRouting JRE OK"

cores="$(nproc)"
# Eval concurrency. In practice ONE eval ~= ONE core: placement is single-threaded
# Python (numpy threads pinned for determinism) and FreeRouting is a ~single-
# threaded JVM, so the old cores/4 left a 24-thread box ~75% idle. This container
# is dedicated to tuning, so size for throughput: ~3/4 of threads, leaving headroom
# for the OS + orchestrator + JVM GC. The eval pool is sized at slots-1
# (see kicraft/tuning/runner.py:default_workers), so slots=18 -> 17 concurrent.
# Override with KICRAFT_BUILD_SLOTS (lower it if you see the box swapping --
# each eval is a pcbnew process + a FreeRouting JVM, roughly ~1 GB).
if [ -z "${KICRAFT_BUILD_SLOTS:-}" ]; then
    KICRAFT_BUILD_SLOTS=$(( cores * 3 / 4 ))
    [ "$KICRAFT_BUILD_SLOTS" -lt 2 ] && KICRAFT_BUILD_SLOTS=2
fi
export KICRAFT_BUILD_SLOTS
echo "[entrypoint] cores=$cores  build_slots=$KICRAFT_BUILD_SLOTS"

cd "$REPO"
if [ "${1:-tune}" = "tune" ]; then
    OUT="${OUT:-$DATA/runs/${RUN_ID:-i10}}"
    mkdir -p "$OUT"
    # Param-selection mode (mutually exclusive, ACTIVE wins):
    #   ACTIVE=<csv>  -> tune EXACTLY these, skip screening (legacy behavior)
    #   PIN=<csv>     -> always tune these, screening fills the rest up to TOPK
    # Default: PIN the Phase 1-2 routing/scorer levers so a noisy single-param
    # screen can't bury them; screening picks the remaining slots over all params.
    PIN="${PIN:-freerouting_max_passes,leaf_freerouting_max_passes,signal_escape_length_mm,psw_bbox_packing,psw_aspect_ratio,psw_topology_structure}"
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
        --run-id "${RUN_ID:-i10}"
fi

# Any other command (e.g. `bash`, or `python -m kicraft.tuning.cli report ...`)
exec "$@"
