#!/usr/bin/env python3
"""Overnight sweep: find a 'draft' quality preset that cuts time-to-gerber.

Rebuilds known-good (previously fab-ready) projects from their staged
state.json -- no LLM stages, no API spend -- across a ladder of place+route
effort levels, and records wall-clock + fab-readiness per cell so we can pick
the fastest preset whose success rate holds up.

Arms (effort ladder):
  good     baseline: autoexperiment 3 leaf rounds x 3 attempts, 3 parent rounds
  2x2      autoexperiment 2x2x2 (the proposed 'draft' shape)
  2x2lite  2x2x2 + autoplacer.json effort cuts (routing passes, SA iters)
  1x1      autoexperiment 1x1x1 (minimum optimized)
  1x1lite  1x1x1 + the same effort cuts
  fast     existing single-pass solve-hierarchy engine

Custom round counts ride the KICRAFT_QUALITY_PRESETS env override consumed by
`kicraft build` (kicraft/design/cli_app.py); solver/router effort cuts ride a
pre-seeded autoplacer.json in the project dir (discover_project_config).
NOTE: the autoexperiment mutation search perturbs sa_refine_iterations /
max_placement_iterations, so those lite cuts only bind in unmutated rounds;
the routing pass caps are not mutated and always bind.

Usage:
  draft_preset_sweep.py                  # run the full matrix (resumable)
  draft_preset_sweep.py --batch-dir D    # resume/extend an existing batch
  draft_preset_sweep.py --dry-run        # print the cell matrix and exit
  draft_preset_sweep.py --limit 1        # run only the first pending cell
  draft_preset_sweep.py --arms fast,1x1  # restrict arms (same for --boards)
  draft_preset_sweep.py --summarize D    # (re)write summary.md for a batch
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KICRAFT_BIN = REPO / ".venv" / "bin" / "kicraft"
SWEEP_ROOT = REPO / "logs" / "draft_sweep"
SOURCE_BATCH = REPO / "logs" / "self_eval" / "20260611T213618Z"

# Effort cuts for the *lite arms. Keys must exist in autoplacer DEFAULT_CONFIG.
LITE_OVERRIDES = {
    "leaf_routing_max_passes": 6,   # default 12
    "routing_max_passes": 10,       # default 20 (parent)
    "sa_refine_iterations": 150,        # default 300 (mutation may override)
    "sa_refine_no_improve_break": 75,   # default 150
    "max_placement_iterations": 1600,   # default 3332 (mutation may override)
}

ARMS: dict[str, dict] = {
    "good": {"quality": "good", "presets": None, "autoplacer": None},
    "2x2": {
        "quality": "draft",
        "presets": {"draft": {"engine": "autoexperiment", "leaf_rounds": 2,
                              "leaf_attempts": 2, "parent_rounds": 2}},
        "autoplacer": None,
    },
    "2x2lite": {
        "quality": "draft",
        "presets": {"draft": {"engine": "autoexperiment", "leaf_rounds": 2,
                              "leaf_attempts": 2, "parent_rounds": 2}},
        "autoplacer": LITE_OVERRIDES,
    },
    "1x1": {
        "quality": "draft",
        "presets": {"draft": {"engine": "autoexperiment", "leaf_rounds": 1,
                              "leaf_attempts": 1, "parent_rounds": 1}},
        "autoplacer": None,
    },
    "1x1lite": {
        "quality": "draft",
        "presets": {"draft": {"engine": "autoexperiment", "leaf_rounds": 1,
                              "leaf_attempts": 1, "parent_rounds": 1}},
        "autoplacer": LITE_OVERRIDES,
    },
    "fast": {"quality": "fast", "presets": None, "autoplacer": None},
}

# (key, source run dir, repeats, per-cell timeout seconds). All three were
# fab-ready in the 20260611T213618Z self-eval batch (grades B / A / C).
BOARDS = [
    ("BENCH", SOURCE_BATCH / "run_07_A_BENCH_BREAKOUT", 3, 3000),
    ("BMP280", SOURCE_BATCH / "run_04_A_BMP280_BAROMETRIC", 3, 3000),
    ("8CH", SOURCE_BATCH / "run_03_AN_8_CHANNEL", 2, 4800),
]

# Build log markers (cli_app.py / build_slots.py). Timestamps on these split
# total wall-clock into synth / slot-wait / layout / verify+export.
MARK_SYNTH = "[build] 1/5 synthesize"
MARK_WAIT = "[build] waiting for a free build slot"
MARK_SLOT = "[build] build slot acquired"
MARK_LAYOUT = "[build] 2/5 place + route"
MARK_PROMOTE = "[build] 3/5 promoted"
MARK_VERIFY = "[build] 4/5 verify:"
MARK_EXPORT = "[build] 5/5 export"
MARK_DONE = "BUILD COMPLETE"

RE_VERIFY = re.compile(
    r"4/5 verify: shorts=(\d+) unconnected=(\d+) traces=(\S+)")
RE_GATE_FAIL = re.compile(
    r"NOT fab-ready -- shorts=(\d+), unconnected=(\d+)")
RE_SEED = re.compile(r"Master seed:\s+(\d+)")
RE_TIMING = re.compile(
    r"\[timing\] round (\d+) (solve_subcircuits_total|parent_route_total)"
    r"=([\d.]+)s")


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def loadavg() -> list[float]:
    try:
        return list(os.getloadavg())
    except OSError:
        return []


def build_matrix(arm_filter, board_filter):
    """Cells ordered in repeat-major blocks: after block k, every (board, arm)
    pair has k samples -- a partial night still covers the whole matrix."""
    cells = []
    max_reps = max(reps for _, _, reps, _ in BOARDS)
    for rep in range(1, max_reps + 1):
        for bkey, run_dir, reps, timeout_s in BOARDS:
            if rep > reps or (board_filter and bkey not in board_filter):
                continue
            for akey in ARMS:
                if arm_filter and akey not in arm_filter:
                    continue
                cells.append({
                    "cell": f"r{rep}_{bkey}_{akey}",
                    "rep": rep, "board": bkey, "arm": akey,
                    "run_dir": run_dir, "timeout_s": timeout_s,
                })
    return cells


def done_cells(results_path: Path) -> set[str]:
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "harness_error":
                done.add(row.get("cell"))
    return done


def stream_build(cmd, env, log_path: Path, timeout_s: float):
    """Run the build, tee output to log_path with elapsed-seconds prefixes,
    and return (rc, timed_out, marks, lines). Kills the whole process group on
    timeout (routing JVMs are grandchildren)."""
    t0 = time.monotonic()
    marks: dict[str, float] = {}
    lines: list[str] = []
    timed_out = threading.Event()
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", start_new_session=True,
    )

    def watchdog():
        deadline = t0 + timeout_s
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                timed_out.set()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
                time.sleep(20)
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                return
            time.sleep(2)

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()
    with log_path.open("w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            el = time.monotonic() - t0
            log.write(f"{el:9.1f}  {line}")
            line = line.rstrip("\n")
            lines.append(line)
            for name, marker in [
                ("synth", MARK_SYNTH), ("wait", MARK_WAIT),
                ("slot", MARK_SLOT), ("layout", MARK_LAYOUT),
                ("promote", MARK_PROMOTE), ("verify", MARK_VERIFY),
                ("export", MARK_EXPORT), ("done", MARK_DONE),
            ]:
                if name not in marks and marker in line:
                    marks[name] = round(el, 1)
    rc = proc.wait()
    return rc, timed_out.is_set(), marks, lines


def parse_metrics(lines):
    m = {"shorts": None, "unconnected": None, "traces": None,
         "seeds": [], "round_timings": []}
    for line in lines:
        v = RE_VERIFY.search(line)
        if v:
            m["shorts"], m["unconnected"] = int(v.group(1)), int(v.group(2))
            m["traces"] = v.group(3)
        g = RE_GATE_FAIL.search(line)
        if g and m["shorts"] is None:
            m["shorts"], m["unconnected"] = int(g.group(1)), int(g.group(2))
        s = RE_SEED.search(line)
        if s:
            m["seeds"].append(int(s.group(1)))
        t = RE_TIMING.search(line)
        if t:
            m["round_timings"].append(
                {"round": int(t.group(1)), "stage": t.group(2),
                 "s": float(t.group(3))})
    return m


def classify(rc: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    return {
        0: "fab_ready", 5: "synth_fail", 6: "route_fail", 7: "gate_fail",
    }.get(rc, f"error_rc{rc}")


def run_cell(cell, batch_dir: Path, results_path: Path, say) -> dict:
    arm = ARMS[cell["arm"]]
    cell_dir = batch_dir / "cells" / cell["cell"]
    if cell_dir.exists():
        shutil.rmtree(cell_dir)  # re-run of a failed/aborted cell: start clean
    cell_dir.mkdir(parents=True)

    src_state = cell["run_dir"] / ".kicraft" / "state.json"
    state_path = cell_dir / "state.json"
    shutil.copy2(src_state, state_path)
    stem = json.loads(state_path.read_text(encoding="utf-8"))["project_stem"]

    out_dir = cell_dir / "generated"
    project_dir = out_dir / stem
    project_dir.mkdir(parents=True)
    if arm["autoplacer"]:
        (project_dir / "autoplacer.json").write_text(
            json.dumps(arm["autoplacer"], indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # marker-line timestamps need per-line flush
    env.pop("KICRAFT_QUALITY_PRESETS", None)
    if arm["presets"]:
        env["KICRAFT_QUALITY_PRESETS"] = json.dumps(arm["presets"])

    cmd = [str(KICRAFT_BIN), "build", str(state_path), str(out_dir),
           "--quality", arm["quality"], "--no-archive"]
    say(f"[{cell['cell']}] start: quality={arm['quality']} "
        f"presets={arm['presets'] and arm['presets'].get('draft')} "
        f"lite={bool(arm['autoplacer'])} timeout={cell['timeout_s']}s")

    load0 = loadavg()
    t0 = time.monotonic()
    started_at = utcnow()
    rc, timed_out, marks, lines = stream_build(
        cmd, env, cell_dir / "build.log", cell["timeout_s"])
    wall_s = round(time.monotonic() - t0, 1)
    metrics = parse_metrics(lines)

    slot_wait_s = None
    if "wait" in marks and "slot" in marks:
        slot_wait_s = round(marks["slot"] - marks["wait"], 1)
    layout_s = None
    if "layout" in marks:
        layout_end = marks.get("promote") or marks.get("done") or wall_s
        layout_s = round(layout_end - marks["layout"], 1)
    synth_s = None
    if "layout" in marks or "wait" in marks:
        synth_s = round(marks.get("wait", marks.get("layout", 0.0)), 1)

    row = {
        "cell": cell["cell"], "rep": cell["rep"], "board": cell["board"],
        "arm": cell["arm"], "quality": arm["quality"],
        "preset": (arm["presets"] or {}).get("draft"),
        "autoplacer_overrides": arm["autoplacer"],
        "started_at": started_at, "rc": rc,
        "status": classify(rc, timed_out),
        "wall_s": wall_s, "synth_s": synth_s, "layout_s": layout_s,
        "slot_wait_s": slot_wait_s, "marks": marks,
        "shorts": metrics["shorts"], "unconnected": metrics["unconnected"],
        "traces": metrics["traces"], "master_seeds": metrics["seeds"],
        "round_timings": metrics["round_timings"],
        "loadavg_start": load0, "loadavg_end": loadavg(),
        "log": str(cell_dir / "build.log"),
    }
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    say(f"[{cell['cell']}] {row['status']} rc={rc} wall={wall_s}s "
        f"layout={layout_s}s shorts={metrics['shorts']} "
        f"unconnected={metrics['unconnected']}")
    return row


def summarize(batch_dir: Path) -> Path:
    results_path = batch_dir / "results.jsonl"
    rows = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    # Last row per cell wins (re-runs append).
    by_cell = {r["cell"]: r for r in rows if r.get("status") != "harness_error"}
    rows = list(by_cell.values())

    def med(vals):
        return round(statistics.median(vals), 1) if vals else None

    arm_order = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    board_order = [b for b, *_ in BOARDS if any(r["board"] == b for r in rows)]
    lines = [f"# Draft-preset sweep summary -- {batch_dir.name}",
             "",
             f"{len(rows)} cells. Success = `fab_ready` (rc=0: routed, DRC "
             "gate clean, fab package exported).", ""]

    baseline_med: dict[str, float] = {}
    for b in board_order:
        ok = [r["wall_s"] for r in rows
              if r["board"] == b and r["arm"] == "good"
              and r["status"] == "fab_ready"]
        if ok:
            baseline_med[b] = statistics.median(ok)

    lines += ["| arm | board | n | fab_ready | median wall | median layout | "
              "speedup vs good | statuses |",
              "|---|---|---|---|---|---|---|---|"]
    for a in arm_order:
        for b in board_order:
            cell_rows = [r for r in rows if r["arm"] == a and r["board"] == b]
            if not cell_rows:
                continue
            n = len(cell_rows)
            ok_rows = [r for r in cell_rows if r["status"] == "fab_ready"]
            walls = [r["wall_s"] for r in ok_rows]
            layouts = [r["layout_s"] for r in ok_rows if r["layout_s"]]
            speed = ""
            if walls and baseline_med.get(b):
                speed = f"{baseline_med[b] / statistics.median(walls):.2f}x"
            statuses = ",".join(
                f"{r['status']}" for r in sorted(cell_rows,
                                                 key=lambda r: r["rep"]))
            lines.append(
                f"| {a} | {b} | {n} | {len(ok_rows)}/{n} | {med(walls)}s | "
                f"{med(layouts)}s | {speed} | {statuses} |")

    lines += ["", "## Per-arm aggregate (all boards)", "",
              "| arm | n | fab_ready | mean wall (success) | "
              "retry-aware E[wall] |", "|---|---|---|---|---|"]
    for a in arm_order:
        cell_rows = [r for r in rows if r["arm"] == a]
        n = len(cell_rows)
        ok_rows = [r for r in cell_rows if r["status"] == "fab_ready"]
        walls = [r["wall_s"] for r in ok_rows]
        mean_ok = round(statistics.mean(walls), 1) if walls else None
        # E[wall] under "retry once on failure": t + (1-p)*t, failures spend
        # roughly a full attempt (timeouts spend the cap; use observed walls).
        all_walls = [r["wall_s"] for r in cell_rows]
        p = len(ok_rows) / n if n else 0
        ew = None
        if all_walls and mean_ok is not None:
            mean_all = statistics.mean(all_walls)
            ew = round(mean_all + (1 - p) * mean_all, 1)
        lines.append(f"| {a} | {n} | {len(ok_rows)}/{n} ({p:.0%}) | "
                     f"{mean_ok}s | {ew}s |")

    lines += ["", "Decision rule (pre-registered): pick the fastest arm whose "
              "fab_ready count is within 1 of the `good` baseline across the "
              "same boards; tie-break on retry-aware E[wall]. Lite-arm SA "
              "cuts can be overridden by the mutation search; routing "
              "pass caps always bind.", ""]
    out = batch_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", default=None,
                    help="existing batch dir to resume (default: new batch)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="run at most N pending cells (0 = all)")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm filter (default: all)")
    ap.add_argument("--boards", default="",
                    help="comma-separated board filter (default: all)")
    ap.add_argument("--summarize", metavar="DIR", default=None,
                    help="only (re)write summary.md for the given batch dir")
    args = ap.parse_args()

    if args.summarize:
        out = summarize(Path(args.summarize))
        print(f"wrote {out}")
        return 0

    arm_filter = {a for a in args.arms.split(",") if a} or None
    board_filter = {b for b in args.boards.split(",") if b} or None
    if arm_filter and arm_filter - set(ARMS):
        print(f"unknown arms: {arm_filter - set(ARMS)}", file=sys.stderr)
        return 2
    cells = build_matrix(arm_filter, board_filter)

    if args.dry_run:
        for c in cells:
            print(c["cell"], f"timeout={c['timeout_s']}s")
        print(f"{len(cells)} cells total")
        return 0

    batch_dir = (Path(args.batch_dir) if args.batch_dir
                 else SWEEP_ROOT / utcnow())
    batch_dir.mkdir(parents=True, exist_ok=True)
    results_path = batch_dir / "results.jsonl"
    sweep_log = (batch_dir / "sweep.log").open("a", encoding="utf-8")

    def say(msg: str) -> None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
        line = f"{stamp} {msg}"
        print(line, flush=True)
        sweep_log.write(line + "\n")
        sweep_log.flush()

    done = done_cells(results_path)
    pending = [c for c in cells if c["cell"] not in done]
    say(f"batch {batch_dir.name}: {len(cells)} cells, {len(done)} done, "
        f"{len(pending)} pending")
    if args.limit:
        pending = pending[: args.limit]

    for i, cell in enumerate(pending, 1):
        say(f"--- cell {i}/{len(pending)}: {cell['cell']} ---")
        try:
            run_cell(cell, batch_dir, results_path, say)
        except Exception as e:  # noqa: BLE001 -- record and keep sweeping
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"cell": cell["cell"],
                                    "status": "harness_error",
                                    "error": repr(e)}) + "\n")
            say(f"[{cell['cell']}] harness_error: {e!r}")
        try:
            summarize(batch_dir)
        except Exception as e:  # noqa: BLE001
            say(f"summarize failed (non-fatal): {e!r}")

    say("sweep complete")
    out = summarize(batch_dir)
    say(f"summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
