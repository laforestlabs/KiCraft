"""Read a tuning run's artifacts into chart-ready series (no UI deps).

Kept separate from the web layer so it is unit-testable: it turns a run dir
(``tuning.db`` + ``checkpoint.json`` + ``report.json``/``screen.json``) into plain
dicts/lists the admin page feeds straight into ECharts. Opens the live sqlite DB
read-only so it never contends with a running tuner (WAL allows concurrent
readers).

Two views the admin page wants:
* time series — per generation, the gen-best train metrics (J, fab-ready rate,
  DRC, wall-time) and the holdout-monitored metrics, so you see results improve;
* parameter convergence — each active param's value in the gen-best config over
  generations (raw + normalized to [0,1]), so you see the search settle.
Plus the Pareto archive (every evaluated config's 3 objectives, non-dominated
flagged) and the baseline (current DEFAULT_CONFIG) for reference.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from kicraft.tuning import space
from kicraft.tuning.reward import CorpusObjectives, pareto_front
from kicraft.tuning.store import config_hash

# A self-contained, chart-ready payload the tuner writes each generation. It is
# the ONLY file that needs to travel to a remote viewer (e.g. the cloud admin
# page): tiny JSON, no live sqlite to sync. The local DB is the source of truth;
# progress.json is its published snapshot.
PROGRESS_NAME = "progress.json"
RUN_MARKERS = (PROGRESS_NAME, "tuning.db", "checkpoint.json", "report.json")


def discover_runs(roots: list[str | Path]) -> list[Path]:
    """Tuning run dirs under ``roots`` (any with a run marker), newest first."""
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            key = str(d.resolve())
            if key in seen:
                continue
            if any((d / m).is_file() for m in RUN_MARKERS):
                seen.add(key)
                out.append(d)
    out.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return out


def _connect_ro(db: Path) -> sqlite3.Connection | None:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        try:  # fallback: plain read (WAL tolerates concurrent readers)
            conn = sqlite3.connect(str(db), timeout=5.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error:
            return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _active_params(run_dir: Path, report: dict | None, checkpoint: dict | None) -> list[str]:
    if report and report.get("active_params"):
        return list(report["active_params"])
    if checkpoint and checkpoint.get("active"):
        return list(checkpoint["active"])
    sj = _read_json(run_dir / "screen.json")
    if sj and sj.get("active"):
        return list(sj["active"])
    return []


def run_overview(run_dir: Path) -> dict:
    """Cheap headline stats for the runs LIST (no full chart build)."""
    run_dir = Path(run_dir)
    # Prefer a published payload (covers a synced remote run with no local DB).
    pub = _read_json(run_dir / PROGRESS_NAME)
    if pub:
        pts = pub.get("points", [])
        finished = bool(pub.get("finished"))
        n_gens = int(pub.get("n_gens", 0))
        return {
            "path": str(run_dir), "name": run_dir.name,
            "mtime": run_dir.stat().st_mtime, "gen": n_gens,
            "n_configs": int(pub.get("n_configs", 0)),
            "scalarization": pub.get("scalarization"),
            "baseline_fab": (pub.get("baseline") or {}).get("fab"),
            "best_fab": (max((p.get("fab", 0.0) for p in pts), default=None)
                         if pts else None),
            "finished": finished, "running": n_gens > 0 and not finished,
        }
    checkpoint = _read_json(run_dir / "checkpoint.json")
    report = _read_json(run_dir / "report.json")
    gen = int(checkpoint.get("gen", 0)) if checkpoint else 0
    archive = (checkpoint or {}).get("archive", [])
    baseline = next((a for a in archive if a.get("baseline")), None)
    best_fab = max((a.get("fab", 0.0) for a in archive), default=None) if archive else None
    return {
        "path": str(run_dir),
        "name": run_dir.name,
        "mtime": run_dir.stat().st_mtime,
        "gen": gen,
        "n_configs": len({a.get("hash") for a in archive}) if archive else 0,
        "scalarization": (checkpoint or report or {}).get("scalarization"),
        "baseline_fab": (baseline or {}).get("fab"),
        "best_fab": best_fab,
        "finished": (run_dir / "report.json").is_file(),
        "running": gen > 0 and not (run_dir / "report.json").is_file(),
    }


def load_run(run_dir: str | Path) -> dict:
    """Chart-ready payload for one run: the published snapshot if present
    (a synced remote run), otherwise computed live from the local DB."""
    run_dir = Path(run_dir)
    pub = _read_json(run_dir / PROGRESS_NAME)
    if pub:
        return pub
    return build_payload(run_dir)


def publish(run_dir: str | Path) -> Path:
    """Write the chart payload to ``progress.json`` (atomic). Called by the tuner
    each generation; the result is the single small file a remote viewer needs."""
    run_dir = Path(run_dir)
    payload = build_payload(run_dir)
    tmp = run_dir / (PROGRESS_NAME + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, run_dir / PROGRESS_NAME)
    return run_dir / PROGRESS_NAME


def build_payload(run_dir: str | Path) -> dict:
    """Compute the full chart-ready payload from the local DB + checkpoint."""
    run_dir = Path(run_dir)
    checkpoint = _read_json(run_dir / "checkpoint.json")
    report = _read_json(run_dir / "report.json")
    active = _active_params(run_dir, report, checkpoint)

    archive = (checkpoint or {}).get("archive", [])
    by_hash: dict[str, dict] = {a["hash"]: a for a in archive if "hash" in a}
    baseline_entry = next((a for a in archive if a.get("baseline")), None)
    baseline_hash = config_hash({})

    # --- time series from the generations table -------------------------
    gens: list[dict] = []
    overlays_by_hash: dict[str, dict] = {}
    db = run_dir / "tuning.db"
    conn = _connect_ro(db) if db.is_file() else None
    if conn is not None:
        try:
            train_rows: dict[int, dict] = {}
            for r in conn.execute(
                "SELECT gen, config_hash, j, fab_ready_rate, mean_drc, mean_wall_s "
                "FROM generations WHERE is_train=1 ORDER BY gen, j"
            ):
                # last row per gen wins => the max-j (best) candidate of that gen
                train_rows[r["gen"]] = {
                    "hash": r["config_hash"], "j": r["j"],
                    "fab": r["fab_ready_rate"], "drc": r["mean_drc"],
                    "wall": r["mean_wall_s"],
                }
            hold_rows: dict[int, dict] = {}
            for r in conn.execute(
                "SELECT gen, j, fab_ready_rate, mean_drc, mean_wall_s "
                "FROM generations WHERE is_train=0 ORDER BY gen"
            ):
                hold_rows[r["gen"]] = {
                    "j": r["j"], "fab": r["fab_ready_rate"],
                    "drc": r["mean_drc"], "wall": r["mean_wall_s"],
                }
            for g in sorted(set(train_rows) | set(hold_rows)):
                gens.append({"gen": g, "train": train_rows.get(g),
                             "holdout": hold_rows.get(g)})
            for row in conn.execute("SELECT config_hash, overlay_json FROM configs"):
                try:
                    overlays_by_hash[row["config_hash"]] = json.loads(row["overlay_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    # --- parameter convergence: gen-best overlay per generation ---------
    param_traces: dict[str, list[dict]] = {p: [] for p in active}
    for gd in gens:
        tb = gd.get("train")
        if not tb:
            continue
        overlay = overlays_by_hash.get(tb["hash"]) or by_hash.get(tb["hash"], {}).get("overlay", {})
        for p in active:
            if p in overlay:
                try:
                    val = float(overlay[p])
                    param_traces[p].append(
                        {"gen": gd["gen"], "value": val, "norm": space.normalize(p, val)})
                except (ValueError, KeyError):
                    pass

    # --- Pareto archive (all evaluated configs) -------------------------
    points: list[dict] = []
    for h, a in by_hash.items():
        if not all(k in a for k in ("fab", "drc", "wall")):
            continue
        points.append({"hash": h, "fab": a["fab"], "drc": a["drc"],
                       "wall": a["wall"],
                       "baseline": bool(a.get("baseline")) or h == baseline_hash})
    front_idx = set(pareto_front([
        CorpusObjectives(p["fab"], p["drc"], p["wall"], p["fab"], 1) for p in points
    ])) if points else set()
    for i, p in enumerate(points):
        p["front"] = i in front_idx

    baseline = None
    if baseline_entry:
        baseline = {"fab": baseline_entry.get("fab"), "drc": baseline_entry.get("drc"),
                    "wall": baseline_entry.get("wall")}

    return {
        "run_dir": str(run_dir), "name": run_dir.name,
        "run_id": (checkpoint or report or {}).get("run_id"),
        "scalarization": (checkpoint or report or {}).get("scalarization"),
        "active_params": active,
        "n_train": (report or {}).get("n_train"),
        "n_holdout": (report or {}).get("n_holdout"),
        "n_gens": len(gens),
        "n_configs": len(points),
        "finished": (run_dir / "report.json").is_file(),
        "baseline": baseline,
        "gens": gens,
        "param_traces": param_traces,
        "points": points,
        "defaults": {p: space.default_value(p) for p in active},
    }
