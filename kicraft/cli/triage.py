#!/usr/bin/env python3
"""Run triage for KiCraft investigations (the /kicraft-investigate engine).

Reads the deterministic artifacts a run leaves behind and prints the failure
picture the investigate skill used to assemble by hand. Four subcommands, each
accepting the same run locator (KC-XXXXXX board code, uid/pid, an explicit
path incl. self-eval ``run_NN_*`` dirs, or nothing = the most recent run):

    locate  — resolve the run dir + its accounts.db row
    run     — unified per-run verdict: pipeline stages, ERC, leaves, parent,
              unconnected-net classification, repair records, KiCad Routing Tools
              failure fingerprints, promote provenance
    scan    — cross-run failure-mode ranking (systematic vs per-design)
    audits  — design-quality audits: part-library provenance, BOM realness +
              substitution ledger, LLM wheel-spin, intent adherence

All subcommands take ``--json``. This module owns the artifact-reading logic
that ``.claude/commands/kicraft-investigate.md`` narrates; the tests in
``tests/test_triage_cli.py`` pin it against artifact-schema drift (the old
inline version rotted silently — e.g. testing ``routed_validation is not
None`` on a field that is always a dict).

Everything degrades to a "not present" note instead of raising: a triage of a
half-dead run must never crash on the very artifact gap it is diagnosing.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fab-blocking DRC/verify categories (mirrors cli_app._verify_routed_board +
# the promote tail). Stranding, minor courtyard clips and utilization are
# warnings, NOT blockers.
DRC_COUNT_KEYS = (
    "shorts", "unconnected", "clearance", "annular_width", "padstack",
    "copper_edge_clearance", "courtyard", "items_not_allowed",
)

# The parent_pipeline.json state keys `run`/`scan` read. The drift-guard test
# asserts these against a freshly serialized ParentCompositionState.to_dict()
# so the next compactor/schema change breaks a test instead of the skill.
PARENT_STATE_KEYS = (
    "interconnect_net_names", "candidate_search", "stamp_drc",
    "geometry_validation", "routed_validation", "phase_timings",
    "packing_metadata", "requested_shape", "shape_fit", "manual_outline",
)
ROUTED_VALIDATION_KEYS = (
    "accepted", "rejection_reasons", "drc",
    # conditional (only when the wrapper ran) — see _compact_routed_validation
    "power_first", "post_route_repairs", "signal_unconnected_repair",
    "illegal_geometry_repair",
)


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def _load(path: Path | None):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def _jdict(path: Path | None) -> dict:
    d = _load(path)
    return d if isinstance(d, dict) else {}


def stem_dir(run: Path) -> Path | None:
    gen = run / "generated"
    if not gen.is_dir():
        return None
    return next((p for p in sorted(gen.iterdir()) if p.is_dir()), None)


def experiments_dir(run: Path) -> Path | None:
    sd = stem_dir(run)
    if sd is None:
        return None
    exp = sd / ".experiments"
    return exp if exp.is_dir() else None


def state_path(run: Path) -> Path | None:
    p = run / ".kicraft" / "state.json"
    if p.is_file():
        return p
    return next(run.rglob("state.json"), None)


def _classify_subcircuit_debug(payload: dict) -> str:
    """LEAF (solved here), REPLICA (identical-leaf reuse stub pointing at its
    class solve), or PARENT (the composed-parent artifact)."""
    if "routing_result" in payload or "composition_state" in payload:
        return "parent"
    if "replicated_from" in payload:
        return "replica"
    return "leaf"


def iter_subcircuit_debugs(exp: Path):
    for dbg in sorted(exp.glob("subcircuits/*/debug.json")):
        payload = _jdict(dbg)
        if payload:
            yield dbg, _classify_subcircuit_debug(payload), payload


def parent_rounds(exp: Path) -> list[tuple[Path, dict]]:
    """(path, state) per hierarchical round, in round order."""
    out = []
    for pp in sorted(exp.glob("hierarchical_autoexperiment/round_*/parent_pipeline.json")):
        st = _jdict(pp).get("state")
        if isinstance(st, dict):
            out.append((pp, st))
    return out


def pick_parent_round(rounds: list[tuple[Path, dict]]):
    """The last round that actually produced a routed board, else the last
    attempted. NOTE: routed_validation is ALWAYS a dict (default {}), so the
    only valid emptiness test is truthiness — never ``is not None``."""
    routed = [(pp, st) for pp, st in rounds if st.get("routed_validation")]
    if routed:
        return routed[-1] + ("routed",)
    if rounds:
        return rounds[-1] + ("last_attempted",)
    return None, {}, "none"


# --------------------------------------------------------------------------
# locate
# --------------------------------------------------------------------------

def resolve_projects_dir() -> Path:
    env = os.environ.get("KICRAFT_PROJECTS_DIR")
    if env:
        return Path(env)
    dotenv = REPO_ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(errors="replace").splitlines():
            m = re.match(r"^KICRAFT_PROJECTS_DIR=(.+)$", line.strip())
            if m:
                return Path(m.group(1).strip().strip("\"'"))
    return Path.home() / ".kicraft" / "projects"


def resolve_db_path(projects: Path) -> Path:
    db = projects.parent / "accounts.db"
    return db if db.is_file() else Path.home() / ".kicraft" / "accounts.db"


def db_row_for_run(db: Path, run: Path) -> dict | None:
    if not db.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT id, user_id, board_code, status, quality, created_at, brief "
                "FROM projects WHERE dir_path=?", (str(run),)).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    keys = ("pid", "uid", "board_code", "status", "quality", "created_at", "brief")
    return dict(zip(keys, row))


def resolve_run(arg: str | None) -> tuple[Path | None, str]:
    """Resolve a locator to a run dir. Returns (run, note)."""
    projects = resolve_projects_dir()
    arg = (arg or "").strip()
    if arg.upper().startswith("KC-"):
        db = resolve_db_path(projects)
        if not db.is_file():
            return None, f"board code given but no accounts.db at {db}"
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                row = con.execute(
                    "SELECT dir_path FROM projects WHERE upper(board_code)=upper(?)",
                    (arg,)).fetchone()
            finally:
                con.close()
        except sqlite3.Error as exc:
            return None, f"accounts.db query failed: {exc}"
        if not row or not row[0]:
            return None, f"no project row with board_code {arg} in {db}"
        run = Path(row[0])
        return (run, "board_code") if run.is_dir() else (None, f"dir_path {run} missing on disk")
    if arg:
        p = Path(arg)
        if p.is_dir():
            return p, "explicit path"
        if (projects / arg).is_dir():
            return projects / arg, "uid/pid"
        hits = [d for d in projects.glob(f"*/{arg}") if d.is_dir()]
        if hits:
            return hits[0], "pid match"
        return None, f"could not resolve {arg!r} under {projects}"
    runs = [d for d in projects.glob("*/*") if d.is_dir()]
    if not runs:
        return None, f"no runs under {projects}"
    return max(runs, key=lambda d: d.stat().st_mtime), "most recent"


def cmd_locate(args) -> int:
    run, note = resolve_run(args.target)
    if run is None:
        print(f"ERROR: {note}", file=sys.stderr)
        return 2
    projects = resolve_projects_dir()
    data = {
        "run": str(run), "note": note, "projects": str(projects),
        "db": str(resolve_db_path(projects)),
        "db_row": db_row_for_run(resolve_db_path(projects), run),
    }
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"RUN={run}")
        print(f"PROJECTS={projects}   (resolved via: {note})")
        row = data["db_row"]
        if row:
            print("DB: " + " ".join(f"{k}={row[k]!r}" for k in
                                    ("pid", "uid", "board_code", "status", "quality", "created_at")))
            print(f"brief: {str(row['brief'])[:160]!r}")
    return 0


# --------------------------------------------------------------------------
# run — the unified per-run verdict
# --------------------------------------------------------------------------

def collect_pipeline(run: Path) -> dict:
    ev = run / "events.jsonl"
    out: dict = {"present": ev.is_file(), "stages": [], "build_done": None, "build_log_tail": []}
    if not ev.is_file():
        return out
    for line in ev.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        kind = e.get("kind")
        if kind == "stage_done":
            out["stages"].append((e.get("stage"), "ok" if e.get("ok") else "FAIL"))
        elif kind == "build_done":
            out["build_done"] = e
        elif kind == "build_log":
            out["build_log_tail"].append(e.get("text"))
    out["build_log_tail"] = out["build_log_tail"][-12:]
    return out


def collect_build_meta(run: Path) -> dict | None:
    """.kicraft/build_meta.json — code SHA/branch/quality stamp (absent on
    runs built before the stamp shipped)."""
    meta = _load(run / ".kicraft" / "build_meta.json")
    return meta if isinstance(meta, dict) else None


def collect_erc(run: Path) -> dict:
    rpt = next(run.rglob("*_erc.rpt"), None)
    if rpt is None:
        return {"present": False}
    d = _jdict(rpt)
    errs = []
    for sheet in d.get("sheets", []) or []:
        for v in sheet.get("violations", []) or []:
            if v.get("severity") != "error":
                continue
            items = []
            for it in v.get("items", []) or []:
                pos = it.get("pos") or {}
                items.append({
                    "description": it.get("description", ""),
                    # ERC pos is 1/100 real mm — report real mm (×100)
                    "x_mm": round(pos.get("x", 0) * 100, 2) if pos else None,
                    "y_mm": round(pos.get("y", 0) * 100, 2) if pos else None,
                })
            errs.append({"sheet": sheet.get("path"), "type": v.get("type"),
                         "description": v.get("description"), "items": items})
    return {"present": True, "path": str(rpt), "errors": errs}


def collect_leaves(exp: Path) -> list[dict]:
    leaves = []
    for dbg, kind, payload in iter_subcircuit_debugs(exp):
        if kind == "parent":
            continue
        if kind == "replica":
            leaves.append({
                "name": payload.get("sheet_name") or dbg.parent.name,
                "kind": "replica",
                "replicated_from": payload.get("replicated_from"),
            })
            continue
        ex = payload.get("extra", {}) or {}
        la_struct = ex.get("leaf_acceptance_structured") or {}
        la_raw = ex.get("leaf_acceptance") or {}
        la = la_struct or la_raw
        gates = la.get("gate_results") or {}
        failed_gates = [g for g, r in gates.items()
                        if isinstance(r, dict) and r.get("passed") is False]
        nu = gates.get("no_unconnected") or {}
        name = ((ex.get("solve_summary") or {}).get("sheet_name")
                or (payload.get("metadata") or {}).get("sheet_name")
                or dbg.parent.name)
        best = ex.get("best_round_routing") or {}
        pdiag = best.get("placement_diagnostics") or {}
        leaves.append({
            "name": name,
            "kind": "leaf",
            "accepted": la.get("accepted"),
            "rejection_reasons": la.get("rejection_reasons"),
            "failed_gates": failed_gates or None,
            "round_reasons": (ex.get("failure_summary") or {}).get("unique_reasons"),
            "unconnected": {
                "total": nu.get("unconnected_total"),
                "nets": nu.get("unconnected_nets"),
                "signal": nu.get("signal_unconnected_nets"),
                "ignored_interface": nu.get("ignored_interface_nets"),
                "ignored_poured": nu.get("ignored_poured_nets"),
            } if nu else None,
            "signal_repair": la_raw.get("signal_unconnected_repair"),
            "placement": {
                "median_pin_mm": pdiag.get("median_pin_mm"),
                "max_pin_mm": pdiag.get("max_pin_mm"),
                "grid_guard": (pdiag.get("grid") or {}).get("guard"),
                "place_quality_gate": pdiag.get("place_quality_gate"),
            } if pdiag else None,
        })
    return leaves


def classify_unconnected(nets, interconnect_names) -> dict:
    """Cross-leaf (parent interconnect) vs leaf-internal split of the parent
    board's unconnected nets. When the artifact predates interconnect_net_names
    the split is unknowable — say so instead of guessing.

    Caveat on the leaf-internal bucket: interconnects are INFERRED from leaf
    copper anchors, so a bare cross-leaf pad (a leaf laid 0 copper on a
    single-pad interface net) drops out of the inference and lands here too —
    recall the bare cross-leaf pad class (run_10 GPIO fan-out, pre-2d6329e)
    before concluding a leaf shipped the open."""
    nets = [str(n) for n in (nets or [])]
    if interconnect_names is None:
        return {"cross_leaf": None, "leaf_internal": None, "unclassified": nets,
                "note": "artifact predates interconnect_net_names — split unknown"}
    inter = {str(n) for n in interconnect_names}
    return {
        "cross_leaf": [n for n in nets if n in inter],
        "leaf_internal": [n for n in nets if n not in inter],
        "unclassified": [],
    }


def collect_parent(exp: Path) -> dict:
    rounds = parent_rounds(exp)
    pp, st, how = pick_parent_round(rounds)
    if pp is None:
        return {"present": False, "rounds": 0}
    rv = st.get("routed_validation") or {}
    drc = rv.get("drc") or {}
    cs = st.get("candidate_search") or {}
    gv = st.get("geometry_validation") or {}
    sd = st.get("stamp_drc") or {}
    inter = st.get("interconnect_net_names")
    outside = [c.get("ref") for c in (gv.get("outside_components") or [])][:6]
    out = {
        "present": True,
        "rounds": len(rounds),
        "routed_rounds": sum(1 for _, s in rounds if s.get("routed_validation")),
        "chosen_round": pp.parent.name,
        "chosen_because": how,
        "candidate_search": {k: cs.get(k) for k in (
            "tried", "accepted", "rejected_drc", "best_seed", "winner_refit",
            "edge_pins_demoted", "winner_from_demoted_wave", "shape_fitted")},
        "stamp_drc": {k: sd.get(k) for k in (
            "shorts", "clearance", "copper_edge_clearance", "courtyard")},
        "geometry": {
            "outside_component_count": gv.get("outside_component_count"),
            "outside_pad_count": gv.get("outside_pad_count"),
            "outside_refs": outside or None,
        },
        "phase_timings": st.get("phase_timings") or None,
        "board_metrics": (st.get("packing_metadata") or {}).get("board_metrics"),
        "shape": {
            "requested": (st.get("requested_shape") or {}).get("shape")
            if st.get("requested_shape") else None,
            "shape_fit": st.get("shape_fit"),
            "manual_outline": bool(st.get("manual_outline")),
        },
        "interconnect_net_names": inter,
        "routed_validation": None,
    }
    if rv:
        out["routed_validation"] = {
            "accepted": rv.get("accepted"),
            "rejection_reasons": rv.get("rejection_reasons"),
            "drc_counts": {k: drc.get(k) for k in DRC_COUNT_KEYS},
            "unconnected_nets": drc.get("unconnected_nets"),
            "unconnected_classified": classify_unconnected(
                drc.get("unconnected_nets"), inter),
            "clearance_footprint_refs": drc.get("clearance_footprint_refs"),
            "copper_edge_footprint_refs": drc.get("copper_edge_footprint_refs"),
            "drc_flags": {k: drc.get(k) for k in
                          ("timed_out", "missing_cli", "skipped_routing") if drc.get(k)},
            "repairs": {k: rv.get(k) for k in (
                "post_route_repairs", "signal_unconnected_repair",
                "illegal_geometry_repair") if rv.get(k) is not None},
        }
    return out


def collect_promotion(run: Path) -> dict:
    sd = stem_dir(run)
    if sd is None:
        return {"present": False}
    from kicraft.cli.artifact_paths import read_provenance
    pcb = sd / f"{sd.name}.kicad_pcb"
    prov = read_provenance(pcb) if pcb.is_file() else None
    routed = sorted((sd / ".experiments").glob("**/parent_routed.kicad_pcb")) \
        if (sd / ".experiments").is_dir() else []
    return {
        "present": pcb.is_file(),
        "promoted_pcb": str(pcb) if pcb.is_file() else None,
        "provenance": {k: prov.get(k) for k in
                       ("source_kind", "fresh", "run_id", "promoted_at")} if prov else None,
        "routed_board": str(routed[-1]) if routed else None,
    }


def _run_family(exp, parent, promotion, pipeline) -> str:
    """Best-effort build outcome. Precedence: the build's own verdict
    (build_done / the [build] verify line) over per-round artifacts — a round
    can end dirty and still be healed/superseded before the promote verify."""
    bd = pipeline.get("build_done") or {}
    rc = bd.get("rc")
    if isinstance(rc, int):
        return f"build_done rc={rc}"
    verify = next((t for t in reversed(pipeline.get("build_log_tail") or [])
                   if isinstance(t, str) and "verify:" in t), None)
    if bd.get("ok") is True:
        return "fab-ready (build_done ok)" + (f" — {verify.strip()}" if verify else "")
    if exp is None:
        return "no .experiments -> build never reached layout (rc<=5 family): investigate the schematic"
    rv = (parent.get("routed_validation") or {}) if parent.get("present") else {}
    kind = (promotion.get("provenance") or {}).get("source_kind")
    if promotion.get("routed_board") or kind == "routed":
        if rv.get("accepted"):
            return "routed + accepted (rc0 family)"
        return "routed but DRC-dirty (rc7 family)"
    return ("no routed parent board (rc6 family) — the promoted preview is a "
            f"{kind or 'partial'} board, NOT a routed one")


def collect_run(run: Path) -> dict:
    exp = experiments_dir(run)
    pipeline = collect_pipeline(run)
    data = {
        "run": str(run),
        "pipeline": pipeline,
        "build_meta": collect_build_meta(run),
        "synthesis_check": _jdict(next(run.rglob("synthesis_check.json"), None)) or None,
        "erc": collect_erc(run),
        "leaves": collect_leaves(exp) if exp else [],
        "parent": collect_parent(exp) if exp else {"present": False},
        "promotion": collect_promotion(run),
    }
    data["verdict"] = _run_family(exp, data["parent"], data["promotion"], pipeline)
    return data


def print_run(d: dict) -> None:
    print(f"=== {d['run']} ===")
    pl = d["pipeline"]
    if pl["present"]:
        print("stages:", pl["stages"] or "(none)")
        if pl["build_done"]:
            print("build_done:", json.dumps(pl["build_done"]))
        for t in pl["build_log_tail"]:
            print("   build_log:", t)
    else:
        print("no events.jsonl (run may not have started the pipeline)")
    if d["build_meta"]:
        bm = d["build_meta"]
        print(f"code: sha={bm.get('git_sha')} branch={bm.get('git_branch')} "
              f"quality={bm.get('quality')} started={bm.get('started_at')}")
    sc = d["synthesis_check"]
    if sc:
        print(f"synthesis_check: status={sc.get('status')} failed={sc.get('failed_checks')}")
    erc = d["erc"]
    if erc.get("present"):
        errs = erc["errors"]
        print(f"\nERC errors: {len(errs)}   ({erc['path']})")
        for e in errs[:20]:
            print(f"  [{e['sheet']}] {e['type']}: {e['description']}")
            for it in e["items"]:
                xy = (f"  @ ({it['x_mm']}, {it['y_mm']}) mm"
                      if it["x_mm"] is not None else "")
                print(f"       - {it['description']}{xy}")
    else:
        print("\nNo ERC report — synthesis likely crashed before ERC ran.")

    if d["leaves"]:
        print("\nLEAVES:")
        for lf in d["leaves"]:
            if lf["kind"] == "replica":
                print(f"  [{lf['name']}] replica of {lf['replicated_from']} "
                      "(solved once per class — see its source leaf)")
                continue
            print(f"  [{lf['name']}] accepted={lf['accepted']} "
                  f"reject={lf['rejection_reasons']} failed_gates={lf['failed_gates'] or '-'}")
            nu = lf["unconnected"]
            if nu and nu["total"]:
                print(f"      unconnected: total={nu['total']} signal={nu['signal']} "
                      f"interface(compose-owned)={nu['ignored_interface']} "
                      f"poured={nu['ignored_poured']}")
                print("      router residue — may close on another seed")
            if lf["placement"]:
                p = lf["placement"]
                print(f"      placement: median_pin={p['median_pin_mm']}mm "
                      f"grid_guard={p['grid_guard']} quality_gate={p['place_quality_gate']}")

    par = d["parent"]
    if par.get("present"):
        print(f"\nPARENT [{par['chosen_round']}, {par['chosen_because']}; "
              f"{par['routed_rounds']}/{par['rounds']} round(s) routed]:")
        print(f"  candidate_search: {par['candidate_search']}")
        print(f"  stamp_drc (PRE-route; shorts>0 = composer stamped overlapping copper): "
              f"{par['stamp_drc']}")
        print(f"  geometry: {par['geometry']}")
        if par["board_metrics"]:
            print(f"  board_metrics: {par['board_metrics']}")
        if par["shape"]["requested"] or par["shape"]["manual_outline"]:
            print(f"  shape: {par['shape']}")
        rv = par["routed_validation"]
        if rv:
            print(f"  routed_validation: accepted={rv['accepted']} "
                  f"reasons={rv['rejection_reasons']}")
            print(f"     DRC (real mm): { {k: v for k, v in rv['drc_counts'].items() if v} }")
            cls = rv["unconnected_classified"]
            if rv["unconnected_nets"]:
                print(f"     unconnected nets: cross-leaf={cls['cross_leaf']} "
                      f"not-in-interconnect={cls['leaf_internal']}"
                      + (f"  [{cls['note']}]" if cls.get("note") else ""))
                if cls.get("leaf_internal"):
                    print("       (not-in-interconnect = leaf-internal open OR a bare "
                          "cross-leaf pad that defeated interconnect inference)")
            if rv["clearance_footprint_refs"]:
                print(f"     clearance refs: {rv['clearance_footprint_refs']}")
            if rv["repairs"]:
                print(f"     repairs (evidence the passes ran): "
                      f"{json.dumps(rv['repairs'], default=str)[:600]}")
            if rv["drc_flags"]:
                print(f"     drc flags: {rv['drc_flags']}")
        else:
            print("  routed_validation: EMPTY -> this round never produced a routed board")
    elif d["leaves"]:
        print("\nPARENT: no parent_pipeline.json round found (compose never ran)")

    promo = d["promotion"]
    if promo.get("present"):
        prov = promo["provenance"]
        print(f"\nPROMOTED: {promo['promoted_pcb']}")
        if prov:
            print(f"  provenance: source_kind={prov['source_kind']} fresh={prov['fresh']} "
                  f"run_id={prov['run_id']}")
            if prov["source_kind"] != "routed":
                print("  ** the board on disk is a PARTIAL/placed preview, not a routed "
                      "board — do not judge routing from it **")
        if promo["routed_board"]:
            print(f"  routed board: {promo['routed_board']}")
    print(f"\nVERDICT: {d['verdict']}")


def cmd_run(args) -> int:
    run, note = resolve_run(args.target)
    if run is None:
        print(f"ERROR: {note}", file=sys.stderr)
        return 2
    data = collect_run(run)
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print_run(data)
    return 0


# --------------------------------------------------------------------------
# scan — cross-run failure-mode ranking
# --------------------------------------------------------------------------

def norm_reason(r) -> str:
    """Collapse per-instance payloads so ONE failure family counts as one row:
    refdes, mm offsets, connector-mouth angles, form-factor/outline summaries."""
    r = str(r)
    r = re.sub(r"@-?\d+(\.\d+)?mm(\((left|right|top|bottom)\))?", "", r)
    r = re.sub(r"\(mouth -?\d+(\.\d+)?deg vs \w+ outward -?\d+(\.\d+)?deg\)", "", r)
    r = re.sub(r"^(form-factor non-conformant|outline-shape non-conformant)\s*\(.*\)$",
               r"\1", r)
    return re.sub(r"\b[A-Z]{1,3}\d+\b", "<ref>", r)


def default_scan_roots() -> list[Path]:
    return [resolve_projects_dir(),
            Path.home() / ".kicraft" / "self_eval",
            REPO_ROOT / "logs" / "self_eval"]


def collect_scan(roots: list[Path]) -> dict:
    run_dirs: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for exp in root.rglob(".experiments"):
            run_dirs.add(exp.parent.parent.parent)
        # runs that died before layout still have an ERC report
        for erc in root.rglob("*_erc.rpt"):
            run_dirs.add(erc.parent.parent.parent)
    runs = sorted(run_dirs)
    tier = collections.Counter()
    when: dict[str, str] = {}
    sha: dict[str, str] = {}
    buckets = {k: collections.defaultdict(set) for k in
               ("erc_types", "reject", "drc", "fp_refs", "nets_cross_leaf",
                "nets_leaf_internal", "nets_unclassified", "leaf_reasons")}
    for run in runs:
        tag = f"{run.parent.name}/{run.name}"
        try:
            when[tag] = _dt.date.fromtimestamp(run.stat().st_mtime).isoformat()
        except OSError:
            when[tag] = "?"
        meta = collect_build_meta(run)
        if meta and meta.get("git_sha"):
            sha[tag] = str(meta["git_sha"])[:9]
        erc = collect_erc(run)
        for e in erc.get("errors") or []:
            buckets["erc_types"][e["type"]].add(tag)
        exp = experiments_dir(run)
        if exp is None:
            tier["no layout (rc<=5 family: synth/ERC)"] += 1
            continue
        rounds = parent_rounds(exp)
        _pp, st, _how = pick_parent_round(rounds)
        rv = st.get("routed_validation") or {}
        routed_board = bool(list(exp.glob("**/parent_routed.kicad_pcb")))
        if not routed_board or not rv:
            tier["route_fail (no routed parent, rc6 family)"] += 1
        elif rv.get("accepted") is False:
            tier["dirty (routed, not fab-ready, rc7 family)"] += 1
        elif rv.get("accepted"):
            tier["clean (fab-ready)"] += 1
        else:
            tier["unknown"] += 1
        if rv:
            for r in rv.get("rejection_reasons") or []:
                buckets["reject"][norm_reason(r)].add(tag)
            drc = rv.get("drc") or {}
            for k in DRC_COUNT_KEYS:
                if (drc.get(k) or 0) > 0:
                    buckets["drc"][k].add(tag)
            for ref in drc.get("clearance_footprint_refs") or []:
                buckets["fp_refs"][ref].add(tag)
            cls = classify_unconnected(drc.get("unconnected_nets"),
                                       st.get("interconnect_net_names"))
            for net in cls["cross_leaf"] or []:
                buckets["nets_cross_leaf"][net].add(tag)
            for net in cls["leaf_internal"] or []:
                buckets["nets_leaf_internal"][net].add(tag)
            for net in cls["unclassified"]:
                buckets["nets_unclassified"][net].add(tag)
        for lf in collect_leaves(exp):
            if lf.get("kind") != "leaf":
                continue
            for r in lf.get("round_reasons") or []:
                buckets["leaf_reasons"][norm_reason(r)].add(tag)
    return {"run_count": len(runs), "tiers": dict(tier), "when": when, "sha": sha,
            **{k: {kk: sorted(vv) for kk, vv in v.items()} for k, v in buckets.items()}}


def print_scan(d: dict) -> None:
    print(f"=== CROSS-RUN SCAN: {d['run_count']} runs with layout or ERC artifacts ===")
    print("tiers:", d["tiers"])
    when, sha = d["when"], d["sha"]

    def show(title, bucket):
        rows = sorted(((k, len(v), max((when.get(t, "") for t in v), default="") or "?", v)
                       for k, v in bucket.items()), key=lambda x: -x[1])
        if not rows:
            return
        print(f"\n{title}  (#designs; >1 = SYSTEMATIC; latest = most recent affected run):")
        for k, n, latest, tags in rows:
            eg = tags[:3]
            shas = sorted({sha[t] for t in tags if t in sha})
            extra = f"  sha={','.join(shas)}" if shas else ""
            print(f"  {k}: {n}  latest={latest}{extra}  e.g. {eg}")

    show("ERC error types (>1 design = systematic synthesis-code bug)", d["erc_types"])
    show("parent rejection reasons", d["reject"])
    show("parent DRC error types", d["drc"])
    show("clearance footprint refs (recurring ref = footprint-library bug)", d["fp_refs"])
    show("unconnected CROSS-LEAF nets (in interconnect_net_names: parent "
         "interconnect problem — escapes/corridors/budget)", d["nets_cross_leaf"])
    show("unconnected nets NOT in interconnect_net_names (leaf-internal open — "
         "OR a bare cross-leaf pad that defeated interconnect inference)",
         d["nets_leaf_internal"])
    show("unconnected nets (UNCLASSIFIED — artifact predates interconnect_net_names)",
         d["nets_unclassified"])
    show("leaf failure reasons", d["leaf_reasons"])


def cmd_scan(args) -> int:
    roots = [Path(r) for r in args.roots] if args.roots else default_scan_roots()
    data = collect_scan(roots)
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print_scan(data)
    return 0


# --------------------------------------------------------------------------
# audits — design-quality (run on EVERY investigation, pass or fail)
# --------------------------------------------------------------------------

def collect_library_provenance(run: Path) -> dict:
    """Which tier does each BOM part's symbol/footprint library resolve at:
    project → curated(vendored) → home-fetched → extra → stock-KiCad."""
    st = _jdict(state_path(run))
    parts = (st.get("bom") or {}).get("parts") or []
    sd = stem_dir(run)
    home = Path.home()
    extra = [Path(p) for p in
             os.environ.get("KICRAFT_EXTRA_PARTS_DIRS", "").split(os.pathsep) if p]

    def tier(lib: str, suffix: str, isdir: bool) -> str:
        cands = []
        if sd:
            cands.append(("project", sd / ".kicraft" / "parts" / lib / f"{lib}{suffix}"))
        cands.append(("curated-default", REPO_ROOT / "kicraft" / "parts_library" / lib / f"{lib}{suffix}"))
        cands.append(("home-fetched", home / ".kicraft" / "parts" / lib / f"{lib}{suffix}"))
        for e in extra:
            cands.append(("extra", e / lib / f"{lib}{suffix}"))
        cands.append(("kicad-standard",
                      Path("/usr/share/kicad/" + ("footprints" if isdir else "symbols"))
                      / f"{lib}{suffix}"))
        for t, p in cands:
            if p.is_dir() if isdir else p.is_file():
                return t
        return "UNKNOWN/MISSING"

    rows, flagged = [], []
    counts = collections.Counter()
    for p in parts:
        sym = p.get("symbol") or ""
        fp = p.get("footprint") or ""
        slib = sym.split(":", 1)[0] if ":" in sym else ""
        flib = fp.split(":", 1)[0] if ":" in fp else ""
        stier = tier(slib, ".kicad_sym", False) if slib else "none"
        ftier = tier(flib, ".pretty", True) if flib else "none"
        counts[stier] += 1
        note = None
        if "UNKNOWN/MISSING" in (stier, ftier):
            note = "LIBRARY NOT FOUND (hallucinated/missing — resolver hole)"
            flagged.append((p.get("ref"), "missing-lib", slib or flib))
        elif stier == "home-fetched":
            note = "auto-fetched (curated library lacks it — coverage gap)"
            flagged.append((p.get("ref"), "home-fetched", slib))
        rows.append({"ref": p.get("ref"), "sym_lib": slib, "sym_tier": stier,
                     "fp_lib": flib, "fp_tier": ftier, "note": note})
    return {"rows": rows, "tiers": dict(counts), "flagged": flagged}


def _mpn_related(a: str, b: str) -> bool:
    """Containment either way, tried raw then normalized (separators stripped,
    zero-padded digit groups collapsed) — 'WJ126V-5.0-2P' must match the
    catalog's 'WJ126V-5.0-02P-14-00A', not flag as a wrong part."""
    au, bu = a.upper(), b.upper()
    if au in bu or bu in au:
        return True

    def norm(s: str) -> str:
        s = re.sub(r"[^A-Z0-9]", "", s.upper())
        return re.sub(r"0+(\d)", r"\1", s)

    an, bn = norm(a), norm(b)
    return bool(an and bn and (an in bn or bn in an))


def _cnum_of(part: dict) -> str | None:
    for s in (part.get("symbol") or "", part.get("footprint") or ""):
        m = re.search(r"(?<![A-Za-z0-9])C\d{4,}", s)
        if m:
            return m.group(0)
    m = re.search(r"\bC\d{4,}\b", part.get("sourcing_note") or "")
    return m.group(0) if m else None


def collect_bom_realness(run: Path) -> dict:
    """Pass A: bom_prices.json resolved C#s vs the offline catalog.
    Pass B: explicit-LCSC BOM parts, catalog MPN vs BOM mpn (wrong-part review).
    Pass C: orphan parts (never priced, no C#, library manifest LCSC check).
    Pass D: the substitution ledger + MCU programming path (deterministic)."""
    from kicraft.parts_library import jlcparts
    st = _jdict(state_path(run))
    bom = st.get("bom") or {}
    parts = bom.get("parts") or []
    catalog = jlcparts.available()
    out: dict = {"catalog_present": catalog,
                 "catalog_age_days": jlcparts.dump_age_days() if catalog else None}

    prices = (_jdict(run / ".kicraft" / "bom_prices.json")).get("prices") or {}
    pass_a, suspects = [], 0
    for key, e in sorted(prices.items()):
        if not isinstance(e, dict):
            continue
        cnum = e.get("lcsc")
        cand = jlcparts.lookup(cnum) if (catalog and cnum) else None
        if catalog and cand is None:
            verdict, suspects = "SUSPECT/HALLUCINATED (C# not in catalog)", suspects + 1
        elif cand is None:
            verdict = f"(no catalog) bom_prices stock={e.get('stock')}"
        elif cand.get("stock"):
            verdict = f"REAL stock={cand['stock']} mpn={cand.get('model')!r}"
        else:
            verdict = f"REAL-but-OUT-OF-STOCK mpn={cand.get('model')!r}"
        pass_a.append({"key": key, "lcsc": cnum, "verdict": verdict})
    out["pass_a"] = pass_a
    out["pass_a_suspects"] = suspects

    pass_b = []
    for p in parts:
        c = _cnum_of(p)
        if not c:
            continue
        cand = jlcparts.lookup(c) if catalog else None
        bm = (p.get("mpn") or "").strip()
        if cand is None:
            tag = "SUSPECT (not in catalog)" if catalog else "(no catalog)"
            pass_b.append({"ref": p.get("ref"), "lcsc": c, "bom_mpn": bm, "tag": tag})
            continue
        cm = (cand.get("model") or "").strip()
        if bm and cm and _mpn_related(bm, cm):
            tag = "MATCH"
        elif bm:
            tag = "MPN-MISMATCH — wrong part?"
        else:
            tag = "no-bom-mpn"
        pass_b.append({"ref": p.get("ref"), "lcsc": c, "bom_mpn": bm, "cat_mpn": cm,
                       "stock": cand.get("stock"), "tag": tag,
                       "cat_desc": (cand.get("description") or "")[:90]})
    out["pass_b"] = pass_b

    covered = set(prices.keys()) | {p.get("ref") for p in parts if _cnum_of(p)}
    pass_c = []
    manifest_by_name = {}
    lib_loaded = False
    sd = stem_dir(run)
    try:
        from kicraft.design.library import _load_library_parts
        # project tier resolves under <stem_dir>/.kicraft/parts — pass the
        # SYNTHESIZED PROJECT dir, not the run dir
        active, _broken = _load_library_parts(sd)
        manifest_by_name = {pm.manifest.name: pm.manifest for pm in active}
        lib_loaded = True
    except Exception as exc:  # library optional for triage
        out["library_note"] = f"parts-library manifests unavailable: {exc}"
    for p in parts:
        ref = p.get("ref")
        if ref in covered:
            continue
        sym = p.get("symbol") or ""
        fp = p.get("footprint") or ""
        lib = (sym.split(":", 1)[0] if ":" in sym
               else (fp.split(":", 1)[0] if ":" in fp else ""))
        if not lib:
            pass_c.append({"ref": ref, "tag": "ORPHAN", "note": "no library prefix"})
            continue
        if not lib_loaded:
            pass_c.append({"ref": ref, "tag": "UNVERIFIED", "note": f"library={lib}"})
            continue
        man = manifest_by_name.get(lib)
        lcsc = (getattr(man, "sourcing", None) or {}).get("lcsc") if man else None
        if man is None:
            pass_c.append({"ref": ref, "tag": "ORPHAN", "note": f"no manifest for {lib}"})
        elif not lcsc:
            pass_c.append({"ref": ref, "tag": "ORPHAN",
                           "note": f"{lib} manifest has no sourcing.lcsc"})
        elif catalog and jlcparts.lookup(lcsc) is None:
            pass_c.append({"ref": ref, "tag": "FABRICATED-LCSC",
                           "note": f"{lib} claims {lcsc}, not in catalog"})
        else:
            pass_c.append({"ref": ref, "tag": "LIBRARY-BUNDLE",
                           "note": f"{lib} lcsc={lcsc} (sourceable, not yet priced)"})
    out["pass_c"] = pass_c

    subs = bom.get("substitutions") or []
    out["pass_d"] = {
        "substitutions": subs,
        "ledger_empty": not subs,
        # An MPN mismatch WITH a ledger entry is a recorded, gated deviation;
        # WITHOUT one it is the silent_substitution class.
        "silent_substitution_suspects": [
            r["ref"] for r in pass_b
            if r.get("tag", "").startswith("MPN-MISMATCH") and not subs],
    }
    try:
        from kicraft.design.models import BOM as _BOM
        from kicraft.design.synthesis.validation import mcu_programming_facts
        facts = mcu_programming_facts(_BOM.model_validate(bom)) if bom else None
        out["pass_d"]["mcu_programming"] = facts
    except Exception:
        out["pass_d"]["mcu_programming"] = None
    return out


_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")


def collect_wheel_spin(run: Path) -> dict:
    """Per-LLM-stage convergence: attempts/rounds/tool loops/recurring errors,
    plus the reconcile-death signature (2026-07-27 class)."""
    st = _jdict(state_path(run))
    ss = st.get("stage_status") or {}
    buckets: dict[str, list] = {s: [] for s in _STAGES}
    cur = None
    ev = run / "events.jsonl"
    if ev.is_file():
        for line in ev.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            k = e.get("kind")
            if k == "stage_start":
                cur = e.get("stage")
            elif k == "stage_done":
                cur = None
            elif cur in buckets:
                buckets[cur].append(e)
    stages, stuck = [], []
    for s in _STAGES:
        evs = buckets[s]
        stat = ss.get(s) or {}
        if not evs and not stat:
            continue
        attempts = stat.get("attempts")
        rounds = stat.get("rounds")
        tools = [(e.get("name"), json.dumps(e.get("args", {}), sort_keys=True))
                 for e in evs if e.get("kind") == "tool"]
        top = collections.Counter(tools).most_common(1)
        errc = collections.Counter(
            str(x)[:100] for e in evs if e.get("kind") == "retry"
            for x in (e.get("errors") or []))
        recurring = [x for x, n in errc.items() if n >= 2]
        reconcile_death = [x for x in errc
                           if "reconcile" in x.lower() or "bom deficit" in x.lower()]
        sig = []
        if attempts and attempts >= (4 if s in ("bom", "wiring") else 3):
            sig.append(f"high_attempts({attempts})")
        if top and top[0][1] >= 3:
            sig.append(f"tool_loop({top[0][0][0]}x{top[0][1]})")
        if recurring:
            sig.append(f"recurring_error(x{len(recurring)})")
        if reconcile_death:
            sig.append("reconcile_death_signature")
        if s == "bom" and rounds and rounds >= 6:
            sig.append(f"bom_rounds_maxed({rounds})")
        if sig:
            stuck.append(s)
        stages.append({"stage": s, "attempts": attempts, "rounds": rounds,
                       "tool_calls": stat.get("tool_calls"), "signals": sig,
                       "recurring_errors": recurring[:3],
                       "reconcile_death": reconcile_death[:2]})
    return {"stages": stages, "stuck": stuck}


_FF_STANDARDS = ["arduino", "uno shield", "mega shield", " shield", "raspberry pi",
                 "rpi ", " hat", "feather", "featherwing", "pi zero", "mikrobus",
                 "m.2", "pmod", "eurocard", "din rail", "qwiic form", "stemma"]
_FF_MECH = ["stacking", "stackable", "form factor", "form-factor", "mounting hole",
            "standoff", "fits ", "enclosure", "faceplate", "front panel",
            "board outline", "keep within", "must be exactly", "footprint of a"]


def collect_intent_adherence(run: Path) -> dict:
    """Does the delivered board honor the brief's mechanical contract?
    Both gates (form-factor + outline-shape) now EXIST at promote — the verdict
    distinguishes: standard not captured / captured but enforcement off
    (advisory) / enforced (a non-conformant survivor = gate regression)."""
    st = _jdict(state_path(run))
    brief = ((st.get("intent") or {}).get("brief") or st.get("brief") or "")
    if not brief and (run / "brief.txt").is_file():
        brief = (run / "brief.txt").read_text(errors="replace")
    low = brief.lower()
    hit_std = sorted({s.strip() for s in _FF_STANDARDS if s in low})
    hit_mech = sorted({s.strip() for s in _FF_MECH if s in low})
    dims = re.findall(
        r"\b\d{1,3}\s?(?:\.\d+)?\s?(?:x|×|by)\s?\d{1,3}\s?(?:\.\d+)?\s?mm\b", low)

    ff = (st.get("intent") or {}).get("form_factor") or st.get("form_factor") or {}
    ff_shape = ff.get("shape") if isinstance(ff, dict) else None
    ff_standard = ff.get("standard") if isinstance(ff, dict) else None

    out: dict = {
        "brief_head": brief[:160], "standard_signals": hit_std,
        "mech_signals": hit_mech, "explicit_dims": dims,
        "captured_shape": ff_shape, "captured_standard": ff_standard,
    }
    sd = stem_dir(run)
    pcb = None
    if sd:
        cand = sd / f"{sd.name}.kicad_pcb"
        if cand.is_file():
            pcb = cand
    out["board"] = str(pcb) if pcb else None

    conformance = None
    template = None
    try:
        from kicraft.form_factors import get_template, match_standard
        template = get_template(ff_standard) or match_standard(brief)
        if template is not None and pcb is not None:
            from kicraft.form_factors.conformance import board_local_pads, check_conformance
            pads, wh = board_local_pads(str(pcb))
            rep = check_conformance(template, pads, wh)
            conformance = {"template": template.display_name,
                           "conformant": rep.conformant, "summary": rep.summary()}
    except Exception as exc:
        conformance = {"error": str(exc)}
    out["conformance"] = conformance

    outline = None
    if pcb is not None and ff_shape and str(ff_shape).lower() not in ("", "rect"):
        try:
            from kicraft.eval.outline_check import evaluate_outline_shape
            outline = evaluate_outline_shape(pcb, str(ff_shape))
        except Exception as exc:
            outline = {"error": str(exc)}
    out["outline_shape"] = outline

    enforce = None
    try:
        from kicraft.form_factors.reconcile import enforce_enabled
        enforce = enforce_enabled()
    except Exception:
        pass
    out["enforcement_enabled"] = enforce

    signals = bool(hit_std or hit_mech or dims)
    conf_ok = (conformance or {}).get("conformant")
    if not signals and not ff_standard:
        verdict = "no mechanical-constraint signal (still eyeball interfaces/parts vs brief)"
    elif template is None and ff_standard is None:
        verdict = ("GAP: brief carries a mechanical signal but no standard was "
                   "captured/matched (detection gap)")
    elif conf_ok is False and enforce:
        verdict = ("GAP (gate regression): standard enforced but the delivered "
                   "board is NON-CONFORMANT — the promote gate should have rc7'd this")
    elif conf_ok is False:
        verdict = ("GAP: board NON-CONFORMANT and form-factor enforcement is OFF "
                   "(advisory-only) — invisible to ERC/DRC")
    elif outline and outline.get("level") == 0:
        verdict = "GAP: requested outline shape delivered as rectangular"
    else:
        verdict = "conformant / no unmet mechanical constraint detected"
    out["verdict"] = verdict
    return out


def collect_eval_report(run: Path) -> dict | None:
    """Self-eval runs persist eval/report.json — read its gate list AND the
    observer-screened entries before believing any historical 'gate fired'."""
    rep = _jdict(run / "eval" / "report.json")
    if not rep:
        return None
    gates = rep.get("gates")
    out = {"final": rep.get("final"), "grade": rep.get("grade"), "gates": gates}
    if isinstance(gates, dict):
        out["observer_rejected"] = gates.get("observer_rejected")
        out["gates_rejected"] = gates.get("gates_rejected")
    return out


def collect_audits(run: Path) -> dict:
    return {
        "run": str(run),
        "library_provenance": collect_library_provenance(run),
        "bom_realness": collect_bom_realness(run),
        "wheel_spin": collect_wheel_spin(run),
        "intent_adherence": collect_intent_adherence(run),
        "eval_report": collect_eval_report(run),
    }


def print_audits(d: dict) -> None:
    print(f"=== design-quality audits: {d['run']} ===")
    lp = d["library_provenance"]
    print("\n[A] part-library provenance:")
    for r in lp["rows"]:
        note = f"   <-- {r['note']}" if r["note"] else ""
        print(f"  {str(r['ref']):5s} sym[{r['sym_lib'] or '-'}]={r['sym_tier']:16s} "
              f"fp[{r['fp_lib'] or '-'}]={r['fp_tier']:16s}{note}")
    print("  tiers:", lp["tiers"], " flagged:", lp["flagged"] or "none")

    br = d["bom_realness"]
    age = br["catalog_age_days"]
    print(f"\n[B] BOM realness (catalog={'present' if br['catalog_present'] else 'MISSING'}"
          + (f", dump {age:.0f}d old" + (" — STALE, spot-check on lcsc.com" if age and age > 14 else "")
             if age is not None else "") + "):")
    for r in br["pass_a"]:
        print(f"  A {r['key']:24s} {str(r['lcsc']):11s} {r['verdict']}")
    for r in br["pass_b"]:
        line = f"  B {str(r['ref']):5s} {r['lcsc']:11s} bom_mpn={r['bom_mpn']!r} [{r['tag']}]"
        if r.get("cat_mpn") is not None:
            line += f" cat_mpn={r['cat_mpn']!r} stock={r.get('stock')}"
        print(line)
        if r.get("cat_desc"):
            print(f"        cat desc: {r['cat_desc']}")
    for r in br["pass_c"]:
        print(f"  C {str(r['ref']):5s} {r['tag']}: {r['note']}")
    pd = br["pass_d"]
    if pd["substitutions"]:
        print("  D substitutions ledger (recorded, NOT silent):")
        for s in pd["substitutions"]:
            print(f"      wanted {s.get('wanted')!r} -> got {s.get('got')!r} ({s.get('reason')})")
    else:
        print("  D substitutions ledger: empty")
    if pd["silent_substitution_suspects"]:
        print(f"      ** MPN mismatch with an EMPTY ledger on {pd['silent_substitution_suspects']} "
              "— the silent_substitution class **")
    mcu = pd.get("mcu_programming")
    if mcu:
        ok = mcu.get("access_ok") and mcu.get("path_ok")
        print(f"  D MCU programming path: {'PASS' if ok else 'GAPS'} "
              f"mcus={mcu.get('mcus')} access={mcu.get('access_parts')}")

    ws = d["wheel_spin"]
    print("\n[C] wheel-spin per LLM stage:")
    for s in ws["stages"]:
        print(f"  {s['stage']:16s} attempts={s['attempts']} rounds={s['rounds']} "
              f"tool_calls={s['tool_calls']} -> {', '.join(s['signals']) or 'OK'}")
        for x in s["recurring_errors"]:
            print(f"        recurring: {x}")
        for x in s["reconcile_death"]:
            print(f"        RECONCILE DEATH: {x}")
    print("  stuck stages:", ws["stuck"] or "none")

    ia = d["intent_adherence"]
    print("\n[D] intent adherence:")
    print(f"  brief: {ia['brief_head']!r}")
    print(f"  signals: std={ia['standard_signals'] or '-'} mech={ia['mech_signals'] or '-'} "
          f"dims={ia['explicit_dims'] or '-'}")
    print(f"  captured: shape={ia['captured_shape']!r} standard={ia['captured_standard']!r} "
          f"enforcement={ia['enforcement_enabled']}")
    if ia["conformance"]:
        print(f"  conformance: {ia['conformance']}")
    if ia["outline_shape"]:
        o = ia["outline_shape"]
        print(f"  outline: level={o.get('level')} {o.get('rationale') or o.get('error')}")
    print(f"  VERDICT: {ia['verdict']}")

    er = d["eval_report"]
    if er:
        print(f"\n[E] eval/report.json: final={er['final']} grade={er['grade']}")
        print(f"  gates: {er['gates']}")
        if er.get("observer_rejected"):
            print(f"  observer_rejected (screened false gates — do NOT cite these): "
                  f"{er['observer_rejected']}")


def cmd_audits(args) -> int:
    run, note = resolve_run(args.target)
    if run is None:
        print(f"ERROR: {note}", file=sys.stderr)
        return 2
    data = collect_audits(run)
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print_audits(data)
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="kicraft.cli.triage",
        description="Investigate-run triage: locate / run / scan / audits")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_target(p):
        p.add_argument("target", nargs="?", default=None,
                       help="KC-XXXXXX | uid/pid | run path (default: most recent run)")
        p.add_argument("--json", action="store_true")

    add_target(sub.add_parser("locate", help="resolve a run locator to its dir + DB row"))
    add_target(sub.add_parser("run", help="unified per-run failure verdict"))
    add_target(sub.add_parser("audits", help="design-quality audits (run on every investigation)"))
    ps = sub.add_parser("scan", help="cross-run failure-mode ranking")
    ps.add_argument("roots", nargs="*", help="scan roots (default: projects + self-eval dirs)")
    ps.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    return {"locate": cmd_locate, "run": cmd_run,
            "scan": cmd_scan, "audits": cmd_audits}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
