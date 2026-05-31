#!/usr/bin/env python3
"""Deterministic (Class-C) scorer for a CircuitChat skill-eval run record.

Two modes:

  score    Read a run-record dir, compute the deterministic metrics, score the
           five Class-C dimensions against rubric.yaml, fire script-detectable
           gates, and write report.json with the Class-J dimensions left null
           for the observer. Prints a human-readable metrics block + partial
           scorecard. Does NOT produce a final number (Class-J pending).

  finalize Read a report.json whose Class-J levels (and any observer gates) the
           observer has filled in, compute the weighted total, apply every
           triggered gate, assign grade + verdict, and write it back. The final
           number is computed by code here, never by the observer's mental math.

Run with the repo venv (PyYAML)::

    .venv/bin/python tests/skill-eval/bin/score_run.py score   <run-dir> [--scenario S02] [-o report.json]
    .venv/bin/python tests/skill-eval/bin/score_run.py finalize <report.json>

Signal sources (all optional; the scorer degrades and flags `partial` when a
source is missing):
  - <run-dir>/**/synthesis_check.json   (status, failed_checks, 9.x checks)
  - <run-dir>/**/*_erc.rpt              (KiCad erc.v1.json; severity counts)
  - <run-dir>/**/state.json            (slots, history, open_questions, bom)
  - <run-dir>/**/settings.local.json   (permission-prompt floor)
  - <run-dir>/transcript.jsonl         (latency, #questions, re-commits/aborts)
  - <run-dir>/run.json                 (target mode, scenario, perm baseline)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rubric_hash import load_rubric  # noqa: E402

SKILL_EVAL_DIR = Path(__file__).resolve().parent.parent
CANONICAL_STAGES = 5  # intent, functional_spec, architecture, bom, wiring


# --------------------------------------------------------------------------- #
# artifact discovery + parsing
# --------------------------------------------------------------------------- #
def _find_one(run_dir: Path, name: str) -> Path | None:
    """First match of an exact filename anywhere under run_dir (shallowest wins)."""
    hits = sorted(run_dir.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _find_glob(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.rglob(pattern), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _load_json(path: Path | None):
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def parse_erc(path: Path | None) -> dict:
    """Count error/warning violations from a KiCad erc.v1.json report."""
    if not path or not path.exists():
        return {"present": False, "errors": None, "warnings": None}
    data = _load_json(path)
    if not isinstance(data, dict):
        return {"present": True, "errors": None, "warnings": None, "note": "unparseable"}
    errors = warnings = 0
    for sheet in data.get("sheets", []):
        for v in sheet.get("violations", []):
            sev = v.get("severity")
            if sev == "error":
                errors += 1
            elif sev == "warning":
                warnings += 1
    return {"present": True, "errors": errors, "warnings": warnings}


def parse_synthesis_check(path: Path | None) -> dict:
    data = _load_json(path)
    if not isinstance(data, dict):
        return {"present": False, "status": None, "failed_checks": None, "checked_at": None}
    checks = data.get("checks", []) or []
    failed = data.get("failed_checks")
    if failed is None:
        failed = [c.get("name") for c in checks if c.get("ok") is False]
    return {
        "present": True,
        "status": data.get("status"),
        "failed_checks": failed,
        "failed_count": len(failed),
        "checked_at": data.get("checked_at"),
        "checks": checks,
    }


def analyze_state(path: Path | None) -> dict:
    s = _load_json(path)
    if not isinstance(s, dict):
        return {"present": False}
    bom = s.get("bom") or {}
    connections = bom.get("connections") or []
    history = s.get("history") or []
    slots = {k: s.get(k) is not None for k in ("intent", "functional_spec", "architecture", "bom")}
    return {
        "present": True,
        "slots": slots,
        "all_slots": all(slots.values()),
        "wiring_done": bool(connections),
        "history_len": len(history),
        "history_first_ts": (history[0].get("timestamp") if history else None),
        "open_questions": len(s.get("open_questions") or []),
        "bom_parts": len(bom.get("parts") or []),
        "project_stem": s.get("project_stem"),
    }


def count_generated(run_dir: Path) -> dict:
    pcb = list(run_dir.rglob("*.kicad_pcb"))
    sch = list(run_dir.rglob("*.kicad_sch"))
    pro = list(run_dir.rglob("*.kicad_pro"))
    return {"pcb": len(pcb), "sch": len(sch), "pro": len(pro), "synthesized": bool(pcb or sch)}


def perm_floor(run_dir: Path, baseline: int) -> dict:
    p = _find_one(run_dir, "settings.local.json")
    data = _load_json(p)
    allow = ((data or {}).get("permissions") or {}).get("allow") or []
    n = len(allow)
    return {"present": p is not None, "count": n, "excess": max(0, n - baseline), "entries": allow}


# --------------------------------------------------------------------------- #
# transcript (best-effort, defensive — schema may drift, never crash)
# --------------------------------------------------------------------------- #
def analyze_transcript(run_dir: Path) -> dict:
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        alt = _find_glob(run_dir, "*.jsonl")
        # ignore experiment logs that aren't the chat transcript
        if alt and "experiments" not in str(alt):
            path = alt
    if not path.exists():
        return {"present": False}

    first_ts = last_ts = synth_ts = None
    stage_commit_calls = failed_commits = ask_questions = synth_attempts = crashes = 0
    ts_re = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        m = ts_re.search(line)
        ts = m.group(1) if m else None
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        if "stage-commit" in line:
            stage_commit_calls += line.count("stage-commit")
        if '"ok": false' in line or '"ok":false' in line:
            failed_commits += 1
        if "AskUserQuestion" in line:
            ask_questions += 1
        if "kicraft-circuitchat synthesize" in line or '"synthesize"' in line:
            synth_attempts += 1
            if ts:
                synth_ts = ts
        if "Traceback (most recent call last)" in line or "ModuleNotFoundError" in line:
            crashes += 1

    return {
        "present": True,
        "path": str(path),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "synth_ts": synth_ts,
        "stage_commit_calls": stage_commit_calls,
        "failed_commits": failed_commits,
        "ask_questions": ask_questions,
        "synth_attempts": synth_attempts,
        "crashes": crashes,
    }


def _parse_ts(s: str | None):
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z") and "T" in s and "-" not in s.split("T")[0][4:]:
            # compact UTC like 20260524T225920Z
            return dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_latency_min(transcript: dict, state: dict, synth: dict) -> tuple[float | None, bool]:
    """Return (minutes, is_approximate). Prefer transcript (consistent tz)."""
    if transcript.get("present"):
        a = _parse_ts(transcript.get("first_ts"))
        b = _parse_ts(transcript.get("synth_ts") or transcript.get("last_ts"))
        if a and b and b > a:
            return round((b - a).total_seconds() / 60, 1), False
    # fallback: history start -> synth checked_at (tz-mismatched -> approximate)
    a = _parse_ts(state.get("history_first_ts"))
    b = _parse_ts(synth.get("checked_at"))
    if a and b:
        if a.tzinfo is None:
            a = a.replace(tzinfo=dt.timezone.utc)
        if b.tzinfo is None:
            b = b.replace(tzinfo=dt.timezone.utc)
        mins = (b - a).total_seconds() / 60
        if mins >= 0:
            return round(mins, 1), True
    return None, True


# --------------------------------------------------------------------------- #
# Class-C dimension scorers  ->  (level|None, partial, rationale)
# --------------------------------------------------------------------------- #
def score_pipeline_completion(m) -> tuple[int | None, bool, str]:
    st = m["state"]
    if not st.get("present"):
        return 0, False, "no state.json"
    slots = st["slots"]
    if not any(slots.values()) or (slots.get("intent") and sum(slots.values()) == 1):
        return 0, False, "no slots beyond intent"
    if not (st["all_slots"] and st["wiring_done"]):
        return 1, False, "incomplete: missing a slot or bom.connections"
    if not m["generated"]["synthesized"]:
        return 2, False, "all slots + wiring, but not synthesized"
    status = m["synth"].get("status")
    if status != "ok":
        return 3, False, f"synthesized but synthesis_check.status={status!r}"
    return 4, False, "synthesized, status ok, files present"


def score_computing_cleanliness(m) -> tuple[int | None, bool, str]:
    erc = m["erc"]
    synth = m["synth"]
    tr = m["transcript"]
    errors = erc.get("errors")
    warnings = erc.get("warnings")
    failed = synth.get("failed_count")
    crashed = bool(tr.get("crashes")) if tr.get("present") else False

    if not m["generated"]["synthesized"]:
        if crashed:
            return 0, False, "synthesis-blocking crash (traceback in transcript)"
        return 2, True, "synthesis not reached; cleanliness unconfirmed (partial)"

    # synthesized: prefer ERC counts; fall back to synth_check failures
    if crashed or (errors is not None and errors > 10):
        return 0, False, f"crash={crashed}, erc_errors={errors}"
    if (errors is not None and errors >= 1) or (failed is not None and failed >= 2):
        return 1, False, f"erc_errors={errors}, failed_checks={failed}"
    if failed == 1:
        return 2, False, "exactly 1 failed synthesis check"
    if errors is None and failed is None:
        return 2, True, "synthesized but no ERC/check signal found (partial)"
    if (warnings or 0) > 0:
        return 3, False, f"clean errors/checks; {warnings} ERC warnings"
    return 4, False, "0 errors, 0 failed checks, 0 warnings"


def score_convergence(m) -> tuple[int | None, bool, str]:
    tr = m["transcript"]
    if tr.get("present"):
        err_recommits = tr.get("failed_commits", 0)
        level = {0: 4, 1: 3, 2: 2, 3: 1}.get(err_recommits, 0)
        return level, False, f"{err_recommits} failed/error-driven commit(s) in transcript"
    extra = max(0, m["state"].get("history_len", 0) - CANONICAL_STAGES)
    if extra == 0:
        return 4, True, "history==5 canonical stages; no transcript to confirm (partial)"
    level = max(0, 4 - extra)
    return level, True, f"{extra} extra history commit(s); cannot classify error vs user-driven without transcript (partial)"


def score_latency(m) -> tuple[int | None, bool, str]:
    mins, approx = m["latency"]
    if mins is None:
        return None, True, "no usable timestamps (transcript absent); unscored"
    # The fallback (history -> synth checked_at) is tz-mismatched and, on archived
    # multi-session records, can span days. Don't let an untrustworthy absolute
    # value drive the score — leave it for the observer to read off the transcript.
    if approx and mins > 60:
        return None, True, (f"fallback latency {mins} min implausible "
                            f"(tz-mismatch / multi-session archive); unscored — use transcript")
    for lvl, hi in ((4, 8), (3, 15), (2, 30), (1, 60)):
        if mins <= hi:
            return lvl, approx, f"{mins} min{' (approx, tz-mismatched fallback)' if approx else ''}"
    return 0, approx, f"{mins} min (>60)"


def score_friction(m) -> tuple[int | None, bool, str]:
    tr = m["transcript"]
    perm = m["perm"]
    band = m["expected_question_band"]  # (lo, hi) or None
    excess = perm["excess"]
    q = tr.get("ask_questions") if tr.get("present") else None

    # question component vs band
    q_state = "unknown"
    if q is not None and band is not None:
        lo, hi = band
        if lo <= q <= hi:
            q_state = "in_band"
        elif abs(q - lo) <= 1 or abs(q - hi) <= 1:
            q_state = "near_band"
        else:
            q_state = "out_of_band"

    partial = q is None or band is None
    # combine with permission excess
    if q_state == "out_of_band" and excess > 3:
        return 0, partial, f"questions out of band (asked {q}, band {band}) and {excess} excess prompts"
    if q_state == "out_of_band" or excess > 3:
        return 1, partial, f"q={q} band={band} ({q_state}); excess_prompts={excess}"
    if q_state == "near_band" or 2 <= excess <= 3:
        return 2, partial, f"q={q} band={band} ({q_state}); excess_prompts={excess}"
    if excess <= 1 and q_state in ("in_band", "unknown"):
        if q_state == "in_band" and excess == 0:
            return 4, partial, f"questions in band ({q}), zero excess prompts"
        return 3, partial, f"q={q} band={band} ({q_state}); excess_prompts={excess}"
    return 2, True, f"q={q} band={band} ({q_state}); excess_prompts={excess} (partial)"


CLASS_C_SCORERS = {
    "pipeline_completion": score_pipeline_completion,
    "computing_error_cleanliness": score_computing_cleanliness,
    "convergence_efficiency": score_convergence,
    "latency": score_latency,
    "interaction_friction": score_friction,
}


# --------------------------------------------------------------------------- #
# gates (script-detectable)
# --------------------------------------------------------------------------- #
def eval_script_gates(m, rubric) -> list[dict]:
    fired = []
    caps = {g["id"]: g["cap"] for g in rubric["gates"]}
    erc_errors = m["erc"].get("errors")
    if erc_errors is not None and erc_errors >= 1:
        fired.append({"id": "erc_errors", "cap": caps["erc_errors"], "by": "script",
                      "why": f"{erc_errors} ERC error(s)"})
    # synthesis_broken only on positive evidence of a failed attempt
    tr = m["transcript"]
    attempted = (tr.get("present") and tr.get("synth_attempts", 0) > 0)
    if attempted and not m["generated"]["synthesized"]:
        fired.append({"id": "synthesis_broken", "cap": caps["synthesis_broken"], "by": "script",
                      "why": "synthesize attempted (transcript) but no project files produced"})
    return fired


# --------------------------------------------------------------------------- #
# build / score
# --------------------------------------------------------------------------- #
def read_scenario_band(scenario_id: str | None) -> tuple[int, int] | None:
    if not scenario_id:
        return None
    for p in (SKILL_EVAL_DIR / "scenarios").glob(f"{scenario_id}*.md"):
        txt = p.read_text()
        m = re.search(r"expected_question_band:\s*\[?\s*(\d+)\s*[,\-]\s*(\d+)", txt)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def collect_metrics(run_dir: Path, scenario_id: str | None, perm_baseline: int) -> dict:
    state = analyze_state(_find_one(run_dir, "state.json"))
    synth = parse_synthesis_check(_find_one(run_dir, "synthesis_check.json"))
    erc = parse_erc(_find_glob(run_dir, "*_erc.rpt"))
    generated = count_generated(run_dir)
    perm = perm_floor(run_dir, perm_baseline)
    transcript = analyze_transcript(run_dir)
    latency = compute_latency_min(transcript, state, synth)
    run_meta = _load_json(_find_one(run_dir, "run.json")) or {}
    band = read_scenario_band(scenario_id or run_meta.get("scenario"))
    return {
        "state": state, "synth": synth, "erc": erc, "generated": generated,
        "perm": perm, "transcript": transcript, "latency": latency,
        "expected_question_band": band, "run_meta": run_meta,
        "scenario": scenario_id or run_meta.get("scenario"),
        "target_mode": run_meta.get("target_mode"),
    }


def dim_by_id(rubric):
    return {d["id"]: d for d in rubric["dimensions"]}


def do_score(args) -> int:
    rubric = load_rubric()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")
    m = collect_metrics(run_dir, args.scenario, args.perm_baseline)
    dims = dim_by_id(rubric)

    report_dims = {}
    for d in rubric["dimensions"]:
        did = d["id"]
        if d["class"] == "C":
            level, partial, why = CLASS_C_SCORERS[did](m)
            report_dims[did] = {"class": "C", "weight": d["weight"], "level": level,
                                "partial": partial, "by": "script", "rationale": why}
        else:
            report_dims[did] = {"class": "J", "weight": d["weight"], "level": None,
                                "partial": False, "by": "observer", "rationale": ""}

    gates = eval_script_gates(m, rubric)
    pending = [k for k, v in report_dims.items() if v["level"] is None]

    report = {
        "scenario": m["scenario"],
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "scored_at": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "rubric_version": rubric["meta"]["version"],
        "rubric_sha256": rubric["_computed_sha256"],
        "target_mode": m["target_mode"],
        "metrics": _public_metrics(m),
        "dimensions": report_dims,
        "gates": {"triggered": gates, "observer_todo": [g["id"] for g in rubric["gates"]
                                                          if g["detected_by"] == "observer"]},
        "score": {"weighted": None, "final": None, "grade": None, "verdict": None,
                  "pending_dimensions": pending,
                  "note": "Class-J dimensions pending observer; run `finalize` after grading."},
    }

    out = Path(args.output) if args.output else run_dir / "report.json"
    out.write_text(json.dumps(report, indent=2))
    _print_score_summary(report, rubric)
    print(f"\nwrote {out}")
    print("Class-J dimensions pending observer — grade them, then run: "
          f"score_run.py finalize {out}")
    return 0


def _public_metrics(m) -> dict:
    tr, st, synth, erc = m["transcript"], m["state"], m["synth"], m["erc"]
    return {
        "synthesized": m["generated"]["synthesized"],
        "generated_files": m["generated"],
        "synthesis_status": synth.get("status"),
        "failed_checks": synth.get("failed_checks"),
        "erc_errors": erc.get("errors"),
        "erc_warnings": erc.get("warnings"),
        "latency_min": m["latency"][0],
        "latency_approx": m["latency"][1],
        "user_questions": tr.get("ask_questions") if tr.get("present") else None,
        "stage_commit_calls": tr.get("stage_commit_calls") if tr.get("present") else None,
        "failed_commits": tr.get("failed_commits") if tr.get("present") else None,
        "crashes": tr.get("crashes") if tr.get("present") else None,
        "history_len": st.get("history_len"),
        "open_questions": st.get("open_questions"),
        "bom_parts": st.get("bom_parts"),
        "permission_floor": m["perm"]["count"],
        "permission_excess": m["perm"]["excess"],
        "expected_question_band": m["expected_question_band"],
        "transcript_present": tr.get("present", False),
    }


# --------------------------------------------------------------------------- #
# finalize
# --------------------------------------------------------------------------- #
def grade_for(score: float, rubric) -> dict:
    for band in rubric["bands"]:
        if score >= band["min"]:
            return {"grade": band["grade"], "verdict": band["verdict"]}
    return {"grade": "F", "verdict": "BROKEN"}


def do_finalize(args) -> int:
    rubric = load_rubric()
    report_path = Path(args.report).resolve()
    report = _load_json(report_path)
    if not report:
        sys.exit(f"cannot read report: {report_path}")
    if report.get("rubric_sha256") != rubric["_computed_sha256"]:
        print(f"WARN: report was scored under rubric {report.get('rubric_sha256','?')[:12]} "
              f"but current rubric is {rubric['_computed_sha256'][:12]} — not comparable.")

    dims = report["dimensions"]
    missing = [k for k, v in dims.items() if v.get("level") is None]
    if missing:
        sys.exit(f"cannot finalize: {len(missing)} dimension(s) ungraded: {', '.join(missing)}")

    points = sum(v["weight"] * v["level"] / 4 for v in dims.values())
    weighted = round(points, 1)

    # gates: script-fired + any observer-fired present in report
    fired = list(report.get("gates", {}).get("triggered", []))
    caps = [g["cap"] for g in fired]
    final = round(min([weighted] + caps), 1)
    g = grade_for(final, rubric)

    report["score"] = {
        "weighted": weighted,
        "final": final,
        "grade": g["grade"],
        "verdict": g["verdict"],
        "gates_applied": [{"id": x["id"], "cap": x["cap"]} for x in fired],
        "pending_dimensions": [],
    }
    report["finalized_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path.write_text(json.dumps(report, indent=2))
    _print_score_summary(report, rubric, final=True)
    print(f"\nfinalized {report_path}")
    return 0


# --------------------------------------------------------------------------- #
# pretty printing
# --------------------------------------------------------------------------- #
def _print_score_summary(report, rubric, final=False):
    m = report["metrics"]
    print(f"\n=== RUN {report['run_id']}  scenario={report.get('scenario')}  "
          f"mode={report.get('target_mode')} ===")
    print(f"rubric v{report['rubric_version']}  sha256:{report['rubric_sha256'][:16]}…")
    print("\n-- deterministic metrics --")
    print(f"  synthesized        : {m['synthesized']}  (status={m['synthesis_status']})")
    print(f"  ERC                : {m['erc_errors']} errors / {m['erc_warnings']} warnings")
    print(f"  failed checks      : {m['failed_checks']}")
    print(f"  latency            : {m['latency_min']} min"
          f"{' (approx)' if m['latency_approx'] else ''}")
    print(f"  user questions     : {m['user_questions']}  (band {m['expected_question_band']})")
    print(f"  stage-commit refs  : {m['stage_commit_calls']} (raw mention count)  "
          f"failed_commits={m['failed_commits']}  crashes={m['crashes']}")
    print(f"  history / open_q   : {m['history_len']} / {m['open_questions']}   bom_parts={m['bom_parts']}")
    print(f"  permission floor   : {m['permission_floor']} (excess {m['permission_excess']})")
    print(f"  transcript present : {m['transcript_present']}")

    print("\n-- scorecard --")
    print(f"  {'dimension':28} {'cls':3} {'wt':>3} {'lvl':>3} {'pts':>5}")
    scored = 0.0
    scored_wt = 0
    for did, v in report["dimensions"].items():
        lvl = v["level"]
        pts = (v["weight"] * lvl / 4) if lvl is not None else None
        if lvl is not None:
            scored += pts
            scored_wt += v["weight"]
        flag = " (partial)" if v.get("partial") else ""
        print(f"  {did:28} {v['class']:3} {v['weight']:>3} "
              f"{('-' if lvl is None else lvl):>3} {('  -' if pts is None else f'{pts:5.1f}')}{flag}")
    print(f"  {'-'*48}")
    if final:
        s = report["score"]
        gates = ", ".join(f"{x['id']}≤{x['cap']}" for x in s.get("gates_applied", [])) or "none"
        print(f"  weighted={s['weighted']}  gates[{gates}]  FINAL={s['final']}  "
              f"grade {s['grade']}  {s['verdict']}")
    else:
        print(f"  Class-C scored: {scored:.1f} / {scored_wt} available pts "
              f"(Class-J + final pending)")
        gates = report.get("gates", {}).get("triggered", [])
        if gates:
            print(f"  script gates fired: " + ", ".join(f"{g['id']}≤{g['cap']} ({g['why']})" for g in gates))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score Class-C from a run-record dir")
    sc.add_argument("run_dir")
    sc.add_argument("--scenario", help="scenario id (e.g. S02) to read expected_question_band")
    sc.add_argument("--perm-baseline", type=int, default=0,
                    help="permission entries considered baseline (default 0)")
    sc.add_argument("-o", "--output", help="report.json path (default <run-dir>/report.json)")
    sc.set_defaults(func=do_score)
    fi = sub.add_parser("finalize", help="compute final score from a fully-graded report.json")
    fi.add_argument("report")
    fi.set_defaults(func=do_finalize)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
