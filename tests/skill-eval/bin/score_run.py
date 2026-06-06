#!/usr/bin/env python3
"""Deterministic (Class-C) scorer for a KiCraft skill-eval run record.

The scoring contract (rubric), the Class-C dimension scorers, the script gates,
and the finalize math now live in the shippable :mod:`kicraft.eval` package, so
the offline harness and the in-app web self-evaluation score identically. This
script is the harness front-end: it builds the metrics dict ``m`` from a harvested
run record (whose run-trace is a ``claude`` ``transcript.jsonl``) and drives the
shared scorer.

Two modes:

  score    Read a run-record dir, compute the deterministic metrics, score the
           five Class-C dimensions against the rubric, fire script-detectable
           gates, and write report.json with the Class-J dimensions left null
           for the observer. Prints a human-readable metrics block + partial
           scorecard. Does NOT produce a final number (Class-J pending).

  finalize Read a report.json whose Class-J levels (and any observer gates) the
           observer has filled in, compute the weighted total, apply every
           triggered gate, assign grade + verdict, and write it back. The final
           number is computed by code (kicraft.eval.scoring), never by the
           observer's mental math.

Run with the repo venv (kicraft + PyYAML)::

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

from kicraft.eval.artifacts import (
    _find_glob,
    _find_one,
    _load_json,
    analyze_state,
    count_generated,
    parse_erc,
    parse_synthesis_check,
)
from kicraft.eval.rubric import load_rubric
from kicraft.eval.scoring import (
    compute_latency_min,
    eval_script_gates,
    finalize_report,
    metrics_block,
    score_class_c_dims,
)

SKILL_EVAL_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# permission floor (claude-only signal; no web analog)
# --------------------------------------------------------------------------- #
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
        if "kicraft synthesize" in line or '"synthesize"' in line:
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


def summarize_token_usage(transcript: dict) -> dict | None:
    """Token totals + estimated cost for the run, or None when unavailable.

    Reuses kicraft.cli.token_report so pricing and requestId de-duplication live
    in one place. Imported lazily and defensively: a missing kicraft install (or
    a parse error) degrades to None, like every other optional signal here,
    rather than crashing the scorer.
    """
    if not transcript.get("present") or not transcript.get("path"):
        return None
    try:
        from kicraft.cli.token_report import summarize_transcripts
    except ImportError:
        print("note: kicraft.cli.token_report not importable; token_usage skipped",
              file=sys.stderr)
        return None
    try:
        return summarize_transcripts([transcript["path"]])
    except (OSError, ValueError) as e:
        print(f"note: token-usage summary failed: {e}", file=sys.stderr)
        return None


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
    token_usage = summarize_token_usage(transcript)
    run_meta = _load_json(_find_one(run_dir, "run.json")) or {}
    band = read_scenario_band(scenario_id or run_meta.get("scenario"))
    return {
        "state": state, "synth": synth, "erc": erc, "generated": generated,
        "perm": perm, "transcript": transcript, "latency": latency,
        "token_usage": token_usage,
        "expected_question_band": band, "run_meta": run_meta,
        "scenario": scenario_id or run_meta.get("scenario"),
        "target_mode": run_meta.get("target_mode"),
    }


def do_score(args) -> int:
    rubric = load_rubric()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")
    m = collect_metrics(run_dir, args.scenario, args.perm_baseline)

    report_dims = score_class_c_dims(m, rubric)
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
        "metrics": metrics_block(m),
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
    print("Class-J dimensions pending observer; grade them, then run: "
          f"score_run.py finalize {out}")
    return 0


# --------------------------------------------------------------------------- #
# finalize
# --------------------------------------------------------------------------- #
def do_finalize(args) -> int:
    rubric = load_rubric()
    report_path = Path(args.report).resolve()
    report = _load_json(report_path)
    if not report:
        sys.exit(f"cannot read report: {report_path}")
    if report.get("rubric_sha256") != rubric["_computed_sha256"]:
        print(f"WARN: report was scored under rubric {report.get('rubric_sha256','?')[:12]} "
              f"but current rubric is {rubric['_computed_sha256'][:12]} (not comparable).")
    try:
        finalize_report(report, rubric)
    except ValueError as e:
        sys.exit(str(e))
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
    tu = m.get("token_usage")
    if tu:
        cost = tu.get("estimated_cost_usd")
        cstr = f"~${cost:,.2f}" if cost is not None else "n/a"
        print(f"  token usage        : {tu['total_tokens']:,} tok over {tu['turns']} call(s)  est {cstr}")
    else:
        print("  token usage        : (no transcript)")

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
