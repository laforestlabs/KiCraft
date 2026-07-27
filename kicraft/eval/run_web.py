"""Evaluate a finished KiCraft web design project end to end.

Ties the pieces together for the in-app, admin-only self-evaluation:

    collect_web_metrics  ->  Class-C scorers + script gates   (deterministic)
    build_run_digest     ->  grade_class_j (LLM judge)        (judgment)
    finalize_report      ->  weighted, gate-capped, graded

and persists the result to ``<project_dir>/eval/report.json`` (the same
``report.schema.json`` shape the harness uses) so it is durable and re-viewable
without re-running the judge.

``evaluate_project`` takes an injected client (the web app passes its capped
OpenRouter client), so this module imports the server only inside ``main`` (the
``kicraft-eval-web`` CLI). ``--no-judge`` scores Class-C only and needs no network
or API key, which is the headless verification path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path

from .artifacts import _find_one, _load_json
from .judge import grade_class_j
from .metrics_web import collect_web_metrics
from .rubric import load_rubric
from .scoring import eval_script_gates, finalize_report, metrics_block, score_class_c_dims


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_digest(project_dir, m, *, budget: int = 16000) -> str:
    """A compact, evidence-only text digest for the judge: the brief, the whole
    committed design state (minus the noisy history log), and the pipeline result.
    Dumping the state wholesale (rather than cherry-picking fields) keeps the judge
    from missing a constraint the design recorded somewhere unexpected."""
    pd = Path(project_dir)
    parts: list[str] = []

    brief = pd / "brief.txt"
    if brief.is_file():
        text = brief.read_text(errors="replace").strip()
        if text:
            parts.append("BRIEF (what the user asked for):\n" + text[:2000])

    state = _load_json(_find_one(pd, "state.json"))
    if isinstance(state, dict):
        trimmed = {k: v for k, v in state.items() if k != "history"}
        parts.append("COMMITTED DESIGN STATE (intent, spec, architecture, bom, wiring, "
                     "assumptions, open_questions):\n"
                     + json.dumps(trimmed, indent=2, default=str)[:budget])

    synth, erc, tr, gen = m["synth"], m["erc"], m["transcript"], m["generated"]
    # Silk-legend evidence rides as a deterministic line: the state dump above
    # is budget-truncated and ``artifacts`` (serialized last) is routinely cut,
    # which zeroed board_self_description on 21/34 runs of the 2026-07-17
    # batch for "no evidence" the state actually held (fix-plan T6).
    silk_line = ""
    if isinstance(state, dict) and isinstance(state.get("artifacts"), dict):
        arts = state["artifacts"]
        placed = arts.get("silk_placed")
        dropped = arts.get("silk_dropped")
        if placed is not None or dropped is not None:
            silk_line = (
                f"\n  silk legend: placed={placed or []} dropped={dropped or []}"
            )
    # Regulator feedback math, computed (not judged): the judge model
    # hallucinated a TPS5430 Vref of 0.8 V (real: 1.221 V) and failed a
    # correct 3.3 V design in the 2026-07-17 batch. Handing it the
    # deterministic number pre-empts the guess (fix-plan T8).
    reg_line = ""
    try:
        from kicraft.design.synthesis.validation import regulator_vout_facts
        bom = state.get("bom") if isinstance(state, dict) else None
        if isinstance(bom, dict):
            facts = regulator_vout_facts(
                bom.get("parts") or [], bom.get("connections") or []
            )
            if facts:
                reg_line = "\n  regulator feedback (computed, authoritative): " + "; ".join(
                    f"{f['ref']} {f['mpn']} Vref={f['vref']}V divider "
                    f"{f['r_top_ref']}/{f['r_bot_ref']} -> Vout={f['vout']}V "
                    f"on net {f['rail_net']!r}"
                    + ("" if f["ok"] is None
                       else (" (matches rail)" if f["ok"]
                             else f" (MISMATCH vs {f['rail_v']}V rail)"))
                    for f in facts
                )
    except Exception:
        reg_line = ""
    # Substitution ledger, surfaced deterministically (2026-07-27 fix-plan
    # P2.5): the silent_substitution gate fires on UNSURFACED swaps, so the
    # judge must see what IS on the record even when the state dump above is
    # budget-truncated.
    sub_line = ""
    try:
        bom = state.get("bom") if isinstance(state, dict) else None
        subs = (bom or {}).get("substitutions") or []
        if subs:
            sub_line = (
                "\n  substitutions (recorded by the design, NOT silent): "
                + "; ".join(
                    f"wanted {s.get('wanted')!r} -> shipped {s.get('got')!r}"
                    + (f" ({s.get('reason')})" if s.get("reason") else "")
                    for s in subs if isinstance(s, dict)
                )
            )
        elif isinstance(bom, dict):
            sub_line = "\n  substitutions ledger: empty (no recorded deviations)"
    except Exception:
        sub_line = ""
    # MCU programming path, computed (not judged): the judge over-fired
    # unprogrammable_mcu on boards §9.29 deliberately accepts (BOOTSEL+USB is
    # the RP2040 ROM UF2 path; a UPDI TP pad satisfies a no-connectors brief)
    # because the digest never carried the deterministic verdict (2026-07-27
    # runs 10/31).
    prog_line = ""
    try:
        from kicraft.design.models import BOM as _BOM
        from kicraft.design.synthesis.validation import mcu_programming_facts
        bom = state.get("bom") if isinstance(state, dict) else None
        if isinstance(bom, dict):
            facts = mcu_programming_facts(_BOM.model_validate(bom))
            if facts:
                verdict = ("PASS -- a workable first-flash path exists"
                           if facts["access_ok"] and facts["path_ok"]
                           else "GAPS: " + "; ".join(
                               facts["access_problems"] + facts["path_problems"]))
                prog_line = (
                    "\n  MCU programming path (computed, authoritative): "
                    f"{verdict}; MCU(s): {', '.join(facts['mcus'])}; "
                    "programming-access parts: "
                    + (", ".join(facts["access_parts"]) or "NONE")
                )
    except Exception:
        prog_line = ""
    parts.append(
        "PIPELINE RESULT (deterministic facts):\n"
        f"  synthesized: {gen['synthesized']} (pcb={gen['pcb']} sch={gen['sch']})\n"
        f"  synthesis_check.status: {synth.get('status')}; failed_checks: {synth.get('failed_checks')}\n"
        f"  ERC: {erc.get('errors')} error(s) / {erc.get('warnings')} warning(s)\n"
        f"  run-trace: {tr.get('failed_commits')} error-driven re-commit(s), "
        f"{tr.get('ask_questions')} clarifying question(s), crashes={tr.get('crashes')}"
        + silk_line
        + reg_line
        + sub_line
        + prog_line
    )
    return "\n\n".join(parts)


def evaluate_project(project_dir, client, *, rubric: dict | None = None,
                     judge_model: str | None = None, judge_client=None,
                     judge_max_tokens: int | None = None,
                     ledger_path=None,
                     started_at: str | None = None, finished_at: str | None = None,
                     skip_judge: bool = False) -> dict:
    """Score one finished web project and write ``eval/report.json``.

    Class-C is always scored from artifacts. Class-J is graded unless
    ``skip_judge`` (or no client) is given, in which case the judgment dimensions
    stay null and the run is not finalized (Class-C only). The judge uses
    ``judge_client`` when supplied (a client with routing relaxed for a stronger,
    steadier judge model that may be off the design provider tier), else
    ``client``.
    """
    rubric = rubric or load_rubric()
    pd = Path(project_dir)

    m = collect_web_metrics(pd, ledger_path=ledger_path,
                            started_at=started_at, finished_at=finished_at)
    dims = score_class_c_dims(m, rubric)
    gates = eval_script_gates(m, rubric)

    judge = None
    if not skip_judge and client is not None:
        digest = build_run_digest(pd, m)
        jkw = {"max_tokens": judge_max_tokens} if judge_max_tokens else {}
        judge = grade_class_j(judge_client or client, digest, rubric, model=judge_model, **jkw)
        for did, jv in judge["dimensions"].items():
            if did in dims:
                dims[did]["level"] = jv["level"]
                dims[did]["rationale"] = jv.get("evidence", "")
                dims[did]["by"] = "observer"  # the automated judge plays the observer role
        have = {g["id"] for g in gates}
        for g in judge["gates"]:
            if g["id"] not in have:
                gates.append(g)
                have.add(g["id"])

    report = {
        "scenario": None,
        "run_id": pd.name,
        "run_dir": str(pd),
        "scored_at": _now(),
        "rubric_version": rubric["meta"]["version"],
        "rubric_sha256": rubric["_computed_sha256"],
        "target_mode": "web",
        "metrics": metrics_block(m),
        "dimensions": dims,
        "gates": {"triggered": gates,
                  "observer_rejected": (judge or {}).get("gates_rejected") or [],
                  "observer_todo": []},
        "judge": {
            "ran": judge is not None,
            "ok": (judge["ok"] if judge else None),
            "model": judge_model,
            "error": (judge["error"] if judge else None),
            "cost_usd": (round(judge["cost_usd"], 6) if judge else None),
        },
        "score": {"weighted": None, "final": None, "grade": None, "verdict": None,
                  "pending_dimensions": [k for k, v in dims.items() if v["level"] is None],
                  "note": ""},
    }

    if judge is not None and judge["ok"]:
        try:
            finalize_report(report, rubric)
        except ValueError as e:  # a Class-C dim came back unscored (e.g. latency)
            report["score"]["note"] = f"not finalized: {e}"
    elif judge is not None and not judge["ok"]:
        report["score"]["note"] = (f"Class-J judge failed ({judge['error']}); "
                                   "Class-C scored, final grade withheld.")
    else:
        report["score"]["note"] = "Class-C only (judge skipped); final grade withheld."

    out_dir = pd / "eval"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _project_times(project_dir, users_db_path=None) -> tuple[str | None, str | None]:
    """Best-effort (created_at, finished_at) for a precise latency, read straight
    from the accounts DB. The project dir is ``.../<uid>/<pid>``; we look up the
    row by id. Fully guarded: any failure yields (None, None) and latency falls
    back to the state-history heuristic."""
    pd = Path(project_dir)
    db = Path(users_db_path) if users_db_path else (Path.home() / ".kicraft" / "accounts.db")
    if not db.is_file() or not pd.name.isdigit():
        return None, None
    try:
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT created_at, finished_at FROM projects WHERE id=?",
                (int(pd.name),)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None, None
    if not row:
        return None, None
    return row[0], row[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate a finished KiCraft web project (Class-C + automated Class-J).")
    ap.add_argument("project_dir", help="projects_dir/<uid>/<pid> of a finished design")
    ap.add_argument("--model", help="judge model override "
                    "(default: Settings.eval_judge_model, else the design model)")
    ap.add_argument("--no-judge", action="store_true",
                    help="score Class-C only; skip the LLM judge (no network / API key)")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="also print the full report JSON")
    args = ap.parse_args(argv)

    pd = Path(args.project_dir)
    if not pd.is_dir():
        raise SystemExit(f"not a directory: {pd}")

    client = None
    judge_client = None
    judge_model = args.model
    judge_max_tokens = None
    ledger_path = None
    users_db = None

    if args.no_judge:
        # Offline: attribute token usage from the default ledger if it happens to
        # exist, but never require an API key.
        default_ledger = Path.home() / ".kicraft" / "spend_ledger.db"
        ledger_path = default_ledger if default_ledger.is_file() else None
    else:
        from kicraft.server.client import CappedOpenRouterClient, make_client
        from kicraft.server.config import Settings
        s = Settings.from_env()
        client = CappedOpenRouterClient(s)
        # Judge defaults to a stronger, steadier model than the design model; it
        # gets a routing-relaxed client when it is not the design model.
        judge_model = (args.model or getattr(s, "eval_judge_model", None)
                       or getattr(s, "review_model", None) or s.model)
        if judge_model and judge_model != s.model:
            judge_client = make_client(s.for_judge())
        judge_max_tokens = getattr(s, "eval_judge_max_tokens", None)
        ledger_path = s.ledger_path
        users_db = s.users_db_path

    started_at, finished_at = _project_times(pd, users_db)
    report = evaluate_project(pd, client, judge_model=judge_model, judge_client=judge_client,
                              judge_max_tokens=judge_max_tokens,
                              ledger_path=ledger_path,
                              started_at=started_at, finished_at=finished_at,
                              skip_judge=args.no_judge)

    s = report["score"]
    j = report["judge"]
    print(f"{pd.name}: weighted={s['weighted']} final={s['final']} "
          f"grade={s['grade']} {s['verdict'] or ''}".rstrip())
    if j["ran"] and not j["ok"]:
        print(f"  judge error: {j['error']}")
    if s["note"]:
        print(f"  note: {s['note']}")
    print(f"wrote {pd / 'eval' / 'report.json'}")
    if args.show:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
