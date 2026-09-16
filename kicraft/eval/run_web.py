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

from kicraft.evidence_digest import EvidenceSection, render_bounded_evidence

from .artifacts import _find_one, _load_json
from .judge import grade_class_j
from .metrics_web import collect_web_metrics
from .rubric import load_rubric
from .scoring import eval_script_gates, finalize_report, metrics_block, score_class_c_dims


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _compact_bom_parts(parts: list[object]) -> str:
    """Complete electrically relevant identity for every committed BOM part."""
    rows = []
    for part in parts:
        if not isinstance(part, dict):
            rows.append(f"  {part!r}")
            continue
        identity = part.get("mpn") or part.get("lcsc") or ""
        rows.append(
            f"  {part.get('ref', '?')}: value={part.get('value', '')!r}; "
            f"symbol={part.get('symbol', '')!r}; footprint={part.get('footprint', '')!r}"
            + (f"; identity={identity!r}" if identity else "")
            + (f"; sheet={part['sheet']!r}" if part.get("sheet") else "")
        )
    return "\n".join(rows) or "No committed parts."


def _compact_bom_nets(connections: list[object], no_connect_pins: list[object]) -> str:
    """Complete pin-to-net evidence, without serializing unrelated state history."""
    rows = []
    for connection in connections:
        if not isinstance(connection, dict):
            rows.append(f"  {connection!r}")
            continue
        endpoints = connection.get("endpoints")
        if isinstance(endpoints, list):
            pins = ", ".join(
                f"{endpoint.get('ref', '?')}.{endpoint.get('pin', '?')}"
                if isinstance(endpoint, dict)
                else repr(endpoint)
                for endpoint in endpoints
            )
            sheet = f" [sheet={connection['sheet']!r}]" if connection.get("sheet") else ""
            rows.append(f"  {connection.get('net_name', '?')}{sheet}: {pins}")
        else:
            # Older persisted states use a/b connection records.  Keep every
            # field instead of pretending they are absent from the netlist.
            rows.append("  legacy connection: " + json.dumps(connection, sort_keys=True, default=str))
    if no_connect_pins:
        pins = ", ".join(
            f"{pin.get('ref', '?')}.{pin.get('pin', '?')}" if isinstance(pin, dict) else repr(pin)
            for pin in no_connect_pins
        )
        rows.append(f"  no-connect: {pins}")
    return "\n".join(rows) or "No committed nets."


def _programming_evidence(state: dict, bom: dict | None) -> str:
    """Emit only evidence-supported MCU conclusions."""
    if bom is None:
        return (
            "No committed BOM is available. MCU programming and delivered-MCU status are "
            "unverified; do not claim a missing or unprogrammable delivered MCU."
        )
    try:
        from kicraft.design.models import BOM as _BOM
        from kicraft.design.synthesis.validation import mcu_programming_facts

        facts = mcu_programming_facts(_BOM.model_validate(bom))
    except Exception as exc:
        return f"Programming analysis unavailable ({type(exc).__name__}); no absence conclusion is supported."
    if facts is None:
        return "No MCU is identified in the COMPLETE committed BOM."
    verdict = (
        "PASS -- a workable first-flash path exists"
        if facts["access_ok"] and facts["path_ok"]
        else "GAPS: " + "; ".join(facts["access_problems"] + facts["path_problems"])
    )
    return (
        f"{verdict}; MCU(s): {', '.join(facts['mcus'])}; "
        "programming-access parts: " + (", ".join(facts["access_parts"]) or "NONE")
    )


def build_run_digest(project_dir, m, *, budget: int = 16000) -> str:
    """Render bounded, structured judge evidence without slicing source records."""
    pd = Path(project_dir)
    state = _load_json(_find_one(pd, "state.json"))
    state = state if isinstance(state, dict) else {}
    bom = state.get("bom") if isinstance(state.get("bom"), dict) else None

    brief_path = pd / "brief.txt"
    brief = brief_path.read_text(errors="replace").strip() if brief_path.is_file() else ""
    requirements = {
        key: state[key]
        for key in ("intent", "functional_spec", "architecture", "assumptions", "open_questions")
        if key in state
    }

    synth, erc, tr, gen = m["synth"], m["erc"], m["transcript"], m["generated"]
    terminal = {
        "pipeline": {
            "synthesized": gen["synthesized"],
            "pcb": gen["pcb"],
            "sch": gen["sch"],
            "synthesis_check": {
                "status": synth.get("status"),
                "failed_checks": synth.get("failed_checks"),
            },
            "erc": {"errors": erc.get("errors"), "warnings": erc.get("warnings")},
            "run_trace": {
                "failed_commits": tr.get("failed_commits"),
                "ask_questions": tr.get("ask_questions"),
                "crashes": tr.get("crashes"),
            },
        },
        "state_terminal": {
            "artifact_status": (state.get("artifacts") or {}).get("status")
            if isinstance(state.get("artifacts"), dict)
            else None,
            "stage_status": state.get("stage_status"),
        },
    }

    deterministic: list[str] = [_programming_evidence(state, bom)]
    if isinstance(state.get("artifacts"), dict):
        artifacts = state["artifacts"]
        if artifacts.get("silk_placed") is not None or artifacts.get("silk_dropped") is not None:
            deterministic.append(
                f"Silk legend: placed={artifacts.get('silk_placed') or []!r}; "
                f"dropped={artifacts.get('silk_dropped') or []!r}"
            )
    if bom is not None:
        try:
            from kicraft.design.synthesis.validation import regulator_vout_facts

            facts = regulator_vout_facts(bom.get("parts") or [], bom.get("connections") or [])
            deterministic.extend(
                f"Regulator {fact['ref']} {fact['mpn']}: Vref={fact['vref']}V; "
                f"{fact['r_top_ref']}/{fact['r_bot_ref']} -> Vout={fact['vout']}V "
                f"on {fact['rail_net']!r}; rail match={fact['ok']!r}"
                for fact in facts
            )
        except Exception:
            deterministic.append("Regulator analysis unavailable; no regulator conclusion is supported.")

    # Put the complete electrical evidence before broad serialized requirements:
    # a large architecture record may be omitted honestly, but must never crowd
    # out the BOM/net facts the judge needs for circuit claims.
    sections = [
        EvidenceSection(
            "BRIEF (what the user asked for)",
            brief or "Brief file is unavailable.",
            complete=bool(brief),
        ),
    ]
    if bom is None:
        sections.append(
            EvidenceSection(
                "BOM / PINS / NETS",
                "No committed BOM. Part, pin, and net absence is unverified.",
                complete=False,
            )
        )
    else:
        sections.extend(
            [
                EvidenceSection(
                    f"BOM PARTS ({len(bom.get('parts') or [])})",
                    _compact_bom_parts(bom.get("parts") or []),
                ),
                EvidenceSection(
                    f"NETS AND PINS ({len(bom.get('connections') or [])})",
                    _compact_bom_nets(
                        bom.get("connections") or [], bom.get("no_connect_pins") or []
                    ),
                ),
                EvidenceSection(
                    "RECORDED SUBSTITUTIONS",
                    json.dumps(bom.get("substitutions") or [], indent=2, sort_keys=True, default=str),
                ),
            ]
        )
    sections.append(
        EvidenceSection(
            "REQUIREMENTS AND OPEN QUESTIONS",
            json.dumps(requirements, indent=2, sort_keys=True, default=str)
            if requirements
            else "No committed requirements are available.",
            complete=bool(requirements),
        )
    )
    sections.append(
        EvidenceSection("TERMINAL PIPELINE FACTS", json.dumps(terminal, indent=2, sort_keys=True))
    )
    sections.append(EvidenceSection("DETERMINISTIC ANALYSES", "\n".join(deterministic)))
    return render_bounded_evidence("STRUCTURED RUN EVIDENCE", sections, budget=budget)


def evaluate_project(
    project_dir,
    client,
    *,
    rubric: dict | None = None,
    judge_model: str | None = None,
    judge_client=None,
    judge_max_tokens: int | None = None,
    ledger_path=None,
    started_at: str | None = None,
    finished_at: str | None = None,
    skip_judge: bool = False,
    run_id: str | None = None,
) -> dict:
    """Score one finished web project and write ``eval/report.json``.

    Class-C is always scored from artifacts. Class-J is graded unless
    ``skip_judge`` (or no client) is given, in which case the judgment dimensions
    stay null and the run is not finalized (Class-C only). The judge uses
    ``judge_client`` when supplied (a client with routing relaxed for a stronger,
    steadier judge model that may be off the design provider tier), else
    ``client``. ``run_id`` selects exact-run ledger metrics and tags judge calls;
    without it, metrics intentionally aggregate all runs of this project.
    """
    rubric = rubric or load_rubric()
    pd = Path(project_dir)

    m = collect_web_metrics(
        pd, ledger_path=ledger_path, run_id=run_id,
        started_at=started_at, finished_at=finished_at
    )
    dims = score_class_c_dims(m, rubric)
    gates = eval_script_gates(m, rubric)

    judge = None
    if not skip_judge and client is not None:
        digest = build_run_digest(pd, m)
        jkw = {"max_tokens": judge_max_tokens} if judge_max_tokens else {}
        meta_ctx = {"run_id": run_id} if run_id else None
        judge = grade_class_j(
            judge_client or client,
            digest,
            rubric,
            model=judge_model,
            meta_ctx=meta_ctx,
            **jkw,
        )
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
        "run_id": run_id if run_id is not None else pd.name,
        "run_dir": str(pd),
        "scored_at": _now(),
        "rubric_version": rubric["meta"]["version"],
        "rubric_sha256": rubric["_computed_sha256"],
        "target_mode": "web",
        "metrics": metrics_block(m),
        "dimensions": dims,
        "gates": {
            "triggered": gates,
            "observer_rejected": (judge or {}).get("gates_rejected") or [],
            "observer_todo": [],
        },
        "judge": {
            "ran": judge is not None,
            "ok": (judge["ok"] if judge else None),
            "model": judge_model,
            "error": (judge["error"] if judge else None),
            "cost_usd": (round(judge["cost_usd"], 6) if judge else None),
        },
        "score": {
            "weighted": None,
            "final": None,
            "grade": None,
            "verdict": None,
            "pending_dimensions": [k for k, v in dims.items() if v["level"] is None],
            "note": "",
        },
    }

    if judge is not None and judge["ok"]:
        try:
            finalize_report(report, rubric)
        except ValueError as e:  # a Class-C dim came back unscored (e.g. latency)
            report["score"]["note"] = f"not finalized: {e}"
    elif judge is not None and not judge["ok"]:
        report["score"]["note"] = (
            f"Class-J judge failed ({judge['error']}); Class-C scored, final grade withheld."
        )
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
                "SELECT created_at, finished_at FROM projects WHERE id=?", (int(pd.name),)
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None, None
    if not row:
        return None, None
    return row[0], row[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate a finished KiCraft web project (Class-C + automated Class-J)."
    )
    ap.add_argument("project_dir", help="projects_dir/<uid>/<pid> of a finished design")
    ap.add_argument(
        "--model",
        help="judge model override (default: Settings.eval_judge_model, else the design model)",
    )
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="score Class-C only; skip the LLM judge (no network / API key)",
    )
    ap.add_argument(
        "--print", dest="show", action="store_true", help="also print the full report JSON"
    )
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
        # Judge identity is independent from electrical review.
        judge_model = args.model or getattr(s, "eval_judge_model", None) or s.model
        if judge_model and judge_model != s.model:
            judge_client = make_client(s.for_judge())
        judge_max_tokens = getattr(s, "eval_judge_max_tokens", None)
        ledger_path = s.ledger_path
        users_db = s.users_db_path

    started_at, finished_at = _project_times(pd, users_db)
    report = evaluate_project(
        pd,
        client,
        judge_model=judge_model,
        judge_client=judge_client,
        judge_max_tokens=judge_max_tokens,
        ledger_path=ledger_path,
        started_at=started_at,
        finished_at=finished_at,
        skip_judge=args.no_judge,
    )

    s = report["score"]
    j = report["judge"]
    print(
        f"{pd.name}: weighted={s['weighted']} final={s['final']} "
        f"grade={s['grade']} {s['verdict'] or ''}".rstrip()
    )
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
