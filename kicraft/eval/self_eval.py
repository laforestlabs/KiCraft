"""Batch self-evaluation: drive every benchmark brief end to end and grade it.

This is the regression harness behind the ``/self-eval`` command (and the
``kicraft-eval-batch`` console script). For each brief in
``kicraft.tuning.benchmark.BENCHMARK_PROMPTS`` (the shared brief corpus that
spans the placement/routing stress archetypes — connector density, fine-pitch
escape, RF keepouts, power/thermal, THT/SMT mix, hierarchy depth, and the
shaped-outline group) it reproduces,
headlessly, exactly what the web app does for a real user and then scores the
result with the existing ``kicraft.eval`` rubric:

  1. drive the five LLM design stages (intent -> functional_spec -> architecture ->
     bom -> wiring) over a fresh workspace, auto-answering any clarifying question
     the pipeline parks on with its own first suggested option (the product's
     suggested-answer UX), so a run reaches completion without a human in the loop;
  2. materialise the scorable ``events.jsonl`` from the driver's progress stream and
     run the deterministic build (synthesise + place + route + fab), exactly as the
     web worker does;
  3. score the finished run dir with ``evaluate_project`` (Class-C metrics + the
     LLM judge -> an A-F grade), which writes ``<rundir>/eval/report.json``;
  4. compile a cross-brief ``summary.json`` + ``summary.md`` (per-brief grade,
     verdict, build/fab-readiness, gates, cost, plus aggregates).

Each brief runs in its own subdir under the report root, which *is* the eval
``project_dir``: the rubric's artifact finders are recursive, so the driver's
``.kicraft/state.json`` and ``generated/<stem>/`` tree are picked up automatically;
this module only has to add the top-level ``events.jsonl`` (the transcript the
Class-C scorers read) and ``brief.txt`` (what the judge digest quotes).

The dominant cost is the design pipelines themselves (BOM part resolution
dominates), not the judge; ``--no-judge`` scores Class-C only and skips the LLM
judge. The capped client's spend guard still applies, so a tripped ceiling fails
the remaining briefs cheaply rather than overspending.

Throughput: briefs run on ``--parallel`` worker threads (default 3 — the heavy
work is build subprocesses and blocking HTTP, both GIL-releasing) with concurrent
build subprocesses capped by ``--build-slots`` (default 2) so single-threaded
routing JVMs don't oversubscribe the cores; every entry point (CLI, ``/self-eval``,
the admin GUI) inherits these defaults. ``--parallel 1`` forces the strictly
sequential baseline. ``summary.json`` is checkpointed after every brief, and
``--resume <batch_dir>`` finishes an interrupted batch by reusing completed
briefs and re-running only errored/missing ones.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kicraft.build_slots import ACQUIRED_MARKER, resolve_build_slots
from kicraft.proc_tree import kill_tree
from kicraft.server.session import (
    bom_reconcile_deficits,
    maybe_bom_reconcile,
    read_state,
    record_answers,
    remaining_stages,
    run_session,
)
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS as BRIEFS
from kicraft.tuning.benchmark import SHAPED_OUTLINE_PROMPTS

from .outline_check import evaluate_outline_shape
from .run_web import evaluate_project


def _find_parent_board(rundir: Path) -> Path | None:
    """The built final/parent board: ``generated/<stem>/<stem>.kicad_pcb``. Leaf
    boards live deeper under ``.experiments/`` so the shallow glob skips them."""
    gen = rundir / "generated"
    cands = [p for p in gen.glob("*/*.kicad_pcb") if ".experiments" not in p.parts]
    return cands[0] if cands else None


# Build outcomes that leave a PROMOTED ROUTED parent at
# ``generated/<stem>/<stem>.kicad_pcb``: rc=0 (fab-ready) and rc=7 (not
# fab-ready, DRC) both composed + routed + promoted a parent, so the shape was
# stamped. Every other outcome -- rc=6 (route/infra abort), rc<=5 (synthesis
# death), None -- leaves only the rectangular synthesis SEED STUB on that path.
_PROMOTED_PARENT_RCS = frozenset({0, 7})


def _outline_check(entry: dict, rundir: Path, build_rc: int | None) -> dict | None:
    """Deterministic outline-shape grade for a shaped brief (entries carrying
    ``outline_shape``); ``None`` for ordinary rectangular briefs.

    Gated on build outcome: only a promoted routed parent carries the stamped
    shape. A build that died in the leaf phase leaves the rectangular seed stub
    (kicad_pcb_stub._draw_board_outline) at the same path; grading THAT reported
    a leaf-phase death as a wrong-shape failure (WS9). Report ``pass=None`` with
    a distinct reason instead, so it never counts as a shape miss."""
    expected = entry.get("outline_shape")
    if not expected:
        return None
    if build_rc not in _PROMOTED_PARENT_RCS:
        return {
            "pass": None,
            "level": None,
            "expected_shape": expected,
            "reason": f"no built parent (build rc={build_rc})",
        }
    board = _find_parent_board(rundir)
    res = (
        evaluate_outline_shape(board, expected)
        if board
        else {
            "level": 0,
            "partial": False,
            "detected_family": None,
            "rationale": "no built board to classify",
            "expected_shape": expected,
        }
    )
    res["pass"] = res.get("level") == 4
    return res


# Deterministic build, mirroring the web worker (kicraft.server.web): relative paths
# resolved against the run dir, no archive sweep. KICRAFT mirrors stage_driver.
_BUILD_CMD = [
    sys.executable,
    "-m",
    "kicraft.design.cli_app",
    "build",
    ".kicraft/state.json",
    "generated",
    "--no-archive",
]

# Only these event kinds belong in the scorable events.jsonl. The capped client
# streams reasoning_delta / answer_delta / tool / tool_result through the SAME
# progress callback; writing those would bloat the log and a stray kind could skew
# the convergence/friction scorers, which key off retry/question/stage_done events.
_EVENT_KINDS = frozenset(
    {
        "stage_start",
        "stage_done",
        "question",
        "retry",
        "serialization_recovery",
        "candidate_decoded",
        "stage_diagnostic",
        "build_start",
        "build_log",
        "build_done",
    }
)

# kicraft.design.cli_app._cmd_build exit code -> a short human label, so the report
# shows routing/fab-readiness distinctly from the rubric grade (which judges the
# schematic / BOM / wiring, not the routed board).
_BUILD_RC_LABEL = {
    0: "fab-ready",
    2: "state unreadable",
    3: "incomplete (no wiring)",
    4: "synthesis input error",
    5: "synthesis check failed",  # refined per failed check by _build_label()
    6: "route/infra failed",
    7: "not fab-ready (DRC)",
}


def _rc5_label(rundir: Path) -> str:
    """Distinguish the rc=5 ("synthesis check failed") sub-causes.

    rc=5 fires for *any* failed §9.x synthesis check, not just ERC — e.g.
    #11 fpc-breakout had 0 ERC errors but failed §9.13 netlist faithfulness.
    Reading ``synthesis_check.json``'s ``failed_checks`` lets the summary name
    the real failure instead of mislabelling every rc=5 as "ERC errors"."""
    try:
        checks = sorted(rundir.rglob("synthesis_check.json"))
        if not checks:
            return _BUILD_RC_LABEL[5]
        failed = json.loads(checks[0].read_text()).get("failed_checks", [])
    except (OSError, json.JSONDecodeError, KeyError):
        return _BUILD_RC_LABEL[5]
    if any("ERC" in name for name in failed):
        return "ERC errors"
    if any("netlist faithfulness" in name for name in failed):
        return "netlist faithfulness"
    return _BUILD_RC_LABEL[5]


def _build_label(build_rc: int | None, rundir: Path) -> str | None:
    if build_rc is None:
        return None
    if build_rc == 5:
        return _rc5_label(rundir)
    return _BUILD_RC_LABEL.get(build_rc, f"rc={build_rc}")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _stable_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_campaign_manifest(
    out_dir: Path,
    *,
    settings,
    judge_model: str | None,
    selected: list[tuple[int, dict]],
    repeats: int,
) -> Path:
    corpus = [
        {"index": index, "slug": entry["slug"], "brief_hash": _stable_hash(entry["brief"])}
        for index, entry in selected
    ]
    immutable = {
        "code_revision": _code_revision(),
        "design_profile": getattr(settings, "design_profile", "custom"),
        "design_model": settings.model,
        "design_provider_order": list(settings.provider_order),
        "design_price_caps_mtok": {
            "prompt": settings.max_price_prompt,
            "completion": settings.max_price_completion,
        },
        "review_model": settings.review_model,
        "review_provider_order": list(settings.review_provider_order),
        "judge_model": judge_model,
        "judge_provider_order": list(settings.judge_provider_order),
        "response_policies": {
            "design": "kicraft_<stage>_response_v1",
            "architecture": "kicraft_architecture_response_v2",
            "bom_and_wiring": "kicraft_<stage>_response_v2",
            "review": "kicraft_electrical_review_v1",
            "judge": "kicraft_eval_judge_v1",
        },
        "caps": {
            "daily_usd": settings.daily_usd_ceiling,
            "total_usd": settings.total_usd_ceiling,
            "max_tokens_per_call": settings.max_tokens_per_call,
            "serialization_retries": settings.serialization_retries,
        },
        "repeats": repeats,
        "corpus_hash": _stable_hash(corpus),
        "corpus": corpus,
    }
    payload = {
        "schema_version": 1,
        "created_at": _now_iso(),
        "immutable": immutable,
    }
    path = out_dir / "campaign_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("immutable") != immutable:
            raise SystemExit(f"campaign manifest mismatch at {path}; source/model/policy changed")
        return path
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stem_for(idx: int, entry: dict) -> str:
    """A stable, filesystem-safe run-dir name; ``p<stem>-`` is also the ledger
    run_id prefix collect_web_metrics groups token usage by. The benchmark slug is
    a unique kebab id, so the dir is human-readable and stable across runs; the
    ``run_NN_`` prefix preserves ordering and keeps the admin's ``run_NN_*`` globs
    working. (Hyphens in the slug are safe in the run_id: the prefix match appends a
    trailing ``-`` and every slug is unique.)"""
    return f"run_{idx:02d}_{entry['slug']}"


def _build_env() -> dict:
    # PYTHONUNBUFFERED (mirroring the web build worker): the build's stdout goes
    # into a pipe, so without it the [build] lines sit block-buffered in the
    # child and a watchdog SIGKILL destroys the whole evidence trail (the empty
    # rc=-9 logs of the 2026-07-07 self-eval batch).
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "KICRAFT_CALLER": os.environ.get("KICRAFT_CALLER", "web"),
    }


def _event_writer(path: Path, *, full: bool = False):
    """A ``progress(event)`` sink that appends design/build events to
    ``events.jsonl``.

    With ``full`` (the batch default) it persists every event the capped client
    streams -- reasoning_delta / answer_delta / tool / tool_result -- so an eval
    run replays in the web viewer exactly like a live web build, whose own
    events.jsonl carries the same stream. With ``full=False`` it keeps only the
    structural kinds (``_EVENT_KINDS``), the lean transcript the Class-C scorers
    were first tuned on. Either is score-safe: the scorers key off specific kinds
    (stage_done / retry / question), so the extra streaming kinds never skew them
    -- the web app's own delta-laden transcript is scored the same way."""

    def progress(ev: dict) -> None:
        if not isinstance(ev, dict):
            return
        if not full and ev.get("kind") not in _EVENT_KINDS:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    return progress


def _auto_answers(questions) -> list[dict]:
    """Answer each parked clarifying question with its first model-suggested option
    (the product's one-click suggested answer), falling back to a defaults
    instruction when the model offered no options."""
    out = []
    for q in questions or []:
        opts = q.get("options") or []
        out.append(
            {
                "text": q.get("text", ""),
                "answer": opts[0]
                if opts
                else "Use sensible engineering defaults; do not ask further questions.",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# drive + build one brief
# --------------------------------------------------------------------------- #
def run_design(
    client,
    brief: str,
    rundir: Path,
    progress,
    *,
    max_park_rounds: int = 12,
    run_id: str | None = None,
) -> dict:
    """Drive the five design stages to completion over ``rundir``, auto-answering
    any parked clarifying question. Returns
    ``{status, cost_usd, questions, rounds, error}`` where status is ``ok`` (every
    stage committed), ``failed`` (a stage exhausted its retry budget), or
    ``parked`` (still awaiting input after ``max_park_rounds`` resumes).

    The driver applies ``answers`` only to the first stage of a chain and never
    re-parks a stage once answered, so each park advances the run by at least one
    stage and the loop converges well inside the round cap.
    """
    cost = 0.0
    n_questions = 0
    pending = None
    bom_passes = 0

    def _add_cost(r_dict: dict) -> None:
        nonlocal cost
        for r in r_dict.get("results") or []:
            c = r.get("cost_usd")
            if isinstance(c, (int, float)):
                cost += c

    for round_no in range(max_park_rounds):
        rem = remaining_stages(read_state(rundir))
        if not rem:
            return {
                "status": "ok",
                "cost_usd": cost,
                "questions": n_questions,
                "rounds": round_no,
                "error": None,
            }
        res = run_session(
            rundir, brief, rem, answers=pending, client=client, progress=progress, run_id=run_id
        )
        _add_cost(res)
        status = res.get("status")
        if status == "ok":
            return {
                "status": "ok",
                "cost_usd": cost,
                "questions": n_questions,
                "rounds": round_no + 1,
                "error": None,
            }
        if status == "awaiting_input":
            # A reconcile_target="bom" park is the pipeline's note-to-self, NOT a
            # user question: wiring cannot add the parts it needs, so plain-
            # answering it loops until the retry budget burns out (all 5
            # synthesis deaths in the 07-10 batch). Re-drive bom+wiring with the
            # concrete shortfall instead -- the same repair the web app runs
            # (WS6). Deficit CHAINS are real (each pass can surface the next
            # genuine shortfall -- fix-plan N3), so keep reconciling while the
            # pass budget advances. Never plain-answer a reconcile park.
            while res.get("status") == "awaiting_input" and bom_reconcile_deficits(res):
                prev = bom_passes
                res, bom_passes = maybe_bom_reconcile(
                    rundir,
                    brief,
                    res,
                    progress=progress,
                    run_id=run_id,
                    client=client,
                    reconcile_passes=bom_passes,
                )
                if bom_passes == prev:
                    break  # budget exhausted; fail honestly below
                _add_cost(res)  # count each reconcile pass
            status = res.get("status")
            if status == "ok":
                return {
                    "status": "ok",
                    "cost_usd": cost,
                    "questions": n_questions,
                    "rounds": round_no + 1,
                    "error": None,
                }
            qs = res.get("questions") or []
            if status == "awaiting_input" and any(q.get("reconcile_target") == "bom" for q in qs):
                # A BOM deficit that survived the reconcile budget cannot be
                # resolved by answering (wiring still can't add parts). Stop
                # honestly instead of looping the retry budget away.
                unresolved = "; ".join(str(q.get("text", "")) for q in qs)[:400]
                return {
                    "status": "failed",
                    "cost_usd": cost,
                    "questions": n_questions,
                    "rounds": round_no + 1,
                    "error": f"unresolved BOM deficit after "
                    f"{bom_passes} reconcile pass(es): {unresolved}",
                }
            if status != "awaiting_input":
                # Reconcile turned the park into a hard failure -- report it.
                last = (res.get("results") or [{}])[-1]
                err = last.get("error") or last.get("commit") or "stage failed to commit"
                return {
                    "status": "failed",
                    "cost_usd": cost,
                    "questions": n_questions,
                    "rounds": round_no + 1,
                    "error": str(err)[:500],
                }
            n_questions += len(qs)
            pending = _auto_answers(qs)
            record_answers(rundir, res.get("last_stage"), pending)
            continue
        last = (res.get("results") or [{}])[-1]
        err = last.get("error") or last.get("commit") or "stage failed to commit"
        return {
            "status": "failed",
            "cost_usd": cost,
            "questions": n_questions,
            "rounds": round_no + 1,
            "error": str(err)[:500],
        }
    return {
        "status": "parked",
        "cost_usd": cost,
        "questions": n_questions,
        "rounds": max_park_rounds,
        "error": "exceeded max park/resume rounds",
    }


def run_build(rundir: Path, progress, *, timeout_s: int = 2400) -> int:
    """Run the deterministic synth+place+route+fab build (mirrors the web worker),
    streaming its log into ``events.jsonl`` via ``progress``. A watchdog kills the
    build after ``timeout_s`` so one stuck route can't stall the whole batch; per
    the ``kicraft.build_slots`` contract the clock restarts at ACQUIRED_MARKER, so
    time spent queued for a host-wide build slot is never billed against the build.
    Returns the build exit code (negative if killed)."""
    progress({"kind": "build_start"})
    # Export a self-limit budget UNDER the watchdog (WS2): the pipeline finalizes
    # a best-so-far board before this SIGKILL fires, so a pathological run yields a
    # graded rc=6/7 partial instead of the empty rc=-9 the watchdog leaves. 90% of
    # the watchdog leaves headroom to finalize + export.
    build_env = _build_env()
    build_env.setdefault("KICRAFT_BUILD_MAX_WALL_S", f"{max(60.0, timeout_s * 0.9):.0f}")
    # start_new_session + kill_tree (not proc.kill): builds fan out to leaf,
    # router, and pcbnew subprocess groups that an outer watchdog must reap.
    proc = subprocess.Popen(
        _BUILD_CMD,
        cwd=str(rundir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=build_env,
        start_new_session=True,
    )
    timer = threading.Timer(timeout_s, kill_tree, args=(proc.pid,))
    timer.start()
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            progress({"kind": "build_log", "text": line.rstrip("\n")})
            if ACQUIRED_MARKER in line:
                timer.cancel()
                timer = threading.Timer(timeout_s, kill_tree, args=(proc.pid,))
                timer.start()
        proc.wait()
    finally:
        timer.cancel()
    rc = proc.returncode
    if rc is not None and rc < 0:
        progress(
            {
                "kind": "build_log",
                "text": f"[self-eval] build killed (timeout {timeout_s}s or signal {-rc})",
            }
        )
    progress({"kind": "build_done", "ok": rc == 0, "rc": rc})
    return rc if rc is not None else -1


def _make_judge_client(s, judge_model, skip_judge: bool):
    """A routing-relaxed client for the judge when it differs from the design
    model (a stronger judge is usually off the fp8 design tier and above the
    price cap), else None (reuse the design client). Built per-thread by the
    caller -- client instances are not safe to share across threads."""
    from kicraft.server.client import make_client

    if skip_judge or not judge_model or judge_model == getattr(s, "model", None):
        return None
    return make_client(s.for_judge())


def evaluate_one(
    client,
    idx: int,
    entry: dict,
    out_dir: Path,
    *,
    judge_model,
    skip_judge: bool,
    judge_client=None,
    judge_max_tokens: int | None = None,
    rep: int | None = None,
    max_park_rounds: int = 12,
    build_timeout_s: int = 2400,
    build_gate=None,
    full_events: bool = True,
) -> dict:
    """Drive + build + score one benchmark brief into ``out_dir/<stem>/``. ``entry``
    is a ``{"slug", "archetype", "brief"}`` dict from ``BENCHMARK_PROMPTS``. Never
    raises: any failure is captured in the returned record so the batch continues.

    ``build_gate`` (a semaphore) caps how many build subprocesses run at once when
    briefs execute concurrently: each route is a single-threaded JVM, so ungated
    builds would oversubscribe the cores and let CPU contention push otherwise-fine
    routes into ``--build-timeout``. The LLM design/judge phases stay ungated —
    they are network-wait and overlap with other briefs' builds for free."""
    t0 = time.time()
    started_at = _now_iso()
    prompt = entry["brief"]
    stem = _stem_for(idx, entry) + (f"__r{rep}" if rep else "")
    rundir = out_dir / stem
    (rundir / ".kicraft").mkdir(parents=True, exist_ok=True)
    (rundir / "brief.txt").write_text(prompt + "\n", encoding="utf-8")
    progress = _event_writer(rundir / "events.jsonl", full=full_events)
    run_id = f"p{stem}-{int(t0)}"

    # ``prompt`` is kept as the record field name (the web admin + report read it) and
    # mirrors entry["brief"]; ``slug``/``archetype`` are the new corpus identity.
    rec: dict = {
        "index": idx,
        "slug": entry["slug"],
        "repeat": rep,
        "archetype": entry["archetype"],
        "prompt": prompt,
        "stem": stem,
        "rundir": str(rundir),
        "run_id": run_id,
    }
    try:
        d = run_design(
            client, prompt, rundir, progress, max_park_rounds=max_park_rounds, run_id=run_id
        )
        rec.update(
            design_status=d["status"],
            design_cost_usd=round(d["cost_usd"], 6),
            questions=d["questions"],
            design_error=d["error"],
        )
        state_doc = {}
        try:
            state_doc = json.loads((rundir / ".kicraft" / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        stage_statuses = state_doc.get("stage_status") if isinstance(state_doc, dict) else {}
        stage_statuses = stage_statuses if isinstance(stage_statuses, dict) else {}
        rec["design_committed"] = all(
            isinstance(stage_statuses.get(stage), dict) and stage_statuses[stage].get("ok") is True
            for stage in ("intent", "functional_spec", "architecture", "bom", "wiring")
        )
        rec["semantic_clean"] = rec["design_committed"] and all(
            stage_statuses[stage].get("semantic_clean") is not False
            for stage in ("intent", "functional_spec", "architecture", "bom", "wiring")
        )
        rec["fab_safe"] = rec["design_committed"] and all(
            stage_statuses[stage].get("fab_safe") is not False
            for stage in ("intent", "functional_spec", "architecture", "bom", "wiring")
        )

        if d["status"] == "ok":
            with build_gate or contextlib.nullcontext():
                build_rc = run_build(rundir, progress, timeout_s=build_timeout_s)
        else:
            build_rc = None
        rec["build_rc"] = build_rc
        rec["build_label"] = _build_label(build_rc, rundir)

        # We know the real design+build wall-clock window (the same span the web app
        # measures), so pass it explicitly: the latency dimension's state-history ->
        # synth-checked_at fallback can return None, which leaves latency ungraded and
        # withholds the whole letter grade. An exact window always scores it.
        report = evaluate_project(
            rundir,
            None if skip_judge else client,
            judge_model=judge_model,
            judge_client=judge_client,
            judge_max_tokens=judge_max_tokens,
            skip_judge=skip_judge,
            started_at=started_at,
            finished_at=_now_iso(),
            run_id=run_id,
        )
        sc, judge = report["score"], report["judge"]
        rec.update(
            grade=sc.get("grade"),
            final=sc.get("final"),
            weighted=sc.get("weighted"),
            verdict=sc.get("verdict"),
            note=sc.get("note"),
            judge_ok=judge.get("ok"),
            judge_cost_usd=judge.get("cost_usd"),
            gates=[g["id"] for g in report["gates"]["triggered"]],
            dims={k: v.get("level") for k, v in report["dimensions"].items()},
            report_path=str(rundir / "eval" / "report.json"),
        )
        # Deterministic outline-shape check (shaped briefs only). Reported
        # alongside the rubric score, not folded into the 100-pt scale -- the
        # rubric's finalize rejects N/A dimensions, and it would only apply to
        # this category anyway.
        oc = _outline_check(entry, rundir, build_rc)
        if oc is not None:
            rec["outline_check"] = oc
    except Exception as e:  # noqa: BLE001 - record and keep the batch going
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
    rec["duration_s"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------- #
# compile the cross-brief report
# --------------------------------------------------------------------------- #
def _run_cost(r: dict) -> float:
    return round((r.get("design_cost_usd") or 0.0) + (r.get("judge_cost_usd") or 0.0), 6)


def _archetype_stats(records: list[dict]) -> dict[str, dict]:
    """Per-archetype rollup, so a regression localized to one stress dimension (e.g.
    fine-pitch escape) is visible even when the overall mean looks fine. Keyed by the
    benchmark archetype; preserves first-seen order."""
    out: dict[str, dict] = {}
    for r in records:
        a = r.get("archetype") or "—"
        st = out.setdefault(
            a, {"n": 0, "graded_n": 0, "fab_ready": 0, "_finals": [], "grade_counts": {}}
        )
        st["n"] += 1
        if isinstance(r.get("final"), (int, float)):
            st["graded_n"] += 1
            st["_finals"].append(r["final"])
        if r.get("build_rc") == 0:
            st["fab_ready"] += 1
        g = r.get("grade") or ("ERROR" if r.get("error") else "—")
        st["grade_counts"][g] = st["grade_counts"].get(g, 0) + 1
    for st in out.values():
        fin = st.pop("_finals")
        st["mean_final"] = round(statistics.fmean(fin), 1) if fin else None
    return out


def _iqr(vals: list[float]) -> float:
    """Inter-quartile range (Q3-Q1); 0.0 with fewer than 2 samples."""
    if len(vals) < 2:
        return 0.0
    q = statistics.quantiles(vals, n=4, method="inclusive")
    return round(q[2] - q[0], 1)


def _per_brief_stats(records: list[dict]) -> dict[str, dict]:
    """Group the N repeats of each brief and summarize its score distribution:
    median (the regression signal that survives the ~12-pt run-to-run noise) and
    IQR (the spread that *measures* that noise). Keyed by slug, first-seen order;
    a no-op shape for a single-repeat batch (median == the one value, IQR 0)."""
    out: dict[str, dict] = {}
    for r in records:
        slug = r.get("slug") or "—"
        st = out.setdefault(
            slug,
            {
                "slug": slug,
                "archetype": r.get("archetype"),
                "n": 0,
                "_finals": [],
                "fab_ready": 0,
                "grades": [],
            },
        )
        st["n"] += 1
        if isinstance(r.get("final"), (int, float)):
            st["_finals"].append(r["final"])
        if r.get("build_rc") == 0:
            st["fab_ready"] += 1
        st["grades"].append(r.get("grade") or ("ERROR" if r.get("error") else "—"))
    for st in out.values():
        fin = st.pop("_finals")
        st["graded_n"] = len(fin)
        st["median_final"] = round(statistics.median(fin), 1) if fin else None
        st["mean_final"] = round(statistics.fmean(fin), 1) if fin else None
        st["iqr"] = _iqr(fin) if fin else None
        st["min_final"] = round(min(fin), 1) if fin else None
        st["max_final"] = round(max(fin), 1) if fin else None
    return out


def compile_report(records: list[dict], out_dir: Path, meta: dict) -> dict:
    """Write ``summary.json`` + ``summary.md`` and return the summary dict."""
    finals = [r["final"] for r in records if isinstance(r.get("final"), (int, float))]
    grade_counts: dict[str, int] = {}
    for r in records:
        g = r.get("grade") or ("ERROR" if r.get("error") else "—")
        grade_counts[g] = grade_counts.get(g, 0) + 1
    gate_counts: dict[str, int] = {}
    for r in records:
        for gid in r.get("gates") or []:
            gate_counts[gid] = gate_counts.get(gid, 0) + 1

    per_brief = _per_brief_stats(records)
    # With repeats, the trustworthy cross-brief signal aggregates each brief's
    # MEDIAN (one vote per brief), so a single noisy run can't sway the headline.
    brief_medians = [
        b["median_final"] for b in per_brief.values() if b.get("median_final") is not None
    ]
    repeats = meta.get("repeats", 1)

    summary = {
        **meta,
        "n": len(records),
        "n_briefs": len(per_brief),
        "graded_n": len(finals),
        "n_errored": sum(1 for r in records if r.get("error")),
        "fab_ready": sum(1 for r in records if r.get("build_rc") == 0),
        "design_committed": sum(1 for r in records if r.get("design_committed") is True),
        "semantic_clean": sum(1 for r in records if r.get("semantic_clean") is True),
        "fab_safe": sum(1 for r in records if r.get("fab_safe") is True),
        "mean_final": round(statistics.fmean(finals), 1) if finals else None,
        "median_final": round(statistics.median(finals), 1) if finals else None,
        # per-brief-median aggregates (== flat mean/median when repeats == 1)
        "brief_median_mean": round(statistics.fmean(brief_medians), 1) if brief_medians else None,
        "brief_median_median": round(statistics.median(brief_medians), 1)
        if brief_medians
        else None,
        "median_iqr": (
            round(
                statistics.median(
                    [b["iqr"] for b in per_brief.values() if b.get("iqr") is not None]
                ),
                1,
            )
            if repeats > 1 and brief_medians
            else None
        ),
        "grade_counts": grade_counts,
        "gate_counts": gate_counts,
        "archetype_stats": _archetype_stats(records),
        "outline_stats": _outline_stats(records),
        "per_brief": per_brief,
        "total_cost_usd": round(sum(_run_cost(r) for r in records), 4),
        "runs": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(_render_md(summary), encoding="utf-8")
    return summary


def _outline_stats(records) -> dict | None:
    """Aggregate the deterministic outline-shape check over shaped briefs."""
    shaped = [r for r in records if r.get("outline_check")]
    if not shaped:
        return None
    # Only briefs that BUILT a promoted parent are graded on shape; ones that
    # failed upstream (pass=None) are counted separately, never as a shape miss
    # (WS9 -- grading the seed stub used to fake a hexagon/snowman "came out
    # rectangular" failure).
    graded = [r for r in shaped if r["outline_check"].get("pass") is not None]
    no_parent = [r for r in shaped if r["outline_check"].get("pass") is None]
    return {
        "n": len(shaped),
        "graded": len(graded),
        "pass": sum(1 for r in graded if r["outline_check"].get("pass")),
        "no_built_parent": len(no_parent),
        "by_brief": {
            r.get("slug"): {
                "expected": r["outline_check"].get("expected_shape"),
                "detected": r["outline_check"].get("detected_family"),
                "level": r["outline_check"].get("level"),
                "pass": r["outline_check"].get("pass"),
                "reason": r["outline_check"].get("reason"),
                "build_rc": r.get("build_rc"),
            }
            for r in shaped
        },
    }


def _render_md(s: dict) -> str:
    L: list[str] = [f"# KiCraft self-eval — {s.get('started_at', '')}", ""]
    repeats = s.get("repeats", 1)
    rep_note = f" ({repeats} repeats, {s['n']} runs)" if repeats > 1 else ""
    L.append(
        f"- briefs: **{s.get('n_briefs', s['n'])}**{rep_note}  ·  graded runs: **{s['graded_n']}**  ·  "
        f"fab-ready builds: **{s['fab_ready']}/{s['n']}**  ·  errored: **{s['n_errored']}**"
    )
    L.append(
        f"- design committed: **{s['design_committed']}/{s['n']}**  ·  "
        f"semantic clean: **{s['semantic_clean']}/{s['n']}**  ·  "
        f"fab safe: **{s['fab_safe']}/{s['n']}**"
    )
    if s.get("mean_final") is not None:
        L.append(f"- score (0–100): mean **{s['mean_final']}** · median **{s['median_final']}**")
    if repeats > 1 and s.get("brief_median_mean") is not None:
        L.append(
            f"- per-brief median (de-noised, 1 vote/brief): mean **{s['brief_median_mean']}** "
            f"· median **{s['brief_median_median']}**  ·  typical IQR **{s.get('median_iqr')}**"
        )
    L.append("- grades: " + "  ".join(f"{g}:{n}" for g, n in sorted(s["grade_counts"].items())))
    if s["gate_counts"]:
        L.append("- gates: " + ", ".join(f"{k}×{v}" for k, v in s["gate_counts"].items()))
    L.append(
        f"- judge: {(s.get('judge_model') if s.get('judge') else 'off (Class-C only)')}"
        f"  ·  design model: {s.get('design_model')}"
    )
    L.append(f"- total spend: **${s['total_cost_usd']}**  ·  report dir: `{s.get('out_dir')}`")
    L.append("")

    arche = s.get("archetype_stats") or {}
    if arche:
        L.append("## By archetype")
        L.append("")
        L.append("| archetype | n | graded | mean | fab-ready | grades |")
        L.append("|-----------|---|--------|------|-----------|--------|")
        for a, st in arche.items():
            grds = " ".join(f"{g}:{n}" for g, n in sorted(st["grade_counts"].items()))
            L.append(
                "| "
                + " | ".join(
                    [
                        a,
                        str(st["n"]),
                        str(st["graded_n"]),
                        (str(st["mean_final"]) if st["mean_final"] is not None else "—"),
                        f"{st['fab_ready']}/{st['n']}",
                        grds,
                    ]
                )
                + " |"
            )
        L.append("")

    per_brief = s.get("per_brief") or {}
    if repeats > 1 and per_brief:
        L.append("## Per brief (median over repeats)")
        L.append("")
        L.append("| slug | archetype | n | median | IQR | min–max | fab-ready | grades |")
        L.append("|------|-----------|---|--------|-----|---------|-----------|--------|")
        for b in per_brief.values():
            med = b.get("median_final")
            rng = f"{b['min_final']}–{b['max_final']}" if b.get("min_final") is not None else "—"
            grds = " ".join(b.get("grades") or [])
            L.append(
                "| "
                + " | ".join(
                    [
                        b.get("slug") or "—",
                        str(b.get("archetype") or "—"),
                        str(b.get("n", 0)),
                        (str(med) if med is not None else "—"),
                        (str(b.get("iqr")) if b.get("iqr") is not None else "—"),
                        rng,
                        f"{b.get('fab_ready', 0)}/{b.get('n', 0)}",
                        grds,
                    ]
                )
                + " |"
            )
        L.append("")

    L.append("## Per run" if repeats > 1 else "## Per brief")
    L.append("")
    L.append("| # | slug | archetype | grade | final | verdict | build | Q | $ |")
    L.append("|---|------|-----------|-------|-------|---------|-------|---|---|")
    for r in s["runs"]:
        if r.get("build_rc") is None:
            build = "—" if not r.get("error") else "—"
        else:
            build = r.get("build_label") or str(r.get("build_rc"))
        verdict = r.get("verdict") or ("ERROR" if r.get("error") else r.get("design_status") or "—")
        final = r.get("final")
        slug_cell = str(r.get("slug") or "—") + (f" r{r['repeat']}" if r.get("repeat") else "")
        L.append(
            "| "
            + " | ".join(
                [
                    str(r["index"]),
                    slug_cell,
                    str(r.get("archetype") or "—"),
                    r.get("grade") or "—",
                    (str(final) if final is not None else "—"),
                    str(verdict),
                    str(build),
                    str(r.get("questions", "—")),
                    f"${_run_cost(r)}",
                ]
            )
            + " |"
        )

    flagged = [
        r
        for r in s["runs"]
        if r.get("error")
        or r.get("gates")
        or (isinstance(r.get("final"), (int, float)) and r["final"] < 60)
    ]
    if flagged:
        L += ["", "## Needs attention"]
        for r in flagged:
            tag = f"**#{r['index']}** {r['stem']}"
            if r.get("error"):
                L.append(f"- {tag}: ERROR — {r['error']}")
            else:
                bits = []
                if r.get("gates"):
                    bits.append(f"gates {r['gates']}")
                if isinstance(r.get("final"), (int, float)) and r["final"] < 60:
                    bits.append(f"final {r['final']} ({r.get('verdict')})")
                if r.get("build_rc") not in (0, None):
                    bits.append(f"build {r.get('build_label')}")
                L.append(f"- {tag}: {', '.join(bits) or 'see report'} → `{r.get('rundir')}`")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_out_dir() -> Path:
    """Report root: a ``self_eval/<ts>/`` sibling of the projects dir if one is
    configured, else under the repo's ``logs/``."""
    root = os.environ.get("KICRAFT_PROJECTS_DIR")
    if not root:
        try:
            from kicraft.server.config import Settings

            root = getattr(Settings.from_env(), "projects_dir", None)
        except Exception:  # noqa: BLE001
            root = None
    base = (Path(root).parent / "self_eval") if root else (Path.cwd() / "logs" / "self_eval")
    return base / _stamp()


def _select(entries: list[dict], limit, only) -> list[tuple[int, dict]]:
    """Pick benchmark entries to run, paired with their 1-based position. ``only`` is
    a comma list of slugs (e.g. ``usb-pd-trigger,buck-3a``); bare integers are still
    accepted as 1-based indices for back-compat."""
    rows = list(enumerate(entries, start=1))
    if only:
        tokens = {t.strip() for t in str(only).split(",") if t.strip()}
        nums = {int(t) for t in tokens if t.isdigit()}
        slugs = {t for t in tokens if not t.isdigit()}
        return [(i, e) for i, e in rows if i in nums or e["slug"] in slugs]
    if limit is not None:
        return rows[:limit]
    return rows


def _run_key(slug: str, rep: int | None) -> str:
    """Stable per-run key. For a single run (rep is None) it is just the brief
    slug (so --resume stays robust to corpus reordering); for an N-repeat run it
    is ``<slug>__r<K>`` so the N runs of one brief don't collide."""
    return slug if rep is None else f"{slug}__r{rep}"


def _load_prior_records(out_dir: Path) -> dict[str, dict]:
    """Per-run records from an existing batch's ``summary.json``, keyed by run key
    (slug, or slug__rK for repeats; robust to corpus reordering between a batch
    and its --resume). Empty when the batch never wrote a checkpoint (or it is
    unreadable)."""
    path = out_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or []
        return {_run_key(r["slug"], r.get("repeat")): r for r in runs if r.get("slug")}
    except Exception:  # noqa: BLE001 - unreadable summary == nothing to reuse
        return {}


def _reusable(rec: dict | None) -> bool:
    """Under ``--resume``, a prior record is kept iff it finished scoring: no harness
    error and its eval report still on disk. Design failures and bad build rcs are
    legitimate *results* (regression signal), not candidates for a re-run."""
    if not rec or rec.get("error"):
        return False
    report = rec.get("report_path")
    return bool(report) and Path(report).exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Drive every benchmark brief end to end and grade it "
        "(the /self-eval regression loop over the 28-brief corpus)."
    )
    ap.add_argument("--limit", type=int, default=None, help="run only the first N benchmark briefs")
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated slugs to run, e.g. 'usb-pd-trigger,buck-3a' "
        "(bare integers are still accepted as 1-based indices)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="report root (default: <projects_dir>/../self_eval/<ts> or ./logs/self_eval/<ts>)",
    )
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="score Class-C only; skip the LLM judge (cheaper, no A-F grade)",
    )
    ap.add_argument(
        "--lean-events",
        action="store_true",
        help="persist only structural events (stage/retry/build) instead of "
        "the full reasoning/answer stream. Default is full fidelity, so a "
        "run replays in the web viewer like a live build (~+0.5-2MB/run)",
    )
    ap.add_argument("--judge-model", default=None, help="judge model override")
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run each brief N times and report per-brief median + IQR "
        "(default 1). N>=3 makes the ~12-pt run-to-run noise floor "
        "legible: a regression is a drop in the per-brief MEDIAN, not "
        "a single noisy run. Cost scales ~N (combine with --parallel).",
    )
    ap.add_argument(
        "--max-park-rounds",
        type=int,
        default=12,
        help="cap on park/auto-answer resume rounds per brief",
    )
    ap.add_argument(
        "--build-timeout",
        type=int,
        default=2400,
        help="seconds before a stuck build is killed (per brief)",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="run N briefs concurrently (threads; default 3 — the measured "
        "sweet spot on a 2-core box; 1 forces the strictly sequential "
        "baseline). Spend ceilings can overshoot by up to N in-flight "
        "calls; per-brief semantics are otherwise unchanged.",
    )
    ap.add_argument(
        "--build-slots",
        type=int,
        default=None,
        help="max concurrent build subprocesses under --parallel. Default "
        "is host-aware (build_slots.resolve_build_slots: max(1, "
        "cpus//6), capped at the core count) -- each build saturates "
        "the CPU, so a value > cores over-subscribes and trips "
        "--build-timeout (the 2026-07-08 rc=-9 kills). An explicit "
        "value > cores is rejected at launch, not silently run.",
    )
    ap.add_argument(
        "--resume",
        default=None,
        metavar="BATCH_DIR",
        help="finish an existing batch dir: reuse completed briefs from its "
        "summary.json, wipe + re-run only errored/missing ones. Combine "
        "with --only/--limit to restrict the considered set.",
    )
    ap.add_argument(
        "--no-shaped",
        action="store_true",
        help="exclude the shaped-outline group (archetype "
        "'shaped_outline') — run only the rectangular briefs",
    )
    ap.add_argument(
        "--shaped-only",
        action="store_true",
        help="run ONLY the shaped-outline group; each board's Edge.Cuts "
        "is graded against its requested shape (deterministic, "
        "reported as outline_check alongside the rubric score)",
    )
    ap.add_argument(
        "--include-shaped",
        action="store_true",
        help="deprecated no-op: the shaped-outline group is part of the "
        "default corpus now (kept so older invocations still run)",
    )
    args = ap.parse_args(argv)

    # Resolve --build-slots to a host-aware, clamped value BEFORE any work: a
    # contended config (build_slots > cores) trips the build watchdog, so reject
    # it loudly at launch instead of running a doomed 5-hour batch.
    try:
        build_slots = resolve_build_slots(args.build_slots)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    corpus = list(BRIEFS)  # the shaped-outline group is included by default now
    if args.shaped_only:
        corpus = list(SHAPED_OUTLINE_PROMPTS)
    elif args.no_shaped:
        corpus = [e for e in BRIEFS if not e.get("outline_shape")]
    selected = _select(corpus, args.limit, args.only)
    if not selected:
        print("no briefs selected (check --limit / --only)", file=sys.stderr)
        return 2

    from kicraft.server.client import make_client
    from kicraft.server.config import Settings

    s = Settings.from_env()
    client = make_client(s)
    # Judge identity is an independent role setting. It never inherits the
    # electrical-review model implicitly.
    judge_model = (
        args.judge_model or getattr(s, "eval_judge_model", None) or getattr(s, "model", None)
    )
    # Headroom for reasoning judges (minimax-m3 burns 10-23k reasoning tokens
    # before the JSON answer); too small a cap truncates and drops the grade.
    judge_max_tokens = getattr(s, "eval_judge_max_tokens", None)

    # Resolve to an absolute path: design stages run subprocesses with cwd=workspace,
    # so a relative rundir would make them resolve state/output paths against the
    # wrong base (writing a nested <rundir>/<rundir>/.kicraft tree).
    resume_dir = Path(args.resume).resolve() if args.resume else None
    out_dir = (
        resume_dir if resume_dir else (Path(args.out) if args.out else _default_out_dir())
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repeats = max(1, args.repeats)
    reps = [None] if repeats == 1 else list(range(1, repeats + 1))

    prior = _load_prior_records(out_dir) if resume_dir else {}
    if resume_dir and prior and not args.only and args.limit is None:
        # default a resume to the batch's own brief set, not the whole catalog
        prior_slugs = {r["slug"] for r in prior.values() if r.get("slug")}
        selected = [(i, e) for i, e in selected if e["slug"] in prior_slugs]
    manifest_path = _write_campaign_manifest(
        out_dir,
        settings=s,
        judge_model=None if args.no_judge else judge_model,
        selected=selected,
        repeats=repeats,
    )
    # Expand to one (idx, entry, rep) per run; reuse / re-run is decided per run.
    all_runs = [(i, e, rep) for i, e in selected for rep in reps]
    reused = {
        _run_key(e["slug"], rep): prior[_run_key(e["slug"], rep)]
        for _, e, rep in all_runs
        if _reusable(prior.get(_run_key(e["slug"], rep)))
    }
    todo = [(i, e, rep) for i, e, rep in all_runs if _run_key(e["slug"], rep) not in reused]
    parallel = max(1, min(args.parallel, len(todo) or 1))

    meta = {
        "started_at": _now_iso(),
        "out_dir": str(out_dir),
        "design_model": getattr(s, "model", None),
        "design_profile": getattr(s, "design_profile", "custom"),
        "design_provider_order": list(getattr(s, "provider_order", [])),
        "campaign_manifest": str(manifest_path),
        "judge": not args.no_judge,
        "judge_model": None if args.no_judge else judge_model,
        "rubric_version": None,
        "repeats": repeats,
        "parallel": parallel,
        "build_slots": build_slots,
        "full_events": not args.lean_events,
    }
    if resume_dir:
        meta["resumed_reused_n"] = len(reused)
    try:
        from .rubric import load_rubric

        meta["rubric_version"] = load_rubric()["meta"]["version"]
    except Exception:  # noqa: BLE001
        pass

    # flush every progress line: batches run for an hour-plus redirected to run.log,
    # and block buffering would otherwise hide per-brief lines until exit
    rep_note = f" x{repeats} repeats ({len(all_runs)} runs)" if repeats > 1 else ""
    print(f"self-eval: {len(selected)} brief(s){rep_note} -> {out_dir}", flush=True)
    if resume_dir:
        print(f"  resume: {len(reused)} reused · {len(todo)} to run", flush=True)
    print(
        f"  design model={meta['design_model']}  "
        f"judge={'off (Class-C only)' if args.no_judge else judge_model}"
        + (f"  parallel={parallel} build_slots={meta['build_slots']}" if parallel > 1 else ""),
        flush=True,
    )

    # A re-run must start from a clean slate: stale .kicraft state would make
    # run_design resume mid-chain and stale events.jsonl would skew the Class-C scorers.
    if resume_dir:
        for idx, entry, rep in todo:
            stale = out_dir / (_stem_for(idx, entry) + (f"__r{rep}" if rep else ""))
            if stale.exists():
                shutil.rmtree(stale)

    t_mono = time.monotonic()
    by_run: dict[str, dict] = dict(reused)
    ckpt_lock = threading.Lock()

    def _checkpoint() -> None:
        # Live partial summary after every run: what --resume reads back when a
        # batch is interrupted, and a progress view for the admin GUI. Call with
        # ckpt_lock held. Ordered by (brief, repeat) so the report is stable.
        recs = [by_run[k] for _, e, rep in all_runs if (k := _run_key(e["slug"], rep)) in by_run]
        compile_report(recs, out_dir, meta)

    if parallel <= 1:
        judge_client = _make_judge_client(s, judge_model, args.no_judge)
        for n, (idx, entry, rep) in enumerate(todo, start=1):
            label = entry["slug"] + (f" r{rep}" if rep else "")
            print(f"\n[{n}/{len(todo)}] #{idx} {label}: {entry['brief']}", flush=True)
            rec = evaluate_one(
                client,
                idx,
                entry,
                out_dir,
                judge_model=judge_model,
                skip_judge=args.no_judge,
                judge_client=judge_client,
                judge_max_tokens=judge_max_tokens,
                rep=rep,
                max_park_rounds=args.max_park_rounds,
                build_timeout_s=args.build_timeout,
                full_events=not args.lean_events,
            )
            if rec.get("error"):
                print(f"   ERROR: {rec['error']}", flush=True)
            else:
                print(
                    f"   grade={rec.get('grade') or '—'} final="
                    f"{rec.get('final') if rec.get('final') is not None else '—'} "
                    f"build={rec.get('build_label')} cost=${_run_cost(rec)} "
                    f"({rec.get('duration_s')}s)",
                    flush=True,
                )
            with ckpt_lock:
                by_run[_run_key(rec["slug"], rec.get("repeat"))] = rec
                _checkpoint()
    else:
        gate = threading.BoundedSemaphore(build_slots)
        print_lock = threading.Lock()

        def _worker(idx: int, entry: dict, rep: int | None) -> dict:
            # One client per run: client instances are not safe to share across
            # threads (construction is ~ms; the spend ledger is WAL sqlite). Routed
            # through make_client so KICRAFT_LLM_MODE=replay drives the corpus at $0.
            wclient = make_client(s)
            wjudge = _make_judge_client(s, judge_model, args.no_judge)
            stem = _stem_for(idx, entry) + (f"__r{rep}" if rep else "")
            with print_lock:
                print(f"[{stem}] start: {entry['brief']}", flush=True)
            rec = evaluate_one(
                wclient,
                idx,
                entry,
                out_dir,
                judge_model=judge_model,
                skip_judge=args.no_judge,
                judge_client=wjudge,
                judge_max_tokens=judge_max_tokens,
                rep=rep,
                max_park_rounds=args.max_park_rounds,
                build_timeout_s=args.build_timeout,
                build_gate=gate,
                full_events=not args.lean_events,
            )
            with print_lock:
                if rec.get("error"):
                    print(f"[{stem}] ERROR: {rec['error']}", flush=True)
                else:
                    print(
                        f"[{stem}] done grade={rec.get('grade') or '—'} final="
                        f"{rec.get('final') if rec.get('final') is not None else '—'} "
                        f"build={rec.get('build_label')} cost=${_run_cost(rec)} "
                        f"({rec.get('duration_s')}s)",
                        flush=True,
                    )
            return rec

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(_worker, idx, entry, rep) for idx, entry, rep in todo]
            for fut in as_completed(futures):
                rec = fut.result()  # evaluate_one never raises
                with ckpt_lock:
                    by_run[_run_key(rec["slug"], rec.get("repeat"))] = rec
                    _checkpoint()

    meta["finished_at"] = _now_iso()
    meta["wall_s"] = round(time.monotonic() - t_mono, 1)
    records = [by_run[k] for _, e, rep in all_runs if (k := _run_key(e["slug"], rep)) in by_run]
    summary = compile_report(records, out_dir, meta)

    print(
        f"\n=== self-eval complete: {summary['graded_n']}/{summary['n']} graded · "
        f"mean={summary['mean_final']} · fab-ready={summary['fab_ready']}/{summary['n']} · "
        f"spend=${summary['total_cost_usd']} ==="
    )
    if summary["gate_counts"]:
        print("    gates: " + ", ".join(f"{k}×{v}" for k, v in summary["gate_counts"].items()))
    print(f"report: {out_dir / 'summary.md'}")
    print(f"        {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
