"""Batch self-evaluation: drive every curated example brief end to end and grade it.

This is the regression harness behind the ``/self-eval`` command (and the
``kicraft-eval-batch`` console script). For each brief in
``kicraft.server.examples.EXAMPLE_PROMPTS`` it reproduces, headlessly, exactly what
the web app does for a real user and then scores the result with the existing
``kicraft.eval`` rubric:

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
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kicraft.build_slots import ACQUIRED_MARKER
from kicraft.server.examples import EXAMPLE_PROMPTS
from kicraft.server.session import read_state, record_answers, remaining_stages, run_session

from .run_web import evaluate_project

# Deterministic build, mirroring the web worker (kicraft.server.web): relative paths
# resolved against the run dir, no archive sweep. KICRAFT mirrors stage_driver.
_BUILD_CMD = [sys.executable, "-m", "kicraft.design.cli_app",
              "build", ".kicraft/state.json", "generated", "--no-archive"]

# Only these event kinds belong in the scorable events.jsonl. The capped client
# streams reasoning_delta / answer_delta / tool / tool_result through the SAME
# progress callback; writing those would bloat the log and a stray kind could skew
# the convergence/friction scorers, which key off retry/question/stage_done events.
_EVENT_KINDS = frozenset({
    "stage_start", "stage_done", "question", "retry",
    "build_start", "build_log", "build_done",
})

# kicraft.design.cli_app._cmd_build exit code -> a short human label, so the report
# shows routing/fab-readiness distinctly from the rubric grade (which judges the
# schematic / BOM / wiring, not the routed board).
_BUILD_RC_LABEL = {
    0: "fab-ready",
    2: "state unreadable",
    3: "incomplete (no wiring)",
    4: "synthesis input error",
    5: "ERC errors",
    6: "route/infra failed",
    7: "not fab-ready (DRC)",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stem_for(idx: int, prompt: str) -> str:
    """A stable, filesystem-safe run-dir name; ``p<stem>-`` is also the ledger
    run_id prefix collect_web_metrics groups token usage by."""
    words = re.findall(r"[A-Za-z0-9]+", prompt.upper())[:3]
    slug = "_".join(words)[:28] or "BRIEF"
    return f"run_{idx:02d}_{slug}"


def _build_env() -> dict:
    return {**os.environ, "KICRAFT_CALLER": os.environ.get("KICRAFT_CALLER", "web")}


def _event_writer(path: Path):
    """A ``progress(event)`` sink that appends the design/build events to
    ``events.jsonl`` (the scorable transcript), dropping the client's high-volume
    streaming kinds so the Class-C scorers see the same shape the web app persists."""
    def progress(ev: dict) -> None:
        if not isinstance(ev, dict) or ev.get("kind") not in _EVENT_KINDS:
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
        out.append({
            "text": q.get("text", ""),
            "answer": opts[0] if opts else
            "Use sensible engineering defaults; do not ask further questions.",
        })
    return out


# --------------------------------------------------------------------------- #
# drive + build one brief
# --------------------------------------------------------------------------- #
def run_design(client, brief: str, rundir: Path, progress, *,
               max_park_rounds: int = 12, run_id: str | None = None) -> dict:
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
    for round_no in range(max_park_rounds):
        rem = remaining_stages(read_state(rundir))
        if not rem:
            return {"status": "ok", "cost_usd": cost, "questions": n_questions,
                    "rounds": round_no, "error": None}
        res = run_session(rundir, brief, rem, answers=pending, client=client,
                          progress=progress, run_id=run_id)
        for r in res.get("results") or []:
            c = r.get("cost_usd")
            if isinstance(c, (int, float)):
                cost += c
        status = res.get("status")
        if status == "ok":
            return {"status": "ok", "cost_usd": cost, "questions": n_questions,
                    "rounds": round_no + 1, "error": None}
        if status == "awaiting_input":
            qs = res.get("questions") or []
            n_questions += len(qs)
            pending = _auto_answers(qs)
            record_answers(rundir, res.get("last_stage"), pending)
            continue
        last = (res.get("results") or [{}])[-1]
        err = last.get("error") or last.get("commit") or "stage failed to commit"
        return {"status": "failed", "cost_usd": cost, "questions": n_questions,
                "rounds": round_no + 1, "error": str(err)[:500]}
    return {"status": "parked", "cost_usd": cost, "questions": n_questions,
            "rounds": max_park_rounds, "error": "exceeded max park/resume rounds"}


def run_build(rundir: Path, progress, *, timeout_s: int = 1200) -> int:
    """Run the deterministic synth+place+route+fab build (mirrors the web worker),
    streaming its log into ``events.jsonl`` via ``progress``. A watchdog kills the
    build after ``timeout_s`` so one stuck route can't stall the whole batch; per
    the ``kicraft.build_slots`` contract the clock restarts at ACQUIRED_MARKER, so
    time spent queued for a host-wide build slot is never billed against the build.
    Returns the build exit code (negative if killed)."""
    progress({"kind": "build_start"})
    proc = subprocess.Popen(_BUILD_CMD, cwd=str(rundir), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=_build_env())
    timer = threading.Timer(timeout_s, proc.kill)
    timer.start()
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            progress({"kind": "build_log", "text": line.rstrip("\n")})
            if ACQUIRED_MARKER in line:
                timer.cancel()
                timer = threading.Timer(timeout_s, proc.kill)
                timer.start()
        proc.wait()
    finally:
        timer.cancel()
    rc = proc.returncode
    if rc is not None and rc < 0:
        progress({"kind": "build_log",
                  "text": f"[self-eval] build killed (timeout {timeout_s}s or signal {-rc})"})
    progress({"kind": "build_done", "ok": rc == 0, "rc": rc})
    return rc if rc is not None else -1


def evaluate_one(client, idx: int, prompt: str, out_dir: Path, *,
                 judge_model, skip_judge: bool, max_park_rounds: int = 12,
                 build_timeout_s: int = 1200, build_gate=None) -> dict:
    """Drive + build + score one brief into ``out_dir/<stem>/``. Never raises: any
    failure is captured in the returned record so the batch continues.

    ``build_gate`` (a semaphore) caps how many build subprocesses run at once when
    briefs execute concurrently: each route is a single-threaded JVM, so ungated
    builds would oversubscribe the cores and let CPU contention push otherwise-fine
    routes into ``--build-timeout``. The LLM design/judge phases stay ungated —
    they are network-wait and overlap with other briefs' builds for free."""
    t0 = time.time()
    started_at = _now_iso()
    stem = _stem_for(idx, prompt)
    rundir = out_dir / stem
    (rundir / ".kicraft").mkdir(parents=True, exist_ok=True)
    (rundir / "brief.txt").write_text(prompt + "\n", encoding="utf-8")
    progress = _event_writer(rundir / "events.jsonl")
    run_id = f"p{stem}-{int(t0)}"

    rec: dict = {"index": idx, "prompt": prompt, "stem": stem, "rundir": str(rundir)}
    try:
        d = run_design(client, prompt, rundir, progress,
                       max_park_rounds=max_park_rounds, run_id=run_id)
        rec.update(design_status=d["status"], design_cost_usd=round(d["cost_usd"], 6),
                   questions=d["questions"], design_error=d["error"])

        if d["status"] == "ok":
            with (build_gate or contextlib.nullcontext()):
                build_rc = run_build(rundir, progress, timeout_s=build_timeout_s)
        else:
            build_rc = None
        rec["build_rc"] = build_rc
        rec["build_label"] = (None if build_rc is None
                              else _BUILD_RC_LABEL.get(build_rc, f"rc={build_rc}"))

        # We know the real design+build wall-clock window (the same span the web app
        # measures), so pass it explicitly: the latency dimension's state-history ->
        # synth-checked_at fallback can return None, which leaves latency ungraded and
        # withholds the whole letter grade. An exact window always scores it.
        report = evaluate_project(rundir, None if skip_judge else client,
                                  judge_model=judge_model, skip_judge=skip_judge,
                                  started_at=started_at, finished_at=_now_iso())
        sc, judge = report["score"], report["judge"]
        rec.update(
            grade=sc.get("grade"), final=sc.get("final"), weighted=sc.get("weighted"),
            verdict=sc.get("verdict"), note=sc.get("note"),
            judge_ok=judge.get("ok"), judge_cost_usd=judge.get("cost_usd"),
            gates=[g["id"] for g in report["gates"]["triggered"]],
            dims={k: v.get("level") for k, v in report["dimensions"].items()},
            report_path=str(rundir / "eval" / "report.json"),
        )
    except Exception as e:  # noqa: BLE001 - record and keep the batch going
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
    rec["duration_s"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------- #
# compile the cross-brief report
# --------------------------------------------------------------------------- #
def _run_cost(r: dict) -> float:
    return round((r.get("design_cost_usd") or 0.0) + (r.get("judge_cost_usd") or 0.0), 6)


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

    summary = {
        **meta,
        "n": len(records),
        "graded_n": len(finals),
        "n_errored": sum(1 for r in records if r.get("error")),
        "fab_ready": sum(1 for r in records if r.get("build_rc") == 0),
        "mean_final": round(statistics.fmean(finals), 1) if finals else None,
        "median_final": round(statistics.median(finals), 1) if finals else None,
        "grade_counts": grade_counts,
        "gate_counts": gate_counts,
        "total_cost_usd": round(sum(_run_cost(r) for r in records), 4),
        "runs": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(_render_md(summary), encoding="utf-8")
    return summary


def _render_md(s: dict) -> str:
    L: list[str] = [f"# KiCraft self-eval — {s.get('started_at', '')}", ""]
    L.append(f"- briefs: **{s['n']}**  ·  graded: **{s['graded_n']}**  ·  "
             f"fab-ready builds: **{s['fab_ready']}/{s['n']}**  ·  errored: **{s['n_errored']}**")
    if s.get("mean_final") is not None:
        L.append(f"- score (0–100): mean **{s['mean_final']}** · median **{s['median_final']}**")
    L.append("- grades: " + "  ".join(f"{g}:{n}" for g, n in sorted(s["grade_counts"].items())))
    if s["gate_counts"]:
        L.append("- gates: " + ", ".join(f"{k}×{v}" for k, v in s["gate_counts"].items()))
    L.append(f"- judge: {(s.get('judge_model') if s.get('judge') else 'off (Class-C only)')}"
             f"  ·  design model: {s.get('design_model')}")
    L.append(f"- total spend: **${s['total_cost_usd']}**  ·  report dir: `{s.get('out_dir')}`")
    L.append("")
    L.append("| # | grade | final | verdict | build | Q | $ | brief |")
    L.append("|---|-------|-------|---------|-------|---|---|-------|")
    for r in s["runs"]:
        if r.get("build_rc") is None:
            build = "—" if not r.get("error") else "—"
        else:
            build = r.get("build_label") or str(r.get("build_rc"))
        verdict = r.get("verdict") or ("ERROR" if r.get("error") else r.get("design_status") or "—")
        final = r.get("final")
        brief = r["prompt"].replace("|", "\\|")
        brief = brief[:58] + "…" if len(brief) > 59 else brief
        L.append("| " + " | ".join([
            str(r["index"]),
            r.get("grade") or "—",
            (str(final) if final is not None else "—"),
            str(verdict),
            str(build),
            str(r.get("questions", "—")),
            f"${_run_cost(r)}",
            brief,
        ]) + " |")

    flagged = [r for r in s["runs"] if r.get("error") or r.get("gates")
               or (isinstance(r.get("final"), (int, float)) and r["final"] < 60)]
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


def _select(prompts: list[str], limit, only) -> list[tuple[int, str]]:
    rows = list(enumerate(prompts, start=1))
    if only:
        want = {int(x) for x in str(only).split(",") if x.strip()}
        return [(i, p) for i, p in rows if i in want]
    if limit is not None:
        return rows[:limit]
    return rows


def _load_prior_records(out_dir: Path) -> dict[int, dict]:
    """Per-brief records from an existing batch's ``summary.json``, keyed by brief
    index. Empty when the batch never wrote a checkpoint (or it is unreadable)."""
    path = out_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or []
        return {r["index"]: r for r in runs if isinstance(r.get("index"), int)}
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
        description="Drive every curated example brief end to end and grade it "
                    "(the /self-eval regression loop).")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N example briefs")
    ap.add_argument("--only", default=None,
                    help="comma-separated 1-based indices to run, e.g. '1,3,5'")
    ap.add_argument("--out", default=None,
                    help="report root (default: <projects_dir>/../self_eval/<ts> or ./logs/self_eval/<ts>)")
    ap.add_argument("--no-judge", action="store_true",
                    help="score Class-C only; skip the LLM judge (cheaper, no A-F grade)")
    ap.add_argument("--judge-model", default=None, help="judge model override")
    ap.add_argument("--max-park-rounds", type=int, default=12,
                    help="cap on park/auto-answer resume rounds per brief")
    ap.add_argument("--build-timeout", type=int, default=1200,
                    help="seconds before a stuck build is killed (per brief)")
    ap.add_argument("--parallel", type=int, default=3,
                    help="run N briefs concurrently (threads; default 3 — the measured "
                         "sweet spot on a 2-core box; 1 forces the strictly sequential "
                         "baseline). Spend ceilings can overshoot by up to N in-flight "
                         "calls; per-brief semantics are otherwise unchanged.")
    ap.add_argument("--build-slots", type=int, default=2,
                    help="max concurrent build subprocesses under --parallel (each route "
                         "is a single-threaded JVM; keep <= CPU cores so --build-timeout "
                         "stays honest)")
    ap.add_argument("--resume", default=None, metavar="BATCH_DIR",
                    help="finish an existing batch dir: reuse completed briefs from its "
                         "summary.json, wipe + re-run only errored/missing ones. Combine "
                         "with --only/--limit to restrict the considered set.")
    args = ap.parse_args(argv)

    selected = _select(list(EXAMPLE_PROMPTS), args.limit, args.only)
    if not selected:
        print("no briefs selected (check --limit / --only)", file=sys.stderr)
        return 2

    from kicraft.server.client import CappedOpenRouterClient
    from kicraft.server.config import Settings
    s = Settings.from_env()
    client = CappedOpenRouterClient(s)
    judge_model = args.judge_model or getattr(s, "eval_judge_model", None) or getattr(s, "model", None)

    # Resolve to an absolute path: design stages run subprocesses with cwd=workspace,
    # so a relative rundir would make them resolve state/output paths against the
    # wrong base (writing a nested <rundir>/<rundir>/.kicraft tree).
    resume_dir = Path(args.resume).resolve() if args.resume else None
    out_dir = (resume_dir if resume_dir
               else (Path(args.out) if args.out else _default_out_dir())).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = _load_prior_records(out_dir) if resume_dir else {}
    if resume_dir and prior and not args.only and args.limit is None:
        # default a resume to the batch's own brief set, not the whole catalog
        selected = [(i, p) for i, p in selected if i in prior]
    reused = {i: prior[i] for i, _ in selected if _reusable(prior.get(i))}
    todo = [(i, p) for i, p in selected if i not in reused]
    parallel = max(1, min(args.parallel, len(todo) or 1))

    meta = {
        "started_at": _now_iso(),
        "out_dir": str(out_dir),
        "design_model": getattr(s, "model", None),
        "judge": not args.no_judge,
        "judge_model": None if args.no_judge else judge_model,
        "rubric_version": None,
        "parallel": parallel,
        "build_slots": max(1, args.build_slots),
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
    print(f"self-eval: {len(selected)} brief(s) -> {out_dir}", flush=True)
    if resume_dir:
        print(f"  resume: {len(reused)} reused · {len(todo)} to run", flush=True)
    print(f"  design model={meta['design_model']}  "
          f"judge={'off (Class-C only)' if args.no_judge else judge_model}"
          + (f"  parallel={parallel} build_slots={meta['build_slots']}" if parallel > 1 else ""),
          flush=True)

    # A re-run brief must start from a clean slate: stale .kicraft state would make
    # run_design resume mid-chain and stale events.jsonl would skew the Class-C scorers.
    if resume_dir:
        for idx, prompt in todo:
            stale = out_dir / _stem_for(idx, prompt)
            if stale.exists():
                shutil.rmtree(stale)

    t_mono = time.monotonic()
    by_idx: dict[int, dict] = dict(reused)
    ckpt_lock = threading.Lock()

    def _checkpoint() -> None:
        # Live partial summary after every brief: what --resume reads back when a
        # batch is interrupted, and a progress view for the admin GUI. Call with
        # ckpt_lock held.
        recs = [by_idx[i] for i, _ in selected if i in by_idx]
        compile_report(recs, out_dir, meta)

    if parallel <= 1:
        for n, (idx, prompt) in enumerate(todo, start=1):
            print(f"\n[{n}/{len(todo)}] #{idx}: {prompt}", flush=True)
            rec = evaluate_one(client, idx, prompt, out_dir, judge_model=judge_model,
                               skip_judge=args.no_judge, max_park_rounds=args.max_park_rounds,
                               build_timeout_s=args.build_timeout)
            if rec.get("error"):
                print(f"   ERROR: {rec['error']}", flush=True)
            else:
                print(f"   grade={rec.get('grade') or '—'} final="
                      f"{rec.get('final') if rec.get('final') is not None else '—'} "
                      f"build={rec.get('build_label')} cost=${_run_cost(rec)} "
                      f"({rec.get('duration_s')}s)", flush=True)
            with ckpt_lock:
                by_idx[idx] = rec
                _checkpoint()
    else:
        gate = threading.BoundedSemaphore(max(1, args.build_slots))
        print_lock = threading.Lock()

        def _worker(idx: int, prompt: str) -> dict:
            # One client per brief: CappedOpenRouterClient instances are not safe to
            # share across threads (construction is ~ms; the spend ledger is WAL sqlite).
            wclient = CappedOpenRouterClient(s)
            stem = _stem_for(idx, prompt)
            with print_lock:
                print(f"[{stem}] start: {prompt}", flush=True)
            rec = evaluate_one(wclient, idx, prompt, out_dir, judge_model=judge_model,
                               skip_judge=args.no_judge, max_park_rounds=args.max_park_rounds,
                               build_timeout_s=args.build_timeout, build_gate=gate)
            with print_lock:
                if rec.get("error"):
                    print(f"[{stem}] ERROR: {rec['error']}", flush=True)
                else:
                    print(f"[{stem}] done grade={rec.get('grade') or '—'} final="
                          f"{rec.get('final') if rec.get('final') is not None else '—'} "
                          f"build={rec.get('build_label')} cost=${_run_cost(rec)} "
                          f"({rec.get('duration_s')}s)", flush=True)
            return rec

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(_worker, idx, prompt) for idx, prompt in todo]
            for fut in as_completed(futures):
                rec = fut.result()  # evaluate_one never raises
                with ckpt_lock:
                    by_idx[rec["index"]] = rec
                    _checkpoint()

    meta["finished_at"] = _now_iso()
    meta["wall_s"] = round(time.monotonic() - t_mono, 1)
    records = [by_idx[i] for i, _ in selected if i in by_idx]
    summary = compile_report(records, out_dir, meta)

    print(f"\n=== self-eval complete: {summary['graded_n']}/{summary['n']} graded · "
          f"mean={summary['mean_final']} · fab-ready={summary['fab_ready']}/{summary['n']} · "
          f"spend=${summary['total_cost_usd']} ===")
    if summary["gate_counts"]:
        print("    gates: " + ", ".join(f"{k}×{v}" for k, v in summary["gate_counts"].items()))
    print(f"report: {out_dir / 'summary.md'}")
    print(f"        {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
