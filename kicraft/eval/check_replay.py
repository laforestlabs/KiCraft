"""Replay archived stage candidates through the current semantic checks.

Vocabulary literalism (``docs/plans/vocabulary-literalism-plan-2026-09-26.md``) is
a claim about *frequency*: a prose-keyed check that refuses a valid design costs a
repair round, and the plan picks its fixes by measurement. This module is that
measurement. It walks an archived self-eval tree, recovers the candidate each
stage committed, and re-runs the deterministic checks over it with today's code,
so a check that fires on a board that shipped is a false-refusal candidate.

Two facts about the archive shape the design (verified 2026-09-26):

* ``.kicraft/state.json`` keeps only the *committed* slots. ``stage_status``
  therefore reads ``diagnostics: []`` for a stage that eventually committed, so
  the diagnostics the run actually saw cannot be read back from the archive --
  they are recomputed here.
* The wiring stage's candidate is not a ``ConversationState`` slot (it becomes the
  generated schematic), so it cannot be replayed from ``state.json`` at all. Only
  ``intent``, ``functional_spec``, ``architecture`` and ``bom`` are replayed; a
  wiring fix needs a live run or the ``answer_delta`` stream, which this module
  deliberately does not attempt.

The replay is a *proxy* for the live path, and says so: it re-runs the checks over
the committed candidate with the committed upstream slots, so it measures "would
this shipped board be refused by today's wording check", not "did this exact
diagnostic fire during the run". Repair rounds and attempts are read straight from
``stage_status``, so the KPI baseline is exact even where the candidate is not.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kicraft.design.stage_semantics import diagnose_stage

#: Stages whose candidate is a durable ``ConversationState`` slot. ``wiring`` is
#: absent on purpose: its output is the generated schematic, not a state slot.
REPLAYABLE_STAGES: tuple[str, ...] = ("intent", "functional_spec", "architecture", "bom")

#: The stage names a committed design must carry, for the ``committed`` roll-up.
DESIGN_STAGES: tuple[str, ...] = ("intent", "functional_spec", "architecture", "bom", "wiring")

STATE_RELATIVE = Path(".kicraft") / "state.json"


@dataclass(frozen=True)
class StageOutcome:
    """The archived operational facts for one stage, read from ``stage_status``."""

    stage: str
    ok: bool | None
    attempts: int | None
    repair_attempted: bool | None
    rounds: int | None
    failure_kind: str | None


@dataclass(frozen=True)
class Hit:
    """One check firing when a committed candidate is replayed."""

    stage: str
    code: str
    severity: str | None
    message: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunReplay:
    """One archived run, replayed."""

    run_dir: str
    campaign: str
    brief: str
    committed: bool
    stages: tuple[StageOutcome, ...] = ()
    hits: tuple[Hit, ...] = ()
    replayed_stages: tuple[str, ...] = ()
    error: str | None = None

    def hits_by_stage(self) -> dict[str, list[Hit]]:
        grouped: dict[str, list[Hit]] = {}
        for hit in self.hits:
            grouped.setdefault(hit.stage, []).append(hit)
        return grouped

    def outcome(self, stage: str) -> StageOutcome | None:
        for row in self.stages:
            if row.stage == stage:
                return row
        return None


# ------------------------------------------------------------------ loading


def iter_run_dirs(root: Path) -> list[Path]:
    """Every archived run directory under ``root`` that carries a ``state.json``."""
    return sorted({p.parent.parent for p in Path(root).rglob(f"*/{STATE_RELATIVE}")})


def _read_brief(run_dir: Path) -> str:
    try:
        return (run_dir / "brief.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _stage_outcomes(doc: Mapping[str, Any]) -> tuple[StageOutcome, ...]:
    status = doc.get("stage_status")
    status = status if isinstance(status, Mapping) else {}
    rows: list[StageOutcome] = []
    for stage in DESIGN_STAGES:
        row = status.get(stage)
        row = row if isinstance(row, Mapping) else None
        if row is None:
            rows.append(StageOutcome(stage, None, None, None, None, None))
            continue
        attempts = row.get("attempts")
        rounds = row.get("rounds")
        rows.append(
            StageOutcome(
                stage=stage,
                ok=row.get("ok") if isinstance(row.get("ok"), bool) else None,
                attempts=int(attempts) if isinstance(attempts, int) else None,
                repair_attempted=(
                    row.get("repair_attempted")
                    if isinstance(row.get("repair_attempted"), bool)
                    else None
                ),
                rounds=int(rounds) if isinstance(rounds, int) else None,
                failure_kind=str(row.get("failure_kind")) if row.get("failure_kind") else None,
            )
        )
    return tuple(rows)


def _is_committed(rows: Sequence[StageOutcome]) -> bool:
    by_stage = {row.stage: row for row in rows}
    return all(by_stage.get(stage) is not None and by_stage[stage].ok for stage in DESIGN_STAGES)


def _upstream_state(doc: Mapping[str, Any], stage: str) -> dict:
    """The state a stage's checks saw: every slot but its own, without bookkeeping.

    A check reads only *prior* stages (``_bom`` reads ``architecture``, ``_wiring``
    reads ``bom``), so dropping the stage's own slot reproduces the fresh-attempt
    semantics without changing any verdict.
    """
    upstream = {
        key: value
        for key, value in doc.items()
        if key not in {"stage_status", stage}
    }
    return upstream


@contextlib.contextmanager
def _offline_research():
    """Disable the one part-completion step that reaches the network and the library.

    ``complete_unavailable_part_classes`` researches an uncovered class and *writes
    a reviewed record into the machine-wide parts library*. A replay must not
    mutate the library or spend a fetch, so it is stubbed to "nothing was
    researched": the candidate is then checked against the library as it stands,
    which is the honest reading of "would today's checks accept this board".
    """
    from kicraft.design import part_research

    original = getattr(part_research, "research_uncovered_classes", None)
    part_research.research_uncovered_classes = lambda classes, limit=2: []  # type: ignore[assignment]
    try:
        yield
    finally:
        if original is not None:
            part_research.research_uncovered_classes = original  # type: ignore[assignment]


def normalize_candidate(
    stage: str, candidate: Mapping[str, Any], brief: str, upstream: Mapping[str, Any]
) -> dict:
    """The runtime's pre-diagnosis normalization, applied exactly once.

    The runtime diagnoses ``_normalize_candidate_for_diagnostics(...)``, not the raw
    provider payload, so a replay that skips it scores a different object than the
    live check did. It is applied *once* because it is deliberately not idempotent
    (``complete_over_rated_supply`` adds the rail a load needs, so a second pass
    duplicates it) -- verified against the archive 2026-09-26.

    ``decider`` is None on purpose: the replay measures the *deterministic* checks,
    never a Jev answer, so the same archive replays to the same table.
    """
    from kicraft.server.stage_runtime import _normalize_candidate_for_diagnostics

    with _offline_research():
        return _normalize_candidate_for_diagnostics(
            stage, dict(candidate), brief, dict(upstream), decider=None
        )


def replay_document(doc: Mapping[str, Any], brief: str) -> tuple[tuple[str, ...], tuple[Hit, ...]]:
    """Re-run the deterministic checks over every replayable committed candidate."""
    replayed: list[str] = []
    hits: list[Hit] = []
    for stage in REPLAYABLE_STAGES:
        candidate = doc.get(stage)
        if not isinstance(candidate, Mapping) or not candidate:
            continue
        replayed.append(stage)
        upstream = _upstream_state(doc, stage)
        try:
            normalized = normalize_candidate(stage, candidate, brief, upstream)
            findings = diagnose_stage(
                stage,
                brief=brief,
                upstream_state=upstream,
                candidate=normalized,
            )
        except Exception as exc:  # a schema-drifted archive row must not stop the sweep
            hits.append(
                Hit(
                    stage=stage,
                    code="<replay_error>",
                    severity=None,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        for finding in findings:
            hits.append(
                Hit(
                    stage=stage,
                    code=str(finding.code),
                    severity=finding.severity,
                    message=str(finding.message),
                    evidence=tuple(str(item) for item in finding.evidence or ()),
                )
            )
    return tuple(replayed), tuple(hits)


def replay_run(run_dir: Path) -> RunReplay:
    """Replay one archived run directory. Never raises: a bad row is reported, not fatal."""
    run_dir = Path(run_dir)
    campaign = run_dir.parent.name
    try:
        doc = json.loads((run_dir / STATE_RELATIVE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return RunReplay(str(run_dir), campaign, "", False, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(doc, Mapping):
        return RunReplay(str(run_dir), campaign, "", False, error="state.json is not an object")
    brief = _read_brief(run_dir)
    stages = _stage_outcomes(doc)
    replayed, hits = replay_document(doc, brief)
    return RunReplay(
        run_dir=str(run_dir),
        campaign=campaign,
        brief=brief,
        committed=_is_committed(stages),
        stages=stages,
        hits=hits,
        replayed_stages=replayed,
    )


# --------------------------------------------------------------- aggregation


@dataclass
class CodeStats:
    """Per-check firing statistics across the replayed corpus."""

    code: str
    severity: str | None = None
    fires: int = 0
    fires_on_committed: int = 0
    runs_firing: int = 0
    stages_firing: dict[str, int] = field(default_factory=dict)
    runs_firing_committed: set[str] = field(default_factory=set)
    examples: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "fires": self.fires,
            "fires_on_committed": self.fires_on_committed,
            "runs_firing": self.runs_firing,
            "runs_firing_on_committed": len(self.runs_firing_committed),
            "stages_firing": dict(self.stages_firing),
            "examples": self.examples[:5],
        }


def _severity_rank(severity: str | None) -> int:
    return {"fab_gate": 0, "repair_required": 1, "advisory": 2}.get(severity or "", 3)


def aggregate(runs: Iterable[RunReplay]) -> dict:
    """The plan's Step-1 table plus the repair-round KPI baseline."""
    runs = list(runs)
    codes: dict[str, CodeStats] = {}
    stage_totals: dict[str, dict[str, float]] = {}
    replayed_runs = 0
    for run in runs:
        if run.error or not run.replayed_stages:
            continue
        replayed_runs += 1
        seen_codes: set[str] = set()
        for hit in run.hits:
            if hit.code == "<replay_error>":
                continue
            stats = codes.setdefault(hit.code, CodeStats(code=hit.code))
            stats.fires += 1
            stats.stages_firing[hit.stage] = stats.stages_firing.get(hit.stage, 0) + 1
            if stats.severity is None or _severity_rank(hit.severity) < _severity_rank(stats.severity):
                stats.severity = hit.severity
            outcome = run.outcome(hit.stage)
            committed_stage = bool(outcome and outcome.ok)
            if committed_stage:
                stats.fires_on_committed += 1
                stats.runs_firing_committed.add(run.run_dir)
            if hit.code not in seen_codes:
                seen_codes.add(hit.code)
                stats.runs_firing += 1
            if len(stats.examples) < 5:
                stats.examples.append((run.run_dir, hit.message[:240]))
        for row in run.stages:
            if row.attempts is None:
                continue
            bucket = stage_totals.setdefault(
                row.stage, {"runs": 0, "attempts": 0.0, "repair_attempted": 0, "failed": 0}
            )
            bucket["runs"] += 1
            bucket["attempts"] += row.attempts
            if row.repair_attempted:
                bucket["repair_attempted"] += 1
            if row.ok is False:
                bucket["failed"] += 1
    kpi = {
        stage: {
            "runs": int(bucket["runs"]),
            "mean_attempts": round(bucket["attempts"] / bucket["runs"], 3) if bucket["runs"] else 0.0,
            "repair_attempted_runs": int(bucket["repair_attempted"]),
            "failed_runs": int(bucket["failed"]),
        }
        for stage, bucket in sorted(stage_totals.items())
    }
    return {
        "runs_total": len(runs),
        "runs_replayed": replayed_runs,
        "codes": {
            code: stats.as_dict()
            for code, stats in sorted(
                codes.items(),
                key=lambda item: (-item[1].fires_on_committed, -item[1].fires, item[0]),
            )
        },
        "kpi": kpi,
    }


def render_table(report: Mapping[str, Any], *, limit: int = 40) -> str:
    """The plan's Step-1 printout: ranked firings, then the repair-round baseline."""
    lines = [
        f"runs: {report['runs_total']} archived, {report['runs_replayed']} replayed "
        f"(candidates present)",
        "",
        "check                                        sev              fires  on-committed  runs",
    ]
    for code, row in list(report["codes"].items())[:limit]:
        lines.append(
            f"{code:44.44} {str(row['severity'] or '-'):16.16} "
            f"{row['fires']:6d} {row['fires_on_committed']:13d} {row['runs_firing']:5d}"
        )
    lines.append("")
    lines.append("stage             runs   mean-attempts  repair-attempted  failed")
    for stage, row in report["kpi"].items():
        lines.append(
            f"{stage:16.16} {row['runs']:5d} {row['mean_attempts']:14.3f} "
            f"{row['repair_attempted_runs']:17d} {row['failed_runs']:7d}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI


def _select(run_dirs: Sequence[Path], *, campaign: str | None, limit: int | None) -> list[Path]:
    selected = [path for path in run_dirs if campaign is None or campaign in path.parent.name]
    if limit is not None:
        selected = selected[:limit]
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="logs/self_eval", help="archived self-eval tree")
    parser.add_argument("--campaign", default=None, help="only runs whose campaign dir contains this")
    parser.add_argument("--limit", type=int, default=None, help="stop after N run dirs")
    parser.add_argument("--workers", type=int, default=1, help="parallel replay processes")
    parser.add_argument("--json", default=None, help="write the full report here")
    parser.add_argument("--dump-hits", default=None, help="write one JSONL row per firing here")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args(argv)

    run_dirs = _select(iter_run_dirs(Path(args.root)), campaign=args.campaign, limit=args.limit)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            runs = list(pool.map(replay_run, run_dirs, chunksize=8))
    else:
        runs = [replay_run(path) for path in run_dirs]

    report = aggregate(runs)
    print(render_table(report, limit=args.top))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.dump_hits:
        with Path(args.dump_hits).open("w", encoding="utf-8") as handle:
            for run in runs:
                if not run.committed:
                    continue
                for hit in run.hits:
                    handle.write(
                        json.dumps(
                            {
                                "run": run.run_dir,
                                "campaign": run.campaign,
                                "brief": run.brief,
                                "stage": hit.stage,
                                "code": hit.code,
                                "severity": hit.severity,
                                "message": hit.message,
                                "evidence": list(hit.evidence),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
