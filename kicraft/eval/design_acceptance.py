"""Fail-closed evidence gate for fresh five-stage design campaigns."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from kicraft.server.stage_contracts import DESIGN_STAGES
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

from .self_eval import _stable_hash


def verify_campaign(root: Path, slugs: list[str] | None = None) -> list[str]:
    """Return evidence defects; an empty list permits the selected campaign."""
    root = root.resolve()
    corpus = {
        entry["slug"]: {
            "index": index,
            "slug": entry["slug"],
            "brief_hash": _stable_hash(entry["brief"]),
        }
        for index, entry in enumerate(BENCHMARK_PROMPTS, 1)
    }
    selected = slugs if slugs is not None else list(corpus)
    if not selected or len(set(selected)) != len(selected) or set(selected) - corpus.keys():
        return ["invalid or duplicate expected corpus selection"]
    if slugs is None and len(selected) != 34:
        return ["production corpus is not exactly 34 briefs"]
    try:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"campaign evidence unreadable: {exc}"]
    errors = []
    if not summary.get("finished_at") or "wall_s" not in summary:
        errors.append("campaign has not finished")
    if summary.get("design_only") is not True or summary.get("judge") is not False:
        errors.append("campaign must be design-only with judge disabled")
    if summary.get("resumed") is not False or summary.get("resumed_reused_n", 0):
        errors.append("campaign freshness is not proven")
    if summary.get("source_unchanged") is not True:
        errors.append("campaign source stability is not proven")
    if summary.get("fresh_output_directory") is not True:
        errors.append("campaign output directory was not fresh")
    if slugs is None and (
        summary.get("requested_only") or summary.get("requested_limit") is not None
    ):
        errors.append("production acceptance forbids corpus selection flags")
    immutable = manifest.get("immutable") or {}
    if not immutable.get("source_fingerprint"):
        errors.append("campaign has no source fingerprint")
    expected = [corpus[slug] for slug in corpus if slug in selected]
    if immutable.get("llm_mode") != "live":
        errors.append("live provider provenance is not proven")
    if immutable.get("corpus") != expected or immutable.get("corpus_hash") != _stable_hash(
        expected
    ):
        errors.append("manifest corpus/brief identities do not match requested corpus")
    if immutable.get("repeats") != 1:
        errors.append("acceptance requires one fresh run per brief")
    cap = (immutable.get("caps") or {}).get("project_usd")
    if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
        errors.append("finite positive project budget is not recorded")
        cap = None
    runs = summary.get("runs") or []
    if Counter(run.get("slug") for run in runs) != Counter(selected):
        errors.append("run corpus has omitted, duplicated, or substituted briefs")
    if summary.get("n") != len(selected) or summary.get("design_committed") != len(selected):
        errors.append("summary does not report every expected design committed")
    for run in runs:
        slug = run.get("slug")
        prefix = f"{slug}: "
        if slug not in corpus:
            continue
        if (
            run.get("index") != corpus[slug]["index"]
            or _stable_hash(run.get("prompt")) != corpus[slug]["brief_hash"]
        ):
            errors.append(prefix + "run brief identity differs from corpus")
        if run.get("design_committed") is not True or any(
            run.get(key) for key in ("error", "design_error", "failure_kind", "design_failure_kind")
        ):
            errors.append(prefix + "terminal design failure")
        cost = run.get("design_cost_usd")
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            errors.append(prefix + "paid design cost is not recorded")
        elif cap is not None and cost > cap + 0.000001:
            errors.append(prefix + "design spend exceeds configured project cap")
        rundir = Path(run.get("rundir") or "")
        if not rundir.is_absolute():
            rundir = root / rundir
        if not rundir.resolve().is_relative_to(root):
            errors.append(prefix + "run evidence is outside fresh campaign")
            continue
        try:
            state = json.loads((rundir / ".kicraft/state.json").read_text(encoding="utf-8"))
            done = []
            with (rundir / "events.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get("kind") == "stage_done":
                        done.append(event)
            statuses = state.get("stage_status") or {}
            if any((statuses.get(stage) or {}).get("ok") is not True for stage in DESIGN_STAGES):
                errors.append(prefix + "state does not commit all five stages")
            successful = [event for event in done if event.get("ok") is True]
            latest = {event.get("stage"): event for event in done}
            if (
                [event.get("stage") for event in successful] != list(DESIGN_STAGES)
                or any((latest.get(stage) or {}).get("ok") is not True for stage in DESIGN_STAGES)
                or any(
                    event.get("failure_kind") or event.get("reused_work_units")
                    for event in successful
                )
            ):
                errors.append(prefix + "events do not prove five fresh successful stages in order")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(prefix + f"run evidence unreadable: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--only", help="diagnostic subset, never full production acceptance")
    args = parser.parse_args(argv)
    errors = verify_campaign(args.campaign, args.only.split(",") if args.only else None)
    if errors:
        print("design acceptance FAILED:\n" + "\n".join(f"- {error}" for error in errors))
        return 1
    print(f"design acceptance passed: {args.only or 'all 34 briefs'}; five fresh stages each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
