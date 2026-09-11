"""Read-only structured failure matrix for completed self-eval campaigns."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .self_eval import _campaign_costs, _design_event_costs, _stage_failure_attribution

_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")


def analyze_campaign(path: Path) -> dict:
    root = path.resolve()
    summary_path = root / "summary.json" if root.is_dir() else root
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    root = summary_path.parent
    stage_counts: Counter[str] = Counter()
    family_counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    runs: list[dict] = []
    cost_records: list[dict] = []
    for record in summary.get("runs") or []:
        run_dir = Path(record.get("rundir") or root / str(record.get("stem") or ""))
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        state_path = run_dir / ".kicraft" / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        attribution = _stage_failure_attribution(state, run_dir / "events.jsonl")
        passed = record.get("design_committed") is True
        stage = (
            "passed"
            if passed
            else str(record.get("failed_stage") or attribution.get("failed_stage") or "unknown")
        )
        family = (
            "passed"
            if passed
            else str(
                record.get("failure_kind")
                or record.get("design_failure_kind")
                or (
                    "budget_refused"
                    if str(record.get("error") or "").startswith("BudgetExceeded:")
                    else attribution.get("failure_kind")
                )
                or "unattributed"
            )
        )
        stage_counts[stage] += 1
        family_counts[(stage, family)] += 1
        slug = str(record.get("slug") or record.get("stem") or "unknown")
        examples[(stage, family)].append(slug)
        stage_costs = record.get("stage_cost_usd") or _design_event_costs(run_dir / "events.jsonl")
        cost_record = {
            **record,
            "stage_cost_usd": stage_costs,
            "design_cost_usd": max(
                float(record.get("design_cost_usd") or 0.0), sum(stage_costs.values())
            ),
        }
        cost_records.append(cost_record)
        runs.append(
            {
                "slug": slug,
                "terminal_stage": stage,
                "failure_kind": family,
                "failure_codes": attribution.get("failure_codes") or [],
                "work_unit_ids": attribution.get("work_unit_ids") or [],
                "accepted_siblings_retained": attribution.get("accepted_siblings_retained"),
                "design_cost_usd": round(cost_record["design_cost_usd"], 6),
                "stage_cost_usd": stage_costs,
                "budget_refusal": record.get("budget_refusal"),
            }
        )
    ordered_stages = (*_STAGES, "unknown", "passed")
    return {
        "campaign": str(root),
        "n": len(runs),
        "stage_counts": {
            stage: stage_counts[stage] for stage in ordered_stages if stage_counts[stage]
        },
        "failure_matrix": [
            {
                "stage": stage,
                "failure_kind": family,
                "count": count,
                "briefs": examples[(stage, family)],
            }
            for (stage, family), count in sorted(family_counts.items())
        ],
        "runs": runs,
        **_campaign_costs(cost_records),
    }


def _render(report: dict) -> str:
    lines = ["terminal_stage\tcount"]
    lines.extend(f"{stage}\t{count}" for stage, count in report["stage_counts"].items())
    lines.append("")
    lines.append("stage\tfailure_kind\tcount\tbriefs")
    lines.extend(
        f"{row['stage']}\t{row['failure_kind']}\t{row['count']}\t{','.join(row['briefs'])}"
        for row in report["failure_matrix"]
    )
    lines.append("")
    lines.append(f"total_cost_usd\t{report['total_cost_usd']}")
    lines.append(f"failed_run_cost_usd\t{report['failed_run_cost_usd']}")
    lines.append(f"cost_per_committed_design_usd\t{report['cost_per_committed_design_usd']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = analyze_campaign(args.campaign)
    print(json.dumps(report, indent=2) if args.as_json else _render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
