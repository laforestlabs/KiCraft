"""Frozen, resumable per-stage reliability campaign and exact release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from kicraft.cli.web_cost_report import load_stage_attempts
from kicraft.design.recipes import recipe_summaries
from kicraft.design.stage_semantics import DETECTOR_VERSION
from kicraft.fsutil import atomic_write_text
from kicraft.server.client import make_client
from kicraft.server.stage_runtime import drive_stage

STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")
CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_FLOOR = 306
CLAIM_MINIMUM = 299
_FORBIDDEN_FIELDS = {
    "brief",
    "messages",
    "reasoning",
    "answer",
    "candidate",
    "response",
    "tool_output",
}


def exact_zero_failure_lower_bound(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return 0.0
    return alpha ** (1.0 / n)


def statistical_release_gate(
    rows: list[dict],
    *,
    baseline_p95_cost: float | None = None,
    baseline_p95_wall_s: float | None = None,
) -> dict:
    valid = [row for row in rows if row.get("valid", True)]
    n = len(valid)
    provider_failures = sum(
        str(row.get("failure_kind") or "").startswith(("provider_", "transport_")) for row in valid
    )
    commit_failures = sum(not row.get("commit_ok", False) for row in valid)
    semantic_failures = sum(row.get("semantic_clean") is not True for row in valid)
    unlogged = sum(bool(row.get("unlogged_defects")) for row in valid)
    fab_escapes = sum(bool(row.get("fab_gate_escape")) for row in valid)
    continuation_failures = sum(
        bool(row.get("advisory_only")) and not row.get("commit_ok", False) for row in valid
    )

    def percentile(values, fraction):
        values = sorted(float(value) for value in values if value is not None)
        if not values:
            return None
        return values[min(len(values) - 1, max(0, int((len(values) - 1) * fraction + 0.999999)))]

    p95_cost = percentile((row.get("cost_usd") for row in valid), 0.95)
    p95_wall = percentile((row.get("wall_s") for row in valid), 0.95)
    cost_ok = baseline_p95_cost is None or (
        p95_cost is not None and p95_cost <= baseline_p95_cost * 1.10
    )
    wall_ok = baseline_p95_wall_s is None or (
        p95_wall is not None and p95_wall <= baseline_p95_wall_s * 1.10
    )
    gates = {
        "sample_size": n >= CLAIM_MINIMUM,
        "provider_availability": provider_failures == 0,
        "commit_availability": commit_failures == 0,
        "semantic_reliability": semantic_failures == 0,
        "corpus_expectations": unlogged == 0,
        "fab_gate_containment": fab_escapes == 0,
        "advisory_continuation": continuation_failures == 0,
        "cost_regression": cost_ok,
        "latency_regression": wall_ok,
    }
    return {
        "n": n,
        "zero_failure_lower_bound": exact_zero_failure_lower_bound(n),
        "failures": {
            "provider": provider_failures,
            "commit": commit_failures,
            "semantic": semantic_failures,
            "unlogged": unlogged,
            "fab_gate_escape": fab_escapes,
            "advisory_continuation": continuation_failures,
        },
        "p95_cost_usd": p95_cost,
        "p95_wall_s": p95_wall,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _canonical_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_corpus(path: Path) -> dict:
    corpus = json.loads(Path(path).read_text(encoding="utf-8"))
    if corpus.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("stage reliability corpus schema version mismatch")
    raw_cases = corpus.get("cases") or []
    templates = corpus.get("upstream_templates") or {}
    identities = [str(case.get("id")) for case in raw_cases]
    if len(identities) != len(set(identities)):
        raise ValueError("stage reliability corpus contains duplicate case IDs")
    counts = defaultdict(int)
    cases = []
    for raw_case in raw_cases:
        expected_hash = raw_case.get("input_hash")
        actual_hash = _canonical_hash(
            {key: value for key, value in raw_case.items() if key != "input_hash"}
        )
        if expected_hash != actual_hash:
            raise ValueError(f"case {raw_case.get('id')!r} input hash mismatch")
        case = dict(raw_case)
        template_id = case.pop("upstream_template", None)
        if template_id is not None:
            if template_id not in templates:
                raise ValueError(f"case {case.get('id')!r} has unknown upstream template")
            case["upstream_state"] = templates[template_id]
        if case.get("stage") not in STAGES:
            raise ValueError(f"unknown stage in case {case.get('id')!r}")
        counts[case["stage"]] += 1
        cases.append(case)
    for stage in STAGES:
        if counts[stage] < CAMPAIGN_FLOOR:
            raise ValueError(f"stage {stage} has {counts[stage]} cases; requires {CAMPAIGN_FLOOR}")
    manifest = corpus.get("manifest") or {}
    if int(manifest.get("detector_version", -1)) != DETECTOR_VERSION:
        raise ValueError("detector version drift")
    return {**corpus, "cases": cases, "_source_hash": _canonical_hash(corpus)}


def _validate_execution_manifest(corpus: dict, client) -> None:
    manifest = corpus.get("manifest") or {}
    settings = getattr(client, "s", None)
    if settings is not None:
        if manifest.get("model") != settings.model:
            raise ValueError("campaign model drift")
        if list(manifest.get("provider_order") or []) != list(settings.provider_order):
            raise ValueError("campaign provider order drift")
    expected_recipes = sorted(manifest.get("recipe_versions") or [])
    actual_recipes = sorted(row["recipe"] for row in recipe_summaries())
    if expected_recipes != actual_recipes:
        raise ValueError("campaign recipe version drift")
    if not isinstance(manifest.get("spend_envelope_usd"), (int, float)):
        raise ValueError("campaign spend envelope is not pinned")


def _assert_redacted_results(results: dict) -> None:
    for case_id, row in results.items():
        leaked = _FORBIDDEN_FIELDS & set(row)
        if leaked:
            raise ValueError(
                f"campaign result {case_id!r} contains private fields: {sorted(leaked)}"
            )


def run_campaign(corpus_path: Path, output_dir: Path, *, client=None) -> dict:
    corpus = load_corpus(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.json"
    completed = {}
    if checkpoint.is_file():
        checkpoint_doc = json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint_doc.get("corpus_hash") != corpus["_source_hash"]:
            raise ValueError("campaign checkpoint corpus identity mismatch")
        completed = checkpoint_doc.get("results", {})
        _assert_redacted_results(completed)
    client = client or make_client()
    _validate_execution_manifest(corpus, client)
    for case in corpus["cases"]:
        case_id = case["id"]
        if case_id in completed:
            continue
        workspace = output_dir / "work" / case_id
        workspace.mkdir(parents=True, exist_ok=True)
        state_path = workspace / ".kicraft" / "state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(state_path, json.dumps(case.get("upstream_state") or {}, indent=2) + "\n")
        result = drive_stage(
            client,
            case["stage"],
            case["brief"],
            state_path,
            workspace,
            meta_ctx={"run_id": case_id},
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        status = (state.get("stage_status") or {}).get(case["stage"], {})
        observed = {row.get("code") for row in status.get("diagnostics") or []}
        expected = set(case.get("expected_diagnostic_codes") or [])
        ledger_path = getattr(getattr(client, "guard", None), "path", None)
        attempt_rows = []
        if ledger_path is not None:
            attempt_rows = [
                row
                for row in load_stage_attempts(ledger_path)
                if row.get("run_id") == case_id and row.get("stage") == case["stage"]
            ]
        completed[case_id] = {
            "id": case_id,
            "stage": case["stage"],
            "commit_ok": bool(result.get("commit_ok")),
            "failure_kind": result.get("failure_kind"),
            "semantic_clean": status.get("semantic_clean"),
            "diagnostic_codes": sorted(observed),
            "unlogged_defects": sorted(expected - observed),
            "fab_gate_escape": bool(
                any(row.get("severity") == "fab_gate" for row in status.get("diagnostics") or [])
                and status.get("fab_safe") is not False
            ),
            "advisory_only": bool(status.get("diagnostics"))
            and all(row.get("severity") == "advisory" for row in status.get("diagnostics") or []),
            "attempt_count": len(attempt_rows) if ledger_path is not None else None,
            "cost_usd": result.get("cost_usd"),
            "wall_s": result.get("wall_s"),
            "valid": ledger_path is None or bool(attempt_rows),
        }
        _assert_redacted_results({case_id: completed[case_id]})
        spent = sum(float(row.get("cost_usd") or 0.0) for row in completed.values())
        if spent > float(corpus["manifest"]["spend_envelope_usd"]):
            raise ValueError("campaign spend envelope exceeded")
        atomic_write_text(
            checkpoint,
            json.dumps({"corpus_hash": corpus["_source_hash"], "results": completed}, indent=2)
            + "\n",
        )
    by_stage = {
        stage: statistical_release_gate(
            [row for row in completed.values() if row["stage"] == stage]
        )
        for stage in STAGES
    }
    report = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "corpus_hash": corpus["_source_hash"],
        "results": completed,
        "release_gates": by_stage,
    }
    atomic_write_text(output_dir / "report.json", json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen per-stage reliability campaign")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).with_name("stage_reliability_corpus.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("stage-reliability-results"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the frozen corpus without provider calls",
    )
    args = parser.parse_args(argv)
    if args.check:
        corpus = load_corpus(args.corpus)
        print(f"valid corpus: {len(corpus['cases'])} cases sha256={corpus['_source_hash']}")
        return 0
    report = run_campaign(args.corpus, args.output)
    print(json.dumps(report["release_gates"], indent=2))
    return 0 if all(row["passed"] for row in report["release_gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
