"""Privacy-safe integrity checker and deterministic LLM canary analyzer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kicraft.cli.web_cost_report import load_rows, load_stage_attempts, load_stage_runs
from kicraft.design.models import Architecture, BOM
from kicraft.design.synthesis.validation import (
    bom_parts_on_unknown_sheets,
    mcu_programming_facts,
)
from kicraft.eval.self_eval import _PROMOTED_PARENT_RCS, _find_parent_board, _stable_hash
from kicraft.server.stage_runtime import _commit_rejection_signature, _offender_identity
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

from .llm_canary import COHORT, ENVELOPE_USD, REFERENCE_BATCH

SCHEMA_VERSION = 2
STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring", "judge")
_CLASSIFICATIONS = {
    "operational",
    "reasoning",
    "serialization",
    "schema_contract",
    "commit_contract",
    "question_or_reconcile",
    "design_complete",
}
_FORBIDDEN_EVENT_FIELDS = {
    "text",
    "output",
    "reasoning",
    "reasoning_delta",
    "answer",
    "answer_delta",
    "candidate",
    "brief",
    "prompt",
}
_WITNESSES = ("1/701", "1/748", "1/749", "1/754")
_COMPARISON_SLUGS = ("rounded-c3-devboard", "snowman-ornament")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def not_observable(reason: str) -> dict:
    return {"status": "not_observable", "reason": reason}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return not_observable("artifact path is outside campaign directory")  # type: ignore[return-value]


def _stream_events(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid events JSON at {path}:{number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"non-object event at {path}:{number}")
            yield event


def _scalar(value):
    return value if isinstance(value, (str, int, float, bool)) or value is None else None


def normalize_tool_signature(event: dict) -> str:
    name = str(event.get("name") or "unknown")
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    normalized = []
    for key, value in sorted(args.items()):
        scalar = _scalar(value)
        if scalar is None and value is not None:
            continue
        if "mpn" in str(key).lower() and isinstance(scalar, str):
            scalar = scalar.lower()
        normalized.append((str(key), scalar))
    return name + "|" + json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)


def _event_signature(event: dict) -> dict | None:
    if event.get("kind") != "retry" or not (event.get("errors") or event.get("offenders")):
        return None
    gates, offenders = _commit_rejection_signature(event)
    normalized_gates = []
    for gate in gates:
        gate_text = str(gate)
        ids = re.findall(r"(?:§\s*)?(9\.\d+)", gate_text)
        if ids:
            normalized_gates.extend(ids)
            continue
        lower = gate_text.lower()
        if "duplicate connection" in lower:
            label = "duplicate_connection"
        elif "already assigned" in lower:
            label = "endpoint_already_assigned"
        elif "validation error" in lower or "wiring patch rejected" in lower:
            label = "patch_validation"
        elif "schema" in lower:
            label = "invalid_patch_schema"
        else:
            label = "commit_rejected"
        if label not in normalized_gates:
            normalized_gates.append(label)
    return {"gates": normalized_gates, "offenders": list(offenders)}


def _signature_key(signature: dict) -> tuple:
    return tuple(signature.get("gates") or ()), tuple(signature.get("offenders") or ())


def _family(gates: list[str], errors: list[str] | None = None) -> str:
    joined = " ".join(gates + (errors or [])).lower()
    if "9.19" in joined:
        return "9.19_multi_net_pin"
    if "9.17" in joined:
        return "9.17_two_terminal_short"
    if "9.15" in joined:
        return "9.15_dangling_net"
    if "9.21" in joined:
        return "9.21_reference_integrity"
    if "9.29" in joined:
        return "9.29_programming_access"
    if "rail" in joined and "ref" in joined:
        return "rail_as_ref"
    if any(word in joined for word in ("unknown", "coverage", "unwired")):
        return "unknown_or_coverage"
    if "reconcile" in joined or "missing bom" in joined:
        return "bom_reconcile"
    return "other"


def _artifact_dir(batch: Path, record: dict) -> Path:
    candidate = Path(record.get("rundir") or batch / str(record.get("stem") or ""))
    if not candidate.is_absolute():
        candidate = batch / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(batch.resolve())
    except ValueError as exc:
        raise ValueError(f"run directory outside campaign: {candidate}") from exc
    return resolved


def _load_ledger(ledger: Path) -> tuple[list[dict], list[dict]]:
    if not ledger.is_file():
        return [], []
    return load_rows(ledger), load_stage_runs(ledger)


def _expected_response_policy(stage: str, frozen: dict) -> set[str]:
    policies = frozen.get("response_policies")
    if not isinstance(policies, dict):
        return set()
    if stage == "judge":
        value = policies.get("judge")
        return {str(value)} if value else set()

    allowed: set[str] = set()
    shared = policies.get("bom_and_wiring") if stage in {"bom", "wiring"} else None
    design = shared or policies.get(stage) or policies.get("design")
    if design:
        allowed.add(str(design).replace("<stage>", stage))
    if stage == "wiring" and policies.get("wiring_patch"):
        allowed.add(str(policies["wiring_patch"]))
    return allowed


def _integrity(batch: Path, spend_rows: list[dict], stage_rows: list[dict]) -> tuple[dict, dict]:
    errors: list[str] = []
    paths = {
        "canary": batch / "canary_manifest.json",
        "campaign": batch / "campaign_manifest.json",
        "summary": batch / "summary.json",
        "designer_preflight": batch / "preflight-designer.json",
        "judge_preflight": batch / "preflight-judge.json",
    }
    loaded: dict[str, dict] = {}
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing {path.name}")
            continue
        try:
            loaded[name] = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {path.name}: {exc}")
    if errors:
        return {"valid": False, "errors": errors}, loaded

    canary = loaded["canary"]
    campaign = loaded["campaign"]
    summary = loaded["summary"]
    immutable = canary.get("immutable") or {}
    frozen = campaign.get("immutable") or {}
    designer = immutable.get("designer") or {}
    judge = immutable.get("judge") or {}
    cohort = immutable.get("cohort") or []
    expected_slugs = [row.get("slug") for row in cohort]
    if canary.get("schema_version") != 1 or campaign.get("schema_version") != 1:
        errors.append("manifest schema version mismatch")
    if canary.get("run_status") != "batch_complete":
        errors.append(f"canary run_status is {canary.get('run_status')!r}, not batch_complete")
    if immutable.get("envelope_usd") != ENVELOPE_USD:
        errors.append("campaign envelope mismatch")
    if expected_slugs != list(COHORT):
        errors.append("canary cohort/order mismatch")
    checkout = immutable.get("checkout") or {}
    if checkout.get("dirty_paths"):
        errors.append("campaign checkout contained uncommitted runtime/config changes")
    if frozen.get("code_revision") != (immutable.get("checkout") or {}).get("commit"):
        errors.append("checkout commit differs between manifests")
    if frozen.get("design_profile") != designer.get("profile"):
        errors.append("designer profile differs between manifests")
    if frozen.get("design_model") != designer.get("model"):
        errors.append("designer model differs between manifests")
    if frozen.get("design_provider_order") != designer.get("provider_order"):
        errors.append("designer providers differ between manifests")
    if frozen.get("judge_model") != judge.get("model"):
        errors.append("judge model differs between manifests")
    if frozen.get("judge_provider_order") != judge.get("provider_order"):
        errors.append("judge providers differ between manifests")
    for stage in STAGES:
        if not _expected_response_policy(stage, frozen):
            errors.append(f"campaign manifest lacks frozen response policy for {stage}")
    if frozen.get("repeats") != 1:
        errors.append("campaign manifest repeats is not one")
    campaign_slugs = [row.get("slug") for row in frozen.get("corpus") or []]
    if campaign_slugs != list(COHORT):
        errors.append("campaign manifest cohort/order mismatch")
    benchmark = {entry["slug"]: entry for entry in BENCHMARK_PROMPTS}
    expected_campaign_hashes = [
        _stable_hash(benchmark[slug]["brief"]) for slug in COHORT if slug in benchmark
    ]
    campaign_hashes = [row.get("brief_hash") for row in frozen.get("corpus") or []]
    if campaign_hashes != expected_campaign_hashes:
        errors.append("campaign brief hashes differ from BENCHMARK_PROMPTS")
    for role in ("designer", "judge"):
        role_artifact = loaded[f"{role}_preflight"]
        expected_role = designer if role == "designer" else judge
        if not role_artifact.get("ok"):
            errors.append(f"{role} preflight is not OK")
        if role_artifact.get("role") != role:
            errors.append(f"{role} preflight role mismatch")
        if role_artifact.get("model") != expected_role.get("model"):
            errors.append(f"{role} preflight model mismatch")
        if role_artifact.get("provider_order") != expected_role.get("provider_order"):
            errors.append(f"{role} preflight providers mismatch")
        reference = (canary.get("preflights") or {}).get(role) or {}
        artifact = paths[f"{role}_preflight"]
        if reference.get("path") != artifact.name:
            errors.append(f"{role} preflight path mismatch")
        elif reference.get("sha256"):
            import hashlib

            if hashlib.sha256(artifact.read_bytes()).hexdigest() != reference["sha256"]:
                errors.append(f"{role} preflight hash mismatch")

    campaign_id = immutable.get("campaign_id")
    preflight_spend = [
        row
        for row in spend_rows
        if (row.get("meta") or {}).get("campaign_id") == campaign_id
        and (row.get("meta") or {}).get("phase") == "model_preflight"
    ]
    expected_preflight_cost = 0.0
    for role in ("designer", "judge"):
        smoke = loaded[f"{role}_preflight"].get("smoke") or {}
        role_cost = float(smoke.get("cost_usd") or 0.0)
        expected_preflight_cost += role_cost
        if role_cost > 0 and not any(
            (row.get("meta") or {}).get("role") == role for row in preflight_spend
        ):
            errors.append(f"{role} preflight has no campaign-attributed spend row")
    actual_preflight_cost = sum(float(row.get("cost_usd") or 0.0) for row in preflight_spend)
    if abs(expected_preflight_cost - actual_preflight_cost) > 0.000002:
        errors.append(
            "preflight artifact/ledger cost mismatch "
            f"({expected_preflight_cost:.6f} != {actual_preflight_cost:.6f})"
        )

    records = summary.get("runs")
    if not isinstance(records, list):
        records = []
        errors.append("summary runs is not a list")
    slugs = [record.get("slug") for record in records if isinstance(record, dict)]
    if len(records) != 9 or len(set(slugs)) != 9 or slugs != list(COHORT):
        errors.append("summary must contain exactly nine unique fixed-cohort records in order")
    checks = {
        "repeats": 1,
        "parallel": 1,
        "build_slots": 1,
        "full_events": True,
        "judge": True,
    }
    for key, expected in checks.items():
        if summary.get(key) != expected:
            errors.append(f"summary {key} must equal {expected!r}")
    if summary.get("design_model") != designer.get("model"):
        errors.append("summary designer model mismatch")
    if summary.get("design_profile") != designer.get("profile"):
        errors.append("summary designer profile mismatch")
    if summary.get("design_provider_order") != designer.get("provider_order"):
        errors.append("summary designer providers mismatch")
    if summary.get("judge_model") != judge.get("model"):
        errors.append("summary judge model mismatch")

    by_run_spend = defaultdict(list)
    for row in spend_rows:
        run_id = row.get("meta", {}).get("run_id")
        if run_id:
            by_run_spend[str(run_id)].append(row)
    by_run_stage = defaultdict(list)
    for row in stage_rows:
        if row.get("run_id"):
            by_run_stage[str(row["run_id"])].append(row)

    def provider_identities(role: str) -> set[str]:
        role_manifest = designer if role == "designer" else judge
        role_preflight = loaded[f"{role}_preflight"]
        identities = {str(value).lower() for value in role_manifest.get("provider_order") or []}
        for endpoint in role_preflight.get("endpoints") or []:
            if not isinstance(endpoint, dict):
                continue
            for key in ("provider", "tag"):
                if endpoint.get(key):
                    identities.add(str(endpoint[key]).lower())
        return identities

    designer_provider_ids = provider_identities("designer")
    judge_provider_ids = provider_identities("judge")

    for record in records:
        if not isinstance(record, dict):
            errors.append("summary contains non-object record")
            continue
        run_id = record.get("run_id")
        if not run_id:
            errors.append(f"{record.get('slug')}: missing exact run_id")
            continue
        try:
            rundir = _artifact_dir(batch, record)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        required = (
            rundir / "brief.txt",
            rundir / "events.jsonl",
            rundir / ".kicraft" / "state.json",
            rundir / "eval" / "report.json",
        )
        for path in required:
            if not path.is_file():
                errors.append(f"{record.get('slug')}: missing {_relative(path, batch)}")
        if not by_run_stage.get(str(run_id)):
            errors.append(f"{record.get('slug')}: no exact stage_runs attribution for {run_id}")
        if (rundir / "events.jsonl").is_file():
            try:
                event_kinds = [
                    event.get("kind") for event in _stream_events(rundir / "events.jsonl")
                ]
            except ValueError as exc:
                errors.append(str(exc))
                event_kinds = []
            if record.get("design_status") == "ok":
                if "build_start" not in event_kinds or "build_done" not in event_kinds:
                    errors.append(f"{record.get('slug')}: design complete without build evidence")
                if record.get("build_rc") is None:
                    errors.append(f"{record.get('slug')}: design complete with null build_rc")
            if (
                record.get("build_rc") in _PROMOTED_PARENT_RCS
                and _find_parent_board(rundir) is None
            ):
                errors.append(f"{record.get('slug')}: promoted build rc without parent board")
        attributed = by_run_spend.get(str(run_id), [])
        expected_cost = round(
            float(record.get("design_cost_usd") or 0.0)
            + float(record.get("judge_cost_usd") or 0.0),
            6,
        )
        actual_cost = round(sum(float(row.get("cost_usd") or 0.0) for row in attributed), 6)
        if abs(expected_cost - actual_cost) > 0.000003:
            errors.append(
                f"{record.get('slug')}: summary/ledger cost mismatch "
                f"({expected_cost:.6f} != {actual_cost:.6f})"
            )
        observed_providers: dict[str, set[str]] = defaultdict(set)
        for row in by_run_spend.get(str(run_id), []):
            meta = row.get("meta") or {}
            stage = str(meta.get("stage") or meta.get("phase") or "")
            expected_model = judge.get("model") if stage == "judge" else designer.get("model")
            if meta.get("provider"):
                observed_providers[stage].add(str(meta["provider"]))
            if row.get("model") and row.get("model") != expected_model:
                errors.append(f"{record.get('slug')}: billed model drift at {stage}")
            if meta.get("profile") and meta.get("profile") != designer.get("profile"):
                errors.append(f"{record.get('slug')}: billed profile drift at {stage}")
            policy_name = meta.get("response_policy_name")
            allowed_policies = _expected_response_policy(stage, frozen)
            if policy_name is not None and policy_name not in allowed_policies:
                errors.append(f"{record.get('slug')}: response policy drift at {stage}")
        for stage, observed in sorted(observed_providers.items()):
            allowed = judge_provider_ids if stage == "judge" else designer_provider_ids
            unexpected = sorted(
                provider for provider in observed if provider.lower() not in allowed
            )
            if unexpected:
                errors.append(
                    f"{record.get('slug')}: billed provider drift at {stage}: observed {unexpected}"
                )
        stages_with_spend = defaultdict(list)
        for row in attributed:
            meta = row.get("meta") or {}
            stages_with_spend[str(meta.get("stage") or meta.get("phase") or "")].append(meta)
        for stage, metas in stages_with_spend.items():
            if stage in STAGES and not any(meta.get("response_policy_name") for meta in metas):
                errors.append(f"{record.get('slug')}: response contract not observable at {stage}")
    loaded["preflight_spend"] = preflight_spend
    integrity = {
        "valid": not errors,
        "errors": errors,
        "record_count": len(records),
        "expected_record_count": 9,
        "exact_run_attribution": not any("attribution" in error for error in errors),
    }
    loaded["spend_by_run"] = by_run_spend
    loaded["stage_by_run"] = by_run_stage
    return integrity, loaded


def _classify(record: dict, state: dict, events: list[dict]) -> tuple[str, str | None, str | None]:
    if record.get("error"):
        return "operational", None, "harness_error"
    statuses = state.get("stage_status") if isinstance(state.get("stage_status"), dict) else {}
    failed_stage = None
    failure_kind = None
    for stage in STAGES[:-1]:
        row = statuses.get(stage) if isinstance(statuses, dict) else None
        if isinstance(row, dict) and row.get("ok") is False:
            failed_stage = stage
            failure_kind = row.get("failure_kind")
            break
    all_done = all(
        isinstance(statuses.get(stage), dict) and statuses[stage].get("ok") is True
        for stage in STAGES[:-1]
    )
    if record.get("design_status") == "ok" or all_done:
        return "design_complete", None, None
    terminal_text = " ".join(
        str(value)
        for value in (
            failure_kind,
            record.get("design_error"),
            record.get("error"),
        )
        if value
    ).lower()
    if failure_kind in {"provider_error", "transport_error"} or any(
        token in terminal_text for token in ("budget", "kill switch", "transport", "provider error")
    ):
        return "operational", failed_stage, failure_kind or "operational"
    if failure_kind == "reasoning_loop" or "reasoning" in terminal_text:
        return "reasoning", failed_stage, failure_kind or "reasoning_loop"
    if (
        failure_kind in {"collection_limit", "truncated_json", "invalid_json"}
        or "no json" in terminal_text
    ):
        return "serialization", failed_stage, failure_kind or "invalid_json"
    if failure_kind == "invalid_schema" or any(
        phrase in terminal_text
        for phrase in ("stage contract failed", "stage-prep failed", "schema")
    ):
        return "schema_contract", failed_stage, failure_kind or "invalid_schema"
    if failure_kind == "commit_rejected":
        return "commit_contract", failed_stage, failure_kind
    questions = state.get("open_questions") or []
    if (
        record.get("design_status") == "needs_input"
        or questions
        or any(event.get("kind") == "question" for event in events)
    ):
        return "question_or_reconcile", failed_stage, failure_kind
    return "operational", failed_stage, failure_kind or "untyped_terminal"


def _stage_analysis(
    stage: str, events: list[dict], spend: list[dict], stage_rows: list[dict]
) -> dict:
    stage_events = [event for event in events if event.get("stage") == stage]
    calls = [
        row
        for row in spend
        if str((row.get("meta") or {}).get("stage") or (row.get("meta") or {}).get("phase"))
        == stage
    ]
    resources = [row for row in stage_rows if row.get("stage") == stage]
    meta = [row.get("meta") or {} for row in calls]
    modes = Counter()
    for item in meta:
        if item.get("serialization"):
            modes["serialization"] += 1
        elif item.get("phase") == "tools-final":
            modes["tool_final"] += 1
        elif item.get("phase") == "tools":
            modes["tool_round"] += 1
        else:
            modes["normal"] += 1
    reasoning_chars = sum(int(item.get("reasoning_chars") or 0) for item in meta)
    return {
        "status": (
            "ok"
            if any(event.get("kind") == "stage_done" and event.get("ok") for event in stage_events)
            else (
                "failed"
                if any(
                    event.get("kind") == "stage_done" and event.get("ok") is False
                    for event in stage_events
                )
                else "not_run"
            )
        ),
        "attempts": max((row.get("attempts") or 0 for row in resources), default=0),
        "billed_provider_calls": len(calls),
        "call_modes": dict(sorted(modes.items())),
        "response_policy_names": sorted(
            {item.get("response_policy_name") for item in meta if item.get("response_policy_name")}
        ),
        "providers": sorted({item.get("provider") for item in meta if item.get("provider")}),
        "models": sorted({row.get("model") for row in calls if row.get("model")}),
        "finish_reasons": sorted(
            {item.get("finish_reason") for item in meta if item.get("finish_reason")}
        ),
        "reasoning_policies": sorted(
            {
                item.get("reasoning_policy_name")
                for item in meta
                if item.get("reasoning_policy_name")
            }
        ),
        "tokens": {
            "input": sum(int(row.get("input_tokens") or 0) for row in calls),
            "output": sum(int(row.get("output_tokens") or 0) for row in calls),
            "reasoning": not_observable(
                "ledger records reasoning characters, not reasoning tokens"
            ),
            "reasoning_chars": reasoning_chars,
            "cache": sum(int(row.get("cached_tokens") or 0) for row in calls),
        },
        "cost_usd": round(sum(float(row.get("cost_usd") or 0.0) for row in calls), 6),
        "wall_s": round(sum(float(row.get("wall_s") or 0.0) for row in resources), 3),
        "cpu_s": round(sum(float(row.get("cpu_s") or 0.0) for row in resources), 3),
        "rounds": sum(int(row.get("rounds") or 0) for row in resources),
        "tool_calls": sum(int(row.get("tool_calls") or 0) for row in resources),
        "failure_kinds": sorted(
            {row.get("failure_kind") for row in resources if row.get("failure_kind")}
        ),
        "emitted_collection_count": max(
            (int(row.get("emitted_collection_count") or 0) for row in resources),
            default=0,
        ),
        "expanded_component_count": max(
            (int(row.get("expanded_component_count") or 0) for row in resources),
            default=0,
        ),
        "response_contract": (
            "observable"
            if calls
            and all(
                (item.get("response_policy_name") is not None)
                for item in meta
                if item.get("phase") not in {"tools"}
            )
            else not_observable("one or more billed rows omit a response policy name")
        ),
    }


def _bom_analysis(state: dict, events: list[dict], stages: dict) -> dict:
    tools = [event for event in events if event.get("kind") == "tool"]
    signatures = [normalize_tool_signature(event) for event in tools]
    counts = Counter(signatures)
    decoded = [
        event
        for event in events
        if event.get("kind") == "candidate_decoded" and event.get("stage") == "bom"
    ]
    recoveries = [
        event
        for event in events
        if event.get("kind") == "serialization_recovery" and event.get("stage") == "bom"
    ]
    bom = state.get("bom") if isinstance(state.get("bom"), dict) else {}
    parts = bom.get("parts") if isinstance(bom.get("parts"), list) else []
    architecture = state.get("architecture") if isinstance(state.get("architecture"), dict) else {}
    known = {
        str(sheet.get("name"))
        for sheet in architecture.get("sheets") or []
        if isinstance(sheet, dict) and sheet.get("name")
    }
    committed_unknown = [
        {"ref": str(part.get("ref") or ""), "sheet": str(part.get("sheet") or "")}
        for part in parts
        if isinstance(part, dict) and str(part.get("sheet") or "") not in known
    ]
    candidate_unknown = [
        violation
        for event in decoded
        for violation in event.get("unknown_sheet_references") or []
        if isinstance(violation, dict)
    ]
    mpn_lookups = sum(
        count
        for signature, count in counts.items()
        if "mpn" in signature.lower() or "part" in signature.lower()
    )
    return {
        "tool_calls": len(tools),
        "tool_signatures": [
            {"signature": key, "count": value} for key, value in sorted(counts.items())
        ],
        "repeated_tool_signatures": sum(value - 1 for value in counts.values() if value > 1),
        "lookup_mpn_count": mpn_lookups,
        "tool_cap_hit": bool(stages.get("bom", {}).get("rounds", 0) > 6 or len(tools) >= 12),
        "recovery_ledger_entries": sum(
            int(event.get("resolution_ledger_entries") or 0) for event in recoveries
        ),
        "emitted_collection_count": int(stages.get("bom", {}).get("emitted_collection_count") or 0),
        "expanded_component_count": max(
            (int(event.get("expanded_component_count") or 0) for event in decoded),
            default=0,
        ),
        "committed_part_count": len(parts),
        "architecture_sheets": sorted(known),
        "candidate_unknown_sheet_references": candidate_unknown,
        "committed_unknown_sheet_references": committed_unknown,
        "unresolved_symbol_footprint": [
            str(part.get("ref") or "")
            for part in parts
            if isinstance(part, dict) and (not part.get("symbol") or not part.get("footprint"))
        ],
        "stock_retail_rejections": not_observable(
            "commit events do not expose a typed stock/retail field"
        ),
        "executor_memo_cache": not_observable("events do not identify executor memo-cache hits"),
    }


def _wiring_analysis(events: list[dict], stages: dict) -> dict:
    decoded = [
        event
        for event in events
        if event.get("kind") == "candidate_decoded" and event.get("stage") == "wiring"
    ]
    signatures = [signature for event in events if (signature := _event_signature(event))]
    keys = [_signature_key(signature) for signature in signatures]
    progression = []
    previous = None
    seen = set()
    for signature, key in zip(signatures, keys):
        offender_count = len(signature["offenders"])
        if previous is None:
            progress = "initial"
        elif offender_count < previous[1] or key != previous[0]:
            progress = "progress"
        else:
            progress = "no_progress"
        if key in seen and progress != "initial":
            progress = "no_progress"
        progression.append({**signature, "offender_count": offender_count, "progress": progress})
        seen.add(key)
        previous = (key, offender_count)
    clean_slate = [event for event in decoded if event.get("clean_slate")]
    no_progress = any(row["progress"] == "no_progress" for row in progression) and bool(clean_slate)
    families = Counter(_family(row["gates"]) for row in progression)
    return {
        "first_call_outcome": (
            "decoded" if decoded else ("rejected_or_unparseable" if signatures else "not_run")
        ),
        "full_calls": len(decoded),
        "correction_calls": max(0, len(decoded) - 1),
        "ordered_rejection_signatures": progression,
        "families": dict(sorted(families.items())),
        "clean_slate_transitions": len(clean_slate),
        "no_progress": no_progress,
        "final_gate": progression[-1] if progression else None,
        "cost_usd": stages.get("wiring", {}).get("cost_usd", 0.0),
    }


def _programming_analysis(state: dict) -> dict:
    architecture = state.get("architecture") if isinstance(state.get("architecture"), dict) else {}
    bom = state.get("bom") if isinstance(state.get("bom"), dict) else {}
    sheets = sorted(
        str(sheet.get("name"))
        for sheet in architecture.get("sheets") or []
        if isinstance(sheet, dict) and sheet.get("name")
    )
    power_nets = sorted(str(net) for net in architecture.get("power_nets") or [])
    facts = None
    unknown = []
    try:
        arch_model = Architecture.model_validate(architecture)
        bom_model = BOM.model_validate(bom)
        unknown = [
            {"ref": ref, "sheet": sheet}
            for ref, sheet in bom_parts_on_unknown_sheets(arch_model, bom_model)
        ]
        facts = mcu_programming_facts(bom_model)
    except Exception:
        facts = None
    texts = " ".join(
        str(value)
        for part in bom.get("parts") or []
        if isinstance(part, dict)
        for value in (part.get("value"), part.get("description"), part.get("mpn"))
        if value
    ).upper()
    if "USB-UART" in texts or any(chip in texts for chip in ("CP210", "CH340", "FT232")):
        usb_path = "usb_uart"
    elif "USB" in texts and facts:
        usb_path = "native_usb"
    else:
        usb_path = not_observable("committed BOM does not identify a USB programming path")
    return {
        "architecture_sheets": sheets,
        "architecture_power_nets": power_nets,
        "power_net_sheet_name_overlap": sorted(set(sheets) & set(power_nets)),
        "bom_parts_on_unknown_sheets": unknown,
        "mcu_programming_facts": facts
        if facts is not None
        else not_observable("no valid committed MCU programming facts"),
        "usb_path": usb_path,
    }


def _question_analysis(state: dict, events: list[dict], record: dict) -> dict:
    question_events = [event for event in events if event.get("kind") == "question"]
    questions = [
        question
        for event in question_events
        for question in event.get("questions") or []
        if isinstance(question, dict)
    ]
    return {
        "blocking_questions": sum(bool(question.get("blocking")) for question in questions),
        "auto_answers": int(record.get("questions") or 0),
        "reconcile_parks": sum(bool(question.get("reconcile_target")) for question in questions),
        "resume_rounds": len(question_events),
        "open_question_count": len(state.get("open_questions") or []),
    }


def _one_run(batch: Path, record: dict, spend: list[dict], stage_rows: list[dict]) -> dict:
    rundir = _artifact_dir(batch, record)
    state = _read_json(rundir / ".kicraft" / "state.json")
    events = list(_stream_events(rundir / "events.jsonl"))
    classification, failed_stage, failure_kind = _classify(record, state, events)
    stages = {stage: _stage_analysis(stage, events, spend, stage_rows) for stage in STAGES}
    bom = _bom_analysis(state, events, stages)
    wiring = _wiring_analysis(events, stages)
    programming = _programming_analysis(state)
    questions = _question_analysis(state, events, record)
    gate_signatures = [signature for event in events if (signature := _event_signature(event))]
    build_rc = record.get("build_rc")
    if build_rc is None:
        build_outcome = "not_run"
    elif build_rc == 0:
        build_outcome = "fab_ready"
    elif build_rc == 7:
        build_outcome = "drc_failed"
    else:
        build_outcome = f"rc_{build_rc}"
    total_cost = round(sum(float(row.get("cost_usd") or 0.0) for row in spend), 6)
    return {
        "identity": {
            "slug": record.get("slug"),
            "archetype": record.get("archetype"),
            "run_id": record.get("run_id"),
            "repeat": record.get("repeat"),
            "attribution_method": "exact_summary_run_id",
        },
        "design_outcome": classification,
        "build_outcome": build_outcome,
        "failed_stage": failed_stage,
        "classification": classification,
        "failure_kind": failure_kind,
        "gate_signatures": gate_signatures,
        "stages": stages,
        "bom": bom,
        "wiring": wiring,
        "programming": programming,
        "questions": questions,
        "cost_usd": total_cost,
        "wall_s": round(float(record.get("duration_s") or 0.0), 1),
        "artifacts": {
            "run_dir": _relative(rundir, batch),
            "events": _relative(rundir / "events.jsonl", batch),
            "state": _relative(rundir / ".kicraft" / "state.json", batch),
            "report": _relative(rundir / "eval" / "report.json", batch),
        },
    }


def _baseline(path: Path) -> dict:
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.is_file():
        return not_observable(f"baseline summary missing: {summary_path}")
    summary = _read_json(summary_path)
    records = [record for record in summary.get("runs") or [] if record.get("slug") in COHORT]
    by_slug = {record.get("slug"): record for record in records}
    complete = sum(record.get("design_status") == "ok" for record in records)
    classes = Counter()
    for record in records:
        text = " ".join(str(record.get(key) or "") for key in ("design_error", "error")).lower()
        if record.get("design_status") == "ok":
            classes["design_complete"] += 1
        elif "trunc" in text:
            classes["serialization"] += 1
        elif "json" in text or "collection_limit" in text:
            classes["serialization"] += 1
        elif "commit" in text or "wiring" in text or "offenders" in text or "9." in text:
            classes["commit_contract"] += 1
        else:
            classes["not_observable"] += 1
    comparisons = {
        slug: next(
            (record for record in summary.get("runs") or [] if record.get("slug") == slug),
            None,
        )
        for slug in _COMPARISON_SLUGS
    }
    return {
        "path": str(path),
        "matched_slugs": sorted(by_slug),
        "design_complete": {"numerator": complete, "denominator": len(records)},
        "failure_classes": dict(sorted(classes.items())),
        "run_attribution": not_observable("legacy baseline summary lacks exact run IDs"),
        "comparison_context": {
            slug: (
                {
                    "design_status": record.get("design_status"),
                    "build_rc": record.get("build_rc"),
                }
                if record
                else not_observable("slug absent from frozen baseline")
            )
            for slug, record in comparisons.items()
        },
    }


def _production(projects_dir: Path) -> dict:
    if not projects_dir.is_dir():
        return not_observable(f"production projects directory missing: {projects_dir}")
    states = sorted(
        projects_dir.glob("*/*/.kicraft/state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    selected = states[:25]
    by_id = {}
    for path in states:
        try:
            project_id = f"{path.parents[2].name}/{path.parents[1].name}"
        except IndexError:
            continue
        by_id[project_id] = path
    for witness in _WITNESSES:
        path = by_id.get(witness)
        if path and path not in selected:
            selected.append(path)
    rows = []
    unknown_total = 0
    class_counts = Counter()
    for path in selected:
        project_id = f"{path.parents[2].name}/{path.parents[1].name}"
        try:
            state = _read_json(path)
        except Exception:
            rows.append(
                {"project_id": project_id, "stage": "unknown", "failure_class": "operational"}
            )
            class_counts["operational"] += 1
            continue
        statuses = state.get("stage_status") or {}
        failed = next(
            (
                (stage, row)
                for stage, row in statuses.items()
                if isinstance(row, dict) and row.get("ok") is False
            ),
            (None, {}),
        )
        architecture = state.get("architecture") or {}
        known = {
            str(sheet.get("name"))
            for sheet in architecture.get("sheets") or []
            if isinstance(sheet, dict) and sheet.get("name")
        }
        unknown = [
            _offender_identity(part.get("ref"))
            for part in (state.get("bom") or {}).get("parts") or []
            if isinstance(part, dict) and str(part.get("sheet") or "") not in known
        ]
        unknown_total += len(unknown)
        failure_kind = failed[1].get("failure_kind") if isinstance(failed[1], dict) else None
        classification = (
            "design_complete"
            if statuses
            and all(isinstance(row, dict) and row.get("ok") is True for row in statuses.values())
            else (
                "serialization"
                if failure_kind in {"invalid_json", "truncated_json", "collection_limit"}
                else "commit_contract"
                if failure_kind == "commit_rejected"
                else "operational"
            )
        )
        class_counts[classification] += 1
        rows.append(
            {
                "project_id": project_id,
                "stage": failed[0],
                "failure_class": classification,
                "failure_kind": failure_kind,
                "unknown_sheet_offenders": sorted(unknown),
                "cost_usd": round(
                    sum(
                        float(row.get("cost_usd") or 0.0)
                        for row in statuses.values()
                        if isinstance(row, dict)
                    ),
                    6,
                ),
                "wall_s": round(
                    sum(
                        float(row.get("wall_s") or 0.0)
                        for row in statuses.values()
                        if isinstance(row, dict)
                    ),
                    3,
                ),
            }
        )
    return {
        "selection": "last_25_plus_fixed_witnesses",
        "selected_count": len(selected),
        "window_count": min(25, len(states)),
        "witnesses_requested": list(_WITNESSES),
        "failure_classes": dict(sorted(class_counts.items())),
        "unknown_sheet_references": unknown_total,
        "projects": rows,
    }


def _quantile(values: list[float], fraction: float):
    if not values:
        return not_observable("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _aggregates(runs: list[dict]) -> dict:
    classes = Counter(run["classification"] for run in runs)
    complete = classes.get("design_complete", 0)
    costs = [float(run.get("cost_usd") or 0.0) for run in runs]
    walls = [float(run.get("wall_s") or 0.0) for run in runs]
    failed_cost = sum(
        float(run.get("cost_usd") or 0.0)
        for run in runs
        if run["classification"] != "design_complete"
    )
    return {
        "design_complete": {"numerator": complete, "denominator": len(runs)},
        "failure_classes": dict(sorted(classes.items())),
        "total_cost_usd": round(sum(costs), 6),
        "terminal_failure_cost_usd": round(failed_cost, 6),
        "cost_per_design_complete_usd": (
            round(sum(costs) / complete, 6)
            if complete
            else not_observable("no design-complete canary runs")
        ),
        "cost_median_usd": statistics.median(costs) if costs else not_observable("no runs"),
        "cost_p90_inclusive_usd": _quantile(costs, 0.9),
        "wall_median_s": statistics.median(walls) if walls else not_observable("no runs"),
        "wall_p90_inclusive_s": _quantile(walls, 0.9),
        "quantile_note": f"descriptive inclusive-method values over {len(runs)} single canary runs",
    }


def _stop_gates(runs: list[dict], integrity: dict) -> list[dict]:
    gates = []

    def add(name: str, triggered: bool, numerator: int, denominator: int):
        gates.append(
            {
                "name": name,
                "triggered": triggered,
                "rate": {"numerator": numerator, "denominator": denominator},
            }
        )

    add("campaign_integrity", not integrity.get("valid"), len(integrity.get("errors") or []), 1)
    terminal = [run for run in runs if run["classification"] != "design_complete"]
    add("terminal_design_failure", bool(terminal), len(terminal), len(runs))
    unknown = [
        run
        for run in runs
        if run["bom"]["candidate_unknown_sheet_references"]
        or run["bom"]["committed_unknown_sheet_references"]
    ]
    add("unknown_sheet_reference", bool(unknown), len(unknown), len(runs))
    serialization = [
        run for run in runs if run["classification"] in {"serialization", "schema_contract"}
    ]
    add("terminal_schema_or_serialization", bool(serialization), len(serialization), len(runs))
    commits = [run for run in runs if run["classification"] == "commit_contract"]
    add("terminal_invalid_commit", bool(commits), len(commits), len(runs))
    no_progress = [run for run in runs if run["wiring"]["no_progress"] and run in terminal]
    add("repeated_no_progress_exhaustion", bool(no_progress), len(no_progress), len(runs))
    return gates


def _recommendation(verdict: str, stop_gates: list[dict]) -> str:
    if verdict == "INVALID_CAMPAIGN":
        return "Repair the first campaign-integrity error and resume the same frozen directory only if identity remains valid."
    keyed = {
        "terminal_design_failure": "Inspect the first terminal design stage and its normalized failure signature; do not change model policy during this canary.",
        "unknown_sheet_reference": "Fix the exact BOM sheet-contract path before any model migration discussion.",
        "terminal_schema_or_serialization": "Inspect the failed response-contract or serialization-recovery evidence without raising token or retry caps.",
        "terminal_invalid_commit": "Inspect the first deterministic commit-gate family and offender progression without weakening gates.",
        "repeated_no_progress_exhaustion": "Inspect typed wiring-patch target stability and repeated offender signatures without adding retries.",
    }
    for gate in stop_gates:
        if gate["triggered"] and gate["name"] in keyed:
            return keyed[gate["name"]]
    return "Run a separate three-repeat campaign with the same frozen policy before considering a model migration."


def stage_reliability_metrics(attempt_rows: list[dict], stage_statuses: list[dict]) -> dict:
    """Aggregate availability and semantic outcomes with separate denominators."""
    by_stage: dict[str, dict] = {}
    statuses_by_key = {
        (str(row.get("run_id")), str(row.get("stage"))): row for row in stage_statuses
    }
    keys = set(statuses_by_key) | {
        (str(row.get("run_id")), str(row.get("stage"))) for row in attempt_rows
    }
    for run_id, stage in sorted(keys):
        status = statuses_by_key.get((run_id, stage), {})
        attempts = [
            row
            for row in attempt_rows
            if str(row.get("run_id")) == run_id and str(row.get("stage")) == stage
        ]
        bucket = by_stage.setdefault(
            stage,
            {
                "n": 0,
                "provider_completed": 0,
                "schema_valid_first_pass": 0,
                "commit_first_pass": 0,
                "semantic_clean_first_pass": 0,
                "semantic_clean_after_repair": 0,
                "user_continued": 0,
                "diagnostics": {},
                "not_observable": 0,
            },
        )
        bucket["n"] += 1
        first = min(attempts, key=lambda row: int(row.get("attempt") or 0)) if attempts else None
        if first is None:
            bucket["not_observable"] += 1
        else:
            outcome = str(first.get("outcome") or "")
            if not outcome.startswith(("provider_", "transport_")):
                bucket["provider_completed"] += 1
            if outcome in {"candidate", "commit_rejected"}:
                bucket["schema_valid_first_pass"] += 1
            if outcome == "candidate":
                bucket["commit_first_pass"] += 1
            if outcome == "candidate" and not first.get("diagnostic_codes"):
                bucket["semantic_clean_first_pass"] += 1
        if status.get("semantic_clean") is True:
            bucket["semantic_clean_after_repair"] += 1
        if status.get("ok") is True:
            bucket["user_continued"] += 1
        for diagnostic in status.get("diagnostics") or []:
            code = diagnostic.get("code") if isinstance(diagnostic, dict) else None
            if code:
                bucket["diagnostics"][code] = bucket["diagnostics"].get(code, 0) + 1
    return by_stage


def _render_markdown(report: dict) -> str:
    verdict = report["verdict"]
    aggregate = report["aggregates"]
    lines = [
        "# Verdict",
        "",
        f"**{verdict}** — {aggregate['design_complete']['numerator']}/{aggregate['design_complete']['denominator']} design-complete; cohort spend ${aggregate['total_cost_usd']:.6f}; campaign spend including preflights ${aggregate['campaign_total_cost_usd']:.6f}.",
        "",
        "# Identity",
        "",
        f"- Campaign: `{report['campaign'].get('campaign_id', 'unknown')}`",
        f"- Commit: `{report['campaign'].get('commit', 'unknown')}`",
        f"- Designer: `{report['campaign'].get('designer_model', 'unknown')}`",
        f"- Judge: `{report['campaign'].get('judge_model', 'unknown')}`",
        "",
        "# Canary table",
        "",
        "| Slug | Design | Failed stage | Failure kind | Build | Cost |",
        "|---|---|---|---|---|---:|",
    ]
    for run in report["runs"]:
        lines.append(
            "| {slug} | {design} | {stage} | {kind} | {build} | ${cost:.6f} |".format(
                slug=run["identity"]["slug"],
                design=run["classification"],
                stage=run["failed_stage"] or "—",
                kind=run["failure_kind"] or "—",
                build=run["build_outcome"],
                cost=run["cost_usd"],
            )
        )
    baseline = report["baseline"]
    lines.extend(["", "# Recent failure-class deltas", ""])
    if isinstance(baseline, dict) and "design_complete" in baseline:
        value = baseline["design_complete"]
        lines.append(
            f"Baseline design-complete: {value['numerator']}/{value['denominator']}; canary: {aggregate['design_complete']['numerator']}/{aggregate['design_complete']['denominator']}."
        )
        baseline_classes = baseline.get("failure_classes") or {}
        live_classes = aggregate.get("failure_classes") or {}
        for name in sorted(set(baseline_classes) | set(live_classes)):
            lines.append(
                f"- `{name}`: baseline {int(baseline_classes.get(name, 0))}/"
                f"{value['denominator']}; canary {int(live_classes.get(name, 0))}/"
                f"{aggregate['design_complete']['denominator']}."
            )
    else:
        lines.append("Baseline: not observable.")
    production = report["production"]
    lines.extend(["", "# Production comparison", ""])
    if isinstance(production, dict) and "selected_count" in production:
        lines.append(
            f"Redacted selection: {production['selected_count']} projects; "
            f"unknown-sheet references: {production['unknown_sheet_references']} structural "
            "references across the selected projects."
        )
    else:
        lines.append("Production comparison: not observable.")
    builds = Counter(run["build_outcome"] for run in report["runs"])
    lines.extend(["", "# Build outcomes", ""])
    lines.extend(
        f"- `{name}`: {count}/{len(report['runs'])}" for name, count in sorted(builds.items())
    )
    lines.extend(
        [
            "",
            "# Recommendation",
            "",
            report["recommendation"],
            "",
            "# Reproduction",
            "",
            "```bash",
            f".venv/bin/python -m kicraft.eval.llm_analysis check {report['campaign'].get('batch_dir')}",
            f".venv/bin/python -m kicraft.eval.llm_analysis analyze {report['campaign'].get('batch_dir')} --baseline {report['baseline'].get('path', REFERENCE_BATCH) if isinstance(report['baseline'], dict) else REFERENCE_BATCH}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_batch(
    batch: Path,
    *,
    baseline: Path | None,
    ledger: Path | None,
    projects_dir: Path | None,
) -> int:
    batch = Path(batch).resolve()
    baseline = Path(baseline or REFERENCE_BATCH)
    ledger = Path(ledger or (Path.home() / ".kicraft" / "spend_ledger.db"))
    projects_dir = Path(projects_dir or (Path.home() / ".kicraft" / "projects"))
    spend_rows, stage_rows = _load_ledger(ledger)
    attempt_rows = load_stage_attempts(ledger) if ledger.is_file() else []
    integrity, loaded = _integrity(batch, spend_rows, stage_rows)
    canary = loaded.get("canary") or {}
    immutable = canary.get("immutable") or {}
    summary = loaded.get("summary") or {}
    runs = []
    if integrity["valid"]:
        for record in summary.get("runs") or []:
            run_id = str(record["run_id"])
            runs.append(
                _one_run(
                    batch,
                    record,
                    loaded["spend_by_run"].get(run_id, []),
                    loaded["stage_by_run"].get(run_id, []),
                )
            )
    aggregates = _aggregates(runs)
    reliability_statuses = [
        {"run_id": run.get("run_id"), "stage": stage, **(data or {})}
        for run in runs
        for stage, data in (run.get("stages") or {}).items()
    ]
    reliability = stage_reliability_metrics(attempt_rows, reliability_statuses)
    preflight_cost = round(
        sum(float(row.get("cost_usd") or 0.0) for row in loaded.get("preflight_spend", [])),
        6,
    )
    aggregates["preflight_cost_usd"] = preflight_cost
    aggregates["campaign_total_cost_usd"] = round(
        float(aggregates["total_cost_usd"]) + preflight_cost,
        6,
    )
    stop_gates = _stop_gates(runs, integrity)
    if not integrity["valid"]:
        verdict = "INVALID_CAMPAIGN"
    elif any(gate["triggered"] for gate in stop_gates if gate["name"] != "campaign_integrity"):
        verdict = "FAIL_LLM"
    else:
        findings = any(
            event
            for run in runs
            for stage in run["stages"].values()
            for event in (
                stage.get("attempts", 0) > 1,
                stage.get("call_modes", {}).get("serialization", 0) > 0,
            )
            if event
        ) or any(run["questions"]["blocking_questions"] for run in runs)
        verdict = "PASS_WITH_LLM_FINDINGS" if findings else "PASS"
    report = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "campaign": {
            "campaign_id": immutable.get("campaign_id"),
            "batch_dir": str(batch),
            "commit": (immutable.get("checkout") or {}).get("commit"),
            "designer_model": (immutable.get("designer") or {}).get("model"),
            "judge_model": (immutable.get("judge") or {}).get("model"),
            "analyzed_at": _now(),
            "ledger": str(ledger),
            "projects_dir": str(projects_dir),
        },
        "integrity": integrity,
        "cohort": immutable.get("cohort") or [{"slug": slug} for slug in COHORT],
        "runs": runs,
        "aggregates": aggregates,
        "stage_reliability": reliability,
        "baseline": _baseline(baseline),
        "production": _production(projects_dir),
        "stop_gates": stop_gates,
        "recommendation": _recommendation(verdict, stop_gates),
    }
    _write_json(batch / "llm_analysis.json", report)
    (batch / "llm_analysis.md").write_text(_render_markdown(report), encoding="utf-8")
    print(
        f"{verdict}: {aggregates['design_complete']['numerator']}/{aggregates['design_complete']['denominator']} design-complete; ${aggregates['total_cost_usd']:.6f}",
        flush=True,
    )
    return 2 if verdict == "INVALID_CAMPAIGN" else 1 if verdict == "FAIL_LLM" else 0


def check_batch(batch: Path, *, ledger: Path | None = None) -> int:
    spend_rows, stage_rows = _load_ledger(
        Path(ledger or (Path.home() / ".kicraft" / "spend_ledger.db"))
    )
    integrity, _ = _integrity(Path(batch).resolve(), spend_rows, stage_rows)
    if integrity["valid"]:
        print("VALID_CAMPAIGN")
        return 0
    print("INVALID_CAMPAIGN", file=sys.stderr)
    for error in integrity["errors"]:
        print(f"- {error}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate campaign integrity")
    check.add_argument("batch_dir", type=Path)
    check.add_argument("--ledger", type=Path)
    analyze = sub.add_parser("analyze", help="write deterministic LLM reports")
    analyze.add_argument("batch_dir", type=Path)
    analyze.add_argument("--baseline", type=Path)
    analyze.add_argument("--ledger", type=Path)
    analyze.add_argument("--projects-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        return check_batch(args.batch_dir, ledger=args.ledger)
    return analyze_batch(
        args.batch_dir,
        baseline=args.baseline,
        ledger=args.ledger,
        projects_dir=args.projects_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
