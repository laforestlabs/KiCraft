"""Fail-closed acceptance gates for design-only canaries and full fulfillment.

``verify_campaign`` preserves the historical five-stage, design-only canary.
``verify_fulfillment_campaign`` is intentionally stricter: it accepts only fresh
full campaigns whose mandatory contract obligations are independently evidenced
by generated artifacts.  Neither function treats a reported ``pass`` flag as
proof.
"""
from __future__ import annotations

import argparse
from kicraft.design.part_identity import declares_package, matches_part_identity
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kicraft.server.stage_contracts import DESIGN_STAGES
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

from .acceptance_contracts import (
    APPROVED_5V_DEVICE_CORPUS_VERSION,
    ARTIFACT_EVIDENCE_KINDS,
    ORIGINAL_CORPUS_VERSION,
    RESULT_STATUSES,
    contract_for,
    reference_obligation_ids,
    validate_contracts,
)

_REFERENCE_EVIDENCE_PATH = Path(__file__).with_name("acceptance_reference_evidence.json")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _corpus() -> dict[str, dict[str, Any]]:
    return {
        entry["slug"]: {"index": index, "slug": entry["slug"], "brief_hash": _stable_hash(entry["brief"])}
        for index, entry in enumerate(BENCHMARK_PROMPTS, 1)
    }


def _selected_corpus(slugs: list[str] | None) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:
    corpus = _corpus()
    selected = slugs if slugs is not None else list(corpus)
    errors: list[str] = []
    if not selected or len(set(selected)) != len(selected) or set(selected) - corpus.keys():
        errors.append("invalid or duplicate expected corpus selection")
    if slugs is None and len(selected) != 34:
        errors.append("production corpus is not exactly 34 briefs")
    return selected, corpus, errors


def _read_campaign(root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    try:
        return (
            json.loads((root / "summary.json").read_text(encoding="utf-8")),
            json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8")),
            [],
        )
    except (OSError, ValueError) as exc:
        return None, None, [f"campaign evidence unreadable: {exc}"]


def _campaign_provenance(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected: list[str],
    corpus: Mapping[str, Mapping[str, Any]],
    *,
    full: bool,
    contract_version: str = ORIGINAL_CORPUS_VERSION,
) -> list[str]:
    errors: list[str] = []
    if not summary.get("finished_at") or "wall_s" not in summary:
        errors.append("campaign has not finished")
    if full:
        if summary.get("design_only") is not False or summary.get("judge") is not True:
            errors.append("fulfillment acceptance requires full execution with judge enabled")
        if summary.get("execution_mode") != "full":
            errors.append("campaign execution mode is not explicitly full")
        lifecycle = summary.get("lifecycle")
        required_lifecycle = ("post_wiring_review", "silkscreen", "judge")
        if not isinstance(lifecycle, Mapping) or any(
            not isinstance(lifecycle.get(name), Mapping)
            or lifecycle[name].get("status") not in {"completed", "pass"}
            or not isinstance(lifecycle[name].get("cost_usd"), (int, float))
            or isinstance(lifecycle[name].get("cost_usd"), bool)
            or not math.isfinite(lifecycle[name]["cost_usd"])
            or lifecycle[name]["cost_usd"] < 0
            for name in required_lifecycle
        ):
            errors.append("full lifecycle evidence or lifecycle cost accounting is incomplete")
    elif summary.get("design_only") is not True or summary.get("judge") is not False:
        errors.append("campaign must be design-only with judge disabled")
    if summary.get("resumed") is not False or summary.get("resumed_reused_n", 0):
        errors.append("campaign freshness is not proven")
    if summary.get("source_unchanged") is not True:
        errors.append("campaign source stability is not proven")
    if summary.get("fresh_output_directory") is not True:
        errors.append("campaign output directory was not fresh")
    if len(selected) == 34 and (summary.get("requested_only") or summary.get("requested_limit") is not None):
        errors.append("production acceptance forbids corpus selection flags")
    immutable = manifest.get("immutable") or {}
    if not immutable.get("source_fingerprint"):
        errors.append("campaign has no source fingerprint")
    expected = [corpus[slug] for slug in corpus if slug in selected]
    if immutable.get("llm_mode") != "live":
        errors.append("live provider provenance is not proven")
    if immutable.get("corpus") != expected or immutable.get("corpus_hash") != _stable_hash(expected):
        errors.append("manifest corpus/brief identities do not match requested corpus")
    if full:
        execution = [
            {
                "index": corpus[slug]["index"],
                "slug": slug,
                "original_brief_hash": corpus[slug]["brief_hash"],
                "execution_brief": contract_for(slug, contract_version)["execution_brief"],
                "execution_brief_hash": _stable_hash(contract_for(slug, contract_version)["execution_brief"]),
                "consent": contract_for(slug, contract_version)["consent"],
            }
            for slug in corpus if slug in selected
        ]
        if (
            immutable.get("contract_version") != contract_version
            or immutable.get("execution_corpus") != execution
            or immutable.get("execution_corpus_hash") != _stable_hash(execution)
        ):
            errors.append("manifest execution corpus or contract version is not proven")
    if immutable.get("repeats") != 1:
        errors.append("acceptance requires one fresh run per brief")
    cap = (immutable.get("caps") or {}).get("project_usd")
    if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
        errors.append("finite positive project budget is not recorded")
    return errors


def _run_dir(root: Path, run: Mapping[str, Any]) -> Path | None:
    rundir = Path(run.get("rundir") or "")
    if not rundir.is_absolute():
        rundir = root / rundir
    try:
        resolved = rundir.resolve()
        if not resolved.is_relative_to(root):
            return None
        return resolved
    except OSError:
        return None


def _stage_evidence(rundir: Path) -> str | None:
    try:
        state = json.loads((rundir / ".kicraft/state.json").read_text(encoding="utf-8"))
        done: list[dict[str, Any]] = []
        with (rundir / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("kind") == "stage_done":
                    done.append(event)
        statuses = state.get("stage_status") or {}
        if any((statuses.get(stage) or {}).get("ok") is not True for stage in DESIGN_STAGES):
            return "state does not commit all five stages"
        successful = [event for event in done if event.get("ok") is True]
        latest = {event.get("stage"): event for event in done}
        if (
            [event.get("stage") for event in successful] != list(DESIGN_STAGES)
            or any((latest.get(stage) or {}).get("ok") is not True for stage in DESIGN_STAGES)
            or any(event.get("failure_kind") or event.get("reused_work_units") for event in successful)
        ):
            return "events do not prove five fresh successful stages in order"
    except (OSError, ValueError, TypeError) as exc:
        return f"run evidence unreadable: {exc}"
    return None


def verify_campaign(root: Path, slugs: list[str] | None = None) -> list[str]:
    """Verify the existing design-only canary; it is not delivery acceptance."""
    root = root.resolve()
    selected, corpus, errors = _selected_corpus(slugs)
    if errors:
        return errors
    summary, manifest, read_errors = _read_campaign(root)
    if read_errors:
        return read_errors
    assert summary is not None and manifest is not None
    errors.extend(_campaign_provenance(summary, manifest, selected, corpus, full=False))
    runs = summary.get("runs") or []
    if Counter(run.get("slug") for run in runs) != Counter(selected):
        errors.append("run corpus has omitted, duplicated, or substituted briefs")
    if summary.get("n") != len(selected) or summary.get("design_committed") != len(selected):
        errors.append("summary does not report every expected design committed")
    cap = ((manifest.get("immutable") or {}).get("caps") or {}).get("project_usd")
    for run in runs:
        slug = run.get("slug")
        if slug not in corpus:
            continue
        prefix = f"{slug}: "
        if run.get("index") != corpus[slug]["index"] or _stable_hash(run.get("prompt")) != corpus[slug]["brief_hash"]:
            errors.append(prefix + "run brief identity differs from corpus")
        if run.get("design_committed") is not True or any(run.get(key) for key in ("error", "design_error", "failure_kind", "design_failure_kind")):
            errors.append(prefix + "terminal design failure")
        cost = run.get("design_cost_usd")
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            errors.append(prefix + "paid design cost is not recorded")
        elif isinstance(cap, (int, float)) and cost > cap + 0.000001:
            errors.append(prefix + "design spend exceeds configured project cap")
        rundir = _run_dir(root, run)
        if rundir is None:
            errors.append(prefix + "run evidence is outside fresh campaign")
            continue
        stage_error = _stage_evidence(rundir)
        if stage_error:
            errors.append(prefix + stage_error)
    return errors


def _evidence_error(evidence: Any, artifact_root: Path | None) -> str | None:
    if not isinstance(evidence, list) or not evidence:
        return "has no artifact evidence"
    for item in evidence:
        if not isinstance(item, Mapping) or item.get("kind") not in ARTIFACT_EVIDENCE_KINDS:
            return "contains invalid evidence kind"
        path = item.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            return "contains evidence without a relative artifact path"
        if artifact_root is not None:
            candidate = (artifact_root / path).resolve()
            if not candidate.is_relative_to(artifact_root) or not candidate.exists():
                return f"references missing or escaping artifact {path!r}"
    return None


def _count(facts: Mapping[str, Any], key: str, name: str) -> int | None:
    values = facts.get(key)
    if not isinstance(values, Mapping):
        return None
    value = values.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _check_facts(check: Mapping[str, Any], facts: Mapping[str, Any]) -> str | None:
    kind = check.get("kind")
    if kind == "part_class_count":
        actual = _count(facts, "part_classes", str(check["part_class"]))
        if actual is None or actual < check["minimum"]:
            return f"requires at least {check['minimum']} {check['part_class']} parts"
    elif kind == "part_class_exact":
        actual = _count(facts, "part_classes", str(check["part_class"]))
        if actual is None or actual != check["value"]:
            return f"requires exactly {check['value']} {check['part_class']} parts"
    elif kind == "part_classes":
        required = check.get("required") or {}
        if not isinstance(required, Mapping) or any((_count(facts, "part_classes", name) or 0) < minimum for name, minimum in required.items()):
            return "does not prove required part classes and quantities"
    elif kind in {"part_identity", "part_identities"}:
        identities = facts.get("part_identities")
        if not isinstance(identities, list):
            return "does not provide part identities"
        actual = {item.get("identity") if isinstance(item, Mapping) else item for item in identities}
        required = set(check.get("identities") or [check.get("identity")])
        if any(
            not isinstance(requested, str)
            or not any(isinstance(candidate, str) and matches_part_identity(requested, candidate) for candidate in actual)
            for requested in required
        ):
            return f"does not prove required identities {sorted(required)}"
        package = check.get("package")
        if package and not any(
            isinstance(item, Mapping)
            and isinstance(item.get("identity"), str)
            and matches_part_identity(str(check.get("identity")), item["identity"])
            and any(declares_package(package, str(item.get(field) or "")) for field in ("package", "footprint"))
            for item in identities
        ):
            return f"does not prove required package {package}"
    elif kind == "part_inventory":
        inventory = facts.get("part_inventory")
        if not isinstance(inventory, list) or not inventory or any(not isinstance(part, Mapping) or not all(part.get(field) for field in ("mpn", "manufacturer", "source_url", "package", "rated_limits")) for part in inventory):
            return "does not prove a sourceable rated part inventory"
    elif kind == "pin_mapping":
        mapping = facts.get("pin_mapping")
        if not isinstance(mapping, list) or not mapping or any(not isinstance(item, Mapping) or not item.get("symbol_pin") or not item.get("footprint_pad") for item in mapping):
            return "does not prove symbol-to-footprint pin mappings"
    elif kind == "net_paths":
        paths = facts.get("net_paths")
        if not isinstance(paths, Mapping):
            return "does not provide net-path facts"
        required = check.get("paths")
        if required:
            if any(paths.get(path) is not True for path in required):
                return f"does not prove required net paths {required}"
        elif not any(value is True for value in paths.values()):
            return "does not prove any required net path"
    elif kind == "connector_map":
        maps = facts.get("connector_maps")
        connector = check.get("connector")
        mapping = maps.get(connector) if isinstance(maps, Mapping) else None
        required = check.get("required") or []
        if not isinstance(mapping, Mapping) or any(name not in mapping for name in required):
            return f"does not prove {connector} connector mapping"
        bindings = [mapping[name] for name in required]
        if any(
            not isinstance(binding, Mapping)
            or not all(isinstance(binding.get(field), str) and binding[field] for field in ("source_pad", "connector_pad", "net"))
            for binding in bindings
        ):
            return f"does not prove actual source/connector pad and net evidence for {connector}"
        if check.get("distinct") and len({binding["connector_pad"] for binding in bindings}) != len(bindings):
            return f"does not prove distinct {connector} connector pads"
        if check.get("source_distinct") and len({binding["source_pad"] for binding in bindings}) != len(bindings):
            return f"does not prove distinct {connector} source pads"
        board_net_members = facts.get("board_net_members")
        if not isinstance(board_net_members, Mapping) or any(
            not isinstance(board_net_members.get(binding["net"]), list)
            or binding["source_pad"] not in board_net_members[binding["net"]]
            or binding["connector_pad"] not in board_net_members[binding["net"]]
            for binding in bindings
        ):
            return f"does not prove {connector} bindings on the delivered board net graph"
    elif kind == "numeric_range":
        fact = check.get("fact")
        value = facts.get(fact)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return f"does not provide finite {fact}"
        if "minimum" in check and value < check["minimum"]:
            return f"{fact} is below required minimum"
        if "maximum" in check and value > check["maximum"]:
            return f"{fact} is above required maximum"
        if "minimum" not in check and "maximum" not in check:
            # An unbounded value proves nothing on its own: the design must
            # declare the range it claims and the value must fall inside it.
            declared = facts.get("declared_ranges")
            bounds = declared.get(fact) if isinstance(declared, Mapping) else None
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or any(not isinstance(bound, (int, float)) or isinstance(bound, bool) or not math.isfinite(bound) for bound in bounds)
                or bounds[0] >= bounds[1]
            ):
                return f"has no declared range for {fact}"
            if not bounds[0] <= value <= bounds[1]:
                return f"{fact} is outside its declared range"
    elif kind == "set_members":
        values = facts.get(check.get("fact"))
        if not isinstance(values, list) or not set(check.get("required") or []) <= set(values):
            return f"does not prove required {check.get('fact')} members"
    elif kind == "channel_count":
        value = _count(facts, "channel_counts", str(check["channel"]))
        if value is None or value < check["minimum"]:
            return f"does not prove {check['minimum']} {check['channel']} channels"
    elif kind == "geometry_count":
        value = _count(facts, "geometry_counts", str(check["feature"]))
        if value is None or value < check["minimum"]:
            return f"does not prove {check['minimum']} {check['feature']} geometry features"
    elif kind == "outline":
        outline = facts.get("outline")
        if not isinstance(outline, Mapping) or outline.get("shape") != check.get("shape"):
            return f"does not prove {check.get('shape')} outline"
        if "diameter_mm" in check and outline.get("diameter_mm") != check["diameter_mm"]:
            return f"does not prove {check['diameter_mm']} mm diameter"
    elif kind == "gate":
        gates = facts.get("gates")
        gate = check.get("gate")
        if not isinstance(gates, Mapping) or gates.get(gate) != "pass":
            return f"does not prove {gate} gate"
    elif kind == "applicable_gate":
        applicable = facts.get("applicable_gates")
        gate = check.get("gate")
        if not isinstance(applicable, Mapping) or not isinstance(applicable.get(gate), bool):
            return f"does not state whether {gate} gate applies"
        if applicable[gate]:
            gates = facts.get("gates")
            if not isinstance(gates, Mapping) or gates.get(gate) != "pass":
                return f"does not prove applicable {gate} gate"
    elif kind == "artifacts":
        artifacts = facts.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return "does not identify exported fabrication artifacts"
    else:
        return f"unknown evaluator {kind!r}"
    return None


def evaluate_obligation(
    obligation: Mapping[str, Any],
    result: Any,
    *,
    artifact_root: Path | None = None,
    facts_override: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a failure reason; a claimed pass never bypasses evidence or facts."""
    if not isinstance(result, Mapping) or result.get("status") not in RESULT_STATUSES:
        return "has invalid result status"
    if result.get("status") != "pass":
        return f"is {result.get('status', 'unverified')}"
    evidence_error = _evidence_error(result.get("evidence"), artifact_root)
    if evidence_error:
        return evidence_error
    facts = result.get("facts", facts_override)
    if not isinstance(facts, Mapping):
        return "has no machine-checkable facts"
    return _check_facts(obligation.get("check") or {}, facts)
def verify_reference_fixture(payload: Mapping[str, Any], *, artifact_root: Path | None = None) -> list[str]:
    """Validate a no-LLM reference fixture against its declared contract.

    A reference proves the reviewed compiler boundary: functional, quantitative,
    and physical obligations that reviewed BOM/wiring/recipe inputs can settle.
    Board-only obligations (pin mapping, connector maps, outline/geometry, build
    gates, fabrication exports) are not proven here; the fixture MUST name them
    in ``deferred_obligations`` so a deferred gate is never mistaken for a pass.
    """
    errors: list[str] = []
    version = payload.get("contract_version")
    slug = payload.get("slug")
    if payload.get("schema_version") != 1 or not isinstance(version, str) or not isinstance(slug, str):
        return ["reference fixture lacks schema/version/slug"]
    try:
        contract = contract_for(slug, version)
    except ValueError as exc:
        return [str(exc)]
    if payload.get("brief_hash") != _stable_hash(contract["original_brief"]):
        errors.append("reference fixture brief identity differs from original contract")
    if not isinstance(payload.get("compiler_boundary"), str) or not payload["compiler_boundary"]:
        errors.append("reference fixture has no compiler boundary")
    results = payload.get("obligations")
    if not isinstance(results, Mapping):
        return [*errors, "reference fixture has no obligation results"]
    reference_ids, deferred_ids = reference_obligation_ids(slug, version)
    declared_deferred = payload.get("deferred_obligations")
    if not isinstance(declared_deferred, list) or Counter(declared_deferred) != Counter(deferred_ids):
        errors.append("reference fixture defers exactly " + ", ".join(sorted(deferred_ids)))
    if Counter(list(results)) != Counter(reference_ids):
        errors.append("reference fixture obligations differ from the contract's reference obligations")
    for obligation_id in reference_ids:
        obligation = next(item for item in contract["obligations"] if item["id"] == obligation_id)
        reason = evaluate_obligation(obligation, results.get(obligation_id), artifact_root=artifact_root)
        if reason:
            errors.append(f"{obligation_id}: {reason}")
    return errors


def build_reference_fixture(
    slug: str,
    obligations: Mapping[str, Mapping[str, Any]],
    *,
    contract_version: str = ORIGINAL_CORPUS_VERSION,
    compiler_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one reviewed reference row and refuse to emit a row that fails.

    ``obligations`` maps the contract's reference obligation ids to their
    evidence-bearing results.  The brief identity, compiler boundary, and the
    exact deferred-obligation list are derived from the contract so a row can
    never silently drop or invent a mandatory obligation.
    """
    contract = contract_for(slug, contract_version)
    reference_ids, deferred_ids = reference_obligation_ids(slug, contract_version)
    acceptance = {
        "schema_version": 1,
        "contract_version": contract_version,
        "slug": slug,
        "brief_hash": _stable_hash(contract["original_brief"]),
        "compiler_boundary": "architecture -> recipe resolution -> deterministic BOM/wiring work units -> stage commit",
        "deferred_obligations": list(deferred_ids),
        "obligations": {obligation_id: dict(obligations[obligation_id]) for obligation_id in reference_ids if obligation_id in obligations},
    }
    row: dict[str, Any] = {"acceptance": acceptance}
    if compiler_input is not None:
        row["compiler_input"] = dict(compiler_input)
    errors = verify_reference_fixture(acceptance)
    if errors:
        raise ValueError(f"{slug}: reference row is incomplete: " + "; ".join(errors))
    return row


def verify_reference_rows(directory: Path) -> list[str]:
    """Validate a directory of single-row reference JSON files."""
    errors: list[str] = []
    paths = sorted(directory.glob("*.json"))
    if not paths:
        return [f"{directory}: no reference rows"]
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: unreadable ({exc})")
            continue
        payload = row.get("acceptance") if isinstance(row, Mapping) else None
        if not isinstance(payload, Mapping):
            errors.append(f"{path.name}: not a reference row")
            continue
        errors.extend(f"{path.name}: {error}" for error in verify_reference_fixture(payload))
    return errors


def verify_reference_fixtures(root: Path | None = None) -> list[str]:
    """Validate every in-repo reference fixture against its declared contract."""
    directory = root or Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "reference_inputs"
    errors: list[str] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8")).get("references")
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: unreadable ({exc})")
            continue
        if not isinstance(rows, list) or not rows:
            errors.append(f"{path.name}: no reference rows")
            continue
        for row in rows:
            payload = row.get("acceptance") if isinstance(row, Mapping) else None
            if not isinstance(payload, Mapping):
                errors.append(f"{path.name}: reference row without an acceptance record")
                continue
            slug = payload.get("slug")
            if slug in seen:
                errors.append(f"{path.name}: duplicate reference row for {slug} (also in {seen[slug]})")
            elif isinstance(slug, str):
                seen[slug] = path.name
            errors.extend(f"{slug}: {error}" for error in verify_reference_fixture(payload))
            errors.extend(f"{slug}: {error}" for error in reference_row_shape_errors(row))
    missing = sorted(set(_corpus()) - set(seen))
    if missing:
        errors.append("missing reference rows: " + ", ".join(missing))
    return errors



_REFERENCE_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "reference_inputs"


def load_reference_rows(root: Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Every in-repo reference row as ``(fixture name, row)``, ordered by fixture."""
    directory = root or _REFERENCE_FIXTURES
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend((path.name, row) for row in (data.get("references") or []))
    return rows


def reference_row_shape_errors(row: Mapping[str, Any]) -> list[str]:
    """Whether a row ships every stage payload the reviewed boundary needs.

    A row that validates but stores no BOM/wiring candidate cannot reproduce its
    boundary at all; that is a row defect, not a boundary result.
    """
    compiler_input = row.get("compiler_input")
    if not isinstance(compiler_input, Mapping):
        return ["row ships no compiler_input"]
    missing = [
        key
        for key in ("intent", "functional_spec", "bom_candidate", "wiring_candidate")
        if not isinstance(compiler_input.get(key), Mapping)
    ]
    if not any(
        isinstance(compiler_input.get(key), Mapping)
        for key in ("architecture", "architecture_intent")
    ):
        missing.append("architecture")
    return [f"compiler_input lacks {missing}"] if missing else []


def replay_reference_row(row: Mapping[str, Any], *, fixture: str = "") -> dict[str, Any]:
    """Drive one row's own stored inputs through the real five-stage chain.

    No provider call is made and no spend is recorded: the row's payloads are
    replayed as the model's answers (`kicraft.loadtest.mockllm`), so the row's
    recorded boundary is re-derived by the real compiler, work-unit, wiring and
    stage-commit path from the inputs it ships.  A row that claims a boundary its
    own inputs cannot reach is therefore refused here (`--reference-replay`).
    """
    from kicraft.loadtest import mockllm
    from kicraft.server.stage_driver import DESIGN_STAGES, drive_chain

    acceptance = row.get("acceptance") or {}
    slug = str(acceptance.get("slug"))
    fixture_errors = verify_reference_fixture(acceptance)
    if fixture_errors:
        return {
            "slug": slug,
            "fixture": fixture,
            "replay": "blocked",
            "blocker": "; ".join(fixture_errors),
        }
    shape_errors = reference_row_shape_errors(row)
    if shape_errors:
        return {"slug": slug, "fixture": fixture, "replay": "row_shape_invalid", "error": shape_errors[0]}
    compiler_input = row["compiler_input"]
    intent = compiler_input.get("intent") or {}
    brief = str(intent.get("goal") or contract_for(slug)["original_brief"])
    transcript = mockllm.transcript_from_reference_row({"references": [dict(row)]}, slug)
    previous_mode = os.environ.get("KICRAFT_LLM_MODE")
    previous_transcript = os.environ.get("KICRAFT_MOCK_TRANSCRIPT")
    try:
        with tempfile.TemporaryDirectory(prefix=f"ref_replay_{slug}_") as tmp:
            slot = Path(tmp) / "transcript.json"
            slot.write_text(json.dumps(transcript), encoding="utf-8")
            os.environ["KICRAFT_LLM_MODE"] = "replay"
            os.environ["KICRAFT_MOCK_TRANSCRIPT"] = str(slot)
            results, _guard, _state = drive_chain(
                list(DESIGN_STAGES), brief, Path(tmp) / "workspace"
            )
    finally:
        for name, value in (
            ("KICRAFT_LLM_MODE", previous_mode),
            ("KICRAFT_MOCK_TRANSCRIPT", previous_transcript),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    failed = [item for item in results if not item.get("commit_ok")]
    entry: dict[str, Any] = {
        "slug": slug,
        "fixture": fixture,
        "stages": {str(item["stage"]): bool(item.get("commit_ok")) for item in results},
        "replay": "committed" if not failed else "refused",
    }
    if failed:
        entry["failed_stage"] = str(failed[0]["stage"])
        entry["first_error"] = str(failed[0].get("error"))[:400]
    return entry


def verify_reference_replay(
    slugs: list[str] | None = None,
    *,
    fixtures_root: Path | None = None,
    report: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Replay every reviewed reference row and require its own boundary.

    A row that ships a complete, validating contract and produces no refusal has
    to commit all five stages from the inputs it stores.  Two recorded cases are
    not failures: a row whose fixture does not validate is a named block (owned by
    ``--references``), and a brief whose contract records a
    ``specification_conflict`` must *refuse* -- a commit there would be an unsafe
    acceptance, not progress.
    """
    selected = set(slugs) if slugs is not None else None
    errors: list[str] = []
    for fixture, row in load_reference_rows(fixtures_root):
        slug = str((row.get("acceptance") or {}).get("slug"))
        if selected is not None and slug not in selected:
            continue
        entry = replay_reference_row(row, fixture=fixture)
        if report is not None:
            report.append(entry)
        conflicts = False
        if entry["replay"] != "blocked":
            try:
                conflicts = (
                    contract_for(slug)["feasibility"]["status"] == "specification_conflict"
                )
            except ValueError:
                conflicts = False
        if entry["replay"] == "blocked":
            continue
        if entry["replay"] == "committed" and conflicts:
            errors.append(
                f"{slug}: committed despite its recorded specification conflict, "
                "which the contract requires it to refuse"
            )
        elif entry["replay"] == "refused" and conflicts:
            continue
        elif entry["replay"] != "committed":
            errors.append(
                f"{slug}: reference row does not reproduce its boundary "
                f"({entry['replay']}"
                + (f" at {entry.get('failed_stage')}" if entry.get("failed_stage") else "")
                + f"): {entry.get('error') or entry.get('first_error') or ''}"
            )
    if selected is not None:
        missing = sorted(selected - {str((row.get("acceptance") or {}).get("slug")) for _name, row in load_reference_rows(fixtures_root)})
        if missing:
            errors.append("no reference row for: " + ", ".join(missing))
    return errors


def load_reference_evidence() -> dict[str, Any]:
    """Load the frozen-witness/source index; it contains no positive fulfillment."""
    return json.loads(_REFERENCE_EVIDENCE_PATH.read_text(encoding="utf-8"))


def new_acceptance_evidence(slug: str, *, contract_version: str = ORIGINAL_CORPUS_VERSION) -> dict[str, Any]:
    """Create an explicitly unverified evidence record for an evaluator to fill.

    The template cannot pass fulfillment until an independent producer supplies
    real artifact paths, extracted facts, and all required evidence.
    """
    contract = contract_for(slug, contract_version)
    return {
        "schema_version": 1,
        "contract_version": contract_version,
        "slug": slug,
        "brief_hash": _stable_hash(contract["original_brief"]),
        "execution_brief_hash": _stable_hash(contract["execution_brief"]),
        "generation": {"status": "unverified"},
        "fabrication": {"status": "unverified", "artifact_paths": []},
        "software_fulfillment": {"status": "unverified"},
        "physical_validation": {"status": "unverified"},
        "common_gates": {},
        "obligations": {
            obligation["id"]: {"status": "unverified", "evidence": [], "facts": {}}
            for obligation in contract["obligations"]
        },
    }


def write_acceptance_evidence(rundir: Path, evidence: Mapping[str, Any]) -> Path:
    """Persist evaluator-produced evidence without promoting it to fulfillment."""
    if evidence.get("schema_version") != 1 or not isinstance(evidence.get("slug"), str):
        raise ValueError("acceptance evidence lacks schema version or slug")
    contract = contract_for(evidence["slug"], str(evidence.get("contract_version")))
    if evidence.get("brief_hash") != _stable_hash(contract["original_brief"]):
        raise ValueError("acceptance evidence brief identity differs from contract")
    if evidence.get("execution_brief_hash") != _stable_hash(contract["execution_brief"]):
        raise ValueError("acceptance evidence execution brief differs from contract")
    path = rundir / "eval" / "acceptance_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact_paths(value: Any, rundir: Path) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(path, str) and path and not Path(path).is_absolute() and (rundir / path).resolve().is_relative_to(rundir) and (rundir / path).exists()
        for path in value
    )


def _fulfillment_run_errors(run: Mapping[str, Any], rundir: Path, version: str) -> list[str]:
    slug = run.get("slug")
    prefix = f"{slug}: "
    errors: list[str] = []
    try:
        contract = contract_for(str(slug), version)
    except ValueError as exc:
        return [prefix + str(exc)]
    if contract["feasibility"]["status"] != "reviewed_feasible":
        errors.append(prefix + f"contract feasibility is {contract['feasibility']['status']}")
    if contract["sourceability"]["status"] != "reviewed_sourceable":
        errors.append(prefix + f"contract sourceability is {contract['sourceability']['status']}")
    if run.get("contract_version") != version:
        errors.append(prefix + "run contract version differs from fulfillment contract")
    if run.get("execution_brief_hash") != _stable_hash(contract["execution_brief"]):
        errors.append(prefix + "run execution brief differs from fulfillment contract")
    evidence_path = rundir / "eval" / "acceptance_evidence.json"
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [prefix + f"acceptance artifact evidence unreadable: {exc}"]
    if run.get("contract_consent") != contract["consent"]:
        errors.append(prefix + "run consent record differs from fulfillment contract")
    if evidence.get("schema_version") != 1 or evidence.get("contract_version") != version:
        errors.append(prefix + "acceptance evidence contract version is missing or wrong")
    if evidence.get("slug") != slug or evidence.get("brief_hash") != _stable_hash(contract["original_brief"]):
        errors.append(prefix + "acceptance evidence brief identity differs from contract")
    if evidence.get("execution_brief_hash") != _stable_hash(contract["execution_brief"]):
        errors.append(prefix + "acceptance evidence execution brief differs from contract")
    for phase in ("generation", "fabrication", "software_fulfillment"):
        phase_record = evidence.get(phase)
        if not isinstance(phase_record, Mapping) or phase_record.get("status") != "pass":
            errors.append(prefix + f"{phase} is not positively verified")
    physical = evidence.get("physical_validation")
    if not isinstance(physical, Mapping) or physical.get("status") not in RESULT_STATUSES:
        errors.append(prefix + "physical validation status is absent or invalid")
    elif physical.get("status") == "pass":
        physical_evidence_error = _evidence_error(physical.get("evidence"), rundir)
        if physical_evidence_error:
            errors.append(prefix + f"physical validation {physical_evidence_error}")
    fabrication = evidence.get("fabrication")
    if not isinstance(fabrication, Mapping) or not _artifact_paths(fabrication.get("artifact_paths"), rundir):
        errors.append(prefix + "fabrication has no delivered artifact evidence")
    common = evidence.get("common_gates")
    if not isinstance(common, Mapping) or any(common.get(gate) != "pass" for gate in ("erc", "drc")):
        errors.append(prefix + "applicable ERC/DRC gates are not positively evidenced")
    results = evidence.get("obligations")
    if not isinstance(results, Mapping):
        return [*errors, prefix + "mandatory obligation evidence is absent"]
    for obligation in contract["obligations"]:
        reason = evaluate_obligation(
            obligation,
            results.get(obligation["id"]),
            artifact_root=rundir,
            facts_override=evidence.get("artifact_facts"),
        )
        if reason:
            errors.append(prefix + f"{obligation['id']}: {reason}")
    return errors


def verify_fulfillment_campaign(root: Path, slugs: list[str] | None = None, *, contract_version: str = ORIGINAL_CORPUS_VERSION) -> list[str]:
    """Fail closed unless every selected mandatory obligation has artifact proof."""
    root = root.resolve()
    selected, corpus, errors = _selected_corpus(slugs)
    if errors:
        return errors
    if contract_version not in {ORIGINAL_CORPUS_VERSION, APPROVED_5V_DEVICE_CORPUS_VERSION}:
        return [f"unknown acceptance contract version: {contract_version}"]
    errors.extend(validate_contracts())
    summary, manifest, read_errors = _read_campaign(root)
    if read_errors:
        return [*errors, *read_errors]
    assert summary is not None and manifest is not None
    errors.extend(_campaign_provenance(summary, manifest, selected, corpus, full=True, contract_version=contract_version))
    runs = summary.get("runs") or []
    if Counter(run.get("slug") for run in runs) != Counter(selected):
        errors.append("run corpus has omitted, duplicated, or substituted briefs")
    if summary.get("n") != len(selected) or summary.get("design_committed") != len(selected):
        errors.append("summary does not report every expected design committed")
    cap = ((manifest.get("immutable") or {}).get("caps") or {}).get("project_usd")
    for run in runs:
        slug = run.get("slug")
        if slug not in corpus:
            continue
        prefix = f"{slug}: "
        if run.get("index") != corpus[slug]["index"] or _stable_hash(run.get("prompt")) != corpus[slug]["brief_hash"]:
            errors.append(prefix + "run brief identity differs from corpus")
        if run.get("design_committed") is not True or any(run.get(key) for key in ("error", "design_error", "failure_kind", "design_failure_kind")):
            errors.append(prefix + "terminal design failure")
        cost = run.get("design_cost_usd")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
            errors.append(prefix + "paid design cost is not recorded")
        elif isinstance(cap, (int, float)) and cost > cap + 0.000001:
            errors.append(prefix + "design spend exceeds configured project cap")
        rundir = _run_dir(root, run)
        if rundir is None:
            errors.append(prefix + "run evidence is outside fresh campaign")
            continue
        stage_error = _stage_evidence(rundir)
        if stage_error:
            errors.append(prefix + stage_error)
        errors.extend(_fulfillment_run_errors(run, rundir, contract_version))
    return errors


def verify_release_campaigns(roots: list[Path], *, contract_version: str = ORIGINAL_CORPUS_VERSION) -> list[str]:
    """Require 102 independently fresh fulfilled runs: all 34 in three campaigns."""
    if len(roots) != 3 or len({root.resolve() for root in roots}) != 3:
        return ["release acceptance requires exactly three distinct fresh campaigns"]
    errors: list[str] = []
    fingerprints: set[str] = set()
    for index, root in enumerate(roots, 1):
        errors.extend(f"campaign {index}: {error}" for error in verify_fulfillment_campaign(root, contract_version=contract_version))
        # The verifier above already validates the manifest.  This read only
        # establishes that all three completed under an identical source tree.
        try:
            immutable = json.loads((root.resolve() / "campaign_manifest.json").read_text(encoding="utf-8")).get("immutable") or {}
            fingerprint = immutable.get("source_fingerprint")
            if isinstance(fingerprint, str):
                fingerprints.add(fingerprint)
        except (OSError, ValueError):
            pass
    if len(fingerprints) != 1:
        errors.append("release campaigns do not share one declared source fingerprint")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="*", help="one canary/full campaign, or three with --release")
    parser.add_argument("--only", help="diagnostic subset, never full production acceptance")
    parser.add_argument("--mode", choices=("design-only", "full"), default="design-only")
    parser.add_argument("--contract-version", default=ORIGINAL_CORPUS_VERSION)
    parser.add_argument("--release", action="store_true", help="require three fresh complete full campaigns")
    parser.add_argument("--references", action="store_true", help="validate the in-repo reviewed reference fixtures")
    parser.add_argument("--reference-rows", type=Path, help="validate single-row reference JSON files in a directory")
    parser.add_argument(
        "--reference-replay",
        action="store_true",
        help="replay every reviewed reference row through the real five-stage chain (no provider calls)",
    )
    args = parser.parse_args(argv)
    slugs = args.only.split(",") if args.only else None
    if args.reference_replay:
        if args.references or args.campaign or args.release or args.reference_rows is not None:
            parser.error("--reference-replay takes no campaign, --references, --reference-rows, or --release")
        report: list[dict[str, Any]] = []
        errors = verify_reference_replay(slugs, report=report)
        for entry in report:
            detail = entry.get("blocker") or entry.get("first_error") or entry.get("error") or ""
            print(f"{entry['slug']:24s} {entry['replay']:<16} {detail}"[:200])
        committed = sum(1 for entry in report if entry["replay"] == "committed")
        blocked = sum(1 for entry in report if entry["replay"] == "blocked")
        label = (
            f"{committed}/{len(report)} reviewed reference rows reproduce their own boundary"
            + (f" ({blocked} recorded block(s) reported)" if blocked else "")
        )
        if errors:
            print("design acceptance FAILED:\n" + "\n".join(f"- {error}" for error in errors))
            return 1
        print(f"design acceptance passed: {label}")
        return 0
    if args.reference_rows is not None:
        if args.references or args.campaign or args.release or slugs is not None:
            parser.error("--reference-rows takes no campaign, --references, --only, or --release")
        errors = verify_reference_rows(args.reference_rows)
        if errors:
            print("design acceptance FAILED:\n" + "\n".join(f"- {error}" for error in errors))
            return 1
        print(f"design acceptance passed: reference rows in {args.reference_rows}")
        return 0
    if args.references:
        if args.campaign or args.release or slugs is not None:
            parser.error("--references takes no campaign, --only, or --release")
        errors = verify_reference_fixtures()
        label = "all 34 reviewed reference inputs at the compiler boundary"
        if errors:
            print("design acceptance FAILED:\n" + "\n".join(f"- {error}" for error in errors))
            return 1
        print(f"design acceptance passed: {label}")
        return 0
    if args.mode == "design-only" and not args.campaign:
        parser.error("a campaign path is required unless --references is used")
    if args.release:
        if args.mode != "full" or slugs is not None:
            parser.error("--release requires --mode full and no --only")
        errors = verify_release_campaigns(args.campaign, contract_version=args.contract_version)
        label = "three full 34-brief campaigns"
    elif len(args.campaign) != 1:
        parser.error("one campaign is required unless --release is used")
    elif args.mode == "design-only":
        errors = verify_campaign(args.campaign[0], slugs)
        label = f"{args.only or 'all 34 briefs'}; five fresh stages each (design-only canary)"
    else:
        errors = verify_fulfillment_campaign(args.campaign[0], slugs, contract_version=args.contract_version)
        label = f"{args.only or 'all 34 briefs'}; artifact-verified fulfillment"
    if errors:
        print("design acceptance FAILED:\n" + "\n".join(f"- {error}" for error in errors))
        return 1
    print(f"design acceptance passed: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
