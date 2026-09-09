"""Exclusive ownership and provider-call coverage accounting for frozen runs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


_SOURCES = ("recipe", "allocator", "lowerer", "reuse", "llm")


def _source(value: object) -> str:
    text = str(value or "llm")
    return {
        "deterministic_architecture_lowering": "lowerer",
        "reused_validated_draft": "reuse",
        "recipe_plus_llm": "llm",
    }.get(text, text if text in _SOURCES else "llm")


def _events_from_path(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            events.append(row)
    return events


def analyze_recipe_coverage(
    state: dict,
    provenance_events: Iterable[dict] = (),
) -> dict:
    """Attribute each canonical part, pin, provider call, and cost exactly once."""
    bom = state.get("bom") or {}
    parts = [part for part in bom.get("parts") or [] if isinstance(part, dict)]
    part_sources: dict[str, str] = {}
    part_counts = Counter({source: 0 for source in _SOURCES})
    for part in parts:
        ref = str(part.get("ref") or "")
        source = "recipe" if part.get("recipe_id") else _source(part.get("resolution_source"))
        if ref in part_sources:
            raise ValueError(f"coverage_duplicate_part: {ref}")
        part_sources[ref] = source
        part_counts[source] += 1

    exact_pin_owners: dict[tuple[str, str], str] = {}
    for manifest in bom.get("recipe_ownership") or []:
        if not isinstance(manifest, dict):
            continue
        for row in manifest.get("pins") or []:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("ref")), str(row.get("pin")))
            if key in exact_pin_owners:
                raise ValueError(f"coverage_duplicate_pin_owner: {key[0]}.{key[1]}")
            exact_pin_owners[key] = _source(row.get("owner"))

    events = [event for event in provenance_events if isinstance(event, dict)]
    wiring_ref_sources: dict[str, str] = {}
    for event in events:
        if event.get("kind") != "work_unit_plan" or event.get("stage") != "wiring":
            continue
        source = _source(event.get("source"))
        for ref in event.get("refs") or []:
            wiring_ref_sources.setdefault(str(ref), source)

    observed_pins: set[tuple[str, str]] = set()
    pin_counts = Counter({source: 0 for source in _SOURCES})
    for connection in bom.get("connections") or []:
        if not isinstance(connection, dict):
            continue
        for endpoint in connection.get("endpoints") or []:
            if not isinstance(endpoint, dict):
                continue
            key = (str(endpoint.get("ref")), str(endpoint.get("pin")))
            if key in observed_pins:
                raise ValueError(f"coverage_pin_counted_twice: {key[0]}.{key[1]}")
            observed_pins.add(key)
            source = exact_pin_owners.get(
                key,
                wiring_ref_sources.get(key[0], part_sources.get(key[0], "llm")),
            )
            pin_counts[source] += 1
    for endpoint in bom.get("no_connect_pins") or []:
        if not isinstance(endpoint, dict):
            continue
        key = (str(endpoint.get("ref")), str(endpoint.get("pin")))
        if key in observed_pins:
            raise ValueError(f"coverage_pin_counted_twice: {key[0]}.{key[1]}")
        observed_pins.add(key)
        source = exact_pin_owners.get(
            key,
            wiring_ref_sources.get(key[0], part_sources.get(key[0], "llm")),
        )
        pin_counts[source] += 1

    calls = Counter({source: 0 for source in _SOURCES})
    costs = Counter({source: 0.0 for source in _SOURCES})
    protected_violations: list[dict] = []
    work_units: dict[tuple[str, str], dict] = {}
    seen_attempts: set[tuple[object, object, object]] = set()
    for event in events:
        stage = str(event.get("stage") or "")
        unit_id = str(event.get("unit_id") or "")
        unit_key = (stage, unit_id)
        if event.get("kind") == "work_unit_plan" and unit_id:
            work_units.setdefault(
                unit_key,
                {
                    "stage": stage,
                    "unit_id": unit_id,
                    "source": _source(event.get("source")),
                    "calls": 0,
                    "cost_usd": 0.0,
                },
            )
        if event.get("kind") == "work_unit_attempt":
            attempt_key = (
                stage,
                unit_id,
                event.get("provider_attempt"),
            )
            if attempt_key not in seen_attempts:
                seen_attempts.add(attempt_key)
                cost = float(event.get("cost_usd") or 0.0)
                calls["llm"] += 1
                costs["llm"] += cost
                unit = work_units.setdefault(
                    unit_key,
                    {
                        "stage": stage,
                        "unit_id": unit_id,
                        "source": "llm",
                        "calls": 0,
                        "cost_usd": 0.0,
                    },
                )
                unit["source"] = "llm"
                unit["calls"] += 1
                unit["cost_usd"] = round(unit["cost_usd"] + cost, 9)
        text = json.dumps(event, sort_keys=True, default=str)
        if "model_authored_protected_identity" in text or "unsupported_protected_variant" in text:
            protected_violations.append(event)

    return {
        "version": 1,
        "parts": {"total": len(parts), **dict(part_counts)},
        "pins": {"total": len(observed_pins), **dict(pin_counts)},
        "calls": {"total": sum(calls.values()), **dict(calls)},
        "cost_usd": {"total": round(sum(costs.values()), 9), **dict(costs)},
        "protected_family_violations": protected_violations,
        "work_units": [
            work_units[key] for key in sorted(work_units)
        ],
    }


def analyze_frozen_run(project_dir: str | Path) -> dict:
    root = Path(project_dir)
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    return analyze_recipe_coverage(state, _events_from_path(root / "provenance.jsonl"))


def analyze_frozen_corpus(project_dirs: Iterable[str | Path]) -> dict:
    runs = [analyze_frozen_run(path) for path in project_dirs]
    aggregate = {
        category: Counter()
        for category in ("parts", "pins", "calls", "cost_usd")
    }
    violations: list[dict] = []
    for run in runs:
        for category in aggregate:
            aggregate[category].update(run[category])
        violations.extend(run["protected_family_violations"])
    return {
        "version": 1,
        "runs": runs,
        **{category: dict(counter) for category, counter in aggregate.items()},
        "protected_family_violations": violations,
    }
