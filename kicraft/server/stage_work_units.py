"""Pure work-unit planning, validation, merging, and draft persistence."""

from __future__ import annotations
import hashlib
import json
import re

from dataclasses import asdict, dataclass
from typing import Literal
from pathlib import Path

from kicraft.fsutil import atomic_write_text
from kicraft.design.recipes import locked_no_connect_pins, locked_pin_assignments
from .stage_contracts import (
    BomArrayGroup,
    BomComponentGroup,
    ConnectedPinAssignment,
    NoConnectPinAssignment,
    _expand_bom_groups,
)

WORK_UNIT_WIRING_PIN_LIMIT = 256


@dataclass(frozen=True)
class StageWorkUnit:
    """Immutable ownership boundary for one bounded BOM or wiring generation call."""

    unit_id: str
    stage: Literal["bom", "wiring"]
    sheet: str
    refs: tuple[str, ...] = ()
    expected_pins: tuple[tuple[str, str], ...] = ()


def _architecture_sheets(prompt_state: dict) -> tuple[str, ...]:
    architecture = prompt_state.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("work-unit planning requires architecture")
    sheets = architecture.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise ValueError("work-unit planning requires nonempty architecture.sheets")
    names = tuple(
        str(sheet.get("name")) for sheet in sheets if isinstance(sheet, dict) and sheet.get("name")
    )
    if len(names) != len(sheets) or len(set(names)) != len(names):
        raise ValueError("architecture.sheets must contain unique named objects")
    return names


def _pin_numbers(info: object) -> tuple[str, ...]:
    if not isinstance(info, dict):
        return ()
    seen: set[str] = set()
    numbers: list[str] = []
    for pin in info.get("pins") or []:
        if not isinstance(pin, dict) or pin.get("number") is None:
            continue
        number = str(pin["number"])
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return tuple(numbers)


def _ref_pin_inventory(prompt_state: dict, extras: dict) -> dict[str, tuple[str, ...]]:
    """Return exact per-reference pin inventories, accepting legacy symbol-keyed extras."""
    bom = prompt_state.get("bom")
    if not isinstance(bom, dict):
        raise ValueError("wiring work-unit planning requires a committed BOM")
    pinouts = extras.get("symbol_pinouts")
    if not isinstance(pinouts, dict):
        raise ValueError("wiring work-unit planning requires extras.symbol_pinouts")
    inventory: dict[str, tuple[str, ...]] = {}
    for part in bom.get("parts") or []:
        if not isinstance(part, dict) or not part.get("ref"):
            continue
        ref = str(part["ref"])
        symbol = str(part.get("symbol") or "")
        info = pinouts.get(ref)
        if info is None:
            info = pinouts.get(symbol)
        inventory[ref] = _pin_numbers(info)
    return inventory


def _locked_pin_sets(
    prompt_state: dict, extras: dict
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    bom = prompt_state.get("bom")
    if not isinstance(bom, dict):
        return set(), set()
    connected = set(locked_pin_assignments(bom))
    no_connect = set(locked_no_connect_pins(bom))
    for row in extras.get("locked_pin_assignments") or extras.get("recipe_locked_pins") or []:
        if isinstance(row, dict) and row.get("ref") is not None and row.get("pin") is not None:
            connected.add((str(row["ref"]), str(row["pin"])))
    for row in extras.get("locked_no_connect_pins") or []:
        if isinstance(row, dict) and row.get("ref") is not None and row.get("pin") is not None:
            no_connect.add((str(row["ref"]), str(row["pin"])))
    return connected, no_connect


def plan_stage_work_units(
    stage: str, prompt_state: dict, extras: dict
) -> tuple[StageWorkUnit, ...]:
    """Plan deterministic sheet-local BOM or bounded exact-pin wiring units."""
    sheets = _architecture_sheets(prompt_state)
    if stage == "bom":
        return tuple(
            StageWorkUnit(unit_id=f"bom-s{index:03d}", stage="bom", sheet=sheet)
            for index, sheet in enumerate(sheets)
        )
    if stage != "wiring":
        raise ValueError(f"work units are unsupported for stage {stage!r}")

    bom = prompt_state.get("bom")
    if not isinstance(bom, dict):
        raise ValueError("wiring work-unit planning requires a committed BOM")
    inventory = _ref_pin_inventory(prompt_state, extras)
    locked_connected, locked_no_connect = _locked_pin_sets(prompt_state, extras)
    locked = locked_connected | locked_no_connect
    parts_by_sheet: dict[str, list[dict]] = {sheet: [] for sheet in sheets}
    for part in bom.get("parts") or []:
        if not isinstance(part, dict) or not part.get("ref"):
            continue
        sheet = str(part.get("sheet") or "")
        if sheet not in parts_by_sheet:
            raise ValueError(f"BOM ref {part['ref']!r} uses unknown architecture sheet {sheet!r}")
        parts_by_sheet[sheet].append(part)

    units: list[StageWorkUnit] = []
    unit_index = 0
    for sheet in sheets:
        pending_refs: list[str] = []
        pending_pins: list[tuple[str, str]] = []

        def flush() -> None:
            nonlocal unit_index
            if not pending_pins:
                return
            units.append(
                StageWorkUnit(
                    unit_id=f"wiring-u{unit_index:03d}",
                    stage="wiring",
                    sheet=sheet,
                    refs=tuple(pending_refs),
                    expected_pins=tuple(pending_pins),
                )
            )
            unit_index += 1
            pending_refs.clear()
            pending_pins.clear()

        for part in parts_by_sheet[sheet]:
            ref = str(part["ref"])
            ref_pins = [(ref, pin) for pin in inventory.get(ref, ()) if (ref, pin) not in locked]
            if not ref_pins:
                continue
            if len(ref_pins) > WORK_UNIT_WIRING_PIN_LIMIT:
                flush()
                for start in range(0, len(ref_pins), WORK_UNIT_WIRING_PIN_LIMIT):
                    pin_slice = tuple(ref_pins[start : start + WORK_UNIT_WIRING_PIN_LIMIT])
                    units.append(
                        StageWorkUnit(
                            unit_id=f"wiring-u{unit_index:03d}",
                            stage="wiring",
                            sheet=sheet,
                            refs=(ref,),
                            expected_pins=pin_slice,
                        )
                    )
                    unit_index += 1
                continue
            if pending_pins and len(pending_pins) + len(ref_pins) > WORK_UNIT_WIRING_PIN_LIMIT:
                flush()
            pending_refs.append(ref)
            pending_pins.extend(ref_pins)
        flush()
    return tuple(units)


class WorkUnitValidationError(ValueError):
    """A locally invalid unit candidate with all ownership defects attached."""

    def __init__(self, unit_id: str, defects: dict[str, list[str]]):
        self.unit_id = unit_id
        self.defects = defects
        detail = "; ".join(f"{name}={values[:12]!r}" for name, values in defects.items() if values)
        super().__init__(f"work unit {unit_id} invalid: {detail}")


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _validate_bom_unit(unit: StageWorkUnit, payload: dict, prompt_state: dict) -> dict:
    groups = [BomComponentGroup.model_validate(group) for group in (payload.get("groups") or [])]
    arrays = [BomArrayGroup.model_validate(array) for array in (payload.get("arrays") or [])]
    group_ids = [group.id for group in groups]
    array_group_ids = [array.group_id for array in arrays]
    architecture = prompt_state.get("architecture") or {}
    from kicraft.design.recipes import expand_selections

    recipe_parts = [
        part
        for expansion in expand_selections(architecture.get("recipe_selections") or [])
        for part in expansion.parts
    ]
    recipe_identities = {(part.symbol.lower(), part.value.lower()) for part in recipe_parts}
    defects = {
        "wrong-sheet": [
            f"{group.id}:{group.sheet}" for group in groups if group.sheet != unit.sheet
        ],
        "duplicate-group": _duplicates(group_ids),
        "bad-array-reference": [
            group_id for group_id in array_group_ids if group_id not in set(group_ids)
        ]
        + [f"duplicate:{group_id}" for group_id in _duplicates(array_group_ids)],
        "recipe-duplicate": [
            group.id
            for group in groups
            if (group.symbol.lower(), group.value.lower()) in recipe_identities
        ],
    }
    if any(defects.values()):
        raise WorkUnitValidationError(unit.unit_id, defects)
    return {
        "groups": [group.model_dump(exclude_none=True) for group in groups],
        "arrays": [array.model_dump(exclude_none=True) for array in arrays],
        "assumptions": [str(value) for value in payload.get("assumptions") or []],
        "substitutions": list(payload.get("substitutions") or []),
    }


def _validate_wiring_unit(
    unit: StageWorkUnit, payload: dict, prompt_state: dict, extras: dict
) -> dict:
    assignments: list[ConnectedPinAssignment | NoConnectPinAssignment] = []
    for row in payload.get("pins") or []:
        model = (
            NoConnectPinAssignment
            if isinstance(row, dict) and "no_connect" in row
            else ConnectedPinAssignment
        )
        assignments.append(model.model_validate(row))
    observed = [(assignment.ref, assignment.pin) for assignment in assignments]
    expected = set(unit.expected_pins)
    observed_set = set(observed)
    bom = prompt_state.get("bom") or {}
    known_refs = {
        str(part["ref"])
        for part in bom.get("parts") or []
        if isinstance(part, dict) and part.get("ref") is not None
    }
    inventory = _ref_pin_inventory(prompt_state, extras)
    known_pins = {(ref, pin) for ref, pins in inventory.items() for pin in pins}
    locked_connected, locked_no_connect = _locked_pin_sets(prompt_state, extras)
    recipe_owned = locked_connected | locked_no_connect
    duplicate_keys = _duplicates([f"{ref}\0{pin}" for ref, pin in observed])
    defects = {
        "duplicate": [key.replace("\0", ".") for key in duplicate_keys],
        "missing": [
            f"{ref}.{pin}" for ref, pin in unit.expected_pins if (ref, pin) not in observed_set
        ],
        "unexpected": [f"{ref}.{pin}" for ref, pin in observed if (ref, pin) not in expected],
        "unknown-ref": [ref for ref, _pin in observed if ref not in known_refs],
        "unknown-pin": [
            f"{ref}.{pin}"
            for ref, pin in observed
            if ref in known_refs and (ref, pin) not in known_pins
        ],
        "recipe-owned": [f"{ref}.{pin}" for ref, pin in observed if (ref, pin) in recipe_owned],
    }
    for key, values in defects.items():
        defects[key] = list(dict.fromkeys(values))
    if any(defects.values()):
        raise WorkUnitValidationError(unit.unit_id, defects)
    by_pin = {
        (assignment.ref, assignment.pin): assignment.model_dump(exclude_none=True)
        for assignment in assignments
    }
    return {"pins": [by_pin[pin] for pin in unit.expected_pins]}


def validate_unit_candidate(
    unit: StageWorkUnit, payload: dict, prompt_state: dict, extras: dict
) -> dict:
    """Validate one complete unit replacement before it can enter the aggregate."""
    if not isinstance(payload, dict):
        raise TypeError("work-unit payload must be an object")
    if unit.stage == "bom":
        return _validate_bom_unit(unit, payload, prompt_state)
    return _validate_wiring_unit(unit, payload, prompt_state, extras)


def _stable_dedupe(values: list) -> list:
    seen: set[str] = set()
    result: list = []
    for value in values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def merge_bom_units(
    units: tuple[StageWorkUnit, ...],
    candidates: dict[str, dict],
    prompt_state: dict,
) -> tuple[dict, dict[str, str]]:
    """Merge sheet replacements and return model-authored ref provenance."""
    sheet_indexes = {sheet: index for index, sheet in enumerate(_architecture_sheets(prompt_state))}
    groups: list[dict] = []
    arrays: list[dict] = []
    assumptions: list[str] = []
    substitutions: list[dict] = []
    group_to_unit: dict[str, str] = {}
    for unit in units:
        if unit.stage != "bom":
            raise ValueError(f"cannot merge {unit.stage!r} unit into BOM")
        candidate = candidates.get(unit.unit_id)
        if candidate is None:
            raise ValueError(f"missing validated candidate for {unit.unit_id}")
        prefix = f"s{sheet_indexes[unit.sheet]:03d}_"
        local_to_global: dict[str, str] = {}
        for group in candidate.get("groups") or []:
            local_id = str(group["id"])
            global_id = f"{prefix}{local_id}"
            local_to_global[local_id] = global_id
            group_to_unit[global_id] = unit.unit_id
            groups.append({**group, "id": global_id})
        for array in candidate.get("arrays") or []:
            local_group = str(array["group_id"])
            if local_group not in local_to_global:
                raise ValueError(
                    f"validated unit {unit.unit_id} has unknown array group {local_group!r}"
                )
            arrays.append({**array, "group_id": local_to_global[local_group]})
        assumptions.extend(candidate.get("assumptions") or [])
        substitutions.extend(candidate.get("substitutions") or [])
    merged = {
        "groups": groups,
        "arrays": arrays,
        "assumptions": _stable_dedupe(assumptions),
        "substitutions": _stable_dedupe(substitutions),
    }
    _response, _parts, _arrays, refs_by_group, _expansions = _expand_bom_groups(
        merged, prompt_state
    )
    ref_to_unit = {
        ref: group_to_unit[group_id] for group_id, refs in refs_by_group.items() for ref in refs
    }
    return merged, ref_to_unit


def merge_wiring_units(
    units: tuple[StageWorkUnit, ...],
    candidates: dict[str, dict],
) -> tuple[
    dict,
    dict[tuple[str, str], str],
    dict[str, tuple[str, ...]],
]:
    """Merge exact pin replacements and reject overlapping unit ownership."""
    pin_to_unit: dict[tuple[str, str], str] = {}
    ref_to_unit_ids: dict[str, list[str]] = {}
    rows_by_pin: dict[tuple[str, str], dict] = {}
    for unit in units:
        if unit.stage != "wiring":
            raise ValueError(f"cannot merge {unit.stage!r} unit into wiring")
        candidate = candidates.get(unit.unit_id)
        if candidate is None:
            raise ValueError(f"missing validated candidate for {unit.unit_id}")
        candidate_by_pin = {
            (str(row["ref"]), str(row["pin"])): row for row in candidate.get("pins") or []
        }
        for pin in unit.expected_pins:
            prior = pin_to_unit.get(pin)
            if prior is not None:
                raise ValueError(
                    f"overlapping wiring ownership for {pin[0]}.{pin[1]}: {prior}, {unit.unit_id}"
                )
            if pin not in candidate_by_pin:
                raise ValueError(f"validated unit {unit.unit_id} is missing {pin[0]}.{pin[1]}")
            pin_to_unit[pin] = unit.unit_id
            rows_by_pin[pin] = candidate_by_pin[pin]
            ids = ref_to_unit_ids.setdefault(pin[0], [])
            if unit.unit_id not in ids:
                ids.append(unit.unit_id)
    pins = [rows_by_pin[pin] for unit in units for pin in unit.expected_pins]
    return (
        {"pins": pins},
        pin_to_unit,
        {ref: tuple(unit_ids) for ref, unit_ids in ref_to_unit_ids.items()},
    )


def route_work_unit_ids(
    evidence: object,
    units: tuple[StageWorkUnit, ...],
    *,
    ref_to_unit: dict[str, str] | None = None,
    pin_to_unit: dict[tuple[str, str], str] | None = None,
    ref_to_unit_ids: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Map bounded gate evidence to owning units; unscoped evidence targets all."""
    text = json.dumps(evidence, sort_keys=True, default=str)

    def mentions(token: str) -> bool:
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])",
                text,
            )
        )

    def mentions_ref(token: str) -> bool:
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9_-]+)",
                text,
            )
        )

    selected: set[str] = set()
    for pin, unit_id in (pin_to_unit or {}).items():
        if mentions(f"{pin[0]}.{pin[1]}"):
            selected.add(unit_id)
    for ref, unit_id in (ref_to_unit or {}).items():
        if mentions_ref(ref):
            selected.add(unit_id)
    for ref, unit_ids in (ref_to_unit_ids or {}).items():
        if mentions_ref(ref):
            selected.update(unit_ids)
    for unit in units:
        if unit.sheet and unit.sheet in text:
            selected.add(unit.unit_id)
    if not selected:
        return tuple(unit.unit_id for unit in units)
    return tuple(unit.unit_id for unit in units if unit.unit_id in selected)


def stage_draft_fingerprint(
    *,
    stage: str,
    brief: str,
    prompt_state: dict,
    answers,
    instruction,
    stage_spec_sha256: str,
    response_contract_names: tuple[str, ...],
    units: tuple[StageWorkUnit, ...],
    extras: dict,
) -> str:
    """Fingerprint every input capable of changing a validated unit candidate."""
    basis = {
        "stage": stage,
        "brief_sha256": hashlib.sha256(brief.encode("utf-8")).hexdigest(),
        "prompt_state": prompt_state,
        "answers": list(answers or []),
        "instruction": instruction,
        "stage_spec_sha256": stage_spec_sha256,
        "response_contract_names": list(response_contract_names),
        "work_units": [asdict(unit) for unit in units],
        "extras": extras,
    }
    encoded = json.dumps(
        basis,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StageDraftStore:
    """Atomic exact-fingerprint storage for validated model-facing unit candidates."""

    SCHEMA_VERSION = 1

    def __init__(self, state_path, stage: str):
        self.stage = stage
        self.path = Path(state_path).parent / "drafts" / f"{stage}-units.json"

    def load(
        self,
        fingerprint: str,
        units: tuple[StageWorkUnit, ...],
    ) -> dict[str, dict]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return {}
        descriptors = json.loads(json.dumps([asdict(unit) for unit in units]))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != self.SCHEMA_VERSION
            or payload.get("stage") != self.stage
            or payload.get("fingerprint") != fingerprint
            or payload.get("work_units") != descriptors
            or not isinstance(payload.get("candidates"), dict)
        ):
            return {}
        known = {unit.unit_id for unit in units}
        return {
            unit_id: candidate
            for unit_id, candidate in payload["candidates"].items()
            if unit_id in known and isinstance(candidate, dict)
        }

    def save(
        self,
        fingerprint: str,
        units: tuple[StageWorkUnit, ...],
        candidates: dict[str, dict],
    ) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "stage": self.stage,
            "fingerprint": fingerprint,
            "work_units": [asdict(unit) for unit in units],
            "candidates": {
                unit.unit_id: candidates[unit.unit_id]
                for unit in units
                if unit.unit_id in candidates
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
