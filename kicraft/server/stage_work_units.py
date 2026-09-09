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
    _requirement_owns_protected_group,
)

WORK_UNIT_WIRING_PIN_LIMIT = 256
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "sixteen": 16,
}


@dataclass(frozen=True)
class StageWorkUnit:
    """Immutable ownership boundary for one bounded BOM or wiring generation call."""

    unit_id: str
    stage: Literal["bom", "wiring"]
    sheet: str
    refs: tuple[str, ...] = ()
    expected_pins: tuple[tuple[str, str], ...] = ()
    requirement_ids: tuple[str, ...] = ()
    owned_roles: tuple[str, ...] = ()
    excluded_refs: tuple[str, ...] = ()
    excluded_pins: tuple[tuple[str, str], ...] = ()
    planned_resolution_source: str = "llm"
    recipe_ids: tuple[str, ...] = ()
    lowerer_ids: tuple[str, ...] = ()


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
    architecture = prompt_state.get("architecture") or {}
    sheets = _architecture_sheets(prompt_state)
    if stage == "bom":
        requirements = [
            requirement
            for requirement in architecture.get("requirements") or []
            if isinstance(requirement, dict)
        ]
        if requirements:
            selected = {
                str(requirement_id)
                for selection in architecture.get("recipe_selections") or []
                if isinstance(selection, dict)
                for requirement_id in selection.get("requirement_ids") or []
            }
            recipe_ids_by_sheet: dict[str, set[str]] = {}
            for selection in architecture.get("recipe_selections") or []:
                if not isinstance(selection, dict):
                    continue
                for sheet in (selection.get("sheets") or {}).values():
                    recipe_ids_by_sheet.setdefault(str(sheet), set()).add(
                        str(selection.get("recipe"))
                    )
            return tuple(
                StageWorkUnit(
                    unit_id=f"bom-r{index:03d}",
                    stage="bom",
                    sheet=str(requirement["sheet"]),
                    requirement_ids=(str(requirement["id"]),),
                    owned_roles=(str(requirement["role"]),),
                    planned_resolution_source="llm",
                    recipe_ids=tuple(
                        sorted(recipe_ids_by_sheet.get(str(requirement["sheet"]), set()))
                    ),
                )
                for index, requirement in enumerate(requirements)
                if str(requirement.get("id")) not in selected
            )
        return tuple(
            StageWorkUnit(
                unit_id=f"bom-s{index:03d}",
                stage="bom",
                sheet=sheet,
            )
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

    recipe_manifests = [
        manifest for manifest in bom.get("recipe_ownership") or [] if isinstance(manifest, dict)
    ]
    excluded_refs = tuple(
        sorted({str(ref) for manifest in recipe_manifests for ref in manifest.get("refs") or []})
    )
    recipe_ids = tuple(sorted({str(manifest.get("recipe")) for manifest in recipe_manifests}))
    units: list[StageWorkUnit] = []
    unit_index = 0
    for sheet in sheets:
        lowerer_buckets: dict[tuple[str, str], list[dict]] = {}
        ordinary_parts: list[dict] = []
        for part in parts_by_sheet[sheet]:
            if (
                part.get("resolution_source") == "lowerer"
                and part.get("resolution_id")
                and part.get("lowering_requirement_id")
            ):
                key = (
                    str(part["resolution_id"]),
                    str(part["lowering_requirement_id"]),
                )
                lowerer_buckets.setdefault(key, []).append(part)
            else:
                ordinary_parts.append(part)
        batches = [
            (parts, lowerer_id, requirement_id)
            for (lowerer_id, requirement_id), parts in lowerer_buckets.items()
        ]
        if ordinary_parts:
            batches.append((ordinary_parts, None, None))

        for batch_parts, lowerer_id, requirement_id in batches:
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
                        excluded_refs=excluded_refs,
                        excluded_pins=tuple(sorted(locked)),
                        planned_resolution_source=("lowerer" if lowerer_id is not None else "llm"),
                        recipe_ids=recipe_ids,
                        lowerer_ids=((lowerer_id,) if lowerer_id is not None else ()),
                        requirement_ids=((requirement_id,) if requirement_id is not None else ()),
                    )
                )
                unit_index += 1
                pending_refs.clear()
                pending_pins.clear()

            for part in batch_parts:
                ref = str(part["ref"])
                ref_pins = [
                    (ref, pin) for pin in inventory.get(ref, ()) if (ref, pin) not in locked
                ]
                if not ref_pins:
                    continue
                if len(ref_pins) > WORK_UNIT_WIRING_PIN_LIMIT:
                    flush()
                    for start in range(
                        0,
                        len(ref_pins),
                        WORK_UNIT_WIRING_PIN_LIMIT,
                    ):
                        pin_slice = tuple(ref_pins[start : start + WORK_UNIT_WIRING_PIN_LIMIT])
                        units.append(
                            StageWorkUnit(
                                unit_id=f"wiring-u{unit_index:03d}",
                                stage="wiring",
                                sheet=sheet,
                                refs=(ref,),
                                expected_pins=pin_slice,
                                excluded_refs=excluded_refs,
                                excluded_pins=tuple(sorted(locked)),
                                planned_resolution_source=(
                                    "lowerer" if lowerer_id is not None else "llm"
                                ),
                                recipe_ids=recipe_ids,
                                lowerer_ids=((lowerer_id,) if lowerer_id is not None else ()),
                                requirement_ids=(
                                    (requirement_id,) if requirement_id is not None else ()
                                ),
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


def _unit_requirement(unit: StageWorkUnit, prompt_state: dict):
    from kicraft.design.models import CircuitRequirement

    if len(unit.requirement_ids) != 1:
        return None
    requirement_id = unit.requirement_ids[0]
    for row in (prompt_state.get("architecture") or {}).get("requirements") or []:
        if isinstance(row, dict) and str(row.get("id")) == requirement_id:
            return CircuitRequirement.model_validate(row)
    return None


def deterministic_bom_candidate(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> dict | None:
    """Return one typed lowerer's combined BOM/wiring artifact."""
    from kicraft.design.lowering import lower_requirement

    requirement = _unit_requirement(unit, prompt_state)
    if requirement is None:
        return None
    artifact = lower_requirement(requirement)
    if artifact is None:
        return None
    return {
        "groups": [
            {
                "id": group.role,
                "reference_prefix": group.reference_prefix,
                "quantity": group.quantity,
                "value": group.value,
                "symbol": group.symbol,
                "footprint": group.footprint,
                "sheet": requirement.sheet,
                **({"mpn": group.mpn} if group.mpn else {}),
            }
            for group in artifact.groups
        ],
        "arrays": [],
        "assumptions": list(artifact.assumptions),
        "substitutions": [],
        "_lowerer_id": artifact.lowerer_id,
        "_lowering_requirement_id": artifact.requirement_id,
        "_lowering_roles": {group.role: {"lowering_role": group.role} for group in artifact.groups},
        "_calculations": [
            calculation.model_dump(mode="json") for calculation in artifact.calculations
        ],
    }


def _normalize_curated_group_identities(
    groups: list[BomComponentGroup],
) -> list[BomComponentGroup]:
    """Use a selected curated bundle's authoritative symbol/footprint pair."""
    from kicraft.parts_library import find_part

    bundles: dict[str, object | None] = {}
    normalized: list[BomComponentGroup] = []
    for group in groups:
        library = group.symbol.partition(":")[0]
        if library not in bundles:
            bundles[library] = find_part(library, project_root=None)
        loaded = bundles[library]
        manifest = getattr(loaded, "manifest", None)
        if manifest is None:
            normalized.append(group)
            continue
        selected_identities = {
            re.sub(r"[^a-z0-9]+", "", str(value).lower())
            for value in (group.mpn, group.value)
            if value
        }
        manifest_identity = re.sub(r"[^a-z0-9]+", "", manifest.mpn.lower())
        if manifest_identity not in selected_identities:
            normalized.append(group)
            continue
        normalized.append(
            group.model_copy(
                update={
                    "symbol": f"{manifest.name}:{manifest.symbol_name}",
                    "footprint": f"{manifest.name}:{manifest.footprint_name}",
                }
            )
        )
    return normalized


def _validate_bom_unit(unit: StageWorkUnit, payload: dict, prompt_state: dict) -> dict:
    groups = _normalize_curated_group_identities(
        [BomComponentGroup.model_validate(group) for group in (payload.get("groups") or [])]
    )
    assumptions = [str(value) for value in payload.get("assumptions") or []]
    lowering_metadata = {key: value for key, value in payload.items() if str(key).startswith("_")}
    if not groups:
        deterministic = deterministic_bom_candidate(unit, prompt_state)
        if deterministic is not None:
            groups = [BomComponentGroup.model_validate(group) for group in deterministic["groups"]]
            assumptions.extend(str(value) for value in deterministic.get("assumptions") or [])
            lowering_metadata = {
                key: value for key, value in deterministic.items() if key.startswith("_")
            }
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
    from kicraft.design.recipes import protected_identity_matches

    requirement = _unit_requirement(unit, prompt_state)
    requirement_id = (
        requirement.get("id") if isinstance(requirement, dict) else getattr(requirement, "id", None)
    )
    requirement_family = (
        requirement.get("family")
        if isinstance(requirement, dict)
        else getattr(requirement, "family", None)
    )
    generic_families = {"", "applicationspecific", "custom", "generic"}
    requires_named_implementation = (
        requirement is not None
        and not lowering_metadata
        and re.sub(r"[^a-z0-9]+", "", str(requirement_family or "").lower()) not in generic_families
    )
    protected_groups = [
        group.id
        for group in groups
        if protected_identity_matches(group.id, group.symbol, group.value, group.mpn)
        and not _requirement_owns_protected_group(
            group, (requirement,) if requirement is not None else ()
        )
    ]
    # A requirement-scoped unit sometimes redundantly emits a protected sibling
    # (for example a USB receptacle beside its PD controller). Another planned
    # unit owns that component. Preserve the owned implementation and discard
    # only the extra group instead of rejecting an otherwise usable unit.
    if protected_groups and len(groups) > len(protected_groups):
        discarded = set(protected_groups)
        groups = [group for group in groups if group.id not in discarded]
        arrays = [array for array in arrays if array.group_id not in discarded]
        protected_groups = []
        group_ids = [group.id for group in groups]
        array_group_ids = [array.group_id for array in arrays]
    recipe_sheets = {part.sheet for part in recipe_parts}
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
        "model_authored_protected_identity": protected_groups,
        "missing-requirement-implementation": (
            [str(requirement_id)]
            if requires_named_implementation
            and not any(
                _requirement_owns_protected_group(group, (requirement,)) for group in groups
            )
            else []
        ),
        "empty-sheet": (
            [unit.sheet]
            if not groups and (unit.requirement_ids or unit.sheet not in recipe_sheets)
            else []
        ),
    }
    if any(defects.values()):
        raise WorkUnitValidationError(unit.unit_id, defects)
    return {
        "groups": [group.model_dump(exclude_none=True) for group in groups],
        "arrays": [array.model_dump(exclude_none=True) for array in arrays],
        "assumptions": list(dict.fromkeys(assumptions)),
        "substitutions": list(payload.get("substitutions") or []),
        **lowering_metadata,
    }


def _natural_key(value: str) -> tuple:
    return tuple(
        int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)
    )


def _sheet_signal_nets(unit: StageWorkUnit, architecture: dict) -> list[dict]:
    power_nets = {str(value) for value in architecture.get("power_nets") or []}
    return [
        net
        for net in architecture.get("inter_sheet_nets") or []
        if isinstance(net, dict)
        and str(net.get("name")) not in power_nets
        and any(
            isinstance(endpoint, dict) and endpoint.get("sheet") == unit.sheet
            for endpoint in net.get("endpoints") or []
        )
    ]


def deterministic_wiring_candidate(
    unit: StageWorkUnit,
    prompt_state: dict,
    extras: dict,
) -> dict | None:
    """Render wiring from the same typed artifact that created the BOM."""
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.models import CircuitRequirement

    bom = prompt_state.get("bom") or {}
    parts = [
        part
        for part in bom.get("parts") or []
        if isinstance(part, dict)
        and str(part.get("ref")) in unit.refs
        and part.get("resolution_source") == "lowerer"
    ]
    lowerer_ids = {str(part.get("resolution_id")) for part in parts}
    requirement_ids = {
        str(part.get("lowering_requirement_id"))
        for part in parts
        if part.get("lowering_requirement_id")
    }
    if len(lowerer_ids) != 1 or len(requirement_ids) != 1:
        return None
    requirement_id = next(iter(requirement_ids))
    requirement_row = next(
        (
            row
            for row in (prompt_state.get("architecture") or {}).get("requirements") or []
            if isinstance(row, dict) and str(row.get("id")) == requirement_id
        ),
        None,
    )
    if requirement_row is None:
        return None
    artifact = lower_requirement(CircuitRequirement.model_validate(requirement_row))
    if artifact is None or artifact.lowerer_id != next(iter(lowerer_ids)):
        return None
    refs = {
        (str(part.get("lowering_role")), int(part.get("lowering_index") or 0)): str(part["ref"])
        for part in parts
        if part.get("lowering_role")
    }
    expected_pins = set(unit.expected_pins)
    try:
        candidate = {
            "pins": [
                {
                    "ref": refs[(pin.role, pin.index)],
                    "pin": pin.pin,
                    "net": pin.net,
                }
                for pin in artifact.pins
                if (refs[(pin.role, pin.index)], pin.pin) in expected_pins
            ]
        }
        return _validate_wiring_unit(unit, candidate, prompt_state, extras)
    except (KeyError, WorkUnitValidationError, TypeError, ValueError):
        return None


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
) -> tuple[dict, dict[str, str], dict[str, dict]]:
    """Merge sheet replacements and return unit/lowerer ref provenance."""
    groups: list[dict] = []
    arrays: list[dict] = []
    assumptions: list[str] = []
    substitutions: list[dict] = []
    group_to_unit: dict[str, str] = {}
    lowering_by_group: dict[str, dict] = {}
    for unit in units:
        if unit.stage != "bom":
            raise ValueError(f"cannot merge {unit.stage!r} unit into BOM")
        candidate = candidates.get(unit.unit_id)
        if candidate is None:
            raise ValueError(f"missing validated candidate for {unit.unit_id}")
        prefix = f"{unit.unit_id.removeprefix('bom-')}_"
        local_to_global: dict[str, str] = {}
        for group in candidate.get("groups") or []:
            local_id = str(group["id"])
            global_id = f"{prefix}{local_id}"
            local_to_global[local_id] = global_id
            group_to_unit[global_id] = unit.unit_id
            groups.append({**group, "id": global_id})
            role_metadata = (candidate.get("_lowering_roles") or {}).get(local_id)
            if role_metadata:
                lowering_by_group[global_id] = {
                    "resolution_id": str(candidate["_lowerer_id"]),
                    "lowering_requirement_id": str(candidate["_lowering_requirement_id"]),
                    **role_metadata,
                }
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
    ref_to_lowering = {
        ref: {
            **lowering_by_group[group_id],
            "lowering_index": index,
        }
        for group_id, refs in refs_by_group.items()
        if group_id in lowering_by_group
        for index, ref in enumerate(refs)
    }
    return merged, ref_to_unit, ref_to_lowering


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
