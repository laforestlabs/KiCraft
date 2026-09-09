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




def _explicit_generic_interface_group(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> BomComponentGroup | None:
    """Lower an explicit generic interface topology without another model call."""
    architecture = prompt_state.get("architecture") or {}
    target = re.sub(r"[^a-z0-9]", "", unit.sheet.lower())
    topology = next(
        (
            str(value)
            for key, value in (architecture.get("topologies") or {}).items()
            if re.sub(r"[^a-z0-9]", "", str(key).lower()) == target
        ),
        "",
    )
    sheet = next(
        (
            row
            for row in architecture.get("sheets") or []
            if isinstance(row, dict) and row.get("name") == unit.sheet
        ),
        {},
    )
    sheet_description = f"{unit.sheet} {topology} {sheet.get('function') or ''}".lower()
    unit_name = unit.sheet.lower()
    if "power input" in unit_name:
        role_terms = ("power input",)
    elif "analog output" in unit_name:
        role_terms = ("analog output",)
    elif "input" in unit_name:
        role_terms = ("input header", "digital input")
    else:
        role_terms = ()
    relevant_assumptions = " ".join(
        str(value)
        for value in architecture.get("assumptions") or []
        if any(term in str(value).lower() for term in role_terms)
    )
    description = f"{sheet_description} {relevant_assumptions}".lower()
    assumptions = " ".join(
        str(value).lower() for value in architecture.get("assumptions") or []
    )
    power_nets = {str(value) for value in architecture.get("power_nets") or []}
    signal_nets = {
        str(net.get("name"))
        for net in architecture.get("inter_sheet_nets") or []
        if isinstance(net, dict)
        and str(net.get("name")) not in power_nets
        and any(
            isinstance(endpoint, dict) and endpoint.get("sheet") == unit.sheet
            for endpoint in net.get("endpoints") or []
        )
    }
    requires_ground = "ground" in description or "gnd" in description
    carries_power = "power input" in unit.sheet.lower() or (
        "input" in description
        and "header" in description
        and bool(
            re.search(
                r"(?:power input|power is supplied|supplied externally).{0,80}"
                r"(?:input )?header",
                assumptions,
            )
        )
    )
    if "header" not in description and "connector" not in description:
        return None
    match = re.search(r"\b1\s*[x×]\s*(\d{1,2})\b", description, re.I)
    pins = int(match.group(1)) if match is not None else 0
    if pins == 0:
        numeric_label = re.search(r"\b(\d{1,2})[- ]+pin\b", description)
        word_pattern = "|".join(_COUNT_WORDS)
        word_label = re.search(rf"\b({word_pattern})[- ]+pin\b", description)
        count_match = re.search(
            r"\b(\d{1,2})\s+(?:parallel\s+)?(?:(?:logic|digital)(?:-level)?\s+)?"
            r"(?:inputs?|pins?)\b",
            description,
        )
        word_match = re.search(
            rf"\b({word_pattern})\s+(?:parallel\s+)?"
            r"(?:(?:logic|digital)(?:-level)?\s+)?(?:inputs?|pins?)\b",
            description,
        )
        if numeric_label is not None:
            pins = int(numeric_label.group(1))
        elif word_label is not None:
            pins = _COUNT_WORDS[word_label.group(1)]
        elif count_match is not None:
            pins = int(count_match.group(1))
        elif word_match is not None:
            pins = _COUNT_WORDS[word_match.group(1)]
        if pins and not signal_nets and requires_ground:
            pins += 1
    inferred_pins = len(signal_nets) + int(requires_ground) + int(carries_power)
    if inferred_pins:
        pins = max(pins, inferred_pins)
    if not 1 <= pins <= 40:
        return None
    suffix = f"{pins:02d}"
    return BomComponentGroup(
        id="generic_pin_header",
        reference_prefix="J",
        quantity=1,
        value=f"PinHeader_1x{suffix}",
        symbol=f"Connector_Generic:Conn_01x{suffix}",
        footprint=(
            "Connector_PinHeader_2.54mm:"
            f"PinHeader_1x{suffix}_P2.54mm_Vertical"
        ),
        sheet=unit.sheet,
    )


def _explicit_low_voltage_opamp_groups(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> list[BomComponentGroup]:
    """Lower an explicit low-voltage unity-gain buffer to the curated default."""
    architecture = prompt_state.get("architecture") or {}
    target = re.sub(r"[^a-z0-9]", "", unit.sheet.lower())
    topology = " ".join(
        str(value)
        for key, value in (architecture.get("topologies") or {}).items()
        if re.sub(r"[^a-z0-9]", "", str(key).lower()) == target
    )
    sheet_function = " ".join(
        str(row.get("function") or "")
        for row in architecture.get("sheets") or []
        if isinstance(row, dict) and row.get("name") == unit.sheet
    )
    description = f"{unit.sheet} {topology} {sheet_function}".lower()
    rails = [
        float(value)
        for value in (architecture.get("rail_voltages") or {}).values()
        if isinstance(value, (int, float))
    ]
    is_buffer = ("op-amp" in description or "opamp" in description) and (
        "buffer" in description or "follower" in description
    )
    if not is_buffer or (rails and max(rails) > 6.0):
        return []
    return [
        BomComponentGroup(
            id="opamp_buffer",
            reference_prefix="U",
            quantity=1,
            value="MCP6001T-I/OT",
            symbol="mcp6001:MCP6001T-I_OT",
            footprint="mcp6001:SOT-23-5_L3.0-W1.7-P0.95-LS2.8-BR",
            sheet=unit.sheet,
            mpn="MCP6001T-I/OT",
            sourcing_note="Curated low-voltage unity-gain buffer default",
        ),
        BomComponentGroup(
            id="opamp_bypass",
            reference_prefix="C",
            quantity=1,
            value="100nF",
            symbol="Device:C",
            footprint="Capacitor_SMD:C_0603_1608Metric",
            sheet=unit.sheet,
        ),
    ]


def _explicit_mcp1700_ldo_groups(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> list[BomComponentGroup]:
    """Lower an explicit MCP1700 3.3 V LDO power-input block."""
    architecture = prompt_state.get("architecture") or {}
    target = re.sub(r"[^a-z0-9]", "", unit.sheet.lower())
    topology = " ".join(
        str(value)
        for key, value in (architecture.get("topologies") or {}).items()
        if re.sub(r"[^a-z0-9]", "", str(key).lower()) == target
    )
    sheet_function = " ".join(
        str(row.get("function") or "")
        for row in architecture.get("sheets") or []
        if isinstance(row, dict) and row.get("name") == unit.sheet
    )
    assumptions = " ".join(str(value) for value in architecture.get("assumptions") or [])
    description = f"{topology} {sheet_function} {assumptions}".lower()
    if "mcp1700" not in description or not re.search(r"\b(?:ldo|regulator)\b", description):
        return []
    interface = _explicit_generic_interface_group(unit, prompt_state)
    groups = [interface] if interface is not None else []
    groups.extend(
        [
            BomComponentGroup(
                id="mcp1700_3v3",
                reference_prefix="U",
                quantity=1,
                value="MCP1700T-3302E/TT",
                symbol="Regulator_Linear:MCP1700x-330xxTT",
                footprint="Package_TO_SOT_SMD:SOT-23",
                sheet=unit.sheet,
                mpn="MCP1700T-3302E/TT",
                sourcing_note="Architecture-selected 3.3 V MCP1700 LDO",
            ),
            BomComponentGroup(
                id="mcp1700_caps",
                reference_prefix="C",
                quantity=2,
                value="1uF",
                symbol="Device:C",
                footprint="Capacitor_SMD:C_0603_1608Metric",
                sheet=unit.sheet,
            ),
        ]
    )
    return groups


def _explicit_r2r_groups(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> list[BomComponentGroup]:
    """Lower an explicitly dimensioned R-2R ladder to two passive groups."""
    architecture = prompt_state.get("architecture") or {}
    target = re.sub(r"[^a-z0-9]", "", unit.sheet.lower())
    topology = " ".join(
        str(value)
        for key, value in (architecture.get("topologies") or {}).items()
        if re.sub(r"[^a-z0-9]", "", str(key).lower()) == target
    )
    sheet_function = " ".join(
        str(row.get("function") or "")
        for row in architecture.get("sheets") or []
        if isinstance(row, dict) and row.get("name") == unit.sheet
    )
    assumptions = " ".join(
        str(value) for value in architecture.get("assumptions") or []
    )
    detail = f"{topology} {sheet_function} {assumptions}"
    bit_match = re.search(
        r"\b(\d+)(?:[- ]bit| digital bits?)\b",
        detail,
        re.IGNORECASE,
    )
    value_match = re.search(
        r"\b(\d+(?:\.\d+)?[km]?)(?:\s*ohm)?\s*(?:/|and)\s*"
        r"(\d+(?:\.\d+)?[km]?)(?:\s*ohm)?\b",
        detail,
        re.IGNORECASE,
    )
    if not re.search(r"\br\s*-\s*2r\b|\br2r\b", detail, re.IGNORECASE):
        return []
    if value_match is None:
        r_value, two_r_value = "10k", "20k"
    else:
        r_value, two_r_value = value_match.groups()
    if bit_match is not None:
        bits = int(bit_match.group(1))
    else:
        word_match = re.search(
            rf"\b({'|'.join(_COUNT_WORDS)})\s+(?:digital\s+)?(?:bits?|inputs?)\b",
            detail,
            re.IGNORECASE,
        )
        bits = _COUNT_WORDS[word_match.group(1).lower()] if word_match else 0
    if not bits:
        bits = sum(
            1
            for net in architecture.get("inter_sheet_nets") or []
            if isinstance(net, dict)
            and any(
                isinstance(endpoint, dict)
                and endpoint.get("sheet") == unit.sheet
                and endpoint.get("direction") == "input"
                for endpoint in net.get("endpoints") or []
            )
        )
    if not 2 <= bits <= 32:
        return []
    return [
        BomComponentGroup(
            id="r2r_series",
            reference_prefix="R",
            quantity=bits - 1,
            value=r_value,
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
            sheet=unit.sheet,
        ),
        BomComponentGroup(
            id="r2r_branches",
            reference_prefix="R",
            quantity=bits + 1,
            value=two_r_value,
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
            sheet=unit.sheet,
        ),
    ]


def deterministic_bom_candidate(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> dict | None:
    """Return a model-free candidate when architecture fully specifies a known block."""
    groups = _explicit_mcp1700_ldo_groups(unit, prompt_state)
    if not groups:
        explicit_interface = _explicit_generic_interface_group(unit, prompt_state)
        if explicit_interface is not None:
            groups = [explicit_interface]
    if not groups:
        groups = _explicit_low_voltage_opamp_groups(unit, prompt_state)
    if not groups:
        groups = _explicit_r2r_groups(unit, prompt_state)
    if not groups:
        return None
    lowered_assumptions = []
    if any(group.id.startswith("r2r_") for group in groups) and not re.search(
        r"\b10k(?:\s*ohm)?\s*(?:/|and)\s*20k",
        " ".join(
            str(value)
            for value in (prompt_state.get("architecture") or {}).get("assumptions") or []
        ),
        re.IGNORECASE,
    ):
        lowered_assumptions.append("R-2R ladder uses 10k/20k resistors (defaulted)")
    return {
        "groups": [group.model_dump(mode="json") for group in groups],
        "arrays": [],
        "assumptions": lowered_assumptions,
        "substitutions": [],
    }


def _validate_bom_unit(unit: StageWorkUnit, payload: dict, prompt_state: dict) -> dict:
    groups = [BomComponentGroup.model_validate(group) for group in (payload.get("groups") or [])]
    assumptions = [str(value) for value in payload.get("assumptions") or []]
    if not groups:
        deterministic = deterministic_bom_candidate(unit, prompt_state)
        if deterministic is not None:
            groups = [
                BomComponentGroup.model_validate(group)
                for group in deterministic["groups"]
            ]
            assumptions.extend(
                str(value) for value in deterministic.get("assumptions") or []
            )
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
        "empty-sheet": [unit.sheet] if not groups and unit.sheet not in recipe_sheets else [],
    }
    if any(defects.values()):
        raise WorkUnitValidationError(unit.unit_id, defects)
    return {
        "groups": [group.model_dump(exclude_none=True) for group in groups],
        "arrays": [array.model_dump(exclude_none=True) for array in arrays],
        "assumptions": list(dict.fromkeys(assumptions)),
        "substitutions": list(payload.get("substitutions") or []),
    }


def _natural_key(value: str) -> tuple:
    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
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


def _deterministic_connector_wiring(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> dict | None:
    bom = prompt_state.get("bom") or {}
    parts = [
        part
        for part in bom.get("parts") or []
        if isinstance(part, dict) and part.get("ref") in unit.refs
    ]
    if len(parts) != 1 or "Connector_Generic:Conn_01x" not in str(parts[0].get("symbol")):
        return None
    ref = str(parts[0]["ref"])
    pins = sorted(
        [pin for pin_ref, pin in unit.expected_pins if pin_ref == ref],
        key=_natural_key,
    )
    architecture = prompt_state.get("architecture") or {}
    signals = sorted(
        [str(net.get("name")) for net in _sheet_signal_nets(unit, architecture)],
        key=_natural_key,
    )
    power = [
        str(net)
        for net in architecture.get("power_nets") or []
        if str(net).upper() != "GND"
    ]
    if "output" in unit.sheet.lower():
        nets = signals + (["GND"] if len(signals) < len(pins) else [])
    else:
        nets = signals + power + (["GND"] if len(signals) + len(power) < len(pins) else [])
    if len(nets) != len(pins):
        return None
    return {
        "pins": [
            {"ref": ref, "pin": pin, "net": net}
            for pin, net in zip(pins, nets, strict=True)
        ]
    }


def _deterministic_opamp_buffer_wiring(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> dict | None:
    bom = prompt_state.get("bom") or {}
    parts = {
        str(part.get("ref")): part
        for part in bom.get("parts") or []
        if isinstance(part, dict) and part.get("ref") in unit.refs
    }
    opamps = [
        ref
        for ref, part in parts.items()
        if str(part.get("symbol")) == "mcp6001:MCP6001T-I_OT"
    ]
    bypass = [
        ref
        for ref, part in parts.items()
        if str(part.get("symbol")) == "Device:C"
    ]
    if len(opamps) != 1 or len(bypass) != 1:
        return None
    architecture = prompt_state.get("architecture") or {}
    sheet_nets = _sheet_signal_nets(unit, architecture)
    inputs = [
        str(net.get("name"))
        for net in sheet_nets
        if any(
            isinstance(endpoint, dict)
            and endpoint.get("sheet") == unit.sheet
            and endpoint.get("direction") == "input"
            for endpoint in net.get("endpoints") or []
        )
    ]
    outputs = [
        str(net.get("name"))
        for net in sheet_nets
        if any(
            isinstance(endpoint, dict)
            and endpoint.get("sheet") == unit.sheet
            and endpoint.get("direction") == "output"
            for endpoint in net.get("endpoints") or []
        )
    ]
    supply = [
        str(net)
        for net in architecture.get("power_nets") or []
        if str(net).upper() != "GND"
    ]
    if len(inputs) != 1 or len(outputs) != 1 or not supply:
        return None
    opamp, cap = opamps[0], bypass[0]
    return {
        "pins": [
            {"ref": opamp, "pin": "1", "net": outputs[0]},
            {"ref": opamp, "pin": "2", "net": "GND"},
            {"ref": opamp, "pin": "3", "net": inputs[0]},
            {"ref": opamp, "pin": "4", "net": outputs[0]},
            {"ref": opamp, "pin": "5", "net": supply[0]},
            {"ref": cap, "pin": "1", "net": supply[0]},
            {"ref": cap, "pin": "2", "net": "GND"},
        ]
    }


def _deterministic_r2r_wiring(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> dict | None:
    bom = prompt_state.get("bom") or {}
    parts = [
        part
        for part in bom.get("parts") or []
        if isinstance(part, dict)
        and part.get("ref") in unit.refs
        and part.get("symbol") == "Device:R"
    ]
    series = sorted(
        [str(part["ref"]) for part in parts if str(part.get("value")).lower() == "10k"],
        key=_natural_key,
    )
    branches = sorted(
        [str(part["ref"]) for part in parts if str(part.get("value")).lower() == "20k"],
        key=_natural_key,
    )
    bits = len(branches) - 1
    if bits < 2 or len(series) != bits - 1:
        return None
    architecture = prompt_state.get("architecture") or {}
    sheet_nets = _sheet_signal_nets(unit, architecture)
    inputs = sorted(
        [
            str(net.get("name"))
            for net in sheet_nets
            if any(
                isinstance(endpoint, dict)
                and endpoint.get("sheet") == unit.sheet
                and endpoint.get("direction") == "input"
                for endpoint in net.get("endpoints") or []
            )
        ],
        key=_natural_key,
    )
    outputs = [
        str(net.get("name"))
        for net in sheet_nets
        if any(
            isinstance(endpoint, dict)
            and endpoint.get("sheet") == unit.sheet
            and endpoint.get("direction") == "output"
            for endpoint in net.get("endpoints") or []
        )
    ]
    if len(inputs) != bits or len(outputs) != 1:
        return None
    nodes = [outputs[0], *(f"R2R_N{index}" for index in range(1, bits))]
    pins: list[dict] = []
    for index, ref in enumerate(series):
        pins.extend(
            [
                {"ref": ref, "pin": "1", "net": nodes[index]},
                {"ref": ref, "pin": "2", "net": nodes[index + 1]},
            ]
        )
    for index, ref in enumerate(branches[:-1]):
        pins.extend(
            [
                {"ref": ref, "pin": "1", "net": inputs[-(index + 1)]},
                {"ref": ref, "pin": "2", "net": nodes[index]},
            ]
        )
    pins.extend(
        [
            {"ref": branches[-1], "pin": "1", "net": nodes[-1]},
            {"ref": branches[-1], "pin": "2", "net": "GND"},
        ]
    )
    return {"pins": pins}


def deterministic_wiring_candidate(
    unit: StageWorkUnit,
    prompt_state: dict,
    extras: dict,
) -> dict | None:
    """Return validated model-free wiring for fully known simple blocks."""
    for lower in (
        _deterministic_connector_wiring,
        _deterministic_opamp_buffer_wiring,
        _deterministic_r2r_wiring,
    ):
        candidate = lower(unit, prompt_state)
        if candidate is None:
            continue
        try:
            return _validate_wiring_unit(unit, candidate, prompt_state, extras)
        except (WorkUnitValidationError, TypeError, ValueError):
            return None
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
