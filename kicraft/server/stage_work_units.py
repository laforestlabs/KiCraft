"""Pure work-unit planning, validation, merging, and draft persistence."""

from __future__ import annotations
import hashlib
from functools import lru_cache
import json
import re

from dataclasses import asdict, dataclass
from typing import Literal
from pathlib import Path

from kicraft.fsutil import atomic_write_text
from kicraft.design.recipes import locked_no_connect_pins, locked_pin_assignments
from kicraft.design.synthesis.symbol_pinout import canonical_symbol_id
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
            if isinstance(requirement, dict) and str(requirement.get("role")) != "power_input"
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
            recipe_part_roles_by_sheet: dict[str, set[str]] = {}
            from kicraft.design.recipes import get_recipe

            for selection in architecture.get("recipe_selections") or []:
                if not isinstance(selection, dict):
                    continue
                definition = get_recipe(str(selection.get("recipe")))
                sheet_bindings = selection.get("sheets") or {}
                for group in definition.parts:
                    bound_sheet = sheet_bindings.get(group.sheet_role)
                    if bound_sheet:
                        recipe_part_roles_by_sheet.setdefault(str(bound_sheet), set()).add(
                            _identity_token(group.role)
                        )

            from kicraft.design.lowering import lower_requirement
            from kicraft.design.models import CircuitRequirement

            # A requirement gets its own bounded unit only when a deterministic
            # lowerer can resolve it. A plain requirement (made-up family, no
            # lowerer) forced the model to invent parts from the requirement
            # alone, which it could not: those units came back empty and
            # exhausted the stage. Plain requirements roll up into ONE
            # sheet-scoped unit per sheet, where the model sees the sheet's
            # function text and can emit a coherent part set.
            combined_connector_sheets = {
                str(sheet.get("name"))
                for sheet in architecture.get("sheets") or []
                if isinstance(sheet, dict)
                and (
                    "fpc" in str(sheet.get("function") or "").lower()
                    or "ffc" in str(sheet.get("function") or "").lower()
                )
                and "header" in str(sheet.get("function") or "").lower()
            }
            determinable_ids: set[str] = set()
            for requirement in requirements:
                if str(requirement.get("id")) in selected:
                    continue
                if str(requirement.get("sheet")) in combined_connector_sheets:
                    continue
                try:
                    if (
                        lower_requirement(CircuitRequirement.model_validate(requirement))
                        is not None
                    ):
                        determinable_ids.add(str(requirement["id"]))
                except (TypeError, ValueError):
                    continue

            units: list[StageWorkUnit] = []
            for requirement in requirements:
                if str(requirement.get("id")) not in determinable_ids:
                    continue
                sheet = str(requirement["sheet"])
                units.append(
                    StageWorkUnit(
                        unit_id=f"bom-r{len(units):03d}",
                        stage="bom",
                        sheet=sheet,
                        requirement_ids=(str(requirement["id"]),),
                        owned_roles=(str(requirement["role"]),),
                        planned_resolution_source="llm",
                        recipe_ids=tuple(sorted(recipe_ids_by_sheet.get(sheet, set()))),
                    )
                )

            plain_by_sheet: dict[str, list[dict]] = {}
            for requirement in requirements:
                requirement_id = str(requirement.get("id"))
                requirement_tokens = {
                    _identity_token(requirement.get("role")),
                    _identity_token(requirement.get("family")),
                }
                if requirement_tokens & recipe_part_roles_by_sheet.get(
                    str(requirement.get("sheet")), set()
                ):
                    continue
                if requirement_id in selected or requirement_id in determinable_ids:
                    continue
                plain_by_sheet.setdefault(str(requirement["sheet"]), []).append(requirement)

            for sheet in sheets:
                sheet_requirements = plain_by_sheet.get(sheet, [])
                if sheet_requirements:
                    units.append(
                        StageWorkUnit(
                            unit_id=f"bom-s{len(units):03d}",
                            stage="bom",
                            sheet=sheet,
                            requirement_ids=tuple(str(req["id"]) for req in sheet_requirements),
                            owned_roles=tuple(
                                dict.fromkeys(str(req["role"]) for req in sheet_requirements)
                            ),
                            planned_resolution_source="llm",
                            recipe_ids=tuple(sorted(recipe_ids_by_sheet.get(sheet, set()))),
                        )
                    )
                elif sheet not in recipe_ids_by_sheet and not any(
                    unit.sheet == sheet for unit in units
                ):
                    units.append(
                        StageWorkUnit(
                            unit_id=f"bom-s{len(units):03d}",
                            stage="bom",
                            sheet=sheet,
                        )
                    )
            return tuple(units)
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


_HIROSE_FH12_05_PIN_COUNTS = {
    6,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    24,
    25,
    26,
    28,
    30,
    32,
    33,
    34,
    35,
    36,
    40,
    45,
    50,
    53,
}


def _standard_connector_sheet_candidate(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> dict | None:
    """Lower mechanically standard connector-only sheets without an LLM."""
    sheets = (prompt_state.get("architecture") or {}).get("sheets") or []
    target = next(
        (
            sheet
            for sheet in sheets
            if isinstance(sheet, dict) and str(sheet.get("name")) == unit.sheet
        ),
        None,
    )
    function = str((target or {}).get("function") or "")
    identity = function.lower()
    has_usb_c = bool(re.search(r"\busb(?:\s*|-)?c\b|\btype(?:\s*|-)?c\b", identity))
    if has_usb_c:
        from kicraft.design.recipes import get_recipe

        recipe_name = (
            "usb-c-usb2-device@1"
            if any(token in identity for token in ("data", "usb2", "usb 2", "esd"))
            else "usb-c-5v-sink@1"
        )
        definition = get_recipe(recipe_name)
        return {
            "groups": [
                {
                    "id": group.role,
                    "reference_prefix": group.reference_prefix,
                    "quantity": group.quantity,
                    "value": group.value,
                    "symbol": group.symbol,
                    "footprint": group.footprint,
                    "sheet": unit.sheet,
                    **({"mpn": group.mpn} if group.mpn else {}),
                }
                for group in definition.parts
            ],
            "arrays": [],
            "assumptions": [
                f"Used deterministic {definition.recipe} parts for the standard USB-C interface"
            ],
            "substitutions": [],
        }

    has_usb_a = bool(re.search(r"\busb(?:\s*|-)?a\b", identity))
    if "bnc" in identity and not any(token in identity for token in ("filter", "low-pass")):
        return {
            "groups": [
                {
                    "id": "bnc",
                    "reference_prefix": "J",
                    "quantity": 1,
                    "value": "BNC",
                    "symbol": "bnc-pcb-jack:KH-BNC50-3511",
                    "footprint": "bnc-pcb-jack:ANT-TH_KH-BNC50-3511",
                    "sheet": unit.sheet,
                    "mpn": "KH-BNC50-3511",
                }
            ],
            "arrays": [],
            "assumptions": ["Selected the curated 50 ohm PCB BNC jack (defaulted)"],
            "substitutions": [],
        }

    if has_usb_a:
        groups = [
            {
                "id": "usb_a_receptacle",
                "reference_prefix": "J",
                "quantity": 1,
                "value": "USB-A receptacle",
                "symbol": "Connector_Generic:Conn_01x04",
                "footprint": "Connector_USB:USB_A_Molex_67643_Horizontal",
                "sheet": unit.sheet,
            }
        ]
        if "current" in identity and ("limit" in identity or "switch" in identity):
            groups.extend(
                [
                    {
                        "id": "current_limit_switch",
                        "reference_prefix": "U",
                        "quantity": 1,
                        "value": "TPS2041B",
                        "symbol": "Power_Management:TPS2041B",
                        "footprint": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                        "sheet": unit.sheet,
                    },
                    {
                        "id": "port_decoupling",
                        "reference_prefix": "C",
                        "quantity": 2,
                        "value": "1uF",
                        "symbol": "Device:C",
                        "footprint": "Capacitor_SMD:C_0603_1608Metric",
                        "sheet": unit.sheet,
                    },
                ]
            )
        if "led" in identity or "status" in identity:
            groups.extend(
                [
                    {
                        "id": "status_led",
                        "reference_prefix": "D",
                        "quantity": 1,
                        "value": "Green",
                        "symbol": "Device:LED",
                        "footprint": "LED_SMD:LED_0603_1608Metric",
                        "sheet": unit.sheet,
                    },
                    {
                        "id": "status_led_resistor",
                        "reference_prefix": "R",
                        "quantity": 1,
                        "value": "1k",
                        "symbol": "Device:R",
                        "footprint": "Resistor_SMD:R_0603_1608Metric",
                        "sheet": unit.sheet,
                    },
                ]
            )
        return {
            "groups": groups,
            "arrays": [],
            "assumptions": ["Used a stock USB-A receptacle and bounded port-power circuit"],
            "substitutions": [],
        }

    architecture_requirements = [
        row
        for row in (prompt_state.get("architecture") or {}).get("requirements") or []
        if isinstance(row, dict) and str(row.get("id")) in set(unit.requirement_ids)
    ]
    if "current" in identity and ("limit" in identity or "switch" in identity):
        return {
            "groups": [
                {
                    "id": "current_limit_switch",
                    "reference_prefix": "U",
                    "quantity": 1,
                    "value": "TPS2041B",
                    "symbol": "Power_Management:TPS2041B",
                    "footprint": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                    "sheet": unit.sheet,
                },
                {
                    "id": "switch_decoupling",
                    "reference_prefix": "C",
                    "quantity": 2,
                    "value": "1uF",
                    "symbol": "Device:C",
                    "footprint": "Capacitor_SMD:C_0603_1608Metric",
                    "sheet": unit.sheet,
                },
            ],
            "arrays": [],
            "assumptions": ["Used a stock current-limited USB power switch (defaulted)"],
            "substitutions": [],
        }
    if "led" in identity:
        quantity = 3 if "leds" in identity else 1
        return {
            "groups": [
                {
                    "id": "status_led",
                    "reference_prefix": "D",
                    "quantity": quantity,
                    "value": "Green",
                    "symbol": "Device:LED",
                    "footprint": "LED_SMD:LED_0603_1608Metric",
                    "sheet": unit.sheet,
                },
                {
                    "id": "status_led_resistor",
                    "reference_prefix": "R",
                    "quantity": quantity,
                    "value": "1k",
                    "symbol": "Device:R",
                    "footprint": "Resistor_SMD:R_0603_1608Metric",
                    "sheet": unit.sheet,
                },
            ],
            "arrays": [],
            "assumptions": ["Used one current-limiting resistor per status LED (defaulted)"],
            "substitutions": [],
        }
    if (
        ("two-pin" in identity or "2-pin" in identity)
        and "power" in identity
        and "input" in identity
    ):
        return {
            "groups": [
                {
                    "id": "power_input",
                    "reference_prefix": "J",
                    "quantity": 1,
                    "value": "Power input",
                    "symbol": "Connector_Generic:Conn_01x02",
                    "footprint": (
                        "TerminalBlock_Phoenix:PhoenixContact_MKDS-1,5-2_1x02_P5.00mm_Horizontal"
                    ),
                    "sheet": unit.sheet,
                }
            ],
            "arrays": [],
            "assumptions": ["Used a stock two-pin power terminal (defaulted)"],
            "substitutions": [],
        }
    requirement_families = {_identity_token(row.get("family")) for row in architecture_requirements}
    if "qspiflash" in requirement_families or "qspi" in identity and "flash" in identity:
        return {
            "groups": [
                {
                    "id": "qspi_flash",
                    "reference_prefix": "U",
                    "quantity": 1,
                    "value": "W25Q16JVSS",
                    "symbol": "Memory_Flash:W25Q16JVSS",
                    "footprint": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                    "sheet": unit.sheet,
                }
            ],
            "arrays": [],
            "assumptions": ["Selected a stock 16-Mbit QSPI NOR flash (defaulted)"],
            "substitutions": [],
        }
    if "ldo" in requirement_families or "ldo" in identity:
        return {
            "groups": [
                {
                    "id": "ldo_3v3",
                    "reference_prefix": "U",
                    "quantity": 1,
                    "value": "AP1117-33",
                    "symbol": "Regulator_Linear:AP1117-33",
                    "footprint": "Package_TO_SOT_SMD:SOT-223-3_TabPin2",
                    "sheet": unit.sheet,
                },
                {
                    "id": "ldo_caps",
                    "reference_prefix": "C",
                    "quantity": 2,
                    "value": "10uF",
                    "symbol": "Device:C",
                    "footprint": "Capacitor_SMD:C_0805_2012Metric",
                    "sheet": unit.sheet,
                },
            ],
            "arrays": [],
            "assumptions": ["Selected a stock fixed 3.3 V LDO implementation (defaulted)"],
            "substitutions": [],
        }
    gpio_requirement = next(
        (
            row
            for row in architecture_requirements
            if _identity_token(row.get("family")) == "gpioheader"
        ),
        None,
    )
    if gpio_requirement is not None:
        count = int((gpio_requirement.get("parameters") or {}).get("pins") or 0)
        if 2 <= count <= 50 and count % 2 == 0:
            rows = count // 2
            return {
                "groups": [
                    {
                        "id": "gpio_header",
                        "reference_prefix": "J",
                        "quantity": 1,
                        "value": f"GPIO 2x{rows}",
                        "symbol": f"Connector_Generic:Conn_02x{rows:02d}_Odd_Even",
                        "footprint": (
                            f"Connector_PinHeader_2.54mm:PinHeader_2x{rows:02d}_P2.54mm_Vertical"
                        ),
                        "sheet": unit.sheet,
                    }
                ],
                "arrays": [],
                "assumptions": ["Used a stock two-row GPIO breakout connector (defaulted)"],
                "substitutions": [],
            }

    count_match = re.search(r"(?<![0-9])(\d{1,2})\s*(?:-|_|\s)?(?:pin|signals?)", identity)
    if count_match is None:
        count = sum(
            1
            for net in (prompt_state.get("architecture") or {}).get("inter_sheet_nets") or []
            if isinstance(net, dict)
            and any(
                isinstance(endpoint, dict) and endpoint.get("sheet") == unit.sheet
                for endpoint in net.get("endpoints") or []
            )
        )
    else:
        count = int(count_match.group(1))
    if not 1 <= count <= 53:
        return None
    has_fpc = "fpc" in identity or "ffc" in identity
    has_header = "header" in identity or "0.1-inch" in identity or "2.54mm" in identity
    groups: list[dict] = []
    if (
        has_fpc
        and count in _HIROSE_FH12_05_PIN_COUNTS
        and ("0.5" in identity or "0.50" in identity)
    ):
        groups.append(
            {
                "id": "fpc",
                "reference_prefix": "J",
                "quantity": 1,
                "value": f"FPC_{count:02d}_0.5mm",
                "symbol": f"Connector_Generic:Conn_01x{count:02d}",
                "footprint": (
                    "Connector_FFC-FPC:"
                    f"Hirose_FH12-{count}S-0.5SH_1x{count:02d}-1MP_P0.50mm_Horizontal"
                ),
                "sheet": unit.sheet,
            }
        )
    if has_header and count <= 50:
        groups.append(
            {
                "id": "header",
                "reference_prefix": "J",
                "quantity": 1,
                "value": f"Header_1x{count:02d}_2.54mm",
                "symbol": f"Connector_Generic:Conn_01x{count:02d}",
                "footprint": (
                    f"Connector_PinHeader_2.54mm:PinHeader_1x{count:02d}_P2.54mm_Vertical"
                ),
                "sheet": unit.sheet,
            }
        )
    if not groups:
        return None
    return {
        "groups": groups,
        "arrays": [],
        "assumptions": ["Used stock KiCad footprints for standard connectors (defaulted)"],
        "substitutions": [],
    }


def deterministic_bom_candidate(
    unit: StageWorkUnit,
    prompt_state: dict,
) -> dict | None:
    """Return deterministic BOM groups from a typed lowerer or standard sheet."""
    from kicraft.design.lowering import lower_requirement

    requirement = _unit_requirement(unit, prompt_state)
    if requirement is not None:
        artifact = lower_requirement(requirement)
        if artifact is not None:
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
                "_lowering_roles": {
                    group.role: {"lowering_role": group.role} for group in artifact.groups
                },
                "_calculations": [
                    calculation.model_dump(mode="json") for calculation in artifact.calculations
                ],
            }
    return _standard_connector_sheet_candidate(unit, prompt_state)


def _identity_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


@lru_cache(maxsize=1)
def _curated_part_indexes() -> tuple[dict[str, object], dict[str, object]]:
    from kicraft.parts_library import load_all_with_overrides

    active, _shadowed, _broken = load_all_with_overrides(project_root=None)
    by_name = {part.manifest.name: part for part in active}
    by_mpn: dict[str, object] = {}
    for part in active:
        by_mpn.setdefault(_identity_token(part.manifest.mpn), part)
    return by_name, by_mpn


def _normalize_curated_group_identities(
    groups: list[BomComponentGroup],
) -> list[BomComponentGroup]:
    """Use curated bundles as authoritative, reusable component defaults."""
    by_name, by_mpn = _curated_part_indexes()
    normalized: list[BomComponentGroup] = []
    for group in groups:
        original_symbol = group.symbol
        group = group.model_copy(update={"symbol": canonical_symbol_id(original_symbol)})
        selected_identities = {
            _identity_token(value) for value in (group.mpn, group.value) if value
        }
        library = group.symbol.partition(":")[0]
        loaded = by_name.get(library)
        if loaded is None:
            loaded = next(
                (by_mpn[identity] for identity in selected_identities if identity in by_mpn),
                None,
            )
        if loaded is None:
            loaded = next(
                (
                    part
                    for identity in selected_identities
                    for mpn, part in by_mpn.items()
                    if mpn and mpn in identity
                ),
                None,
            )
        if loaded is None and original_symbol.lower().startswith("potentiometer:"):
            loaded = by_name.get("trim-pot-3296w-10k")
        identity_text = " ".join(
            str(value or "") for value in (group.id, group.value, group.symbol, group.footprint)
        ).lower()
        if loaded is None and "bnc" in identity_text:
            loaded = by_name.get("bnc-pcb-jack")
        if loaded is None and "screw" in identity_text and "terminal" in identity_text:
            loaded = by_name.get("screw-terminal-5mm-2p")
        if loaded is None and group.reference_prefix in {"J", "P"}:
            # Models often invent a library namespace for mechanically standard
            # connectors instead of using KiCad's stock generic symbol plus a
            # concrete stock footprint. Canonicalize only obvious invented
            # identities; preserve already-valid Connector_* selections.
            stock_connector_library = library.startswith("Connector")
            if not stock_connector_library and ("fpc" in identity_text or "ffc" in identity_text):
                pin_match = re.search(r"(?<![0-9])(\d{1,2})(?:pin|p)(?![0-9])", identity_text)
                if (
                    pin_match
                    and int(pin_match.group(1)) == 24
                    and ("0.5" in identity_text or "05mm" in _identity_token(identity_text))
                ):
                    group = group.model_copy(
                        update={
                            "symbol": "Connector_Generic:Conn_01x24",
                            "footprint": (
                                "Connector_FFC-FPC:"
                                "Hirose_FH12-24S-0.5SH_1x24-1MP_P0.50mm_Horizontal"
                            ),
                        }
                    )
                    normalized.append(group)
                    continue
            if not stock_connector_library and "header" in identity_text:
                pin_match = re.search(r"1x(\d{1,2})(?![0-9])", identity_text)
                if (
                    pin_match
                    and 1 <= int(pin_match.group(1)) <= 50
                    and ("2.54" in identity_text or "254mm" in _identity_token(identity_text))
                ):
                    pin_count = int(pin_match.group(1))
                    group = group.model_copy(
                        update={
                            "symbol": f"Connector_Generic:Conn_01x{pin_count:02d}",
                            "footprint": (
                                "Connector_PinHeader_2.54mm:"
                                f"PinHeader_1x{pin_count:02d}_P2.54mm_Vertical"
                            ),
                        }
                    )
                    normalized.append(group)
                    continue
        manifest = getattr(loaded, "manifest", None)
        if manifest is None:
            normalized.append(group)
            continue
        update = {
            "symbol": f"{manifest.name}:{manifest.symbol_name}",
            "footprint": f"{manifest.name}:{manifest.footprint_name}",
            "mpn": manifest.mpn,
        }
        lcsc = (manifest.sourcing or {}).get("lcsc")
        if lcsc:
            update["sourcing_note"] = f"LCSC {lcsc}"
        normalized.append(group.model_copy(update=update))
    return normalized


def _validate_bom_unit(unit: StageWorkUnit, payload: dict, prompt_state: dict) -> dict:
    groups = _normalize_curated_group_identities(
        [BomComponentGroup.model_validate(group) for group in (payload.get("groups") or [])]
    )
    assumptions = [str(value) for value in payload.get("assumptions") or []]
    used_deterministic_candidate = payload.get("_trusted_deterministic_candidate") is True
    lowering_metadata = {
        key: value
        for key, value in payload.items()
        if str(key).startswith("_") and key != "_trusted_deterministic_candidate"
    }
    if not groups:
        deterministic = deterministic_bom_candidate(unit, prompt_state)
        if deterministic is not None:
            groups = [BomComponentGroup.model_validate(group) for group in deterministic["groups"]]
            used_deterministic_candidate = True
            assumptions.extend(str(value) for value in deterministic.get("assumptions") or [])
            lowering_metadata = {
                key: value
                for key, value in deterministic.items()
                if key.startswith("_") and key != "_trusted_deterministic_candidate"
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
    recipe_identities = {
        (part.sheet, part.symbol.lower(), part.value.lower()) for part in recipe_parts
    }
    from kicraft.design.recipes import protected_identity_matches

    architecture_requirements = [
        row for row in architecture.get("requirements") or [] if isinstance(row, dict)
    ]
    unit_requirement_ids = set(unit.requirement_ids)
    unit_requirements = tuple(
        row for row in architecture_requirements if str(row.get("id")) in unit_requirement_ids
    )
    requirement = unit_requirements[0] if len(unit_requirements) == 1 else None
    requirement_id = (
        requirement.get("id") if isinstance(requirement, dict) else getattr(requirement, "id", None)
    )
    requirement_family = (
        requirement.get("family")
        if isinstance(requirement, dict)
        else getattr(requirement, "family", None)
    )
    requirement_role = (
        requirement.get("role")
        if isinstance(requirement, dict)
        else getattr(requirement, "role", None)
    )
    connector_prefixes = {"J", "P"}
    # Ownership pruning only makes sense for a bounded single-requirement unit.
    # A sheet-scoped unit legitimately owns a mix of connector and circuit
    # roles, so pruning its J/P groups would silently drop owned connectors.
    target_sheet = next(
        (
            sheet
            for sheet in architecture.get("sheets") or []
            if isinstance(sheet, dict) and sheet.get("name") == unit.sheet
        ),
        {},
    )
    target_identity = str(target_sheet.get("function") or "").lower()
    target_owns_connector = bool(
        re.search(r"\b(?:connector|receptacle|header|terminal|usb|bnc|fpc|ffc)\b", target_identity)
    )
    if len(unit.requirement_ids) == 1:
        if requirement_role == "connector":
            owned_group_ids = {
                group.id for group in groups if group.reference_prefix in connector_prefixes
            }
            if owned_group_ids:
                groups = [group for group in groups if group.id in owned_group_ids]
        elif (
            any(row.get("role") == "connector" for row in architecture_requirements)
            and not target_owns_connector
        ):
            non_connector_groups = [
                group for group in groups if group.reference_prefix not in connector_prefixes
            ]
            if non_connector_groups:
                groups = non_connector_groups
                retained_group_ids = {group.id for group in groups}
                arrays = [array for array in arrays if array.group_id in retained_group_ids]
    group_ids = [group.id for group in groups]
    array_group_ids = [array.group_id for array in arrays]
    requirement_identities = " ".join(
        str(value or "") for value in (requirement_id, requirement_family)
    )
    requirement_exact_part = (
        requirement.get("exact_part")
        if isinstance(requirement, dict)
        else getattr(requirement, "exact_part", None)
    )
    requires_named_implementation = (
        requirement is not None
        and not lowering_metadata
        and (
            bool(requirement_exact_part)
            or "controller" in re.split(r"[^a-z0-9]+", requirement_identities.lower())
        )
    )
    protected_groups = [
        group.id
        for group in groups
        if protected_identity_matches(group.id, group.symbol, group.value, group.mpn)
        and not used_deterministic_candidate
        and not _requirement_owns_protected_group(group, unit_requirements)
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
            if (group.sheet, group.symbol.lower(), group.value.lower()) in recipe_identities
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


def _standard_connector_wiring_candidate(
    unit: StageWorkUnit,
    prompt_state: dict,
    extras: dict,
) -> dict | None:
    if len(unit.refs) != 1 or not unit.expected_pins:
        return None
    ref = unit.refs[0]
    part = next(
        (
            row
            for row in (prompt_state.get("bom") or {}).get("parts") or []
            if isinstance(row, dict) and str(row.get("ref")) == ref
        ),
        None,
    )
    if part is None:
        return None
    symbol = str(part.get("symbol") or "").lower()
    architecture = prompt_state.get("architecture") or {}
    nets = [
        str(net.get("name"))
        for net in architecture.get("inter_sheet_nets") or []
        if isinstance(net, dict)
        and any(
            isinstance(endpoint, dict) and endpoint.get("sheet") == unit.sheet
            for endpoint in net.get("endpoints") or []
        )
    ]
    pins = sorted(
        (pin for owned_ref, pin in unit.expected_pins if owned_ref == ref),
        key=_natural_key,
    )
    if not nets and "connector_generic:conn_01x" in symbol:
        sheet_function = next(
            (
                str(sheet.get("function") or "").lower()
                for sheet in architecture.get("sheets") or []
                if isinstance(sheet, dict) and sheet.get("name") == unit.sheet
            ),
            "",
        )
        if ("fpc" in sheet_function or "ffc" in sheet_function) and "header" in sheet_function:
            nets = [f"SIG{index}" for index in range(1, len(pins) + 1)]
    candidate_pins: list[dict] = []
    if "connector_generic:conn_01x" in symbol and len(nets) == len(pins):
        candidate_pins = [
            {"ref": ref, "pin": pin, "net": net} for pin, net in zip(pins, nets, strict=True)
        ]
    elif "bnc" in symbol:
        signal = next((net for net in nets if net.upper() != "GND"), None)
        if signal is not None and "GND" in nets and "1" in pins:
            candidate_pins = [
                {"ref": ref, "pin": pin, "net": signal if pin == "1" else "GND"} for pin in pins
            ]
    if not candidate_pins:
        return None
    try:
        return _validate_wiring_unit(
            unit,
            {"pins": candidate_pins},
            prompt_state,
            extras,
        )
    except (WorkUnitValidationError, TypeError, ValueError):
        return None


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
        return _standard_connector_wiring_candidate(unit, prompt_state, extras)
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
        return _standard_connector_wiring_candidate(unit, prompt_state, extras)
    artifact = lower_requirement(CircuitRequirement.model_validate(requirement_row))
    if artifact is None or artifact.lowerer_id != next(iter(lowerer_ids)):
        return _standard_connector_wiring_candidate(unit, prompt_state, extras)
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
        return _standard_connector_wiring_candidate(unit, prompt_state, extras)


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
