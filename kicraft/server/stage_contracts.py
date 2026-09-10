"""Per-invocation stage response contracts and response normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kicraft.design import models

from .config import (
    BOM_ARRAY_LIMIT,
    BOM_GROUP_LIMIT,
    BOM_NOTE_LIMIT,
    BOM_SHEET_PART_LIMIT,
    BOM_TOTAL_PART_LIMIT,
)

# Canonical stage -> slot model, mirroring cli_app._apply_slot's owned-field map.
SLOT_MODEL = {
    "intent": models.IntentSlot,
    "functional_spec": models.FunctionalSpec,
    "architecture": models.Architecture,
    "bom": models.BOM,
}
# wiring is not a standalone slot model: it sets bom.connections + bom.no_connect_pins.
SUPPORTED_STAGES = (*SLOT_MODEL.keys(), "wiring")
# Full design order from a brief to a synthesizable state.
DESIGN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")


class ConnectedPinAssignment(BaseModel):
    """Final connected state for one component pin."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    pin: str
    net: str = Field(min_length=1)


class NoConnectPinAssignment(BaseModel):
    """Final intentionally-unconnected state for one component pin."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    pin: str
    no_connect: Literal[True]


PinAssignment = Annotated[
    ConnectedPinAssignment | NoConnectPinAssignment,
    Field(union_mode="left_to_right"),
]


class WiringStageResponse(BaseModel):
    """Model-facing wiring contract: one final assignment per pin."""

    model_config = ConfigDict(extra="forbid")

    pins: list[PinAssignment] = Field(max_length=5000)

    @model_validator(mode="after")
    def _pins_unique(self):
        seen: set[tuple[str, str]] = set()
        for assignment in self.pins:
            key = (assignment.ref, assignment.pin)
            if key in seen:
                raise ValueError(f"duplicate pin assignment {assignment.ref}.{assignment.pin}")
            seen.add(key)
        return self


class InterSheetNetRange(BaseModel):
    """Model-facing compact numeric range expanded before canonical validation."""

    model_config = ConfigDict(extra="forbid")

    name_pattern: str = Field(pattern=r"^[^{}]*\{n\}[^{}]*$")
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    endpoints: list[models.SheetPin] = Field(min_length=2)

    @model_validator(mode="after")
    def _ordered_and_bounded(self):
        if self.end < self.start:
            raise ValueError("inter-sheet net range end must be >= start")
        if self.end - self.start + 1 > 5000:
            raise ValueError("inter-sheet net range expansion exceeds 5000 nets")
        return self


class ArchitectureStageResponse(models.Architecture):
    model_config = ConfigDict(extra="forbid")

    inter_sheet_net_ranges: list[InterSheetNetRange] = Field(default_factory=list)


class IntentStageResponse(models.IntentSlot):
    model_config = ConfigDict(extra="forbid")

    project_stem: str = Field(pattern=r"^[A-Z0-9_]{1,32}$")


class StageQuestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[models.Question] = Field(min_length=1, max_length=5)


class BomComponentGroup(BaseModel):
    """One component type expanded into deterministic references."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,71}$")
    reference_prefix: str = Field(pattern=r"^[A-Z]+$")
    quantity: int = Field(ge=1, le=BOM_SHEET_PART_LIMIT)
    value: str
    symbol: str = Field(pattern=r"^[A-Za-z0-9_.+-]+:[A-Za-z0-9_.+-]+$")
    footprint: str = Field(pattern=r"^[A-Za-z0-9_.+-]+:[A-Za-z0-9_.,+-]+$")
    sheet: str
    mpn: str | None = None
    datasheet: str | None = None
    sourcing_note: str | None = None
    side: Literal["front", "back"] | None = None


def _requirement_owns_protected_group(group: BomComponentGroup, requirements) -> bool:
    """Whether a requirement-scoped group id names its implementation."""

    def value(requirement, field: str):
        if isinstance(requirement, dict):
            return requirement.get(field)
        return getattr(requirement, field, None)

    def token(raw: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(raw).lower())

    def words(raw: object) -> set[str]:
        aliases = {
            "header": "connector",
            "receptacle": "connector",
            "socket": "connector",
            "terminal": "connector",
            "jumper": "switch",
        }
        return {
            aliases.get(word, word)
            for word in re.split(r"[^a-z0-9]+", str(raw).lower())
            if len(word) >= 2
        }

    group_token = token(group.id)
    group_words = words(group.id)
    for requirement in requirements or ():
        for raw in (
            value(requirement, "id"),
            value(requirement, "family"),
            value(requirement, "exact_part"),
        ):
            if not raw:
                continue
            requirement_token = token(raw)
            if requirement_token and requirement_token in group_token:
                return True
            if len(group_words & words(raw)) >= 2:
                return True
    return False


class BomArrayGroup(BaseModel):
    """Placement pattern covering every member of one component group."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    pattern: Literal["grid", "ring"] = "grid"
    rows: int | None = Field(default=None, gt=0)
    cols: int | None = Field(default=None, gt=0)
    pitch_mm: float | None = Field(default=None, gt=0)
    serpentine: bool = True
    radius_mm: float | None = Field(default=None, gt=0)
    start_angle_deg: float = 0.0


class BomStageResponse(BaseModel):
    """Model-facing BOM contract: exactly one group-first representation."""

    model_config = ConfigDict(extra="forbid")

    groups: list[BomComponentGroup] = Field(default_factory=list, max_length=BOM_GROUP_LIMIT)
    arrays: list[BomArrayGroup] = Field(default_factory=list, max_length=BOM_ARRAY_LIMIT)
    assumptions: list[str] = Field(default_factory=list, max_length=BOM_NOTE_LIMIT)
    substitutions: list[models.Substitution] = Field(
        default_factory=list,
        max_length=BOM_NOTE_LIMIT,
    )

    @model_validator(mode="after")
    def _ids_unique(self):
        ids = [group.id for group in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("BOM group ids must be unique")
        array_groups = [array.group_id for array in self.arrays]
        if len(array_groups) != len(set(array_groups)):
            raise ValueError("a BOM group may appear in at most one array")
        unknown = set(array_groups) - set(ids)
        if unknown:
            raise ValueError(f"BOM arrays reference unknown groups: {sorted(unknown)}")
        return self


def _expand_bom_groups(
    payload: dict, prompt_state: dict | None = None
) -> tuple[
    BomStageResponse,
    list[models.BomPart],
    list[models.ArraySpec],
    dict[str, list[str]],
    list,
]:
    """Validate and deterministically allocate recipe and model-authored BOM groups."""
    response = BomStageResponse.model_validate(payload)
    total = sum(group.quantity for group in response.groups)
    if total > BOM_TOTAL_PART_LIMIT:
        raise ValueError(f"BOM has {total} parts; maximum is {BOM_TOTAL_PART_LIMIT}")

    from kicraft.design.recipes import (
        expand_selections,
        protected_identity_matches,
    )
    from kicraft.design.recipes.registry import next_reference_numbers

    architecture = (prompt_state or {}).get("architecture") or {}
    expansions = expand_selections(architecture.get("recipe_selections") or [])
    recipe_parts = [part for expansion in expansions for part in expansion.parts]
    if not response.groups and not recipe_parts:
        raise ValueError("BOM must contain at least one component group or circuit-recipe part")
    recipe_identities = {
        (part.sheet, part.symbol.lower(), part.value.lower()) for part in recipe_parts
    }
    requirements = architecture.get("requirements") or []
    for group in response.groups:
        protected = protected_identity_matches(
            group.id,
            group.symbol,
            group.value,
            group.mpn,
        )
        if protected and not _requirement_owns_protected_group(group, requirements):
            raise ValueError(
                "model_authored_protected_identity: "
                f"BOM group {group.id!r} contains {list(protected)!r}"
            )
        if (group.sheet, group.symbol.lower(), group.value.lower()) in recipe_identities:
            raise ValueError(f"BOM group {group.id!r} duplicates a locked circuit-recipe role")

    per_sheet: dict[str, int] = {}
    next_number = next_reference_numbers(recipe_parts)
    refs_by_group: dict[str, list[str]] = {}
    parts: list[models.BomPart] = list(recipe_parts)
    for group in response.groups:
        per_sheet[group.sheet] = per_sheet.get(group.sheet, 0) + group.quantity
        start = next_number.get(group.reference_prefix, 1)
        refs = [
            f"{group.reference_prefix}{number}" for number in range(start, start + group.quantity)
        ]
        next_number[group.reference_prefix] = start + group.quantity
        refs_by_group[group.id] = refs
        shared = group.model_dump(exclude={"id", "reference_prefix", "quantity"}, exclude_none=True)
        parts.extend(models.BomPart.model_validate({"ref": ref, **shared}) for ref in refs)

    oversized = {sheet: count for sheet, count in per_sheet.items() if count > BOM_SHEET_PART_LIMIT}
    if oversized:
        raise ValueError(f"BOM exceeds {BOM_SHEET_PART_LIMIT} parts on a sheet: {oversized}")
    arrays = [
        models.ArraySpec.model_validate(
            {
                "refs": refs_by_group[array.group_id],
                **array.model_dump(exclude={"group_id"}, exclude_none=True),
            }
        )
        for array in response.arrays
    ]
    return response, parts, arrays, refs_by_group, expansions


def _normalize_bom_stage_response(
    payload: dict, prompt_state: dict | None = None
) -> tuple[dict, int]:
    """Expand recipe parts first, then model component groups."""
    response, parts, arrays, _refs_by_group, expansions = _expand_bom_groups(payload, prompt_state)
    recipe_part_count = sum(len(expansion.parts) for expansion in expansions)
    canonical = models.BOM(
        parts=parts,
        arrays=arrays,
        assumptions=response.assumptions,
        substitutions=response.substitutions,
        connections=[
            connection for expansion in expansions for connection in expansion.connections
        ],
        no_connect_pins=[pin for expansion in expansions for pin in expansion.no_connect_pins],
        edge_interfaces=[
            interface for expansion in expansions for interface in expansion.edge_interfaces
        ],
        recipe_ownership=[expansion.ownership for expansion in expansions],
    )
    model_part_count = sum(group.quantity for group in response.groups)
    return canonical.model_dump(exclude_none=True), model_part_count + recipe_part_count


def _normalize_wiring_stage_response(payload: dict, prompt_state: dict) -> dict:
    """Merge project-owned assignments with immutable recipe wiring."""
    response = WiringStageResponse.model_validate(payload)
    bom = prompt_state.get("bom")
    if not isinstance(bom, dict):
        raise ValueError("wiring response requires a committed BOM")
    from kicraft.design.recipes import locked_no_connect_pins, locked_pin_assignments

    locked = locked_pin_assignments(bom)
    locked_no_connects = locked_no_connect_pins(bom)
    ref_sheets = {
        str(part.get("ref")): str(part.get("sheet"))
        for part in bom.get("parts") or []
        if isinstance(part, dict) and part.get("ref") and part.get("sheet")
    }
    grouped: dict[tuple[str, str], list[models.PinEndpoint]] = {}
    no_connect_pins: list[models.PinEndpoint] = [
        models.PinEndpoint(ref=ref, pin=pin) for ref, pin in sorted(locked_no_connects)
    ]
    for assignment in response.pins:
        if assignment.ref not in ref_sheets:
            raise ValueError(f"wiring references unknown component {assignment.ref!r}")
        if (assignment.ref, assignment.pin) in locked or (
            assignment.ref,
            assignment.pin,
        ) in locked_no_connects:
            raise ValueError(
                f"wiring attempted to overwrite recipe-owned pin {assignment.ref}.{assignment.pin}"
            )
        endpoint = models.PinEndpoint(ref=assignment.ref, pin=assignment.pin)
        if isinstance(assignment, NoConnectPinAssignment):
            no_connect_pins.append(endpoint)
            continue
        key = (ref_sheets[assignment.ref], assignment.net)
        grouped.setdefault(key, []).append(endpoint)
    for (ref, pin), net in locked.items():
        key = (ref_sheets[ref], net)
        grouped.setdefault(key, []).append(models.PinEndpoint(ref=ref, pin=pin))
    connections = [
        models.NetConnection(sheet=sheet, net_name=net, endpoints=endpoints)
        for (sheet, net), endpoints in grouped.items()
    ]
    return {
        "connections": [connection.model_dump() for connection in connections],
        "no_connect_pins": [endpoint.model_dump() for endpoint in no_connect_pins],
    }


def _json_response_format(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def _slot_response_schema(stage: str) -> dict:
    if stage == "intent":
        return IntentStageResponse.model_json_schema()
    if stage == "bom":
        return BomStageResponse.model_json_schema()
    if stage == "wiring":
        return WiringStageResponse.model_json_schema()
    if stage == "architecture":
        return ArchitectureStageResponse.model_json_schema()
    return SLOT_MODEL[stage].model_json_schema()


def _response_schema(stage: str) -> dict:
    slot = dict(_slot_response_schema(stage))
    question = dict(StageQuestionResponse.model_json_schema())
    definitions = {
        **(slot.pop("$defs", {}) or {}),
        **(question.pop("$defs", {}) or {}),
    }
    schema = {"anyOf": [slot, question]}
    if definitions:
        schema["$defs"] = definitions
    return schema


@dataclass(frozen=True)
class StageResponseContract:
    stage: str
    schema: dict
    response_format: dict


def _architecture_sheet_names(prompt_state: dict) -> tuple[str, ...]:
    architecture = prompt_state.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("BOM response contract requires architecture object")
    sheets = architecture.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise ValueError("BOM response contract requires nonempty architecture.sheets")
    names: list[str] = []
    for sheet in sheets:
        if not isinstance(sheet, dict):
            raise ValueError("architecture.sheets entries must be objects")
        name = sheet.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("architecture sheet names must be nonempty strings")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("architecture sheet names must be unique")
    return tuple(names)


def build_stage_response_contract(
    stage: str,
    prompt_state: dict,
    *,
    bom_sheet: str | None = None,
    allow_questions: bool = True,
    wiring_refs: tuple[str, ...] | None = None,
) -> StageResponseContract:
    schema = _response_schema(stage) if allow_questions else _slot_response_schema(stage)
    if stage == "bom":
        architecture_names = _architecture_sheet_names(prompt_state)
        if bom_sheet is not None and bom_sheet not in architecture_names:
            raise ValueError(f"unknown BOM work-unit sheet {bom_sheet!r}")
        names = [bom_sheet] if bom_sheet is not None else list(architecture_names)
        definitions = schema.get("$defs")
        if not isinstance(definitions, dict):
            raise ValueError("BOM response schema is missing $defs")
        definition = definitions.get("BomComponentGroup")
        properties = definition.get("properties") if isinstance(definition, dict) else None
        sheet = properties.get("sheet") if isinstance(properties, dict) else None
        if not isinstance(sheet, dict):
            raise ValueError("BOM response schema is missing BomComponentGroup.sheet")
        sheet["enum"] = names
        architecture = prompt_state.get("architecture") or {}
        selections = [
            selection
            for selection in architecture.get("recipe_selections") or []
            if isinstance(selection, dict)
        ]
        recipe_sheets = {
            str(selected_sheet)
            for selection in selections
            for selected_sheet in (selection.get("sheets") or {}).values()
        }
        resolved_requirement_ids = {
            str(requirement_id)
            for selection in selections
            for requirement_id in selection.get("requirement_ids") or []
        }
        sheet_has_unresolved_requirement = any(
            isinstance(requirement, dict)
            and requirement.get("sheet") == bom_sheet
            and str(requirement.get("id")) not in resolved_requirement_ids
            for requirement in architecture.get("requirements") or []
        )
        require_groups = bom_sheet is not None and (
            bom_sheet not in recipe_sheets or sheet_has_unresolved_requirement
        )
        if require_groups:
            variants = schema.get("anyOf") or [schema]
            for variant in variants:
                variant_properties = (
                    variant.get("properties") if isinstance(variant, dict) else None
                )
                groups = (
                    variant_properties.get("groups")
                    if isinstance(variant_properties, dict)
                    else None
                )
                if not isinstance(groups, dict):
                    continue
                groups["minItems"] = 1
                required = list(variant.get("required") or [])
                if "groups" not in required:
                    required.append("groups")
                variant["required"] = required
    elif stage == "wiring" and wiring_refs is not None:
        definitions = schema.get("$defs")
        if not isinstance(definitions, dict):
            raise ValueError("wiring response schema is missing $defs")
        for name in ("ConnectedPinAssignment", "NoConnectPinAssignment"):
            definition = definitions.get(name)
            properties = definition.get("properties") if isinstance(definition, dict) else None
            ref = properties.get("ref") if isinstance(properties, dict) else None
            if not isinstance(ref, dict):
                raise ValueError(f"wiring response schema is missing {name}.ref")
            ref["enum"] = list(wiring_refs)
    version = 3 if stage in {"bom", "wiring"} else (2 if stage == "architecture" else 1)
    contract_name = f"kicraft_{stage}_response_v{version}"
    if not allow_questions:
        contract_name += "_noninteractive"
    response_format = _json_response_format(contract_name, schema)
    return StageResponseContract(stage=stage, schema=schema, response_format=response_format)


def schema_json(contract: StageResponseContract) -> str:
    return json.dumps(contract.schema)


class StageSchemaError(ValueError):
    pass


def _inter_sheet_net_endpoint_signature(endpoints: list[dict]) -> tuple[tuple[str, str], ...]:
    """Order-independent, multiplicity-preserving endpoint identity.

    Endpoint order is not semantically meaningful for an inter-sheet net, but
    repeated endpoints are, so the signature is a sorted tuple, never a set.
    """
    return tuple(
        sorted((str(endpoint["sheet"]), str(endpoint["direction"])) for endpoint in endpoints)
    )


def _complete_connector_requirements(payload: dict) -> dict:
    """Auto-bind empty connector ports from the sheet's inter-sheet interface.

    A header/screw-terminal requirement whose ``ports`` are empty cannot be
    lowered deterministically, and the model's BOM unit then emits nothing
    (empty-sheet exhaustion). The pin order of such a connector is exactly the
    nets that cross its sheet, in declaration order, plus GND for a signal
    header. Bind those here so the connector lowerer owns both BOM and wiring.
    """
    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        return payload
    nets_by_sheet: dict[str, list[str]] = {}
    for net in payload.get("inter_sheet_nets") or []:
        if not isinstance(net, dict):
            continue
        name = net.get("name")
        for endpoint in net.get("endpoints") or []:
            if isinstance(endpoint, dict) and endpoint.get("sheet"):
                nets_by_sheet.setdefault(str(endpoint["sheet"]), []).append(str(name))
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        if requirement.get("role") != "connector" or requirement.get("ports"):
            continue
        sheet = requirement.get("sheet")
        signals = nets_by_sheet.get(sheet) or []
        family = str(requirement.get("family") or "").lower()
        is_terminal = "screw" in family or "terminal" in family
        nets = list(signals) if is_terminal else [*signals, "GND"]
        if not nets:
            continue
        requirement["ports"] = {f"PIN{index + 1}": net for index, net in enumerate(nets)}
    return payload


def _normalize_architecture_sheet_aliases(payload: dict) -> dict:
    """Canonicalize harmless sheet identifier representation differences.

    ``Sheet.name`` is the uppercase human label (spaces); ``Sheet.stem`` is the
    uppercase filesystem identifier (underscores). Structured-output schemas
    cannot express the Architecture model's cross-field endpoint check, and
    providers commonly vary case or interchange those two spellings. Normalize
    those representation-only differences locally instead of spending another
    provider call; punctuation and missing fields remain schema errors.
    """
    raw_sheets = payload.get("sheets")
    if not isinstance(raw_sheets, list):
        return payload

    def alias_key(value: str) -> str:
        return re.sub(r"\s+", " ", value.replace("_", " ")).strip().upper()

    aliases: dict[str, str] = {}
    sheets: list[object] = []
    for raw_sheet in raw_sheets:
        if not isinstance(raw_sheet, dict):
            sheets.append(raw_sheet)
            continue
        sheet = dict(raw_sheet)
        raw_name = sheet.get("name")
        canonical_name = None
        if isinstance(raw_name, str):
            canonical_name = alias_key(raw_name)
            sheet["name"] = canonical_name
            aliases[alias_key(raw_name)] = canonical_name

        raw_stem = sheet.get("stem")
        if isinstance(raw_stem, str) and raw_stem.strip():
            canonical_stem = re.sub(r"\s+", "_", raw_stem.strip()).upper()
        elif canonical_name:
            canonical_stem = canonical_name.replace(" ", "_")
        else:
            canonical_stem = None
        if canonical_stem is not None:
            sheet["stem"] = canonical_stem
            if canonical_name is not None:
                if isinstance(raw_stem, str) and raw_stem:
                    aliases[alias_key(raw_stem)] = canonical_name
                aliases[alias_key(canonical_stem)] = canonical_name
        sheets.append(sheet)

    normalized = dict(payload)
    normalized["sheets"] = sheets

    def normalize_reference(value: str) -> str:
        return aliases.get(alias_key(value), value)

    for field in ("inter_sheet_nets", "inter_sheet_net_ranges"):
        raw_nets = payload.get(field)
        if not isinstance(raw_nets, list):
            continue
        nets: list[object] = []
        for raw_net in raw_nets:
            if not isinstance(raw_net, dict):
                nets.append(raw_net)
                continue
            net = dict(raw_net)
            raw_endpoints = net.get("endpoints")
            if isinstance(raw_endpoints, list):
                endpoints: list[object] = []
                for raw_endpoint in raw_endpoints:
                    if not isinstance(raw_endpoint, dict):
                        endpoints.append(raw_endpoint)
                        continue
                    endpoint = dict(raw_endpoint)
                    raw_ref = endpoint.get("sheet")
                    if isinstance(raw_ref, str):
                        endpoint["sheet"] = normalize_reference(raw_ref)
                    endpoints.append(endpoint)
                net["endpoints"] = endpoints
            nets.append(net)
        normalized[field] = nets

    raw_requirements = payload.get("requirements")
    if isinstance(raw_requirements, list):
        requirements: list[object] = []
        for raw_requirement in raw_requirements:
            if not isinstance(raw_requirement, dict):
                requirements.append(raw_requirement)
                continue
            requirement = dict(raw_requirement)
            raw_ref = requirement.get("sheet")
            if isinstance(raw_ref, str):
                requirement["sheet"] = normalize_reference(raw_ref)
            requirements.append(requirement)
        normalized["requirements"] = requirements

    raw_selections = payload.get("recipe_selections")
    if isinstance(raw_selections, list):
        selections: list[object] = []
        for raw_selection in raw_selections:
            if not isinstance(raw_selection, dict):
                selections.append(raw_selection)
                continue
            selection = dict(raw_selection)
            raw_mapping = selection.get("sheets")
            if isinstance(raw_mapping, dict):
                selection["sheets"] = {
                    role: normalize_reference(sheet_ref)
                    if isinstance(sheet_ref, str)
                    else sheet_ref
                    for role, sheet_ref in raw_mapping.items()
                }
            selections.append(selection)
        normalized["recipe_selections"] = selections

    return normalized


def _normalize_usb_c_requirements(payload: dict) -> dict:
    """Route generic USB-C architecture requirements through verified recipes."""
    normalized = dict(payload)
    nets = {
        str(name)
        for name in [
            *(payload.get("power_nets") or []),
            *(
                row.get("name")
                for row in payload.get("inter_sheet_nets") or []
                if isinstance(row, dict)
            ),
        ]
        if name
    }

    def named_net(*aliases: str) -> str | None:
        wanted = {re.sub(r"[^a-z0-9]+", "", alias.lower()) for alias in aliases}
        return next(
            (net for net in nets if re.sub(r"[^a-z0-9]+", "", net.lower()) in wanted),
            None,
        )

    usb_dp = named_net("USB_DP", "USB_D+")
    usb_dm = named_net("USB_DM", "USB_D-")
    vbus = named_net("VBUS", "+5V", "5V")
    requirements = []
    for row in payload.get("requirements") or []:
        if not isinstance(row, dict):
            requirements.append(row)
            continue
        requirement = dict(row)
        family = re.sub(r"[^a-z0-9]+", "", str(requirement.get("family") or "").lower())
        if "usbc" not in family and "typec" not in family:
            requirements.append(requirement)
            continue
        ports = dict(requirement.get("ports") or {})
        ports["gnd"] = "GND"
        if vbus:
            ports["vbus"] = vbus
        if usb_dp and usb_dm:
            requirement["family"] = "usb-c-usb2-device"
            ports["usb_dp"] = usb_dp
            ports["usb_dm"] = usb_dm
        else:
            requirement["family"] = "usb-c-power-sink"
        requirement["ports"] = ports
        requirements.append(requirement)
    requirement_sheets = {
        str(row.get("sheet"))
        for row in requirements
        if isinstance(row, dict)
        and "usbc" in re.sub(r"[^a-z0-9]+", "", str(row.get("family") or "").lower())
    }
    for sheet in payload.get("sheets") or []:
        if not isinstance(sheet, dict):
            continue
        sheet_name = str(sheet.get("name") or "")
        identity = f"{sheet_name} {sheet.get('function') or ''}".lower()
        if sheet_name in requirement_sheets or not re.search(
            r"\busb(?:\s*|-)?c\b|\btype(?:\s*|-)?c\b", identity
        ):
            continue
        ports = {"gnd": "GND"}
        if vbus:
            ports["vbus"] = vbus
        has_data = usb_dp is not None and usb_dm is not None
        if has_data:
            ports.update({"usb_dp": usb_dp, "usb_dm": usb_dm})
        stem = re.sub(r"[^a-z0-9]+", "_", sheet_name.lower()).strip("_") or "input"
        requirements.append(
            {
                "id": f"auto_usb_c_{stem}",
                "sheet": sheet_name,
                "role": "connector" if has_data else "power_input",
                "family": "usb-c-usb2-device" if has_data else "usb-c-power-sink",
                "parameters": {},
                "ports": ports,
                "interfaces": [],
            }
        )
    normalized["requirements"] = requirements
    return normalized


def _fold_recipe_covered_sheets(architecture: models.Architecture) -> models.Architecture:
    """Fold model-created sub-sheets already owned by a selected circuit recipe."""
    from kicraft.design.recipes import get_recipe

    data = architecture.model_dump(exclude_none=True)
    requirements = [row for row in data.get("requirements") or [] if isinstance(row, dict)]
    selections = [row for row in data.get("recipe_selections") or [] if isinstance(row, dict)]
    claimed: dict[str, tuple[str, dict]] = {}

    def token(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    for selection in selections:
        definition = get_recipe(str(selection.get("recipe")))
        bindings = selection.get("sheets") or {}
        for requirement in requirements:
            requirement_id = str(requirement.get("id"))
            if requirement_id in set(selection.get("requirement_ids") or []):
                continue
            family = token(requirement.get("family"))
            for group in definition.parts:
                role = token(group.role)
                covered = (
                    role == family
                    or (role and role in family)
                    or (
                        definition.family == "rp2040"
                        and family == "gpioheader"
                        and role.startswith("cast")
                    )
                )
                target_sheet = bindings.get(group.sheet_role)
                if covered and target_sheet and requirement.get("sheet") != target_sheet:
                    claimed[requirement_id] = (str(target_sheet), selection)
                    break

    requirements_by_sheet: dict[str, list[dict]] = {}
    for requirement in requirements:
        requirements_by_sheet.setdefault(str(requirement.get("sheet")), []).append(requirement)
    bound_sheets = {
        str(sheet) for selection in selections for sheet in (selection.get("sheets") or {}).values()
    }
    folds: dict[str, str] = {}
    for sheet, sheet_requirements in requirements_by_sheet.items():
        targets = {
            claimed[str(requirement.get("id"))][0]
            for requirement in sheet_requirements
            if str(requirement.get("id")) in claimed
        }
        if (
            sheet not in bound_sheets
            and len(targets) == 1
            and all(str(requirement.get("id")) in claimed for requirement in sheet_requirements)
        ):
            folds[sheet] = next(iter(targets))
    if not folds:
        return architecture

    for requirement in requirements:
        source_sheet = str(requirement.get("sheet"))
        if source_sheet not in folds:
            continue
        requirement["sheet"] = folds[source_sheet]
        requirement_id = str(requirement.get("id"))
        selection = claimed[requirement_id][1]
        selection["requirement_ids"] = list(
            dict.fromkeys([*(selection.get("requirement_ids") or []), requirement_id])
        )
    data["sheets"] = [
        sheet for sheet in data.get("sheets") or [] if str(sheet.get("name")) not in folds
    ]
    rewritten_nets = []
    for net in data.get("inter_sheet_nets") or []:
        endpoints = []
        observed = set()
        for endpoint in net.get("endpoints") or []:
            rewritten = dict(endpoint)
            rewritten["sheet"] = folds.get(str(rewritten.get("sheet")), rewritten.get("sheet"))
            signature = (rewritten.get("sheet"), rewritten.get("direction"))
            if signature not in observed:
                observed.add(signature)
                endpoints.append(rewritten)
        if len({endpoint["sheet"] for endpoint in endpoints}) >= 2:
            rewritten_nets.append({**net, "endpoints": endpoints})
    data["inter_sheet_nets"] = rewritten_nets
    data["requirements"] = requirements
    data["recipe_selections"] = selections
    data["unresolved_requirement_ids"] = [
        requirement_id
        for requirement_id in data.get("unresolved_requirement_ids") or []
        if requirement_id not in claimed
    ]
    return models.Architecture.model_validate(data)


def _normalize_stage_response(stage: str, payload: dict, prompt_state: dict) -> tuple[dict, int]:
    try:
        if isinstance(payload.get("questions"), list):
            return StageQuestionResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "intent":
            return IntentStageResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "architecture":
            payload = _normalize_architecture_sheet_aliases(payload)
            payload = _complete_connector_requirements(payload)
            payload = _normalize_usb_c_requirements(payload)
            named_parts = (prompt_state.get("intent") or {}).get("named_parts") or []
            for requirement in payload.get("requirements") or []:
                if not isinstance(requirement, dict) or requirement.get("exact_part"):
                    continue
                requirement_identity = re.sub(
                    r"[^a-z0-9]+",
                    "",
                    f"{requirement.get('id', '')} {requirement.get('family', '')}".lower(),
                )
                for named_part in named_parts:
                    named_identity = re.sub(r"[^a-z0-9]+", "", str(named_part).lower())
                    if named_identity and named_identity in requirement_identity:
                        requirement["exact_part"] = str(named_part)
                        break
            architecture_assumptions = " ".join(
                str(value) for value in payload.get("assumptions") or []
            ).lower()
            for requirement in payload.get("requirements") or []:
                if not isinstance(requirement, dict):
                    continue
                family = re.sub(r"[^a-z0-9]+", "", str(requirement.get("family") or "").lower())
                if family == "opampbuffer" and "mcp6001" in architecture_assumptions:
                    requirement["family"] = "mcp6001-follower"
                    requirement["exact_part"] = "MCP6001T-I/OT"
                    ports = dict(requirement.get("ports") or {})
                    ports.setdefault("gnd", "GND")
                    rail = next(
                        (
                            net
                            for net in payload.get("power_nets") or []
                            if str(net).upper() != "GND"
                        ),
                        None,
                    )
                    if rail:
                        ports.setdefault("vdd", str(rail))
                    requirement["ports"] = ports
            response = ArchitectureStageResponse.model_validate(payload)
            canonical = response.model_dump(exclude={"inter_sheet_net_ranges"}, exclude_none=True)
            explicit_nets = canonical.get("inter_sheet_nets") or []
            explicit_by_name = {str(net["name"]): net for net in explicit_nets}
            # Every name a range emits, whether kept or deduplicated: a second
            # range over the same name is redundant and stays rejected.
            range_covered = set()
            expanded = []
            for net_range in response.inter_sheet_net_ranges:
                range_signature = _inter_sheet_net_endpoint_signature(
                    [endpoint.model_dump() for endpoint in net_range.endpoints]
                )
                for number in range(net_range.start, net_range.end + 1):
                    name = net_range.name_pattern.replace("{n}", str(number))
                    if name in range_covered:
                        raise ValueError(f"duplicate/overlapping inter-sheet net {name!r}")
                    explicit = explicit_by_name.get(name)
                    if (
                        explicit is not None
                        and _inter_sheet_net_endpoint_signature(explicit["endpoints"])
                        != range_signature
                    ):
                        raise ValueError(f"duplicate/overlapping inter-sheet net {name!r}")
                    range_covered.add(name)
                    if explicit is not None:
                        # Semantically identical to the explicit canonical net
                        # (same endpoints, any order): keep that one and drop
                        # only this redundant range expansion.
                        continue
                    expanded.append(
                        models.InterSheetNet(name=name, endpoints=net_range.endpoints).model_dump()
                    )
            canonical["inter_sheet_nets"] = explicit_nets + expanded
            from kicraft.design.recipes import apply_architecture_recipe_resolution

            validated = models.Architecture.model_validate(canonical)
            resolved = apply_architecture_recipe_resolution(
                validated,
                prompt_state.get("intent") or {},
            )
            resolved = _fold_recipe_covered_sheets(resolved)
            return resolved.model_dump(exclude_none=True), len(expanded)
        if stage == "bom":
            return _normalize_bom_stage_response(payload, prompt_state)
        if stage == "wiring":
            return _normalize_wiring_stage_response(payload, prompt_state), 0
        return SLOT_MODEL[stage].model_validate(payload).model_dump(exclude_none=True), 0
    except (TypeError, ValueError) as exc:
        raise StageSchemaError(str(exc)) from exc


def _extract_json(text: str) -> dict:
    """Parse exactly ONE complete JSON object from ``text``.

    Tolerates optional markdown fences and leading/trailing whitespace, but NOT
    trailing prose or a second object: a complete object followed by
    non-whitespace is a malformed answer (the caller classifies it
    ``invalid_json``), never a silent success that drops content
    (bom-stage-programming-and-json-gaps plan).
    """
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    a = text.find("{")
    if a == -1:
        raise json.JSONDecodeError("no JSON object in reply", text, 0)
    obj, end = json.JSONDecoder().raw_decode(text[a:])
    if text[a + end :].strip():
        raise json.JSONDecodeError("trailing content after JSON object", text, a + end)
    return obj
