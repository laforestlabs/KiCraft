"""Per-invocation stage response contracts and response normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kicraft.design import models

from .config import BOM_SHEET_PART_LIMIT, BOM_TOTAL_PART_LIMIT

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


class IntentStageResponse(models.IntentSlot):
    model_config = ConfigDict(extra="forbid")

    project_stem: str = Field(pattern=r"^[A-Z0-9_]{1,32}$")


class StageQuestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[models.Question] = Field(min_length=1, max_length=5)


class BomComponentGroup(BaseModel):
    """One component type expanded into deterministic references."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    reference_prefix: str = Field(pattern=r"^[A-Z]+$")
    quantity: int = Field(ge=1, le=BOM_SHEET_PART_LIMIT)
    value: str
    symbol: str
    footprint: str
    sheet: str
    mpn: str | None = None
    datasheet: str | None = None
    sourcing_note: str | None = None
    side: Literal["front", "back"] | None = None


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

    groups: list[BomComponentGroup] = Field(min_length=1, max_length=BOM_TOTAL_PART_LIMIT)
    arrays: list[BomArrayGroup] = Field(default_factory=list, max_length=100)
    assumptions: list[str] = Field(default_factory=list)
    substitutions: list[models.Substitution] = Field(default_factory=list)

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


def _normalize_bom_stage_response(payload: dict) -> tuple[dict, int]:
    """Expand component groups into the canonical per-part BOM."""
    response = BomStageResponse.model_validate(payload)
    total = sum(group.quantity for group in response.groups)
    if total > BOM_TOTAL_PART_LIMIT:
        raise ValueError(f"BOM has {total} parts; maximum is {BOM_TOTAL_PART_LIMIT}")

    per_sheet: dict[str, int] = {}
    next_number: dict[str, int] = {}
    refs_by_group: dict[str, list[str]] = {}
    parts: list[models.BomPart] = []
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

    arrays = []
    for array in response.arrays:
        data = array.model_dump(exclude={"group_id"}, exclude_none=True)
        arrays.append(
            models.ArraySpec.model_validate({"refs": refs_by_group[array.group_id], **data})
        )
    canonical = models.BOM(
        parts=parts,
        arrays=arrays,
        assumptions=response.assumptions,
        substitutions=response.substitutions,
    )
    return canonical.model_dump(exclude_none=True), total


def _normalize_wiring_stage_response(payload: dict, prompt_state: dict) -> dict:
    """Derive canonical connection rows from final pin assignments."""
    response = WiringStageResponse.model_validate(payload)
    bom = prompt_state.get("bom")
    if not isinstance(bom, dict):
        raise ValueError("wiring response requires a committed BOM")
    ref_sheets = {
        str(part.get("ref")): str(part.get("sheet"))
        for part in bom.get("parts") or []
        if isinstance(part, dict) and part.get("ref") and part.get("sheet")
    }
    grouped: dict[tuple[str, str], list[models.PinEndpoint]] = {}
    no_connect_pins: list[models.PinEndpoint] = []
    for assignment in response.pins:
        if assignment.ref not in ref_sheets:
            raise ValueError(f"wiring references unknown component {assignment.ref!r}")
        endpoint = models.PinEndpoint(ref=assignment.ref, pin=assignment.pin)
        if isinstance(assignment, NoConnectPinAssignment):
            no_connect_pins.append(endpoint)
            continue
        key = (ref_sheets[assignment.ref], assignment.net)
        grouped.setdefault(key, []).append(endpoint)
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


def build_stage_response_contract(stage: str, prompt_state: dict) -> StageResponseContract:
    schema = _response_schema(stage)
    if stage == "bom":
        names = list(_architecture_sheet_names(prompt_state))
        definitions = schema.get("$defs")
        if not isinstance(definitions, dict):
            raise ValueError("BOM response schema is missing $defs")
        definition = definitions.get("BomComponentGroup")
        properties = definition.get("properties") if isinstance(definition, dict) else None
        sheet = properties.get("sheet") if isinstance(properties, dict) else None
        if not isinstance(sheet, dict):
            raise ValueError("BOM response schema is missing BomComponentGroup.sheet")
        sheet["enum"] = names
    version = 2 if stage in {"bom", "wiring"} else 1
    response_format = _json_response_format(f"kicraft_{stage}_response_v{version}", schema)
    return StageResponseContract(stage=stage, schema=schema, response_format=response_format)


def schema_json(contract: StageResponseContract) -> str:
    return json.dumps(contract.schema)


class StageSchemaError(ValueError):
    pass


def _normalize_stage_response(stage: str, payload: dict, prompt_state: dict) -> tuple[dict, int]:
    try:
        if isinstance(payload.get("questions"), list):
            return StageQuestionResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "intent":
            return IntentStageResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "bom":
            return _normalize_bom_stage_response(payload)
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
