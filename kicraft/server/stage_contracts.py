"""Per-invocation stage response contracts and response normalization."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kicraft.design import models

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


class WiringSlotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connections: list[models.NetConnection]
    no_connect_pins: list[models.PinEndpoint]


class IntentStageResponse(models.IntentSlot):
    model_config = ConfigDict(extra="forbid")

    project_stem: str = Field(pattern=r"^[A-Z0-9_]{1,32}$")


class StageQuestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[models.Question] = Field(min_length=1, max_length=5)


class BomPartRun(BaseModel):
    """Compact stage-only declaration for identical BOM members."""

    model_config = ConfigDict(extra="forbid")

    refs: list[str] | None = Field(default=None, min_length=1, max_length=450)
    ref_prefix: str | None = None
    start: int | None = Field(default=None, ge=1)
    end: int | None = Field(default=None, ge=1)
    value: str
    symbol: str
    footprint: str
    sheet: str
    mpn: str | None = None
    datasheet: str | None = None
    sourcing_note: str | None = None
    side: str | None = None

    @model_validator(mode="after")
    def _one_reference_form(self):
        explicit = self.refs is not None
        ranged = self.ref_prefix is not None or self.start is not None or self.end is not None
        if explicit == ranged:
            raise ValueError("part run requires exactly one of refs or ref_prefix/start/end")
        if ranged:
            if not self.ref_prefix or self.start is None or self.end is None:
                raise ValueError("part run range requires ref_prefix, start, and end")
            if not re.fullmatch(r"[A-Z]+", self.ref_prefix):
                raise ValueError("part run ref_prefix must contain uppercase letters only")
            if self.end < self.start:
                raise ValueError("part run end must be greater than or equal to start")
            if self.end - self.start + 1 > 450:
                raise ValueError("part run may contain at most 450 references")
        return self

    def reference_count(self) -> int:
        if self.refs is not None:
            return len(self.refs)
        assert self.start is not None and self.end is not None
        return self.end - self.start + 1

    def iter_refs(self):
        if self.refs is not None:
            yield from self.refs
            return
        assert self.ref_prefix is not None and self.start is not None and self.end is not None
        for number in range(self.start, self.end + 1):
            yield f"{self.ref_prefix}{number}"


class BomStageResponse(BaseModel):
    """Schema-bound BOM response; expanded into canonical ``BOM.parts``."""

    model_config = ConfigDict(extra="forbid")

    parts: list[models.BomPart] = Field(default_factory=list, max_length=500)
    part_runs: list[BomPartRun] = Field(default_factory=list, max_length=500)
    ic_groups: dict[str, list[str]] = Field(default_factory=dict)
    group_labels: dict[str, str] = Field(default_factory=dict)
    thermal_refs: list[str] = Field(default_factory=list)
    signal_flow_order: list[str] = Field(default_factory=list)
    component_zones: dict[str, dict[str, str]] = Field(default_factory=dict)
    arrays: list[models.ArraySpec] = Field(default_factory=list)
    placement_hints: list[models.PlacementHint] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    substitutions: list[models.Substitution] = Field(default_factory=list)
    connections: list[models.NetConnection] = Field(default_factory=list)
    no_connect_pins: list[models.PinEndpoint] = Field(default_factory=list)

def _expand_bom_stage_response(payload: dict) -> tuple[dict, int]:
    """Validate compact runs and return the unchanged canonical BOM shape."""
    response = BomStageResponse.model_validate(payload)
    total = len(response.parts) + sum(run.reference_count() for run in response.part_runs)
    if total > 500:
        raise ValueError(f"BOM has {total} parts; maximum is 500")

    seen = {part.ref for part in response.parts}
    if len(seen) != len(response.parts):
        raise ValueError("BOM ordinary parts contain duplicate references")
    per_sheet: dict[str, int] = {}
    for part in response.parts:
        per_sheet[part.sheet] = per_sheet.get(part.sheet, 0) + 1
    run_refs: list[tuple[BomPartRun, list[str]]] = []
    for run in response.part_runs:
        refs = list(run.iter_refs())
        for ref in refs:
            if not models.REF_RE.fullmatch(ref):
                raise ValueError(f"invalid part run reference {ref!r}")
            if ref in seen:
                raise ValueError(f"duplicate or overlapping part run reference {ref!r}")
            seen.add(ref)
        per_sheet[run.sheet] = per_sheet.get(run.sheet, 0) + len(refs)
        run_refs.append((run, refs))
    oversized = {sheet: count for sheet, count in per_sheet.items() if count > 450}
    if oversized:
        raise ValueError(f"BOM exceeds 450 parts on a sheet: {oversized}")

    expanded = list(response.parts)
    for run, refs in run_refs:
        shared = run.model_dump(exclude={"refs", "ref_prefix", "start", "end"}, exclude_none=True)
        expanded.extend(models.BomPart.model_validate({"ref": ref, **shared}) for ref in refs)
    canonical = response.model_dump(exclude={"part_runs"})
    canonical["parts"] = [part.model_dump(exclude_none=True) for part in expanded]
    return models.BOM.model_validate(canonical).model_dump(exclude_none=True), total - len(
        response.parts
    )


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
        return WiringSlotResponse.model_json_schema()
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
        for definition_name in ("BomPart", "BomPartRun"):
            definition = definitions.get(definition_name)
            properties = definition.get("properties") if isinstance(definition, dict) else None
            sheet = properties.get("sheet") if isinstance(properties, dict) else None
            if not isinstance(sheet, dict):
                raise ValueError(f"BOM response schema is missing {definition_name}.sheet")
            sheet["enum"] = names
    response_format = _json_response_format(f"kicraft_{stage}_response_v1", schema)
    return StageResponseContract(stage=stage, schema=schema, response_format=response_format)


def schema_json(contract: StageResponseContract) -> str:
    return json.dumps(contract.schema)

class StageSchemaError(ValueError):
    pass


def _normalize_stage_response(stage: str, payload: dict) -> tuple[dict, int]:
    try:
        if isinstance(payload.get("questions"), list):
            return StageQuestionResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "intent":
            return IntentStageResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "bom":
            return _expand_bom_stage_response(payload)
        if stage == "wiring":
            return (
                WiringSlotResponse.model_validate(payload).model_dump(exclude_none=True),
                0,
            )
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
