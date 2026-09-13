"""Per-invocation stage response contracts and response normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kicraft.design import models
from kicraft.design.lowering import lowerer_parameter_diagnostics
from kicraft.design.part_identity import is_part_family, matches_part_identity

from .config import (
    BOM_ARRAY_LIMIT,
    BOM_GROUP_LIMIT,
    BOM_NOTE_LIMIT,
    BOM_SHEET_PART_LIMIT,
    BOM_TOTAL_PART_LIMIT,
    STAGE_COLLECTION_BOUNDS,
    CollectionBound,
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
    """Match an implementing component to its typed requirement, not just its label."""

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
    from kicraft.design.recipes import registered_recipes

    component_tokens = {
        token(group.value),
        token(group.mpn or ""),
        token(group.symbol.partition(":")[2]),
    } - {""}
    component_families = [
        words(registered.definition.family) - {"fixed", "minimal"}
        for registered in registered_recipes()
        if token(registered.definition.exact_part) in component_tokens
    ]
    physical_identity = group.mpn or group.value
    for requirement in requirements or ():
        exact_part = value(requirement, "exact_part")
        if exact_part:
            if matches_part_identity(str(exact_part), physical_identity):
                return True
            # A matching group label or broad family cannot waive an explicit
            # physical identity, especially when the BOM declares another MPN.
            continue
        family_identity = str(value(requirement, "family") or "")
        if matches_part_identity(family_identity, physical_identity):
            return True
        if is_part_family(family_identity):
            # A typed product family needs verified concrete hardware. Neither
            # the family token itself nor a matching group label is a device.
            continue
        family_words = words(value(requirement, "family")) - {"fixed", "minimal", "selectable"}
        if family_words and any(
            family_words == family or family_words <= family for family in component_families
        ):
            return True
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
    trusted_lowering_groups = set((prompt_state or {}).get("_trusted_lowering_group_ids") or ())
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
        if (
            protected
            and group.id not in trusted_lowering_groups
            and not _requirement_owns_protected_group(group, requirements)
        ):
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


def _is_free_form_object(schema: object) -> bool:
    """A map-typed object (``dict[str, X]``): keys are data, not a fixed shape."""
    return (
        isinstance(schema, dict)
        and schema.get("type") == "object"
        and "properties" not in schema
        and isinstance(schema.get("additionalProperties"), dict)
    )


def _strict_provider_schema(node):
    """Deep-copy a Pydantic JSON schema into OpenAI's strict structured-output form.

    OpenAI (and any endpoint that enforces ``strict: true``) demands that every
    fixed-shape object close itself with ``additionalProperties: false`` and list
    *every* property in ``required``. Pydantic omits fields that carry defaults,
    so the raw ``model_json_schema()`` is rejected verbatim (live 2026-09-12:
    HTTP 400 ``invalid_json_schema`` on the first luna intent call). Free-form
    map objects (``dict[str, int]``) must instead stay open and MUST NOT be
    required -- OpenAI rejects them in ``required`` ("Extra required key").

    Root-level ``anyOf``/``oneOf``/``allOf`` are forbidden by OpenAI; the
    interactive contracts already express the slot/questions alternation as one
    object (see ``_response_schema``), so this only has to complete ``required``.

    Only the provider envelope is transformed; ``StageResponseContract.schema``
    (prompt text, in-stream guard allowlist, tests) keeps the canonical schema.
    """
    if isinstance(node, list):
        return [_strict_provider_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict = {}
    for key, value in node.items():
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: _strict_provider_schema(subschema) for name, subschema in value.items()}
        elif key in ("items", "additionalProperties", "not", "contains"):
            out[key] = _strict_provider_schema(value)
        elif key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(value, list):
            out[key] = [_strict_provider_schema(item) for item in value]
        else:
            out[key] = value
    properties = out.get("properties")
    if isinstance(properties, dict):
        out["additionalProperties"] = False
        required = set(out.get("required") or [])
        for name, subschema in properties.items():
            if _is_free_form_object(subschema):
                required.discard(name)
            else:
                required.add(name)
        out["required"] = sorted(required)
    return out


def _json_response_format(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": _strict_provider_schema(schema)},
    }


def _slot_response_schema(stage: str) -> dict:
    if stage == "intent":
        return IntentStageResponse.model_json_schema()
    if stage == "bom":
        return BomStageResponse.model_json_schema()
    if stage == "wiring":
        return WiringStageResponse.model_json_schema()
    if stage == "architecture":
        schema = ArchitectureStageResponse.model_json_schema()
        properties = schema["properties"]
        for field in ("recipe_resolution", "unresolved_requirement_ids", "protected_identities"):
            properties.pop(field, None)
        schema["$defs"].pop("RecipeResolutionRecord", None)
        properties["requirements"]["minItems"] = 1
        schema["required"] = [*schema.get("required", []), "requirements"]
        requirement = schema["$defs"]["CircuitRequirement"]
        requirement["properties"]["functional_blocks"]["minItems"] = 1
        # Canonical validation enforces uniqueness; Alibaba rejects array uniqueItems.
        requirement["required"] = [*requirement.get("required", []), "functional_blocks"]
        return schema
    return SLOT_MODEL[stage].model_json_schema()


def _response_schema(stage: str) -> dict:
    """One object holding the slot plus an optional (possibly empty) ``questions``.

    OpenAI strict structured outputs reject ``anyOf``/``oneOf``/``allOf`` at the
    schema root and require every typed property in ``required``, so the
    interactive "slot OR clarifying questions" union is expressed as a single
    object: the slot fields plus ``questions``. ``questions`` is required (the
    decoder must always emit it) but ``minItems`` is 0, so ``[]`` is the normal
    answer and a non-empty array is how the model asks. Slot-answer behavior is
    unchanged: the state contract still accepts the slot shape.
    """
    slot = dict(_slot_response_schema(stage))
    question = dict(StageQuestionResponse.model_json_schema())
    definitions = {
        **(slot.pop("$defs", {}) or {}),
        **(question.pop("$defs", {}) or {}),
    }
    properties = dict(slot.get("properties") or {})
    questions = dict((question.get("properties") or {}).get("questions") or {})
    questions["minItems"] = 0
    properties["questions"] = questions
    required = [*slot.get("required", [])]
    if "questions" not in required:
        required.append("questions")
    schema = {**slot, "type": "object", "properties": properties, "required": required}
    if definitions:
        schema["$defs"] = definitions
    return schema


@dataclass(frozen=True)
class StageResponseContract:
    stage: str
    schema: dict
    contract_name: str
    allow_questions: bool = True

    @property
    def response_format(self) -> dict:
        """The provider envelope, derived from the *current* ``schema``.

        A property (not a stored copy) so callers that tighten the schema after
        construction -- work units apply invocation-local collection bounds --
        send the same limits they enforce.
        """
        return _json_response_format(self.contract_name, self.schema)


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


def apply_collection_bounds(schema: dict, bounds: tuple[CollectionBound, ...]) -> None:
    """Constrain provider arrays without relaxing existing model or unit limits."""
    for variant in schema.get("anyOf") or [schema]:
        properties = variant.get("properties") or {}
        for bound in bounds:
            collection = properties.get(bound.field)
            if isinstance(collection, dict) and collection.get("type") == "array":
                collection["maxItems"] = min(collection.get("maxItems", bound.total), bound.total)


def build_stage_response_contract(
    stage: str,
    prompt_state: dict,
    *,
    bom_sheet: str | None = None,
    allow_questions: bool = True,
    wiring_refs: tuple[str, ...] | None = None,
) -> StageResponseContract:
    schema = _response_schema(stage) if allow_questions else _slot_response_schema(stage)
    apply_collection_bounds(schema, STAGE_COLLECTION_BOUNDS.get(stage, ()))
    if stage == "architecture":
        functional_spec = prompt_state.get("functional_spec")
        if isinstance(functional_spec, dict):
            spec = models.FunctionalSpec.model_validate(functional_spec)
            names = [block.name for block in spec.blocks]
            if not names:
                raise ValueError("architecture response contract requires functional block names")
            ownership = schema["$defs"]["CircuitRequirement"]["properties"]["functional_blocks"]
            ownership["items"]["enum"] = names
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
            if bom_sheet is None or bom_sheet in (selection.get("sheets") or {}).values()
            for requirement_id in selection.get("requirement_ids") or []
        }
        unresolved_requirement_ids = {
            str(requirement_id)
            for requirement_id in architecture.get("unresolved_requirement_ids") or []
        }
        sheet_has_unresolved_requirement = any(
            isinstance(requirement, dict)
            and requirement.get("sheet") == bom_sheet
            and (
                str(requirement.get("id")) not in resolved_requirement_ids
                or str(requirement.get("id")) in unresolved_requirement_ids
            )
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
    return StageResponseContract(
        stage=stage,
        schema=schema,
        contract_name=contract_name,
        allow_questions=allow_questions,
    )


def schema_json(contract: StageResponseContract) -> str:
    return json.dumps(contract.schema)


class StageSchemaError(ValueError):
    def __init__(self, message: str, *, diagnostic: dict | None = None):
        self.diagnostic = diagnostic
        super().__init__(message)


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
        if family.replace("_", "-") not in {
            "pin-header",
            "generic-header",
            "header",
            "spi-header",
            "screw-terminal",
        }:
            continue
        is_terminal = "screw" in family or "terminal" in family
        nets = list(dict.fromkeys(signals if is_terminal else [*signals, "GND"]))
        if not nets:
            continue
        requirement["ports"] = {f"PIN{index + 1}": net for index, net in enumerate(nets)}
    return payload


def _normalize_architecture_sheet_aliases(payload: dict) -> dict:
    """Canonicalize harmless sheet identifier representation differences.

    ``Sheet.name`` is the uppercase human label (spaces); ``Sheet.stem`` is the
    uppercase filesystem identifier (underscores). Structured-output schemas
    cannot express the Architecture model's cross-field endpoint check, and
    providers commonly vary case, punctuation, or interchange those two
    spellings. Normalize those representation-only differences locally instead
    of spending another provider call.
    """
    raw_sheets = payload.get("sheets")
    if not isinstance(raw_sheets, list):
        return payload

    def alias_key(value: str) -> str:
        return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()

    aliases: dict[str, str] = {}
    canonical_names: set[str] = set()

    def register_alias(raw: str, canonical: str) -> None:
        key = alias_key(raw)
        prior = aliases.get(key)
        if prior is not None and prior != canonical:
            raise ValueError(
                f"sheet alias {raw!r} ambiguously names both {prior!r} and {canonical!r}"
            )
        aliases[key] = canonical

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
            if canonical_name in canonical_names:
                raise ValueError(f"duplicate canonical sheet name {canonical_name!r}")
            canonical_names.add(canonical_name)
            sheet["name"] = canonical_name
            register_alias(raw_name, canonical_name)

        raw_stem = sheet.get("stem")
        if isinstance(raw_stem, str) and raw_stem.strip():
            canonical_stem = re.sub(r"[^A-Z0-9]+", "_", raw_stem.upper()).strip("_")
        elif canonical_name:
            canonical_stem = canonical_name.replace(" ", "_")
        else:
            canonical_stem = None
        if canonical_stem is not None:
            sheet["stem"] = canonical_stem
            if canonical_name is not None:
                if isinstance(raw_stem, str) and raw_stem:
                    register_alias(raw_stem, canonical_name)
                register_alias(canonical_stem, canonical_name)
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


def _sheet_owns_usb_c_connector(sheet: dict) -> bool:
    """Distinguish a receptacle from sheets merely carrying its signals."""
    name = str(sheet.get("name") or "")
    function = str(sheet.get("function") or "")
    if re.search(r"\b(?:header|gpio|expansion)\b", name, re.I) or re.search(
        r"\b(?:mcu|microcontroller)\b|\b(?:pd|power delivery)\b.*\btrigger\b",
        function,
        re.I,
    ):
        return False
    usb = r"\b(?:usb[\s-]*c|type[\s-]*c)\b"
    return bool(
        re.search(usb, name, re.I)
        or re.search(usb + r"\s+(?:receptacle|connector|port)\b", function, re.I)
    )


# Generic USB-C identities whose recipe is a power-only 5 V sink. When such a
# connector's own sheet carries the MCU's D+/D- pair the sink identity is stale
# and, under the `completing` ladder arm, is replaced by the USB2 device recipe
# (see _normalize_usb_c_requirements and O6 in
# docs/plans/architecture-contract-correction-ladder.md).
_GENERIC_USB_C_SINK_IDENTITIES = frozenset({"usbc5vsink", "usbcpowersink"})


def _normalize_usb_c_requirements(
    payload: dict, *, complete_native_usb: bool = False
) -> dict:
    """Complete generic connector contracts without changing their hardware role."""
    from kicraft.design.recipes.registry import get_recipe

    normalized = dict(payload)
    power_nets = {str(net) for net in payload.get("power_nets") or []}
    sheet_nets: dict[str, set[str]] = {}
    for row in payload.get("inter_sheet_nets") or []:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        for endpoint in row.get("endpoints") or []:
            if isinstance(endpoint, dict) and endpoint.get("sheet"):
                sheet_nets.setdefault(str(endpoint["sheet"]), set()).add(str(row["name"]))

    def named_net(nets: set[str], *aliases: str) -> str | None:
        # Polarity is electrical meaning: D+ and D- must never share a key.
        wanted = {re.sub(r"[^a-z0-9+-]+", "", alias.lower()) for alias in aliases}
        matches = sorted(net for net in nets if re.sub(r"[^a-z0-9+-]+", "", net.lower()) in wanted)
        if len(matches) > 1:
            raise StageSchemaError(f"ambiguous connector net aliases: {matches}")
        return matches[0] if matches else None

    def ports_for(sheet: str, explicit: dict[str, str] | None = None) -> dict[str, str]:
        nets = sheet_nets.get(sheet, set())
        ports = dict(explicit or {})
        aliases = {
            "gnd": ("GND",),
            "vbus": ("VBUS", "+5V", "5V"),
            "usb_dp": ("USB_DP", "USB_D+", "USB_D_P", "D+"),
            "usb_dm": ("USB_DM", "USB_D-", "USB_D_N", "D-"),
            "cc1": ("CC1",),
            "cc2": ("CC2",),
            "sbu1": ("SBU1",),
            "sbu2": ("SBU2",),
        }
        for port, names in aliases.items():
            # A typed binding already identifies the net; aliases only complete
            # missing ports, never second-guess that binding.
            if port in ports:
                continue
            net = named_net(nets, *names)
            # Power rails may be global rather than inter-sheet interfaces.
            # Data and auxiliary signals must belong to this connector's sheet.
            if net is None and port in {"gnd", "vbus"}:
                net = named_net(power_nets, *names)
            if net is not None:
                ports[port] = net
        return ports

    generic_families = {
        "usbc",
        "typec",
        "usbcconnector",
        "typecconnector",
        "usbcreceptacle",
        "usbcpowersink",
        "usbcusb2device",
    }
    requirements = []
    for row in payload.get("requirements") or []:
        if not isinstance(row, dict):
            requirements.append(row)
            continue
        requirement = dict(row)
        family = re.sub(r"[^a-z0-9]+", "", str(row.get("family") or "").lower())
        if family in generic_families:
            ports = ports_for(str(row.get("sheet") or ""), row.get("ports"))
            has_data = bool(ports.get("usb_dp") and ports.get("usb_dm"))
            sink = get_recipe("usb-c-usb2-device@1" if has_data else "usb-c-5v-sink@1")
            exposed_auxiliary = ports.keys() - {port.name for port in sink.ports}
            exact = re.sub(r"[^a-z0-9]+", "", str(row.get("exact_part") or "").lower())
            if exposed_auxiliary and exact in {
                "usbc5vsink",
                "usbcpowersink",
                "usbcusb2device",
            }:
                raise StageSchemaError(
                    "USB connector exposes CC/SBU signals but requests an active sink recipe; "
                    "use a passive usb-c-breakout with the actual connector identity, "
                    "not a sink block identity"
                )
            requirement["family"] = "usb-c-breakout" if exposed_auxiliary else sink.family
            requirement["ports"] = ports
            if (
                complete_native_usb
                and has_data
                and not exposed_auxiliary
                and exact in _GENERIC_USB_C_SINK_IDENTITIES
            ):
                # O6: a generic sink identity cannot carry the D+/D- pair this
                # connector's own sheet exposes, and a stale exact part outranks
                # the upgraded family downstream (the sink recipe is then
                # unresolved, which blocks the MCU's mandatory native-USB
                # companion). The connector wired to the MCU's USB pair *is* the
                # USB2 device; bind the identity the resolver must select.
                requirement["exact_part"] = sink.exact_part
        requirements.append(requirement)
    requirement_sheets = {
        str(row.get("sheet")) for row in requirements if isinstance(row, dict) and row.get("sheet")
    }
    for sheet in payload.get("sheets") or []:
        if not isinstance(sheet, dict):
            continue
        sheet_name = str(sheet.get("name") or "")
        if sheet_name in requirement_sheets or not _sheet_owns_usb_c_connector(sheet):
            continue
        ports = ports_for(sheet_name)
        remaining = sheet_nets.get(sheet_name, set()) - power_nets - set(ports.values())
        # Arbitrary signal names are preserved losslessly as connector port keys.
        # They are not recipe pin names until an explicit typed binding exists.
        for net in sorted(remaining):
            key = net.lower().replace("+", "_plus").replace("-", "_minus")
            key = re.sub(r"[^a-z0-9_]+", "_", key).strip("_")
            if key in ports and ports[key] != net:
                raise StageSchemaError(f"connector port alias collision: {ports[key]!r}, {net!r}")
            ports[key] = net
        has_data = bool(ports.get("usb_dp") and ports.get("usb_dm"))
        sink = get_recipe("usb-c-usb2-device@1" if has_data else "usb-c-5v-sink@1")
        exposed_auxiliary = ports.keys() - {port.name for port in sink.ports}
        stem = re.sub(r"[^a-z0-9]+", "_", sheet_name.lower()).strip("_") or "input"
        requirements.append(
            {
                "id": f"auto_usb_c_{stem}",
                "sheet": sheet_name,
                "role": "connector" if has_data or exposed_auxiliary else "power_input",
                "family": "usb-c-breakout" if exposed_auxiliary else sink.family,
                "parameters": {},
                "ports": ports,
                "interfaces": [],
            }
        )
    normalized["requirements"] = requirements
    return normalized


def _fold_recipe_covered_sheets(architecture: models.Architecture) -> models.Architecture:
    """Fold only explicitly recipe-owned requirements with one unambiguous target."""
    data = architecture.model_dump(exclude_none=True)
    requirements = data["requirements"]
    selections = data["recipe_selections"]
    unresolved = set(architecture.unresolved_requirement_ids)
    targets_by_requirement: dict[str, set[str]] = {}
    for selection in selections:
        target_sheets = set(selection["sheets"].values())
        for requirement_id in selection["requirement_ids"]:
            if requirement_id not in unresolved:
                targets_by_requirement.setdefault(requirement_id, set()).update(target_sheets)
    claimed = {
        requirement_id: next(iter(targets))
        for requirement_id, targets in targets_by_requirement.items()
        if len(targets) == 1
    }

    requirements_by_sheet: dict[str, list[dict]] = {}
    for requirement in requirements:
        requirements_by_sheet.setdefault(str(requirement.get("sheet")), []).append(requirement)
    bound_sheets = {
        str(sheet) for selection in selections for sheet in (selection.get("sheets") or {}).values()
    }
    folds: dict[str, str] = {}
    for sheet, sheet_requirements in requirements_by_sheet.items():
        targets = {
            claimed[str(requirement.get("id"))]
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
        return models.Architecture.model_validate(data)

    for requirement in requirements:
        source_sheet = str(requirement.get("sheet"))
        if source_sheet not in folds:
            continue
        requirement["sheet"] = folds[source_sheet]
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
    return models.Architecture.model_validate(data)


def _validate_power_requirement_contracts(architecture: dict, prompt_state: dict) -> None:
    from kicraft.design.stage_semantics import architecture_power_requirement_diagnostics

    diagnostics = architecture_power_requirement_diagnostics(prompt_state, architecture)
    if diagnostics:
        message = " ".join(diagnostic.message for diagnostic in diagnostics)
        raise StageSchemaError(
            message,
            diagnostic={
                "code": "unrealizable_power_requirement",
                "message": message,
                "evidence": [
                    diagnostic.model_dump(exclude_none=True) for diagnostic in diagnostics
                ],
            },
        )


def _validate_lowerer_parameter_contracts(architecture: models.Architecture) -> None:
    diagnostics = [
        diagnostic
        for requirement in architecture.requirements
        for diagnostic in lowerer_parameter_diagnostics(requirement)
    ]
    if not diagnostics:
        return
    evidence = []
    for diagnostic in diagnostics:
        value = "<missing>" if diagnostic.missing else repr(diagnostic.value)
        evidence.append(
            f"requirement {diagnostic.requirement_id!r} on sheet {diagnostic.sheet!r} "
            f"({diagnostic.family}, {diagnostic.lowerer_id}): "
            f"parameters.{diagnostic.parameter}={value}; "
            + (
                diagnostic.constraint
                if diagnostic.constraint
                else f"choices={list(diagnostic.choices)!r}"
            )
        )
    message = (
        "Invalid deterministic lowerer parameter contract. Supply explicit supported values "
        "before BOM; do not infer missing hardware policy or substitute pull direction for "
        "internal/external implementation. " + "; ".join(evidence)
    )
    raise StageSchemaError(
        message,
        diagnostic={
            "code": "invalid_lowerer_parameters",
            "message": message,
            "evidence": [diagnostic.model_dump(mode="json") for diagnostic in diagnostics],
        },
    )


def _validate_typed_inter_sheet_contracts(
    architecture: models.Architecture, functional_spec: dict | None
) -> None:
    """Check shared typed signals only when explicit functional ownership links them.

    Net names alone do not establish scope. Neither power-only graph links nor
    ambiguous replicated owners prove that two sheet-local signals are one wire.
    Recipe peer completion and sheet folding must run before this check.
    """
    if not functional_spec:
        return
    spec = models.FunctionalSpec.model_validate(functional_spec)
    global_nets = {"GND", *architecture.power_nets}
    bindings: dict[str, dict[str, list[models.CircuitRequirement]]] = {
        block.name: {} for block in spec.blocks
    }
    for requirement in architecture.requirements:
        signals = set(requirement.ports.values()) - global_nets
        for block in requirement.functional_blocks:
            if block not in bindings:
                continue  # The block-sheet mapping gate owns invalid membership.
            for net in signals:
                bindings[block].setdefault(net, []).append(requirement)
    boundaries = {
        net.name: {endpoint.sheet for endpoint in net.endpoints}
        for net in architecture.inter_sheet_nets
    }
    missing: dict[str, dict[str, models.CircuitRequirement]] = {}
    for connection in spec.connections:
        if connection.signal_type in {"power", "ground"}:
            continue
        source = bindings[connection.from_block]
        target = bindings[connection.to_block]
        for net in source.keys() & target.keys():
            source_sheets = {owner.sheet for owner in source[net]}
            target_sheets = {owner.sheet for owner in target[net]}
            if len(source_sheets) != 1 or len(target_sheets) != 1:
                continue
            if source_sheets == target_sheets:
                continue
            if source_sheets | target_sheets <= boundaries.get(net, set()):
                continue
            owners = missing.setdefault(net, {})
            owners.update((owner.id, owner) for owner in source[net])
            owners.update((owner.id, owner) for owner in target[net])
    if not missing:
        return
    evidence = []
    for net, owners in sorted(missing.items()):
        owner_labels = ", ".join(
            f"{owner.id!r} on sheet {owner.sheet!r}"
            for owner in sorted(owners.values(), key=lambda owner: owner.id)
        )
        declared_sheets = boundaries.get(net, set())
        missing_sheets = sorted({owner.sheet for owner in owners.values()} - declared_sheets)
        evidence.append(
            f"net {net!r}: signal-linked typed owners {owner_labels}; "
            f"declared inter-sheet endpoints {sorted(declared_sheets)}, "
            f"missing owning endpoints {missing_sheets}"
        )
    message = (
        "Typed signal bindings cross functional-owner sheets without a complete "
        "inter_sheet_nets contract. Declare each named net with its owning sheet endpoints "
        "before BOM/wiring; preserve the typed port bindings. " + "; ".join(evidence)
    )
    raise StageSchemaError(
        message,
        diagnostic={
            "code": "missing_typed_inter_sheet_contract",
            "message": message,
            "evidence": evidence,
        },
    )


def _complete_hub75_optional_address(payload: dict) -> dict:
    """O8: tie the HUB75 `addr_d` channel low when the model leaves it unused.

    `addr_d` is the 13th address line a 1/32-scan panel needs; a 1/8- or
    1/16-scan panel does not. Under the `addr_d_optional` arm an interface that
    omits it gets the channel bound to GND, so the spare '245 input is tied low
    (never floating) and the channel's buffer output keeps driving the connector
    position — instead of failing the whole stage on an undeclared HUB75_D net.
    """
    requirements = []
    for row in payload.get("requirements") or []:
        if not isinstance(row, dict):
            requirements.append(row)
            continue
        requirement = dict(row)
        family = re.sub(r"[^a-z0-9]+", "", str(row.get("family") or "").lower())
        ports = dict(row.get("ports") or {})
        if family == "hub75levelshiftinterface" and not ports.get("addr_d"):
            ports["addr_d"] = "GND"
            requirement["ports"] = ports
        requirements.append(requirement)
    return {**payload, "requirements": requirements}


def _declared_net_names(payload: dict) -> set[str]:
    names = {"GND"}
    names.update(str(net) for net in payload.get("power_nets") or [])
    names.update(str(net) for net in (payload.get("rail_voltages") or {}))
    names.update(
        str(net["name"])
        for net in payload.get("inter_sheet_nets") or []
        if isinstance(net, dict) and net.get("name")
    )
    for net_range in payload.get("inter_sheet_net_ranges") or []:
        if not isinstance(net_range, dict):
            continue
        pattern = str(net_range.get("name_pattern") or "")
        try:
            numbers = range(int(net_range["start"]), int(net_range["end"]) + 1)
        except (KeyError, TypeError, ValueError):
            continue
        names.update(pattern.replace("{n}", str(number)) for number in numbers)
    return names


def _recipe_ports_for_requirement(family: str, exact_part: object):
    """The recipe ports a requirement's family/exact part selects (or nothing)."""
    from kicraft.design.recipes.registry import registered_recipes
    from kicraft.design.recipes.resolver import _family_recipes, _recipe_for_exact

    recipes = tuple(registered_recipes())
    matches = list(_family_recipes(str(family or ""), recipes))
    exact = _recipe_for_exact(exact_part, recipes)
    if exact is not None and exact not in matches:
        matches.append(exact)
    return matches[0].definition.ports if matches else ()


def _complete_bound_port_nets(payload: dict) -> dict:
    """O9: declare the nets a port binding already names.

    A recipe port bound to a net the architecture never declares is the dominant
    rung-1 rejection (`unknown_recipe_port_net`): the model asserts "this port is
    on net X" and then loses the stage on a declaration it omitted. Under the
    `bound_nets` arm the architecture completes what it was told: net X is
    declared with the endpoint its requiring sheet needs, plus the peer sheets
    any other requirement binding X already implies. A signal that leaves the
    board from one sheet (`data_out` on a WS2812 string driver, whose brief says
    "include an output for driving addressable LED string") additionally gets the
    physical output connector the net needs to have a pin — otherwise the net is
    one-pin and the build fails on a dangling net, which is exactly the defect it
    would have papered over.

    Nothing is invented: the net name comes from the binding, the endpoints from
    the requiring/peer sheets, and the connector binds the driver's own rails.
    Requirements whose recipe cannot be resolved, power ports, and nets already
    covered by a declaration are all left to the existing diagnostics.
    """
    requirements = list(payload.get("requirements") or [])
    if not requirements:
        return payload
    declared = _declared_net_names(payload)
    nets = list(payload.get("inter_sheet_nets") or [])
    added_nets: list[dict] = []
    added_requirements: list[dict] = []
    for row in requirements:
        if not isinstance(row, dict):
            continue
        ports = {}
        for port in _recipe_ports_for_requirement(row.get("family"), row.get("exact_part")):
            ports[port.name] = port
        bindings = row.get("ports") or {}
        for name, port in ports.items():
            net = bindings.get(name)
            if not net or net in declared or port.direction == "power":
                continue
            if re.fullmatch(r"[+-]?\d+(\.\d+)?V(\d+)?", str(net)):
                continue  # a rail written as a name, not a declaration to invent
            peer_sheets = sorted(
                {
                    str(other.get("sheet"))
                    for other in requirements
                    if isinstance(other, dict)
                    and other is not row
                    and net in (other.get("ports") or {}).values()
                }
                - {str(row.get("sheet"))}
            )
            endpoints = [
                {"sheet": row.get("sheet"), "direction": port.direction},
                *(  # a peer sheet's direction is not ours to claim
                    {"sheet": sheet, "direction": "bidirectional"} for sheet in peer_sheets
                ),
            ]
            if not peer_sheets and port.direction == "output":
                requirement_id = f"{row.get('id')}_{name}_connector"
                if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", requirement_id):
                    continue
                connector_ports = {"pin1": net}
                for rail_port in ("gnd", "vdd", "vdd_5v"):
                    rail = bindings.get(rail_port)
                    if rail:
                        connector_ports[f"pin{len(connector_ports) + 1}"] = rail
                added_requirements.append(
                    {
                        "id": requirement_id,
                        "sheet": row.get("sheet"),
                        "role": "connector",
                        "family": "pin-header",
                        "parameters": {"rows": 1, "gender": "male"},
                        "ports": connector_ports,
                        "interfaces": [],
                        "functional_blocks": list(row.get("functional_blocks") or []),
                    }
                )
                endpoints.append({"sheet": row.get("sheet"), "direction": "input"})
            if len(endpoints) < 2:
                continue  # an input with no peer keeps its diagnostic
            added_nets.append({"name": str(net), "endpoints": endpoints})
            declared.add(str(net))
    if not added_nets and not added_requirements:
        return payload
    return {
        **payload,
        "inter_sheet_nets": [*nets, *added_nets],
        "requirements": [*requirements, *added_requirements],
    }


def _normalize_stage_response(
    stage: str,
    payload: dict,
    prompt_state: dict,
    *,
    ladder: frozenset[str] = frozenset(),
) -> tuple[dict, int]:
    try:
        questions = payload.get("questions")
        if isinstance(questions, list) and questions:
            return (
                StageQuestionResponse.model_validate({"questions": questions}).model_dump(
                    exclude_none=True
                ),
                0,
            )
        if isinstance(questions, list):
            # The strict provider envelope requires the key on every answer, so a
            # slot response carries `questions: []`. It is not part of the slot.
            payload = {key: value for key, value in payload.items() if key != "questions"}
        if stage == "intent":
            return IntentStageResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "architecture":
            payload = _normalize_architecture_sheet_aliases(payload)
            payload = _complete_connector_requirements(payload)
            payload = _normalize_usb_c_requirements(
                payload, complete_native_usb="completing" in ladder
            )
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
                    if is_part_family(str(named_part)):
                        continue
                    named_identity = re.sub(r"[^a-z0-9]+", "", str(named_part).lower())
                    if named_identity and named_identity in requirement_identity:
                        requirement["exact_part"] = str(named_part)
                        break
            # These are server-derived persisted fields, not provider claims.
            for field in (
                "recipe_resolution",
                "unresolved_requirement_ids",
                "protected_identities",
            ):
                payload.pop(field, None)
            if "addr_d_optional" in ladder:
                payload = _complete_hub75_optional_address(payload)
            if "bound_nets" in ladder:
                payload = _complete_bound_port_nets(payload)
            response = ArchitectureStageResponse.model_validate(payload)
            _validate_lowerer_parameter_contracts(response)
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
            _validate_power_requirement_contracts(canonical, prompt_state)
            from kicraft.design.recipes import (
                RecipeResolutionError,
                apply_architecture_recipe_resolution,
            )

            validated = models.Architecture.model_validate(canonical)
            try:
                resolved = apply_architecture_recipe_resolution(
                    validated,
                    prompt_state.get("intent") or {},
                    complete_native_usb="completing" in ladder,
                )
            except RecipeResolutionError as exc:
                evidence = [row.model_dump(exclude_none=True) for row in exc.diagnostics]
                diagnostic = (
                    evidence[0]
                    if len(evidence) == 1
                    else {
                        "code": "multiple_recipe_contracts",
                        "message": str(exc),
                        "evidence": evidence,
                    }
                )
                raise StageSchemaError(str(exc), diagnostic=diagnostic) from exc
            resolved = _fold_recipe_covered_sheets(resolved)
            _validate_typed_inter_sheet_contracts(resolved, prompt_state.get("functional_spec"))
            return resolved.model_dump(exclude_none=True), len(expanded)
        if stage == "bom":
            return _normalize_bom_stage_response(payload, prompt_state)
        if stage == "wiring":
            return _normalize_wiring_stage_response(payload, prompt_state), 0
        return SLOT_MODEL[stage].model_validate(payload).model_dump(exclude_none=True), 0
    except StageSchemaError:
        raise
    except (TypeError, ValueError) as exc:
        diagnostic = getattr(exc, "diagnostic", None)
        raise StageSchemaError(
            str(exc),
            diagnostic=(
                diagnostic.model_dump(exclude_none=True) if diagnostic is not None else None
            ),
        ) from exc


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
