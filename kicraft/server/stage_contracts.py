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
from .question_policy import normalize_question_options


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

    @model_validator(mode="after")
    def _normalize_ordinary_question_options(self):
        self.questions = [
            question
            if question.reconcile_target == "bom"
            else question.model_copy(
                update={"options": normalize_question_options(question.options)}
            )
            for question in self.questions
        ]
        return self


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
    assembly: bool = True
    mpn: str | None = None
    datasheet: str | None = None
    sourcing_note: str | None = None
    side: Literal["front", "back"] | None = None


def _requirement_owns_protected_group(group: BomComponentGroup, requirements) -> bool:
    """Match an implementing component to its typed requirement, not just its label.

    BOM groups are model-facing and therefore cannot carry provenance.  Canonical
    :class:`~kicraft.design.models.BomPart` rows can: when a deterministic lowerer
    made such a row, re-derive that lowerer's artifact before treating its ownership
    stamp as evidence.  This makes a stamped FPC owner visible without allowing a
    model-authored ``recipe_id`` or ``resolution_id`` to waive physical identity.
    """

    def value(row, field: str):
        if isinstance(row, dict):
            return row.get(field)
        return getattr(row, field, None)

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

    def has_verified_lowerer_provenance(requirement) -> bool:
        if value(group, "resolution_source") != "lowerer":
            return False
        if value(group, "lowering_requirement_id") != value(requirement, "id"):
            return False
        lowerer_id = value(group, "resolution_id")
        if not lowerer_id:
            return False
        try:
            from kicraft.design.lowering import lower_requirement
            from kicraft.design.models import CircuitRequirement

            artifact = lower_requirement(CircuitRequirement.model_validate(requirement))
        except (TypeError, ValueError):
            return False
        if artifact is None or artifact.lowerer_id != lowerer_id:
            return False
        prefix = re.match(r"[A-Z]+", str(value(group, "ref") or ""))
        index = value(group, "lowering_index")
        if prefix is None or not isinstance(index, int):
            return False
        return any(
            candidate.reference_prefix == prefix.group()
            and value(group, "lowering_role") == candidate.role
            and 0 <= index < candidate.quantity
            and candidate.value == value(group, "value")
            and candidate.symbol == value(group, "symbol")
            and candidate.footprint == value(group, "footprint")
            and candidate.mpn == value(group, "mpn")
            for candidate in artifact.groups
        )

    def has_verified_recipe_provenance(requirement) -> bool:
        if value(group, "resolution_source") != "recipe":
            return False
        recipe_id = value(group, "recipe_id")
        if not recipe_id or value(group, "resolution_id") != recipe_id:
            return False
        try:
            from kicraft.design.recipes.registry import get_recipe

            definition = get_recipe(str(recipe_id))
        except (KeyError, ValueError):
            return False
        role = value(group, "recipe_role")
        ref = str(value(group, "ref") or "")
        prefix = re.match(r"[A-Z]+", ref)
        if (
            not role
            or prefix is None
            or not any(
                candidate.role == role
                and candidate.reference_prefix == prefix.group()
                and candidate.value == value(group, "value")
                and candidate.symbol == value(group, "symbol")
                and candidate.footprint == value(group, "footprint")
                and candidate.mpn == value(group, "mpn")
                for candidate in definition.parts
            )
        ):
            return False
        exact_part = value(requirement, "exact_part")
        if exact_part:
            return bool(
                definition.exact_part
                and matches_part_identity(str(exact_part), definition.exact_part)
            )
        family = str(value(requirement, "family") or "")
        return bool(
            definition.family
            and family
            and (
                matches_part_identity(family, definition.exact_part or "")
                or token(family) == token(definition.family)
            )
        )

    from kicraft.design.recipes import registered_recipes

    physical_identity = group.mpn or group.value
    component_tokens = {token(physical_identity)} - {""}
    component_families = [
        words(registered.definition.family) - {"fixed", "minimal"}
        for registered in registered_recipes()
        if token(registered.definition.exact_part) in component_tokens
    ]
    for requirement in requirements or ():
        if has_verified_lowerer_provenance(requirement) or has_verified_recipe_provenance(
            requirement
        ):
            return True
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


def _recipe_parts_with_identity(expansions, get_recipe) -> list[models.BomPart]:
    """Every recipe's parts, each carrying the curated identity the design named.

    §9.33 requires each ``Requirement.exact_part`` to appear in the BOM's own text or in
    its substitution/assumption ledger. A recipe's identity is not always the orderable
    part it ships (``HUB75-SN74HCT245`` ships ``SN74HCT245PWR-JSM``) and nothing was
    substituted — the recipe *is* the part — so the pairing is recorded on the part that
    carries the identity. Without it a BOM whose sheets are all recipe- or
    lowerer-covered has no provider call in which to write that ledger, and the gate
    fails a board that is in fact exactly what the architecture named.
    """
    parts: list[models.BomPart] = []
    for expansion in expansions:
        own = list(expansion.parts)
        parts.extend(own)
        try:
            definition = get_recipe(expansion.selection.recipe)
        except (KeyError, ValueError):
            continue
        identity = definition.exact_part
        if not identity or not own:
            continue
        if any(
            identity.lower() in f"{part.value} {part.mpn or ''} {part.sourcing_note or ''}".lower()
            for part in own
        ):
            continue
        tail = identity.lower().rsplit("-", 1)[-1]
        index = next(
            (
                position
                for position, part in enumerate(own)
                if tail and tail in f"{part.value} {part.mpn or ''}".lower()
            ),
            0,
        )
        carrier = own[index]
        parts[len(parts) - len(own) + index] = carrier.model_copy(
            update={
                "sourcing_note": (
                    f"curated recipe {expansion.selection.recipe} ships "
                    f"{carrier.value or carrier.mpn} for identity {identity}"
                )
            }
        )
    return parts


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
    from kicraft.design.recipes.registry import get_recipe, next_reference_numbers

    architecture = (prompt_state or {}).get("architecture") or {}
    trusted_lowering_groups = set((prompt_state or {}).get("_trusted_lowering_group_ids") or ())
    expansions = expand_selections(architecture.get("recipe_selections") or [])
    recipe_parts = _recipe_parts_with_identity(expansions, get_recipe)
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

    ``oneOf`` is forbidden *anywhere* in a strict schema, and Pydantic emits it
    for every discriminated union (the typed ``RequirementObligation`` union is
    the one that reaches the intent/functional_spec/architecture slots). Its
    branches are disjoint by their literal ``kind`` discriminator, so the
    provider envelope rewrites ``oneOf`` to ``anyOf`` -- otherwise OpenAI 400s
    ``invalid_json_schema`` before the model ever runs.

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
            out[key] = {
                name: _strict_provider_schema(subschema) for name, subschema in value.items()
            }
        elif key in ("items", "additionalProperties", "not", "contains"):
            out[key] = _strict_provider_schema(value)
        elif key in ("anyOf", "oneOf") and isinstance(value, list):
            # OpenAI strict structured outputs accept `anyOf` and reject `oneOf`.
            # Merge rather than overwrite so a node carrying both keys keeps every
            # branch (Pydantic never emits both; this only guards the merge).
            out.setdefault("anyOf", []).extend(_strict_provider_schema(item) for item in value)
        elif key in ("allOf", "prefixItems") and isinstance(value, list):
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
        # The architecture stage asks for intent; the canonical shape (net names,
        # port bindings, endpoints, connector exposure) is derived from it
        # (`kicraft.design.architecture_intent`), never hand-written by the model.
        from kicraft.design.architecture_intent import ArchitectureIntent

        schema = ArchitectureIntent.model_json_schema()
        properties = schema["properties"]
        properties["requirements"]["minItems"] = 1
        schema["required"] = [*schema.get("required", []), "requirements"]
        requirement = schema["$defs"]["IntentRequirement"]
        requirement["properties"]["functional_blocks"]["minItems"] = 1
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
    slot_schema = dict(_slot_response_schema(stage))
    question = dict(StageQuestionResponse.model_json_schema())
    definitions = {
        **(slot_schema.pop("$defs", {}) or {}),
        **(question.pop("$defs", {}) or {}),
    }
    properties = dict(slot_schema.get("properties") or {})
    questions = dict((question.get("properties") or {}).get("questions") or {})
    questions["minItems"] = 0
    properties["questions"] = questions
    required = [*slot_schema.get("required", [])]
    if "questions" not in required:
        required.append("questions")
    schema = {**slot_schema, "type": "object", "properties": properties, "required": required}
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
    if stage in {"functional_spec", "architecture"}:
        source_slots = ("intent",) if stage == "functional_spec" else ("intent", "functional_spec")
        source_keys = {
            (row["kind"], row["original_obligation_id"])
            for slot in source_slots
            for row in (prompt_state.get(slot) or {}).get("obligations") or []
        }
        if source_keys:
            for variant in schema.get("anyOf") or [schema]:
                properties = variant.get("properties") or {}
                if "obligations" not in properties:
                    continue
                properties["obligations"]["minItems"] = len(source_keys)
                required = list(variant.get("required") or [])
                if "obligations" not in required:
                    required.append("obligations")
                variant["required"] = required
    if stage == "architecture":
        functional_spec = prompt_state.get("functional_spec")
        if isinstance(functional_spec, dict):
            spec = models.FunctionalSpec.model_validate(functional_spec)
            names = [block.name for block in spec.blocks]
            if not names:
                raise ValueError("architecture response contract requires functional block names")
            ownership = schema["$defs"]["IntentRequirement"]["properties"]["functional_blocks"]
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


def _normalize_board_outline_obligation_rows(rows) -> list:
    """Retype only semantically proven board-outline pseudo-parts as fabrication facts."""
    from kicraft.design.part_identity import board_outline_fabrication_feature

    normalized = []
    for row in rows or []:
        if not isinstance(row, dict) or row.get("kind") != "physical":
            normalized.append(row)
            continue
        feature = board_outline_fabrication_feature(str(row.get("component_class") or ""))
        normalized.append(
            {
                "kind": "fabrication",
                "original_obligation_id": row.get("original_obligation_id"),
                "feature": feature,
            }
            if feature is not None
            else row
        )
    return normalized


def normalize_board_outline_obligations(payload: dict) -> dict:
    """Normalize board-outline facts in source and requirement rows before ownership checks."""
    normalized = dict(payload)
    if "obligations" in normalized:
        normalized["obligations"] = _normalize_board_outline_obligation_rows(
            normalized["obligations"]
        )
    requirements = normalized.get("requirements")
    if isinstance(requirements, list):
        normalized["requirements"] = [
            {
                **requirement,
                "obligations": _normalize_board_outline_obligation_rows(
                    requirement.get("obligations")
                ),
            }
            if isinstance(requirement, dict) and "obligations" in requirement
            else requirement
            for requirement in requirements
        ]
    return normalized


def _canonical_obligations(rows) -> list[dict]:
    """Validate obligation rows and normalise them for comparison."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(list[models.RequirementObligation])
    return [
        row.model_dump(mode="json", exclude_none=True)
        for row in adapter.validate_python(_normalize_board_outline_obligation_rows(rows))
    ]


def source_obligation_rows(prompt_state: dict, *, slots: tuple[str, ...]) -> list[dict]:
    """The typed obligations the named committed stages carry, in declaration order.

    Raises when two committed stages disagree about one obligation: that is a defect in the
    owning stage, not something a later stage can repair.
    """
    rows: list[dict] = []
    by_key: dict[tuple[str, str], dict] = {}
    for slot in slots:
        for row in _canonical_obligations((prompt_state.get(slot) or {}).get("obligations")):
            key = (row["kind"], row["original_obligation_id"])
            if key in by_key:
                if by_key[key] != row:
                    raise StageSchemaError(
                        f"source obligation {key!r} disagrees between committed stages",
                        diagnostic={
                            "code": "conflicting_source_obligation",
                            "message": "Committed source obligations disagree; repair the owning stage.",
                            "evidence": [by_key[key], row],
                        },
                    )
                continue
            by_key[key] = row
            rows.append(row)
    return rows


def restore_source_obligations(payload: dict, prompt_state: dict) -> dict:
    """Write the architecture's top-level `obligations` from the committed intent/spec set.

    That list is the design's typed obligation set, and the draft's job is to say *where* each row
    is implemented, so the verbatim copy is the compiler's to write. What the draft has to get right
    — every committed row attached to the requirement that implements it — is checked by
    `validate_obligation_retention` below.
    """
    try:
        rows = source_obligation_rows(prompt_state, slots=("intent", "functional_spec"))
    except ValueError:
        return payload  # an unreadable source row is reported by validate_obligation_retention
    if not rows:
        return payload
    return {**payload, "obligations": rows}


def _requirement_proves_physical_obligation(requirement: dict, component_class: str) -> bool:
    """Whether reviewed compiler evidence proves this requirement realizes one class."""
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.part_identity import (
        canonical_physical_features,
        physical_inventory_record,
        reviewed_part,
    )
    from kicraft.design.recipes import registered_recipes

    wanted = canonical_physical_features(component_class)
    if not wanted:
        return False

    def realizes(*, mpn=None, symbol=None, footprint=None) -> bool:
        record = physical_inventory_record(mpn=mpn, symbol=symbol, footprint=footprint)
        return record is not None and bool(wanted & record.physical_features)

    exact_part = str(requirement.get("exact_part") or "").strip()
    exact_record = reviewed_part(exact_part) if exact_part else None
    if exact_record is not None and wanted & exact_record.physical_features:
        return True

    family = str(requirement.get("family") or "")
    for registered in registered_recipes():
        definition = registered.definition
        if definition.family != family or (
            exact_part
            and definition.exact_part
            and definition.exact_part.casefold() != exact_part.casefold()
        ):
            continue
        if exact_part and definition.exact_part and realizes(mpn=definition.exact_part):
            return True
        if any(
            realizes(mpn=part.mpn, symbol=part.symbol, footprint=part.footprint)
            for part in definition.parts
        ):
            return True

    try:
        artifact = lower_requirement(requirement)
    except (TypeError, ValueError):
        return False
    return artifact is not None and any(
        realizes(mpn=group.mpn, symbol=group.symbol, footprint=group.footprint)
        for group in artifact.groups
    )


def physical_obligation_candidate_requirement_ids(payload: dict, row: dict) -> list[str]:
    """Sorted requirements whose reviewed recipe/lowerer evidence realizes one physical row."""
    component_class = str(row.get("component_class") or "")
    if row.get("kind") != "physical" or not component_class:
        return []
    return sorted(
        str(requirement.get("id"))
        for requirement in payload.get("requirements") or []
        if isinstance(requirement, dict)
        and _requirement_proves_physical_obligation(requirement, component_class)
    )


def attach_uniquely_provable_physical_obligations(payload: dict, prompt_state: dict) -> dict:
    """Attach an omitted physical row only when one reviewed implementation proves it."""
    try:
        source_rows = source_obligation_rows(prompt_state, slots=("intent", "functional_spec"))
    except (StageSchemaError, ValueError):
        return payload
    normalized = dict(payload)
    requirements = list(normalized.get("requirements") or [])
    owned = {
        (row["kind"], row["original_obligation_id"])
        for requirement in requirements
        if isinstance(requirement, dict)
        for row in _canonical_obligations(requirement.get("obligations"))
    }
    derived_notes: list[str] = []
    for row in source_rows:
        key = (row["kind"], row["original_obligation_id"])
        if row["kind"] != "physical" or key in owned:
            continue
        candidate_ids = physical_obligation_candidate_requirement_ids(normalized, row)
        if len(candidate_ids) != 1:
            continue
        requirement_id = candidate_ids[0]
        for index, requirement in enumerate(requirements):
            if isinstance(requirement, dict) and requirement.get("id") == requirement_id:
                requirements[index] = {
                    **requirement,
                    "obligations": [*(requirement.get("obligations") or []), row],
                }
                owned.add(key)
                derived_notes.append(
                    f"{requirement_id}: physical obligation {row['original_obligation_id']!r} "
                    "attached from unique reviewed recipe/lowerer evidence (derived)"
                )
                break
    if not derived_notes:
        return normalized
    normalized["requirements"] = requirements
    normalized["assumptions"] = [*(normalized.get("assumptions") or []), *derived_notes]
    return normalized


def validate_obligation_retention(stage: str, payload: dict, prompt_state: dict) -> None:
    """Reject a slot that drops or misplaces the obligations the committed stages carry.

    `functional_spec` must repeat the committed rows verbatim; `architecture` carries the same rows
    at the top level (written by `restore_source_obligations`) and must attach each one to the
    requirement that implements it.
    """
    if stage not in {"functional_spec", "architecture"}:
        return
    source_slots = ("intent",) if stage == "functional_spec" else ("intent", "functional_spec")
    expected = {
        (row["kind"], row["original_obligation_id"]): row
        for row in source_obligation_rows(prompt_state, slots=source_slots)
    }
    if not expected:
        return
    if stage == "architecture":
        # The top-level list is written from the sources before this runs; what the draft owns is
        # where each obligation is implemented, so check the requirement rows instead of asking the
        # model to copy the same rows twice.
        owned: dict[tuple[str, str], int] = {}
        for requirement in payload.get("requirements") or []:
            if not isinstance(requirement, dict):
                continue
            for row in _canonical_obligations(requirement.get("obligations")):
                key = (row["kind"], row["original_obligation_id"])
                owned[key] = owned.get(key, 0) + 1
        # Only board-wide facts may remain ownerless. Quantitative rows qualify only when their
        # subject and unit prove they are board-outline geometry; electrical and part limits stay
        # on their realizing requirement.
        unowned = [
            row
            for key, row in expected.items()
            if key not in owned and models.obligation_requires_requirement_owner(row)
        ]
        if unowned:
            raise StageSchemaError(
                "source obligations must be owned by a requirement",
                diagnostic={
                    "code": "source_obligation_not_retained",
                    "message": (
                        "Attach every committed typed obligation to the requirement that implements "
                        "it through `requirements[].obligations`; the architecture's top-level "
                        "`obligations` list is written from the committed intent and functional "
                        "spec, and a paraphrased copy is restored from the committed row. A "
                        "`quantity`, `fabrication`, or `negative` board-wide fact may stay at the "
                        "top level. A `quantitative` row may do so only when it explicitly measures "
                        "a board/PCB outline in a geometric unit or names the board's own build "
                        "stack-up in `layers`/`plies`; electrical and part limits need "
                        "their realizing requirement."
                    ),
                    "evidence": unowned,
                    "candidate_requirement_ids": {
                        row[
                            "original_obligation_id"
                        ]: physical_obligation_candidate_requirement_ids(payload, row)
                        for row in unowned
                        if row["kind"] == "physical"
                    },
                },
            )
        return
    actual = {}
    duplicates = []
    for row in _canonical_obligations(payload.get("obligations")):
        key = (row["kind"], row["original_obligation_id"])
        if key in actual:
            duplicates.append(row)
        actual[key] = row
    missing_or_changed = [row for key, row in expected.items() if actual.get(key) != row]
    if missing_or_changed or duplicates:
        raise StageSchemaError(
            "mandatory source obligations were omitted, changed, or duplicated",
            diagnostic={
                "code": "source_obligation_not_retained",
                "message": "Retain the complete original typed obligation at this stage.",
                "evidence": missing_or_changed + duplicates,
            },
        )


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


def _apply_authoritative_standard_form_factor(payload: dict, prompt_state: dict) -> dict:
    """Map the original user-owned standard into the architecture response."""

    from kicraft.design.architecture_intent import apply_authoritative_standard_form_factor

    try:
        return apply_authoritative_standard_form_factor(payload, prompt_state.get("intent"))
    except ValueError as exc:
        candidate = payload.get("standard_form_factor")
        intent = prompt_state.get("intent")
        form_factor = intent.get("form_factor") if isinstance(intent, dict) else None
        standard = form_factor.get("standard") if isinstance(form_factor, dict) else None
        raise StageSchemaError(
            str(exc),
            diagnostic={
                "code": "contradictory_standard_form_factor",
                "message": str(exc),
                "evidence": [f"intent={standard}", f"architecture={candidate}"],
            },
        ) from exc


def _intent_shaped(payload: dict) -> bool:
    """An intent-shaped architecture answer: it declares signals and no canonical net list."""
    return "signals" in payload and "inter_sheet_nets" not in payload


def _aggregate_intent_diagnostic(exc) -> dict:
    """One durable row for a refusal that names several contract findings.

    A single finding is its own row. Two or more are wrapped, and the wrapper must still be a
    `StageDiagnostic`: an aggregate that named no severity and left the rows in ``evidence`` as
    bare dicts made *every* state saved after an architecture refusal with two or more defects
    fail ``ConversationState.model_validate`` (live runs KC-HPD3YF and KC-P4E2PH, 119 states in
    the corpus) -- so ``stage_driver replay`` could not open the runs it exists to iterate on.
    The wrapper keeps the findings as typed `findings` rows and repeats their headline text in
    ``evidence``, which is the shape every reader already accepts.
    """
    rows = [row.model_dump(exclude_none=True) for row in exc.diagnostics]
    if len(rows) == 1:
        return rows[0]
    return {
        "code": "multiple_intent_contracts",
        "severity": "repair_required",
        "message": str(exc),
        "evidence": [f"{row.code}: {row.message}" for row in exc.diagnostics],
        "findings": rows,
    }


def _derive_intent_payload(payload: dict, functional_spec: object | None = None) -> dict:
    """Intent slot -> canonical slot; every refusal is carried as one diagnostic.

    ``functional_spec`` is the committed spec slot, when there is one. A board
    feature the spec names as its own block (the prototyping pad field) has no
    architecture counterpart to claim it, so the derivation needs the spec to
    give the derived requirement that block's exact name; with no spec, or with
    no matching block, the derived requirement owns none.
    """
    from kicraft.design.architecture_intent import ArchitectureIntentError, derive_architecture

    try:
        return derive_architecture(payload, functional_spec).model_dump(exclude_none=True)
    except ArchitectureIntentError as exc:
        raise StageSchemaError(
            str(exc), diagnostic=_aggregate_intent_diagnostic(exc)
        ) from exc


def _schema_error_detail(exc: Exception) -> str:
    """Name the fix when a schema error is an unknown slot field.

    The provider schema is generated from the slot models, so an ``extra_forbidden`` error is a key
    the draft invented; the repair is to delete it, which the pydantic loc/message does not say.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        unknown = sorted(
            {
                ".".join(str(part) for part in row.get("loc") or ()) or "<root>"
                for row in exc.errors()
                if row.get("type") == "extra_forbidden"
            }
        )
        if unknown:
            return (
                f"{exc} — unknown slot field(s): {', '.join(unknown)}. The slot accepts only the "
                "fields the published schema lists: remove each unknown key (nested keys are named "
                "by their path) and re-emit."
            )
    return str(exc)


def _normalize_stage_response(
    stage: str,
    payload: dict,
    prompt_state: dict,
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
        payload = normalize_board_outline_obligations(payload)
        if stage == "architecture":
            payload = _apply_authoritative_standard_form_factor(payload, prompt_state)
            payload = attach_uniquely_provable_physical_obligations(payload, prompt_state)
            payload = restore_source_obligations(payload, prompt_state)
        validate_obligation_retention(stage, payload, prompt_state)
        if stage == "intent":
            return IntentStageResponse.model_validate(payload).model_dump(exclude_none=True), 0
        if stage == "architecture":
            # These are server-derived persisted fields, not provider claims. Rebuild
            # the mapping instead of popping: a caller that re-normalizes its own
            # previous result must not see it mutated out from under it.
            server_derived = (
                "recipe_resolution",
                "unresolved_requirement_ids",
                "protected_identities",
                "declared_interfaces",
            )
            payload = {key: value for key, value in payload.items() if key not in server_derived}
            if _intent_shaped(payload):
                # The answer states the design; the canonical shape (net names, port
                # bindings, endpoints, connector exposure) is derived from it here.
                payload = _derive_intent_payload(payload, prompt_state.get("functional_spec"))
            response = models.Architecture.model_validate(payload)
            _validate_lowerer_parameter_contracts(response)
            canonical = response.model_dump(exclude_none=True)
            _validate_power_requirement_contracts(canonical, prompt_state)
            from kicraft.design.recipes import (
                RecipeResolutionError,
                apply_architecture_recipe_resolution,
            )

            try:
                resolved = apply_architecture_recipe_resolution(
                    canonical,
                    prompt_state.get("intent") or {},
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
            return resolved.model_dump(exclude_none=True), 0
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
            _schema_error_detail(exc),
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
