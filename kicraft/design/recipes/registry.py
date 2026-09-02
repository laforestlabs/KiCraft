"""Exact, deterministic circuit-recipe registry and expansion."""

from __future__ import annotations

import re
from collections import defaultdict

from kicraft.design.models import (
    BomPart,
    EdgeInterface,
    NetConnection,
    PinEndpoint,
    RecipeSelection,
)

from .models import RecipeDefinition, RecipeExpansion

_REGISTRY: dict[str, RecipeDefinition] = {}


def register_recipe(definition: RecipeDefinition) -> None:
    if definition.recipe in _REGISTRY:
        raise ValueError(f"duplicate recipe {definition.recipe!r}")
    _REGISTRY[definition.recipe] = definition


def get_recipe(recipe: str) -> RecipeDefinition:
    try:
        return _REGISTRY[recipe]
    except KeyError as exc:
        raise ValueError(f"unknown circuit recipe {recipe!r}") from exc


def recipe_summaries() -> list[dict]:
    return [
        {
            "recipe": definition.recipe,
            "required_sheet_roles": list(definition.required_sheet_roles),
            "parameters": definition.parameter_defaults,
        }
        for definition in sorted(_REGISTRY.values(), key=lambda item: item.recipe)
    ]


def _validated_parameters(definition: RecipeDefinition, selection: RecipeSelection) -> dict:
    unknown = set(selection.parameters) - set(definition.parameter_defaults)
    if unknown:
        raise ValueError(f"recipe {definition.recipe} has unknown parameters: {sorted(unknown)}")
    parameters = {**definition.parameter_defaults, **selection.parameters}
    for name, allowed in definition.allowed_parameters.items():
        if parameters.get(name) not in allowed:
            raise ValueError(
                f"recipe {definition.recipe} parameter {name!r} must be one of {list(allowed)!r}"
            )
    missing_roles = set(definition.required_sheet_roles) - set(selection.sheets)
    extra_roles = set(selection.sheets) - set(definition.required_sheet_roles)
    if missing_roles or extra_roles:
        raise ValueError(
            f"recipe {definition.recipe} sheet roles mismatch; missing={sorted(missing_roles)}, "
            f"unknown={sorted(extra_roles)}"
        )
    return parameters


def expand_recipe(
    selection: RecipeSelection,
    *,
    next_reference: dict[str, int] | None = None,
) -> RecipeExpansion:
    definition = get_recipe(selection.recipe)
    parameters = _validated_parameters(definition, selection)
    del parameters  # validation is the only generic behavior; definitions are immutable.
    next_number: dict[str, int] = defaultdict(lambda: 1, next_reference or {})
    refs_by_role: dict[str, list[str]] = {}
    parts: list[BomPart] = []
    for group in definition.parts:
        refs = [
            f"{group.reference_prefix}{number}"
            for number in range(
                next_number[group.reference_prefix],
                next_number[group.reference_prefix] + group.quantity,
            )
        ]
        next_number[group.reference_prefix] += group.quantity
        if next_reference is not None:
            next_reference[group.reference_prefix] = next_number[group.reference_prefix]
        refs_by_role[group.role] = refs
        for ref in refs:
            parts.append(
                BomPart(
                    ref=ref,
                    value=group.value,
                    symbol=group.symbol,
                    footprint=group.footprint,
                    sheet=selection.sheets[group.sheet_role],
                    assembly=group.assembly,
                    recipe_id=definition.recipe,
                    recipe_instance=selection.instance,
                    recipe_role=group.role,
                )
            )
    by_sheet_net: dict[tuple[str, str], list[PinEndpoint]] = defaultdict(list)
    part_by_ref = {part.ref: part for part in parts}
    for spec in definition.pins:
        refs = refs_by_role.get(spec.role)
        if not refs:
            raise ValueError(f"recipe pin references unknown role {spec.role!r}")
        if spec.index >= len(refs):
            raise ValueError(
                f"recipe pin role {spec.role!r} index {spec.index} exceeds {len(refs)} parts"
            )
        ref = refs[spec.index]
        by_sheet_net[(part_by_ref[ref].sheet, spec.net)].append(PinEndpoint(ref=ref, pin=spec.pin))
    connections = [
        NetConnection(sheet=sheet, net_name=net, endpoints=endpoints)
        for (sheet, net), endpoints in sorted(by_sheet_net.items())
    ]
    no_connect_pins: list[PinEndpoint] = []
    for spec in definition.no_connects:
        refs = refs_by_role.get(spec.role)
        if not refs or spec.index >= len(refs):
            raise ValueError(f"recipe no-connect role {spec.role!r} index {spec.index} is invalid")
        no_connect_pins.append(PinEndpoint(ref=refs[spec.index], pin=spec.pin))
    edge_interfaces = [
        EdgeInterface(
            name=f"{selection.instance}:{edge.role}",
            refs=refs_by_role[edge.role],
            side=edge.side,
            pitch_mm=edge.pitch_mm,
            behavior="castellated",
        )
        for edge in definition.edges
    ]
    return RecipeExpansion(
        selection=selection,
        parts=parts,
        connections=connections,
        no_connect_pins=no_connect_pins,
        edge_interfaces=edge_interfaces,
    )


def expand_selections(selections) -> list[RecipeExpansion]:
    next_reference: dict[str, int] = {}
    return [
        expand_recipe(RecipeSelection.model_validate(selection), next_reference=next_reference)
        for selection in selections
    ]


def locked_pin_assignments(bom: dict) -> dict[tuple[str, str], str]:
    """Recipe-owned assignments reconstructed from canonical recipe provenance."""
    # Canonical BOM stores recipe IDs/instances, while recipe expansion itself is
    # deterministic. Reconstruct role/pin/net directly from definitions.
    parts = [part for part in bom.get("parts") or [] if isinstance(part, dict)]
    by_key_role: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for part in parts:
        if part.get("recipe_id") and part.get("recipe_instance") and part.get("recipe_role"):
            by_key_role[
                (
                    str(part["recipe_id"]),
                    str(part["recipe_instance"]),
                    str(part["recipe_role"]),
                )
            ].append(str(part["ref"]))
    locked: dict[tuple[str, str], str] = {}
    for (recipe_id, instance, role), refs in by_key_role.items():
        definition = get_recipe(recipe_id)
        for pin in definition.pins:
            if pin.role == role and pin.index < len(refs):
                locked[(refs[pin.index], pin.pin)] = pin.net
    return locked


def locked_no_connect_pins(bom: dict) -> set[tuple[str, str]]:
    """Recipe-owned deliberate no-connects reconstructed from provenance."""
    parts = [part for part in bom.get("parts") or [] if isinstance(part, dict)]
    by_key_role: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for part in parts:
        if part.get("recipe_id") and part.get("recipe_instance") and part.get("recipe_role"):
            by_key_role[
                (
                    str(part["recipe_id"]),
                    str(part["recipe_instance"]),
                    str(part["recipe_role"]),
                )
            ].append(str(part["ref"]))
    locked: set[tuple[str, str]] = set()
    for (recipe_id, _instance, role), refs in by_key_role.items():
        for spec in get_recipe(recipe_id).no_connects:
            if spec.role == role and spec.index < len(refs):
                locked.add((refs[spec.index], spec.pin))
    return locked


def next_reference_numbers(parts) -> dict[str, int]:
    result: dict[str, int] = {}
    for part in parts:
        match = re.match(r"([A-Z]+)([0-9]+)$", part.ref)
        if match:
            result[match.group(1)] = max(result.get(match.group(1), 1), int(match.group(2)) + 1)
    return result
