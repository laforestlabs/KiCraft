"""Exact, deterministic circuit-recipe registry and expansion."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from kicraft.design.models import (
    BomPart,
    EdgeInterface,
    NetConnection,
    PinEndpoint,
    PinOwnership,
    RecipeOwnershipManifest,
    RecipeSelection,
)

from .models import (
    RecipeDefinition,
    RecipeExpansion,
    RegisteredRecipe,
    ResolvedRecipeSelection,
)
from .pin_allocator import FIXED_INTERFACES, allocatable_capability_counts

_REGISTRY: dict[str, RegisteredRecipe] = {}
_SELECTORS: dict[str, str] = {}


def _identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def register_recipe(
    definition: RecipeDefinition,
    expand=None,
) -> None:
    """Register one immutable version and reject ambiguous exact selectors."""
    if definition.recipe in _REGISTRY:
        raise ValueError(f"duplicate recipe {definition.recipe!r}")
    if not re.search(r"@[1-9][0-9]*$", definition.recipe):
        raise ValueError(f"recipe {definition.recipe!r} must end in an immutable @N version")
    selectors = {
        _identity(value) for value in (definition.exact_part, *definition.identity_aliases) if value
    }
    for selector in selectors:
        prior = _SELECTORS.get(selector)
        if prior is not None:
            raise ValueError(
                f"recipe selector {selector!r} overlaps {prior!r} and {definition.recipe!r}"
            )
    registered = RegisteredRecipe(definition=definition, expand=expand)
    _REGISTRY[definition.recipe] = registered
    for selector in selectors:
        _SELECTORS[selector] = definition.recipe


def get_registered_recipe(recipe: str) -> RegisteredRecipe:
    try:
        return _REGISTRY[recipe]
    except KeyError as exc:
        raise ValueError(f"unknown circuit recipe {recipe!r}") from exc


def get_recipe(recipe: str) -> RecipeDefinition:
    return get_registered_recipe(recipe).definition


def registered_recipes() -> tuple[RegisteredRecipe, ...]:
    return tuple(_REGISTRY[key] for key in sorted(_REGISTRY))


def recipe_summaries(
    allowed_maturities: frozenset[str] = frozenset({"production"}),
) -> list[dict]:
    return [
        {
            "recipe": definition.recipe,
            "family": definition.family,
            "exact_part": definition.exact_part,
            "maturity": definition.maturity,
            "required_sheet_roles": list(definition.required_sheet_roles),
            "parameter_defaults": definition.parameter_defaults,
            "external_ports": [port.model_dump() for port in definition.ports],
            "owned_parts": [
                {"role": part.role, "quantity": part.quantity, "value": part.value}
                for part in definition.parts
            ],
            "allocatable_gpios": sorted(
                {
                    pin.gpio
                    for pin in definition.allocatable_pins
                    if pin.gpio is not None and not pin.reserved
                }
            ),
            "allocatable_capabilities": (
                capabilities := allocatable_capability_counts(definition.allocatable_pins)
            ),
            "interfaces": {
                interface: [
                    {"keys": list(keys), "capability": capability}
                    for keys, capability in members
                ]
                for interface, members in FIXED_INTERFACES.items()
                if all(capability in capabilities for _, capability in members)
            },
        }
        for definition in (registered.definition for registered in registered_recipes())
        if definition.maturity in allowed_maturities
    ]


def protected_identities() -> frozenset[str]:
    """Return identities owned by production-enabled deterministic recipes."""
    return frozenset(
        _identity(value)
        for registered in registered_recipes()
        if registered.definition.maturity == "production"
        for value in (
            registered.definition.family,
            registered.definition.exact_part,
            *registered.definition.identity_aliases,
            *registered.definition.protected_aliases,
        )
        if value
    )


def protected_identity_matches(*values: object) -> tuple[str, ...]:
    """Return protected identities contained in model-authored identity fields."""
    observed = {_identity(value) for value in values if value}
    matches = {
        protected
        for protected in protected_identities()
        if any(protected == value or protected in value for value in observed)
    }
    return tuple(sorted(matches))


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
            f"recipe {definition.recipe} sheet roles mismatch; "
            f"missing={sorted(missing_roles)}, unknown={sorted(extra_roles)}"
        )
    declared_ports = {port.name: port for port in definition.ports}
    unknown_ports = set(selection.port_bindings) - set(declared_ports)
    missing_ports = {
        name
        for name, port in declared_ports.items()
        if port.required and not selection.port_bindings.get(name)
    }
    if unknown_ports or missing_ports:
        raise ValueError(
            f"recipe {definition.recipe} port bindings mismatch; "
            f"missing={sorted(missing_ports)}, unknown={sorted(unknown_ports)}"
        )
    return parameters


def _scope_internal_net(recipe: str, instance: str, net: str) -> str:
    recipe_name = recipe.rsplit("@", 1)[0].replace("-", "_")
    return f"__KICRAFT_RECIPE__{recipe_name}__{instance}__{net}"


def expand_static_definition(
    definition: RecipeDefinition,
    resolved: ResolvedRecipeSelection,
) -> RecipeExpansion:
    """Expand one ordinary definition; custom builders may vary a frozen copy."""
    selection = resolved.selection
    refs_by_role: dict[str, list[str]] = {}
    next_number: dict[str, int] = defaultdict(lambda: 1)
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
                    mpn=group.mpn,
                    datasheet=group.datasheet,
                    recipe_id=definition.recipe,
                    resolution_source="recipe",
                    resolution_id=definition.recipe,
                    recipe_instance=selection.instance,
                    recipe_role=group.role,
                )
            )

    declared_ports = {port.name: port for port in definition.ports}

    def bound_net(logical: str) -> str:
        # A pin may name a parameter-controlled strap (e.g. an I2C address pin
        # tied to the supply instead of ground); resolve it through the same
        # binding rules so the chosen value must be a declared port or rail.
        if logical.startswith("@parameter:"):
            name = logical.split(":", 1)[1]
            value = selection.parameters.get(name, definition.parameter_defaults.get(name))
            if value is not None:
                return bound_net(str(value))
            return logical
        if logical in selection.port_bindings:
            return selection.port_bindings[logical]
        port = declared_ports.get(logical)
        if port is not None and port.default_tie:
            target = selection.port_bindings.get(port.default_tie)
            if target:
                return target
        if logical in definition.internal_nets:
            return _scope_internal_net(definition.recipe, selection.instance, logical)
        return logical

    by_sheet_net: dict[tuple[str, str], list[PinEndpoint]] = defaultdict(list)
    part_by_ref = {part.ref: part for part in parts}
    owned: list[PinOwnership] = []
    requirement_id = selection.requirement_ids[0] if selection.requirement_ids else None
    for spec in definition.pins:
        refs = refs_by_role.get(spec.role)
        if not refs:
            raise ValueError(f"recipe pin references unknown role {spec.role!r}")
        if spec.index >= len(refs):
            raise ValueError(
                f"recipe pin role {spec.role!r} index {spec.index} exceeds {len(refs)} parts"
            )
        ref = refs[spec.index]
        net = bound_net(spec.net)
        by_sheet_net[(part_by_ref[ref].sheet, net)].append(PinEndpoint(ref=ref, pin=spec.pin))
        owned.append(
            PinOwnership(
                ref=ref,
                pin=spec.pin,
                net=net,
                owner="recipe",
                owner_id=definition.recipe,
                requirement_id=requirement_id,
            )
        )
    allocated_keys: set[tuple[str, int, str]] = set()
    for allocation in selection.pin_allocations:
        allocatable = next(
            (pin for pin in definition.allocatable_pins if pin.pin == allocation.pin),
            None,
        )
        if allocatable is None:
            raise ValueError(
                f"recipe {definition.recipe} allocation uses unavailable pin {allocation.pin!r}"
            )
        ref = refs_by_role[allocatable.role][allocatable.index]
        allocated_keys.add((allocatable.role, allocatable.index, allocatable.pin))
        by_sheet_net[(part_by_ref[ref].sheet, allocation.net)].append(
            PinEndpoint(ref=ref, pin=allocation.pin)
        )
        owned.append(
            PinOwnership(
                ref=ref,
                pin=allocation.pin,
                net=allocation.net,
                owner="allocator",
                owner_id=f"{definition.recipe}:pin-allocator@1",
                requirement_id=requirement_id,
            )
        )
    connections = [
        NetConnection(sheet=sheet, net_name=net, endpoints=endpoints)
        for (sheet, net), endpoints in sorted(by_sheet_net.items())
    ]
    no_connect_pins: list[PinEndpoint] = []
    for spec in definition.no_connects:
        refs = refs_by_role.get(spec.role)
        if not refs or spec.index >= len(refs):
            raise ValueError(f"recipe no-connect role {spec.role!r} index {spec.index} is invalid")
        endpoint = PinEndpoint(ref=refs[spec.index], pin=spec.pin)
        no_connect_pins.append(endpoint)
        owned.append(
            PinOwnership(
                ref=endpoint.ref,
                pin=endpoint.pin,
                owner="recipe",
                owner_id=definition.recipe,
                requirement_id=requirement_id,
            )
        )
    no_connect_keys = {(spec.role, spec.index, spec.pin) for spec in definition.no_connects}
    recipe_pin_keys = {(spec.role, spec.index, spec.pin) for spec in definition.pins}
    for allocatable in definition.allocatable_pins:
        key = (allocatable.role, allocatable.index, allocatable.pin)
        if key in allocated_keys or key in no_connect_keys or key in recipe_pin_keys:
            continue
        ref = refs_by_role[allocatable.role][allocatable.index]
        endpoint = PinEndpoint(ref=ref, pin=allocatable.pin)
        no_connect_pins.append(endpoint)
        owned.append(
            PinOwnership(
                ref=ref,
                pin=allocatable.pin,
                owner="allocator",
                owner_id=f"{definition.recipe}:pin-allocator@1",
                requirement_id=requirement_id,
            )
        )
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
    internal_nets = [
        _scope_internal_net(definition.recipe, selection.instance, net)
        for net in definition.internal_nets
    ]
    ownership = RecipeOwnershipManifest(
        recipe=definition.recipe,
        instance=selection.instance,
        requirement_ids=selection.requirement_ids,
        refs=[part.ref for part in parts],
        pins=owned,
        internal_nets=internal_nets,
        port_bindings=selection.port_bindings,
        pin_allocations=selection.pin_allocations,
        placement_constraints=[
            constraint.model_dump(mode="json") for constraint in definition.placement_constraints
        ],
        assertions=[assertion.code for assertion in definition.electrical_assertions],
    )
    return RecipeExpansion(
        selection=selection,
        parameters=resolved.parameters,
        parts=parts,
        connections=connections,
        no_connect_pins=no_connect_pins,
        edge_interfaces=edge_interfaces,
        ownership=ownership,
    )


def _renumber_expansion(
    expansion: RecipeExpansion,
    next_reference: dict[str, int] | None,
) -> RecipeExpansion:
    if next_reference is None:
        return expansion
    mapping: dict[str, str] = {}
    counters = dict(next_reference)
    for part in expansion.parts:
        match = re.match(r"([A-Z]+)[0-9]+$", part.ref)
        if match is None:
            raise ValueError(
                f"recipe {expansion.selection.recipe} emitted invalid local ref {part.ref!r}"
            )
        prefix = match.group(1)
        number = counters.get(prefix, 1)
        mapping[part.ref] = f"{prefix}{number}"
        counters[prefix] = number + 1
    payload = expansion.model_dump(mode="json")
    for part in payload["parts"]:
        part["ref"] = mapping[part["ref"]]
    for connection in payload["connections"]:
        for endpoint in connection["endpoints"]:
            endpoint["ref"] = mapping[endpoint["ref"]]
    for endpoint in payload["no_connect_pins"]:
        endpoint["ref"] = mapping[endpoint["ref"]]
    for interface in payload["edge_interfaces"]:
        interface["refs"] = [mapping[ref] for ref in interface["refs"]]
    ownership = payload["ownership"]
    ownership["refs"] = [mapping[ref] for ref in ownership["refs"]]
    for pin in ownership["pins"]:
        pin["ref"] = mapping[pin["ref"]]
    next_reference.clear()
    next_reference.update(counters)
    return RecipeExpansion.model_validate(payload)


def expand_recipe(
    selection: RecipeSelection,
    *,
    next_reference: dict[str, int] | None = None,
) -> RecipeExpansion:
    registered = get_registered_recipe(selection.recipe)
    parameters = _validated_parameters(registered.definition, selection)
    resolved = ResolvedRecipeSelection(
        selection=selection,
        parameters=parameters,
    )
    expansion = (
        registered.expand(resolved)
        if registered.expand is not None
        else expand_static_definition(registered.definition, resolved)
    )
    if expansion.selection != selection:
        raise ValueError(f"recipe builder {selection.recipe} returned a different selection")
    return _renumber_expansion(expansion, next_reference)


def expand_selections(selections: Iterable[object]) -> list[RecipeExpansion]:
    next_reference: dict[str, int] = {}
    return [
        expand_recipe(
            RecipeSelection.model_validate(selection),
            next_reference=next_reference,
        )
        for selection in selections
    ]


def locked_pin_assignments(bom: dict) -> dict[tuple[str, str], str]:
    """Read deterministic connected ownership from the canonical BOM manifest."""
    locked: dict[tuple[str, str], str] = {}
    manifests = bom.get("recipe_ownership") or []
    for manifest in manifests:
        for row in manifest.get("pins") or []:
            if row.get("net") is not None:
                locked[(str(row["ref"]), str(row["pin"]))] = str(row["net"])
    if manifests:
        return locked

    # Legacy state compatibility for static recipe expansions created before
    # ownership manifests existed.
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
    for (recipe_id, _instance, role), refs in by_key_role.items():
        for pin in get_recipe(recipe_id).pins:
            if pin.role == role and pin.index < len(refs):
                locked[(refs[pin.index], pin.pin)] = pin.net
    return locked


def locked_no_connect_pins(bom: dict) -> set[tuple[str, str]]:
    """Read deterministic no-connect ownership from the canonical BOM manifest."""
    locked: set[tuple[str, str]] = set()
    manifests = bom.get("recipe_ownership") or []
    for manifest in manifests:
        for row in manifest.get("pins") or []:
            if row.get("net") is None:
                locked.add((str(row["ref"]), str(row["pin"])))
    if manifests:
        return locked

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
            result[match.group(1)] = max(
                result.get(match.group(1), 1),
                int(match.group(2)) + 1,
            )
    return result
