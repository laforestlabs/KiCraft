"""Architecture-time deterministic recipe resolution and protected-family blocking."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from kicraft.design.lowering import (
    lowerer_contract_diagnostic,
    lower_requirement,
    registered_lowerers,
)
from kicraft.design.models import (
    Architecture,
    CircuitRequirement,
    RecipeResolutionRecord,
    RecipeSelection,
)
from kicraft.design.part_identity import (
    canonical_physical_features,
    is_part_family,
    matches_part_identity,
    physical_inventory_record,
)
from kicraft.design.synthesis.validation import _net_voltage, named_part_tokens

from .models import RecipeDefinition, RegisteredRecipe
from .pin_allocator import PinAllocationError, allocate_requirement_pins
from .registry import protected_identities, registered_recipes

_PROGRAMMING_PEER_PORTS = {
    "uart_tx": "rx",
    "uart_rx": "tx",
    "dtr_n": "dtr_n",
    "rts_n": "rts_n",
}


def _identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


class ResolutionDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    requirement_id: str | None = None
    recipe: str | None = None
    sheet: str | None = None
    message: str
    evidence: list[str] = Field(default_factory=list)


class ResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selections: list[RecipeSelection] = Field(default_factory=list)
    exact_parts: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    blocking: list[ResolutionDiagnostic] = Field(default_factory=list)
    protected_identities: list[str] = Field(default_factory=list)
    records: list[RecipeResolutionRecord] = Field(default_factory=list)
    requirements: list[CircuitRequirement] = Field(default_factory=list)


class RecipeResolutionError(ValueError):
    def __init__(self, diagnostics: list[ResolutionDiagnostic]):
        self.diagnostics = diagnostics
        super().__init__("; ".join(f"{row.code}: {row.message}" for row in diagnostics))


def _exact_selectors(recipe: RegisteredRecipe) -> set[str]:
    definition = recipe.definition
    return {
        _identity(value) for value in (definition.exact_part, *definition.identity_aliases) if value
    }


def _protected_selectors(recipe: RegisteredRecipe) -> set[str]:
    definition = recipe.definition
    return {
        _identity(value)
        for value in (
            definition.family,
            definition.exact_part,
            *definition.identity_aliases,
            *definition.protected_aliases,
        )
        if value
    }


def _recipe_requirement_choice(definition: RecipeDefinition) -> str:
    return (
        f"canonical choice: recipe={definition.recipe}; "
        f"family={definition.family}; exact_part={definition.exact_part}; "
        "ports=" + ",".join(port.name for port in definition.ports)
    )


def _recipe_edge_companion(
    definition: RecipeDefinition,
    requirement: CircuitRequirement,
    requirements: list[CircuitRequirement],
    existing_requirement_ids: frozenset[str],
) -> tuple[str, CircuitRequirement] | None:
    """Return the synthetic edge header physically covered by recipe pads.

    An architecture edge opened from an MCU can initially appear as a generic
    pin-header requirement.  A recipe that already declares matching
    castellated pads is the physical implementation of that edge, not an
    additional header.  The match is intentionally narrow: the recipe must
    expose one edge sheet role, and the generated connector must carry only
    those edge nets for the same functional block.
    """
    edge_roles = {edge.role for edge in definition.edges}
    edge_sheet_roles = {part.sheet_role for part in definition.parts if part.role in edge_roles}
    if len(edge_sheet_roles) != 1 or not edge_sheet_roles <= set(definition.required_sheet_roles):
        return None
    edge_nets = {_identity(pin.net) for pin in definition.pins if pin.role in edge_roles}
    if not edge_nets:
        return None
    candidates = [
        candidate
        for candidate in requirements
        if candidate.id != requirement.id
        and candidate.id not in existing_requirement_ids
        and candidate.compiler_origin == "edge_connector"
        and candidate.role == "connector"
        and candidate.family == "pin-header"
        and candidate.sheet != requirement.sheet
        and candidate.exact_part is None
        and candidate.standard_stacking_role is None
        and not candidate.obligations
        and candidate.ports
        and set(candidate.functional_blocks) & set(requirement.functional_blocks)
        and {_identity(net) for net in candidate.ports.values()} <= edge_nets
    ]
    if len(candidates) != 1:
        return None
    return edge_sheet_roles.pop(), candidates[0]


def _coalesce_mcu_support_requirements(
    result: ResolutionResult,
    requirements: list[CircuitRequirement],
    recipes: tuple[RegisteredRecipe, ...],
) -> None:
    """Attach proven support obligations to one identically configured MCU.

    A clock or flash requirement can describe a member of the MCU circuit,
    not another complete MCU. Identity alone is insufficient: the selected
    circuit, physical sheet map, parameters, external nets and pin allocation
    must agree, and reviewed non-MCU groups must implement every physical
    class. Ambiguous ownership remains an architecture error.
    """
    by_id = {requirement.id: requirement for requirement in requirements}
    definitions = {recipe.definition.recipe: recipe.definition for recipe in recipes}

    def core_ids(selection: RecipeSelection) -> list[str]:
        return [
            identity
            for identity in selection.requirement_ids
            if by_id[identity].role == "mcu_core"
        ]

    cores = [selection for selection in result.selections if core_ids(selection)]
    retained = []
    for selection in result.selections:
        definition = definitions[selection.recipe]
        if "mcu" not in definition.required_sheet_roles or core_ids(selection):
            retained.append(selection)
            continue
        if len(selection.requirement_ids) != 1:
            retained.append(selection)
            continue
        requirement = by_id[selection.requirement_ids[0]]
        physical = [
            obligation
            for obligation in requirement.obligations
            if obligation.kind == "physical"
        ]
        if not physical:
            retained.append(selection)
            continue
        support_features: set[str] = set()
        core_features: set[str] = set()
        for group in definition.parts:
            record = physical_inventory_record(
                mpn=group.mpn, symbol=group.symbol, footprint=group.footprint
            )
            if record is not None:
                target = core_features if group.role == "mcu" else support_features
                target.update(record.physical_features)
        demands = [canonical_physical_features(row.component_class) for row in physical]
        if any(demand & core_features for demand in demands):
            # This requirement explicitly demands a processor, not merely
            # one of its support parts; never erase a second physical MCU.
            retained.append(selection)
            continue
        candidates = [
            core
            for core in cores
            if len(core_ids(core)) == 1
            and core.recipe == selection.recipe
            and core.sheets == selection.sheets
            and core.parameters == selection.parameters
            and core.port_bindings == selection.port_bindings
            and core.pin_allocations == selection.pin_allocations
            and by_id[core_ids(core)[0]].sheet == requirement.sheet
            and _identity(by_id[core_ids(core)[0]].exact_part)
            == _identity(requirement.exact_part)
        ]
        if (
            len(candidates) != 1
            or not all(demand & support_features for demand in demands)
            or requirement.interfaces
            or requirement.declared_interface is not None
            or requirement.standard_stacking_role is not None
        ):
            result.blocking.append(
                ResolutionDiagnostic(
                    code="mcu_support_owner_unproven",
                    requirement_id=requirement.id,
                    recipe=selection.recipe,
                    message=(
                        "MCU support hardware requires one matching physical core and "
                        "reviewed support groups; do not expand another MCU to satisfy it"
                    ),
                    evidence=[core.instance for core in candidates],
                )
            )
            result.unresolved_requirements.append(requirement.id)
            continue
        owner = candidates[0]
        owner.requirement_ids.extend(selection.requirement_ids)
        result.assumptions.append(
            f"{requirement.id}: support hardware is owned by MCU instance {owner.instance}"
        )
    result.selections = retained


def _recipe_for_exact(
    identity: str | None,
    recipes: tuple[RegisteredRecipe, ...],
) -> RegisteredRecipe | None:
    token = _identity(identity) if identity else ""
    if not token:
        return None
    matches = [recipe for recipe in recipes if token in _exact_selectors(recipe)]
    if len(matches) > 1:
        raise ValueError(f"ambiguous exact recipe identity {identity!r}")
    return matches[0] if matches else None


def _recipe_for_unique_family_prefix(
    identity: str | None,
    recipes: tuple[RegisteredRecipe, ...],
) -> RegisteredRecipe | None:
    token = _identity(identity) if identity else ""
    if len(token) < 5:
        return None
    matches = {
        recipe.definition.recipe: recipe
        for recipe in recipes
        if any(selector.startswith(token) for selector in _protected_selectors(recipe))
    }
    if len(matches) == 1:
        return next(iter(matches.values()))
    # An exact family request may use its explicitly reviewed default ordering
    # code. A partial prefix or an exact variant never gains this fallback.
    if matches and all(
        _identity(recipe.definition.family or "") == token for recipe in matches.values()
    ):
        defaults = [recipe for recipe in matches.values() if recipe.definition.default_for_family]
        if len(defaults) == 1:
            return defaults[0]
    return None


def _is_mcu_recipe(recipe: RegisteredRecipe) -> bool:
    return "mcu" in recipe.definition.required_sheet_roles


def _family_recipes(
    family: str,
    recipes: tuple[RegisteredRecipe, ...],
) -> list[RegisteredRecipe]:
    token = _identity(family)
    return [
        recipe
        for recipe in recipes
        if recipe.definition.family and _identity(recipe.definition.family) == token
    ]


def _looks_protected(value: str, recipes: tuple[RegisteredRecipe, ...]) -> bool:
    token = _identity(value)
    return bool(
        token
        and any(
            selector in token or token in selector
            for recipe in recipes
            for selector in _protected_selectors(recipe)
            if len(selector) >= 5
        )
    )


def _looks_exact_variant(value: str) -> bool:
    lowered = value.lower()
    return bool(
        re.search(
            r"(?:mini[-_ ]?1|wroom|esp32[-_ ]?c3|c8t6|tiny(?:402|412|1614)|ch32v003)",
            lowered,
        )
    )


def _registered_variant_for(
    identity: str | None,
    recipes: tuple[RegisteredRecipe, ...],
) -> RegisteredRecipe | None:
    """The single registered recipe that may serve ``identity``'s order code.

    Membership is explicit and reviewed in ``part_identity`` — never inferred
    from the name (``ESP32-S3-WROOM-1-N16R8`` is served by the recipe pinning the
    ``-N8R8`` module because the datasheet lists them as one module differing only
    in flash). An identity with no reviewed member, or one spanning two families,
    returns None so the caller blocks instead of guessing.
    """
    if not identity or not identity.strip():
        return None
    matches = [
        recipe
        for recipe in recipes
        if recipe.definition.exact_part
        and recipe.definition.exact_part.strip().casefold() != identity.strip().casefold()
        and matches_part_identity(identity, recipe.definition.exact_part)
    ]
    # Exactly one serving recipe in exactly one family: a second candidate means
    # the identity is ambiguous, and guessing which variant to ship is not this
    # function's call.
    if len(matches) != 1:
        return None
    return matches[0]


def _substitution_note(identity: str, recipe: RegisteredRecipe) -> str:
    """Record, in the resolved architecture's own assumptions, that a requested
    order code is served by a different registered variant — the deviation must
    reach ``bom.substitutions`` (§9.33), never be silent."""
    definition = recipe.definition
    return (
        f"{identity!r} is an unregistered ordering code of {definition.family!r}; "
        f"served by {definition.recipe} ({definition.exact_part}). Record the "
        f"deviation in bom.substitutions: wanted={identity!r}, "
        f"got={definition.exact_part!r}, reason=<why>"
    )


def _registered_identity_choices(
    identity: str,
    recipes: tuple[RegisteredRecipe, ...],
    *,
    limit: int = 6,
) -> list[str]:
    """Canonical choice lines for the registered families that share a name stem
    with ``identity``.

    A blocking ``unsupported_protected_variant`` must name what CAN be bound, not
    only what was refused — but dumping all ~35 registered families into the
    correction feedback buries the answer. Only the plausible candidates (same
    5-character identity stem) are offered, so a foreign ESP32-S3 order code is
    answered with the ESP32-S3 families and a truly unknown part with nothing.
    """
    stem = _identity(identity)[:5]
    if not stem:
        return []
    seen: dict[str, RegisteredRecipe] = {}
    for recipe in recipes:
        family = recipe.definition.family
        if not family or family in seen:
            continue
        if any(
            len(selector) >= 5 and selector[:5] == stem for selector in _protected_selectors(recipe)
        ):
            seen[family] = recipe
    return [_recipe_requirement_choice(recipe.definition) for recipe in list(seen.values())[:limit]]


def _mcu_sheet(architecture: Architecture) -> str | None:
    owners = [
        sheet.name
        for sheet in architecture.sheets
        if re.search(r"\b(?:mcu|microcontroller)\b", f"{sheet.name} {sheet.function}", re.I)
    ]
    return owners[0] if len(owners) == 1 else None


_USB_PORT_ALIASES = {
    "usb_dm": ("USB_DM", "USB_D-", "USB_D_N", "D-"),
    "usb_dp": ("USB_DP", "USB_D+", "USB_D_P", "D+"),
}

# Net-label aliases per semantic port. The port's own name is always accepted in
# addition to these (see _port_bindings), so an architecture may label a channel
# either bare ('OE') or as the connector signal ('HUB75_OE').
_HUB75_PORT_ALIASES = {
    "r0": ("HUB75_R0",),
    "g0": ("HUB75_G0",),
    "b0": ("HUB75_B0",),
    "r1": ("HUB75_R1",),
    "g1": ("HUB75_G1",),
    "b1": ("HUB75_B1",),
    "addr_a": ("HUB75_A",),
    "addr_b": ("HUB75_B",),
    "addr_c": ("HUB75_C",),
    "addr_d": ("HUB75_D",),
    "clk": ("HUB75_CLK", "HUB75_CLOCK"),
    "lat": ("HUB75_LAT", "HUB75_LATCH", "HUB75_STB"),
    "oe": ("HUB75_OE",),
}
_PORT_ALIASES = {**_USB_PORT_ALIASES, **_HUB75_PORT_ALIASES}


def _net_identity(value: str) -> str:
    """Ignore naming separators without erasing electrical polarity."""
    return re.sub(r"[^a-z0-9+-]", "", value.lower())


def _net_by_tokens(architecture: Architecture, sheet: str, *tokens: str) -> str | None:
    wanted = {_net_identity(token) for token in tokens}
    matches = {
        net.name
        for net in architecture.inter_sheet_nets
        if _net_identity(net.name) in wanted
        and any(endpoint.sheet == sheet for endpoint in net.endpoints)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _supply_binding(architecture: Architecture, sheet: str) -> str | None:
    """Complete the conventional 3.3V logic supply only from unique rail evidence."""
    endpoint_nets = {
        net.name
        for net in architecture.inter_sheet_nets
        if any(endpoint.sheet == sheet for endpoint in net.endpoints)
    }
    candidates = {
        name
        for name in set(architecture.power_nets) | endpoint_nets
        if (
            abs(architecture.rail_voltages[name] - 3.3) <= 0.05
            if name in architecture.rail_voltages
            else bool(re.fullmatch(r"\+?(?:(?:vcc|vdd)[_-]?)?3(?:v3|\.3v)", name, re.I))
        )
        and not name.startswith("-")
    }
    owned = candidates & endpoint_nets
    if owned:
        candidates = owned
    return next(iter(candidates)) if len(candidates) == 1 else None


def _named_mcu_required(intent: dict, recipes: tuple[RegisteredRecipe, ...]) -> bool:
    return any(
        re.fullmatch(
            r"(?:attiny|atmega|stm32|esp32|rp20|rp23|nrf5|ch32)[a-z0-9]*", _identity(named)
        )
        or any(_is_mcu_recipe(recipe) for recipe in _family_recipes(named, recipes))
        for named in intent.get("named_parts") or []
    )


def _named_mcu_sheets(
    architecture: Architecture, recipes: tuple[RegisteredRecipe, ...]
) -> dict[str, RegisteredRecipe]:
    """Recognize a chip-domain owner, not a programming header mentioning it."""
    matches: dict[str, RegisteredRecipe] = {}
    mcu_recipes = tuple(recipe for recipe in recipes if _is_mcu_recipe(recipe))
    for sheet in architecture.sheets:
        text = f"{sheet.name} {sheet.function}"
        if not re.search(r"\b(?:mcu|microcontroller|controller)\b", text, re.I):
            continue
        candidates = {
            recipe.definition.recipe: recipe
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
            if (
                recipe := _recipe_for_exact(token, mcu_recipes)
                or _recipe_for_unique_family_prefix(token, mcu_recipes)
            )
            is not None
        }
        if len(candidates) == 1:
            matches[sheet.name] = next(iter(candidates.values()))
    return matches


def _legacy_requirements(
    architecture: Architecture,
    intent: dict,
    recipes: tuple[RegisteredRecipe, ...],
) -> tuple[list[CircuitRequirement], list[ResolutionDiagnostic]]:
    if architecture.requirements:
        requirements = list(architecture.requirements)
        if (architecture.mcu_present or _named_mcu_required(intent, recipes)) and not any(
            row.role == "mcu_core" for row in requirements
        ):
            inferred, diagnostics = _legacy_requirements(
                architecture.model_copy(update={"requirements": []}),
                intent,
                tuple(recipe for recipe in recipes if _is_mcu_recipe(recipe)),
            )
            used_ids = {row.id for row in requirements}
            requirements.extend(row for row in inferred if row.id not in used_ids)
            return requirements, diagnostics
        return requirements, []
    named_parts = [str(value) for value in intent.get("named_parts") or []]
    architecture_text = " ".join(
        [
            *named_parts,
            *(f"{sheet.name} {sheet.function}" for sheet in architecture.sheets),
            *(f"{key} {value}" for key, value in architecture.topologies.items()),
        ]
    )
    candidates = named_parts or [architecture_text]
    requirements: list[CircuitRequirement] = []
    diagnostics: list[ResolutionDiagnostic] = []
    seen_recipes: set[str] = set()
    mcu_sheet = _mcu_sheet(architecture)
    for candidate in candidates:
        exact = _recipe_for_exact(candidate, recipes) or _recipe_for_unique_family_prefix(
            candidate, recipes
        )
        if exact is not None and exact.definition.recipe not in seen_recipes:
            sheet = mcu_sheet
            definition = exact.definition
            if not _is_mcu_recipe(exact):
                matching_sheets = [
                    row.name
                    for row in architecture.sheets
                    if any(
                        selector in _identity(f"{row.name} {row.function}")
                        for selector in _exact_selectors(exact)
                    )
                ]
                if len(matching_sheets) != 1:
                    diagnostics.append(
                        ResolutionDiagnostic(
                            code="missing_recipe_requirement",
                            recipe=definition.recipe,
                            message=f"named part {candidate!r} needs an explicit owning requirement and sheet",
                            evidence=[candidate, _recipe_requirement_choice(definition)],
                        )
                    )
                    continue
                sheet = matching_sheets[0]
            if sheet is None:
                diagnostics.append(
                    ResolutionDiagnostic(
                        code="missing_mcu_requirement",
                        recipe=definition.recipe,
                        message=f"named MCU {candidate!r} needs one explicit owning MCU sheet and requirement",
                        evidence=[candidate],
                    )
                )
                continue
            ports: dict[str, str] = {}
            ports["gnd"] = "GND"
            usb_dm = _net_by_tokens(architecture, sheet, *_USB_PORT_ALIASES["usb_dm"])
            usb_dp = _net_by_tokens(architecture, sheet, *_USB_PORT_ALIASES["usb_dp"])
            interfaces: list[str] = []
            if usb_dm and usb_dp:
                ports.update({"usb_dm": usb_dm, "usb_dp": usb_dp})
                interfaces.append("usb_device")
            requirements.append(
                CircuitRequirement(
                    id=f"auto_{definition.family or 'circuit'}".replace("-", "_"),
                    sheet=sheet,
                    role="mcu_core" if _is_mcu_recipe(exact) else "bus_interface",
                    family=definition.family or definition.recipe.rsplit("@", 1)[0],
                    exact_part=definition.exact_part,
                    ports=ports,
                    interfaces=interfaces,
                )
            )
            seen_recipes.add(definition.recipe)
            continue
        if _looks_protected(candidate, recipes) and _looks_exact_variant(candidate):
            diagnostics.append(
                ResolutionDiagnostic(
                    code="unsupported_protected_variant",
                    message=f"protected variant {candidate!r} has no verified recipe",
                    evidence=[candidate],
                )
            )
    if not requirements and _looks_protected(architecture_text, recipes) and mcu_sheet:
        family_defaults = [
            recipe
            for recipe in recipes
            if recipe.definition.default_for_family
            and any(
                selector in _identity(architecture_text)
                for selector in _protected_selectors(recipe)
            )
        ]
        if len(family_defaults) == 1:
            definition = family_defaults[0].definition
            requirements.append(
                CircuitRequirement(
                    id=f"auto_{definition.family}".replace("-", "_"),
                    sheet=mcu_sheet,
                    role="mcu_core",
                    family=definition.family or "mcu",
                    ports={"gnd": "GND"},
                )
            )
    return requirements, diagnostics


def _port_bindings(
    definition: RecipeDefinition,
    requirement: CircuitRequirement,
    architecture: Architecture,
) -> tuple[dict[str, str], list[ResolutionDiagnostic]]:
    declared = {port.name: port for port in definition.ports}
    bindings = {name: net for name, net in requirement.ports.items() if name in declared}
    if "gnd" in declared:
        bindings.setdefault("gnd", "GND")
    if "vdd" in declared:
        vdd = _supply_binding(architecture, requirement.sheet)
        if vdd:
            bindings.setdefault("vdd", vdd)
    local_nets = {
        net.name: [
            endpoint.direction for endpoint in net.endpoints if endpoint.sheet == requirement.sheet
        ]
        for net in architecture.inter_sheet_nets
        if any(endpoint.sheet == requirement.sheet for endpoint in net.endpoints)
    }
    for port_name, port in declared.items():
        if port.direction == "power" or port_name in bindings:
            continue
        if "auto_reset" in definition.parameter_defaults and port_name in _PROGRAMMING_PEER_PORTS:
            continue  # Programming polarity needs explicit typed ownership, not label matching.
        # A net label is not ownership, and a peer's TX label is not our TX.
        candidates = sorted(
            net
            for net, directions in local_nets.items()
            if not (
                port.direction in {"input", "output"}
                and ("output" if port.direction == "input" else "input") in directions
            )
        )
        port_identity = _net_identity(port_name)
        aliases = {
            _net_identity(port_name),
            *(_net_identity(alias) for alias in _PORT_ALIASES.get(port_name, ())),
        }
        exact_matches = [net for net in candidates if _net_identity(net) in aliases]
        matches = exact_matches or [
            net
            for net in candidates
            if len(port_identity) >= 2
            and (
                _net_identity(net).endswith(port_identity)
                or _net_identity(net).startswith(port_identity)
            )
        ]
        if len(matches) == 1:
            bindings[port_name] = matches[0]
    diagnostics: list[ResolutionDiagnostic] = []
    context = (
        f"requirement {requirement.id!r}, recipe {definition.recipe}, sheet {requirement.sheet!r}"
    )
    power_bindings = {
        name
        for name, net in bindings.items()
        if declared[name].direction == "power"
        or (declared[name].allow_ground and net == bindings.get("gnd"))
    }
    signal_power_nets = sorted(
        {
            net
            for name, net in bindings.items()
            if name not in power_bindings and net in architecture.power_nets
        }
    )
    if signal_power_nets:
        diagnostics.append(
            ResolutionDiagnostic(
                code="recipe_signal_in_power_nets",
                requirement_id=requirement.id,
                recipe=definition.recipe,
                sheet=requirement.sheet,
                message=f"{context}: signal nets {signal_power_nets} belong in explicit inter-sheet contracts, not power_nets",
                evidence=signal_power_nets,
            )
        )
    return dict(sorted(bindings.items())), diagnostics


def _bound_rail_voltage(architecture: Architecture, net: str | None) -> float | None:
    if net is None:
        return None
    if net in architecture.rail_voltages:
        return architecture.rail_voltages[net]
    if net.startswith("-"):
        return None
    return _net_voltage(net)


def _complete_ch340c_supply_mode(
    definition: RecipeDefinition,
    requirement: CircuitRequirement,
    architecture: Architecture,
) -> tuple[CircuitRequirement, list[ResolutionDiagnostic]]:
    """Select the V3 circuit from the actual supply, never from a voltage default."""
    if definition.exact_part != "CH340C":
        return requirement, []
    bindings, _diagnostics = _port_bindings(definition, requirement, architecture)
    net = bindings.get("vdd")
    voltage = _bound_rail_voltage(architecture, net)
    explicit = requirement.parameters.get("supply_voltage")
    evidence = [
        f"{requirement.id} ({requirement.sheet}).ports.vdd={net!r}; voltage={voltage!r} V",
        f"{requirement.id}.parameters.supply_voltage={explicit!r}",
        (
            f"architecture.rail_voltages[{net!r}]={voltage!r}"
            if net in architecture.rail_voltages
            else f"rail-name voltage evidence: {net!r} -> {voltage!r}"
        ),
    ]
    error = None
    if voltage is None or not math.isfinite(voltage):
        error = "CH340C supply voltage is unproven"
    elif voltage not in (3.3, 5.0):
        error = "CH340C bound supply has no reviewed 3.3 V or 5.0 V operating mode"
    elif "supply_voltage" in requirement.parameters and (
        isinstance(explicit, bool)
        or not isinstance(explicit, (int, float))
        or not math.isfinite(explicit)
        or explicit not in (3.3, 5.0)
        or explicit != voltage
    ):
        error = "CH340C explicit supply_voltage is unsupported or conflicts with its bound rail"
    if error is not None:
        return requirement, [
            ResolutionDiagnostic(
                code="conflicting_recipe_supply",
                requirement_id=requirement.id,
                recipe=definition.recipe,
                sheet=requirement.sheet,
                message=(
                    f"{error}: {'; '.join(evidence)}. Declare the actual rail voltage and "
                    "a matching numeric supply_voltage (3.3 or 5.0), or explicitly repair "
                    "the physical supply binding; do not guess VBUS voltage or silently move rails."
                ),
                evidence=evidence,
            )
        ]
    if "supply_voltage" in requirement.parameters:
        return requirement, []
    return requirement.model_copy(
        update={"parameters": {**requirement.parameters, "supply_voltage": voltage}}
    ), []


def _ch340c_esp32_domain_diagnostics(
    definition: RecipeDefinition,
    requirement: CircuitRequirement,
    architecture: Architecture,
    recipes: tuple[RegisteredRecipe, ...],
) -> list[ResolutionDiagnostic]:
    """Check only direct typed CH340C UART conductors reaching a known ESP32."""
    if not (definition.exact_part or "").startswith("ESP32"):
        return []
    bindings, _diagnostics = _port_bindings(definition, requirement, architecture)
    mcu_rail = bindings.get("vdd")
    mcu_voltage = _bound_rail_voltage(architecture, mcu_rail)
    diagnostics = []
    for peer in architecture.requirements:
        exact = _recipe_for_exact(peer.exact_part or peer.family, recipes)
        matches = [exact] if exact is not None else _family_recipes(peer.family, recipes)
        if len(matches) != 1 or matches[0].definition.exact_part != "CH340C":
            continue
        direct = {
            name: net
            for name, net in peer.ports.items()
            if name in {"tx", "rx"}
            and (
                (peer.sheet == requirement.sheet and net in bindings.values())
                or (
                    peer.sheet != requirement.sheet
                    and any(
                        row.name == net
                        and {requirement.sheet, peer.sheet} <= {ep.sheet for ep in row.endpoints}
                        for row in architecture.inter_sheet_nets
                    )
                )
            )
        }
        if not direct:
            continue
        peer_bindings, _peer_diagnostics = _port_bindings(matches[0].definition, peer, architecture)
        peer_rail = peer_bindings.get("vdd")
        peer_voltage = _bound_rail_voltage(architecture, peer_rail)
        if (
            mcu_rail is not None
            and peer_rail == mcu_rail
            and mcu_voltage == peer_voltage == 3.3
            and bindings.get("gnd") is not None
            and bindings["gnd"] == peer_bindings.get("gnd")
        ):
            continue
        evidence = [
            f"{peer.id} ({peer.sheet}, CH340C).ports.vdd={peer_rail!r}; voltage={peer_voltage!r} V; "
            f"gnd={peer_bindings.get('gnd')!r}",
            f"{requirement.id} ({requirement.sheet}, {definition.exact_part}).ports.vdd="
            f"{mcu_rail!r}; voltage={mcu_voltage!r} V; gnd={bindings.get('gnd')!r}",
            *(
                f"direct conductor {net!r}: {peer.id}.ports.{name} and {requirement.id} "
                f"({requirement.sheet}); MCU bindings="
                f"{sorted(port for port, bound in bindings.items() if bound == net)!r}"
                for name, net in sorted(direct.items())
            ),
            "WCH CH340DS1 v3D sections 5.1, 6.2, 7.6-7.7: unified supply recommended; "
            "at VCC=5.0 V, non-USB VOH minimum is VCC-0.6=4.4 V",
        ]
        if (definition.exact_part or "").startswith("ESP32-C3-MINI-1"):
            evidence.append(
                "Espressif ESP32-C3-MINI-1 datasheet v2.2 table 6-3: input high maximum "
                "VDD+0.3 V (3.6 V at VDD=3.3 V); "
                "https://documentation.espressif.com/esp32-c3-mini-1_datasheet_en.pdf"
            )
        diagnostics.append(
            ResolutionDiagnostic(
                code="incompatible_programming_power_domain",
                requirement_id=requirement.id,
                recipe=definition.recipe,
                sheet=requirement.sheet,
                message=(
                    "Direct CH340C-to-ESP32 UART requires the reviewed common 3.3 V rail "
                    f"and common ground: {'; '.join(evidence)}. Explicitly choose a real common "
                    "3.3 V supply binding for both devices, or implement typed isolation/level "
                    "translation with distinct conductors on each side. Renaming signals or "
                    "correcting UART/GPIO ownership does not repair this electrical domain."
                ),
                evidence=evidence,
            )
        )
    return diagnostics


def _complete_typed_programming_peer(
    definition: RecipeDefinition,
    requirement: CircuitRequirement,
    architecture: Architecture,
    recipes: tuple[RegisteredRecipe, ...],
) -> tuple[CircuitRequirement, list[ResolutionDiagnostic]]:
    """Cross UART directions only for one verified bridge with local typed nets."""
    if requirement.role != "mcu_core" or "auto_reset" not in definition.parameter_defaults:
        return requirement, []
    directions = {port.name: port.direction for port in definition.ports}
    nets = {net.name: net for net in architecture.inter_sheet_nets}
    candidates: list[dict[str, str]] = []
    peer_contracts: list[str] = []
    errors: list[str] = []
    for peer in architecture.requirements:
        if peer.sheet == requirement.sheet or peer.role not in {"bus_interface", "programming"}:
            continue
        exact = _recipe_for_exact(peer.exact_part or peer.family, recipes)
        matches = [exact] if exact is not None else _family_recipes(peer.family, recipes)
        if len(matches) != 1:
            continue
        peer_directions = {port.name: port.direction for port in matches[0].definition.ports}
        if not {"usb_dm", "usb_dp"} <= peer_directions.keys() or not (
            peer_directions.get("tx") == "output" and peer_directions.get("rx") == "input"
        ):
            continue
        expected = {
            name: peer.ports[peer_name]
            for name, peer_name in _PROGRAMMING_PEER_PORTS.items()
            if peer_name in peer.ports
            and peer_directions.get(peer_name)
            == ("input" if directions[name] == "output" else "output")
        }
        local = {
            name: net
            for name, net in expected.items()
            if net in nets
            and {requirement.sheet, peer.sheet} <= {ep.sheet for ep in nets[net].endpoints}
        }
        if not local:
            continue
        for name, net in expected.items():
            peer_name = _PROGRAMMING_PEER_PORTS[name]
            mcu_port = f"{requirement.id}.ports.{name}"
            bridge_port = f"{peer.id}.ports.{peer_name}"
            source, sink = (
                (mcu_port, bridge_port) if directions[name] == "output" else (bridge_port, mcu_port)
            )
            peer_contracts.append(
                f"{mcu_port}={requirement.ports.get(name, '<unbound>')!r}; "
                f"{bridge_port}={net!r}; required {mcu_port}={net!r} "
                f"with {source} (output) -> {sink} (input)"
            )
        needed = (
            set(_PROGRAMMING_PEER_PORTS)
            if {"dtr_n", "rts_n"} & local.keys()
            else {"uart_tx", "uart_rx"}
        )
        missing = needed - local.keys()
        if missing:
            errors.append(
                f"{peer.id}: missing local typed programming ports {sorted(missing)}; "
                "bind each missing bridge port and declare its net on both "
                f"{peer.sheet!r} and {requirement.sheet!r} with the typed directions"
            )
            continue
        for name, net in local.items():
            for endpoint in nets[net].endpoints:
                wanted = (
                    directions[name]
                    if endpoint.sheet == requirement.sheet
                    else peer_directions[_PROGRAMMING_PEER_PORTS[name]]
                    if endpoint.sheet == peer.sheet
                    else None
                )
                if (
                    wanted
                    and endpoint.direction in {"input", "output"}
                    and endpoint.direction != wanted
                ):
                    errors.append(
                        f"{net}: {endpoint.sheet} direction {endpoint.direction} conflicts with typed {wanted}"
                    )
            if name in requirement.ports and requirement.ports[name] != net:
                errors.append(f"{name}: explicit MCU binding conflicts with bridge {peer.id}")
        candidates.append(local)
    if len(candidates) > 1:
        errors.append(
            "multiple local UART programming peers; select one unambiguous typed contract"
        )
    ports = dict(requirement.ports)
    if len(candidates) == 1:
        ports.update(candidates[0])
    for name in _PROGRAMMING_PEER_PORTS:
        net = nets.get(ports.get(name, ""))
        if net is not None and any(
            endpoint.sheet == requirement.sheet
            and endpoint.direction in {"input", "output"}
            and endpoint.direction != directions[name]
            for endpoint in net.endpoints
        ):
            errors.append(f"{name}: local endpoint conflicts with MCU {directions[name]} direction")
    programming_nets = {net for name, net in ports.items() if name in _PROGRAMMING_PEER_PORTS}
    aliases = [
        name
        for name, net in requirement.ports.items()
        if name not in _PROGRAMMING_PEER_PORTS and net in programming_nets
    ]
    if aliases:
        errors.append(
            f"programming nets also request non-programming ports "
            f"{[f'{name}={requirement.ports[name]!r}' for name in sorted(aliases)]}; "
            "use fixed uart_tx/uart_rx/dtr_n/rts_n, never allocate reset handshakes to GPIO"
        )
    if len(programming_nets) != sum(name in ports for name in _PROGRAMMING_PEER_PORTS):
        errors.append(
            "programming ports alias the same net; give UART TX, UART RX, DTR and RTS "
            "distinct physical nets in both typed peers and their sheet endpoints"
        )
    parameters = dict(requirement.parameters)
    if {"dtr_n", "rts_n"} & ports.keys():
        if parameters.get("auto_reset") is False:
            errors.append(
                f"{requirement.id}.parameters.auto_reset=False conflicts with typed "
                f"dtr_n={ports.get('dtr_n', '<unbound>')!r}/"
                f"rts_n={ports.get('rts_n', '<unbound>')!r}; "
                "set auto_reset=True to implement these controls through the fixed reset circuit"
            )
        elif set(_PROGRAMMING_PEER_PORTS) <= ports.keys():
            parameters["auto_reset"] = True
    if errors:
        if peer_contracts:
            errors.append(
                "Repair the explicit bindings and sheet directions to agree with the typed "
                "physical connections (do not allocate programming signals to GPIO): "
                + "; ".join(peer_contracts)
            )
        return requirement, [
            ResolutionDiagnostic(
                code="conflicting_programming_contract",
                requirement_id=requirement.id,
                recipe=definition.recipe,
                sheet=requirement.sheet,
                message="; ".join(errors),
                evidence=errors,
            )
        ]
    return requirement.model_copy(update={"ports": ports, "parameters": parameters}), []


def _complete_typed_mcu_interfaces(
    requirement: CircuitRequirement,
    architecture: Architecture,
    recipes: tuple[RegisteredRecipe, ...],
) -> tuple[CircuitRequirement, dict[str, str]]:
    """Constrain MCU functions from typed direct pads and verified CAN peers."""
    if requirement.role != "mcu_core":
        return requirement, {}
    peers = []
    required_capabilities: dict[str, str] = {}
    direct_pad_nets = {
        peer.ports["signal"]
        for peer in architecture.requirements
        if peer.role == "sensor"
        and peer.family == "touch-pad"
        and set(peer.ports) == {"signal"}
        and peer.exact_part is None
        and not peer.interfaces
    }
    for peer in architecture.requirements:
        if peer.sheet == requirement.sheet or peer.role != "bus_interface":
            continue
        matches = _family_recipes(peer.family, recipes)
        exact = _recipe_for_exact(peer.exact_part or peer.family, recipes)
        if exact is not None:
            matches = [exact]
        if len(matches) != 1 or not {"tx", "rx", "canh", "canl"} <= {
            port.name for port in matches[0].definition.ports
        }:
            continue
        peer_bindings, _diagnostics = _port_bindings(matches[0].definition, peer, architecture)
        if all(
            any(
                net.name == peer_bindings.get(port)
                and {requirement.sheet, peer.sheet} <= {ep.sheet for ep in net.endpoints}
                for net in architecture.inter_sheet_nets
            )
            for port in ("tx", "rx")
        ):
            required_capabilities.update(
                {
                    peer_bindings["tx"]: "can-tx",
                    peer_bindings["rx"]: "can-rx",
                }
            )
            peers.append(peer)
    # Both devices must bind the same conductor. A controller/front-end output
    # on another net, or a sheet/net label alone, proves no direct MCU sensor.
    for net in sorted(direct_pad_nets.intersection(requirement.ports.values())):
        required_capabilities[net] = "touch"
    if len(peers) != 1 or not {"tx", "rx"} <= peers[0].ports.keys():
        return requirement, required_capabilities
    ports = dict(requirement.ports)
    expected = {"can_tx": peers[0].ports["tx"], "can_rx": peers[0].ports["rx"]}
    if any(name in ports and ports[name] != net for name, net in expected.items()):
        return requirement, required_capabilities
    ports.update(expected)
    return requirement.model_copy(
        update={
            "ports": ports,
            "interfaces": list(dict.fromkeys([*requirement.interfaces, "can_controller"])),
        }
    ), required_capabilities


def _unowned_endpoint_diagnostics(
    definition: RecipeDefinition,
    requirement: CircuitRequirement,
    architecture: Architecture,
    bindings: dict[str, str],
    allocated_capabilities: dict[str, str],
    required_capabilities: dict[str, str],
) -> list[ResolutionDiagnostic]:
    owned = set(bindings.values()) | set(allocated_capabilities) | set(architecture.power_nets)
    # A separate typed block on the same sheet may implement a support circuit.
    owned.update(
        net
        for peer in architecture.requirements
        if peer.id != requirement.id and peer.sheet == requirement.sheet
        for net in peer.ports.values()
    )
    missing = sorted(
        {
            net.name
            for net in architecture.inter_sheet_nets
            if net.name not in owned and any(ep.sheet == requirement.sheet for ep in net.endpoints)
        }
        | {
            net
            for net, capability in required_capabilities.items()
            if allocated_capabilities.get(net) != capability
        }
    )
    if not missing:
        return []
    is_mcu = requirement.role == "mcu_core"
    return [
        ResolutionDiagnostic(
            code="missing_mcu_application_contract" if is_mcu else "unsupported_recipe_endpoint",
            requirement_id=requirement.id,
            recipe=definition.recipe,
            sheet=requirement.sheet,
            message=(
                f"requirement {requirement.id!r}, recipe {definition.recipe}, sheet {requirement.sheet!r}: "
                f"endpoints {missing} have no compatible physical recipe port or peripheral pin allocation. "
                + (
                    "Bind MCU ports and interfaces explicitly (can_controller: can_tx/can_rx; "
                    "output_<id>/input_<id> for digital I/O; touch_<id> requires verified touch capability). "
                    "Alternatively declare explicit tx/rx nets on one typed CAN transceiver peer. "
                    if is_mcu
                    else "Use the listed physical ports or declare the actual separate supporting circuit and its ports. "
                )
                + "Repair the device/interface contract; never drop committed nets or invent a chip pin."
            ),
            evidence=[
                *missing,
                *(
                    f"{net}: requires {capability}"
                    for net, capability in sorted(required_capabilities.items())
                ),
                "available_ports=" + ",".join(port.name for port in definition.ports),
                "available_capabilities="
                + ",".join(
                    sorted(
                        {
                            cap
                            for pin in definition.allocatable_pins
                            if not pin.reserved
                            for cap in pin.capabilities
                        }
                    )
                ),
            ],
        )
    ]


def resolve_architecture_recipes(
    architecture: Architecture | dict,
    intent: dict | BaseModel | None = None,
    registry: Iterable[RegisteredRecipe] | None = None,
    *,
    allowed_maturities: frozenset[str] = frozenset({"production"}),
) -> ResolutionResult:
    """Resolve supported requirements stably; protected misses are blocking."""
    architecture_model = Architecture.model_validate(architecture)
    intent_payload = (
        intent.model_dump(exclude_none=True)
        if isinstance(intent, BaseModel)
        else dict(intent or {})
    )
    # Retain exact user identities even when an older intent omitted classification.
    named_parts = list(intent_payload.get("named_parts") or [])
    supplied = {_identity(part) for part in named_parts}
    for token in named_part_tokens(
        [intent_payload.get("goal", ""), *(intent_payload.get("constraints") or [])]
    ).values():
        if _identity(token) not in supplied:
            named_parts.append(token)
            supplied.add(_identity(token))
    intent_payload["named_parts"] = named_parts
    recipes = tuple(registry) if registry is not None else registered_recipes()
    requirements, legacy_diagnostics = _legacy_requirements(
        architecture_model, intent_payload, recipes
    )
    result = ResolutionResult(
        blocking=list(legacy_diagnostics),
        protected_identities=sorted(protected_identities()),
    )
    used_ids = {row.id for row in requirements}
    for sheet, recipe in _named_mcu_sheets(architecture_model, recipes).items():
        if any(row.role == "mcu_core" and row.sheet == sheet for row in requirements):
            continue
        requirement_id = "auto_mcu_" + sheet.lower().replace(" ", "_")
        if requirement_id in used_ids:
            result.blocking.append(
                ResolutionDiagnostic(
                    code="missing_mcu_requirement",
                    sheet=sheet,
                    message=f"named MCU on {sheet!r} needs an explicit mcu_core requirement with a unique id",
                )
            )
            continue
        requirements.append(
            CircuitRequirement(
                id=requirement_id,
                sheet=sheet,
                role="mcu_core",
                family=recipe.definition.family or "mcu",
                exact_part=recipe.definition.exact_part,
            )
        )
        used_ids.add(requirement_id)
    if (architecture_model.mcu_present or _named_mcu_required(intent_payload, recipes)) and not any(
        row.role == "mcu_core" for row in requirements
    ):
        result.blocking.append(
            ResolutionDiagnostic(
                code="missing_mcu_requirement",
                message=(
                    "The declared MCU or typed upstream MCU identity requires an explicit mcu_core "
                    "requirement on its implementing sheet; keep it model-owned when no verified recipe fits"
                ),
                evidence=[sheet.name for sheet in architecture_model.sheets],
            )
        )
    architecture_model = architecture_model.model_copy(update={"requirements": requirements})
    primitive_families = {
        family for lowerer in registered_lowerers() for family in lowerer.families
    }
    lowered = {}
    for requirement in requirements:
        diagnostic = lowerer_contract_diagnostic(requirement)
        if diagnostic is None:
            if requirement.family in primitive_families:
                lowered[requirement.id] = lower_requirement(requirement)
            continue
        result.blocking.append(
            ResolutionDiagnostic(
                code="unsupported_lowerer_contract",
                requirement_id=requirement.id,
                sheet=requirement.sheet,
                message=diagnostic.message,
                evidence=diagnostic.evidence,
            )
        )
    primitive_identities = {
        requirement_id: {
            _identity(identity)
            for identity in (
                *artifact.named_part_identities,
                *(group.mpn for group in artifact.groups if group.mpn),
            )
        }
        for requirement_id, artifact in lowered.items()
        if artifact is not None
    }
    user_named = [str(value) for value in intent_payload.get("named_parts") or []]
    for named in user_named:
        named_identity = _identity(named)
        selected = _recipe_for_exact(named, recipes) or _recipe_for_unique_family_prefix(
            named, recipes
        )
        sibling = _registered_variant_for(named, recipes) if selected is None else None
        if (
            selected is None
            and sibling is None
            and _looks_protected(named, recipes)
            and _looks_exact_variant(named)
            and not any(
                len(matches := _family_recipes(row.family, recipes)) == 1
                and any(
                    part.mpn and _identity(part.mpn) == named_identity
                    for part in matches[0].definition.parts
                )
                for row in requirements
            )
        ):
            result.blocking.append(
                ResolutionDiagnostic(
                    code="unsupported_protected_variant",
                    message=f"protected variant {named!r} has no verified recipe",
                    evidence=[named, *_registered_identity_choices(named, recipes)],
                )
            )
            continue
        if sibling is not None:
            note = _substitution_note(named, sibling)
            if note not in result.assumptions:
                result.assumptions.append(note)
        if selected is None:
            # Unregistered parts remain model-owned, but an exact user token
            # still needs a typed owner rather than a mention in sheet prose.
            if not named_part_tokens([named]):
                continue
            owners = [
                row
                for row in requirements
                if named_identity in primitive_identities.get(row.id, ())
                or (
                    row.family not in primitive_families
                    and (
                        matches_part_identity(named, row.exact_part or row.family)
                        or (
                            # A typed architecture family owns the requested
                            # design block without pretending to be a BOM SKU.
                            row.exact_part is None
                            and is_part_family(named)
                            and _identity(row.family) == named_identity
                        )
                    )
                )
                or any(
                    selector.startswith(named_identity)
                    for recipe in _family_recipes(row.family, recipes)
                    for selector in _protected_selectors(recipe)
                )
                or any(
                    _identity(part.mpn or "") == _identity(named)
                    for recipe in _family_recipes(row.family, recipes)
                    for part in recipe.definition.parts
                )
            ]
        else:
            owners = [
                row
                for row in requirements
                if named_identity in primitive_identities.get(row.id, ())
                or (
                    row.family not in primitive_families
                    and (
                        _recipe_for_exact(row.exact_part or row.family, recipes)
                        or _recipe_for_unique_family_prefix(row.exact_part or row.family, recipes)
                    )
                    == selected
                )
                or selected in _family_recipes(row.family, recipes)
                or any(
                    _identity(part.mpn or "") == _identity(named)
                    for recipe in _family_recipes(row.family, recipes)
                    for part in recipe.definition.parts
                )
                # An unregistered ordering code of a registered family is owned
                # by a requirement bound to that family (the deviation is
                # recorded as an assumption and ledgers at BOM under §9.33).
                or (sibling is not None and row.family == sibling.definition.family)
            ]
        if not owners:
            choices = (
                [_recipe_requirement_choice(selected.definition)] if selected is not None else []
            )
            if sibling is not None:
                choices = [_recipe_requirement_choice(sibling.definition)]
            result.blocking.append(
                ResolutionDiagnostic(
                    code="missing_recipe_requirement",
                    recipe=selected.definition.recipe if selected is not None else None,
                    message=f"named part {named!r} needs an explicit owning requirement and sheet",
                    evidence=[named, *choices],
                )
            )
    existing_by_requirement = {
        requirement_id: selection
        for selection in architecture_model.recipe_selections
        for requirement_id in selection.requirement_ids
    }
    output_nets: dict[str, list[str]] = defaultdict(list)

    for requirement in sorted(requirements, key=lambda item: item.id):
        family_matches = _family_recipes(requirement.family, recipes)
        sheet = next(
            (row for row in architecture_model.sheets if row.name == requirement.sheet),
            None,
        )
        requirement_context = _identity(
            " ".join(
                (
                    requirement.id,
                    requirement.family,
                    sheet.function if sheet is not None else "",
                    architecture_model.topologies.get(requirement.sheet, ""),
                )
            )
        )
        if not family_matches and requirement.role in {"mcu_core", "bus_interface"}:
            contextual_matches = [
                recipe
                for recipe in recipes
                if any(
                    len(selector) >= 4 and selector in requirement_context
                    for selector in _protected_selectors(recipe)
                )
            ]
            if len(contextual_matches) == 1:
                family_matches = contextual_matches
        user_exact = next(
            (
                recipe
                for named in user_named
                if (recipe := _recipe_for_exact(named, recipes)) is not None
                and recipe in family_matches
            ),
            None,
        )
        named_exact = next(
            (
                recipe
                for named in user_named
                if (recipe := _recipe_for_exact(named, recipes)) is not None
                and _is_mcu_recipe(recipe)
            ),
            None,
        )
        if (
            named_exact is not None
            and named_exact not in family_matches
            and requirement.role == "mcu_core"
        ):
            result.blocking.append(
                ResolutionDiagnostic(
                    code="conflicting_exact_variant",
                    requirement_id=requirement.id,
                    message="user exact MCU part conflicts with architecture requirement family",
                    evidence=[
                        named_exact.definition.exact_part or named_exact.definition.recipe,
                        requirement.exact_part or requirement.family,
                    ],
                )
            )
            continue
        architecture_exact = _recipe_for_exact(
            requirement.exact_part, recipes
        ) or _recipe_for_unique_family_prefix(requirement.exact_part, recipes)
        if user_exact and architecture_exact and user_exact != architecture_exact:
            result.blocking.append(
                ResolutionDiagnostic(
                    code="conflicting_exact_variant",
                    requirement_id=requirement.id,
                    message="user and architecture exact parts select different recipes",
                    evidence=[
                        user_exact.definition.exact_part or user_exact.definition.recipe,
                        architecture_exact.definition.exact_part
                        or architecture_exact.definition.recipe,
                    ],
                )
            )
            continue
        # An explicitly selected composite family may contain the exact controller
        # as a component while exposing a different circuit-level contract.
        contained = [
            recipe
            for recipe in family_matches
            if (
                recipe.definition.exact_part is None
                or (
                    len(family_matches) == 1
                    and _identity(requirement.family) == _identity(recipe.definition.family)
                )
            )
            and requirement.exact_part
            and any(
                _identity(part.mpn) == _identity(requirement.exact_part)
                for part in recipe.definition.parts
                if part.mpn
            )
        ]
        if len(contained) == 1:
            architecture_exact = contained[0]
            if user_exact is not None and any(
                _identity(part.mpn) in {_identity(named) for named in user_named}
                for part in contained[0].definition.parts
                if part.mpn
            ):
                user_exact = contained[0]
        selected = user_exact or architecture_exact
        artifact = lowered.get(requirement.id)
        lowerer_owns_exact = artifact is not None and any(
            group.mpn and _identity(group.mpn) == _identity(requirement.exact_part)
            for group in artifact.groups
        )
        assumption_rows: list[str] = []
        if (
            requirement.exact_part
            and selected is None
            and not lowerer_owns_exact
            and (family_matches or _looks_protected(requirement.exact_part, recipes))
        ):
            sibling = _registered_variant_for(requirement.exact_part, recipes)
            if sibling is None:
                result.blocking.append(
                    ResolutionDiagnostic(
                        code="unsupported_protected_variant",
                        requirement_id=requirement.id,
                        message=(
                            f"protected variant {requirement.exact_part!r} has no verified recipe"
                        ),
                        evidence=list(
                            dict.fromkeys(
                                [
                                    requirement.exact_part,
                                    *(
                                        _recipe_requirement_choice(row.definition)
                                        for row in family_matches
                                    ),
                                    *_registered_identity_choices(requirement.exact_part, recipes),
                                ]
                            )
                        ),
                    )
                )
                continue
            # A registered family can serve this ordering code (same symbol,
            # footprint and pin map): bind that family's recipe and surface the
            # deviation as an assumption so BOM must ledger it (§9.33), instead
            # of hard-blocking on an order code the registry has no reason to
            # enumerate separately.
            sibling_recipes = _family_recipes(sibling.definition.family, recipes)
            if sibling_recipes:
                family_matches = sibling_recipes
            assumption_rows.append(_substitution_note(requirement.exact_part, sibling))
        if selected is None and family_matches:
            defaults = [recipe for recipe in family_matches if recipe.definition.default_for_family]
            if len(defaults) == 1:
                selected = defaults[0]
                assumption_rows.append(
                    f"{requirement.id}: selected {selected.definition.exact_part} as the registered family default"
                )
            elif len(family_matches) == 1:
                selected = family_matches[0]
                assumption_rows.append(
                    f"{requirement.id}: selected the only registered exact family variant"
                )
        if selected is None:
            result.unresolved_requirements.append(requirement.id)
            continue
        definition = selected.definition
        if definition.maturity not in allowed_maturities:
            result.blocking.append(
                ResolutionDiagnostic(
                    code="protected_recipe_unavailable",
                    requirement_id=requirement.id,
                    message=(
                        f"recipe {definition.recipe} maturity {definition.maturity!r} "
                        "is disabled in this execution context"
                    ),
                    evidence=[definition.recipe, definition.maturity],
                )
            )
            continue
        requirement, programming_diagnostics = _complete_typed_programming_peer(
            definition, requirement, architecture_model, recipes
        )
        domain_diagnostics = _ch340c_esp32_domain_diagnostics(
            definition, requirement, architecture_model, recipes
        )
        if programming_diagnostics or domain_diagnostics:
            result.blocking.extend([*programming_diagnostics, *domain_diagnostics])
            continue
        requirement, supply_diagnostics = _complete_ch340c_supply_mode(
            definition, requirement, architecture_model
        )
        if supply_diagnostics:
            result.blocking.extend(supply_diagnostics)
            continue
        parameters = {
            **definition.parameter_defaults,
            **{
                name: requirement.parameters[name]
                for name in definition.parameter_defaults
                if name in requirement.parameters
            },
        }
        invalid_parameters = [
            name
            for name, choices in definition.allowed_parameters.items()
            if parameters.get(name) not in choices
        ]
        if invalid_parameters:
            result.blocking.append(
                ResolutionDiagnostic(
                    code="conflicting_recipe_parameter",
                    requirement_id=requirement.id,
                    recipe=definition.recipe,
                    sheet=requirement.sheet,
                    message=f"recipe {definition.recipe} has unsupported parameter values: {invalid_parameters}",
                    evidence=[
                        f"{name}={parameters.get(name)!r}; allowed={definition.allowed_parameters[name]!r}"
                        for name in invalid_parameters
                    ],
                )
            )
            continue
        for name, value in definition.parameter_defaults.items():
            if name not in requirement.parameters:
                assumption_rows.append(
                    f"{requirement.id}: {name}={value!r} from registered circuit recipe"
                )
        if "native_usb" in definition.parameter_defaults and "usb_device" in requirement.interfaces:
            if requirement.parameters.get("native_usb") is False:
                result.blocking.append(
                    ResolutionDiagnostic(
                        code="conflicting_recipe_parameter",
                        requirement_id=requirement.id,
                        message="native_usb=False conflicts with required usb_device interface",
                        evidence=["native_usb", "usb_device"],
                    )
                )
                continue
            parameters["native_usb"] = True
        if requirement.role == "mcu_core" and definition.allocatable_pins:
            available_gpios = {
                pin.gpio
                for pin in definition.allocatable_pins
                if pin.gpio is not None and not pin.reserved
            }
            invalid_gpios = sorted(
                net.name
                for net in architecture_model.inter_sheet_nets
                if (match := re.fullmatch(r"GPIO[_-]?(\d+)", net.name, re.IGNORECASE))
                and int(match.group(1)) not in available_gpios
                and any(endpoint.sheet == requirement.sheet for endpoint in net.endpoints)
            )
            if invalid_gpios:
                result.blocking.append(
                    ResolutionDiagnostic(
                        code="unavailable_recipe_gpio",
                        requirement_id=requirement.id,
                        recipe=definition.recipe,
                        sheet=requirement.sheet,
                        message=(
                            f"requirement {requirement.id!r}, recipe {definition.recipe}, "
                            f"sheet {requirement.sheet!r}: GPIO contracts {invalid_gpios} "
                            "have no allocatable recipe pin; the reviewed allocatable GPIO set is "
                            f"{sorted(available_gpios)}. Correct ranges and typed peer pin bindings "
                            "to that set, or repair the device/interface choice if more signals "
                            "are required; do not silently drop required nets"
                        ),
                        evidence=[
                            *invalid_gpios,
                            "allocatable_gpios=" + ",".join(map(str, sorted(available_gpios))),
                        ],
                    )
                )
                continue
            inferred_ports = dict(requirement.ports)
            for net in architecture_model.inter_sheet_nets:
                match = re.fullmatch(r"GPIO[_-]?(\d+)", net.name, re.IGNORECASE)
                if (
                    match is not None
                    and int(match.group(1)) in available_gpios
                    and any(endpoint.sheet == requirement.sheet for endpoint in net.endpoints)
                ):
                    inferred_ports.setdefault(f"gpio{match.group(1)}", net.name)
            requirement = requirement.model_copy(update={"ports": inferred_ports})
        requirement, required_capabilities = _complete_typed_mcu_interfaces(
            requirement, architecture_model, recipes
        )
        if "can_controller" in requirement.interfaces and parameters.get("can_remap"):
            assumption_rows.append(
                f"{requirement.id}: firmware must enable CAN remap {parameters['can_remap']}"
            )
        bindings, port_diagnostics = _port_bindings(definition, requirement, architecture_model)
        if "auto_reset" in definition.parameter_defaults:
            programming = {
                name: bindings[name] for name in _PROGRAMMING_PEER_PORTS if name in bindings
            }
            incomplete = parameters["auto_reset"] and len(programming) != len(
                _PROGRAMMING_PEER_PORTS
            )
            disabled = not parameters["auto_reset"] and {"dtr_n", "rts_n"} & programming.keys()
            aliased = len(set(programming.values())) != len(programming)
            if incomplete or disabled or aliased:
                result.blocking.append(
                    ResolutionDiagnostic(
                        code="conflicting_programming_contract",
                        requirement_id=requirement.id,
                        recipe=definition.recipe,
                        sheet=requirement.sheet,
                        message=(
                            "Automatic reset requires auto_reset=True and four distinct typed "
                            "uart_tx, uart_rx, dtr_n, rts_n bindings; incomplete or aliased controls "
                            "cannot be wired directly to EN/BOOT or allocated to GPIO."
                        ),
                        evidence=[f"{name}={net}" for name, net in sorted(programming.items())],
                    )
                )
                continue
        if "native_usb" in definition.parameter_defaults and {"usb_dm", "usb_dp"} <= set(bindings):
            parameters["native_usb"] = True
        if port_diagnostics:
            result.blocking.extend(port_diagnostics)
            continue
        existing = existing_by_requirement.get(requirement.id)
        allocation_requirement = requirement
        if "uart" in requirement.interfaces:
            # A proved fixed UART already implements this interface. Only an
            # additional application tx/rx contract needs allocatable MCU pins.
            uart_ports = (
                ("uart_tx", "uart_rx")
                if requirement.role == "mcu_core" and not {"tx", "rx"} & requirement.ports.keys()
                else ("tx", "rx")
            )
            fixed_directions = {port.name: port.direction for port in definition.ports}
            if (
                all(
                    name in bindings and fixed_directions.get(name) == direction
                    for name, direction in zip(uart_ports, ("output", "input"), strict=True)
                )
                and bindings[uart_ports[0]] != bindings[uart_ports[1]]
                and not any(
                    endpoint.direction in {"input", "output"}
                    and endpoint.direction != fixed_directions[name]
                    for name in uart_ports
                    for net in architecture_model.inter_sheet_nets
                    if net.name == bindings[name]
                    for endpoint in net.endpoints
                    if endpoint.sheet == requirement.sheet
                )
            ):
                allocation_requirement = requirement.model_copy(
                    update={
                        "interfaces": [name for name in requirement.interfaces if name != "uart"]
                    }
                )
        try:
            allocations = allocate_requirement_pins(
                definition,
                allocation_requirement,
                existing=(existing.pin_allocations if existing is not None else None),
                # CAN retains its endpoint-contract diagnostics; direct pads
                # additionally strengthen generic input allocation requests.
                required_capabilities={
                    net: capability
                    for net, capability in required_capabilities.items()
                    if capability == "touch"
                },
            )
        except PinAllocationError as exc:
            result.blocking.append(
                ResolutionDiagnostic(
                    code=exc.code,
                    requirement_id=requirement.id,
                    message=str(exc),
                    evidence=exc.evidence
                    + [
                        value
                        for value in (exc.capability, str(exc.count) if exc.count else None)
                        if value
                    ],
                )
            )
            continue
        endpoint_diagnostics = _unowned_endpoint_diagnostics(
            definition,
            requirement,
            architecture_model,
            bindings,
            {allocation.net: allocation.capability for allocation in allocations},
            required_capabilities,
        )
        if endpoint_diagnostics:
            result.blocking.extend(endpoint_diagnostics)
            continue
        edge_companion = _recipe_edge_companion(
            definition,
            requirement,
            requirements,
            frozenset(existing_by_requirement),
        )
        sheet_map = {role: requirement.sheet for role in definition.required_sheet_roles}
        selection_requirement_ids = [requirement.id]
        if edge_companion is not None:
            edge_sheet_role, companion = edge_companion
            sheet_map[edge_sheet_role] = companion.sheet
            selection_requirement_ids.append(companion.id)
        selection = RecipeSelection(
            recipe=definition.recipe,
            instance=(existing.instance if existing is not None else requirement.id),
            sheets=sheet_map,
            parameters=parameters,
            port_bindings=bindings,
            requirement_ids=selection_requirement_ids,
            pin_allocations=allocations,
        )
        result.selections.append(selection)
        exact_part = definition.exact_part or requirement.exact_part
        if exact_part is not None:
            result.exact_parts[requirement.id] = exact_part
        result.assumptions.extend(assumption_rows)
        result.records.append(
            RecipeResolutionRecord(
                requirement_id=requirement.id,
                recipe=definition.recipe,
                exact_part=exact_part or definition.recipe,
                assumptions=assumption_rows,
            )
        )
        result.requirements.append(
            requirement.model_copy(
                update={"exact_part": exact_part, "ports": {**requirement.ports, **bindings}}
            )
        )
    _coalesce_mcu_support_requirements(result, requirements, recipes)
    definitions = {recipe.definition.recipe: recipe.definition for recipe in recipes}
    for selection in result.selections:
        for port in definitions[selection.recipe].ports:
            if port.direction == "output" and port.name in selection.port_bindings:
                output_nets[selection.port_bindings[port.name]].append(selection.instance)
    for net, owners in sorted(output_nets.items()):
        if len(owners) > 1:
            result.blocking.append(
                ResolutionDiagnostic(
                    code="recipe_output_port_collision",
                    message=f"multiple output-only recipe ports bind net {net!r}",
                    evidence=owners,
                )
            )
    result.selections.sort(key=lambda selection: (selection.recipe, selection.instance))
    resolved_requirements = {row.id: row for row in result.requirements}
    result.requirements = [resolved_requirements.get(row.id, row) for row in requirements]
    result.records.sort(key=lambda record: record.requirement_id)
    selected_requirement_ids = {
        requirement_id
        for selection in result.selections
        for requirement_id in selection.requirement_ids
    }
    result.unresolved_requirements = sorted(
        set(result.unresolved_requirements) - selected_requirement_ids
    )
    # The same deviation note can be raised by the named-part pass and the
    # requirement pass; assumptions are a set-like ledger, not a log.
    result.assumptions = list(dict.fromkeys(result.assumptions))
    result.blocking.sort(
        key=lambda diagnostic: (
            diagnostic.code != "recipe_signal_in_power_nets",
            diagnostic.code,
            diagnostic.requirement_id or "",
        )
    )
    return result


def apply_architecture_recipe_resolution(
    architecture: Architecture | dict,
    intent: dict | BaseModel | None = None,
    *,
    allowed_maturities: frozenset[str] = frozenset({"production"}),
) -> Architecture:
    architecture_model = Architecture.model_validate(architecture)
    result = resolve_architecture_recipes(
        architecture_model,
        intent,
        allowed_maturities=allowed_maturities,
    )
    if result.blocking:
        raise RecipeResolutionError(result.blocking)
    requirements = result.requirements
    return architecture_model.model_copy(
        update={
            "mcu_present": architecture_model.mcu_present
            or any(row.role == "mcu_core" for row in requirements),
            "requirements": requirements,
            "recipe_selections": result.selections,
            "recipe_resolution": result.records,
            "unresolved_requirement_ids": result.unresolved_requirements,
            "protected_identities": result.protected_identities,
            "assumptions": list(
                dict.fromkeys([*architecture_model.assumptions, *result.assumptions])
            ),
        }
    )
