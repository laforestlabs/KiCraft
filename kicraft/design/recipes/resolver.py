"""Architecture-time deterministic recipe resolution and protected-family blocking."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from kicraft.design.models import (
    Architecture,
    CircuitRequirement,
    RecipeResolutionRecord,
    RecipeSelection,
)

from .models import RegisteredRecipe
from .pin_allocator import PinAllocationError, allocate_requirement_pins
from .registry import protected_identities, registered_recipes


def _identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


class ResolutionDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    requirement_id: str | None = None
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


class RecipeResolutionError(ValueError):
    def __init__(self, diagnostic: ResolutionDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic.code}: {diagnostic.message}")


def _exact_selectors(recipe: RegisteredRecipe) -> set[str]:
    definition = recipe.definition
    return {
        _identity(value)
        for value in (definition.exact_part, *definition.identity_aliases)
        if value
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
    token = _identity(value)
    return any(
        marker in token
        for marker in (
            "mini1",
            "wroom",
            "c3",
            "c8t6",
            "tiny402",
            "tiny412",
            "tiny1614",
            "ch32v003",
        )
    )


def _mcu_sheet(architecture: Architecture) -> str | None:
    ranked = sorted(
        architecture.sheets,
        key=lambda sheet: (
            0
            if re.search(r"\b(?:mcu|controller|esp32|rp2040|stm32)\b", f"{sheet.name} {sheet.function}", re.I)
            else 1,
            architecture.sheets.index(sheet),
        ),
    )
    return ranked[0].name if ranked else None


def _net_by_tokens(architecture: Architecture, *tokens: str) -> str | None:
    names = [*architecture.power_nets, *(net.name for net in architecture.inter_sheet_nets)]
    wanted = {_identity(token) for token in tokens}
    return next((name for name in names if _identity(name) in wanted), None)


def _legacy_requirements(
    architecture: Architecture,
    intent: dict,
    recipes: tuple[RegisteredRecipe, ...],
) -> tuple[list[CircuitRequirement], list[ResolutionDiagnostic]]:
    if architecture.requirements:
        return list(architecture.requirements), []
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
    sheet = _mcu_sheet(architecture)
    for candidate in candidates:
        exact = _recipe_for_exact(candidate, recipes)
        if exact is not None and exact.definition.recipe not in seen_recipes and sheet:
            definition = exact.definition
            ports: dict[str, str] = {}
            vdd = _net_by_tokens(architecture, "+3V3", "3V3", "VDD")
            if vdd:
                ports["vdd"] = vdd
            ports["gnd"] = "GND"
            usb_dm = _net_by_tokens(architecture, "USB_DM", "USB_D-")
            usb_dp = _net_by_tokens(architecture, "USB_DP", "USB_D+")
            interfaces: list[str] = []
            if usb_dm and usb_dp:
                ports.update({"usb_dm": usb_dm, "usb_dp": usb_dp})
                interfaces.append("usb_device")
            requirements.append(
                CircuitRequirement(
                    id=f"auto_{definition.family or 'circuit'}".replace("-", "_"),
                    sheet=sheet,
                    role="mcu_core",
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
    if not requirements and _looks_protected(architecture_text, recipes) and sheet:
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
                    sheet=sheet,
                    role="mcu_core",
                    family=definition.family or "mcu",
                    ports={
                        "gnd": "GND",
                        **(
                            {"vdd": vdd}
                            if (vdd := _net_by_tokens(architecture, "+3V3", "3V3", "VDD"))
                            else {}
                        ),
                    },
                )
            )
    return requirements, diagnostics


def _port_bindings(
    definition,
    requirement: CircuitRequirement,
    architecture: Architecture,
) -> tuple[dict[str, str], list[ResolutionDiagnostic]]:
    bindings = dict(requirement.ports)
    declared = {port.name: port for port in definition.ports}
    bindings = {name: net for name, net in bindings.items() if name in declared}
    if "gnd" in declared:
        bindings.setdefault("gnd", "GND")
    if "vdd" in declared:
        vdd = _net_by_tokens(architecture, "+3V3", "3V3", "VDD")
        if vdd:
            bindings.setdefault("vdd", vdd)
    known_nets = {
        "GND",
        *architecture.power_nets,
        *(net.name for net in architecture.inter_sheet_nets),
    }
    diagnostics: list[ResolutionDiagnostic] = []
    missing = [
        name for name, port in declared.items() if port.required and not bindings.get(name)
    ]
    if missing:
        diagnostics.append(
            ResolutionDiagnostic(
                code="missing_recipe_port",
                requirement_id=requirement.id,
                message=f"recipe {definition.recipe} requires ports {sorted(missing)}",
                evidence=sorted(missing),
            )
        )
    unknown = sorted({net for net in bindings.values() if net not in known_nets})
    if unknown:
        diagnostics.append(
            ResolutionDiagnostic(
                code="unknown_recipe_port_net",
                requirement_id=requirement.id,
                message="recipe port bindings reference undeclared architecture nets",
                evidence=unknown,
            )
        )
    return dict(sorted(bindings.items())), diagnostics

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
    recipes = tuple(registry) if registry is not None else registered_recipes()
    requirements, legacy_diagnostics = _legacy_requirements(
        architecture_model, intent_payload, recipes
    )
    result = ResolutionResult(
        blocking=list(legacy_diagnostics),
        protected_identities=sorted(protected_identities()),
    )
    user_named = [str(value) for value in intent_payload.get("named_parts") or []]
    existing_by_requirement = {
        requirement_id: selection
        for selection in architecture_model.recipe_selections
        for requirement_id in selection.requirement_ids
    }
    output_nets: dict[str, list[str]] = defaultdict(list)

    for requirement in sorted(requirements, key=lambda item: item.id):
        family_matches = _family_recipes(requirement.family, recipes)
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
                        named_exact.definition.exact_part
                        or named_exact.definition.recipe,
                        requirement.exact_part or requirement.family,
                    ],
                )
            )
            continue
        architecture_exact = _recipe_for_exact(requirement.exact_part, recipes)
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
        selected = user_exact or architecture_exact
        if requirement.exact_part and selected is None and (
            family_matches or _looks_protected(requirement.exact_part, recipes)
        ):
            result.blocking.append(
                ResolutionDiagnostic(
                    code="unsupported_protected_variant",
                    requirement_id=requirement.id,
                    message=(
                        f"protected variant {requirement.exact_part!r} has no verified recipe"
                    ),
                    evidence=[requirement.exact_part],
                )
            )
            continue
        assumption_rows: list[str] = []
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
        parameters = {
            **definition.parameter_defaults,
            **{
                name: requirement.parameters[name]
                for name in definition.parameter_defaults
                if name in requirement.parameters
            },
        }
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
        bindings, port_diagnostics = _port_bindings(
            definition, requirement, architecture_model
        )
        if parameters.get("native_usb") and not {
            "usb_dm",
            "usb_dp",
        } <= set(bindings):
            result.blocking.append(
                ResolutionDiagnostic(
                    code="missing_recipe_port",
                    requirement_id=requirement.id,
                    message="native USB requires usb_dm and usb_dp architecture bindings",
                    evidence=["usb_dm", "usb_dp"],
                )
            )
            continue
        if port_diagnostics:
            result.blocking.extend(port_diagnostics)
            continue
        existing = existing_by_requirement.get(requirement.id)
        try:
            allocations = allocate_requirement_pins(
                definition,
                requirement,
                existing=(existing.pin_allocations if existing is not None else None),
            )
        except PinAllocationError as exc:
            result.blocking.append(
                ResolutionDiagnostic(
                    code=exc.code,
                    requirement_id=requirement.id,
                    message=str(exc),
                    evidence=[
                        value
                        for value in (exc.capability, str(exc.count) if exc.count else None)
                        if value
                    ],
                )
            )
            continue
        sheet_map = {
            role: requirement.sheet for role in definition.required_sheet_roles
        }
        selection = RecipeSelection(
            recipe=definition.recipe,
            instance=(existing.instance if existing is not None else requirement.id),
            sheets=sheet_map,
            parameters=parameters,
            port_bindings=bindings,
            requirement_ids=[requirement.id],
            pin_allocations=allocations,
        )
        for port in definition.ports:
            if port.direction == "output" and port.name in bindings:
                output_nets[bindings[port.name]].append(requirement.id)
        result.selections.append(selection)
        exact_part = definition.exact_part or definition.recipe
        result.exact_parts[requirement.id] = exact_part
        result.assumptions.extend(assumption_rows)
        result.records.append(
            RecipeResolutionRecord(
                requirement_id=requirement.id,
                recipe=definition.recipe,
                exact_part=exact_part,
                assumptions=assumption_rows,
            )
        )
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
    result.records.sort(key=lambda record: record.requirement_id)
    result.unresolved_requirements.sort()
    result.blocking.sort(key=lambda diagnostic: (diagnostic.code, diagnostic.requirement_id or ""))
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
        raise RecipeResolutionError(result.blocking[0])
    requirements, _diagnostics = _legacy_requirements(
        architecture_model,
        (
            intent.model_dump(exclude_none=True)
            if isinstance(intent, BaseModel)
            else dict(intent or {})
        ),
        registered_recipes(),
    )
    return architecture_model.model_copy(
        update={
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
