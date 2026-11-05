"""Typed deterministic circuit lowering from ``CircuitRequirement`` objects.

A lowerer emits BOM groups and their wiring from one immutable artifact. Matching
is exact on the requirement family; no prose, regex, or ordered first-match path
participates.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from kicraft.design.models import BOMAccessory, CircuitRequirement, JsonScalar, RequirementObligation
from kicraft.design.part_identity import quantity_subject_binds, reviewed_part


class LoweringArray(BaseModel):
    """The placement pattern one lowered group's members must be laid out on.

    A board feature whose geometry *is* the deliverable (a prototyping pad
    field on a 0.1 inch grid) cannot be handed to the force/simulated-annealing
    placer: the solver would scatter the members, and the delivered board would
    no longer carry the pitch the field is for. The lowerer therefore declares
    the pattern here, and the BOM stage turns it into a ``models.ArraySpec`` so
    the array placer lays the group out programmatically.
    """

    model_config = ConfigDict(extra="forbid")

    pattern: Literal["grid", "ring"] = "grid"
    rows: int | None = Field(default=None, gt=0)
    cols: int | None = Field(default=None, gt=0)
    pitch_mm: float | None = Field(default=None, gt=0)
    serpentine: bool = True
    radius_mm: float | None = Field(default=None, gt=0)
    start_angle_deg: float = 0.0


class LoweringGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    reference_prefix: str = Field(pattern=r"^[A-Z]+$")
    quantity: int = Field(default=1, ge=1, le=500)
    value: str
    symbol: str
    footprint: str
    mpn: str | None = None
    datasheet: str | None = None
    sourcing_note: str | None = None
    # Hand-fitted, non-PCB items purchased with each instance of this part.
    accessories: tuple[BOMAccessory, ...] = ()
    # Board-fabricated copper features remain on the PCB but are omitted from
    # assembly BOM and position exports.
    assembly: bool = True
    # Declared only when the members' pattern is part of the deliverable.
    array: LoweringArray | None = None


class LoweringPin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    index: int = Field(default=0, ge=0)
    pin: str
    net: str


class LoweringNoConnect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    index: int = Field(default=0, ge=0)
    pin: str
    reason: str


class LoweringCalculation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: str
    equation: str
    inputs: dict[str, float | str]
    chosen_value: str
    tolerance: str


class LoweringLocalBypass(BaseModel):
    """Owned bypass pad pairs whose PCB distances are mandatory, in millimetres."""

    model_config = ConfigDict(extra="forbid")

    owner_role: str
    owner_index: int = Field(default=0, ge=0)
    bypass_role: str
    bypass_index: int = Field(default=0, ge=0)
    power_pins: tuple[str, str]
    ground_pins: tuple[str, str]
    max_distance_mm: float = Field(gt=0)



class LoweringArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lowerer_id: str
    requirement_id: str
    groups: tuple[LoweringGroup, ...]
    pins: tuple[LoweringPin, ...]
    no_connects: tuple[LoweringNoConnect, ...] = ()
    assumptions: tuple[str, ...] = ()
    calculations: tuple[LoweringCalculation, ...] = ()
    local_bypasses: tuple[LoweringLocalBypass, ...] = ()
    # Compiler-proven semantic identities, never copied from requirement labels.
    named_part_identities: tuple[str, ...] = ()


class LoweringParameterDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    sheet: str
    family: str
    lowerer_id: str
    parameter: str
    value: JsonScalar
    choices: tuple[JsonScalar, ...]
    missing: bool
    constraint: str | None = None


class LoweringContractDiagnostic(BaseModel):
    """A known lowerer's refused contract, reported at architecture ownership."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    sheet: str
    family: str
    lowerer_id: str
    message: str
    evidence: list[str] = Field(default_factory=list)


_LOWERER_PORT_DIRECTIONS = {
    "gnd": "bidirectional",
    "negative": "bidirectional",
    "shield": "bidirectional",
    "vdd": "input",
    "vcc": "input",
    "vbus": "input",
    "vin": "input",
    "vm": "input",
    "input": "input",
    "positive": "input",
    "output": "output",
    "drive": "input",
    "signal": "input",
    "sda": "bidirectional",
    "scl": "bidirectional",
}

@dataclass(frozen=True)
class RegisteredLowerer:
    lowerer_id: str
    families: frozenset[str]
    build: Callable[[CircuitRequirement], LoweringArtifact | None]
    parameter_keys: tuple[str, ...] = ()
    port_keys: tuple[str, ...] = ()
    required_port_keys: tuple[str, ...] = ()
    parameter_choices: tuple[tuple[str, tuple[JsonScalar, ...]], ...] = ()
    required_parameter_keys: tuple[str, ...] = ()
    # Exact logical ports and regular keyed vocabularies are both part of the
    # lowerer's public architecture contract. A known lowerer must not become
    # model-owned merely because its contract was malformed.
    port_directions: tuple[tuple[str, str], ...] = ()
    port_patterns: tuple[tuple[str, str], ...] = ()
    #: Spelling groups a build accepts for one published contact: canonical key -> every spelling
    #: that denotes it. A required contact bound under any of its spellings *is* bound, so the
    #: contract diagnostic must read the same groups instead of reporting the canonical key
    #: unbound while the build itself accepts the draft (live rc-lowpass: `output` *is* the
    #: trimmer's published wiper spelling, and the refusal named `wiper`).
    port_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: Published contacts that *are* the part's return, spelled as the family names them.
    #: `_reference_port_name` reads the conventional tokens (`gnd`, `vss`); a family whose
    #: contract names its return something else (`negative` on a coin cell, `sleeve` on a
    #: TRS jack) publishes it here, so the compiler answers it with the design's ground
    #: instead of asking a draft to restate a fact the reviewed part already holds.
    reference_port_keys: tuple[str, ...] = ()
    # The reviewed ordering code the lowerer realizes. A build that refuses every other code
    # (including none) publishes it here, so the refusal can name the part the draft must state.
    required_exact_part: str | None = None
    # The reviewed ordering code a build realizes *when one is stated*; the family still accepts a
    # requirement with no `exact_part`. Published so a draft names a part the lowerer can build.
    reviewed_exact_part: str | None = None
    # The reviewed ordering codes a composition can realize when its own finite
    # implementation supports more than one carrier. This is the same exact_part
    # contract as `reviewed_exact_part`, not a generic part-family substitution.
    reviewed_exact_parts: tuple[str, ...] = ()
    # Cross-parameter limits that individual choice lists cannot express.
    parameter_constraints: tuple[str, ...] = ()
    # Compiler-owned circuit nodes: valid in a completed lowering requirement,
    # but not contacts a draft may wire or discover as public architecture ports.
    derived_port_patterns: tuple[tuple[str, str], ...] = ()


_REGISTRY: dict[str, RegisteredLowerer] = {}
_FAMILIES: dict[str, str] = {}


def register_lowerer(lowerer: RegisteredLowerer) -> None:
    if lowerer.lowerer_id in _REGISTRY:
        raise ValueError(f"duplicate lowerer {lowerer.lowerer_id!r}")
    if "@" not in lowerer.lowerer_id:
        raise ValueError("lowerer ids must be immutable @N versions")
    overlap = set(lowerer.families) & set(_FAMILIES)
    if overlap:
        raise ValueError(f"lowerer family overlap: {sorted(overlap)}")
    _REGISTRY[lowerer.lowerer_id] = lowerer
    for family in lowerer.families:
        _FAMILIES[family] = lowerer.lowerer_id


def registered_lowerers() -> tuple[RegisteredLowerer, ...]:
    return tuple(_REGISTRY[key] for key in sorted(_REGISTRY))


def lowerer_family_for_class(family: str, parameters: object) -> str | None:
    """The lowerer family that owns the support parts a class-named requirement needs, or None.

    A draft may name the reviewed *part class* it wants (`led-0603`) where the stage's reference
    data lists lowerer *families* (`status-led`). The class is what the reviewed record carries, so
    the name reads as legitimate -- and the build then emits the bare part with none of the support
    the circuit needs: the live 2026-09-25 converter draft wired a LED straight across the 3.3 V
    rail, the audit caught it at 0.98 confidence, and two repair rounds could not talk the writer
    out of the name. No deterministic gate covered it either, because §9.36 (the typed LED
    current-path check) keys on the lowerer family.

    Deliberately narrow: the class must be one the candidate lowerer family denotes, every
    parameter the draft states must be one that lowerer publishes, and the draft must state every
    parameter that lowerer requires. Anything else is left exactly as written.
    """
    from kicraft.design.part_identity import canonical_physical_features

    wanted = str(family or "").strip().casefold()
    stated = {str(key) for key in (parameters or {})}
    if not wanted or not stated:
        return None
    for lowerer in registered_lowerers():
        for candidate in sorted(lowerer.families):
            if wanted not in canonical_physical_features(candidate):
                continue
            if not stated <= set(lowerer.parameter_keys):
                continue
            if not set(lowerer.required_parameter_keys) <= stated:
                continue
            return candidate
    return None


def lowerer_summaries() -> list[dict]:
    return [
        {
            "lowerer": lowerer.lowerer_id,
            "families": sorted(lowerer.families),
            "parameter_keys": list(lowerer.parameter_keys),
            "port_keys": list(lowerer.port_keys),
            **(
                {"port_directions": dict(lowerer.port_directions)}
                if lowerer.port_directions
                else {}
            ),
            **(
                {"required_port_keys": list(lowerer.required_port_keys)}
                if lowerer.required_port_keys
                else {}
            ),
            **(
                {
                    "parameter_choices": {
                        key: list(values) for key, values in lowerer.parameter_choices
                    }
                }
                if lowerer.parameter_choices
                else {}
            ),
            **(
                {"parameter_constraints": list(lowerer.parameter_constraints)}
                if lowerer.parameter_constraints
                else {}
            ),
            **(
                {"required_parameter_keys": list(lowerer.required_parameter_keys)}
                if lowerer.required_parameter_keys
                else {}
            ),
            **(
                {"port_patterns": [pattern for pattern, _direction in lowerer.port_patterns]}
                if lowerer.port_patterns
                else {}
            ),
            **(
                {
                    "port_aliases": {
                        key: list(spellings) for key, spellings in lowerer.port_aliases
                    }
                }
                if lowerer.port_aliases
                else {}
            ),
            **(
                {"required_exact_part": lowerer.required_exact_part}
                if lowerer.required_exact_part
                else {}
            ),
            **(
                {"reviewed_exact_part": lowerer.reviewed_exact_part}
                if lowerer.reviewed_exact_part
                else {}
            ),
            **(
                {"reviewed_exact_parts": list(lowerer.reviewed_exact_parts)}
                if lowerer.reviewed_exact_parts
                else {}
            ),
        }
        for lowerer in registered_lowerers()
    ]


def lowerer_port_direction(
    lowerer: RegisteredLowerer, name: str, *, include_derived: bool = False
) -> str | None:
    """Return a registered direction; compiler-owned nodes are opt-in."""

    name = name.lower()
    directions = {key.lower(): direction for key, direction in lowerer.port_directions}
    if name in directions:
        return directions[name]
    # Concrete `port_keys` from older registrations remain part of the public
    # contract. Their directions come from this reviewed semantic table; prose
    # placeholders still require an explicit registered pattern below.
    concrete = {key.lower() for key in lowerer.port_keys if key.isidentifier()}
    if name in concrete:
        return _LOWERER_PORT_DIRECTIONS.get(name, "bidirectional")
    for pattern, direction in lowerer.port_patterns:
        if re.fullmatch(pattern, name):
            return direction
    if include_derived:
        for pattern, direction in lowerer.derived_port_patterns:
            if re.fullmatch(pattern, name):
                return direction
    return None

def lowerer_parameter_diagnostics(
    requirement: CircuitRequirement,
) -> list[LoweringParameterDiagnostic]:
    """Report only declared finite/required contracts, not model-owned parameters."""
    lowerer_id = _FAMILIES.get(requirement.family)
    if lowerer_id is None:
        return []
    lowerer = _REGISTRY[lowerer_id]
    if not lowerer.parameter_choices and not lowerer.required_parameter_keys:
        return []
    choices_by_key = dict(lowerer.parameter_choices)
    diagnostics = []
    for key in dict.fromkeys((*lowerer.required_parameter_keys, *choices_by_key)):
        missing = key not in requirement.parameters
        if missing and key not in lowerer.required_parameter_keys:
            continue
        value = requirement.parameters.get(key)
        choices = choices_by_key.get(key, ())
        # JSON booleans must not compare equal to numeric choices such as rows=1.
        if not missing and (
            not choices
            or any(
                value == choice and isinstance(value, bool) == isinstance(choice, bool)
                for choice in choices
            )
        ):
            continue
        diagnostics.append(
            LoweringParameterDiagnostic(
                requirement_id=requirement.id,
                sheet=requirement.sheet,
                family=requirement.family,
                lowerer_id=lowerer_id,
                parameter=key,
                value=value,
                choices=choices,
                missing=missing,
            )
        )
    if lowerer_id == "led-current-resistor@1" and not diagnostics:
        try:
            _led_resistor_parameters(requirement)
        except ValueError as exc:
            key, constraint = exc.args
            diagnostics.append(
                LoweringParameterDiagnostic(
                    requirement_id=requirement.id,
                    sheet=requirement.sheet,
                    family=requirement.family,
                    lowerer_id=lowerer_id,
                    parameter=key,
                    value=requirement.parameters.get(key),
                    choices=(),
                    missing=key not in requirement.parameters,
                    constraint=constraint,
                )
            )
    return diagnostics


def lowerer_contract_diagnostic(
    requirement: CircuitRequirement,
) -> LoweringContractDiagnostic | None:
    """Explain why a registered lowerer cannot own a claimed requirement."""

    lowerer_id = _FAMILIES.get(requirement.family)
    if lowerer_id is None:
        return None
    lowerer = _REGISTRY[lowerer_id]
    unknown_ports = sorted(
        name for name in requirement.ports
        if lowerer_port_direction(lowerer, name, include_derived=True) is None
    )
    parameter_diagnostics = lowerer_parameter_diagnostics(requirement)
    failure = None
    if not unknown_ports and not parameter_diagnostics:
        try:
            if lower_requirement(requirement) is not None:
                return None
        except ValueError as exc:
            failure = str(exc)
    evidence = [
        "ports=" + ",".join(name for name, _direction in lowerer.port_directions),
        *(
            "port_patterns=" + ",".join(pattern for pattern, _direction in lowerer.port_patterns),
        ),
        *(["port_contract=" + "; ".join(lowerer.port_keys)] if lowerer.port_keys else []),
        "requirement_ports=" + ",".join(sorted(requirement.ports)),
        "parameters=" + ",".join(lowerer.parameter_keys),
    ]
    if lowerer.required_port_keys:
        evidence.insert(0, "required_ports=" + ",".join(lowerer.required_port_keys))
    if lowerer.parameter_choices:
        evidence.append(
            "parameter_choices="
            + ",".join(f"{key}={list(values)}" for key, values in lowerer.parameter_choices)
        )
    if lowerer.required_parameter_keys:
        evidence.append("required_parameters=" + ",".join(lowerer.required_parameter_keys))
    if lowerer.required_exact_part:
        evidence.append("required_exact_part=" + lowerer.required_exact_part)
    if lowerer.reviewed_exact_part:
        evidence.append("reviewed_exact_part=" + lowerer.reviewed_exact_part)
    if lowerer.reviewed_exact_parts:
        evidence.append("reviewed_exact_parts=" + ",".join(lowerer.reviewed_exact_parts))
    if failure:
        evidence.append(failure)
    # A published contact is bound when the draft spells it any way its own build accepts: the
    # alias groups are the build's, published, so the check reads them instead of reporting the
    # canonical key unbound while the build would take the draft.
    bound = {str(name).lower() for name in requirement.ports}
    for canonical, spellings in lowerer.port_aliases:
        if any(str(spelling).lower() in bound for spelling in (*spellings, canonical)):
            bound.add(str(canonical).lower())
    missing_required = sorted(
        key for key in lowerer.required_port_keys if str(key).lower() not in bound
    )
    unpublished_parameters = sorted(set(requirement.parameters) - set(lowerer.parameter_keys))
    if not requirement.ports and (
        lowerer.required_port_keys or lowerer.port_directions or lowerer.port_patterns
    ):
        # The lowerer owns this family, so a model-authored fallback is refused; the
        # requirement has simply declared no contacts. Name the exact ports its build
        # code wants (published as required_port_keys) instead of the generic
        # "cannot realize this port/parameter combination" a model cannot act on.
        wanted = (
            "bind every contact — " + ", ".join(lowerer.required_port_keys) + " —"
            if lowerer.required_port_keys
            else "bind every contact using its published port contract (see evidence)"
        )
        message = (
            f"known lowerer {lowerer_id} needs this requirement's ports declared, but it declares "
            f"none; {wanted} so the lowerer can build the part"
        )
    elif unknown_ports:
        message = (
            f"known lowerer {lowerer_id} does not support ports {unknown_ports}; "
            "use its published port contract or choose a genuinely model-owned family"
        )
    elif unpublished_parameters:
        # A parameter the published contract does not hold is what `lower_requirement` refuses
        # *before* it ever builds, so it is the repair the draft owes. Name the keys, the accepted
        # keys, and the accepted values: the live rc-lowpass drafts stated
        # `capacitance_f`/`capacitor_exact_part` on a bare trimmer and were told their `wiper` was
        # unbound instead, which is not what stopped the build.
        accepted = ", ".join(lowerer.parameter_keys) or "none — this part takes no parameters"
        choices = (
            "; accepted values: "
            + ", ".join(f"{key}={list(values)}" for key, values in lowerer.parameter_choices)
            if lowerer.parameter_choices
            else (
                "; required parameters: " + ", ".join(lowerer.required_parameter_keys)
                if lowerer.required_parameter_keys
                else ""
            )
        )
        message = (
            f"known lowerer {lowerer_id} does not accept parameters "
            f"{', '.join(unpublished_parameters)}; it accepts {accepted}{choices} "
            "— delete each parameter this part's published contract does not hold"
        )
    elif missing_required:
        # The draft named part of the contract and left a contact unbound. Name the ports the
        # published contract still requires rather than the generic "cannot realize" sentence.
        message = (
            f"known lowerer {lowerer_id} needs every published contact bound, but "
            f"{missing_required} {'is' if len(missing_required) == 1 else 'are'} unbound; its "
            f"required ports are {', '.join(lowerer.required_port_keys)}"
        )
    elif parameter_diagnostics:
        # Name the offending key, the value the draft gave it, and the accepted values, so the
        # refusal is repairable without reading the lowerer's source.
        detail = "; ".join(
            f"parameters.{row.parameter}="
            + ("<missing>" if row.missing else repr(row.value))
            + (
                f" ({row.constraint})"
                if row.constraint
                else (
                    f" (accepted: {list(row.choices)!r})"
                    if row.choices
                    else " (published contract)"
                )
            )
            for row in parameter_diagnostics[:4]
        )
        message = (
            f"known lowerer {lowerer_id} has unsupported parameters: {detail}; "
            "use its published parameter contract"
        )
    elif lowerer.required_exact_part and (
        requirement.exact_part or ""
    ).casefold() != lowerer.required_exact_part.casefold():
        message = (
            f"known lowerer {lowerer_id} realizes exactly the reviewed part "
            f"{lowerer.required_exact_part!r}; set exact_part to that ordering code"
        )
    elif (
        lowerer.reviewed_exact_part
        and requirement.exact_part
        and requirement.exact_part.casefold() != lowerer.reviewed_exact_part.casefold()
    ):
        message = (
            f"known lowerer {lowerer_id} realizes the reviewed part "
            f"{lowerer.reviewed_exact_part!r}, not {requirement.exact_part!r}; state that ordering "
            "code or omit exact_part"
        )
    elif (
        lowerer.reviewed_exact_parts
        and requirement.exact_part
        and requirement.exact_part.casefold()
        not in {part.casefold() for part in lowerer.reviewed_exact_parts}
    ):
        message = (
            f"known lowerer {lowerer_id} realizes one of the reviewed parts "
            f"{list(lowerer.reviewed_exact_parts)!r}, not {requirement.exact_part!r}; state one "
            "of those ordering codes or omit exact_part"
        )
    else:
        message = (
            f"known lowerer {lowerer_id} cannot realize this port/parameter combination"
            # The build's own refusal is the actionable half: "does not implement the exact part
            # 'LTST-C190KGKT'" names the one edit that repairs the draft, while the generic
            # sentence alone leaves it to guess. Live run KC-HPD3YF re-emitted the same named
            # exact part through all three rungs against this generic message.
            + (
                f": {failure}"
                if failure
                else (
                    "; use a published satisfying configuration or choose a genuinely "
                    "model-owned family"
                )
            )
        )
    return LoweringContractDiagnostic(
        requirement_id=requirement.id,
        sheet=requirement.sheet,
        family=requirement.family,
        lowerer_id=lowerer_id,
        message=message,
        evidence=evidence,
    )


def lower_requirement(requirement: CircuitRequirement | dict) -> LoweringArtifact | None:
    requirement = CircuitRequirement.model_validate(requirement)
    lowerer_id = _FAMILIES.get(requirement.family)
    if lowerer_id is None:
        return None
    lowerer = _REGISTRY[lowerer_id]
    if requirement.parameters.keys() - lowerer.parameter_keys or lowerer_parameter_diagnostics(
        requirement
    ):
        raise ValueError(
            f"requirement {requirement.id}: known lowerer {lowerer_id} rejects these parameters; "
            f"accepted keys={list(lowerer.parameter_keys)}, choices={dict(lowerer.parameter_choices)}"
        )
    artifact = lowerer.build(requirement)
    if artifact is None:
        raise ValueError(
            f"requirement {requirement.id}: known lowerer {lowerer_id} cannot realize "
            "this identity/port/parameter combination; model-owned fallback is not permitted. "
            f"Received exact_part={requirement.exact_part!r}, ports={requirement.ports}, "
            f"parameters={requirement.parameters}. "
            + " ".join(lowerer.parameter_constraints)
        )
    if artifact is not None and requirement.exact_part:
        # Only a physically identified group can prove an exact-part request.
        # Generic values/footprints must not claim an unverified manufacturer part.
        identity = requirement.exact_part.casefold()
        if not any(
            group.mpn and identity in {group.mpn.casefold(), group.value.casefold()}
            for group in artifact.groups
        ):
            raise ValueError(
                f"requirement {requirement.id}: lowerer {lowerer_id} does not implement "
                f"the exact part {requirement.exact_part!r}"
            )
    return artifact


def _artifact(
    lowerer_id: str,
    requirement: CircuitRequirement,
    groups: tuple[LoweringGroup, ...],
    pins: tuple[LoweringPin, ...],
    *,
    no_connects: tuple[LoweringNoConnect, ...] = (),
    assumptions: tuple[str, ...] = (),
    calculations: tuple[LoweringCalculation, ...] = (),
    local_bypasses: tuple[LoweringLocalBypass, ...] = (),
    named_part_identities: tuple[str, ...] = (),
) -> LoweringArtifact:
    roles = {group.role: group.quantity for group in groups}
    seen: set[tuple[str, int, str]] = set()
    for pin in (*pins, *no_connects):
        if pin.role not in roles or pin.index >= roles[pin.role]:
            raise ValueError(
                f"{lowerer_id}: pin references unknown role/index {pin.role}[{pin.index}]"
            )
        key = (pin.role, pin.index, pin.pin)
        if key in seen:
            raise ValueError(f"{lowerer_id}: duplicate pin ownership {key}")
        seen.add(key)
    for locality in local_bypasses:
        for owner_pin, bypass_pin in (locality.power_pins, locality.ground_pins):
            owner = next((p for p in pins if (p.role, p.index, p.pin) ==
                          (locality.owner_role, locality.owner_index, owner_pin)), None)
            bypass = next((p for p in pins if (p.role, p.index, p.pin) ==
                           (locality.bypass_role, locality.bypass_index, bypass_pin)), None)
            if owner is None or bypass is None or owner.net != bypass.net:
                raise ValueError(f"{lowerer_id}: invalid local bypass pin ownership")
    return LoweringArtifact(
        lowerer_id=lowerer_id,
        requirement_id=requirement.id,
        groups=groups,
        pins=pins,
        no_connects=no_connects,
        assumptions=assumptions,
        calculations=calculations,
        local_bypasses=local_bypasses,
        named_part_identities=named_part_identities,
    )


def _require_ports(
    requirement: CircuitRequirement, names: tuple[str, ...]
) -> dict[str, str] | None:
    if set(requirement.ports) != set(names):
        return None
    if any(not requirement.ports[name] for name in names):
        return None
    return requirement.ports

def _dual_source_battery_power(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Lower the bounded host-5-V/single-cell-Li-ion supply without source tying."""
    from kicraft.design.recipes.dual_source_battery_power import (
        DUAL_SOURCE_BATTERY_POWER,
    )

    required_ports = ("host_input", "battery_input", "output", "gnd")
    ports = _require_ports(requirement, required_ports)
    expected = DUAL_SOURCE_BATTERY_POWER.parameter_defaults
    if (
        ports is None
        or set(requirement.parameters) != set(expected)
        or (
            requirement.exact_part is not None
            and requirement.exact_part.casefold() != "tps2113apwr"
        )
    ):
        return None
    if len(set(ports.values())) != len(ports):
        return None

    include_host_connector = bool(requirement.parameters["include_host_connector"])
    prefix = f"__KICRAFT_DUAL_SOURCE_{requirement.id.upper()}__"
    net_map = {
        **ports,
        **{
            net: f"{prefix}{net.upper()}"
            for net in DUAL_SOURCE_BATTERY_POWER.internal_nets
        },
    }
    groups = tuple(
        LoweringGroup.model_validate(group.model_dump(exclude={"sheet_role"}))
        for group in DUAL_SOURCE_BATTERY_POWER.parts
        if include_host_connector or group.role != "host_connector"
    )
    pins = tuple(
        LoweringPin(
            role=pin.role,
            index=pin.index,
            pin=pin.pin,
            net=net_map[pin.net],
        )
        for pin in DUAL_SOURCE_BATTERY_POWER.pins
        if include_host_connector or pin.role != "host_connector"
    )
    no_connects = tuple(
        LoweringNoConnect(
            role=pin.role,
            index=pin.index,
            pin=pin.pin,
            reason=(
                "Unconnected connector mounting pad"
                if pin.role == "battery_connector"
                else "TPS61088 manufacturer-designated no-connect pin"
            ),
        )
        for pin in DUAL_SOURCE_BATTERY_POWER.no_connects
    )
    return _artifact(
        "dual-source-battery-power@1",
        requirement,
        groups,
        pins,
        no_connects=no_connects,
        assumptions=(
            "Unspecified battery chemistry defaults to one external protected single-cell "
            "Li-ion pack, 3.0 V to 4.2 V. No charger is included.",
            "Host 5 V is a separate physical interface constrained to 4.75 V to 5.25 V; "
            "it is not a surrogate battery source.",
            "The 400 Ohm TPS2113A ILIM resistor has a datasheet-characterized 0.95 A "
            "minimum current limit. The 0.50 A load contract is below that bound; host-path "
            "output remains at least 4.675 V using the 150 mOhm 125 C maximum switch resistance.",
        ),
        calculations=(
            LoweringCalculation(
                output="battery_path_output_range",
                equation=(
                    "VOUT = VREF * (1 + RTOP / RBOTTOM), then VLOAD = VOUT - "
                    "0.50 A * 0.150 ohm"
                ),
                inputs={
                    "vref_min_v": 1.186, "vref_max_v": 1.222,
                    "top_resistance_ohm_nominal": 178000.0,
                    "bottom_resistance_ohm_nominal": 56000.0,
                    "resistor_tolerance_fraction": 0.01,
                    "load_current_a": 0.5, "mux_rds_on_max_ohm": 0.150,
                },
                chosen_value="4.806 V minimum, 5.188 V maximum",
                tolerance="TPS61088 VREF min/max and 1% feedback-divider worst case; TPS2113A 125 C maximum RDS(on)",
            ),
            LoweringCalculation(
                output="boost_compensation_equations",
                equation=(
                    "R5 = 2*pi*VOUT*RSENSE*fC*COUT / ((1-D)*VREF*GEA); "
                    "C5 = RO*COUT/(2*R5); C8 = RESR*COUT/R5"
                ),
                inputs={
                    "vout_v": 5.0, "duty_cycle": 0.4, "load_resistance_ohm": 10.0,
                    "switching_frequency_hz": 600000.0, "selected_crossover_hz": 20000.0,
                    "inductance_h": 2.2e-6, "rhpz_hz": 260435.0,
                    "rhpz_crossover_limit_hz": 52087.0,
                    "sense_resistance_ohm": 0.08, "vref_v_typ": 1.204,
                    "error_amplifier_gm_a_per_v_typ": 190e-6,
                    "design_output_capacitance_uf": 192.0,
                    "initial_output_capacitance_min_uf": 240.0,
                    "maximum_output_esr_ohm": 0.011667,
                },
                chosen_value="R5 69.8 kohm, C5 15 nF C0G, C8 33 pF C0G",
                tolerance="TPS61088 equations 18--20; GEA is datasheet typical, so delivered-graph evidence proves the authored network but cannot prove dynamic phase/gain margin without electrical characterization.",
            ),
            LoweringCalculation(
                output="mux_current_limit_minimum",
                equation="TPS2113A datasheet current-limit characterization at RILIM = 400 ohm",
                inputs={"rilim_ohm": 400.0, "load_current_a": 0.5},
                chosen_value="0.95 A minimum",
                tolerance="The 0.50 A load contract is below the documented minimum; 1.25 A is only the nominal adjustment value, not a guaranteed capability.",
            ),
            LoweringCalculation(
                output="host_path_minimum",
                equation="4.75 V - 0.50 A * 0.150 ohm",
                inputs={"host_min_v": 4.75, "load_current_a": 0.5, "mux_rds_on_max_ohm": 0.150},
                chosen_value="4.675 V",
                tolerance="TPS2113A 125 C maximum RDS(on) from datasheet",
            ),
        ),
        named_part_identities=(
            "tps2113apwr",
            "tps61088rhlr",
            "s2b-ph-sm4-tb(lf)(sn)",
            "tsw-102-07-g-s",
        ),
    )


_E24 = (
    10,
    11,
    12,
    13,
    15,
    16,
    18,
    20,
    22,
    24,
    27,
    30,
    33,
    36,
    39,
    43,
    47,
    51,
    56,
    62,
    68,
    75,
    82,
    91,
)
_E12 = (10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82)


def _standard_value(target: float, series: tuple[int, ...], *, ceiling: bool = False) -> float:
    if target <= 0:
        raise ValueError("standard values require a positive target")
    exponent = math.floor(math.log10(target)) - 1
    candidates = tuple(
        base * (10.0**power) for power in range(exponent - 1, exponent + 3) for base in series
    )
    if ceiling:
        return min(value for value in candidates if value >= target)
    return min(candidates, key=lambda value: (abs(math.log(value / target)), value))


def _resistance(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}R"


def _capacitance(value: float) -> str:
    if value >= 1e-6:
        return f"{value * 1e6:g}uF"
    if value >= 1e-9:
        return f"{value * 1e9:g}nF"
    return f"{value * 1e12:g}pF"


def _passive(role: str, prefix: str, value: str, *, quantity: int = 1) -> LoweringGroup:
    symbol = "Device:C" if prefix == "C" else "Device:R" if prefix == "R" else "Device:LED"
    footprint = {
        "C": "Capacitor_SMD:C_0603_1608Metric",
        "R": "Resistor_SMD:R_0603_1608Metric",
        "D": "LED_SMD:LED_0603_1608Metric",
    }[prefix]
    return LoweringGroup(
        role=role,
        reference_prefix=prefix,
        quantity=quantity,
        value=value,
        symbol=symbol,
        footprint=footprint,
    )


def _reviewed_group_for_identity(
    role: str,
    prefix: str,
    requirement: CircuitRequirement,
    *,
    expected_port_pins: Mapping[str, str],
) -> LoweringGroup | None:
    """The reviewed record's own identity, when the draft named one this build can place.

    A demanded physical class is realized only by a reviewed part, and the architecture stage
    is told to name `exact_part` from the reviewed list. A build that emits its own generic
    symbol would then refuse *following that instruction* -- the draft names a reviewed LED and
    the refusal is "does not implement the exact part" -- so a build that can place the named
    record does: the group carries the record's symbol/footprint pair and its ordering code.

    Adopted only when the record's own pin map agrees with the contacts this build places
    (`expected_port_pins`, port function -> contact number): a record whose pins land elsewhere
    is a different part, and the caller's own refusal is the honest answer. Returns None when no
    usable identity was stated, so a caller falls back to its generic part.
    """
    identity = str(requirement.exact_part or "").strip()
    if not identity:
        return None
    record = reviewed_part(identity)
    if record is None or not record.symbol or not record.footprint:
        return None
    pins = {str(key): str(value) for key, value in dict(record.port_pins or {}).items()}
    if any(pins.get(key) != str(value) for key, value in expected_port_pins.items()):
        return None
    ordering_code = record.identity.upper()
    return LoweringGroup(
        role=role,
        reference_prefix=prefix,
        value=ordering_code,
        symbol=record.symbol,
        footprint=record.footprint,
        mpn=ordering_code,
    )


_CONTACT_KEY_RE = re.compile(r"(?P<prefix>pin|p)(?P<index>[1-9][0-9]*)")

#: Contact key prefixes a cable-side connector family publishes, in match order.
_CONTACT_KEY_PREFIXES = ("pin", "p")

#: The reviewed family the screw-terminal lowerer realizes.
_SCREW_TERMINAL_FAMILY = "screw-terminal"


def _contact_net_map(
    ports: Mapping[str, str], *, prefixes: tuple[str, ...] = _CONTACT_KEY_PREFIXES
) -> dict[int, str] | None:
    """``{contact number: net}`` from explicit ``pinN``/``pN`` keys, or None.

    JSON objects have no physical ordering, so a contact number is only ever what the
    draft states: the key names the position, and a family publishes one key alphabet
    (`pin`) or accepts both.
    """
    if not ports:
        return None
    indexed: dict[int, str] = {}
    prefix: str | None = None
    for key, net in ports.items():
        match = _CONTACT_KEY_RE.fullmatch(str(key))
        if match is None or match["prefix"] not in prefixes:
            return None
        if prefix is None:
            prefix = match["prefix"]
        elif match["prefix"] != prefix:
            return None
        index = int(match["index"])
        if index in indexed or not str(net).strip():
            return None
        indexed[index] = str(net)
    return indexed


def _contact_nets(
    ports: Mapping[str, str], *, prefixes: tuple[str, ...] = _CONTACT_KEY_PREFIXES
) -> tuple[str, ...] | None:
    """Ordered nets of explicit contacts, or None when the numbering is ambiguous.

    A draft states a contact number; a gap, a duplicate, or a second key alphabet leaves
    which physical position carries which net undecidable, so the family refuses instead
    of guessing the part's size from the highest number it happened to see.
    """
    indexed = _contact_net_map(ports, prefixes=prefixes)
    if indexed is None:
        return None
    if set(indexed) != set(range(1, len(indexed) + 1)):
        return None
    return tuple(indexed[index] for index in range(1, len(indexed) + 1))

#: DE-9 names that intentionally denote this nine-contact connector class.  The
#: `dsub-9` spelling is the architecture's existing CAN-interface requirement;
#: the others are precise DE-9/DB9 connector spellings, not a catch-all for serial
#: D-Sub connectors with an unspecified contact convention.
_REVIEWED_DE9_FAMILIES = (
    "dsub-9",
    "dsub9",
    "db9-connector",
    "de-9-connector",
    "de9-connector",
)
_REVIEWED_DE9_IDENTITY = "ds1034-09funsi44"
_DE9_CAN_PORTS = frozenset({"can_h", "can_l", "gnd"})


def _reviewed_de9_connector(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Realize the reviewed DE-9, never infer an arbitrary serial protocol.

    A CAN binding is limited to the published CiA 303-1 map in the reviewed
    record.  Other uses must name every numbered signal contact, including every
    unused one as ``NC``; a partially specified D-Sub has no safe generic
    serial convention to infer.  The record's 9 signal contacts are distinct
    from its two mechanical mounting posts, which have pad number 0 and no
    schematic contact.
    """
    record = reviewed_part(_REVIEWED_DE9_IDENTITY)
    contacts = tuple(str(contact) for contact in record.contacts) if record is not None else ()
    mapping = {str(key): str(value) for key, value in dict(record.port_pins or {}).items()} if record else {}
    if (
        record is None
        or record.family != "de-9-connector"
        or not record.symbol
        or not record.footprint
        or contacts != tuple(str(number) for number in range(1, 10))
        or {mapping.get(port) for port in _DE9_CAN_PORTS} != {"2", "3", "7"}
    ):
        return None
    if requirement.exact_part and requirement.exact_part.casefold() != record.identity:
        return None

    ports = {str(key): str(net) for key, net in requirement.ports.items()}
    by_contact: dict[str, str]
    assumption: str
    if set(ports) == _DE9_CAN_PORTS:
        if (
            any(not ports[port].strip() or ports[port] == "NC" for port in _DE9_CAN_PORTS)
            or len({ports[port] for port in _DE9_CAN_PORTS}) != len(_DE9_CAN_PORTS)
        ):
            return None
        by_contact = {mapping[port]: ports[port] for port in _DE9_CAN_PORTS}
        assumption = (
            f"{record.identity}: CiA 303-1 CAN-over-DE-9 maps CAN_H to contact "
            f"{mapping['can_h']}, CAN_L to contact {mapping['can_l']}, and CAN_GND "
            f"to contact {mapping['gnd']}"
        )
    else:
        numbered = _contact_net_map(ports, prefixes=("pin",))
        if numbered is None or set(numbered) != set(range(1, 10)):
            return None
        by_contact = {str(contact): net for contact, net in numbered.items()}
        assumption = (
            f"{record.identity}: every DE-9 signal contact is explicitly bound by "
            "the requirement; no serial protocol was inferred"
        )

    group = LoweringGroup(
        role="connector",
        reference_prefix="J",
        value=record.identity.upper(),
        mpn=record.identity.upper(),
        symbol=record.symbol,
        footprint=record.footprint,
        datasheet=record.manufacturer_sources[0] if record.manufacturer_sources else None,
    )
    pins = tuple(
        LoweringPin(role="connector", pin=contact, net=by_contact[contact])
        for contact in contacts
        if contact in by_contact and by_contact[contact] != "NC"
    )
    no_connects = tuple(
        LoweringNoConnect(
            role="connector",
            pin=contact,
            reason="unused DE-9 signal contact is intentionally unconnected",
        )
        for contact in contacts
        if contact not in by_contact or by_contact[contact] == "NC"
    )
    return _artifact(
        "reviewed-de9-connector@1",
        requirement,
        (group,),
        pins,
        no_connects=no_connects,
        assumptions=(assumption,),
    )


#: Families whose reviewed record is a two-contact DC power inlet (centre pin plus barrel).
#: Distinct families so a draft that names this connector family reaches this build instead of
#: being refused as "no curated recipe and declares no interface".
_REVIEWED_JACK_FAMILIES = ("barrel-jack", "barrel-jack-connector", "dc-barrel-jack")

#: Draft-side spellings of the jack's two wires. The reviewed record owns the *contact numbers*
#: (`tip`/`sleeve`), not the polarity: which net is the positive one is the draft's statement.
_JACK_POSITIVE_PORTS = ("positive", "vbus", "vin", "input", "input_12v", "power", "v+")
_JACK_RETURN_PORTS = ("gnd", "negative", "sleeve", "ret", "return", "v-")


def _reviewed_connector(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """A reviewed DC power inlet, on the record's own symbol/footprint pair.

    The jack is a two-wire passive connector: the build wires the draft's positive net to the
    record's tip contact and its return to the sleeve. The reviewed record's pin map is the only
    source of those contact numbers, so a record without one is refused rather than guessed. The
    third contact of a switched jack (the normally-closed switch) is emitted as a no-connect when
    the record publishes one: it is closed to the tip only while no plug is inserted, which is not
    a net any design statement asks for.
    """
    record = reviewed_part(str(requirement.exact_part or "").strip())
    if record is None or not record.symbol or not record.footprint:
        return None
    pins_by_function = {str(k): str(v) for k, v in dict(record.port_pins or {}).items()}
    tip = pins_by_function.get("tip")
    sleeve = pins_by_function.get("sleeve")
    if not tip or not sleeve:
        return None
    ports = {str(key).lower(): net for key, net in requirement.ports.items()}
    positive = next((ports[key] for key in _JACK_POSITIVE_PORTS if key in ports), None)
    reference = next((ports[key] for key in _JACK_RETURN_PORTS if key in ports), None)
    if positive is None or reference is None or len(ports) != 2:
        return None
    if not positive or not reference:
        return None
    group = _reviewed_group_for_identity(
        "connector", "J", requirement, expected_port_pins={"tip": tip, "sleeve": sleeve}
    )
    if group is None:
        return None
    pins = (
        LoweringPin(role="connector", pin=tip, net=positive),
        LoweringPin(role="connector", pin=sleeve, net=reference),
    )
    switch = pins_by_function.get("switch")
    no_connects = (
        (
            LoweringNoConnect(
                role="connector",
                pin=switch,
                reason=(
                    "the normally-closed switch contact is closed to the tip only while no plug "
                    "is inserted; the design draws no net from it"
                ),
            ),
        )
        if switch
        else ()
    )
    return _artifact(
        "reviewed-connector@1",
        requirement,
        (group,),
        pins,
        no_connects=no_connects,
        assumptions=(
            f"{record.identity}: tip contact {tip} carries "
            f"{positive!r}, sleeve contact {sleeve} carries {reference!r} "
            "(contact numbers from the reviewed record's pin map, polarity from the draft)",
        ),
    )


#: Families a draft uses for a trim potentiometer. Distinct spellings so a draft that names one
#: reaches the build instead of being refused as a family with no curated recipe.
_REVIEWED_TRIMMER_FAMILIES = ("trim-potentiometer", "trimpot", "trimmer-pot", "trimmer_pot")

#: The two travel ends, as drafts spell them. The corpus uses `end_a`/`end_b`, `end1`/`end2`,
#: `contact1`/`contact3`, and (for a rheostat with one end grounded) `gnd`.
_TRIMMER_END_A_PORTS = ("end_a", "end1", "contact1", "ccw", "a", "gnd", "vss")
_TRIMMER_END_B_PORTS = ("end_b", "end2", "contact3", "cw", "b", "vcc", "vdd")
_TRIMMER_WIPER_PORTS = ("wiper", "wipe", "out", "output", "signal", "sig")

#: The same groups as the build reads them, published: each is one physical contact, spelled the
#: way the corpus spells it. The build takes any spelling; so does the contract check.
_TRIMMER_PORT_ALIASES = (
    ("wiper", _TRIMMER_WIPER_PORTS),
    ("end_a", _TRIMMER_END_A_PORTS),
    ("end_b", _TRIMMER_END_B_PORTS),
)


def _reviewed_potentiometer(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """The reviewed trimmer on its own symbol/footprint pair, from the record's wiper/end map.

    The class is realized only by a reviewed part and the architecture stage is told to name one,
    so the build places the record the draft named instead of emitting a generic device it would
    then have to refuse. A trimmer used as a rheostat binds one end and the wiper, so an unbound
    end is a no-connect rather than a refusal.
    """
    record = reviewed_part(str(requirement.exact_part or "").strip())
    if record is None or not record.symbol or not record.footprint:
        return None
    pins_by_function = {str(k): str(v) for k, v in dict(record.port_pins or {}).items()}
    wiper = pins_by_function.get("wiper")
    ccw = pins_by_function.get("ccw")
    cw = pins_by_function.get("cw")
    if not (wiper and ccw and cw):
        return None
    ports = {str(key).lower(): net for key, net in requirement.ports.items()}
    if not any(ports.get(name) for name in _TRIMMER_WIPER_PORTS):
        return None
    bound_a = next((name for name in _TRIMMER_END_A_PORTS if ports.get(name)), None)
    bound_b = next((name for name in _TRIMMER_END_B_PORTS if ports.get(name)), None)
    if bound_a is None and bound_b is None:
        return None
    group = _reviewed_group_for_identity(
        "trimmer", "RV", requirement, expected_port_pins={"ccw": ccw, "wiper": wiper, "cw": cw}
    )
    if group is None:
        return None
    wiper_net = next(ports[name] for name in _TRIMMER_WIPER_PORTS if ports.get(name))
    pins = [LoweringPin(role="trimmer", pin=wiper, net=wiper_net)]
    no_connects = []
    for name, net, pin, label in (
        (bound_a, ports.get(bound_a or ""), ccw, "counter-clockwise"),
        (bound_b, ports.get(bound_b or ""), cw, "clockwise"),
    ):
        if name and net:
            pins.append(LoweringPin(role="trimmer", pin=pin, net=net))
        else:
            no_connects.append(
                LoweringNoConnect(
                    role="trimmer",
                    pin=pin,
                    reason=f"the {label} end is unused: the trimmer is wired as a rheostat",
                )
            )
    return _artifact(
        "trim-potentiometer@1",
        requirement,
        (group,),
        tuple(pins),
        no_connects=tuple(no_connects),
        assumptions=(
            f"{record.identity}: wiper on contact {wiper}, ends on {ccw}/{cw} "
            "(contacts from the reviewed record's pin map)",
        ),
    )


def _reviewed_terminal_part(identity: str) -> tuple[int, str, str, str] | None:
    """``(contacts, ordering code, symbol, footprint)`` for a reviewed screw terminal.

    The reviewed library keys a part by its canonical identity -- the shipped ordering
    code in lower case, which is what the manifest and the BOM both carry -- while a
    draft states that code in its shipped case. Resolving through the library realizes
    both spellings, and every reviewed contact count, instead of only the two codes a
    private table happened to hold.
    """
    record = reviewed_part(identity)
    if (
        record is None
        or record.family != _SCREW_TERMINAL_FAMILY
        or record.lcsc is None
        or not record.symbol
        or not record.footprint
    ):
        return None
    return len(record.contacts), record.identity.upper(), record.symbol, record.footprint


def _connector(
    requirement: CircuitRequirement,
    *,
    terminal: bool = False,
    assumptions: tuple[str, ...] = (),
) -> LoweringArtifact | None:
    if not requirement.ports or len(requirement.ports) > 40:
        return None
    try:
        rows = int(requirement.parameters.get("rows", 1))
    except (TypeError, ValueError):
        return None
    if terminal:
        semantic_order = (
            ("positive", "negative")
            if len(requirement.ports) == 2
            else ("positive", "common", "negative")
        )
        if set(requirement.ports) == set(semantic_order):
            ordered_nets = tuple(requirement.ports[key] for key in semantic_order)
        else:
            ordered_nets = _contact_nets(requirement.ports)
    else:
        ordered_nets = _contact_nets(requirement.ports, prefixes=("pin",))
    if ordered_nets is None:
        # These ports and parameters ARE the published contract (the caller checks that first), so
        # this is the one refusal left: the declared contacts cannot be numbered. Saying so is what
        # lets the draft repair itself -- live run KC-HPD3YF offered a terminal spelled
        # `p2,positive` and was told only "cannot realize this identity/port/parameter combination".
        raise ValueError(
            "the declared contacts cannot be numbered: give every contact a net and spell them "
            "either as contiguous contacts (`pin1..pinN`) or as the semantic names "
            "(`positive`/`negative`, or `positive`/`common`/`negative`); got "
            + ", ".join(sorted(requirement.ports))
        )
    count = len(ordered_nets)
    if count % rows:
        raise ValueError(f"{count} declared contacts do not divide into rows={rows}")
    per_row = count // rows
    mpn: str | None = None
    # How many physical parts of `block_contacts` contacts realize the declared contacts: one
    # for a single part that exactly covers them, `ceil(count / block_contacts)` when the part
    # the draft pinned carries fewer positions than it declares (an eight-contact demand on a
    # four-position block is two blocks). The last block's surplus positions are reserved
    # unconnected, exactly as a fixed reviewed part with unused contacts is handled elsewhere.
    blocks = 1
    block_contacts = count
    if terminal:
        if not 2 <= count <= 12:
            raise ValueError(
                f"a screw terminal block carries 2 to 12 contacts and this requirement declares "
                f"{count}; "
                + (
                    # No reviewed block is larger than twelve positions, so a bigger demand is a
                    # composition: several reviewed blocks whose positions cover it. Naming the
                    # realizable shape is what lets the draft repair itself (the sixteen-input
                    # held-out brief declares 16 analog inputs plus ground).
                    "split the demand across several terminal requirements, each a reviewed "
                    "block of at most 12 contacts, so that their positions cover it"
                    if count > 12
                    else "spell a two-wire terminal as `positive`/`negative`, or an N-position "
                    "block as `pin1..pinN`"
                )
            )
        lowerer_id = "screw-terminal@1"
        if requirement.exact_part is not None:
            selected = _reviewed_terminal_part(str(requirement.exact_part))
            if selected is None:
                raise ValueError(
                    f"{requirement.exact_part!r} is not a reviewed screw terminal; name a reviewed "
                    "ordering code or omit `exact_part` for the generic part"
                )
            # A reviewed position-array block IS its contacts, so a requirement declaring more
            # contacts than one block carries is a composition of that block, and one declaring
            # fewer leaves the surplus positions reserved. Refusing the width mismatch made the
            # recorded held-out drafts unrepairable (r3 rp2040-dual-adc-usb: an eight-contact
            # `terminals` requirement pinning the four-position WJ126V-5.0-04P, and a two-contact
            # `terminal-ground` pinning the three-position WJ126V-5.0-03P). Nothing is invented:
            # the draft's own pinned reviewed part is instantiated as many times as its positions
            # require, the same way `test-points@1`/`connector_bank` publish N parts in one group.
            block_contacts = selected[0]
            blocks = -(-count // block_contacts)
            _, mpn, symbol, footprint = selected
            value = mpn
        else:
            # A quantity obligation counts positions, not housings. Preserve any
            # contacts without declared nets as explicit no-connects rather than
            # inventing signals. The exact-part branch above never substitutes a
            # different housing for a pinned ordering code.
            positions = max(
                (row.minimum for row in requirement.obligations
                 if row.kind == "quantity"
                 and quantity_subject_binds(row.subject, _SCREW_TERMINAL_FAMILY)),
                default=count,
            )
            positions = max(count, positions)
            default_identity = (
                "WJ126V-5.0-03P-14-00A" if positions % 3 == 0 and positions % 2
                else "WJ126V-5.0-02P-14-00A"
            )
            selected = _reviewed_terminal_part(default_identity)
            if selected is None:
                raise ValueError("the reviewed generic screw terminal is unavailable")
            block_contacts, mpn, symbol, footprint = selected
            blocks = -(-positions // block_contacts)
            if blocks > 500:
                raise ValueError("the screw-terminal bank exceeds 500 housings")
            value = mpn
    else:
        gender = str(requirement.parameters.get("gender", "male")).lower()
        if gender not in ("male", "female"):
            return None
        selected_header = None
        male_parts = {
            2: "tsw-102-07-g-s",
            3: "2.54-1*3p针",
            4: "b-2100s04p-a110",
            6: "68000-406hlf",
        }
        female_parts = {
            2: "pm254v-11-02-h85",
            4: "pm254v-11-04-h85",
            8: "pm254v-11-08-h85",
        }
        header_parts = male_parts if gender == "male" else female_parts
        if requirement.exact_part is not None:
            selected_header = reviewed_part(str(requirement.exact_part))
            from kicraft.design.part_identity import reviewed_header_geometry

            geometry = reviewed_header_geometry(selected_header) if selected_header else None
            if geometry is None or len(selected_header.contacts) != count:
                return None
            record_rows, record_gender = geometry
            if (
                ("rows" in requirement.parameters and rows != record_rows)
                or ("gender" in requirement.parameters and gender != record_gender)
            ):
                return None
            rows, gender = record_rows, record_gender
            per_row = count // rows
        elif rows == 1 and count in header_parts:
            selected_header = reviewed_part(header_parts[count])
        if selected_header is not None:
            mpn = selected_header.identity
            symbol = selected_header.symbol
            footprint = selected_header.footprint
            value = selected_header.identity
        else:
            symbol = f"Connector_Generic:Conn_{rows:02d}x{per_row:02d}"
            if rows == 2 and per_row > 1:
                symbol += "_Odd_Even"
            if gender == "female":
                footprint = f"Connector_PinSocket_2.54mm:PinSocket_{rows}x{per_row:02d}_P2.54mm_Vertical"
                value = f"PinSocket_{rows}x{per_row:02d}"
            else:
                footprint = f"Connector_PinHeader_2.54mm:PinHeader_{rows}x{per_row:02d}_P2.54mm_Vertical"
                value = f"PinHeader_{rows}x{per_row:02d}"
        lowerer_id = "pin-header@1"
    group = LoweringGroup(
        role="connector",
        reference_prefix="J",
        quantity=blocks,
        value=value,
        symbol=symbol,
        footprint=footprint,
        mpn=mpn,
    )
    # Each declared contact binds to one contact of one block; a block's surplus trailing
    # positions are reserved unconnected, exactly as a reviewed fixed-size part's unused
    # contacts already are.
    placements: list[LoweringPin | LoweringNoConnect] = []
    for number in range(1, blocks * block_contacts + 1):
        index, contact = divmod(number - 1, block_contacts)
        net = ordered_nets[number - 1] if number <= count else "NC"
        if net == "NC":
            placements.append(
                LoweringNoConnect(
                    role="connector",
                    index=index,
                    pin=str(contact + 1),
                    reason="reserved contact is intentionally unconnected",
                )
            )
        else:
            placements.append(
                LoweringPin(role="connector", index=index, pin=str(contact + 1), net=net)
            )
    pins = tuple(item for item in placements if isinstance(item, LoweringPin))
    no_connects = tuple(item for item in placements if isinstance(item, LoweringNoConnect))
    if blocks > 1 or block_contacts != count:
        assumptions = (
            *assumptions,
            f"{count} declared contact(s) are realized by {blocks} reviewed "
            f"{value} block(s) of {block_contacts} position(s) each",
        )
    return _artifact(
        lowerer_id, requirement, (group,), pins, no_connects=no_connects, assumptions=assumptions
    )

def _reviewed_jst_xh_connector(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Lower one reviewed JST XH carrier whose contact count exactly matches its interface."""
    nets = _contact_nets(requirement.ports, prefixes=("pin",))
    if nets is None:
        return None
    from kicraft.design.part_identity import reviewed_parts_for_feature

    exact = str(requirement.exact_part or "").strip()
    if exact:
        record = reviewed_part(exact)
        candidates = (record,) if record is not None else ()
    else:
        candidates = reviewed_parts_for_feature("jst-xh-connector")
    matches = [
        record
        for record in candidates
        if record is not None
        and record.family == "jst-xh-connector"
        and len(record.contacts) == len(nets)
        and record.symbol
        and record.footprint
    ]
    if len(matches) != 1:
        return None
    (record,) = matches
    group = LoweringGroup(
        role="connector",
        reference_prefix="J",
        value=record.identity.upper(),
        symbol=record.symbol,
        footprint=record.footprint,
        mpn=record.identity.upper(),
    )
    return _artifact(
        "reviewed-jst-xh-connector@1",
        requirement,
        (group,),
        tuple(
            LoweringPin(role="connector", pin=contact, net=net)
            for contact, net in zip(record.contacts, nets, strict=True)
        ),
    )


def _qwiic_connector(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Publish the reviewed four-contact I2C pinout, never ordinal GPIO pins."""
    record = reviewed_part(requirement.exact_part or "SM04B-SRSS-TB(LF)(SN)")
    if record is None or record.family != "qwiic-i2c-connector":
        return None
    ports = _require_ports(requirement, ("ground", "vcc", "sda", "scl"))
    if ports is None or set(requirement.ports) != set(record.port_pins):
        return None
    if len(set(ports.values())) != 4:
        raise ValueError("Qwiic supply, return, SDA and SCL must use four distinct nets")
    group = LoweringGroup(
        role="connector", reference_prefix="J", value=record.identity.upper(),
        symbol=record.symbol, footprint=record.footprint, mpn=record.identity.upper(),
    )
    return _artifact(
        "qwiic-i2c-connector@1", requirement, (group,),
        tuple(
            LoweringPin(role="connector", pin=pin, net=ports[key])
            for key, pin in record.port_pins.items()
        ),
    )


def _bnc_connector(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("signal", "gnd"))
    if (
        ports is None
        or requirement.parameters
        or (
            requirement.exact_part is not None
            and requirement.exact_part.casefold() != "kh-bnc50-3511"
        )
    ):
        return None
    group = LoweringGroup(
        role="bnc",
        reference_prefix="J",
        value="KH-BNC50-3511",
        mpn="KH-BNC50-3511",
        symbol="bnc-pcb-jack:KH-BNC50-3511",
        footprint="bnc-pcb-jack:ANT-TH_KH-BNC50-3511",
        datasheet="https://www.lcsc.com/datasheet/C2837587.pdf",
    )
    return _artifact(
        "bnc-connector@1",
        requirement,
        (group,),
        (
            LoweringPin(role="bnc", pin="1", net=ports["signal"]),
            *(LoweringPin(role="bnc", pin=pin, net=ports["gnd"]) for pin in ("2", "3", "4")),
        ),
    )


def _audio_jack(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Reviewed no-switch SJ1-3533NG TRS jack; contacts are named S/T/R."""

    ports = _require_ports(requirement, ("sleeve", "tip", "ring"))
    if ports is None or requirement.parameters or requirement.exact_part != "SJ1-3533NG":
        return None
    jack = LoweringGroup(
        role="audio_jack",
        reference_prefix="J",
        value="SJ1-3533NG",
        mpn="SJ1-3533NG",
        symbol="sj1-3533ng:SJ1-3533NG",
        footprint="sj1-3533ng:Jack_3.5mm_CUI_SJ1-3533NG_Horizontal",
        datasheet="https://www.sameskydevices.com/product/resource/sj1-353xng.pdf",
    )
    pins = tuple(
        LoweringPin(role="audio_jack", pin=pin, net=ports[key])
        for key, pin in (("sleeve", "S"), ("tip", "T"), ("ring", "R"))
        if ports[key] != "NC"
    )
    no_connects = tuple(
        LoweringNoConnect(
            role="audio_jack",
            pin=pin,
            reason="mono TS channel leaves the SJ1-3533NG ring contact intentionally open",
        )
        for key, pin in (("sleeve", "S"), ("tip", "T"), ("ring", "R"))
        if ports[key] == "NC"
    )
    return _artifact("audio-jack@1", requirement, (jack,), pins, no_connects=no_connects)


#: The reviewed receptacle this build places, and every documented spelling the library carries
#: for it. Amphenol documents one 12401610E4 receptacle whose order code appears as both
#: ``12401610E4#2A`` and ``12401610E4-2A``; the reviewed inventory keeps a record per spelling on
#: one symbol/footprint pair ("the same part and the same pair, deliberately separate identities
#: so a design may assert either documented order code"), so either spelling names this device.
_REVIEWED_USB_C_RECEPTACLE_CODES = ("12401610E4#2A", "12401610E4-2A")


def _reviewed_usb_c_receptacle_code(identity: object) -> str | None:
    """The reviewed receptacle a stated ``exact_part`` names, in the design's own spelling.

    Only the one device this build places answers: any other stated code -- a different reviewed
    connector, or an unreviewed one -- returns None, and the build then declines the requirement
    exactly as before.
    """
    from kicraft.design.part_identity import reviewed_part

    stated = str(identity or "").strip()
    if not stated:
        return None
    record = reviewed_part(stated)
    published = reviewed_part(_REVIEWED_USB_C_RECEPTACLE_CODES[0])
    if (
        record is None
        or published is None
        or record.family != published.family
        or (record.symbol, record.footprint) != (published.symbol, published.footprint)
    ):
        return None
    return stated.upper()


def _usb_c_breakout(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Expose declared contacts of a verified receptacle, without terminations."""
    ports = requirement.ports
    superspeed = {
        "tx1p": "A2",
        "tx1n": "A3",
        "tx2p": "B2",
        "tx2n": "B3",
        "rx1p": "B11",
        "rx1n": "B10",
        "rx2p": "A11",
        "rx2n": "A10",
    }
    supported = {"vbus", "gnd", "shield", "usb_dp", "usb_dm", "cc1", "cc2", "sbu1", "sbu2"}
    supported.update(superspeed)
    if (
        not {"vbus", "gnd"} <= ports.keys()
        or ports.keys() - supported
        or any(not net.strip() for net in ports.values())
        or ports["vbus"] == ports["gnd"]
        or requirement.parameters
    ):
        # Missing bindings are not rails; unknown hardware constraints stay unresolved.
        return None
    superspeed_mpn = "12401610E4#2A"
    stated_receptacle = _reviewed_usb_c_receptacle_code(requirement.exact_part)
    use_superspeed = bool(ports.keys() & superspeed.keys()) or stated_receptacle is not None
    if use_superspeed:
        # Amphenol drawing C12401610 rev C contact table; the installed land
        # pattern uses A1..A12/B1..B12 and four shell pads sharing S1. The design's own
        # spelling of the order code is carried, so the group states the identity asserted.
        group_mpn = stated_receptacle or superspeed_mpn
        group = LoweringGroup(
            role="connector",
            reference_prefix="J",
            value=group_mpn,
            mpn=group_mpn,
            symbol="Connector:USB_C_Receptacle",
            footprint="Connector_USB:USB_C_Receptacle_Amphenol_12401610E4-2A",
            datasheet="https://cdn.amphenol-cs.com/media/wysiwyg/files/drawing/c12401610_c.pdf",
        )
        pin_ports = [
            *((number, "gnd") for number in ("A1", "A12", "B1", "B12")),
            *((number, "vbus") for number in ("A4", "A9", "B4", "B9")),
            ("A5", "cc1"),
            ("B5", "cc2"),
            ("S1", "shield"),
            *((number, port) for port, number in superspeed.items()),
        ]
    else:
        from kicraft.design.recipes import get_recipe

        definition = get_recipe("usb-c-5v-sink@1")
        connector = next(group for group in definition.parts if group.role == "connector")
        group = LoweringGroup(
            role="connector",
            reference_prefix=connector.reference_prefix,
            value=connector.value,
            symbol=connector.symbol,
            footprint=connector.footprint,
            mpn=connector.mpn,
        )
        pin_ports = [(pin.pin, pin.net) for pin in definition.pins if pin.role == "connector"]
    if requirement.exact_part and requirement.exact_part.casefold() != (group.mpn or "").casefold():
        return None
    pin_ports.extend(
        (number, port)
        for port, numbers in (
            ("usb_dp", ("A6", "B6")),
            ("usb_dm", ("A7", "B7")),
            ("sbu1", ("A8",)),
            ("sbu2", ("B8",)),
        )
        for number in numbers
    )
    pins = tuple(
        LoweringPin(role="connector", pin=number, net=ports[port])
        for number, port in pin_ports
        if port in ports
    )
    no_connects = tuple(
        LoweringNoConnect(
            role="connector",
            pin=number,
            reason=f"{port} is not exposed by this receptacle contract",
        )
        for number, port in pin_ports
        if port not in ports
    )
    return _artifact(
        "usb-c-breakout@1",
        requirement,
        (group,),
        pins,
        no_connects=no_connects,
        assumptions=(
            "Unbound USB-C shell pads are intentionally unconnected; no CC pull-downs are fitted.",
        ),
    )


def _stated_contact_count(obligations: list[RequirementObligation]) -> int:
    """The contact count the requirement's own obligations state, or 0.

    A brief's "6-pin 0.1 inch header" reaches the parts step as a quantitative row owned by the
    header requirement ("header pins = 6"). The connector's physical size otherwise comes from the
    contacts the circuit happens to use -- two of six here, so the parts step emitted a 2-way
    header for a brief that demands six, and no gate anywhere said a word (live 2026-09-25).
    Equality and minimum rows about the connector's own contacts set its minimum size.
    """
    for row in obligations:
        if row.kind != "quantitative" or row.relation not in {"equal", "minimum"} or row.value is None:
            continue
        if str(row.unit or "").strip().casefold() not in {
            "count", "pin", "pins", "contact", "contacts", "position", "positions",
        }:
            continue
        quantity = str(row.quantity or "").casefold()
        if not any(term in quantity for term in ("pin", "contact", "position")):
            continue
        try:
            count = int(row.value)
        except (TypeError, ValueError):
            continue
        if 2 <= count <= 64:
            return count
    return 0


def _numbered_connector_ports(
    requirement: CircuitRequirement, *, fill_gaps: bool = False, minimum: int = 0
) -> dict[str, str] | None:
    """``pin1..pinN`` for a generic header, or None when the numbering is ambiguous.

    JSON objects have no physical ordering, so only explicit ``pinN`` bindings establish
    contact numbers. The header's physical size is its **highest declared contact**: a
    draft that declares only the positions its circuit uses (the corpus's "8-pin header"
    arriving as ``pin1,pin2,pin3,pin4,pin8``) describes the same part as one that spelled
    out every position, and the undeclared positions are that part's no-connects.
    ``fill_gaps`` is what accepts such a draft -- it emits the missing positions as
    ``NC``, which `_connector` turns into no-connects. Without it the numbering must be
    exactly ``1..N`` and any gap is undecidable, so the family refuses. ``minimum`` is the
    size the requirement's own obligations state, which the part may be larger than.
    """
    indexed = _contact_net_map(requirement.ports, prefixes=("pin",))
    if indexed is None:
        return None
    highest = max({*indexed, int(minimum or 0)} or {0})
    if not fill_gaps and set(indexed) != set(range(1, highest + 1)):
        return None
    return {f"pin{index}": indexed.get(index, "NC") for index in range(1, highest + 1)}


def _pin_header(requirement: CircuitRequirement) -> LoweringArtifact | None:
    stated = _stated_contact_count(requirement.obligations)
    ports = _numbered_connector_ports(requirement, fill_gaps=True, minimum=stated)
    if ports is None:
        return None
    filled = tuple(
        index for index in range(1, len(ports) + 1) if f"pin{index}" not in requirement.ports
    )
    declared = max(
        (
            int(key[3:])
            for key in requirement.ports
            if key.startswith("pin") and key[3:].isdigit()
        ),
        default=0,
    )
    notes: list[str] = []
    if stated > declared:
        notes.append(
            f"the header is sized to the {stated} contacts its own requirement states, "
            f"which the circuit uses {declared} of"
        )
    if filled:
        notes.append(
            f"the header's physical size is its highest declared contact ({len(ports)}); "
            f"undeclared contact(s) {', '.join(str(index) for index in filled)} are "
            "emitted as no-connects"
        )
    return _connector(
        requirement.model_copy(update={"ports": ports}),
        assumptions=tuple(notes),
    )


def _screw_terminal(requirement: CircuitRequirement) -> LoweringArtifact | None:
    return _connector(requirement, terminal=True)


def _test_points(requirement: CircuitRequirement) -> LoweringArtifact | None:
    if not requirement.ports or len(requirement.ports) > 64:
        return None
    group = LoweringGroup(
        role="testpoint",
        reference_prefix="TP",
        quantity=len(requirement.ports),
        value="TEST",
        symbol="Connector:TestPoint",
        footprint="TestPoint:TestPoint_Pad_D1.0mm",
    )
    pins = tuple(
        LoweringPin(role="testpoint", index=index, pin="1", net=net)
        for index, net in enumerate(requirement.ports.values())
    )
    return _artifact("test-points@1", requirement, (group,), pins)


def _connector_bank(requirement: CircuitRequirement) -> LoweringArtifact | None:
    try:
        channels = int(requirement.parameters.get("channels", 0))
    except (TypeError, ValueError):
        return None
    needed = ("vdd", "gnd", *(f"signal{i}" for i in range(channels)))
    ports = _require_ports(requirement, needed)
    if ports is None or not 1 <= channels <= 32:
        return None
    group = LoweringGroup(
        role="connector",
        reference_prefix="J",
        quantity=channels,
        value="Servo_1x03",
        symbol="Connector_Generic:Conn_01x03",
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
    )
    pins = tuple(
        pin
        for index in range(channels)
        for pin in (
            LoweringPin(role="connector", index=index, pin="1", net=ports["gnd"]),
            LoweringPin(role="connector", index=index, pin="2", net=ports["vdd"]),
            LoweringPin(role="connector", index=index, pin="3", net=ports[f"signal{index}"]),
        )
    )
    return _artifact("connector-bank@1", requirement, (group,), pins)


def _fpc_connector(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Lower the reviewed 24-contact FPC connector, never a hidden header pair."""

    ports = _numbered_connector_ports(requirement)
    if (
        ports is None
        or len(ports) != 24
        or requirement.parameters != {"pitch_mm": 0.5}
        or (
            requirement.exact_part is not None
            and requirement.exact_part.casefold() != "kh-fg0.5-h2.0-24pin"
        )
    ):
        return None
    fpc = LoweringGroup(
        role="fpc",
        reference_prefix="J",
        value="KH-FG0.5-H2.0-24PIN",
        mpn="KH-FG0.5-H2.0-24PIN",
        symbol="fpc-24-0-5-kinghelm:KH-FG0.5-H2.0-24PIN",
        footprint="fpc-24-0-5-kinghelm:FPC-SMD_KH-FG0.5-H2.0-24PIN",
    )
    pins = tuple(
        LoweringPin(role="fpc", pin=str(index), net=net)
        for index, net in enumerate(ports.values(), 1)
    )
    return _artifact("fpc-connector@1", requirement, (fpc,), pins)




def _r2r(requirement: CircuitRequirement) -> LoweringArtifact | None:
    try:
        if (
            len(requirement.parameters.keys() & {"r_value", "r_series", "r"}) > 1
            or len(requirement.parameters.keys() & {"two_r_value", "r_shunt"}) > 1
        ):
            return None
        bits = int(requirement.parameters.get("bits", 0))
    except (TypeError, ValueError):
        return None
    r_raw = requirement.parameters.get(
        "r_value",
        requirement.parameters.get("r_series", requirement.parameters.get("r", "")),
    )
    two_r_raw = requirement.parameters.get(
        "two_r_value",
        requirement.parameters.get("r_shunt", ""),
    )
    if not two_r_raw and isinstance(r_raw, (int, float)):
        two_r_raw = float(r_raw) * 2
    r_value = _resistance(float(r_raw)) if isinstance(r_raw, (int, float)) else str(r_raw)
    two_r_value = (
        _resistance(float(two_r_raw)) if isinstance(two_r_raw, (int, float)) else str(two_r_raw)
    )
    if not 2 <= bits <= 32 or not r_value or not two_r_value:
        return None
    series = LoweringGroup(
        role="series",
        reference_prefix="R",
        quantity=bits - 1,
        value=r_value,
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
    )
    branch = LoweringGroup(
        role="branch",
        reference_prefix="R",
        quantity=bits + 1,
        value=two_r_value,
        symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric",
    )
    names = (*(f"bit{i}" for i in range(bits)), "output", "gnd")
    ports = _require_ports(requirement, names)
    if ports is None:
        # The ladder's published contract names its two spellings of the output contact: the
        # expanded form's `output` and the compact form's `analog_output`. A draft that binds the
        # expanded bit contacts and spells the output `analog_output` states exactly that part, so
        # the published synonym is taken rather than refused (live r2r-dac r13: `bit0..bit7` plus
        # `analog_output`). The compact form below is untouched: it is only reached when the
        # expanded bit contacts are absent.
        if "analog_output" in requirement.ports and "digital_inputs" not in requirement.ports:
            spelt = {
                key: value for key, value in requirement.ports.items() if key != "analog_output"
            }
            spelt["output"] = requirement.ports["analog_output"]
            ports = _require_ports(
                requirement.model_copy(update={"ports": spelt}), names
            )
    if ports is None:
        compact_ports = {name.lower(): value for name, value in requirement.ports.items()}
        if set(compact_ports) != {"digital_inputs", "analog_output"}:
            return None
        bus = str(compact_ports["digital_inputs"]).strip()
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*?)(\d+)\s*-\s*(?:\1)?(\d+)",
            bus,
        )
        if match is not None:
            prefix, first_text, last_text = match.groups()
            first, last = int(first_text), int(last_text)
            step = 1 if last >= first else -1
            input_nets = [f"{prefix}{index}" for index in range(first, last + step, step)]
        else:
            input_nets = [item for item in re.split(r"[\s,;]+", bus) if item]
        if len(input_nets) != bits:
            return None
        ports = {
            **{f"bit{index}": net for index, net in enumerate(input_nets)},
            "output": str(compact_ports["analog_output"]),
            "gnd": "GND",
        }
    nodes = [ports["output"], *(f"__lowerer__{requirement.id}__r2r_n{i}" for i in range(1, bits))]
    pins: list[LoweringPin] = []
    for index in range(bits - 1):
        pins.extend(
            (
                LoweringPin(role="series", index=index, pin="1", net=nodes[index]),
                LoweringPin(role="series", index=index, pin="2", net=nodes[index + 1]),
            )
        )
    for index in range(bits):
        pins.extend(
            (
                LoweringPin(
                    role="branch", index=index, pin="1", net=ports[f"bit{bits - index - 1}"]
                ),
                LoweringPin(role="branch", index=index, pin="2", net=nodes[index]),
            )
        )
    pins.extend(
        (
            LoweringPin(role="branch", index=bits, pin="1", net=nodes[-1]),
            LoweringPin(role="branch", index=bits, pin="2", net=ports["gnd"]),
        )
    )
    return _artifact("r2r-ladder@1", requirement, (series, branch), tuple(pins))


_LED_PARAMETER_KEYS = ("rail_voltage", "led_vf", "target_current_ma")

#: The family spellings the reviewed indicator-LED records themselves carry.
#:
#: A draft names the LED by its reviewed part class instead of by the topology ("warm-white-led",
#: "led-0603" -- read from `kicraft/parts_library/e6c0805wway1uda-1-1t-m`), and a family key with no
#: registered lowerer silently drops the requirement out of compiler-owned realization: the BOM
#: unit is then handed to the model, which must re-emit a reviewed identity it cannot see.
#: Live cohort 2026-09-30, brief 2: the architecture committed `power_led` with family
#: `warm-white-led` and the reviewed exact part, the BOM unit `bom-s002` came back model-owned
#: (`planned_resolution_source="llm", lowerer_ids=[]`), and the draft failed the ownership gate
#: identically in two passes ("missing-requirement-implementation=['power_led'];
#: declared-interface-unrealized=['power_led: declared interface needs one identified hardware
#: owner']") until the stage exhausted its repair attempts. Registering the class spellings routes
#: the requirement to this build, which places the reviewed part the draft named plus the series
#: resistor its own rail/current parameters compute.
_REVIEWED_INDICATOR_LED_FAMILIES = ("warm-white-led", "led-0603")


def _led_resistor_parameters(requirement: CircuitRequirement) -> tuple[float, float, float]:
    values = []
    for key in _LED_PARAMETER_KEYS:
        raw = requirement.parameters.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(key, "must be a finite number") from None
        if isinstance(raw, bool) or not math.isfinite(value):
            raise ValueError(key, "must be a finite number")
        values.append(value)
    rail, vf, current_ma = values
    if not rail > vf > 0:
        raise ValueError(
            "rail_voltage",
            f"must exceed positive led_vf={vf:g} V; a passive current limiter needs voltage headroom",
        )
    if not 1 <= current_ma <= 30:
        raise ValueError("target_current_ma", "must be between 1 and 30 mA")
    return rail, vf, current_ma


def _led_resistor(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("drive", "gnd"))
    try:
        rail, vf, current_ma = _led_resistor_parameters(requirement)
    except ValueError:
        return None
    if ports is None:
        return None
    ideal = (rail - vf) / (current_ma / 1000.0)
    value = _standard_value(ideal, _E24, ceiling=True)
    resistor = _passive("resistor", "R", _resistance(value))
    # The build places the LED's anode on the drive node and its cathode on the return; a
    # reviewed indicator record carries exactly that map (`anode`/`cathode`), so honour the
    # identity the draft named instead of refusing it for naming one.
    led = _reviewed_group_for_identity(
        "led", "D", requirement, expected_port_pins={"anode": "2", "cathode": "1"}
    ) or _passive("led", "D", str(requirement.parameters.get("color", "LED")))
    node = f"__lowerer__{requirement.id}__led_anode"
    pins = (
        LoweringPin(role="resistor", pin="1", net=ports["drive"]),
        LoweringPin(role="resistor", pin="2", net=node),
        LoweringPin(role="led", pin="2", net=node),
        LoweringPin(role="led", pin="1", net=ports["gnd"]),
    )
    calculation = LoweringCalculation(
        output="series_resistance",
        equation="R=(Vrail-Vf)/I",
        inputs={"rail_voltage": rail, "led_vf": vf, "target_current_ma": current_ma},
        chosen_value=_resistance(value),
        tolerance="E24, chosen at or above ideal",
    )
    return _artifact(
        "led-current-resistor@1", requirement, (resistor, led), pins, calculations=(calculation,)
    )


def _adjustable_rc_filter(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Reviewed 3296W rheostat plus capacitor low-pass path."""

    ports = _require_ports(requirement, ("input", "output", "gnd"))
    try:
        capacitance = float(requirement.parameters["capacitance_f"])
        capacitor_exact_part = str(requirement.parameters["capacitor_exact_part"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        ports is None
        or capacitor_exact_part.casefold() != "c0805c103j5gactu"
        or not math.isclose(capacitance, 10e-9, rel_tol=0.0, abs_tol=1e-15)
        or (
            requirement.exact_part is not None
            and requirement.exact_part.casefold() != "3296w-1-103lf"
        )
    ):
        return None
    chosen_capacitance = 10e-9
    minimum_cutoff = 1.0 / (2.0 * math.pi * 10_000.0 * chosen_capacitance)
    pot = LoweringGroup(
        role="trim_pot",
        reference_prefix="RV",
        value="3296W-1-103LF",
        mpn="3296W-1-103LF",
        symbol="trim-pot-3296w-10k:3296W-1-103LF",
        footprint="trim-pot-3296w-10k:RES-ADJ-TH_3296W",
        datasheet="https://www.lcsc.com/datasheet/C34846.pdf",
    )
    capacitor = LoweringGroup(
        role="capacitor",
        reference_prefix="C",
        value="10nF 50V C0G/NP0 ±5%",
        mpn="C0805C103J5GACTU",
        symbol="c0805c103j5gactu:C0805C103J5GACTU",
        footprint="c0805c103j5gactu:C_0805_2012Metric",
        datasheet="https://www.lcsc.com/datasheet/lcsc/2304131050_KEMET-C0805C103J5GACTU_C2167597.pdf",
    )
    return _artifact(
        "adjustable-rc-lowpass@1",
        requirement,
        (pot, capacitor),
        (
            LoweringPin(role="trim_pot", pin="1", net=ports["input"]),
            LoweringPin(role="trim_pot", pin="2", net=ports["output"]),
            LoweringPin(role="trim_pot", pin="3", net=ports["output"]),
            LoweringPin(role="capacitor", pin="1", net=ports["output"]),
            LoweringPin(role="capacitor", pin="2", net=ports["gnd"]),
        ),
        calculations=(
            LoweringCalculation(
                output="minimum_cutoff_frequency",
                equation="fc_min=1/(2*pi*Rmax*C)",
                inputs={"rmax_ohm": 10_000.0, "capacitance_f": chosen_capacitance},
                chosen_value=f"{minimum_cutoff:g}Hz",
                tolerance="E12 capacitor; 3296W-1-103LF adjusted from 0 to 10 kOhm",
            ),
        ),
    )



def _voltage_divider(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("input", "output", "gnd"))
    try:
        vin = float(requirement.parameters["input_voltage"])
        vout = float(requirement.parameters["target_voltage"])
        bottom = float(requirement.parameters["bottom_resistance_ohm"])
        max_error = float(requirement.parameters.get("max_error_percent", 5.0))
    except (KeyError, TypeError, ValueError):
        return None
    if ports is None or not (vin > vout > 0 and 100 <= bottom <= 1_000_000):
        return None
    ideal_top = bottom * (vin / vout - 1.0)
    if not 100 <= ideal_top <= 10_000_000:
        return None
    chosen_bottom = _standard_value(bottom, _E24)
    chosen_top = _standard_value(ideal_top, _E24)
    achieved = vin * chosen_bottom / (chosen_top + chosen_bottom)
    error_percent = abs(achieved - vout) / vout * 100.0
    if not 0 <= max_error or error_percent > max_error:
        return None
    groups = (
        _passive("top", "R", _resistance(chosen_top)),
        _passive("bottom", "R", _resistance(chosen_bottom)),
    )
    pins = (
        LoweringPin(role="top", pin="1", net=ports["input"]),
        LoweringPin(role="top", pin="2", net=ports["output"]),
        LoweringPin(role="bottom", pin="1", net=ports["output"]),
        LoweringPin(role="bottom", pin="2", net=ports["gnd"]),
    )
    calc = LoweringCalculation(
        output="divider_voltage",
        equation="Vout=Vin*Rbottom/(Rtop+Rbottom)",
        inputs={
            "input_voltage": vin,
            "target_voltage": vout,
            "top_resistance_ohm": chosen_top,
            "bottom_resistance_ohm": chosen_bottom,
        },
        chosen_value=f"{achieved:g}V",
        tolerance=f"{error_percent:.3g}% error using E24 values",
    )
    return _artifact("voltage-divider@1", requirement, groups, pins, calculations=(calc,))


def _rc_filter(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("input", "output", "gnd"))
    try:
        cutoff = float(requirement.parameters["cutoff_hz"])
        resistance = float(requirement.parameters["resistance_ohm"])
        max_error = float(requirement.parameters.get("max_error_percent", 10.0))
    except (KeyError, TypeError, ValueError):
        return None
    if ports is None or not (cutoff > 0 and 100 <= resistance <= 1_000_000):
        return None
    ideal_capacitance = 1.0 / (2.0 * math.pi * resistance * cutoff)
    if not 10e-12 <= ideal_capacitance <= 100e-6:
        return None
    chosen_resistance = _standard_value(resistance, _E24)
    chosen_capacitance = _standard_value(ideal_capacitance, _E12)
    achieved_cutoff = 1.0 / (2.0 * math.pi * chosen_resistance * chosen_capacitance)
    error_percent = abs(achieved_cutoff - cutoff) / cutoff * 100.0
    if not 0 <= max_error or error_percent > max_error:
        return None
    r = _passive("resistor", "R", _resistance(chosen_resistance))
    c = _passive("capacitor", "C", _capacitance(chosen_capacitance))
    pins = (
        LoweringPin(role="resistor", pin="1", net=ports["input"]),
        LoweringPin(role="resistor", pin="2", net=ports["output"]),
        LoweringPin(role="capacitor", pin="1", net=ports["output"]),
        LoweringPin(role="capacitor", pin="2", net=ports["gnd"]),
    )
    calc = LoweringCalculation(
        output="cutoff_frequency",
        equation="fc=1/(2*pi*R*C)",
        inputs={
            "target_cutoff_hz": cutoff,
            "resistance_ohm": chosen_resistance,
            "capacitance_f": chosen_capacitance,
        },
        chosen_value=f"{achieved_cutoff:g}Hz",
        tolerance=f"{error_percent:.3g}% error using E24/E12 values",
    )
    return _artifact("rc-lowpass@1", requirement, (r, c), pins, calculations=(calc,))


def _i2c_pullups(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("vdd", "sda", "scl"))
    try:
        speed = int(requirement.parameters.get("speed_hz", 0))
        capacitance = float(requirement.parameters.get("bus_capacitance_pf", 0))
        voltage = float(requirement.parameters.get("voltage", 0))
    except (TypeError, ValueError):
        return None
    if (
        ports is None
        or voltage not in (1.8, 3.3, 5.0)
        or speed not in (100000, 400000, 1000000)
        or not 10 <= capacitance <= 400
    ):
        return None
    max_r = (1000e-9 if speed == 100000 else 300e-9 if speed == 400000 else 120e-9) / (
        0.8473 * capacitance * 1e-12
    )
    chosen = next((value for value in (10000, 4700, 2200) if value <= max_r), None)
    if chosen is None:
        return None
    group = _passive("pullup", "R", _resistance(chosen), quantity=2)
    pins = tuple(
        pin
        for index, signal in enumerate(("sda", "scl"))
        for pin in (
            LoweringPin(role="pullup", index=index, pin="1", net=ports["vdd"]),
            LoweringPin(role="pullup", index=index, pin="2", net=ports[signal]),
        )
    )
    calc = LoweringCalculation(
        output="pullup_resistance",
        equation="Rp(max)=tr/(0.8473*Cbus)",
        inputs={"speed_hz": speed, "bus_capacitance_pf": capacitance, "voltage": voltage},
        chosen_value=group.value,
        tolerance="standard conservative value below rise-time maximum",
    )
    return _artifact("i2c-pullups@1", requirement, (group,), pins, calculations=(calc,))


def _pullup(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("vdd", "signal"))
    value = str(requirement.parameters.get("resistance", ""))
    if ports is None or not value:
        return None
    group = _passive("pullup", "R", value)
    pins = (
        LoweringPin(role="pullup", pin="1", net=ports["vdd"]),
        LoweringPin(role="pullup", pin="2", net=ports["signal"]),
    )
    return _artifact("open-drain-pullup@1", requirement, (group,), pins)


def _switch_input(requirement: CircuitRequirement) -> LoweringArtifact | None:
    from kicraft.design.part_identity import canonical_physical_features, class_key

    mechanisms = {
        feature
        for obligation in requirement.obligations
        if obligation.kind == "adjustability"
        for feature in canonical_physical_features(class_key(obligation.mechanism))
    }
    unsupported = mechanisms - {"momentary-button", "jumper"}
    if unsupported or {"momentary-button", "jumper"} <= mechanisms:
        mechanism = sorted(unsupported or mechanisms)
        raise ValueError(
            "switch-input@1 cannot combine a momentary pushbutton with "
            f"adjustment mechanism(s) {mechanism!r}; retain the requested mechanism and select "
            "hardware that actually implements it"
        )
    ports = _require_ports(requirement, ("signal", "gnd", "vdd"))
    policy = str(requirement.parameters.get("pull_policy", "external"))
    active_level = str(requirement.parameters.get("active_level", "low"))
    if (
        ports is None
        or len(set(ports.values())) != 3
        or any(not net.strip() for net in ports.values())
        or active_level not in {"low", "high"}
    ):
        return None
    if "jumper" in mechanisms:
        if policy != "external":
            raise ValueError(
                "a removable enable jumper requires an external pull so its removed state is "
                "deterministic"
            )
        raw_supply = re.sub(r"[^A-Z0-9]", "", ports["vdd"].upper())
        explicitly_high = (
            re.fullmatch(r"(?:VIN|VBAT|RAW)(?:1[89]|2[0-9]|[3-9][0-9])V?", raw_supply)
            or re.fullmatch(r"(?:1[89]|2[0-9]|[3-9][0-9])V", raw_supply)
        )
        voltage = requirement.parameters.get("voltage")
        if explicitly_high or (
            raw_supply in {"VIN", "VBAT", "RAW"}
            and (type(voltage) not in (int, float) or not 0 < voltage <= 5.5)
        ):
            raise ValueError(
                "a removable enable jumper must use a verified logic/control rail, not an "
                f"unbounded raw supply ({ports['vdd']!r})"
            )
        identity = "TSW-102-07-G-S"
        if requirement.exact_part and requirement.exact_part.casefold() != identity.casefold():
            return None
        record = reviewed_part(identity)
        if record is None or not record.symbol or not record.footprint:
            return None
        active_net = ports["gnd" if active_level == "low" else "vdd"]
        idle_net = ports["vdd" if active_level == "low" else "gnd"]
        pull_role = "pullup" if active_level == "low" else "pulldown"
        jumper = LoweringGroup(
            role="jumper",
            reference_prefix="J",
            value=identity,
            symbol=record.symbol,
            footprint=record.footprint,
            mpn=identity,
            accessories=(BOMAccessory(
                mpn="SNT-100-BL-G", lcsc="C6680974", quantity=1,
                datasheet="https://www.lcsc.com/datasheet/C6680974.pdf",
            ),),
            sourcing_note=(
                "Fit Samtec SNT-100-BL-G removable shunt for enabled state; leave shunt off "
                "for disabled state"
            ),
        )
        pull = _passive(pull_role, "R", str(requirement.parameters.get("resistance", "10k")))
        return _artifact(
            "switch-input@1",
            requirement,
            (jumper, pull),
            (
                LoweringPin(role="jumper", pin="1", net=ports["signal"]),
                LoweringPin(role="jumper", pin="2", net=active_net),
                LoweringPin(role=pull_role, pin="1", net=idle_net),
                LoweringPin(role=pull_role, pin="2", net=ports["signal"]),
            ),
            assumptions=(
                f"SNT-100-BL-G fitted on {identity}: {ports['signal']}={active_net}; removed: "
                f"{ports['signal']}={idle_net} through the {pull.value} {pull_role}",
            ),
        )
    if policy == "internal" and "resistance" in requirement.parameters:
        return None
    if requirement.exact_part:
        record = reviewed_part(requirement.exact_part)
        if (
            record is None
            or "momentary-button" not in record.physical_features
            or set(record.contacts) != {"1", "2"}
        ):
            return None
    else:
        record = reviewed_part("KMR232NGULCLFS")
    if record is None:
        return None
    # Contact ratings constrain declared voltage, not a voltage guessed from a net label.
    voltage = requirement.parameters.get("voltage")
    if voltage is not None:
        try:
            voltage = float(voltage)
        except (TypeError, ValueError):
            return None
        limit = record.operating_limits.get("voltage_v")
        if limit is None or not 0 < voltage <= limit:
            return None
    switch = _reviewed_group_for_identity(
        "switch", "SW", requirement, expected_port_pins={}
    )
    if requirement.exact_part and switch is None:
        return None
    groups = [
        switch or LoweringGroup(
            role="switch",
            reference_prefix="SW",
            value="BUTTON",
            symbol="kmr232ngulclfs:KMR232NGULCLFS",
            footprint="kmr232ngulclfs:KMR232NGULCLFS",
            mpn="KMR232NGULCLFS",
            datasheet="https://datasheet.lcsc.com/datasheet/pdf/b5da7fd4e81211d634f1332fb1486a1f.pdf",
        )
    ]
    pins = [
        LoweringPin(role="switch", pin="1", net=ports["signal"]),
        LoweringPin(role="switch", pin="2", net=ports["gnd" if active_level == "low" else "vdd"]),
    ]
    if policy == "external":
        pull_role = "pullup" if active_level == "low" else "pulldown"
        groups.append(
            _passive(pull_role, "R", str(requirement.parameters.get("resistance", "10k")))
        )
        pins.extend(
            (
                LoweringPin(
                    role=pull_role, pin="1", net=ports["vdd" if active_level == "low" else "gnd"]
                ),
                LoweringPin(role=pull_role, pin="2", net=ports["signal"]),
            )
        )
    return _artifact("switch-input@1", requirement, tuple(groups), tuple(pins))


def _voltage_selector_switch(
    requirement: CircuitRequirement,
) -> LoweringArtifact | None:
    ports = requirement.ports
    contacts = (("throw_1", "1"), ("common", "2"), ("throw_2", "3"), ("throw_3", "4"))
    if (
        requirement.parameters.get("positions") != 3
        or not {"common", "gnd"} <= ports.keys()
        or set(ports) - {"common", "gnd", "throw_1", "throw_2", "throw_3"}
        or not any(name in ports for name in ("throw_1", "throw_2", "throw_3"))
        or any(not net.strip() or net == "NC" for net in ports.values())
    ):
        return None
    group = LoweringGroup(
        role="voltage_selector_switch",
        reference_prefix="SW",
        value="SS13D07VG4",
        symbol="ss13d07vg4:SS13D07VG4",
        footprint="ss13d07vg4:SW-TH_SS13D07VG4",
        mpn="SS13D07VG4",
        datasheet="https://datasheet.lcsc.com/datasheet/pdf/39fcef34462917ff9922c33e708581d0.pdf?productCode=C2681578",
    )
    pins = tuple(
        LoweringPin(role=group.role, pin=pin, net=ports[name])
        for name, pin in contacts if name in ports
    ) + tuple(
        LoweringPin(role=group.role, pin=pin, net=ports["gnd"])
        for pin in ("5", "6")
    )
    return _artifact(
        "voltage-selector-switch@1",
        requirement,
        (group,),
        pins,
        no_connects=tuple(
            LoweringNoConnect(
                role=group.role, pin=pin,
                reason="Unbound SP3T throw is intentionally open",
            )
            for name, pin in contacts if name not in ports
        ),
    )


def _coin_cell_holder(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("positive", "negative"))
    if (
        ports is None
        or any(not net.strip() for net in ports.values())
        or ports["positive"] == ports["negative"]
        or set(requirement.parameters) != {"cell_format"}
        or requirement.parameters["cell_format"] != "CR2032"
        or (requirement.exact_part is not None and requirement.exact_part != "BS-07-A1BJ001")
    ):
        return None
    holder = LoweringGroup(
        role="holder",
        reference_prefix="BT",
        value="CR2032 holder",
        symbol="Device:Battery_Cell",
        footprint="Battery:BatteryHolder_MYOUNG_BS-07-A1BJ001_CR2032",
        mpn="BS-07-A1BJ001",
        datasheet="https://www.lcsc.com/datasheet/C2979167.pdf",
        sourcing_note="LCSC C2979167; MYOUNG BS-07-A1BJ001 holder; cell supplied separately",
    )
    # The installed MYOUNG land pattern marks pad 1 positive (the retaining
    # contact), pad 2 negative. Device:Battery_Cell uses the same 1+/2- convention.
    return _artifact(
        "coin-cell-holder@1",
        requirement,
        (holder,),
        (
            LoweringPin(role="holder", pin="1", net=ports["positive"]),
            LoweringPin(role="holder", pin="2", net=ports["negative"]),
        ),
        assumptions=("CR2032 holder only; the removable cell is supplied separately.",),
        named_part_identities=("CR2032",),
    )


# The reviewed bare-board pad field: one 1.5 mm through-hole pad per 2.54 mm
# grid position. The pad is vendored (`parts_library/prototyping-area`) because
# no stock single-pad footprint can be laid at 0.1 inch: the array placer floors
# a requested pitch to the member's courtyard plus its gap, and every stock pad's
# courtyard is wider than 1.94 mm. The vendored courtyard is 1.93 mm, so
# 1.93 + 0.6 = 2.53 stays inside the declared 2.54 mm pitch.
_PROTOTYPING_AREA_FAMILY = "prototyping-area"
_PROTOTYPING_AREA_SYMBOL = "prototyping-area:PrototypingPad"
_PROTOTYPING_AREA_FOOTPRINT = "prototyping-area:PrototypingPad_1.5mm_Drill0.8mm"
_PROTOTYPING_AREA_PITCH_MM = 2.54
_PROTOTYPING_AREA_DEFAULT_ROWS = 5
_PROTOTYPING_AREA_DEFAULT_COLS = 5
# The smallest field a user can solder a part into, and the size the
# `prototyping_area` acceptance gate counts as usable (5x5 at 0.1 inch).
_PROTOTYPING_AREA_MIN_PADS = 25
_PROTOTYPING_AREA_MAX_PADS = 500


def _prototyping_area(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Lower a bare-board 2.54 mm pad field: real pads, no nets, one grid.

    The field is a fabrication feature, not a component: it owns no contact, so
    the requirement declares no ports and this is the only lowerer that builds
    anything from one. Every pad is deliberately unwired and is declared as a
    no-connect (the same deterministic path the other lowerers use), so net
    coverage (§9.11) is satisfied without any provider involvement. The grid
    travels with the group as a declared array, so the members are laid out as
    one uniform grid at the requested pitch (the placer floors that pitch to the
    pad's courtyard plus its gap) instead of being scattered by the solver.
    """

    rows = requirement.parameters.get("rows", _PROTOTYPING_AREA_DEFAULT_ROWS)
    cols = requirement.parameters.get("cols", _PROTOTYPING_AREA_DEFAULT_COLS)
    pitch = requirement.parameters.get("pitch_mm", _PROTOTYPING_AREA_PITCH_MM)
    # JSON booleans are ints in Python; a boolean is not a declared dimension.
    if type(rows) is not int or type(cols) is not int:
        return None
    if not 1 <= rows <= _PROTOTYPING_AREA_MAX_PADS:
        return None
    if not 1 <= cols <= _PROTOTYPING_AREA_MAX_PADS:
        return None
    pads = rows * cols
    if not _PROTOTYPING_AREA_MIN_PADS <= pads <= _PROTOTYPING_AREA_MAX_PADS:
        return None
    if isinstance(pitch, bool) or not isinstance(pitch, (int, float)):
        return None
    # Only the 0.1 inch grid is realized: the pad's courtyard and the array
    # placer's own grid arithmetic are both built around it, so a different
    # declared pitch is refused rather than silently redrawn at 2.54 mm.
    if abs(float(pitch) - _PROTOTYPING_AREA_PITCH_MM) > 1e-6:
        return None
    field = LoweringGroup(
        role="pad_field",
        reference_prefix="PB",
        quantity=pads,
        value=f"Prototyping pad, {_PROTOTYPING_AREA_PITCH_MM} mm pitch",
        symbol=_PROTOTYPING_AREA_SYMBOL,
        footprint=_PROTOTYPING_AREA_FOOTPRINT,
        assembly=False,
        array=LoweringArray(
            pattern="grid",
            rows=rows,
            cols=cols,
            pitch_mm=float(pitch),
        ),
    )
    return _artifact(
        "prototyping-area@1",
        requirement,
        (field,),
        (),
        no_connects=tuple(
            LoweringNoConnect(
                role="pad_field",
                index=index,
                pin="1",
                reason="bare prototyping pad; the user wires the field by hand",
            )
            for index in range(pads)
        ),
        assumptions=(
            "The field is a grid of bare 1.5 mm through-hole pads (0.8 mm drill) on a "
            "2.54 mm pitch; no pad carries a net, because the user wires the field.",
            "The pads are placed as a declared grid, never by the placement solver, so "
            "the delivered board keeps the 2.54 mm pitch.",
        ),
    )


def _capacitive_touch_pad(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Lower board-fabricated PTC electrodes with explicit no-underlay copper."""

    count = requirement.parameters.get("count")
    pin_names = requirement.parameters.get("pins")
    no_underlay = requirement.parameters.get("no_copper_underlay")
    if type(count) is not int or not 1 <= count <= 500 or no_underlay is not True:
        return None
    if type(pin_names) is int:
        # `pins` is the number of electrode pins the field needs, which is one per electrode. A
        # draft that states that number instead of spelling each pin's own name states the same
        # requirement: the pad field owns no connector pin, and the two-electrodes-on-one-contact
        # defect is still refused below, on distinct nets.
        if pin_names != count:
            return None
    elif isinstance(pin_names, str):
        pins = tuple(name.strip() for name in pin_names.split(","))
        if len(pins) != count or len(set(pins)) != count or any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in pins
        ):
            return None
    else:
        return None
    port_names = tuple(f"touch{index}" for index in range(1, count + 1))
    ports = _require_ports(requirement, port_names)
    if ports is None or len(set(ports.values())) != count:
        return None
    electrode = LoweringGroup(
        role="electrode",
        reference_prefix="PAD",
        quantity=count,
        value="Capacitive touch electrode, 12 mm",
        symbol="capacitive-touch-pad:CapacitiveTouchPad",
        footprint="capacitive-touch-pad:TouchPad_12mm_Front_NoUnderlay",
        datasheet=(
            "https://onlinedocs.microchip.com/oxy/"
            "GUID-A8A0085D-58D1-4E41-A07D-B93BFDE11AFE-en-US-4/"
            "GUID-51E19131-0329-4E41-9FE1-E3905434030B.html"
        ),
        assembly=False,
    )
    artifact_pins = tuple(
        LoweringPin(
            role="electrode",
            index=index,
            pin="1",
            net=ports[f"touch{index + 1}"],
        )
        for index in range(count)
    )
    return _artifact(
        "capacitive-touch-pad@1",
        requirement,
        (electrode,),
        artifact_pins,
        assumptions=(
            "Each electrode is a 12 mm round F.Cu copper pad covered by solder mask, "
            "with a 16 mm B.Cu no-underlay rule area; neither dimensions nor the "
            "clearance are user-specified.",
            "Microchip AN2934 treats 12 mm as a typical button height and recommends "
            "4 mm plus touch-cover thickness between electrodes; acquisition settings "
            "and thresholds require stackup/cover validation.",
        ),
    )


def _decoupling(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("vdd", "gnd"))
    try:
        count = int(requirement.parameters.get("count", 0))
    except (TypeError, ValueError):
        return None
    value = str(requirement.parameters.get("value", ""))
    if ports is None or not 1 <= count <= 64 or not value:
        return None
    group = _passive("capacitor", "C", value, quantity=count)
    pins = tuple(
        pin
        for index in range(count)
        for pin in (
            LoweringPin(role="capacitor", index=index, pin="1", net=ports["vdd"]),
            LoweringPin(role="capacitor", index=index, pin="2", net=ports["gnd"]),
        )
    )
    return _artifact("explicit-decoupling@1", requirement, (group,), pins)


def _analog_power_cycle_reset(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Power-cycle one regulated analog rail with supervised external isolation."""
    if requirement.exact_part is not None or set(requirement.parameters) != {
        "rail_voltage_v", "source_current_a"
    }:
        return None
    try:
        rail_voltage = float(requirement.parameters["rail_voltage_v"])
        source_current = float(requirement.parameters["source_current_a"])
    except (TypeError, ValueError):
        return None
    if rail_voltage != 5.0 or source_current != 0.5:
        return None
    required_ports = {"input", "output", "gnd"}
    if not required_ports <= set(requirement.ports) or set(requirement.ports) - (
        required_ports | {"isolation_enable"}
    ):
        return None
    ports = {key: str(value) for key, value in requirement.ports.items()}
    if (
        any(not ports[key].strip() for key in required_ports)
        or any(net.startswith("__KICRAFT_ANALOG_POWER_CYCLE__") for net in ports.values())
        or len({ports[key] for key in required_ports}) != 3
        or (
            "isolation_enable" in ports
            and ports["isolation_enable"] in {ports["input"], ports["output"], ports["gnd"]}
        )
    ):
        return None
    source = ports["input"]
    prefix = f"__KICRAFT_ANALOG_POWER_CYCLE__{requirement.id}__"
    enable = f"{prefix}ENABLE"
    supervisor_reset = f"{prefix}SUPERVISOR_RESET"
    isolation_enable = ports.get("isolation_enable", f"{prefix}ISOLATION_ENABLE")
    supervisor = "TPS3808G50DBVR"
    supervisor_threshold = 4.65
    supervisor_accuracy = 0.02
    groups = (
        LoweringGroup(
            role="power_cycle_switch", reference_prefix="U", value="TPS22948DCKR",
            symbol="tps22948dckr:TPS22948DCKR",
            footprint="tps22948dckr:SC-70-6_L2.0-W1.3-P0.65-LS2.1-BR",
            mpn="TPS22948DCKR", datasheet="https://www.ti.com/lit/ds/symlink/tps22948.pdf",
        ),
        LoweringGroup(
            role="power_cycle_supervisor", reference_prefix="U", value=supervisor,
            symbol="Power_Supervisor:TPS3808DBV",
            footprint="Package_TO_SOT_SMD:SOT-23-6", mpn=supervisor,
            datasheet="https://www.ti.com/lit/ds/symlink/tps3808.pdf",
        ),
        LoweringGroup(
            role="isolation_control_buffer", reference_prefix="U", value="SN74LVC1G17DBVR",
            symbol="74xGxx:74LVC1G17", footprint="Package_TO_SOT_SMD:SOT-23-5",
            mpn="SN74LVC1G17DBVR", datasheet="https://www.ti.com/lit/ds/symlink/sn74lvc1g17.pdf",
        ),
        LoweringGroup(
            role="isolation_control_decoupling", reference_prefix="C", value="100nF",
            symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric",
            mpn="GRM188R71C104KA01D",
        ),
        LoweringGroup(
            role="power_cycle_button", reference_prefix="SW", value="POWER CYCLE",
            symbol="kmr232ngulclfs:KMR232NGULCLFS", footprint="kmr232ngulclfs:KMR232NGULCLFS",
            mpn="KMR232NGULCLFS",
            datasheet="https://datasheet.lcsc.com/datasheet/pdf/b5da7fd4e81211d634f1332fb1486a1f.pdf",
        ),
        LoweringGroup(
            role="reset_enable_series", reference_prefix="R", value="5.9k",
            symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", mpn="FRC0603F5901TS",
        ),
        LoweringGroup(
            role="reset_enable_pulldown", reference_prefix="R", value="60k",
            symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        LoweringGroup(
            role="supervisor_reset_pullup", reference_prefix="R", value="100k",
            symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        LoweringGroup(
            role="switch_input_decoupling", reference_prefix="C", value="1uF",
            symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric",
        ),
        LoweringGroup(
            role="switch_output_decoupling", reference_prefix="C", value="18nF",
            symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric",
        ),
        LoweringGroup(
            role="supervisor_decoupling", reference_prefix="C", value="100nF",
            symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric",
            mpn="GRM188R71C104KA01D",
        ),
        LoweringGroup(
            role="output_discharge", reference_prefix="R", value="1k",
            symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", mpn="AC0603FR-071KL",
        ),
    )
    pins = (
        LoweringPin(role="power_cycle_switch", pin="1", net=source),
        LoweringPin(role="power_cycle_switch", pin="2", net=ports["gnd"]),
        LoweringPin(role="power_cycle_switch", pin="3", net=enable),
        LoweringPin(role="power_cycle_switch", pin="6", net=ports["output"]),
        LoweringPin(role="power_cycle_supervisor", pin="1", net=supervisor_reset),
        LoweringPin(role="power_cycle_supervisor", pin="2", net=ports["gnd"]),
        LoweringPin(role="power_cycle_supervisor", pin="3", net=source),
        LoweringPin(role="power_cycle_supervisor", pin="5", net=ports["output"]),
        LoweringPin(role="power_cycle_supervisor", pin="6", net=source),
        LoweringPin(role="isolation_control_buffer", pin="2", net=supervisor_reset),
        LoweringPin(role="isolation_control_buffer", pin="3", net=ports["gnd"]),
        LoweringPin(role="isolation_control_buffer", pin="4", net=isolation_enable),
        LoweringPin(role="isolation_control_buffer", pin="5", net=source),
        LoweringPin(role="isolation_control_decoupling", pin="1", net=source),
        LoweringPin(role="isolation_control_decoupling", pin="2", net=ports["gnd"]),
        LoweringPin(role="reset_enable_series", pin="1", net=source),
        LoweringPin(role="reset_enable_series", pin="2", net=enable),
        LoweringPin(role="reset_enable_pulldown", pin="1", net=enable),
        LoweringPin(role="reset_enable_pulldown", pin="2", net=ports["gnd"]),
        LoweringPin(role="supervisor_reset_pullup", pin="1", net=supervisor_reset),
        LoweringPin(role="supervisor_reset_pullup", pin="2", net=enable),
        LoweringPin(role="power_cycle_button", pin="1", net=enable),
        LoweringPin(role="power_cycle_button", pin="2", net=ports["gnd"]),
        LoweringPin(role="switch_input_decoupling", pin="1", net=source),
        LoweringPin(role="switch_input_decoupling", pin="2", net=ports["gnd"]),
        LoweringPin(role="switch_output_decoupling", pin="1", net=ports["output"]),
        LoweringPin(role="switch_output_decoupling", pin="2", net=ports["gnd"]),
        LoweringPin(role="supervisor_decoupling", pin="1", net=source),
        LoweringPin(role="supervisor_decoupling", pin="2", net=ports["gnd"]),
        LoweringPin(role="output_discharge", pin="1", net=ports["output"]),
        LoweringPin(role="output_discharge", pin="2", net=ports["gnd"]),
    )
    return _artifact(
        "analog-power-cycle-reset@1", requirement, groups, pins,
        no_connects=(
            LoweringNoConnect(
                role="power_cycle_switch", pin="4",
                reason="TPS22948 NC is deliberately left floating.",
            ),
            LoweringNoConnect(
                role="power_cycle_switch", pin="5",
                reason="TPS22948 FLT is unused; power-cycle recovery is controlled by the supervisor.",
            ),
            LoweringNoConnect(
                role="power_cycle_supervisor", pin="4",
                reason="TPS3808 CT is deliberately open for its documented 20 ms nominal reset delay.",
            ),
            LoweringNoConnect(
                role="isolation_control_buffer", pin="1",
                reason="SN74LVC1G17 NC has no internal connection.",
            ),
        ),
        assumptions=(
            f"TPS22948 operates only from the already-regulated {rail_voltage:g} V input rail; "
            "a 24 V source must be converted before this circuit.",
            f"{supervisor} is independently powered from the unswitched input; its SENSE "
            "pin monitors the switched analog rail. Its open-drain RESET feeds an unswitched "
            "SN74LVC1G17 Schmitt buffer, which drives isolation_enable low until the rail "
            "exceeds the reviewed threshold and CT-open delay. The buffer regenerates the "
            "slow 100 kOhm pull-up edge instead of driving the SN74LVC1G66 controls directly. "
            "MR is held high from the unswitched source. RESET behavior is guaranteed only "
            "after its 1.7 V minimum VDD is reached; below that startup region isolation is not "
            "advertised as safe or functional.",
            "Released EN is set by 5.9 kOhm to source and 60 kOhm to return; pressing the "
            "momentary button grounds EN and opens the TPS22948. Its guaranteed continuous "
            "rating is 130 mA; the integrated current-limit range is 130--350 mA. "
            "The connected analog loads and discharge resistor are limited to <=100 mA.",
            "The switched-domain capacitance limit is 1 uF. A 1 kOhm 1% bleed bounds "
            "sixteen SN74LVC1G66 off leaks, TPS22948 forward-off input current with OUT "
            "grounded, and TPS3808 SENSE/RESET leakage to <=23.23 mV, below 1% of "
            "the supported 5 V rail. Unknown external drives or loads remain unverified "
            "unless every such path is isolated by a reviewed switch controlled by isolation_enable.",
        ),
        calculations=(
            LoweringCalculation(
                output="power_cycle_discharge_to_1_percent",
                equation="t=Rmax*Cmax*ln((Vrail-Ileak*Rmax)/(0.01*Vrail-Ileak*Rmax)) after the power switch and signal isolators are off",
                inputs={"rail_voltage_v": rail_voltage, "bleed_resistance_ohm_max": 1_010, "off_leakage_a_max": 23e-6, "total_switched_capacitance_uf_max": 1},
                chosen_value="<=0.006s after power and signal isolation",
                tolerance="This bounds passive discharge after isolation, not control propagation. TPS22948 SLVSEZ7A gives typical, not maximum, switching delay; no guaranteed total button-to-discharge time is claimed. Capacitance is an integration limit.",
            ),
            LoweringCalculation(
                output="isolation_release_threshold",
                equation="TPS3808 fixed SENSE threshold with the selected threshold variant",
                inputs={"sense_threshold_v": supervisor_threshold, "threshold_accuracy_fraction": supervisor_accuracy, "hysteresis_fraction_max": 0.025},
                chosen_value=f"{supervisor_threshold:g} V nominal",
                tolerance=f"-40..125 C threshold accuracy +/-{supervisor_accuracy * 100:g}%; fixed hysteresis 1% typical, 2.5% maximum.",
            ),
            LoweringCalculation(
                output="isolation_reconnect_delay",
                equation="TPS3808 CT-open reset delay after SENSE rises above the reviewed threshold",
                inputs={"ct_open_delay_min_ms": 12, "ct_open_delay_typ_ms": 20, "ct_open_delay_max_ms": 28},
                chosen_value="20 ms typical; 12--28 ms bounded",
                tolerance="TPS3808 SBVS050N; RESET remains asserted before and during this interval.",
            ),
            LoweringCalculation(
                output="off_state_leakage_voltage",
                equation="1.01k*(16*1uA + 5uA TPS22948 + 1.7uA SENSE + 0.3uA RESET)",
                inputs={"bleed_resistance_ohm_max": 1_010, "isolator_count_max": 16, "isolator_off_leakage_ua_max": 1, "switch_forward_off_current_ua_max": 5, "supervisor_sense_leakage_ua_max": 1.7, "supervisor_reset_leakage_ua_max": 0.3},
                chosen_value="23.23 mV maximum",
                tolerance="TPS22948 off-current bound is specified at VIN=5 V; no bound is claimed at other supply voltages. External leakage is not bounded by this circuit.",
            ),
        ),
        named_part_identities=("TPS22948DCKR", supervisor, "SN74LVC1G17DBVR"),
    )

def _common_anode_rgb_indicator(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Build one 5-V common-anode RGB indicator with three 3.3-V GPIO controls.

    The common anode uses 5 V, while each cathode is sunk by a discrete NPN
    whose base is resistor-driven from a 3.3-V GPIO. A base-emitter pull-down
    makes the power-up/reset state off.
    """
    ports = _require_ports(requirement, ("vdd_5v", "gnd", "red", "green", "blue"))
    if (
        ports is None
        or set(requirement.parameters) != {"target_current_ma"}
        or requirement.parameters["target_current_ma"] != 2.0
        or (requirement.exact_part or "xl-1615rgbc-rf").casefold()
        not in {"aptf1616lseezgkqbkc", "xl-1615rgbc-rf"}
        or len(set(ports.values())) != len(ports)
        or any(net.startswith("__KICRAFT_RGB_INDICATOR_") for net in ports.values())
    ):
        return None
    led = reviewed_part(requirement.exact_part or "XL-1615RGBC-RF")
    assert led is not None
    mpn = "APTF1616LSEEZGKQBKC" if led.identity == "aptf1616lseezgkqbkc" else "XL-1615RGBC-RF"


    groups = (
        LoweringGroup(
            role="rgb_led",
            reference_prefix="D",
            value=mpn,
            symbol=led.symbol,
            footprint=led.footprint,
            mpn=mpn,
            datasheet=led.manufacturer_sources[0],
        ),
        LoweringGroup(
            role="red_current_resistor",
            reference_prefix="R",
            value="1.5k 1%",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        LoweringGroup(
            role="green_blue_current_resistor",
            reference_prefix="R",
            quantity=2,
            value="820R 1%",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        LoweringGroup(
            role="channel_driver",
            reference_prefix="Q",
            quantity=3,
            value="MMBT3904",
            symbol="Transistor_BJT:MMBT3904",
            footprint="Package_TO_SOT_SMD:SOT-23",
            mpn="MMBT3904",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8586176249693941760",
        ),
        LoweringGroup(
            role="base_resistor",
            reference_prefix="R",
            quantity=3,
            value="4.7k 1%",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        LoweringGroup(
            role="base_pulldown",
            reference_prefix="R",
            quantity=3,
            value="100k 1%",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
    )
    prefix = f"__KICRAFT_RGB_INDICATOR_{requirement.id}__"
    cathodes = tuple(f"{prefix}{color}_K" for color in ("RED", "GREEN", "BLUE"))
    collectors = tuple(f"{prefix}COLLECTOR_{color}" for color in range(3))
    bases = tuple(f"{prefix}BASE_{color}" for color in range(3))
    controls = (ports["red"], ports["green"], ports["blue"])
    pins: list[LoweringPin] = [
        LoweringPin(role="rgb_led", pin=led.port_pins["common_anode"], net=ports["vdd_5v"]),
        *(LoweringPin(role="rgb_led", pin=led.port_pins[f"{color}_cathode"], net=cathodes[index])
          for index, color in enumerate(("red", "green", "blue"))),
    ]
    for index, (cathode, collector, base, control) in enumerate(
        zip(cathodes, collectors, bases, controls, strict=True)
    ):
        resistor_role = "red_current_resistor" if index == 0 else "green_blue_current_resistor"
        resistor_index = 0 if index == 0 else index - 1
        pins.extend(
            (
                LoweringPin(role=resistor_role, index=resistor_index, pin="1", net=cathode),
                LoweringPin(role=resistor_role, index=resistor_index, pin="2", net=collector),
                LoweringPin(role="channel_driver", index=index, pin="1", net=base),
                LoweringPin(role="channel_driver", index=index, pin="2", net=ports["gnd"]),
                LoweringPin(role="channel_driver", index=index, pin="3", net=collector),
                LoweringPin(role="base_resistor", index=index, pin="1", net=control),
                LoweringPin(role="base_resistor", index=index, pin="2", net=base),
                LoweringPin(role="base_pulldown", index=index, pin="1", net=base),
                LoweringPin(role="base_pulldown", index=index, pin="2", net=ports["gnd"]),
            )
        )
    return _artifact(
        "common-anode-rgb-indicator@2",
        requirement,
        groups,
        tuple(pins),
        assumptions=(
            f"{mpn} is a four-contact common-anode RGB LED: "
            "the 5 V common anode is not connected to any MCU GPIO.",
            "The 1.5 kOhm red and 820 Ohm green/blue resistors limit individual "
            "channels; each channel's 4.7 kOhm base resistor limits its own GPIO load.",
            "Light output at the actual operating current is unmeasured; "
            "the LED's forward-voltage and luminous-intensity specifications are not a 2 mA guarantee.",
            "A qualified MCU output must use its manufacturer-documented 3.3 V loaded-output "
            "condition and source-current rating for that base load. Such rows are per GPIO; "
            "this circuit makes no simultaneous-pin or aggregate GPIO-current claim.",
            "Each 100 kOhm base-emitter pull-down holds its NPN off while GPIOs are reset or "
            "unpowered. GPIO-output configuration remains a firmware responsibility.",
        ),
        calculations=(
            LoweringCalculation(
                output="channel_current_upper_bound",
                equation="I <= 5.25 V / (820 Ohm * 0.99)",
                inputs={"supply_voltage_v_max": 5.25, "resistance_ohm_min": 811.8},
                chosen_value="6.47 mA per channel maximum",
                tolerance="Conservative zero LED and transistor voltage drop; 1% resistor, 5% supply. No brightness claim.",
            ),
        ),
        named_part_identities=(led.identity, "mmbt3904"),
    )


_DISTRIBUTION_OUTPUT_RE = re.compile(r"output([1-8])$")


def _fused_power_distribution(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """Build a source-owned, separately fused low-to-mid-voltage distribution block.

    Each typed output port is the branch after its own fuse. The architecture
    owns board-edge connector selection and maps each requested positive/return
    contact onto these ports. The optional removable jumper actuates a high-side
    PMOS; it never exposes a raw supply on an unprotected logic/GPIO control,
    and its fitted and removed states are visible in the generated netlist.
    """
    allowed = {
        "channels",
        "input_voltage_v",
        "source_current_a",
        "enable_jumper",
        "reset_button",
        "rgb_status",
    }
    if set(requirement.parameters) - allowed or requirement.exact_part is not None:
        return None
    try:
        channels = requirement.parameters["channels"]
        input_voltage = float(requirement.parameters["input_voltage_v"])
        source_current = float(requirement.parameters["source_current_a"])
    except (KeyError, TypeError, ValueError):
        return None
    enable_jumper = requirement.parameters.get("enable_jumper", False)
    reset_button = requirement.parameters.get("reset_button", False)
    rgb_status = requirement.parameters.get("rgb_status", False)
    if (
        type(enable_jumper) is not bool
        or type(reset_button) is not bool
        or type(rgb_status) is not bool
        or type(channels) is not int
        or channels not in {2, 4, 8}
        or input_voltage not in {5.0, 6.0, 7.4, 8.4, 12.0, 18.0, 24.0}
        or source_current not in {0.5, 1.0, 2.0}
    ):
        return None
    required_ports = {"input", "gnd"} | {f"output{index}" for index in range(1, channels + 1)}
    indexed_outputs = {
        int(match.group(1)): net
        for key, net in requirement.ports.items()
        if (match := _DISTRIBUTION_OUTPUT_RE.fullmatch(key))
    }
    if (
        set(requirement.ports) != required_ports
        or set(indexed_outputs) != set(range(1, channels + 1))
        or any(not str(net).strip() for net in requirement.ports.values())
        or any(net.startswith(("__KICRAFT_DISTRIBUTION__", "__KICRAFT_EFUSE_")) for net in requirement.ports.values())
        or len(set(indexed_outputs.values())) != channels
        or requirement.ports["input"] == requirement.ports["gnd"]
        or requirement.ports["gnd"] in set(indexed_outputs.values())
        or requirement.ports["input"] in set(indexed_outputs.values())
    ):
        return None
    high_voltage_reset = reset_button and input_voltage != 5.0
    if (
        (not high_voltage_reset and source_current != 0.5)
        or (high_voltage_reset and source_current not in {0.5, 1.0, 2.0})
        or (high_voltage_reset and source_current > channels * 0.5)
    ):
        # The 5 V path has only PTC branch protection. Its source budget stays
        # at the conservative 0.5 A default. Higher-current high-voltage
        # distributions use the TPS26600 eFuse below; 3 A remains unsupported
        # instead of being silently clamped to that part's 2.23 A maximum.
        return None
    # Each branch has a 0.5 A-hold, 30 V PTC for fault protection. A PTC hold
    # value is not a precision current limit or a continuous-load guarantee.
    # The 30 V rating, unlike the earlier proposed 16 V part, remains valid at
    # the 24 V input maximum.
    branch_hold_current = 0.5
    fuse = LoweringGroup(
        role="branch_fuse",
        reference_prefix="F",
        quantity=channels,
        value="MF-MSMF050/30X-2",
        symbol="mf-msmf050-30x-2:Polyfuse_1812",
        footprint="mf-msmf050-30x-2:MF-MSMF-1812",
        mpn="MF-MSMF050/30X-2",
        datasheet="https://www.bourns.com/docs/product-datasheets/mf-msmf.pdf",
        sourcing_note=(
            "Bourns MF-MSMF050/30X-2 resettable fuse: 0.5 A hold, 30 V maximum; "
            "one is fitted in every delivered branch."
        ),
    )
    source_fuse = LoweringGroup(
        role="source_fuse",
        reference_prefix="F",
        value="MF-MSMF075/33X-2",
        symbol="mf-msmf075-33x-2:Polyfuse_1812",
        footprint="mf-msmf075-33x-2:MF-MSMF-1812",
        mpn="MF-MSMF075/33X-2",
        datasheet="https://www.bourns.com/docs/product-datasheets/mf-msmf.pdf",
        sourcing_note=(
            "Bourns MF-MSMF075/33X-2 resettable input fuse: 0.75 A hold, 33 V "
            "maximum; it limits faults on the 0.5 A non-eFuse source path."
        ),
    )

    resistor_count = max(2, math.ceil((input_voltage - 2.0) / 2.0))
    groups: list[LoweringGroup] = [
        *(() if high_voltage_reset else (source_fuse,)),
        fuse,
        LoweringGroup(
            role="indicator_resistor",
            reference_prefix="R",
            quantity=resistor_count,
            value="1k",
            symbol="Device:R",
            footprint="Resistor_SMD:R_0603_1608Metric",
            mpn="AC0603FR-071KL",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8754852008425861120",
        ),
        LoweringGroup(
            role="indicator_led",
            reference_prefix="D",
            value="LTST-C190KGKT",
            symbol="Device:LED",
            footprint="LED_SMD:LED_0603_1608Metric",
            mpn="LTST-C190KGKT",
            datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8588884651616583680",
        ),
    ]
    source_bus = (
        "__KICRAFT_DISTRIBUTION__EFUSE_OUT"
        if high_voltage_reset
        else "__KICRAFT_DISTRIBUTION__SOURCE_FUSED"
    )
    protected_source_bus = source_bus
    pins: list[LoweringPin] = []
    if not high_voltage_reset:
        pins.extend(
            (
                LoweringPin(role="source_fuse", pin="1", net=requirement.ports["input"]),
                LoweringPin(role="source_fuse", pin="2", net=source_bus),
            )
        )
    no_connects: list[LoweringNoConnect] = []
    assumptions = [
        (
            f"Each branch uses a {branch_hold_current:g} A-hold PTC at 23 °C, "
            "with a 1 A trip rating and temperature derating; this is not a precision current limit. "
            f"The {source_current:g} A source budget is aggregate, not a per-branch guarantee."
        ),
        (
            "Each output port is the unique post-fuse branch rail. Connector "
            "ownership remains with the architecture’s explicitly requested I/O."
        ),
    ]
    if reset_button and not high_voltage_reset:
        groups.extend(
            (
                *((LoweringGroup(
                    role="reset_enable_jumper",
                    reference_prefix="J",
                    value="TSW-102-07-G-S",
                    symbol="Connector_Generic:Conn_01x02",
                    footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                    mpn="TSW-102-07-G-S",
                    sourcing_note="Fit SNT-100-BK-G to enable all output switches; remove to turn all outputs off.",
                ),) if enable_jumper else ()),
                LoweringGroup(
                    role="reset_enable_series", reference_prefix="R", value="5.9k",
                    symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", mpn="0603WAF5901T5E",
                ),
                LoweringGroup(
                    role="reset_enable_pulldown", reference_prefix="R", quantity=6, value="10k",
                    symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", mpn="0603WAF1002T5E",
                ),
                LoweringGroup(
                    role="power_cycle_button", reference_prefix="SW", value="POWER CYCLE",
                    symbol="kmr232ngulclfs:KMR232NGULCLFS", footprint="kmr232ngulclfs:KMR232NGULCLFS",
                    mpn="KMR232NGULCLFS",
                ),
                LoweringGroup(
                    role="branch_switch", reference_prefix="U", quantity=channels, value="TPS2553DBVR",
                    symbol="tps2553dbvr:TPS2553DBVR", footprint="tps2553dbvr:SOT-23-6_DBV",
                    mpn="TPS2553DBVR", datasheet="https://www.ti.com/lit/ds/symlink/tps2553.pdf",
                ),
                LoweringGroup(
                    role="switch_ilim_top", reference_prefix="R", quantity=channels, value="10k",
                    symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", mpn="0603WAF1002T5E",
                ),
                LoweringGroup(
                    role="switch_ilim_bottom", reference_prefix="R", quantity=channels, value="5.9k",
                    symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", mpn="0603WAF5901T5E",
                ),
                LoweringGroup(
                    role="switch_input_decoupling", reference_prefix="C", quantity=channels, value="100nF",
                    symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric", mpn="GRM188R71C104KA01D",
                ),
                LoweringGroup(
                    role="output_discharge", reference_prefix="R", quantity=channels, value="10k",
                    symbol="Device:R", footprint="Resistor_SMD:R_0603_1608Metric", mpn="0603WAF1002T5E",
                ),
            )
        )
        pins.extend(
            (
                *((
                    LoweringPin(role="reset_enable_jumper", pin="1", net=protected_source_bus),
                    LoweringPin(role="reset_enable_jumper", pin="2", net="__KICRAFT_DISTRIBUTION__ENABLE_FEED"),
                ) if enable_jumper else ()),
                LoweringPin(
                    role="reset_enable_series", pin="1",
                    net="__KICRAFT_DISTRIBUTION__ENABLE_FEED" if enable_jumper else protected_source_bus,
                ),
                LoweringPin(role="reset_enable_series", pin="2", net="__KICRAFT_DISTRIBUTION__ENABLE"),
                LoweringPin(role="power_cycle_button", pin="1", net="__KICRAFT_DISTRIBUTION__ENABLE"),
                LoweringPin(role="power_cycle_button", pin="2", net=requirement.ports["gnd"]),
            )
        )
        for index in range(6):
            pins.extend(
                (
                    LoweringPin(
                        role="reset_enable_pulldown", index=index, pin="1",
                        net="__KICRAFT_DISTRIBUTION__ENABLE" if index == 0 else f"__KICRAFT_DISTRIBUTION__RESET_PULL_{index}",
                    ),
                    LoweringPin(
                        role="reset_enable_pulldown", index=index, pin="2",
                        net=requirement.ports["gnd"] if index == 5 else f"__KICRAFT_DISTRIBUTION__RESET_PULL_{index + 1}",
                    ),
                )
            )
        assumptions.append(
            "The enabled state drives TPS2553 EN≈4.55 V through 5.9 kOhm against "
            "the 60 kOhm pulldown. Pressing the button, or removing the optional enable "
            "shunt when present, drives EN low. Each TPS2553 is limited by a 15.9 kOhm "
            "ILIM network to about 1.4 A and provides reverse-current blocking; each output "
            "has a 10 kOhm discharge path."
        )
    if high_voltage_reset:
        # TPS26600's back-to-back FETs are the sole global switching element.
        # GND is the source/load return; RTN remains an isolated control return,
        # including the PowerPAD, dividers and reset button, per TI SLVSDG2G
        # sections 9.3.5.5 and 10.4.  Tying those returns together would defeat
        # reverse-polarity protection.
        divider = {
            6.0: (6, 0, 10.0),
            7.4: (8, 0, 10.0),
            8.4: (8, 0, 10.0),
            12.0: (11, 0, 5.9),
            18.0: (17, 0, 5.9),
            24.0: (23, 1, 5.9),
        }[input_voltage]
        top_10k_count, top_5k9_count, divider_mid_kohm = divider
        # Bounds use 1% resistor extrema and TI's specified rising thresholds:
        # UVLO 1.175--1.225 V, OVP 1.170--1.225 V.
        divider_bounds = {
            6.0: (4.630, 4.974, 9.175, 9.998),
            7.4: (5.782, 6.224, 11.468, 12.497),
            8.4: (5.782, 6.224, 11.468, 12.497),
            12.0: (9.143, 9.871, 14.439, 15.734),
            18.0: (13.489, 14.587, 21.320, 23.233),
            24.0: (18.263, 19.767, 28.877, 31.469),
        }[input_voltage]
        ilim = {
            0.5: ("22.6k", "RS-03K2262FT"),
            1.0: ("11.3k", "0603WAF1132T5E"),
            2.0: ("5.36k", "0603WAF5361T5E"),
        }[source_current]
        groups.extend(
            (
                LoweringGroup(
                    role="global_efuse",
                    reference_prefix="U",
                    value="TPS26600PWPR",
                    symbol="tps26600:TPS26600PWPR",
                    footprint="tps26600:HTSSOP-16_L5.0-W4.4-P0.65-LS6.4-BL-EP3.3",
                    mpn="TPS26600PWPR",
                    datasheet="https://www.ti.com/lit/ds/symlink/tps2660.pdf",
                ),
                LoweringGroup(
                    role="efuse_input_ceramic",
                    reference_prefix="C",
                    value="100nF 100V X7R",
                    symbol="Device:C",
                    footprint="Capacitor_SMD:C_0603_1608Metric",
                    mpn="GRM188R72A104KA35D",
                ),
                LoweringGroup(
                    role="efuse_output_ceramic",
                    reference_prefix="C",
                    value="1uF 50V X7R",
                    symbol="Device:C",
                    footprint="Capacitor_SMD:C_0603_1608Metric",
                    mpn="CL10B105KB8NQNC",
                ),
                LoweringGroup(
                    role="efuse_ilim",
                    reference_prefix="R",
                    value=f"{ilim[0]} 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn=ilim[1],
                ),
                LoweringGroup(
                    role="efuse_dvdt",
                    reference_prefix="C",
                    value="1.5nF 50V C0G",
                    symbol="Device:C",
                    footprint="Capacitor_SMD:C_0603_1608Metric",
                    mpn="GRM1885C1H152JA01D",
                ),
                LoweringGroup(
                    role="efuse_uvlo_ovp_top_10k",
                    reference_prefix="R",
                    quantity=top_10k_count,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="0603WAF1002T5E",
                ),
                *(
                    (
                        LoweringGroup(
                            role="efuse_uvlo_ovp_top_5k9",
                            reference_prefix="R",
                            value="5.9k 1%",
                            symbol="Device:R",
                            footprint="Resistor_SMD:R_0603_1608Metric",
                            mpn="0603WAF5901T5E",
                        ),
                    )
                    if top_5k9_count
                    else ()
                ),
                LoweringGroup(
                    role="efuse_uvlo_ovp_mid",
                    reference_prefix="R",
                    value=f"{divider_mid_kohm:g}k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn=(
                        "0603WAF1002T5E"
                        if divider_mid_kohm == 10.0
                        else "0603WAF5901T5E"
                    ),
                ),
                LoweringGroup(
                    role="efuse_uvlo_ovp_bottom",
                    reference_prefix="R",
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="0603WAF1002T5E",
                ),
                LoweringGroup(
                    role="efuse_power_cycle_button",
                    reference_prefix="SW",
                    value="POWER CYCLE",
                    symbol="kmr232ngulclfs:KMR232NGULCLFS",
                    footprint="kmr232ngulclfs:KMR232NGULCLFS",
                    mpn="KMR232NGULCLFS",
                ),
                *(
                    (
                        LoweringGroup(
                            role="efuse_disable_jumper",
                            reference_prefix="J",
                            value="EFUSE_DISABLE",
                            symbol="Connector_Generic:Conn_01x02",
                            footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                            mpn="TSW-102-07-G-S",
                            sourcing_note=(
                                "Fit the shunt to pull SHDN to RTN and turn all delivered "
                                "rails off; remove it for the TPS26600 internal-pullup on state."
                            ),
                        ),
                    )
                    if enable_jumper
                    else ()
                ),
                LoweringGroup(
                    role="efuse_output_discharge",
                    reference_prefix="R",
                    quantity=channels,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="0603WAF1002T5E",
                ),
                LoweringGroup(
                    role="efuse_bus_discharge",
                    reference_prefix="R",
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="0603WAF1002T5E",
                ),
            )
        )
        control_rtn = "__KICRAFT_DISTRIBUTION__EFUSE_RTN"
        top_end = (
            "__KICRAFT_DISTRIBUTION__EFUSE_TOP_10K"
            if top_5k9_count
            else "__KICRAFT_DISTRIBUTION__EFUSE_UVLO"
        )
        for index in range(top_10k_count):
            pins.extend(
                (
                    LoweringPin(
                        role="efuse_uvlo_ovp_top_10k",
                        index=index,
                        pin="1",
                        net=(
                            requirement.ports["input"]
                            if index == 0
                            else f"__KICRAFT_DISTRIBUTION__EFUSE_TOP_{index}"
                        ),
                    ),
                    LoweringPin(
                        role="efuse_uvlo_ovp_top_10k",
                        index=index,
                        pin="2",
                        net=(
                            top_end
                            if index == top_10k_count - 1
                            else f"__KICRAFT_DISTRIBUTION__EFUSE_TOP_{index + 1}"
                        ),
                    ),
                )
            )
        if top_5k9_count:
            pins.extend(
                (
                    LoweringPin(
                        role="efuse_uvlo_ovp_top_5k9",
                        pin="1",
                        net=top_end,
                    ),
                    LoweringPin(
                        role="efuse_uvlo_ovp_top_5k9",
                        pin="2",
                        net="__KICRAFT_DISTRIBUTION__EFUSE_UVLO",
                    ),
                )
            )
        pins.extend(
            (
                *(LoweringPin(role="global_efuse", pin=pin, net=requirement.ports["input"])
                  for pin in ("1", "2")),
                LoweringPin(role="global_efuse", pin="3", net="__KICRAFT_DISTRIBUTION__EFUSE_UVLO"),
                LoweringPin(role="global_efuse", pin="5", net="__KICRAFT_DISTRIBUTION__EFUSE_OVP"),
                LoweringPin(role="global_efuse", pin="7", net="__KICRAFT_DISTRIBUTION__EFUSE_SHDN"),
                LoweringPin(role="global_efuse", pin="8", net=control_rtn),
                LoweringPin(role="global_efuse", pin="9", net=requirement.ports["gnd"]),
                LoweringPin(role="global_efuse", pin="11", net="__KICRAFT_DISTRIBUTION__EFUSE_ILIM"),
                LoweringPin(role="global_efuse", pin="12", net="__KICRAFT_DISTRIBUTION__EFUSE_DVDT"),
                *(LoweringPin(role="global_efuse", pin=pin, net=source_bus) for pin in ("15", "16")),
                LoweringPin(role="global_efuse", pin="17", net=control_rtn),
                LoweringPin(role="efuse_input_ceramic", pin="1", net=requirement.ports["input"]),
                LoweringPin(role="efuse_input_ceramic", pin="2", net=requirement.ports["gnd"]),
                LoweringPin(role="efuse_output_ceramic", pin="1", net=source_bus),
                LoweringPin(role="efuse_output_ceramic", pin="2", net=requirement.ports["gnd"]),
                LoweringPin(role="efuse_bus_discharge", pin="1", net=source_bus),
                LoweringPin(role="efuse_bus_discharge", pin="2", net=requirement.ports["gnd"]),
                LoweringPin(role="efuse_ilim", pin="1", net="__KICRAFT_DISTRIBUTION__EFUSE_ILIM"),
                LoweringPin(role="efuse_ilim", pin="2", net=control_rtn),
                LoweringPin(role="efuse_dvdt", pin="1", net="__KICRAFT_DISTRIBUTION__EFUSE_DVDT"),
                LoweringPin(role="efuse_dvdt", pin="2", net=control_rtn),
                LoweringPin(role="efuse_uvlo_ovp_mid", pin="1", net="__KICRAFT_DISTRIBUTION__EFUSE_UVLO"),
                LoweringPin(role="efuse_uvlo_ovp_mid", pin="2", net="__KICRAFT_DISTRIBUTION__EFUSE_OVP"),
                LoweringPin(role="efuse_uvlo_ovp_bottom", pin="1", net="__KICRAFT_DISTRIBUTION__EFUSE_OVP"),
                LoweringPin(role="efuse_uvlo_ovp_bottom", pin="2", net=control_rtn),
                LoweringPin(role="efuse_power_cycle_button", pin="1", net="__KICRAFT_DISTRIBUTION__EFUSE_SHDN"),
                LoweringPin(role="efuse_power_cycle_button", pin="2", net=control_rtn),
            )
        )
        if enable_jumper:
            pins.extend(
                (
                    LoweringPin(
                        role="efuse_disable_jumper",
                        pin="1",
                        net="__KICRAFT_DISTRIBUTION__EFUSE_SHDN",
                    ),
                    LoweringPin(role="efuse_disable_jumper", pin="2", net=control_rtn),
                )
            )
        for index in range(channels):
            pins.extend(
                (
                    LoweringPin(
                        role="efuse_output_discharge",
                        index=index,
                        pin="1",
                        net=indexed_outputs[index + 1],
                    ),
                    LoweringPin(
                        role="efuse_output_discharge",
                        index=index,
                        pin="2",
                        net=requirement.ports["gnd"],
                    ),
                )
            )
        no_connects.extend(
            (
                LoweringNoConnect(role="global_efuse", pin="4", reason="TPS26600 PWP pin 4 is NC"),
                LoweringNoConnect(role="global_efuse", pin="6", reason="MODE is intentionally open for TPS26600 circuit-breaker auto-retry"),
                LoweringNoConnect(role="global_efuse", pin="10", reason="IMON is unused on a standalone distribution"),
                LoweringNoConnect(role="global_efuse", pin="13", reason="TPS26600 PWP pin 13 is NC"),
                *(
                    ()
                    if rgb_status
                    else (LoweringNoConnect(
                        role="global_efuse",
                        pin="14",
                        reason="FLT has no external monitor when RGB status is absent",
                    ),)
                ),
            )
        )
        nominal_current_limit = {
            0.5: 12.0 / 22.6,
            1.0: 12.0 / 11.3,
            2.0: 2.23,
        }[source_current]
        assumptions.append(
            f"TPS26600 PWPR has an isolated RTN control return and a {ilim[0]} ILIM "
            f"resistor (nominal {nominal_current_limit:.3g} A; this is fault limiting, "
            f"not a continuous-load rating). The 1%-resistor/TI-threshold bounds are "
            f"UVLO {divider_bounds[0]:.3f}--{divider_bounds[1]:.3f} V and OVP "
            f"{divider_bounds[2]:.3f}--{divider_bounds[3]:.3f} V. Its 1.5 nF dVdT "
            "capacitor produces about 272 us nominal ramp; 1 uF COUT therefore contributes "
            "about 88 mA inrush at 24 V. Each delivered output is post-eFuse/post-PTC and "
            "has a 10 kOhm 0.1 W discharge path (24 V: 57.6 mW, tau about 10 ms per 1 uF)."
        )
        assumptions.append(
            "The reset button spans TPS26600 SHDN and isolated RTN, not the raw input: "
            "TI specifies SHDN output 2--3.4 V at 0.1 uA and a -10 uA minimum leakage "
            "entry at 0.4 V, not a guaranteed 10 uA contact current. KMR232NGULCLFS "
            "ULC is specified at 1 uA/1.8 V; contact-life qualification below that "
            "test current remains unverified."
        )
    if high_voltage_reset and rgb_status:
        # FLT is a 0--60 V active-low open drain (TI SLVSDG2G table 7.5);
        # the red LED is current-limited directly to it. This does not join
        # external system GND to the isolated RTN control return.
        rgb_led = reviewed_part("XL-1615RGBC-RF")
        assert rgb_led is not None
        rgb_series = max(1, math.ceil(input_voltage / 20.0))
        rgb_base_series = max(1, math.ceil((input_voltage - 0.7) / 5.0))
        groups.extend(
            (
                LoweringGroup(
                    role="efuse_rgb_status_led",
                    reference_prefix="D",
                    value="XL-1615RGBC-RF",
                    symbol=rgb_led.symbol,
                    footprint=rgb_led.footprint,
                    mpn="XL-1615RGBC-RF",
                    datasheet=rgb_led.manufacturer_sources[0],
                ),
                LoweringGroup(
                    role="efuse_rgb_current_resistor",
                    reference_prefix="R",
                    quantity=3 * rgb_series,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="0603WAF1002T5E",
                ),
                LoweringGroup(
                    role="efuse_rgb_driver",
                    reference_prefix="Q",
                    quantity=2,
                    value="MMBT3904",
                    symbol="Transistor_BJT:MMBT3904",
                    footprint="Package_TO_SOT_SMD:SOT-23",
                    mpn="MMBT3904",
                ),
                LoweringGroup(
                    role="efuse_rgb_base_resistor",
                    reference_prefix="R",
                    quantity=2 * rgb_base_series,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="0603WAF1002T5E",
                ),
                LoweringGroup(
                    role="efuse_rgb_base_pulldown",
                    reference_prefix="R",
                    quantity=2,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="0603WAF1002T5E",
                ),
            )
        )
        rgb_cathodes = ("__KICRAFT_EFUSE_RGB_R", "__KICRAFT_EFUSE_RGB_G", "__KICRAFT_EFUSE_RGB_B")
        for color, endpoint in enumerate(("__KICRAFT_EFUSE_FLT", "__KICRAFT_EFUSE_RGB_Q0", "__KICRAFT_EFUSE_RGB_Q1")):
            for index in range(rgb_series):
                group_index = color * rgb_series + index
                pins.extend((
                    LoweringPin(role="efuse_rgb_current_resistor", index=group_index, pin="1", net=rgb_cathodes[color] if index == 0 else f"__KICRAFT_EFUSE_RGB_LED_{color}_{index}"),
                    LoweringPin(role="efuse_rgb_current_resistor", index=group_index, pin="2", net=endpoint if index == rgb_series - 1 else f"__KICRAFT_EFUSE_RGB_LED_{color}_{index + 1}"),
                ))
        for driver, drive_net in enumerate((indexed_outputs[1], requirement.ports["input"])):
            base = f"__KICRAFT_EFUSE_RGB_B{driver}"
            for index in range(rgb_base_series):
                group_index = driver * rgb_base_series + index
                pins.extend((
                    LoweringPin(role="efuse_rgb_base_resistor", index=group_index, pin="1", net=drive_net if index == 0 else f"__KICRAFT_EFUSE_RGB_BASE_{driver}_{index}"),
                    LoweringPin(role="efuse_rgb_base_resistor", index=group_index, pin="2", net=base if index == rgb_base_series - 1 else f"__KICRAFT_EFUSE_RGB_BASE_{driver}_{index + 1}"),
                ))
            pins.extend((
                LoweringPin(role="efuse_rgb_driver", index=driver, pin="1", net=base),
                LoweringPin(role="efuse_rgb_driver", index=driver, pin="2", net=requirement.ports["gnd"]),
                LoweringPin(role="efuse_rgb_driver", index=driver, pin="3", net=f"__KICRAFT_EFUSE_RGB_Q{driver}"),
                LoweringPin(role="efuse_rgb_base_pulldown", index=driver, pin="1", net=base),
                LoweringPin(role="efuse_rgb_base_pulldown", index=driver, pin="2", net=requirement.ports["gnd"]),
            ))
        pins.extend((
            LoweringPin(role="global_efuse", pin="14", net="__KICRAFT_EFUSE_FLT"),
            LoweringPin(role="efuse_rgb_status_led", pin=rgb_led.port_pins["common_anode"], net=requirement.ports["input"]),
            *(LoweringPin(role="efuse_rgb_status_led", pin=rgb_led.port_pins[f"{color}_cathode"], net=rgb_cathodes[index])
              for index, color in enumerate(("red", "green", "blue"))),
        ))
        assumptions.append(
            f"eFuse RGB channels use {rgb_series}x10 kOhm current resistors; at "
            f"{input_voltage:g} V their zero-drop 1%-resistor current bound is "
            f"{input_voltage / (rgb_series * 9900):.3g} A. The red channel is sunk by FLT; "
            "green and blue use separate low-side drivers. Light output at the actual current "
            "is unmeasured; XINGLIGHT optical and forward-voltage specifications are at 20 mA."
        )
    if enable_jumper and not reset_button:
        source_bus = "__KICRAFT_DISTRIBUTION__SWITCHED"
        groups.extend(
            (
                LoweringGroup(
                    role="global_switch",
                    reference_prefix="Q",
                    value="AONR21357",
                    symbol="aonr21357:AONR21357",
                    footprint="aonr21357:DFN-8_L3.0-W3.0-P0.65-BL",
                    mpn="AONR21357",
                    datasheet="https://www.lcsc.com/datasheet/C431196.pdf",
                ),
                LoweringGroup(
                    role="gate_pullup",
                    reference_prefix="R",
                    quantity=6,
                    value="10k",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="FRC0603F1002TS",
                    datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8763490791224557568",
                ),
                LoweringGroup(
                    role="gate_drive",
                    reference_prefix="R",
                    value="5.9k",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="FRC0603F5901TS",
                    datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8763495553592778752",
                ),
                LoweringGroup(
                    role="gate_clamp",
                    reference_prefix="D",
                    value="BZT52C10",
                    symbol="Device:D_Zener",
                    footprint="Diode_SMD:D_SOD-123",
                    mpn="BZT52C10",
                    datasheet="https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8756679576573890560",
                ),
                LoweringGroup(
                    role="enable_jumper",
                    reference_prefix="J",
                    value="TSW-102-07-G-S",
                    symbol="Connector_Generic:Conn_01x02",
                    footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                    mpn="TSW-102-07-G-S",
                    sourcing_note=(
                        "Fit Samtec SNT-100-BK-G shunt for enabled output; remove for "
                        "the deterministic PMOS-off state."
                    ),
                ),
            )
        )
        pins.extend(
            [
                *(LoweringPin(role="global_switch", pin=pin, net=protected_source_bus)
                  for pin in ("1", "2", "3")),
                *(LoweringPin(role="global_switch", pin=pin, net=source_bus)
                  for pin in ("5", "6", "7", "8", "9")),
                LoweringPin(role="global_switch", pin="4", net="__KICRAFT_DISTRIBUTION__GATE"),
                LoweringPin(role="gate_clamp", pin="1", net=protected_source_bus),
                LoweringPin(role="gate_clamp", pin="2", net="__KICRAFT_DISTRIBUTION__GATE"),
                LoweringPin(role="gate_drive", pin="1", net="__KICRAFT_DISTRIBUTION__GATE"),
                LoweringPin(role="gate_drive", pin="2", net="__KICRAFT_DISTRIBUTION__ENABLE"),
                LoweringPin(role="enable_jumper", pin="1", net="__KICRAFT_DISTRIBUTION__ENABLE"),
                LoweringPin(role="enable_jumper", pin="2", net=requirement.ports["gnd"]),
            ]
        )
        for index in range(6):
            pins.extend(
                (
                    LoweringPin(
                        role="gate_pullup",
                        index=index,
                        pin="1",
                        net=(
                            protected_source_bus
                            if index == 0
                            else f"__KICRAFT_DISTRIBUTION__PULLUP_{index}"
                        ),
                    ),
                    LoweringPin(
                        role="gate_pullup",
                        index=index,
                        pin="2",
                        net=(
                            "__KICRAFT_DISTRIBUTION__GATE"
                            if index == 5
                            else f"__KICRAFT_DISTRIBUTION__PULLUP_{index + 1}"
                        ),
                    ),
                )
            )
        assumptions.append(
            "Enable jumper fitted: gate is pulled low through 5.9 kOhm and the PMOS delivers "
            "power; removed: six series 10 kOhm pull-up resistors hold gate at source and the "
            "output is off. At 5 V they set VGS≈-4.55 V; at 24 V BZT52C10 clamps |VGS| near "
            "10 V, with about 2.4 mA gate-drive current (about 24 mW in the clamp and 33 mW in "
            "the 5.9 kOhm resistor), all below reviewed ratings."
        )
    if rgb_status and not high_voltage_reset:
        # These are discrete status channels, not an addressable LED pretending
        # to be useful without a controller. Blue proves protected input power,
        # green is sampled after the first branch PTC, and a separate output
        # sense inverter lights red only when that output is absent. The PMOS
        # gate is never loaded by status circuitry.
        rgb_led = reviewed_part("XL-1615RGBC-RF")
        assert rgb_led is not None
        rgb_current_series = max(1, math.ceil(input_voltage / 20.0))
        rgb_base_series = max(1, math.ceil((input_voltage - 0.7) / 5.0))
        groups.extend(
            (
                LoweringGroup(
                    role="rgb_status_led",
                    reference_prefix="D",
                    value="XL-1615RGBC-RF",
                    symbol=rgb_led.symbol,
                    footprint=rgb_led.footprint,
                    mpn="XL-1615RGBC-RF",
                    datasheet=rgb_led.manufacturer_sources[0],
                ),
                LoweringGroup(
                    role="rgb_channel_resistor",
                    reference_prefix="R",
                    quantity=3 * rgb_current_series,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="FRC0603F1002TS",
                ),
                LoweringGroup(
                    role="rgb_driver",
                    reference_prefix="Q",
                    quantity=3,
                    value="MMBT3904",
                    symbol="Transistor_BJT:MMBT3904",
                    footprint="Package_TO_SOT_SMD:SOT-23",
                    mpn="MMBT3904",
                ),
                LoweringGroup(
                    role="rgb_base_resistor",
                    reference_prefix="R",
                    quantity=3 * rgb_base_series,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="FRC0603F1002TS",
                ),
                LoweringGroup(
                    role="rgb_base_pulldown",
                    reference_prefix="R",
                    quantity=3,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="FRC0603F1002TS",
                ),
                LoweringGroup(
                    role="rgb_output_sense",
                    reference_prefix="Q",
                    value="MMBT3904",
                    symbol="Transistor_BJT:MMBT3904",
                    footprint="Package_TO_SOT_SMD:SOT-23",
                    mpn="MMBT3904",
                ),
                LoweringGroup(
                    role="rgb_sense_base_resistor",
                    reference_prefix="R",
                    quantity=rgb_base_series,
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="FRC0603F1002TS",
                ),
                LoweringGroup(
                    role="rgb_sense_base_pulldown",
                    reference_prefix="R",
                    value="10k 1%",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric",
                    mpn="FRC0603F1002TS",
                ),
            )
        )
        rgb_cathodes = ("__KICRAFT_DISTRIBUTION__RGB_RED_K", "__KICRAFT_DISTRIBUTION__RGB_GREEN_K", "__KICRAFT_DISTRIBUTION__RGB_BLUE_K")
        rgb_sources = (
            protected_source_bus,
            indexed_outputs[1],
            protected_source_bus,
        )
        for channel, (cathode, drive_source) in enumerate(zip(rgb_cathodes, rgb_sources, strict=True)):
            collector = f"__KICRAFT_DISTRIBUTION__RGB_COLLECTOR_{channel}"
            base = f"__KICRAFT_DISTRIBUTION__RGB_BASE_{channel}"
            for resistor_index in range(rgb_current_series):
                group_index = channel * rgb_current_series + resistor_index
                pins.extend(
                    (
                        LoweringPin(
                            role="rgb_channel_resistor",
                            index=group_index,
                            pin="1",
                            net=(
                                cathode
                                if resistor_index == 0
                                else f"__KICRAFT_DISTRIBUTION__RGB_LED_{channel}_{resistor_index}"
                            ),
                        ),
                        LoweringPin(
                            role="rgb_channel_resistor",
                            index=group_index,
                            pin="2",
                            net=(
                                collector
                                if resistor_index == rgb_current_series - 1
                                else f"__KICRAFT_DISTRIBUTION__RGB_LED_{channel}_{resistor_index + 1}"
                            ),
                        ),
                    )
                )
            for resistor_index in range(rgb_base_series):
                group_index = channel * rgb_base_series + resistor_index
                pins.extend(
                    (
                        LoweringPin(
                            role="rgb_base_resistor",
                            index=group_index,
                            pin="1",
                            net=(
                                drive_source
                                if resistor_index == 0
                                else f"__KICRAFT_DISTRIBUTION__RGB_BASE_R_{channel}_{resistor_index}"
                            ),
                        ),
                        LoweringPin(
                            role="rgb_base_resistor",
                            index=group_index,
                            pin="2",
                            net=(
                                base
                                if resistor_index == rgb_base_series - 1
                                else f"__KICRAFT_DISTRIBUTION__RGB_BASE_R_{channel}_{resistor_index + 1}"
                            ),
                        ),
                    )
                )
            pins.extend(
                (
                    LoweringPin(role="rgb_driver", index=channel, pin="1", net=base),
                    LoweringPin(role="rgb_driver", index=channel, pin="2", net=requirement.ports["gnd"]),
                    LoweringPin(role="rgb_driver", index=channel, pin="3", net=collector),
                    LoweringPin(role="rgb_base_pulldown", index=channel, pin="1", net=base),
                    LoweringPin(role="rgb_base_pulldown", index=channel, pin="2", net=requirement.ports["gnd"]),
                )
            )
        sense_base = "__KICRAFT_DISTRIBUTION__RGB_SENSE_BASE"
        red_base = "__KICRAFT_DISTRIBUTION__RGB_BASE_0"
        for resistor_index in range(rgb_base_series):
            pins.extend(
                (
                    LoweringPin(
                        role="rgb_sense_base_resistor",
                        index=resistor_index,
                        pin="1",
                        net=(
                            indexed_outputs[1]
                            if resistor_index == 0
                            else f"__KICRAFT_DISTRIBUTION__RGB_SENSE_R_{resistor_index}"
                        ),
                    ),
                    LoweringPin(
                        role="rgb_sense_base_resistor",
                        index=resistor_index,
                        pin="2",
                        net=(
                            sense_base
                            if resistor_index == rgb_base_series - 1
                            else f"__KICRAFT_DISTRIBUTION__RGB_SENSE_R_{resistor_index + 1}"
                        ),
                    ),
                )
            )
        pins.extend(
            (
                LoweringPin(role="rgb_output_sense", pin="1", net=sense_base),
                LoweringPin(role="rgb_output_sense", pin="2", net=requirement.ports["gnd"]),
                LoweringPin(role="rgb_output_sense", pin="3", net=red_base),
                LoweringPin(role="rgb_sense_base_pulldown", pin="1", net=sense_base),
                LoweringPin(role="rgb_sense_base_pulldown", pin="2", net=requirement.ports["gnd"]),
            )
        )
        pins.extend(
            (
                LoweringPin(role="rgb_status_led", pin=rgb_led.port_pins["common_anode"], net=protected_source_bus),
                *(LoweringPin(role="rgb_status_led", pin=rgb_led.port_pins[f"{color}_cathode"], net=rgb_cathodes[index])
                  for index, color in enumerate(("red", "green", "blue"))),
            )
        )
        assumptions.append(
            f"Discrete RGB status: blue is protected-input present, green is post-PTC output1 "
            f"present, and red is output1-absent through a separate MMBT3904 inverter. "
            f"{rgb_current_series}x10 kOhm per LED channel bounds current at 24 V to "
            f"{24.0 / (rgb_current_series * 9900):.3g} A with zero LED and switch drop; "
            f"{rgb_base_series}x10 kOhm base chains limit sensing and driver base current. "
            "Light output at the actual current is unmeasured.",
        )
    for index in range(channels):
        output = indexed_outputs[index + 1]
        if reset_button and not high_voltage_reset:
            fused = f"__KICRAFT_DISTRIBUTION__FUSED_{index + 1}"
            ilim_mid = f"__KICRAFT_DISTRIBUTION__ILIM_{index + 1}"
            pins.extend(
                (
                    LoweringPin(role="branch_fuse", index=index, pin="1", net=protected_source_bus),
                    LoweringPin(role="branch_fuse", index=index, pin="2", net=fused),
                    LoweringPin(role="branch_switch", index=index, pin="1", net=fused),
                    LoweringPin(role="branch_switch", index=index, pin="2", net=requirement.ports["gnd"]),
                    LoweringPin(role="branch_switch", index=index, pin="3", net="__KICRAFT_DISTRIBUTION__ENABLE"),
                    LoweringPin(role="branch_switch", index=index, pin="5", net=ilim_mid),
                    LoweringPin(role="branch_switch", index=index, pin="6", net=output),
                    LoweringPin(role="switch_ilim_top", index=index, pin="1", net=ilim_mid),
                    LoweringPin(role="switch_ilim_top", index=index, pin="2", net=f"{ilim_mid}_R"),
                    LoweringPin(role="switch_ilim_bottom", index=index, pin="1", net=f"{ilim_mid}_R"),
                    LoweringPin(role="switch_ilim_bottom", index=index, pin="2", net=requirement.ports["gnd"]),
                    LoweringPin(role="switch_input_decoupling", index=index, pin="1", net=fused),
                    LoweringPin(role="switch_input_decoupling", index=index, pin="2", net=requirement.ports["gnd"]),
                    LoweringPin(role="output_discharge", index=index, pin="1", net=output),
                    LoweringPin(role="output_discharge", index=index, pin="2", net=requirement.ports["gnd"]),
                )
            )
            no_connects.append(
                LoweringNoConnect(
                    role="branch_switch", index=index, pin="4",
                    reason="TPS2553 FAULT is intentionally unused on this standalone power-cycle board",
                )
            )
        else:
            pins.extend(
                (
                    LoweringPin(role="branch_fuse", index=index, pin="1", net=source_bus),
                    LoweringPin(role="branch_fuse", index=index, pin="2", net=output),
                )
            )
    for index in range(resistor_count):
        pins.extend(
            (
                LoweringPin(
                    role="indicator_resistor",
                    index=index,
                    pin="1",
                    net=source_bus if index == 0 else f"__KICRAFT_DISTRIBUTION__LED_R_{index}",
                ),
                LoweringPin(
                    role="indicator_resistor",
                    index=index,
                    pin="2",
                    net=(
                        "__KICRAFT_DISTRIBUTION__LED_A"
                        if index == resistor_count - 1
                        else f"__KICRAFT_DISTRIBUTION__LED_R_{index + 1}"
                    ),
                ),
            )
        )
    pins.extend(
        (
            LoweringPin(role="indicator_led", pin="2", net="__KICRAFT_DISTRIBUTION__LED_A"),
            LoweringPin(role="indicator_led", pin="1", net=requirement.ports["gnd"]),
        )
    )
    # Internal nodes belong to this requirement, not every distribution on a sheet.
    for pin in pins:
        for prefix in ("__KICRAFT_DISTRIBUTION__", "__KICRAFT_EFUSE_"):
            if pin.net.startswith(prefix):
                pin.net = f"{prefix}{requirement.id}__{pin.net[len(prefix):]}"
                break
    return _artifact(
        "fused-power-distribution@1",
        requirement,
        tuple(groups),
        tuple(pins),
        no_connects=tuple(no_connects),
        assumptions=tuple(assumptions),
        calculations=(
            LoweringCalculation(
                output="power_indicator_current",
                equation="I=(Vin-Vf)/(N*1k)",
                inputs={
                    "input_voltage_v": input_voltage,
                    "led_forward_voltage_v": 2.0,
                    "series_resistors": resistor_count,
                },
                chosen_value=f"{(input_voltage - 2.0) / (resistor_count * 1_000):.3g}A",
                tolerance="LTST-C190KGKT nominal 2 V forward voltage; E24-equivalent 1 kOhm series chain",
            ),
        ),
    )
def _tmp102_temperature(requirement: CircuitRequirement) -> LoweringArtifact | None:
    """TMP102AIDRLR with ADD0 grounded, bypass and independent I2C pullups."""
    ports = requirement.ports
    if (requirement.exact_part and requirement.exact_part.casefold() != "tmp102aidrlr") or (
        set(ports) != {"vdd", "gnd", "sda", "scl"}
    ):
        return None
    if any(not value or value == "NC" for value in ports.values()) or len(set(ports.values())) != 4:
        return None
    groups = (
        LoweringGroup(role="sensor", reference_prefix="U", value="TMP102AIDRLR",
                      symbol="Sensor_Temperature:TMP102xxDRL", footprint="tmp102aidrlr:TI_DRL0006A_TMP102",
                      mpn="TMP102AIDRLR", datasheet="https://www.ti.com/lit/ds/symlink/tmp102.pdf",
                      sourcing_note="LCSC C99269"),
        _passive("bypass", "C", "100nF"),
        _passive("sda_pullup", "R", "4.7k"),
        _passive("scl_pullup", "R", "4.7k"),
    )
    pins = (
        LoweringPin(role="sensor", pin="1", net=ports["scl"]),
        LoweringPin(role="sensor", pin="2", net=ports["gnd"]),
        LoweringPin(role="sensor", pin="4", net=ports["gnd"]),
        LoweringPin(role="sensor", pin="5", net=ports["vdd"]),
        LoweringPin(role="sensor", pin="6", net=ports["sda"]),
        LoweringPin(role="bypass", pin="1", net=ports["vdd"]),
        LoweringPin(role="bypass", pin="2", net=ports["gnd"]),
        LoweringPin(role="sda_pullup", pin="1", net=ports["vdd"]),
        LoweringPin(role="sda_pullup", pin="2", net=ports["sda"]),
        LoweringPin(role="scl_pullup", pin="1", net=ports["vdd"]),
        LoweringPin(role="scl_pullup", pin="2", net=ports["scl"]),
    )
    return _artifact(
        "tmp102-i2c-temperature@1", requirement, groups, pins,
        no_connects=(LoweringNoConnect(role="sensor", pin="3", reason="ALERT unused"),),
        local_bypasses=(LoweringLocalBypass(
            owner_role="sensor", bypass_role="bypass",
            power_pins=("5", "1"), ground_pins=("2", "2"), max_distance_mm=2.5,
        ),),
        assumptions=("ADD0 grounded selects I2C address 0x48; supply 1.4 to 3.6 V. "
                     "Verify bus capacitance and pull-up rise time for the actual layout.",
                     "TI TMP102 SBOS397I sections 4 and 7.2."),
    )



from kicraft.design import actuator_circuits as _actuator_circuits  # noqa: F401
from kicraft.design import analog_conditioner as _analog_conditioner  # noqa: F401


for _lowerer in (
    RegisteredLowerer(
        "tmp102-i2c-temperature@1",
        frozenset({"tmp102-temperature", "temperature-sensor", "i2c-temperature-sensor"}),
        _tmp102_temperature,
        (),
        ("vdd", "gnd", "sda", "scl"),
        required_port_keys=("vdd", "gnd", "sda", "scl"),
        reviewed_exact_part="TMP102AIDRLR",
    ),
    RegisteredLowerer(
        "usb-c-breakout@1",
        frozenset({"usb-c-breakout"}),
        _usb_c_breakout,
        (),
        (
            "vbus",
            "gnd",
            "shield",
            "usb_dp",
            "usb_dm",
            "cc1",
            "cc2",
            "sbu1",
            "sbu2",
            "tx1p",
            "tx1n",
            "tx2p",
            "tx2n",
            "rx1p",
            "rx1n",
            "rx2p",
            "rx2n",
        ),
        required_port_keys=("vbus", "gnd"),
        reviewed_exact_part="12401610E4#2A",
    ),
    RegisteredLowerer(
        "bnc-connector@1",
        frozenset({"bnc-connector"}),
        _bnc_connector,
        (),
        ("signal", "gnd"),
        required_port_keys=("signal", "gnd"),
        reviewed_exact_part="KH-BNC50-3511",
    ),
    RegisteredLowerer(
        "audio-jack@1",
        frozenset({"audio-jack"}),
        _audio_jack,
        (),
        ("sleeve", "tip", "ring"),
        port_directions=(("sleeve", "bidirectional"), ("tip", "bidirectional"), ("ring", "bidirectional")),
        required_port_keys=("sleeve", "tip", "ring"),
        reference_port_keys=("sleeve",),
        required_exact_part="SJ1-3533NG",
    ),
    RegisteredLowerer(
        "reviewed-de9-connector@1",
        frozenset(_REVIEWED_DE9_FAMILIES),
        _reviewed_de9_connector,
        ("positions", "gender"),
        (
            "<can_h, can_l, gnd: CiA 303-1 CAN contacts>",
            "<pin1..pin9: every physical signal contact, with unused contacts spelled NC>",
        ),
        parameter_choices=(("positions", (9,)), ("gender", ("female",))),
        port_directions=(
            ("can_h", "bidirectional"),
            ("can_l", "bidirectional"),
            ("gnd", "bidirectional"),
        ),
        port_patterns=((r"pin[1-9]", "bidirectional"),),
        reviewed_exact_part="DS1034-09FUNSi44",
    ),
    RegisteredLowerer(
        "pin-header@1",
        frozenset(
            {
                "pin-header",
                "pin_header",
                "generic-header",
                "header",
                "spi-header",
                "spi_header",
                "i2c-header",
                "i2c_header",
            }
        ),
        _pin_header,
        ("rows", "gender"),
        (
            "<pin1..pinN: explicit physical pin numbers; the highest declared contact sets "
            "the physical size, and every undeclared position becomes a no-connect>",
        ),
        parameter_choices=(("rows", (1, 2)), ("gender", ("male", "female"))),
        port_patterns=((r"pin[1-9][0-9]*", "bidirectional"),),
    ),
    RegisteredLowerer(
        "reviewed-jst-xh-connector@1",
        frozenset({"jst-connector", "jst-xh-connector"}),
        _reviewed_jst_xh_connector,
        (),
        ("<pin1..pinN: every real JST XH contact, with a reviewed exact-width carrier>",),
        port_patterns=((r"pin[1-9][0-9]*", "bidirectional"),),
    ),
    RegisteredLowerer(
        "qwiic-i2c-connector@1",
        frozenset({"qwiic-i2c-connector", "qwiic-receptacle", "qwiic-header"}),
        _qwiic_connector,
        (),
        ("ground", "vcc", "sda", "scl"),
        required_port_keys=("ground", "vcc", "sda", "scl"),
        reference_port_keys=("ground",),
        port_directions=(
            ("ground", "bidirectional"), ("vcc", "input"),
            ("sda", "bidirectional"), ("scl", "input"),
        ),
        reviewed_exact_part="SM04B-SRSS-TB(LF)(SN)",
    ),
    RegisteredLowerer(
        "reviewed-connector@1",
        frozenset(_REVIEWED_JACK_FAMILIES),
        _reviewed_connector,
        (),
        ("positive; negative/gnd (the inlet's two wires)",),
        # Every spelling this build accepts is published: an unpublished name is a refusal the
        # draft cannot repair from the message alone, and these families were previously
        # model-owned, so drafts in the corpus spell the inlet both ways.
        port_directions=tuple(
            (name, "input" if name in _JACK_POSITIVE_PORTS else "bidirectional")
            for name in (*_JACK_POSITIVE_PORTS, *_JACK_RETURN_PORTS)
        ),
    ),
    RegisteredLowerer(
        "trim-potentiometer@1",
        frozenset(_REVIEWED_TRIMMER_FAMILIES),
        _reviewed_potentiometer,
        (),
        ("wiper; one or both travel ends (end_a/end_b, end1/end2, contact1/contact3)",
         "ccw/cw or gnd/vcc)"),
        port_directions=tuple(
            (name, "bidirectional")
            for name in (*_TRIMMER_WIPER_PORTS, *_TRIMMER_END_A_PORTS, *_TRIMMER_END_B_PORTS)
        ),
        port_aliases=_TRIMMER_PORT_ALIASES,
        # A trimmer always has its wiper on a net; naming it makes the "declares no ports"
        # refusal say the one thing the draft has to add (live run KC-5UG8UR's `gain_pot`).
        required_port_keys=("wiper",),
    ),
    RegisteredLowerer(
        "screw-terminal@1",
        frozenset({"screw-terminal", "screw_terminal"}),
        _screw_terminal,
        ("rows",),
        ("<contiguous pinN/pN contacts, positive/negative, or positive/common/negative; "
         "generic terminals may use multiple reviewed housings, with undeclared positions unconnected>",),
        parameter_choices=(("rows", (1,)),),
        port_patterns=((r"(?:pin|p)[1-9][0-9]*|positive|common|negative", "bidirectional"),),
    ),
    RegisteredLowerer(
        "fpc-connector@1",
        frozenset({"fpc-connector"}),
        _fpc_connector,
        ("pitch_mm",),
        ("<pin1..pin24: contiguous explicit physical contacts>",),
        port_patterns=((r"pin(?:[1-9]|1[0-9]|2[0-4])", "bidirectional"),),
        reviewed_exact_part="KH-FG0.5-H2.0-24PIN",
    ),
    RegisteredLowerer(
        "test-points@1",
        frozenset({"test-points"}),
        _test_points,
        (),
        ("<one named port per test point>",),
        port_patterns=((r"[a-z][a-z0-9_]*", "bidirectional"),),
    ),
    RegisteredLowerer(
        "connector-bank@1",
        frozenset({"connector-bank", "servo-connector-bank"}),
        _connector_bank,
        ("channels",),
        ("vdd", "gnd", "signal0..signalN"),
        port_patterns=((r"signal(?:[0-9]|[12][0-9]|3[0-2])", "input"),),
    ),
    RegisteredLowerer(
        "r2r-ladder@1",
        frozenset(
            {
                "r2r-ladder",
                "r2r_ladder",
                "resistor-ladder",
                "resistor_ladder",
                "resistor-network",
                "resistor_network",
                # The corpus also names this topology `r-2r-resistor-ladder`; the family key is
                # exact, so the hyphenated and unhyphenated spellings must both be listed or the
                # requirement resolves to no lowerer (`part_identity`'s witness set carries the
                # matching normalised key).
                "r-2r-resistor-ladder",
                "r2r-resistor-ladder",
            }
        ),
        _r2r,
        ("bits", "r_value", "two_r_value", "r_series", "r_shunt", "r"),
        ("bit0..bitN/output/gnd or digital_inputs/analog_output",),
        port_directions=(
            ("output", "output"),
            ("gnd", "bidirectional"),
            ("digital_inputs", "input"),
            ("analog_output", "output"),
        ),
        port_patterns=((r"bit(?:[0-9]|[12][0-9]|3[0-2])", "input"),),
    ),
    RegisteredLowerer(
        "common-anode-rgb-indicator@2",
        frozenset({"rgb-led", "rgb-status-indicator", "common-anode-rgb-indicator"}),
        _common_anode_rgb_indicator,
        ("target_current_ma",),
        ("vdd_5v", "gnd", "red", "green", "blue"),
        required_port_keys=("vdd_5v", "gnd", "red", "green", "blue"),
        required_parameter_keys=("target_current_ma",),
        parameter_choices=(("target_current_ma", (2.0,)),),
        reviewed_exact_parts=("XL-1615RGBC-RF", "APTF1616LSEEZGKQBKC"),
        parameter_constraints=(
            "This circuit uses a 5 V common anode and separate 3.3 V MCU GPIO controls "
            "for red, green and blue. Bind all five public ports; do not connect the common "
            "anode to 3.3 V or to an MCU GPIO.",
            "The supported nominal design target is 2.0 mA per channel; actual current and "
            "light output are not guaranteed. Each channel has its own low-side MMBT3904, "
            "series resistor and base-emitter pull-down; no external MOSFET drive is assumed.",
        ),
        port_directions=(
            ("vdd_5v", "input"), ("gnd", "bidirectional"),
            ("red", "input"), ("green", "input"), ("blue", "input"),
        ),
    ),
    RegisteredLowerer(
        "led-current-resistor@1",
        frozenset({"led-current-resistor", "status-led", *_REVIEWED_INDICATOR_LED_FAMILIES}),
        _led_resistor,
        ("rail_voltage", "led_vf", "target_current_ma", "color"),
        ("drive", "gnd"),
        required_port_keys=("drive", "gnd"),
        required_parameter_keys=_LED_PARAMETER_KEYS,
    ),
    RegisteredLowerer(
        "voltage-divider@1",
        frozenset({"voltage-divider"}),
        _voltage_divider,
        ("input_voltage", "target_voltage", "bottom_resistance_ohm", "max_error_percent"),
        ("input", "output", "gnd"),
        required_port_keys=("input", "output", "gnd"),
    ),
    RegisteredLowerer(
        "rc-lowpass@1",
        frozenset({"rc-lowpass", "rc_filter"}),
        _rc_filter,
        ("cutoff_hz", "resistance_ohm", "max_error_percent"),
        ("input", "output", "gnd"),
        required_port_keys=("input", "output", "gnd"),
    ),
    RegisteredLowerer(
        "adjustable-rc-lowpass@1",
        frozenset({"adjustable-rc-lowpass"}),
        _adjustable_rc_filter,
        ("capacitance_f", "capacitor_exact_part"),
        ("input", "output", "gnd"),
        required_port_keys=("input", "output", "gnd"),
        required_parameter_keys=("capacitance_f", "capacitor_exact_part"),
        parameter_choices=(
            ("capacitance_f", (10e-9,)),
            ("capacitor_exact_part", ("C0805C103J5GACTU",)),
        ),
        reviewed_exact_part="3296W-1-103LF",
    ),
    RegisteredLowerer(
        "i2c-pullups@1",
        frozenset({"i2c-pullups"}),
        _i2c_pullups,
        ("speed_hz", "bus_capacitance_pf", "voltage"),
        ("vdd", "sda", "scl"),
        required_port_keys=("vdd", "sda", "scl"),
    ),
    RegisteredLowerer(
        "open-drain-pullup@1",
        frozenset({"open-drain-pullup"}),
        _pullup,
        ("resistance",),
        ("vdd", "signal"),
        required_port_keys=("vdd", "signal"),
    ),
    RegisteredLowerer(
        "switch-input@1",
        frozenset({"switch-input"}),
        _switch_input,
        ("pull_policy", "resistance", "active_level", "voltage"),
        ("signal", "gnd", "vdd"),
        required_port_keys=("signal", "gnd", "vdd"),
        port_directions=(("signal", "bidirectional"),),
        parameter_choices=(
            ("pull_policy", ("internal", "external")),
            ("active_level", ("low", "high")),
        ),
        reviewed_exact_parts=("KMR232NGULCLFS", "TL3342F260QG"),
        parameter_constraints=(
            "An unspecified pull policy uses an external pull resistor; explicit internal bias remains internal.",
            "When known, voltage is the switch supply in volts and must not exceed the selected contact rating.",
        ),
    ),
    RegisteredLowerer(
        "voltage-selector-switch@1",
        frozenset({"voltage_selector_switch"}),
        _voltage_selector_switch,
        ("positions",),
        ("common", "throw_1", "throw_2", "throw_3", "gnd"),
        required_port_keys=("common", "gnd"),
        parameter_choices=(("positions", (3,)),),
        required_parameter_keys=("positions",),
        port_directions=tuple(
            (name, "bidirectional")
            for name in ("common", "throw_1", "throw_2", "throw_3")
        ),
    ),
    RegisteredLowerer(
        "coin-cell-holder@1",
        frozenset({"coin-cell-holder"}),
        _coin_cell_holder,
        ("cell_format",),
        ("positive", "negative"),
        required_port_keys=("positive", "negative"),
        reference_port_keys=("negative",),
        reviewed_exact_part="BS-07-A1BJ001",
    ),
    RegisteredLowerer(
        "capacitive-touch-pad@1",
        frozenset({"capacitive-touch-pad"}),
        _capacitive_touch_pad,
        ("count", "pins", "no_copper_underlay"),
        ("<touch1..touchN: contiguous electrically distinct electrodes>",),
        required_parameter_keys=("count", "pins", "no_copper_underlay"),
        port_patterns=((r"touch[1-9][0-9]*", "bidirectional"),),
    ),
    # The one lowerer whose requirement declares no ports: a pad field owns no
    # contact, so its contract is the field's dimensions alone.
    RegisteredLowerer(
        "prototyping-area@1",
        frozenset({_PROTOTYPING_AREA_FAMILY}),
        _prototyping_area,
        ("rows", "cols", "pitch_mm"),
    ),
    RegisteredLowerer(
        "analog-power-cycle-reset@1",
        frozenset({"analog-power-cycle-reset", "analog_power_cycle_reset"}),
        _analog_power_cycle_reset,
        ("rail_voltage_v", "source_current_a"),
        ("input", "output", "gnd", "isolation_enable"),
        required_port_keys=("input", "output", "gnd"),
        required_parameter_keys=("rail_voltage_v", "source_current_a"),
        parameter_choices=(("rail_voltage_v", (5.0,)), ("source_current_a", (0.5,))),
        parameter_constraints=(
            "Input is an already-regulated 5 V rail, output is the sole switched "
            "analog-load rail, and gnd is the common return. Never connect TPS22948 input to "
            "24 V; lower the source first.",
            "The connected analog load must not exceed 95 mA and total output-to-return "
            "capacitance must not exceed 1 uF. The 1 kOhm bleed consumes the remaining 5 mA "
            "of the 100 mA switched-load budget. If `isolation_enable` is published, it is "
            "the TPS3808 RESET output: it remains low while the output is below the selected "
            "threshold and for the documented 12--28 ms CT-open delay after recovery.",
        ),
        port_directions=(("input", "input"), ("output", "output"), ("gnd", "bidirectional"), ("isolation_enable", "output")),
    ),
    RegisteredLowerer(
        "fused-power-distribution@1",
        frozenset(
            {
                "fused-power-distribution",
                "fused_power_distribution",
                "power-distribution",
                "power_distribution",
            }
        ),
        _fused_power_distribution,
        (
            "channels",
            "input_voltage_v",
            "source_current_a",
            "enable_jumper",
            "reset_button",
            "rgb_status",
        ),
        (
            "input; gnd; output1..outputN (N is channels, each output is post-fuse)",
        ),
        required_port_keys=("input", "gnd"),
        required_parameter_keys=("channels", "input_voltage_v", "source_current_a"),
        parameter_choices=(
            ("channels", (2, 4, 8)),
            ("input_voltage_v", (5.0, 6.0, 7.4, 8.4, 12.0, 18.0, 24.0)),
            ("source_current_a", (0.5, 1.0, 2.0)),
            ("enable_jumper", (False, True)),
            ("reset_button", (False, True)),
            ("rgb_status", (False, True)),
        ),
        parameter_constraints=(
            "source_current_a must be 0.5 A unless reset_button=true and input_voltage_v>5. "
            "That high-voltage reset configuration also supports 1 or 2 A, but no more than "
            "channels*0.5 A. These are supported circuit limits, not permission to reduce "
            "an explicit current requirement.",
            "Bind exactly input, gnd and output1 through outputN; N=channels. "
            "All output nets must be distinct from each other, input and gnd. "
            "The enable jumper and reset button are owned controls, not additional public ports.",
            "This circuit owns the fuses, power indicator and enabled optional controls, "
            "but does not build output connectors. Preserve the requested connector types "
            "and quantities in separate requirements. For two-contact screw-terminal outputs, "
            "use one screw-terminal requirement per output: positive=outputN net, negative=gnd. "
            "Do not number contacts across separate connectors or omit their return contacts.",
        ),
        port_directions=(("input", "input"), ("gnd", "bidirectional")),
        port_patterns=((r"output[1-8]", "output"),),
    ),
    RegisteredLowerer(
        "dual-source-battery-power@1",
        frozenset({"dual-source-battery-power"}),
        _dual_source_battery_power,
        (
            "host_voltage_v",
            "host_voltage_min_v",
            "host_voltage_max_v",
            "battery_chemistry",
            "battery_voltage_min_v",
            "battery_voltage_max_v",
            "load_current_a",
            "include_host_connector",
        ),
        ("host_input", "battery_input", "output", "gnd"),
        required_port_keys=("host_input", "battery_input", "output", "gnd"),
        required_parameter_keys=(
            "host_voltage_v",
            "host_voltage_min_v",
            "host_voltage_max_v",
            "battery_chemistry",
            "battery_voltage_min_v",
            "battery_voltage_max_v",
            "load_current_a",
            "include_host_connector",
        ),
        parameter_choices=(
            ("host_voltage_v", (5.0,)),
            ("host_voltage_min_v", (4.75,)),
            ("host_voltage_max_v", (5.25,)),
            ("battery_chemistry", ("single-cell-liion",)),
            ("battery_voltage_min_v", (3.0,)),
            ("battery_voltage_max_v", (4.2,)),
            ("load_current_a", (0.5,)),
            ("include_host_connector", (True, False)),
        ),
        reviewed_exact_part="TPS2113APWR",
        parameter_constraints=(
            "This is a bounded 0.5 A load supply only: physical host and battery "
            "connectors must bind separately to host_input and battery_input. Never join "
            "those nets outside the TPS2113A.",
            "The host source contract is 5.00 V nominal, 4.75 V to 5.25 V. The "
            "external battery contract is a 3.0 V to 4.2 V single-cell Li-ion source; "
            "this circuit does not charge it.",
        ),
        port_directions=(
            ("host_input", "input"),
            ("battery_input", "input"),
            ("output", "output"),
            ("gnd", "bidirectional"),
        ),
    ),
    RegisteredLowerer(
        "explicit-decoupling@1",
        frozenset({"explicit-decoupling"}),
        _decoupling,
        ("count", "value"),
        ("vdd", "gnd"),
        required_port_keys=("vdd", "gnd"),
    ),
):
    register_lowerer(_lowerer)
