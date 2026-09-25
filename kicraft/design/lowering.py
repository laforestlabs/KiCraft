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

from kicraft.design.models import CircuitRequirement, JsonScalar
from kicraft.design.part_identity import reviewed_part


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


class LoweringArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lowerer_id: str
    requirement_id: str
    groups: tuple[LoweringGroup, ...]
    pins: tuple[LoweringPin, ...]
    no_connects: tuple[LoweringNoConnect, ...] = ()
    assumptions: tuple[str, ...] = ()
    calculations: tuple[LoweringCalculation, ...] = ()
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
                {"required_exact_part": lowerer.required_exact_part}
                if lowerer.required_exact_part
                else {}
            ),
            **(
                {"reviewed_exact_part": lowerer.reviewed_exact_part}
                if lowerer.reviewed_exact_part
                else {}
            ),
        }
        for lowerer in registered_lowerers()
    ]


def lowerer_port_direction(lowerer: RegisteredLowerer, name: str) -> str | None:
    """Return a published lowerer port direction; never infer one from its name."""

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
        name for name in requirement.ports if lowerer_port_direction(lowerer, name) is None
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
    if failure:
        evidence.append(failure)
    missing_required = sorted(
        key for key in lowerer.required_port_keys if key not in requirement.ports
    )
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
            "this identity/port/parameter combination; model-owned fallback is not permitted"
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
    return LoweringArtifact(
        lowerer_id=lowerer_id,
        requirement_id=requirement.id,
        groups=groups,
        pins=pins,
        no_connects=no_connects,
        assumptions=assumptions,
        calculations=calculations,
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
    if terminal:
        if not 2 <= count <= 12:
            raise ValueError(
                f"a screw terminal block carries 2 to 12 contacts and this requirement declares "
                f"{count}; spell a two-wire terminal as `positive`/`negative`, or an N-position "
                "block as `pin1..pinN`"
            )
        lowerer_id = "screw-terminal@1"
        if requirement.exact_part is not None:
            selected = _reviewed_terminal_part(str(requirement.exact_part))
            if selected is None:
                raise ValueError(
                    f"{requirement.exact_part!r} is not a reviewed screw terminal; name a reviewed "
                    "ordering code or omit `exact_part` for the generic part"
                )
            if selected[0] != count:
                raise ValueError(
                    f"the reviewed terminal {requirement.exact_part!r} carries {selected[0]} "
                    f"contacts and this requirement declares {count}"
                )
            _, mpn, symbol, footprint = selected
            value = mpn
        else:
            symbol = f"Connector:Screw_Terminal_01x{count:02d}"
            footprint = (
                f"TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-{count}_1x{count:02d}"
                "_P5.00mm_Horizontal"
            )
            value = f"ScrewTerminal_1x{count:02d}"
    else:
        gender = str(requirement.parameters.get("gender", "male")).lower()
        if gender not in ("male", "female"):
            return None
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
        value=value,
        symbol=symbol,
        footprint=footprint,
        mpn=mpn,
    )
    pins = tuple(
        LoweringPin(role="connector", pin=str(index), net=net)
        for index, net in enumerate(ordered_nets, 1)
        if net != "NC"
    )
    no_connects = tuple(
        LoweringNoConnect(
            role="connector",
            pin=str(index),
            reason="reserved contact is intentionally unconnected",
        )
        for index, net in enumerate(ordered_nets, 1)
        if net == "NC"
    )
    return _artifact(
        lowerer_id, requirement, (group,), pins, no_connects=no_connects, assumptions=assumptions
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
    use_superspeed = bool(ports.keys() & superspeed.keys()) or (
        requirement.exact_part is not None
        and requirement.exact_part.casefold() == superspeed_mpn.casefold()
    )
    if use_superspeed:
        # Amphenol drawing C12401610 rev C contact table; the installed land
        # pattern uses A1..A12/B1..B12 and four shell pads sharing S1.
        group = LoweringGroup(
            role="connector",
            reference_prefix="J",
            value=superspeed_mpn,
            mpn=superspeed_mpn,
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


def _stated_contact_count(requirement: CircuitRequirement) -> int:
    """The contact count the requirement's own obligations state, or 0.

    A brief's "6-pin 0.1 inch header" reaches the parts step as a quantitative row owned by the
    header requirement ("header pins = 6"). The connector's physical size otherwise comes from the
    contacts the circuit happens to use -- two of six here, so the parts step emitted a 2-way
    header for a brief that demands six, and no gate anywhere said a word (live 2026-09-25). Only
    an equality row about the connector's own contacts counts, and it only ever grows the part.
    """
    for row in requirement.obligations:
        if row.kind != "quantitative" or row.relation != "equal" or row.value is None:
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
    stated = _stated_contact_count(requirement)
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
    chosen = 10000 if max_r >= 10000 else 4700 if max_r >= 4700 else 2200
    group = _passive(
        "pullup", "R", f"{chosen // 1000}k" if chosen >= 1000 else f"{chosen}R", quantity=2
    )
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
    ports = _require_ports(requirement, ("signal", "gnd", "vdd"))
    policy = str(requirement.parameters.get("pull_policy", ""))
    active_level = str(requirement.parameters.get("active_level", "low"))
    if (
        ports is None
        or len(set(ports.values())) != 3
        or any(not net.strip() for net in ports.values())
        or (policy == "internal" and "resistance" in requirement.parameters)
        or requirement.exact_part
    ):
        return None
    groups = [
        LoweringGroup(
            role="switch",
            reference_prefix="SW",
            value="BUTTON",
            symbol="Switch:SW_Push",
            footprint="Button_Switch_SMD:SW_SPST_TL3342",
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
    if not isinstance(pin_names, str):
        return None
    pins = tuple(name.strip() for name in pin_names.split(","))
    if len(pins) != count or len(set(pins)) != count or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in pins
    ):
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


for _lowerer in (
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
        "pin-header@1",
        frozenset(
            {
                "pin-header",
                "pin_header",
                "generic-header",
                "header",
                "spi-header",
                "spi_header",
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
        # A trimmer always has its wiper on a net; naming it makes the "declares no ports"
        # refusal say the one thing the draft has to add (live run KC-5UG8UR's `gain_pot`).
        required_port_keys=("wiper",),
    ),
    RegisteredLowerer(
        "screw-terminal@1",
        frozenset({"screw-terminal", "screw_terminal"}),
        _screw_terminal,
        ("rows",),
        ("<contiguous pinN/pN contacts, positive/negative, or positive/common/negative>",),
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
        "led-current-resistor@1",
        frozenset({"led-current-resistor", "status-led"}),
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
        ("pull_policy", "resistance", "active_level"),
        ("signal", "gnd", "vdd"),
        required_port_keys=("signal", "gnd", "vdd"),
        parameter_choices=(
            ("pull_policy", ("internal", "external")),
            ("active_level", ("low", "high")),
        ),
        required_parameter_keys=("pull_policy",),
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
        "explicit-decoupling@1",
        frozenset({"explicit-decoupling"}),
        _decoupling,
        ("count", "value"),
        ("vdd", "gnd"),
        required_port_keys=("vdd", "gnd"),
    ),
):
    register_lowerer(_lowerer)
