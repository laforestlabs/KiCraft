"""Typed deterministic circuit lowering from ``CircuitRequirement`` objects.

A lowerer emits BOM groups and their wiring from one immutable artifact. Matching
is exact on the requirement family; no prose, regex, or ordered first-match path
participates.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from kicraft.design.models import CircuitRequirement, JsonScalar


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
        "parameters=" + ",".join(lowerer.parameter_keys),
    ]
    if failure:
        evidence.append(failure)
    if not requirement.ports and (lowerer.port_directions or lowerer.port_patterns):
        # The lowerer owns this family, so a model-authored fallback is refused; the
        # requirement has simply declared no contacts. Say that, rather than the generic
        # "cannot realize this port/parameter combination" a model cannot act on.
        message = (
            f"known lowerer {lowerer_id} needs this requirement's ports declared, but it declares "
            "none; bind every contact using its published port contract (see evidence) so the "
            "lowerer can build the part"
        )
    elif unknown_ports:
        message = (
            f"known lowerer {lowerer_id} does not support ports {unknown_ports}; "
            "use its published port contract or choose a genuinely model-owned family"
        )
    elif parameter_diagnostics:
        message = (
            f"known lowerer {lowerer_id} has unsupported parameters; "
            "use its published parameter contract"
        )
    else:
        message = (
            f"known lowerer {lowerer_id} cannot realize this port/parameter combination; "
            "use a published satisfying configuration or choose a genuinely model-owned family"
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


def _connector(
    requirement: CircuitRequirement, *, terminal: bool = False
) -> LoweringArtifact | None:
    if not requirement.ports or len(requirement.ports) > 40:
        return None
    try:
        rows = int(requirement.parameters.get("rows", 1))
    except (TypeError, ValueError):
        return None
    count = len(requirement.ports)
    if count % rows:
        return None
    per_row = count // rows
    mpn: str | None = None
    if terminal:
        if not 2 <= count <= 12:
            return None
        lowerer_id = "screw-terminal@1"
        semantic_order = (
            ("positive", "negative") if count == 2 else ("positive", "common", "negative")
        )
        if set(requirement.ports) == set(semantic_order):
            ordered_nets = tuple(requirement.ports[key] for key in semantic_order)
        else:
            prefix = next(
                (
                    candidate
                    for candidate in ("pin", "p")
                    if set(requirement.ports)
                    == {f"{candidate}{index}" for index in range(1, count + 1)}
                ),
                None,
            )
            if prefix is None:
                return None
            ordered_nets = tuple(
                requirement.ports[f"{prefix}{index}"] for index in range(1, count + 1)
            )
        if requirement.exact_part is not None:
            reviewed = {
                "WJ126V-5.0-02P-14-00A": (
                    2,
                    "screw-terminal-5mm-2p:WJ126V-5.0-2P",
                    "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
                ),
                "WJ126V-5.0-03P-14-00A": (
                    3,
                    "screw-terminal-5mm-3p:WJ126V-5.0-3P",
                    "screw-terminal-5mm-3p:CONN-TH_3P-P5.00_WJ126V-5.0-3P",
                ),
            }
            selected = reviewed.get(requirement.exact_part)
            if selected is None or selected[0] != count:
                return None
            _, symbol, footprint = selected
            value = requirement.exact_part
            mpn = value
        else:
            symbol = f"Connector:Screw_Terminal_01x{count:02d}"
            footprint = (
                f"TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-{count}_1x{count:02d}"
                "_P5.00mm_Horizontal"
            )
            value = f"ScrewTerminal_1x{count:02d}"
    else:
        ordered_nets = tuple(requirement.ports.values())
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
    return _artifact(lowerer_id, requirement, (group,), pins, no_connects=no_connects)


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


def _numbered_connector_ports(requirement: CircuitRequirement) -> dict[str, str] | None:
    # JSON objects have no physical ordering. Only explicit, contiguous pinN
    # bindings establish the contact numbers of a generic header.
    ports = requirement.ports
    if set(ports) != {f"pin{index}" for index in range(1, len(ports) + 1)}:
        return None
    if any(not net.strip() for net in ports.values()):
        return None
    return {f"pin{index}": ports[f"pin{index}"] for index in range(1, len(ports) + 1)}


def _pin_header(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _numbered_connector_ports(requirement)
    if ports is None:
        return None
    return _connector(requirement.model_copy(update={"ports": ports}))


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
    led = _passive("led", "D", str(requirement.parameters.get("color", "LED")))
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
    ),
    RegisteredLowerer(
        "audio-jack@1",
        frozenset({"audio-jack"}),
        _audio_jack,
        (),
        ("sleeve", "tip", "ring"),
        port_directions=(("sleeve", "bidirectional"), ("tip", "bidirectional"), ("ring", "bidirectional")),
        required_port_keys=("sleeve", "tip", "ring"),
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
        ("<pin1..pinN: contiguous explicit physical pin numbers>",),
        parameter_choices=(("rows", (1, 2)), ("gender", ("male", "female"))),
        port_patterns=((r"pin[1-9][0-9]*", "bidirectional"),),
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
        required_parameter_keys=_LED_PARAMETER_KEYS,
    ),
    RegisteredLowerer(
        "voltage-divider@1",
        frozenset({"voltage-divider"}),
        _voltage_divider,
        ("input_voltage", "target_voltage", "bottom_resistance_ohm", "max_error_percent"),
        ("input", "output", "gnd"),
    ),
    RegisteredLowerer(
        "rc-lowpass@1",
        frozenset({"rc-lowpass", "rc_filter"}),
        _rc_filter,
        ("cutoff_hz", "resistance_ohm", "max_error_percent"),
        ("input", "output", "gnd"),
    ),
    RegisteredLowerer(
        "adjustable-rc-lowpass@1",
        frozenset({"adjustable-rc-lowpass"}),
        _adjustable_rc_filter,
        ("capacitance_f", "capacitor_exact_part"),
        ("input", "output", "gnd"),
        required_parameter_keys=("capacitance_f", "capacitor_exact_part"),
    ),
    RegisteredLowerer(
        "i2c-pullups@1",
        frozenset({"i2c-pullups"}),
        _i2c_pullups,
        ("speed_hz", "bus_capacitance_pf", "voltage"),
        ("vdd", "sda", "scl"),
    ),
    RegisteredLowerer(
        "open-drain-pullup@1",
        frozenset({"open-drain-pullup"}),
        _pullup,
        ("resistance",),
        ("vdd", "signal"),
    ),
    RegisteredLowerer(
        "switch-input@1",
        frozenset({"switch-input"}),
        _switch_input,
        ("pull_policy", "resistance", "active_level"),
        ("signal", "gnd", "vdd"),
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
    RegisteredLowerer(
        "explicit-decoupling@1",
        frozenset({"explicit-decoupling"}),
        _decoupling,
        ("count", "value"),
        ("vdd", "gnd"),
    ),
):
    register_lowerer(_lowerer)
