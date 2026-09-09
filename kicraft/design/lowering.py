"""Typed deterministic circuit lowering from ``CircuitRequirement`` objects.

A lowerer emits BOM groups and their wiring from one immutable artifact. Matching
is exact on the requirement family; no prose, regex, or ordered first-match path
participates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from kicraft.design.models import CircuitRequirement


class LoweringGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    reference_prefix: str = Field(pattern=r"^[A-Z]+$")
    quantity: int = Field(default=1, ge=1, le=500)
    value: str
    symbol: str
    footprint: str
    mpn: str | None = None


class LoweringPin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    index: int = Field(default=0, ge=0)
    pin: str
    net: str


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
    assumptions: tuple[str, ...] = ()
    calculations: tuple[LoweringCalculation, ...] = ()


@dataclass(frozen=True)
class RegisteredLowerer:
    lowerer_id: str
    families: frozenset[str]
    build: Callable[[CircuitRequirement], LoweringArtifact | None]
    parameter_keys: tuple[str, ...] = ()
    port_keys: tuple[str, ...] = ()


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
            "parameters": list(lowerer.parameter_keys),
            "ports": list(lowerer.port_keys),
        }
        for lowerer in registered_lowerers()
    ]


def lower_requirement(requirement: CircuitRequirement | dict) -> LoweringArtifact | None:
    requirement = CircuitRequirement.model_validate(requirement)
    lowerer_id = _FAMILIES.get(requirement.family)
    if lowerer_id is None:
        return None
    return _REGISTRY[lowerer_id].build(requirement)


def _artifact(
    lowerer_id: str,
    requirement: CircuitRequirement,
    groups: tuple[LoweringGroup, ...],
    pins: tuple[LoweringPin, ...],
    *,
    assumptions: tuple[str, ...] = (),
    calculations: tuple[LoweringCalculation, ...] = (),
) -> LoweringArtifact:
    roles = {group.role: group.quantity for group in groups}
    seen: set[tuple[str, int, str]] = set()
    for pin in pins:
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
        assumptions=assumptions,
        calculations=calculations,
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
    if rows not in (1, 2) or count % rows:
        return None
    per_row = count // rows
    if terminal and rows != 1:
        return None
    if terminal:
        if not 2 <= count <= 12:
            return None
        symbol = f"Connector:Screw_Terminal_01x{count:02d}"
        footprint = f"TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-{count}_1x{count:02d}_P5.00mm_Horizontal"
        lowerer_id = "screw-terminal@1"
        value = f"ScrewTerminal_1x{count:02d}"
    else:
        symbol = f"Connector_Generic:Conn_{rows:02d}x{per_row:02d}"
        footprint = f"Connector_PinHeader_2.54mm:PinHeader_{rows}x{per_row:02d}_P2.54mm_Vertical"
        lowerer_id = "pin-header@1"
        value = f"PinHeader_{rows}x{per_row:02d}"
    group = LoweringGroup(
        role="connector", reference_prefix="J", value=value, symbol=symbol, footprint=footprint
    )
    pins = tuple(
        LoweringPin(role="connector", pin=str(index), net=net)
        for index, net in enumerate(requirement.ports.values(), 1)
    )
    return _artifact(lowerer_id, requirement, (group,), pins)


def _pin_header(requirement: CircuitRequirement) -> LoweringArtifact | None:
    return _connector(requirement)


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


def _fpc_breakout(requirement: CircuitRequirement) -> LoweringArtifact | None:
    if not requirement.ports or not 10 <= len(requirement.ports) <= 19:
        return None
    count = len(requirement.ports)
    try:
        pitch = float(requirement.parameters.get("pitch_mm", 0.5))
    except (TypeError, ValueError):
        return None
    if pitch not in (0.5, 1.0):
        return None
    series = "1-1734839" if pitch == 0.5 else "1-84952"
    fpc = LoweringGroup(
        role="fpc",
        reference_prefix="J",
        value=f"FPC_{count:02d}",
        symbol=f"Connector_Generic:Conn_01x{count:02d}",
        footprint=(
            "Connector_FFC-FPC:"
            f"TE_{series}-{count - 10}_1x{count:02d}-1MP_P{pitch:.1f}mm_Horizontal"
        ),
    )
    header = LoweringGroup(
        role="header",
        reference_prefix="J",
        value=f"Header_{count:02d}",
        symbol=f"Connector_Generic:Conn_01x{count:02d}",
        footprint=f"Connector_PinHeader_2.54mm:PinHeader_1x{count:02d}_P2.54mm_Vertical",
    )
    pins = tuple(
        pin
        for index, net in enumerate(requirement.ports.values(), 1)
        for pin in (
            LoweringPin(role="fpc", pin=str(index), net=net),
            LoweringPin(role="header", pin=str(index), net=net),
        )
    )
    return _artifact("fpc-header-breakout@1", requirement, (fpc, header), pins)


def _r2r(requirement: CircuitRequirement) -> LoweringArtifact | None:
    try:
        bits = int(requirement.parameters.get("bits", 0))
    except (TypeError, ValueError):
        return None
    r_value = str(requirement.parameters.get("r_value", ""))
    two_r_value = str(requirement.parameters.get("two_r_value", ""))
    names = (*(f"bit{i}" for i in range(bits)), "output", "gnd")
    ports = _require_ports(requirement, names)
    if ports is None or not 2 <= bits <= 32 or not r_value or not two_r_value:
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


def _led_resistor(requirement: CircuitRequirement) -> LoweringArtifact | None:
    ports = _require_ports(requirement, ("drive", "gnd"))
    try:
        rail = float(requirement.parameters["rail_voltage"])
        vf = float(requirement.parameters["led_vf"])
        current_ma = float(requirement.parameters["target_current_ma"])
    except (KeyError, TypeError, ValueError):
        return None
    if ports is None or not (rail > vf > 0 and 1 <= current_ma <= 30):
        return None
    ideal = (rail - vf) / (current_ma / 1000.0)
    value = _standard_value(ideal, _E24, ceiling=True)
    resistor = _passive("resistor", "R", _resistance(value))
    led = _passive("led", "D", str(requirement.parameters.get("color", "LED")))
    node = f"__lowerer__{requirement.id}__led_anode"
    pins = (
        LoweringPin(role="resistor", pin="1", net=ports["drive"]),
        LoweringPin(role="resistor", pin="2", net=node),
        LoweringPin(role="led", pin="1", net=node),
        LoweringPin(role="led", pin="2", net=ports["gnd"]),
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
    if ports is None or policy not in ("internal", "external"):
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
        LoweringPin(role="switch", pin="2", net=ports["gnd"]),
    ]
    if policy == "external":
        groups.append(_passive("pullup", "R", str(requirement.parameters.get("resistance", "10k"))))
        pins.extend(
            (
                LoweringPin(role="pullup", pin="1", net=ports["vdd"]),
                LoweringPin(role="pullup", pin="2", net=ports["signal"]),
            )
        )
    return _artifact("switch-input@1", requirement, tuple(groups), tuple(pins))


def _voltage_selector_switch(
    requirement: CircuitRequirement,
) -> LoweringArtifact | None:
    try:
        positions = int(requirement.parameters.get("positions", 0))
    except (TypeError, ValueError):
        return None
    if positions != 3 or {name.lower() for name in requirement.ports} != {
        "sel0",
        "sel1",
    }:
        return None
    group = LoweringGroup(
        role="voltage_selector_switch",
        reference_prefix="SW",
        value="MSK13C02-SZ",
        symbol="sp3t-switch-msk13c02:MSK13C02-SZ",
        footprint="sp3t-switch-msk13c02:SW-SMD_MSK13C02-SZ",
        mpn="MSK13C02-SZ",
    )
    return _artifact(
        "voltage-selector-switch@1",
        requirement,
        (group,),
        (),
        assumptions=("Three-position selector uses the curated MSK13C02-SZ part.",),
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
        "pin-header@1",
        frozenset({"pin-header", "generic-header"}),
        _pin_header,
        ("rows",),
        ("<one named port per pin>",),
    ),
    RegisteredLowerer(
        "screw-terminal@1",
        frozenset({"screw-terminal"}),
        _screw_terminal,
        (),
        ("<one named port per terminal>",),
    ),
    RegisteredLowerer(
        "fpc-header-breakout@1",
        frozenset({"fpc-header-breakout"}),
        _fpc_breakout,
        ("pitch_mm",),
        ("<one named port per pin>",),
    ),
    RegisteredLowerer(
        "test-points@1",
        frozenset({"test-points"}),
        _test_points,
        (),
        ("<one named port per test point>",),
    ),
    RegisteredLowerer(
        "connector-bank@1",
        frozenset({"connector-bank", "servo-connector-bank"}),
        _connector_bank,
        ("channels",),
        ("vdd", "gnd", "signal0..signalN"),
    ),
    RegisteredLowerer(
        "r2r-ladder@1",
        frozenset({"r2r-ladder"}),
        _r2r,
        ("bits", "r_value", "two_r_value"),
        ("bit0..bitN", "output", "gnd"),
    ),
    RegisteredLowerer(
        "led-current-resistor@1",
        frozenset({"led-current-resistor", "status-led"}),
        _led_resistor,
        ("rail_voltage", "led_vf", "target_current_ma", "color"),
        ("drive", "gnd"),
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
        frozenset({"rc-lowpass"}),
        _rc_filter,
        ("cutoff_hz", "resistance_ohm", "max_error_percent"),
        ("input", "output", "gnd"),
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
        ("pull_policy", "resistance"),
        ("signal", "gnd", "vdd"),
    ),
    RegisteredLowerer(
        "voltage-selector-switch@1",
        frozenset({"voltage_selector_switch"}),
        _voltage_selector_switch,
        ("positions",),
        ("SEL0", "SEL1"),
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
