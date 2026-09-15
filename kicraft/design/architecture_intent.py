"""The architecture stage's intent-shaped slot, and its deterministic derivation.

See `docs/plans/architecture-constructive-slot-2026-09-14.md`. The slot used to ask the model for
derived data: canonical net names, `requirements[].ports` values that had to equal those names, an
endpoint list per declared net, rail nets, and the connector exposure a signal needs. Every one of
those is a *consequence* of design intent, so the model now states only the intent and the compiler
derives the rest here:

  sheets[]        physical sheets (the model: there is no other source)
  requirements[]  the part per sheet: family, exact part, parameters, supply rail, and -- only for a
                  part with no curated recipe -- the interface the model claims it has
  signals[]       one entry per named application signal: source port, peer port(s), or `edge:`
  power.rails{}   rail name -> voltage, and the port that generates it

`derive_architecture` produces the canonical `Architecture` the rest of the pipeline already
consumes, shaped like a hand-written valid slot, so no downstream consumer changes. It refuses,
once and by name, only what it cannot derive from: an uncurated part whose interface nobody stated,
a port key that does not exist, a rail no part generates. Everything else -- net names, port
bindings, inter-sheet endpoints and directions, rail nets, board-edge connectors -- is written
once, here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .lowering import RegisteredLowerer, registered_lowerers
from .models import (
    SHEET_NAME_RE,
    SHEET_STEM_RE,
    Architecture,
    CircuitRequirement,
    CircuitRole,
    InterSheetNet,
    Sheet,
    SheetPin,
)
from .recipes.models import RegisteredRecipe
from .recipes.pin_allocator import FIXED_INTERFACES
from .recipes.registry import registered_recipes
from .recipes.resolver import (
    _PORT_ALIASES,
    _family_recipes,
    _net_identity,
    _recipe_for_exact,
)

EDGE_PREFIX = "edge:"
"""Signals whose peer is off the board: `"to": "edge:LED_STRING"`."""

GND_NET = "GND"
_USB_DATA_PORTS = frozenset({"usb_dm", "usb_dp"})
_USB_DEVICE_CONNECTOR_FAMILY = "usb-c-usb2-device"
_USB_DEVICE_CONNECTOR_PART = "USB-C-USB2-DEVICE"
# Reviewed factory-native-programmable families: their USB pair must reach one physical
# data connector, never a UART header.
_NATIVE_USB_MCU_RECIPES = frozenset(
    {
        "esp32-s3-mini-1-minimal@1",
        "esp32-s3-wroom-1-minimal@1",
        "esp32-c3-mini-1-minimal@1",
        "rp2040-minimal@2",
    }
)
# Recipes whose own expansion already carries the 22R MCU-side series pair, so the
# data connector must not add a second one.
_MCU_OWNED_USB_SERIES_RECIPES = frozenset(
    {
        "esp32-s3-mini-1-minimal@1",
        "esp32-s3-wroom-1-minimal@1",
        "esp32-c3-mini-1-minimal@1",
    }
)
_HEADER_FAMILY = "pin-header"
# Port a requirement's `supply` rail feeds, in preference order: a logic part its vdd, a motor
# driver its vm, a regulator its input.
_SUPPLY_PORTS = ("vdd", "vm", "vin", "input", "vdd_5v")
_USB_VBUS_RAIL_NAMES = ("VBUS", "+5V")
_APPLICATION_PREFIXES = ("input", "output", "touch", "parallel", "pwm", "adc", "gpio")
# Capability-prefixed application pins: the recipe does not enumerate them, the allocator does.
_SIGNAL_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("output", "output"),
    ("parallel", "output"),
    ("pwm", "output"),
    ("can_tx", "output"),
    ("mosi", "output"),
    ("sclk", "output"),
    ("cs", "output"),
    ("sda", "bidirectional"),
    ("scl", "bidirectional"),
    ("input", "input"),
    ("touch", "input"),
    ("adc", "input"),
    ("miso", "input"),
    ("can_rx", "input"),
    ("rx", "input"),
)
_SINK_DIRECTION = {
    "output": "input",
    "input": "output",
    "bidirectional": "bidirectional",
    "passive": "passive",
    "power": "bidirectional",
}


class IntentDeclaredPort(BaseModel):
    """One pin of an uncurated part as the model claims it: a recorded claim, not a verified pin."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    direction: Literal["input", "output", "bidirectional", "passive", "power"]
    function: str


class IntentSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    stem: str
    role: str = Field(pattern=r"^[a-z][a-z0-9_]{1,31}$")
    function: str
    from_library: str | None = None
    library_instance: int | None = None
    replication_group: str | None = None
    replication_instance: int | None = None

    @model_validator(mode="after")
    def _shape(self):
        if not SHEET_NAME_RE.match(self.name):
            raise ValueError(
                f"sheet name {self.name!r} must be uppercase letters/digits with spaces"
            )
        if not SHEET_STEM_RE.match(self.stem):
            raise ValueError(
                f"sheet stem {self.stem!r} must be uppercase letters/digits/underscores"
            )
        if (self.from_library is None) != (self.library_instance is None):
            raise ValueError("from_library and library_instance must both be set or both null")
        if (self.replication_group is None) != (self.replication_instance is None):
            raise ValueError(
                "replication_group and replication_instance must both be set or both null"
            )
        return self


class IntentRequirement(BaseModel):
    """The part that implements a block: identity, parameters, supply, and its interface claim."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    sheet: str
    role: CircuitRole
    family: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    exact_part: str | None = None
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    # Rail this part is powered from: the compiler binds it to the recipe's supply port and makes
    # this sheet an endpoint of that rail.
    supply: str | None = None
    # How the part is programmed when it is programmable (`native_usb`, `usb_uart_bridge`, `swd`,
    # `updi`, `bootsel`, `none`). Declared intent the design-level checks read back.
    programming: str | None = None
    interfaces: list[str] = Field(default_factory=list)
    functional_blocks: list[str] = Field(default_factory=list)
    # Port key -> declared net this part ties directly: a connector shell to GND, a spare input
    # held low, an enable pin fed from a rail. Design facts no signal names, still never invented
    # nets: the net must be a declared rail, GND, or a net a signal already names.
    ties: dict[str, str] = Field(default_factory=dict)
    # Only for a part with no curated recipe: the interface the model claims. Recorded as a claim.
    declared_ports: list[IntentDeclaredPort] = Field(default_factory=list)


class IntentSignal(BaseModel):
    """One named application signal: its source port, its peers, and any rails a peer needs."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    from_ref: str = Field(alias="from")
    to: str | list[str]
    # Rails an off-board peer needs on its connector (a 5 V LED string behind 3.3 V logic, say).
    rails: list[str] = Field(default_factory=list)
    # Optional compact range: with start/end set, `{n}` in name/from/to expands over the range.
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _range_complete(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("signal start and end must both be set or both null")
        if self.start is not None:
            if self.end < self.start:
                raise ValueError("signal range end must be >= start")
            if self.end - self.start + 1 > 5000:
                raise ValueError("signal range expansion exceeds 5000 nets")
            for text in (self.name, self.from_ref, *self.peers()):
                if text.count("{n}") != 1:
                    raise ValueError(f"signal range {text!r} must contain exactly one '{{n}}'")
        return self

    def peers(self) -> list[str]:
        return [self.to] if isinstance(self.to, str) else list(self.to)

    def expanded(self) -> list[IntentSignal]:
        if self.start is None:
            return [self]
        return [
            self.model_copy(
                update={
                    "name": self.name.replace("{n}", str(number)),
                    "from_ref": self.from_ref.replace("{n}", str(number)),
                    "to": [peer.replace("{n}", str(number)) for peer in self.peers()],
                    "start": None,
                    "end": None,
                }
            )
            for number in range(self.start, self.end + 1)
        ]


class IntentRail(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    voltage: float
    # `<requirement_id>.<port>` generating the rail, or null: a rail whose source no part claims.
    # The design-level rail-source check reports the latter; the derivation never invents one.
    from_ref: str | None = Field(default=None, alias="from")


class IntentPower(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rails: dict[str, IntentRail] = Field(default_factory=dict)


class ArchitectureIntent(BaseModel):
    """The architecture stage's intent-shaped slot (see the module docstring)."""

    model_config = ConfigDict(extra="forbid")

    topologies: dict[str, str] = Field(default_factory=dict)
    comms_protocols: list[str] = Field(default_factory=list)
    mcu_present: bool = False
    sheets: list[IntentSheet]
    requirements: list[IntentRequirement]
    signals: list[IntentSignal] = Field(default_factory=list)
    power: IntentPower = Field(default_factory=IntentPower)
    assumptions: list[str] = Field(default_factory=list)


class IntentDiagnostic(BaseModel):
    """One blocking refusal: the part, port or peer the derivation cannot derive from."""

    model_config = ConfigDict(extra="forbid")

    code: str
    requirement_id: str | None = None
    sheet: str | None = None
    message: str
    evidence: list[str] = Field(default_factory=list)


class ArchitectureIntentError(ValueError):
    def __init__(self, diagnostics: list[IntentDiagnostic]):
        self.diagnostics = diagnostics
        super().__init__("; ".join(f"{row.code}: {row.message}" for row in diagnostics))


@dataclass(frozen=True)
class _Catalog:
    """Every port key one requirement may bind, and where that knowledge comes from."""

    directions: dict[str, str]
    source: Literal["recipe", "lowerer", "declared"]
    recipe: RegisteredRecipe | None = None
    open_ports: bool = False
    # Recipe ports the part cannot work without, and the ones a reviewed recipe allows tying to
    # ground when the design does not use them (a spare bus or address channel, a shell).
    required: frozenset[str] = frozenset()
    groundable: frozenset[str] = frozenset()

    @property
    def choices(self) -> str:
        return ",".join(sorted(self.directions)) or "(none)"


@dataclass(frozen=True)
class _SheetRef:
    """One resolved signal endpoint: its requirement, port key and declared direction."""

    requirement: IntentRequirement
    port: str
    direction: str


def _application_direction(name: str) -> str | None:
    """Direction of a capability-prefixed application port, or None when the name is not one."""
    for prefix, direction in _SIGNAL_DIRECTIONS:
        if name == prefix or name.startswith(f"{prefix}_"):
            return direction
    return None


def _allows_application_port(catalog: _Catalog, requirement: IntentRequirement, name: str) -> bool:
    """MCU application pins are capability-named, not enumerated by the recipe."""
    if not catalog.recipe or not catalog.recipe.definition.allocatable_pins:
        return False
    if any(name == prefix or name.startswith(f"{prefix}_") for prefix in _APPLICATION_PREFIXES):
        return True
    if re.fullmatch(r"gpio\d+", name):
        return True
    # A bus member key is a request for that bus: the interface list is derived from the
    # ports the design binds, so the key itself is what has to be checked here.
    return any(
        name in keys for members in FIXED_INTERFACES.values() for keys, _capability in members
    )


# Interfaces whose whole contract is "this many pins of this capability": the bound ports state
# them, so a declaration with no matching port is not a request.
_CAPABILITY_INTERFACES = frozenset({"parallel_output", "pwm", "adc"})


def _derived_interfaces(
    declared: list[str],
    bound: set[str],
    parameters: dict,
) -> list[str]:
    """The interface list a requirement's own bound ports imply.

    An interface is a *consequence* of the ports the design binds, the same way a net name is a
    consequence of a signal: an interface declared with no member port bound is not a request
    (the allocator would refuse the requirement for a binding the model never meant to make), and
    a bound member port needs its interface declared. `parallel_output` additionally carries the
    count its own `parallel_<i>` ports state.
    """
    derived = [
        name
        for name, members in FIXED_INTERFACES.items()
        if any(any(port in bound for port in keys) for keys, _capability in members)
    ]
    parallel = sorted(
        int(match.group(1)) for key in bound if (match := re.fullmatch(r"parallel_(\d+)", key))
    )
    if parallel:
        derived.append("parallel_output")
        if not isinstance(parameters.get("parallel_output_count"), int):
            parameters["parallel_output_count"] = parallel[-1] + 1
    for name, prefix in (("pwm", "pwm"), ("adc", "adc")):
        if any(key == prefix or key.startswith(f"{prefix}_") for key in bound):
            derived.append(name)
    for name in declared:
        if (
            name not in derived
            and name not in FIXED_INTERFACES
            and name not in _CAPABILITY_INTERFACES
        ):
            derived.append(name)  # an interface this compiler does not model: keep the claim
    return list(dict.fromkeys(derived))


def _direction(catalog: _Catalog, requirement: IntentRequirement, name: str) -> str | None:
    if name in catalog.directions:
        return catalog.directions[name]
    if _allows_application_port(catalog, requirement, name):
        return _application_direction(name) or "output"
    if catalog.open_ports:
        return "bidirectional"
    return None


def _catalog(
    requirement: IntentRequirement,
    recipes: tuple[RegisteredRecipe, ...],
    lowerers: dict[str, RegisteredLowerer],
) -> _Catalog | None:
    """One requirement's port vocabulary: curated recipe, lowerer family, or declared interface."""
    matches = list(_family_recipes(requirement.family, recipes))
    selected = _recipe_for_exact(requirement.exact_part, recipes)
    if selected is not None and selected not in matches:
        matches.append(selected)
    if selected is None and matches:
        defaults = [recipe for recipe in matches if recipe.definition.default_for_family]
        selected = (
            defaults[0] if len(defaults) == 1 else (matches[0] if len(matches) == 1 else None)
        )
    if selected is not None:
        definition = selected.definition
        return _Catalog(
            directions={port.name: port.direction for port in definition.ports},
            source="recipe",
            recipe=selected,
            required=frozenset(port.name for port in definition.ports if port.required),
            groundable=frozenset(port.name for port in definition.ports if port.allow_ground),
        )
    if requirement.family in lowerers:
        # A lowerer's `port_keys` are prose placeholders for keyed families (`pin1..pinN`), so its
        # vocabulary is open: the model names the keys, the lowerer validates them at BOM.
        return _Catalog(directions={}, source="lowerer", open_ports=True)
    if requirement.declared_ports:
        return _Catalog(
            directions={port.key: port.direction for port in requirement.declared_ports},
            source="declared",
        )
    return None


def _resolve_port(
    requirement: IntentRequirement, catalog: _Catalog, name: str
) -> tuple[str, str] | None:
    """The canonical (port key, direction) `name` denotes on this requirement, or None."""
    direction = _direction(catalog, requirement, name)
    if direction is not None:
        return name, direction
    for port, aliases in _PORT_ALIASES.items():
        if port not in catalog.directions:
            continue
        if _net_identity(name) in {_net_identity(alias) for alias in (port, *aliases)}:
            return port, catalog.directions[port]
    return None


def _edge_label(target: str) -> str | None:
    if not target.startswith(EDGE_PREFIX):
        return None
    label = target[len(EDGE_PREFIX) :].strip().upper().replace(" ", "_").replace("-", "_")
    return label or None


def derive_architecture(intent: ArchitectureIntent | dict) -> Architecture:
    """Turn an intent-shaped slot into the canonical `Architecture`.

    Raises `ArchitectureIntentError` carrying *every* blocking refusal at once, so a rejected draft
    names all the parts, ports and rails it could not derive wiring from in a single answer.
    """
    intent = ArchitectureIntent.model_validate(intent)
    recipes = tuple(registered_recipes())
    lowerers = {family: row for row in registered_lowerers() for family in row.families}
    diagnostics: list[IntentDiagnostic] = []

    def _fail(code: str, message: str, **kw) -> None:
        diagnostics.append(IntentDiagnostic(code=code, message=message, **kw))

    sheets = [
        Sheet(
            name=row.name,
            stem=row.stem,
            function=row.function,
            from_library=row.from_library,
            library_instance=row.library_instance,
            replication_group=row.replication_group,
            replication_instance=row.replication_instance,
        )
        for row in intent.sheets
    ]
    sheet_by_stem = {row.stem: row for row in sheets}
    sheet_names = {row.name for row in sheets}

    models_by_id: dict[str, IntentRequirement] = {}
    for row in intent.requirements:
        if row.id in models_by_id:
            _fail(
                "duplicate_requirement_id",
                f"requirement id {row.id!r} is declared twice",
                requirement_id=row.id,
            )
        models_by_id[row.id] = row
    for row in intent.requirements:
        if row.sheet not in sheet_names:
            _fail(
                "unknown_requirement_sheet",
                f"requirement {row.id!r} names sheet {row.sheet!r}, which is not declared",
                requirement_id=row.id,
                sheet=row.sheet,
                evidence=sorted(sheet_names),
            )

    catalogs: dict[str, _Catalog] = {}
    requirements: dict[str, CircuitRequirement] = {}
    for row in intent.requirements:
        if row.sheet not in sheet_names:
            continue
        catalog = _catalog(row, recipes, lowerers)
        if catalog is None:
            continue  # model-owned, or refused below once we know whether a signal needs its pins
        catalogs[row.id] = catalog
        requirements[row.id] = CircuitRequirement(
            id=row.id,
            sheet=row.sheet,
            role=row.role,
            family=row.family,
            # The recipe's own exact part is the identity the resolver would settle on anyway;
            # stating it here keeps the resolved payload explicit instead of inferred.
            exact_part=row.exact_part
            or (catalog.recipe.definition.exact_part if catalog.recipe is not None else None),
            parameters=dict(row.parameters),
            interfaces=list(row.interfaces),
            functional_blocks=list(row.functional_blocks),
        )

    signals = [row for item in intent.signals for row in item.expanded()]
    referenced = {row.from_ref.partition(".")[0] for row in signals}
    referenced |= {
        peer.partition(".")[0]
        for row in signals
        for peer in row.peers()
        if not peer.startswith(EDGE_PREFIX)
    }
    referenced |= {
        rail.from_ref.partition(".")[0]
        for rail in intent.power.rails.values()
        if rail.from_ref is not None
    }
    for requirement_id in sorted(referenced):
        if requirement_id in catalogs or requirement_id not in models_by_id:
            continue
        row = models_by_id[requirement_id]
        if row.declared_ports:
            continue
        _fail(
            "unknown_part_refused",
            (
                f"requirement {row.id!r} ({row.family}"
                + (f", {row.exact_part}" if row.exact_part else "")
                + ") has no curated recipe and declares no interface; state the part's interface "
                "(`declared_ports`: one entry per pin function with a direction) so its wiring can "
                "be derived, or use a family with a curated recipe"
            ),
            requirement_id=row.id,
            sheet=row.sheet,
            evidence=[row.family, *([row.exact_part] if row.exact_part else [])],
        )

    # Endpoint bookkeeping: every port binding, with the direction it contributes to its net.
    bindings: dict[str, dict[str, str]] = {requirement_id: {} for requirement_id in requirements}
    endpoints: dict[str, list[tuple[str, str]]] = {}

    def _bind(requirement_id: str, port: str, net: str, direction: str, *, context: str) -> None:
        # A recipe's `power` port direction is not a sheet-boundary direction:
        # an endpoint that carries a rail on this sheet is bidirectional there.
        direction = "bidirectional" if direction == "power" else direction
        row = bindings.setdefault(requirement_id, {})
        existing = row.get(port)
        if existing is not None and existing != net:
            _fail(
                "conflicting_port_binding",
                f"{context}: port {port!r} of {requirement_id!r} is already bound to {existing!r}",
                requirement_id=requirement_id,
                sheet=requirements[requirement_id].sheet,
            )
            return
        bindings[requirement_id][port] = net
        sheet = requirements[requirement_id].sheet
        rows = endpoints.setdefault(net, [])
        # One sheet may touch a net more than one way (a driver and the connector it
        # feeds on the same sheet): keep each direction once, never merge them away.
        if (sheet, direction) not in rows:
            rows.append((sheet, direction))

    def _lookup(reference: str, *, context: str) -> _SheetRef | None:
        """`<requirement_id>.<port>`, any accepted alias -> the canonical endpoint."""
        requirement_id, separator, port_name = reference.partition(".")
        if not separator or not requirement_id or not port_name:
            _fail(
                "malformed_signal_ref",
                (
                    f"{context}: reference {reference!r} must be '<requirement_id>.<port>' or "
                    f"'{EDGE_PREFIX}<name>'"
                ),
            )
            return None
        if requirement_id not in models_by_id:
            # A reference may abbreviate the requirement it means (`hub` for `hub75`). Exactly
            # one declared id may match; an ambiguous or absent one stays a refusal.
            matches = [row.id for row in intent.requirements if row.id.startswith(requirement_id)]
            if len(matches) == 1:
                requirement_id = matches[0]
        requirement = requirements.get(requirement_id)
        if requirement is None:
            if requirement_id not in models_by_id:
                _fail(
                    "unknown_signal_requirement",
                    f"{context}: reference {reference!r} names an undeclared requirement",
                    requirement_id=requirement_id,
                    evidence=sorted(models_by_id),
                )
            return None  # a model-owned requirement already carries its own refusal, if any
        catalog = catalogs[requirement_id]
        resolved = _resolve_port(models_by_id[requirement_id], catalog, port_name.lower().strip())
        if resolved is None:
            _fail(
                "unknown_interface_port",
                (
                    f"{context}: {reference!r} names no port of requirement {requirement_id!r} "
                    f"({catalog.source} interface)"
                ),
                requirement_id=requirement_id,
                sheet=requirement.sheet,
                evidence=[f"ports={catalog.choices}", f"requested={port_name}"],
            )
            return None
        port, direction = resolved
        return _SheetRef(models_by_id[requirement_id], port, direction)

    rail_names = set(intent.power.rails)
    # Connectors that expose a rail rather than draw from it (`supply` on a connector family).
    connector_rails: dict[str, str] = {}
    for net, rail in intent.power.rails.items():
        if rail.from_ref is None:
            continue
        source = _lookup(rail.from_ref, context=f"rail {net!r}")
        if source is None:
            continue
        _bind(source.requirement.id, source.port, net, "output", context=f"rail {net!r}")

    for requirement_id, requirement in requirements.items():
        row = models_by_id[requirement_id]
        catalog = catalogs[requirement_id]
        if "gnd" in catalog.directions:
            _bind(requirement_id, "gnd", GND_NET, "bidirectional", context="ground")
        if row.supply is None:
            continue
        if row.supply not in rail_names:
            _fail(
                "unknown_supply_rail",
                (
                    f"requirement {requirement_id!r} declares supply {row.supply!r}, which is not a "
                    "declared rail under power.rails"
                ),
                requirement_id=requirement_id,
                sheet=requirement.sheet,
                evidence=sorted(rail_names),
            )
            continue
        port = next((name for name in _SUPPLY_PORTS if name in catalog.directions), None)
        if port is None:
            if requirement.role == "connector":
                # A connector's `supply` is not a port it draws from: it is the rail the
                # connector exposes to the peer. Add the pin (GND joins in `_close_connectors`).
                connector_rails[requirement_id] = row.supply
                continue
            _fail(
                "unsupported_supply_port",
                (
                    f"requirement {requirement_id!r} declares supply {row.supply!r}, but its "
                    f"{catalog.source} interface has no supply port; bind the rail with a signal"
                ),
                requirement_id=requirement_id,
                sheet=requirement.sheet,
                evidence=[f"ports={catalog.choices}"],
            )
            continue
        _bind(requirement_id, port, row.supply, "input", context="supply")

    # Off-board peers: one connector per `edge:` label, carrying only what named it.
    edge_connectors: dict[str, tuple[str, str, list[str]]] = {}  # label -> (id, mode, rails)
    # A label that cannot be opened says so once: two signals asking for the same
    # impossible edge must not each repeat the same refusal.
    failed_edges: set[str] = set()
    renamed: list[str] = []
    # Derived statements the review should see (a tied spare port, an exposed output).
    derived_notes: list[str] = []

    def _open_edge(
        label: str, source: _SheetRef, requirement_id: str | None = None
    ) -> tuple[str, str] | None:
        existing = edge_connectors.get(label)
        if existing is not None:
            return existing[0], existing[1]
        if label in failed_edges:
            return None
        sheet = sheet_by_stem.get(label)
        if sheet is None:
            sheet = Sheet(
                name=label.replace("_", " "),
                stem=label,
                function=f"Board-edge connector for {label.replace('_', ' ').title()}",
            )
            sheets.append(sheet)
            sheet_by_stem[label] = sheet
        # An edge label that names a sheet the model already declared puts the connector
        # there (the sheet owns that interface); a fresh label gets its own sheet.
        requirement_id = requirement_id or f"{source.requirement.id}_{label.lower()}"
        if requirement_id in requirements:
            _fail(
                "duplicate_edge_connector",
                f"edge {label!r} would need connector {requirement_id!r}, which already exists",
            )
            failed_edges.add(label)
            return None
        source_recipe = catalogs[source.requirement.id].recipe
        source_recipe_name = (
            source_recipe.definition.recipe if source_recipe is not None else None
        )
        mode = (
            "usb"
            if source.port in _USB_DATA_PORTS and source_recipe_name in _NATIVE_USB_MCU_RECIPES
            else "header"
        )
        if mode == "usb":
            vbus = _usb_vbus_rail(intent)
            if vbus is None:
                _fail(
                    "usb_connector_supply_unknown",
                    (
                        f"edge {label!r} carries the native USB data pair but no 5 V rail is "
                        f"declared (named {_USB_VBUS_RAIL_NAMES} or at ~5 V); declare the rail the "
                        "USB socket exposes under power.rails"
                    ),
                    requirement_id=source.requirement.id,
                    evidence=sorted(rail_names),
                )
                failed_edges.add(label)
                return None
            catalogs[requirement_id] = _Catalog(
                directions=dict.fromkeys(("vbus", "gnd", "usb_dm", "usb_dp"), "bidirectional"),
                source="recipe",
            )
            requirements[requirement_id] = CircuitRequirement(
                id=requirement_id,
                sheet=sheet.name,
                role="connector",
                family=_USB_DEVICE_CONNECTOR_FAMILY,
                exact_part=_USB_DEVICE_CONNECTOR_PART,
                functional_blocks=list(source.requirement.functional_blocks),
                # A recipe that already expands the 22R MCU-side pair owns it; the
                # socket must not add a second pair in series with the same lines.
                parameters=(
                    {"series_resistors": False}
                    if source_recipe_name in _MCU_OWNED_USB_SERIES_RECIPES
                    else {}
                ),
            )
            _bind(requirement_id, "vbus", vbus, "input", context=f"edge {label!r}")
            _bind(requirement_id, "gnd", GND_NET, "bidirectional", context=f"edge {label!r}")
        else:
            catalogs[requirement_id] = _Catalog(directions={}, source="lowerer", open_ports=True)
            requirements[requirement_id] = CircuitRequirement(
                id=requirement_id,
                sheet=sheet.name,
                role="connector",
                family=_HEADER_FAMILY,
                parameters={"rows": 1, "gender": "male"},
                functional_blocks=list(source.requirement.functional_blocks),
            )
        edge_connectors[label] = (requirement_id, mode, [])
        return requirement_id, mode

    for signal in signals:
        source = _lookup(signal.from_ref, context=f"signal {signal.name!r}")
        if source is None:
            continue
        # A signal out of a port that already carries ground or a declared rail IS that net: the
        # design said the same thing twice (the implicit binding plus a named wire), and one pin
        # cannot be on two nets. The existing net owns the connection, the peers join it, and a
        # rail rename is recorded — nothing is invented. Two *different* rails on one pin, or a
        # ground signal out of a rail port, stay refusals.
        source_net = bindings.get(source.requirement.id, {}).get(source.port)
        if source_net in rail_names or source_net == GND_NET:
            if signal.name in rail_names and signal.name != source_net:
                _fail(
                    "signal_conflicts_with_rail",
                    (
                        f"signal {signal.name!r} leaves {signal.from_ref}, which the design already "
                        f"ties to rail {source_net!r}; one port cannot carry two rails"
                    ),
                    requirement_id=source.requirement.id,
                    sheet=source.requirement.sheet,
                )
                continue
            if source_net in rail_names and signal.name == GND_NET:
                _fail(
                    "signal_conflicts_with_rail",
                    (
                        f"signal {signal.name!r} leaves {signal.from_ref}, which the design already "
                        f"ties to rail {source_net!r}; a rail port is not a ground connection"
                    ),
                    requirement_id=source.requirement.id,
                    sheet=source.requirement.sheet,
                )
                continue
            if signal.name != source_net and source_net in rail_names:
                renamed.append(
                    f"{signal.name}: peers of {signal.from_ref} join rail {source_net} "
                    "(declared rail owns the net) (derived)"
                )
            for peer in signal.peers():
                label = _edge_label(peer)
                if label is not None:
                    opened = _open_edge(label, source)
                    if opened is not None and source_net in rail_names:
                        edge_connectors[label][2].append(source_net)
                    continue
                target = _lookup(peer, context=f"signal {signal.name!r}")
                if target is None:
                    continue
                _bind(
                    target.requirement.id,
                    target.port,
                    source_net,
                    "bidirectional" if source_net == GND_NET else "input",
                    context=f"signal {signal.name!r}",
                )
            continue
        if signal.name in rail_names or signal.name == GND_NET:
            _fail(
                "signal_names_rail",
                (
                    f"signal {signal.name!r} is a declared rail; rails are wired from `power.rails` "
                    "and each requirement's `supply`, not from `signals`"
                ),
                requirement_id=source.requirement.id,
            )
            continue
        _bind(
            source.requirement.id,
            source.port,
            signal.name,
            source.direction,
            context=f"signal {signal.name!r}",
        )
        for peer in signal.peers():
            label = _edge_label(peer)
            if label is not None:
                opened = _open_edge(label, source)
                if opened is None:
                    continue
                connector_id, mode = opened
                if mode == "usb":
                    port = "usb_dm" if source.port == "usb_dm" else "usb_dp"
                    _bind(
                        connector_id,
                        port,
                        signal.name,
                        "bidirectional",
                        context=f"signal {signal.name!r}",
                    )
                else:
                    port = f"pin{len(bindings.setdefault(connector_id, {})) + 1}"
                    _bind(
                        connector_id,
                        port,
                        signal.name,
                        _SINK_DIRECTION[source.direction],
                        context=f"signal {signal.name!r}",
                    )
                edge_connectors[label][2].extend(signal.rails)
                continue
            target = _lookup(peer, context=f"signal {signal.name!r}")
            if target is None:
                continue
            _bind(
                target.requirement.id,
                target.port,
                signal.name,
                target.direction,
                context=f"signal {signal.name!r}",
            )

    def _next_pin(requirement_id: str) -> str:
        taken = bindings.setdefault(requirement_id, {})
        index = 1
        while f"pin{index}" in taken:
            index += 1
        return f"pin{index}"

    # A required recipe output nobody bound is, by the recipe's own contract, an output that
    # leaves the board (a WS2812 driver's `data_out` feeding the next LED). Name it after its own
    # requirement and port, expose it with the part's supply rail and ground, and say so: the net
    # is one the recipe requires, and its name is reported rather than silently chosen.
    for requirement_id, catalog in sorted(catalogs.items()):
        if catalog.source != "recipe" or catalog.recipe is None:
            continue
        supply = bindings.get(requirement_id, {}).get(
            next((name for name in _SUPPLY_PORTS if name in catalog.directions), "")
        )
        for port in sorted(catalog.required):
            if catalog.directions.get(port) != "output" or bindings.get(requirement_id, {}).get(
                port
            ):
                continue
            net = f"{requirement_id}_{port}".upper()
            if net in endpoints:
                _fail(
                    "derived_output_name_collision",
                    (
                        f"requirement {requirement_id!r} port {port!r} needs a net, but {net!r} is "
                        "already declared by the design; bind the port with a signal instead"
                    ),
                    requirement_id=requirement_id,
                    sheet=requirements[requirement_id].sheet,
                )
                continue
            source = _SheetRef(models_by_id[requirement_id], port, "output")
            label = f"{requirement_id}_{port}".upper()
            opened = _open_edge(label, source, f"{requirement_id}_{port}")
            if opened is None:
                continue
            connector_id, _mode = opened
            _bind(requirement_id, port, net, "output", context="derived output")
            _bind(
                connector_id,
                _next_pin(connector_id),
                net,
                "input",
                context=f"derived output {requirement_id}.{port}",
            )
            if supply:
                edge_connectors[label][2].append(supply)
            derived_notes.append(
                f"{requirement_id}.{port}: required output with no peer exposed as {net} on its own "
                "board-edge connector (derived)"
            )

    # Header connectors close their pin order with GND and the rails the peer asked for.
    for label, (connector_id, mode, rails) in edge_connectors.items():
        unknown_rails = sorted({rail for rail in rails if rail not in rail_names})
        if unknown_rails:
            _fail(
                "unknown_edge_rail",
                (
                    f"edge {label!r} names rails {unknown_rails} the architecture does not declare; "
                    "declare each one under power.rails or drop it from the signal"
                ),
                requirement_id=connector_id,
                evidence=sorted(rail_names),
            )
            continue
        if not bindings[connector_id]:
            _fail(
                "empty_edge_connector",
                f"edge {label!r} was named by a signal that could not be resolved to a source port",
            )
            continue
        if mode == "usb":
            missing = sorted(
                port for port in _USB_DATA_PORTS if not bindings[connector_id].get(port)
            )
            if missing:
                _fail(
                    "incomplete_usb_edge",
                    (
                        f"edge {label!r} is a USB data connector but no signal bound {missing}; a "
                        "USB socket needs both usb_dm and usb_dp"
                    ),
                    requirement_id=connector_id,
                    evidence=sorted(bindings[connector_id]),
                )
            continue
        for net in [GND_NET, *dict.fromkeys(rails)]:
            port = f"pin{len(bindings.setdefault(connector_id, {})) + 1}"
            _bind(
                connector_id,
                port,
                net,
                "bidirectional" if net == GND_NET else "input",
                context=f"edge {label!r}",
            )

    # Ports the design ties to a declared net (ground, a rail, a signal's own net).
    signal_names = {row.name for row in signals}
    for requirement_id, requirement in requirements.items():
        model = models_by_id.get(requirement_id)
        if model is None:
            continue  # a connector the compiler built has no intent of its own
        for port_name, net in model.ties.items():
            catalog = catalogs[requirement_id]
            resolved = _resolve_port(model, catalog, port_name.lower().strip())
            if resolved is None:
                _fail(
                    "unknown_interface_port",
                    (
                        f"tie {port_name!r} of requirement {requirement_id!r} names no port of its "
                        f"{catalog.source} interface"
                    ),
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                    evidence=[f"ports={catalog.choices}", f"requested={port_name}"],
                )
                continue
            if net != GND_NET and net not in rail_names and net not in signal_names:
                _fail(
                    "unknown_tie_net",
                    (
                        f"tie {port_name!r} of requirement {requirement_id!r} names net {net!r}, "
                        "which no rail, no ground and no signal declares"
                    ),
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                    evidence=[GND_NET, *sorted(rail_names), *sorted(signal_names)],
                )
                continue
            _bind(
                requirement_id,
                resolved[0],
                net,
                resolved[1],
                context=f"tie {port_name!r}",
            )

    # A required port the reviewed recipe allows grounding and the design does not use (HUB75's
    # spare address channel, a connector shell) is tied low here, once, instead of being asked for
    # and then validated: an unbound spare input floats, which is the defect, not the fix.
    for requirement_id, catalog in catalogs.items():
        if catalog.source != "recipe":
            continue
        for port in sorted(catalog.required & catalog.groundable):
            if bindings.get(requirement_id, {}).get(port):
                continue
            _bind(requirement_id, port, GND_NET, "bidirectional", context="unused port tie")
            derived_notes.append(f"{requirement_id}: unused {port} tied to {GND_NET} (derived)")

    # A connector closes its pin order with ground and the rails it exposes: the signals the
    # design bound first, then GND, then any rail the connector declares as its supply.
    for requirement_id, rail in connector_rails.items():
        if not any(bound == GND_NET for bound in bindings.get(requirement_id, {}).values()):
            _bind(
                requirement_id,
                _next_pin(requirement_id),
                GND_NET,
                "bidirectional",
                context="connector ground",
            )
        _bind(requirement_id, _next_pin(requirement_id), rail, "input", context="connector supply")

    # A port the reviewed recipe requires and nothing wired is a missing design statement, not
    # a bookkeeping slip: the part cannot work without it. Name it here, once, instead of
    # letting the part reach BOM expansion and fail on a port binding mismatch.
    for requirement_id, catalog in sorted(catalogs.items()):
        if catalog.source != "recipe":
            continue
        bound = bindings.get(requirement_id, {})
        for port in sorted(catalog.required):
            if bound.get(port) or catalog.directions.get(port) == "output":
                continue  # a required output is exposed on its own edge connector above
            recipe = catalog.recipe.definition.recipe if catalog.recipe is not None else "recipe"
            _fail(
                "unbound_required_port",
                (
                    f"requirement {requirement_id!r} port {port!r} is required by {recipe} and "
                    "nothing wires it; state a signal that uses the port, or tie it to a declared "
                    "net"
                ),
                requirement_id=requirement_id,
                sheet=requirements[requirement_id].sheet,
                evidence=[f"ports={catalog.choices}"],
            )

    if diagnostics:
        raise ArchitectureIntentError(diagnostics)

    # `interfaces` follows the ports the design bound (see `_derived_interfaces`), so the
    # allocator is asked for exactly the interfaces this design actually uses.
    for requirement_id, requirement in requirements.items():
        model = models_by_id.get(requirement_id)
        if model is None or catalogs[requirement_id].source != "recipe":
            continue
        parameters = dict(requirement.parameters)
        interfaces = _derived_interfaces(
            list(requirement.interfaces), set(bindings.get(requirement_id, {})), parameters
        )
        requirements[requirement_id] = requirement.model_copy(
            update={"interfaces": interfaces, "parameters": parameters}
        )

    final_requirements = [
        requirements[requirement_id].model_copy(update={"ports": bindings[requirement_id]})
        for requirement_id in sorted(requirements)
    ]
    declared = sorted(
        requirement_id
        for requirement_id, catalog in catalogs.items()
        if catalog.source == "declared"
    )
    notes = [
        (
            f"{requirement_id}: interface declared by the model for {models_by_id[requirement_id].family}"
            + (
                f" ({models_by_id[requirement_id].exact_part})"
                if models_by_id[requirement_id].exact_part
                else ""
            )
            + "; its pin functions are a claim, not verified against a curated recipe (derived)"
        )
        for requirement_id in declared
    ]
    # A declared programming path is the design decision the checks look for in the slot's own
    # text; recording it here keeps the statement the model made instead of asking for it twice.
    notes.extend(
        f"{requirement_id}: programming and reset path {models_by_id[requirement_id].programming} "
        "(declared)"
        for requirement_id in sorted(models_by_id)
        if models_by_id[requirement_id].programming
    )
    return Architecture(
        topologies=dict(intent.topologies),
        rail_voltages={
            **{net: rail.voltage for net, rail in intent.power.rails.items()},
            GND_NET: 0.0,
        },
        comms_protocols=list(intent.comms_protocols),
        mcu_present=intent.mcu_present,
        sheets=sheets,
        power_nets=[GND_NET, *sorted(rail_names)],
        inter_sheet_nets=_inter_sheet_nets(final_requirements, endpoints, rail_names),
        assumptions=[*intent.assumptions, *derived_notes, *renamed, *notes],
        requirements=final_requirements,
        declared_interfaces=declared,
    )


def _usb_vbus_rail(intent: ArchitectureIntent) -> str | None:
    """The 5 V rail a USB socket exposes, never a guess.

    Preference: the rail the board's own input requirement generates (that is the
    host's VBUS, exactly what a USB socket's VBUS pin carries), then a single
    rail named VBUS/+5V, then a single rail at 5 V. Two candidates at any step
    mean the design has not said which rail the socket exposes, and the model is
    asked rather than guessed at.
    """
    input_ids = {row.id for row in intent.requirements if row.role == "power_input"}
    from_input = [
        net
        for net, rail in intent.power.rails.items()
        if rail.from_ref and rail.from_ref.partition(".")[0] in input_ids
    ]
    if len(from_input) == 1:
        return from_input[0]
    named = [name for name in _USB_VBUS_RAIL_NAMES if name in intent.power.rails]
    if len(named) == 1:
        return named[0]
    volts = [name for name, rail in intent.power.rails.items() if abs(rail.voltage - 5.0) <= 0.5]
    return volts[0] if len(volts) == 1 else None


def _inter_sheet_nets(
    requirements: list[CircuitRequirement],
    endpoints: dict[str, list[tuple[str, str]]],
    rail_names: set[str],
) -> list[InterSheetNet]:
    """One record per net with ≥2 sheet endpoints, GND and rails first, in declaration order."""
    order = [GND_NET, *sorted(rail_names)]
    order.extend(name for name in endpoints if name not in set(order))
    return [
        InterSheetNet(
            name=name,
            endpoints=[SheetPin(sheet=sheet, direction=direction) for sheet, direction in rows],
        )
        for name in order
        if len(rows := endpoints.get(name, [])) >= 2
    ]
