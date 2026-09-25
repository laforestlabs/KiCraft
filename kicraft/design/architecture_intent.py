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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .lowering import (
    RegisteredLowerer,
    lowerer_contract_diagnostic,
    lowerer_port_direction,
    registered_lowerers,
)
from .models import (
    SHEET_NAME_RE,
    SHEET_STEM_RE,
    Architecture,
    ArchitectureAdvisory,
    CircuitRequirement,
    CircuitRole,
    DeclaredInterfaceClaim,
    DeclaredInterfacePort,
    RequirementObligation,
    InterSheetNet,
    Sheet,
    SheetPin,
    is_power_or_ground_name,
    obligation_requires_requirement_owner,
)
from .recipes.models import RegisteredRecipe
from .recipes.pin_allocator import FIXED_INTERFACES
from kicraft.form_factors import get_template
from .recipes.registry import registered_recipes
from .recipes.resolver import (
    _PORT_ALIASES,
    _family_recipes,
    _net_identity,
    _recipe_for_exact,
)
from .synthesis.board_features import (
    PROTOTYPING_AREA_FEATURE,
    PROTOTYPING_AREA_OBLIGATION_ID,
    has_prototyping_area,
    prototyping_area_requested,
)

EDGE_PREFIX = "edge:"
"""Signals whose peer is off the board: `"to": "edge:LED_STRING"`."""

GND_NET = "GND"

# The board fabricated from a `fabrication` obligation is a pad field, a thing no component
# class names and no pin draws a net through. Its whole shape is determined by the obligation,
# so the derivation writes the sheet, the requirement and the field's default geometry -- the
# same way it writes a board-edge connector from an `edge:` peer -- instead of asking the model
# for a part it cannot name. The obligation itself stays the design's own statement.
PROTOTYPING_AREA_SHEET_NAME = "PROTOTYPING AREA"
PROTOTYPING_AREA_SHEET_STEM = "PROTOTYPING_AREA"
PROTOTYPING_AREA_FAMILY = "prototyping-area"
PROTOTYPING_AREA_FUNCTION = (
    "Bare pad field the user solders through-hole parts into; no net, no part, board fabricated"
)
# The smallest field that is still a usable prototyping area, on the 2.54 mm breadboard pitch
# the lowerer drills. A brief rarely states a size, and the obligation records only the fact.
PROTOTYPING_AREA_PARAMETERS: dict[str, int | float] = {
    "rows": 5,
    "cols": 5,
    "pitch_mm": 2.54,
}


def _committed_block_names(functional_spec: object | None) -> list[str]:
    """The block names a committed `functional_spec` carries, in committed order.

    Accepts the slot as the committed mapping a stage state holds, or as a `FunctionalSpec`; a
    caller with neither (a derivation run on its own) passes nothing and gets no names.
    """
    if functional_spec is None:
        return []
    rows = (
        functional_spec.get("blocks")
        if isinstance(functional_spec, Mapping)
        else getattr(functional_spec, "blocks", None)
    )
    names: list[str] = []
    for row in rows or ():
        name = row.get("name") if isinstance(row, Mapping) else getattr(row, "name", None)
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _names_prototyping_area(name: str) -> bool:
    """Whether a slot name (a functional block, a sheet stem) names the pad field.

    The brief-level phrase rule (`kicraft.design.synthesis.board_features`) is the one canonical
    reading of "the user asked for a prototyping area"; a slot name is the same words with
    separators (`PROTOTYPING_AREA`, `PROTO BOARD`, `PAD_FIELD`), so it is read by that same rule
    rather than a second vocabulary that could drift away from it.
    """
    return prototyping_area_requested(name.replace("_", " ")) is not None


def _prototyping_area_block(functional_spec: object | None) -> str | None:
    """The committed functional block that asks for the pad field, or None.

    The derived requirement must claim the *exact* committed block name, or the block-coverage
    check reports that block unowned. The name only exists in the functional spec, so it is read
    off it rather than invented; the first block that names the field in committed order wins.
    """
    for name in _committed_block_names(functional_spec):
        if _names_prototyping_area(name):
            return name
    return None


def apply_authoritative_standard_form_factor(
    architecture_payload: dict,
    original_intent: object | None,
) -> dict:
    """Carry the user-owned standard into an architecture candidate.

    This boundary is shared by provider normalization and direct commits: a
    provider may omit a server-owned mechanical fact, but may not replace it.
    """

    form_factor = (
        original_intent.get("form_factor")
        if isinstance(original_intent, dict)
        else getattr(original_intent, "form_factor", None)
    )
    standard = (
        form_factor.get("standard")
        if isinstance(form_factor, dict)
        else getattr(form_factor, "standard", None)
    )
    if not isinstance(standard, str) or not standard.strip():
        return architecture_payload
    standard = standard.strip()
    candidate = architecture_payload.get("standard_form_factor")
    if candidate is not None and (
        not isinstance(candidate, str) or candidate.strip().casefold() != standard.casefold()
    ):
        raise ValueError(
            "architecture standard_form_factor contradicts the "
            "user-approved intent.form_factor.standard"
        )
    return {**architecture_payload, "standard_form_factor": standard}


_USB_DATA_PORTS = frozenset({"usb_dm", "usb_dp"})
_USB_DEVICE_CONNECTOR_FAMILY = "usb-c-usb2-device"
_USB_DEVICE_CONNECTOR_PART = "USB-C-USB2-DEVICE"
_USB_DEVICE_CONNECTOR_RECIPE = "usb-c-usb2-device@1"
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
# driver its vm, a regulator its input, a socket its vbus.
_SUPPLY_PORTS = ("vdd", "vm", "vin", "input", "vdd_5v", "vcc", "vbus", "supply")
# Supply/ground tokens a *qualified* pin name is built from (`vdd_logic`, `vbus_5v`, `gnd_field`).
_SUPPLY_TOKENS = ("vdd", "vcc", "vbus", "vin", "vm", "supply", "positive")
# The ports that *are* a part's reference (return) connection. A qualified name (`gnd_field`,
# `vss_digital`) is still the reference; `negative`/`common` are not, because a terminal named
# that way is just as often a floating output pair (a relay contact, a motor return).
_REFERENCE_PORTS = ("gnd", "vss", "ground", "0v", "agnd", "dgnd")
_REFERENCE_TOKENS = ("gnd", "vss", "ground")
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


class IntentDeclaredPort(DeclaredInterfacePort):
    """One claimed pin of an uncurated part, retained as a canonical claim."""

    model_config = ConfigDict(extra="forbid")


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
    # Optional refinement for multi-supply devices. Unlike `supply`, these name the exact port
    # rather than the conventional one; the compiler derives the binding when they are absent,
    # and falls back to the family's published supply port when the named one does not exist.
    supply_bindings: dict[str, str] = Field(default_factory=dict)
    # Optional refinement for a part with several reference domains: port -> declared zero-volt
    # rail such as GND_LOGIC or GND_FIELD; names never merge domains. Absent, the family's own
    # reference port takes GND.
    reference_bindings: dict[str, str] = Field(default_factory=dict)
    # Role from an approved form-factor template. This is not a freeform
    # connector label: derivation checks its exact pin inventory and nets.
    standard_stacking_role: str | None = None
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
    obligations: list[RequirementObligation] = Field(default_factory=list)


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
    # Standard template selected from the user-approved form-factor contract.
    # Only declared fixed connector roles of this template may use
    # `standard_stacking_role`.
    standard_form_factor: str | None = None
    # The architecture receives the original typed facts, not only prose
    # constraints; each one must have a requirement-local owner below.
    obligations: "list[RequirementObligation]" = Field(default_factory=list)

    @model_validator(mode="after")
    def _obligations_are_owned_once(self):
        owned_keys = [
            (row.kind, row.original_obligation_id)
            for requirement in self.requirements
            for row in requirement.obligations
        ]
        listed = {(row.kind, row.original_obligation_id) for row in self.obligations}
        # Board-wide counts, fabrication/negative facts, and evidence-backed board-outline
        # measurements are top-level facts. A quantitative component or electrical limit must
        # remain on the requirement that can prove it.
        unowned = sorted(
            key
            for key in listed - set(owned_keys)
            if obligation_requires_requirement_owner(
                next(
                    row for row in self.obligations if (row.kind, row.original_obligation_id) == key
                )
            )
        )
        if unowned:
            raise ValueError(
                "ArchitectureIntent obligations must be owned exactly once: each top-level "
                "`obligations` row must also appear on the requirement that implements it, "
                "because the top-level list is the union of the requirements' own rows. "
                f"listed_at_top_level_only={unowned} (attach each to its implementing requirement)"
            )
        # Several requirements may carry the *same* row when the design implements one obligation
        # in more than one place (three binding posts, three identical axis drivers): the rows are
        # identical, so ownership is unambiguous and each requirement's copy is normalised from the
        # design's own row below.
        rows = {
            (row.kind, row.original_obligation_id): row
            for requirement in self.requirements
            for row in requirement.obligations
        }
        missing = [rows[key] for key in dict.fromkeys(owned_keys) if key not in listed]
        if missing:
            self.obligations = [*self.obligations, *missing]
        # A typed obligation is written once, by the stage that owns it. Restore the requirement's
        # copy from the design's own row, so a paraphrased or trimmed copy commits the obligation
        # the user actually stated rather than a markdown summary of it.
        by_key = {(row.kind, row.original_obligation_id): row for row in self.obligations}
        for requirement in self.requirements:
            requirement.obligations = [
                by_key.get((row.kind, row.original_obligation_id), row)
                for row in requirement.obligations
            ]
        return self


class IntentDiagnostic(BaseModel):
    """One finding the derivation records: a blocking refusal, or a recorded advisory.

    ``severity`` is required-in-practice like ``StageDiagnostic.severity``, and each helper below
    states which kind of row it builds. Rows used to be stored without it, and a durable status
    carrying them could not be loaded back at all (119 saved states, live runs KC-HPD3YF and
    KC-P4E2PH among them).
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    requirement_id: str | None = None
    sheet: str | None = None
    message: str
    evidence: list[str] = Field(default_factory=list)
    severity: Literal["advisory", "repair_required"] = "repair_required"


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
    lowerer: RegisteredLowerer | None = None
    open_ports: bool = False
    # Recipe ports the part cannot work without, and the ones a reviewed recipe allows tying to
    # ground when the design does not use them (a spare bus or address channel, a shell).
    required: frozenset[str] = frozenset()
    groundable: frozenset[str] = frozenset()
    #: Ports the recipe ties to another of its own ports when the design wires neither
    #: (`RecipePort.default_tie`): port name -> the port it is tied to.
    default_ties: Mapping[str, str] = field(default_factory=dict)

    @property
    def choices(self) -> str:
        concrete = ",".join(sorted(self.directions))
        if concrete:
            return concrete
        # A lowerer whose ports are a pattern (pin1..pinN) or a named set publishes its
        # contract in words, not concrete keys. Every port refusal embeds this menu, so an
        # empty "(none)" teaches the draft nothing about what it may declare.
        if self.lowerer is not None:
            described = "; ".join(
                [key for key in self.lowerer.port_keys if key]
                + [pattern for pattern, _direction in self.lowerer.port_patterns]
            )
            if described:
                return described
        return "(none)"


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
            derived.append(name)
    return list(dict.fromkeys(derived))


def _direction(catalog: _Catalog, requirement: IntentRequirement, name: str) -> str | None:
    if name in catalog.directions:
        return catalog.directions[name]
    if catalog.lowerer is not None:
        return lowerer_port_direction(catalog.lowerer, name)
    if _allows_application_port(catalog, requirement, name):
        return _application_direction(name) or "output"
    return None


def _name_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def _supply_port(catalog: _Catalog, rail: str) -> str | None:
    """The port a requirement's declared `supply` rail feeds, or None.

    Exact conventional names first (`vdd`, `vcc`, `vbus`, ...). A design that qualifies the pin
    (`vdd_logic`, `vbus_5v`) is still naming its supply: take the single qualified candidate, and
    when several exist let the rail's own name pick (`+5V_LOGIC` -> `vdd_logic`). Anything else
    stays a refusal — the compiler does not guess which pin takes the rail.
    """
    for name in _SUPPLY_PORTS:
        if name in catalog.directions:
            return name
    qualified = sorted(
        name
        for name in catalog.directions
        if any(name == token or name.startswith(f"{token}_") for token in _SUPPLY_TOKENS)
    )
    if len(qualified) == 1:
        return qualified[0]
    if qualified:
        rail_tokens = _name_tokens(rail)
        scored = [name for name in qualified if _name_tokens(name) & rail_tokens]
        if len(scored) == 1:
            return scored[0]
    return None


def _reference_port(catalog: _Catalog) -> str | None:
    """The port a requirement's reference (ground) connection lands on, or None.

    Exact conventional names first, then the single qualified candidate (`gnd_field`). Anything
    else stays unbound: the compiler does not guess which pin is the return.
    """
    for name in _REFERENCE_PORTS:
        if name in catalog.directions:
            return name
    qualified = sorted(
        name
        for name in catalog.directions
        if any(name == token or name.startswith(f"{token}_") for token in _REFERENCE_TOKENS)
    )
    return qualified[0] if len(qualified) == 1 else None


def _token(value) -> str:
    """Casefolded, punctuation-free token, for family comparisons."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


#: Lowerer parameters that mean nothing to a reviewed part's family.
_LOWERER_ONLY_PARAMETERS = ("rows", "gender")

#: The family the compiler uses for the USB socket it writes itself from a data pair sent to an
#: edge. A requirement declaring that family duplicates the socket.
_DERIVED_SOCKET_FAMILY = "usb-c-usb2-device"


def _referenced_ports(payload: dict, requirement_id: str) -> set[str]:
    """Every port of this requirement the draft names: in rails, signals and ties."""
    ports: set[str] = set()
    for row in ((payload.get("power") or {}).get("rails") or {}).values():
        text = str((row or {}).get("from") or "")
        if text.startswith(f"{requirement_id}."):
            ports.add(text.split(".", 1)[1])
    for signal in payload.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        peers = signal.get("to")
        references = [signal.get("from"), *(peers if isinstance(peers, list) else [peers])]
        for reference in references:
            text = str(reference or "")
            if text.startswith(f"{requirement_id}."):
                ports.add(text.split(".", 1)[1])
    requirement = next(
        (row for row in payload.get("requirements") or []
         if isinstance(row, dict) and str(row.get("id")) == requirement_id),
        None,
    )
    ports.update(str(key) for key in (requirement or {}).get("ties") or {})
    return ports


def _lowerer_families() -> frozenset[str]:
    """The generic lowerer families, which implement their own parts and not a reviewed one."""
    from kicraft.design.lowering import lowerer_summaries

    return frozenset(
        str(name).casefold()
        for row in lowerer_summaries()
        if isinstance(row, dict)
        for name in (row.get("families") or [])
    )


def _carrier_family_for_class(component_class: str) -> str | None:
    """The family that realizes this class, when the library has exactly one carrier family."""
    from kicraft.design.part_identity import reviewed_parts_for_feature

    families = sorted(
        {part.family for part in reviewed_parts_for_feature(component_class) if part.family}
    )
    return families[0] if len(families) == 1 else None


def _declared_ports_from_signals(payload: dict, requirement_id: str, contacts: tuple) -> list[dict]:
    """Interface for a reviewed part with no recipe, built from the signals the draft already sends.

    The compiler needs a `pin` per declared port and refuses a function claimed on a contact the
    part does not have, so each port takes the carrier's own contact in the order the draft wires
    them, and its function names the net the draft carries. Nothing here is invented: the port
    names are the draft's, the contacts are the part's, and the function names its own signal.
    """
    ports: list[dict] = []
    index = 0
    for signal in payload.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        peers = signal.get("to")
        references = peers if isinstance(peers, list) else [peers]
        for reference in references:
            text = str(reference or "")
            if not text.startswith(f"{requirement_id}."):
                continue
            key = text.split(".", 1)[1]
            pin = str(contacts[index]) if index < len(contacts) else key
            ports.append(
                {
                    "key": key,
                    "pin": pin,
                    "direction": "passive",
                    "function": f"carries {signal.get('name')}",
                }
            )
            index += 1
    return ports


def _adopt_carrier_families(payload: dict) -> dict:
    """Give a requirement the family its reviewed carrier answers to.

    The writer reaches for a generic lowerer (`pin-header`) for a class that has a reviewed
    carrier (`jst-xh-connector`), or names the reviewed exact part on that lowerer. Both are
    refused, and the parts stage may not change a family later (live walkthrough 2026-09-25:
    "lowerer pin-header@1 does not implement the exact part 'B2B-XH-A(LF)(SN)'", and the BOM
    exhausted its rounds on "missing-requirement-implementation=['motor_a']").
    """
    from kicraft.design.lowering import lowerer_family_for_class
    from kicraft.design.part_identity import canonical_physical_features, reviewed_part

    requirements = [row for row in payload.get("requirements") or [] if isinstance(row, dict)]
    if not requirements:
        return payload
    changed = False
    lowerers = _lowerer_families()
    for requirement in requirements:
        family = str(requirement.get("family") or "")
        if family.casefold() not in lowerers:
            # A requirement that names a reviewed part *class* instead of the lowerer family that
            # builds it gets the family that owns the class's support parts -- the part is the
            # reviewed one either way, but only the family emits the series element the circuit
            # needs (a bare LED across the rail was the live 2026-09-25 shape).
            adopting = lowerer_family_for_class(family, requirement.get("parameters"))
            if adopting is not None and adopting != family:
                requirement["family"] = adopting
                changed = True
            # A curated family that names a reviewed part is the compiler's own business.
            continue
        exact = str(requirement.get("exact_part") or "").strip()
        record = reviewed_part(exact) if exact else None
        target = None
        if (
            record is not None
            and record.family
            and family != record.family
            # A lowerer whose own family vocabulary already denotes the named part's class *is* the
            # builder for it: the LED lowerer honours the reviewed LED identity and adds its series
            # resistor, so rewriting `status-led` to the part's own class (`led-0603`) dropped the
            # resistor and took §9.36 -- which keys on the lowerer family -- out of play (live
            # 2026-09-25). A lowerer that cannot answer the class still adopts the carrier's family.
            and record.family not in canonical_physical_features(family)
        ):
            target = record.family
        else:
            for obligation in requirement.get("obligations") or []:
                if not isinstance(obligation, dict) or obligation.get("kind") != "physical":
                    continue
                component_class = str(obligation.get("component_class") or "")
                carrier_family = _carrier_family_for_class(component_class) if component_class else None
                if (
                    carrier_family
                    and family != carrier_family
                    and carrier_family not in canonical_physical_features(family)
                ):
                    target = carrier_family
                    if record is None:
                        from kicraft.design.part_identity import reviewed_parts_for_feature

                        record = next(
                            (
                                part
                                for part in reviewed_parts_for_feature(component_class)
                                if part.family == carrier_family
                            ),
                            None,
                        )
                    break
        if target is None or target == family:
            continue
        requirement["family"] = target
        parameters = dict(requirement.get("parameters") or {})
        for key in _LOWERER_ONLY_PARAMETERS:
            parameters.pop(key, None)
        requirement["parameters"] = parameters
        if record is not None and not requirement.get("declared_ports"):
            declared = _declared_ports_from_signals(payload, str(requirement.get("id") or ""), record.contacts)
            if declared:
                requirement["declared_ports"] = declared
        changed = True
    return payload


def _complete_terminal_returns(payload: dict) -> dict:
    """Give a screw terminal its return contact when the draft names only the live one.

    The compiler refuses a terminal whose contacts cannot be numbered, and a board input terminal
    has two: the live contact and the return. The return is the ground net, so it is a tie, and it
    keeps the spelling the draft already used ("positive" -> "negative", "pin1" -> "pin2").
    """
    requirements = [row for row in payload.get("requirements") or [] if isinstance(row, dict)]
    for requirement in requirements:
        if "screw" not in _token(str(requirement.get("family") or "")):
            continue
        referenced = sorted(_referenced_ports(payload, str(requirement.get("id") or "")))
        if len(referenced) >= 2:
            continue
        if not referenced:
            continue
        live = referenced[0]
        if live.startswith("pin") and live[3:].isdigit():
            return_key = f"pin{int(live[3:]) + 1}"
        elif live == "positive":
            return_key = "negative"
        else:
            continue
        ties = dict(requirement.get("ties") or {})
        if return_key in ties:
            continue
        ties[return_key] = "GND"
        requirement["ties"] = ties
    return payload


def complete_architecture_payload(payload: dict) -> dict:
    """Drop the duplicate power statements the deterministic contracts can only refuse.

    The writer states one supply twice -- a port's `supply` (or a declared rail) *and* a signal
    to the same pin -- or sets a tie field on a port a signal already names. The compiler
    refuses both (`conflicting_port_binding`, `declared_signal_port_tied`), and a refusal at
    decode costs a whole ladder round: three of the four rejections across the 2026-09-25
    seed-37 architecture drafts were these two shapes, and one of those drafts failed outright.

    Neither correction changes the design. A port a rail owns keeps that net and the ground net
    is implicit on every part, so a signal to such a pin is the duplicate and is dropped; the
    net a tie field would name is already named by the signal, so the field goes. Everything
    else -- the rails, the parts, the nets the writer meant -- is left exactly as written.
    """
    if not isinstance(payload, dict):
        return payload
    # Family and interface repairs first: they change what the later steps see.
    _adopt_carrier_families(payload)
    _complete_terminal_returns(payload)
    requirements = [row for row in payload.get("requirements") or [] if isinstance(row, dict)]
    signals = [row for row in payload.get("signals") or [] if isinstance(row, dict)]
    if not requirements or not signals:
        return payload
    by_id = {str(row.get("id") or ""): row for row in requirements}

    def _endpoint(reference) -> tuple[str, str] | None:
        text = str(reference)
        if text.startswith("edge:") or "." not in text:
            return None
        requirement_id, port = text.split(".", 1)
        return requirement_id, port

    def _peers(signal: dict) -> list[str]:
        peers = signal.get("to")
        rows = peers if isinstance(peers, list) else [peers]
        return [str(row) for row in rows if row is not None]

    def _duplicates_a_binding(reference: str) -> bool:
        parts = _endpoint(reference)
        if parts is None:
            return False
        requirement_id, port = parts
        requirement = by_id.get(requirement_id)
        if requirement is None:
            return False
        key = port.casefold()
        if _reference_port_name(key):
            return True
        # The compiler's own notion of a supply input, not just the token list: a regulator's
        # `input` is one, so `_supply_port_name` alone would miss it.
        supply_shaped = _supply_port_name(key) or key in _SUPPLY_PORTS
        return bool(requirement.get("supply")) and supply_shaped

    kept: list[dict] = []

    def _rail_entry_the_signal_carries(signal: dict) -> tuple[str, str, str] | None:
        """(rail, requirement id, port) when this duplicate signal is a rail's only entry.

        A signal that restates a rail its peer requirement already declares is dropped -- unless
        that rail names no source of its own (`from` null), in which case this signal *is* how the
        rail reaches the board, and dropping it takes the contact with it. Live 2026-09-25: the
        host 5 V input contact of a header disappeared exactly this way, leaving the board's input
        rail with no physical source and nothing in the diagnostics to show for it.
        """
        rail: str | None = None
        for peer in _peers(signal):
            parts = _endpoint(peer)
            requirement = by_id.get(parts[0]) if parts else None
            if requirement is None:
                continue
            key = parts[1].casefold()
            if (_supply_port_name(key) or key in _SUPPLY_PORTS) and requirement.get("supply"):
                rail = str(requirement["supply"])
        if rail is None:
            return None
        declared = ((payload.get("power") or {}).get("rails") or {}).get(rail)
        if not isinstance(declared, dict) or declared.get("from"):
            return None
        source = _endpoint(signal.get("from"))
        if source is None or source[0] not in by_id:
            return None
        return rail, source[0], source[1]

    dropped: list[str] = []
    rail_entries: list[tuple[str, str, str]] = []
    for signal in signals:
        peers = _peers(signal)
        if peers and all(_duplicates_a_binding(peer) for peer in peers):
            entry = _rail_entry_the_signal_carries(signal)
            if entry is not None:
                rail_entries.append(entry)
            dropped.append(str(signal.get("name") or ""))
            continue
        kept.append(signal)
    if not dropped and not any(
        isinstance(entry, dict)
        and (entry.get("supply_rail") or entry.get("reference_domain"))
        for row in requirements
        for entry in (row.get("declared_ports") or [])
    ):
        return payload

    carried = set()
    for signal in kept:
        for reference in [signal.get("from"), *_peers(signal)]:
            parts = _endpoint(reference) if reference is not None else None
            if parts is not None:
                carried.add((parts[0].casefold(), parts[1].casefold()))
    # A rail whose only entry was a signal that restated it keeps that contact: the tie names the
    # rail on the contact the signal named, which is the one statement the compiler cannot derive
    # from anywhere else. A contact a kept signal still names is left alone (the tie normalizer
    # would clear it and the compiler would refuse the two names on one pin).
    for rail, requirement_id, port in rail_entries:
        if (requirement_id.casefold(), port.casefold()) in carried:
            continue
        ties = by_id[requirement_id].setdefault("ties", {})
        if isinstance(ties, dict):
            ties.setdefault(port, rail)
    completed_requirements: list[dict] = []
    for requirement in requirements:
        declared = requirement.get("declared_ports")
        if not isinstance(declared, list):
            completed_requirements.append(requirement)
            continue
        requirement_id = str(requirement.get("id") or "").casefold()
        rows = []
        for entry in declared:
            if (
                isinstance(entry, dict)
                and (requirement_id, str(entry.get("key") or "").casefold()) in carried
                and (entry.get("supply_rail") or entry.get("reference_domain"))
            ):
                rows.append(
                    {
                        key: value
                        for key, value in entry.items()
                        if key not in ("supply_rail", "reference_domain")
                    }
                )
            else:
                rows.append(entry)
        completed_requirements.append({**requirement, "declared_ports": rows})
    completed = dict(payload)
    completed["signals"] = kept
    completed["requirements"] = completed_requirements
    return completed


def _supply_port_name(port: str) -> bool:
    """Whether a published port key *is* a supply input (`vdd`, `vbus_5v`)."""
    return any(port == token or port.startswith(f"{token}_") for token in _SUPPLY_TOKENS)


def _reference_port_name(port: str) -> bool:
    """Whether a published port key *is* the return (`gnd`, `gnd_field`)."""
    return port in _REFERENCE_PORTS or any(
        port.startswith(f"{token}_") for token in _REFERENCE_TOKENS
    )


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
            default_ties={
                port.name: port.default_tie
                for port in definition.ports
                if port.default_tie is not None
            },
        )
    if requirement.family in lowerers:
        lowerer = lowerers[requirement.family]
        names = {
            key.lower()
            for key in (
                *lowerer.port_keys,
                *lowerer.required_port_keys,
                *(key for key, _direction in lowerer.port_directions),
            )
            if key.isidentifier()
        }
        return _Catalog(
            directions={
                name: direction
                for name in names
                if (direction := lowerer_port_direction(lowerer, name)) is not None
            },
            source="lowerer",
            lowerer=lowerer,
        )
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


#: Two rails sourced from one port are the same node when their voltages agree this closely; the
#: module already reads the USB 5 V rail with this tolerance.
_RAIL_ALIAS_TOLERANCE_V = 0.5


def _merge_alias_rails(
    intent: ArchitectureIntent,
) -> tuple[ArchitectureIntent, list[tuple[str, str, str]]]:
    """Collapse rails that name one node twice: same normalized source port, same voltage.

    ``{"VBUS": {"from": "usb.vbus", "voltage": 5.0}, "+5V": {"from": "usb.vbus", ...}}`` is one
    node declared twice, and a port carries one net: the second rail's bind is refused
    (`conflicting_port_binding`), so a model that keeps re-emitting the same draft never gets a
    different answer. The first-declared key wins, every alias key is dropped, and every rail
    reference is rewritten to it. Returns the merged intent plus one ``(alias, canonical,
    from_ref)`` row per collapsed alias, for the caller's advisory.

    Deliberately narrow: only rails that declare a source port are considered, their voltages
    must agree within `_RAIL_ALIAS_TOLERANCE_V`, and an alias a signal names is left alone (that
    conflict is the separate `signal_conflicts_with_rail` refusal). A rail-vs-signal conflict, a
    voltage conflict, and two genuinely different source ports refuse exactly as before.
    """
    by_source: dict[str, list[str]] = {}
    for name, rail in intent.power.rails.items():
        if rail.from_ref is None:
            continue
        by_source.setdefault(_net_identity(rail.from_ref), []).append(name)
    signal_names = {_net_identity(signal.name) for signal in intent.signals}
    aliases: dict[str, str] = {}
    merges: list[tuple[str, str, str]] = []
    for names in by_source.values():
        canonical = names[0]
        for alias in names[1:]:
            if _net_identity(alias) in signal_names:
                continue
            if (
                abs(intent.power.rails[alias].voltage - intent.power.rails[canonical].voltage)
                > _RAIL_ALIAS_TOLERANCE_V
            ):
                continue
            aliases[alias] = canonical
            merges.append(
                (alias, canonical, str(intent.power.rails[canonical].from_ref))
            )
    if not aliases:
        return intent, []

    def canonical(name: str | None) -> str | None:
        return aliases.get(name, name)

    def rewrite(mapping: dict[str, str]) -> dict[str, str]:
        return {key: canonical(value) for key, value in mapping.items()}

    requirements = [
        row.model_copy(
            update={
                "supply": canonical(row.supply),
                "supply_bindings": rewrite(row.supply_bindings),
                "reference_bindings": rewrite(row.reference_bindings),
                "ties": rewrite(row.ties),
                "declared_ports": [
                    port.model_copy(
                        update={
                            "supply_rail": canonical(port.supply_rail),
                            "reference_domain": canonical(port.reference_domain),
                        }
                    )
                    for port in row.declared_ports
                ],
            }
        )
        for row in intent.requirements
    ]
    signals = [
        signal.model_copy(update={"rails": [canonical(name) for name in signal.rails]})
        for signal in intent.signals
    ]
    return (
        intent.model_copy(
            update={
                "requirements": requirements,
                "signals": signals,
                "power": intent.power.model_copy(
                    update={
                        "rails": {
                            name: rail
                            for name, rail in intent.power.rails.items()
                            if name not in aliases
                        }
                    }
                ),
            }
        ),
        merges,
    )


def derive_architecture(
    intent: ArchitectureIntent | dict,
    functional_spec: object | None = None,
) -> Architecture:
    """Turn an intent-shaped slot into the canonical `Architecture`.

    `functional_spec` is the committed functional-spec slot when the caller has it: a derived
    requirement claims the exact block that asked for it, and only that slot names the block.

    Raises `ArchitectureIntentError` carrying *every* blocking refusal at once, so a rejected draft
    names all the parts, ports and rails it could not derive wiring from in a single answer.
    """
    intent = ArchitectureIntent.model_validate(intent)
    recipes = tuple(registered_recipes())
    lowerers = {family: row for row in registered_lowerers() for family in row.families}
    diagnostics: list[IntentDiagnostic] = []
    advisories: list[IntentDiagnostic] = []

    def _fail(code: str, message: str, **kw) -> None:
        diagnostics.append(IntentDiagnostic(code=code, message=message, **kw))

    def _advise(code: str, message: str, **kw) -> None:
        """Record a property that could not be *proven*; the design still compiles.

        RECORD half of the BLOCK-vs-RECORD bar (design-yield-recovery plan §4.3): the board is
        very likely fine, what is missing is proof. The row is carried on
        `Architecture.advisories` and mirrored into `assumptions`, so the artifact says what the
        board shipped with; it is never a refusal and never capped (plan D6).
        """
        row = IntentDiagnostic(code=code, message=message, severity="advisory", **kw)
        advisories.append(row)
        derived_notes.append(f"advisory [{code}]: {message}")

    # Statements the compiler derived rather than read off the draft, and the signal renames a
    # rail that owns a pin forced: both are carried into the review's assumptions.
    renamed: list[str] = []
    derived_notes: list[str] = []

    # One node declared twice (two rails from one source port) is normalized before anything
    # reads `intent.power.rails`, so the draft designs the board it described instead of being
    # refused for saying the same thing twice.
    intent, rail_merges = _merge_alias_rails(intent)
    for alias, canonical_rail, from_ref in rail_merges:
        _advise(
            "rail_alias_merged",
            f"rail {alias!r} is the same node as {canonical_rail!r} "
            f"(both declared from {from_ref!r}); the design carries one rail",
        )

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
            standard_stacking_role=row.standard_stacking_role,
            family=row.family,
            # The recipe's own exact part is the identity the resolver would settle on anyway;
            # stating it here keeps the resolved payload explicit instead of inferred.
            exact_part=row.exact_part
            or (catalog.recipe.definition.exact_part if catalog.recipe is not None else None),
            parameters=dict(row.parameters),
            interfaces=list(row.interfaces),
            functional_blocks=list(row.functional_blocks),
            declared_interface=(
                DeclaredInterfaceClaim(ports=row.declared_ports)
                if catalog.source == "declared"
                else None
            ),
            obligations=list(row.obligations),
        )
    standard_port_bindings: dict[str, dict[str, str]] = {}
    # Requirements whose stacking pin/net map was refused: their authored `ties` are the same
    # wrong map, so reporting each net once more teaches nothing.
    refused_stacking: set[str] = set()
    stacking_requirements = [
        row for row in intent.requirements if row.standard_stacking_role is not None
    ]
    if intent.standard_form_factor is not None or stacking_requirements:
        template = get_template(intent.standard_form_factor)
        if template is None or not template.validated:
            _fail(
                "unsupported_standard_stacking_interface",
                "stacking headers require one approved, validated standard_form_factor template",
                evidence=[intent.standard_form_factor or "(none)"],
            )
        else:
            expected = {connector.role: connector for connector in template.fixed_connectors}
            actual = {row.standard_stacking_role: row for row in stacking_requirements}
            if len(actual) != len(stacking_requirements) or set(actual) != set(expected):
                _fail(
                    "incomplete_standard_stacking_interface",
                    (
                        f"standard {template.key!r} requires exactly its fixed connector roles; "
                        "do not replace them with a composite or arbitrary header"
                    ),
                    evidence=[*sorted(expected), *sorted(actual)],
                )
            # The four fixed connectors are ONE board interface, and the wiring stage may permute
            # which owner carries which template geometry, so their functional-block membership must
            # be identical: the proto-shield architecture gave the power connector an extra block
            # (that connector also serves POWER_INPUT) and the migration then refused the whole
            # netlist with "cannot redistribute standard ports across distinct functional owners".
            # Normalising to the block they all implement keeps the interface coherent; when they
            # agree on none, the refusal stays and names the disagreement.
            stacked = [requirements[row.id] for row in actual.values() if row.id in requirements]
            shared_blocks = (
                set.intersection(*(set(requirement.functional_blocks) for requirement in stacked))
                if stacked
                else set()
            )
            # Only an extra block that ANOTHER requirement still implements may be dropped:
            # a functional block with no implementation requirement is refused by the commit
            # ("functional block 'POWER_INPUT' has no implementation requirement on a sheet"),
            # so normalising must never orphan one. When the connector is that block's only
            # owner, its membership stays and the migration's own per-owner checks decide.
            orphans = {
                block
                for requirement in stacked
                for block in requirement.functional_blocks
                if block not in shared_blocks
                and not any(
                    block in other.functional_blocks
                    for other in requirements.values()
                    if other is not requirement
                )
            }
            if (
                shared_blocks
                and not orphans
                and any(
                    sorted(requirement.functional_blocks) != sorted(shared_blocks)
                    for requirement in stacked
                )
            ):
                for requirement in stacked:
                    requirement.functional_blocks = sorted(shared_blocks)
                derived_notes.append(
                    f"standard stacking connectors share the {sorted(shared_blocks)[0]!r} "
                    "interface block (derived)"
                )
            for role, connector in expected.items():
                row = actual.get(role)
                if row is None:
                    continue
                expected_ports = {
                    f"pin{index}": net for index, net in enumerate(connector.net_by_pin, start=1)
                }
                if (
                    row.id not in requirements
                    or row.role != "connector"
                    or row.family != "pin-header"
                    or row.exact_part is not None
                    or row.parameters != {"rows": 1, "gender": "female"}
                    or not row.functional_blocks
                ):
                    _fail(
                        "invalid_standard_stacking_owner",
                        (
                            f"stacking role {role!r} must be a generic female one-row pin-header "
                            "owned by an explicit stacking functional block"
                        ),
                        requirement_id=row.id,
                        sheet=row.sheet,
                    )
                    refused_stacking.add(row.id)
                    continue
                if row.ties and row.ties != expected_ports:
                    _fail(
                        "invalid_standard_stacking_pinmap",
                        f"stacking role {role!r} must use the template's exact pin/net map",
                        requirement_id=row.id,
                        sheet=row.sheet,
                        evidence=[f"{pin}={net}" for pin, net in expected_ports.items()],
                    )
                    refused_stacking.add(row.id)
                    continue
                if not row.ties:
                    # The template is the pin/net map; the role is the design statement. Derive the
                    # map the standard requires rather than asking for it back verbatim.
                    derived_notes.append(
                        f"{row.id}: stacking role {role!r} pinned from the "
                        f"{template.key!r} template (derived)"
                    )
                standard_port_bindings[row.id] = expected_ports

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
        # BLOCK, not RECORD, and measured rather than argued: with this downgraded the
        # requirement is not carried at all (no curated recipe, no declared interface, so it has
        # no catalog and never enters `requirements`), so the design ships with the part absent
        # from the netlist — and the obligation it owned then fails the ownership check one
        # validation later, under a message that no longer names the cause (2026-09-20: three of
        # three production drafts whose only advisory was this code died exactly that way). A
        # part with nothing implementing it is §4.1's BLOCK case, so the check that says so
        # stands; recording this class needs the architecture to carry an unverified interface
        # first, which is a move of its own.
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
            catalog = catalogs.get(requirement_id)
            # A supply or reference pin conflict has one repair, and it is not "bind another
            # port": the rail/ground already tied to this pin IS its net, so the duplicate
            # connection is what goes. Say that instead of sending the draft port-shopping.
            rail_bound = existing in intent.power.rails or existing == GND_NET
            if rail_bound and _supply_port_name(port):
                tail = (
                    f" and {existing!r} is the rail that feeds it: delete the connection that "
                    "duplicates the rail (a rail is distributed by declaring it under "
                    "power.rails, never by a second signal on the same pin)"
                )
            elif rail_bound and _reference_port_name(port):
                tail = (
                    f" and {existing!r} is the net tied to it: delete the connection that "
                    "duplicates it"
                )
            else:
                tail = ", so bind this connection to another of this requirement's ports"
            _fail(
                "conflicting_port_binding",
                (
                    f"{context}: port {port!r} of {requirement_id!r} is already bound to "
                    f"{existing!r} — one port carries one net"
                    + tail
                    + (f" ({catalog.choices})" if catalog is not None else "")
                ),
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

    # A standard template's pin names are the HOST's labels (D5, A2, IOREF, RESET...), not
    # nets of this board: the shield only realizes a name when its own circuits use it (the
    # rails it draws from) or when another requirement already binds that net. Binding the
    # rest would put a template label on a net with a single pin, which is exactly the
    # dangling label §9.15 refuses -- and wiring an unused host pin to nothing is the honest
    # netlist. The port stays (a connector's physical size comes from its port count), so the
    # pin becomes a no-connect instead of disappearing from the header.
    realized_nets = {
        net
        for requirement_id, ports in bindings.items()
        if requirement_id not in standard_port_bindings
        for net in ports.values()
    }
    rail_names = set(intent.power.rails)
    for requirement_id, ports in standard_port_bindings.items():
        for port, net in ports.items():
            # A power/ground pin of a standard shield IS that rail (the board is powered
            # through it), so it stays bound even when this design names its own rail
            # differently; only the signal labels are host-side names this board may never
            # realize.
            host_label = net != "NC" and not (
                net in realized_nets or net in rail_names or is_power_or_ground_name(net)
            )
            if net == "NC" or host_label:
                bindings[requirement_id][port] = "NC"
            else:
                _bind(
                    requirement_id,
                    port,
                    net,
                    "bidirectional",
                    context="standard stacking interface",
                )

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
        # Which ports the design's own signals bind, before any statement about them is read.
        signal_ports = {
            resolved[0]
            for item in signals
            for peer in (item.from_ref, *item.peers())
            if peer.partition(".")[0] == requirement_id
            and (resolved := _resolve_port(row, catalog, peer.partition(".")[2])) is not None
        }
        # `supply_rail`/`reference_domain` on a declared port name the net the PIN IS TIED TO.
        # A draft that writes `reference_domain: GND` (and `supply_rail: <its own supply>`) on
        # every port means the *domain* the signal belongs to, and the field name cannot tell the
        # two meanings apart. Where the compiler already knows the owner it takes it and reports
        # the domain statement instead of refusing the draft for it:
        #   * a port the design's own signals bind belongs to that signal — and a rail that *feeds*
        #     the requirement (`supply`), or ground, is exactly a domain statement about it;
        #   * a port that *is* a supply input keeps its rail and drops the reference, which is the
        #     precedence this compiler already applies to `supply_bindings`.
        # A statement the compiler cannot place either way stays a refusal below, and an unrelated
        # net on a signal's pin stays `conflicting_port_binding` (two nets, one pin).
        domain_statements: set[str] = set()
        reference_dropped_by_supply: set[str] = set()
        for port in row.declared_ports:
            if _supply_port_name(port.key.casefold()) and port.supply_rail:
                if port.reference_domain:
                    reference_dropped_by_supply.add(port.key)
                continue
            if port.key not in signal_ports:
                continue
            if port.supply_rail and port.supply_rail == row.supply:
                domain_statements.add(port.key)
            elif port.reference_domain == GND_NET:
                domain_statements.add(port.key)
        for port in row.declared_ports:
            if port.key in reference_dropped_by_supply:
                derived_notes.append(
                    f"{requirement_id}.{port.key}: stated as both a supply input "
                    f"({port.supply_rail!r}) and a reference; the supply input owns the pin (derived)"
                )
            elif port.key in domain_statements and port.key not in bindings[requirement_id]:
                derived_notes.append(
                    f"{requirement_id}.{port.key}: stated "
                    + ", ".join(
                        statement
                        for statement in (
                            f"supply_rail={port.supply_rail!r}" if port.supply_rail else "",
                            f"reference_domain={port.reference_domain!r}"
                            if port.reference_domain
                            else "",
                        )
                        if statement
                    )
                    + " on a port the design's own signals bind: the signal owns the pin, and the "
                    "statement is its domain (derived)"
                )
        declared_supply = {
            port.key: port.supply_rail
            for port in row.declared_ports
            if port.supply_rail and port.key not in domain_statements
        }
        declared_references = {
            port.key: port.reference_domain
            for port in row.declared_ports
            if port.reference_domain
            and port.key not in domain_statements
            and port.key not in reference_dropped_by_supply
        }
        supply_bindings = {**declared_supply, **row.supply_bindings}
        stated_references = {**declared_references, **row.reference_bindings}
        # Both fields are optional refinements of a binding the compiler derives anyway. When one
        # draft states both for a pin, the supply input is the load-bearing fact (the pin is the
        # rail it draws from) and the reference statement is dropped, reported, never refused.
        reference_bindings = {
            port: net for port, net in stated_references.items() if port not in supply_bindings
        }
        for port in sorted(set(stated_references) - set(reference_bindings)):
            derived_notes.append(
                f"{requirement_id}.{port}: stated as both a supply input "
                f"({supply_bindings[port]!r}) and a reference; the supply input owns the pin (derived)"
            )
        if set(declared_supply) & set(row.supply_bindings) and any(
            declared_supply[port] != row.supply_bindings[port]
            for port in set(declared_supply) & set(row.supply_bindings)
        ):
            _fail(
                "conflicting_supply_binding",
                f"requirement {requirement_id!r} gives a declared port two supply rails",
                requirement_id=requirement_id,
                sheet=requirement.sheet,
            )
        if set(declared_references) & set(row.reference_bindings) and any(
            declared_references[port] != row.reference_bindings[port]
            for port in set(declared_references) & set(row.reference_bindings)
        ):
            _fail(
                "conflicting_reference_binding",
                f"requirement {requirement_id!r} gives a declared port two reference domains",
                requirement_id=requirement_id,
                sheet=requirement.sheet,
            )
        # What the compiler could not place above stays a refusal, so the misuse the field name
        # invites is still named where it is written: a pin is one net, and a statement the design
        # never attaches to a signal is a claim about the net the pin carries.
        for port in row.declared_ports:
            if port.key in domain_statements or port.key in reference_dropped_by_supply:
                continue  # the compiler placed this statement: the owner is known
            if port.supply_rail and port.reference_domain:
                _fail(
                    "declared_port_double_bound",
                    f"declared port {port.key!r} of {requirement_id!r} sets both supply_rail "
                    f"({port.supply_rail!r}) and reference_domain ({port.reference_domain!r}), but a pin "
                    "is tied to ONE net: keep only the net this pin is actually tied to, and leave "
                    "reference_domain null on a signal port",
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                )
                continue
            if port.key in signal_ports and (port.supply_rail or port.reference_domain):
                _fail(
                    "declared_signal_port_tied",
                    f"declared port {port.key!r} of {requirement_id!r} carries a signal but sets "
                    f"supply_rail={port.supply_rail!r} / reference_domain={port.reference_domain!r}; "
                    "these name the net this PIN IS TIED TO (a supply input or a ground pin), not the "
                    "domain a signal is referenced to — leave both null on a signal port, or record a "
                    "genuinely strapped pin in `ties`",
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                )
                continue
        reference_port = _reference_port(catalog)
        for port_name, net in sorted(reference_bindings.items()):
            if net != GND_NET and net not in rail_names:
                _fail(
                    "unknown_reference_domain",
                    f"reference port {port_name!r} of {requirement_id!r} names undeclared domain {net!r}",
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                    evidence=[GND_NET, *sorted(rail_names)],
                )
                continue
            if net != GND_NET and intent.power.rails[net].voltage != 0:
                _fail(
                    "reference_domain_not_zero_volt",
                    f"reference port {port_name!r} of {requirement_id!r} names non-zero rail {net!r}",
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                )
                continue
            resolved = _resolve_port(row, catalog, port_name)
            if resolved is None:
                # The pin the draft named is not one the family publishes; the family's own
                # reference port is the pin it means, so the stated domain lands there instead of
                # refusing a name the compiler can derive.
                if reference_port is not None and reference_port not in bindings[requirement_id]:
                    _bind(
                        requirement_id,
                        reference_port,
                        net,
                        "bidirectional",
                        context="reference domain",
                    )
                    derived_notes.append(
                        f"{requirement_id}.{reference_port}: {net} (stated as {port_name!r}, "
                        "which the family does not publish) (derived)"
                    )
                continue
            _bind(requirement_id, resolved[0], net, resolved[1], context="reference domain")
        if reference_port is not None and reference_port not in bindings[requirement_id]:
            _bind(requirement_id, reference_port, GND_NET, "bidirectional", context="legacy ground")
        deferred_rails: list[tuple[str, str]] = []
        for port_name, rail in sorted(supply_bindings.items()):
            if rail not in rail_names:
                _fail(
                    "unknown_supply_rail",
                    f"supply port {port_name!r} of {requirement_id!r} names undeclared rail {rail!r}",
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                    evidence=sorted(rail_names),
                )
                continue
            resolved = _resolve_port(row, catalog, port_name)
            if resolved is None:
                # The rail is the intent and the pin name is bookkeeping: take the family's own
                # supply port for it below rather than refusing a name the compiler derives.
                deferred_rails.append((rail, port_name))
                continue
            _bind(requirement_id, resolved[0], rail, "input", context="explicit supply")
        if row.supply is not None:
            if row.supply not in rail_names:
                _fail(
                    "unknown_supply_rail",
                    (
                        f"requirement {requirement_id!r} declares supply {row.supply!r}, which is not "
                        "a declared rail under power.rails"
                    ),
                    requirement_id=requirement_id,
                    sheet=requirement.sheet,
                    evidence=sorted(rail_names),
                )
                continue
            if requirement.role == "connector":
                # A connector exposes the rail on its supply pin later, with its ground.
                connector_rails[requirement_id] = row.supply
            else:
                deferred_rails.append((row.supply, "supply"))
        for rail, stated in dict.fromkeys(deferred_rails):
            port = _supply_port(catalog, rail)
            if port is None:
                # The family publishes no supply port at all: a status LED draws its current from
                # its own drive signal, not from a rail pin. Nothing is invented and nothing is
                # refused — the part simply has no pin on this rail.
                derived_notes.append(
                    f"{requirement_id}: declares supply {rail} ({stated}), but its "
                    f"{catalog.source} interface publishes no supply port ({catalog.choices}); "
                    "no pin bound (derived)"
                )
                continue
            bound = bindings[requirement_id].get(port)
            if bound is not None:
                # The authored binding owns the pin; a stated supply that would need the same pin
                # for another rail is reported rather than silently dropped.
                if bound != rail:
                    derived_notes.append(
                        f"{requirement_id}: supply {rail} ({stated}) is not bound — its supply port "
                        f"{port} already carries {bound} (derived)"
                    )
                continue
            _bind(requirement_id, port, rail, "input", context="legacy supply")
            if stated != "supply":
                derived_notes.append(
                    f"{requirement_id}: supply {rail} stated on unpublished port {stated!r}; bound "
                    f"to the family's own {port} (derived)"
                )

    # A requirement that must be realized by a reviewed part may not pin an exact identity the
    # reviewed library does not hold: the BOM stage is deterministic for these requirements, so
    # it cannot repair the choice and refuses with "found 0 ... exact MPN/symbol/footprint
    # evidence" and no correction round left (the proto-shield runs answered the demanded
    # `voltage-regulator` with the familiar but unreviewed `AMS1117-3.3`). Refusing here keeps
    # the correction where the part is still being chosen, and the evidence names the reviewed
    # options. A class with no reviewed coverage at all is left alone: naming the exact part is
    # legitimately its only route.
    from kicraft.design.part_identity import (
        canonical_physical_features,
        reviewed_part,
        reviewed_parts_for_feature,
    )

    for requirement_id, requirement in requirements.items():
        exact = str(requirement.exact_part or "").strip()
        if not exact or reviewed_part(exact) is not None:
            continue
        for obligation in requirement.obligations:
            component_class = str(getattr(obligation, "component_class", "") or "").casefold()
            if obligation.kind != "physical" or not component_class:
                continue
            options = {
                part.identity: part
                for feature in canonical_physical_features(component_class)
                for part in reviewed_parts_for_feature(feature)
            }
            if not options:
                continue  # a class the library does not cover: the exact part is its route
            _advise(
                "unreviewed_exact_part",
                (
                    f"requirement {requirement_id!r} pins exact_part {exact!r}, which the reviewed "
                    f"library does not hold, while its {component_class!r} obligation must be "
                    "realized by a reviewed part; recorded (not refused) — the part is shipped as "
                    "chosen and the reviewed alternatives are listed for review"
                ),
                requirement_id=requirement_id,
                sheet=requirement.sheet,
                evidence=[f"exact_part={exact}", *sorted(options)],
            )
            break

    # A committed `fabrication` obligation naming the prototyping area is a board feature no
    # requirement can implement: it names no component class, owns no pin and draws no net, so a
    # model that states it as prose only leaves its sheet with nothing to build. The obligation is
    # the whole statement, so the sheet, the requirement that makes the field buildable, and the
    # field's default geometry are derived here -- deterministic input, deterministic output, the
    # same way a board-edge connector is written from an `edge:` peer. A model that already
    # declared the sheet or the requirement keeps exactly what it wrote, and a design whose
    # obligations never named the feature is untouched. The requirement carries no port: the pad
    # field shares no net with the circuit, and the obligation itself stays a top-level row
    # (`fabrication` owns no requirement).
    stated_obligations = [
        *intent.obligations,
        *(row for requirement in intent.requirements for row in requirement.obligations),
    ]
    modelled_prototyping_area = any(
        row.id == PROTOTYPING_AREA_OBLIGATION_ID or row.family == PROTOTYPING_AREA_FAMILY
        for row in intent.requirements
    )
    if has_prototyping_area(stated_obligations) and not modelled_prototyping_area:
        sheet = sheet_by_stem.get(PROTOTYPING_AREA_SHEET_STEM)
        if sheet is None:
            sheet = Sheet(
                name=PROTOTYPING_AREA_SHEET_NAME,
                stem=PROTOTYPING_AREA_SHEET_STEM,
                function=PROTOTYPING_AREA_FUNCTION,
            )
            sheets.append(sheet)
            sheet_by_stem[sheet.stem] = sheet
            sheet_names.add(sheet.name)
        # The block that asked for the field is this requirement's implementation claim, so the
        # derivation claims its committed name rather than leaving the block unowned. A functional
        # spec that never emitted one leaves the list empty: no block name may be invented.
        block = _prototyping_area_block(functional_spec)
        requirements[PROTOTYPING_AREA_OBLIGATION_ID] = CircuitRequirement(
            id=PROTOTYPING_AREA_OBLIGATION_ID,
            sheet=sheet.name,
            role="user_io",
            family=PROTOTYPING_AREA_FAMILY,
            parameters=dict(PROTOTYPING_AREA_PARAMETERS),
            functional_blocks=[block] if block is not None else [],
        )
        bindings[PROTOTYPING_AREA_OBLIGATION_ID] = {}
        derived_notes.append(
            f"{PROTOTYPING_AREA_OBLIGATION_ID}: sheet {sheet.name!r} and a "
            f"{PROTOTYPING_AREA_PARAMETERS['rows']}x{PROTOTYPING_AREA_PARAMETERS['cols']} pad "
            f"field at {PROTOTYPING_AREA_PARAMETERS['pitch_mm']} mm derived from the committed "
            f"{PROTOTYPING_AREA_FEATURE!r} fabrication obligation (derived)"
        )
    elif has_prototyping_area(stated_obligations):
        # The model declared the field itself: its sheet, id and role are kept, but the field's
        # SIZE and its netless nature are the reviewed contract, not a free choice. A real run
        # asked for 10x15 (150 plated pads) and its board was the one the layout engine could not
        # route ("no routed parent board"), while the acceptance check asks for a usable field of
        # at least 25 positions. A port bound to the field is dropped for the same reason: the pad
        # field shares no net with the circuit, and no requirement can own one.
        for requirement_id, requirement in list(requirements.items()):
            # Only a requirement that IS the pad field (its family), never one that merely
            # reuses the obligation id while declaring some other family: that is a different
            # model choice, reported by the gates that own it.
            if requirement.family != PROTOTYPING_AREA_FAMILY:
                continue
            if dict(requirement.parameters) != dict(PROTOTYPING_AREA_PARAMETERS):
                requirement.parameters = dict(PROTOTYPING_AREA_PARAMETERS)
                derived_notes.append(
                    f"{requirement_id}: pad field set to the reviewed "
                    f"{PROTOTYPING_AREA_PARAMETERS['rows']}x{PROTOTYPING_AREA_PARAMETERS['cols']} "
                    f"at {PROTOTYPING_AREA_PARAMETERS['pitch_mm']} mm (derived)"
                )
            if requirement.ports or bindings.get(requirement_id):
                requirement.ports = {}
                bindings[requirement_id] = {}
                derived_notes.append(
                    f"{requirement_id}: pad field declares no port — a bare pad shares no net "
                    "with the circuit (derived)"
                )

    # Off-board peers: one connector per `edge:` label, carrying only what named it.
    edge_connectors: dict[str, tuple[str, str, list[str]]] = {}  # label -> (id, mode, rails)
    # A label that cannot be opened says so once: two signals asking for the same
    # impossible edge must not each repeat the same refusal.
    failed_edges: set[str] = set()

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
        source_recipe_name = source_recipe.definition.recipe if source_recipe is not None else None
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
                compiler_origin="edge_connector",
                parameters={"rows": 1, "gender": "male"},
                functional_blocks=list(source.requirement.functional_blocks),
            )
        edge_connectors[label] = (requirement_id, mode, [])
        return requirement_id, mode

    def _next_pin(requirement_id: str) -> str:
        taken = bindings.setdefault(requirement_id, {})
        index = 1
        while f"pin{index}" in taken:
            index += 1
        return f"pin{index}"

    def _join_peers(
        signal: IntentSignal,
        source: _SheetRef,
        net: str,
        *,
        peer_direction: str | None = None,
        bind_edge_pins: bool = True,
    ) -> None:
        """Bind this signal's peers to `net`: a connector contact, or the peer's own port.

        `peer_direction` overrides the peer's own direction (a net that is a rail or ground lands
        bidirectional/input on the sheet it reaches); `bind_edge_pins` is false when the net
        already owns the connector — a rail is exposed by that connector's own pin plan.
        """
        for peer in signal.peers():
            label = _edge_label(peer)
            if label is not None:
                opened = _open_edge(label, source)
                if opened is None:
                    continue
                connector_id, mode = opened
                if not bind_edge_pins:
                    if net in rail_names:
                        edge_connectors[label][2].append(net)
                    continue
                if mode == "usb":
                    port = "usb_dm" if source.port == "usb_dm" else "usb_dp"
                    _bind(
                        connector_id,
                        port,
                        net,
                        "bidirectional",
                        context=f"signal {signal.name!r}",
                    )
                else:
                    _bind(
                        connector_id,
                        _next_pin(connector_id),
                        net,
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
                net,
                peer_direction or target.direction,
                context=f"signal {signal.name!r}",
            )

    for signal in signals:
        source = _lookup(signal.from_ref, context=f"signal {signal.name!r}")
        if source is None:
            continue
        # A port that already carries a net: the design said the same connection twice, and one
        # pin cannot be on two nets. A declared rail or ground owns the net (the implicit binding
        # plus a named wire); two *signals* out of one physical pin are one net under two names, so
        # the first name owns it and this signal's peers join it, reported rather than refused.
        # Two different *sinks* naming one port is untouched below and stays a refusal: that is two
        # nets on one pin.
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
            _join_peers(
                signal,
                source,
                source_net,
                peer_direction="bidirectional" if source_net == GND_NET else "input",
                bind_edge_pins=False,
            )
            continue
        if source_net is not None:
            if signal.name in rail_names or signal.name == GND_NET:
                _fail(
                    "signal_conflicts_with_rail",
                    (
                        f"signal {signal.name!r} leaves {signal.from_ref}, which the design already "
                        f"ties to net {source_net!r}; a pin is one net, and a rail is wired from "
                        "`power.rails`, not from `signals`"
                    ),
                    requirement_id=source.requirement.id,
                    sheet=source.requirement.sheet,
                )
                continue
            if signal.name != source_net:
                renamed.append(
                    f"{signal.name}: peers of {signal.from_ref} join net {source_net} "
                    "(the net that already owns the pin) (derived)"
                )
            _join_peers(signal, source, source_net)
            continue
        if signal.name in rail_names or signal.name == GND_NET:
            # A connector contact that the design's own signal names after a declared net *is*
            # that net's exposure on the board: the compiler already models exactly this shape
            # when `power.rails[net].from_ref` names the connector pin, and the draft stating it
            # with a signal names a declared net on a pin it also named. Take it — the contact
            # carries the net and the signal's peers join it — instead of refusing the exposure
            # and then refusing the connector again for the contact the refusal left unbound.
            # Everywhere else a signal still may not *be* a rail.
            if source.requirement.role == "connector" and signal.name != GND_NET:
                _bind(
                    source.requirement.id,
                    source.port,
                    signal.name,
                    "input",
                    context=f"signal {signal.name!r}",
                )
                _join_peers(
                    signal,
                    source,
                    signal.name,
                    peer_direction="input",
                    bind_edge_pins=False,
                )
                derived_notes.append(
                    f"{source.requirement.id}.{source.port}: carries rail {signal.name!r} "
                    f"(named by signal {signal.name!r}) (derived)"
                )
                continue
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
        _join_peers(signal, source, signal.name)

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
        # A connector the design named can carry signals, rails, or both: a signal whose source
        # port already carries a declared rail names the edge for that rail, so "no signal pins"
        # is not "empty" while there is a rail to expose. Only a connector with neither is empty.
        pinned = bindings.get(connector_id) or {}
        if not pinned and not rails:
            _fail(
                "empty_edge_connector",
                f"edge {label!r} was named by a signal that could not be resolved to a source port",
            )
            continue
        if mode == "usb":
            missing = sorted(port for port in _USB_DATA_PORTS if not pinned.get(port))
            if missing:
                _fail(
                    "incomplete_usb_edge",
                    (
                        f"edge {label!r} is a USB data connector but no signal bound {missing}; a "
                        "USB socket needs both usb_dm and usb_dp"
                    ),
                    requirement_id=connector_id,
                    evidence=sorted(pinned),
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
        if requirement_id in standard_port_bindings:
            continue  # exact template map was validated and bound above
        if requirement_id in refused_stacking:
            continue  # the pinmap refusal above already names this requirement's map
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

    # A factory-native-programmable MCU's USB data pair is not an optional signal: the part is
    # flashed and recovered over it, and the reviewed satisfier is a registered recipe. Asking
    # the draft for the connector refused live run 1 -- an RP2040 current-sense brief that never
    # mentioned USB -- with `unbound_required_port: requirement 'rp2040' port 'usb_dm' is
    # required by rp2040-minimal@2 and nothing wires it`. The draft was not wrong; the brief
    # never named USB. The compiler therefore completes the connector the same way it completes
    # a published supply contact, and records why the part is on the board. A design that wires
    # its own USB data pair is untouched: `_open_edge` already built that connector from its
    # signal, and an authored tie owns its port before this runs.
    if not any(row.family == _USB_DEVICE_CONNECTOR_FAMILY for row in requirements.values()):
        for requirement_id, catalog in sorted(catalogs.items()):
            if catalog.source != "recipe" or catalog.recipe is None:
                continue
            if catalog.recipe.definition.recipe not in _NATIVE_USB_MCU_RECIPES:
                continue
            if all(bindings.get(requirement_id, {}).get(port) for port in _USB_DATA_PORTS):
                continue
            connector_id = f"{requirement_id}_usb"
            opened = _open_edge(
                connector_id.upper(),
                _SheetRef(models_by_id[requirement_id], "usb_dm", "bidirectional"),
                connector_id,
            )
            if opened is None:
                continue  # `_open_edge` named the missing 5 V rail it needs
            connector_id, _mode = opened
            for port, net in (("usb_dm", "USB_DM"), ("usb_dp", "USB_DP")):
                # A partially wired pair keeps the net the design chose for the line it did
                # state; only the line nothing wired gets the conventional name.
                already = bindings.get(requirement_id, {}).get(port)
                if already is None:
                    _bind(requirement_id, port, net, "bidirectional", context="derived USB data")
                    already = net
                _bind(connector_id, port, already, "bidirectional", context="derived USB data")
            derived_notes.append(
                f"{requirement_id}: native-USB MCU needs a data connector; bound "
                f"{_USB_DEVICE_CONNECTOR_RECIPE} ({connector_id}) (derived)"
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
    # design bound first, then GND, then the rail the connector declares as its supply — on the
    # pin the family (or the model's declared interface) documents for it, when the design has
    # not already claimed that pin for something else.
    for requirement_id, rail in connector_rails.items():
        if not any(bound == GND_NET for bound in bindings.get(requirement_id, {}).values()):
            _bind(
                requirement_id,
                _next_pin(requirement_id),
                GND_NET,
                "bidirectional",
                context="connector ground",
            )
        pinned = bindings.get(requirement_id) or {}
        documented = _supply_port(catalogs[requirement_id], rail)
        port = documented if documented is not None and documented not in pinned else None
        _bind(
            requirement_id,
            port or _next_pin(requirement_id),
            rail,
            "input",
            context="connector supply",
        )

    # A lowerer publishes the contacts its own build code needs. Bind the ones the design's own
    # rails and references already answer before the contract check reads the port set: a published
    # supply contact takes the requirement's rail, a published reference contact takes ground.
    # Anything else stays unbound, and the contract diagnostic below names what is missing — the
    # compiler never invents a signal net for a port the model did not wire.
    for requirement_id, catalog in sorted(catalogs.items()):
        if catalog.source != "lowerer" or catalog.lowerer is None:
            continue
        model = models_by_id.get(requirement_id)
        if model is None:
            continue  # a connector the compiler built has no intent of its own
        rail = model.supply if model.supply in rail_names else None
        bound = bindings.setdefault(requirement_id, {})
        for key in catalog.lowerer.required_port_keys:
            port = key.lower()
            if port in bound or port not in catalog.directions:
                continue
            if _supply_port_name(port) and rail is not None:
                _bind(requirement_id, port, rail, "input", context="published supply port")
                derived_notes.append(f"{requirement_id}: published {port} carries {rail} (derived)")
            elif _reference_port_name(port) or port in catalog.lowerer.reference_port_keys:
                _bind(
                    requirement_id,
                    port,
                    GND_NET,
                    "bidirectional",
                    context="published reference port",
                )

    # A port the reviewed recipe requires and nothing wired is a missing design statement, not
    # a bookkeeping slip: the part cannot work without it. Name it here, once, instead of
    # letting the part reach BOM expansion and fail on a port binding mismatch.
    # Registered lowerers own a finite published contract. Once all architecture
    # bindings exist, reject an invalid known contract here rather than treating
    # its deterministic coverage miss as model-owned BOM work.
    for requirement_id, catalog in sorted(catalogs.items()):
        if catalog.source != "lowerer":
            continue
        candidate = requirements[requirement_id].model_copy(
            update={"ports": bindings.get(requirement_id, {})}
        )
        diagnostic = lowerer_contract_diagnostic(candidate)
        if diagnostic is not None:
            _fail(
                "unsupported_lowerer_contract",
                diagnostic.message,
                requirement_id=requirement_id,
                sheet=candidate.sheet,
                evidence=diagnostic.evidence,
            )

    # A published control port whose recipe declares a default tie is tied to the port it names
    # when the design does not wire it: the fleet's always-on regulator keeps EN on its input
    # rail and the follower's inverting input on its output, which is what the recipes emitted
    # before the port existed. A brief that asks for the controllable configuration binds the
    # port instead, and that binding wins.
    for requirement_id, catalog in sorted(catalogs.items()):
        if catalog.source != "recipe":
            continue
        bound = bindings.setdefault(requirement_id, {})
        for port, target in sorted(catalog.default_ties.items()):
            if bound.get(port):
                continue
            net = bound.get(target)
            if net is None:
                continue  # the tie target is itself unwired: its own refusal names it
            _bind(requirement_id, port, net, "bidirectional", context="recipe default tie")
            derived_notes.append(f"{requirement_id}: {port} tied to {target} ({net}) (derived)")

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
        standard_form_factor=intent.standard_form_factor,
        mcu_present=intent.mcu_present,
        sheets=sheets,
        power_nets=[GND_NET, *sorted(rail_names)],
        inter_sheet_nets=_inter_sheet_nets(final_requirements, endpoints, rail_names),
        assumptions=[*intent.assumptions, *derived_notes, *renamed, *notes],
        advisories=[
            ArchitectureAdvisory(
                code=row.code,
                message=row.message,
                evidence=[
                    *([f"requirement={row.requirement_id}"] if row.requirement_id else []),
                    *([f"sheet={row.sheet}"] if row.sheet else []),
                    *row.evidence,
                ],
            )
            for row in advisories
        ],
        requirements=final_requirements,
        obligations=list(intent.obligations),
        declared_interfaces=declared,
    )


def _usb_vbus_rail(intent: ArchitectureIntent) -> str | None:
    """The 5 V rail a USB socket exposes, never a guess.

    Preference: the rail the board's own *USB* power input generates (that is the
    host's VBUS, exactly what a USB socket's VBUS pin carries), then a single rail
    named VBUS/+5V, then a single rail at ~5 V. Two candidates at any step mean the
    design has not said which rail the socket exposes, and the model is asked rather
    than guessed at.

    The board's other inputs are deliberately NOT candidates: a 12 V barrel jack or a
    screw terminal generating a rail is not a USB supply, and putting the jack's rail on
    a socket's VBUS describes a board that does not survive being plugged in (live run
    904's RP2040 brief is powered from 12 V DC). Such a design is refused with the rail
    it must declare instead.
    """
    usb_input_ids = {
        row.id
        for row in intent.requirements
        if row.role == "power_input" and "usb" in row.family.casefold()
    }
    from_input = [
        net
        for net, rail in intent.power.rails.items()
        if rail.from_ref and rail.from_ref.partition(".")[0] in usb_input_ids
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


ArchitectureIntent.model_rebuild()
