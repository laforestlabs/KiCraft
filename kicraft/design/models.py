"""Pydantic models for the KiCraft conversation state.

One slot per stage (intent / functional_spec / architecture / bom) plus a
Question type used by every stage to surface clarifications and an
ArtifactPaths type set by the synthesis stage.

Validation rules mirror the hard requirements from
`docs/kicraft_schematic_prompt.md` so invalid state cannot reach synthesis.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REF_RE = re.compile(r"^[A-Z]+[0-9]+[A-Z0-9_-]*$")
FOOTPRINT_RE = re.compile(r"^[A-Za-z0-9_.+-]+:[A-Za-z0-9_.,+-]+$")
SYMBOL_RE = re.compile(r"^[A-Za-z0-9_.+-]+:[A-Za-z0-9_.+-]+$")
SHEET_NAME_RE = re.compile(r"^[A-Z0-9](?:[A-Z0-9 ]*[A-Z0-9])?$")
SHEET_STEM_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]*$")
# `'` allows prime-notation pin numbers (e.g. a transformer's 1'/2', or LCSC
# symbols that label mirrored pads 1'/2'); the net-coverage check still enforces
# that the pin actually exists on the symbol, so widening this can't mask a typo.
PIN_NUMBER_RE = re.compile(r"^[A-Za-z0-9+~_/.\-']+$")

POWER_NET_PATTERNS = [
    # A leading sign is optional so negative rails (-12V, -5V, -3.3V) classify
    # as power too — the op-amp/audio/analog dual-supply case. Without the `-`
    # the negative rail never gets a PWR_FLAG and KiCad ERC flags VCC- as
    # undriven (self-eval #28 audio-jack-buffer). See is_power_or_ground_name.
    re.compile(r"^[+-]?\d+\.?\d*V$", re.IGNORECASE),
    re.compile(r"^[+-]?\d+V\d+$", re.IGNORECASE),  # covers -3V3
    # VEE = negative supply, VSS = negative/ground reference; router and
    # placement already special-case both for the ground-symbol choice.
    re.compile(r"^V(CC|DD|BAT|BUS|SYS|IN|OUT|EE|SS)\b", re.IGNORECASE),
    # VDD_3V3, VCC_5V, VBUS_RAW, etc. — locally-named supply nets that the
    # canonical patterns (bare 3V3 / ^VDD\b) miss because _ is a word char.
    re.compile(r"^V(CC|DD|BAT|BUS|SYS|IN|OUT|EE|SS)_", re.IGNORECASE),
]
GND_NET_PATTERNS = [
    re.compile(r"^(P|A|D)?GND$", re.IGNORECASE),
    re.compile(r"_GND$"),
]


PinDirection = Literal["input", "output", "bidirectional", "passive"]
BlockCategory = Literal["sense", "process", "drive", "power", "interface", "mechanical"]
# How a 2-pin passive relates to the IC it serves — drives schematic placement
# (which side of the anchor it sits on, which way it's rotated, and what its far
# pin ties to). See PlacementHint and synthesis/placement.py.
PlacementRole = Literal[
    "decoupling",  # local bypass cap: rail pin <-> gnd pin, hugs a power pin
    "bulk",  # large reservoir cap: rail <-> gnd, like decoupling
    "pullup",  # resistor: signal pin <-> a positive rail
    "pulldown",  # resistor: signal pin <-> ground
    "series",  # in-line R/L/ferrite in a signal/power path
    "feedback",  # divider / compensation around the IC
    "other",  # cluster near the anchor, no special orientation
]
SignalType = Literal["power", "ground", "digital", "analog", "clock", "bus", "rf", "other"]
EdgeZone = Literal["left", "right", "top", "bottom"]
CornerZone = Literal["top-left", "top-right", "bottom-left", "bottom-right"]
BoardZone = Literal["top", "bottom"]


def is_power_or_ground_name(name: str) -> bool:
    """Match §2.5 of the contract doc — names KiCraft auto-classifies as power."""
    stripped = name.lstrip("/")
    for pat in POWER_NET_PATTERNS + GND_NET_PATTERNS:
        if pat.search(stripped):
            return True
    return False


class Question(BaseModel):
    """A clarification a stage wants the user to answer.

    blocking: user must answer before the stage can produce useful output.
    material: not blocking but should be raised at the next stage boundary.
    cosmetic questions (blocking=False, material=False) are silently
    defaulted; the chosen default is recorded in default_applied AND in the
    owning slot's `assumptions` list.

    options: suggested answers the UI may offer as buttons. The UI always also
    offers a freeform text answer, so options are never exhaustive.
    answer: the user's response once given (None while still open).

    reconcile_target: when set (only "bom" today), this is NOT a question for
    the user — it is a stage-internal deficit the pipeline can discharge itself
    by re-driving the named stage. The wiring stage sets it to "bom" when the
    only thing blocking full net coverage is that the BOM lacks supporting
    passives an IC requires (e.g. too few decoupling caps); the driver then
    re-runs the BOM stage to provision the parts and re-runs wiring, instead of
    stalling on a clarifying question KiCraft can answer for itself.
    """

    text: str
    stage: str
    blocking: bool = False
    material: bool = True
    default_applied: str | None = None
    options: list[str] = Field(default_factory=list)
    answer: str | None = None
    reconcile_target: str | None = None


class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------- Stage 1: Intent ----------


# Parametric outline shapes that map one-to-one onto
# ``layout_editor.outline.OutlineSpec``. Anything else in ``FormFactor.shape``
# (``hexagon``, ``snowman``, ...) is a named-library shape the shapes module
# expands to a polygon downstream; unknown names are tolerated here and degrade
# to a warning at synthesis rather than bricking the intent commit.
PARAMETRIC_OUTLINE_SHAPES: tuple[str, ...] = (
    "rect",
    "rounded_rect",
    "circle",
    "chamfered_rect",
)


class FormFactor(BaseModel):
    """Requested board outline shape, captured from the brief at the intent
    stage and resolved to concrete ``Edge.Cuts`` geometry downstream
    (autoplacer + parent compose). ``shape`` is either a parametric shape
    (:data:`PARAMETRIC_OUTLINE_SHAPES`) or a named-library shape. Validation is
    deliberately lenient: an unknown shape name is NOT rejected (a brief may ask
    for a novel shape), mirroring the placement-rules leniency -- it degrades to
    a warning when the shapes module cannot resolve it at synthesis."""

    model_config = ConfigDict(extra="forbid")

    shape: str = "rect"
    corner_radius_mm: float | None = None  # rounded_rect
    chamfer_mm: float | None = None  # chamfered_rect
    size_mm: float | None = None  # headline dimension the brief stated (advisory)
    note: str | None = None  # the phrase that triggered the classification
    # A named standard mechanical form factor (e.g. "arduino_uno_shield") the
    # brief requested -- a HARD outline + fixed-connector-position contract,
    # unlike the advisory ``shape``/``size_mm``. Resolved via
    # ``kicraft.form_factors``; None means no standard was requested. PR2 honors
    # it in placement/compose; until then it only surfaces the intent.
    standard: str | None = None

    @field_validator("shape")
    @classmethod
    def _normalize_shape(cls, v: str) -> str:
        v = (v or "rect").strip().lower()
        return v or "rect"

    @field_validator("corner_radius_mm", "chamfer_mm", "size_mm")
    @classmethod
    def _non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("form-factor dimensions must be >= 0 mm")
        return v


class IntentSlot(BaseModel):
    goal: str
    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Every explicit package, quantity, voltage, frequency, interface, "
            "inclusion, exclusion, and mechanical requirement stated by the user. "
            "Do not leave empty when the brief contains any such requirement."
        ),
    )
    named_parts: list[str] = Field(
        default_factory=list,
        description=(
            "Every exact MPN, IC, module, connector, battery, or named component "
            "family stated by the user. Do not leave empty when one is named."
        ),
    )
    inferred_expertise: Literal["beginner", "intermediate", "expert"] = "intermediate"
    assumptions: list[str] = Field(default_factory=list)
    # Requested non-rectangular board shape, when the brief asks for one. Set by
    # the intent stage (LLM + a deterministic extractor at stage-commit). None /
    # shape "rect" means a conventional rectangular board.
    form_factor: FormFactor | None = None
    # Typed requirements captured from the original brief. Constraints/named
    # parts remain explanatory; they are not a substitute for these facts.
    obligations: "list[RequirementObligation]" = Field(default_factory=list)


# ---------- Stage 2: Functional spec ----------


class FunctionalBlock(BaseModel):
    name: str
    category: BlockCategory
    purpose: str
    # Number of identical instances of this block the design needs (e.g. 3 for
    # "3 axes of stepper drivers"). The architecture stage expands a count>1
    # block into ``count`` sheets sharing a ``replication_group`` so the layout
    # solves ONE and reuses its placement+routing for the rest.
    count: int = 1
    # The original obligations this functional block is responsible for.
    obligation_ids: list[str] = Field(default_factory=list)

    @field_validator("count")
    @classmethod
    def _count_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"FunctionalBlock.count must be >= 1, got {v}")
        return v


class BlockConnection(BaseModel):
    from_block: str
    to_block: str
    signal_type: SignalType
    description: str = ""


class FunctionalSpec(BaseModel):
    blocks: list[FunctionalBlock]
    connections: list[BlockConnection] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    # The functional decomposition carries the same typed original facts into
    # architecture; the architecture model verifies requirement ownership.
    obligations: "list[RequirementObligation]" = Field(default_factory=list)

    @model_validator(mode="after")
    def _block_names_unique(self):
        names = [b.name for b in self.blocks]
        if len(names) != len(set(names)):
            raise ValueError("FunctionalSpec block names must be unique")
        return self

    @model_validator(mode="after")
    def _connections_reference_known_blocks(self):
        names = {b.name for b in self.blocks}
        for c in self.connections:
            if c.from_block not in names:
                raise ValueError(f"connection from unknown block {c.from_block!r}")
            if c.to_block not in names:
                raise ValueError(f"connection to unknown block {c.to_block!r}")
        return self

    @model_validator(mode="after")
    def _connections_unique(self):
        seen: set[tuple[str, str, SignalType]] = set()
        for connection in self.connections:
            key = (connection.from_block, connection.to_block, connection.signal_type)
            if key in seen:
                raise ValueError(
                    "duplicate functional connection "
                    f"{connection.from_block!r} -> {connection.to_block!r} "
                    f"({connection.signal_type})"
                )
            seen.add(key)
        return self


# ---------- Stage 3: Architecture ----------


class Sheet(BaseModel):
    name: str
    stem: str
    function: str
    # Set when the LLM elects to reuse a Leaf Library entry verbatim for
    # this sheet. ``from_library`` is "<name>@<version>"; ``library_instance``
    # disambiguates multiple uses of the same leaf (1 for the first, 2 for
    # the second, etc.). Both None for from-scratch sheets.
    from_library: str | None = None
    library_instance: int | None = None
    # Replication: from-scratch sheets that are structurally identical (e.g.
    # STEPPER_AXIS_X/Y/Z) share a ``replication_group`` key and carry a 1-based
    # ``replication_instance``. The layout solves instance 1 (the representative)
    # and reuses its placement+routing for the rest, remapping refs/nets. Each
    # sheet still has its own distinct refs and nets (so ERC sees N independent
    # circuits) -- only the geometry is shared. Both None for unique sheets.
    replication_group: str | None = None
    replication_instance: int | None = None

    @field_validator("name")
    @classmethod
    def _name_shape(cls, v: str) -> str:
        if not SHEET_NAME_RE.match(v):
            raise ValueError(
                f"Sheet.name {v!r} must contain only uppercase letters, digits, and optional spaces (e.g. 'USB INPUT' or '5V BUCK')"
            )
        return v

    @field_validator("stem")
    @classmethod
    def _stem_shape(cls, v: str) -> str:
        if not SHEET_STEM_RE.match(v):
            raise ValueError(
                f"Sheet.stem {v!r} must contain only uppercase letters, digits, and underscores (e.g. 'USB_INPUT' or '5V_BUCK')"
            )
        return v

    @model_validator(mode="after")
    def _library_fields_paired(self):
        if (self.from_library is None) != (self.library_instance is None):
            raise ValueError(
                "Sheet.from_library and Sheet.library_instance must both be set or both be None"
            )
        if self.library_instance is not None and self.library_instance < 1:
            raise ValueError(f"Sheet.library_instance must be >= 1, got {self.library_instance}")
        if self.from_library is not None and "@" not in self.from_library:
            raise ValueError(f"Sheet.from_library {self.from_library!r} must be '<name>@<version>'")
        return self

    @model_validator(mode="after")
    def _replication_fields_paired(self):
        if (self.replication_group is None) != (self.replication_instance is None):
            raise ValueError(
                "Sheet.replication_group and Sheet.replication_instance must "
                "both be set or both be None"
            )
        if self.replication_instance is not None and self.replication_instance < 1:
            raise ValueError(
                f"Sheet.replication_instance must be >= 1, got {self.replication_instance}"
            )
        if self.replication_group is not None and self.from_library is not None:
            raise ValueError(
                "Sheet cannot be both a library reuse (from_library) and a "
                "replication instance (replication_group)"
            )
        return self


class SheetPin(BaseModel):
    sheet: str
    direction: PinDirection


class InterSheetNet(BaseModel):
    """A signal that crosses sheet boundaries. Lists every endpoint."""

    name: str
    endpoints: list[SheetPin]

    @model_validator(mode="after")
    def _at_least_two_endpoints(self):
        if len(self.endpoints) < 2:
            raise ValueError(
                f"InterSheetNet {self.name!r} needs at least 2 endpoints, got {len(self.endpoints)}"
            )
        return self


JsonScalar = str | int | float | bool | None

CircuitRole = Literal[
    "mcu_core",
    "power_input",
    "regulator",
    "programming",
    "bus_interface",
    "sensor",
    "driver",
    "analog_block",
    "user_io",
    "connector",
]


InterfacePortDirection = Literal["input", "output", "bidirectional", "passive", "power"]


class DeclaredInterfacePort(BaseModel):
    """A model-declared physical port preserved for later pin and domain checks."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    direction: InterfacePortDirection
    function: str = Field(min_length=1)
    # These two name a NET THIS PIN IS TIED TO — the pin that IS the part's supply
    # input or its reference/ground pin (or a strap pin held at that net). They are a
    # claim about this exact port, never inferred from a port name, and the compiler
    # verifies the binding exists. A SIGNAL port carries neither: leaving
    # `reference_domain` on a signal port ties the pin to the reference, and the
    # signal's own binding then conflicts.
    supply_rail: str | None = Field(
        default=None,
        description=(
            "Only on a port that IS this part's supply input for that rail (a 'vdd' "
            "pin, or a strap pin held at the rail). Leave null on signal ports."
        ),
    )
    reference_domain: str | None = Field(
        default=None,
        description=(
            "Only on a port that IS this part's reference/ground pin tied to that "
            "zero-volt domain (a 'gnd' pin, or a strap pin held low). Do NOT set it "
            "on signal ports to mean 'this signal is ground-referenced' — leave signal "
            "ports null."
        ),
    )

    # Pin selector as published by the claimed part's symbol/datasheet. It is
    # intentionally separate from the logical key so BOM/wiring can verify a
    # declared interface against the actual component pin inventory.
    pin: str | None = Field(default=None, pattern=PIN_NUMBER_RE.pattern)


class DeclaredInterfaceClaim(BaseModel):
    """Canonical persisted interface for hardware without a curated recipe."""

    model_config = ConfigDict(extra="forbid")

    ports: list[DeclaredInterfacePort] = Field(min_length=1)

    @model_validator(mode="after")
    def _port_keys_unique(self):
        keys = [port.key for port in self.ports]
        if len(keys) != len(set(keys)):
            raise ValueError("DeclaredInterfaceClaim port keys must be unique")
        return self


class PhysicalObligation(BaseModel):
    """A requested physical component class retained from the original intent."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["physical"]
    original_obligation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    component_class: str = Field(
        min_length=1,
        description="Canonical lowercase kebab-case hardware class, for example "
        "bnc-connector, trim-potentiometer, pin-header, screw-terminal, "
        "audio-jack-3-5mm, or microcontroller. Use a physical class, not a component label.",
    )

    @field_validator("component_class", mode="before")
    @classmethod
    def _canonical_class(cls, value: object) -> object:
        return value.strip().casefold().replace("_", "-") if isinstance(value, str) else value


class QuantityObligation(BaseModel):
    """A required number of independently present implementation items."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["quantity"]
    original_obligation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    subject: str = Field(min_length=1)
    minimum: int = Field(ge=1)

    @field_validator("subject", mode="before")
    @classmethod
    def _canonical_subject(cls, value: object) -> object:
        return value.strip().casefold().replace("_", "-") if isinstance(value, str) else value


class AdjustabilityObligation(BaseModel):
    """A user-adjustable electrical behavior the implementation must retain."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["adjustability"]
    original_obligation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    parameter: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)


class ConversionObligation(BaseModel):
    """A required input-to-output conversion behavior."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["conversion"]
    original_obligation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    input_kind: str = Field(min_length=1)
    output_kind: str = Field(min_length=1)
    behavior: str = Field(min_length=1)


class QuantitativeObligation(BaseModel):
    """A numerical limit with its unit and comparison direction."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["quantitative"]
    original_obligation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    quantity: str = Field(min_length=1)
    relation: Literal["equal", "minimum", "maximum", "range"]
    value: float | None = Field(default=None, allow_inf_nan=False)
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def _shape(self):
        if self.relation == "range":
            if (
                self.value is not None
                or self.minimum is None
                or self.maximum is None
                or self.minimum > self.maximum
            ):
                raise ValueError("range quantitative obligation needs ordered minimum and maximum")
        elif self.value is None or self.minimum is not None or self.maximum is not None:
            raise ValueError("scalar quantitative obligation needs value only")
        return self


class FabricationObligation(BaseModel):
    """A board fabrication feature the design must carry, never a component demand.

    A printed copper area, a thermal-via field or an edge treatment is a property of the
    board itself: the PCB-side checks own it and no BOM line can satisfy it, so it is not
    an implementation claim a requirement owns.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["fabrication"]
    original_obligation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    feature: str = Field(
        min_length=1,
        description="Canonical lowercase kebab-case fabrication feature, for example "
        "copper-area, thermal-via-field, castellated-edge, or edge-plating. Use a board "
        "feature, not a component class.",
    )
    minimum: int | None = Field(default=None, ge=1)
    unit: str | None = Field(
        default=None,
        min_length=1,
        description="Unit the stated minimum is counted in (mm2, vias, A); a unit without a "
        "minimum states no limit.",
    )

    @field_validator("feature", mode="before")
    @classmethod
    def _canonical_feature(cls, value: object) -> object:
        return value.strip().casefold().replace("_", "-") if isinstance(value, str) else value

    @model_validator(mode="after")
    def _limit_needs_a_minimum(self):
        if self.unit is not None and self.minimum is None:
            raise ValueError("fabrication obligation unit needs a minimum")
        return self


class NegativeObligation(BaseModel):
    """A physical class the design must NOT contain, never a component demand.

    An absence is a constraint on the whole board, not a part: no BOM line implements it,
    so it is not an implementation claim a requirement owns.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["negative"]
    original_obligation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    absent_class: str = Field(
        min_length=1,
        description="the canonical physical class that must NOT appear, for example "
        "microcontroller or bnc-connector.",
    )

    @field_validator("absent_class", mode="before")
    @classmethod
    def _canonical_absent_class(cls, value: object) -> object:
        return value.strip().casefold().replace("_", "-") if isinstance(value, str) else value


# Obligation kinds that can state a board-level fact instead of naming the requirement that
# implements it. `quantitative` is exempt only when its subject and unit prove it is board-level
# geometry: an outline dimension in a geometric unit, or the board's own build stack-up in
# `layers`/`plies`. Electrical and component limits still require an implementing requirement (see
# `obligation_requires_requirement_owner`). A `quantity` counts a class across the design, a
# `fabrication` feature is a printed-board property the PCB side owns rather than any part, and a
# `negative` forbids a class.
# `Architecture._obligations_are_owned_verbatim`,
# `ArchitectureIntent._obligations_are_owned_once` and
# `stage_contracts.validate_obligation_retention` all read this one set.
OWNERSHIP_EXEMPT_OBLIGATION_KINDS: frozenset[str] = frozenset(
    {"quantity", "quantitative", "fabrication", "negative"}
)

_BOARD_SUBJECT_TERMS = frozenset({"board", "pcb"})
_BOARD_DIMENSION_TERMS = frozenset(
    {"width", "height", "length", "diameter", "radius", "perimeter", "area", "thickness"}
)
_BOARD_DIMENSION_UNITS = frozenset(
    {"mm", "cm", "m", "in", "inch", "inches", "mil", "mm2", "cm2", "in2"}
)
# The board's own build stack-up ("PCB copper layers", unit `layers`): a property of the printed
# board, so no requirement can implement it. The build unit is what separates it from a part
# count that merely mentions copper.
_BOARD_STACKUP_TERMS = frozenset({"layer", "layers", "stackup", "stack", "ply", "plies", "copper"})
_BOARD_STACKUP_UNITS = frozenset({"layer", "layers", "ply", "plies"})


def _obligation_field(obligation: object, field: str) -> object:
    return (
        obligation.get(field) if isinstance(obligation, dict) else getattr(obligation, field, None)
    )


def is_board_level_quantitative_obligation(obligation: object) -> bool:
    """Whether a quantitative row is a board-level measurement, not a part limit.

    Two board-level shapes are admissible, each on its own evidence:

    - an explicit board/PCB subject with an outline dimension in a geometric unit (board width in
      mm);
    - an explicit board/PCB subject with a build stack-up term in a build unit ("PCB copper
      layers" in `layers`).

    Everything else still requires all three pieces of semantic evidence to be attached to its
    realizing requirement: a current, voltage, frequency, contact pitch, or component value must
    remain on the requirement whose realization proves it.
    """
    if _obligation_field(obligation, "kind") != "quantitative":
        return False
    quantity_terms = set(
        re.findall(r"[a-z0-9]+", str(_obligation_field(obligation, "quantity") or "").casefold())
    )
    unit = re.sub(r"\s+", "", str(_obligation_field(obligation, "unit") or "").casefold())
    if not (_BOARD_SUBJECT_TERMS & quantity_terms):
        return False
    if unit in _BOARD_DIMENSION_UNITS and _BOARD_DIMENSION_TERMS & quantity_terms:
        return True
    return unit in _BOARD_STACKUP_UNITS and bool(_BOARD_STACKUP_TERMS & quantity_terms)


def obligation_requires_requirement_owner(obligation: object) -> bool:
    """Whether an obligation must be attached to a realizable architecture requirement."""
    kind = str(_obligation_field(obligation, "kind") or "")
    if kind not in OWNERSHIP_EXEMPT_OBLIGATION_KINDS:
        return True
    return kind == "quantitative" and not is_board_level_quantitative_obligation(obligation)


RequirementObligation = Annotated[
    PhysicalObligation
    | QuantityObligation
    | AdjustabilityObligation
    | ConversionObligation
    | QuantitativeObligation
    | FabricationObligation
    | NegativeObligation,
    Field(discriminator="kind"),
]


class CircuitRequirement(BaseModel):
    """Bounded implementation requirement emitted by architecture.

    Requirements describe what a circuit must provide. Exact component pins and
    support networks remain owned by recipes, lowerers, and pin allocators.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    sheet: str
    role: CircuitRole
    family: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    exact_part: str | None = None
    # Provenance written only by the deterministic architecture compiler.  It is
    # intentionally absent from IntentRequirement, so an intent-declared physical
    # connector cannot impersonate a compiler-generated board edge.
    compiler_origin: Literal["edge_connector"] | None = None
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)
    # Canonical fixed-connector role when this generic header is the explicitly
    # owned host stacking interface of an approved standard form factor.
    standard_stacking_role: str | None = None
    ports: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Logical port key to actual net NAME binding, e.g. {'usb_dp': 'D_P', "
            "'gnd': 'GND'}, not port directions or electrical types. Every declared "
            "inter-sheet net endpoint must have its exact net name bound as a value "
            "by a requirement on that sheet, including model-owned requirements."
        ),
    )
    interfaces: list[str] = Field(default_factory=list)
    # A complete pin-level claim for uncurated hardware. This is canonical
    # architecture data, rather than an advisory architecture-level id list.
    declared_interface: DeclaredInterfaceClaim | None = None
    # Requirement-local links to original acceptance obligations. Validators
    # consume these facts rather than trying to recover them from prose.
    obligations: list[RequirementObligation] = Field(default_factory=list)
    # Exact committed FunctionalSpec block names, never inferred from sheet prose.
    # Empty remains valid for standalone primitive contracts; stage coverage
    # requires explicit membership for a complete architecture.
    functional_blocks: list[str] = Field(default_factory=list)

    @field_validator("functional_blocks")
    @classmethod
    def _functional_blocks_unique(cls, names: list[str]) -> list[str]:
        if any(not name.strip() for name in names):
            raise ValueError("CircuitRequirement.functional_blocks names must be nonempty")
        if len(names) != len(set(names)):
            raise ValueError("CircuitRequirement.functional_blocks names must be unique")
        return names

    @field_validator("obligations")
    @classmethod
    def _obligations_have_unique_source_ids(
        cls, obligations: list[RequirementObligation]
    ) -> list[RequirementObligation]:
        keys = [(obligation.kind, obligation.original_obligation_id) for obligation in obligations]
        if len(keys) != len(set(keys)):
            raise ValueError("CircuitRequirement obligation kind/source pairs must be unique")
        return obligations


class RecipePinAllocation(BaseModel):
    """One deterministic application-net to exact component-pin assignment."""

    model_config = ConfigDict(extra="forbid")

    net: str
    pin: str
    capability: str


class RecipeResolutionRecord(BaseModel):
    """Durable, provider-free explanation of architecture recipe resolution."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    recipe: str
    exact_part: str
    assumptions: list[str] = Field(default_factory=list)


class PinOwnership(BaseModel):
    """One pin excluded from model wiring by a deterministic owner."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    pin: str
    net: str | None = None
    owner: Literal["recipe", "allocator", "lowerer"]
    owner_id: str
    requirement_id: str | None = None


class RecipeOwnershipManifest(BaseModel):
    """Complete ordinary-state ownership emitted by one recipe expansion."""

    model_config = ConfigDict(extra="forbid")

    recipe: str
    instance: str
    requirement_ids: list[str] = Field(default_factory=list)
    refs: list[str]
    pins: list[PinOwnership] = Field(default_factory=list)
    internal_nets: list[str] = Field(default_factory=list)
    port_bindings: dict[str, str] = Field(default_factory=dict)
    pin_allocations: list[RecipePinAllocation] = Field(default_factory=list)
    placement_constraints: list[dict[str, object]] = Field(default_factory=list)
    assertions: list[str] = Field(default_factory=list)


class RecipeSelection(BaseModel):
    """Explicit versioned circuit recipe selected by architecture."""

    model_config = ConfigDict(extra="forbid")

    recipe: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*@[1-9][0-9]*$")
    instance: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    sheets: dict[str, str]
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)
    port_bindings: dict[str, str] = Field(default_factory=dict)
    requirement_ids: list[str] = Field(default_factory=list)
    pin_allocations: list[RecipePinAllocation] = Field(default_factory=list)


class ArchitectureAdvisory(BaseModel):
    """A property the derivation could not *prove* about an otherwise buildable design.

    The RECORD half of the BLOCK-vs-RECORD bar: recorded on the artifact and counted in the
    scorecard, never a refusal. Not part of the provider slot schema (the architecture stage
    answers `ArchitectureIntent`).
    """

    code: str
    message: str = ""
    evidence: list[str] = Field(default_factory=list)


class Architecture(BaseModel):
    topologies: dict[str, str] = Field(default_factory=dict)
    rail_voltages: dict[str, float] = Field(default_factory=dict)
    comms_protocols: list[str] = Field(default_factory=list)
    mcu_present: bool = False
    sheets: list[Sheet]
    power_nets: list[str]
    inter_sheet_nets: list[InterSheetNet]
    assumptions: list[str] = Field(default_factory=list)
    # RECORD-class findings (design-yield-recovery plan §4): properties that could not be
    # proven, carried so the artifact and the scorecard say what a board shipped with.
    advisories: list[ArchitectureAdvisory] = Field(default_factory=list)
    # Approved standard whose fixed connector map is owned by requirements.
    standard_form_factor: str | None = None
    recipe_selections: list[RecipeSelection] = Field(default_factory=list)
    requirements: list[CircuitRequirement] = Field(default_factory=list)
    # Canonical original facts survive recipe resolution and all later stage
    # normalization. Every row must be retained verbatim by one requirement.
    obligations: list[RequirementObligation] = Field(default_factory=list)
    recipe_resolution: list[RecipeResolutionRecord] = Field(default_factory=list)
    unresolved_requirement_ids: list[str] = Field(default_factory=list)
    protected_identities: list[str] = Field(default_factory=list)
    # Requirements whose interface the model declared because no curated recipe covers the part
    # (`docs/plans/architecture-constructive-slot-2026-09-14.md` §4.2). Compiler-derived, never a
    # provider claim: it marks a recorded claim, so review and BOM can flag it instead of assuming
    # the pin functions were verified.
    declared_interfaces: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sheets_unique(self):
        names = [s.name for s in self.sheets]
        stems = [s.stem for s in self.sheets]
        if len(names) != len(set(names)):
            raise ValueError("Architecture.sheets must have unique names")
        if len(stems) != len(set(stems)):
            raise ValueError("Architecture.sheets must have unique stems")
        return self

    @model_validator(mode="after")
    def _recipe_instances_unique_and_mapped(self):
        instances = [selection.instance for selection in self.recipe_selections]
        if len(instances) != len(set(instances)):
            raise ValueError("Architecture.recipe_selections instances must be unique")
        sheet_names = {sheet.name for sheet in self.sheets}
        for selection in self.recipe_selections:
            unknown = set(selection.sheets.values()) - sheet_names
            if unknown:
                raise ValueError(
                    f"recipe {selection.instance!r} maps unknown sheets: {sorted(unknown)}"
                )
        return self

    @model_validator(mode="after")
    def _requirements_unique_and_mapped(self):
        ids = [requirement.id for requirement in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("Architecture.requirements ids must be unique")
        sheet_names = {sheet.name for sheet in self.sheets}
        unknown = {
            requirement.sheet
            for requirement in self.requirements
            if requirement.sheet not in sheet_names
        }
        if unknown:
            raise ValueError(f"Architecture.requirements map unknown sheets: {sorted(unknown)}")
        requirement_ids = set(ids)
        selected = {
            requirement_id
            for selection in self.recipe_selections
            for requirement_id in selection.requirement_ids
        }
        unknown_selected = selected - requirement_ids
        if unknown_selected:
            raise ValueError(
                f"recipe selections reference unknown requirements: {sorted(unknown_selected)}"
            )
        unknown_unresolved = set(self.unresolved_requirement_ids) - requirement_ids
        if unknown_unresolved:
            raise ValueError(
                f"unresolved requirement ids are unknown: {sorted(unknown_unresolved)}"
            )
        return self

    @model_validator(mode="after")
    def _obligations_are_owned_verbatim(self):
        expected = {(row.kind, row.original_obligation_id): row for row in self.obligations}
        if len(expected) != len(self.obligations):
            raise ValueError("Architecture obligations have duplicate kind/source pairs")
        owned_rows = [row for requirement in self.requirements for row in requirement.obligations]
        owned = {(row.kind, row.original_obligation_id) for row in owned_rows}
        # Board-wide counts, fabrication/negative facts, and evidence-backed board-outline
        # measurements may stay at the top level. Every electrical or physical limit still needs
        # a requirement that can realize it.
        unowned = sorted(
            key
            for key in set(expected) - owned
            if obligation_requires_requirement_owner(expected[key])
        )
        if unowned:
            raise ValueError(
                "Architecture obligations must be owned by a requirement: "
                f"listed_at_top_level_only={unowned} — attach each obligation to the requirement "
                "that implements it"
            )
        # The top-level list is the union of the requirements' own rows: a row only a requirement
        # carries is added here, a row several requirements carry is normalised from the design's own
        # row, and nothing is refused for a bookkeeping difference.
        invented = sorted(owned - set(expected))
        if invented:
            self.obligations = [
                *self.obligations,
                *(
                    row
                    for row in owned_rows
                    if (row.kind, row.original_obligation_id) in set(invented)
                ),
            ]
            expected = {(row.kind, row.original_obligation_id): row for row in self.obligations}
        for requirement in self.requirements:
            requirement.obligations = [
                expected.get((row.kind, row.original_obligation_id), row)
                for row in requirement.obligations
            ]
        return self

    @model_validator(mode="after")
    def _inter_sheet_nets_reference_known_sheets(self):
        sheet_names = {s.name for s in self.sheets}
        for net in self.inter_sheet_nets:
            for ep in net.endpoints:
                if ep.sheet not in sheet_names:
                    raise ValueError(
                        f"InterSheetNet {net.name!r} references unknown sheet {ep.sheet!r}"
                    )
        return self


# ---------- Stage 4: BOM ----------


class BomPart(BaseModel):
    """One placeable component."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    value: str
    symbol: str  # KiCad symbol "Library:Name"
    footprint: str  # KiCad footprint "Library:Name"
    sheet: str  # Sheet.name this part belongs to
    mpn: str | None = None
    datasheet: str | None = None
    sourcing_note: str | None = None
    # Which copper side the part mounts on. "back" places it on B.Cu (the
    # footprint is flipped at stamp time); None/"front" is the default F.Cu.
    # Set "back" for back-mounted parts (e.g. a "header on the back side").
    side: Literal["front", "back"] | None = None
    # Set on parts that came from a Leaf Library entry. "<name>@<version>".
    # None for LLM-emitted parts. Synthesis applies the renumber map to
    # parts where this is set; LLM parts pass through unchanged.
    source_leaf: str | None = None
    # Set True on parts the deterministic BOM-reconcile pass added (a cloned
    # donor or an offline-catalog pick). Distinguishes a prior reconcile add
    # that is still unconsumed from a pre-existing part of the same value:
    # only the former may suppress a repeat add of the same ask.
    reconcile_added: bool = False
    # Deterministic circuit-recipe provenance. Model-facing BOM groups cannot
    # author these fields.
    recipe_id: str | None = None
    recipe_instance: str | None = None
    recipe_role: str | None = None
    # Canonical ownership source. Model-facing BOM groups cannot author these.
    resolution_source: Literal["recipe", "lowerer", "llm", "reuse"] | None = None
    resolution_id: str | None = None
    lowering_requirement_id: str | None = None
    lowering_role: str | None = None
    lowering_index: int | None = Field(default=None, ge=0)
    # False means a routed/validated board-fabricated feature omitted from
    # assembly BOM and position exports.
    assembly: bool = True

    @field_validator("ref")
    @classmethod
    def _ref_pattern(cls, v: str) -> str:
        if not REF_RE.match(v):
            raise ValueError(
                f"Reference {v!r} must match ^[A-Z]+[0-9]+[A-Z0-9_-]*$ (e.g. U1, C12, RT1)"
            )
        return v

    @field_validator("footprint")
    @classmethod
    def _footprint_shape(cls, v: str) -> str:
        if not FOOTPRINT_RE.match(v):
            raise ValueError(
                f"Footprint {v!r} must be in 'Library:Name' form (KiCraft does no lookup)"
            )
        return v

    @field_validator("symbol")
    @classmethod
    def _symbol_shape(cls, v: str) -> str:
        if not SYMBOL_RE.match(v):
            raise ValueError(f"Symbol {v!r} must be in 'Library:Name' form")
        return v


class PinEndpoint(BaseModel):
    """One pin's participation in a net.

    ``ref`` matches a BomPart.ref. ``pin`` is the pin number as defined
    in the KiCad symbol (matches the ``(pin "<number>" …)`` token in
    .kicad_sym). For multi-unit symbols, this addresses unit 1 only.
    """

    model_config = ConfigDict(extra="forbid")

    ref: str
    pin: str

    @field_validator("ref")
    @classmethod
    def _ref_pattern(cls, v: str) -> str:
        if not REF_RE.match(v):
            raise ValueError(f"PinEndpoint.ref {v!r} must match {REF_RE.pattern}")
        return v

    @field_validator("pin")
    @classmethod
    def _pin_pattern(cls, v: str) -> str:
        if not PIN_NUMBER_RE.match(v):
            raise ValueError(f"PinEndpoint.pin {v!r} must match {PIN_NUMBER_RE.pattern}")
        return v


class NetConnection(BaseModel):
    """One electrical net inside a leaf sheet.

    ``net_name`` is either an ``Architecture.power_nets`` entry, an
    ``Architecture.inter_sheet_nets`` name, or a sheet-local descriptive
    name. Pin-level mappings live here; the synthesis stage renders them
    as PCB nets (Stage A) and as schematic wires + power symbols
    (Stage B).
    """

    model_config = ConfigDict(extra="forbid")

    net_name: str
    endpoints: list[PinEndpoint]
    sheet: str

    @model_validator(mode="after")
    def _has_endpoints(self):
        if len(self.endpoints) < 1:
            raise ValueError(f"NetConnection {self.net_name!r} has no endpoints")
        return self


class ArraySpec(BaseModel):
    """A regular repeated-component pattern (LED matrix, LED ring, ...).

    Carries the pattern from design intent through synthesis into the
    autoplacer, which lays the members out programmatically instead of running
    the force/simulated-annealing solver over them (which does not converge at
    array scale).

    ``refs`` are listed in data-chain / logical order; the placer fills the
    pattern in that order, so consecutive members are physical neighbours and
    the daisy-chain routes stay short.

    Two patterns:

    - ``grid`` (default): ``rows`` x ``cols`` serpentine matrix. ``rows`` and
      ``cols`` are REQUIRED and ``rows*cols`` must equal ``len(refs)``.
    - ``ring``: members evenly spaced on a circle ("12 LEDs in a ring").
      ``rows``/``cols`` must be omitted. ``radius_mm`` sets the placement
      circle's radius; leave null to derive the tightest legal radius from the
      member size. ``start_angle_deg`` rotates where the chain starts
      (0 = +x axis, clockwise-positive like KiCad).
    """

    model_config = ConfigDict(extra="forbid")

    refs: list[str]
    pattern: Literal["grid", "ring"] = "grid"
    rows: int | None = Field(default=None, gt=0)
    cols: int | None = Field(default=None, gt=0)
    pitch_mm: float | None = Field(
        default=None,
        description=(
            "Centre-to-centre member spacing in millimetres (grid pitch, or the "
            "chord between ring neighbours). SET THIS whenever the design "
            "specifies a pitch (e.g. a brief asking for 'LEDs at 3mm pitch' -> "
            "3.0). Leave null ONLY when no pitch is given, to derive one from "
            "the footprint courtyard plus a default gap."
        ),
    )
    serpentine: bool = True
    radius_mm: float | None = Field(
        default=None,
        description=(
            "ring only: radius of the placement circle in millimetres. SET THIS "
            "when the brief fixes the ring/board size (a '60 mm ring board' "
            "wants the LEDs near the edge -> radius ~24). Leave null to derive "
            "the tightest legal radius from member size."
        ),
    )
    start_angle_deg: float = Field(
        default=0.0,
        description="ring only: angle of the first chain member (deg, 0 = +x).",
    )

    @model_validator(mode="after")
    def _shape_matches(self):
        if self.pattern == "ring":
            if self.rows is not None or self.cols is not None:
                raise ValueError(
                    "ArraySpec pattern='ring' takes no rows/cols (members are "
                    "evenly spaced on a circle); remove them"
                )
            if len(self.refs) < 3:
                raise ValueError(f"ArraySpec pattern='ring' needs >= 3 refs, got {len(self.refs)}")
            if self.radius_mm is not None and self.radius_mm <= 0:
                raise ValueError(f"ArraySpec radius_mm must be > 0, got {self.radius_mm}")
        else:
            if self.rows is None or self.cols is None:
                raise ValueError("ArraySpec pattern='grid' requires rows and cols")
            if self.rows * self.cols != len(self.refs):
                raise ValueError(
                    f"ArraySpec rows*cols ({self.rows}x{self.cols}="
                    f"{self.rows * self.cols}) != len(refs) ({len(self.refs)})"
                )
            if self.radius_mm is not None:
                raise ValueError("ArraySpec radius_mm applies only to pattern='ring'")
        if len(self.refs) != len(set(self.refs)):
            dupes = sorted({r for r in self.refs if self.refs.count(r) > 1})
            raise ValueError(f"ArraySpec has duplicate refs: {dupes}")
        if self.pitch_mm is not None and self.pitch_mm <= 0:
            raise ValueError(f"ArraySpec pitch_mm must be > 0, got {self.pitch_mm}")
        return self


class PlacementHint(BaseModel):
    """Schematic-placement intent for one 2-pin passive (optional).

    The deterministic placer (synthesis/placement.py) clusters each passive
    next to the anchor pin it serves and rotates it so its far pin points
    into open space. It can INFER all of this from ``connections``; a hint
    just makes the intent explicit when inference would be ambiguous (a cap
    between two rails, an RC where the "served" pin isn't obvious, a passive
    that should hug a different IC than the netlist implies).

    Every field except ``ref``/``role`` is optional — the placer fills any
    gap from the netlist. ``anchor_ref`` is the IC the passive belongs with;
    ``anchor_pin`` is the specific pin number on that IC it sits beside;
    ``rail_net`` is the power/ground net its far pin ties to (for pull-ups
    and decoupling caps, so the placer points it at the rail).
    """

    model_config = ConfigDict(extra="forbid")

    ref: str
    role: PlacementRole
    anchor_ref: str | None = None
    anchor_pin: str | None = None
    rail_net: str | None = None

    @field_validator("ref", "anchor_ref")
    @classmethod
    def _ref_pattern(cls, v: str | None) -> str | None:
        if v is not None and not REF_RE.match(v):
            raise ValueError(f"PlacementHint ref {v!r} must match {REF_RE.pattern}")
        return v


class Substitution(BaseModel):
    """One surfaced part substitution: the BOM deviates from a part the
    spec/architecture named (or the user asked for) and says so.

    The 2026-07-27 self-eval batch gated 6 runs on ``silent_substitution`` --
    an architecture-named RECOM converter silently swapped for a quarter of
    its output current, a brief-stated SMT OLED shipped as through-hole. The
    substitution itself is often a fine engineering call; the defect is
    silence. §9.33 enforces that a spec-named MPN missing from the BOM has a
    ledger entry here, and the eval digest surfaces the ledger to the judge
    and the user."""

    model_config = ConfigDict(extra="forbid")

    wanted: str  # the part the spec/user named (MPN or description)
    got: str  # the part the BOM ships instead
    reason: str = ""


class EdgeInterface(BaseModel):
    """Ordered fabrication interface constrained to one board edge."""

    model_config = ConfigDict(extra="forbid")

    name: str
    refs: list[str] = Field(min_length=1)
    side: Literal["top", "bottom", "left", "right"]
    pitch_mm: float = Field(gt=0)
    behavior: Literal["castellated"]


class BOM(BaseModel):
    parts: list[BomPart]
    ic_groups: dict[str, list[str]] = Field(default_factory=dict)
    group_labels: dict[str, str] = Field(default_factory=dict)
    thermal_refs: list[str] = Field(default_factory=list)
    signal_flow_order: list[str] = Field(default_factory=list)
    component_zones: dict[str, dict[str, str]] = Field(default_factory=dict)
    arrays: list[ArraySpec] = Field(default_factory=list)
    placement_hints: list[PlacementHint] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    substitutions: list[Substitution] = Field(default_factory=list)
    connections: list[NetConnection] = Field(default_factory=list)
    no_connect_pins: list[PinEndpoint] = Field(default_factory=list)
    edge_interfaces: list[EdgeInterface] = Field(default_factory=list)
    recipe_ownership: list[RecipeOwnershipManifest] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refs_unique(self):
        refs = [p.ref for p in self.parts]
        if len(refs) != len(set(refs)):
            dupes = [r for r in refs if refs.count(r) > 1]
            raise ValueError(f"BOM has duplicate refs: {sorted(set(dupes))}")
        return self

    @model_validator(mode="after")
    def _ic_group_refs_known(self):
        ref_set = {p.ref for p in self.parts}
        for ic, members in self.ic_groups.items():
            if ic not in ref_set:
                raise ValueError(f"ic_groups leader {ic!r} not in BOM parts")
            for m in members:
                if m not in ref_set:
                    raise ValueError(f"ic_groups[{ic!r}] member {m!r} not in BOM parts")
        return self

    @model_validator(mode="after")
    def _named_refs_known(self):
        ref_set = {p.ref for p in self.parts}
        for ref in self.thermal_refs:
            if ref not in ref_set:
                raise ValueError(f"thermal_refs entry {ref!r} not in BOM parts")
        for ref in self.signal_flow_order:
            if ref not in ref_set:
                raise ValueError(f"signal_flow_order entry {ref!r} not in BOM parts")
        for ref in self.component_zones:
            if ref not in ref_set:
                raise ValueError(f"component_zones entry {ref!r} not in BOM parts")
        seen_array_refs: set[str] = set()
        for spec in self.arrays:
            for ref in spec.refs:
                if ref not in ref_set:
                    raise ValueError(f"arrays ref {ref!r} not in BOM parts")
                if ref in seen_array_refs:
                    raise ValueError(f"arrays ref {ref!r} appears in more than one array")
                seen_array_refs.add(ref)
        for hint in self.placement_hints:
            if hint.ref not in ref_set:
                raise ValueError(f"placement_hints ref {hint.ref!r} not in BOM parts")
            if hint.anchor_ref is not None and hint.anchor_ref not in ref_set:
                raise ValueError(
                    f"placement_hints[{hint.ref!r}].anchor_ref {hint.anchor_ref!r} not in BOM parts"
                )
        for interface in self.edge_interfaces:
            unknown = set(interface.refs) - ref_set
            if unknown:
                raise ValueError(
                    f"edge interface {interface.name!r} references unknown refs: {sorted(unknown)}"
                )
        return self

    @model_validator(mode="after")
    def _recipe_ownership_is_disjoint_and_known(self):
        ref_set = {part.ref for part in self.parts}
        owned_refs: set[str] = set()
        owned_pins: set[tuple[str, str]] = set()
        internal_nets: set[str] = set()
        for manifest in self.recipe_ownership:
            unknown_refs = set(manifest.refs) - ref_set
            if unknown_refs:
                raise ValueError(
                    f"recipe ownership references unknown refs: {sorted(unknown_refs)}"
                )
            overlap_refs = owned_refs & set(manifest.refs)
            if overlap_refs:
                raise ValueError(f"recipe ownership overlaps refs: {sorted(overlap_refs)}")
            owned_refs.update(manifest.refs)
            overlap_nets = internal_nets & set(manifest.internal_nets)
            if overlap_nets:
                raise ValueError(f"recipe ownership internal nets collide: {sorted(overlap_nets)}")
            internal_nets.update(manifest.internal_nets)
            for ownership in manifest.pins:
                key = (ownership.ref, ownership.pin)
                if ownership.ref not in ref_set:
                    raise ValueError(
                        f"recipe ownership pin references unknown ref {ownership.ref!r}"
                    )
                if key in owned_pins:
                    raise ValueError(
                        f"recipe ownership pin overlaps: {ownership.ref}.{ownership.pin}"
                    )
                owned_pins.add(key)
        return self

    @model_validator(mode="after")
    def _connection_refs_known(self):
        ref_set = {p.ref for p in self.parts}
        for c in self.connections:
            for ep in c.endpoints:
                if ep.ref not in ref_set:
                    raise ValueError(
                        f"NetConnection {c.net_name!r} references unknown ref {ep.ref!r}"
                    )
        for ep in self.no_connect_pins:
            if ep.ref not in ref_set:
                raise ValueError(f"no_connect_pins references unknown ref {ep.ref!r}")
        return self

    @model_validator(mode="after")
    def _connection_sheets_known(self):
        part_sheets = {p.sheet for p in self.parts}
        unknown = {c.sheet for c in self.connections} - part_sheets
        if unknown:
            raise ValueError(
                f"NetConnection.sheet values not represented in BOM.parts: {sorted(unknown)}"
            )
        return self


# ---------- Placement (user rules, deterministic; no LLM) ----------

# Anchor vocabulary for per-component placement rules. Single source of
# truth: the layout editor's rules layer and the web/offline UIs import
# these (kicraft.layout_editor.rules aliases them).
PLACEMENT_ANCHOR_VALUES: dict[str, list[str]] = {
    "edge": ["left", "right", "top", "bottom"],
    "corner": ["top-left", "top-right", "bottom-left", "bottom-right"],
    "zone": [
        "center",
        "top",
        "bottom",
        "left",
        "right",
        "center-top",
        "center-bottom",
        "center-left",
        "center-right",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ],
}


class PlacementBoard(BaseModel):
    """Fixed board dimensions for the auto placer. When width/height are
    set (and ``size_search`` is off) they land in the generated
    autoplacer.json as board_width_mm/board_height_mm with the size
    search disabled, so the solver fits the user's chosen board."""

    model_config = ConfigDict(extra="forbid")

    width_mm: float | None = None
    height_mm: float | None = None
    size_search: bool = True

    @field_validator("width_mm", "height_mm")
    @classmethod
    def _positive(cls, v: float | None) -> float | None:
        if v is not None and v < 10.0:
            raise ValueError("board dimensions must be >= 10 mm")
        return v


class PlacementSection(BaseModel):
    """User placement rules. Deterministic (committing this section never
    runs an LLM stage and invalidates nothing upstream); merged OVER the
    BOM's LLM-derived hints into the generated ``<stem>_autoplacer.json``
    at synthesis time. Refs are deliberately NOT validated against the
    BOM here: parts churn across BOM re-runs, and a stale rule must
    degrade to a warning at synthesis, not brick the commit."""

    model_config = ConfigDict(extra="forbid")

    component_zones: dict[str, dict[str, str | float]] = Field(default_factory=dict)
    thermal_refs: list[str] = Field(default_factory=list)
    backside_through_hole_leaves: list[str] = Field(default_factory=list)
    board: PlacementBoard | None = None

    @model_validator(mode="after")
    def _zone_specs_well_formed(self):
        allowed_keys = {"edge", "corner", "zone", "rotation"}
        for ref, spec in self.component_zones.items():
            extra = set(spec.keys()) - allowed_keys
            if extra:
                raise ValueError(
                    f"component_zones[{ref!r}]: unknown keys {sorted(extra)}; "
                    f"allowed: {sorted(allowed_keys)}"
                )
            anchors = [k for k in ("edge", "corner", "zone") if k in spec]
            if len(anchors) > 1:
                raise ValueError(
                    f"component_zones[{ref!r}]: at most one anchor of "
                    f"edge/corner/zone, got {anchors}"
                )
            for key in anchors:
                value = spec[key]
                if value not in PLACEMENT_ANCHOR_VALUES[key]:
                    raise ValueError(
                        f"component_zones[{ref!r}].{key}: {value!r} not in "
                        f"{PLACEMENT_ANCHOR_VALUES[key]}"
                    )
            if "rotation" in spec:
                try:
                    rot = float(spec["rotation"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"component_zones[{ref!r}].rotation must be a number") from exc
                if not 0.0 <= rot <= 360.0:
                    raise ValueError(f"component_zones[{ref!r}].rotation must be in 0..360")
        return self


# ---------- Silkscreen plan (authored post-wiring, placed at build tail) ----------


class SilkAnchor(BaseModel):
    """Semantic placement hint for one silk label. The LLM never emits
    coordinates: ``ref`` names a BOM component and the build-tail placer
    turns it into geometry (or drops the label honestly)."""

    ref: str | None = None  # BOM refdes the label belongs beside
    prefer: Literal["above", "below", "left", "right"] | None = None


class SilkPinText(BaseModel):
    """One per-pin entry inside a ``pinout`` label. ``pin`` names a real
    pad number on the anchor footprint; ``text`` is the short single-line
    function label (e.g. ``VIN``, ``GND``, ``12V OUT``)."""

    pin: str
    text: str

    @field_validator("pin")
    @classmethod
    def _pin_pattern(cls, v: str) -> str:
        if not PIN_NUMBER_RE.match(v):
            raise ValueError(f"SilkPinText.pin {v!r} must match {PIN_NUMBER_RE.pattern}")
        return v


class SilkLabel(BaseModel):
    """One functional silkscreen text block (an IO rating, a DIP-switch
    table, a per-pin connector pinout, a usage note). Content is linted
    before commit: anchors must exist in the BOM and numeric claims must be
    corroborated by the design state — an uncorroborated voltage on silk is
    worse than none."""

    id: str
    kind: Literal["io", "table", "note", "pinout"] = "note"
    text: str  # ASCII; '\n' separates lines
    anchor: SilkAnchor | None = None
    priority: int = Field(default=2, ge=1, le=3)  # 1 must-have .. 3 nice
    pins: list[SilkPinText] = Field(default_factory=list)


class SilkPlan(BaseModel):
    """Top-level silkscreen content slot (like ``review_findings``, it is
    authored in the web process BEFORE the build; the no-LLM build tail
    consumes it deterministically). Absent slot => legend-only fallback."""

    version: int = 1
    title: str | None = None  # short board title for the legend line
    board_code: str | None = None  # KC-XXXXXX; server-side knowledge
    rev: str = "1.0"
    labels: list[SilkLabel] = Field(default_factory=list)
    # Lint honesty: labels the deterministic lint rejected, with reasons —
    # surfaced so a missing table is a visible decision, not a silent drop.
    dropped_at_lint: list[str] = Field(default_factory=list)
    # Coverage report: IO connectors that got no label (visibility only —
    # never auto-generated text). Surfaced in the web inspector.
    uncovered_connectors: list[str] = Field(default_factory=list)
    author_model: str | None = None
    cost_usd: float = 0.0


# ---------- Artifacts (set after synthesis) ----------


class ReviewFinding(BaseModel):
    """A single electrical-review finding, persisted so the GUI inspector can
    render it richly (severity badge, area, issue, suggestion) even on reopen."""

    severity: Literal["blocker", "warning", "note"]
    area: str = ""
    issue: str
    suggestion: str = ""


class PcbViolation(BaseModel):
    """One bounded, coordinate-bearing PCB failure fact from a DRC report."""

    type: str
    x_mm: float | None = None
    y_mm: float | None = None
    net1: str | None = None
    net2: str | None = None
    footprint_refs: list[str] = Field(default_factory=list)
    description: str = ""


class PcbError(BaseModel):
    """Durable explanation for one terminal place/route or verify failure."""

    stage: Literal["place_route", "verify"]
    code: str
    title: str
    explanation: str
    details: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    nets: list[str] = Field(default_factory=list)
    footprint_refs: list[str] = Field(default_factory=list)
    violations: list[PcbViolation] = Field(default_factory=list)
    next_action: str
    overlay_path: Path | None = None


class ArtifactPaths(BaseModel):
    project_dir: Path
    project_stem: str
    root_sch: Path
    leaf_schs: list[Path]
    kicad_pro: Path
    autoplacer_json: Path
    custom_footprint_dir: Path | None = None
    routed_pcb: Path | None = None  # set by `build`: promoted fully-routed board
    fab_zip: Path | None = None  # set by `build`: zipped Gerber/drill/CPL/BOM package
    step_file: Path | None = None  # set by `build`: STEP export of the assembled board
    board_3d_png: Path | None = None  # set by `build`: rendered 3D view of the board
    status: str = "ok"  # "ok" if all §9 checks passed, else "failed"
    # Non-blocking fab-readiness warnings (e.g. a minor, fraction-of-a-mm
    # courtyard clip). The board IS fab-exported + 3D-rendered; these surface as
    # a yellow caution in the UI rather than a red failure.
    build_warnings: list[str] = Field(default_factory=list)
    # Electrical-review findings persisted for the GUI inspector (structured,
    # with suggestions, vs the build_log lines which are bare text).
    review_findings: list[ReviewFinding] = Field(default_factory=list)
    # PCB place/route and verify failures persisted for live/reopened diagnosis.
    pcb_errors: list[PcbError] = Field(default_factory=list)
    # Silk-legend honesty: what the build-tail placer actually did. Placed ids
    # include "legend:N"; dropped entries are "id: reason" strings.
    silk_placed: list[str] = Field(default_factory=list)
    silk_dropped: list[str] = Field(default_factory=list)


# ---------- Conversation state ----------


class StageDiagnostic(BaseModel):
    """Versioned, redacted deterministic finding for one committed stage.

    Additive readability, like ``StageStatus``: a field a historical writer did
    not record reads as ``None``/empty instead of making the whole state
    unloadable. ``_commit_rejection_diagnostics`` wrote ``gate_codes`` without a
    ``detector_version``, so *every* state saved after a deterministic commit
    gate refused a candidate (a block-sheet mapping failure, a wiring gate)
    failed ``ConversationState.model_validate`` -- which is exactly the state
    ``stage_driver replay`` exists to iterate on.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,127}$")
    # None = the writer that produced this row did not record one. Every consumer already reads
    # a row as "severe" only when it names `repair_required`/`fab_gate`, so an unrecorded severity
    # ranks as not-severe rather than making the whole state unloadable (119 saved states carried
    # rows from the architecture gate, which recorded no severity at all).
    severity: Literal["advisory", "repair_required", "fab_gate"] | None = None
    message: str
    evidence: list[str] = Field(default_factory=list)
    # None = the writer that produced this row did not record a detector version.
    detector_version: int | None = Field(default=None, ge=1)
    attempt: int | None = Field(default=None, ge=1)
    # The stable gate ids (``§9.9``, ``9.15``, ...) a commit rejection names, so the durable
    # status names every reason a candidate was refused and not just the first error line.
    gate_codes: list[str] = Field(default_factory=list)
    # A work-unit refusal is scoped to ONE unit of a multi-unit stage (bom/wiring), so the
    # row carries which unit and that unit's per-check offender lists. Both are optional:
    # a stage-scoped row (the commit-rejection and semantic findings) records neither.
    unit_id: str | None = None
    defects: dict[str, list[str]] | None = None
    # The scope a design-contract refusal names: which requirement, on which sheet, from which
    # resolved recipe. The architecture gate records all three and its rows are stored here, so
    # the durable status carries them; before this they were `extra_forbidden` and the states
    # holding them could not be loaded back (runs 905/907/913, and two self-eval campaigns).
    requirement_id: str | None = None
    sheet: str | None = None
    recipe: str | None = None
    # The per-obligation candidate requirements the obligation-retention refusal names, written
    # into the diagnostic by `validate_obligation_retention` for `physical` rows. Same defect
    # class as the fields above: the writer stored it and `extra="forbid"` made every state saved
    # after that refusal unloadable — the live stack-up refusals KC-CTBW6M (44/917) and KC-9FPA59
    # (1/919) could not be replayed at all. Read as empty, never a load refusal.
    candidate_requirement_ids: dict[str, list[str]] | None = None
    # The individual findings an aggregate row wraps -- an outer ``multiple_intent_contracts``
    # over the contract refusals it collected. Typed and recursive: `_derive_intent_payload` used
    # to put the same rows in `evidence` as bare dicts with no `severity`, which made *every* state
    # saved after an architecture refusal naming two or more defects fail
    # `ConversationState.model_validate` (live runs KC-HPD3YF and KC-P4E2PH) -- the one state
    # `stage_driver replay` exists to iterate on.
    findings: list[StageDiagnostic] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _read_legacy_evidence_rows(cls, value):
        """Read a nested row written into ``evidence`` by an earlier writer.

        Historical states carry the findings as dicts there (the shape `findings` now types), and
        the reader keeps them as one readable line rather than refusing the whole state. Losing a
        nested row's structure is a report detail; refusing to load the state loses the run.
        """
        return [
            (
                item
                if isinstance(item, str)
                else " — ".join(
                    str(part)
                    for part in (
                        ": ".join(str(v) for v in (item.get("code"), item.get("message")) if v)
                        if isinstance(item, dict)
                        else item,
                    )
                )
            )
            for item in (value or [])
        ]


class StageStatus(BaseModel):
    """Durable outcome of one pipeline stage, keyed by stage name in
    ConversationState.stage_status. Written by the server stage driver at
    commit/fail time (and by manual slot edits), so a reopened project can
    restore pipeline progress without replaying the ephemeral event stream."""

    ok: bool
    # Outcome dimensions are additive so old state remains readable. ``None``
    # means the historical writer did not observe the dimension.
    provider_ok: bool | None = None
    schema_ok: bool | None = None
    semantic_clean: bool | None = None
    repair_required: bool = False
    fab_safe: bool | None = None
    repair_attempted: bool = False
    repair_adopted: bool = False
    diagnostics: list[StageDiagnostic] = Field(default_factory=list)
    cost_usd: float | None = None
    attempts: int | None = None
    finished_at: str | None = None  # UTC ISO timestamp
    wall_s: float | None = None
    cpu_s: float | None = None
    rounds: int | None = None  # BOM tool-loop rounds (None for single-shot stages)
    tool_calls: int | None = None  # total BOM tool calls (None for non-BOM stages)
    # Terminal failure classification for a failed stage: one of
    # collection_limit / reasoning_loop / truncated_json / invalid_json /
    # invalid_schema / contract_rejected / commit_rejected / provider_error /
    # transport_error. None for a committed project written before the field
    # existed. Derived, never free-form. invalid_schema is unusable provider
    # output; contract_rejected is a schema-clean candidate that a semantic or
    # recipe contract refused (its `diagnostic` names the contract).
    failure_kind: str | None = None


class ConversationState(BaseModel):
    """Single mutable object passed to every stage and the orchestrator."""

    project_stem: str | None = None
    intent: IntentSlot | None = None
    functional_spec: FunctionalSpec | None = None
    architecture: Architecture | None = None
    bom: BOM | None = None
    # User placement rules (deterministic; not a design stage). Edited
    # by the web rules panel via `stage-commit placement`; consumed by
    # write_autoplacer_json with the highest merge precedence.
    placement: PlacementSection | None = None
    open_questions: list[Question] = Field(default_factory=list)
    history: list[ChatMsg] = Field(default_factory=list)
    artifacts: ArtifactPaths | None = None
    # Electrical-review findings from the post-wiring review. Lives at the top
    # level (not on artifacts) because the review runs BEFORE the build, when
    # artifacts is still None. The GUI electrical-review inspector reads this
    # first, falling back to artifacts.review_findings for legacy projects.
    review_findings: list[ReviewFinding] = Field(default_factory=list)
    # Silkscreen content plan, authored post-wiring in the web process (same
    # lifecycle as review_findings). The build tail places it; None => the
    # deterministic legend only.
    silk_plan: SilkPlan | None = None
    expert_mode: bool = False
    stage_status: dict[str, StageStatus] = Field(default_factory=dict)

    def replace_open_questions_for_stage(self, stage: str, new: list[Question]) -> None:
        """Stages overwrite their own slot — questions are slot-scoped too."""
        kept = [q for q in self.open_questions if q.stage != stage]
        self.open_questions = kept + list(new)


# These upstream classes refer to the architecture obligation union declared
# later in this module. Resolve the forward annotations once the complete
# canonical vocabulary is available.
IntentSlot.model_rebuild()
FunctionalSpec.model_rebuild()
