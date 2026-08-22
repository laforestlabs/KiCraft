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
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REF_RE = re.compile(r"^[A-Z]+[0-9]+[A-Z0-9_-]*$")
FOOTPRINT_RE = re.compile(r"^[A-Za-z0-9_.+-]+:[A-Za-z0-9_.+-]+$")
SYMBOL_RE = re.compile(r"^[A-Za-z0-9_.+-]+:[A-Za-z0-9_.+-]+$")
SHEET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9 ]*[A-Z0-9]$")
SHEET_STEM_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
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
BlockCategory = Literal["sense", "process", "drive", "power", "interface"]
# How a 2-pin passive relates to the IC it serves — drives schematic placement
# (which side of the anchor it sits on, which way it's rotated, and what its far
# pin ties to). See PlacementHint and synthesis/placement.py.
PlacementRole = Literal[
    "decoupling",  # local bypass cap: rail pin <-> gnd pin, hugs a power pin
    "bulk",        # large reservoir cap: rail <-> gnd, like decoupling
    "pullup",      # resistor: signal pin <-> a positive rail
    "pulldown",    # resistor: signal pin <-> ground
    "series",      # in-line R/L/ferrite in a signal/power path
    "feedback",    # divider / compensation around the IC
    "other",       # cluster near the anchor, no special orientation
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
    constraints: list[str] = Field(default_factory=list)
    named_parts: list[str] = Field(default_factory=list)
    inferred_expertise: Literal["beginner", "intermediate", "expert"] = "intermediate"
    assumptions: list[str] = Field(default_factory=list)
    # Requested non-rectangular board shape, when the brief asks for one. Set by
    # the intent stage (LLM + a deterministic extractor at stage-commit). None /
    # shape "rect" means a conventional rectangular board.
    form_factor: FormFactor | None = None


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
                f"Sheet.name {v!r} must be uppercase with optional spaces (e.g. 'USB INPUT')"
            )
        return v

    @field_validator("stem")
    @classmethod
    def _stem_shape(cls, v: str) -> str:
        if not SHEET_STEM_RE.match(v):
            raise ValueError(
                f"Sheet.stem {v!r} must be uppercase with underscores (e.g. 'USB_INPUT')"
            )
        return v

    @model_validator(mode="after")
    def _library_fields_paired(self):
        if (self.from_library is None) != (self.library_instance is None):
            raise ValueError(
                "Sheet.from_library and Sheet.library_instance must both "
                "be set or both be None"
            )
        if self.library_instance is not None and self.library_instance < 1:
            raise ValueError(
                f"Sheet.library_instance must be >= 1, got {self.library_instance}"
            )
        if self.from_library is not None and "@" not in self.from_library:
            raise ValueError(
                f"Sheet.from_library {self.from_library!r} must be "
                f"'<name>@<version>'"
            )
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
                f"Sheet.replication_instance must be >= 1, got "
                f"{self.replication_instance}"
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


class Architecture(BaseModel):
    topologies: dict[str, str] = Field(default_factory=dict)
    rail_voltages: dict[str, float] = Field(default_factory=dict)
    comms_protocols: list[str] = Field(default_factory=list)
    mcu_present: bool = False
    sheets: list[Sheet]
    power_nets: list[str]
    inter_sheet_nets: list[InterSheetNet]
    assumptions: list[str] = Field(default_factory=list)

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
            raise ValueError(
                f"PinEndpoint.pin {v!r} must match {PIN_NUMBER_RE.pattern}"
            )
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
                raise ValueError(
                    f"ArraySpec pattern='ring' needs >= 3 refs, got {len(self.refs)}"
                )
            if self.radius_mm is not None and self.radius_mm <= 0:
                raise ValueError(
                    f"ArraySpec radius_mm must be > 0, got {self.radius_mm}"
                )
        else:
            if self.rows is None or self.cols is None:
                raise ValueError("ArraySpec pattern='grid' requires rows and cols")
            if self.rows * self.cols != len(self.refs):
                raise ValueError(
                    f"ArraySpec rows*cols ({self.rows}x{self.cols}="
                    f"{self.rows * self.cols}) != len(refs) ({len(self.refs)})"
                )
            if self.radius_mm is not None:
                raise ValueError(
                    "ArraySpec radius_mm applies only to pattern='ring'"
                )
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
    got: str     # the part the BOM ships instead
    reason: str = ""


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
                    raise ValueError(
                        f"arrays ref {ref!r} appears in more than one array"
                    )
                seen_array_refs.add(ref)
        for hint in self.placement_hints:
            if hint.ref not in ref_set:
                raise ValueError(f"placement_hints ref {hint.ref!r} not in BOM parts")
            if hint.anchor_ref is not None and hint.anchor_ref not in ref_set:
                raise ValueError(
                    f"placement_hints[{hint.ref!r}].anchor_ref "
                    f"{hint.anchor_ref!r} not in BOM parts"
                )
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
                raise ValueError(
                    f"no_connect_pins references unknown ref {ep.ref!r}"
                )
        return self

    @model_validator(mode="after")
    def _connection_sheets_known(self):
        part_sheets = {p.sheet for p in self.parts}
        unknown = {c.sheet for c in self.connections} - part_sheets
        if unknown:
            raise ValueError(
                f"NetConnection.sheet values not represented in BOM.parts: "
                f"{sorted(unknown)}"
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
        "center", "top", "bottom", "left", "right",
        "center-top", "center-bottom", "center-left", "center-right",
        "top-left", "top-right", "bottom-left", "bottom-right",
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
                    raise ValueError(
                        f"component_zones[{ref!r}].rotation must be a number"
                    ) from exc
                if not 0.0 <= rot <= 360.0:
                    raise ValueError(
                        f"component_zones[{ref!r}].rotation must be in 0..360"
                    )
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
            raise ValueError(
                f"SilkPinText.pin {v!r} must match {PIN_NUMBER_RE.pattern}"
            )
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


class StageStatus(BaseModel):
    """Durable outcome of one pipeline stage, keyed by stage name in
    ConversationState.stage_status. Written by the server stage driver at
    commit/fail time (and by manual slot edits), so a reopened project can
    restore pipeline progress without replaying the ephemeral event stream."""

    ok: bool
    cost_usd: float | None = None
    attempts: int | None = None
    finished_at: str | None = None  # UTC ISO timestamp
    wall_s: float | None = None
    cpu_s: float | None = None
    rounds: int | None = None  # BOM tool-loop rounds (None for single-shot stages)
    tool_calls: int | None = None  # total BOM tool calls (None for non-BOM stages)


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
