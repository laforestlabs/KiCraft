"""Shared data structures for the autoplacer system.

All types are plain Python dataclasses — no pcbnew imports.
These serve as the interchange format between Brain and Hardware layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import atan2, hypot
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .subcircuit_composer import LeafBlockerSet


class Layer(IntEnum):
    FRONT = 0  # F.Cu
    BACK = 1  # B.Cu


@dataclass(slots=True)
class Point:
    x: float  # mm
    y: float  # mm

    def dist(self, other: Point) -> float:
        return hypot(self.x - other.x, self.y - other.y)

    def angle_to(self, other: Point) -> float:
        """Angle in radians from self to other."""
        return atan2(other.y - self.y, other.x - self.x)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> Point:
        return Point(self.x * s, self.y * s)

    def __hash__(self):
        return hash((round(self.x, 4), round(self.y, 4)))


@dataclass(slots=True)
class Pad:
    ref: str  # component reference, e.g. "U2"
    pad_id: str  # pad number/name, e.g. "1"
    pos: Point  # absolute position of pad center, in mm
    net: str  # net name
    layer: Layer
    # Pad copper extent (width, height) in mm. None = legacy artifact with
    # no recorded size; bbox() then returns the center as a degenerate point.
    # Populated by hardware.adapter from KiCad's Pad.GetSize() during board
    # extraction and carried through solved_layout.json round-trip.
    size_mm: Point | None = None

    def bbox(self) -> tuple[Point, Point]:
        """Pad copper bbox (top_left, bottom_right) in absolute coords."""
        if self.size_mm is None:
            return (self.pos, self.pos)
        hw = self.size_mm.x / 2.0
        hh = self.size_mm.y / 2.0
        return (
            Point(self.pos.x - hw, self.pos.y - hh),
            Point(self.pos.x + hw, self.pos.y + hh),
        )


@dataclass
class Component:
    ref: str
    value: str
    pos: Point
    rotation: float  # degrees
    layer: Layer
    width_mm: float  # courtyard bbox width
    height_mm: float  # courtyard bbox height
    pads: list[Pad] = field(default_factory=list)
    locked: bool = False
    kind: str = ""  # "connector", "mounting_hole", "ic", "passive", "misc"
    is_through_hole: bool = False  # True if footprint has PTH pads
    body_center: Point | None = None  # courtyard/body bbox center (absolute coords)
    opening_direction: float | None = (
        None  # LOCAL-frame angle (0/90/180/270) where opening faces
    )
    block_blocker_set: LeafBlockerSet | None = None
    block_artifact_origin_offset: Point | None = None
    block_side: str | None = None
    allowed_rotations: list[float] | None = None

    @property
    def area(self) -> float:
        return self.width_mm * self.height_mm

    def bbox(self, clearance: float = 0.0) -> tuple[Point, Point]:
        """Courtyard bbox (top_left, bottom_right) with optional clearance.

        This is the keep-out / repulsion target -- the area routing tries to
        leave clear of other parts. It does NOT include pad copper that
        sticks out past the courtyard. Use ``physical_bbox()`` when the
        question is "where is the actual physical extent of this component
        including its copper" (e.g. board-edge containment, parent frame
        sizing, packing density).

        Centers the bbox on body_center when available, falling back to pos
        (footprint origin). Critical for components whose origin differs
        from the courtyard center (battery holders, some connectors).
        """
        hw = self.width_mm / 2 + clearance
        hh = self.height_mm / 2 + clearance
        cx = self.body_center.x if self.body_center else self.pos.x
        cy = self.body_center.y if self.body_center else self.pos.y
        return (
            Point(cx - hw, cy - hh),
            Point(cx + hw, cy + hh),
        )

    def physical_bbox(self, clearance: float = 0.0) -> tuple[Point, Point]:
        """Union of courtyard bbox and every pad's copper bbox.

        This is the SINGLE source of truth for "where is this component
        physically present in board coordinates", used wherever the answer
        must include pad copper that extends past the courtyard:
        board-edge containment, parent frame sizing, packing density,
        outside-the-board geometry validation.

        For pads with no recorded ``size_mm`` (legacy artifacts), the pad
        contributes its center point only -- behaviour identical to the
        old courtyard ∪ pad-centers heuristic. Re-extract the leaf from
        its PCB to get pad sizes captured.
        """
        body_tl, body_br = self.bbox(clearance)
        min_x, min_y = body_tl.x, body_tl.y
        max_x, max_y = body_br.x, body_br.y
        for pad in self.pads:
            pad_tl, pad_br = pad.bbox()
            min_x = min(min_x, pad_tl.x - clearance)
            min_y = min(min_y, pad_tl.y - clearance)
            max_x = max(max_x, pad_br.x + clearance)
            max_y = max(max_y, pad_br.y + clearance)
        return (Point(min_x, min_y), Point(max_x, max_y))


@dataclass
class Net:
    name: str
    pad_refs: list[tuple[str, str]] = field(default_factory=list)  # [(ref, pad_id)]
    priority: int = 0  # higher = route first
    width_mm: float = 0.127  # trace width
    is_power: bool = False

    @property
    def component_refs(self) -> set[str]:
        return {ref for ref, _ in self.pad_refs}


@dataclass(slots=True)
class TraceSegment:
    start: Point
    end: Point
    layer: Layer
    net: str
    width_mm: float

    @property
    def length(self) -> float:
        return self.start.dist(self.end)


@dataclass(slots=True)
class Via:
    pos: Point
    net: str
    drill_mm: float = 0.3
    size_mm: float = 0.6


@dataclass(slots=True)
class SilkscreenElement:
    """A silkscreen graphic that travels with a subcircuit through composition."""

    kind: str  # "poly" or "text"
    layer: str  # "F.SilkS" or "B.SilkS"
    points: list[Point] = field(default_factory=list)
    stroke_width: float = 0.15
    text: str = ""
    pos: Point = field(default_factory=lambda: Point(0.0, 0.0))
    font_height: float = 1.0
    font_width: float = 1.0
    font_thickness: float = 0.15


@dataclass
class BoardState:
    """Complete snapshot -- the interchange format between Brain and Hardware."""

    components: dict[str, Component] = field(default_factory=dict)  # ref -> Component
    nets: dict[str, Net] = field(default_factory=dict)  # name -> Net
    traces: list[TraceSegment] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    silkscreen: list[SilkscreenElement] = field(default_factory=list)
    board_outline: tuple[Point, Point] = field(
        default_factory=lambda: (Point(0, 0), Point(90, 58))
    )

    @property
    def board_width(self) -> float:
        return self.board_outline[1].x - self.board_outline[0].x

    @property
    def board_height(self) -> float:
        return self.board_outline[1].y - self.board_outline[0].y

    @property
    def board_center(self) -> Point:
        tl, br = self.board_outline
        return Point((tl.x + br.x) / 2, (tl.y + br.y) / 2)


@dataclass
class PlacementIterationSnapshot:
    """Snapshot of placement state at one iteration."""

    iteration: int = 0
    score: float = 0.0
    max_displacement: float = 0.0
    stagnant_count: int = 0
    overlap_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "score": round(self.score, 2),
            "max_displacement": round(self.max_displacement, 2),
            "stagnant": self.stagnant_count,
            "overlaps": self.overlap_count,
        }


@dataclass
class PlacementScore:
    """Scores a placement configuration before routing.
    Higher is better for all fields (0-100 scale)."""

    total: float = 0.0
    net_distance: float = 0.0  # how close connected components are
    crossover_count: int = 0  # estimated ratsnest crossings
    crossover_score: float = 0.0  # 100 = zero crossings
    compactness: float = 0.0  # board utilization
    edge_compliance: float = 0.0  # connectors/holes on edges
    rotation_score: float = 0.0  # pad alignment quality
    board_containment: float = 0.0  # % of pads/bodies inside board outline
    courtyard_overlap: float = 0.0  # 100 = no overlaps
    smt_opposite_tht: float = 100.0  # SMT-over-THT board space utilization
    group_coherence: float = 100.0  # functional group compactness (100 = perfect)
    aspect_ratio: float = 100.0  # 100 = square board, penalized for elongated boards
    topology_structure: float = (
        100.0  # 100 = topology-aware passive chains stay ordered around anchors
    )
    block_opposite_side: float = 0.0  # parent-side: reward stacking of
    # blocker-compatible pairs (front-only x back-only). 100 = every
    # compatible pair fully overlaps; 0 = none overlap. Stays at 0 for
    # leaf placement (no synthetic blocks present).

    def compute_total(self, weights: Optional[dict[str, float]] = None) -> float:
        w = weights or {
            "net_distance": 0.20,  # connected parts close together
            "crossover_score": 0.17,  # fewer crossings = easier routing
            "compactness": 0.01,  # tighter layouts = smaller boards
            "edge_compliance": 0.10,
            "rotation_score": 0.00,
            "board_containment": 0.12,
            "courtyard_overlap": 0.10,
            "smt_opposite_tht": 0.15,  # SMT on opposite side of THT
            "group_coherence": 0.08,  # functional groups stay compact
            "aspect_ratio": 0.02,  # penalize elongated board shapes
            "topology_structure": 0.05,  # reward topology-aware passive ordering
            "block_opposite_side": 0.0,  # parent-side: reward stacking
            # blocker-compatible (front-only x back-only) block pairs
            # so SMT leaves migrate onto large back-side THT footprints.
            # Plumbing is in place but the default weight is 0 -- the
            # _place_clusters initial placement already puts SMT blocks
            # in a connectivity-driven cluster, and SA refinement
            # consistently finds no nearby improvement that would
            # actually start the stacking. Achieving stacking requires
            # either a stronger initial placement hint that seeds SMT
            # blocks inside large back-side block bboxes, or a much
            # higher weight here paired with a stronger force-phase
            # attraction. Track as follow-up.
        }
        self.total = sum(getattr(self, k) * v for k, v in w.items())
        return self.total


@dataclass
class DRCScore:
    """DRC violation penalties. Higher = fewer violations. 0-100 scale."""

    total: float = 100.0
    shorts: float = 100.0
    unconnected: float = 100.0
    clearance: float = 100.0
    courtyard: float = 100.0

    @staticmethod
    def from_counts(drc_dict: dict[str, Any]) -> "DRCScore":
        """Convert quick_drc() output dict to DRCScore on 0-100 scale."""
        import math

        def _violation_score(count: int, weight: float) -> float:
            if count == 0:
                return weight
            return max(0.0, weight * (1 - math.log10(1 + count) / math.log10(100)))

        s = DRCScore()
        s.shorts = _violation_score(drc_dict.get("shorts", 0), 40)
        s.unconnected = _violation_score(drc_dict.get("unconnected", 0), 30)
        s.clearance = _violation_score(drc_dict.get("clearance", 0), 20)
        s.courtyard = _violation_score(drc_dict.get("courtyard", 0), 10)
        s.total = s.shorts + s.unconnected + s.clearance + s.courtyard
        return s



# ---------------------------------------------------------------------------
# Hierarchical group placement data structures
# ---------------------------------------------------------------------------


class InterfaceRole(str, Enum):
    POWER_IN = "power_in"
    POWER_OUT = "power_out"
    GROUND = "ground"
    SIGNAL_IN = "signal_in"
    SIGNAL_OUT = "signal_out"
    BIDIR = "bidir"
    DIFF_P = "diff_p"
    DIFF_N = "diff_n"
    BUS = "bus"
    ANALOG = "analog"
    TEST = "test"
    MECHANICAL = "mechanical"
    UNKNOWN = "unknown"


class InterfaceDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    PASSIVE = "passive"
    UNKNOWN = "unknown"


class InterfaceSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ANY = "any"


class SubcircuitAccessPolicy(str, Enum):
    INTERFACE_ONLY = "interface_only"
    OPEN_ACCESS = "open_access"


@dataclass(frozen=True, slots=True)
class SubCircuitId:
    """Stable identity for a schematic sheet instance."""

    sheet_name: str
    sheet_file: str
    instance_path: str
    parent_instance_path: str | None = None

    @property
    def path_key(self) -> str:
        return self.instance_path or self.sheet_file


@dataclass(slots=True)
class InterfacePort:
    """Normalized external interface for a subcircuit."""

    name: str
    net_name: str
    role: InterfaceRole = InterfaceRole.BIDIR
    direction: InterfaceDirection = InterfaceDirection.UNKNOWN
    preferred_side: InterfaceSide = InterfaceSide.ANY
    access_policy: SubcircuitAccessPolicy = SubcircuitAccessPolicy.INTERFACE_ONLY
    cardinality: int = 1
    bus_index: int | None = None
    required: bool = True
    description: str = ""
    raw_direction: str = ""
    source_uuid: str | None = None
    source_kind: str = "sheet_pin"


@dataclass(slots=True)
class InterfaceAnchor:
    """Physical anchor point for a normalized interface on a solved layout."""

    port_name: str
    pos: Point
    layer: Layer = Layer.FRONT
    pad_ref: tuple[str, str] | None = None


@dataclass
class SubCircuitDefinition:
    """Logical subcircuit definition derived from schematic hierarchy."""

    id: SubCircuitId
    schematic_path: str = ""
    component_refs: list[str] = field(default_factory=list)
    ports: list[InterfacePort] = field(default_factory=list)
    child_ids: list[SubCircuitId] = field(default_factory=list)
    parent_id: SubCircuitId | None = None
    is_leaf: bool = True
    sheet_uuid: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.id.sheet_name


@dataclass
class SubCircuitLayout:
    """Frozen solved layout artifact for a subcircuit."""

    subcircuit_id: SubCircuitId
    components: dict[str, Component] = field(default_factory=dict)
    traces: list[TraceSegment] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    silkscreen: list[SilkscreenElement] = field(default_factory=list)
    bounding_box: tuple[float, float] = (0.0, 0.0)
    ports: list[InterfacePort] = field(default_factory=list)
    interface_anchors: list[InterfaceAnchor] = field(default_factory=list)
    score: float = 0.0
    artifact_paths: dict[str, str] = field(default_factory=dict)
    frozen: bool = True

    @property
    def width(self) -> float:
        return self.bounding_box[0]

    @property
    def height(self) -> float:
        return self.bounding_box[1]

    @property
    def area(self) -> float:
        return self.bounding_box[0] * self.bounding_box[1]


@dataclass(slots=True)
class SolveRoundResult:
    """One local placement-search round for a leaf subcircuit."""

    round_index: int
    seed: int
    score: float
    placement: PlacementScore
    components: dict[str, Component] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    routed: bool = False
    timing_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        routing = {
            key: value for key, value in self.routing.items() if not key.startswith("_")
        }
        return {
            "round_index": self.round_index,
            "seed": self.seed,
            "score": self.score,
            "routed": self.routed,
            "placement": {
                "total": self.placement.total,
                "net_distance": self.placement.net_distance,
                "crossover_count": self.placement.crossover_count,
                "crossover_score": self.placement.crossover_score,
                "compactness": self.placement.compactness,
                "edge_compliance": self.placement.edge_compliance,
                "rotation_score": self.placement.rotation_score,
                "board_containment": self.placement.board_containment,
                "courtyard_overlap": self.placement.courtyard_overlap,
                "smt_opposite_tht": self.placement.smt_opposite_tht,
                "group_coherence": self.placement.group_coherence,
                "aspect_ratio": self.placement.aspect_ratio,
            },
            "routing": routing,
            "timing_breakdown": dict(self.timing_breakdown),
            "preview_paths": {
                "pre_route_front": routing.get("round_preview_pre_route_front", ""),
                "pre_route_back": routing.get("round_preview_pre_route_back", ""),
                "pre_route_copper": routing.get("round_preview_pre_route_copper", ""),
                "routed_front": routing.get("round_preview_routed_front", ""),
                "routed_back": routing.get("round_preview_routed_back", ""),
                "routed_copper": routing.get("round_preview_routed_copper", ""),
            },
            "board_paths": {
                "illegal_pre_stamp": routing.get("round_board_illegal_pre_stamp", ""),
                "pre_route": routing.get("round_board_pre_route", ""),
                "routed": routing.get("round_board_routed", ""),
            },
            "log_summary": {
                "router": routing.get("router", ""),
                "reason": routing.get("reason", ""),
                "failed": bool(routing.get("failed", False)),
                "skipped": bool(routing.get("skipped", False)),
                "traces": int(routing.get("traces", 0) or 0),
                "vias": int(routing.get("vias", 0) or 0),
                "total_length_mm": float(routing.get("total_length_mm", 0.0) or 0.0),
                "failed_internal_nets": list(
                    routing.get("failed_internal_nets", []) or []
                ),
                "routed_internal_nets": list(
                    routing.get("routed_internal_nets", []) or []
                ),
            },
        }


@dataclass(slots=True)
class SubCircuitInstance:
    """Placed instance of a frozen subcircuit inside a parent composition."""

    layout_id: SubCircuitId
    origin: Point
    rotation: float = 0.0
    access_policy: SubcircuitAccessPolicy = SubcircuitAccessPolicy.INTERFACE_ONLY
    transformed_bbox: tuple[float, float] = (0.0, 0.0)


@dataclass
class HierarchyLevelState:
    """Composition state for one hierarchy level."""

    subcircuit: SubCircuitDefinition
    child_instances: list[SubCircuitInstance] = field(default_factory=list)
    local_components: dict[str, Component] = field(default_factory=dict)
    interconnect_nets: dict[str, Net] = field(default_factory=dict)
    board_outline: tuple[Point, Point] = field(
        default_factory=lambda: (Point(0, 0), Point(0, 0))
    )
    constraints: dict[str, object] = field(default_factory=dict)


@dataclass
class FunctionalGroup:
    """A functional group of components that belong together (e.g. one IC and
    its supporting passives, as defined by a schematic sub-sheet)."""

    name: str  # Human-readable name (e.g. "USB INPUT")
    leader_ref: str  # Primary component reference (e.g. "U1")
    member_refs: list[str]  # All component refs including leader
    inter_group_nets: list[str] = field(
        default_factory=list
    )  # Nets connecting to other groups


@dataclass
class GroupSet:
    """Complete set of functional groups for a project."""

    groups: list[FunctionalGroup] = field(default_factory=list)
    ungrouped_refs: list[str] = field(
        default_factory=list
    )  # Components not in any group
    source: str = "auto"  # "schematic", "netlist", "manual", "auto"

    def ref_to_group(self) -> dict[str, FunctionalGroup]:
        """Build reverse map: component ref -> its FunctionalGroup."""
        mapping = {}
        for group in self.groups:
            for ref in group.member_refs:
                mapping[ref] = group
        return mapping

    def ref_to_leader(self) -> dict[str, str]:
        """Build reverse map: component ref -> group leader ref."""
        mapping = {}
        for group in self.groups:
            for ref in group.member_refs:
                mapping[ref] = group.leader_ref
        return mapping


@dataclass
class PlacedGroup:
    """A functional group after intra-group placement.

    Component positions are stored relative to the group origin (0, 0).
    The bounding_box gives the overall envelope of the placed group.
    """

    group: FunctionalGroup
    bounding_box: tuple[float, float]  # (width, height) in mm
    component_positions: dict[
        str, tuple[float, float, float]
    ]  # ref -> (rel_x, rel_y, rotation)
    component_layers: dict[str, Layer] = field(default_factory=dict)  # ref -> layer

    @property
    def width(self) -> float:
        return self.bounding_box[0]

    @property
    def height(self) -> float:
        return self.bounding_box[1]

    @property
    def area(self) -> float:
        return self.bounding_box[0] * self.bounding_box[1]
