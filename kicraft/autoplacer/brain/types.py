"""Shared data structures for the autoplacer system.

All types are plain Python dataclasses — no pcbnew imports.
These serve as the interchange format between Brain and Hardware layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import atan2, hypot
from typing import Any, Optional


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
    pos: Point  # absolute position in mm
    net: str  # net name
    layer: Layer


@dataclass
class Component:
    ref: str
    value: str
    pos: Point
    rotation: float  # degrees
    layer: Layer
    width_mm: float  # bounding box width
    height_mm: float  # bounding box height
    pads: list[Pad] = field(default_factory=list)
    locked: bool = False
    kind: str = ""  # "connector", "mounting_hole", "ic", "passive", "misc"
    is_through_hole: bool = False  # True if footprint has PTH pads
    body_center: Point | None = None  # courtyard/body bbox center (absolute coords)
    opening_direction: float | None = (
        None  # LOCAL-frame angle (0/90/180/270) where opening faces
    )

    @property
    def area(self) -> float:
        return self.width_mm * self.height_mm

    def bbox(self, clearance: float = 0.0) -> tuple[Point, Point]:
        """Return (top_left, bottom_right) with optional clearance margin.

        Centers the bounding box on body_center (courtyard geometric center)
        when available, falling back to pos (footprint origin).  This is
        critical for components where the origin differs from the courtyard
        center (e.g. battery holders, some connectors) — using pos would
        produce a shifted bbox that misses real overlaps.
        """
        hw = self.width_mm / 2 + clearance
        hh = self.height_mm / 2 + clearance
        cx = self.body_center.x if self.body_center else self.pos.x
        cy = self.body_center.y if self.body_center else self.pos.y
        return (
            Point(cx - hw, cy - hh),
            Point(cx + hw, cy + hh),
        )


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


@dataclass
class BoardState:
    """Complete snapshot — the interchange format between Brain and Hardware."""

    components: dict[str, Component] = field(default_factory=dict)  # ref -> Component
    nets: dict[str, Net] = field(default_factory=dict)  # name -> Net
    traces: list[TraceSegment] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
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

    def compute_total(self, weights: Optional[dict[str, float]] = None) -> float:
        w = weights or {
            "net_distance": 0.20,  # connected parts close together
            "crossover_score": 0.17,  # fewer crossings = easier routing
            "compactness": 0.02,  # tighter layouts = smaller boards
            "edge_compliance": 0.10,
            "rotation_score": 0.01,
            "board_containment": 0.12,
            "courtyard_overlap": 0.10,
            "smt_opposite_tht": 0.10,  # SMT on opposite side of THT
            "group_coherence": 0.08,  # functional groups stay compact
            "aspect_ratio": 0.05,  # penalize elongated board shapes
            "topology_structure": 0.05,  # reward topology-aware passive ordering
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


@dataclass
class ExperimentScore:
    """Unified score combining placement + routing quality.
    Single metric for the outer optimization loop. Higher = better."""

    placement: PlacementScore = field(default_factory=PlacementScore)
    routed_nets: int = 0
    total_nets: int = 0
    failed_nets: int = 0
    trace_count: int = 0
    via_count: int = 0
    total_trace_length_mm: float = 0.0
    total: float = 0.0
    # Phase timing (ms)
    placement_ms: float = 0.0
    routing_ms: float = 0.0
    failed_net_names: list[str] = field(default_factory=list)
    drc_score: DRCScore = field(default_factory=DRCScore)
    pipeline_drc: dict[str, Any] = field(default_factory=dict)
    skipped_routing: bool = False

    def compute(
        self,
        weights: Optional[dict[str, float]] = None,
        drc_dict: Optional[dict[str, Any]] = None,
        board_area_mm2: Optional[float] = None,
    ) -> float:
        """Compute unified score. Route completion dominates, then placement + DRC.

        If board_area_mm2 is provided (from board size search), an area bonus
        rewards smaller boards.
        """
        w = weights or {}
        w_placement = w.get("placement", 0.15)
        w_route = w.get("route_completion", 0.50)
        w_via = w.get("via_penalty", 0.10)
        w_contain = w.get("containment", 0.05)
        w_drc = w.get("drc", 0.20)

        # Route completion: most important — must get all nets routed
        if self.total_nets > 0:
            route_pct = ((self.total_nets - self.failed_nets) / self.total_nets) * 100
        else:
            # No nets counted = routing was skipped or failed. Score 0, not 100.
            route_pct = 0.0

        # Trace length efficiency: penalize long traces relative to board perimeter
        # Good routing should have total length roughly proportional to
        # (num_nets * avg_half_perimeter). Use board_area to estimate.
        if self.total_trace_length_mm > 0 and self.routed_nets > 0:
            # Estimate optimal: ~15mm baseline + sqrt(board_area)/2 per net
            if board_area_mm2 and board_area_mm2 > 0:
                import math as _m

                est_half_perim = _m.sqrt(board_area_mm2) * 2
                optimal_estimate = self.routed_nets * (est_half_perim / 4 + 10)
            else:
                optimal_estimate = self.routed_nets * 25.0
            length_ratio = min(1.0, optimal_estimate / self.total_trace_length_mm)
            trace_length_score = length_ratio * 100.0
        else:
            trace_length_score = 50.0

        # Via penalty: fewer vias per routed net = better
        # Blended with trace length efficiency for joint penalization
        if self.routed_nets > 0:
            vias_per_net = self.via_count / self.routed_nets
            raw_via_score = max(0, min(100, 100 - vias_per_net * 20))
            # Blend: 40% via penalty + 60% trace length efficiency
            via_score = 0.4 * raw_via_score + 0.6 * trace_length_score
        elif self.skipped_routing:
            via_score = 0.0  # No routing = no via credit
        else:
            via_score = 50.0

        # DRC score — no credit when routing was skipped
        if drc_dict:
            self.drc_score = DRCScore.from_counts(drc_dict)
        if self.skipped_routing:
            drc_val = 0.0
        else:
            drc_val = self.drc_score.total

        self.total = (
            w_placement * self.placement.total
            + w_route * route_pct
            + w_via * via_score
            + w_contain * self.placement.board_containment
            + w_drc * drc_val
        )

        # Smooth score gating: continuous cap based on route completion
        # Smoothly ramps from cap=40 at route_pct=50 to cap=100 at route_pct=95
        if route_pct < 95.0:
            t = max(0.0, min(1.0, (route_pct - 50.0) / 45.0))
            # Smooth hermite interpolation (smoothstep)
            smooth_t = t * t * (3.0 - 2.0 * t)
            cap = 40.0 + smooth_t * 60.0
            self.total = min(self.total, cap)

        # Area bonus: reward smaller boards (only when board size search is active)
        if board_area_mm2 is not None:
            import math as _math

            # Nonlinear area scoring: exponential decay rewards being closer
            # to the minimum viable area.  A board at min_area scores ~85,
            # at 2x min_area scores ~35, at max_area scores ~10.
            max_area = 120.0 * 80.0  # generous upper bound
            # Use a reference area proportional to a reasonable min (40% of max)
            ref_area = max_area * 0.4
            area_score = max(
                0.0, min(100.0, 100.0 * _math.exp(-board_area_mm2 / (1.8 * ref_area)))
            )
            w_area = w.get("area", 0.15)
            # Scale other weights down proportionally
            scale = 1.0 - w_area
            self.total = self.total * scale + w_area * area_score

        return self.total

    def summary(self) -> str:
        return (
            f"score={self.total:.1f} "
            f"routed={self.routed_nets}/{self.total_nets} "
            f"failed={self.failed_nets} "
            f"traces={self.trace_count} vias={self.via_count} "
            f"length={self.total_trace_length_mm:.0f}mm "
            f"placement={self.placement.total:.1f}"
        )


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
