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


# --- Edge-connector orientation helpers -----------------------------------
# Single source of truth for "which way should a connector face at a board
# edge". Shared by the single-board solver (_best_rotation_for_edge), the
# multi-leaf composer's leaf-rotation filter, and the post-compose
# orientation gate so the three can never drift apart.

# Board-space angle (deg, Y-down: 0=+X, 90=+Y, 180=-X, 270=-Y) that points
# OUTWARD from each named board edge. On B.Cu the local X-axis is mirrored
# by Flip(), so left/right swap.
_OUTWARD_FRONT = {"left": 180.0, "right": 0.0, "top": 270.0, "bottom": 90.0}
_OUTWARD_BACK = {"left": 0.0, "right": 180.0, "top": 270.0, "bottom": 90.0}


def edge_outward_angle(layer: "Layer", edge: str) -> float:
    """Board-space angle that points away from the named board edge."""
    table = _OUTWARD_BACK if layer == Layer.BACK else _OUTWARD_FRONT
    return table[edge]


def opening_board_angle(opening_direction: float, rotation: float) -> float:
    """Footprint-local ``opening_direction`` expressed in board space.

    Inverse of ``detect_opening_direction``'s ``local = (board + rotation)``:
    ``board = (local - rotation) % 360``. ``rotation`` is the footprint's
    board-space orientation (``GetOrientationDegrees``), which after a leaf
    transform is the connector's parent-space rotation.
    """
    return (opening_direction - rotation) % 360.0


def angles_close(a: float, b: float, tol: float = 1.0) -> bool:
    """True when two angles are equal modulo 360 within ``tol`` degrees."""
    return abs(((a - b + 180.0) % 360.0) - 180.0) <= tol


@dataclass(slots=True, frozen=True)
class BlockRotationGeometry:
    """Per-rotation bbox dimensions for a synthetic leaf block.

    Leaf blocks rotate around their body_center (the rotation pivot), so
    the body_center offset is rotation-invariant; only the AABB swaps
    width/height for 90°/270° rotations of axis-aligned content. The
    placement solver swaps these fields when trying alternate rotations
    of a block, since blocks have no pads to rotate via the IC/connector
    rotation path.
    """
    width_mm: float
    height_mm: float


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
    # Pad copper extent in mm -- WORLD-AXIS-ALIGNED width and height of the
    # pad's bounding box, with footprint rotation and pad-local rotation
    # already applied. Stored this way so bbox() can return the world AABB
    # by centering on ``pos`` without any rotation math.
    #
    # None = legacy artifact with no recorded size; bbox() then returns the
    # center as a degenerate point. Populated by hardware.adapter from
    # KiCad's Pad.GetBoundingBox() during board extraction and carried
    # through solved_layout.json round-trip. _transform_pad rotates this
    # again when a leaf is placed onto the parent at some rotation, so a
    # leaf-frame world-AABB becomes a parent-frame world-AABB.
    size_mm: Point | None = None

    def bbox(self) -> tuple[Point, Point]:
        """Pad copper bbox (top_left, bottom_right) in absolute world coords.

        ``size_mm`` is already world-axis-aligned (post-rotation), so the
        bbox is simply ``pos ± size_mm / 2``. No rotation math here -- the
        rotation was applied upstream when ``size_mm`` was populated by
        the adapter (or rotated by ``_rotate_size`` during placement).
        """
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
    # True for members of a programmatically-placed array/matrix grid (see
    # brain/array_placement.py). Such grids are intentionally positioned and
    # self-legal by construction; the overlap legalizer must NOT try to resolve
    # overlaps among them (a dense locked grid would thrash the O(n^2) escape
    # passes and never reach a "legal" state).
    array_member: bool = False
    kind: str = ""  # "connector", "mounting_hole", "ic", "passive", "misc"
    is_through_hole: bool = False  # True if footprint has PTH pads
    body_center: Point | None = None  # courtyard/body bbox center (absolute coords)
    opening_direction: float | None = (
        None  # LOCAL-frame angle (0/90/180/270) where opening faces
    )
    block_blocker_set: LeafBlockerSet | None = None
    block_artifact_origin_offset: Point | None = None
    block_side: str | None = None
    # Project-level override: when True, can_overlap_sparse treats this
    # leaf as having NO front-side intent regardless of what its blocker
    # set looks like. Set from
    # cfg["parent_placement"]["backside_through_hole_leaves"] for
    # leaves whose ``dominant_blocker_side`` cannot be inferred from
    # geometry alone (a pure-THT screw terminal has no real F.Cu or
    # B.Cu intent — annular ring shadow lives in ``front_tht_pads`` /
    # ``back_tht_pads`` and intentionally doesn't count as layer
    # commitment, so the auto-detection returns "none" and the user
    # must declare which side the leaf is meant to live on). The flag
    # rides on the synthetic block component so the bridge between
    # project config and the leaf-pair compatibility check is local
    # to compose-time setup; everything downstream (placement solver,
    # scorer) reads it via the predicate.
    block_force_back_only: bool = False
    # Reference of the locked anchor this candidate was deliberately
    # row/col-packed onto by ``_stack_compatible_blocks``. When set,
    # ``_resolve_overlaps`` must skip its full-bbox escape against
    # that anchor: the stacking placement is intentional, the
    # candidate's pads are guaranteed to clear the anchor's pads at
    # placement time, and any tiny pad-vs-corner-ring drift from the
    # free-free push pass would otherwise flip the position-dependent
    # predicate False and trigger a 30-50 mm escape that unwinds the
    # whole stacking pass. Cleared on un-stacking (e.g. when the
    # candidate is moved by SA outside its anchor's bbox by intent).
    block_stacked_anchor: str | None = None
    # Reference of the host block whose ENCLOSED interior hole this block
    # was deliberately nested into by ``_nest_blocks_in_interior_holes``
    # (shaped compose -- docs/plans/shaped-compose-leaf-nesting.md). Both
    # partners are locked at nest time, so later passes cannot drift the
    # pair into the partial overlap the same-side veto exists to prevent;
    # the field additionally exempts the pair from the courtyard-overlap
    # separation pass (block bboxes overlap by design; the real
    # per-footprint courtyards do not -- that is what the containment
    # allowance in ``can_overlap_sparse`` guarantees).
    block_nested_anchor: str | None = None
    allowed_rotations: list[float] | None = None
    # Per-rotation bbox dimensions for synthetic leaf blocks. Keyed by
    # rotation degrees (0/90/180/270). For 90° and 270° the width/height
    # values swap relative to 0°/180°. Used by the placement solver to
    # try alternate block rotations without recomputing geometry.
    block_rotation_geometry: dict[float, "BlockRotationGeometry"] | None = None

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


@dataclass(slots=True)
class KeepoutRect:
    """Axis-aligned keep-out region (board coords) the placer keeps parts out of.

    Used for RF antenna near-field clearances (and any future keep-clear). The
    ``owner_ref`` component is exempt -- it is the part the keep-out belongs to
    (e.g. the ESP32 whose antenna this protects), so its own courtyard may
    overlap. ``source`` is "preserve" (a footprint-internal rule-area) or
    "inject" (a config antenna_keepouts family-spec rect), for diagnostics.

    ``tl``/``br`` are board coords sampled at extraction time. ``owner_origin``
    is the owner footprint's position at that same instant: because the keep-out
    is rigidly attached to the owner, the placer translates the rect by the
    owner's displacement since extraction (``owner.pos - owner_origin``) so the
    rect tracks the owner as the solve moves it -- without this the rect goes
    stale the moment the (unlocked) owner is nudged, and parts get pushed out of
    where the antenna *was*, not where it *is*.
    """

    tl: Point
    br: Point
    owner_ref: str
    source: str = ""
    owner_origin: Point | None = None

@dataclass(slots=True, frozen=True)
class AntennaEdgeIntent:
    """Serializable footprint-local antenna anchor and requested board edge.

    ``local_polygon`` and ``local_anchor_midpoint`` stay in the unrotated
    footprint frame.  This lets leaf and parent placement apply the same
    KiCad-clockwise transform without rediscovering semantics from a ref or
    footprint name.
    """

    owner_ref: str
    source: str
    source_id: str
    local_direction: str
    local_anchor_mm: float
    local_anchor_midpoint: Point
    local_polygon: tuple[Point, ...]
    target_edge: str
    inset_mm: float = 0.0
    explicit_edge: bool = False
    explicit_rotation: bool = False




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
    # RF antenna keep-clear rects (board coords), populated by adapter.load via
    # hardware.keepout_extract. The placer pushes non-owner parts out of each.
    keepout_rects: list[KeepoutRect] = field(default_factory=list)
    # Accepted antenna edge contracts. Geometry remains footprint-local so it
    # survives placement, leaf re-basing, and rigid parent composition.
    antenna_edge_intents: list[AntennaEdgeIntent] = field(default_factory=list)

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



# Default weighting of the PlacementScore sub-scores -- the single source of
# truth for what "good placement" means. Tunable per-config via ``psw_<key>``
# overlays (see ``placement_weights_from_config``); the literal values here are
# the baseline so a config that overrides nothing scores byte-identically.
DEFAULT_PLACEMENT_WEIGHTS: dict[str, float] = {
    "net_distance": 0.20,  # connected parts close together
    "crossover_score": 0.17,  # fewer crossings = easier routing
    "compactness": 0.00,  # absorbed by bbox_packing (was 0.01); seed-frame
    # ratio is constant within a solve so SA cannot move it.
    "edge_compliance": 0.10,
    "rotation_score": 0.00,
    "board_containment": 0.12,
    "courtyard_overlap": 0.10,
    "smt_opposite_tht": 0.15,  # SMT on opposite side of THT
    "group_coherence": 0.08,  # functional groups stay compact
    "aspect_ratio": 0.02,  # penalize elongated board shapes. NOTE
    # (area-compaction Phase 2): a raise to ~0.08 was tried and REGRESSED
    # parent compose/route on multi-leaf boards (535/530 replay A/B) --
    # the weight also steers the parent block placement, which was tuned
    # against 0.02. Re-raise only via the CMA-ES tuner re-run, not by hand.
    "topology_structure": 0.05,  # reward topology-aware passive ordering
    "bbox_packing": 0.15,  # tight packing vs placed bbox (dynamic
    # under SA, unlike compactness which is fixed for the solve).
    # Bumped from 0.01 because the post-rotation-fix solver was
    # producing sprawling placements (board height 170-250 mm) on
    # most seeds -- with the predicate fix preventing leaf overlap,
    # nothing else in the score function was pulling leaves
    # together, so SA had no signal to compact. 0.15 is on par
    # with smt_opposite_tht so compactness competes with stacking.
    "pin_locality": 0.0,  # do passives hug the anchor pins they connect to?
    # 0 by default (byte-identical until a config opts in via psw_pin_locality);
    # the connectivity-first leaf path sets a real weight. This is the objective
    # the discrete-grid assignment optimizes -- a decap next to its IC power/GND
    # pins, not floating tidily 6-20 mm away. Unlike net_distance (which excludes
    # GND from the wirelength MST), this term DOES pull toward the IC's GND pin.
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


def placement_weights_from_config(cfg: Optional[dict] = None) -> dict[str, float]:
    """Build a PlacementScore weight dict from a project config.

    Returns ``DEFAULT_PLACEMENT_WEIGHTS`` overlaid with any ``psw_<key>`` entries
    present in ``cfg`` (e.g. ``psw_bbox_packing`` overrides the ``bbox_packing``
    weight). A config that sets no ``psw_*`` keys yields the defaults verbatim, so
    placement scoring is unchanged until a tuned config supplies weights.
    """
    weights = dict(DEFAULT_PLACEMENT_WEIGHTS)
    if cfg:
        for key in DEFAULT_PLACEMENT_WEIGHTS:
            cv = cfg.get(f"psw_{key}")
            if cv is not None:
                weights[key] = float(cv)
    return weights


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
    bbox_packing: float = 100.0  # tight packing vs placed bbox; 100 when <2 comps
    pin_locality: float = 100.0  # passives hug the anchor pins they connect to;
    # 100 = every decap ~0 mm from its IC power/GND pins, 100 when nothing scorable.
    # Weight 0 by default; the connectivity-first leaf path drives it.

    def compute_total(self, weights: Optional[dict[str, float]] = None) -> float:
        w = weights or DEFAULT_PLACEMENT_WEIGHTS
        # Normalize by the weight sum so total is a true 0-100 weighted
        # average. The literal weights above sum to ~1.14, so without this
        # a strong placement can score >100 (observed: 103.67), which both
        # breaks the "0-100 scale" contract and lets an unrouted leaf
        # out-score a routed one. Dividing by the weight sum preserves the
        # relative ordering SA optimizes while bounding the result to 100.
        weight_sum = sum(w.values()) or 1.0
        self.total = sum(getattr(self, k) * v for k, v in w.items()) / weight_sum
        return self.total



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
    antenna_edge_intents: list[AntennaEdgeIntent] = field(default_factory=list)
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
    # Placement-quality evidence for this round: the discrete grid's slot
    # provisioning + accept-if-better verdict, and the honest median/max
    # pad->pin distance the router was handed.
    placement_diagnostics: dict[str, Any] = field(default_factory=dict)

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
            "placement_diagnostics": dict(self.placement_diagnostics),
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


