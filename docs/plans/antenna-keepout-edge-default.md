# Antenna keepout edge-default placement

**Status:** implemented; focused verification and one cold production-project replay complete; remaining replay corpus pending
**Scope:** generic antenna-bearing footprints in every generated design, with first-class support for ESP32 modules carrying footprint-local antenna rule areas  
**Supersedes:** the antenna-placement phases of `docs/plans/kc-va3uu7-keepout-gates-and-esp32-antenna-edge-default.md` and `docs/plans/antenna-edge-flush-plan.md`

## 1. Goal

When a footprint has authoritative antenna keepout geometry, KiCraft should by default:

1. infer which side of the footprint radiates outward;
2. rotate the footprint so that side faces an exterior board edge;
3. place the outward boundary of the antenna keepout directly on that board edge;
4. preserve the same antenna direction and anchor through leaf routing, parent composition, outline repair, promotion, and verification;
5. keep all pads and ordinary copper inboard while keeping other components out of the RF keepout;
6. apply this to every design without relying on `ANT*`, `U1`, `ESP32`, connector classification, or a hard-coded module list;
7. preserve explicit user/manual/form-factor placement as higher-precedence intent.

The default target edge is `top` for deterministic results. Projects may select another edge per antenna or globally.

“Directly on the edge” has one exact meaning: after placement, the keepout polygon’s outward support line and the selected `Edge.Cuts` side are coincident within tolerance. It does **not** mean body-bbox flush, courtyard flush, connector-mouth flush, or “within a few millimetres.”

## 2. Why this is a separate change

The keepout-copper correctness work already prevents and rejects illegal tracks/vias. Antenna edge placement is a different contract: it changes orientation, placement, leaf outline ownership, parent rotation candidates, and final board outline. It should consume the shared keepout extraction layer but must not be coupled to DRC acceptance plumbing.

The earlier plans are too ESP32-specific in intent and too connector-centric in implementation. The generic source of truth is antenna keepout geometry plus explicit antenna semantics. Reference prefixes and footprint-name heuristics are insufficient and misclassify unrelated parts.

## 3. Required invariants

### 3.1 Detection

- A footprint is automatically antenna-bearing only when at least one authoritative source identifies antenna keepout geometry:
  1. a footprint-local rule area whose name matches configured semantic patterns such as `*antenna*keepout*`, `*antenna_keepout*`, or `*rf*keepout*`;
  2. a match in the existing `cfg["antenna_keepouts"]` family map;
  3. an explicit project antenna declaration.
- A generic unnamed track/via rule area is **not** enough. Mounting-hole, high-voltage, mechanical, and connector rule areas must not become antenna constraints.
- Detection is independent of reference, component kind, BOM description, and connector opening detection.
- Footprint-local geometry wins over an injected family rectangle for the edge anchor when both exist. The injected rectangle remains an additional component/RF exclusion region.
- Ambiguous geometry must produce a bounded diagnostic and no inferred default; it must never guess a direction.

### 3.2 Geometry

- Preserve the actual footprint-local polygon in its unrotated local frame. Do not reduce the antenna anchor contract to a board-coordinate AABB.
- Infer the local outward cardinal direction from the antenna keepout’s displacement relative to the footprint’s physical/pad envelope:
  - choose the dominant signed axis from envelope center to keepout centroid;
  - require a configurable dominance ratio and minimum displacement;
  - otherwise mark the direction ambiguous.
- The local anchor is the keepout polygon’s outward support line:
  - local left: minimum X;
  - local right: maximum X;
  - local top: minimum Y;
  - local bottom: maximum Y.
- The anchor should retain a midpoint for diagnostics and an entire support-line coordinate for placement. A midpoint alone is insufficient for non-rectangular polygons.
- Use KiCad clockwise-positive transforms through `brain.geometry.rotate_vector` / `transform_point` only.
- At the default inset of `0.0 mm`, transformed support line and selected board edge coincide.
- Positive inset moves the keepout inward. Negative inset is an explicit overhang and must not be enabled by default.

### 3.3 Placement precedence

Highest precedence first:

1. manual layout / user-authored outline and coordinates;
2. locked form-factor scaffold placement;
3. explicit `component_zones[ref]` edge and rotation;
4. explicit antenna declaration for the ref;
5. inferred antenna edge default;
6. generic solver heuristics.

Rules:

- The inference layer must build an effective config without mutating the user-loaded config.
- An explicit edge with no explicit rotation still gets antenna-aware outward orientation.
- An explicit rotation is preserved verbatim. If it points the antenna away from the explicit edge, return `antenna_edge_orientation_conflict`; do not silently rotate it.
- Manual or scaffold placement is not moved by the default. Verification should record the measured antenna condition but must not reinterpret an explicit placement as inferred intent.
- `antenna_edge_pin_enabled=false` disables only inferred defaults, not explicit antenna constraints.

### 3.4 Leaf placement

- Inferred antennas participate in existing edge-group packing, including multiple antennas on one edge.
- Rotation is selected from `0/90/180/270` so the transformed local antenna vector matches the selected outward board-edge vector.
- Placement perpendicular to the edge is solved from the antenna support line and `antenna_edge_inset_mm`, not body/courtyard half extents or connector inset/overhang.
- Motion parallel to the edge may be packed/jittered; motion perpendicular to the edge is locked.
- `_pinned_targets` must restore both position and rotation after all legalization/compaction passes.
- All electrical pads must remain inside `Edge.Cuts` by `pad_inset_margin_mm`. If keepout-edge flush conflicts with pad containment, reject the placement round as `antenna_edge_pad_conflict`; never pull the antenna inboard silently.
- Existing owner exemption remains: the antenna footprint may overlap its own keepout. Other components may not.

### 3.5 Leaf and parent outlines

- The selected antenna side is anchor-authoritative.
- `_outline_around_geometry` and leaf-size reduction must set that side to the transformed antenna support line plus configured inset. Generic silk/body margin must not grow material past it.
- Other outline sides retain normal margin and copper/pad containment behavior.
- Parent composition must preserve the antenna direction when rotating the leaf block. Rotation candidates that turn the antenna inward or toward another side are invalid.
- The antenna anchor must flow through `AttachmentConstraint`/`PlacementConstraintEntry`; parent composition must not rediscover it from a transformed board, ref name, or footprint name.
- `edge_zoned_outline_sides` must include antenna-owned sides so `_repair_parent_outline` does not bury the antenna behind generic margin.
- `_ensure_edge_blocks_extremal` must keep the antenna-bearing leaf exterior. No sibling leaf may extend beyond the antenna support line on the RF side.
- Pads, tracks, and vias remain subject to normal inboard/copper-edge containment. Only non-copper antenna/body geometry may overhang when explicitly configured.

### 3.6 Verification

For every persisted antenna edge intent:

- anchor gap outside tolerance: `antenna_stranded:<ref>@<gap>(<edge>)`;
- transformed direction does not face the selected edge: `antenna_misoriented:<ref>(<actual>-><expected>)`;
- pad containment failure: existing geometry blocker plus `antenna_edge_pad_conflict` at placement time;
- missing footprint/intent owner: `antenna_constraint_owner_missing:<ref>`.

Inferred constraints are hard build contracts once accepted into a placement round. A board must not claim successful antenna edge defaulting while exporting an inboard or inward-facing antenna.

## 4. Data model and configuration

### 4.1 Defaults

Add to `kicraft/autoplacer/config.py`:

```python
"antenna_edge_pin_enabled": True,
"antenna_default_edge": "top",
"antenna_edge_inset_mm": 0.0,
"antenna_edge_tolerance_mm": 0.10,
"antenna_direction_min_offset_mm": 0.5,
"antenna_direction_dominance_ratio": 1.25,
"antenna_rule_area_name_patterns": [
    "*antenna*keepout*",
    "*antenna_keepout*",
    "*rf*keepout*",
],
"antenna_components": {},
```

`antenna_components` is the explicit escape hatch and custom-footprint contract:

```json
{
  "U1": {"edge": "right"},
  "ANT1": {
    "edge": "top",
    "local_direction": "top",
    "anchor_mm": -3.5,
    "rotation": 0
  }
}
```

Do not reuse connector inset, connector overhang, opening direction, or stranded tolerances.

### 4.2 Serializable intent

Add a compact value object in `brain/types.py`:

```python
@dataclass(frozen=True, slots=True)
class AntennaEdgeIntent:
    owner_ref: str
    source: Literal["footprint_rule_area", "family_config", "explicit"]
    source_id: str
    local_direction: Literal["left", "right", "top", "bottom"]
    local_anchor_mm: float
    local_anchor_midpoint: Point
    target_edge: Literal["left", "right", "top", "bottom"]
    inset_mm: float
    explicit_edge: bool
    explicit_rotation: bool
```

If polygon points are needed by diagnostics, persist them separately in bounded form; do not put pcbnew objects into this model.

Persist the effective intent with each solved leaf and parent result. JSON loading must reject malformed direction/edge values and tolerate artifacts produced before this field existed by treating them as having no antenna intent.

## 5. Implementation sequence

### Phase A — semantic antenna geometry extraction

**Files**

- `kicraft/autoplacer/hardware/keepout_extract.py`
- `kicraft/autoplacer/brain/types.py`
- `kicraft/autoplacer/config.py`
- `tests/test_keepout_extract.py`
- `tests/test_library_antenna_keepouts.py`

**Changes**

1. Add a pure extraction result carrying owner ref, source, local polygon/support line, inferred direction, and ambiguity diagnostic.
2. For footprint-local zones, read the rule-area name and exact placed polygon, then inverse-transform it into the unrotated footprint-local frame.
3. Match names case-insensitively against configured antenna semantic patterns.
4. For `antenna_keepouts` family rectangles, reuse `_footprint_name_candidates`, `_matches_family`, and the existing local rectangle values.
5. When both sources exist for one owner, use the named footprint-local antenna zone for direction/anchor and retain injected rectangles for placement exclusion only.
6. Support explicit `antenna_components` values as overrides or as complete geometry for custom parts.
7. Emit one bounded diagnostic per ambiguous owner; avoid logging full board/config data.

**Tests**

- ESP32-S3-MINI-1 and ESP32-S3-WROOM-1 built-in zones infer local top.
- ESP32-WROOM-32 family metadata infers local left.
- 0/90/180/270 placed rotations recover identical local semantics.
- A rotated/non-rectangular antenna polygon preserves its support line.
- Named mechanical/HV/mounting keepouts do not infer antenna intent.
- An unnamed track/via rule area does not infer intent.
- Built-in geometry wins over a larger injected near-field rectangle for anchoring.
- Centered/tied geometry is diagnosed and skipped.
- Explicit geometry resolves an otherwise ambiguous custom footprint.

### Phase B — effective leaf intent and persistence

**Files**

- `kicraft/autoplacer/hardware/adapter.py`
- `kicraft/cli/solve_subcircuits.py`
- leaf artifact serialization in `solve_subcircuits.py`
- `kicraft/autoplacer/brain/subcircuit_instances.py`
- artifact round-trip tests

**Changes**

1. Extract antenna geometry in the same board/config load that extracts placement keepouts.
2. Build an effective leaf constraint map:
   - start with explicit user `component_zones` and `antenna_components`;
   - add inferred `{edge: antenna_default_edge}` only for eligible owners absent from explicit placement;
   - preserve explicit rotations and locked/manual placement markers;
   - do not mutate shared config.
3. Carry `AntennaEdgeIntent` beside the effective component-zone entry rather than encoding semantics into arbitrary extra zone keys.
4. Persist intents in `metadata.json`, `debug.json`, and the solved-layout artifact field used by parent composition.
5. Load the persisted field in `load_solved_artifact`; do not rerun inference in the parent.
6. Include one concise diagnostic line per inferred antenna.

**Tests**

- Explicit component edge beats default top.
- Explicit antenna declaration beats family inference.
- Explicit rotation marker survives serialization.
- Kill switch removes inferred entries only.
- Old artifacts load with an empty intent list.
- User config is byte/deep-equality unchanged after effective config construction.

### Phase C — antenna-aware leaf edge placement

**Files**

- `kicraft/autoplacer/brain/placement_solver.py`
- `kicraft/autoplacer/brain/leaf_routing.py`
- `kicraft/autoplacer/brain/leaf_size_reduction.py`
- new `tests/test_antenna_edge_placement.py`

**Changes**

1. Extend `_pin_edge_components` with an antenna branch keyed only by effective intents.
2. Add a pure direction-to-rotation helper using KiCad clockwise rotation.
3. Preserve explicit rotation and reject incompatible edge/direction combinations.
4. Add antenna anchor versions of `_connector_edge_x/_connector_edge_y`; do not overload connector semantics.
5. Pack antennas through the existing edge-group parallel-axis machinery.
6. Restore antenna position and rotation after every pass that can move/rotate pinned parts.
7. Check pad containment after pinning and after final restoration.
8. Make `_outline_around_geometry` and leaf-size reduction anchor-authoritative on antenna sides.
9. Recompute measured direction/gap from final leaf geometry and persist it.

**Tests**

Parameterize all four target edges and local top/left directions:

- transformed antenna direction points outward;
- support-line gap is `0.0 ± tolerance`;
- every pad is inboard by required clearance;
- another component cannot overlap the moved RF keepout;
- later overlap/compaction/restore passes do not move or rotate the antenna;
- leaf outline adds no generic margin on the antenna side;
- explicit compatible rotation is preserved;
- explicit incompatible rotation rejects visibly;
- manual/scaffold placement remains unchanged;
- multiple antennas pack without overlap while all remain flush.

### Phase D — parent propagation and outline ownership

**Files**

- `kicraft/autoplacer/brain/subcircuit_composer.py`
- `kicraft/autoplacer/brain/parent_adapter.py`
- `kicraft/cli/compose_subcircuits.py`
- `kicraft/cli/_compose_state.py`
- parent attachment/outline tests

**Changes**

1. Extend `AttachmentConstraint` with explicit anchor kind and antenna direction/line information; do not infer antenna semantics from `ref` or `kind`.
2. Merge persisted antenna intents into derived child constraints. Explicit current-project placement wins on conflict.
3. Compute local anchor offset from the persisted support line in the same local frame as the leaf artifact.
4. Filter child rotation candidates so the transformed antenna direction matches its requested parent edge.
5. Carry antenna sides into `edge_zoned_outline_sides` and extremal-block repair.
6. Make final parent outline repair preserve the antenna anchor line exactly while still expanding other sides for pads/tracks/vias.
7. Persist final parent gap, actual direction, selected edge, and owning leaf.

**Tests**

- ESP32 leaf stays top-facing and flush after parent rotation/translation.
- Invalid 90/180/270 child rotations are removed.
- Parent repair does not add margin beyond antenna support line.
- A sibling leaf cannot become the RF-side extremity.
- Explicit side change propagates to the parent.
- USB-C, barrel, screw-terminal, mounting-hole, and ordinary edge-zone behavior is unchanged.

### Phase E — final antenna gates

**Files**

- new focused antenna geometry verifier or `connector_edge_gap.py` only if terminology is generalized cleanly
- promote verification in `kicraft/design/cli_app.py`
- gate tests

**Changes**

1. Measure final gap and direction from persisted intent plus promoted board geometry.
2. Add hard rejection categories listed in section 3.6.
3. Never populate the gate by scanning references or footprint names.
4. Surface measured results in parent debug output and the final build line.
5. Keep existing connector-stranding policy unchanged.

## 6. Verification matrix

### Focused tests

```bash
.venv/bin/pytest \
  tests/test_keepout_extract.py \
  tests/test_library_antenna_keepouts.py \
  tests/test_antenna_edge_placement.py \
  tests/test_placement_keepout.py \
  tests/test_edge_constraint_applied.py \
  tests/test_connector_edge_gap.py
```

Add the exact parent/gate test modules selected during implementation.

### Frozen replay corpus

Replay scratch copies only, through the real build tail:

1. KC-VA3UU7 / project `1/766`: ESP32-S3-MINI-1 with built-in `antenna_keepout`.
2. Project `1/707`: repeated ESP32 environmental-sensor case.
3. A project containing ESP32-S3-WROOM-1.
4. An ESP32-WROOM-32 board to exercise local-left geometry.
5. KC-69TGAP / project `1/660`: discrete chip antenna with existing explicit edge intent.
6. One negative control containing a non-antenna footprint rule area.
7. One project with `antenna_edge_pin_enabled=false`.
8. One project with explicit non-top edge and explicit compatible rotation.

For each replay record:

- owner ref and source identifier;
- source kind;
- local direction and anchor;
- selected edge and final rotation;
- leaf and parent anchor gaps;
- actual final direction;
- outside-pad count;
- `items_not_allowed` count;
- final rc and rejection categories.

Acceptance for each enabled/inferred antenna:

- leaf gap and parent gap within tolerance;
- direction outward;
- pads inboard;
- no ordinary component overlap with antenna keepout;
- no keepout DRC errors;
- no new shorts, unconnected nets, or connector/outline regressions.

Run at least two cold replays of KC-VA3UU7 because routing is best-effort stable. Compare invariants within each replay, not artifact bytes across independent runs.

## 7. Rollout and observability

- Default on after the focused suite and replay corpus pass.
- Keep `antenna_edge_pin_enabled` as an emergency/project-specific kill switch.
- Log exactly one line per intent, for example:

  ```text
  antenna-edge: U1 source=footprint_rule_area:antenna_keepout local=top target=top gap=0.00mm rotation=0
  ```

- Persist ambiguity and precedence decisions so triage can distinguish “not detected,” “explicit placement won,” “orientation conflict,” and “outline moved.”
- Update both superseded plans to link to this implementation and mark their antenna sections closed only after the replay corpus passes.

## 8. Non-goals

- No `ANT*`, `U1`, `ESP32`, or connector-kind heuristic.
- No changes to track/via keepout legality or `items_not_allowed` acceptance gates.
- No special GND carveout; existing rule areas remain authoritative.
- No automatic RF tuning, impedance matching, feed routing, or antenna model validation.
- No default overhang.
- No silent override of manual, scaffold, explicit edge, or explicit rotation intent.
- No parent-side rediscovery from transformed PCB geometry.

## 9. Completion checklist

- [x] Semantic antenna keepout extraction is generic and polygon-aware.
- [x] Built-in ESP32 antenna zones infer stable local direction and anchor.
- [x] Non-antenna rule areas are negative-tested.
- [x] Effective inferred constraints do not mutate user config.
- [x] Explicit/manual/form-factor precedence is tested.
- [x] Leaf placement orients outward and flushes the keepout support line.
- [x] All pads remain inboard and other components remain outside the RF keepout.
- [x] Leaf outline and size reduction preserve antenna-owned sides.
- [x] Intent survives artifact serialization and loading.
- [x] Parent rotation, extremity, anchor, and outline remain antenna-correct.
- [x] Final hard gap/direction gates read persisted intent only.
- [x] Existing connector and mounting-hole behavior is unchanged.
- [x] Focused tests pass.
- [ ] Frozen ESP32/discrete-antenna/negative-control corpus passes.
- [ ] Superseded antenna plans link to the shipped implementation and verification evidence.

## 10. Implementation analysis and evidence

Implemented in the shared placement pipeline rather than as an ESP32 or
connector special case:

- `hardware/keepout_extract.py` now separates semantic antenna extraction from
  the existing placement-exclusion rectangles. Named footprint polygons are
  retained in the unrotated footprint frame; the existing AABB representation
  remains only for collision pushing.
- `AntennaEdgeIntent` travels on `BoardState` and `SubCircuitLayout`, is stored
  in `solved_layout.json`, and is loaded without reinference. Old artifacts
  yield an empty intent list; malformed side values are rejected.
- The leaf solver builds a private effective `component_zones` map, rotates the
  local antenna vector with KiCad clockwise transforms, pins the support line,
  preserves rotation through restore passes, checks full pad bounding boxes,
  and packs the antenna polygon's parallel span.
- Leaf outline generation replaces generic margin only on antenna-owned sides.
  Parent attachment derivation consumes persisted intent, filters incompatible
  rigid-child rotations, uses the persisted support midpoint as its anchor, and
  propagates intent to the parent artifact.
- `brain/antenna_edge.py` verifies final gap, direction, owner presence, and pad
  containment. The build-tail fab gate reads current-run parent intent rather
  than scanning references or footprint names.

Critical corrections to the original sequence:

1. A second keepout collision model was not introduced. Exact polygons are
   needed for semantic direction, anchoring, and parallel packing; the existing
   conservative `KeepoutRect` path remains the established exclusion mechanism.
2. Inference belongs at board adaptation, where footprint-local zones and the
   already-classified `Component` envelope are simultaneously available.
   Encoding inferred semantics into arbitrary `component_zones` keys would lose
   the contract during artifact composition.
3. Parent propagation reuses the existing attachment anchor and
   anchor-authoritative outline machinery. A parallel antenna-only composer
   would duplicate transforms and create a second outline convention.
4. Final verification uses the persisted parent artifact. Rediscovery on the
   promoted PCB would violate precedence and could silently reinterpret manual
   placement.

Focused evidence:

```text
194 passed, 3 skipped
```

The focused run covered semantic extraction, four source rotations, all four
target edges, explicit-rotation conflict, multi-antenna polygon packing,
serialization compatibility, final geometry verification, placement keepouts,
edge/connector regressions, leaf placement, artifact loading, parent
composition, parent adaptation, leaf intent filtering, and non-zero parent
origin anchoring.

Cold replay evidence for KC-VA3UU7 / project `1/766`, quality `good`, seed `0`:

```text
Leafs: 2/2 accepted
Parent: routed
verify: shorts=0 unconnected=0 courtyard=0 keepout=0 traces=362
REPLAY COMPLETE
provenance: run_id=f896339e1ebf source_kind=routed fresh=True
```

The first cold replay exposed a dropped `antenna_edge_intents` field when a
full `BoardState` was extracted into a leaf; the ESP32 was therefore placed at
`-90°` and failed routed geometry. After preserving owned intents in leaf
extraction, the next replay exposed a parent-only transform bug: the antenna
anchor's child origin was added twice, stranding it `1.22 mm` inboard. The
antenna path now consumes the already-world-frame transformed anchor while the
legacy connector body-anchor convention remains unchanged. The final cold
replay passed the complete build tail and fresh-artifact check.

The full frozen corpus in section 6 remains unchecked: the second KC-VA3UU7
cold replay, project `1/707`, discrete-antenna cases, and negative controls have
not been run.
