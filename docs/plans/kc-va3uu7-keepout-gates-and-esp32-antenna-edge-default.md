# KC-VA3UU7 keepout legality, acceptance gates, replay robustness, and ESP32 antenna-edge default

**Status:** implemented and verified by focused tests and three cold production-workspace replays
**Origin:** KC-VA3UU7 (`/home/kicraft/.kicraft/projects/1/766`)
**Primary owners:** `kicraft/autoplacer/brain/breakout_stubs.py`, `kicraft/autoplacer/routing_board.py`, `kicraft/autoplacer/hardware/keepout_extract.py`, `kicraft/autoplacer/brain/placement_solver.py`, `kicraft/autoplacer/brain/subcircuit_composer.py`, `kicraft/cli/compose_subcircuits.py`, `kicraft/design/cli_app.py`

## 1. Goal

Land one coherent fix set that:

1. Never stamps a breakout track or end via into a board-level or footprint-local rule area that forbids that item.
2. Treats every nonzero KiCad `items_not_allowed` count as illegal routed geometry at leaf and parent acceptance gates.
3. Rejects deterministic pre-route keepout damage before spending KRT time or auto-pinning a bad leaf.
4. Makes project-only `replay --project ...` return its real rc7 verdict instead of crashing while persisting diagnostics with `state_path=None`.
5. Adds a universal default for ESP32 modules with antenna keepout geometry: orient the antenna outward and place that antenna-facing side directly on an exterior leaf/board edge. No ordinary part or ground pour may occupy the RF side.
6. Preserves explicit user placement as the highest-precedence instruction and keeps every ESP32 pad fabricable/inboard.

This plan supersedes the ESP32/module portion of `docs/plans/antenna-edge-flush-plan.md`. Do not implement a second `ANT*`/footprint-name heuristic beside the keepout-driven mechanism below. The older plan remains relevant for discrete chip antennas that have no module-family keepout metadata.

## 2. Evidence and current failure

KC-VA3UU7 is an ESP32-S3-MINI-1 environmental sensor node. Its authoritative build verdict is rc7:

```text
shorts=0 unconnected=0 courtyard=0 keepout=36
reasons=['keepout_intrusion']
```

All 36 errors are GND tracks/vias in U1's footprint-local `antenna_keepout`. The ESP32 leaf already contains 30 errors in `leaf_placed.kicad_pcb`, before KRT. Routing increases the count to 36; parent placement and routing preserve it.

The defect reproduced in three independent cold current-code replays with `--quality good --seed 0`. Every replay reached the same promote verdict: 36 keepout intrusions, zero shorts, zero unconnected nets, zero courtyard errors. Every parent `debug.json` nevertheless stored:

```json
{
  "accepted": true,
  "drc": {"items_not_allowed": 36}
}
```

Breadth from `triage scan --json`: four affected runs across three distinct briefs and four deployed SHAs:

- `1/707` — `73ea1b1fe`
- `1/717` — `c97d49e25`
- `1/746` — `39ca3a9f2`
- `1/766` — `ddbc66771`

Exact prior art exists as open item B31 in `docs/plans/codebase-review-2026-07-06.md`. The June KC-S8PC37 fix protects only the post-route GND finisher; its regression test does not exercise pre-route `gnd_escape_specs -> add_breakout_stubs`.

## 3. Current mechanisms to reuse

Do not create parallel concepts.

- `kicraft/autoplacer/config.py:DEFAULT_CONFIG["antenna_keepouts"]` already defines local-frame RF keep-clear rectangles for:
  - ESP32-S3-WROOM-1: antenna at local `-Y`
  - ESP32-S3-MINI-1: antenna at local `-Y`
  - ESP32-WROOM-32: antenna at local `-X`
- `kicraft/autoplacer/hardware/keepout_extract.py` already:
  - matches footprint family names case-insensitively;
  - preserves footprint-local rule areas;
  - injects configured RF near-field rectangles;
  - transforms local rectangles with the canonical KiCad rotation convention.
- `kicraft/autoplacer/brain/gnd_pour.py:_collect_keepout_zones` already finds board- and footprint-local track/via rule areas.
- `PlacementSolver._pin_edge_components` already groups explicit edge parts, locks them, orients them, and restores their positions after later placement passes.
- `AttachmentConstraint`, `attachment_constraints_to_zones`, `edge_zoned_outline_sides`, and `_repair_parent_outline` already propagate a constrained child part to the final parent outline.
- `connector_edge_gap.py` and the promote facings/gap gates already measure whether an explicitly edge-zoned part is actually exterior-facing.
- `run_kicad_cli_drc` already parses `items_not_allowed`; only acceptance plumbing is missing.

## 4. Required invariants

The implementation is incomplete unless all invariants hold.

### Copper legality

- A candidate track is blocked only by a rule area with `GetDoNotAllowTracks()`.
- A candidate via is blocked only by a rule area with `GetDoNotAllowVias()`.
- Both board-level and footprint-local zones participate.
- Use the placed zone geometry in board coordinates. Do not assume a footprint rule area is axis-aligned or unrotated.
- Account for physical width: track half-width and via radius may not overlap the protected polygon even when their centerlines are outside it.
- A multi-segment `BreakoutSpec` is atomic. If any segment or required end via is illegal, stamp none of it.
- Same-net status does not waive a rule area.

### Acceptance

- `items_not_allowed > 0` always means `obviously_illegal_routed_geometry=true`, `accepted=false`, and rejection reason `illegal_routed_geometry`.
- No footprint-internal waiver applies. A footprint's rule area is an intentional constraint, not an intrinsic pad-spacing false positive.
- Leaf and parent gates use the same rule.
- The promote verifier remains the final backstop, not the first detector.

### ESP32 antenna placement

- The trigger is antenna keepout metadata/geometry on a matched module, not reference prefix and not `kind == connector`.
- Explicit `component_zones`, explicit rotation, manual layout, and form-factor scaffold coordinates win over the default.
- With no explicit override, default edge is `top`. This intentionally separates the antenna default from the existing default edge-connector side (`bottom`) and is deterministic across replays.
- Rotate the module so the local antenna direction points outward through the selected board edge.
- Place the module's antenna-facing outer line flush with the board edge by default (`0.0 mm` inset/overhang). A project may request a small overhang through a dedicated antenna setting; do not reuse connector barrel overhang.
- Clamp the leaf and final parent outlines to the antenna anchor line on that side. Generic bbox/silkscreen margin must not add carrier-board material beyond the RF-facing module edge.
- All copper pads remain inside the outline by the normal pad-edge clearance. The antenna body may touch or project past the edge only when an explicit antenna overhang is configured.
- Ground/copper keepout remains authoritative even after edge placement. Edge pinning is an RF-layout default, not a substitute for keepout enforcement.
- Other components are kept out of the existing injected/preserved RF rectangle; the owner module remains exempt from its own keepout.
- Multiple inferred modules may share the top edge and use the existing edge-group packing. If a design needs different sides, explicit `component_zones` overrides each module.

## 5. Implementation sequence

Implement in this order. Each phase establishes a contract consumed by the next.

### Phase A — one reusable track/via rule-area geometry layer

**Files**

- `kicraft/autoplacer/hardware/keepout_extract.py`
- `kicraft/autoplacer/brain/gnd_pour.py`
- `tests/test_keepout_extract.py`

**Change**

1. Promote the board/footprint track-via zone collection currently private to `gnd_pour.py` into `hardware.keepout_extract` (or a narrowly named sibling in `hardware` if the file becomes unclear). Suggested API:

   ```python
   collect_track_via_rule_areas(board) -> list[RuleArea]
   ```

   `RuleArea` must retain the pcbnew zone plus independent `blocks_tracks` and `blocks_vias` flags. Do not flatten both flags into one boolean.

2. Add geometry predicates used by every deterministic copper stamper:

   ```python
   track_intersects_rule_area(a, b, half_width_mm, area) -> bool
   via_intersects_rule_area(center, radius_mm, area) -> bool
   ```

3. Predicates must handle containment and boundary crossing. At minimum:
   - center/endpoints inside polygon;
   - segment crossing any polygon edge;
   - track/via radius within the relevant distance of a polygon edge;
   - rotated footprint-local zones;
   - non-rectangular rule areas.

   Use pcbnew polygon primitives when they expose reliable collision/distance operations. If wrapper compatibility forces Python geometry, enumerate polygon outlines and use the existing canonical segment-distance helpers; keep this implementation in one module.

4. Migrate `gnd_pour.py`'s post-route `_in_keepout` behavior to the shared collector/predicates. Preserve its existing behavior and KC-S8PC37 regression test. Delete the old private collector after migration.

**Tests**

Extend `tests/test_keepout_extract.py` with:

- board-level track-only area;
- footprint-local via-only area;
- rotated footprint-local area;
- centerline outside but copper width/radius overlapping the boundary;
- legal geometry immediately outside the physical clearance;
- independent track/via flags.

### Phase B — enforce rule areas in every breakout stamp

**Files**

- `kicraft/autoplacer/brain/breakout_stubs.py`
- `tests/test_power_pour.py`
- optionally a focused breakout-stub test module if existing tests there are the stronger convention

**Change**

1. In `add_breakout_stubs`, collect rule areas once after loading the board.
2. Before accepting a waypoint path or radial candidate:
   - reject the candidate when any segment, including its physical width, intersects a track-blocking area;
   - reject a required end via when its barrel intersects a via-blocking area.
3. Run this check before adding any `PCB_TRACK`/`PCB_VIA`. Preserve the current whole-spec atomicity.
4. Record stable skip reasons, for example:

   ```text
   U1.7:track_keepout
   U1.7:via_keepout
   ```

   Do not collapse this into `no_safe_radial_escape`; operators need to distinguish a deliberate RF keepout from pad/copper congestion.
5. Keep `gnd_escape_specs` as the producer of requested bonds. The universal materialization boundary owns legality so signal ties, shield ties, array ties, and future callers receive the same protection.

**Regression test**

Reuse `tests/test_power_pour.py::_module_keepout_board`:

1. Generate the pre-route specs with `gnd_escape_specs`.
2. Materialize them through `add_breakout_stubs`.
3. Assert zero GND tracks/vias overlap `antenna_keepout`.
4. Add a small legal GND pad far from the keepout and assert its escape still stamps.
5. Assert skip diagnostics identify track/via keepout rather than a generic failure.

This is the missing pre-route twin of `test_gnd_finisher_keeps_copper_out_of_footprint_keepout`.

### Phase C — make `items_not_allowed` a hard acceptance category

**Files**

- `kicraft/autoplacer/routing_board.py`
- `kicraft/autoplacer/brain/leaf_routing.py`
- `tests/test_routing_board.py`
- leaf-routing tests that already cover pre-route rejection/persistence

**Change**

1. Immediately after `run_kicad_cli_drc` in `validate_routed_board`:

   ```python
   if drc.get("items_not_allowed", 0) > 0:
       validation["obviously_illegal_routed_geometry"] = True
   ```

   Let the existing reason plumbing append exactly `illegal_routed_geometry`.

2. Keep this outside the clearance and connector-shield waiver logic.

3. Move or split leaf pre-route validation so it executes immediately after all deterministic breakout/ring/shield stamps and before `route_with_kicad_routing_tools`. The current source order validates `pre_route_board` only after KRT has already run.

4. If pre-route validation finds `items_not_allowed > 0`:
   - do not invoke KRT;
   - persist the pre-route board and bounded DRC evidence;
   - return a rejected leaf result with a specific stage such as `leaf_pre_route_drc` and reason `illegal_routed_geometry`;
   - never auto-pin it.

5. Other pre-route violations retain current semantics unless already hard blockers. Do not accidentally turn warning-only silk/via-diameter/footprint-baseline findings into pre-route aborts.

**Tests**

- Mock DRC with `items_not_allowed=1`; assert `accepted is False`, `obviously_illegal_routed_geometry is True`, and reasons equal/include `illegal_routed_geometry`.
- Assert zero retains existing acceptance.
- Leaf flow: stamped pre-route keepout violation means KRT mock is never called and the leaf rejection stage is persisted.
- Parent validation receives the same hard verdict.

### Phase D — infer and persist ESP32 antenna-edge intent

**Files**

- `kicraft/autoplacer/config.py`
- `kicraft/autoplacer/brain/types.py`
- `kicraft/autoplacer/hardware/keepout_extract.py`
- `kicraft/autoplacer/hardware/adapter.py`
- `kicraft/cli/solve_subcircuits.py`
- artifact metadata persistence/loading code in `leaf_routing.py` / `subcircuit_instances.py`
- `tests/test_keepout_extract.py`
- `tests/test_library_antenna_keepouts.py`

**Configuration**

Add boring, explicit defaults:

```python
"antenna_edge_pin_enabled": True,
"antenna_default_edge": "top",
"antenna_edge_inset_mm": 0.0,
```

Do not reuse `connector_edge_inset_mm` or `connector_edge_overhang_mm`; connector mouths and RF radiators have different physical meanings.

Use the existing `antenna_keepouts` family specs as the module allowlist. The current default entries are ESP32 modules, so no second footprint-pattern table is needed. Project-specific antenna family specs automatically opt into the same mechanism unless `antenna_edge_pin_enabled` is false.

**Intent model**

Add a small serializable value object, e.g. `AntennaEdgeIntent`, containing:

- `owner_ref`
- matched footprint/family identifier
- local outward cardinal direction (`left|right|top|bottom` in the footprint's unrotated local frame)
- local antenna anchor line or midpoint
- source (`inject` or `preserve`)
- selected target board edge
- explicit-vs-inferred marker

Derive local outward direction from the configured local keepout rectangle:

- dominant negative X center -> local left;
- dominant positive X center -> local right;
- dominant negative Y center -> local top;
- dominant positive Y center -> local bottom.

For a tie or a keepout centered on the footprint origin, do not guess. Skip inference with a bounded diagnostic. The three default ESP32 specs are unambiguous.

Use the RF-facing outer boundary of the module/keepout as the default flush anchor, not the keepout's inward boundary. This keeps the module body/pads inboard while placing the antenna-facing side directly at the carrier edge. The rule area still prevents copper beneath/behind the antenna. A future explicit negative `antenna_edge_inset_mm` may overhang it; default remains zero.

**Data flow**

1. `adapter.load` extracts both placement keepout rectangles and antenna intents from the same board/config pass.
2. `solve_subcircuits` builds an effective leaf-zone map:
   - start with explicit project `component_zones`;
   - add `{owner_ref: {"edge": antenna_default_edge}}` only for inferred owners absent from explicit zones;
   - do not mutate the shared/user config in place.
3. Persist the inferred intent and chosen edge in leaf metadata/debug output. The parent composer must not rediscover this from a possibly transformed PCB or from a ref-name heuristic.
4. `load_solved_artifact` exposes the persisted intent to composition.

**Tests**

- ESP32-S3-MINI and ESP32-S3-WROOM infer local top/`-Y`.
- ESP32-WROOM-32 infers local left/`-X`.
- 0/90/180/270 initial footprint rotations do not change the local semantic direction.
- A non-ESP footprint with an unrelated rule area does not infer an antenna edge unless it is explicitly present in `antenna_keepouts`.
- Explicit `component_zones[U1]` prevents default edge injection.
- Kill switch disables inference.
- Ambiguous centered rect produces no intent and a bounded diagnostic.

### Phase E — orient and flush the antenna side at leaf placement

**Files**

- `kicraft/autoplacer/brain/placement_solver.py`
- `kicraft/autoplacer/brain/leaf_routing.py`
- `kicraft/autoplacer/brain/leaf_size_reduction.py` if it independently rewrites Edge.Cuts
- `tests/test_usb_edge_connector_placement.py` or a new focused `tests/test_antenna_edge_placement.py`
- `tests/test_placement_keepout.py`

**Change**

1. Extend `_pin_edge_components` through an antenna-specific branch driven by `AntennaEdgeIntent`, not `comp.kind` or `ANT*`.
2. For an inferred/explicit target edge, select the 0/90/180/270 rotation that maps the intent's local outward vector to that edge under `geometry.transform_point`'s KiCad convention.
3. If the user supplied an explicit rotation, keep it. If that rotation points the antenna inward, surface a placement diagnostic/gate failure; do not silently override explicit intent.
4. Compute the pinned coordinate from the antenna anchor line plus `antenna_edge_inset_mm`, rather than `_connector_edge_x/_connector_edge_y`'s body-center/mouth rules.
5. Reuse grouped edge packing for motion parallel to the edge and `_pinned_targets` for restoration.
6. Run `_shift_pads_inside` and then assert all module pads satisfy pad-edge clearance. If the RF flush and pad containment constraints are incompatible, reject the placement round with explicit `antenna_edge_pad_conflict`; do not pull U1 inboard and pretend it remains RF-flush.
7. Update `_outline_around_geometry` and any leaf-size-reduction outline rewrite so the selected antenna side is anchor-authoritative: set that Edge.Cuts side to the antenna line, not `bbox ± leaf_edge_margin_mm`. Other three sides retain normal margins.
8. Ensure the injected/preserved `KeepoutRect` moves with U1 and still pushes non-owner components inward. The owner remains exempt.

**Tests**

Parameterize all four target edges and at least two module-local directions:

- antenna vector points outward after placement;
- antenna anchor gap is `0.0 ± tolerance`;
- all pads remain inside with clearance;
- no other component overlaps the moved RF rectangle;
- later overlap/compaction/restore passes do not move U1 off the edge;
- leaf Edge.Cuts stays on the antenna anchor instead of adding generic margin;
- explicit edge overrides default top;
- explicit compatible rotation is preserved;
- explicit incompatible rotation is rejected visibly.

### Phase F — propagate antenna edge intent through parent composition

**Files**

- `kicraft/autoplacer/brain/subcircuit_composer.py`
- `kicraft/autoplacer/brain/parent_adapter.py`
- `kicraft/cli/compose_subcircuits.py`
- `kicraft/cli/_compose_state.py` only if an additional persisted field is genuinely needed
- parent attachment/outline tests

**Change**

1. Merge persisted inferred antenna constraints into the input of `derive_attachment_constraints`. Explicit project `component_zones` win on ref conflicts.
2. Extend `AttachmentConstraint` with the minimum semantic information needed to distinguish an antenna anchor from connector body/pad anchors. Avoid encoding this in ref prefixes.
3. `_compute_local_anchor_offset` must use the persisted antenna anchor line.
4. Filter child block rotation candidates so the transformed antenna direction still points through the requested parent edge. The leaf may not rotate 90/180 degrees and turn a top-facing antenna inward while retaining a nominal `edge:top` constraint.
5. Feed the resulting block zone through existing `attachment_constraints_to_zones` and `_ensure_edge_blocks_extremal`.
6. Include the antenna side in `edge_zoned_outline_sides`. `_repair_parent_outline` must keep that side flush rather than growing it by generic board margin.
7. Geometry validation must use the existing edge-constrained pad-aware policy: pads/tracks/vias stay inboard; an explicitly configured antenna overhang may place body-only geometry outboard.
8. The final GND fill must stop at the parent Edge.Cuts and continue honoring the antenna rule area. Do not add a special GND-zone carveout beside the existing rule area.
9. Persist the final antenna anchor gap/direction in parent debug metadata so `triage` can distinguish “edge default absent,” “orientation wrong,” and “outline grew past anchor.”

**Tests**

- A leaf containing only U1 plus passives composes with U1 antenna side on final parent top edge.
- Allowed child rotations exclude inward-facing antenna rotations.
- Parent outline repair does not add margin beyond the antenna anchor.
- Pads remain inside; body/keepout ownership is preserved.
- A sibling leaf cannot become the top extremity and bury the antenna behind it; existing extremal-block repair keeps the antenna leaf exterior.
- Explicit user edge changes the final parent side.
- Existing USB/screw-terminal edge behavior remains unchanged.

### Phase G — deterministic edge/facing backstop

**Files**

- `kicraft/autoplacer/brain/connector_edge_gap.py` or a narrowly named antenna companion if connector terminology becomes misleading
- promote verification in `kicraft/design/cli_app.py`
- corresponding gate tests

**Change**

Add a hard antenna-edge verdict for every persisted inferred/explicit antenna constraint:

- anchor gap exceeds configured tolerance -> `antenna_stranded:<ref>@<gap>(<edge>)`;
- outward vector does not face selected edge -> `antenna_misoriented:<ref>(...)`;
- pads outside outline -> existing geometry blocker, not a new waiver.

These are fab/RF blockers for this default, not warning-only `connector_stranded` behavior. The pipeline claimed a deliberate RF placement and must not export a board that violated it.

Do not infer the gate population from `ANT*`. Read the persisted antenna constraint set.

### Phase H — fix project-only replay diagnostics persistence

**Files**

- `kicraft/design/cli_app.py`
- `tests/test_replay_command.py`

**Change**

1. Correct the type contract:

   ```python
   def _persist_artifacts(state, state_path: Path | None, artifacts) -> None:
       state.artifacts = artifacts
       if state_path is None:
           return
       ...
   ```

2. Propagate `Path | None` annotations through `_persist_pcb_diagnostics` and other relevant callers. Do not invent a fake state path in project-only mode.
3. Ensure the rc7 branch completes diagnostics and returns `7`; it must not throw `AttributeError` after printing the decisive verdict.
4. Keep `artifacts --project` filesystem/provenance-based. It already resolves the promoted/routed boards without state.json.

**Tests**

- `_persist_artifacts` updates the in-memory state and does not write/crash when path is `None`.
- Project-only replay with a mocked rc7 verify path returns 7 and does not raise.
- State-backed build still atomically persists artifacts.
- The root schematic no-mutation replay invariant remains active on rc7.

## 6. Verification matrix

Run focused tests first:

```bash
.venv/bin/pytest \
  tests/test_keepout_extract.py \
  tests/test_power_pour.py \
  tests/test_routing_board.py \
  tests/test_placement_keepout.py \
  tests/test_antenna_edge_placement.py \
  tests/test_edge_constraint_applied.py \
  tests/test_replay_command.py
```

Use the actual test filename chosen by the implementation if antenna tests are added to an existing module.

### KC-VA3UU7 replay

Never replay in place. For each of three samples:

```bash
set -a && source .env && set +a
WORK=$(mktemp -d)
cp -a /home/kicraft/.kicraft/projects/1/766/generated/ESP32S3_ENV_SENSOR_NODE_1_0_0_0_ "$WORK/replay"
rm -rf "$WORK/replay/.experiments"
.venv/bin/python -m kicraft.design.cli_app replay \
  --project "$WORK/replay" --quality good --seed 0
.venv/bin/python -m kicraft.design.cli_app artifacts --project "$WORK/replay"
```

Expected in at least two of three completed replays, preferably 3/3:

- replay exits normally, not by traceback;
- `items_not_allowed: 36 -> 0` at leaf placed, leaf routed, parent placed, and parent routed stages;
- promote `keepout: 36 -> 0`;
- shorts remain 0;
- unconnected remains 0;
- courtyard remains 0;
- rc7 -> rc0 unless a newly exposed unrelated blocker is reported honestly;
- U1 antenna side is flush with the selected final board edge and faces outward;
- all U1 pads are inboard;
- no GND copper exists in the footprint-local antenna rule area;
- the far legal GND escape still stamps.

If rc remains nonzero for another category, report the exact new category; do not weaken its gate to force rc0.

#### Observed result — 2026-09-01

The focused contract suite completed with `188 passed, 2 skipped`:

```text
tests/test_array_placement.py
tests/test_usb_edge_connector_placement.py
tests/test_breakout_stubs.py
tests/test_keepout_extract.py
tests/test_power_pour.py
tests/test_routing_board.py
tests/test_placement_keepout.py
tests/test_antenna_edge_placement.py
tests/test_edge_constraint_applied.py
tests/test_replay_command.py
```

Three fresh scratch copies of production project `1/766` were replayed with
`--quality good --seed 0` after deleting only the copied `.experiments`
directory. All three exited `rc0`; artifact resolution reported fresh routed
parents with run IDs `b0c25acebc32`, `95d9af380886`, and `40e641148f6e`.
Every replay promoted the same honest verdict:

```text
shorts=0 unconnected=0 courtyard=0 keepout=0
items_not_allowed=0
traces=362 vias=77
```

The selected U1 leaf also reported `items_not_allowed=0` both before and after
routing. Its persisted antenna intent is inferred from the footprint-local
`antenna_keepout`, with local direction `top`, target edge `top`, and inset
`0.0 mm`. U1 remained at rotation `0°`; its physical top edge is flush with the
leaf outline, while its nearest pad is `6.875 mm` inboard. Parent DRC remained
clean, so no GND copper entered the antenna rule area. No unrelated rejection
category was exposed.


### RF-placement corpus

Replay or run placement-only checks on:

- KC-VA3UU7 / run `1/766` — ESP32-S3-MINI-1
- run `1/707` — repeated ESP32 environmental-sensor brief
- self-eval `run_12_esp32-s3-sensor` — ESP32-S3 family
- KC-69TGAP / run `1/660` — existing discrete-antenna edge case; must not regress and should continue through its existing explicit constraint path
- one ESP32-WROOM-32 board to exercise local `-X` antenna geometry

For each board, record:

- module ref and matched family;
- inferred local antenna direction;
- selected edge;
- final rotation;
- leaf and parent antenna gap;
- outside-pad count;
- keepout DRC count;
- final rc.

### Regression checks

- USB-C, barrel jack, BNC, screw-terminal, and mounting-hole edge tests remain unchanged.
- A project with `antenna_edge_pin_enabled=false` preserves previous placement behavior.
- A project with explicit `component_zones[U1]` is not silently rewritten.
- A board with a non-antenna footprint rule area is not edge-pinned.
- Warning-only silk clips/via-dangling findings do not become blockers.

## 7. Rollout and diagnostics

1. Keep the universal antenna behavior default-on; add the kill switch only for emergency A/B and atypical carrier constraints.
2. Surface inferred antenna constraints in leaf and parent debug JSON. A hidden default is hard to triage.
3. Add one concise build line per inferred module, for example:

   ```text
   antenna-edge: U1 family=ESP32-S3-MINI-1 local=top target=top gap=0.00mm source=default
   ```

4. Do not log secrets or full configuration.
5. Update the open B31 checklist item after implementation and verification. Link this plan and record the three-replay result.
6. Update `docs/plans/antenna-edge-flush-plan.md` status/link so future agents do not implement competing antenna heuristics.

## 8. Non-goals

- Do not edit KC-VA3UU7's generated board by hand.
- Do not relax or delete the ESP32 footprint's `antenna_keepout`.
- Do not teach KRT to repair deterministic locked copper; prevent it before routing.
- Do not waive footprint-local `items_not_allowed`.
- Do not use `ANT*`, `U1`, or generic `ESP32` substring checks as the primary placement contract.
- Do not add a second ground-pour carveout; Edge.Cuts plus the existing rule area own copper exclusion.
- Do not change connector-stranding warning policy while implementing the separate antenna hard gate.
- Do not combine this with general board compaction or routing-budget tuning.

## 9. Completion checklist

- [x] Shared polygon-aware track/via rule-area predicates landed.
- [x] `gnd_pour` migrated to the shared predicates; duplicate collector removed.
- [x] `add_breakout_stubs` rejects illegal tracks and end vias atomically.
- [x] Pre-route GND regression proves legal escapes survive and antenna intrusions do not.
- [x] `validate_routed_board` rejects every nonzero `items_not_allowed`.
- [x] Leaf pre-route hard violations stop before KRT and cannot auto-pin.
- [x] ESP32 antenna intent inferred from existing keepout family metadata.
- [x] Explicit placement/manual/form-factor intent has precedence.
- [x] Module antenna direction rotates outward and flushes to leaf Edge.Cuts.
- [x] Parent composition preserves direction, anchor, extremity, and final outline flush.
- [x] All module pads remain inboard; no component occupies RF keepout.
- [x] Antenna stranded/misoriented hard gates landed.
- [ ] Project-only rc7 replay returns 7 without traceback.
- [x] Focused tests pass.
- [x] Three cold KC-VA3UU7 replays completed and recorded.
- [ ] RF corpus shows no connector/outline regressions.
- [ ] B31 and the older antenna-edge plan updated to point at the implemented fix.
