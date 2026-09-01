# KC-AHW6GA: root-cause plan for edge-connector array placement

**Status:** implemented and verified by frozen-workspace replay

## Objective

Prevent an array of side-entry connectors assigned to one board edge from being
oriented as a generic signal-chain array. The leaf placer must produce a
one-dimensional bank parallel to the assigned edge, with every detectable
connector mouth facing that edge. Parent composition should then be able to pin
the rigid leaf without rotating the bank into a second, inaccessible row.

This plan fixes placement. It does **not** change the later policy that preserves
or exports a `connector_stranded` board; that is a separate release-gate issue.

## Critical review outcome

The root-cause analysis and specialization boundary were correct. Implementation
made two geometry requirements stricter than the original steps:

- Bank members align by their outward **physical bbox extremity**, not merely by
  footprint origin or body-center coordinate. This covers unequal bodies and
  footprints whose origin is offset from the courtyard center.
- Tangent packing uses each adjacent member's rotated `physical_bbox()`
  (courtyard plus pad copper). Explicit pitch is a center-distance lower bound;
  physical non-overlap plus the configured array gap remains mandatory. A single
  maximum courtyard width would be insufficient for heterogeneous or
  origin-offset connectors.

The impossible-shape diagnostic is raised before any member is mutated, so a
2-D same-edge connector specification cannot leave a partially rotated/locked
candidate behind. The existing exception path preserves the stable diagnostic;
no generic-array fallback was added.

## Verification result

On 2026-08-31, the production workspace for project `1/772`
(`HIGH_SIDE_LOAD_SWITCH_10A_THERM_`, board `KC-AHW6GA`) was copied to a cold
scratch directory and replayed through the real `good`-quality build tail with
seed `0`, pinned KiCadRoutingTools, and no synthesis.

- The solved INPUT OUTPUT leaf placed J1/J2 at X `5.57`/`16.26` mm with common
  Y `4.57` mm. Both rotations are `90°`; their local `180°` mouths therefore
  face bottom before composition.
- The selected parent was routed and accepted with no `connector_stranded` or
  `connector_misoriented` finding and an empty rejection-reason list.
- Final verification reported shorts `0`, unconnected `0`, courtyard overlaps
  `0`, keepout intrusions `0`, 97 traces, and 8 vias.
- Fab export completed. Inspection of `fab/board_3d.png` confirms both terminal
  wire-entry faces form one row at the same board edge and are independently
  accessible from outside the board.

## Reproduced failure and root cause

Board `KC-AHW6GA` generated this placement intent:

```json
{
  "component_zones": {
    "J1": {"edge": "bottom"},
    "J2": {"edge": "bottom"}
  },
  "arrays": [{
    "refs": ["J1", "J2"],
    "pattern": "grid",
    "rows": 1,
    "cols": 2,
    "pitch_mm": 7.5
  }]
}
```

The INPUT OUTPUT leaf initially placed the terminals correctly as a horizontal
row:

- J1 body center: `(4.47, 5.57)`
- J2 body center: `(12.96, 5.57)`

Both WJ126V terminals have `opening_direction=180°`. The generic
`_orient_array_grid` path left both at rotation `0°`, so both mouths faced left,
parallel to the row instead of toward the requested bottom edge.

The parent composer then evaluated the rigid leaf:

| Leaf rotation | Mouths face bottom | J1/J2 both bottom extremities |
|---|---:|---:|
| 0° | no | yes |
| 90° | yes | no |
| 180° | no | yes |
| 270° | no | no |

No transform satisfied both invariants. `_filter_rotations_for_connector_opening`
therefore took its documented mouth-only fallback and retained only `90°`.
Rotating the entire leaf by 90° turned the horizontal bank into a vertical bank.
J1 reached the bottom edge and J2 landed 8.71 mm behind it.

The parent detector behaved correctly: all three parent rounds reported
`connector_stranded:J2@-8.71mm(bottom)`. The root defect was already frozen into
the leaf before parent placement.

### Root cause in code

`kicraft/autoplacer/brain/array_placement.py::_orient_array_grid` assumes an
array is a routed data chain such as a WS2812 matrix. It derives member rotation
from a non-power net shared with the next member. The function is applied to all
grid arrays, including a synthesized grouping of screw terminals. For a bank of
edge-access connectors, shortest chain routing is subordinate to the mechanical
access invariant.

The existing parent mechanisms are safety nets, not the correct repair site:

- `subcircuit_composer._filter_rotations_for_connector_opening` can rotate only
  the whole rigid leaf; it cannot rotate J1/J2 independently.
- `parent_adapter.attachment_constraints_to_zones` represents a child as one
  synthetic block with one primary edge anchor; it cannot turn a two-row block
  back into a one-row connector bank.
- `connector_edge_gap` detects the result only after parent stamping.

## Required invariants

For an array whose present members are all explicitly assigned to the same edge
and are edge-access connectors:

1. Members form exactly one bank along the edge tangent:
   - `top`/`bottom`: common Y extremity, members distributed along X.
   - `left`/`right`: common X extremity, members distributed along Y.
2. Every member with a known `opening_direction` satisfies:

   ```python
   opening_board_angle(member.opening_direction, member.rotation)
       == edge_outward_angle(member.layer, assigned_edge)
   ```

3. Mechanical orientation wins over `_orient_array_grid` data-chain
   orientation. The generic LED/ring behavior remains byte-for-byte unchanged
   for arrays that are not same-edge connector banks.
4. A two-dimensional array cannot satisfy one shared edge-access constraint.
   Reject such an impossible specification during leaf placement with a precise
   error; do not silently create an inboard row or fall back to generic array
   orientation.
5. Rotation must carry pads and body-center geometry through the existing
   `rotate_component_in_place` path. Do not assign `Component.rotation`
   directly.
6. Explicit pitch remains a lower bound. Actual tangent spacing must also clear
   the rotated physical extents plus `array_gap_mm`/placement clearance.

## Implementation plan

### 1. Classify same-edge connector banks before generic grid orientation

**File:** `kicraft/autoplacer/brain/array_placement.py`

Add a small private classifier used by `place_array_leaves` after resolving the
present `refs` and before grid coordinates/orientation are finalized.

The classifier should return an edge only when:

- every member has a `component_zones[ref]` dictionary with the same valid
  `edge` value;
- every member is a connector by existing project conventions (`kind ==
  "connector"` or connector reference/value classification already used by the
  placer); and
- the array is a grid, not a ring.

Do not infer an edge for unzoned arrays. Do not add a second connector taxonomy.
Reuse the established component classification available to the leaf solver.

If the common-edge array has both `rows > 1` and `cols > 1`, raise a diagnostic
that names the refs, shape, and edge. All members cannot physically occupy the
same access edge in that shape.

### 2. Add a dedicated edge-bank placement/orientation path

**File:** `kicraft/autoplacer/brain/array_placement.py`

For the classified bank:

1. Rotate each known-mouth member to the absolute rotation computed from the
   shared helpers in `brain/types.py`:

   ```python
   target_rotation = (
       member.opening_direction
       - edge_outward_angle(member.layer, edge)
   ) % 360.0
   ```

   Apply only the delta through `rotate_component_in_place`.

2. Place members in `refs` order on the edge tangent:
   - horizontal for top/bottom;
   - vertical for left/right.

3. Derive tangent pitch after target rotation from each member's physical AABB.
   Honor the explicit `pitch_mm` when it is larger, but floor it to the required
   non-overlap distance exactly as the current grid path does.

4. Mark members `locked=True` and `array_member=True`, and preserve the existing
   grid metadata/centers needed by routing and artifact serialization.

5. Skip `_orient_array_grid` for this bank. It must not overwrite the
   mechanically required rotation.

Keep the ordinary grid and ring paths unchanged. Avoid a general array-engine
rewrite.

### 3. Centralize the opening-to-edge rotation formula

**Files:**

- `kicraft/autoplacer/brain/types.py`
- `kicraft/autoplacer/brain/placement_solver.py`
- `kicraft/autoplacer/brain/array_placement.py`

The formula currently lives inside
`PlacementSolver._best_rotation_for_edge`. Add one shared, boring helper beside
`edge_outward_angle`, `opening_board_angle`, and `angles_close`, for example
`opening_rotation_for_edge(opening_direction, layer, edge)`.

Migrate `_best_rotation_for_edge`'s known-mouth branch to that helper and use the
same helper in the new edge-bank path. This prevents the single-board placer and
array placer from drifting on KiCad's clockwise rotation convention. Preserve
`_best_rotation_for_edge`'s existing fallback for connectors with no detectable
mouth.

### 4. Make the leaf-level impossibility observable

**File:** `kicraft/autoplacer/brain/array_placement.py` and the existing caller
that records leaf rejection diagnostics, only if propagation is currently
opaque.

An impossible same-edge 2-D connector array must fail the leaf candidate with a
stable reason such as:

```text
edge_connector_array_not_one_dimensional:J1,J2,...@bottom(2x2)
```

Use the existing leaf failure/rejection channel. Do not catch the error and run
the generic array path. No new fallback.

## Regression tests

### Unit: edge-bank geometry and facing

**File:** `tests/test_array_placement.py`

Add a WJ126V-shaped two-terminal helper using ordinary `Component` objects:

- two pads on a 5 mm pitch;
- `kind="connector"`;
- body approximately `7.89 × 10.09 mm`;
- `opening_direction=180°`;
- initial rotation `0°`.

Add these tests:

1. **KC-AHW6GA bottom-bank regression**
   - `rows=1`, `cols=2`, both refs `edge:bottom`.
   - Assert the centers share Y and differ on X.
   - Assert both rotations are `90°` for local opening `180°`.
   - Assert `opening_board_angle(...)` equals bottom's outward angle for both.
   - Assert members remain locked and non-overlapping.

2. **All four edges**
   - Parameterize top, bottom, left, and right.
   - Assert top/bottom banks are horizontal and left/right banks vertical.
   - Assert every mouth faces the assigned edge.

3. **Generic LED array unchanged**
   - Retain the existing DOUT-facing and serpentine tests unchanged; add no
     compatibility switch. These tests prove the specialized branch does not
     capture data-chain arrays.

4. **Mixed/missing zones do not specialize**
   - An ordinary connector array without one common explicit edge follows the
     existing generic behavior.

5. **Impossible 2-D same-edge bank fails loudly**
   - A 2×2 grid of four connectors all assigned to bottom raises/records the
     stable leaf rejection reason instead of creating two rows.

### Unit: parent composer no longer needs mouth-only fallback

**File:** `tests/test_usb_edge_connector_placement.py` or a focused addition to
`tests/test_subcircuit_composer.py`

Construct a solved leaf from the fixed two-connector bottom bank and call
`derive_attachment_constraints` with both refs zoned bottom. Assert:

- the only allowed parent rotation is the one that leaves the bank parallel to
  the bottom edge (`[0.0]` for the fixture geometry);
- that rotation makes both connector mouths outward;
- `_edge_zoned_is_leaf_extremity` is true for both refs;
- no `"no rotation places every edge-zoned part"` warning is emitted.

This test is essential. A leaf-only assertion can pass while the rigid parent
transform still admits the old vertical result.

## End-to-end verification

Use the frozen workspace for `KC-AHW6GA` through the real place/route/build tail,
not a hand-edited PCB and not a parent-only compose replay. A parent-only replay
would reuse the already-wrong solved leaf and cannot verify this source fix.

Expected evidence:

1. INPUT OUTPUT solved leaf:
   - J1/J2 remain a horizontal bank for the bottom assignment;
   - both mouths face bottom before parent composition.
2. Parent pipeline:
   - selected INPUT OUTPUT leaf rotation keeps the bank tangent to bottom;
   - J1 and J2 are both bottom extremities.
3. Connector gates:
   - no `connector_stranded:J1` or `connector_stranded:J2`;
   - no `connector_misoriented:J1` or `connector_misoriented:J2`;
   - connector edge gaps are each at least `-connector_edge_inboard_tol_mm`.
4. Electrical/layout checks remain clean:
   - shorts `0`;
   - unconnected `0`;
   - courtyard overlaps `0`;
   - keepout intrusions `0`.
5. Inspect the generated 3D render and confirm both wire-entry faces are
   independently accessible from outside the same board edge.

Run the focused tests first, then the applicable array/composer/connector suites.
For the production replay, follow the repository's `verify` skill so the frozen
workspace is exercised through the honest build tail.

## Acceptance criteria

- The KC-AHW6GA geometry cannot recur: two same-edge side-entry connectors never
  become a bank normal to that edge.
- The specialized behavior is derived solely from existing `component_zones`
  and connector mouth metadata; no board-code, footprint-name, or WJ126V special
  case exists.
- Generic LED matrices and rings retain their current chain orientation and
  placement.
- Impossible multi-row same-edge connector arrays fail during leaf placement,
  before routing or parent composition.
- Parent composition finds a transform satisfying both mouth-facing and
  extremity constraints; it does not enter the mouth-only fallback.
- The real KC-AHW6GA replay finishes electrically clean with both terminal mouths
  accessible in the 3D render and no connector placement findings.

## Non-goals

- Changing `connector_stranded` from warning to a fab-export blocker.
- Altering connector footprint mouth detection.
- Reworking synthesis array inference.
- Replacing the parent synthetic-block model with multi-anchor constraint
  solving.
- Improving routing or board scoring unrelated to the connector bank.
