# Fix per-child transformed-copper accounting mismatch

## Problem

A real KRT parent route preserves every authoritative input copper item, but the older per-child transformed-copper ledger reports complete loss:

- KRT adapter custody: 117/117 input traces and 13/13 input vias matched, zero missing.
- `metadata.json.copper_verification`: 0/117 child traces and 0/13 child vias matched.
- Electrical result: zero shorts, zero unconnected, zero genuine routed-copper clearance violations.

Do not weaken or remove either check. The KRT adapter check proves physical input/output custody at the routing boundary. The per-child ledger should independently attribute preserved copper to each composed child and distinguish it from newly routed parent copper.

## Reproduction evidence

Scratch replay used during the KRT cutover:

```text
/tmp/kicraft-krt-sole-router/USB_PD_TRIGGER/
```

Relevant artifacts:

```text
.experiments/subcircuits/subcircuit__8a5edab282/debug.json
.experiments/subcircuits/subcircuit__8a5edab282/metadata.json
.experiments/subcircuits/subcircuit__8a5edab282/parent_placed.kicad_pcb
.experiments/subcircuits/subcircuit__8a5edab282/parent_routed.kicad_pcb
```

In `debug.json`, inspect `routing_result.routing_stats.input_copper_preservation`:

- traces: `matched_count=117`, `missing_count=0`
- vias: `matched_count=13`, `missing_count=0`

In `metadata.json`, inspect `copper_verification`:

- `expected_child_traces=117`, `matched_child_traces=0`
- `expected_child_vias=13`, `matched_child_vias=0`
- every child reports 0% preservation
- `new_route_traces=235`, incorrectly classifying all routed-board traces as parent additions

## Likely root cause

`kicraft/autoplacer/brain/copper_accounting.py` makes manifest fingerprints relative to the minimum coordinate of **child traces only**:

```python
_origin = _trace_set_origin(all_child_traces)
```

`verify_copper_preservation` makes routed fingerprints relative to the minimum coordinate of **all post-route traces**, including newly added parent interconnects:

```python
_post_origin = _trace_set_origin(post_route_traces)
```

If any new parent trace has a lower X or Y coordinate than the child-copper minimum, `_post_origin` differs from the manifest origin. Every child fingerprint shifts by the same nonzero delta, producing the observed all-or-nothing 0/117 result. Existing unit tests add parent traces away from the child minimum, so they do not cover this case.

The unused `final_child_bboxes` argument to `build_copper_manifest` suggests an incomplete earlier attempt to preserve a stable composed-child frame. Confirm this hypothesis against the real artifact before choosing the implementation.

## Required investigation

1. Load the real manifest inputs at the end of parent composition and print or test-capture:
   - child-only trace origin;
   - routed-all-trace origin;
   - the coordinate delta;
   - one expected child fingerprint;
   - the corresponding absolute routed trace fingerprint.
2. Prove whether the mismatch is a single uniform translation or also includes rotation, endpoint reversal, layer naming, rounding, or post-route trace splitting.
3. Compare the manifest against `parent_placed.kicad_pcb` before KRT. If it already reports 0/117 there, the bug is composition/stamping frame selection, not routing.
4. Compare the absolute input fingerprints recorded by KRT with the manifest's expected child fingerprints. Use this only as evidence; do not couple the per-child ledger to KRT-specific stats.

## Design constraints

- Keep `copper_accounting.py` router-independent and pure Python.
- Preserve per-child attribution and multiset consumption; duplicate geometry must not be double-counted across children.
- Preserve endpoint direction normalization behavior as currently defined. If direction reversal is discovered, fix it explicitly and test it.
- A uniform board translation must not change preservation results.
- Newly routed parent copper before, after, or between child coordinates must not change the child reference frame.
- Parent traces and vias must still be counted as `new_route_*` after child matches are consumed.
- Do not fall back to count-only custody when a manifest exists.
- Do not copy the KRT adapter result into the per-child result; the two checks defend different boundaries.
- Do not loosen coordinate precision globally without evidence. A tolerance change can alias distinct short segments.

## Preferred implementation direction

Use a stable reference frame derived only from copper that exists on both sides of the comparison.

One boring approach:

1. Store the manifest's absolute child fingerprints and a manifest origin explicitly in `CopperManifest`.
2. During verification, solve the uniform translation by matching translation-invariant geometric features that exclude absolute position:
   - layer;
   - width/diameter/drill;
   - segment delta `(dx, dy)` with endpoint-order normalization;
   - candidate translation from an expected endpoint to a routed endpoint.
3. Select the translation that maximizes multiset child matches, then apply it once to all expected child traces and vias.
4. Consume matches per child from routed multisets and classify the remainder as new parent copper.

A simpler alternative is acceptable if investigation proves a stable parent board origin is already available at both manifest-build and post-route import time. In that case, pass that explicit origin through `CopperManifest` and never recompute it from the routed trace set.

Do not use the minimum of all routed traces as the verification origin.

## Implementation steps

### 1. Add a failing pure-Python regression

In `tests/test_copper_accounting.py`, construct:

- one child trace and via around `(20, 20)`;
- a manifest from that child;
- a post-route board translated uniformly, if translation is part of the real path;
- a new parent trace with a lower X/Y coordinate than every child trace.

Assert:

- child trace and via preservation remain 100%;
- `new_route_traces == 1`;
- only the new parent trace remains unmatched;
- per-child counts are correct.

Add a second case with two children and duplicate-looking trace geometry so multiset consumption and attribution remain deterministic.

### 2. Make the reference frame explicit

Update `CopperManifest` and its serialization only as needed. If adding fields:

- include a schema-stable serialized representation;
- update `to_dict` tests;
- update any fixture goldens through `scripts/replay_corpus.py --mode parent --update`, not by hand.

Remove `final_child_bboxes` if it remains genuinely unused after the fix; otherwise use it with a documented invariant. Do not leave another dead compatibility parameter.

### 3. Fix matching

Update `build_copper_manifest` and `verify_copper_preservation` so parent-added copper cannot shift the child reference frame. Keep fingerprint multisets and per-child consumption.

Add diagnostic fields useful for future failures, bounded in size:

- chosen translation/reference origin;
- unmatched expected count by child;
- a few unmatched expected fingerprints;
- a few unmatched routed fingerprints.

Do not serialize every fingerprint into normal metadata.

### 4. Add an integration contract

Add or extend a parent composition test that:

1. builds a manifest from transformed child copper;
2. stamps/imports the parent board;
3. adds parent copper with coordinates outside the child minimum;
4. verifies all child copper and only the added copper is classified as new.

Use native KiCad geometry if the existing test conventions provide pcbnew; otherwise keep the permanent contract pure Python and use the real scratch replay as the smoke test.

### 5. Verify against the real routed board

Run the focused tests:

```bash
.venv/bin/python -m pytest -q \
  tests/test_copper_accounting.py \
  tests/test_kicad_routing_tools.py \
  tests/test_artifact_paths.py
```

Then replay a cold scratch copy with the pinned KRT runtime, or reuse the documented workspace only for diagnosis. The final fresh run must show:

- adapter custody: 117/117 traces, 13/13 vias, zero missing;
- per-child ledger: 117/117 traces, 13/13 vias;
- `new_route_traces == routed_total_traces - 117` after multiset matching;
- `new_route_vias == routed_total_vias - 13`;
- every child has 100% trace and via preservation;
- `copper_verification.status == "PASS"`;
- zero shorts and zero unconnected.

Regenerate committed parent replay goldens only after the behavior is proven:

```bash
.venv/bin/python scripts/replay_corpus.py --mode parent --update
```

## Acceptance criteria

- The new regression fails on the current implementation and passes after the fix.
- Adding a parent trace before the child-coordinate minimum cannot invalidate child matches.
- Uniform translation between composed and saved board frames is handled explicitly.
- Duplicate fingerprints are consumed once and attributed to one child only.
- The real USB-PD parent reports 117/117 child traces and 13/13 child vias preserved.
- KRT adapter custody remains unchanged and independently green.
- No count-only fallback, tolerance inflation, router-specific dependency, or compatibility alias is introduced.
