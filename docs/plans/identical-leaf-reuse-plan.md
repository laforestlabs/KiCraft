# Identical-subcircuit leaf reuse (isomorphic dedup)

**Status:** SHIPPED (2026-08): `solve_subcircuits.py` drives `plan_leaf_replication` / `materialize_sibling` and `autoexperiment.py` runs `finalize_leaf_replication` after pinning (kill switch `cfg['leaf_replication']`, default on). History: IN PROGRESS 2026-07-17 on `placement-streamline`. User directive: boards with
repeated identical functionality (KC-AXHQTP: OPTO CH1–4, RELAY CH1–4) must place the **exact
same leaf** for every instance — solve the best version once, duplicate it identically so all
instances function identically. See [[kicraft-identical-leaf-reuse]].

## The problem (confirmed on run 1/626)

The synthesis emitter writes each channel as its OWN sheet file (`OPTO_CH1.kicad_sch` …
`OPTO_CH4.kicad_sch`), **byte-identical except ref designators** (U2→U3, R1→R4…) **and
channel-numbered net names** (IN1→IN2, OPTO_OUT1→OPTO_OUT2). The solve then treats them as 10
independent leaves — each gets a distinct `slug__hash(instance_path)` dir and is solved with a
different seed, producing four *different* placements of the electrically identical channel.

Root cause of the missed dedup: `compute_source_hash` (`subcircuit_artifacts.py:396`) fingerprints
a leaf by its instance-specific labels (sorted `component_refs` + `interface_ports[].net_name`), so
isomorphic channels never share identity. But even a canonical hash would be inert: **nothing reads
these hashes** — there is no cache-hit / reuse path anywhere (verified by exploration).

## Key architecture facts (from the exploration pass)

1. **The ref/net remap primitive already exists, fully unit-tested, but is UNWIRED:**
   `kicraft/cli/_replicate_leaves.py` — `build_replication_maps(rep_refs, sib_refs, rep_comps,
   sib_comps)` (strict structural check → `(ref_map, net_map)` or `None`) and
   `remap_solved_layout(rep_layout, ref_map, net_map, sib_identity)` (deep-copies the rep's
   solved_layout dict, rewrites every ref/net-bearing field, preserves geometry verbatim). Its only
   referent today is its own test.
2. **The reuse seam is artifact generation, NOT compose.** Compose discovers one artifact dir per
   instance on disk (`_discover_artifact_dirs`) and loads it 1:1 by `instance_path`
   (`subcircuit_instances.load_solved_artifact`). It needs `metadata.json` + `debug.json` +
   `solved_layout.json`; when `solved_layout.json` is present `_normalize_to_canonical` returns it
   **as-is** (debug content is bypassed). So a materialized sibling dir Just Works downstream.
3. **The final board's refs come from the SEED PCB, not the leaf mini_pcbs.** The parent stamp
   (`_parent_stamp_subprocess.py:65`) loads the seed parent PCB (already carrying every instance's
   real refs from synthesis) and only *moves* footprints to composed positions; positions come from
   each leaf's `solved_layout` components. So sibling identity flows correctly as long as its
   `solved_layout` maps sibling-ref → the rep's relative position.
4. **The mini_pcb is read only for blocker GEOMETRY, never nets** (`_extract_blockers_from_pcb`
   builds ref-keyed pad/courtyard/anchor rects — reads refs + geometry, no net logic). So a sibling
   mini_pcb needs a **ref-only** remap (its ref-keyed connector edge anchors must match the sibling's
   refs); net names are irrelevant there. `leaf_library/sexpr_edit.renumber_pcb_text(text, ref_map)`
   does exactly the ref remap.
5. **The round loop keys leaf acceptance off DISK** (`autoexperiment._accepted_leaf_artifacts`
   scans `solved_layout.json.validation.accepted`). A materialized sibling inherits the rep's
   `validation.accepted = True` via the deep-copy, so it counts as accepted with no round-loop change.

## Design

All in `kicraft/cli/solve_subcircuits.py` + a new `kicraft/cli/_leaf_replication.py`; the
`_replicate_leaves.py` primitives and `renumber_pcb_text` are reused verbatim.

**Step 1 — group (new `_leaf_replication.plan_leaf_replication`).** For each leaf being solved this
invocation, extract its board state (cheap, no solve), take its components in schematic order
(`node.definition.component_refs` ∩ extracted) + `serialize_components`. Bucket by a cheap key
(sorted multiset of `_component_footprint_signature`); within a bucket pick a deterministic
representative and verify every other member with `build_replication_maps` — matches become
`(sibling_node, ref_map, net_map)`; non-matches fall back to their own representative. Returns
`list[LeafGroup]`. Kill switch `cfg["leaf_replication"]` (default on). Grouping is scoped to the
leaves present in THIS invocation, so `--only` / rescue rounds stay self-consistent.

**Step 2 — solve only representatives.** The solve loop iterates `representatives` (was every leaf).

**Step 3 — materialize siblings (new `_leaf_replication.materialize_sibling`).** After a rep is
solved + persisted, for each sibling: `remap_solved_layout` → `solved_layout.json`; rewrite
`metadata.json` (sibling identity + `artifact_paths` → sibling dir, `mini_pcb` → sibling routed pcb);
copy `debug.json`; `renumber_pcb_text(rep_routed_text, ref_map)` → sibling `leaf_routed.kicad_pcb`.
Append a `(SolvedLeafSubcircuit via dataclasses.replace(rep, node=sib), persisted_dict)` pair so the
JSON/human summaries report all channels.

**Step 4 — post-pin finalize (new `_leaf_replication.finalize_leaf_replication`).** The leaf phase
runs N rounds and `_auto_pin_best_leaves` pins each leaf to its OWN best round by copying that
round's snapshot into the canonical `solved_layout.json`. A sibling has no round snapshots (it is
materialized, not solved), so it keeps whatever round last wrote it — which can differ from the
representative's finally-pinned round (observed: rep U rot=180, sibling rot=0). So after pinning we
re-materialize every sibling from its representative's now-pinned `solved_layout.json` + `mini_pcb`,
using the `(ref_map, net_map)` stashed in the sibling's `metadata.json`. Siblings are also given a
minimal `debug.json` (no `all_rounds`) so `_auto_pin_best_leaves` skips them. Called right after
`_auto_pin_best_leaves` in `autoexperiment` (leaves-only phase, before the parent phase composes).

**Step 5 — compose unchanged.** Discovers + loads the sibling dirs identically; the stamp moves the
seed PCB's sibling footprints to the (identical) remapped positions.

## Verification (all $0 replay)

1. Unit: isomorphic channels group together + produce valid `(ref_map, net_map)`; a topology
   difference (extra/relabelled component) does NOT group.
2. Replay 1/626 KC-AXHQTP: exactly ONE solve per class (OPTO ×1, RELAY ×1) in the log; OPTO CH1–4
   leaf-internal geometry byte-identical (same relative component positions/rotations); same for
   RELAY CH1–4; DRC 0 / 0 preserved; rc0; util still ~good (compaction fix intact).
3. Regression: a board with NO repeats (e.g. 1/622 buck) solves every leaf as its own representative
   (no behaviour change), rc0 DRC 0/0.

## Risks / watchpoints

- **Schematic order must align isomorphic components** (`build_replication_maps` pairs by position).
  Guaranteed for identically-generated sheets; the strict check returns `None` (→ independent solve)
  on any mismatch, so a bad pairing can never corrupt a board — worst case the optimisation just
  doesn't fire.
- **Ref-keyed blocker anchors:** the sibling mini_pcb must carry sibling refs (hence the
  `renumber_pcb_text` step) so edge-pinned connectors anchor correctly.
- **Round-to-round representative stability:** grouping is per-invocation; within a round all
  siblings match their round's rep (identical geometry → identical acceptance → pinned together).
