# KiCraft board artifacts — the source of truth

This is the canonical answer to the two questions that keep biting: **where is the
current board on disk, and was it produced by this run (or is it stale)?**

If you take one thing from this doc: **don't glob `.experiments` by hand and don't
measure `<stem>.kicad_pcb` for placement work. Run `kicraft artifacts`** (or call
`kicraft/cli/artifact_paths.py`). One resolver, one provenance record, one truth.

```bash
kicraft artifacts --project <synthesized-project-dir> --kind all      # human
kicraft artifacts --project <dir> --kind placed --json                # machine
```

## On-disk layout

A synthesized project keeps every board the layout engine produces under
`<project_dir>/.experiments/subcircuits/<slug>/`. The top-level `<stem>.kicad_pcb`
is a **promoted copy** of the best board a build/replay reached — never edit or
measure it as if it were the engine's output.

```
<project_dir>/
├── <stem>.kicad_pcb            # PROMOTED copy of the best board this run reached
├── <stem>.provenance.json      # which run promoted it, from what source, md5, fresh?
├── <stem>.kicad_sch / .kicad_pro / .kicad_prl
├── .experiments/pre_promote_seed.kicad_pcb   # full-component seed saved before the
│                               # promote clobbers <stem>.kicad_pcb; replay restores it (rc6)
└── .experiments/subcircuits/
    └── <slug>/                 # one dir per leaf or parent subcircuit
        ├── metadata.json       # parent: carries run_id + generated_at (routed compose only)
        ├── debug.json, solved_layout.json
        ├── parent_placed.kicad_pcb   # PLACED parent (what --no-route produces)
        ├── parent_routed.kicad_pcb            # ROUTED parent (post-KiCad Routing Tools)
        ├── leaf_placed.kicad_pcb     # PLACED leaf
        ├── leaf_routed.kicad_pcb              # ROUTED leaf
        ├── leaf_illegal_pre_stamp.kicad_pcb   # legality-REJECTED placement (rc6 preview only)
        ├── round_NNNN_*.kicad_pcb             # per-round snapshots (autoexperiment)
        ├── _search/cand_NN.kicad_pcb          # candidate placements (rmtree'd after the winner)
        └── renders/*.png
```

Names are defined **once**, in `kicraft/cli/artifact_paths.py`
(`PARENT_ROUTED`, `PARENT_PLACED`, `LEAF_ROUTED`, …). Do not hard-code these
literals in new code — import them.

## Resolving a board (intent-based)

`kicraft/cli/artifact_paths.py` is the only resolver. It is **intent-based**, not
"richest board wins":

| Want | Call | Returns |
|---|---|---|
| the routed parent | `resolve_parent_board(dir, kind="routed")` | `parent_routed.kicad_pcb` or `None` |
| the **placed** parent | `resolve_parent_board(dir, kind="placed")` | `parent_placed.kicad_pcb` only — **never** a routed board |
| an rc6 preview leaf | `resolve_best_leaf_board(dir)` | richest leaf (routed > placed > rejected) |

`kind="placed"` **never** falls back to the routed board. That is the fix for the
old `replay --no-route` trap, where the resolver returned a *routed* board from a
previous run and the placement-only run silently promoted it. When several parent
dirs exist, the newest is chosen by board/metadata **mtime** (deterministic), not
`iterdir` order or alphabetical `sorted()[-1]`.

## Freshness & provenance — "did this run produce it?"

Every command that promotes a board (`build`, `replay`, `manual-route`) calls
`artifact_paths.ensure_run_context()` at entry, which sets two env vars inherited
by the compose/route subprocesses:

- `KICRAFT_RUN_ID` — a short id for this run (honors one the web driver injected;
  same convention as `parts_library/query_log.py`, so board provenance and
  part-lookup logs correlate).
- `KICRAFT_RUN_STARTED_AT` — wall-clock start, the freshness reference.

The promote tail **gates on freshness** (`produced_by_this_run`): a board the
current run did not produce is **never silently promoted**.

- **Routed fab promote** and **`--no-route` placement promote**: if the resolved
  board isn't from this run → **loud failure** (the routed case falls through to
  the rc6 inspection preview; the `--no-route` case errors `rc6`). No more "my
  change had no effect" because a stale board was copied.
- **rc6 best-partial preview**: this path *exists* to show "whatever this run
  got" on failure, so a non-fresh partial is a **warning**, not a hard error —
  the board is still shown for inspection.
- **Parent search failure**: a fresh `parent_routed.kicad_pcb` can be a
  retained router failure, not a selected result. If no successful round can
  be pinned, the search returns rc6; the build shows a placed/leaf partial and
  does not verify or export that rejected file.
- **Selected parent evidence**: the winning PCB, project/rule sidecars and
  contemporaneous diagnostics are snapshotted together. `parent_custody.json`
  binds their hashes to the run and round; re-pinning restores the whole unit,
  so copper accounting cannot silently describe a later discarded composition.
- **Antenna edge geometry**: parent composition aligns antenna banks by the
  antenna support line, not the intentionally outboard keepout/courtyard.
  Other blocks must remain inboard; outline repair still checks real copper
  and pad containment. Final antenna placement remains a hard export gate.
- **Seeded cluster placement**: graph construction, community ties and cluster
  traversal use canonical reference order before seeded random choices.
  Initial and intra-cluster placement do not assign random draws or force
  accumulation order according to Python's per-process set order.
- **Input copper custody**: KRT is given a temporary lock on one input item per
  previously unlocked copper net. Its locked-net protocol forbids automatic
  rip-up, while ordinary routing may still add connections to those nets.
  Temporary locks are removed from the output; authored locks remain. The
  post-route copper comparison remains authoritative: a router's "dead-end"
  removal count is not proof that missing input copper was expendable.
- **Leaf *inputs* are never gated.** `replay` and the autoexperiment
  `--parents-only` phase reuse frozen leaves from a prior run on purpose.

`produced_by_this_run` trusts a **positive `run_id` match** (authoritative; immune
to clock skew and mtime-preserving copies) and otherwise falls back to
`mtime >= run_started_at`. It never *rejects* on a `run_id` mismatch, because
`metadata.json` (the run_id source) is only rewritten on a **routed** compose —
on a stamp-only `--no-route` run the board is freshly `Save()`'d (fresh mtime) but
`metadata.json` is not, so mtime is the correct signal there.

Promotion uses `shutil.copy` (not `copy2`), so `<stem>.kicad_pcb`'s mtime reflects
**when it was promoted** — `find -newer` / "did it change?" work again.

`<stem>.provenance.json` records the last promote: `run_id`, `source_board`,
`source_kind`, `source_md5`, `md5`, `fresh`. The writer runs after final board
mutations, including the silk legend: `md5` binds the delivered board, while
`source_md5` retains the selected router output's identity. Those bytes can
legitimately differ after post-processing.

`read_provenance` checks the current board without rewriting the historical
sidecar. It reports `current_md5` and `hash_matches`; a changed or missing board
cannot read as `fresh`. In `kicraft artifacts --json`, `matches_promoted_run`
compares run identities only, never byte equality.

## Reviewed power layout and fabrication process

Local power contracts are reconstructed from the selected reviewed construction
and actual BOM ownership. Every indexed capacitor-bank member is bounded; input
HF, both bootstrap terminals and switch-to-inductor routes have separate native
length/width checks. Output-bank current paths start at the inductor output,
not the feedback pin. Native copper contacts, layers and continuous pad escapes
are checked after promotion; nearby pad centers alone do not prove a short route.
Output-bank placement is anchored to the actual inductor output as well. An
explicit `ground_return.owner_ref` independently identifies the converter's
ground pad; the placement owner is not assumed to have a ground terminal.
Ground-return proof requires a shared native filled-copper island or a locally
bonded via within the contract distance, not merely a nearby same-net via.
Nested placement owners reserve in dependency order and use the same hard
courtyard clearance throughout their connected ownership family. A single
coupled bank tries interior anchor positions first; independent groups retain
edge-first packing. Neither ordering removes legal candidate positions.
Within each ownership-group search, unchanged obstacle envelopes are reused;
candidate bounds are rebuilt per pose and applicable gap. The cache is rebuilt
after installing a group's poses, so nested owners retain current geometry.
CH224K VDD bypasses use the same plane-backed ground proof: the supply terminal
stays within 3 mm of VDD, while the buried exposed ground pad and capacitor ground
must bond to common filled copper. A 3 mm ground-pad-centre rule would require
the external capacitor to intrude into this package's physical envelope.
Narrow escape lengths count copper outside actual native pad boundaries,
excluding drill holes; pad-covered intervals retain their physical route length.
Finite-width tangencies and shallow contacts cannot masquerade as full-width
current paths. Continuous exposed neckdowns remain limited to 1.5 mm.
Offset track/via contacts qualify only when an interior transverse section proves
the entire actual track width lies inside the annulus, excluding the drill with
two-native-unit margins. The measured section joins the existing via layer node
at its actual distance; unproved offset contacts retain connectivity only.
The pinned router runs width-homogeneous non-ground groups widest first, then
configured plane-ground groups at their original widths. Each pass has an
explicit base width. This prevents router cleanup from silently replacing
a power-class segment with the narrower default-class width. Every pass retains
the complete preceding copper and actual project rules; the passes share one
deadline. Ambiguous cross-width net selection is rejected before routing.
Per-pass commands, logs and custody evidence are retained in routing statistics.
Parent passes default to geometry-derived `inside_out` ordering; leaves retain
`mps`. Keeping parent ordering independent avoids closing dense child-port
access with other interconnects before routing those ports. The setting is
`kicad_routing_tools_parent_ordering`; widths, custody and safety gates are unchanged.
The pinned router also checks physical connectivity when recovering multipoint
victims and restores previously closed victims if recovery would disconnect them.
Routed leaves are checked against native current-width, route-length and
locally reconstructible ground-return contracts before candidate acceptance,
including after copper repairs and on the trivial routed-leaf path. Thermal
requirements whose owners are present are also checked before pinning. Both
ground faces are refilled; a nearby via on a separate island does not qualify.
External ground owners defer to composition, where all canonical current,
ground and thermal requirements are checked again.

Contracted ordinary ground witnesses are constructed and protected before leaf
signal escapes can fragment them. Signal exits then precede remaining ground
escapes and power ties, including local signals throughout multipad SMD fields
and wide cross-leaf SMT pads. Pad width/pitch alone must not exclude those
signals. Canonical `power_nets` remain excluded from signal-width fanout even
when their names do not match a voltage heuristic.
Every disconnected external signal component
must reach a native-connected via or PTH landing; a short copper nub or a via on
a different same-net island is insufficient. This also applies to singleton
ports with no local DRC open and to trivial leaves.
Leaf routing keeps in-plane connector wiring behind its native mating datum
before escapes and routing begin. Author markers, long-barrel pad anchors and
body mouths follow the parent attachment convention. Final silk and Edge.Cuts
retain that boundary; outboard routed copper remains a native failure rather
than growing the board past the connector opening.
After routing, one guarded completion pass targets only interface pads that
still lack physical landings. An off-board planned waypoint may use the same
guarded short exit as other blocked plans; all edge, copper, pad, hole and
keepout checks remain required. Final landing and physical audits—not the
presence of anchor names—determine acceptance.
Signal-width repair excludes every canonical power rail, including rails with
non-voltage-like names and explicitly requested extra nets. An unresolved
power connection may instead be repaired by a separate guarded native mesh at
its resolved net-class track and via dimensions; unresolved connections remain
blocking failures. Power repair runs before signal repair within the shared
accept-or-revert pass, whose summary retains both per-kind results.
Zone reports identify a zone object, not a copper contact. Power repair joins
its physically disconnected filled-copper components through real pads or vias;
it never routes from the reported zone origin. Mixed zone/item membership keeps
native identity and the report's coordinate precision, rejecting ambiguity.
Reported signal track endpoints may reuse existing via or PTH access only through
their native-connected copper component. A same-net via on another island is not
access; ambiguous rounded report coordinates do not authorize borrowing it.
Common existing back-layer access is preferred without increasing the guarded
stamping budget, endpoint-gap cap, or any copper/clearance/keepout permissions.
Track-to-pad guards apply solder copper only on the track's actual face.
PTH/NPTH holes still block both faces; a new through via must clear solder
lands on either face. An undrilled front-only pad is not a back-track obstacle.
Mesh via windows reserve the same solder-mask expansion (at least 0.1 mm)
as the stamper, including connector lands; copper-only clearance is insufficient.
Drilled-pad obstacles also reserve the stamper's hole-overhang margin on both
faces, even where a narrow solder land understates the slot's drill envelope.
When simple paths fail, signal repair may join native clearance-mesh components
through legal via windows on both faces. Ground reservation plans remain
hash-validated obstacles; existing pours yield on refill. The candidate uses the
same stamping budget and endpoint-gap cap, retains every copper/clearance/hole
guard, and is byte-restored unless all legs stamp. Final native DRC and current,
ground-return and thermal audits still determine acceptance.

If a local current contract still fails, shallow same-net track/via contacts
receive explicit full-width landing segments through the existing breakout
guards. These do not add drills or relax annular-contact proof, width, length,
clearance, hole or authored-keepout requirements. Both faces are refilled and
all current/ground contracts are audited again before acceptance.
The mask-free input DRC copy is also refilled after signal escapes, so obsolete
ground-fill polygons are not mistaken for actual shorts. Board/project/rule
custody is preserved around both operations.

Size-reduction reroutes pass the same structured leaf gates as ordinary rounds.
Rejected candidates restore their prior board snapshots; accepted candidates
replace the winning round's validation and layout records with the reduced
board's evidence before pinning.
Before reduction or persistence, canonical boards are materialized from the
selected round's snapshots, not left holding a later rejected attempt.

Default leaf convergence is invocation-local, including the production
`--leaves-only` phase. A reused leaf must have a current solve's accepted,
selected PCB/layout/metadata snapshot unit and matching canonical bytes and pin.
Changed source, project inputs, solve configuration or custody invalidates it.
Qualified units keep a fixed leaf solve configuration while unresolved leaves
retry within the original limits; complete qualification stops leaf cohorts.
Parent search retains its own mutated configuration and configured limits.
Parent placement retains leaf-qualified interface-pad identities and independent
copies of their real positions. Rotation moves those pads with the measured
child-body geometry. Packing and rotation choices therefore include actual
inter-child wiring costs; rejected individual or replica-group poses restore
both body geometry and pad positions without modifying the child artifacts.
Child-copper accounting still maximizes the complete multiset match under one
uniform translation, with the same coordinate tie-break. Candidate votes order
work but never prune it. A candidate stops only when its remaining-item bound
cannot beat the current match or tie; full-board copies do not trigger exhaustive
fingerprinting of every losing same-shape translation.
Parent ground- and unconnected-repair subprocesses retain the complete fabrication and routing
configuration. Runtime sets use the shared JSON configuration conversion;
individual guard settings must not disappear at the process boundary.

Ground pours default to the fabrication clearance floor, preserve explicitly
wider clearances, and never accept a configured value below that floor. The
merged configuration leaves zone clearance unset unless a project overrides it;
ground writers reapply that policy to reused zones as well as new ones. Power
pours retain their 0.30 mm default. This allows qualified fill between fine-pitch
ground pads without an inherited 0.30 mm gap excluding their required bonds.
Thermal area, connected F/B spreading and local via requirements remain unchanged.
The normal ground finisher materializes both faces for native-owned thermal
contracts after routing and after outline/silk restamping clears zones.
It shares the final thermal audit's direct-island and nearby connected-via
predicates: a filled path around a hole need not be straight or have a track.
Missing local plane stitches are created at guarded off-pad points in actual
two-face filled overlap. Existing qualified witnesses do not gain extra vias.
After competing power pours, deficient local thermal copper can reclaim bounded
ground fill at a higher, deterministic zone priority. These are actual copper
zones, not routing masks; native filling still clears foreign copper, holes,
edges and authored keepouts. Owned zones follow current owner/pad geometry and
requirements rather than surviving moves or removal at stale coordinates.
Global ground finishing uses native island intersections, guarded off-pad
stitches and guarded detours. Its conservative geometric cluster estimate does
not replace the final native connectivity, current, thermal and DRC verdicts.
Stitch and escape guards retain configured hole-to-copper clearance independently
of drilled-hole spacing; ordinary solder openings remain undrilled.
Thermal spreading is built from the finished copper, rather than solid routing
windows that can cage signal fanout vias. Final area/via gates remain unchanged.
After power-strand repair and final thermal-zone construction, parent finishing
rebuilds guarded thermal stitches against the current fill. Earlier qualified
vias are not assumed to retain their two-face plane bond after later pours.
Leaf input preparation and parent routing protect thin connected ordinary return
paths. B.Cu is preferred; a qualifying F.Cu return remains valid when no protected
B.Cu path exists. Native holes, exits outside witnesses and unrelated areas remain
available. Reservations remain active across every width pass and are removed
before publication; parent planning stays inside the unchanged shared routing
deadline. Ground preparation and mask cleanup preserve authoritative project and
custom-rule sidecars. Final shared-region, local-area/via, copper-custody and DRC
gates remain authoritative.
Parent preparation also preserves the complete connected physical ground-pad
network with a thin forest through native-eroded fill and fixed copper. Closest
qualified contacts lead; equal-distance choices prefer B.Cu rather than forcing
a longer back-face bond ahead of a shorter front-face one. Duplicate pad numbers
remain separate physical contacts. Meshes and contact anchors are reused; an
incomplete witness forest blocks routing.
Witness widths retain each source zone's native minimum thickness, even when
the fabrication track floor is smaller.
Via-based local returns also reserve any fill-dependent pad-to-via landing on
the pad's own face. A protected opposite-face plane does not protect that landing;
an already-qualified immutable copper path needs no additional landing mask.
Qualified via choices and forest-contact ties use physical coordinates and copper
properties, not native serialization or generated UUID order. Equivalent copper
must not reserve different signal-routing space merely because item order changed.
The pinned router allocates real interconnects jointly across both copper faces.
A ground net leaves explicit trace-routing targets only after its complete
native all-pad forest is proved; a configured plane name alone is insufficient.
Plane-only work retains an immutable hash-bound plan and verifies input copper
without inventing an external-router run. Final electrical and DRC gates still apply.
The router also returns a hash-bound, tagged reservation-plan board. Subsequent
power pours exclude those exact ground corridors, expanded for the actual
ground/power clearance, from their native outlines. A late higher-priority power
plane must not erase ground paths protected during routing. Power-repair IPC
retains the complete configuration and plan; missing or mismatched supplied
plans fail closed. Final native filling and electrical audits remain required.
The existing absolute routing deadline covers native ground preparation and
loader retries as well as external width passes. Expiry terminates the owned
native process group and reports a deadline, not geometric impossibility.
Self-touching native contours may be decomposed for planning only when all
vertices remain on the native grid and native Boolean XOR proves equivalence.
Finite-width witnesses are still checked against the original native copper.
Ground repair stops when an iteration adds no copper rather than retrying
identical geometry. Final native connectivity remains authoritative.
Reservations compensate source-zone clearances wider than the router's clearance;
their planning boundary may extend beyond the protected fill, without adding copper.
Witness shortcut search rejects impossible straight shortcuts against the
prepared polygon before running the unchanged native finite-width bond proof.
Rejected shortcuts retain the full mesh route; they do not prune reachability.
The router also receives a positive same-net pad/via clearance at the project
fabrication floor. This disables pad-centre swap/rescue vias that can bypass its
via-only assembly rule areas. Authored copper and reviewed thermal VIPs remain
subject to the same input-custody checks; qualified ground finishing, not signal
routing, owns new thermal VIPs. Ordinary mask/paste drill guards remain unchanged.
Assembly-opening contours retain owned integer coordinates before the temporary
KiCad shape is replaced for another face. Dual-face connector pads must preserve
both actual mask outlines, not borrowed native point references.

Required 2 oz outer copper is emitted before placement/routing, not added only
to the final instructions. The supported two-layer stack is 1.600 mm including
0.070 mm F/B copper, 0.010 mm masks and 1.440 mm dielectric. Its trace/spacing
floor is 0.170 mm, conservatively above the manufacturer's published 0.160 mm
2 oz minimum: https://jlcpcb.com/help/article/jlcpcb-copper-weight.
Project rules, netclasses, escape/repair subprocesses and router overrides retain
that process floor. Actual narrow tracks/arcs and clearance violations block
fabrication; a correct rule declaration cannot qualify narrower copper.
Plated component holes require at least a 0.254 mm annular ring, separately from
the via annulus, and drilled holes retain at least 0.200 mm edge-to-edge spacing.

Thermal contracts require connected ground spreading on both faces, local
connected fill and nearby qualified stitching. These are geometry screens, not
a measured thermal resistance or a guaranteed junction temperature. Fabrication
instructions explicitly require resin filling, copper capping and planarization
where vias occupy SMT lands; an exported CAD package does not claim those
manufacturing operations have already occurred. Standalone export derives the
heavy-copper requirement from the actual board and requires fresh native DRC.

## Fabrication package and STEP body coverage

Fresh export receipts bind the exact PCB, ZIP and every ZIP member by SHA-256.
The package's `step_model_coverage.json` records top-level assembly references
observed in the actual STEP, with explicit missing references or an unknown
result when the structure cannot be interpreted safely. The persisted
`step_model_missing_refs` is `null` for unmeasured/unknown, `[]` when all expected
assembly references occur, or the list of missing references.

Reference presence is not proof of model accuracy, placement, render fidelity
or assembled mechanical clearance. Missing bodies are shown in the GUI and CLI;
they are not silently replaced by generic shapes. Bundled model attribution
is staged alongside the model and retained in downloaded KiCad projects.

Paired WRL and STEP assets must use the same footprint-local coordinate frame.
Audit actual exported terminal registration and body seating against native
footprint transforms; matching reference counts cannot detect a shifted, floating
or PCB-penetrating body. Correct source model frames, preserve the failed package,
and publish a fresh byte-bound export rather than patching the delivered STEP.
Standalone candidate inspection reports are not necessarily reports of the
final promoted layout. Bind any final raw-DRC claim to the delivered PCB hash;
a selected-category fabrication gate does not establish zero warnings.


Hand-fitted non-PCB items, such as a removable enable shunt, are typed
accessories of their owning BOM part. Their exact order codes, quantities and
positive-stock qualification are included in the sourcing receipt and its BOM
digest. The fabrication ZIP includes a separate `accessories.csv` procurement
list with unique accessory IDs and owner references. These items do not become
fake PCB footprints, netlist components, placement rows or PCB assembly-BOM rows.
Missing or changed accessory procurement evidence invalidates its product proof.

An assembled chip passive without a selected compatible catalog package is not
a complete procurement row. Missing candidates, insufficient stock and
unverified retail availability must not silently leave an assembled part out
of the sourcing receipt. Generic header geometry likewise does not qualify a
purchased ordering identity: the exact header, pitch, contact map and nominal
manufacturer land pattern must agree.

## Invariants (do not break)

- **Provenance lives only in JSON sidecars — never inside a `.kicad_pcb`.**
  Boards are byte-stable; the geometry goldens (`scripts/replay_corpus.py`,
  `tests/test_replay_command.py`) compare footprint x/y/rotation only.
- **Never measure `<stem>.kicad_pcb` for placement A/B.** Resolve the placed board
  (`kind="placed"` / `kicraft artifacts --kind placed`). The A/B scripts
  (`replay_corpus.py`, `ab_compose.py`) already do, and follow the
  "delete the canonical output → compose → measure" discipline.
- **`.experiments` is not wiped at run start.** Staleness is handled by the
  freshness gate, which lets `replay` reuse frozen leaves and preserves the rc6
  preview. Don't add a blanket wipe.

## See also

- `kicraft/cli/artifact_paths.py` — the implementation and the precedence rules.
- `docs/bug-replay-no-route-promotes-stale-routed-board.md` — the investigation
  this resolves (kept for history).
