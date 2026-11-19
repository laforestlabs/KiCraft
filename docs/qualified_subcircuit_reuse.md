# Automatic qualified subcircuit reuse

## Product path

Ordinary Design, Surprise Me and the offline product runner share the design-stage
orchestration in `kicraft/server/stage_pipeline.py`. AI interprets the brief and
proposes functional requirements. Architecture resolution, recipe expansion,
lowering and pin allocation own deterministic electrical circuitry; the model is
not asked to recreate its support networks.

Synthesis separates complete deterministic owners into existing hierarchy leaves
(`isolate_owned_blocks` in `design/synthesis/sheet_partition.py`). It never splits
recipe-private connectivity, a lowerer support network, an array or explicitly
coupled placement. Unowned circuitry cannot inherit qualification from a reviewed
part sharing its original sheet. Existing per-sheet net/ownership transfer helpers
preserve the electrical graph and published interfaces.

After synthesis checks pass, `design/synthesize.py` writes
`.kicraft/qualified-leaf-requests.json` using
`leaf_library.qualification.synthesis_requests`. The local solver consumes this
artifact-bound request rather than optional model-selected `from_library` hints.
Matching uses the engineering contract, not the original brief or sheet name.
A compatible physical asset bypasses local solving; a miss uses the ordinary
local solve. Composition and board-level acceptance remain necessary in both
cases. Only an independently qualified candidate may be automatically published.

Selected-parent metadata retains each installed qualified block's geometry and
the preservation receipt bound to the selected routed PCB hash. The final
promotion gate checks native geometry again **after cleanup and silkscreen**.
Missing, stale, duplicated or changed custody fails fabrication export even when
ordinary board DRC is clean.

Input copper custody includes its original layers. The adapter requests both
`--keep-input-copper` and `--no-stub-layer-swap`, but does not trust these flags
as proof: native rescue paths may still relayer stubs. Native before/after
geometry checks reject such output. Dense public fanout reserves the parent's
resolved normal netclass via envelope, not merely the smaller escape-via
diameter, so attachment does not depend on moving inherited launches.

Global ground-plane reservations replace explicit routing only when a complete
native witness forest exists. An incomplete forest adds no partial global masks
and leaves that net in the ordinary routing worklist; mandatory local-return
masks and final native connectivity/DRC checks remain in force.
Dense two-layer source interfaces may request a both-face GND-plane proof before
private routing. Only a complete native forest replaces explicit GND routing.
Its checked corridors remain protected through recentering and power pour; final
native ground connectivity must pass again. Incomplete proof keeps GND in the
ordinary worklist.

Dense singleton signal interfaces reserve normal-width opposite-face launches
as well as fabricable vias. Source fanout leaves space for sibling launches
before private routing; native leaf acceptance rejects a bare connected via
when its required launch is absent. Dogbone, via and launch stamp atomically,
under the existing copper, hole, mask, outline and keepout guards.

Cleanup stays within its provenance scope. A leaf acknowledges previously
retired compiler escapes only through a native-readback receipt bound to its
exact router input and current output. Fresh native DRC may then authorize
router-born terminal fragments exposed by that cleanup. Every surviving input
identity remains inherited and protected; the parent cannot strip imported
private copper to repair an unfinished leaf.
A compiler pad-anchor segment is retained when it provides an interior landing
for a foreign route endpoint outside the pad. A diagonal grazing a pad corner
does not justify deleting that proper landing merely because finite copper
connectivity would survive.
Within a fresh local solve, load-bearing compiler joins may be canonicalized
before qualification: trim an overhanging terminal to its actual routed endpoint,
or replace a single-face via bridge with a same-width track. Exact ownership,
unchanged unrelated copper, retained connectivity, fabrication/contract guards
and complete native DRC are required. This does not authorize changing an
imported qualified asset.
After board integration, the same proof may normalize a compiler-owned public
join only when the selected parent's ledger explicitly declares that net as an
inter-block interface. The ledger also binds the applicable electrical/fab
constraints and preceding cleanup hashes. Missing authority permits no change;
private source nets and qualified asset geometry remain excluded.

Post-fill source cleanup carries raw router identities through the exact native
restamp transform. Only unique, unchanged router-born geometries remain eligible;
compiler replacements and new finisher copper are protected. Empty or remote
same-net zones do not create a fictitious dependency, but actual filled-copper
contact, inherited copper and engineering contracts remain protected. Saved zone
geometry is checked again. Native mutations run in isolated subprocesses.

Required contact/function annotations take precedence over ordinary reference
designators. The existing native reference legalizer keeps visible references on
silk where possible and otherwise preserves them on Fab for assembly. The
silkscreen receipt records those IDs; component/pad/copper geometry is not moved
to make room for text.
Contact annotations try horizontal and orthogonal text axes so rigidly rotated
placement reservations remain usable. Full verified text, minimum type size,
clearance, bounded locality and unique nearest-contact checks are unchanged;
the receipt records the actual text angle.

## Qualification and ownership

Electrical recipes, qualified leaf packages and board acceptance are distinct:

- A recipe defines electrical circuitry; its presence alone is not a physical
  qualification or a reuse hit.
- A qualified package must bind the exact circuit, parameters, interfaces,
  supported operating point, fabrication rules, dependencies and evaluated
  artifacts to independent electrical and physical evidence.
- Unsupported functional claims, missing operating evidence and incomplete local
  connectivity cannot be interpreted as wildcard compatibility.
- Qualification is automated design checking, **not hardware bench validation**.
- Renumbering and permitted rigid transforms may not change connectivity or
  internal relative geometry/copper. Reflection, substitution, layer reassignment
  and rerouting are not implicitly qualified transformations.
- Block qualification does not certify board power budgets, inter-block routes,
  address/pin resources, required labels, connector mechanics or final exports.

In particular, generic nominal capacitors and a regulator's absolute maximum
power figure do not establish effective capacitance, ESR, stability or a thermally
derated load envelope. A successful board check must not manufacture that evidence.

Current physical qualification is deliberately bounded: front-side, two-layer
native footprint/track/via geometry with independently closed local connectivity.
Unsupported keepouts, private copper areas, layer changes or unproved functional
claims produce explicit ineligibility/failure, not a fresh solve labelled as reuse.
Qualification method `@2` includes dangling-track/via warnings; obsolete `@1`
evidence is excluded without modifying the old package.

Independent functional evaluators currently cover the reviewed fixed-5 V USB-C
sink identification network, fully bound 0–5 V passive headers, reviewed
externally biased momentary switches and source-backed low-power indicators.
They do not establish USB PD/data behavior, a header current rating, debounce,
brightness or hardware validation. Other supported electrical recipes may still
solve normally without being eligible for automatic physical publication.
User-authored `.kicad_dru` rules are explicitly ineligible; adding one after
synthesis cannot bypass the rule dependency check.

The temperature conditioner additionally supports an explicit 3.3 V operating
variant for untrimmed MCP6002/MCP9700A channels. Its consumer bound is the actual
3.3 V supply, not a nominal temperature endpoint that ignores component errors.
The existing 5 V variant cannot match a 3.3 V-only consumer. This follows
[Microchip's operating ranges](https://ww1.microchip.com/downloads/en/DeviceDoc/20001060AQ.pdf)
and [temperature-interface reference](https://ww1.microchip.com/downloads/en/AppNotes/21895c.pdf);
it does not establish assembled accuracy or calibration.

The active `rp2040-minimal@5` contract separates logical GPIO availability from
physical breakout geometry. Its default `breakout_style=none` is an integrated
MCU core with complete power, clock, flash and recovery circuitry; it does not
invent a castellated module boundary. Explicit `breakout_style=castellated`
retains the two 17-pad, 2.54 mm banks and their exact board-edge requirements.
Pinned `@4` graphs remain expandable but are not offered for new selection.

The active 3.3 V TPS54331 families use `tps54331-adjustable@4` and
`tps54331-3v3-shunt-controls@3`, with a separate, non-drilled native land bundle.
The land and stencil follow the exact device drawings in
[TI SLVS839H](https://www.ti.com/lit/ds/symlink/tps54331.pdf)
(4206322-5/L and 4208951-5/D), not a generic DDA land or untented
via-in-paste pattern. Existing geometry contracts require an adjacent ground
stitch and connected solid copper on both faces. These CAD continuity minima
are not a measured thermal rating. Old bundles and pinned BOMs are not rewritten;
regenerate architecture/BOM/wiring when their recipe identity is obsolete.

## Storage, discovery and reproduction

The library root is `$KICRAFT_LEAF_LIB`, defaulting to
`~/.kicraft/leaves`. Use an isolated override for experiments; do not clear the
production library.

New packages use `<name>@<version>/`. Versions are immutable. Publication stages
the complete package on the same filesystem, serializes the final rename with
`.publish.lock`, synchronizes payload files and the library directory, and never
replaces an existing identity. Legacy `<name>/` entries remain readable without
being rewritten or silently shadowed by a new copy of the same version.

Readers enumerate published directories on every discovery; dot-prefixed staging
and quarantine directories are not entries. A long-running `LeafLibrary` therefore
sees new publications without a restart. A new process reads the same on-disk
identities. Missing/corrupt packages have explicit reasons; a catalogue access
failure is not reported as an empty library.

Packages carry the existing native PCB, schematic, BOM, solved layout, metadata
and autoplacer fragment. Qualification evidence is included in the content hash.
The project installer retains the remapped round PCB/layout/metadata and the
selected manifest/qualification evidence, then pins the round. Build artifacts
must not depend on a mutable catalogue entry continuing to exist.

Inspect packages without promotion:

```sh
.venv/bin/python -m kicraft.cli.leaf path
.venv/bin/python -m kicraft.cli.leaf list
.venv/bin/python -m kicraft.cli.leaf show '<name>@<version>'
```

A bare name is accepted only when unambiguous. Publication is an ordinary workflow
action, not an operator promotion command. Production service restarts are not
needed for catalogue discovery.

## Reuse/privacy policy

The automatic shared catalogue is an engineering-asset library, not a project
archive or a whole-board result cache. Automatic publication may carry only
independently verified reusable circuitry and necessary sanitized source-reference,
dependency and qualification evidence. Original briefs, user/project identifiers,
secrets, proprietary descriptive text, unrelated parts and unrelated board assets
must not be copied into it. Project-specific bindings and instance provenance
remain in the private project artifacts. A caller-supplied description or an
unqualified legacy package does not authorize automatic cross-project publication.

## Evidence

The implementation record and actual cold/warm/original-brief outcomes are kept in
`plans/automatic-qualified-subcircuit-reuse-2026-11-19.md`. Neither a manifest flag
nor a speedup proves reuse: compare normalized electrical connectivity and native
relative physical geometry/copper, while independently checking the newly composed
board.

## Implementation entrypoints

Paths below are relative to `kicraft/`.

- `design/architecture_intent.py`, `design/lowering.py` and `design/recipes/`:
  complete logical contracts, exact owned circuitry and resource allocation.
- `design/synthesis/sheet_partition.py`, `design/synthesize.py`:
  ownership-preserving leaves and artifact-bound requests.
- `leaf_library/{qualification,loader,extractor,installer}.py`:
  compatibility, independent checks, immutable publication and project snapshots.
- `cli/solve_subcircuits.py`, `cli/_compose_*.py`, `cli/autoexperiment.py`:
  local-solve bypass, protected composition and truthful progress/history.
- `autoplacer/hardware/escape_receipt.py`,
  `autoplacer/brain/{copper_accounting,escape_cleanup}.py`:
  exact native provenance and transactional cleanup.
- `parts_library/footprint_opening.py`, `autoplacer/brain/subcircuit_composer.py`:
  reviewed connector datums and separate mating/packing geometry.
- `autoplacer/hardware/ground_reservations.py`, `autoplacer/kicad_routing_tools.py`:
  atomic ground reservations and explicit routing of unproved ground nets.
- `design/synthesis/{interface_facts,silk_plan}.py`,
  `autoplacer/hardware/_silk_legend_subprocess.py`, `design/cli_app.py`:
  verified physical functions, readable annotations and final delivery gates.
- `parts_library/tps54331-dda-no-vip/`: the versioned ordinary-assembly native land.
- `autoplacer/brain/{signal_escape,breakout_stubs,leaf_routing}.py`:
  source-owned usable interface launches and native landing acceptance.

## Rebuilding a product state

Use the same runtime environment and pinned router configuration as the build
worker. For an isolated library and a **fresh** output directory:

```sh
KICRAFT_LEAF_LIB=/absolute/path/to/isolated-library \
  .venv/bin/python -m kicraft.design.cli_app build \
  /absolute/path/to/.kicraft/state.json /absolute/path/to/fresh-output \
  --no-archive --quality good
```

The input state comes from ordinary product orchestration, not an edited PCB or
manually pinned leaf. There is no seeding or promotion step. Keep the original
brief, committed stage artifacts, build output and native acceptance receipts;
an ineligible block or rejected parent must not be reported as qualified reuse.
