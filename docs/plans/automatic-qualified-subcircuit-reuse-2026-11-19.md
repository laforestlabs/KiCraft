# Automatic qualified subcircuit reuse — implementation handoff

Date: 2026-11-19
Status: implemented; acceptance A–F passed with retained native artifacts. Production activation recorded below.

## 1. Owner's intent — authoritative

The owner identified KiCraft's distinguishing value as decomposing circuit designs into simpler subcircuits whose known-good electrical designs and layouts can be reused across projects without repeated engineering.

The owner explicitly agreed that:

> Every newly solved subcircuit becomes an asset subsequent projects actually consume, rather than merely another successful project on disk.

And clarified:

> Missing blocks that are created need to be automatically made available for future projects.

Implement the complete loop:

1. Interpret the original brief and decompose it into functional blocks.
2. Retrieve compatible, qualified existing blocks and instantiate them.
3. Where a block is missing, design and qualify it automatically during the project workflow.
4. Publish each successfully qualified new block automatically into the durable reusable library.
5. Make that block discoverable and usable by later projects, including after process restart.
6. Compose the selected blocks into a board and verify their integration and delivered artifacts.

No manual promotion, operator curation, or separate user approval should be required for this ordinary learning loop. Qualification failures must remain explicit failures; automatic publication is not permission to publish unqualified designs.

This is not a request for a document-only investigation, another brief-specific patch campaign, a recipe catalogue alone, or a whole-board result cache. The implementation agent must deliver and demonstrate the loop through the real product path.

## 2. Evidence gathered in the strategy session

These observations were read from production files. No provider runs, tests, restarts, or code changes were performed during the investigation. Revalidate relevant integration details before implementation because the checkout may change.

### The three latest Surprise Me projects

Project roots: `/home/kicraft/.kicraft/projects/1/{1186,1187,1188}`. Evidence is in `brief.txt`, `events.jsonl`, `provenance.jsonl`, and the persisted state/build artifacts where present.

| Project | Original brief | Observed failure |
|---|---|---|
| 1186 | A CH32V003 development board with a USB-C 5 V input, a Qwiic/STEMMA QT I2C connector, a 4-pin JST connector, a reset button, and an RGB status LED. Put all the connectors on one edge. | All five design stages passed; synthesis produced an ERC-clean schematic, but the pre-layout physical contract gate rejected unverified required silkscreen functions for SW1 and J1–J4. |
| 1187 | A data logger with eight temperature channels using an RP2040, powered from a 24 V DC screw terminal, with an SPI header and a 4-pin JST connector. Use a two-layer stack-up. | Architecture rejected: `pin-header@1` required declared contact ports, but the requirement declared none. |
| 1188 | A 24 V DC screw terminal to 3.3 V regulator board rated 0.5 A with three push buttons, a power LED, and an enable jumper. Add a reset button. | Architecture rejected: `switch-input@1` signal unbound; enable jumper did not bind to exactly one recipe-published enable port. |

Project 1186 persisted these recipe selections:

- `ch32v003f4u6-minimal@1` (`mcu`)
- `me6211-3v3@2` (`reg`)
- `usb-c-5v-sink@1` (`usb`)

Its BOM had 11 recipe-sourced parts and 17 lowerer-sourced parts. All six sheets had `from_library: null`.

The running web process had no `KICRAFT_LEAF_LIB` override in its environment. Its default directory, `/home/kicraft/.kicraft/leaves`, did not exist. This establishes that these projects did not consume a populated golden-leaf library; it does not establish that all existing import/stamping code is broken.

### Diagnosis, not yet a proven root cause for every failure

There is meaningful deterministic electrical generation already. Do not describe the pipeline as inventing every part and wire anew.

However, deterministic recipes are not equivalent to stored, qualified physical subcircuits. The observed path still has composition-contract failures and no populated golden-layout library. Merely fixing the three diagnostics would not prove the owner's intended reuse architecture.

## 3. Reuse the existing machinery; establish the actual call path

Inspect these existing areas before designing replacements:

| Area | Existing implementation to inspect |
|---|---|
| Versioned electrical definitions, ports, constraints, expansion | `kicraft/design/recipes/models.py`, `registry.py`, `resolver.py`, and individual recipe definitions |
| Architecture contracts and deterministic small circuits | `kicraft/design/architecture_intent.py`, `lowering.py`, `models.py` |
| Live design orchestration and contracts | `kicraft/server/stage_pipeline.py`, `stage_runtime.py`, `stage_prompts.py`, `stage_contracts.py`, `stage_work_units.py`, `reconciliation.py` |
| Leaf package model, loading, extraction, installation | `kicraft/leaf_library/{manifest,loader,extractor,installer,renumber,sexpr_edit}.py` |
| Existing public leaf API | `kicraft/leaf_library/__init__.py`: `LeafLibrary`, `PromoteRequest`, `extract_leaf`, `install_leaf`, manifest and hash helpers |
| Architecture-side legacy leaf context | `kicraft/design/library.py`, relevant callers in `kicraft/design/cli_app.py` |
| Synthesis-side import | `kicraft/design/synthesize.py`, especially `_install_library_sheets`; `kicraft/design/synthesis/autoplacer.py` |
| Physical solve, replication, stamp, compose | `kicraft/cli/solve_subcircuits.py`, `compose_subcircuits.py`, `_leaf_replication.py`, `_compose_stamp.py`, `_compose_validate.py`, `_compose_route.py` |
| Router and PCB integration | `kicraft/autoplacer/kicad_routing_tools.py`, relevant `autoplacer/brain/` code |
| Read-only library CLI | `kicraft/cli/leaf.py` |
| Existing focused coverage | `tests/leaf_library/`, `test_design_recipes.py`, leaf replication and composition tests |

Known details worth checking:

- `RecipeDefinition` includes parts, pins, ports, parameters, placement constraints, electrical assertions and source documents. Its inspected schema did not itself carry a pinned PCB fragment.
- `_format_available_leaves_block` describes optional model-selected sheet reuse and otherwise designing from scratch. An optional prompt hint alone does not satisfy required reuse-first behavior.
- `_install_library_sheets` installs explicitly selected `from_library` sheets through `install_leaf` and emits fragment/library records. Trace those records through the current solver and composer; do not assume their mere existence proves preserved copper in final output.
- The current extractor and installer provide substantial existing infrastructure. Inspect their actual safety, dependency, publication, and version semantics before extending them for unattended use.
- `docs/leaf_library_spec.md` is explicitly superseded. It supplies historical context, not authority for manual-only promotion, replacing versions, or description-only matching.
- Some old roadmap documents describe a prototype without valuable user projects. That is not the current operating context: this is production.

Use available language-server code intelligence where supported. In the strategy session only Ruff was configured, and it did not support references.

Produce a concise current-path map as part of the implementation record: brief → decomposition → resolution → electrical instantiation → local solve/import → compose → verify → automatic publication → next-project retrieval. Identify the specific missing links, then implement them using existing conventions. Do not build a parallel block system beside recipes and leaves.

## 4. Required block contract

Use one coherent contract connecting electrical recipes, physical assets, and qualification. Exact types and storage layout are implementation decisions; the following invariants are not optional.

### Identity and payload

- Stable block identity, immutable version/content identity, and source provenance.
- A declared function and supported operating envelope, not just a descriptive name.
- Complete internal parts, actual symbols/footprints, net connectivity, and required support circuitry.
- Published interface with typed signal/power roles, constraints, required versus optional contacts, and resource requirements.
- Qualified local placement and routing, including relevant copper/vias, keepouts, constraints and boundary geometry. Preserve dependency information for custom assets actually used by production designs.
- Explicit allowed variants, transformations and parameter changes. Changing an electrically or physically significant input outside the qualified envelope creates a new candidate/variant requiring qualification.
- Evidence identifying exactly which artifacts, parameters, dependencies, tool/check versions and conditions were evaluated.

Ref/net renumbering and permitted rigid transforms must not change electrical topology or internal relative geometry. Reflection, layer reassignment, substitutions, rerouting or parameter changes are not implicitly safe transformations.

Do not make whole schematic sheets the mandatory reuse granularity solely because the old importer works at sheet level. Establish electrically meaningful ownership and boundaries that work with the existing hierarchy. Tiny deterministic controls and larger multi-part support networks must not fall through an unowned gap.

### Compatibility and selection

- AI interprets requirements and proposes functional decomposition/topology where judgment is needed.
- Code validates candidate compatibility, instantiates complete published contracts, and resolves mechanical details from structured data.
- Search by function and constraints, not project wording, reference designators, or exact original brief text.
- Evaluate relevant electrical limits, interface semantics, resource availability, footprints, physical constraints, fabrication rules and stack-up. Missing evidence is not a wildcard match.
- Select a compatible qualified entry predictably; record selection and rejection reasons. Repeated ordinary projects must not leave reuse up to an optional model suggestion.
- If the operating point genuinely needs a new variant, qualify that variant; do not mutate the published block.

### Composition

Keep AI from repeatedly specifying boilerplate contacts and support wiring already owned by a block. Code should instantiate complete header/switch/enable interfaces and check their connections.

Validate board-level power budgets, voltage domains, bus addressing/loading, MCU pin allocation, grounding, programming access, mechanical/connector requirements, required labeling and inter-block routing. A qualified block does not guarantee a qualified board.

Maintain a clear boundary between block-internal geometry and board-level interface routes. The composer/router must preserve the former. If a context requires altering qualified internals, invalidate that qualification or create and qualify a new variant rather than report an unchanged reuse hit.

## 5. Automatic creation, qualification and publication

### Missing block

An actual library miss enters automatic creation within the existing project workflow and spend/resource controls. It is not a request for the user to curate a library or run a different tool.

Reuse existing deterministic electrical recipes where applicable. A missing qualified layout for a known recipe is a physical qualification gap, not permission to redesign the electrical circuit unnecessarily.

Design/solve the missing block, establish its boundary contract, and run the applicable qualification checks. Only after qualification may the entry become selectable by ordinary future projects.

### Qualification evidence

Qualification must cover the specific block's claimed use, including:

- Electrical correctness and complete interfaces/support networks, using applicable existing deterministic checks and reference evidence.
- Schematic/netlist-to-PCB connectivity correspondence and required internal connections.
- Local placement, clearance, routing completion and relevant electrical/physical constraints.
- The declared fabrication/stack-up and operating envelope.
- Extraction integrity: schematic, BOM and PCB agree after isolating the block; no external dependency or boundary-crossing copper is silently lost.

ERC/DRC alone do not prove circuit functionality. A provider's approval alone does not prove it either. Use relevant reference-design, calculation or simulation evidence where required, and fail qualification when a claimed function cannot be supported.

Label evidence honestly: automated design qualification is not hardware bench validation. Do not claim measured performance or tested hardware without that evidence.

### Publication and availability

- Publish automatically and durably after successful block qualification, not after an operator clicks Promote.
- Make it discoverable by subsequent projects without restarting services or regenerating a hard-coded catalogue. Verify availability also survives restart.
- Use staged/atomic publication and immutable versions so readers cannot consume partial entries. Concurrent attempts to qualify the same block must not corrupt or overwrite a good published version.
- Persist the selected version and enough self-contained project assets to reproduce builds even if library contents later change.
- Do not expose project/user identifiers, secrets, proprietary descriptive text, or unrelated board assets through the reusable catalogue. Store the reusable engineering asset and necessary sanitized provenance under an explicit reuse/privacy policy. Cross-project reuse must not become cross-user project disclosure.
- Quarantine failed/corrupt/incompatible entries with an actionable reason. Do not silently label a fresh solve as a reuse hit or repeatedly hide a broken catalogue behind an empty-library fallback.

Block qualification and final board acceptance are distinct. A block may remain reusable when an unrelated board-level failure occurs, provided its independent qualification is complete and valid. Conversely, a successful board is not sufficient to auto-publish an arbitrary extracted fragment without its own boundary and qualification evidence.

No universal automation can guarantee arbitrary briefs succeed. On unsupported designs, exhausted existing budgets or failed qualification, report the actual gap. Do not weaken gates, omit requested behavior, substitute easier briefs, or publish candidates merely to keep the pipeline green.

## 6. Delivery sequence — all steps belong to this task

1. Trace the actual live path and identify reuse/automatic-publication gaps. Establish a minimal coherent contract by extending existing mechanisms rather than creating duplicate architectures.
2. Connect deterministic compatibility/resolution and complete interface instantiation to the live Surprise Me and ordinary Design paths.
3. Connect qualified physical import/stamping to the actual local solve and board composition path; protect internals through routing/export.
4. Implement missing-block creation, qualification, automatic durable publication, and subsequent-project discovery.
5. Carry block identity, creation/reuse decisions, qualification and failure reasons through existing state/events and the user-visible progress/history surface. Keep wording precise; do not report published/reused until those actions occurred.
6. Demonstrate cold-library creation and warm-library reuse across different complete projects. Then exercise the three original failure cases without rewriting their briefs, fixing shared composition defects they expose.
7. Document the implemented contract, operational library location, evidence semantics and observed results. Deliver the end-to-end loop, not a plan to finish it later.

Do not spend the session rewriting the placement/router core, adding speculative abstractions, or applying isolated brief regexes instead of fixing ownership and interface contracts.

## 7. Acceptance — observable consumer behavior

### A. Cold-library creation

Start with an isolated empty reusable library, not by deleting production data. Submit a project through the same orchestration used by the product.

Observe a real miss, automatic block creation/qualification and automatic publication. No manual extraction/promotion or preseeded fixture may substitute for these actions. The board must complete its required integration checks and deliver real artifacts.

### B. Different project, same solved subcircuit

Use a genuinely different project that needs the same compatible block; change surrounding circuitry or requirements so this cannot pass as whole-project caching.

Prove that it selects the published identity and reuses the electrical internals and qualified local layout. Observe the local-solving path being bypassed for that block while the new board's composition and inter-block routing still execute.

Compare normalized connectivity and relative physical geometry/copper in the resulting artifacts, accounting only for declared ref/net mapping and allowed transforms. A manifest field, matching screenshot, speedup or log message alone is not sufficient proof.

### C. Durable discovery

Demonstrate a fresh process can discover and use the automatically published entry with the same identity. Confirm the live long-running discovery path also sees new entries without a restart. Use isolated processes/services where possible; do not restart production merely to test durability.

### D. Correct misses and failed candidates

Exercise at least one meaningful out-of-envelope change and one failed qualification. The incompatible asset must not be silently reused; the failed candidate must not enter qualified search results. Successful new variants must become available automatically too.

### E. Multiple instances

Place two instances of the same block on a board. Prove distinct internal references/nets, correct external bindings, preserved internals, and no accidental cross-instance short or geometric corruption.

### F. Existing product failures

Carry projects 1186, 1187 and 1188's exact briefs through the live product-equivalent path to complete, independently checked board artifacts. The three cases are regression targets, not proof by themselves of general cross-project reuse.

- Header contacts must have complete meaningful published bindings.
- Buttons/reset/enable requirements must be satisfied as requested; expose genuine ambiguity rather than delete the control or invent an unconnected substitute.
- Required connector/control labels must follow verified functions into physical output, without disabling the physical contract gate.
- Preserve original requirements such as channel count, current rating, stack-up and connector-edge placement.

### Verification discipline

Use existing focused tests first. Add deterministic regression tests only for plausible consumer-visible gaps not covered already: compatibility boundaries, failed publication, concurrency/atomicity, preservation/renumbering, multi-instance isolation and restart discovery as applicable.

Run the narrow checks that exercise the implemented change and record actual commands/results. Mock-only success is not the end-to-end proof. Use the existing provider budget/spend controls for real-provider scenarios; do not casually launch the entire canary cohort. Observe the actual UI when changing the product surface, and retain machine-readable artifact evidence for electrical/layout preservation.

## 8. Production constraints and final report

This checkout is the production Hetzner host. Follow current repository operating instructions. Do not delete user data, clear caches globally, mutate existing library versions, expose secrets, disable safety gates, or restart services opportunistically.

If deployment is needed, use `deploy/deploy-production.sh`, not systemd. That script performs the canonical web/worker restart and health checks; the 34-brief provider canary is separate and is not an automatic prerequisite for deployment.

Final report must include:

- The implemented path and files changed.
- The shared failure causes repaired, not only the final diagnostics disappearing.
- A compact cold/warm evidence table: project, block identity, created or reused, qualification evidence, published status, local solve performed or bypassed, and final board outcome.
- Evidence of durable automatic availability and preserved internal electrical/physical assets across projects.
- Exact original-brief outcomes, actual verification commands/results, and any genuine unresolved limitations.

Do not call the work complete if future projects still need manual promotion, if only electrical recipes are reused, if the layout is silently regenerated, or if the demonstration never traversed the product's real creation and retrieval path.

## 9. Implementation record

Operational contract: [`../qualified_subcircuit_reuse.md`](../qualified_subcircuit_reuse.md).

This is a chronological engineering record. Earlier “next build” and pending
statements describe that point in the investigation, not the current outcome.
The status at the top and the original-brief artifact outcome table are current.

### Traced path and missing links

`server/stage_pipeline.run_pipeline` and the web session drive the same design
stages. `architecture_intent` compiles functional requirements; recipe resolution,
recipe expansion, lowerers and allocation instantiate deterministic circuitry.
`design/synthesize.run` emits the native schematic/PCB/config and validates it.
The build then drives local `solve_subcircuits`, parent composition/routing and
final acceptance/export.

The former library path depended on optional model-authored `from_library`
selections. Its importer installed pinned physical snapshots, but the ordinary
solver still selected representatives for fresh solving. Extraction replaced
older versions under one name. Neither extraction nor existence in the catalogue
established independent functional qualification, operating compatibility or
automatic publication.

The cutover connects complete electrical ownership to existing hierarchy leaves,
artifact-bound qualification requests at synthesis, deterministic physical
lookup before local solving, independent qualification/publication after a miss,
and subsequent-project discovery through `LeafLibrary`. Parent composition must
still verify its board-level integration and preserve qualified internals.

### Storage and ownership checks

New versions publish to `name@version/`, staged and synchronized before the
locked atomic rename. Existing identities cannot be overwritten or shadowed.
Readers ignore staging/quarantine directories; legacy directories are read-only
inputs. Project snapshot installation retains the mapped triad and source
manifest/qualification evidence before pinning.

Ownership partitioning uses complete recipe manifests and lowerer instances,
not sheet wording. It transfers requirements and recipe-sheet bindings and
rebuilds hierarchy connections. Private nets, arrays and explicit placement
dependencies stay indivisible.

Observed focused checks:

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q tests/leaf_library/test_loader.py tests/leaf_library/test_installer.py tests/test_sheet_partition.py` | 29 passed |
| `.venv/bin/python -m pytest -q tests/leaf_library/test_installer.py tests/leaf_library/test_architecture_validation.py tests/test_sheet_partition.py` | 21 passed after directory durability and automatic-resolution prompt changes |

Text-only tests pinning the obsolete optional catalogue prompt were removed;
electrical interface contract tests remain.

### Isolated product-path runs

Evidence root: `/tmp/kicraft-qualified-reuse.xmHjXM`. Production data was not
deleted or changed to seed the catalogue. Runs use an isolated `library/`,
`run_pipeline`, the production provider profile and spend guard, and the
configured $0.60 per-run ceiling.

Accepted cold/multiple-instance artifacts and their native comparison/discovery/
negative evidence are also retained under
`logs/qualified_reuse_20261119.xmHjXM/` on this host; they are not dependent on
the temporary working directory surviving.

The first cold project committed all five design stages. Its initial build
encountered an in-progress qualifier syntax error before publication. That
attempt is not a successful cold-library demonstration; the deterministic build
is resumed from its committed state after correction, without repeating provider
calls or changing the brief.

Qualification investigation also found that `me6211-3v3@2` does not establish
effective capacitance/ESR or thermally derated performance from its generic
nominal capacitors and absolute-maximum dissipation figure. Those missing facts
must remain explicit, not become an automatically published performance claim.

### Additional observed integration findings

The first stable board build routed all three leaves and the parent with zero
shorts, opens, courtyard overlaps or keepout violations. Final fabrication
acceptance correctly rejected one dangling CC1 via already present in the
USB leaf. The earlier independent DRC had checked errors but not the warning
class used by this final gate. This was a qualification defect, not permission
to weaken board acceptance.

Qualification method `@2` now checks all native severities, rejects dangling
tracks/vias and excludes obsolete `@1` evidence without rewriting those entries.
Fresh local routing cleans only compiler-owned private escape tails with native
connectivity/partition proof before acceptance and publication. The `@1`
publications above are historical observations, not current qualified results.

Additional focused results:

| Command/scenario | Observed result |
|---|---|
| `PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/leaf_library/test_qualification.py tests/test_escape_cleanup.py` | 13 passed, including native dangling-copper regression |
| `.venv/bin/python -m pytest -q tests/test_architecture_intent.py -k power_output_headers` | 6 passed |
| Real accepted local USB snapshot with 29 private copper items removed, passed to `qualify_and_publish` in a fresh native process using method `@2` | Rejected: independent no-plane DRC reported 5 opens and 2 dangling vias; failed library had zero selectable entries |
| 9 V request against the actual published fixed-5 V USB asset | No selection; explicit incompatible operating/dependency/interface/fabrication contract |

Current-method negative-case machine evidence: `negative-current-evidence.json`
under the evidence root. The initial mutation script exposed an in-process native-binding issue; the
fresh qualifier process used by the successful negative check rejected the
electrical defect as intended.

The initial combined integration check had 262 passes and four new test-fixture
failures (wrong recipe sheet role and inconsistent test input rails). Those
fixtures were corrected. Targeted reruns passed; reset ownership also retains
the original functional block rather than replacing it with the reset block.

### Control clarification

The owner clarified the under-specified regulator-only brief through the normal
interactive decision surface:

- Reset disables the 3.3 V regulator while held; release restarts its output.
- The other three buttons provide independently biased 3.3 V signals on a
  dedicated output header.

These are functional constraints, not permission to short EN to its own switched
output or invent an MCU reset pin. The exact original brief remains unchanged in
the evidence workspace; clarification is supplied through stage instructions and
typed intent.

### Accepted cold creation and warm multi-instance reuse

The corrected cold project completed the real deterministic build, including
native verification and the fabrication package, with the isolated `library-v2`
initially empty. Existing failed boards were retained; the successful attempt
uses a fresh output directory, not an overwrite bypass:

```sh
.venv/bin/python -u /tmp/kicraft-qualified-reuse.xmHjXM/build.py cold-v2 generated-qualified
```

| Project | Qualified block | Creation/publication | Local solve | Delivered outcome |
|---|---|---|---|---|
| `cold-v2` | `usb-c-5v-sink@1.0.0+f4827bf87a1c601c29ac0a29e7e17078e96d74f33a17fe5f135b19ea2be842e9` | Automatically published after independent full native DRC | Performed for initial creation; subsequent cohorts consumed the published snapshot | Build rc 0; 0 shorts/opens/courtyard/keepout violations; Gerbers, drill, BOM, CPL, STEP and 3D render exported |
| `multi` (different warm project, two independent inputs) | Same immutable USB identity, twice | Existing independently qualified publication selected in a fresh build process | Bypassed for both USB instances; zero local-router logs for each; parent composition and inter-block routing executed | Build rc 0; 12/12 components, 192 traces, 13 vias; 0 shorts/opens/courtyard/keepout violations; complete fab package |
| `variants` (USB input, 3.3 V LDO and test header) | Same USB identity plus `pin-header@1.0.0+f40ee6e02d7a96d03359e0ce8aa5ed9a10353e7c17216608f1a985a138e01539` | New 3.3 V header independently qualified and automatically published | USB bypassed; header created locally, then reused by subsequent solver processes | Build rc 0; 7/7 components, 140 traces, 9 vias; complete fab package |

Content identity:
`sha256:2b3b5f4d94fec4473d426c1949760efece96a6c5981c6f73ddafda8474a6ecfc`.

The final delivered-board receipt (after cleanup and silkscreen) independently
checked all 73 expected tracks and four expected vias of this instance: zero
missing copper, zero added private copper and no changed footprints. The receipt
binds the delivered PCB hash
`6ce8a33ff5aa199024fe9f88e913f7d48e6abeb5cffe086ddf0311df0a5b8953`.
Location:
`cold-v2/generated-qualified/USB_C_POWER/USB_C_POWER.verification.receipt.json`
under the evidence root. Its fabrication package is
`USB_C_POWER_fab_20261119.zip` in the same directory.

A `LeafLibrary` reader created against the empty directory before this
publication discovered the new identity using the same object, without restart.
Machine evidence: `live-discovery-evidence.json`. A separate
`KICRAFT_LEAF_LIB=/tmp/kicraft-qualified-reuse.xmHjXM/library-v2 .venv/bin/python -m kicraft.cli.leaf list`
process also discovered the published identity and payload hash.

Additional observed checks:

- Native qualification, normalization, obsolete-pin and cleanup regressions:
  `PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/leaf_library/test_qualification.py tests/test_auto_pin_acceptance.py tests/test_escape_cleanup.py`
  — 23 passed.
- Final-delivery custody and ordinary-button ownership:
  `PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/test_promote_freshness_gate.py tests/test_architecture_intent.py -k 'delivery or ordinary_button or regulator_reset'`
  — 9 passed.
- The explicit 3.3 V temperature variant and spare-JST-contact contract:
  `.venv/bin/python -m pytest -q tests/test_analog_conditioner.py tests/test_design_lowering.py::test_fixed_jst_carrier_preserves_declared_wires_and_marks_spare_contact_nc tests/test_design_lowering.py::test_jst_repair_identifies_missing_real_contacts_without_changing_part`
  — 78 passed.

Neither a generic LED without forward-voltage evidence nor a header assigned an
unproved conversion obligation was published. Their ineligibility remains visible
in the build log; local acceptance is deliberately distinct from qualification.

### Native cross-project preservation proof

`native-cross-project-evidence.json` compares the actual delivered cold PCB and
both actual USB instances in the delivered `multi` PCB against the published
native PCB. It does not infer reuse from manifests or elapsed time. Normalization
uses the declared reference/net maps and each connector's rigid coordinate frame.
Pad connectivity, footprint/pad relative geometry, track geometry and both-layer
via diameters/drills are read through the native PCB API.

All three actual instances have equal normalized graph, relative-geometry and
preserved-copper digests. Each contains the same 73 tracks and four vias; no
required copper is absent and no private copper is added. The two warm instances
use disjoint references and private nets, bind VBUS to distinct `VBUS_A`/`VBUS_B`
rails and share only the declared ground. Their final delivered PCB hash is
`99a4157b115f4d8f6bf108b370fb27dafb21c9120169d347bbc15949ebb28f10`.
The accepted package is `multi/generated/DUAL_USB_C/DUAL_USB_C_fab_20261119.zip`.

The separate, larger `warm` experiment was electrically clean and preserved its
qualified block, but correctly failed required header-title placement. It did
not export a fab package and is not counted as an accepted board. The accepted
`multi` project is the genuinely different-project reuse and repeated-instance
demonstration.

### Shared original-brief repairs still under verification

The exact 1186 board initially failed parent composition because the reviewed
JST SH four-contact side-entry footprint lacked a native opening datum.
Orientation is now measured from the manufacturer drawing and registered native
geometry, not guessed from the brief. The targeted four-rotation opening test
passed: `PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/test_screw_terminal_orientation.py -k qwiic_side_entry`
— 4 passed. Final original-brief board acceptance remains pending.

Qwiic power completion now uses only physically bound reviewed peers sharing the
same complete SDA/SCL bus, sourced 3.3 V rail and return. Mixed, isolated or
incomplete domains remain unresolved. Sourceable reviewed passives replace the
unavailable TPS54331 BOMs through new `tps54331-adjustable@3` and
`tps54331-3v3-shunt-controls@2` identities. Persisted old BOMs require regeneration;
published assets and the unrelated 12 V variant are not rewritten.

Focused interface, recipe, supply/current and artifact checks:

```sh
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_architecture_intent.py tests/test_power_converter_recipes.py \
  tests/test_design_recipes.py tests/test_converter_current_evidence.py \
  tests/test_electrical_artifact_evidence.py tests/test_silk_plan_lint.py \
  tests/test_data_logging_storage.py
```

Observed: **835 passed**. This is not a provider or final-board acceptance result.

The shared edge-bank investigation separately found packing envelopes containing
silkscreen reservation lanes being used as physical mating/copper support lines.
Those are now distinct: packing space is retained without shifting the actual
connector bank inboard. Exact board-normal connectors are not assigned an
in-plane mouth direction from generic pad asymmetry. The Qwiic housing mouth is
not an authored PCB cut; its solder tabs still require copper-safe board support.
Existing edge-gap thresholds remain unchanged.

```sh
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_screw_terminal_orientation.py tests/test_subcircuit_composer.py \
  tests/test_connector_edge_gap.py
```

Observed: **113 passed, 3 skipped**.

Both original 1187 and clarified 1188 now commit architecture, BOM and wiring
through the real provider/orchestration path. Cumulative recorded provider spend
is $0.091950 and $0.209552 respectively, below the unchanged $0.60 project cap.
1186's original five-stage run committed at $0.010493. Board builds remain the
acceptance requirement; committed stages alone are not counted as completion.

### First sourceable regulator-board build

1188 reached clean synthesis, but its regulator leaf was correctly rejected
before routing: four imported pad-9 drills penetrate the exposed pad's continuous
assembly opening. Sourceable MPNs alone do not make that land pattern
manufacturable under the ordinary assembly profile. The mask/paste guard is not
being waived; a source-backed, non-VIP TPS54331 land pattern and actual adjacent
ground path are required.

A separate button-leaf rejection exposed receipt quantization, not lost copper.
All 12 generated signal-escape tracks/vias were retained at exact native
coordinates; seven recorded planned-float coordinates differed by one nanometre
from KiCad's actual setter readback. Ownership receipts must capture native
geometry at both leaf and parent stamping. The exact matching/uniqueness and
unresolved-ownership gates remain necessary; broad numeric tolerance or ignoring
missing provenance is not the repair.

### Accepted new operating-point variant

The `variants` project's successful attempt uses
`variants/generated-qualified-2/USB_3V3_LDO/`. Its delivered PCB hash is
`76db8414caa5c24164c2162fbbdbf77f27d46bcc4f07710ed88e67e30938b696`;
the complete package is `USB_3V3_LDO_fab_20261119.zip`.

The existing header entry has a 5 V/GND operating point. The new entry binds
3.3 V/GND and has content hash
`sha256:8d8287108decf8ac9c0cfb3671e37cf803b33781fcf6421bcebfc126748736ed`.
It was published by the ordinary first local solve and consumed from the library
by subsequent solver processes. This is a conservative exact-operating-point
variant of the passive-contact contract, not a new current-rating claim.

The same `LeafLibrary` object created before cold publication now discovers all
three entries with no broken packages. A separate `leaf show` process discovers
the new identity and identical payload hash. Machine record:
`variant-discovery-evidence.json`. Native comparison was expanded to the
`variants` delivered PCB: its USB block preserves the same 77 copper items, and
the header's native pad connectivity/relative geometry matches its own published
source. A single passive header has no internal tracks/vias to preserve.

Further focused results:

- Exact native escape provenance, publication and pin acceptance:
  `PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/test_breakout_stubs.py tests/test_escape_cleanup.py tests/leaf_library/test_qualification.py tests/test_auto_pin_acceptance.py`
  — 70 passed.
- Updated sourceable native regulator and interface contracts:
  `PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/test_architecture_intent.py tests/test_power_converter_recipes.py tests/test_design_recipes.py tests/test_converter_current_evidence.py tests/test_electrical_artifact_evidence.py tests/test_silk_plan_lint.py`
  — 829 passed.
- Final multi-branch cleanup and unchanged local-copper contracts:
  `PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/test_escape_cleanup.py tests/test_local_copper_contract.py`
  — 77 passed.

The native cleanup now proves all planned removals before mutation, applies one
batch and reopens the saved candidate in a fresh native process. Exact surviving
copper and affected connectivity partitions must match; full native DRC must
still approve before atomic commit. Required labels retain their physical
clearance, minimum type size and nearest-contact checks; constrained contacts
are packed before flexible titles, and interior contacts before exterior ones.

Later native consumer checks found two integration details not covered by recipe
unit tests: the parent route accepted only 2 oz thermal contracts and then
unconditionally promoted thermal boards to 2 oz. Both paths now honor the
declared 1/2 oz process; final manufacturing floors still follow actual native
copper weight. The focused one-ounce dual-ground-path case and existing
two-ounce narrow-track rejection both passed.

The final remaining development-board annotation slot was occupied only by
ordinary reference text, not physical copper/courtyards. Required functions now
precede the existing reference legalizer, which preserves references on Fab
when they cannot remain on silk. Native annotation replay placed all 17 required
labels with no missing entries; no component/pad/copper motion was used.
`tests/test_silk_legend_placer.py tests/test_silk_refdes.py` — 20 passed.
Full original-brief builds remain the acceptance gate.

### Original-brief development-board acceptance

1186 completed `generated-qualified-4/CH32V003_DEV_BOARD` through the ordinary
build command: rc 0, 26/26 components, 519 traces and 58 vias, zero shorts,
opens, courtyard or keepout violations, no missing required annotations and no
same-edge violations. Its delivered PCB hash is
`755844d1d7cb0c653347d1d731c00feea1566b73fc115841ac363b229e3b7a0c`.
The exported package is `CH32V003_DEV_BOARD_fab_20261119.zip`; artifacts and the
build log are retained under `logs/qualified_reuse_20261119.xmHjXM/`.

Native contacts confirm USB 5 V input, sourced 3.3 V Qwiic SDA/SCL, JST pins
1/2 = 3.3 V/GND and pins 3/4 = the two actual MCU GPIO signals, RGB drive and
the reviewed NRST-to-ground reset switch. All 17 required function/contact
annotations are placed. STEP export reports one absent assembly body, U1;
the package is exported and fabrication-accepted, but this is not complete
3D-body coverage or hardware bench validation.

1188's next full board was electrically/mechanically clean but correctly failed
one full-length `BUTTON_3` contact annotation. Native measurement found no
horizontal slot and 193 valid orthogonal-text candidates under the unchanged
font, clearance, outline, locality and nearest-contact rules. An isolated
annotation experiment placed every required label using one 90-degree text
rotation, with no physical geometry change. That diagnostic is not a delivered
board acceptance result. The shared placer now tries both text axes under those
same physical rules and records the actual angle. Native replay through the real
renderer placed all 14 required labels plus both legend lines, with no drops.
`tests/test_silk_legend_placer.py tests/test_silk_refdes.py` — **21 passed**.

1187's first full build failed: the MCU leaf was not accepted and both parent
rounds could not construct a complete native-filled GND witness forest. No fab
package was exported. The MCU architecture had also introduced a 30-GPIO fanout
not requested by either the original brief or committed functional spec; only
GPIO0–5 carry the actual SPI/I2C interfaces. Normal provider regeneration now
selects the supported six-GPIO exposure while preserving every original
obligation, all eight channels, storage and SWD. Native MCU and ground-witness
failures remain under investigation; neither gates nor budgets are relaxed.

The six-GPIO regeneration committed architecture, BOM and wiring through the
real provider path (architecture cost $0.020154, three attempts). The prior
30-GPIO fanout was an implementation default, not a user obligation.

Native investigation also distinguished two genuine pinned-router failures
from compiler errors: one width pass introduced a SWCLK/RUN short; another
retracted an existing 3.3 V castellation segment despite `--keep-input-copper`.
The existing DRC and exact copper-custody gates rejected both. No installed
KiCadRoutingTools files, tolerance or fabrication gates were changed.

The parent GND failure was not a false native-connectivity rejection: two
actual disconnected filled-copper clusters remained. The global reservation
stage incorrectly aborted before the existing explicit-ground routing path
could handle an unproved net. The correction must keep incomplete nets on that
worklist, avoid partial reservation masks and retain final native connectivity
checks; it must not label disconnected fill as a complete ground plane.

The ground staging correction is implemented. A typed incomplete-forest result
leaves GND on the existing explicit routing worklist; malformed geometry and
required local-return failures still fail closed. Forests are planned completely
before masks are stamped. A native raw-fill upper-bound cut check rejects
impossible forests early but cannot qualify a successful witness.

Focused native reservation and real-router checks passed:

```sh
KICRAFT_KICAD_ROUTING_TOOLS_PATH=/home/kicraft/KiCadRoutingTools-f6fa93122db4-usb \
KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/home/kicraft/krt-venv/bin/python \
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_local_copper_contract.py tests/test_kicad_routing_tools.py \
  -k 'ground_forest or ground_witness or ground_connectivity_requires_actual_plane_proof_or_routing or ground_reservations_ignore'
```

Observed: **13 passed**, including actual explicit routing across split fill.
The initial check exposed a missing search-rejection-set initialization, fixed
before this result. Its native cases also exposed a concurrent, externally made
router-pin/configuration update: source now expects `f6fa93122db463ff3f3a1617605c69f7959d2cd9`
and production `.env` selects the matching checkout above. Those changes are
preserved; the tests and next full builds use that actual configured runtime,
not a reverted pin or a bypassed preflight. Earlier accepted reuse/1186 artifacts
retain their original runtime provenance.

Fresh `generated-qualified-5` full builds for 1188 and 1187 are now the remaining
acceptance gate, under the unchanged quality, provider cap and build deadline.

### Final original-brief artifact outcomes

| Exact brief | Build output | Components | Traces / vias | Missing required labels | Outcome |
|---|---|---:|---:|---:|---|
| 1186 | `generated-qualified-4/CH32V003_DEV_BOARD` | 26 | 519 / 58 | 0 | rc 0; fab package exported |
| 1187 | `generated-qualified-14/TEMPERATURE_DATA_LOGGER` | 90 | 2370 / 269 | 0 | rc 0; fab package exported |
| 1188 | `generated-qualified-5/3V3_REGULATOR` | 31 | 437 / 46 | 0 | rc 0; fab package exported |

1188 has zero shorts, opens, courtyard and keepout violations; all 14 required
function/contact labels are placed, and every expected STEP assembly reference
is present. Its delivered PCB hash is
`cf81574b6ff7f0609547085c975c22531d6fbc70f3335109e0a1b018d3afc0c9`.
The package is `3V3_REGULATOR_fab_20261119.zip`. The committed contract retains
24 V input, 3.3 V/0.5 A conversion, independent regulator reset and disable
jumper, power LED and three separately biased button outputs plus 3.3 V/GND on
their dedicated header. The accepted board and build log are retained under
`logs/qualified_reuse_20261119.xmHjXM/1188/` and its evidence-root sibling log.
The actual exported 3D surface was inspected; this is still CAD, not bench proof.

Independent native verification of both accepted original boards additionally
matched every declared BOM connection and unused contact to actual delivered
PCB pads, checked the actual two-layer boards and receipt hashes, retained the
exact brief bytes, and verified fabrication ZIP CRCs and required members.
Machine evidence: `original-brief-native-evidence-1186-1188.json` in the retained
evidence root. Cumulative recorded provider spend remains below the unchanged
$0.60 cap for each project. This verification did not treat a log success line
or a missing library-preservation field as electrical proof.

1187's next full build accepted all ten leaves and selected a routed parent:
zero shorts, opens, courtyard or keepout violations, with all required annotations
placed. Final acceptance still correctly failed. The RP2040 recipe unconditionally
emitted 34 board-edge castellation pads, even for an integrated logger with only
six allocated GPIOs; composition placed those board-fabricated terminals inside
the parent. Neither the brief nor its committed functional specification requests
a castellated module. This requires an explicit versioned integrated-core versus
castellated-breakout contract, not removal of the edge gate.

The other failure is one native 0.0422 mm dangling feedback track on
`__KICRAFT_ANALOG__conditioner__FB_3`, UUID
`a51d4a0b-285a-44ef-a93f-47f0921e2df0`. Its birth/ownership must be resolved at the
producer or originating router boundary; final-board stripping of unknown
inherited copper is not an acceptable repair. The failed board remains retained.

The versioned boundary correction is implemented as `rp2040-minimal@5`, default
`breakout_style=none`; explicit `castellated` mode retains both exact banks.
Integrated mode removes only those duplicate board-fabricated terminals and
their edge constraints, not allocatable MCU pins, power/clock/flash, SWD,
RESET or BOOTSEL. Pinned `@4` expansion retains its original graph.

The feedback fragment was created by the temperature leaf's second KRT width
pass, not by parent stamping. Both of its graph endpoints had incidental
finite-width overlaps, so an endpoint-degree prefilter incorrectly prevented
the existing native-DRC/partition proof from considering it. Router-delta cleanup
now uses that proof at the originating stage, plans read-only and verifies
saved-board readback after batched removal. Inherited, protected and zoned copper
remain excluded. Native DRC removal of the actual fragment cleared only its
warning and preserved all 69 retained anchors.

Focused verification:

```sh
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_design_recipes.py tests/test_touch_pin_allocation.py \
  tests/test_silk_plan_lint.py tests/test_castellated_interface.py \
  -k 'rp2040 or castellat or usb_contract_proves_reviewed_chain or real_pin_header or recipe_expansion_is_deterministic'
# 41 passed
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_escape_cleanup.py
# 12 passed
```

The normal provider chain is regenerating the logger with the supported
integrated-core parameter. No persisted state or failed PCB was hand-edited.

Integrated-core regeneration committed all three requested stages through the
provider ($0.023062, four architecture attempts), with the explicit supported
parameter and 90 actual parts. The next full build accepted all ten leaves;
the MCU core has 21 parts rather than forcing 34 additional board-edge pads.
All three parent attempts nevertheless remained genuinely unconnected: the
last has open SPI MOSI and CS. Native routing diagnostics identify preserved
SPI/private-QSPI/power copper around the MCU endpoints as blockers. Neither
larger placement candidates nor the existing native repair found a legal path.

The next investigation is therefore the shared physical interface-access
contract and any reservation interference, not another unmodified search run
or permission to rip inherited internals. `generated-qualified-6` is retained
as rejected evidence, not a delivered board.

Native interface tracing found the physical gap: the compiler's dense MCU
dogbones and vias survived composition exactly, but some normal-width parent
traces cannot pass between adjacent preserved escape vias. Local DRC and a valid
via do not establish a usable published outlet. The source fanout must reserve
the actual parent trace/clearance envelope before private copper is solved;
relabeling anchors or permitting parent rip-up would not repair that contract.

Independent native mask analysis confirmed that the complete GND witness forest
matches the staged masks exactly. Removing ground/assembly restrictions from
the in-memory hard-obstacle map did not restore either SPI path. Omitting only
new parent I2C SDA copper did, with all restrictions retained: that route seals
the shared SPI mouth. No PCB was written by these probes. Ground masks,
fabrication policy and inherited-copper protection remain unchanged.

The source correction now reserves spaced interface vias and normal-width
opposite-face launches before private routing. Dogbone, via and launch are one
atomic stamp; the native interface audit rejects a connected via without its
required launch. All inherited-copper, hole, mask, outline and keepout guards
remain unchanged. Native RP2040 regressions verify concurrent usable outlets,
the old crossing's conflict with preserved launch copper, missing-launch
rejection and blocked-launch atomicity.

The module check
`PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/test_signal_escape.py tests/test_breakout_stubs.py`
passed 77 cases. Its new blocked-launch fixture initially blocked the via too;
the fixture was corrected to clear the via and intersect only the launch, and
the exact consumer check
`PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q tests/test_signal_escape.py::test_rp2040_opposite_face_crossing_blocks_entire_launch`
then passed. No gate or expected diagnostic was re-pinned to hide the failure.

The unchanged committed 90-component logger is undergoing a fresh
`generated-qualified-7` full build with these source-owned launches.

The first full launch-enabled run exposed a serialization defect in the new
native-audit boundary: the solver stores `power_nets` as a set, and JSON rejected
it before the native audit ran. All seven signal-bearing leaves were therefore
falsely marked unlanded. The boundary now normalizes that typed collection to a
sorted list; no audit exception is suppressed.

The native RP2040 regression now exercises the actual audit subprocess with
the solver's set-valued configuration and passed. Read-only re-audits of all
seven retained real leaves also reported no missing landings. No failed artifact
was promoted or modified by those checks. The unchanged logger is now in a
fresh `generated-qualified-8` full build.

The next complete search selected three physically routed parents: zero shorts,
opens, courtyard and keepout violations. Delivery still correctly rejected the
selected board because cleanup was not closed over the new launch dependencies.
It tried vias before unused opposite-face launches, so retiring a launch later
created a newly single-face via and the full-DRC transaction rolled back.
Owned branches now retire in their emitted terminal-to-pad order; used
multilayer vias and all remaining connectivity stay protected.

Native refill also exposed a router-born VIN24 terminal fragment that was not
reported at the early router-output boundary. The same original-stamp-scoped
cleanup now runs after finishing. Native DRC authorizes each step of a finite,
strictly copper-removing terminal cascade; fresh saved-board readback, protected
net exclusions and the final unchanged-except-retired DRC gate remain required.

Focused cleanup and parent ground-custody checks:

```sh
KICRAFT_KICAD_ROUTING_TOOLS_PATH=/home/kicraft/KiCadRoutingTools-f6fa93122db4-usb \
KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/home/kicraft/krt-venv/bin/python \
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_escape_cleanup.py tests/test_parent_ground_custody.py
```

Observed: **20 passed**. A scratch replay of the actual selected logger retired
two new routing-transaction items and 52 owned escape groups with fresh native
readback, zero opens and zero dangling warnings. Machine record:
`cleanup-complete-proof/summary.json`. That replay is explicitly not a delivered
board; a fresh `generated-qualified-9` full build is the remaining acceptance gate.

### Leaf cleanup handoff after generation 9

The complete generation-9 logger selected a 90-component parent with 2,122
traces and zero shorts, opens, courtyard or keepout violations. Parent escape
cleanup passed fresh native readback. Delivery still correctly rejected one
0.7071 mm private bootstrap track; no fabrication package was exported.

The same warning exists in the selected regulator leaf. Native UUID comparison
proves the fragment was absent from that round's actual router input and present
in its selected output. The recorded leaf cleanup retired an unused compiler
fanout via before import, exposing the router-born tail after the early spur
cleanup had already run. This is not authorization for parent-side removal of
inherited private copper.

The leaf now closes that handoff before import. Its subsequent router-only
cleanup accepts already-retired input escapes only when the preceding receipt
has successful native readback and binds both the exact input hash and current
output hash. All original input UUIDs remain protected from deletion. Contracted,
zoned and qualified-private net exclusions, retained-copper partitions, fresh
native saved-board readback and unchanged-except-retired full DRC remain in force.
The selected source input/output, early cleanup, exact four retired bootstrap
escape items and native dangling report are retained under `bootstrap-proof/`
in the evidence root; the complete rejected generation-9 project is retained too.

The focused cleanup/ground-custody invocation above initially observed 21 passes
and one new fixture failure. The fixture was corrected to retain a real pad at
the surviving local path endpoint, rather than inadvertently creating an
inherited dangling endpoint. The two new handoff/stale-receipt cases then passed:

```sh
KICRAFT_KICAD_ROUTING_TOOLS_PATH=/home/kicraft/KiCadRoutingTools-f6fa93122db4-usb \
KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/home/kicraft/krt-venv/bin/python \
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_escape_cleanup.py -k 'owned-cleanup'
```

Observed: **2 passed, 16 deselected**. A fresh complete
`generated-qualified-10` logger build is running under the unchanged product
profile, quality, provider cap and build deadline; these unit results alone do
not satisfy the outstanding original-brief acceptance gate.

### Source exit capacity and pad-anchor correction after generation 10

Generation 10 accepted all ten leaves, but all three parents retained four
I²C/SPI opens. Native round-3 checkpoints were 37 → 20 → 4 → 4 opens; generation
9's corresponding sequence was 37 → 20 → 0 → 0. All six signals were in both
worklists and their source launches survived composition.

Native source-stage comparison isolated the closure: the GND-only leaf pass
added two 0.50 mm B.Cu tracks, enclosing the six public exits against private
QSPI/DVDD copper. The parent could individually escape through a narrow F.Cu
neck, but newly routed MOSI/CS copper consumed that shared mouth. Nominal
0.1 mm-grid/0.20 mm-track/0.60 mm-via probes with all original masks reproduced
the before/after failure. Removing only new signal-pass copper in memory restored
reachability; removing rule-area masks alone did not. A diagnostic `mps` ordering
also retained four opens, so no global ordering or mask waiver was introduced.
Records: `1187/native-source-port-diagnosis.json` and `parent-port-proof/`.

The source now requests existing native both-face ground proof for stamped dense
interfaces on an enabled two-layer ground-plane design. A complete 22-pad,
21-mask forest preserved more than 7.5 mm of normal-width B.Cu reach from all six
exits. Only complete proof removes GND from the explicit worklist; incomplete
proof routes it normally. The checked reservation plan is translated with the
source frame and protects ground from the higher-priority power pour. Final
filled source ground connectivity is independently required again.

The earlier bootstrap diagnosis also needed correction: the 0.7071 mm routed
diagonal is **load-bearing**, not removable dead copper. It grazes the pad corner.
Removing the entire compiler pad-anchor segment leaves that foreign route's
endpoint dangling, despite preserving finite copper connectivity. Cleanup now
retains an original compiler segment when a foreign endpoint lands on its
interior outside the actual owner pad; unused outer segments and the single-face
via may still retire.

A disposable reconstruction of the selected native leaf reversed its checked
rigid translation and restored the four authentic original compiler items.
Fresh cleanup retained the pad anchor and routed diagonal, retired only the
other three compiler items, and passed saved-board readback plus complete native
DRC with zero violations and opens. Record:
`bootstrap-replay-proof/native-anchor-summary.json`. This is explicitly a
reconstructed source-stage diagnostic, **not a delivered project**.

Focused checks after both shared fixes:

```sh
KICRAFT_KICAD_ROUTING_TOOLS_PATH=/home/kicraft/KiCadRoutingTools-f6fa93122db4-usb \
KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/home/kicraft/krt-venv/bin/python \
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_escape_cleanup.py tests/test_parent_ground_custody.py \
  tests/test_kicad_routing_tools.py \
  -k 'not test_kicad_routing_tools or ground_connectivity_requires_actual_plane_proof_or_routing or ground_last_uses_native_selection_and_preserves_class_widths'
```

Observed: **30 passed, 64 deselected**, including actual native pad-anchor
preservation and ordinary-plane complete/incomplete routing cases. A fresh
`generated-qualified-11` full logger build is running under the same product
profile, quality and caps. No scratch board has been promoted.

### Generation-11 product result

The fresh complete build now routed all three parent candidates successfully;
round 2 was selected. The delivered inspection board has 90/90 components,
1,997 traces, 244 vias, zero shorts/opens/courtyard/keepout violations and all 21
required labels (23 total function/contact annotations) placed. The actual
selected MCU leaf carries the complete 22-pad/21-mask GND forest, and final
native plane-ground audits are complete for the MCU and both ADC leaves.
Record: `1187/source-plane-native-evidence-11.json`.

The unchanged final gate still correctly refused fabrication export for six
dangling-copper warnings: two compiler-owned QSPI/SWD items and four +3V3
items. Parent cleanup passed native readback and retired 45 owned groups; no
warning was waived. The complete rejected project and log are retained as
`1187/generated-qualified-11/` and `1187-qualified-11-build.log`.
These remaining items are being traced at their originating source/rail
producers; functional parent routing alone does not satisfy acceptance F.

### Native source join and rail closure

The two signal warnings were load-bearing joins, not disposable dead copper.
The QSPI terminal extended past a router endpoint landing 3.088 µm off its
interior centerline; the SWD via's same-face disk bridged track caps that did not
otherwise meet. Fresh local source generation now trims the former to the exact
landing and replaces the latter with a same-width track. The operation is
private/source-owned only, preserves unrelated copper and retained partitions,
and requires full native DRC to clear exactly the targeted warnings. The native
selected-source proof preserved 718 unrelated items and zero opens:
`signal-join-proof/native-signal-summary.json`.

Three rail warnings and their exposed terminal tail were born in the MCU's
native router output, before silk or pour finishing. The fourth warning was
born in the parent router. Actual same-net filled copper was either empty or
remote; the pour finisher did not create these items. Router-only cleanup now
tests the complete native copper envelope against same-layer filled polygons,
including tangencies, instead of protecting an entire net merely because it has
a zone definition. Same-net saved zone outlines/fill and all retained copper/
pad partitions remain unchanged; current, thermal, USB, qualified-private and
inherited exclusions remain in force.

The leaf captures raw copper against the first actual native router input,
filters surviving identities after compiler work, then rebinds unique exact
native geometry through recentering. Coordinates and via/track dimensions use
the actual stamper's `FromMM(ToMM(value) + delta)` conversion, not integer-delta
approximation. Post-fill cleanup's protection fence includes every non-router
item, including newly created finisher copper. This is a provenance fence, not
a fabricated design result or a parent-side deletion of inherited leaf copper.
The native mutation/readback paths are isolated subprocesses.

The focused module invocation was:

```sh
KICRAFT_KICAD_ROUTING_TOOLS_PATH=/home/kicraft/KiCadRoutingTools-f6fa93122db4-usb \
KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/home/kicraft/krt-venv/bin/python \
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_escape_cleanup.py tests/test_signal_escape.py
```

The first attempt crashed in an in-process native fill fixture. Fill and new
mutating helpers now use the existing isolated native runner. The next attempt
observed **62 passed, 3 failed**: one fixture did not actually produce empty
fill, and two wrote fine-via fabrication settings before native Save reset them.
The empty-fill fixture now uses a real pour-blocking rule area; fabrication
settings are stamped after Save. The three corrected native cases then passed.
An additional exact-restamp test pair passed (**2 passed, 23 deselected**),
covering fresh track/via identities and preservation of newly added finisher
copper. These are focused-run results, not a claim of one green combined rerun.
A throwaway four-layer native scenario also confirmed explicit ineligibility
for the new two-layer frame proof, rather than changing ordinary four-layer
routing behavior.

A fresh `generated-qualified-12` full product build is running with unchanged
profile, quality and caps. Acceptance F still requires its final export gate.

### Generation-12 boundary result

Generation 12 selected a functional parent with 90/90 components, 2,359 traces,
262 vias, zero shorts/opens/courtyard/keepout violations and all 21 required
labels. Other search rounds that changed inherited input copper were correctly
rejected; their output was not promoted. Final export still refused one
`TEMP_B2` via warning.

That via is a load-bearing, compiler-owned public analogue join, retained by the
deletion-only cleanup's connectivity proof. It is not private block circuitry.
The composer now records its graph-derived interconnect net names and applicable
electrical/fab constraints in the exact routed ownership ledger. Final integration
may apply the same proved join canonicalization only to those declared boundary
nets, after checking ownership/cleanup/board hash continuity. Missing boundary
authority is deny-by-default; no imported private or qualified copper is granted
new authority.

A disposable copy of the actual selected parent, with the new ledger fields
derived from its recorded composition state, passed ordinary cleanup plus
boundary canonicalization: exactly the remaining via was replaced, full native
DRC had zero violations and opens. Record: `boundary-join-proof/summary.json`.
This diagnostic was not promoted or counted as fabrication acceptance.

Focused native join checks:

```sh
KICRAFT_KICAD_ROUTING_TOOLS_PATH=/home/kicraft/KiCadRoutingTools-f6fa93122db4-usb \
KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/home/kicraft/krt-venv/bin/python \
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_signal_escape.py -k source_owned_landing_canonicalization
```

Observed **10 passed, 35 deselected**, including front/back private, declared
boundary and unauthorized cases. The missing-authority variant was then checked
separately: **2 passed, 43 deselected**, with the actual PCB bytes unchanged.
The complete generation-12 project/log remain retained. A fresh
`generated-qualified-13` product build is running with unchanged profile,
quality and caps; final fabrication acceptance is still the outstanding gate.

### Generation-13 native layer custody and parent access

Generation 13 accepted all ten leaves, but all three parent rounds changed
inherited track layers. The preservation gate rejected every parent (rc 6);
there is no selected final board or fabrication package for this attempt.
Its complete project and build log are retained.

Native input/output comparison identified the same UUID
`c3b4fa62-2794-4d2a-938d-648ce53a0919`, width 0.2 mm, on `I2C_SDA`:
the backend moved its existing B.Cu launch to F.Cu, without changing endpoints.
`--keep-input-copper` protects against rip-up, not this relayering. The adapter
now also requests `--no-stub-layer-swap`; an isolated replay of the actual
failed input nevertheless exposed a phase-three rescue path in the pinned
backend that ignores that option. The replay was correctly rejected too.
Neither output was promoted and no custody check was relaxed.

The source access mismatch was a 0.36 mm escape-via row presented to a parent
using its normal 0.6 mm via class. Public launch planning now reserves the
resolved normal netclass via envelope as well as normal track width, while
retaining the fabrication-profile escape vias. Private-only fanout is unchanged.
This gives parent attachment room rather than relying on mutation of inherited
fanout. The external checkout and the user's router pin are unchanged.

Observed checks:

```sh
KICRAFT_KICAD_ROUTING_TOOLS_PATH=/home/kicraft/KiCadRoutingTools-f6fa93122db4-usb \
KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/home/kicraft/krt-venv/bin/python \
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_kicad_routing_tools.py::test_native_route_preserves_inherited_opposite_face_launch

PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH .venv/bin/python -m pytest -q \
  tests/test_signal_escape.py::test_dense_interface_ports_admit_parent_via_envelope \
  tests/test_signal_escape.py::test_rp2040_dense_singletons_reserve_concurrent_native_launches \
  tests/test_signal_escape.py::test_rp2040_opposite_face_crossing_blocks_entire_launch
```

Results: **1 passed** and **3 passed**, respectively. The first actually routes
with the pinned native backend and checks inherited B.Cu plus native closure.
The second batch checks native clearance with normal-class via envelopes,
concurrent source launches and atomic rejection of an obstructed launch.
A fresh `generated-qualified-14` build is running through the same product CLI
with unchanged quality, provider state and caps; fabrication acceptance remains
outstanding.

### Final acceptance and retained delivery

Generation 14 completed through the ordinary product build CLI (quality `good`,
unchanged provider state/caps). All ten leaves and all three parent candidates
were accepted; round 1 was selected. Its input copper survived the native passes.
Final cleanup, declared-boundary join canonicalization, silkscreen, independent
native verification and fabrication export all passed without promoting a
diagnostic PCB or invoking a private export tail.

```sh
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH \
  .venv/bin/python /tmp/kicraft-qualified-reuse.xmHjXM/build.py \
  1187 generated-qualified-14
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH \
  .venv/bin/python /tmp/kicraft-qualified-reuse.xmHjXM/verify_originals.py 1187
```

Observed: build **rc 0**, 1,320.95 seconds; native artifact checks passed.
The build harness called the public command
`python -m kicraft.design.cli_app build .kicraft/state.json generated-qualified-14 --no-archive --quality good`
under the production runtime configuration and isolated library. The independent
checker verified the exact original brief, every one of 283 declared BOM
endpoints, 31 declared NC pins, two native copper layers, all 21 required labels,
hash-bound acceptance receipt, ZIP CRC and required fabrication members.
It also checked eight MCP9700 sensors, four MCP6002 conditioners, two ADS1115s
with distinct GND/VDD address straps and eight distinct ADC inputs, RP2040,
EEPROM and no incidental castellated breakout. Cumulative provider spend was
**$0.206435125**, below the unchanged $0.60 project cap.

Final logger PCB SHA-256:
`dbc93a0d4f76fb12d4fec02918dcbc1ca2c10659045b69e8e0327ec7ca3ac9ab`.
Package `TEMPERATURE_DATA_LOGGER_fab_20261119.zip`, SHA-256:
`dd13871601920a04ea7c04dfe7c04956c2d5d723602116766985c7b6ba384cd5`.
The actual exported render was inspected. Known limitations: U7
(`K7803M-1000R3`) has no STEP body; board utilization is 14.3% on an approximately
94 × 101 mm outline, reported as a nonblocking placement warning. Neither
limitation was hidden or waived.

All A–F acceptance cases now have observed consumer-level evidence:

- A: empty-library cold creation, independent qualification and automatic
  publication, canonical self-installation, completed final board.
- B/E: a different warm board installs the same USB asset twice, bypasses both
  local solves, preserves actual native internal graph/geometry/copper, isolates
  references/private nets and passes parent routing/export.
- C: fresh CLI/build processes and the already-live discovery object see the
  immutable publications without manually refreshing or restarting a service.
- D: a 9 V request misses the fixed-5 V asset; physically broken USB copper
  fails independent qualification and stays out of search. A newly qualified
  3.3 V header variant publishes automatically and is reused by later processes.
- F: all three exact original briefs have rc-0 complete artifacts, native graph
  checks and fabrication packages. The current outcome table above is final.

Retained private evidence root: `logs/qualified_reuse_20261119.xmHjXM/`
(mode 700). Accepted outputs are `cold-v2/`, `multi/`, `variants/`,
`1186/generated-qualified-4/`, `1187/generated-qualified-14/` and
`1188/generated-qualified-5/`; original product states/provider records and build
logs accompany them. `library-v2/` retains the immutable published assets.
`original-brief-native-evidence-all.json` consolidates all three checked briefs;
`native-cross-project-evidence.json`, `live-discovery-evidence.json`,
`variant-discovery-evidence.json` and `negative-current-evidence.json` record
reuse/availability/rejection proofs.
Rejected intermediate boards and native diagnostic evidence remain clearly
separate. Throwaway build/check scripts were removed after use; the public
reproduction command is documented in `../qualified_subcircuit_reuse.md`.

Qualification remains automated, bounded CAD/design evidence, not hardware bench
validation. Independently reviewed automatic functional eligibility currently
covers fixed-5 V USB-C sinks, 0–5 V passive headers, externally biased SPST
controls and bounded low-power LEDs; other families report ineligible and take
ordinary solving, not a fabricated qualification. The development board also
retains its disclosed missing CH32V003 STEP body. No claim of measured regulator
current/thermal performance or calibrated temperature accuracy is made.

### Production activation

After acceptance and throwaway-script cleanup, ran exactly:

```sh
./deploy/deploy-production.sh
```

Observed **exit 0**: web restarted on port 8080 (PID 2377472), build worker
restarted (PID 2377513), and the canonical script reported
`production health passed: web and newly started build worker are ready`.
No dependency changes were needed. No catalogue seeding, service restart for
discovery proof, global cache clearing, existing-version mutation, budget
increase or 34-brief provider canary was performed.
