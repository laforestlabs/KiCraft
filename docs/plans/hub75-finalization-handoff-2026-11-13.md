# HUB75 board finalization handoff — 2026-11-13

## Recorded verdict — November 15–16

**Project 1161 source corrections were implemented; no corrected fabrication
package was accepted.** The November 15 cold `good`/seed-0 cohort ran the older
strict-approval path. All three tails rejected missing fabricator-approved
90 Ω geometry for the retained two-layer, 2 oz USB stackup before placement.
That verdict belongs to those pinned inputs and source, not to a replay with
today's default engineering-review policy. See the [USB routing integration
notes](../kicad-routing-tools-experiment.md) for the draft/fabrication
distinction and the separate November 17 project 1167 replay, which still did
not select a routed parent. No source correction, review copy, or partial
replay here is evidence of manufacturing or hardware qualification.

Review copies published2026-11-16, without starting builds or changing project1161:

- [Project1164 — corrected full schematics and BOM](https://kicraft.io/admin/projects/view?project=1164). The109-component PCB is explicitly the unplaced/unrouted input, not a completed corrected layout.
- [Project1165 — latest placed, labeled programming-USB section](https://kicraft.io/admin/projects/view?project=1165). Four-component leaf only, with its corresponding schematic; not the full board.

Both records retain `status=failed`, `quality=unverified` and no fabrication
ZIP, with their review scope stated prominently in the visible brief. Admin
sign-in is required. Eleven published schematic/PCB files were fetched through
live HTTPS and matched their local SHA256 hashes; both live KiCanvas PCB
viewers loaded without browser errors. No admin login credentials were
available to this verification browser, so the authenticated admin page itself
was not exercised. Publication evidence is retained in
`board1161-ee-20261115-2GqCJ1/published-review-projects.json` and
`publication-viewer-check.json`.

|Evidence boundary|Established result|Still missing|
|---|---|---|
|Source29 physical cohort|Three recorded qualified cold tails: replays74,75,76|These are not source30 qualifications|
|Source29 GUI project1158 / KC-Q7U775|Database status `ok`; ZIP exists|Electrically rejected: independent HUB75 channels merged; no accepted download|
|Source30 planner|All five stages committed; compiled proof has 14 independent display/control nets on 14 MCU pads|Physical build and real GUI delivery were not exercised by this probe|
|Source30 deployment|Commit/push/deploy recorded complete; HTTP200 and worker-ready recorded|Zero qualified full cold tails and no new source30 GUI project in retained/current account45 evidence|

Deployment record: `22b36af7b09ae36506d8c553724d29ab099a4b49`, branch `simplify/bom-wiring-pipeline`, completed `2026-11-13T03:16:11.725297+00:00`. Deployment before full qualification was explicitly requested in that earlier campaign. It is not a manufacturing approval.

The November13 deployment evidence is historical. The November15 sections below record source changes and copied-workspace verification, not production deployment or hardware measurements.

### Project1161 review-evidence correction — 2026-11-15

The electrical-review input now uses a compact electrical projection with a
96,000-character ceiling and typed mandatory-coverage accounting. The caller
does not invoke a provider with incomplete evidence. Coverage, provider/schema
outcomes and review findings are separate durable facts; missing historical
coverage remains unknown. Source-stage replacement invalidates the old review.
Connector/control silk evidence is projected independently.

Read-only projection of project1161 retained all109 parts, all103 connection
records (74 distinct net names), explicit NCs and resolved endpoint functions in
63,150 characters. Its connector projection retained J1–J5 and both switches in
31,508 characters. No provider was called and this is not a completed electrical
review of the circuit.

Verification: 109 tests passed across electrical-review, gate, lifecycle,
web-assurance and silk-lint tests. A temporary NiceGUI instance rendered the real
review inspector's complete/no-findings, incomplete and historical/unknown
outcomes, including reload. Production services and project1161 were untouched.
USB routing, physical silk, presentation and new-build qualification remain
separate acceptance boundaries.

### Project1161 USB physical-contract correction — 2026-11-15

Reviewed native USB and USB bridge recipes now publish impedance, skew and
transition targets. Synthesis validates exact recipe-owned roles/pins and emits
ordered connector/ESD/series/MCU pair segments. Canonical private names use
paired `_DP`/`_DM` suffixes. Old generated BOMs must be re-expanded upstream;
renaming only board copper is not supported.

The pair owner excludes its nets from independent dog-bone escapes, generic
single-ended routing and generic copper repair. Local proximity constraints and
cross-leaf placement affinity preserve the chain. The pinned differential
backend routes first with preserved-input-copper checks; partial, swapped or
single-ended fallback results are rejected.

Promoted-board receipts bind native path lengths, layer transitions,
paired-span/reference-plane proof and grounded-return evidence to the PCB hash.
Incomplete paths have unknown totals/skew, not misleading partial sums.
Read-only examination of the old main interconnect reports DP137.699742 mm,
DM101.729547 mm, skew35.970195 mm and16/12 transitions; it is rejected.

Provider-free normalization of saved1161 retained all109 exact part identities,
all pin identities and every non-USB net assignment while changing only private
USB pair names. The first affected regression run passed531 tests; four fixture
failures were corrected (configured router path and unavailable native count
API). The corrected backend/native/generic-repair subset passed57 tests.

The two-layer,2oz stackup remains unchanged and **unqualified for controlled
impedance**: no fabricator approval is on file. JLCPCB's
[calculator guide](https://jlcpcb.com/help/article/user-guide-to-the-jlcpcb-impedance-calculator)
excludes2oz external copper because its trace tolerances hinder impedance
control. Fabrication acceptance requires supplied geometry bound to the actual
stackup and checked at 90 Ω ±10%; calculated draft widths are not approval.
No successor board or full-tail qualification is claimed here.

### Project1161 usability and release correction — 2026-11-15

Required labels now come from canonical recipe expansion, actual owned pin
assignments and explicit operating assumptions, independently of optional model
wording. They cover the five connector functions, BOOT/RESET, HUB75 pin1,
floating speaker polarity and LED supply/data/return. Numeric facts belong to
their interface; unrelated prose cannot authorize a rating. The0.25 A LED
allocation is explicitly assumed, and0.087 W speaker output is an unmeasured
low-frequency circuit estimate, not a2 W output promise.

Compiler-owned text reservations use KiCad glyph measurements and persist in
component envelopes through rigid composition. Pre-layout checks reject missing
reservations. Post-route required-placement receipts, not authored label counts,
gate delivery. Required text cannot be discarded as cosmetic.

The operating sheet includes separate panel power/common ground, separate board
power for programming USB, BOOT/RESET sequence, exact pin/function/net mappings,
unverified hardware behavior and observed missing STEP references. Sourcing
commits populate canonical catalog IDs even when stock is unverified; identity
disagreement is rejected. Export consumes that selection without substituting.
Release archives select one promoted PCB/project, include hash-bound operating
and fabrication data and physical receipts, and reject intermediate/stale boards.

Verification: the initial affected run passed616 tests with four fixture/wording
failures; corrected content, physical-placer and release subsets passed50 tests.
The regenerated in-memory1161 design provides all seven reviewed interfaces and
ten required text/pinout blocks, including the LED allocation and speaker
estimate. Applying labels to a **copy of the old, unreserved board** placed all
seven role/control blocks but failed four required pin labels (HUB75 pin1,
both speaker pins and LED DATA). Those named delivery failures are not waived;
the successor must resolve them through its new placement reservations.

Original evidence is preserved in
`/home/kicraft/.kicraft/debug/board1161-ee-20261115-2GqCJ1/original-snapshot`;
the original project's656 file hashes are recorded alongside it. No corrected
layout, downloaded successor package or hardware qualification is claimed yet.


### Project1161 materialization and cleanup evidence — 2026-11-15

Canonical BOM and wiring commits on the isolated copy, followed by synthesis,
retained109 parts and all355 wired endpoints. Native synthesis checks report
zero ERC errors and netlist faithfulness. Ten endpoint assignments changed only
the four private USB pair names; public USB and every other net stayed intact.

The strict sourcing gate exposed L1/L2's old shared15uH bundle identity
C177243 conflicting with the selected4.7uH FXL0630-4R7-M/C167220. The producer now
uses an exact4.7uH bundle; selected MPN/value/catalog code and native land geometry
are unchanged. The [manufacturer FXL family specification](https://xonstorage.z8.web.core.windows.net/pdf/changjiangmicroelectronics_fxl10501r0m_apr22_xonlink.pdf)
provides shared FXL0630 dimensions on p4 and4.7uH characteristics on p10.
No15uH substitution or manifest-disagreement waiver was made. Canonical commits
still disclose stale catalog data, unavailable CH224K/PAM8302A assembly stock and
unverified AP63205/MCU/HUB75 stock.

Routed-round selection now uses measured whole-board area, native via count and
copper length; only accepted custody-bound routed parents compete. Dominated
measured candidates cannot beat the frontier by packing score. Missing
measurements stay unknown/incomparable, not predicted savings. A native probe of
the old copied board measured8839.250273 mm²,496 vias and4608.445929 mm of copper.
This is a baseline, not evidence of successor improvement.

Compiler escape ownership now survives import, serialization, rigid transforms,
stamping and selected-parent custody. Cleanup requires the exact selected routed
PCB hash, stamped provenance, complete unambiguous branch ownership, no
unmodelled same-net zone, and unchanged physical-pad/remaining-anchor
connectivity partitions. Whole-branch removal requires single-pad dead-end
contacts. Via-only removal also handles a used escape stub whose via has
contacts on only one copper layer; genuine layer-changing vias remain.
Removed native UUIDs and before/after hashes form an authorized transition,
without rewriting the historical router preservation ledger. Subsequent
final-silk and verification receipts bind the final board hash. Ten
cleanup/ownership/ground-custody tests passed, including native unused-branch
and via-only removal, retention of load-bearing copper and unrelated ground,
stale/ambiguous ownership rejection and board-specific dangling proof.

Final dangling audit consumes complete native DRC JSON with all severities and
all track errors, not the bounded UI violation list. A real DRC of the old-board
silk preview found61 unproven dangling entries and correctly rejected it.
Ground-finisher regression injection now identifies a pre-existing UUID rather
than accidentally selecting a newly added thermal via by iteration order;
destructive removal of existing ground copper remains rejected.

Initial viewer inspection caught remaining MCU text/body and power-caption
collisions; those were fixed in the producer rather than waived. The final
candidate uses actual symbol-graphics bounds and cached/batched KiCad stroke-font
measurements, wraps capacitor banks by converter/rail, reserves visible power
values in horizontal lanes and wraps the root into rows. Nineteen drawing
regressions passed. Native SVG glyph bounds, including stroke, have no positive
different-text intersections above0.1px across root, MCU and POWER
(`native-svg-all-text-proof-candidate4.json`). Synthesis still reports zero ERC
errors; native PCB inspection matches every one of355 wired endpoints and
retains14 distinct MCU-side HUB75 nets. The canonical review projection is
complete at65,704 characters, without a provider call.

Native interface proof confirms seven measured label envelopes survive
serialization and90° rigid transformation, and U1/U3/U7 plus L1/L2 export their
selected catalog IDs. Release packaging now requires affirmative, hash-bound
silk/cleanup/dangling/verification receipts; failed or unknown fabrication
verdicts cannot be archived. Eleven release-contract tests passed. The viewed
native3D seed is an **unrouted component grid**, not a qualified layout.
No routed-board or fabrication acceptance follows from these source checks.

### Project1161 frozen copied-tail cohort — 2026-11-15

Three byte-identical cold copies of the final materialized input are staged
under `board1161-ee-20261115-2GqCJ1/cold-tail-{1,2,3}`. The state+output replay
form enables fabrication export if all gates pass; quality is `good`, seed0,
without providers, archive writes, retries added by this correction or changes
to project1161. Runs are sequential to avoid multiplying production-box load.

`frozen-custody.json` records1391 runtime-source files,684 router files, four
native dependency files, installed distribution versions and the private
environment file's hash only. Its SHA256 is
`adc17558bff3056e096251119afa7c928124666b98de24901debf938b965f90a`.
Native/application/router/environment custody is checked around the cohort.
Canonical preflight observes router commit
`8eeee55fd4ba4ed600b11bd8214de9994e8f5396`, source0.20.2/native0.20.1, the
configured KRT venv and KiCad9.0.9. Stage semantics remains the production
`observe` setting; the new USB/silk/cleanup/dangling physical gates are
unconditional. This initial cohort completed two tails, both rc6 with no final
receipts or fabrication qualification. Its third tail was cancelled after the
first closed failure exposed source defects. `aborted-cohort.json` records the
two terminal results and unchanged application/router/native/environment
custody at cancellation; interrupted work is not a third completed verdict.
The initial cohort remains preserved, not overwritten.

The closed witness separates two causes. MCU placement reached routing but
retried missing USB fabrication approval12 times. Programming USB rejected its
owner before ESD placement: the configured connector inset was1.8876mm while
the locality reservation incorrectly required the ordinary2mm edge margin.
After aligning that check with the existing assigned-edge policy, the measured
title envelope itself obstructed the connector's required3mm ESD approach.
The adapter now reserves that title beside the coupled pad row, and edge
packing accounts for the resulting asymmetric physical-envelope center.
Pad geometry,3mm locality, ordinary component margins and copper rules were
not relaxed.

The actual programming-USB leaf subsequently completed placement with the
same canonical input and3mm contract. Four native regressions exercise the
title/coupling/assigned-inset geometry at all four board edges. The targeted
suite reported119 passes and two opt-in skips; two native backend cases
initially selected the old checkout without the environment loader, then both
passed under `deploy/run-with-env.sh` and the configured pinned backend.

A replacement byte-identical three-tail cohort is frozen under
`cohort-final/`, with1391 application files,684 router files and the same four
native dependencies. Its manifest SHA256 is
`c15fed7ae764918fc24ddd4f3ab7507fb7ffe5902a7fa738bf4820348d2d7ab3`.
Missing fabrication approval now rejects before placement/routing search and
persists failed artifact status rather than consuming repeated routing rounds.
All three final `good`/seed0 cold tails completed with rc7:
`USB controlled impedance requires fabricator approval`. Their persisted
artifact statuses are `failed`, fabrication ZIPs are null and final physical
receipts are absent. Each actual-project `artifacts` inspection completed with
rc0 but found no routed, placed or leaf output; that rc0 is inspection success,
not manufacturing acceptance. The first tail's recorded elapsed time includes
waiting for the separately cancelled probe's build slot; the other tails
finished in1.148s and0.988s.

Application/router/native/environment custody remained unchanged across the
final cohort. `cohort-final/cold-cohort-results.json` and
`qualification-verdict.json` retain the per-run records and explicit negative
qualification. Full-board USB geometry, final DRC/thermal/antenna/custody,
all-interface placed silk, and improvement over496 vias/4.608m without area
growth remain **unproved**, not passed. Hardware qualification is also unrun.
No corrected fabrication package was produced.

The isolated placement proof measures actual native J4.A6→U9.3 and
J4.A7→U9.1 pad centers at2.163331mm each, below the unchanged3mm limit.
The required programming/external-board-power title physically places at0.9mm
text height. The actual KiCanvas PCB and native3D render were inspected;
the USB connector model is resolved from the copied project's model directory.
This is a four-component, **unrouted leaf diagnostic**, not the successor board.
Its full native DRC still records15 unconnected items, five dangling vias and
three intrinsic J4-footprint silk warnings; none is waived or represented as a
fabrication pass. Evidence: `placement-proof/native-usb-placement-proof.json`,
`usb-placement.drc.json` and `usb-placement-3d-with-model.png`.

Whole-board preview limits were checked rather than hidden. An extra
`good --no-route --no-fab` probe was cancelled after its CLI explicitly
reported that only the fast engine honors `--no-route`; good still routes.
The separately copied supported `fast --no-route --no-fab` probe then
completed with rc6. It found all seven leaf artifacts, but parent composition
refused POWER INPUT, POWER, MCU, DISPLAY and PROGRAMMING USB because they had
no accepted routed artifacts. No composition guard was bypassed.
`fast-placement-proof/inspection-verdict.json` records that refusal; it is
not a replacement for the three frozen good-quality results.

Actual visual proof therefore covers the regenerated seed PCB/3D and
root/MCU/POWER drawings, the corrected programming-USB leaf PCB/3D, and the
electrical-review inspector's initial/reopened complete/incomplete states.
There is no corrected full-parent PCB/3D or accepted download to inspect.
The temporary viewer and probe scripts were removed after inspection.

Original-project custody check: `original-custody-final.json` compares
all656 files against the pre-change snapshot and reports unchanged, with zero
added, removed or modified files. Replay inputs/outputs are isolated debug
copies, not project1161.

### Project1161 retained choices and hardware qualification protocol

The first correction retains the exact N16R8 module, independent HUB75 channels,
9V CH224K/TPS26631 protection topology, both AP6320x rails, conservative capacitor
banks and HCT245 LED driver. External panel power/common ground, the assumed0.25A
LED allocation and an8Ω floating speaker remain explicit choices. Neither the
generic32V capacitor provisioning nor one-channel use of an eight-bit buffer is
silently cost-reduced.

Before a9V-specific capacitor profile, calculate effective capacitance from the
selected parts' DC-bias, tolerance and temperature data; check converter loop
stability and RMS ripple against the actual protected input envelope, including
protection tolerances/transients. A buffer replacement needs reviewed input
thresholds, output drive and power-off/backfeed behavior. Mounting geometry
requires an actual enclosure/panel drawing. Bring-up review must establish
accessible, labeled rail/GND test access; no dedicated test-point addition or
bare-pad validation is claimed here.

All assembled-hardware work below is **unrun**. Record board/revision, BOM
substitutions, firmware hash, setup/cable/load identities, calibrated instrument
models, ambient conditions, waveforms and raw results. Pass limits must come from
the committed requirements and selected-component specifications; a CAD pass is
not a measurement.

|Qualification|Setup and evidence required|
|---|---|
|PD/startup/fallback|PD analyzer, representative9V sources and scope; capture negotiated PDO/current, cold/hot startup,5V fallback behavior, protection/enable sequence and rail overshoot against device limits. Do not assume fallback supports normal operation.|
|Concurrent power/thermal|Current-limited supply, dynamic loads, scope and thermocouples/IR with emissivity recorded; exercise intended simultaneous MCU/radio/display logic/LED/audio loads, rail droop/ripple/transients and component temperatures. Panel power stays external.|
|USB programming|Representative hosts/cables, USB analyzer or scope as appropriate; record enumeration, BOOT/RESET entry, repeated programming/reset and disconnect/power-order behavior. Obtain actual fabricator-controlled stackup/geometry and inspect return continuity before accepting the routed pair.|
|Actual HUB75 panel|Identify exact panel, scan arrangement and firmware; verify pin1/orientation, common ground, safe OE startup, all colors/rows, test patterns, refresh, brightness and ghosting at intended conditions.|
|LED output|Identify string/controller and cable, scope data at the receiver and measure current; verify logic levels/timing and power sequencing within the explicitly assumed0.25A allocation. Higher-current strip power needs a separate reviewed specification.|
|Floating audio|8Ω noninductive load with adequate dissipation, differential probe/audio analyzer; never ground either BTL output. Record differential power, clipping/THD, noise and startup/shutdown pop. Approximately0.087W is an unmeasured low-frequency estimate, not2W delivered.|
|RF/mechanics/test access|Test the actual antenna/enclosure/cable environment; record radio performance against a stated reference and inspect connector access, mounting and safe labeled rail/GND probing. Missing mechanical requirements are decisions to obtain, not invented failures.|
|Shield/ESD/EMC|Make an explicit J4 shield-to-chassis/ground decision first; define applicable product requirements and test levels, then perform corresponding ESD and conducted/radiated tests. The current shell-only net and CAD clearance are not an EMC qualification.|

## Locations and custody

All shell examples below use working directory `/home/kicraft/KiCraft`.

```bash
S=/home/kicraft/.kicraft/debug/hub75-gui-20261106T042027Z
C="$S/continuation-20261110T172718Z"
F="$S/finalization-20261113"
```

`F` is a proposed **new** evidence directory for the next campaign, not an existing successful run. If occupied, use a fresh timestamped sibling and record that absolute path. Preserve all existing witnesses.

Read these retained records first:

- `C/source30-candidate-status.json`: recorded stop verdict and deployment; its storage figures are stale.
- `C/production-cutover-source30-user-requested.json`: commit/push/deployment evidence.
- `C/source30-planning-compiled-contracts.json`: actual compiled channel, converter and floating-speaker proof.
- `C/source30-real-provider-five-stage-1/result.json` and its `workspace/.kicraft/state.json`: retained complete new planning seed.
- `C/gui-source29-project1158-electrical-identity-failure.json`: why a green GUI result is not acceptable.
- `C/qualified-full-tails.json`: source29 qualification cohort; do not relabel it source30.
- `S/physical-replay-76/fabrication-qualification.json` and `finite-ground-qualification.json`: retained source29 physical witness.
- `S/physical-replay-59/.kicraft/state.json`: older complete design seed for historical comparisons, not the preferred fresh source30 seed.

### What changed since the recorded stop

Read-only filesystem observation during handoff preparation: **45,855,981,568 bytes available**. The earlier 5,290,917,888-byte storage shortage is no longer established. Keep the runner's **5 GiB minimum free reserve**, estimate writes before launching, and recheck live free space; do not delete evidence to solve the obsolete shortage.

These recorded paths are now absent:

- `S/synthesis-probe-12`
- `C/frozen-tail-source-30`
- `C/isolated-kicraft-integration`

Do not follow the old instruction to copy synthesis-probe12 or assume a recorded source archive still exists. Old records can retain paths to removed evidence; missing bytes are not proof of a failed build, but prevent rechecking those exact bytes.

The source30 manifest `C/source30-planner-probe-source.json` contains 2,241 entries and has SHA256:

```text
e95a1c49351dc19b05c99940cba00da4df5e7ded616444394b1bdf601c667bb8
```

The comparison made during that handoff was limited to files in the historical
manifest; it found documentation drift and did not enumerate added paths.
The old manifest is lineage, not proof of current tree identity.

The historical November 13/15 runs used router commit
`8eeee55fd4ba4ed600b11bd8214de9994e8f5396` (Python source `0.20.2`, native
`0.20.1`). `C/router-frozen-tails.json` and its SHA256
`af2b402946db145b67be69ff2dfca7ce4cfb3016c2dd5e332e478d2fa659b16a`
describe that historical router and file hashes, not the effective runtime for
future runs. Do not restore it over the active router. Freeze the actual
configured revision, patched file set, interpreter and native binary together
with the application source and input hashes for each new qualification cohort.

## Literal requirement and electrical acceptance

Submit this exact brief through the production GUI, with **Auto default questions? checked**:

> make a board with USB C PD power input to an ESP32-S3-WROOM-1-N16R8 that drives a 64x32 p2.5 HUB75 display. Include header outputs for driving addressable LED string and a speaker.

No added hints, substituted MCU, hand-edited committed state, or CLI-only replacement for GUI acceptance.

Retain the reviewed design requirements from the earlier campaign and verify them on the newly generated state and native artifacts:

- Exact ESP32-S3-WROOM-1-N16R8; reserved module pads28/29/30 unused. Native programming USB uses GPIO19/20 through its series resistors; programming VBUS remains separate from board/PD power.
- Fixed9 V CH224K PD input, TPS26631 protected path, AP63205 5 V and AP63203 3.3 V supplies, actual support components and ratings. Source30 proof puts both converter VIN pins on `PD_PROTECTED` and enable pins on `PD_PATH_POWER_GOOD`, not raw `PD_SOURCE`.
- Thirteen independently buffered HUB75 signals **plus a separate enable control**: `r0,g0,b0,r1,g1,b1,addr_a,addr_b,addr_c,addr_d,clk,lat,oe,enable`. These require14 distinct MCU-side nets/pads. Thirteen buffered channels and fourteen MCU controls are not contradictory counts. Safe OE/startup, pulldowns, decoupling, and connector endpoints must survive synthesis and routing.
- LED output: real3-pin header, 5 V buffering and series resistor. Default external string allocation0.25 A; do not turn an unspecified load into unlimited power.
- PAM8302A BTL output: real floating2-pin speaker header, neither speaker terminal tied to GND. Default8 ohm speaker rated at least2 W; this does not promise2 W delivered audio. Preserve the PWM/filter, shutdown, decoupling and short floating-lead constraints.
- Panel LEDs use an external5 V supply by default. Source30 assumptions allocate9 V/2 A external PD capability,0.97 A aggregate5 V loads including0.6 A speaker reserve, and1 A3.3 V capacity. Check the compiled protection/current/thermal contracts; assumptions are not measured panel consumption or thermal qualification.
- Two-layer1.600 mm stack:0.070 mm copper per outer layer,0.010 mm masks,1.440 mm dielectric. Track/space floor0.170 mm, component PTH annulus at least0.254 mm, drilled-hole edge spacing at least0.200 mm. Keep local-current/ground/thermal gates and RF keepouts. Required via-in-pad instructions must specify resin fill, capping and planarization.

### The last electrical failure and its fix boundary

Project1158 merged `HUB75_R0/G0/B0/ENABLE` onto one MCU control and `HUB75_R1/G1/B1` onto another. DRC and a completed ZIP did not establish the intended channel identities. It remains rejected even though its database status is `ok`.

Source30 changed `kicraft/design/architecture_intent.py`, `kicraft/server/stage_prompts.py`, `.agents/skills/kicraft/stages/architecture.md`, and `tests/test_architecture_intent.py`: independent named MCU signals must not silently share a source port; intentional fan-out is one signal with multiple destinations. Reuse this compiler/prompt contract rather than patching1158's nets.

The retained source30 planning state SHA256 was rechecked:

```text
364e1682682352af6afbff894e158e02e41604950c7c0992e36c4e8b7e9485ac
```

It has stem `ESP32_HUB75_CONTROLLER`,109 parts in the compiled proof, all five stage statuses true, and wiring complete. Its evidence maps14 channels to14 MCU pads, both converters to the protected/power-good nets, and J3 to distinct floating speaker nets. This is one successful real-provider planning-chain smoke, with deterministic recipe compilation for downstream work—not five separately provider-authored stages, three planner successes, or a physical result.

Recorded checks:275 architecture/prompt checks and5 focused production-source checks passed. They were not rerun for this handoff and do not replace board qualification. If this compiler behavior is changed during resumption, exercise its existing regressions in `tests/test_architecture_intent.py`, then repeat the affected real pipeline proof before qualifying the changed source.

## Resume in dependency order

### 1. Establish a new frozen campaign without undoing user changes

Read the matching `kicraft`, `kicraft-investigate`, and `verify` skills; read `kicraft-debug` before stage debugging. Inspect current production work, disk and budget headroom before competing for resources. Preserve quotas, spend accounting and uncertain reservations; do not reset them. The recorded probe requested a$0.25 client budget while its compiled proof reports a$0.60 project guard; inspect effective limits rather than treating either historical number as a new spending authorization.

Create `F` and a new full maintained-source hash inventory, including current docs and this handoff. Retain the old source30 manifest separately as lineage, not as the new manifest. Verify and copy the current router manifest. Do not restore missing unrelated docs to satisfy an old hash. Do not edit source while a qualification tail runs. A later application/router change starts a new qualification cohort; never mix its passes with old ones.

The source30 deployment record applies to November 13 only. Do not redeploy or
restart merely to reproduce it. If a new deployment is authorized and needed,
finish narrow verification and drain active production work before following
the production deploy procedure; verify local/public HTTP and worker readiness.

### 2. Synthesize a new cold seed from retained source30 planning

Copy the complete `C/source30-real-provider-five-stage-1/workspace` to a new `F/seed` without altering the original. Recheck its source state hash and fourteen-channel/power/speaker identities first. Do not copy the rejected1158 state or a routed/partial PCB as the seed. Synthesize with current frozen source and the production environment:

```bash
./deploy/run-with-env.sh python -m kicraft.design.cli_app synthesize \
  "$F/seed/.kicraft/state.json" "$F/seed/generated" --no-archive
```

Require successful synthesis/ERC and complete KiCad project/rule/library sidecars. Preserve all emitted files and `.kicraft/state.json`; a bare PCB copy is invalid. Inventory the complete fresh seed with the existing `run_tail.py:inventory` and `map_fingerprint` semantics: absolute `seed_root`, per-file SHA256 `files`, and canonical-map `fingerprint`. It must have no `.experiments` or previous tail markers. This synthesis is required because the preferred source30 planning input was never physically built; do not spend on new architecture calls merely to recover a deleted synthesized workspace.

If the retained state fails current schema/recipe validation, preserve the failure and correct its owning stage via supported stage tooling, regenerating invalidated downstream stages. Do not hand-edit state or transplant accepted JSON/PCB geometry around a gate.

### 3. Rebind existing helpers, then complete three cold tails

Copy `C/run_tail.py`, `verify_tail.py`, `qualify_finite_ground.py`, and `verify_gui_delivery.py` into `F`. These are operational evidence helpers, not new production abstractions. Retain originals unchanged. Copy `C/live-status-source12-full34-started.json` for the GUI helper's historical-account exclusion, and retain `S/physical_replay.py` in its current parent location.

In `F`, write fresh `source-frozen-tails.json`, `router-frozen-tails.json`, `cold-tail-inputs.json`, and `tail-runs.json`. The source manifest must describe the actual final source, not source29; the input manifest must describe `F/seed`, not absent synthesis-probe12. Map indices `1`,`2`,`3` to three new full seed copies `F/tail-1`,`F/tail-2`,`F/tail-3`; each row has absolute `copy` and `input_manifest` paths. Never reuse the old map to replay74/75/76.

In the copied `run_tail.py`, use its existing `root/.kicraft/state.json` and
`root/generated` positional arguments instead of `--project`. Use strict
fabrication mode for qualification, not the automatic draft path, and confirm
stackup-bound approval first. Preserve all source/router/pathset/native/cold-input/
exclusive-output/5 GiB checks. If the historical helper cannot pass the strict
flag through, update the copied helper before running it. The effective call is:

```bash
./deploy/run-with-env.sh python "$S/physical_replay.py" \
  "$F/tail-1/.kicraft/state.json" "$F/tail-1/generated" \
  --quality good --seed 0 --no-archive --require-fabrication-approval
```

Use the runner rather than this bare call for recorded acceptance. Execute serially, substituting indices2 and3 and their corresponding paths after1 succeeds:

```bash
./deploy/run-with-env.sh python "$F/run_tail.py" 1 --preflight-only
./deploy/run-with-env.sh python "$F/run_tail.py" 1
./deploy/run-with-env.sh python "$F/verify_tail.py" "$F/tail-1"
./deploy/run-with-env.sh python "$F/qualify_finite_ground.py" "$F/tail-1"
```

Use a named/background job without an arbitrary short outer diagnostic deadline; native pipeline budgets still apply. Retain every failed attempt. A preflight failure means resolve custody/environment/input admission, not weaken the check. A routing/qualification failure means inspect the actual native witness, fix the producing generic source and earliest detector, then start a new frozen cohort; do not manually route the witness, loosen fabrication gates or merely raise timeouts.

Only aggregate three completed passing `fabrication-qualification.json` records into **`F/qualified-full-tails.json`**, with their matching finite-ground records. Require unchanged application/router/input custody, fresh selected/routed/promoted provenance, all native ERC/DRC/current/thermal/ground/manufacturing gates, and actual fab archive receipts. Do not count an interrupted run, a clean leaf, source29's three passes, or a no-fab replay.

Historical reference: replay76 recorded43 local contracts,19 ground-return contracts and3 thermal owners; board SHA256 `571d42f148ecca38ee54a8b09f8795927d1a36b91a24f558a8d67b142c18b476`. Its full qualification belongs to source29 manifest `1206c4e118c98ff06d98e808ed6263305aad5dfc4c7e5ee010bd8fdd88068d97`. The old U2.6/C31.1/leaf-timeout failures are not the current leading blocker merely because the old handoff listed them; reopen only if new witnesses reproduce them.

### 4. Audit electrical meaning and disclosure before GUI acceptance

For every candidate used for acceptance, verify channel identities through actual compiled wiring and native schematic/PCB endpoints, not counts alone. Validate all fourteen distinct MCU nets/pads, each corresponding HUB75 input/enable endpoint, the thirteen translated output paths, safe startup, protected converter feed/enable, exact MCU/programming path, LED header and floating BTL speaker output. Reconcile current allocations, component ratings, startup and thermal evidence against the emitted board; ERC/DRC alone cannot reject the1158 defect class.

Keep provenance and procurement limitations visible. Replay76's qualification includes `unreviewed_exact_part` and missing STEP references J2/U9; current source30 architecture also contains exact-part advisories for interface identities. Review actual BOM identities/support/provenance against the reviewed implementations and original brief. An unresolved material electrical/identity mismatch blocks acceptance. Missing STEP geometry or unverified stock must be disclosed accurately; neither a model-coverage status of `observed` nor a CAD pass is an assembly-stock guarantee. Do not invent a requirement to fabricate and power hardware before delivering the CAD package, and do not claim hardware qualification.

### 5. Create a genuinely fresh production GUI project and download

After the three frozen tails pass and electrical review is satisfactory, use an
authorized account and normal login. Do not print credentials, reset passwords,
forge authentication or adjust quota. Check current projects/jobs and effective
spend guards before submitting. Create a genuinely new project at
`https://kicraft.io/` using the literal brief and automatic defaults. Do not
retry or relabel historical projects 1155–1158.

Observe the real browser: capture the brief/checkbox before submission, new project ID/code/URL/generation, all five committed stages, final build result, actual schematic and PCB views, and the completed download click. Recheck the newly generated electrical identities independently; source30's earlier probe is not proof about this stochastic GUI run. If it is wrong, preserve/reject it and repair the generic source rather than accepting a green status.

Before using the copied `F/verify_gui_delivery.py`, extend its explicit excluded-project set to `{1155,1156,1157,1158}`. Bind it to `F`'s new source/router/cohort evidence. Preserve its native electrical/current/thermal/finite-ground/manufacturing, receipt, provenance, source-custody and ZIP content checks. It does **not** replace the fourteen-channel/intent audit; save that independent state/board-bound result alongside its output.

Write an actual-browser proof JSON at `F/gui-browser-proof.json` from observations, not inferred flags. The helper consumes: `kind="actual_browser_gui"`, `created_via_gui`, `download_clicked`, `brief_dom_value`, `auto_default_before_submit`, `account_id`, new `project_id`, `project_url`, `attempt_generation`, `board_code`, `download_path`, and source/router fingerprints (`source_fingerprint`, `router_inventory_sha256`, `router_commit`). Bind any historical account checks in the copied helper to the account actually authorized for the new campaign. The Boolean observations must really be true. Run:

```bash
./deploy/run-with-env.sh python "$F/verify_gui_delivery.py" "$F/gui-browser-proof.json"
```

The actual GUI download must match the server's current project ZIP, contain the same promoted board and nested fab archive/receipt, and be fresh for the same attempt/run. Validate Gerbers, PTH/NPTH drill files, BOM, placement CSV, STEP and its coverage report, and fabrication instructions. Inspect receipt hashes and archive member integrity; a file merely existing is insufficient. Close the browser after evidence capture.

## Final acceptance and delivery

The board is ready to report as completed only when all of these are true:

1. Three full cold fabrication tails qualify under one unchanged final source/router inventory, with preserved input and output custody.
2. A fresh production GUI project uses the exact brief and automatic defaults, commits all five stages, and produces an electrically correct native board with required fabrication gates passing.
3. Its actual GUI-downloaded package passes freshness, source/run/generation binding, native audit, manufacturing and receipt/content checks.
4. Delivery includes the new project URL/code, actual downloaded package path/hash, final source/router fingerprints, qualification evidence paths, electrical/default/load/manufacturing caveats, and measured spend/source-change counts with their baseline.

Recompute financial and change-count evidence at delivery. `C/campaign-source-change-counts-source30-probe.json` and `continuation-source-change-counts-source30-probe.json` are historical comparisons; `C/source-counts-binding.json` is source29-bound. Do not publish them as current counts without a new binding. Keep opaque CAD/model assets separate from code/text counts, and never add outstanding budget reservation tables together or clear them manually.

Until these conditions hold, report the exact failed boundary and witness. Project1158, a passing old physical replay, a deployed fix, or a green unit suite is not the requested completed HUB75 board.
