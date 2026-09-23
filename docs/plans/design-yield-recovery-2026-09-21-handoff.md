# Design-yield recovery — evidence and execution (2026-09-21)

## Decision

Recover **manufacturable boards**, not stage commits or optimistic build labels.
Use the stronger pinned native design session with **current manufacturing**, while
keeping their state schemas isolated. Do not deploy the old manufacturing tail:
independent DRC has disproved some of its `fab-ready` labels.

Execution is in progress. No completed new 34-brief result is claimed below.
Production remains on `current` until the corrected path is verified and measured.

## Critical review of the previous plan

1. **The primary metric was wrong.** Design-only screens cannot establish fab-ready
   yield. Full synthesis, routing, ERC, DRC and fabrication export are required.
2. **Denominators were mixed.** Historical 4/34 and 21/34 were distinct-brief coverage
   across repeats, not first-pass probabilities. The first-pass baselines are 2/34
   and 19/34, respectively. Keep attempts, first-pass yield and cumulative coverage
   separate; never erase failed attempts with a successful retry.
3. **Build success was over-trusted.** Native final gates omit several real copper
   manufacturing violations. An archive and `build_rc=0` are insufficient evidence.
4. **Projected improvements were unsupported.** Remove speculative yield increments
   and claims that several simultaneous changes isolate one cause. Measure the
   deployed combination and distinguish frozen reproductions from live results.
5. **Provenance was too trusting.** Model-authored recipe/lowerer identifiers cannot
   prove physical realization. Re-derive reviewed artifacts and compare identities,
   quantities and topology before accepting ownership.
6. **Quantity exemptions were too broad.** Board dimensions are fabrication duties;
   electrical limits and component quantities still need owners and realizations.
7. **“Empty deterministic BOM” misdiagnosed the saved RP2040 failure.** The actual
   candidate contained 65 parts. A synthetic assembly header conflicted with the
   recipe's castellated edge interface; fixing that requires compiler provenance.
8. **The native bridge was not its native session.** The old CLI bridge lost answers
   and repair instructions and omitted native BOM reconciliation. Running current
   post-wiring serialization over native state then made it unreadable by native
   models. Native design and current manufacturing need an explicit boundary, not
   accidental cross-tree imports.

## Baseline and measurement definitions

| Saved campaign | First pass: reported rc0 | All attempts: reported rc0 | Distinct briefs across repeats |
|---|---:|---:|---:|
| `movea_a1_current_20260919T1431Z` | 2/34 | 5/102 | 4/34 |
| `movea_a2_legacy_bc6a2f8` | 19/34 | 30/68 | 21/34 |

Fresh independent inspection confirms both current first-pass successes
(`fpc-breakout`, `audio-jack-buffer`): zero DRC errors, zero unconnected nets,
passing ERC, and verified Gerber/drill archives. Evidence:
`logs/self_eval/yield_recovery_20260921/baseline_current_independent.json`.
The 19 native first-pass labels have not been independently certified and must not
be presented as 19 proven manufacturing successes.

New campaigns retain the **34 original briefs**, original contract version,
per-attempt costs, source fingerprints and every failure. Use one manufacturing
slot on this two-core production host. No LLM judge is needed for manufacturing
proof. Brief fulfillment is a separate result: clean ERC/DRC does not certify
analog performance, firmware, sourcing, or every requested feature.

## Implemented design-source repairs

### Physical ownership and realization

- `stage_contracts.py`, `stage_work_units.py`, `part_identity.py`: recognize a
  declared interface from re-derived lowerer groups covering its committed nets.
  Require exact part/value/symbol/footprint identity and valid role/index bounds.
- Reviewed pushbutton, LED and USB-receptacle identities are recognized without
  accepting arbitrary model provenance.
- Resistor networks require the complete canonical R2R realization, exactly once;
  one resistor is not a DAC and one topology cannot satisfy quantity greater than
  one. Thermocouple terminal recognition uses actual reviewed physical evidence.
- Exact primary-IC requirements may own their verified recipe support parts;
  unrelated same-sheet parts do not acquire that ownership.
- Bare Keystone `8734` is recognized only with its reviewed symbol/footprint pair.

### Obligation classification and attachment

- `models.py`, `architecture_intent.py`, `stage_semantics.py`: exempt only explicit
  board/PCB geometry from component ownership. Preserve electrical quantities.
- Normalize a board-outline pseudo-component into a fabrication obligation only
  when no reviewed realizable component class/variant applies.
- Repair omitted ownership only when a unique reviewed realization proves the
  owner. Ambiguous ownership still fails; record deterministic attachments.

### Castellated edges

- Compiler-generated edge headers carry `compiler_origin="edge_connector"`.
  Provider-authored requirements cannot supply that field.
- A unique recipe edge role may absorb only that synthetic companion when its
  block and complete edge-net coverage match. Explicit user headers remain real
  requirements. The RP2040 recipe retains 34 non-assembly castellated pads rather
  than gaining an unintended assembly header.

### BOM validation termination

- Preserve meaningful manufacturer codes even when `mpn == value`.
- Normalize optional metadata consistently on both sides of deterministic identity
  comparisons; bound deterministic fallback with an internal one-shot guard.
- Frozen USB-C reproduction previously recursed roughly 990 times. It now returns
  an honest ownership error promptly: the USB connector does not realize the
  requested header. This proves termination, **not** that the design now passes.

### Canonical recipe identities and shared MCU ownership (current design pipeline)

Two provenance defects made the canonical RP2040/STM32 recipes fail their own
§9.42 physical-realization gate, and neither was fixable by weakening it.

- **Unresolvable identities.** `rp2040-minimal@2` emitted its MCU and QSPI flash
  with no MPN at all, and `W25Q16JVSS` (SS/208-mil SOIC-8) was stamped with the
  smaller 150-mil `SOIC-8_3.9x4.9mm` land pattern, while both recipes' crystals
  used the unreviewed `ABM8-272-T3` / generic `Device:Crystal_GND24` pair. The
  reviewed inventory has no record for any of those triples, so zero parts
  realized and the BOM commit was refused. The MCU, flash, RP2040 12 MHz and
  STM32 8 MHz crystals now carry resolvable identities: `RP2040`, `W25Q16JVSS`
  with its true package, and reviewed records for `abm8-272-t3` (Abracon
  ABM8-272-T3 datasheet) and `x32258msb4si` (the vendored `crystal-8mhz-3225`
  bundle, LCSC C2682774). No gate, tolerance or reviewed record was relaxed.
- **Support obligations were expanded as whole MCUs.** The architecture can emit
  `clock`, `flash` and `rp2040` requirements that all name the same family and
  exact part; each one independently selected `rp2040-minimal@2`, so a single
  RP2040 board became three complete 51-part MCU circuits (162 parts). A new
  coalescing pass attaches such a requirement to one existing MCU instance only
  when the recipe, physical sheet map, parameters, external nets and pin
  allocation all agree, the requirement is not itself a processor demand, and
  reviewed non-MCU groups implement every physical class it declares. Anything
  ambiguous stays an explicit `mcu_support_owner_unproven` error instead of
  silently merging two physical cores.

Deterministic replay of the two frozen current-pipeline architectures
(`staged/recipe_realization_probe.json`) now reports one
`rp2040-minimal@2` selection owning `rp2040`+`clock`+`flash` with **60** parts,
`physical_realization_ok=true` and closed net coverage, and the STM32 design
realizes its crystal with 26 parts.

### Router hole clearance and manufactured copper

- **NPTH clearance was never binding.** The pinned router hard-codes a 0.20 mm
  NPTH-to-track floor and exposes no knob, while the emitted project rule is
  KiCad's default 0.25 mm. USB-C leaves therefore failed every round on one
  `hole_clearance` violation at 0.2077–0.2474 mm from J1's alignment posts.
  Before routing, the adapter now stages a copy of the input plus its
  authoritative `.kicad_pro`/`.kicad_dru` and stamps named all-copper rule areas
  around every footprint NPTH drill, dilated by exactly
  `min_hole_clearance − router_clearance`. The finished board is then stripped of
  those router-side areas, so shipped artifacts carry only authored geometry and
  the normal DRC gate still measures the real rule.
- **A vendored footprint shorted itself.** Every `PrototypingPad_1.5mm_Drill0.8mm`
  carried an unnetted `F.Cu` `fp_circle` concentric with its own netless pad, so
  KiCad DRC reported one shorting item per pad — 25 for the R-2R pad field. That
  made `shorts == 0` unreachable, aborted the parent candidate search, and
  promoted a partial leaf preview instead of a board. The stray copper graphic is
  removed and the bundle's `content_hash` refreshed.
- **Leaves with no internal nets were never validated.** `_stamp_trivial_leaf`
  hard-coded `accepted: true`, so the pad-field defect above could not surface and
  a missing source board silently counted as success. Such leaves now run the same
  real DRC as every other leaf (with the project rules propagated first), report
  their true verdict, and fail honestly when there is no board to stamp.

### Connector mating evidence

`_directional_edge_candidate` used body depth as a proxy for an in-plane mouth, so
the frozen USB3 breakout header J6 (a through-hole 2×05 vertical header, 12.70 ×
5.00 mm courtyard, no `PCB Edge` marker) was classified as an unmeasured
side-entry connector and rejected every candidate at compose. Mating axis is now
extracted from the footprint's mechanical contract (`PinHeader_*_Vertical`,
vendored `HDR-TH_*‑V‑*`) and persisted with the leaf artifact; the same classifier
drives the compose gate and the final fab gate. A verified board-normal header is
omitted from both; every other connector without a measured in-plane opening stays
**blocking**, never a warning — the previous `unknown_mouth` warning path is gone.

## Native design / current manufacturing boundary

Native design still runs under pinned `bc6a2f8`, in its own interpreter. The bridge
passes answers, repair instructions and run identity to the native session and
uses one capped client across native BOM reconciliation. Missing/malformed result
packets and contradictory exit status fail closed. Real user questions remain
parked rather than becoming generic failures.

Native synthesis first compiles the snapshot against the same symbol/footprint
library that validated the design, including ERC. Current manufacturing then
consumes the compiled KiCad project through the real replay/build tail. This
boundary matters: direct current synthesis of the frozen FPC design rejected
native no-connect pins 25/26 because the current library's same-named symbol no
longer has those pins. Do not erase pin declarations or assume cross-version
library identities are interchangeable.

The separate `build_state.json` receives current manufacturing serialization;
only the shared `artifacts` result is published back into
canonical native state. Authored stages, questions, history and native BOM schema
remain native and resumable. A source change during the build must not be silently
overwritten. Worker, web fallback and evaluation use the same dispatch boundary.
Current post-wiring authorship remains disabled for native-owned design state.

This is deliberate composition, not a claim that the native tree acquired current
typed design work units. Provenance and the admin label identify **native design
plus current manufacturing**. Obsolete native routing/export is removed from new
build dispatch; native compilation retains ownership of its part-library version.
The real build tail has now exercised this boundary on all three frozen designs.
All three preserved authored state and remained readable by native models.
ESP32 changed from 36 keepout intrusions and rc7 to **rc0, zero DRC errors, zero
unconnected nets, passing ERC and a verified fabrication archive**. FPC now has
zero DRC errors but two unrouted connections and remains blocked. USB-C stops at
an illegal-geometry leaf gate; its partial preview is not a finished board.
Evidence: `current_tail_smoke/summary.json` and
`current_tail_smoke/esp32-manufacturing-evidence.json`.

## Frozen replays on the corrected source (verified)

Both previously blocked frozen workspaces were replayed end-to-end through the real
build tail against an isolated copy of the corrected tree (no provider calls, no
production edits during the campaign):

| Frozen design | Before | After |
|---|---|---|
| `recovery_current_r2r/run_02_r2r-dac` (current design) | parent candidate search aborted: every candidate `shorts=25` | promoted routed parent, `shorts=0 unconnected=0`, fab package; independent `kicad-cli` DRC **0 errors** |
| `current_tail_smoke/usb-c-full-breakout` (native design) | `connector_orientation_unmeasured:J6`; no routed parent, partial preview promoted | `REPLAY COMPLETE`, promoted routed parent, `shorts=0 unconnected=0 courtyard=0 keepout=0`, fab package; independent DRC **0 errors** |

The USB-C leaves also stopped failing: the USB C INPUT leaf (two J1 NPTH posts) is
now DRC-clean instead of rejecting 12 consecutive rounds on one `hole_clearance`
violation, and the shipped leaf no longer carries the router-side keepout areas.
Evidence: `replay_r2r_clean.log`, `replay_usb_clean.log`,
`replay_*_independent.json`, and the isolated-overlay test run
(`staged/overlay-full-tests-final.log`: **4357 passed**, 18 skipped, 1 xfailed, one
pre-existing failure that also fails on the deployed tree:
`test_vendored_bundles_are_not_prototype`, caused by the unrelated
`ams1117-5v0-fixed` bundle still declaring `prototype`).

## Deployed default and measured baseline

The source repairs were integrated and deployed (`deploy/deploy-production.sh`:
web HTTP 200, build worker ready). Evidence for the default: with the **same**
current manufacturing tail, the pinned native design engine completed a full
34-brief campaign with **16/34 first-pass fab-ready** and every one of the 16
independently certified (0 DRC errors, 0 unconnected, passing ERC, verified
Gerber/drill export) — against the previously saved current-engine first pass of
2/34. The admin pipeline knob was therefore set to native design + current
manufacturing and re-deployed; `pipeline.selected()` reports `legacy`.

Campaign `recovery_full` (baseline, pre-fix manufacturing, `$0.93`): 16/34
certified. Its 18 failures were dominated by design-stage deficits (missing
decoupling/charge-pump networks, non-orderable picks, dangling nets) rather than
the manufacturing classes repaired here, so the next section measures the same
briefs on the corrected source rather than extrapolating.

Campaign `recovery_fixed` (same briefs, corrected source, `$0.95`): 17/34
reported and all 17 independently certified, with 7 briefs newly passing
(`r2r-dac`, `usb-c-full-breakout`, `usb-a-power-splitter`, `rs485-terminal`,
`stm32-min`, `highside-switch-10a`, `dual-rail-supply`) against 6 that regressed.
Five of those six regressions are design-stage variance (orderability, BOM
deficits, a self-shorted two-terminal part). The sixth class was **real and
self-inflicted**: the first cut of the connector gate reported every mouthless
edge-zoned part as `unverified_directional`, so a fab-clean board whose only such
part was a slide switch (`SW1`/`SW2`) failed the terminal gate. The gate was
corrected to block on connector *evidence* — a recognized horizontal terminal
row, or a through-hole body deeper than the calibrated 4 mm cut — while keeping
the historical no-verdict reading for shallow parts; a shallow/deep pair is pinned
by `test_facing_ignores_shallow_part_without_connector_evidence`, and the
correction was re-deployed before any further measurement.

The six regressed briefs were then re-run as a **separate, visible retry
campaign** (`recovery_fixed_retry`, `--only` those six) rather than folded back
into the first pass. Four passed (`speaker-crossover`, `hex-env-sensor`,
`star-ornament`, `snowman-ornament`) and all four were independently certified;
`led-cc-driver` (thermal-pad ground deficit) and `daq-8ch` (self-shorted
two-terminal part) still fail on legacy-engine design deficits.

| Measurement | Value |
|---|---|
| Baseline first pass (pre-fix manufacturing) | 16/34, all certified |
| Corrected source, first pass | 17/34, all certified |
| Corrected source, distinct briefs passing | **21/34**, all certified |
| Certified false positives in any campaign | 0 |

Independent certification for every claimed success: zero `kicad-cli` DRC
errors, zero unconnected nets, passing ERC and a verified Gerber/drill archive.
New live provider spend for all of today's work: **$2.95** against the authorized
`+$18`. Evidence: `recovery_full/summary.json`,
`recovery_fixed/summary.json`, `recovery_fixed_retry/summary.json`, and the
matching `*_independent.json` files.

The `current` design engine remains weaker than the pinned native one and its
recipe repairs do not change that today: two live design-only attempts on the
corrected source still fail `rp2040-min`/`stm32-min` at other gates (an unproven
regulator power-transfer contract, and a model-authored crystal that conflicts
with the recipe-owned one), whereas the deterministic replay above proves the
identity/multiplicity defects themselves are gone. No yield claim is made for
that engine.

## What the live investigations actually showed

All paths below are relative to `logs/self_eval/yield_recovery_20260921/`.

| Investigation | Outcome | Interpretation |
|---|---|---|
| `current_firstpass` | Interrupted after 11 completed runs; 1 reported rc0 | USB-C BOM validation entered recursive deterministic fallback. Not a complete 34-brief result. |
| `legacy_bridge_smoke` | Native session/ledger exercised; wiring failed | Protocol proof only, not a successful board. |
| `legacy_full` | Interrupted after 15 completed runs; 8 reported rc0 | Independent DRC disproved two labels; do not continue routing with an unsafe old tail. |
| `independent_fab_verification.json` | 6 of those 8 candidates independently DRC-clean | Partial diagnostic evidence, not final 34-brief yield. |

The rejected native candidates are concrete:

- **USB-C full breakout:** 57 DRC errors — clearance, hole clearance, annular width
  and via diameter. Vias were 0.25 mm despite the project's 0.36 mm minimum; rings
  were 0.05 mm despite its 0.105 mm minimum.
- **FPC breakout:** 11 DRC errors of the same manufacturing classes.
- **ESP32-S3 sensor:** native final build correctly failed 36 antenna-keepout
  intrusions, although its intermediate routed validation incorrectly accepted
  them. Pre-route GND escape stubs and tip vias crossed the footprint rule area.

Current source already contains the relevant producer fixes:
`autoplacer/kicad_routing_tools.py::_project_routing_floors` binds adaptive routing
to actual project limits; `brain/breakout_stubs.py` checks track/via rule areas;
`hardware/keepout_extract.py` supplies the geometry; `routing_board.py` blocks
keepout intrusions. Use these implementations rather than relaxing constraints or
exporting the rejected boards. Frozen copies for current-tail verification are in
`current_tail_smoke/`; original failed artifacts remain untouched.

Remaining native design failures examined so far are genuine five-attempt
exhaustions, not missing feedback: dangling `FAULT`, an incompatible isolated
DC-DC symbol/footprint pair, and a nonexistent header footprint. Native feedback
has weaknesses, but arbitrary substitution or deleting required nets is not an
acceptable fix. Targeted fresh attempts must remain visible as retries.

## Verification and spend

The final integrated focused suite passed **573 tests**, with two opt-in replay
tests skipped and five dependency deprecation warnings. Evidence:
`integrated-tests.log`. The affected frozen reference slice passed **5/5**: R2R,
FPC, RP2040, round LED ring and snowman. A broader reference run timed out at
300 seconds; it is not a full-suite pass.

Today's repaired set was verified on the deployed tree with the **whole suite**:
`4357 passed, 18 skipped, 1 xfailed`, and one failure —
`test_vendored_bundles_are_not_prototype` (`ams1117-5v0-fixed` still declares
`prototype`) — which fails identically on the untouched baseline and is therefore
pre-existing rather than introduced here. Evidence:
`integrated-tests-final.log`, plus the same run on the isolated copy of the tree
before deployment (`staged/overlay-full-tests-final.log`).

Suite-order fixture failures were corrected to patch module objects rather than
package-attribute strings. An existing build-concurrency test also failed to mock
post-wiring provider calls; it is now isolated. The final combined suite incurred
**$0 additional LLM spend**, verified against the ledger in
`test-spend-verification.json`. Earlier incidental test charges remain included
in the authorized spend total.

The actual admin routing page was rendered in an isolated server with temporary
storage and an authentication-only preview seam. The new pipeline label and
trade-off text were visually verified; no production config was saved. Screenshot:
`admin-routing.webp`.

Authorized new live LLM spend: **$20 maximum**. Starting shared ledger total:
`130.73234997348`. Enforced total ceiling: `148.73234997348` (+$18), reserving $2 for
in-flight/shared activity; daily ceiling $19. Profile snapshots and settings are in
`budget.json`, `routing-current.json`, `routing-legacy.json`. New spend at the last
stopped-campaign checkpoint was **$0.66164683**, including unsuccessful calls.
Final accounting must use the shared ledger delta, not just successful run costs.

## Completion criteria and next execution

1. Prove the isolated current build against frozen FPC, USB-C and ESP32 native
   designs. Inspect real promoted boards with their project rules, verify native
   state remains readable, and retain failures rather than overriding a gate.
2. Run a fresh full 34-original-brief campaign on the corrected composition under
   the cap. Keep source/config fingerprints fixed during that campaign.
3. Independently inspect every reported success: zero DRC errors and unconnected
   nets, passing ERC, current manufacturing gates and verified Gerber/drill export.
   Report first-pass yield separately from any targeted-retry coverage.
4. Investigate remaining failures and use justified targeted attempts or source
   repairs while preserving the original briefs and budget. Do not promise 34/34
   from estimates or replace missing outcomes with design-only results.
5. Record complete results here, select the proven default, deploy with
   `deploy/deploy-production.sh`, and verify HTTP 200 plus build-worker readiness.
   Remove throwaway runners after measurement; retain reports and failed evidence.

## Status of those criteria

1. **Done.** Frozen R2R and USB-C native/current designs were replayed through the
   real build tail on an isolated copy of the corrected source (see *Frozen
   replays*); both now promote routed parents with zero DRC errors and exported
   fabrication packages. The native state handoff is unchanged by these repairs.
2. **Done.** Two full 34-brief campaigns were run under the cap with fixed source
   fingerprints (`recovery_full`, `recovery_fixed`), plus a targeted six-brief
   retry campaign.
3. **Done.** Every claimed success in all three campaigns was independently
   inspected: 16/16, 17/17 and 4/4 certified, with zero false positives.
4. **Done.** Failures were investigated individually; the only source-level
   regression found (the connector gate over-blocking shallow parts) was fixed,
   re-tested and re-deployed, and its briefs were re-measured as visible retries.
5. **Done.** The native-design + current-manufacturing default was selected on
   measured evidence, deployed, and verified (web HTTP 200, build worker ready);
   `pipeline.selected()` reports `legacy`.

Remaining known gaps, stated plainly: 13 briefs still fail at the design stage of
the pinned engine (missing decoupling/charge-pump networks, non-orderable or
insufficiently stocked parts, dangling nets, self-shorted two-terminal parts),
`current`-engine MCU briefs still fail later gates, one pre-existing unrelated test
failure (`test_vendored_bundles_are_not_prototype`, `ams1117-5v0-fixed`) remains,
and legacy-engine design outcomes are stochastic run to run — none of that is
hidden by the numbers above.
