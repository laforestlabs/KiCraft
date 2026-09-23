# Path to 34/34 — remaining design-yield failures and how to clear them (2026-09-22)

> **SUPERSEDED — do not execute this plan.** The product objective was restated on
> 2026-09-22: see [general-brief-90-percent-yield-plan-2026-09-22.md](general-brief-90-percent-yield-plan-2026-09-22.md),
> which replaces "34/34 briefs eventually passed" with >90% delivery on unseen briefs and
> states that clearing the last outliers of the known 34 is **not** a prerequisite. The
> measurement that motivated the replacement is in
> [../handoff-general-brief-yield-2026-09-22.md](../handoff-general-brief-yield-2026-09-22.md)
> (0/60 on unseen briefs, every failure in the pinned engine's parts-list stage). This
> document is retained as the diagnostic record of that retired target.

## Goal and definition of done

Every one of the 34 original briefs reaches a **certified fab-ready board** with the
deployed composition (native design engine at pin `bc6a2f8` + current manufacturing),
verified by the same independent bar used for the 21 successes: zero `kicad-cli` DRC
errors, zero unconnected nets, passing ERC, and a verified Gerber/drill archive.

Two numbers are tracked separately and neither is allowed to absorb the other:

| Metric | Meaning | Now |
|---|---|---|
| First-pass yield | briefs that pass on the first attempt of one full campaign | 17/34 |
| Distinct-brief coverage | briefs that have passed on any attempt, retries visible as retries | **21/34** |

Target: 34/34 distinct-brief coverage, then 34/34 first pass.

## Where the 13 remaining failures actually come from

Evidence: `logs/self_eval/yield_recovery_20260921/{recovery_full,recovery_fixed,recovery_fixed_retry}/`,
plus `current_firstpass/`, `legacy_full/`, `movea_a2_legacy_bc6a2f8/`.

| # | Brief | Failure | Observed in | Verdict |
|---|---|---|---|---|
| 1 | rp2040-min | wiring park: "RP2040 needs nine 100nF decoupling caps + 1uF VREG" | fixed | deterministic park, reconcile exhausted 3/3 |
| 2 | nrf52-beacon | park: decoupling on DEC1–DEC6/DECUSB/DCCH | full, fixed | deterministic park, 3/3 |
| 3 | stepper-a4988 | park: A4988 VCP charge-pump + VREG network | full, fixed | deterministic park |
| 4 | led-cc-driver | park: PAM2804A application circuit / TPS54160 thermal pad | full(pass), fixed, retry | stochastic part count, deterministic class |
| 5 | audio-jack-buffer | park: MCP6004 channel count + coupling/bias | full, fixed | deterministic park |
| 6 | round-led-ring | park: 11× 100nF WS2812 bypass; also §9.25 polarity pair | fixed; full | deterministic class (two defects) |
| 7 | gpio-expander | park: 16-pin connector described as one GPIO bank | fixed | contract/identity mismatch |
| 8 | usb-pd-trigger | §9.26: `C970725` retail stock 0 (assembly stock 7051) | full, fixed | deterministic given stock snapshot |
| 9 | daq-8ch | §9.17/§9.19: series R shorted onto one net | fixed, retry | model wiring error, both attempts |
| 10 | esp32-s3-sensor | §9.33 named-part accountability / BME280 ports | full, fixed | contract: semantic ports vs raw pin names |
| 11 | encoder-oled-panel | rc5 §9.9: `MOUNTING_STRUCTURE.kicad_sch: 4 components, 0 wires` | full, fixed | deterministic false positive |
| 12 | esp32-dual-motor | rc7: `connector_stranded:J2@-8.39mm`, `antenna_stranded:U2@4.99mm` | fixed | deterministic placement, post-route discovery |
| 13 | rounded-c3-devboard | rc7: `connector_orientation_unmeasured:SW1/SW2` | fixed (pre-correction) | **likely already fixed** — see Phase 0 |

Two mechanical corrections to earlier analysis, both verified:

- **The "0 reconcile passes" label is wrong.** `legacy_session_runner` reports
  `reconcile_passes: 3` (budget exhausted) in the saved packet; `self_eval.py:564-575`
  prints its own local counter, which stays 0 for native-owned state
  (`current_tree_owns_design_state` is false for `legacy`). Every deficit park listed
  above therefore spent the full three-pass budget and still could not satisfy the ask.
- **The parked asks are protocol, not electrical checkers.** Legacy wiring prompts the
  model for one `{text, blocking, reconcile_target:"bom"}` obligation when a missing
  support part is the only blocker (`KiCraft-legacy/kicraft/server/stage_driver.py:373-400`),
  and `bom_reconcile_deficits` selects exactly that park shape
  (`KiCraft-legacy/kicraft/server/session.py:328-335`). The gate is legitimate; the
  deficit text is model-authored.

## The structural constraint that shapes everything

`bc6a2f8` **is an ancestor of the current HEAD**, and the pinned tree simply lacks the
machinery this repo has since grown:

| Path | legacy @ bc6a2f8 | current |
|---|---|---|
| `kicraft/design/part_identity.py` | *absent* | 141 KB |
| `kicraft/design/lowering.py` | *absent* | 71 KB |
| `kicraft/server/stage_work_units.py` | *absent* | 101 KB |
| `kicraft/design/synthesis/validation.py` | 124 KB | 215 KB |
| `kicraft/design/models.py` | 40 KB | 68 KB |

So most design-stage fixes have two possible landing zones, and the choice must be
explicit rather than accidental:

- **M1 — move the pin forward.** Commit the current working tree, re-point
  `LEGACY_COMMIT` to it, and drop the second interpreter. Highest leverage and removes
  the two-tree split, but it changes what "pinned engine" means and risks the current
  engine's own weaker record (2/34 first pass historically, and today's `current` MCU
  briefs still fail at §9.39/§9.42-adjacent gates). Not the recommended first step.
- **M2 — keep the pin, add deterministic completion in the tree we own.** Extend
  `legacy_session_runner`'s reconcile loop to call **current-tree, native-schema-aware,
  no-model repairs** (support-circuit completion, typed wiring invariants) and re-drive
  legacy wiring. This is the only way to fix the parked classes without editing the
  pinned checkout, and it keeps every existing commit gate as the guard.

**Recommendation: M2 for the design-stage classes; treat M1 as a follow-up once the
current engine's own gaps (`§9.39` power-transfer, `§9.42` port mapping) are closed.**

## Plan

### Phase 0 — re-measure the 13 on the deployed code (do this first)

`rounded-c3-devboard` failed on `connector_orientation_unmeasured:SW1/SW2`, and those
parts are the **SMD** stock slide switch `SW_DIP_SPSTx01_Slide_6.7x4.1mm…_JPin`
(`(attr smd)`, no hole). The deployed correction only reports an unmeasured mouth for
parts with through-hole evidence, so that rejection should already be gone. Several
other briefs were also run before that correction.

Action: run `--only` over the 13 with the live config; independently certify whatever
passes. Expected: +1 to +3 to distinct-brief coverage at zero source risk, and an
honest, current failure set for Phases 1–4.

### Phase 1 — support-circuit ownership (the single biggest lever: up to 6 briefs)

Today the *model* must author the MCU/IC support network (per-pin decoupling,
charge-pump, bypass), and when it does not, the design parks and the 3-pass reconcile
cannot rescue it. The current tree already owns that knowledge deterministically
(`design/recipes/rp2040_minimal.py` io/dvdd/bulk groups and pins;
`ws2812-output@1` per-pixel bypass; `part_identity.ReviewedPart.support_network`;
per-brief expectations in `eval/acceptance_contracts.py`, e.g. "per-pixel 100 nF
decoupling" at line ~1310).

1. **1a — counted and per-pin asks (S).** The deterministic passive-add parser cannot
   represent "nine" (quantity words cover a/an/one/two/three/four and `_ask_qty` caps at
   8) and has no notion of "one on each of N named pins/pages", so a many-instance ask is
   provisioned once and judged unfulfilled. Support counted asks, `one per <pin list>`,
   and per-sheet repetition. This is a narrow, testable fix in the add path.
2. **1b — deterministic support completion (L).** For an architecture requirement that
   resolves to a reviewed identity or a registered recipe, inject that identity's
   support circuit (groups + pins + nets) into the native BOM before wiring, exactly as
   recipe expansion already does for the current engine. Where a reviewed record lacks a
   support network (e.g. `rp2040`, `nrf52840-qiaa-r`, `a4988settr-t`, `drv8833pwpr`,
   `ws2812b-b/t` — only 7 of 84 records carry `support_network` today), author the
   reviewed support template from the manufacturer datasheet and record it in the
   reviewed inventory; the missing-cap asks above are exactly those parts.
3. **1c — honest accounting (S).** Adopt `packet["reconcile_passes"]` in `self_eval`
   instead of the dead local counter, so a park never again reads as "0 passes" and the
   next investigation starts from the truth.

Unblocks: rp2040-min, nrf52-beacon, stepper-a4988, led-cc-driver, audio-jack-buffer,
round-led-ring (its WS2812 half).

### Phase 2 — mechanical-only leaves are not unwired sheets (1 brief)

`encoder-oled-panel` fails §9.9 with `4 components, 0 wires, 0 power symbols` on a
mounting-hardware leaf; the BOM proves all four are `Mechanical:MountingHole`. The
module already defines exactly this safe taxonomy (`_PINLESS_MECHANICAL_SYMBOLS`,
`kicraft/design/synthesis/validation.py:884-909`) but the connectivity check does not
consult it. Exempt a leaf only when every non-power symbol is in that set; four
ordinary components with no wiring must still fail. Land in the current tree; proof is
a four-`MountingHole` fixture that passes while `LOST.kicad_sch` still fails.

### Phase 3 — wire the placement intent into feasibility, not post-route verification (1 brief)

`esp32-dual-motor` routes fine and then fails promotion on
`connector_stranded:J2@-8.39mm(top)` + `antenna_stranded:U2@4.99mm(top)` — repeated
across all three rounds and all seeds. Edge-mating and antenna-edge intents are checked
*after* the board is routed; they belong in parent candidate scoring, so a candidate
that would strand a connector never wins a round. Land in `cli/compose_subcircuits.py`
candidate scoring; keep the terminal gate unchanged as the backstop.

### Phase 4 — contract and sourcing gaps (4 briefs)

1. **§9.17/§9.19 typed series components (M, daq-8ch).** Both attempts wired a USB
   series resistor onto a single net. Give the wiring repair contract the offending
   terminal/net facts and a typed invariant: every two-terminal R/L/diode/fuse has each
   terminal exactly once, on distinct nets; a connector→MCU link is
   connector-net → series part → MCU-net. Gates unchanged.
2. **Reviewed port maps for exact identities (M, esp32-s3-sensor, gpio-expander).** A
   BME280 declares `gnd/sda/scl` while the symbol exposes `GND=1/7, SDI=3, SCK=4`; the
   reviewed record already carries `{ground:1, sda:3, scl:4}` but the work-unit contract
   only sees raw symbol names. Expose the reviewed `port_pins` for an exact
   bundle/symbol/footprint match and canonicalize accepted semantic ports to those
   numeric selectors — never infer aliases for arbitrary symbols. The same mechanism
   covers the connector-bank count mismatch in `gpio-expander`, and the §9.33 case
   (prose names a different exact part) stays an explicit `bom.substitutions` ledger row.
3. **Reviewed alternate for the PD controller (M, usb-pd-trigger).** The only failed
   predicate is *retail* stock on `C970725` (assembly stock 7051). Review a
   pin/topology-compatible selectable-PD controller bundle + recipe and let the resolver
   offer it as an auditable substitution; never blind-swap an explicit C-number.
   Contingent on that alternate genuinely being in stock at both storefronts.
4. **Polarized-pair convention (S, round-led-ring).** `Device:C_Polarized` was paired
   with a non-polarized `C_Elec_10x10.2_P12.50` land pattern. Offer only prevalidated
   matching pairs (polarized symbol ⇒ `CP_*` footprint) in the BOM producer/recovery
   contract; §9.25 stays.

## Sequencing and expected cumulative coverage

| Step | Change | Expected distinct-brief coverage |
|---|---|---|
| start | — | 21/34 |
| Phase 0 | re-measure 13 on deployed code | 22–24/34 |
| Phase 1 | support-circuit ownership (1a+1b+1c) | 28–30/34 |
| Phase 2 | mechanical-leaf exemption | 29–31/34 |
| Phase 3 | placement intent in feasibility | 30–32/34 |
| Phase 4 | typing, port maps, alternates, polarity pairs | 33–34/34 |

Cumulative ranges, not promises: Phases 1 and 4 each contain several independent
mechanisms, and the legacy engine's design stage remains stochastic, so a brief can pass
in one attempt and fail in the next. The plan reports first-pass and coverage separately
for exactly that reason.

## Verification protocol (every change)

1. **Producer-level test** — fails before, passes after, asserting the observable
   contract (a rejected plan, an emitted pin graph, a certified board), never wiring or
   implementation shape.
2. **Frozen replay** — replay the affected saved workspace through the real build tail
   from an isolated copy of the tree; inspect the promoted board with `kicad-cli` and
   confirm 0 errors / 0 unconnected / fab package. Keep the reproduction as the
   regression fixture when it is a real defect.
3. **Targeted retry** — re-run only the affected briefs against the deployed code, in a
   separate campaign directory so first-pass numbers are never rewritten.
4. **Full suite** on the deployed tree; one pre-existing unrelated failure
   (`test_vendored_bundles_are_not_prototype`, `ams1117-5v0-fixed`) is expected and must
   not be "fixed" by flipping a maturity flag that would enable an unreviewed part.
5. **Deploy discipline** — `deploy/deploy-production.sh`, verify web HTTP 200 + build
   worker ready, and confirm the effective pipeline before measuring.
6. **Spend** — every live attempt stays inside the authorized ledger delta; report the
   shared-ledger delta, not just successful run costs.

## Guardrails

- No gate is weakened, deleted, or downgraded to a warning; no brief, slug, ref or part
  is hardcoded; no measured evidence is replaced by an inference.
- Design-stage fixes that must reach the deployed engine land as M2 (current-tree,
  no-model completion driven by `legacy_session_runner`), so the pinned checkout stays
  byte-identical. If M1 (re-pin) is ever chosen, it is a separate, explicit decision with
  its own measurement.
- Failure text is treated as model-authored unless proven deterministic; each class above
  states which it is, with the artifact that decided it.

## Non-goals

- Raising the 3-pass reconcile budget: every park above already burned all three passes
  and re-parked on the same ask, so a bigger budget buys repeated identical failures.
- Publishing design-only screens as progress, or counting a board that skips a requested
  feature (the recovered `usb-c-full-breakout` earlier exposed exactly that trap).
- Relaxing §9.9/§9.17/§9.19/§9.25/§9.26/§9.33/§9.42 to make a brief pass.
