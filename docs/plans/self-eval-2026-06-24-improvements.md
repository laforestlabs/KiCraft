> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# KiCraft improvement plan — from self-eval 2026-06-24

Source run: `logs/self_eval/20260624T120631Z` (28 briefs, design+judge `deepseek/deepseek-v4-flash`,
parallel=3, build_slots=2, wall ~3.1 h, spend $0.84). Baseline for comparison:
`logs/self_eval/20260623T210902Z`. Deployed on `main@cf5ef4b`.

## Headline

| metric | 2026-06-23 (base) | 2026-06-24 (this) | Δ |
|---|---|---|---|
| fab-ready builds | 12/28 | 11/28 | −1 |
| mean final | 69.7 | 71.1 | +1.4 |
| median final | 72.8 | 73.5 | +0.7 |
| grades | A1 B12 C9 D5 F1 | B13 C7 D6 ERR1 F1 | — |
| gates fired | erc×2, silent_sub×1, unprog_mcu×2 | unprog_mcu×2, erc×2 | — |

Aggregate is **flat**. The single ERROR (#24 daq-8ch) is a transient OpenRouter **HTTP 503**, not a
code defect. The real story is in the per-brief breakdown, where two things are tangled together:
**(a) a handful of reproducible, systematic bugs**, and **(b) heavy run-to-run nondeterminism** that
makes single-run per-brief deltas mostly noise.

## The noise floor (read this before trusting any single-brief delta)

Design stages run `deepseek-v4-flash` at temp 0.2; the judge is the *same* model. Across the two
full runs (27 briefs graded in both, identical input, same branch):

- mean |Δfinal| = **11.8 pts**, median 10.5, max **42** (r2r-dac 45→87)
- **59% (16/27)** of briefs crossed a letter-grade bucket
- **5/27** flipped fab-ready ↔ not-fab-ready
- only **26%** moved ≤5 pts

So a −25 swing on one brief in one run is, by itself, ~no evidence of a regression. Every "regression"
below was confirmed to be either a reproducible code bug or a *nondeterministic design choice*, by
reading both runs' artifacts — not inferred from the score delta. **De-noising the eval (Workstream D)
is a prerequisite for trusting future regression calls.**

---

## Workstream A — Deterministic source bugs (cheap, high-confidence, do first)

These are confirmed code/data defects with reproducible failures. Each is a single-point "fix at source"
change, not a post-hoc band-aid. Verified directly against the tree on `main@cf5ef4b`.

### A1. Negative supply rails are not classified as power nets  ⭐ highest ROI
- **Symptom:** #28 audio-jack-buffer — grade **F / BROKEN / 39.5 in BOTH runs**, gate `erc_errors`.
  Exactly one ERC error: `power_pin_not_driven` on the op-amp `VCC-` pin, net `-12V`.
- **Root cause:** `is_power_or_ground_name()` in `kicraft/design/models.py` — `POWER_NET_PATTERNS[0]`
  is `^[+]?\d+\.?\d*V$` (optional `+`, **no `-`**). Verified: `-12V`, `-5V`, `-3V3`, `VEE`, `VSS`
  all return `False`, so the negative rail never gets a `PWR_FLAG` (emitter) nor a power symbol
  (router) → KiCad ERC correctly flags the `VCC-` power-input pin as undriven. This is the single
  shared classification choke point (router split, PWR_FLAG, driver scan, placement, gnd_pour,
  breakout_stubs, validation), so the miss propagates everywhere.
- **Blast radius:** every design with a negative or dual (±) supply — op-amp front-ends, audio,
  analog/sensor boards. Systematic, not model nondeterminism.
- **Fix:** widen the regex to accept a leading sign — `^[+-]?\d+\.?\d*V$` and `^[+-]?\d+V\d+$`
  (covers `-3V3`); add `VEE`/`VSS` to the named-rail set (note `router.power_symbol_for` and
  `placement.py` already special-case VSS/VEE for the *ground-symbol* choice but are currently dead
  because this gate returns False first). Add a regression test asserting True for
  `-12V/-5V/-3.3V/-3V3/VEE/VSS`.
- **Confidence:** High. **Effort:** ~1 line + test.

### A2. TPS54331 switch-node pin mistyped, trips `power_pin_not_driven`
- **Symptom:** #18 dual-rail-supply this run — ERC error `power_pin_not_driven` on **U1 pin 8 "PH"**
  (the buck switch/phase node), net correctly wired `{U1.8, D3.1, L2.1, C9.2}`.
- **Root cause:** `kicraft/parts_library/tps54331/tps54331.kicad_sym` types pin 8 "PH" as `power_in`
  (verified at line 113/116). PH is an *output* (it drives the inductor) → should be `power_out`.
  Pins 7/9 (GND/EP) are also `power_in` but don't trip (GND is flagged).
- **Fix (two layers):** (1) correct the vendored symbol: PH `power_in`→`power_out` (one-line data fix,
  zero masking risk); (2) generalize — add a sibling to `_normalize_passive_device_pins` in
  `kicraft/design/synthesis/symbol_library.py` that retypes regulator switch-node pins
  (`PH/SW/LX/PHASE/SWITCH`) `power_in`→`power_out` at the `extract_symbol_block` choke point, so
  other switchers don't repeat this.
- **Note:** the #18 *score* regression (B→D) was mostly a **nondeterministic topology choice** — the
  baseline used a self-contained `WRA2412S-3WR2` ±12V module; this run chose 2× discrete TPS54331
  and then hit the symbol bug. The symbol fix is real and worth doing regardless.
- **Confidence:** High. **Effort:** 1-line data + small normalizer + test.

### A3. Surface the synthesis reviewer's findings into user-facing state (failure_honesty)
- **Symptom:** `unprogrammable_mcu` gate (#10 rp2040-min, #12 esp32-s3) and others scored
  `failure_honesty: 0` because a real defect was known internally but hidden behind a healthy build.
- **Root cause:** `kicraft/design/synthesis/electrical_review.py` **already emits** a `[programming]`
  WARNING (and other findings) every run, but they stay as build-log lines — never copied into
  `state.bom.assumptions` / `open_questions`, and not escalated to a blocker.
- **Fix:** at stage-commit, copy reviewer findings (≥WARNING) into the user-facing `assumptions` /
  `open_questions` slots; consider treating `programming-path` (already in `_BLOCKER_ELIGIBLE`) as a
  fab-readiness caveat. This alone lifts `failure_honesty` off 0 and softens the gate's "gap not
  surfaced" clause.
- **Confidence:** High. **Effort:** small.

### A4. Honest build label for rc=5
- **Symptom:** #11 fpc-breakout shown as "ERC errors" but its ERC report has **0 errors** — the real
  failure is §9.13 netlist faithfulness.
- **Root cause:** `kicraft/eval/self_eval.py:87` maps `build_rc=5` unconditionally to `"ERC errors"`,
  but rc=5 = "synthesis-check failed" which also covers §9.13. Verified.
- **Fix:** split the rc=5 label by which check failed (read `synthesis_check.json` failed_checks):
  "ERC errors" vs "netlist faithfulness" vs "synthesis check failed".
- **Confidence:** High. **Effort:** tiny.

---

## Workstream B — Synthesis correctness guarantees (medium)

Deterministic gates/normalizers at the wiring stage-commit, in the established style of
`reconcile_inter_sheet_nets` / `bridge_duplicate_pins` / the §9.16–9.20 semantic-miswire family
(`kicraft/design/synthesis/validation.py` + `kicraft/design/cli_app.py:_cmd_stage_commit`).

### B1. Guarantee a first-flash / programming path for every programmable MCU  ⭐
- **Why:** `unprogrammable_mcu` (cap 50) is a **true positive** in every case examined. #10 rp2040-min:
  `QSPI_CS` not grounded by any BOOTSEL button/test-point, SWD pins in `no_connect`. #12 esp32-s3:
  GPIO0 and EN hard-tied directly to `+3V3` (no resistor/button), so the ROM bootloader can't be
  entered. The esp32-s3 B→D "regression" was the model **deleting the two 10k boot-strap resistors**
  between runs — pure nondeterminism. A deterministic guarantee is immune to that.
- **Fix:** new check in `validation.py` (alongside §9.16–9.20), wired into `_cmd_stage_commit`. For each
  MCU (detect via `extras.symbol_pinouts` / `symbol_pinout.lookup_pins`, already used at
  validation.py:961), assert a reachable first-flash path by family:
  - ESP32-S3: GPIO0/IO0 must be drivable LOW (pull-down + button or RC-jumper), not hard-tied; EN/CHIP_PU RC reset.
  - RP2040: `QSPI_CS` reaches a BOOTSEL button/test-point to ground, **or** SWD (SWCLK/SWDIO) is broken out.
  - Generic: require ≥1 of {SWD header, UART boot header, drivable boot strap + reset}.
  Emit a deterministic open_question / blocker when absent. The per-family boot-pin roles need a small
  amount of MCU pinout data added to the table the semantic checks already consume.
- **Confidence:** High on diagnosis, med-high on the exact per-family assertions. **Effort:** medium.

### B2. "Breakout" intent gate (nets must bridge the connectors)
- **Why:** #11 fpc-breakout emitted 49 nets, **none spanning both connectors** — J1 (FPC) and J2
  (header) on mutually disconnected nets, so the breakout's whole job (J1.k ↔ J2.k) was left undone.
- **Fix:** a §9.20-style heuristic detector — when the brief is a "breakout"/"adapter" and two
  connectors share **zero** bridging nets, flag it as an intent failure at stage-commit. This is a
  detector, not a normalizer (the pin mapping is a synthesis-intent decision, not mechanically derivable).
- **Confidence:** Medium. **Effort:** medium.

---

## Workstream C — Place/route convergence on dense / connector boards (load-bearing — surgical only)

The dominant non-fab category. Whole archetypes fail in **both** runs: `connector_dense_io` 0/4,
`hi_pin_hierarchical` 0/3. Ground-truth `kicad-cli pcb drc` on the promoted boards (servo-driver-16,
can-node, esp32-dual-motor, rs485-terminal) shows **`unconnected` is the universal gate failure**, with
three independent causes. Leaves route cleanly (90–96, 0 ERC); the failure is entirely at the parent /
inter-leaf stage. **`autoplacer/` is load-bearing — fixes must be surgical and collision-guarded.**

### C1. GND pour island closure  (~60% of unconnecteds — highest leverage in this workstream)
- **Symptom:** unconnected items are mostly `Zone[GND]↔Zone[GND]` island fragments clustered at one
  spot (servo 7, esp32 2). The union-find via-stitcher runs but leaves residual islands.
- **Fix:** in `kicraft/autoplacer/brain/gnd_pour.py`, add a final closure pass — after stitching,
  re-check connectivity and drop a **collision-checked** stitching via (or escape track) from each
  residual GND island to the main cluster, iterating until GND unconnected == 0 or no legal site
  remains. **Guard:** prior memory — GND thermal vias stamped through routed B.Cu caused shorts; any
  new via must keep the existing collision guard.
- **Confidence:** High diagnosis, medium fix. **Effort:** medium, careful.

### C2. Parent inter-leaf routing convergence  (~40% of unconnecteds)
- **Symptom:** cross-leaf signal nets left open by the parent freerouter (rs485 4 nets across 3 leaves;
  can-node CAN_RX; esp32 MOTOR_B2_DIR spanning ~76 mm). Parent phase reports `tier=not_routed`,
  best_score 0.00 every round; a dirty best-effort parent is promoted per the "show most-complete" policy.
- **Fix (surgical):** in `kicraft/cli/compose_subcircuits.py` + `kicraft/autoplacer/freerouting_runner.py`
  — (a) raise the **parent** `freerouting_max_passes` budget (default 20) for boards with high
  cross-leaf net counts; (b) add a parent-level rip-up/retry or escape-stub for still-unrouted
  inter-leaf nets before promote. Freerouting 1.9.0 is pinned and quirky on DSN keepouts, so bumping
  passes is the lowest-risk lever to try first.
- **Confidence:** Medium. **Effort:** medium.

### C3. Multi-connector same-edge packing
- **Symptom:** `connector_stranded` (5/9 batch-wide: edge-zoned connector landing inboard of its edge —
  can J2, esp32 J3) and `courtyards_overlap` (servo J18↔J16 adjacent headers; rs485 C8↔J3 decap on
  screw terminal). The new wrinkle vs prior connector-edge fixes is **many** connectors competing for
  one edge.
- **Fix:** extend the connector-edge family (`compose_subcircuits.py` `_ensure_edge_blocks_extremal` /
  `connector_edge_*` config + leaf companion clamp) with same-edge multi-connector packing/spacing, so
  N connectors on one edge are distributed instead of colliding.
- **Confidence:** Medium. **Effort:** medium (well-trodden area).

---

## Workstream D — Make the eval signal trustworthy (do alongside A)

Without this, future runs can't tell a real regression from sampling noise (see "noise floor"). All
knobs verified in source.

### D1. N-of-3 repeats per brief, report median + IQR  ⭐
- The only option that *measures and beats* variance. Today `self_eval.py compile_report` computes only
  `mean_final`. Add an outer repeat loop keyed per brief, aggregate **median**; a regression becomes
  "median dropped across N×2 runs." With N=3 the ±15 swings become legible. Cost ~3× (cheap run; build
  compute dominates — combine with D4).

### D2. Pin a stronger, separate judge model
- The judge is *also* deepseek-v4-flash, adding its own sampling noise to the class-J dimensions
  (electrical_soundness, intent_fidelity, part_selection). `judge_model` is already independently
  selectable (`--judge-model` / `eval_judge_model`). Point it at a stronger, steadier model — removes
  judge-side variance at no design cost. Lowest-cost win; do with D1.

### D3. Lower design temperature → 0 (+ seed)
- Design runs at temp 0.2 hardcoded (`kicraft/server/client.py`), no override. Add a configurable
  `design_temperature` (mirror the existing `review_temperature` in `config.py`), thread it through
  `stage_driver.py`, set 0; pass a `seed` if the OpenRouter deepseek route honors it. Partial — discrete
  topology/part swings won't fully vanish — but near-free variance reduction.

### D4. Use `replay` to attribute deltas (route-only A/B)
- `kicraft replay` (`cli_app.py:_cmd_replay`) re-runs place+route on a frozen `state.json` with no LLM,
  byte-deterministic via `_pin_deterministic_placement_env`. Doesn't help design variance, but lets us
  separate "synthesis changed" from "routing changed" when hunting a place/route regression — pair with
  D1 by replaying the same N synthesized states.

### D5. Infra resilience: retry transient OpenRouter errors
- #24 daq-8ch errored on a one-off **HTTP 503**. Add bounded retry/backoff on 5xx in the LLM client so a
  transient blip doesn't drop a brief from the batch (and doesn't masquerade as a design failure).

---

## Suggested sequencing

1. **Now (1 PR):** A1, A2, A4 — three confirmed source fixes + the honest label. Each recovers a known
   reproducible failure (audio-jack-buffer F, dual-rail ERC) at near-zero risk. Add A3 if quick.
2. **Next (1 PR):** D1+D2 (+D3, D5) — stand up trustworthy, de-noised eval before chasing softer
   regressions, so the A-fixes can be *measured*.
3. **Then:** B1 (MCU programming-path guarantee) — biggest remaining synthesis-correctness lever; B2.
4. **Ongoing, careful:** C1 → C2 → C3 — the place/route convergence work on `autoplacer/`; surgical,
   collision-guarded, re-verified per change. Highest effort, but the only path to lifting the
   `connector_dense_io` / `hi_pin_hierarchical` archetypes off 0/N fab-ready.

## Per-run evidence pointers (for the `kicraft-investigate` skill)
- A1: `logs/self_eval/20260624T120631Z/run_28_audio-jack-buffer`
- A2: `logs/self_eval/20260624T120631Z/run_18_dual-rail-supply`
- B1: `…/run_12_esp32-s3-sensor`, `logs/self_eval/20260623T210902Z/run_10_rp2040-min`
- B2: `…/run_11_fpc-breakout`
- C: `…/run_26_servo-driver-16`, `…/run_23_can-node`, `…/run_22_esp32-dual-motor`, `…/run_08_rs485-terminal`
- D5: `…/run_24_daq-8ch` (HTTP 503)
