# Self-eval 2026-07-27 batch — analysis & fix plan

## P4 RE-DIAGNOSIS + FIX (2026-07-28 session — critical analysis of this plan's own P4)

**P4 as written was stale and mis-scoped; the re-batch artifacts point at one
mechanism this plan never named.** Three findings against the plan itself:

1. **P4's board list is the PRE-fix batch.** usb-c-full-breakout ("the only
   remaining illegal_routed_geometry emitter") and stm32-min are **fab-ready in
   the re-batch** this same document reports; the actual rc7 set is run_10, 12,
   22, 23, 24, 26, 27, 30 (+ rc6 run_13, run_25). A plan that names target
   boards must be refreshed from its own re-batch table.
2. **"Dense-SoC escape residue" is no longer the owning class.** The dense-SoC
   leaf work *worked*: on run_10 the leaves are 5/5 accepted in every round.
   The residue moved UP: all 8 rc7 boards fail at the **parent** on
   **interface pads** — a pad whose net crosses leaves has no on-leaf partner,
   so leaf routing lays **zero copper** on it (verified: 25/26 bare GPIO
   interconnect pads on run_10's accepted MCU leaf; CAN_TX/RX bare on both
   run_23 leaves; SDA/SCL/ALERT_* bare on run_24's ADC leaves; IO4/IO7 bare on
   run_30; USB_DN/DP + I2C bare on run_12's MCU leaf — while every
   interconnect net that happened to have 2 on-leaf pads, e.g. run_27's MS1,
   has copper). By parent time the bare pad sits behind the pin-adjacent
   companion wall (the connectivity-first placement win made this structurally
   worse) plus the leaf's locked traces: FreeRouting abandons it and the
   repair pass reports `no_clear_path` on every edge (run_10:
   `GPIO12:U1.15->J2.13:no_clear_path` × 26). The source is a documented,
   now-falsified assumption in `auto_signal_escape_specs`: *"single-pad signal
   nets … close at compose"* and *"ICs route fine without them"* — and the
   escape planner deliberately scopes to non-outer-row pads, whose `open`
   verdict ("router's job") is true at leaf scope and false at parent scope.
3. **A second, compounding budget bug this plan's P3.2 didn't reach:** the
   parent attempt-1 probe cap (`parent_gnd_plane_probe_timeout_s`, flat 120 s)
   clamps BELOW the dense-interconnect budget the scaler just computed (180 s
   for run_10's 39 nets). A timeout-killed FreeRouting that already wrote a
   partial SES does not raise, so the GND-skip fallback never fires and the
   round ends 26-30 unconnected. P3.2 fixed crash *pricing*; this is crash
   *budgeting*.

**Fixes implemented (branch `placement-streamline`):**

- **Interface escapes** (owning fix, leaf-side): new
  `escape_planner.plan_interface_escapes` — pure geometry, sweeps rays against
  the PLACED environment (every other pad as obstacle, footprint bodies as
  dead-end zones for endpoints), dog-bone fanout via fallback for fully walled
  pads (run_10's GPIO21/22/23 class), honest `infeasible` (never a nub) —
  plus `breakout_stubs.interface_escape_specs` (SMD-only, dense footprints
  >= `interface_escape_min_pads`, single-pad interface nets only, GND/power
  pour nets excluded) wired into `leaf_routing` after the escape planner with
  the same no-silent-handoff stamp-skip recording; the per-pad verdict
  travels in the leaf validation record as `interface_escapes`. Kill switch
  `interface_escape_enabled` (default on). On run_10's MCU leaf: 35/35
  cross-leaf pads escape (27 ray + 8 via at the tight canvas).
- **Probe scaling**: `_compose_route._probe_timeout_s` — attempt-1 cap =
  `min(route_budget, max(120, n_interconnect * parent_probe_s_per_interconnect))`
  (default 5 s/net), so a hang is still caught fast on sparse parents and a
  39-net fan-out is no longer starved.
- Tests: `tests/test_interface_escapes.py` (12: geometry verdicts incl. the
  walled/sideways/via/infeasible/courtyard/THT-exclusion cases + board-level
  spec gen) and 4 probe-cap tests in `tests/test_parent_route_budget.py`.
  Leaf-level A/B at rounds=1 seed=0 (same command, kill-switch off vs on):
  baseline 4 unconnected / no accepted round, fix 3 / no accepted round —
  parity within noise, with all 35 pads now carrying escape copper.

**$0 A/B replays (same code both sides; baseline = kill switches in the
project autoplacer.json; `--quality good --seed 0`, one replay per side, both
sides measured by ONE script):**

- **run_23 can-node (the 1-2-unc tail class): flipped decisively.** Baseline:
  parent route succeeded 1 of 3 parent-phase rounds (rc0 only on the last
  try); every CAN_TX/CAN_RX interface pad bare on both leaves. Fix: parent
  route **3 of 3**, rc0, shorts=0 unconnected=0; every CAN interface pad
  carries exactly one escape track. The batch's rc7 was this coin landing
  tails — with the escapes it stops being a coin.
- **run_10 rp2040-min (the 26-30-unc hard case): mechanism moved, board not
  yet flipped.** Baseline round-1 board: copper on **2** of 35 interconnect
  nets. Fix round-1 board: copper on **all 35** (598 segs, 61 vias, 0 bare
  pads on the MCU leaf) but 27 still open — FreeRouting engaged every net and
  was killed mid-route at the 180 s dense floor (both A/B sides were also
  clipped by the harness' 25-min timeout after 1-2 parent rounds; the batch
  itself gave this board 6 failed parent rounds, so the baseline verdict
  stands). Bottleneck moved from *reachability* to *budget* → third fix:
  `parent_s_per_interconnect` (8 s/net above the dense threshold; 39 nets →
  312 s, <= 22-net parents unchanged). Untimed re-replay pending below.

- **run_10 untimed full-stack replay (all 3 parent rounds, 312 s budgets):
  still rc7 — 29 unc, shorts=0.** Every round: FreeRouting engages all
  interconnects, completes ~10, abandons 23-29 (`no_clear_path` at repair on
  every edge; now even VBUS/+3V3/GND appear among the opens). With
  reachability (escapes) and budget (scalers) both removed as causes, what
  remains is the **composition**: 26+ GPIO nets must flow from the MCU leaf
  across the board into the 40-pin header leaf's wall, with the QSPI/USB/LDO
  leaves in between — a corridor/mouth-alignment problem. Owner: the open
  **N5b compose-level mouth-line alignment** workstream (place the MCU leaf's
  GPIO mouth adjacent and pin-order-aligned to the header leaf), not more
  routing budget and not an in-house router. The fix stack stays net-positive:
  the tail class (can-node genre — also run_24/27/30's 1-9-unc shapes by
  mechanism) flips, and run_10's failure is now attributable per pad from the
  leaf record.

### Post-deploy batch 20260728T120442Z (34 briefs, $0.99, 5.6 h — first batch on `2d6329e`)

| | 7-20 baseline | 7-27 re-batch | **7-28 post-escape** |
| --- | --- | --- | --- |
| fab-ready | 24/34 | 22/34 | **25/34 (new high)** |
| mean / median | 75.3 / — | 75.8 / 78.8 | 73.7 / 75.8 |
| grades | 23B/9C/2D | 26B/4C/4D | 21B/10C/2D/1F |
| observer gates | 1 | 2 (legit) | 1 (run_13, dead build) |
| synthesis deaths | 0 | 1 | **3** (13, 15, 18 — model-side) |

**8 of the re-batch's 12 open boards flipped to fab-ready** (12, 22, 23, 25,
26, 28, 29, 30) — six of them the interface-escape class exactly. Still open:
run_10 (22 unc GPIO fan-out — N5b compose alignment, as predicted), run_24
(9→5 unc; VBUS/GND pour + 2 signals), run_27 (2 unc, on-leaf-partnered nets),
run_13 (moved to a synthesis death). New downs, all attributed: run_06 rc6 =
USB-C receptacle footprint geometry at a 0.153 local override whose B-row
pads need 0.127 (the escape plan's named "untested risk" knob; the stamped
rays appear ONLY as expected dangling warnings, not in any error), run_09
2 unc = J1.A6/A7 declared honestly `infeasible` by the planner (per-pad
attribution working), run_14 1 unc GND = the known pour/strand class. The
mean dip is the three deaths (model variance; the death briefs are ~$0.10 to
re-drive); place/route pre-gate quality is flat-to-up. ≥26/34 target missed
by one. Next owners: fine-pitch local-override selection (run_06+run_09, one
mechanism), N5b mouth alignment (run_10), GND strand (run_14), reconcile
advancing-chain + crystal deterministic-donor (the deaths).

## RESULTS (implementation, 2026-07-27 — updated as items land)

- **P0 SHIPPED.** Judge contract now demands per-gate `triggered: true|false`;
  `_validate` keeps only affirmed gates, screens legacy self-negating evidence
  (`_GATE_NEGATION_RE`), and records screened entries under
  `gates.observer_rejected` in report.json. 7 new polarity tests (verbatim
  run_17/run_34 payloads). **$0 re-score of the batch confirms surgical
  precision: exactly run_17 (D 55 → B 80.5) and run_34 (D 50 → C 73)
  screened, nothing else touched. Corrected batch: mean 69.5, 16B/8C/10D.**
- **P1 SHIPPED.** Kind-then-value supplement regex (`_PASSIVE_KIND_VALUE_RE`,
  ask-verb-sentence-gated so descriptive prose can't provision duplicates),
  replacement-aware `bom_reconcile_instruction` (drops the "do NOT drop"
  framing when the deficit names a swap), per-deficit stuck detection
  (`_deficit_key` = refs+kinds, stable across rewordings; a changed-nothing
  pass on deficit A no longer denies deficit B its budget), and a pointed
  retry when a pass changed the BOM without resolving the same deficit.
  `test point/pad` + `screw terminal` added to the non-passive remainder
  list (§5.8). 11 new tests on the verbatim batch deficit texts.
- **P2 SHIPPED.** `bom.substitutions` ledger (models.py `Substitution`);
  §9.33 spec-named-MPN accountability + §9.34 brief-stated mount type, both
  HARD at BOM commit (cli_app identity_checks); STM32 BOOT0 strap rule in
  §9.29 `_family_strap_gaps` (run_24's exact gap, now a BOM-commit retry);
  `mcu_programming_facts` + SUBSTITUTIONS/PROGRAMMING-PATH lines in
  `build_run_digest`; §9.26 rejection messages point at the ledger; bom.md
  spec + worked example updated. 30 new tests.
- **P3.1 ROOT-CAUSED + FIXED.** The round-led-ring "crash" is FreeRouting
  1.9.0 **hanging forever when a net's locked wiring forms a closed loop**
  ("The normalization of net '5V' failed." then silence until the rc=-1
  watchdog kill). The LED-ring leaf routed its 5V bus as a closed ring;
  bisection over the 31 locked 5V wires pinned one loop-closing segment.
  Fix at source: `_break_locked_wire_cycles` DSN sanitizer (opens each loop
  with a width/10 gap in FreeRouting's view only; the .kicad_pcb copper is
  untouched; removing one edge of a cycle never disconnects the net).
  **Validated end-to-end: the exact poisoned DSN routes in 18 s with a SES
  (was 397 s of hangs → rc6).** Fixture `tests/data/fr_hang_5v_loop.dsn`
  (verbatim) + 4 tests.
- **P3.2 SHIPPED.** `observe_duration(crashed_route_s=...)` half-discounts
  infra-crashed route time in the wall-budget EMA; autoexperiment detects the
  no-SES/crash signature and passes the route seconds. 2 new scheduler tests
  (the run_29 numbers verbatim: one more round now runs, a second crash
  exhausts honestly).
- **P5.1 SHIPPED.** ESP32-S3-MINI-1 pins 23/24 renamed `IO19/USB_D-` /
  `IO20/USB_D+`; bundle re-validated, content_hash updated.
- Suite: full run green except the 6 documented pre-existing reds
  (test_build_zero_leaf ×3, fine-pitch USB-C param, provider_bench plotly,
  lookup_lcsc easyeda fall-through — the last reproduced on clean HEAD).
- **P3.1 end-to-end replay PROOF:** full $0 replay of run_29 with the cycle
  guard — every parent round now routes (`parent_route=ok` ×3, best 56.32,
  shorts=0 unconnected=0, 237 traces) where the batch had 100% route
  failure. The board remains not-fab-ready on a DIFFERENT, pre-existing
  gate: outline-shape conformance (requested ⌀60 mm circle; the ring leaf's
  ~58×58 mm bbox circumscribes 86 mm → rect fallback). Round-led-ring was
  conformant at the 7-20 baseline, so the ring-leaf growth is a suspect for
  the uncommitted placement work — run_29 added to the A/B evidence set.
  (Also note for the shaped-nesting workstream: circumscription is computed
  from bbox CORNERS, which is maximally pessimistic for ring-shaped content
  whose actual copper sits at ~radius+LED size from center.)
- **P3.3 DONE — no systematic regression; branch committed.** A/B replays
  (same frozen project, seed 0, sequential; B = clean 1638a27 via the
  editable-finder stub): servo-driver-16 **2 vs 5** unconnected (working
  tree BETTER), esp32-s3-sensor 1 vs 1 (tie), can-node 1 vs 0 (one-net cost
  at this seed). The batch down-flips were design variance. run_29 shape
  check: clean HEAD **also** rect-falls-back (63.5×64.6 vs A's 64.4×63.8) —
  the ⌀60 mm non-conformance is the synthesized design's ring size, owned by
  the shaped-nesting workstream, not the placement work. Extra P3.2
  corroboration: on HEAD the hung parent round 1 priced out rounds 2-3,
  which route fine when allowed to run. Committed as `62c4738` (placement
  wave) + `a48563d` (this fix wave).
- **P1/P2 re-drive VERIFIED (3 briefs, $0.18):** all three previously
  reconcile-dead briefs now SYNTHESIZE (was 3/3 deaths), zero observer
  gates, and the events show the new deterministic kind-then-value path
  working ("deterministically provisioned R2 ... no model BOM pass" on
  esp32-dual-motor's FB-resistor ask; daq-8ch's BOOT0 story resolves at BOM
  commit under the new §9.29 rule instead of dying in reconcile).
  rs485-terminal: fab-ready B/78 (baseline parity). esp32-dual-motor B/77
  and daq-8ch C/68.5 route but keep 2 / 4 unconnected — the known
  dense-board residue class, not reconcile. Synthesis-convergence target
  met (3/3); fab-ready 1/3 (residue owns the rest).
- **Full re-batch DONE (`logs/self_eval/20260727T151918Z`, $1.05, 5.7 h):**

  | | 7-20 baseline | 7-27 pre-fix | 7-27 post-fix |
  | --- | --- | --- | --- |
  | fab-ready | 24/34 | 22/34 | 22/34 |
  | mean / median | 75.3 / — | 68.1 / 72.8 | **75.8 / 78.8** |
  | grades | 23B/9C/2D | 15B/7C/12D | **26B/4C/4D** |
  | observer gates | 0 | 10 (2 false) | **2 (both verified legit)** |
  | reconcile deaths | 0 | 3 | **0** |

  Targets: mean ≥74 **HIT** (75.8, above baseline); 0 false gates **HIT**
  (both applied gates verified: speaker-crossover's 39 µF/680 µH defaults
  give a ~few-Hz crossover vs the asked 2 kHz; usb-c-full-breakout's 16-pin
  receptacle physically lacks the SuperSpeed pairs the brief demanded);
  ≤2 legit observer gates **HIT**; ≥26/34 fab-ready **MISSED** (22).
  The whole gate architecture is working in the wild: 6 judge-refuted gates
  recorded under `observer_rejected` with rationales quoting the new
  SUBSTITUTIONS/PROGRAMMING-PATH digest lines ("explicitly recorded in
  bom.substitutions ... not silent", "MCU programming path computed PASS
  with J2 (SWD 4-pin header)"); dual-rail-supply — the pre-fix poster child
  — ledgered its 1A→125 mA converter swap and scored B/75 instead of D/55;
  unprogrammable_mcu went 4 → 0.

  The fab-ready gap is now cleanly place/route-owned: 8 DRC-residue boards
  (shorts=0 everywhere; rp2040-min 30 unc is the dense-SoC hard case,
  the rest 1-9 unc), 2 leaf-acceptance/parent-route failures (nrf52-beacon,
  gpio-expander — no crash signatures, ordinary variance), 1 ERC flap
  (audio-jack-buffer), 1 synthesis death (round-led-ring §9.17 two-terminal
  self-short — an honest early death, new variance class). Owners: the P4
  dense-escape workstream (§ above) + the 1-3-unconnected tail as the
  cheapest next fab-ready yield.

  Commits: `62c4738` (placement wave) + `a48563d` (fix wave), branch
  `placement-streamline`, NOT pushed / NOT deployed (deploy = restart both
  web + build worker when it goes live).

Batch: `logs/self_eval/20260727T045000Z` (34 briefs, design=deepseek-v4-flash,
judge=minimax-m3, wall 6.5 h, $0.93). Ran on branch `placement-streamline`
**including the uncommitted** dense-SoC escape + placement-streamline work.
Baseline: `logs/self_eval/20260720T113207Z` (same models, same rubric — rubric.yaml
unchanged since Jul 7, eval code unchanged since PR-F/PR-G which BOTH batches ran).

## Headline

| | 7-20 baseline | 7-27 this batch |
| --- | --- | --- |
| fab-ready | 24/34 | **22/34** |
| grades | 23B / 9C / 2D | 15B / 7C / **12D** |
| mean final | 75.3 | 68.1 (pre-gate weighted mean: 73.5) |
| gates | erc_errors ×1 | **silent_substitution ×6, unprogrammable_mcu ×4** |
| errored | 0 | 0 (but 3 synthesis deaths graded as failed) |

The D explosion is **gates, not scores**: 10 of 12 D's are gate-capped runs whose
pre-gate weighted scores average 71.2 (would have been C/B). The fab-ready drop is
3 synthesis deaths (BOM reconcile non-convergence) + 3 place/route down-flips,
partially offset by 4 up-flips (usb-pd-trigger, lora-node, encoder-oled-panel,
audio-jack-buffer — the latter +26 pts, ERC gate fixed).

## Root causes (5 classes, all evidence-verified in the run dirs)

### Class 1 — EVAL BUG: polarity-blind observer-gate acceptance (2 false D's)

`kicraft/eval/judge.py:143-169` `_validate` counts ANY entry the judge lists in
`triggered_gates`, without checking the evidence affirms the gate. minimax-m3
sometimes enumerates gates with a *verdict* in the evidence field:

- run_17 led-cc-driver, gated silent_substitution, evidence: *"No named parts were
  specified by the user, so there is nothing to silently substitute against …
  recorded openly."* → capped 80.5 → 55 (B → D).
- run_34 snowman-ornament, gated unprogrammable_mcu, evidence literally ends
  *"Gate does not trigger."* → capped 73.0 → 50 (C → D).

The prompt (`judge.py:63`, `:78`) says "list ONLY with concrete evidence" but the
output contract has no per-gate boolean, so a compliant-looking enumeration
passes `_validate` untouched.

### Class 2 — BOM reconcile non-convergence (3 dead builds: rs485-terminal,
### esp32-dual-motor, daq-8ch — all were building at baseline, two were B/80)

All three died identically: `unresolved BOM deficit after 3 reconcile pass(es)`
with the stuck-loop detector (`server/session.py:603-617`) firing on a pass that
recommitted a byte-identical BOM. Four distinct code-side flaws compound:

1. **`_PASSIVE_ASK_RE` (session.py:307-312) only matches value-then-kind.**
   run_22's ask was *"Add a bottom resistor (e.g., 3.9k, typical for 3.3V
   output)"* — kind-then-value with an `(e.g., …)` — no match, so the trivially
   satisfiable single-resistor ask fell through to the LLM pass, which failed
   twice.
2. **`bom_reconcile_instruction` (session.py:280-287) is add-only and actively
   forbids replacements.** run_08's deficit was *"Replace U1 (ADuM1301ARWZ) with
   two ADuM1201ARZ …"* while the instruction commands "do NOT drop any part
   already present — just provision what's missing". The model obeyed the
   instruction over the deficit and recommitted unchanged. (Upstream trigger:
   deepseek picked ADuM1301 then claimed its channels are unidirectional — LLM
   knowledge noise we can't fix, but the harness must be able to *execute* a
   replacement ask.)
3. **Stuck-loop detection exhausts the whole budget even when the NEXT park is a
   NEW deficit.** run_24's chain was: terminals deficit (pass 1, changed
   something) → terminals again (pass 2, BOM signature unchanged → passes set to
   MAX) → wiring re-parked on a *brand-new* deficit ("10k resistor between BOOT0
   and GND" — a clone-donor ask `apply_deterministic_bom_adds` would have
   satisfied from existing R5) → budget already burned, run dies. A changed-nothing
   pass on deficit A must not deny deficit B its deterministic attempt.
4. **No post-pass satisfaction check.** run_22's LLM pass committed "ok" twice
   without adding the asked-for FB resistor; nothing verifies the specific ask
   was met before re-driving wiring.

Secondary finding from run_22: wiring repeatedly asked (rt=None) how to wire USB
because **the ESP32-S3-MINI-1 symbol exposes no USB D+/D− pins** (module GPIO19/20
are USB) — parts-library symbol gap worth its own check.

### Class 3 — real silent substitutions (synthesis gap; 3 clear + 2 borderline)

The judge caught genuine unsurfaced spec→BOM deviations:

- run_18 dual-rail-supply: architecture named RECOM RP12-2412DA (~500 mA/rail),
  BOM shipped WRA2412S-3WR2 (~125 mA/rail) — a capability downgrade, unsurfaced.
- run_20 encoder-oled-panel: brief said **SMT** I2C OLED; BOM footprint is
  `OLED-TH_…` through-hole. Mount-type contradiction with the user's own words.
- run_5 usb-pd-trigger: spec assumption "3-position rotary switch (defaulted)",
  BOM ships an SP3T slide switch; no assumption delta recorded.
- run_23 (borderline): symbol field says STM32F103C8T6, committed MPN CBT6
  (different flash density) — metadata inconsistency more than a downgrade.
- run_32 (borderline): connector switched after a commit rejection; the note
  exists in BOM assumptions but never surfaced as a user-visible delta.

Common shape: the BOM stage deviates from an architecture-named part or a
brief-stated attribute and nothing forces the deviation into a surfaced,
user-visible record. The gate condition is "…without surfacing it" — surfacing
is the fix, not banning substitution.

### Class 4 — MCU programmability: one real gap + two judge over-fires

- **Real (run_24 daq-8ch):** STM32F042 intended to program via native-USB DFU,
  but no BOOT0 strap/jumper/TP in BOM. §9.29
  (`design/synthesis/validation.py:1783` `check_mcu_programming_access`) has
  family strap requirements for **RP2040 and ESP32 only** — STM32
  DFU-via-BOOT0 isn't covered, so the gap surfaced late as a wiring deficit
  (which then died in Class 2).
- **Over-fires (run_10 rp2040-min, run_31 chamfered-badge):** run_10 HAS a
  BOOTSEL button + USB (the ROM UF2 path §9.29 explicitly accepts; the judge's
  own evidence says "BOOTSEL/reset buttons present"); run_31 HAS a UPDI TP pad +
  a recorded architecture assumption (§9.29 deliberately accepts TP pads). The
  judge re-derives programmability from a digest that never mentions the §9.29
  verdict, and guesses harsher than the deterministic check.

### Class 5 — place/route

- **round-led-ring (rc6, was fab-ready at baseline): NEW FreeRouting crash
  class.** Every invocation — power-first phase, single-phase fallback, GND-skip
  retry — died with `FreeRouting crash (rc=-1) … no SES output after 2 attempts`
  (`autoplacer/freerouting_runner.py:1547` retry loop). NOT the known Ω-DSN
  deadlock: the stamped parent has zero non-ASCII bytes. The crashed round
  burned 398 s, then the wall-budget estimator (`cli/_round_scheduler.py:188`)
  extrapolated "next round ≈ 398 s > 648 s budget" and finalized after ONE
  round — a crash-priced estimate denying all retry rounds. Compose itself was
  fine (144/144 child traces preserved; shape fit legitimately rejected
  86 mm ⌀ ring > 60 mm request, rect fallback).
- **Down-flips vs baseline (unconnected):** servo-driver-16 0→7, can-node 0→2,
  esp32-s3-sensor 0→1. Prime suspect: the uncommitted placement-streamline /
  escape-planner changes this batch ran on (grid legality clearance, escape
  loop). Needs $0 A/B replay before committing.
- **Persistent dense-escape class (pre-existing, owned by the active dense-SoC
  workstream):** rp2040-min 24→29 unc, usb-c-full-breakout 5→6 unc (+
  `illegal_routed_geometry`, `cli/_compose_route.py:446`), nrf52-beacon 6→5,
  stepper-a4988 10→4, stm32-min 1→1.

---

## Fix plan

### P0 — eval integrity: affirmative gate contract (small, do first; 2 D's are false)

**Files:** `kicraft/eval/judge.py`, `tests/test_judge_parsing.py`,
`kicraft/eval/rubric.yaml` (prompt text only, no cap/id changes).

1. Change the output contract (`_output_contract`) to
   `"triggered_gates": [{"id", "triggered": true|false, "evidence"}]` and the
   prompt line to: *"For each observer gate, include it ONLY if it fires; if you
   mention a gate at all, set `triggered` explicitly."* In `_validate`, keep only
   entries with `triggered is True` (accept legacy entries lacking the key for
   robustness, see 2).
2. Defense-in-depth negation screen in `_validate` for entries without an
   explicit `triggered: true`: drop when evidence matches
   `(?i)\b(does not|doesn't|not) (trigger|hold|apply)\b|nothing to .*substitute|no named part`
   and record it under a `gates.observer_rejected` list in report.json so a
   screened gate is visible, never silently discarded.
3. Unit tests: the exact run_17 and run_34 payloads (verbatim evidence strings)
   must produce zero applied gates; an affirmative payload must still cap.
4. Re-score the 10 gated runs of THIS batch offline from their saved digests
   (judge re-call ≈ $0.02) to get corrected batch numbers; append a RESULTS
   section here.

**Acceptance:** re-scored batch shows no gate whose evidence self-negates;
run_17 → B, run_34 → C.

### P1 — BOM reconcile convergence (3 dead builds; highest fab-ready yield)

**Files:** `kicraft/server/session.py`, tests (new
`tests/test_bom_reconcile_asks.py` or extend existing session tests).

1. **Widen `parse_passive_deficits`:** add a kind-then-value alternative to
   `_PASSIVE_ASK_RE` covering "resistor (e.g., 3.9k…)", "resistor of 10k",
   "resistor, 3.3k to 4.7k" (take the first value of a range). Keep the
   confidence rule: no parse → LLM pass. Unit-test with the three verbatim
   deficit texts from this batch (run_22 FB resistor, run_24 BOOT0 10k,
   run_08 decap sentence).
2. **Replacement-aware instruction:** in `bom_reconcile_instruction`, detect
   `(?i)\breplace\s+(?P<ref>[A-Z]+\d+)\b.*\bwith\b` in the deficit text. When
   present, emit a replacement variant: *"Apply the replacement described below
   exactly: remove ONLY the named part(s), add the named successors (fresh refs,
   correct value/footprint/sheet, ic_groups updated), keep every other part."*
   Only blanket-forbid drops in the pure-add variant.
3. **Per-deficit stuck detection, not global:** carry the last deficit text (or
   a normalized hash) alongside the BOM signature. A changed-nothing pass only
   exhausts the budget when the CURRENT deficit is the same as the one that
   produced no change; a new deficit re-enters the loop (and gets its
   deterministic-add attempt) as long as `bom_passes < MAX`. This directly
   rescues run_24's clone-R5 BOOT0 ask.
4. **Post-pass satisfaction check (cheap, deterministic):** after an LLM
   bom+wiring pass, re-run `parse_passive_deficits` against the new BOM: if the
   pass "changed something" but the parsed asks are still unmet, retry ONCE with
   a pointed instruction naming exactly the missing part(s) ("the BOM still has
   no bottom FB resistor; add R_bottom 3.9k on the BUCK 3V3 sheet") before
   burning a full pass. Unparseable asks keep today's behavior.
5. **Verification:** unit tests for 1-4 (fake-client harness); then re-drive the
   three dead briefs only:
   `python -m kicraft.eval.self_eval --only rs485-terminal,esp32-dual-motor,daq-8ch --out logs/self_eval/<new>` (~$0.10).
   Target: 3/3 synthesize; ≥2/3 fab-ready.

### P2 — surface substitutions + close the STM32 DFU strap gap (kills the two
### legit gate classes at the source)

**Files:** `kicraft/design/synthesis/validation.py`, BOM stage prompt/schema
(`design/` synthesis stage definitions), `kicraft/eval/run_web.py`
(`build_run_digest`), web stagetabs (display only, optional).

1. **Substitution ledger:** add optional `bom.substitutions:
   [{wanted, got, reason}]`. New deterministic check at BOM commit: scan
   architecture + functional_spec text for MPN-like tokens (regex for
   part-number shapes, e.g. `[A-Z]{2,}[0-9]{2,}[A-Z0-9-]*`, filtered against the
   BOM's own MPNs); any spec-named MPN absent from the BOM without a ledger
   entry → commit retry error naming it ("architecture names RP12-2412DA; BOM
   ships WRA2412S-3WR2 — record it in bom.substitutions with a reason, or use
   the named part"). This converts run_18/run_23 into surfaced deltas.
2. **Mount-type lint:** when the brief/spec says SMT/SMD (resp. through-hole)
   for a named block and the chosen footprint contradicts it
   (`_TH`/`THT`/`P2.54` heuristics vs SMD footprint families), require a ledger
   entry the same way (run_20).
3. **Commit-rejection substitutions** (run_32's connector swap after a 9.26
   retry): when a BOM retry changes an MPN for a part class the user or spec
   named, auto-append the ledger entry (reason = the rejection) rather than
   relying on the model to remember.
4. **STM32 DFU strap requirement in §9.29:** for STM32-family MCUs whose only
   programming story is native-USB DFU (USB part present, no SWD access part),
   require a BOOT0 access part (strap resistor/jumper/TP) at BOM commit, same
   pattern as the existing RP2040 BOOTSEL / ESP32 BOOT+EN rules
   (validation.py §9.29 "family strap requirements"). This turns run_24's
   late wiring deficit into an early, reconcile-free BOM retry.
5. **Digest surfacing (feeds P0):** in `build_run_digest`, add two short
   sections — `SUBSTITUTIONS:` (the ledger, or "none recorded") and
   `PROGRAMMING PATH:` (the §9.29/§9.21 verdict: which access parts satisfied
   it, e.g. "RP2040: BOOTSEL button SW2 + USB → ROM UF2; SWD on J3"). The judge
   then grades from facts instead of re-deriving; run_10/run_31-style over-fires
   lose their footing, and surfaced-but-suboptimal substitutions score on the
   quality dimensions instead of a cap.
6. **Verification:** offline `synthesize` subcommand on the five affected briefs
   (replay can't verify synthesis changes — memory: KC-UFHJ42); assert ledger
   entries/commit-retries appear. Unit tests for the MPN scanner + mount-type
   lint with this batch's verbatim spec/BOM values.

### P3 — place/route regressions (this batch's new failures)

**Files:** `kicraft/autoplacer/freerouting_runner.py`,
`kicraft/cli/_round_scheduler.py`, uncommitted work on branch.

1. **round-led-ring FreeRouting rc=-1 (investigate first, then fix):** replay
   `run_29` ($0, frozen seed) and capture the freerouting JVM
   stdout/stderr/log for the parent DSN; classify: JVM crash vs our
   per-attempt watchdog kill. The board is small (24 comps, 3 interconnect
   nets, 144 pre-routed child traces) so 130 s+ per attempt is itself the
   anomaly. Known non-cause: non-ASCII DSN (checked, clean). Suspects, in
   order: (a) preserved-wiring DSN sections from the ring leaf's 100 traces
   (power-first `_restrict_dsn_routing_to_nets` keeps wiring — interaction
   with a wiring-heavy, hole-free rect board), (b) a degenerate geometry from
   the rect-fallback outline + ring-shaped courtyard union, (c) memory/JVM
   flags. Fix at the source once classified; add the failing DSN as a
   regression fixture next to the Ω one.
2. **Crash-aware wall-budget estimate:** in `_round_scheduler.py` (both
   estimate sites, :188 and :243), when the previous round's parent route
   FAILED with the infra signature (no SES / rc=-1 crash), do not use its
   duration as the next-round estimate — use the last *successful* round's
   duration or the pre-crash average. Rationale: a crashed 398 s round priced
   the estimator out of ALL remaining rounds on a 648 s budget; a placement
   mutation may well route in ~90 s. Unit-testable via the scheduler's
   injected callables.
3. **A/B the uncommitted placement work before committing it:** replay
   servo-driver-16, can-node, esp32-s3-sensor from THIS batch (frozen seeds,
   $0) twice — working tree vs `git stash` — same-script measurement per run
   (memory: never compare across separate replay invocations for metrics; here
   we compare verdict/unconnected of the same frozen project under two code
   states, which is the standard verify loop). If the uncommitted changes own
   the 0→7 / 0→2 / 0→1 flips, fix or gate them before commit; if not, the
   flips are design-variance (different synthesized netlists batch-to-batch)
   and the dense-escape workstream owns them.
4. **Commit hygiene:** the batch measured an uncommitted tree. After 3's A/B,
   commit the placement-streamline + escape-planner work (it is otherwise
   unprotected), then deploy both services per the standard two-service rule
   when it goes live.

### P4 — persistent dense-escape residue (existing workstream, re-scoped, not new)

rp2040-min (29 unc), usb-c-full-breakout (6 unc + illegal_routed_geometry),
nrf52-beacon (5), stepper-a4988 (4), stm32-min (1) stay with the active
dense-SoC escape plan (`docs/plans/dense-soc-escape-routing-plan.md`) — this
batch adds fresh evidence that the failure mode is unchanged (ordinary router
residue, shorts=0 across the board). Two additions from this batch:

1. usb-c-full-breakout is the only remaining `illegal_routed_geometry` emitter
   (`cli/_compose_route.py:446`) — pull its validation record during the next
   escape-plan iteration; the KC-69TGAP leaf-verdict fix removed this class for
   the beacon, so whatever geometry survives here is a parent-level variant.
2. The 1-2-unconnected tail (stm32-min, esp32-s3, can-node) is the cheapest
   remaining fab-ready yield (+3 boards) and per the no-in-house-router
   principle must come from placement/escape changes, not routing patches.

### P5 — small follow-ups

1. **ESP32-S3-MINI-1 symbol: expose USB D+/D− (GPIO19/20)** in the parts
   library (run_22's wiring interrogated it 3×; costs retries on every ESP32-S3
   USB brief). Re-run `validate-part --update-hash` after the edit (vendored
   hash gate).
2. Judge-noise note: B-run deltas of ±3-6 pts (rc-lowpass −5.5, fpc −5.5,
   highside −6.5, star −9) are within the documented single-run noise floor —
   no action, N-of-3 before believing any of them.
3. After P0-P2 land: full re-batch (~$1, ~6 h). Targets: ≥26/34 fab-ready,
   0 false gates, ≤2 legit gates, mean ≥74.

## Sequencing

P0 (hours, eval-only, no service restart) → P1 (day, session.py + tests,
restart web for live reconcile parity) → P2 (1-2 days, synthesis validation +
digest) → P3.1-3.2 (investigation + scheduler fix) and P3.3-3.4 (A/B, then
commit the branch) can run parallel to P1/P2 → P4 continues as the standing
workstream → P5 opportunistic. Re-batch after P0-P2.

## Cost note

Full batch $0.93; per-brief re-drives ~$0.03. P0's re-score of 10 digests
~$0.02. All place/route verification is $0 replay.
