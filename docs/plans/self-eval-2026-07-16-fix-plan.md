# Self-eval 2026-07-16 fix plan — batch `20260716T011056Z`

**For the implementing agent.** Self-contained: every finding names its evidence dir, the
owning file:line, and a $0 verification where one exists. Prior workstream numbering:
`docs/plans/self-eval-2026-07-11-fix-plan.md` (N1–N12); this plan starts a new S-series.

## 0. Scorecard and what this batch proves

| | baseline `20260713T051225Z` | this batch `20260716T011056Z` |
|---|---|---|
| fab-ready | 20/34 | **22/34** (target was ≥24; rect 17/27, shaped 5/7) |
| mean / median | 70.9 / — | **73.6 / 74.5** (34/34 graded) |
| grades | — | B:16 C:15 D:3 |
| gates | — | unprogrammable_mcu×1 (run_10), silent_substitution×1 (run_20) |
| spend | — | $1.11 |

Code provenance: committed **`302d08d`**, branch `placement-streamline`, clean tree for the
whole batch (no mid-batch contamination). Judge `minimax/minimax-m3`, design
`deepseek/deepseek-v4-flash`, `--parallel 3 --build-slots 1`.

### Batch incident: lifetime spend ceiling reached mid-batch

`BudgetExceeded: total spend $50.0019 >= ceiling $50.00` killed runs 28–34 on first pass.
This is the **lifetime** cap (`Settings.total_usd_ceiling`, `kicraft/server/config.py:72`)
in the shared ledger `~/.kicraft/spend_ledger.db` — NOT the daily cap ($5 default / $20 in
`.env`; today's eval spend was only ~$1). Crossing it also blocks **the live site** (web +
build worker preflight the same ledger). The batch was finished via `--resume` with a
process-scoped `KICRAFT_TOTAL_USD_CEILING=52`. Operator actions in S10.

### Wins vs baseline (fixes that held)

- run_08 rs485-terminal: ERC 45 → **fab-ready B 81.5** (the 07-13 RS-485 ERC D-class is gone).
- run_19 relay-quad & run_20 encoder-oled-panel: dead (rc=1 / unprogrammable_mcu reconcile
  starvation) → **fab-ready** (N3 bounded-reconcile `4d87b74` held; run_20 now gated D on an
  honest substitution instead of crashing).
- run_10 rp2040-min: dead → routed (rc7, 6 unconnected) — progress, not yet fab-ready.
- run_14 lora-node, run_23 can-node: DRC → **fab-ready**.
- run_30 rounded-c3-devboard: DRC → **fab-ready B 76.5**; shaped runs 31–34 all held fab-ready
  (31 B 76.5, 32 B 83.5, 33 C 67.5, 34 B 82.0).
- Zero shorts across all 34 boards; zero ERC-blocked briefs; zero synthesis crashes.

### Regressions vs baseline (all single-run — see noise caveat)

- run_15 buck-3a: fab-ready → **rc6** (compose rejects every candidate; S2 mechanism found).
- run_22 esp32-dual-motor: fab-ready → **rc6** (compound: S2 + S4 + one leaf never routes).
- run_18 dual-rail-supply: fab-ready → rc7 (unconnected=1, S1).
- run_26 servo-driver-16: fab-ready → rc7 (courtyard=1, S7).
- run_28 audio-jack-buffer: rc7 (DRC) → **build never ran** (`build=None`, D 59.5): wiring
  parked 5× on the same unanswerable question — the vendored **TL072 dual-op-amp symbol only
  exposes unit-A pins 1,2,3,4,8; unit-B pins 5,6,7 are missing** — until park rounds ran out
  (S4b). Its budget-killed first pass reached fab-ready (different BOM pick), so the symbol
  defect bites stochastically.

**Noise caveat** (`kicraft-self-eval-2026-06-24-findings`): single-run flips cross grade
buckets; the ~12-pt noise floor is real and PYTHONHASHSEED perturbs builds. S2/S3/S4/S5 are
*mechanism* fixes verified by replay regardless of noise; do NOT treat the run_18/26 flips
alone as code regressions without an N-of-3 median.

### Where the 12 non-fab-ready runs go

| Cluster | Runs | Owning fix |
|---|---|---|
| rc7 unconnected=1–6, 0 shorts (genuine `no_clear_path` + debris slivers) | 06(3), 10(6), 12(1), 13(5), 18(1), 27(5) | S1 (C1 v2) |
| rc7 missing via at layer change (+ `illegal_routed_geometry`) | 24(1) | S3 |
| rc6 compose stamp-DRC reject: leaf copper vs edge-snapped outline | 15, 22 | S2 |
| footprint/symbol defects (neg. annular; drill < min hole; missing unit-B pins) | 06 (J1), 22 (U2), 28 (TL072) | S4 |
| near-miss clearance 5.9µm under rule (guard is 5µm) | 13 | S5 |
| connector_stranded | 24 (J2–J9, −1.2/−20.4mm), 13 (J1 −1.33mm) | S6 |
| courtyard overlap in connector array | 26 (J5×J7) | S7 |
| observer gates (fab quality, not build) | 10 unprogrammable_mcu, 20 silent_substitution | S8 |
| shaped conformance (electrically 0/0) | 29 | S9 |
| design never reached build (wiring park exhaustion on symbol defect) | 28 | S4b |

Run dirs: `/home/kicraft/KiCraft/logs/self_eval/20260716T011056Z/run_NN_<slug>/` — each holds
`.kicraft/{state.json,build.log}`, `events.jsonl`, `eval/report.json`, and the generated tree
with `.experiments/`. Best-round evidence:
`generated/<stem>/.experiments/hierarchical_autoexperiment/round_000N/parent_pipeline.json`
(`state.routed_validation` / `state.stamp_drc`), promoted boards under
`.experiments/subcircuits/subcircuit__*/parent_routed.kicad_pcb`.

---

## Workstreams, priority order

### S1 (P0) — C1 v2: richer pathfinding/rip-up for the walled-off unconnected family

**Breadth owner: 6 of 9 rect failures** (06, 10, 12, 13, 18, 27 — 21 unconnected items).
This batch *re-confirms* the 07-13 measured verdict: the remaining rc7 edges are genuinely
`no_clear_path`, not budget. Fresh forensics (via-aware same-net clustering on the promoted
boards):

- run_27 MS1: 12 segs → 2 clusters **14.8mm apart**; the 0.6µm/92µm "tracks" in the DRC pair
  are rip-up debris, not micro-gaps.
- run_13 DEC3/DECUSB: sliver stubs sit **9.4/12.9mm from the target pad** — same shape.
- run_12 USB_DN: **zero copper** on the net (R6 pad ↔ J1 pad never attempted/always ripped).
- run_10: 3×QSPI (U4 RP2040 ↔ U5 flash), USB_D±, and pad-22→+3V3 **zone not reaching a pad**
  (zone-reach strand, the +3V3 analog of the GND strand).
- run_27 also has GND **zone-to-zone** strand — `kicraft-gnd-plane-strand-walled-off-breadth`
  (GND edge-spine lever for connectors-along-edge) applies.

Do NOT spend this workstream on stub/anchor tweaks (N5 pre-filter `77def64` already measured
that out). The owning fix is the deferred **C1 v2**: rip-up + reroute of blocking traces for
the straight/L/dogleg family, plus zone-reach (pour must reach every same-net pad or emit a
spine). Debris slivers (<10µm same-net segments) should be swept post-import so DRC pairs
point at the real gap — cosmetic but makes every future investigation faster.

$0 verify: `kicraft replay` any of run_10/12/13/27 (frozen workspaces above), assert
unconnected drops to 0 on the touched genre without shorts regressions.

> **IMPLEMENTATION PASS 2026-07-16 (same day).** S2 LANDED + replay-verified
> (run_15 rc6 → **routed 0/0, fab package exported**). S3 DISPROVED (see below).
> S4 LANDED (three parts of it; §9.30 gate + PTH fab-floor normalization at the
> footprint-load seam + validate-part check 6; 5 curated parts repaired).
> S5 LANDED. S7 LANDED (locked-pair slide, tested) — but run_26's true owner
> turned out to be S6 (see consolidated note in S6/S7). S6 root-caused,
> fix deferred to its own PR. Details inline per workstream.

### S2 (P0) — compose edge-snap ignores leaf copper: whole-board rc6

> **DONE + VERIFIED 2026-07-16.** Deeper mechanism than first written: the
> repair pass `_repair_parent_outline` ALREADY had the correct
> pads+traces+bodies containment rules (incl. the connector-side
> `pad_edge_clearance_mm` floor) — but Phase 3A (`c26ffe7`, 2026-07-02) turned
> BOTH call sites verify-only, betting the bbox-level clamp in
> `_compute_final_outline` covered everything. It doesn't (child bboxes only;
> "no pad data here"; connector sides skipped): run_15's round-3 geometry
> extended 0.545mm past the snapped outline on both x sides (3 traces +
> parent-local C4's body outside; `geometry_validation.accepted=false` in
> every round). Fix = restore mutating repair at both call sites
> (`_compose_stamp.py:137`, `compose_subcircuits.py:~3300`) + new
> `state.outline_authoritative` flag so form-factor scaffold outlines are
> still never grown (manual outlines already early-return). Replay of run_15:
> parent routed, **0 shorts / 0 unconnected**, fab zip exported.

run_15 buck-3a: **all 3 leaves route and accept (77–97)**, then every parent candidate in
all 6 rounds is rejected by the stamp gate with 6× `copper_edge_clearance` — leaf tracks
(+3V3, VSENSE) measured **0.0000mm from Edge.Cuts** on `parent_pre_freerouting.kicad_pcb`
(i.e. *stamped leaf* copper, before parent routing). run_22 shows the same signature
(5× copper_edge_clearance) among its other problems.

Mechanism (`kicraft/cli/compose_subcircuits.py`): `_compute_final_outline` (line 772) snaps
edge-constrained sides to the connector anchor — `_resolve_min` returns `c_val` whenever the
anchor is within `spacing_mm + 2.0` slack of geometry — **without checking where leaf COPPER
ends**. A leaf whose interior tracks run near its own canvas edge gets its copper cut by (or
inside 0.2mm of) the snapped parent edge. The per-leaf `copper_manifest` already in the
compose state has the data to prevent this.

Fix at the source: when snapping an edge-constrained side, compute the leaf copper extent on
that side from the copper manifest and (a) nudge the non-flush leaves inboard, or (b) reject
only that *candidate placement* with a loud reason — never let a copper-cutting outline reach
the stamp gate 6 rounds in a row. Add the measured deficit to the near-miss log so
autoexperiment mutation can react (`edge_margin_mm` is already a mutable param).

$0 verify: `kicraft replay` run_15 (`logs/self_eval/20260716T011056Z/run_15_buck-3a`),
assert parent_composed=True and fab gate reached; run_22 should progress past stamp.

### S3 — ~~layer change without a via~~ **DISPROVED 2026-07-16 (analysis artifact)**

The original finding (AIN0 clusters meeting at 0.00µm across F.Cu/B.Cu with "no via") came
from pad-blind text clustering. With pads included (pcbnew HitTest): the joint at
(157.5337, 97.1493) sits **inside J2 pad 1 (PTH, net AIN0)** — a through-hole pad is a legal
layer transition, and the two "clusters" are connected through it. A heal implementation was
built and dry-run against **all 32 batch parent_routed boards: zero true bare layer jumps
anywhere** (every cross-layer joint is bridged by a via or a same-net PTH pad) — so the code
was reverted, not landed.

run_24's real `unconnected=1`: **pad U4.4 + its three breakout stubs isolated at
(162.8, 94.1)** a few mm from the rest of the net — the classic C1 walled-off stub family.
Reclassified to S1. (Its `illegal_routed_geometry` flag co-occurs with connector_stranded →
S6.) Lesson recorded: any track-connectivity analysis MUST include pads before claiming a
break.

### S4 (P1) — footprint defects reaching board DRC: add validate-part lints

> **DONE 2026-07-16 — with corrected provenance.** run_22's U2 is the FETCHED
> `esp32-s3-wroom-1-n8` module (12× 0.25mm thermal-via drills), not the
> curated DRV8833 (already 0.30 since WS10); run_06's J1 is the FETCHED
> `usb-c-24p` (round shell legs drill==size → zero annular; its oval pad-25
> slots are conformant per-axis). Landed as
> `normalize_pth_pads_for_fab` (`parts_library/footprint_courtyard.py`):
> grow-only per-axis floors (drill ≥ 0.30, annular ≥ 0.13), applied at BOTH
> seams — the every-footprint-enters-here load in
> `design/synthesis/kicad_pcb_stub.py` (heals already-cached bundles in
> memory, no hash churn) and vendor-time `add-part` hygiene. Verified on both
> fetched parts (14 + 24 changes, idempotent second pass = 0). Curated-library
> sweep found 5 more sub-floor footprints (incl. run_20's OLED and tps61088's
> thermal vias) — normalized on disk + rehashed. `validate-part` gained check
> (6) so regressions can't re-enter the library.
>
> **S4b — corrected root cause:** the fetched TL072 symbol is NOT defective
> (all 8 pins present across 2 units). The pipeline is: `lookup_pins` defaults
> to unit 1 and `_emit_symbol_instance` hardcodes `(unit 1)`, so unit-B pins
> are unreachable BY DESIGN — wiring then park-loops on an unanswerable
> question. Landed **§9.30 multi-unit-symbol gate** (parts-only, BOM commit,
> `validation.check_multi_unit_symbols` + `cli_app` hook): rejects any pick
> whose symbol has functional pins beyond unit 1, with directive re-pick
> feedback (TL071/OPA344-class singles). Verified: fires on run_28's real BOM
> (all 4 TL072s), zero false positives across the other 33 batch BOMs. The
> REAL feature (emitter instantiates all units; `lookup_pins(all_units=True)`
> groundwork already exists) is a follow-up PR — remove §9.30 when it lands.

#### (original analysis below)

Two distinct part-library defects produced DRC errors this batch:

- run_06 usb-c-full-breakout J1: **14× `annular_width` (−0.0034mm, i.e. hole larger than
  pad ring) + 2× `padstack`** on the USB-C receptacle's PTH pads.
- run_22 esp32-dual-motor U2: **`drill_out_of_range`** — 0.25mm holes (`<no net>` PTH pads =
  thermal vias baked into the footprint) vs board min hole 0.30mm.

**S4b — missing-unit symbol pins killed run_28 outright:** the vendored TL072 (dual op-amp,
SOIC-8) symbol exposes only pins 1,2,3,4,8 — unit B (5,6,7) is absent. Wiring correctly
refused to short around it, parked the same clarifying question **5 times** (2 failed
commits, $0.055 — pure wheel-spin, cf. `kicraft-pipeline-cost-bom-retries`), exhausted
`--max-park-rounds`, and the run died with `build=None`. Two fixes: (a) the lint — a
multi-unit part's symbol must cover all units / symbol pin numbers must cover the footprint's
pad numbers (TL072 SOIC-8: 8 pads vs 5 pins → fail at bind time); (b) the loop — when the
same park text repeats N times verbatim, escalate to a BOM re-pick instead of re-parking.

These are static properties of the footprint/symbol versus the board's standing constraints —
catchable at part-bind time, long before a $-costing build. Extend `validate-part`
(`kicraft/parts_library/`) with: (a) annular ring = (pad size − drill)/2 ≥ board min annular
(0.127mm), (b) every drill ≥ board min hole (0.30mm), (c) symbol-pin coverage vs footprint
pads (S4b), (d) re-run `--update-hash` after any fix
(`kicraft-vendored-hash-mismatch-silent-skip`). This is the same lint family as the open
BNC follow-up (markerless asymmetric TH connectors) — implement as one lint pass.

$0 verify: lint the vendored TL072 symbol + two footprints; fix/replace; replay run_06 (its
remaining unconnected=3 is S1, but annular/padstack errors must vanish). run_28 needs a
synthesis re-run (~$0.05) after the symbol fix.

### S5 (P1) — DSN clearance guard margin: 5µm is too tight for the tail

> **DONE 2026-07-16:** `freerouting_clearance_guard_um` 5 → 10 in
> `autoplacer/config.py` + the `freerouting_runner.py` fallback literal;
> guard tests green. (Replay of run_13 checks the clearance pair is gone;
> its unconnected=5 is S1 and will keep it rc7.)

run_13 nrf52-beacon: 1× clearance violation, **0.1521 vs 0.1530mm rule = 5.9µm under**,
pad P2 of U1 (rotated aQFN) vs track. The KC-9G4YPT guard (`_apply_dsn_clearance_guard`,
`kicraft/autoplacer/freerouting_runner.py:868`, default +5µm via `clearance_guard_um`) was
sized for ~1µm pad-approximation error; this instance shows ≥5.9µm skew on rotated-pad
geometry. Raise the default guard to **10µm** (routing-space cost is negligible; the guard
lives only in the DSN so KiCad DRC still verifies the true rule — it cannot mask anything).

$0 verify: replay run_13, assert the clearance pair is gone (its unconnected=5 is S1).

### S6 (P1) — connector stranding: ROOT-CAUSED to block-level edge-capacity overflow

> **Root-caused 2026-07-16; fix deferred to its own PR (deepest-risk change of
> the set).** The −20.37mm family is NOT "constraint never attached": run_24's
> 8 screw terminals live 4-each inside the two ADC **leaves** (20×52mm blocks),
> both leaves' connectors claim the right edge, and the composer placed the
> blocks side-by-side in x — the inner leaf's connectors sit exactly one
> leaf-width inboard. A same-edge column of both blocks needs ~105mm; the
> seed outline is ~74mm. Candidate phase-timing breadcrumbs show early solve
> phases at placed_h=111mm (the column WAS formed) collapsing to a 73.7mm
> final outline — the block-level path lacks (or loses) the grow-to-fit that
> the leaf-level edge spread got for run_19 ("column taller than the leaf
> overflows → 2nd column → stranded", `placement_solver.py:~1280`).
> **run_26 is the same owner** (see S7): 16 servo-header leaves × 4.68mm ≈
> 75mm vs a ~66mm edge. Next step: instrument `_compose_artifacts`' solve on
> run_24/run_26 replays to find where the grown outline is discarded
> (seed-outline recompute? clamp? `_shift_pads_inside`?), then make the
> block-level same-edge group grow the seed outline the way the leaf path
> does. The −1.20mm family (J6–J9) is the residual flush gap (Bucket A3),
> secondary once the capacity fix lands.

### S7 (P2) — courtyard overlap inside a connector array

> **PARTIALLY DONE 2026-07-16 — slide fix landed; run_26's flip belongs to S6.**
> Mechanism: each servo header is its own single-connector LEAF (pure THT, 0
> traces) — copper-transparent to `can_overlap_sparse` (no committed side), so
> nothing upstream holds the blocks apart, and the final-guarantee
> `_resolve_courtyard_overlaps` pass counted locked-locked pairs as
> "unresolved" and shipped them (its own comment names exactly this THT
> pin-header case). LANDED: `_slide_locked_pair` in that pass — edge pins fix
> only the perpendicular coordinate, so one part slides ALONG the shared edge
> (flushness preserved; mounting holes exempt). Tests updated
> (`test_courtyard_overlap_resolution.py`: slide + mounting-hole cases).
> BUT the replay shows run_26's real problem is capacity, not resolution:
> 16 header-leaves ≈ 75mm of edge vs a ~66mm board — sliding cannot create
> space that doesn't exist (clamped → honest unresolved). run_26 flips when
> the S6 block-level edge-capacity grow lands. The slide fix still hardens
> every board where the edge HAS room (the batch's own J5×J7 was a 0.23mm
> overlap on an 86mm board — that case now resolves).

### S8 (P2) — synthesis-side gates that capped grades on fab-ready-adjacent boards

- run_10 `unprogrammable_mcu` (cap 50): RP2040 with USB+QSPI but **no BOOTSEL button / RUN
  strap in BOM**, and it wasn't surfaced as an open question. The §9.29 programmability gate
  (from `86e2a17`) is judging correctly — the *synthesis* side needs the rule: architecture/
  BOM must include a boot-entry mechanism for RP2040 (BOOTSEL switch or exposed strap), or
  explicitly surface the omission. Add to the architecture-stage IC-domain hardening
  (PR1 of `kicraft-reconcile-stage-plan`) or the BOM §-rules.
- run_20 `silent_substitution` (cap 55): brief said **"SMT I2C OLED"**, BOM picked
  HS96L03W2C03 with **OLED-TH** (through-hole) and never surfaced the substitution. Add a
  cheap deterministic check at BOM commit: mounting-technology / package keywords present in
  the brief ("SMT", "0805", "through-hole", …) must match the bound footprint's tech, else
  force an open_question. This is the third silent_substitution in three batches (×3 on
  07-10, ×1 on 07-13) — it is systematic, not noise.

Verify: re-run the two briefs' synthesis stages (LLM $, ~$0.06) or unit-test the BOM check
against run_20's state.json.

### S9 (P2) — shaped group: 5/7 fab-ready; run_29 is conformance-only

Final shaped results: 30 (rounded-rect) **fab-ready B 76.5** (flip from baseline DRC-fail),
31 chamfered / 32 hex / 33 star / 34 snowman all **fab-ready** — the shaped-outline +
nesting stack (PR-N1..N5) is holding on 5 of 7.

run_29 round-led-ring: **electrically perfect — 0 shorts / 0 unconnected / 0 courtyard** —
fails ONLY outline conformance: `shape fit rejected: circumscribed 75.8×75.8mm exceeds
requested 60×60mm → RECT FALLBACK` → non-conformant verdict. This is the known post-nesting
gate (`kicraft-shaped-compose-nesting`): flip reliability is gated by guest-leaf size
variance; run_29 additionally splits into 3 leaves (synthesis sheet-merge finding), so the
nest sees ring+2 guests and the circumscribed content (75.8mm) far exceeds the 60mm circle.
Owner: leaf grid-assignment tuning (shrink guest leaves below the ~21.9mm boundary) +
sheet-merge so the ring genre yields ring+1 guest. (run_28 is not a shaped failure — see
S4b.)

### S10 (P0-operational) — spend-guard: unblock the live site, fail batches fast

1. **Operator (user) decision needed:** lifetime ledger total is $50.00 ≥ ceiling; the live
   web app + build worker refuse every model call until `KICRAFT_TOTAL_USD_CEILING` in
   `/home/kicraft/KiCraft/.env:13` is raised (e.g. 250) and both services restart
   (`deploy/restart-web.sh` + `restart-build-worker.sh`). Daily cap ($20) still bounds burn.
2. `kicraft.eval.self_eval` should **preflight the guard at batch start** (call
   `SpendGuard.status()`; refuse to launch if `total_remaining_usd` < a per-brief estimate ×
   briefs) instead of erroring brief-by-brief mid-batch, and print remaining headroom in the
   startup banner. ~15 lines in `kicraft/eval/self_eval.py`.
3. Consider a `--budget-usd` flag that sets a scoped ceiling for the batch process (the
   manual `KICRAFT_TOTAL_USD_CEILING=52` env override worked; make it first-class).

---

## Priority order & expected yield

| WS | Cost | Boards flipped (this batch) | Notes |
|---|---|---|---|
| S2 | small, surgical | 15 (+ helps 22) | deterministic mechanism, replay-verified |
| S3 | small | 24 (with S6) | deterministic repair + root-cause |
| S5 | one-line | part of 13 | guard bump |
| S4/S4b | small | 28, part of 06, 22 | lint, prevents whole class |
| S6 | medium | 24, part of 13 | needs root-cause first |
| S7 | small | 26 | single courtyard pair |
| S1 | large (deferred C1 v2) | 06, 10, 12, 13, 18, 27 | the breadth owner; plan separately |
| S8 | prompt/schema | grade-only (10, 20) | systematic silent_substitution |
| S9 | tuning | 29 | continues nesting workstream |

Realistic near-term yield: S2–S7 + S4b flip **4–6 boards** (15, 24, 26, 28, parts of
06/13/22) → **~26/34, past the ≥24 target** without touching the hard C1 v2 work; C1 v2
then owns the path to ≥30.

### Post-implementation status (2026-07-16, same day)

| WS | Status | Measured |
|---|---|---|
| S2 | **LANDED + replay-verified** | run_15 rc6 → routed 0/0 + fab zip |
| S3 | **disproved, reverted** | 0 true cases in 32 boards; run_24 → S1 |
| S4 | **LANDED** (§9.30 + PTH floors both seams + validate-part(6) + 5 curated parts repaired/rehashed) | §9.30: run_28 caught, 0 false pos ×33; normalizer idempotent on both fetched parts |
| S5 | **LANDED** | guard 10µm; tests green |
| S6 | root-caused, **deferred** (own PR: block-level edge-capacity grow; owns run_24 AND run_26) | phase-timing breadcrumbs: 111mm column collapsed to 73.7mm outline |
| S7 | **LANDED** (locked-pair slide) | tests green; run_26 gated by S6 capacity, not resolution |

Immediate expected flips on a re-run: **run_15** (verified) + **run_28** (§9.30 re-pick;
needs $ synthesis) → 24/34. run_24/26 wait on the S6 PR; 06/13/22 lose their footprint/
clearance DRC noise but stay rc7 on S1 unconnected.

## Re-baseline rule

After landing any S-series fix, re-run ONLY the touched briefs via
`--resume <new-batch> --only <slugs>` on a fresh batch dir, N-of-3 for anything
noise-adjacent (18, 26). Full-corpus re-baseline once S2–S7 land together.
