> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Self-eval 2026-07-17 fix plan (batch 20260717T035619Z: 18/34 fab-ready)

**Batch**: `logs/self_eval/20260717T035619Z` — all 34 briefs, judge on
(minimax-m3), design model deepseek-v4-flash, parallel=3, build_slots=1,
wall 5.8h, spend $1.20. No harness errors; **zero watchdog SIGKILLs**
(validates `5e70ea8`).

**Headline**: 18/34 fab-ready, DOWN from 22/34 (batch 20260716T011056Z).
Grades 13B/12C/9D, mean 68.6, median 73.0. The S-series fixes worked —
**4 boards flipped green** (#6 usb-c-breakout, #15 buck-3a, #26 servo-driver-16,
#28 audio-jack) and without the new regressions the batch lands at **24/34,
exactly the S-series prediction**. Two new failure surfaces ate the gain:

| Surface | Boards | Net |
| --- | --- | --- |
| Parent-compose tight-outline regression (from `e972275`) | #2, #9, #11, #14, #25, #30 — ALL fab-ready last batch | −6 |
| Reconcile-deficit hard fail (3-pass bound exhausted on trivial passives) | #10, #18, #22 — previously synthesized OK, failed later | moved earlier (0 net, blocks recovery) |
| Model/env noise | #20 (retail stock went 0), #21 (zero-wire sheet) | −2 |

Code-state caveat: `2985ccb` (identical-leaf reuse) landed **05:26 UTC, 90 min
into the batch** (separate session). run_02 regressed BEFORE it (04:32, pure
`e972275`); runs 09/11/14/25/30 after. Regression spans both states → common
factor is `e972275`. Reuse itself engaged and worked (#19 `replicated_from`
markers, fab-ready B; #26 flipped green).

---

## T1 (P0) — compose outline margins collapse below the 0.2 mm copper-to-edge DRC floor

**Evidence** (all measured from batch artifacts, `parent_pipeline.json` per round):
- run_02: winner outline 38.748 mm vs placed-content bbox 38.501 mm → **~0.12 mm
  per side** where board-setup edge clearance requires 0.2 mm. DRC: "actual
  0.1000 mm", 3× copper_edge_clearance, present at **stamp time**
  (pre-freerouting) — child leaf traces, not new routes.
- run_09: same signature, "actual **0.0000 mm**", stamp-time. The winning
  candidate was a **pass-1 (non-refit)** candidate (even index 4) → Fix 1's
  per-edge-bank seeds (`_seed_outline_dimensions`, dropped `sum*0.6` floor)
  produce the violation **without** the re-fit. The `parent_refit` kill switch
  alone is NOT a sufficient mitigation.
- run_11: stamp clean, routed edge_clr=3 @0.1786 mm — freerouting routed against
  the new tighter edge (DSN boundary gives it no margin to respect; freerouting
  1.9.0 ignores boundary for wires — known gotcha).
- All 3 rounds fail identically per board; `kept_count=0`; score tier
  `not_routed` → promoted best-effort board → rc7.

**Causal verification (replays, $0)**: replay of run_02 at the exact batch
commit (`e972275` worktree) + batch master seed 1360911876 **passes clean**
(0/0, 31×53 mm) — because `replay` pins `PYTHONHASHSEED=0` and the batch
`build` path does NOT pin it (see T5). The new margins sit AT the DRC
boundary; the per-process hash salt decides which side. Salt probes
(`PYTHONHASHSEED=1,2` at `e972275`) — see addendum at end of file.
HEAD replays (seed 0 and batch seed, refit on and off): all clean; `2985ccb`
shifted the leaf-solve trajectory so HEAD non-repro is expected and proves
nothing either way.

**Fix (at the single source of the bad value — `kicraft/cli/compose_subcircuits.py`)**:
1. `_compute_final_outline` / `_repair_parent_outline`: every outline side that
   is not connector-flush (edge/barrel-anchored) must clear the child **copper**
   extent (traces/vias included, not just courtyards) by
   `max(existing spacing rules, 0.2 mm board edge clearance + 0.1 mm guard)`.
   Today the width axis of run_02 resolves to ~0.12 mm/side — trace which
   branch (`_resolve_min/_resolve_max` anchor path vs geometry path) emits it
   and floor it there, not at a downstream patch point.
2. `_refit_seed_from_placement`: add the same copper-margin term to `need_w/h`
   floors so a re-fit seed can never demand a sub-clearance fit.
3. DSN export: the parent boundary given to freerouting must carry the same
   0.2 mm+guard inset (run_11's routed-only violations; lineage of the 5→10 µm
   DSN guards `6e78597`/`7b07d79`).

**Tests**: extend `tests/test_parent_seed_edge_banks.py` — synthetic placement
whose content bbox touches the seed edge must yield outline ≥0.3 mm out on
non-connector sides; refit floors respect the copper margin.

## T2 (P0, same PR) — candidate screen never gates on stamp-time edge clearance; tighter = higher score

**Evidence**: `candidate_search.rejected_drc=0` while the accepted winner has
stamp `copper_edge_clearance=3` (run_02/09). Hard gates today are only
`shorts>0` / `geometry_accepted=False` (compose_subcircuits.py:3060-3066).
The packing term rewards tightness, so violating candidates **win** the pool.

**Fix**: a candidate is `accepted` only if its stamp DRC also shows
`copper_edge_clearance == 0`. Zero added cost (stamp DRC already runs per
candidate); keep the fail-loud `_rejected_candidates.json` diagnostics.

## T3 (P0, same PR) — no fallback when the routed winner fails validation; re-fit trades fab for compactness with no recovery

**Evidence**:
- run_25: pass-1 = 213.2×74.5 mm (routes fine — was fab-ready last batch at
  ~this size); re-fit = 213.2×**26.8 mm** wins on score, routes to
  `unconnected=1` → whole round dies. Compaction working as designed, but the
  promised "pass-1 = congestion fallback" only exists at *scoring* time — the
  re-fit always outscores it, and after routed-validation rejection there is no
  second chance.
- run_14/30: re-fit winners strand edge connectors
  (`connector_stranded:J1@-24.79mm(top)` etc., unconn 6/15) — the second solve
  breaks the edge-flush/outline agreement; `_compute_final_outline`'s
  containment growth then leaves the connector inboard.

**Fix**:
1. On `routed_validation.accepted=False` for the winner, route the best-scoring
   accepted **pass-1** candidate once before failing the round (bounded: 1
   extra freerouting run, only when the winner was a re-fit).
2. After the re-fit re-solve, re-derive edge-anchor positions from the re-fit
   placement so outline and anchors agree (kills the stranding face).

**Expected recovery from T1+T2+T3**: #2, #9, #11, #14, #25, #30 (~+6).

**Coordination**: another agent is merging further compactness work to main
right now (2026-07-17). T1–T3 touch the same functions — land as ONE PR
coordinated with that merge; whoever lands second re-runs BOTH suites:
- compactness KPI fleet ($0 replay: projects 618/614/622 — util must stay ≥
  the e972275 gains),
- this batch's six regressed workspaces as a frozen $0 regression suite
  (`logs/self_eval/20260717T035619Z/run_{02,09,11,14,25,30}_*/generated/<STEM>`),
  replayed at 2–3 salts (see T5) — pass = rc0 at every salt.

**Interim mitigation only if the live site needs relief before the PR**:
`candidate_search.parent_refit=false` removes the re-fit faces (#25-style and
stranding) but NOT run_09-style pass-1 violations — it is a partial dial, not
a fix, and costs the KC-AXHQTP compactness win.

## T4 (P1) — reconcile executor: deterministically add fully-specified passives instead of 3 LLM round-trips

**Evidence** (all: `unresolved BOM deficit after 3 reconcile pass(es)`,
design-stage death, build never ran; LLM cost 1.8–2.6× batch median = retry churn):
- #10 rp2040-min: "RP2040 (U2) VREG_VOUT (pin 45) requires a 1uF capacitor to
  GND … Add one 1uF 0402/0603 … clustered with U2."
- #18 dual-rail-supply: TPS54160 ×2 missing FB bottom resistors — "Add two 10k
  resistors (0402/0603) on the DC DC CONVERTER …"
- #22 esp32-dual-motor: BOOT-PH 0.1µF, COMP RC to GND, DRV8833 VCP-VM 0.1µF ×2.

The deficit message already carries **kind + value + package + pins + sheet +
cluster hint**. The bounded chain (N3 `4d87b74`) correctly stops the infinite
loop but turns "model didn't apply a mechanical edit" into a hard death. Last
batch all three synthesized fine — this is the model failing a round-trip the
pipeline can do itself.

**Fix**: in the reconcile chain, parse deficits that fully specify a passive
(R/C/L + value + 2 endpoints) and execute the BOM row + wiring add
deterministically (same emitter path the synthesis stages use), then re-run the
check; LLM round-trip remains for everything else; 3-pass bound and fail-loud
behavior unchanged. Owner: `design/cli_app` reconcile stage.

**Expected recovery**: +1–2 (note #10/#18 were rc7 at route last batch — they
also need T1-T3 to go green; #22 was route/infra-failed).

## T5 (P1) — pin PYTHONHASHSEED in the production `build` path

**Evidence**: `_pin_deterministic_placement_env()` is called ONLY by `replay`
(cli_app.py:5225). T1's verification shows the salt **flips fab outcomes** on
margin-boundary boards: batch (unpinned) fails run_02 3/3 rounds; replay at the
same commit+seed (salt 0) passes. Web/live builds have the same roulette today.

**Fix**: call the same helper at the top of the `build` command (worker and
self-eval inherit). `setdefault` semantics keep explicit salt probes working.
This also restores honest batch↔replay comparability for every future
investigation.

## T6 (P2) — commit silk-legend results to state.json (judge floors board_self_description on 21/34 runs)

**Evidence**: build log prints `[build] silk legend: placed=2 dropped=0`
(run_02) yet `state.silk_plan/silk_placed/silk_dropped` are `None` → the judge
(no-evidence rule) scores the weight-4 dimension 0 on 21/34 runs, including
fab-ready boards. The work happens; the evidence never reaches the digest.

**Fix**: the build tail (promote step) commits silk results into state.json via
the existing stage-commit path (`server/stage_driver.py` seam). ~1–2 rubric
points/run across most of the fleet.

## T7 (P2) — §9.26 orderability retry: feed offline-catalog alternates

**Evidence**: #20 encoder-oled-panel died: OLED C5248080 has JLC assembly stock
but **0 retail stock**; model churned $0.075 without converging on an
alternative (gate correct, retry loop unproductive). Partially environmental
(stock drifts), so this WILL recur across the catalog.

**Fix**: when §9.26 flags an offender, query the offline jlcparts catalog for
in-stock same-category/spec alternates and inject the top 3 (LCSC id + stock +
price) into the retry prompt; for passives, auto-substitute deterministically.
Owner: `server/parts_catalog.py` + BOM retry prompt assembly.

## T8 (P3) — electrical-soundness lints the judge caught but no gate did

1. **Regulator feedback-divider check** — #15 buck-3a shipped fab-ready with
   Vout = 0.8×(1+16.9k/10k) = **2.15 V against a 3.3 V brief** (judge caught
   it; grade D). For regulators with known Vref (part metadata), compute the
   divider Vout at reconcile and fail/fix on >5% mismatch with the
   architecture rail. Deterministic, no LLM.
2. **Per-channel connectivity coverage** — #28 audio-jack: 4 declared channels,
   **only channel 1 wired** (J3/J4/J5 signal pins NC, 6/8 header pins NC);
   fab-ready, grade D. §9.9 catches only the degenerate zero-wire sheet
   (#21). Extend it: N declared instances of a repeated functional block →
   each instance's signal path must be wired (the identical-leaf topo-hash
   machinery from `2985ccb` can supply the instance grouping).

## Known/deferred — dedupe, do NOT re-own here

- #12, #13, #27 (unconnected walled-off routing) → **C1 v2** rip-up/pathfinding
  (existing owner; unchanged from last batch).
- #24 daq-8ch (+ its `unprogrammable_mcu` gate hit) → **S6** block-level
  edge-capacity grow (own PR, already planned).
- #29 round-led-ring: last round composed DRC-clean but circle fit rejected
  (circumscribed 91.0 vs requested 60.0) → shaped-nesting guest-leaf-size
  variance, owner = leaf grid-assignment tuning (existing).
- Identical-leaf-reuse siblings missing `leaf_routed.kicad_pcb` (manual editor
  broken on repeated-channel boards) → **PR-M1** of
  `docs/plans/manual-layout-usability-plan.md` (other session, already P0 there).
- #4 speaker-crossover `silent_substitution` gate: fired as designed (grade
  capped at D); the substitution itself is model behavior — no code action.
- #21 proto-shield zero-wire STACKING_HEADERS sheet: §9.9 caught it correctly;
  wiring it is the form-factor rail-binding feature's job
  (`KICRAFT_FORM_FACTOR_ENFORCE`, default OFF) — model-noise otherwise.

## Sequencing

1. **PR-1 (T1+T2+T3)** — compose margins + candidate gate + pass-1 fallback.
   Coordinate with the in-flight compactness merge; dual-suite verification
   (compactness KPI fleet + 6-board regression suite at 2–3 salts).
2. **PR-2 (T5)** — one-line build pin + a determinism note in docs; ship with
   PR-1 or immediately after (it changes live-site behavior distribution).
3. **PR-3 (T4)** — deterministic reconcile adds; verify by re-running the three
   dead briefs' synthesis ($ small) or unit-fixture the three deficit strings.
4. **PR-4 (T6, T7)** — silk commit + orderability alternates.
5. **T8** — after PR-1..3 land; each is an independent small gate.

**Expected next batch**: 24–27/34 fab-ready (T1-T3 recover ~6, T4 up to 2, T7
1; C1 v2/S6 remain the structural rc7 tail).

---

## Addendum: replay evidence log (2026-07-17)

| Replay | Code | Seed | Salt | Result |
| --- | --- | --- | --- | --- |
| r02 baseline | HEAD (2985ccb) | 0 | 0 (pinned) | rc0, 0/0 |
| r02 refit-off | HEAD | 0 | 0 | rc0, 0/0 |
| r02 baseline | HEAD | 1360911876 (batch) | 0 | rc0, 0/0 |
| r02 refit-off | HEAD | 1360911876 | 0 | rc0, 0/0 |
| r02 baseline | e972275 (batch code for run_02) | 1360911876 | 0 | rc0, 0/0 |
| r02 salt probe | e972275 | 1360911876 | 1 | rc0, 0/0 — util 15.3%, aspect 1.15 |
| r02 salt probe | e972275 | 1360911876 | 2 | rc0, 0/0 — util 13.0%, aspect 2.03 |

Batch build path is salt-unpinned (T5), so batch failures are not expected to
reproduce under pinned replay; the in-batch artifacts (stamp-time DRC counts,
candidate records, measured 0.10–0.12 mm margins across 3 rounds × 4 seeds on
run_02 alone) are the primary evidence for T1–T3.

## Implementation + verification results (2026-07-17, same day)

All of T1-T8 implemented (plus review fixes and a congestion-growth valve that
emerged from verification). $0 regression suite — the six frozen batch
workspaces replayed at their batch master seeds on the final code:

| Board | Batch verdict | Post-fix replay |
| --- | --- | --- |
| run_02 r2r-dac | rc7 copper_edge_clearance | **rc0, DRC 0/0** |
| run_09 stm32-min | rc7 copper_edge_clearance | **rc0, DRC 0/0** |
| run_11 fpc-breakout | rc7 copper_edge_clearance | **rc0, DRC 0/0** |
| run_14 lora-node | rc7 stranded x3 + unconn 6 | **rc0, DRC 0/0** |
| run_25 gpio-expander | rc7 unconn (re-fit routability) | **rc0, DRC 0/0** — refit-backoff + congestion-growth engaged (overhead 2.68→4.55) |
| run_30 rounded-c3 | rc7 unconn 15 + stranded | rc7 unconn 17 — **known remaining**: shaped-outline congestion, C1-v2/walled-off family owner |

Congestion-growth valve (implemented during verification): rounds whose routed
parent is rejected for unconnected nets scale `parent_seed_area_overhead`
x1.3/round (cap 2x) via the RoundScheduler — compactness is traded back for
routability only on designs that failed to route. This is what recovered
run_25.

Unit suite: 2745 passed; 8 failures pre-existing (identical on pre-diff code).
A medium code review over the diff pre-verification caught and fixed: a
non-atomic state.json write, the rescue round dropping the refit backoff,
`1.25V`→25V net-voltage parsing, lowercase-m milliohm/mega confusion,
missing wrong-value screen in §9.26 alternates, the strand screen conflicting
with shaped-genre demoted waves, lost single-axis refit tightening, and a
shape-fit/edge-clean precedence inversion.

Follow-ups deliberately NOT in this pass: `_catalog_passive` retail-verdict
check (donor-clone path dominates; §9.26 still gates), deriving
`_COPPER_EDGE_MARGIN_MM` from the board's actual edge rule, digest
truncation architecture (serialize `artifacts` before the budget cut),
run_30's shaped-congestion (C1 v2 owner).

---

Salt probes 1 and 2 both pass but produce **materially different boards**
(aspect 1.15 vs 2.03, util 15.3% vs 13.0%, 168 vs 189 traces) — direct
demonstration that the hash salt steers the placement trajectory (T5's
mechanism). The batch's random 64-bit salt landed run_02 in a failing pocket
for all 3 rounds of that process; small-integer probes did not re-enter it.
Consequence for verification: the 6-board regression suite must be judged by
the T1 margin INVARIANT (outline ≥0.3 mm from child copper on non-connector
sides, asserted on the stamped candidate records) — not by pass/fail at a
handful of salts, which under-samples the pocket.
