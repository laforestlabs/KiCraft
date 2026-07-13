# Self-eval 2026-07-11 fix plan — batch `20260710T211406Z` (resumed)

**For the implementing agent.** This plan is self-contained: every finding names its evidence
dir, the owning file:line, and a $0 verification. Read `docs/plans/self-eval-2026-07-10-fix-plan.md`
first for the prior workstream numbering (WS1–WS13) — this plan continues from its re-baseline.

## 0. Scorecard and what this batch proves

| | baseline `20260710T041015Z` (pre-fix) | this batch `20260710T211406Z` (post-WS1–11) |
|---|---|---|
| fab-ready | 11/34 | **17/34** (target was ≥20) |
| mean / median | — | 68.6 / 72.5 |
| grades | — | A:1 B:13 C:11 D:9 |
| gates | — | silent_substitution×3, unprogrammable_mcu×1, erc_errors×1 |
| spend | — | $1.33 |

- **10 briefs improved** to fab-ready (usb-pd-trigger, fpc-breakout ← A 90, esp32-s3-sensor,
  lora-node, encoder-oled-panel, gpio-expander, rounded-c3-devboard, chamfered-badge,
  hex-env-sensor, snowman-ornament). The WS1–11 fixes broadly worked — **do not revert WS1/WS2
  wholesale; the fixes below tune them.**
- **4 briefs regressed** from baseline fab-ready: rc-lowpass-bnc (rc6), stm32-min (rc7),
  round-led-ring (rc6), star-ornament (rc5/ERC). All four are root-caused below.

### Code provenance (matters for replays)

- The batch ran on **committed `eec54f2`** (= `bdcdd25` WS1–11 + Phase 6 perpendicular connector
  banks). The WS1–11 fixes are *committed*, not working-tree.
- The **uncommitted working tree** is the KC-HN59RJ shape-fit work (ring `ArraySpec`,
  `inscribed_rect_bound` shape-seed cap, §9.29 MCU-programmability gate), landed **17:53 UTC
  2026-07-11 — mid-batch**. Only runs finishing after 17:53 could see it (runs 31–34), and only
  in subprocesses spawned after that time. Proven contaminated: **run_33 star-ornament** (its
  BOM used `pattern:"ring"`, which only exists in the uncommitted schema).
- **Before any replay work: commit or stamp the tree.** Mixed-code replays are unattributable.

### Where the 17 non-fab-ready runs go

| Cluster | Runs | Owning fix |
|---|---|---|
| rc7 DRC-blocked (all 0 shorts, 1–11 unconnected) | 02, 06, 09, 10, 15, 17, 23, 26, 27 | N5 (repair pass); N2 for 09/10; N9 for 06; N5+outline-shrink for 26 |
| rc6 route/infra failed | 01, 24, 29 | N1 (01), N6 (24), N4-verify (29) |
| design failed (build never ran) | 13, 18, 21, 22 | N3 (all), N7 (21), N8 (18), N10 (13, 22) |
| rc5 ERC errors | 33 | N4 |
| fab-ready but observer-gated D | 12, 14, 20 | N11 |

Run dirs: `/home/kicraft/KiCraft/logs/self_eval/20260710T211406Z/run_NN_<slug>/` — each holds
`.kicraft/state.json` + `build.log`, `events.jsonl`, `eval/report.json`, and the generated KiCad
tree with `.experiments/`.

---

## Workstreams, priority order

### N1 (P0) — `leaf_solve_deadline` bypasses the "can't-regress" seed-bbox fallback

> **DONE 2026-07-12 — and the evidence line hid a second, deeper root cause.** Both halves
> landed: `leaf_solve_deadline` removed from `_STRUCTURAL_UNROUTABLE_REASONS`, and the ladder
> now JUMPS to the terminal seed-bbox rung on deadline (25% budget reserve via
> `leaf_solve_seed_bbox_reserve_frac`, first fallback round guaranteed even over budget).
> Replay then exposed why the leaf never routed at ALL: **FreeRouting 1.9.0 deadlocks on the
> 'Ω' character in DSN PN fields** (run_01's `10kΩ`/`BNC 50Ω` values; proven by A/B on the
> identical DSN — Ω removed → routes in seconds, rc=0). Fixed at the DSN boundary:
> `_sanitize_dsn_part_numbers` in `export_dsn` transliterates non-ASCII in PN fields (cosmetic,
> not SES round-tripped) + loud warning for residual non-ASCII in names. Re-replay of run_01:
> 2/2 leaves, parent routed, **0 shorts / 0 unconnected**. Fleet note: 15/34 batch runs carry
> non-ASCII values but almost all are 'µ', which FR tolerates — Ω was unique to run_01.

**Evidence:** run_01_rc-lowpass-bnc (baseline fab-ready → rc6). `events.jsonl`:
`No accepted routed leaf artifact … after 4 round(s) across 2 canvas attempt(s) (0.25, 0.22):
leaf_solve_deadline,routing_exception` → `[abort] leaf … is structurally unroutable`.

**Root cause (two halves, both WS2):**
1. `kicraft/cli/autoexperiment.py:183` — `leaf_solve_deadline` was added to
   `_STRUCTURAL_UNROUTABLE_REASONS`, so a merely *slow* leaf is treated as structurally
   unroutable and aborts the whole build.
2. `kicraft/cli/solve_subcircuits.py:846-899` — the per-leaf wall deadline can expire **before
   the ladder reaches the terminal seed-bbox fallback**, the pure historical solve whose comment
   says it is "what makes 'fab-ready rate can't regress' hold by construction". The guarantee is
   currently violated.

**Fix (at source, no gate weakening):** a deadline expiry must never classify as structural.
Remove `leaf_solve_deadline` from `_STRUCTURAL_UNROUTABLE_REASONS`; and when the deadline is
near/hit, **jump directly to the seed-bbox fallback attempt** (skip remaining compaction rungs)
instead of abandoning the leaf. The fallback is cheap relative to the compaction rungs; reserve
budget for it explicitly.

**Verify ($0):** replay run_01's workspace through the build tail (`verify` skill); expect the
`/1eac9743…` leaf (R1, C1, RV1 trimpot, J2 BNC) to accept via fallback and the board to reach
fab-ready as in baseline. Unit test: a leaf solve whose deadline fires mid-ladder must still run
the seed-bbox rung.

### N2 (P0) — Wall-budget starvation drops the heaviest (MCU) leaf → rc7 + stranded connectors

> **DONE 2026-07-12 — the evidence again decomposed differently than framed.** Reproduction on
> current code showed run_09's MCU leaf was NOT budget/speed-bound (rounds ~50–120 s, not 500 s
> — 500 s was the per-leaf deadline being consumed by MANY rejected rounds). Every round died
> for geometry, in two modes: (1) tight-canvas rounds fail placement legality (force loop
> converges U2 into locked J2 etc. — residual, see below); (2) seed-bbox rounds routed but were
> deterministically DRC-rejected by OUR OWN breakout stub: `_foreign_pad_margins` path margins
> bounded the segment CENTERLINE with bare pair clearance, omitting the track half-width, so a
> diagonal radial stub stamped copper 0.095 mm from the neighboring LQFP pad vs the 0.153 rule,
> every round (also the mechanism behind KC-UXASHQ-style diagonal grazes). Fixes:
> **(a) LANDED** — `_foreign_pad_margins` path margins are now edge-guarded (`pair + half_width`
> — the via-obstacle derivation always assumed this); with N1's ladder jump the seed-bbox rung
> now accepts (leaf best_score 62.99, routed). This was the fix that actually pins run_09's MCU
> leaf. **(b) N2a RECAST, not landed** — the rescue round was implemented and then pulled in
> review: it added a seventh inline policy (3 mutables + an env clamp) to `main()`'s round loop.
> The working implementation + tests are preserved in `docs/plans/patches/n2a-wall-rescue.patch`;
> it re-lands as a small policy inside the `RoundScheduler` refactor
> (`docs/plans/autoexperiment-round-scheduler.md`, step 3). **Residual (tracked, not blocking):**
> tight-canvas legality thrash (assignment/force loop vs locked parts) still rejects the
> compaction rungs on dense MCU leaves — the connectivity-first tuning TODO in the handoff doc;
> and the +3V3 strand-repair skip (`U2.9:no_clear_path`) is N5/GND-strand territory.

**Evidence:** run_09_stm32-min (regression; `leafs=1/2`, LQFP-48 leaf never pinned, `+3V3`
unconnected, SW1/SW2 stranded 1.21 mm off the bottom edge) and run_10_rp2040-min (`leafs=3/4`,
11 unconnected incl. GND + QSPI/USB bus, J2 stranded off right edge). Both logs show
`~496-500 s per leaf-solve round` and `[wall-budget] … finalizing after 2 round(s)` (budget 1404 s).

**Root cause:** connectivity-first leaf placement (`leaf_grid_assignment`, WS1/`0c48cc7`) is
~3-5× slower per round on dense MCU leaves; the WS2 wall budget then cuts the run before the
heavy leaf ever gets an accepted round. The known TODO "needs assignment-search tuning"
(`docs/plans/placement-reconsider-connectivity-first-handoff.md`) is now the direct cause of two
regressions.

**Fix (both halves):**
1. **Budget-shape:** when finalizing on wall budget, an *unpinned* leaf must get priority over
   re-solving already-pinned leaves — spend remaining budget on leaves with zero accepted
   artifacts before any re-rounds. A build should almost never end with an unpinned leaf while
   pinned leaves consumed multiple rounds.
2. **Speed:** profile `leaf_grid_assignment` on run_09's MCU leaf (the workspace is frozen —
   replay one leaf solve) and tune the assignment search (candidate pruning / early accept), see
   the handoff doc. Target: dense-MCU leaf round back under ~250 s.

**Verify ($0):** replay run_09 and run_10 workspaces; expect all leaves accepted and the
connector-stranding to disappear with them (the off-edge SW1/SW2/J2 are symptoms of promoting a
partial board, not a separate placement bug — confirm this while there).

### N3 (P0) — One-shot BOM-reconcile cap makes any deficit chain ≥2 unwinnable

**Evidence:** all four design-failed runs (13 nrf52-beacon, 18 dual-rail-supply, 21 proto-shield,
22 esp32-dual-motor) recorded `unresolved BOM deficit after reconcile: …`. In runs 13 and 22 the
single reconcile pass **succeeded** (parts really added — verified in state.json) and wiring then
surfaced a *genuinely new, real* deficit (nRF52840 DCCH cap; DRV8833 VCP–VM charge-pump cap);
the harness fails the run anyway.

**Root cause:** `kicraft/server/session.py:304-305` — `maybe_bom_reconcile` hard-guards to a
single pass via `already_reconciled`; `kicraft/eval/self_eval.py:302-313` then fails the run on
any remaining `reconcile_target=="bom"` park. Chains of length ≥2 cannot succeed by construction.
Note this is **shared session code** — real web users hit the same cap.

**Fix (at source, in session.py):** replace the boolean with a bounded counter (≤3 passes) and
abort only when a pass **changes nothing** (compare the committed BOM ref-set/part-count before
vs after). That converts runs 13/22 (and likely 18) into successes while still stopping a truly
stuck loop (run_21 would stop after one no-change pass). Adjust `self_eval.py:290-313` to match.

**Verify:** unit test the counter + no-change abort in session.py; then a paid spot-check of
run_13's brief (single brief, `--only nrf52-beacon`) is acceptable, or wait for the batch re-run.

### N4 (P0 — blocks landing KC-HN59RJ) — `isolate_array_sheets` mis-wires ring arrays with series companions → 30 ERC errors

> **DONE 2026-07-12.** Both halves landed in `array_decaps.py`: (1) a member's series/parallel
> 2-pin companion (shares a SIGNAL net with a same-sheet member) is allowed on the array sheet;
> (2) `_declare_cross_sheet_signal_nets` now reconciles PRE-EXISTING stale declarations after a
> move (kept sheet keeps direction, 1-for-1 swap inherits, new sheets join bidirectional,
> collapsed nets drop their declaration). Verified: run_33's pre-isolation state reconstructed
> and re-synthesized on fixed code → R1–R5 stay put, **ERC 0 errors** (was 30); regression tests
> in `tests/test_array_sheet_isolation.py`. 598 is untouched by construction (its committed
> state has `arrays: []`, so isolation is a no-op there). Still open from this item: the paid
> shaped-outline group re-run (29–34) to de-contaminate batch results.

**Evidence:** run_33_star-ornament (rc5, 30 ERC errors, ERC-clean fab-ready in baseline).
`build.log`: `[synth] moved 1 non-array part(s) (R1) off array sheet 'LED 1' onto a dedicated
sheet 'SUPPORT'` (×5, R1–R5). ERC: 20× `label_dangling` + 10× `hier_label_mismatch`, all on the
five `LED_x_CTRL` nets; §9.13 confirms each net split at `Rx.1`/`U1.pin`.

**Root cause:** the new KC-HN59RJ ring `ArraySpec` (`kicraft/design/models.py:467-553`,
uncommitted) let the LLM declare `arrays=[{refs:[D1..D5], pattern:"ring"}]` — for the first time
populating `bom.arrays` for this brief — which triggered the *old, latent* bug in
`kicraft/design/synthesis/array_decaps.py:327-389` (`isolate_array_sheets`, commit `13463aa`):
it strands each LED's series resistor onto a `SUPPORT*` sheet, and
`_resplit_connections_by_sheet`/`_declare_cross_sheet_signal_nets` emit dangling labels and
unmatched sheet-pins on the crossing nets.

**Fix (both, in array_decaps.py):**
1. Treat a member's series/parallel 2-pin companion (per-LED current-limit R — same allowance
   the decap logic at lines 339-342 already has) as **allowed on the array sheet**; don't
   relocate it.
2. Fix the cross-sheet net emission so any isolation produces matching sheet-pin +
   hierarchical-label pairs on both sides (the current output fails ERC deterministically).
   Regression test: a ring array whose members carry series resistors must come out ERC-clean.

**Also:** this is the gating item for committing the KC-HN59RJ tree. After the fix, replay
run_33 AND re-verify KC-HN59RJ's own replay (598) still gets its ⌀60 circle; then re-run the
shaped-outline group (runs 29–34 slugs) to de-contaminate their results.

### N5 (P1) — Parent airwire repair pass (deferred C1 v2, now the single biggest fab-ready lever)

**Evidence:** all 9 rc7 runs have **zero shorts** and 1–4 open nets (run_10: 11, but that's N2).
7/9 are within 4 airwires of fab-ready: run_02 `LADDER_OUT`, run_15 `VIN`, run_17 `CC1,CC2`,
run_23 `CAN_TX,CAN_RX` (reached `routed_dirty` in rounds 1–2!), run_27
`SENSE1,SENSE2,STEP,ENABLE`, run_26 `GND`, run_06 `CC1,SBU1,D+,D-` (leaf-level, see N9).
Leaf scores 77–100 throughout — the loss is entirely at parent compose/route.

**Fix:** after freerouting returns, run an explicit repair pass on still-unconnected nets:
rip up blocking neighbors along the airwire corridor, allow a via detour, re-invoke routing on
just those nets (freerouting 1.9.0 ignores DSN keepouts for wires — see the known gotcha — so
the repair likely needs to be our own geometry pass or a constrained re-route). Owning modules:
`kicraft/autoplacer/brain/leaf_routing.py` / `freerouting_runner.py` / compose tail in
`kicraft/cli/compose_subcircuits.py`. This is the deferred "C1 v2 rip-up reroute" — it is now
quantified at **~6 of 9 rc7 boards this batch**; do it properly, not as a band-aid.

**Verify ($0):** replay run_02 and run_15 first (single-net near-misses), then run_17/23/27.
Measure leaf+parent in ONE replay per run (never across runs).

### N6 (P1) — Wall budget allows only one parent-route attempt on large boards

**Evidence:** run_24_daq-8ch: parent compose succeeded, freerouting burned 610 s and failed,
`[wall-budget] elapsed 611s + est. next round 611s > budget 648s; finalizing after 1 round(s)`.
Also `leafs=4/5`: leaf `261f7b0a` (ANALOG INPUT 2: U4, C8, J3) rejected by
`leaf_routed_artifact_validation` **all 12 rounds** despite `historically_trivial_candidate=True`.

**Fix:** (a) the WS2 EMA budget (`kicraft/cli/autoexperiment.py:2456-2470`) should guarantee a
minimum of 2 parent-route attempts when compose succeeded (scale budget with net count instead
of a flat 648 s); (b) investigate why a historically-trivial 3-component leaf fails routed-artifact
validation 12/12 — prime suspect is the WS7 `tracks_crossing`-counts-as-short strictness in
`kicraft/autoplacer/freerouting_runner.py`; find the actual rejection reason in the leaf's
`metadata.json` / debug artifacts (if `rejection_reasons` are missing, that's WS13.1 — fix that
observability gap first, it's small).

### N7 (P1) — KC-WFFXZ3 inter-sheet-contract recurrence via the park path

**Evidence:** run_21_proto-shield. Architecture emitted orphan sheets `HEADER`/`HEADER 2`
(`function:"interface"`) that appear in **no** `inter_sheet_nets`; wiring parks asking to *move*
J1/J3 onto the main sheet; the bom-reconcile (a "add parts" mechanism) can't move parts; fail.
The c51864e fix (`reconcile_inter_sheet_nets`) never ran because it only runs at wiring
**stage-commit** and a park short-circuits the commit.

**Fix (two levers, do both):**
1. **Architecture stage:** forbid/merge degenerate sheets that carry parts but no
   `inter_sheet_nets` entry (an interface sheet with a connector and no declared nets is always
   a wiring dead-end).
2. **Wiring stage:** before parking on a sheet-assignment/net-contract issue, run the c51864e
   normalizer path — wire the crossing and let commit-time reconciliation add undeclared
   inter-sheet nets. A park for "please move part X to sheet Y" should be unreachable.

### N8 (P1) — Reconcile picks electrically-wrong parts and leaves superseded ones

**Evidence:** run_18_dual-rail-supply. Auto-answer correctly chose "replace BOM with isolated
converter module"; the reconcile pass then picked **WRB2412S-3WR2 — a single-output 12 V module
that cannot make −12 V** — and left the now-redundant MT3608 boost in the BOM. Wiring correctly
re-parked; one-pass cap killed it (N3 also applies).

**Fix:** the reconcile instruction built in `kicraft/server/session.py:265-279` must carry the
electrical constraint verbatim (here: "dual-output ±12 V isolated module") and require removal
of superseded parts. Deeper (cheap, worth it): an architecture-stage check that a requested
bipolar/negative rail has a topology capable of producing it (single-ended buck/boost → cannot),
so the wrong BOM never gets built.

### N9 (P2) — USB-C fine-pitch fan-out needs leaf-time escape stubs (WS12.1, quantified again)

**Evidence:** run_06_usb-c-full-breakout — single-leaf board; the leaf itself left `CC1, SBU1,
D+, D-` unrouted out of the TYPE-C-31-M-12 pad field (132 traces routed). Same class WS12.1
identified (5/7 DRC runs last batch). Fix as WS12.1 specifies: leaf-time breakout stubs for
external nets on fine-pitch connector pads (`kicraft/autoplacer/brain/breakout_stubs.py`).
Verify ($0) by replaying run_06.

### N10 (P2) — BOM stage: provision datasheet-mandated IC companions up front

**Evidence:** run_13 (nRF52840: needs 7 DEC caps + DECUSB 4.7 µF + DCCH 4.7 µF — BOM shipped 6
generic 100 nF), run_22 (DRV8833: needs VCP–VM 0.1 µF charge-pump cap per chip, distinct from VM
decoupling). Each missing companion costs a park round; with N3 fixed these become recoverable,
but the first BOM pass should be complete. Fix: extend the curated IC-companion rules
(bom stage prompt/core-defaults — `kicraft/server/stage_driver.py` + `.claude/skills/kicraft/stages/bom.md`)
with per-family support-pin checklists (nRF52 DEC*/DCCH, DRV88xx VCP). Keep it curated data, not
LLM vibes.

### N11 (P2) — Close the three observer-gate holes (substitution + programmability)

All three signals were deterministically present in the BOM slot; each needs one cheap check in
`kicraft/design/synthesis/validation.py`:

1. **run_14 lora-node (SX1276→SX1278):** §9.23 `check_named_part_substitutions`
   (`validation.py:2056`) never reads `p.footprint`. Extend its scanned corpus to the footprint
   name and flag family-token mismatches (value/mpn says `SX1276`, footprint says `SX1278…`).
2. **run_20 encoder-oled-panel (SMT→TH):** brief said "SMT"; footprint is `OLED-TH_…P2.54`.
   §9.23 never ran (`named_parts` empty). Add a package-type conformance check: SMT/SMD vs
   THT/through-hole tokens in the brief/intent vs footprint-name markers (`-TH_`, `P2.54`,
   `PinHeader` vs `_SMD`, `Metric`).
3. **run_12 esp32-s3-sensor (unprogrammable):** the new §9.29 gate
   (`check_mcu_programming_access`, `validation.py:1487`, uncommitted KC-HN59RJ) **would not
   catch this** — its rule "a USB connector satisfies programming access" (lines 1521-1522) is
   wrong for ESP32-family: native USB without a GPIO0-to-GND affordance (button/jumper/TP) or an
   auto-reset bridge cannot enter first-flash download mode. §9.21 `_esp_boot_problem` also
   fails-open when GPIO0 is simply unwired. Tighten §9.29 for ESP32/ESP8266 accordingly —
   coordinate with the KC-HN59RJ owner; re-run its 100-BOM false-positive sweep after.

### N12 (P3) — Observability carried over (WS13 still open) + one new item

- WS13.1 (persist `rejection_reasons` for zero-success leaves) directly blocks N6(b) — do first.
- New: run_01's `.experiments` `run_status.txt` stayed `phase: running / round 1` after a clean
  abort — flush a terminal status on the abort path (`autoexperiment.py` abort branch), it cost
  investigation time.
- WS13.2/13.3 (silent no-JSON wiring attempts; wiring token floor) remain open, unchanged.

---

## Suggested order & batching

1. **N4 first** — it un-blocks committing the KC-HN59RJ tree (currently sitting uncommitted in
   the shared checkout, which caused this batch's contamination; getting the tree stamped makes
   every later replay attributable).
2. **N1 + N2** — the three build regressions; each has a frozen workspace and a $0 replay.
3. **N3 (+N8 session-side)** — recovers up to 4 design-failed briefs; shared session.py code.
4. **N5** — the big rc7 lever; scope it as its own PR (autoplacer surgery — load-bearing, go
   surgical).
5. **N6, N7, N9** — independent, each with a $0 replay or unit test.
6. **N10, N11, N12** — small data/gate/logging changes.
7. **Re-run the full self-eval** (`--no-judge` fine for build-outcome). Expected: the 17 holds,
   +3 regression recoveries (01, 09/10 via N1/N2), +2–4 from N3, +4–6 from N5 →
   **target ≥ 24/34 fab-ready**, stretch 26. Remember the ~12-pt judge-score noise floor:
   build-outcome flips are evidence, single-run score deltas are not.

## Ground rules (project working preferences — binding)

- Fix at the single point that sets the bad value. No masking gates, no band-aid exemptions, no
  weakening of correct gates (the wall budget and fab gate are correct; the bugs are upstream).
- Verify every fix with a $0 replay of the exact failing run dir before paid re-runs (`verify`
  skill). Measure leaf+parent within ONE replay — never compare artifacts across separate runs.
- `kicraft/autoplacer/` is load-bearing: surgical changes only (N5 especially).
- Commit/stamp the working tree before running replays; coordinate with the KC-HN59RJ owner —
  do not clobber their uncommitted work in the shared checkout (use a worktree under
  `~/KiCraft-worktrees/` if it's still uncommitted).
- Deploy = restart BOTH `kicraft-web` and the build worker for pipeline changes.
- KiCad rotation is clockwise; ERC report coords are ×100 (1/100 mm).
