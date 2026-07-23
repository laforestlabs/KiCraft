# Dense-SoC single-leaf unconnected — plan (v2, rediagnosed 2026-07-23)

## RESULTS (P0–P3 implemented 2026-07-23, branch `placement-streamline`)

**The owning defect is fixed; the leaf is not yet clean.** Placement now delivers
pin-adjacency; what is left on KC-69TGAP is a routing/escape problem that belongs
to C1-v2, not here.

$0 A/B on the KC-69TGAP MCU leaf — one `solve_subcircuits --only MCU --rounds 3
--route` per side, same seed, same workspace copy, HEAD (`c7dd0cc`) vs this branch:

| | before | after |
| --- | --- | --- |
| median decap→pin (best round) | ~30 mm (probe: 35.2 mm) | **4.2 mm** |
| max decap→pin | 53.9 mm | 11.9–13.2 mm |
| leaf signal-unconnected (best round) | 7 | **5** |
| rounds rejected `illegal_unrepaired_leaf_placement` | 1 of 12 | 0 of 9 |
| rounds that paid for freerouting | 11 of 12 | 6 of 9 |
| leaf accepted (`no_unconnected`) | no | no |

Where the adjacency was being lost (measured per solver phase with
`KICRAFT_TRACE_PIN_LOCALITY=1`), and what each fix bought:

1. `build_anchor_grid` could not *represent* pin-adjacency: slot pitch was tied to
   `leaf_placement_clearance_mm` (2.84 mm) and every slot was treated as a square
   of the passive's LONG side. Decoupling the pitch (0.5 mm courtyard-legal gap)
   and making slots orientation-aware (ring step = the occupant's SHORT side)
   moved ring 1 from 2.03 mm to 1.28 mm off the pad and roughly tripled the
   ring-1 density.
2. Slot starvation was real and now measured: 25 slots for 23 passives (1.1×) →
   72–82 slots (3.1–3.6×), with the growth loop honoring `leaf_grid_min_provision`.
3. Anchors were only ic/regulator/connector, so X1/X2/SW1/BT1 companions had no
   slots near them. `is_pin_anchor` now anchors any non-passive with ≥2 netted pads.
4. The assignment matched on ANY shared net, so a GND-adjacent slot looked as good
   as the DEC pin it had to hug. Point-net (non-plane) matching plus
   scarcest-first seating took the greedy init from 26.7 mm to 4.3 mm.
5. The accept-if-better guard reverted the grid silently. It now records both
   metrics and has a pin-locality floor; on this leaf it reports
   `accept_pin_locality` every round (10.9 mm → 3.8 mm).
6. **The escape loop then scattered it all back** (3.8 mm → 6.5 mm, and the leaf
   was rejected `illegal_unrepaired_leaf_placement`): `legality_diagnostics` /
   `_resolve_overlaps` / `legalize_components` inflate every bbox by the 2.84 mm
   *placement* clearance, so a pin-adjacent decap reads as an overlap. On a
   gridded leaf, legality now means "courtyards don't overlap"
   (`leaf_legality_clearance_mm`, 0.2 mm); spacing is structural.

Residual: the same five nets fail every round — `DEC3, DECUSB, X2_OSC1, RESET,
BUTTON`. The leaf repair (P1.6) runs, reports, and honestly reverts: every
candidate tie is `no_clear_path` after ~2 000 screened paths. Two of the five are
hauls to edge-pinned parts (`RESET` J1↔U1, `BUTTON` SW1↔U1) — the
congested-escape class C1-v2 owns.

**Tested and disproved (so nobody repeats it):** the unconnected edges read
`Track [DEC4] length 1.5000 mm ↔ Pad 1 of C5` — i.e. 4 of the 5 opens on the
run_13 replay are KiCraft's OWN pre-stamped escape stubs dangling beside the pad
they should reach, which made "the stub is now the obstacle" look compelling
(a fixed 1.5 mm escape helps when the partner is 20 mm away, not when it is 2 mm
away). Skipping the escape when a same-net partner pad is within 2.5× the escape
length dropped 2–3 stubs per round and moved the best round from 6 to 5 signal
opens while another round went 6 → 7: noise. Reverted. The blockage is the
surrounding copper, not our stub.

End-to-end replays (`--quality good --seed 0`, matching the runs' own quality):

* **buck-3a — the guard's motivating fixture: rc=0, 0 shorts / 0 unconnected**, same
  as its 20260720 batch result. There the grid is correctly DISCARDED (only 3–4
  legal slots for 11 passives on that geometry); both `discard_score` and the new
  `discard_pin_locality_floor` were observed firing on the real board.
* **run_13 nrf52-beacon: rc=7, 0 shorts / 5 unconnected** (batch baseline recorded
  6). Do not read that as a −1: single-replay deltas are noise
  (`kicraft-replay-cross-run-contamination`). The board is otherwise DRC-clean
  (0 violations besides the opens) and 4× denser than the batch board
  (util 31.9% vs 7.9%).

Full suite: 6 failures, all pre-existing and reproduced on a stashed clean tree
(build_zero_leaf ×3, fine_pitch USB-C param, lookup_lcsc, provider_bench plotly).
+19 new tests.

Not done: P1.7 (targeted freerouting re-pass on the failing nets) — explicitly
optional in the plan, and it buys wall clock back that P2 just saved. P3 ships
DEFAULT OFF behind `KICRAFT_SPLIT_DENSE_SHEETS=1`: it restructures every dense
design's sheets, and enabling it in the same self-eval batch that measures P0–P2
would make neither attributable.

Still open, unchanged by this work: the breadth claim needs a self-eval batch
(real $, ask the user) against the 60-design leaf / 68-design parent tiers.

---

**Status:** open — the #1 leaf-routing failure mode by breadth. **v2 replaces the
two-lever framing (architecture split + C1-v2 A\*) after a $0 root-cause repro
showed both levers miss the owning defect: leaf placement pin-adjacency.**
The v1 plan's evidence sections were partly wrong; corrected below with
measurements from the actual KC-69TGAP artifacts and a reproduced single-leaf
solve.

## Symptom

A whole MCU subsystem (SoC + decoupling farm + crystals + debug header +
button) lands in one hierarchy sheet. Every solve round fails `no_unconnected`
on a **seed-dependent handful** of short 2-pin nets (decaps, crystal load caps,
RESET, BUTTON), the leaf burns `leaf_solve_deadline`, is rejected, and the
parent inherits the opens → `unconnected>0` at the fab gate.

## Root cause (verified, KC-69TGAP `~/.kicraft/projects/1/660`, MCU leaf)

The causal chain has three links; the first is the owner:

### 1. Placement never delivers pin-adjacency — the owning defect

Measured in the failing rounds (round 10, compact 49.5×53.5 mm canvas): **every
decoupling cap sits 8–51 mm from the U1 pin it bridges** (median ~30 mm;
C3/DEC3 at 46.9 mm, SW1/BUTTON 21 mm, on the seed-bbox canvas 95 mm). The
connectivity-first grid (`leaf_grid_assignment`, shipped default, built exactly
to put a decap 1–2 mm from its pins) fails on this leaf class for three
concrete reasons, all reproduced at $0 with an instrumented single-leaf solve
(`solve_subcircuits --only MCU --rounds 1`):

- **Slot starvation:** `build_anchor_grid` produced **25–26 slots for 23
  gridable passives** (~1.1× provisioning; `leaf_grid_overprovision=10` is
  aspirational, never reached). Cause: slot pitch is tied to
  `leaf_placement_clearance_mm` (~2.84 mm) via `leaf_grid_pitch_gap_mm`, so
  rings around the 7×7 mm aQFN sit ~6–12 mm out at ~4.5 mm pitch → ~20 U1
  slots total, per-net coverage 1–6 slots. A decap belongs 1–2 mm off the
  package edge at ~1.2 mm pitch; the clearance-derived grid cannot represent
  the correct answer at all.
- **Missing anchors:** only U1 and J1 got slots. X1/X2 (whose load caps
  C12–C15 must hug them), SW1, BT1 are not in `_ANCHOR_KINDS` → their
  companions have no slots anywhere near them.
- **Silent all-or-nothing revert:** `grid_assignment_sa`'s accept-if-better
  guard compares **total** placement score; when the (already-crippled) grid
  assignment loses to the force-loop scatter, the entire grid is discarded and
  the input kept verbatim — with **zero telemetry**. Reproduced both branches
  under PYTHONHASHSEED variance: guard-reverted run → median cap→pin
  **32.7 mm**; grid-won run → median **11.6 mm** (max 27.8 — still 5–10× the
  target, per the slot-starvation point). This flip is why the failing-net set
  changes per round/seed.

Even "success" (11.6 mm) hands freerouting a hairball of 20+ crossing 5–30 mm
two-pin nets instead of 21 local hops. And a board where C1-v2-style repair
later closes a 46 mm decap trace is still **electrically wrong** — routing
completion cannot own this defect; adjacency can.

### 2. Freerouting silently abandons a subset; the wrapper reports success

Pre/post track-count diff on rounds 10/11: freerouting added **zero copper** on
the failing nets (the dangling `track_dangling` items the v1 plan attributed to
"partial escapes" are KiCraft's own pre-stamped breakout stubs, untouched). It
CAN route these (DEC2/DEC5/SWDIO closed 30 mm hauls in the same rounds); it
drops a seed-dependent handful and reports "Auto-routing was completed".
Meanwhile `leaf_routing.py` **hardcodes** `routed_internal_nets = all` /
`failed_internal_nets = []` on the success path (~line 1300) — per-net truth
exists only in the acceptance DRC and never reaches the round record, so the
retry loop can't react per-net and the honest verdict arrives only after
30–100 s of routing + validation per round.

### 3. No leaf-level repair; retries are blind full re-rolls

`repair_unconnected_signals` is invoked from `cli/_compose_route.py` only.
A leaf with 1–9 residual opens gets no tie attempt — the only lever is a new
seed/canvas, including the counterproductive seed-bbox fallback (177×72 mm →
even longer hauls). 12 rounds ≈ 395 s total, all wasted on hopeless placements.

## What the v1 plan got wrong (kept for the record)

- "FreeRouting lays partial escapes that never land" — false; stubs are ours,
  freerouting adds nothing on those nets. "The SES log reports the net routed"
  — the success path hardcodes it; nothing parses per-net results.
- Lever B (C1-v2 parent A\*) as "headline beneficiary" — wrong altitude: it
  would close electrically-useless long decap traces at the most expensive,
  least reliable stage. C1-v2 remains owner of the genuinely congested escape
  class (run_10 QSPI / GPIO fan-out), not this.
- Lever A (architecture split) as co-owner — the failing nets (decaps,
  crystal caps, RESET/BUTTON) are exactly the nets that **must stay inside the
  SoC leaf**. v1's example split ("power/decoupling" as its own sheet) would be
  electrically wrong and explode cross-leaf interconnect. Splitting detachable
  subfunctions (SWD, button, RF chain) trims 29→~24 parts and helps the parent,
  but the decap-farm hairball — the thing that fails — remains. Demoted to P3.
- "2 solve rounds, ~395 s/round" — actually 12 rounds / 395 s total, across
  two canvases; the compact 0.28-utilization canvas fails identically to the
  seed-bbox one, so canvas size is not the discriminator.

## Breadth (cross-run scan, `latest=2026-07-22`) — unchanged, still the target

- leaf `no_unconnected`: **60 designs**; leaf `leaf_solve_deadline`: **15**
- parent `unconnected_nets`: **68 designs** (the #1 rc7 tier)

## Plan

### P0 — make the grid able to express pin-adjacency (owning fix)

In `leaf_grid_assignment.py` (+ `local_solver_config` defaults):

1. **Decouple slot pitch from placement clearance.** Pin-adjacent rings use a
   pad/courtyard-legal gap (~0.3–0.6 mm off the anchor courtyard, slot pitch ≈
   passive body + DRC clearance), not `leaf_placement_clearance_mm`. The
   courtyard-legality culling already in `build_anchor_grid` keeps it legal by
   construction; the 2.84 mm blanket was a policy accident, not a requirement.
2. **Honor over-provisioning.** Target ≥3× slots per gridable passive (rings +
   per-pin slots generated opposite each anchor power/dec pad from pad
   geometry). Instrument: emit `slots_total / slots_per_anchor / provisioning
   ratio` into round metadata — the starvation was invisible.
3. **Extend the anchor set** to any part with dedicated companions: crystals,
   buttons, battery holders (reuse the companion notion from the KC-HN59RJ
   work). Load caps then get X1/X2-adjacent slots.
4. **Make the guard honest, not silent.** Keep accept-if-better (buck-3a
   regression is real — A/B against it), but (a) record
   `grid_discarded/input_score/grid_score` in round metadata, and (b) add a
   pin-locality floor to the comparison: a candidate whose median decap→pin
   distance is worse cannot win on compactness/crossings alone. The scorer's
   `pin_locality` term already exists (`psw_pin_locality=0.25`) — it was
   outvoted, not absent; retune with the metric visible.

**Acceptance target (leaf-local, $0):** KC-69TGAP MCU leaf median decap→pin
≤ 2.5 mm, max ≤ 5 mm, leaf unconnected = 0.

### P1 — honest per-net verdict + leaf-level repair

5. `leaf_routing.py`: derive `routed/failed_internal_nets` from the
   post-import connectivity/DRC that acceptance already computes; delete the
   hardcoded lists. The round record then names the actual failing nets.
6. Run the existing `repair_unconnected_signals` wrapper (with its byte-revert
   containment) on the leaf board when a round has ≤N residual opens —
   post-P0 they are 1–5 mm ties, the candidate family's sweet spot. Parent-only
   today for no principled reason.
7. Optional third rung: targeted freerouting re-pass restricted to the failing
   nets via the power-first `_restrict_dsn_routing_to_nets` machinery.

### P2 — stop burning the wall clock on hopeless rounds

8. **Place-quality gate before routing:** if median decap→pin distance exceeds
   a threshold (~4 mm), skip freerouting (30–100 s/round) and re-place. Most
   `leaf_solve_deadline` pressure on this class evaporates; the v1 deadline
   questions (seed-bbox reserve slice) become footnotes.
9. Don't escalate to the seed-bbox canvas when the compact canvas failed on
   `no_unconnected` — a bigger board makes hauls longer, not shorter.

### P3 — architecture partition (reframed, secondary)

10. Split only **detachable subfunctions** (SWD/debug, user-IO, battery, RF
    chain, power conversion) — never the decap farm or crystals away from
    their IC. Trigger on routable-part count (~>15); guard test: nRF52840 /
    RP2040 / ESP32 briefs yield ≥3 sheets, no sheet >15 routable parts, and
    every decap shares a sheet with its IC. This is reconcile-stage PR1 with
    the electrical constraint made explicit.

## Verify

- **$0 leaf repro first** (this is how the diagnosis was made): copy the
  project's sch/pcb/pro + `*_autoplacer.json` to a scratch dir, run
  `python -m kicraft.cli.solve_subcircuits <sch> --only MCU --rounds 1` with
  the P0 instrumentation; read slots/seated/guard/pin-locality from the round
  metadata. Set PYTHONHASHSEED to pin the guard branch.
- Then `replay` KC-69TGAP **and** `run_13_nrf52-beacon`: leaf unconnected → 0,
  parent unconnected → 0, `track_dangling` → 0.
- A/B buck-3a (the guard's motivating fixture) to prove no regression.
- Gate any breadth claim on a self-eval batch (real $, ask the user) against
  the 60-design leaf / 68-design parent tiers; single-run deltas are noise.

## Prior art / links

- Handoff that predicted exactly this failure shape and locked the grid
  decisions: `docs/plans/placement-reconsider-connectivity-first-handoff.md`
  (pin-locality metric, decision (d): "too sparse ≈ layouts blow up" — that is
  what shipped).
- `kicraft-placement-streamline-plan` (memory) — "needs assignment-search
  tuning" was this, plus slot starvation.
- C1-v2: `docs/plans/c1-v2-pathfinding-design.md` — stays owner of the
  congested-escape class (run_10), explicitly **not** this plan's owner.
- `kicraft-reconcile-stage-plan` (P3), `kicraft-wiring-park-integrated-soc`.
- Congestion-growth fix `6e18f79` already stops these leaf-internal opens from
  bloating the parent while this plan lands.
