# HANDOFF: the routing bottleneck is now KiCraft's dominant fab-ready limiter

**Audience:** an agent picking up routing/place-and-route quality.
**Status:** research + fix needed. Schematic-level correctness is in good shape
(see the §9.x semantic gates + the Layer-3/4 electrical review, all merged); the
thing now stopping boards from reaching *fab-ready* is **routing/placement**, not
the netlist.

---

## UPDATE 2026-06-18 — Bucket A re-diagnosed + two fixes landed (branch `fix/connector-stranding-edge-flush`)

The original Bucket-A hypothesis below ("the edge-extremity guarantee isn't
holding for *parent-level loose connectors*") was **wrong** — verified against the
actual boards (sheet membership + leaf geometry + pcbnew). **None of the 8
stranded boards has a parent-local loose connector.** Bucket A is **three**
distinct mechanisms:

- **A1 — spurious back-side flip** (run_07; contributes run_20/run_25).
  `placement_solver._assign_layers` auto-flips any THT part ≥50 mm² to B.Cu. A
  USB-C / terminal-block connector qualifies → its pad X mirrors *and*
  `edge_outward_angle` swaps left↔right (`types.py`), corrupting the compose
  rotation filter → stranded. **No guard protected edge-zoned connectors.**
- **A2 — leaf packing leaves same-edge connectors inboard** (run_08/19/21/25).
  Several connectors share one edge; the column/row is longer than the leaf, so
  the overflow wraps into a 2nd row/column inboard (run_19: 4 terminal blocks
  need ~67 mm in a 55.7 mm leaf → TB2 strands; under the fast engine the whole
  RELAY_OUTPUT leaf failed to place). run_08/25 are the "two-row, not co-aligned"
  variant.
- **A3 — cross-leaf same-edge competition** (run_09/22). Two leaves both claim
  one edge; only one can be the parent extremity (run_22 MOTOR_1/MOTOR_2 both
  right). **Not yet fixed** — see remaining work.

**Binding-constraint table (ground truth — count from `rejection_reasons` AND the
strict `unconnected==0` gate, not the rc):** the final gate
`_verify_routed_board` returns `ok = accepted and shorts==0 and unconnected==0`,
and `unconnected` is tracked *separately* from `rejection_reasons`. So Bucket C
co-binds on 09/13/22/25 but is the *sole* binding constraint only on run_06 —
fixing A/B clears more boards than chasing C's retry budget.

| Constraint | Runs | |
|---|---|---|
| A only | 07, 08, 19, 20 | A clears |
| A + C | 09, 25 | |
| B only | 01, 05, 16 | B clears |
| B + C | 13 | |
| A + B | 21 | |
| A + B + C | 22 | |
| C only | 06 | completeness |

**Fixes landed (validated by deterministic `replay --no-route` + `connector_edge_gaps`, $0):**
1. **FIX A** — `_assign_layers` exempts edge-zoned connectors / parts with
   `opening_direction` / `kind=="connector"` from the back-flip. run_07 J2/J3:
   −5.76/−6.55 (stranded) → +0.38/+0.47 (flush). Test:
   `test_assign_layers_keeps_edge_connectors_on_front`.
2. **FIX B** — `_pin_edge_components` grows the leaf board along the edge's
   parallel axis so all same-edge connectors fit flush in ONE line. run_19: all
   4 TBs −17.62→−0.03 (and the leaf now *places*, vs failing); run_25: TB1
   −11.87→−0.03 (all 4 flush). Test: `test_edge_group_grows_board.py`.

Net deterministic placement result: **run_07/19/25 connectors fully flush, run_03
(the fab-ready board) unchanged, 0 new unit-test failures** (534 passed; the 5
fails are pre-existing: `test_build_zero_leaf ×3`, `test_kicraft_stage_cli` 1,
`test_best_round_to_layout` 1).

**Remaining Bucket-A work (precise, with the boards that need it):**
- **run_08 residual (−2.02 mm).** FIX B co-aligned H1+J2 along the bottom, but a
  ~2 mm dead leaf margin remains below the (now-extremal) connectors. Needs a
  *leaf-outline clamp* on the zoned side — but do NOT just shrink the global
  `connector_edge_inset`/margin: that band IS the copper-edge clearance Bucket B
  depends on. Clamp only the dead margin where the zoned connector is already the
  extremity (nothing past it).
- **A3 cross-leaf (run_09/22).** `_ensure_edge_blocks_extremal` (compose) shifts
  blocks one-axis and can't co-arrange two same-edge leaves along an edge. Lower
  value: both boards ALSO have `unconnected>0`, so stranding alone isn't the
  blocker. See FIX C in the session task list.

---

## 1. Why this matters now

The schematic side is handled: deterministic wiring gates (§9.16–§9.20) catch
shorts/reversals/self-shorts at the wiring commit, and an LLM electrical review
(Layer 4, on by default) blocks electrically-wrong-but-DRC-clean boards. With
those landed, the **routing/place-and-route stage is where most briefs now die.**

### Evidence (self-eval `logs/self_eval/20260618T142304Z/`, 28 briefs, gate on)

Separating *review-blocks* (the gate working) from *true routing failures*:

| Outcome | Count | Notes |
|---|---|---|
| FAB-READY | 1 | run_03 |
| REVIEW-BLOCKED (structurally clean) | 6 | the Layer-4 gate, not a routing problem |
| **Routing/structural failures** | **~18** | **this handoff** |
| ERC errors | 5 | schematic-side, separate |

The routing failures (read off each run's `events.jsonl` `build_log` →
`[build] 4/5 verify:` + `error: ... NOT fab-ready -- reasons=[...]`) fall into
**three actionable buckets** plus two tails:

**Bucket A — `connector_stranded` (placement; the single biggest cause, ~8 runs).**
A board-edge connector lands *inboard* of its board edge, so its port can't be
accessed. Examples this run:
- run_07 `connector_stranded:J2@-11.45mm(right)`
- run_08 `H1@-5.83mm(bottom)`, run_09 `SW1@-10.85mm(top)`
- run_19 `TB2@-17.62mm(right)`, run_20 `J1@-1.37mm(left)`, run_25 `TB1@-11.87mm(bottom)`
- run_21, run_22 (mixed with bucket B)
These boards are otherwise clean (shorts=0, unconnected=0, all components present).
The defect is **placement geometry**, fixed before/at compose time, not routing.

**Bucket B — `illegal_routed_geometry` (freerouting output; ~6 runs).**
shorts=0, unconnected=0, all parts present, but the routed copper violates a
geometry rule — overwhelmingly **copper-to-board-edge clearance**. Root cause is
known (see memory `kicraft-array-leaf-purity-and-backside-header`): **FreeRouting
1.9.0 honours the DSN `(boundary)`/`(keepout)` for vias and components but
IGNORES it for WIRES**, so traces run right up to the board edge. Examples:
run_01, run_05, run_13, run_16, run_21, run_22. A previous "edge-clearance Fix 1"
was reverted because of this; the open item is a **post-route geometry pass**.

**Bucket C — route incompleteness / `unconnected` on dense boards (~5 runs).**
FreeRouting doesn't finish every net within its pass/time budget: run_06
(7 unconnected), run_09 (2), run_13 (3), run_22 (3), run_25 (1). Correlates with
trace count / density (run_22 = 446 traces / 41 components).

**Tail 1 — total route failure / "no routed parent" (run_10, run_24).** The
parent compose/route produced nothing routable (timeout or exception). rc=6.

**Tail 2 — rc=1 (run_26, 16-ch servo).** A build/infra crash during P&R (see
the `_auto_pin_best_leaves` "serialized no result" family in memory).

> Note the metric trap, now fixed in the admin dashboard: a review-block and a
> routing failure both return **rc 7**. Count true routing failures from the
> `build_log` reasons, not the rc. `kicraft.server.web._build_review_outcome`
> does this split; mirror it in any analysis.

---

## 2. Where the code is

- **Acceptance gate + reason strings:** `kicraft/autoplacer/freerouting_runner.py`
  - `validate_routed_board()` (~L1338) — produces `rejection_reasons`:
    `illegal_routed_geometry`, `connector_stranded:*`, `unconnected`, `empty_board`.
  - `obviously_illegal_routed_geometry` logic (~L1398–1452): shorts, multi-footprint
    clearance, and **`copper_edge_clearance` on non-edge components** (bucket B).
  - FreeRouting invocation / passes / DSN export live in this file too
    (v1.9.0; v2.1.0 has a `max_passes` regression — see the header comment).
- **Build verify/promote tail:** `kicraft/design/cli_app.py`
  - `_verify_routed_board()` (~L2452) wraps `validate_routed_board`.
  - `_promote_verify_fab()` (~L2590) — the gate; returns rc 7 on failure, keeps
    the board for inspection (no-fallback policy — do NOT reintroduce a
    last-good-board restore).
- **Compose + placement + stranding re-stamp:** `kicraft/cli/compose_subcircuits.py`
  (3.1k lines) and the autoplacer (`kicraft/autoplacer/…`). Edge-connector
  placement and the "edge-zoned part must be the leaf extremity" logic live here.
- **Tuning surface (routing knobs are already searchable):**
  `kicraft/tuning/` — freerouting passes and `signal_escape_length` are tunable;
  see memory `kicraft-tuning-framework`.

## 3. Prior art — READ THESE memory notes first (they save days)

- `kicraft-connector-stranding-root-cause-v2` + `-transform-local-point` — the
  connector-stranding family (bucket A): edge-zoned parts must be the leaf
  extremity; `+rot` vs `-rot` convention; the `_wrap_loose_parent_components_as_leaves`
  path. Much of A has been fixed before and regressed; understand why.
- `kicraft-array-leaf-purity-and-backside-header` — the FreeRouting-ignores-
  boundary-for-wires fact behind bucket B, and the reverted edge-clearance fix.
- `kicraft-dense-leaf-route-fail`, `kicraft-compose-drops-child-copper-rc6-cluster`,
  `kicraft-offboard-power-tie-freerouting-hang` — dense-board / compose / hang modes.
- `kicraft-autopin-safety-net-misfire` — the rc=1 `_auto_pin_best_leaves` crash (tail 2).
- `kicraft-tuning-framework` — how to measure routing changes at $0 via replay.

## 4. How to reproduce / inspect (no LLM spend)

- The corpus boards are already built in `logs/self_eval/20260618T142304Z/run_*/`.
  Each has `generated/<STEM>/` with the routed `.kicad_pcb`, `.experiments/…`
  (rejected candidates), and `events.jsonl` (the `build_log` with the verify reason).
- Re-run placement+route on a frozen design deterministically (NO synth, $0):
  `kicraft replay --project <run_dir>/generated/<STEM>` (see memory
  `kicraft-replay-command-and-determinism`; pin `PYTHONHASHSEED`).
- Validate a board directly: `validate_routed_board(pcb, cfg=DEFAULT_CONFIG)` →
  inspect `rejection_reasons` / `drc`.
- A/B a compose change deterministically: `scripts/ab_compose.py`.

## 5. Rough plan (sequence by leverage; each is a research spike, not a spec)

1. **Bucket A — connector edge placement (highest count, most fixable).**
   Audit why edge connectors still land inboard despite the v2 stranding fixes.
   Likely the edge-extremity guarantee isn't holding for *parent-level* loose
   connectors or specific edge zones (the `@-Xmm(side)` offsets show which side).
   Add a deterministic A/B harness over the 8 stranded runs; target 0 stranded.
   *This alone could move ~8 boards toward fab-ready.*

2. **Bucket B — post-route copper-to-edge geometry pass.**
   Since FreeRouting won't respect the boundary for wires, add a post-route pass
   that (a) detects copper within the edge-clearance band and (b) either pulls it
   in or shrinks/!grows Edge.Cuts to legalize, then re-validates. The prior
   attempt was reverted — understand why before retrying. Alternatively evaluate
   a newer/patched router or a KiCad-native autorouter.

3. **Bucket C — route completeness on dense boards.**
   Tune FreeRouting pass budget / `signal_escape_length` / per-net effort via the
   existing `kicraft/tuning/` surface against the dense runs (run_06/22/25);
   consider a retry-with-more-passes on `unconnected>0` before failing.

4. **Tails — run_10/24 (no routed parent) and run_26 (rc=1).** Triage
   separately; likely compose timeout/exception and the `_auto_pin_best_leaves`
   crash respectively.

## 6. Constraints / gotchas

- **No-fallback policy** (memory `kicraft-no-fallback-previews`): never ship the
  raw uncomposed board or a last-good restore; a failed board is kept for
  inspection at rc 7. Keep this.
- **Determinism:** P&R must stay byte-deterministic under a pinned
  `PYTHONHASHSEED` (memory `kicraft-replay-cross-run-contamination`) — measure
  leaf+parent in ONE replay run, never across runs.
- **FreeRouting 1.9.0 specifics** (jar pinned; v2 regressions) — header of
  `freerouting_runner.py`.
- **Measure with the corpus + replay ($0).** Re-run the full self-eval only to
  confirm end-to-end (~$0.8, ~2.3h); use replay/A-B for iteration.

## 7. Definition of done

True routing/structural failures (buckets A–C + tails) cut materially on the
28-brief corpus, measured by `validate_routed_board` `rejection_reasons` over a
clean replay — **with zero new shorts and no regression in the structurally-clean
count.** Bucket A (connector stranding) reaching ~0 is the highest-value single
win.
