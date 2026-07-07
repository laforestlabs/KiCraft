# C1 — constrained bend/via repair for walled-off unconnected signal nets

**Status:** prototype implemented (`kicraft/autoplacer/brain/unconnected_repair.py`),
hooked into the parent compose path behind `signal_unconnected_repair_enabled`.
**Evidence base:** self-eval 2026-07-07 (`logs/self_eval/20260706T224451Z/`):
5/28 boards routed but not fab-ready on `unconnected_nets` (run_09 stm32-min,
run_13 nrf52-beacon, run_14 lora-node, run_22 esp32-dual-motor, run_24 daq-8ch)
— ~18 points of fab-ready rate, the single biggest lever in the batch.

## Measured verdict (2026-07-07, all five frozen boards)

The prototype was run against a copy of each frozen routed parent
(before/after DRC measured inside the same run). Result: **17/17 unconnected
edges detected (exactly matching DRC), 0 closed, 0 regressions** (shorts
stayed 0 everywhere; accept-or-revert restores the board byte-for-byte).

The skips tell the real story — this batch's `unconnected_nets` family is
NOT last-millimetre walls:

- **run_09** `USB_D+ J1.A6→U2.33`: pad-to-pad with **no copper at all** — a
  fully abandoned ~22 mm route across a routed, area-compacted board.
- **run_14 / run_22** `DIO0/DIO1`, `M1_IN1..4`: 65–81 mm abandoned
  pad-to-pad routes (over any local-repair cap by design).
- **run_13 / run_24**: partial routes whose track ends sit 12–22 mm from the
  target pad, in copper too dense for any guarded polyline (every one of
  ~120 candidate shapes per edge — 8 escape directions × lengths ×
  L-bends/doglegs on both layers — hit a foreign pad/track margin).

Conclusion: waypoint stamping is the wrong tool for THIS family. What these
boards need is **C1 v2: selective rip-up reroute** — on the final routed
board, unlock ONLY the blocked nets plus the copper crossing their corridor
(leaf copper stays fixed), and re-run freerouting on that sub-problem. The
autoexperiment rounds do not cover this: rounds re-randomize configs from
scratch; none does targeted rip-up on the best board. That is a
freerouting-level change (DSN export with selective wire locking) and is the
recommended next step.

The prototype stays enabled (`signal_unconnected_repair_enabled`, default
on): it is provably regression-free, costs <1 min only on already-failing
rounds, and will close the genuinely-local subset (the earlier
"unconnected=1" 8-board cluster shape) when it recurs.

## What this is (and is not)

After freerouting, a handful of signal nets are sometimes left unrouted because
the router **walled itself off**: earlier passes laid copper across the only
corridor to a pad, and freerouting 1.9.0's rip-up never recovers
(memory `kicraft-unconnected-1-cluster-walled-off-signal-power`: these are
`no_clear_path` cases, NOT last-mm snaps). The prior decision stands: no
island-removal, no evidence deletion, no gate-waiving. This repair works at the
same abstraction level the accepted pour repairs do
(`repair_stranded_net`, memory `kicraft-power-plane-strand-no-repair`): stamp
real, guarded copper; verify; give up cleanly and keep the honest
not-fab-ready label when it doesn't land.

## Design

### Detection

Reuse `gnd_pour._collect_net_clusters` (geometric union-find over one net's
pads/vias/tracks/fill islands). For every board net that is not the GND pour
net and not a power-plane net (those have their own pour-aware repairs), a net
with ≥2 electrically-disjoint clusters is unrouted-broken. The largest cluster
is "main"; every other cluster needs a tie.

### Candidate paths (the "bend/via" part)

`breakout_stubs.add_breakout_stubs` already provides polyline stamping with
every guard the pour repairs rely on (foreign-pad margins, stamped/board
copper clearance on the target layer, netclass floors, board-outline inset,
hole-to-hole for tip vias) — a candidate that cannot stamp legally is skipped,
never forced. The straight tie `repair_stranded_net` uses is exactly what
`no_clear_path` rules out here, so per (stranded pad → nearest-K main-cluster
targets) we try, in order, on every feasible start layer:

1. **straight** — `[tgt]` (cheapest; occasionally the wall was another net's
   copper that a different layer dodges);
2. **two L-bends** — `[(sx,ty), tgt]` and `[(tx,sy), tgt]` (routes around a
   rectangular wall corner);
3. **doglegs** — 3-segment paths through midpoints offset perpendicular to
   the straight line by ±1/±2/±4 mm (the "off-grid bend through the blocked
   corridor": freerouting's copper follows its grid habits, and a small
   perpendicular offset frequently threads the gap it left);
4. **layer-change tie** — when the target's copper is not on the start layer,
   the same shapes stamp with `via_at_end=True` (the "one extra via" budget;
   the via inherits the full hole-to-hole/foreign-pad guard set and the spec
   is dropped whole when the via cannot land).

Targets are capped (`signal_repair_max_targets`, default 5) and ties are
distance-capped (`signal_repair_max_mm`, default 40) — this is a LOCAL repair,
not a rerouter. Nets are processed smallest-gap first so cheap wins land
before congested ones consume stamped-copper budget.

### Verification (accept-or-revert)

The pass runs **after** the parent's pour/strand repairs, on the routed board,
from `cli/_compose_route.py`:

1. run `validate_routed_board` (existing call); if `drc.unconnected == 0`,
   nothing to do;
2. snapshot the `.kicad_pcb`, run the repair in a pcbnew subprocess (SWIG
   isolation, like every sibling repair), **refill zones** (new tracks must
   cut clearance through the pours), re-run `validate_routed_board`;
3. accept iff `unconnected` strictly decreased AND `shorts` did not increase
   AND the board still parses; otherwise restore the snapshot byte-for-byte
   and keep the original validation — the round is then rejected with the
   same honest `unconnected_nets` reason as before the repair existed.

The acceptance gate is deliberately outside the repair module: the module
reports what it stamped; the caller owns the evidence-grade verdict.

### Why this is not a band-aid

- It adds real copper connectivity at the netlist's own terms — the same
  operation a human would perform in pcbnew to finish the route.
- Nothing is deleted, waived, or relabeled; a failed repair leaves the board
  byte-identical.
- Every stamped segment passes the same clearance guards the approved pour
  repairs use, then the whole board must survive a full re-DRC to be kept.

## Verify plan (per the ground rules)

Replay the five frozen workspaces (one replay each, measured inside that
replay — never across runs). Target: ≥3/5 reach `unconnected=0` with 0
shorts; none may regress to shorts>0. Then a fresh eval batch for the
fab-ready rate.

## Knobs

| key | default | meaning |
| --- | --- | --- |
| `signal_unconnected_repair_enabled` | `True` | master switch (parent compose path) |
| `signal_repair_max_mm` | `40.0` | max tie length (mm) |
| `signal_repair_max_targets` | `5` | nearest main-cluster targets tried per strand |
| `signal_repair_dogleg_offsets_mm` | `(1.0, 2.0, 4.0)` | perpendicular corridor offsets |
