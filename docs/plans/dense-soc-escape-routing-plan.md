> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Dense-SoC inner-pad escape — plan (v1, 2026-07-23)

## RESULTS (2026-07-23, implementation session — read this first)

**All five phases implemented**, branch `placement-streamline`, uncommitted, not
deployed. Kill switches: `escape_planner_enabled` (default on) and the
`fab_capability` config block (set it back to the legacy values for HEAD
behaviour — that is exactly how the baseline below was produced).

**Witness A/B — KC-RYVSQV (1/661).** One `replay --quality fast --seed 0` per
side, two independent workspace copies, identical code both sides (the baseline
is this plan's own kill switches: legacy floors, 0.6/0.3 netclass via, planner
off). Both parent boards DRC'd against their own project rules:

| | baseline (= HEAD) | with the plan |
| --- | --- | --- |
| parent DRC errors | 0 | 0 |
| parent unconnected items | **5** | **4** |
| parent unconnected nets | DEC3, **DECUSB**, GND, **XL1**, **nRESET** | ANT_RF, DEC3, GND, VBAT |
| NRF52840_MCU leaf, local-signal opens | **4** (DEC3, DECUSB, XL1, nRESET) | **2** |
| `escape_infeasible` | n/a | **[]** — nothing geometrically stranded |
| escape plan on U1 | n/a | 14 open, 4 tie, 7 lane, **1 via**, 0 infeasible |

**The verdict is not the count, it is the failure mode.** Baseline: *every* leaf
round fails on *the same four nets* — DEC3, DECUSB, XL1, nRESET — because they
are geometrically unreachable, exactly as §1 diagnosed. With the plan, `nRESET`
and `DECUSB` never appear again in any round, and the residue **changes every
round** (`SWDIO` / `DEC1,DEC3,XL2` / `VBAT,DEC6,XL2` / `VBAT,DEC3,DCC`): an
invariant geometry failure has become ordinary seed-dependent router residue.
That is the thing this plan set out to remove.

**It is not rc0, and the remaining opens are a PLACEMENT problem, not a routing
one.** I traced DEC3 on the routed leaf, and the earlier "open-field congestion,
leave it to a router" framing was wrong on the facts. DEC3 escapes U1's north
edge cleanly to ~`(148.2, 103.4)`; its capacitor C9 sits ~4.6 mm away at
`(152.2, 100.9)`; and the straight line between them runs directly through a pad
of C3 (a crystal load-cap, a 0.9 mm GND pad). That corner is a *second* dense
cluster, not open field: the chip's whole north outer pad row, the crystal caps,
and the escapes for five other nets (DEC5, DEC6, XC1, ANT_RF, GND) all thread the
same ~5×4 mm patch, with C9 parked on the far side of it from where DEC3 exits.
FreeRouting gives up there not because it is weak — on genuinely open copper it
would close this instantly — but because we handed it an over-congested corner.
The fix is squarely on the KiCraft side and respects "leave routing to
FreeRouting": **decongest the landing** so a legal slot for the companion
actually survives next to where the net emerges. (There is no in-house router to
fall back on — that plan was scrapped by design principle, not shelved. Any
earlier note here or in the code comments implying a C1-v2 / A\* owner for this
residue is stale.) P3 already aims the cap at the landing — the pad and the exit
are ~0.8 mm apart, essentially the same spot, so "pad vs landing" is not the
issue and C9 did move a lot closer (11.25 → 4.87 mm off U1.D23). What P3 does NOT
do is make *room* there: the landing sits in the leaf's most packed corner
(U1's outer pad row + the crystal load-caps + five other escaped nets in one
~5×4 mm patch), so no legal cap slot survives at the exit and the placer bumps
C9 to the nearest free slot — across the crystal caps, where a C3 GND pad then
lands square on the straight path.

**Tried the obvious lever (reserve + pin the companion at its landing) — it does
NOT work, and the measurement says why.** Clean single-round, same-seed A/B on
the U1 leaf (`solve_subcircuits --only "NRF52840 MCU" --rounds 1 --seed 0
--route`, one workspace, only the reservation toggled):

| | reservation off | reserve + pin C9 at the landing |
| --- | --- | --- |
| C9 → DEC3 pin | 6.90 mm | **2.88 mm** (held ~0.5 mm off the escape tip) |
| DEC3 routed? | no | **still no** |
| leaf unconnected | 3 | **4** (pinning stranded DEC2) |

Two facts fall out, both fatal to the "seat the companion" idea. First, **the
seed alone is erased by the SA** — pin-locality is only ~18 % of the placement
score, so the Metropolis wander drags C9 5–8 mm back off the landing every seed;
holding it there needs a hard *pin*. Second, and decisively, **even pinned
0.5 mm from the escape tip DEC3 still will not route, and pinning strands DEC2**.
So companion *distance* was never the blocker — the congested corner is, and it
blocks the short hop just as it blocked the long one; meanwhile forcing caps into
fixed landing slots over-constrains the placement and pushes the strand elsewhere.
This attempt (grid landing-reservation + SA pinning + a kill switch) was
implemented, measured, and **reverted** as a net regression.

**The lever that remains is genuine decongestion of the corner, and it is
unbuilt:** spread U1's north-edge crystal/decap pile so the escape corridors are
physically clear, rather than relocating one cap into a jam. That is a real
placement change (edge-cluster spreading / escape-corridor keep-outs), not the
one-line seating tweak this attempt hoped it was. Left as the honest next step.

Other verification: 39 new golden tests on the checked-in aQFN-73 pad table; full
suite **8 failed / 2968 passed**, every one of the 8 confirmed pre-existing by
re-running them at HEAD (two further tests pinned the old 0.153 floor and were
updated to read it from the profile, since the floor deliberately moved); fresh
`synthesize` confirms the synthesis-side floors and that the netclasses are
untouched; escapability lint swept over 44 vendored + 157 home-fetched bundles;
all four kill-switch paths exercised — with the legacy `fab_capability` restored
the planner reports `U1.AC13:nRESET` INFEASIBLE by name instead of stamping the
nub, which is the P4 honesty win in a single call.

**Blast radius is narrow by construction.** The planner only acts on pads that
are NOT on their pad field's outer row. On `run_10_rp2040-min` — the RP2040
QFN-56, a single-ring package — it emits **zero** specs and zero infeasible
across every leaf (U1: all 54 netted pads `open`; J1 USB-C and J2 header
likewise), so that genre isolates the floors change from the escape machinery.

**Genre check, `run_10_rp2040-min` A/B (same method).** At the deterministic
leaf level: J1 leaf accepted 0/0 both sides; the U1+U3 leaf improves **30 → 26**
unconnected — from the finer stamped escape copper alone (every stamper now lays
0.127 mm instead of 0.153 mm), since the planner contributes nothing here. Both
sides then fail at rc6 on the parent; the *same* FreeRouting crash
(`rc=-1` → retry → GND-skipped retry) happens on both, and only the baseline
happened to recover from it. That is router nondeterminism on a June-30
workspace that was already broken at HEAD, not a regression — but it is also not
a clean regression test, and should not be read as one.

**Untested risk, stated plainly:** the fine-pitch clearance auto-lower never
fired in any of the four replays, so moving `freerouting_min_clearance_mm` from
0.153 to 0.127 had **no observed effect at all**. Its blast radius (a board with
a genuinely sub-0.153 mm pad gap — USB-C class — where FreeRouting would now be
allowed to squeeze to 0.127 board-wide) is unexercised here, not disproved. If
anything regresses on the next batch, that knob is the first suspect and reverts
independently.

### Three findings that changed the plan's own numbers

The plan was right about the shape of the problem and wrong about two constants.
Both were caught by building the geometry honestly and measuring it.

1. **0.4/0.2 does not fit. The fanout class is 0.36/0.15.** Two bounds, pulling
   opposite ways, pin it — see `autoplacer/fab_profile.py` for the derivation and
   the full table.
   *Upper:* the 0.4 via has a real 17 µm AC13 window at the bare 0.153 rule, but
   `breakout_stubs` holds a **+10 µm geometry guard** above every clearance rule
   (added for a measured KiCad/HitTest skew on *this very footprint*), and that
   closes it.
   *Lower:* a thin annular ring creates a rule FreeRouting cannot see. It knows
   only the copper clearance, so a track it places legally against a via's
   annulus can sit under KiCad's hole-to-copper minimum. The invariant is
   `netclass_clearance + annular_ring >= hole_to_copper`, which forces a drill of
   0.18 mm or finer. A first attempt at 0.35/0.2 (ring 0.075, 22 µm short) proved
   it immediately: FreeRouting ran a B.Cu track 0.2417 mm from the nRESET fanout
   hole against the 0.25 mm rule, and an otherwise perfect **zero-unconnected**
   leaf round was discarded for it.
2. **0.5/0.3 has no AC13 window at any clearance.** §1 expected one to open at
   0.127 (`o ∈ [0.052, 0.065]`); honest geometry says the exposed pad needs
   `o >= 0.052` while the diagonal to AD12/AD14 needs `o <= 0.019`. The small
   fanout class is *required*, not preferred — which is why the fab-capability
   check was load-bearing. Verified 2026-07-23 against JLC's published 2-layer
   1 oz capability: 0.10/0.10 mm track/space, 0.15 mm minimum via hole, 0.25 mm
   minimum via diameter. Our floors sit at **0.127** — deliberately above their
   minimum, and exactly 127 µm so the DSN's whole-micron rounding trap that
   motivated 0.153 does not apply.
3. **Lane-first, not via-first.** §P2 specified via-first, arguing the dog-bone
   needs no lane coordination. With the lane assignment actually built (capacity
   *and* position rationing), that stops paying: via-first drops all 12 inner
   netted pads — VBAT, ANT_RF and the crystal included — onto B.Cu, twelve
   punctures through the GND plane of a 2-layer board. Order is now
   same-net tie → lane → via, and on the witness leaf that is 4 ties, 7 lanes and
   **one** via: AC13, the wall-locked pad nothing else can solve.

### What each phase became

- **P0** `autoplacer/brain/escape_planner.py` — pure geometry, no pcbnew. Per
  netted pad: `open` (outer row — the router's job, unchanged) / `tie` / `lane` /
  `via` / `infeasible`. Lanes are gaps in the outer pad row, rationed by
  `floor((gap − c) / (track + c))` **and** by position, so two escapes cannot be
  assigned 0.26 mm apart inside one 1.5 mm lane. Golden fixture
  `tests/data/aqfn73_pads.json` is the real KC-RYVSQV pad table.
- **P1** `autoplacer/fab_profile.py` — one source of truth, read by
  `autoplacer/config.py` and `design/synthesis/kicad_pro.py`. Floors are stamped
  into every project `.kicad_pro` at `cli_app._run_layout`, which covers `build`
  AND `replay`: the first attempt hooked `autoexperiment.main`, and
  `replay --quality fast` never enters it — the run silently tested the old
  rules. The fanout class is added to `via_dimensions` so it round-trips through
  the DSN; each netclass keeps its own `use_via`, so FreeRouting's own vias are
  unchanged (scope guard confirmed: netclasses still 0.153 / 0.2 / 0.6 after a
  fresh `synthesize`).
- **P2** `breakout_stubs.escape_planner_specs` + `leaf_routing` wiring. Planner
  specs stamp first; pads it owns are removed from the legacy radial stampers, so
  an `infeasible` pad gets **no copper** rather than the 0.2 mm nub. Escapes the
  board-level guards drop are recorded, not hidden. `BreakoutSpec` grew per-spec
  via sizing (the netclass via cannot express a dog-bone).
- **P3** `leaf_grid_assignment.escape_landings` — slot origins move to where a
  net actually surfaces. Ties are excluded: their landing is the exposed pad,
  *inward*, so using it would delete slots rather than relocate them.
- **P4** `escape_infeasible` is a distinct leaf rejection reason and a member of
  `_STRUCTURAL_UNROUTABLE_REASONS`, so an unreachable pad aborts in one round
  instead of nine. `validate-part` gained check (11).

### Open / deliberately not done

- **validate-part (11) is a WARNING, not the hard failure §P4 specified.** A
  library footprint has no nets, so the check runs the pessimistic
  all-distinct-nets model: it answers "is this POSITION escapable", a fact about
  the package, not about any board. Swept over the library: **0 of 44 vendored**
  and **3 of 157 home-fetched** bundles hit — all three the nRF52840, naming
  AC15/B15/M2/V23, four ring positions no design has ever netted. Failing a
  working part on a hypothetical would be wrong; the hard gate belongs at the
  leaf, where `escape_infeasible` names a pad that demonstrably carries a net.
  (Useful new fact regardless: the nRF52840 has four pins that cannot be routed
  on two layers at all.)
- **The recurring home-fetched bundles were NOT vendored** (`nrf52840`,
  `cr2032-holder`, `chip-ant-2450at43b100e`); they were linted in place instead.
  `cr2032-holder` fails the pre-existing check (10) model-frame test.
- **Self-eval batch not run** — real money, needs the user's go-ahead. The 53-run
  class is the target population.
- **Observation, 1 of 4 rounds:** FreeRouting emitted a `tracks_crossing` pair
  (GND × SWDCLK, its own 0.127 mm traces, one segment routed twice) on the
  witness leaf. The leaf gate caught it and the round was discarded; the shipped
  rounds have 0 DRC errors. Not planner copper — but the finer escape track does
  change what FreeRouting's fine-pitch mode does, so it is worth watching.

Owns the **residual** of `dense-soc-leaf-unconnected-plan.md` (P0–P3 shipped `1638a27`:
placement now delivers pin-adjacency, the leaf still rejects `no_unconnected`). Witness run:
**KC-RYVSQV** (`~/.kicraft/projects/1/661`, built 2026-07-23 13:16 UTC **with** `1638a27` —
`grid_score`/`grid_discarded:false` markers present in the leaf debug). Brief: nRF52840 BLE
beacon. rc7: shorts=0, unconnected=6; the NRF52840-MCU leaf fails `no_unconnected` on
`DEC3, DECUSB, XL1, nRESET` in all 9 rounds; parent adds 2 GND opens (U1 inner pads B7/F23).

**Class definition:** a fine-pitch SoC package with an inner pad ring (nRF52840 aQFN-73;
same family: ESP32 `XTAL_N`/`EN`, RP2040 `XIN`/`DVDD`, STM32 `OSC`/`RESET`) has netted pads
whose only exits are narrow designed lanes. KiCraft's escape stamping is lane-blind and its
rule floors are one fab-generation too coarse, so the pads are **geometrically unreachable
before freerouting ever runs**. Breadth: **53 of 574 runs** with layout artifacts have an
MCU/SoC leaf rejected `no_unconnected`; the DEC/XL/RESET net family recurs on every nRF52
board (`run_13_nrf52-beacon` on 4 self-eval batches, live `1/620`, `1/660`, `1/661`),
latest hit 2026-07-23 — live **after** all deployed fixes.

**Not the owner:** C1-v2 phase-3 grid A\* (SHELVED 2026-07-21 by user ruling: no in-house
router). This plan needs no router — the defects below are in deterministic pre-route
stamping and rule configuration, both established KiCraft practice (`breakout_stubs.py`).
The honest open-field residue that remains after this plan (see P3) is what stays with the
shelved A\* fallback.

## 1. Measured evidence (KC-RYVSQV MCU leaf, `AQFN-73_L7.0-W7.0-P0.50-BL-EP4.8`)

Package geometry (all mm, relative to U1 center; pads 0.25×0.25 on 0.5 pitch):

- Outer ring at ±3.25, inner ring at ±2.75, EP 4.85×4.85 at center.
- Inner-pad↔EP gap **0.20**; inner↔outer ring channel **0.25**; adjacent-pad gap **0.25**.
- Depopulated lane (missing ring position) **0.75** edge-to-edge; same-row diagonal
  opening **0.457**; corner regions (rows end early) **≥0.5**.

Corridor feasibility — a trace needs `w + 2c`; two sharing a lane need `2w + 3c`:

| corridor | width | today (w=0.153, c=0.153) | JLC capability (w=c=0.127) |
| --- | --- | --- | --- |
| adjacent pads / ring channel | 0.25 | never (0.459 > 0.25) | never (0.381 > 0.25) |
| inner-pad ↔ EP gap | 0.20 | never | never |
| same-row diagonal | 0.457 | ✗ **by 2 µm** (0.459) | ✓ (0.381) |
| depopulated lane, 1 trace | 0.75 | ✓ (0.459) | ✓ |
| depopulated lane, 2 traces | 0.75 | ✗ **by 15 µm** (0.765) | ✓ (0.635) |

The four failed pads, from the routed board (`leaf_routed.kicad_pcb`, stub widths 0.153 =
the pre-stamped escapes; freerouting traces are 0.2):

| pad | net | what the stamper did | why it failed | feasible? |
| --- | --- | --- | --- | --- |
| D23 (−2.0,−2.75) | DEC3 | full escape through its private B24–E24 lane to (−2.38,−3.28) ✓ | freerouting failed the **open-field** remainder to C9 (5.9 mm away, congested north side) | today |
| D2 (−2.0,+2.75) | XL1 | **0.2 mm nub** into the dead ring channel | its radial ray hits pad C1; the open C1–G1 lane center is 0.36 mm sideways — and XL2's escape already fills that lane (2 traces need 0.765 > 0.75) | at 0.127 (lane fits two) |
| AC5 (+2.75,+2.0) | DECUSB | **0.2 mm nub** into the dead ring channel | radial ray points at the AD4/AD2 wall; the open 0.5 mm AB2–AD2 corner exit is southward, non-radial | today, with a dogleg |
| AC13 (+2.75, 0.0) | nRESET | **0.2 mm nub** facing the fully-populated AD column | no on-layer exit exists at ANY rule; freerouting built a 14 mm approach to within 0.3 mm and gave up | via fanout only: a **0.4/0.2 via at (+2.80, 0.00)** clears EP by 0.175, AC11/AC15 by 0.177, AD12/AD14 by 0.190 — **legal even at today's c=0.153**. (A 0.45 via needs c=0.127: EP gap drops to 0.150. The current 0.6/0.3 netclass via has NO legal position at any offset.) |

Parent-level GND opens B7 (−2.75,+1.5) / F23 (−1.5,−2.75): same trapped-inner-pad geometry;
the F.Cu pour cannot enter the 0.25 channel, and no plane-bond via was stamped.

XL2 (F2) routed only because its fixed radial ray *happens* to thread the C1–G1 lane. That
is the entire difference between the nets that route and the nets that fail.

**Footprint verified correct (2026-07-23, do not chase this):** the home-fetched
`AQFN-73_L7.0-W7.0-P0.50-BL-EP4.8` was diffed pad-for-pad against the official KiCad
`Package_DFN_QFN:Nordic_AQFN-73-1EP_7x7mm_P0.5mm` — 73 pads + EP, **identical names,
sizes, and occupancy** (our copy is the official pattern rotated 90°, name-consistent;
the apparent per-pad "mismatches" are that global rotation). The symbol pin map also
matches the official `MCU_Nordic:nRF52840` exactly (AC13 = `P0.18/~RESET`, AC5 = DECUSB,
D23 = DEC3, D2/F2 = XL1/XL2, B7/F23 = VSS/VSS_PA). The fully-populated AD-column wall and
the sparse depopulated lanes are the **real package** — there is no extra routing room to
reclaim by fixing the footprint, and the intended escape for wall-locked inner pads is a
via fanout (§P2).

## 2. Root-cause chain (all live at HEAD, each with its source point)

- **D1 — lane-blind escape stamping.** `breakout_stubs.py:_radial_escape_end` (~305)
  marches in ONE fixed direction (radial from footprint center) and returns `best_short` —
  the farthest legal point — when blocked. For a ring package that stamps an unreachable
  0.2 mm nub inside a dead channel: it satisfies nothing, becomes a foreign-copper obstacle,
  and turns the DRC edge into `Track [net] ↔ Pad` (the "stub is the obstacle" note in
  dense-soc-leaf-unconnected-plan.md's disproved section was this defect seen from the
  other side). No lane detection, no dogleg, no per-footprint coordination, no
  infeasibility verdict. `auto_signal_escape_specs` (~698) also keys on the "spread-power
  connector" signature — an SoC only gets escapes by accident of its VBAT pad count.
- **D2 — oversized via class blocks the standard escape.** The netclass via is
  **0.6/0.3**. A 0.6 via has *no legal position* anywhere near the ring — EP and
  outer-pad clearances cannot both clear at any offset, and in a 0.75 lane it needs
  0.906 mm — so neither freerouting (which only has this via) nor any stamper can do the
  classic dog-bone fanout that fine-pitch inner rings are *designed* for. A **0.4/0.2
  via fits beside every inner-ring pad at today's 0.153 clearance** (offset window
  o ∈ [0.03, 0.09] outward; the AC13 numbers in §1). This, not the clearance floor, is
  the binding constraint for the primary fix.
- **D3 — legacy fab floor one generation too coarse (secondary).**
  `design/synthesis/kicad_pro.py` DEFAULTS `min_clearance`/`min_track_width` = 0.153 and
  `autoplacer/config.py` `freerouting_min_clearance_mm`/`freerouting_fine_pitch_track_mm`
  = 0.153 encode the **OSH Park 6 mil** floor. The pipeline's actual fab target is JLC
  (the BOM gates on JLC assembly + LCSC retail), whose 2-layer capability is **5 mil =
  0.127** track/clearance. The 6-mil floor closes the *on-layer* corridors: lane-sharing
  misses by 15 µm, the same-row diagonal by 2 µm; it also decides whether a 0.45-or-larger
  fanout via is legal (EP gap 0.150) and how tight the 0.5/0.3-via fallback window is.
  (`types.py:BreakoutSpec.width_mm` already defaults 0.127 — raised to 0.153 only to clear
  the legacy floor. 0.127 = exactly 127 µm: the DSN integer-µm rounding trap that motivated
  0.153 does not exist at 0.127.) With via-fanout as the primary strategy this becomes an
  enabling margin, not a hard prerequisite — unless JLC's 2-layer via minimum turns out to
  be 0.5/0.3 (see P1), in which case 0.127 clearance becomes required again.
- **D4 — no feasibility verdict anywhere.** Infeasibility is discovered as freerouting
  exhaustion at leaf round 9-of-9 (~2 min/round), reported indistinguishably from router
  failure, and autoexperiment keeps mutating *placement* to fix a *geometry* constant.
  Nothing at validate-part time checks that a footprint's netted pads are escapable at the
  fab profile — the nrf52840 bundle (home-fetched, not even vendored) shipped with an
  inner ring that today's rules cannot route, and no gate said so.

## 3. Fix program

Order matters: P0 is the shared engine; P1 is a config truth fix; P2/P3 consume both;
P4 closes the gates. Each phase lands independently with its own kill switch.

### P0 — escape planner (new module, pure geometry)

`kicraft/autoplacer/brain/escape_planner.py`. Input: a footprint's pad field (positions,
sizes, nets, EP) + a rule set (`track_mm, clearance_mm, via_diameter_mm, via_drill_mm`).
No pcbnew mutation — analyzable and unit-testable standalone.

1. **Trapped-pad detection:** a netted pad is *trapped* when no straight ray of required
   width (`w+2c`) reaches open copper (outside the footprint courtyard + margin) without
   crossing a foreign pad. Replaces the "spread-power connector" heuristic — any package
   qualifies by geometry alone (QFN crystal pins, aQFN rings, dense connectors).
2. **Exit enumeration per trapped pad — via first:** (a) **via fanout** (the primary,
   uniform strategy — the classic dog-bone): search a legal via center in a small disc
   around the pad (the AC13 calc in §1 is the template: clearance to EP, ring neighbors,
   foreign pads, and previously-placed fanout vias all ≥ c; EP thermal vias are B.Cu
   obstacles). The disc search naturally handles adjacent netted pads — two 0.4 vias at
   0.5 pitch violate via-via clearance inline, but shifting one along the ring into an
   empty neighboring position (e.g. D2's via shifts toward the empty (−2.5,+2.75) slot)
   separates them legally; no special-casing, just honest clearance checks in the search.
   (b) on-layer fallback: straight rays swept at fine angular steps (the radial-only
   assumption is D1), then one-bend doglegs through *lane centers* — lanes found by
   scanning each ring row/col for gaps ≥ `w+2c`, corners included, with capacity
   `floor((lane − c) / (w + c))` and scarcest-first assignment.
3. **Verdict per pad:** `FEASIBLE_VIA(center, landing)` | `FEASIBLE(polyline)` |
   `INFEASIBLE` — computed at *both* the current rules and the capability rules, so the
   caller can report "feasible only at 0.127" distinctly from "never feasible".

Golden unit fixture: the exact aQFN-73 pad table from §1 (checked in as data). Expected
at (via 0.4/0.2, c = 0.153): **all six trapped pads FEASIBLE_VIA** (AC13, D2, AC5, D23,
B7, F23), incl. the D2/F2 adjacent-pair shift; lane verdicts: D23/AC5 also FEASIBLE
on-layer, D2 lane-share only at 0.127 (0.765 > 0.75 by 15 µm), same-row diagonal flips at
0.127 (0.459 vs 0.457); at via 0.5/0.3 the AC13 window exists only at c = 0.127
(o ∈ [0.052, 0.065]); at via 0.6/0.3 everything is INFEASIBLE_VIA. This pins the 2 µm/15 µm
margins so a future "harmless" constant bump fails a test instead of a batch.

### P1 — fab capability profile (single source of truth)

New block in `autoplacer/config.py`:

```python
"fab_capability": {           # JLCPCB 2-layer 1oz standard — verify against the
    "min_track_mm": 0.127,    # current JLC capability page at implementation time
    "min_clearance_mm": 0.127,
    "min_via_diameter_mm": 0.4,   # fanout-via class; THE load-bearing verification:
    "min_via_drill_mm": 0.2,      # if JLC's 2-layer minimum is really 0.5/0.3, the
},                                # AC13 window needs c=0.127 (P0 goldens cover both)
```

The **fanout-via class (0.4/0.2)** is the unlock for the primary strategy (D2): expose it
as a via padstack in the DSN export too, so freerouting can place dog-bones *itself* where
the deterministic stamps leave anything open — this keeps "outsource routing to
freerouting" true while P2 guarantees the trapped pads deterministically. The single
load-bearing external check before merge: JLC's current 2-layer via minimum (0.4/0.2
comfortable everywhere at today's clearance; 0.5/0.3 workable only with the 0.127 floor
and a razor-thin AC13 window; if even that fails, the honest fallback is the P4 4-layer
escalation for aQFN-class parts).

- `design/synthesis/kicad_pro.py` DEFAULTS floors read from/mirror this block (synthesis
  side — **replay cannot verify this edit**, the seed `.kicad_pro` is frozen; verify with
  the offline `synthesize` subcommand or a full `build`).
- The autoplacer additionally **stamps the floors into the leaf/parent board
  `design_settings` it writes** (locate where the leaf `.kicad_pro` is copied in
  `solve_subcircuits`; stamp there) so `replay` exercises the change.
- **Scope guard:** only the *floors* (`min_track_width`, `min_clearance`) and the stamped
  escape copper move to capability values. The Default netclass stays 0.2/0.153 —
  freerouting's own routing behavior is unchanged, so this cannot regress general routing.
  `freerouting_clearance_guard_um=10` semantics unchanged (DSN-only guard above the rule).
- `freerouting_fine_pitch_track_mm` 0.153 → 0.127 (its comment block documents why 0.153
  existed; 127 µm has no rounding trap). `BreakoutSpec.width_mm` stays 0.127, now legal.

### P2 — coordinated escape stamping (replaces the radial nubs)

In the `leaf_routing.py` stub orchestration (~589–737):

- For every footprint with trapped netted pads, drive stubs from the P0 planner —
  **via-fanout first** (per the 2026-07-23 re-scope): stamp the dog-bone (short pad stub +
  0.4/0.2 via + short B.Cu landing stub into open space under/around the chip) for every
  `FEASIBLE_VIA` pad. This is uniform, placement-independent, and needs no lane
  coordination — the simplest correct mechanism, and the one the package was designed for.
- On-layer polyline escapes (extend `BreakoutSpec`/`add_breakout_stubs` to multi-segment
  paths) remain the fallback for pads where no legal via spot exists, using P0's lane
  assignment so two pads never fight over one lane.
- Trapped **GND** pads get the same fanout via, bonded to the B.Cu plane (fold the
  existing GND pre-escape for fine-pitch pads into the same planner path).
- **Never stamp a nub:** an `INFEASIBLE` pad gets *no* copper (today's nub is an obstacle
  plus a false "partially routed" signal) and is recorded in the leaf debug + acceptance
  as `escape_infeasible:<ref>.<pad>:<net>` — honest at round 1, not freerouting
  exhaustion at round 9.
- Keep the existing collision guard as the final legality check on every stamped segment.

### P3 — partner seating at the lane exit + router handoff

- `leaf_grid_assignment` currently targets the *pad center* when seating a trapped pad's
  companion (C9 ended up 5.9 mm away on the wrong side of the chip). Target the planner's
  **escape landing point** instead — the decap belongs next to where the net actually
  emerges. Small change, large effect on the remaining open-field hop.
- After P2, freerouting's job on these nets is stub-landing ↔ adjacent partner over open
  copper. Whatever still fails there is the true C1 walled-off residue — measure it after
  P2/P3 land; only if it stays material does the shelved A\* fallback re-enter discussion.

### P4 — gates + honesty (the "which gate should have caught it" answers)

- **validate-part lint (new numbered check):** run the P0 planner over every vendored
  footprint's netted-pad-capable pads at the capability rule set; any `INFEASIBLE` pad
  fails `add-part`/`validate-part`. This catches the AC13 class the day a part is
  vendored — the earliest deterministic gate. Run it over the existing vendored library
  once; also vendor the recurring home-fetched bundles this board used
  (`nrf52840`, `cr2032-holder`, `chip-ant-2450at43b100e`) and lint them.
- **Leaf acceptance:** split `no_unconnected` failure annotation into
  `escape_infeasible` vs `router_fail` so autoexperiment stops burning placement rounds
  on geometry constants (`escape_infeasible` is invariant under mutation — fail fast).
- **Recorded, out of scope:** sourcing policy for BLE briefs (certified module vs bare
  aQFN73 — modules sidestep antenna layout + certification entirely) and a 4-layer
  escalation hook when a board's parts are infeasible at the 2-layer capability profile.

## 4. Verification

1. **Unit ($0, seconds):** P0 goldens on the §1 fixture, including the rule-set flips.
2. **Leaf integration ($0):** copy KC-RYVSQV's workspace; one
   `solve_subcircuits --only "NRF52840 MCU" --rounds 3 --route` per side (HEAD vs plan),
   same seed, same copy. Expect: leaf unconnected 4 → 0 (accept 4 → ≤1 if DEC3's
   open-field hop still fails — then P3's seating is the follow-up), zero dangling stubs
   on failed nets, `escape_infeasible` empty at capability rules. Never compare across
   separate replays; measure both sides inside one script.
3. **Genre replays ($0):** full `replay` of 1/661 (expect rc7 → rc0: the parent GND pair
   closes via the P2 GND fanouts), `run_13_nrf52-beacon`, `1/660`, `run_10_rp2040-min`
   (XIN class). P1's kicad_pro floor change needs a fresh `build`/`synthesize` to be
   visible — replay alone under-tests it (frozen seed).
4. **Batch (real $, ask user):** self-eval; the 53-run class is the target population;
   success = nrf52-beacon genre flips to fab-ready and no regression on the clean 208.

## 5. Rollout / kill switches / risks

- `escape_planner_enabled` (default on once fixtures pass; off = legacy radial stamper),
  `fab_capability` floors behind config. P0→P4 are independent commits in order.
- **Risk: capability constants wrong for the real fab** → P1 explicitly verifies against
  JLC's published 2-layer capability before merge; the profile is one block, trivially
  revertible, and the DRC still gates at whatever floor is configured.
- **Risk: via-in-pad-adjacent fanout and assembly** (solder wicking on 0.25 mm pads):
  fanout vias are *beside* the pad (0.05 mm offset in the AC13 calc), not in it; flag for
  eyeball on the first 3D render. Tent-via note in the exporter if it proves real.
- **Risk: 0.127 floors let freerouting legally squeeze elsewhere** → floors ≠ netclass;
  the router still routes at 0.2/0.153 defaults (scope guard in P1).
- **Concurrent session:** `1638a27` and this run share the checkout with another active
  session; land this plan's code on `placement-streamline` after coordinating, and re-run
  the §4.2 A/B at whatever HEAD is current then.

## 6. Appendix — investigation audit trail (KC-RYVSQV)

- ERC clean; synthesis stages all ok; BOM 15/15 parts REAL + in stock (0 hallucinated);
  no mechanical-intent constraints; BOM matches the brief (nRF52840-QIAA, 2450AT43B100E
  chip antenna, CR2032 holder, tactile button, SWD header).
- §8 wheel-spin: `bom rounds=6` (maxed) — known cost driver
  (`kicraft-pipeline-cost-bom-retries`), not this board's failure.
- §6 provenance: `nrf52840`, `cr2032-holder`, `chip-ant-2450at43b100e` are home-fetched
  (coverage gaps; vendor per P4).
- Parent DRC: shorts 0 / unconnected 6 (`GND×2, DEC3, DECUSB, XL1, nRESET`), clearance 0,
  courtyard 0 — the *only* thing between this board and fab-ready is this plan's class.
