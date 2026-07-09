# Placement pipeline — reconsideration handoff (2026-07-09)

**Read this first.** This supersedes the *direction* in
[`placement-streamline-handoff.md`](./placement-streamline-handoff.md) (which is now
the history of how we got here). The soft-tidiness path made leaves *look* tidier but
visual inspection showed the tidiness is **cosmetic and often electrically wrong**. We
are stepping back to reconsider placement as a whole. Theme for the next session:
**simplify the pipeline AND make the visual layout emerge from connectivity, not be
imposed on top of it.**

## TL;DR — the verdict

The tidiness work (soft-tidiness scorer term + anchor-less array grouping + the
compaction/clearance fix) succeeded at its stated metric — passives in rows, 100%
orientation consensus, arrays packed 7% → 37% fill, no *aggregate* routing regression on
most designs. **But the metric was measuring the wrong thing.** On inspection, the rows
are tidy and wrong: decoupling caps land **6–20 mm from the IC pins they bridge** (they
belong ~1–2 mm away, snug against the pins), at orientations that fight their routing.
Tidiness was bolted on as a **scoring term over an accreted global optimizer**, so it
pulls a cap into a pretty row and *away* from the pins it exists to serve.

**Recommendation: stop layering tidiness on top. Make tidiness STRUCTURAL — restrict
placement to a discrete grid of allowed slots, and change SA from continuous positioning
to *assignment* (which component goes in which slot), driven by connectivity so each
passive lands in the slot next to the pins it connects to.** Tidiness, legality, and
packing then come *for free* (everything is grid-aligned and pre-spaced), and the
~1000-line "fake tidiness on continuous positions" stack — packer / group-rigid /
orderedness / passive-ordering / alignment / re-snap / the `psw_tidiness` term — becomes
dead code. The simplification and the visual fix are the same change. (Full model below.)

## The evidence (grounded — measured on the shipped `soft` layouts)

Distance from each decoupling cap's body to the nearest IC pin it shares a net with. A
decoupling cap's whole purpose is to sit against that pin; ~1–2 mm is right, 6–20 mm is
broken:

| Leaf | part | nearest connected IC pin | distance |
|---|---|---|---|
| RP2040 MCU | C13 | U1 / +3V3 | 6.2 mm |
| RP2040 MCU | C5  | U1 / GND  | 6.5 mm |
| RP2040 MCU | C10 | U1 / +3V3 | 8.3 mm |
| RP2040 MCU | C8  | U1 / +3V3 | 8.8 mm |
| RP2040 MCU | C11 | U1 / +3V3 | 10.9 mm |
| RP2040 MCU | C9  | U1 / +3V3 | 14.5 mm |
| RP2040 MCU | C6  | U1 / GND  | 15.2 mm |
| RP2040 MCU | C7  | U1 / +3V3 | 18.5 mm |
| RP2040 MCU | C12 | U1 / GND  | 19.6 mm |
| RS485 XCVR | R2  | U3 / B    | 7.0 mm |
| RS485 XCVR | C9  | U3 / ISO_5V | 11.0 mm |
| RS485 XCVR | R1  | U3 / ISO_5V | 15.3 mm |

Reproduce: solve any design via `scripts/soft_tidiness_ab.py`, then for each passive
compute body-center distance to the nearest IC pad on a shared net (the leaf
`solved_layout.json` has per-pad `pos` + `net`). This "pin-locality" number is the
metric we *should* have been optimizing.

## Why the current pipeline produces this

The optimizer never has a per-passive "hug your pins" objective:

- **`net_distance` is a global MST ratsnest, and it excludes GND** (`placement_scorer.py`:
  `n.name not in ("GND", "/GND")`). A cap on `+3V3`/`GND` therefore has almost no locality
  pull — the MST only wants it *somewhere* on the big `+3V3` tree, and its `GND` pad pulls
  nothing at all. So a 9-cap MCU leaf has 9 caps floating on two huge nets with no force
  tying each to *its* pin pair.
- **No pin-pair locality term.** Nothing says "C13's two pads should straddle U1's +3V3
  and GND pins at *this* location." Placement optimizes aggregate wirelength + crossings +
  compactness + tidiness — none of which is pin-adjacency.
- **The tidiness term (`psw_tidiness=0.15`) actively fights locality.** It rewards a
  straight, uniformly-oriented row, which for 9 caps means a neat line — necessarily
  *away* from the 9 different pin pairs they each belong to.
- **The accreted tail** (force + SA + ~26 deterministic passes) legalizes and compacts
  globally; nothing preserves or creates pin-adjacency.
- The **RP2040 base-term routing gap** we found (soft N-of-3 median 23 unconnected vs
  classic 15; pre-existing, not from this session's changes) is the *symptom*: tidiness
  trades away routability because tidy ≠ routable when tidy ignores pins.

## Scoring findings — justifiable improvements (carry into the new model)

Concrete problems in the *current* scoring this investigation surfaced. These are the
objective the grid-assignment SA must optimize (and what to delete):

1. **`net_distance` excludes GND from the ratsnest MST**
   (`placement_scorer._score_net_distance`: `n.name not in ("GND", "/GND")`). Correct for
   *wirelength* — GND is a pour, not routed point-to-point — but it means a decoupling cap,
   whose second pad is GND, gets **zero placement pull from its GND pad**. Its only locality
   signal is the other net (e.g. `+3V3`), which is usually large, so the MST is content with
   the cap anywhere on that tree. Net effect: power/ground passives float away from their
   pins (the 6–20 mm measured above). Fix is NOT to un-exclude GND from wirelength (that
   distorts routing cost) — add a dedicated **pin-locality term** that *does* count GND-pin
   proximity (the cap must sit next to the IC's GND pin for a short via to the plane).
2. **`net_distance` is aggregate MST, not pin-pair.** Even on signal nets, "minimize total
   ratsnest" only asks a part to be near *somewhere* on its net's tree, not against the
   specific pin it bridges. There is no "hug your pins" objective in the scorer at all. Add
   one: per-passive Σ(distance from each pad to the nearest same-net pin on its anchor) + an
   **orientation-to-span** term (reward the part rotated so its two pads line up with its two
   target pins). This doubles as the evaluation metric (see "fix the metric first").
3. **`psw_tidiness` rewards rows regardless of pins** — the term that produced
   tidy-but-wrong. In the grid model tidiness is structural, so it is deleted, not retuned.
4. **Plane nets in the locality term:** because GND (and other planes) are poured, a decap's
   slot should score as "near enough" when a GND via *fits*, not by distance to the nearest
   GND pad — treat plane-net pads as reachable-by-via, not point-to-point.
5. **Already fixed this session (keep):** the tidiness alignment reward was a linear clamp
   that saturated to 0 past `ref_mm` (no gradient, SA only improved orientation) → replaced
   with `100·exp(-resid/ref_mm)` (handoff author); and the leaf clearance 2.84/3.0 two-place
   split → one `leaf_placement_clearance_mm` policy (`a7df6f9`) — which becomes the grid
   **pitch** source.

## What shipped this session (context — keep vs reconsider)

On branch `placement-streamline` (pushed), base = `main` which carries the admin viewer.

**Keep — genuinely useful and orthogonal to the tidiness question:**
- **Admin A/B gallery viewer** (`main`, `654407b`, deployed): `/admin/tidiness-ab`. The
  render+metric loop is how we'll evaluate any new approach — extend it, don't discard it.
- **The compaction / clearance insight** (`a7df6f9`): leaf placement clearance was decided
  in two places with different defaults (canvas 2.84 mm vs solve 3.0 mm); unified into one
  policy (`is_anchorless_passive_array` / `leaf_placement_clearance_mm`). Packing passives
  tight is *correct* and part of a connectivity-first world (a cap hugging its pins IS
  tight). This stays.
- The `$0` A/B harness (`scripts/soft_tidiness_ab.py`) and the diagnostic renderer.

**Reconsider / likely delete once connectivity-first lands:**
- The soft-tidiness scorer term (`PlacementScore.tidiness` / `_score_tidiness`,
  `psw_tidiness`) — optimizes rows blind to pins.
- Anchor-less array grouping's *purpose* (`assign_passive_groups` sub-rows, `245054b`) —
  the R-2R ladder still wants order, but as an emergent property of "each R next to the
  nodes it connects," not an imposed row. The grouping code may survive as an analysis
  tool.
- The dormant packer (`leaf_structured_layout.py`) + group-rigid (`leaf_group_rigid.py`),
  already default-off and falsified.
- Legacy orderedness (Step 8.5) + `apply_leaf_passive_ordering` + `placement_alignment.py`
  + re-snaps — the ~1000-line win, now for the *right* reason.

## Proposed reconsideration — a DISCRETE PLACEMENT GRID, SA as assignment (the direction)

Core idea (user's, 2026-07-09): **tidiness must be structural, not scored.** Define a
limited set of allowed placement locations on a grid; a component may sit *only* on one of
those slots. Everything else follows:

- **Tidiness is free / by construction.** Every part is grid-aligned, so straight rows and
  uniform spacing are guaranteed — there is nothing to *score*. The whole "make it look
  tidy" stack (the `psw_tidiness` term, alignment, re-snap, orderedness, passive-ordering,
  packer, group-rigid) exists only to fake this on continuous positions; the grid makes all
  of it dead code.
- **SA's job flips from *continuous* positioning to *slot* positioning — it keeps a real
  say in where things go, just quantized to the grid.** The grid is deliberately
  **over-provisioned**: for an IC with 20 passives, generate ~200 grid slots around all four
  sides of the IC (most stay empty) and let SA search which passive lands in which slot and
  at what rotation. So SA still explores position + orientation freely and can scatter parts
  around all four sides of the IC if that routes best — but because every candidate is a
  grid slot, the result still reads tidy. What SA *loses* is only the "unusual in-between"
  continuous position that looks messy and helps marginally — those don't exist in the slot
  set. (Grid density is the knob that trades SA freedom against tidiness: too dense ≈
  continuous/messy, too sparse ≈ layouts blow up — see decision (d).)
- **Connectivity drives the assignment.** The routable choice is the tidy choice: a
  decoupling cap is assigned to the grid slot adjacent to its IC power-pin pair; a series /
  filter part to a slot inline on its net. The grid *supplies* pin-adjacent slots around
  each anchor, and SA picks the best occupant for each — so pin-locality (the thing broken
  today, caps 6–20 mm from their pins) becomes the optimizer's actual objective.
- **Legality is by construction too.** Slots are pre-spaced at a courtyard-legal pitch
  (pitch informed by the `leaf_placement_clearance_mm` work), so overlaps *cannot* occur →
  the ~7-step overlap / courtyard / clamp / re-snap tail collapses to nothing.

This is the honest, structural version of "connectivity-first, tidiness emergent": the
simplification and the visual improvement are the **same change**, and both fall out of the
*model* rather than being bolted on.

## Decisions locked (user, 2026-07-09)

- **(a) Anchor-relative grid.** Slots are generated *around each anchor from its pad
  geometry* (pin-adjacent slots on all four sides), not a single global lattice — this is
  what makes "the cap in the slot next to its pins" the natural, tidy, *and* routable choice.
- **(b) Hybrid, anchors first.** Place ICs / edge connectors first (continuous, driven by
  inter-anchor connectivity + edge zones, ~as today), *then* derive the passive grid from
  the placed anchors' pins. SA-assignment is over the passives; anchors seed the grid.
- **(c) Slot orientation is configurable — horizontal, vertical, or both.** A slot declares
  which rotations it admits; the user configures the behaviour (force H lanes, V lanes, or
  let SA choose). Orientation can be structural OR a free SA choice, by config.
- **(d) Grid density is configurable with a sensible default.** Over-provisioned (many more
  slots than parts, so SA has positional latitude) but bounded: the default is tuned so it
  is neither so dense it behaves like continuous placement (messy) nor so sparse that
  layouts blow up in area. One knob, good default, user-overridable.

## Mechanics still to work out (next session)

- **Assignment search.** SA over slot occupancy: swap two parts' slots, move a part to a
  free slot, change a part's rotation within its slot's admitted set — scored by
  routability + pin-locality only (no tidiness term). Likely no continuous refinement
  remains; if any does, it must not leave the grid.
- **Slot generation** around an anchor: how far the pin-adjacent rings extend, how the grid
  fills the content canvas between anchors, and how connectors / large parts that don't fit
  a passive grid are handled (probably continuous, like anchors).
- **Multi-pin nets:** a cap on `+3V3` with several candidate `+3V3` pins → its pin-locality
  target is the nearest admissible pin pair; the assignment resolves this naturally.
- **Fix the metric FIRST** (next section): land the pin-locality metric before the grid so
  the assignment optimizes — and is judged on — the right objective.

Parent compose stays out of scope — leaf-path only (`local_solver_config`); parent `solve()`
untouched.

## The metric we must fix first (do this before any placement change)

The tidiness metric (orientation / residual / fill) rewarded rows regardless of pins and
is *why* we shipped tidy-but-wrong. Add to `leaf_tidiness.py` + the A/B harness a
**pin-locality** metric: per-passive distance from each pad to its nearest same-net IC pad
(and an orientation-to-span term), aggregated per leaf. Then a layout is "good" when
passives hug their pins **and** it routes — and every A/B page shows it. Land this metric
first so the next placement change is measured against the right thing.

## Pointers

- Branch `placement-streamline`: `245054b` (grouping), `a7df6f9` (compaction/clearance).
- `kicraft/autoplacer/brain/placement_solver.py` — `solve()` force+SA core + ~26-step tail.
- `kicraft/autoplacer/brain/placement_scorer.py` — `_score_net_distance` (GND excluded!),
  `_score_tidiness`.
- `kicraft/autoplacer/brain/leaf_size_reduction.py` — `local_solver_config` (leaf-only cfg),
  `leaf_placement_clearance_mm`.
- A/B: `scripts/soft_tidiness_ab.py` → `logs/tidiness_ab/run-*/index.html` → admin
  `/admin/tidiness-ab`.
- Memory: `kicraft-placement-streamline-plan` (running log of the whole effort).
