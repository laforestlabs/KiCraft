# Handoff: dense power-leaf sprawls / doesn't route (USB-C + LDO)

**Status:** diagnosed end-to-end, not fixed. One build-orchestration change landed
(`feat/build-two-phase-leaf-parent`, two-phase `_run_layout`) — necessary but not
sufficient. This doc is the full diagnosis + a prioritized fix plan for the next
agent.

**Symptom (user-reported):** generated PCBs "look terrible" — a sparse power leaf
(USB-C receptacle + LDO + ESD + CC resistors + decoupling) is placed strung-out
across a ~150 mm strip, and `kicraft build` exits **rc=6** (parent compose/route
fails). Multi-IC sensor boards reportedly placed compactly before.

**Canonical repro design:** the ESP32-S3 plant monitor (5 sheets: `POWER`,
`ESP32 S3`, `BME280`, `OLED`, `SOIL MOISTURE`). A saved `state.json` exists under
`logs/self_eval/20260608T151858Z_rerun/run_01_AN_ESP32_S3/.kicraft/state.json`
(logs/ is gitignored; regenerate via the kicraft pipeline if absent). The `POWER`
sheet = `J1` USB-C `TYPE-C-31-M-12`, `U1` LDO `AP2112K-3.3`, `D1` ESD `USBLC6`,
`R1`/`R2` 5.1k CC pulldowns, `C1`–`C4` caps. 2 interface ports, low net density.

---

## TL;DR of the root cause

It is **not** that force-directed/SA placement is off, and **not** (only) that the
scorer prefers sprawl. The real chain:

1. The leaf solver produces BOTH tight (~16 mm) and sprawled (~104 mm) placements
   across its rounds. Placement works.
2. The leaf **round score** discards any routing-FAILED round (`-inf`):
   `solve_subcircuits.py:562-572`. A round is "failed" when the routed board is
   rejected for `illegal_routed_geometry`.
3. The **tight placements fail routing** — FreeRouting routes a **VBUS track
   straight across the LDO `U1`'s pads** (a real short), so every tight round is
   `-inf` and thrown out.
4. The **sprawl routes "well enough"** (no shorts, ~5 unconnected) → it gets a real
   score and is **pinned by default** (`autoexperiment.py:_auto_pin_best_leaves`,
   line 231, ranks by `(routed, score)`, skips `-inf`).
5. So the pinned `POWER` leaf is the sprawl → parent can't compose → rc=6.

So "A" (scoring prefers tight) and "B" (route the tight cluster) collapse into one
problem: **the router shorts on the tight placement, so the tight placement is
(correctly) rejected and the sprawl wins**. The fix must make the tight, compact
placement *route cleanly* — not just bias the score toward it (biasing alone would
pin a shorting board).

---

## Evidence

Re-solve just the `POWER` leaf to reproduce (fast, ~1 min, no full build):

```bash
# project_dir = a generated/<stem>/ tree with the root + child .kicad_sch and the seed .kicad_pcb
solve-subcircuits <root>.kicad_sch --pcb <seed>.kicad_pcb --only "POWER" --rounds 6 --route
```

Observed:

- Per-round placement is good (`Final placement score: 67–77`), spans ~16 mm.
- Every routed round: `Routed DRC rejected placement: ... illegal_routed_geometry`.
- `kicad-cli pcb drc` on a tight round's `round_NNNN_leaf_routed.kicad_pcb`:
  - **`shorting_items` / `solder_mask_bridge` (the killers):**
    `Track [VBUS] on F.Cu ↔ Pad 4 [<no net>] of U1` and `↔ Pad 5 [+3V3] of U1`.
    → FreeRouting routed a VBUS track **across U1's pads**. A router must never
    cross a pad of another (or no) net — so **U1's pads are not obstacles** in the
    routing for this leaf.
  - **`clearance` (15, mostly false positives):** the bulk reference a single
    footprint `J1` (USB-C) — its own intrinsic fine-pitch pads (`B8`, `A4B9`,
    `A1B12`, `B4A9`, …) sitting closer than the 2.84 mm placement clearance. A few
    are inter-track (USB_DP_RAW/USB_DN_RAW/VBUS density). These are largely benign.

For comparison, in the full build the per-round `POWER` placements were:
`round0 = 17 mm (tight)`, `round1 = 16 mm (tight)`, `round2 = 104 mm (sprawl)` — and
the **104 mm sprawl was pinned** because rounds 0/1 routed-failed (illegal geometry).

The accepted MCU leaf (`ESP32 S3`, net-dense) clusters to 43×43 mm and routes — so
the engine is fine on dense leaves; the sparse, connector-heavy power leaf is the
hard case.

---

## How the pieces fit (file/line map)

| Concern | Location |
|---|---|
| Build → layout entry (now two-phase) | `kicraft/design/cli_app.py` `_run_layout` (~1653), `_QUALITY_PRESETS` (~1646) |
| Leaf solve, per-leaf seed search | `kicraft/cli/solve_subcircuits.py` `main` (~1414); `--rounds` = attempts/leaf (seeds, no param mutation) |
| **Round score blend** | `solve_subcircuits.py:580-589`: `score = 0.5*placement.total + (50 - 10*unconnected - 25*shorts)`; **routing-failed round → `-inf`** at `562-572` |
| Auto-pin best per leaf | `kicraft/cli/autoexperiment.py` `_auto_pin_best_leaves` (231-320): ranks `(routed, score)`, skips `-inf`. **Only runs under `--leaves-only`** (line 2979-2982) |
| Param mutation (autoexperiment only) | `autoexperiment.py` `_mutate_config` (~1749); applied per parent round, passed to the leaf solve via `--config` |
| **Routed-board validation / illegal-geometry** | `kicraft/autoplacer/freerouting_runner.py` `validate_routed_board` (~1330-1490): `shorts>0 → illegal` at **1369 (no footprint-internal escape)**; clearance footprint-internal handling at **1372-1400**; copper-edge at 1401-1420 |
| Leaf acceptance gates | `kicraft/autoplacer/brain/leaf_acceptance.py`: `_gate_no_unconnected` (`max_unconnected=0`), `_gate_no_illegal_geometry` (255), `allow_footprint_internal_clearance` |
| Placement scorer (compactness OK) | `kicraft/autoplacer/brain/placement_scorer.py`: `_score_compactness` (87, divides by *seed* area → constant), `_score_bbox_packing` (98, rewards tight placed bbox — works), aspect-ratio (44-50) |
| Leaf canvas/envelope | `kicraft/autoplacer/brain/subcircuit_extractor.py` `_derive_local_envelope`: canvas = bbox of the leaf's parts (at seed positions) + margin |
| Seed placement | `kicraft/design/synthesis/kicad_pcb_stub.py:88-108`: scatters ALL parts on a 200×150 grid by index → every leaf starts on a wide canvas |
| DSN export / routing setup | `freerouting_runner.py` — `route_with_freerouting`, the Specctra DSN export, `_inject_netclass_clearances` (~579). **This is where pad keepouts must be enforced.** |

---

## Fix plan (prioritized for the implementing agent)

### 1. (HIGHEST LEVERAGE) Make every pad a routing keepout in the DSN export
The decisive bug: a `VBUS` track was routed **over `U1` pad 4 `[<no net>]`** and
near pad 5 `[+3V3]`. The autorouter is treating same-board pads (especially
**no-net** pads, and pads of other nets) as free copper. Fix the Specctra DSN
export in `freerouting_runner.py` so that **all footprint pads are obstacles** for
nets that don't own them (no-net pads must be keepouts on all layers). A track
crossing another part's pad should be physically impossible.
- This is the real "B", and it fixes shorts on **every** dense leaf, not just POWER.
- Verify: after the fix, a tight `POWER` round routes with **0 shorts**; it then
  gets a real (non-`-inf`) score and is pinned — so **A falls out for free**.
- Watch for: FreeRouting DSN representation of THT/shield pads, no-net pads, and
  multi-net connector pads. Confirm KiCad's DRC `shorting_items` goes to 0.

### 2. Give the router channels on tight placements
Even with pad keepouts, a maximally-packed placement may leave no routable channel
for the cross-net trace (e.g. VBUS J1→U1). Options:
- A placement term / constraint that keeps a clearance channel around dense ICs
  (the LDO, the connector) — i.e. tight, but not *touching*.
- Bump `placement_clearance_mm` for connector/IC-adjacent passives only.
- Re-route after a small "spread just enough" relaxation when a tight round
  route-fails, instead of discarding it outright.

### 3. Footprint-internal escape for the *shorts* gate (secondary cleanup)
`freerouting_runner.py:1369` flags `shorts>0 → illegal` with **no** footprint-internal
escape (unlike `clearance`, which has one at 1372-1400). Genuinely intrinsic
connector geometry can produce nuisance short/mask flags. Add the same
single-footprint escape used for clearance. NOTE: the POWER `U1` short here is a
*real* routing error, so this alone will NOT fix POWER — it's a robustness cleanup
to avoid false rejections on legal fine-pitch parts.

### 4. Don't bias the score toward "tight" (anti-goal)
The scorer's compactness/`bbox_packing` terms already reward tight placements and
work. Do **not** "fix A" by down-weighting routing or up-weighting tightness — that
would pin shorting, non-fab-able boards. The tight rounds are *correctly* rejected
until #1 makes them route.

### 5. Fallback: a curated USB-C power-input leaf in the library
If #1–#2 prove too hard for this connector class, a pre-routed `from_library`
USB-C+LDO power-input leaf is exactly what the leaf library is for — it sidesteps
the autorouter for the hard block. Lower-effort, higher-reliability, but one-off.

---

## What already changed (committed)

- **`feat/build-two-phase-leaf-parent` (merged):** `kicraft build`'s autoexperiment
  qualities now run **leaf phase (`--leaves-only`, auto-pin best per leaf) → parent
  phase (`--parents-only`)** instead of one combined loop. `good = 3×3 leaf / 3
  parent`, `best = 6×3 leaf / 6 parent`. This is the manual `solve-leaves → pin →
  compose` flow the user trusted, and it correctly pins each leaf's best — but on
  its own it does NOT fix the sprawl (the best non-failed round is still the
  sprawl, because the tight rounds route-fail per #1).

Earlier, related (already on `main`): bidirectional IC-domain sheet partitioning
(`architecture.md`), the connector signal-pad escape + interface-net leaf gate
(`breakout_stubs.auto_signal_escape_specs`, `leaf_acceptance` interface exclusion),
and the `/self-eval` harness + admin dashboard. The signal-pad escape helped CC1
route but not CC2 on the sprawled placement.

---

## Definition of done / verification

Re-run `kicraft build <state.json> <out> --no-archive` on the plant monitor.
Success criteria:
1. `POWER` leaf routes with **0 shorts and 0 unconnected** → passes
   `leaf_acceptance` (`max_unconnected=0`) → **accepted**, not rejected.
2. The auto-pin pins the **tight** (~16–25 mm) `POWER` placement, not the sprawl.
3. Parent composes + routes → build **rc=0** (fab-ready).
4. Render the `POWER` leaf (`kicad-cli pcb export svg ... | convert` or
   `kicad-cli pcb render`) and confirm it's a compact, clean power-input block.

Fast inner loop (no full build): `solve-subcircuits <root>.kicad_sch --pcb
<seed>.kicad_pcb --only "POWER" --rounds 6 --route` and check that tight rounds
route with 0 shorts (no `illegal_routed_geometry` rejection).
