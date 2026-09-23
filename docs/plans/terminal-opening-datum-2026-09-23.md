# Terminal opening datum: the vendored WJ128V family is unreviewed

**2026-09-23.** Found while driving the production Surprise-me flow (board
`KC-CG58R4`, project id 892). Status: **open — needs a reviewed physical datum,
not a code heuristic.**

## What happens

Every candidate of the parent compose is rejected before routing:

```
[candidate-search] cand=3 rejected before routing: connector_orientation_unmeasured:J2
error: failed to compose subcircuits: every one of 4 candidate(s) was rejected
       before routing for connector orientation: connector_orientation_unmeasured:J2
```

`9.9`/`9.12` are clean, all six leaves route, and the build reports honestly
(`3/5 no routed parent; promoted best partial board for inspection`). The run is
`projects.id=892`, workspace
`~/.kicraft/projects/42/892/generated/USB_AMBIENT_LIGHT_AMP`.

## Why

J2 is the terminal block `screw-terminal-5mm-4p-128:WJ128V-4P-5.0-14-00A`
(KANGNEX WJ128V-5.0-4P, LCSC C192769; footprint
`CONN-TH_4P-P5.00_WJ128V-4P-5.0-14-00A.kicad_mod`). The design was authored by the
**pinned legacy backend** (`pipeline.json` → `legacy`, commit `bc6a2f8`), which
picks this vendored part; the current tree's own lowerer emits the reviewed
Phoenix MKDS family instead.

`kicraft/parts_library/footprint_opening.py` is the single place that answers
"which way does the wire enter". Its doctrine is explicit: reviewed family
metadata keyed on the exact footprint name, and **an unreviewed terminal is
reported unmeasured rather than guessed** (see its docstring, the KC-DZQ76R
story: a 0.61 mm courtyard asymmetry beat the 0.5 mm body-overhang threshold and
aimed screwdriver mouths into the board). `_REVIEWED_OPENINGS` covers only
`TerminalBlock_Phoenix_MKDS-1,5-N_1xNN_P5.00mm_Horizontal` for n=2..12, so
`reviewed_connector_opening("…WJ128V-4P-5.0-14-00A")` returns None and the
footprint carries no `PCB Edge` marker — nothing to measure.

## Measured geometry (from the vendored footprint)

Pad row: THT pads on `y ≈ 0` at `x = -7.50, -2.50, 2.50, 7.50` (5.00 mm pitch,
pad-cluster centre `x = 0`).

| feature | layer | extent |
|---|---|---|
| courtyard | `F.CrtYd` | y `-5.15 … +5.42`, x `±10.44` |
| body outline | `F.SilS` | y `-5.02 … +5.29` |
| **wire-entry glyphs** (square + direction arrow per position) | `F.SilS` | y `+1.99 … +5.01` |
| screw/pad circles | `F.Fab` | y `≈ 0` |
| value text | `F.Fab` | y `+4.015` |
| reference text | `F.SilS` | y `-4.015` |

A 3D model ships with the part (`3d/CONN-TH_WJ5.00-LI-4P-128.step`, `.wrl`), which
is what the reviewed-datum doctrine wants the datum verified against.

**Two signals disagree, which is why this is not a code fix:**

* the entry glyphs (arrow + rectangle at every position) sit on **+Y → mouth = 90°**;
* the courtyard asymmetry says the *smaller* side is `-Y` (`5.15 < 5.42`), and for
  the reviewed MKDS family the **entry face is the smaller side** → mouth = 270°.

For MKDS the true mouth is +Y with front `5.10` vs rear `5.71`, so "smaller
courtyard side is the mouth" is the rule the module explicitly rejects as a
threshold guess. Guessing here is fab-blocking (a mouth aimed into the board is
what 8857b8b was written to stop).

## The two clean fixes

1. **Review the datum.** Check the part's STEP model / drawing for the wire-entry
   face, then add an entry to `_REVIEWED_OPENINGS` keyed on the exact vendored
   footprint name(s) — this repo has several WJ128V siblings
   (`screw-terminal-5mm-4p-128`, `screw-terminal-5mm-4p-wj128v`,
   `screw-terminal-5mm-5p-wj128v` under `~/.kicraft/parts/`), each needing its own
   row. Cite the model evidence in the comment, as the MKDS entries do.
2. **Emit a reviewed family from the design path**, so a legacy-authored BOM's
   terminal choice cannot land outside the reviewed set. Design-side; a
   `bom.substitutions` row would make the swap visible to the user.

Either way, verify with the board that exposed it:

```bash
.venv/bin/python kicraft/cli/compose_subcircuits.py \
  --project ~/.kicraft/projects/42/892/generated/USB_AMBIENT_LIGHT_AMP \
  --parent / --spacing-mm 4.0 \
  --pcb ~/.kicraft/projects/42/892/generated/USB_AMBIENT_LIGHT_AMP/USB_AMBIENT_LIGHT_AMP.kicad_pcb \
  --route --output /tmp/parent_pp.json --seed 4908357 \
  --config ~/.kicraft/projects/42/892/generated/USB_AMBIENT_LIGHT_AMP/.experiments/hierarchical_autoexperiment/round_0002/round_config.json
```

Expect `connector_orientation_unmeasured:J2` to disappear from the
`cand=N rejected before routing` line and the parent to proceed to routing.
