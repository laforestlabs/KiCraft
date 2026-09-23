# Terminal opening datum: the vendored WJ128V family is unreviewed

**2026-09-23.** Found while driving the production Surprise-me flow (board
`KC-CG58R4`, project id 892). Status: **resolved** — see "Resolution" at the
end; the reviewed datum for the whole WJ128V family is in
`kicraft/parts_library/footprint_opening.py`.

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

## Resolution (2026-09-23): fix 1 landed

Fix 1, reviewed from the model rather than guessed. The extra signal the two
footprint heuristics could not supply is the **wire path**: material along Y at
the window heights, measured on the part's vendored mesh registered on the
footprint's own through-hole pins (mesh pin x centres `-7.5/-2.5/2.5/7.5` and
pin y `-0.43..0.37` == the `y≈0` pad row, so the frame is the footprint frame;
the mesh outline `-5.03..+5.27` against the `F.SilkS` outline `-5.02..+5.29`
fixes the otherwise-ambiguous sign):

| measurement (4P, at every pin position) | value |
|---|---|
| outer `+Y` wall | absent (wire window) over z `2.5..6.0` of the 14.1 mm body |
| outer `-Y` wall | present to z `4.75`, no wire window |
| material along Y at z `3..5` | clamp cage back plate only, y `-3.83..-3.03` |
| wire reach from `+Y` | `8.6..9.1 mm` (datasheet strip length `6~7 mm`) |
| wire reach from `-Y` | `1.2..2.0 mm` — stops on the cage plate, never crosses the clamp |
| screwdriver access | hole in the TOP face per pin position (screws vertical) |

So the mouth is **+Y = 90°**, with the marker on the wire-entry wall at
`y = +5.27 mm` (the mesh-measured face; `F.SilkS` draws it at `+5.29`). The
vendor customer drawing (LCSC `C192769`) agrees: its front view shows the
square openings low on the body and the PCB LAYOUT view puts the pin row
`4.90 mm` from one long edge and `5.38 mm` from the other, the openings on the
`5.38` side. The 5P (`C192770`) carries the same window in `+Y` and the same
drawing profile.

Landed in `kicraft/parts_library/footprint_opening.py`
(`_WJ128V_FOOTPRINT_NAMES`: `CONN-TH_4P-P5.00_WJ128V-4P-5.0-14-00A`,
`CONN-TH_5P-P5.00_WJ128V-5P-5.0-14-00A`) with regression tests in
`tests/test_screw_terminal_orientation.py`.

**The command above is not a valid post-fix check by itself**: the leaf
artifacts it composes were solved before the fix, and a component's
`opening_direction` is persisted in the leaf artifact — the compose gate reads
that frozen field, so re-running compose on this workspace still reports
`connector_orientation_unmeasured:J2`. The datum takes effect where the leaves
are re-solved: `cli_app replay` (or **Rebuild board**) with a cold
`.experiments`. Evidence for the fix is therefore the replay of this workspace
plus `detect_opening_direction` on the shipped footprint.

Fix 2 (emit a reviewed family from the design path) is still open. The live run
in the same session shipped a `KC-MQNE7R` whose `J1` is
`CONN-TH_WJ126V-5.0-2P` — the *other* vendored Kangnex family — and it was
correctly certified: both the repo bundle and the pinned tree's copy carry a
`PCB Edge` marker at footprint-local `(-3.8, 0)` (→ 180°) added with the
KC-YJ7Q69 fix `7c82a8d`, and the leaf board carries that same marker, so the
placer and the gate read reviewed data, not a heuristic.

The gap is narrower than "the vendored set": after this session the reviewed
datums are the two WJ128V rows above, `CONN-TH_WJ126V-5.0-2P` (180°) and
`CONN-TH_3P-P5.00_WJ126V-5.0-3P` (90°, marker `(0, 4.13)`) — and these three
footprints stay **unmeasured** (blocked at compose, honestly, rather than
guessed):

- `CONN-TH_4P-P5.00_WJ126V-5.0-4P-1` and `CONN-TH_5P-P5.00_WJ126V-5.0-5P` — no
  marker in-repo and no 3D model at all, so a datum needs the vendor drawing;
- `CONN-TH_5P-P5.00_WJ127-5.0-5P` — no marker anywhere, and its mesh does not
  register cleanly on its own outline.

Two traps found while checking this, both worth knowing before the next review:

1. The **home-tier fetch cache** (`~/.kicraft/parts/screw-terminal-5mm-{2p,3p}`)
   holds older copies of those bundles with no marker (same manifest `version`,
   different content). Harmless — the loader prefers curated over cache
   (`loader.py`: "curated repo content beats auto-fetched caches") — but a
   `detect_opening_direction` result taken directly from that tier is `None`
   and is *not* evidence about what a board runs.
2. Models may be authored rotated against their own footprint: the WJ126V-2P's
   `.kicad_mod` declares `(rotate (xyz 0 0 270))` for its `.wrl`, so any
   mesh-based review must apply the declared transform (my first pass did not,
   which is why its spans looked muddled). Compare the WJ128V-4P, whose
   footprint declares no rotation — that is why the 4P review registered on a
   bare Y-flip.
