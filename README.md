# KiCraft

KiCad PCB automation toolkit — automated placement, routing, scoring, and
experiment management for KiCad projects via the pcbnew Python API.

## Layers

KiCraft is a multi-layer pipeline. Top-down:

1. **start-new-project** (LLM-driven, opencode plugin) -- turns a natural-language
   project description into a topology-level `project_plan.json`. See
   [`opencode-plugin/`](opencode-plugin/).
2. **formalize-design** -- not yet implemented. Will take a topology plan and
   produce a concrete schematic-level design.
3. **select-parts** -- not yet implemented. Will resolve generic part classes
   to specific MPNs based on price and availability.
4. **placement + routing + scoring** (Python, this repo) -- everything below.

## Installation

```bash
# Development install (editable)
pip install -e .

# With GUI support
pip install -e ".[gui]"

# With all optional dependencies
pip install -e ".[gui,scoring,experiment,dev]"
```

## Quick Start

KiCraft is project-agnostic. Create a `<project>_autoplacer.json` (or
`autoplacer.json`) in your KiCad project root to configure IC groups,
component zones, signal flow, and other project-specific settings.

See `examples/llups_autoplacer.json` for a full example.

### Solve subcircuits

```bash
cd /path/to/your/kicad-project
solve-subcircuits project.kicad_sch --pcb project.kicad_pcb --rounds 3 --route
```

### Run experiment loop

```bash
autoexperiment project.kicad_pcb project.kicad_sch --rounds 20 --workers 2
```

### Two-phase guided experiment (leaf pinning)

Explore leaf candidates first, lock the ones you like, then iterate on the
parent only:

```bash
# 1. Solve only leaves -- snapshots every round to .experiments/subcircuits/<leaf>/round_NNNN_*
autoexperiment project.kicad_pcb --schematic project.kicad_sch --rounds 30 --leaves-only

# 2. Pin chosen rounds via the GUI Analysis page (Hierarchical Progression -> Accepted
#    Leaf Gallery -> "Pin from prior experiment-round snapshots"), or write
#    .experiments/pins.json by hand.

# 3. Run only the parent compose phase against the pinned leaves
autoexperiment project.kicad_pcb --schematic project.kicad_sch --rounds 10 --parents-only
```

The composer calls `pins.ensure_applied()` before loading artifacts, so
pinned leaves stay locked even if a stray leaf-solve overwrites the
canonical files. The best parent of any run is also copied to
`.experiments/best/parent_routed.kicad_pcb` and `<projectname>_best.kicad_pcb`
at the project root for fab handoff.

### Score a layout

```bash
score-layout project.kicad_pcb
```

### Launch GUI

```bash
python -m kicraft.gui
```

### Configuring Placement

Placement constraints live in `<project>_autoplacer.json` at the project
root. The autoplacer auto-discovers it; the file is project-agnostic --
it references components by ref designator (e.g. `J1`) or subcircuit
sheet name (e.g. `BATT`), nothing is hardcoded to a particular project.
See `examples/llups_autoplacer.json` for a full working file.

#### Pinning components to edges, corners, or zones

`component_zones` constrains where individual refs land on the board.
Three constraint types, all applied at solve time:

```json
{
  "component_zones": {
    "J1":  { "edge":   "left" },
    "J2":  { "edge":   "right" },
    "H4":  { "corner": "top-left" },
    "H86": { "corner": "bottom-right" },
    "BT1": { "zone":   "bottom" },
    "BT2": { "zone":   "bottom" }
  }
}
```

* `edge: left|right|top|bottom` — connector body flush with the named
  edge. Pads face inward, housing overhangs outward by
  `connector_edge_inset_mm`.
* `corner: top-left|top-right|bottom-left|bottom-right` — typically
  mounting holes. Locks at the corner with `mounting_holes.keepout.size_mm`
  reserved (see [Mounting holes](#mounting-holes) below).
* `zone: top|bottom|left|right|...` — soft confinement to a region.
  The component is locked inside the zone; the solver may place it
  anywhere within it.

When a ref is part of a child subcircuit (e.g. `BT1` inside the `BATT`
subcircuit), the constraint is propagated to the *block* during parent
composition, so the BATT block ends up zone-bottom-pinned at parent
solve time.

#### Backside THT components: enabling stacking

By default, the parent composer prevents two leaves from overlapping
when both have any same-layer copper -- a strict same-layer-outline
gate that catches CHARGER-style continuous-F.Cu shorts. PTH pads
register as copper on **both** F.Cu and B.Cu, so a leaf whose only
front-side copper is the shadow of through-hole pads (battery
holders, screw terminals, large barrel jacks) is treated as having
real F.Cu occupancy and rejects SMT-on-front stacking inside its
bbox -- even when the body centre is empty and stacking would be
physically fine.

To re-enable opposite-side stacking on top of such a leaf, open the
GUI's **Setup → Per-Component Placement Rules** tab, expand the leaf,
and toggle the sheet-level switch:

> ☐ → ☑ **Backside THT anchor (allow SMT-front leaves to stack on top)**

Click **Save to project JSON…**, review the diff modal, and confirm.
The toggle persists as a sheet-name entry under
`parent_placement.backside_through_hole_leaves`. For headless or CI
workflows the same key can be set by hand:

```json
{
  "parent_placement": {
    "backside_through_hole_leaves": ["BATT", "TERMINAL_BLOCK"]
  }
}
```

What the override does:

* Marks the leaf as having no F.Cu intent for the same-layer-outline
  gate, so SMT-on-front leaves may overlap inside its bbox.
* Tells the post-SA stack pass (`_stack_compatible_blocks`) to use
  it as a stacking anchor, actively row-packing front-only candidates
  inside its body.
* Leaves the per-pad sparse-rect overlap checks in place -- if a
  candidate's pads happen to land on the THT pads themselves
  (corner positions), the predicate still rejects that specific
  geometry.

The flag is keyed by `sheet_name` so any project can list its own
THT-back leaves; nothing is hardcoded to a single board. CHARGER-style
leaves with continuous F.Cu routing **must not** be enabled -- the
override there would let the original same-layer shorts return.

#### Candidate search

`parent_placement.candidate_search` controls the K-candidate search
loop. The picker stamps each candidate, runs DRC, scores by a
composite of opposite-side packing, courtyard overlap, ratsnest
length, and bbox density, and picks the highest-scoring candidate
whose stamped DRC reports `shorts == 0`:

```json
{
  "parent_placement": {
    "candidate_search": {
      "k": 4,
      "time_budget_s": 240.0
    }
  }
}
```

`shorts == 0` is the only hard gate. Geometry-validation results
(components/pads outside the auto-grown outline) are recorded on each
candidate for diagnosis but no longer reject -- letting routing run
on a violating layout produces a routed PNG showing exactly where the
problem is, which is more actionable than aborting the round.

#### Mounting holes

The `mounting_holes` config block parameterizes the corner-anchored
mechanical holes (H4, H86, etc.):

```json
{
  "mounting_holes": {
    "count": 2,
    "screw": "M3",
    "hole_diameter_mm": 3.2,
    "pad":     { "shape": "hexagon", "size_mm": 3.0 },
    "keepout": { "shape": "hexagon", "size_mm": 4.0 }
  }
}
```

| Field | Meaning |
|---|---|
| `count` | Target number of holes (informational; actual count is whatever the source PCB project ships). |
| `screw` | Free-form screw reference (`"M2.5"`, `"M3"`, `"#4-40"`). Documents intent; not parsed. |
| `hole_diameter_mm` | Drill clearance diameter for the screw. M3 default = 3.2 mm (clearance fit + fab tolerance). |
| `pad.shape` | `hexagon` (default), `circle`, or `square`. Reserved for the future footprint generator. |
| `pad.size_mm` | Outer extent of the exposed copper / annular ring around the hole, measured from hole center to the closest edge of the shape (radius for `circle`, half-width-across-flats for `hexagon` / `square`). Reserved for the future footprint generator. |
| `keepout.shape` | Same vocabulary as `pad.shape`. Reserved -- placer currently treats keepout as rectangular. |
| `keepout.size_mm` | Component-free zone radius (face-to-flat distance for `hexagon` / `square`). The placer reads this directly: a corner-anchored hole sits this far inboard from each board edge. Default 4 mm = M3 head 3 mm + 1 mm fab slack; bump to ~5 mm if you ship M3 + washer. |

Only `keepout.size_mm` currently affects geometry. The shape fields and
`pad.*` are plumbing for an upcoming change that lets KiCraft generate
or validate mounting-hole footprints directly from this block.

### Mutation Search Bounds

The GUI Setup tab exposes 39 searchable parameters that the evolutionary
optimizer mutates during `autoexperiment` runs. Users can narrow or widen
the search range (min/max) for each parameter to focus exploration.

Bounds persist across sessions automatically and flow to the optimizer via
`gui_param_ranges.json`. A shared `normalize_bounds()` helper validates all
bound inputs (rejects NaN/Infinity, clamps to spec domain, swaps inverted
ranges). The only enforced cross-parameter constraint is physical:
`via_drill_mm < via_size_mm` (annular ring requirement).

**Parameter groups:**

| Group | Params | Controls |
|-------|--------|----------|
| Placement Physics | 8 | Force strengths, cooling, iterations, convergence |
| Board Geometry | 6 | Dimensions, margins, grid snap, clearances |
| Edge & Connectors | 6 | Courtyard, connector gaps, insets, pad margins |
| SA Refinement | 6 | Temperature, cooling, move radius, swap/rotation |
| Routing | 7 | Trace widths, via dimensions, zone margin, FreeRouting |
| Thermal | 1 | Keep-away radius around hot components |
| Component Behavior | 1 | THT backside area threshold |
| Zone Pour | 4 | Clearance, fill thickness, thermal relief |

Run `autoexperiment` with `--param-ranges <file.json>` to override bounds
from the command line.

## CLI Commands

### Core Pipeline
- `solve-subcircuits` — Hierarchical subcircuit placement and routing
- `compose-subcircuits` — Assemble solved subcircuits into parent boards
- `solve-hierarchy` — Full hierarchical solve (leaves → parents)

### Experiment Management
- `autoexperiment` — Automated experiment loop with parameter search
- `clean-experiments` — Clean up experiment artifacts (before/after/nuke)
- `watch-status` — Live terminal monitor for running experiments

### Scoring & Rendering
- `score-layout` — Score PCB layout quality
- `render-pcb` — Render PCB layers to PNG
- `render-drc-overlay` — DRC violation overlay on PCB render
- `render-failure-heatmap` — Routing failure heatmap

### Analysis
- `plot-results` — Plot experiment or scoring dashboards (auto-detects format)
- `diff-rounds` — Diff between experiment rounds
- `generate-report` — Generate scoring report

### Board Manipulation
- `move-component` — Move a component to absolute position
- `align-components` — Align components along an axis
- `arrange-grid` — Arrange components in a grid
- `add-gnd-zone` — Add GND copper zone
- `cleanup-routing` — Clean up routing artifacts
- `split-schematic` — Split flat schematic into hierarchical sheets

### Inspection
- `list-footprints` — List all footprints with positions
- `check-trace-widths` — Check trace widths against minimum
- `run-drc` — Run design rule check
- `net-report` — Network connectivity report
- `inspect-subcircuits` — Inspect subcircuit hierarchy
- `inspect-solved-subcircuits` — Inspect solved subcircuit artifacts

## Package Structure

```
kicraft/
├── autoplacer/          # Placement and routing engine
│   ├── config.py        # Default config + project config loader
│   ├── freerouting_runner.py
│   ├── brain/           # Pure algorithms (no pcbnew dependency)
│   └── hardware/        # KiCad pcbnew API adapter
├── scoring/             # Layout quality scoring checks
├── gui/                 # NiceGUI experiment manager
├── cli/                 # CLI entry-point scripts
└── logging_config.py    # Structured logging setup
```

## Requirements

- Python 3.10+
- KiCad 9 with pcbnew Python bindings (on system PATH)
- FreeRouting JAR (for automated routing)

## License

MIT
