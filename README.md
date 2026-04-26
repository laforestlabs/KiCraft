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
