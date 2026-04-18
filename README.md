# kicad-helper

KiCad PCB automation toolkit — automated placement, routing, scoring, and
experiment management for KiCad projects via the pcbnew Python API.

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

kicad-helper is project-agnostic. Create a `<project>_autoplacer.json` (or
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

### Score a layout

```bash
score-layout project.kicad_pcb
```

### Launch GUI

```bash
python -m kicad_helper.gui
```

## CLI Commands

### Core Pipeline
- `solve-subcircuits` — Hierarchical subcircuit placement and routing
- `compose-subcircuits` — Assemble solved subcircuits into parent boards
- `solve-hierarchy` — Full hierarchical solve (leaves → parents)
- `export-subcircuit-artifacts` — Export subcircuit placement artifacts

### Experiment Management
- `autoexperiment` — Automated experiment loop with parameter search
- `clean-experiments` — Clean up experiment artifacts (before/after/nuke)
- `watch-status` — Live terminal monitor for running experiments

### Scoring & Rendering
- `score-layout` — Score PCB layout quality
- `render-pcb` — Render PCB layers to PNG
- `render-drc-overlay` — DRC violation overlay on PCB render
- `render-failure-heatmap` — Routing failure heatmap

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
kicad_helper/
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
