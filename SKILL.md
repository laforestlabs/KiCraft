---
name: KiCraft
description: Use when the user asks to "check trace widths", "audit my layout", "list footprints", "rearrange footprints", "arrange LEDs in a grid", "move component", "run DRC", "check clearances", "align components", or discusses KiCad PCB layout automation. Provides Python scripts using the KiCad 9 pcbnew API to parse and modify .kicad_pcb files.
---

# KiCad PCB Helper

Automate KiCad 9 PCB tasks using Python scripts that call the `pcbnew` API.

## Available Scripts

All scripts are in this skill's `scripts/` directory. Run them with `python3`.

### Inspection

| Script | Usage | Description |
|--------|-------|-------------|
| `list_footprints.py` | `python3 scripts/list_footprints.py <pcb>` | List all footprints with reference, value, position, layer |
| `check_trace_widths.py` | `python3 scripts/check_trace_widths.py <pcb> [--min-mm 0.2]` | Find traces narrower than a minimum width |
| `run_drc.py` | `python3 scripts/run_drc.py <pcb>` | Run Design Rule Check and report violations |
| `net_report.py` | `python3 scripts/net_report.py <pcb>` | List all nets with pad counts and connectivity |

### Modification

| Script | Usage | Description |
|--------|-------|-------------|
| `move_component.py` | `python3 scripts/move_component.py <pcb> <ref> <x_mm> <y_mm> [--rotate-deg N]` | Move a footprint to absolute position |
| `arrange_grid.py` | `python3 scripts/arrange_grid.py <pcb> <ref_prefix> --cols N --spacing-mm S [--start-x X --start-y Y]` | Arrange matching footprints in a grid |
| `align_components.py` | `python3 scripts/align_components.py <pcb> <refs...> --axis x|y` | Align footprints along an axis |
| `add_group_labels.py` | `python3 scripts/add_group_labels.py <pcb> --config <config.py> [--in-place] [--dry-run]` | Add/update silkscreen group labels on PCB from ic_groups config. Idempotent. |
| `split_schematic.py` | `python3 scripts/split_schematic.py <sch> --config <config.py> [--dry-run] [--backup]` | Split flat schematic into hierarchical sheets by ic_groups. Creates sub-sheets with hierarchical labels for cross-group nets. |

### Scoring & Visual Analysis

| Script | Usage | Description |
|--------|-------|-------------|
| `score_layout.py` | `python3 scripts/score_layout.py <pcb> [--compare prev.json]` | Score layout quality (traces, DRC, connectivity, placement, vias, routing) |
| `render_pcb.py` | `python3 scripts/render_pcb.py <pcb> [--views front_all back_copper]` | Render PCB layers to PNG for visual review |
| `layout_session.py` | `python3 scripts/layout_session.py summary` | Track layout progress, token usage, and change classification across iterations |

### Observability & Analysis

| Script | Usage | Description |
|--------|-------|-------------|
| `render_drc_overlay.py` | `python3 scripts/render_drc_overlay.py <pcb> <round.json> [-o overlay.png]` | Render PCB with DRC violations highlighted (red X for shorts, orange circles for unconnected, yellow for clearance) |
| `render_failure_heatmap.py` | `python3 scripts/render_failure_heatmap.py <experiments_dir> <pcb> [-o heatmap.png]` | Board-space heatmap of routing failure hotspots across all rounds |
| `diff_rounds.py` | `python3 scripts/diff_rounds.py <experiments_dir> <A> <B> [--format text\|json\|html]` | Side-by-side comparison of two experiment rounds (config, scores, nets, DRC) |
| `generate_report.py` | `python3 scripts/generate_report.py <experiments_dir> [-o report.html]` | Self-contained interactive HTML report with score timeline, round browser, net failure analysis, shorts dashboard |
| `plot_experiments.py` | `python3 scripts/plot_experiments.py [experiments.jsonl] [output.png]` | Static matplotlib dashboard: score trend, category breakdown, DRC bars, phase timing, config heatmap |
| `dashboard_app.py` | `python3 scripts/dashboard_app.py [--port 5000]` | Live Flask dashboard for monitoring running experiments (auto-refreshing status, history, DRC counts) |

The scoring framework automatically renders PCB images alongside JSON results.

#### Visual Review Workflow

After running `score_layout.py`, you MUST complete a visual review:

1. **Find renders**: The JSON result contains `metrics.render_paths` in the `visual` category. The renders are in `results/renders_<timestamp>/`.
2. **Read each PNG**: Use the Read tool to view each rendered PNG (`front_all.png`, `back_copper.png`, `copper_both.png`).
3. **Review checklist** — evaluate and report on each:
   - [ ] **Connector access**: USB, barrel jacks, headers, test points at board edges with correct orientation (not facing inward)
   - [ ] **Component grouping**: Related components (e.g. buck converter + inductor + caps) placed close together
   - [ ] **Trace routing**: No unnecessary detours, avoid 90-degree corners, clean flow
   - [ ] **Ground plane**: B.Cu copper pour intact, no fragmentation from traces cutting through
   - [ ] **Thermal management**: Power ICs have thermal vias, adequate copper area
   - [ ] **Silkscreen**: Readable, not overlapping pads or other text
   - [ ] **Board utilization**: Components spread out efficiently, no wasted space
   - [ ] **Mechanical fit**: Mounting holes accessible, no components blocking board edges
4. **Report findings**: Include specific component references and locations for any issues found.

#### Session Tracking

Each `score_layout.py` run automatically records a board state snapshot in `results/session.json`. This tracks:

- **Change classification**: Each iteration is classified as `NO_CHANGE`, `MINOR_TWEAK`, `MODERATE_REWORK`, or `MAJOR_REDESIGN` based on component movement distances and counts.
- **Token budget**: Cumulative token usage across all iterations, plus tokens-per-score-point efficiency.
- **Stagnation detection**: If score spread is <1 point over 3 consecutive runs, a warning is printed suggesting a major redesign instead of continued tweaking.

Use `python3 scripts/layout_session.py summary` to review the full session history. Use `--no-track` on `score_layout.py` to skip recording.

#### Experiment Observability

The autoexperiment system collects detailed per-round data for post-run analysis:

- **Round detail JSONs**: `.experiments/rounds/round_NNNN.json` — full config, per-net routing results (timing, layer split, failure reasons), phase timings, DRC violations with (x,y) coordinates, placement scores
- **Enriched JSONL**: `experiments.jsonl` includes `placement_ms`, `routing_ms`, `failed_net_names`, `grid_occupancy_pct`
- **GIF frames**: Short-circuit markers (red X) overlaid on PCB snapshots; border color indicates kept (green), shorts (red), or discarded (gray)

**Post-run analysis workflow:**

1. **Generate HTML report**: `python3 scripts/generate_report.py .experiments/` — interactive report with score timeline, round browser (click to expand per-net details), net failure analysis, shorts dashboard
2. **Compare rounds**: `python3 scripts/diff_rounds.py .experiments/ 5 20` — shows config changes, score deltas, nets that changed routing status, DRC diff
3. **DRC overlay**: `python3 scripts/render_drc_overlay.py board.kicad_pcb .experiments/rounds/round_0015.json` — highlights violations on board render
4. **Failure heatmap**: `python3 scripts/render_failure_heatmap.py .experiments/ board.kicad_pcb` — shows where routing consistently fails
5. **Live monitoring**: `python3 scripts/dashboard_app.py` — Flask dashboard with auto-refreshing status, history table (with shorts/DRC columns), stagnation warnings

## Important Rules

1. **Always back up before modifying**: Scripts that modify the PCB save to `<filename>_modified.kicad_pcb` by default. Pass `--in-place` to overwrite.
2. **Units**: The pcbnew API uses nanometers internally. Scripts accept millimeters and convert with `pcbnew.FromMM()` / `pcbnew.ToMM()`.
3. **After modification**: Tell the user to reload the PCB in KiCad (`File > Revert`).
4. **Do NOT modify .kicad_pcb files with text editing** — always use these scripts or the pcbnew API.

## Extending

To add a new script, write Python using `pcbnew.LoadBoard(path)` to load and `board.Save(path)` to save. Key API patterns:

```python
import pcbnew

board = pcbnew.LoadBoard("file.kicad_pcb")

# Iterate footprints
for fp in board.Footprints():
    ref = fp.GetReferenceAsString()
    pos = fp.GetPosition()
    x_mm, y_mm = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)

# Iterate tracks
for track in board.GetTracks():
    width_mm = pcbnew.ToMM(track.GetWidth())
    layer = track.GetLayerName()
    net = track.GetNetname()

# Move a footprint
fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
fp.SetOrientationDegrees(angle)

# Save
board.Save("output.kicad_pcb")
```
