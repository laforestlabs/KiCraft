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

### Observability & Analysis

| Script | Usage | Description |
|--------|-------|-------------|
| `render_drc_overlay.py` | `python3 scripts/render_drc_overlay.py <pcb> <round.json> [-o overlay.png]` | Render PCB with DRC violations highlighted (red X for shorts, orange circles for unconnected, yellow for clearance) |
| `render_failure_heatmap.py` | `python3 scripts/render_failure_heatmap.py <experiments_dir> <pcb> [-o heatmap.png]` | Board-space heatmap of routing failure hotspots across all rounds |
| `diff_rounds.py` | `python3 scripts/diff_rounds.py <experiments_dir> <A> <B> [--format text\|json\|html]` | Side-by-side comparison of two experiment rounds (config, scores, nets, DRC) |
| `generate_report.py` | `python3 scripts/generate_report.py <experiments_dir> [-o report.html]` | Self-contained interactive HTML report with score timeline, round browser, net failure analysis, shorts dashboard |
| `plot_results.py` | `python3 scripts/plot_results.py [experiments.jsonl] [output.png]` | Static matplotlib dashboard: score trend, category breakdown, DRC bars, phase timing, config heatmap |

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

#### Two-phase guided experiments and leaf pinning

A normal `autoexperiment` run re-solves every leaf and recomposes the parent on every round, so the user can't lock in a leaf they like and only iterate on the parent. The two-phase mode + pin manifest fixes that:

1. **Explore leaves** — `python3 -m kicraft.cli.autoexperiment <pcb> --schematic <sch> --rounds 30 --leaves-only` runs only the leaf solve phase. Each round's leaf state is snapshotted to `.experiments/subcircuits/<leaf>/round_NNNN_*.kicad_pcb` (plus `metadata.json` and `solved_layout.json`).
2. **Pin chosen leaves** — open the GUI Analysis page → "Pinned leaves" panel at the top of Hierarchical Progression. Each leaf in the gallery has a "Pin from prior experiment-round snapshots" expansion with a thumbnail per snapshotted round and a Pin button. Pins are stored in `.experiments/pins.json`.
3. **Compose against pinned leaves** — `python3 -m kicraft.cli.autoexperiment <pcb> --schematic <sch> --rounds 10 --parents-only` (or use the "Run parent compose with these pins" button in the GUI). The composer calls `pins.ensure_applied()` before loading artifacts, so pinned leaves are guaranteed to be the canonical state on disk.

**Persistent artifacts:**
- Per-round leaf snapshots: `.experiments/subcircuits/<leaf>/round_NNNN_{leaf_routed,metadata,solved_layout,debug}.json` and `.kicad_pcb`
- Per-round parent snapshots: `.experiments/subcircuits/subcircuit__<hash>/round_NNNN_parent_{routed,pre_freerouting}.kicad_pcb`
- Best parent (fab-ready): `.experiments/best/parent_routed.kicad_pcb` and `<projectname>_best.kicad_pcb` at the project root, refreshed every time a round is "kept"
- Pin manifest: `.experiments/pins.json` — schema `pins.v1`. Commit this if you want pins to survive across machines.

`pins.ensure_applied()` is idempotent and a no-op when `pins.json` doesn't exist, so adding `--parents-only` doesn't change behavior of normal runs.

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
