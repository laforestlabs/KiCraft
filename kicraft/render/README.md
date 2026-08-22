# `kicraft.render` — PCB rasterization pipeline

One core function (`render_pcb`) renders a `.kicad_pcb` to a PNG clipped
to the board's Edge.Cuts AABB. One preset registry (`VIEWS`) names the
layer set + monitor styling for each named view. One multi-preset
wrapper (`render_views`) renders several views to one directory. One
freshness index (`RenderIndex`) tells every consumer whether a PNG is
fresh in the current run.

If you're looking for "where does this PCB image come from?" — this is
the only place. Everything else (the GUI monitor, the manual layout
canvas, the score-time visual check, the DRC overlay, the parent
compose stamper, the subcircuit diagnostics bundle) flows through
`render_pcb` / `render_views`, so the rendered pixels cannot drift
between consumers.

## Modules

| Module | Role |
|---|---|
| `pcb_renderer.py` | `render_pcb`, `render_views`, `MonitorStyle`, `EdgeCutsExtent` — the kicad-cli + magick core. One call site for `kicad-cli pcb export svg` in the production code. |
| `views.py` | `VIEWS` — the single named-preset registry shared across all consumers. |
| `edge_cuts.py` | `parse_edge_cuts_aabb` — parses the board outline from a `.kicad_pcb` so the renderer can clip the SVG viewBox to the physical board. |
| `index.py` | `RenderIndex` — the four freshness signals (`run_started_at` floor, `run_phase`, `pins.json`, `RENDERER_VERSION`) plus `parent_render` / `leaf_render` / `round_renders` lookups. |
| `inspector_overlay.py` | PIL-drawn diagnostic diagrams (`render_annotated_top`, `render_stacking_heatmap`). Draws from analysis data, NOT from PCB pixels — explicitly NOT part of the kicad-cli pipeline. Output goes to `inspect/`, not `renders/`. |

## PNG basename map

Every PNG that lands in `.experiments/` has one producer and one set of
consumers. Freshness is gated by `RenderIndex` for everything the
monitor surfaces.

| Basename | Producer | Consumer(s) | Freshness gate |
|---|---|---|---|
| `renders/routed_front_all.png` | `render_views` in `subcircuit_render_diagnostics.render_leaf_board_views` after KiCad Routing Tools | monitor leaf thumbnail (`RenderIndex.leaf_render`), manual layout fallback | `run_started_at` floor + pins bypass in `parents_only` |
| `renders/routed_back_all.png` | Same | Same | Same |
| `renders/routed_copper_both.png` | Same | Same | Same |
| `renders/pre_route_front_all.png` | Same, stage='pre_route' before routing | Same fallback chain when routed is missing | Same |
| `renders/pre_route_back_all.png` | Same | Same | Same |
| `renders/pre_route_copper_both.png` | Same | Same | Same |
| `renders/round_NNNN_routed_front_all.png` | `promote_to_round_snapshot` hardlinks canonical → round snapshot at the end of each leaf-routing pass | per-round scrubber (`RenderIndex.round_renders`), pinned-round source for `pin_leaf` | Per-round floor in `RenderIndex.leaf_render(round_index=NNNN)` |
| `renders/round_NNNN_pre_route_*.png` | Same (also produced by the illegal-pre-stamp fast-path for failed rounds) | Same | Same |
| `renders/illegal_pre_stamp_*.png` | `subcircuit_render_diagnostics.render_leaf_board_views(stage='illegal_pre_stamp')` when a leaf is rejected pre-route | leaf_routing copies → `round_NNNN_pre_route_*` so the monitor scrubber shows *something* for rejected rounds | `run_started_at` floor |
| `renders/leaf_canvas.png` | `gui.pages.leaf_canvas_render.render_leaf_canvas` (style=None, F.Cu+F.SilkS+Edge.Cuts) | Manual layout canvas JS drops as `<img>` background | sidecar `RENDERER_VERSION` + `pins.json` mtime + PCB mtime |
| `renders/leaf_canvas.png.extent.json` | Same | Same — gives the canvas the leaf-local mm extent for placement | Same |
| `renders/{pre_route,routed}_drc_overlay.png` | `subcircuit_render_diagnostics.render_leaf_drc_overlay` via `cli.render_drc_overlay.render_overlay` (base from `render_pcb`) | monitor detail panel | `RenderIndex.leaf_render` |
| `renders/pre_vs_routed_contact_sheet.png` | `subcircuit_render_diagnostics.build_leaf_contact_sheet` (magick montage) when enabled | optional debug-only artifact | none |
| `renders/parent_stamped.png` | `compose_subcircuits._render_parent_board_views` via `render_views` after stamping | monitor root node when routing failed (`RenderIndex.parent_render(prefer_routed=False)`) | `run_started_at` floor |
| `renders/parent_routed.png` | Same, after routing | monitor root node when routing succeeded | Same |
| `hierarchical_autoexperiment/round_NNNN/parent_*.png` | `autoexperiment.py` snapshot-copies the per-round canonical parent renders | per-round root-node thumbnail (first probe layer in `RenderIndex.parent_render`) | `run_started_at` floor |
| `frames/frame_NNNN.png`, `frame_latest.png` | `autoexperiment.py` copies the best preview from each round | none (Progression viewer removed; frames still written to disk) | none (always picks latest) |
| `best_preview.png` | Same | Promoted top-level preview | none |
| `inspect/annotated_top.png` | `render.inspector_overlay.render_annotated_top` (PIL, NOT kicad-cli) | `inspect_parent` CLI Markdown summary | none — generated on demand |
| `inspect/stacking_heatmap.png` | `render.inspector_overlay.render_stacking_heatmap` (PIL, NOT kicad-cli) | Same | none |
| `failure_heatmap.png` | `cli.render_failure_heatmap` (matplotlib, NOT kicad-cli) | tools / dashboards | none |

## Freshness signals

All freshness logic lives in `RenderIndex` (`kicraft.render.index`).
The four signals it gates on:

1. **`.experiments/run_started_at`** — float mtime epoch, stamped by the
   runner the instant a new run starts (after purge, before subprocess
   launch). Any PNG with mtime older than this floor belongs to a prior
   run and is hidden.
2. **`.experiments/run_phase`** — `"leaves_only" | "parents_only" |
   "full" | None`. When `"parents_only"`, leaves are not re-solved, so
   pinned-leaf renders bypass the `run_started_at` floor (the prior
   run's leaf renders are still valid). In `"leaves_only"` and
   `"full"`, every leaf is about to be re-rendered, so even pinned
   leaves wait for fresh renders to land.
3. **`.experiments/pins.json`** — `{leaf_key: {round: N, ...}}`. The
   pinned round drives `RenderIndex.leaf_render(leaf_key)` to surface
   that round's snapshot instead of the canonical render. `pin_leaf`
   in `kicraft.autoplacer.brain.pins` overwrites canonical files from
   the pinned-round snapshot atomically and busts the canvas cache
   via PNG + sidecar deletion (it does NOT re-render).
4. **`RENDERER_VERSION`** — int in
   `gui.pages.leaf_canvas_render.RENDERER_VERSION`. Bumping forces
   every cached canvas PNG + sidecar to regenerate on next page load;
   use when the renderer's output shape changes (DPI, layers,
   background, viewBox semantics).

## Atomic-write contract

`render_pcb` writes to a temp file in the destination directory and
finishes with `os.replace(stage, out_png)` so the canonical PNG path
gets a fresh inode on every render. This makes
`promote_to_round_snapshot`'s hardlinks safe: a later render that
overwrites canonical does not reach back through a shared inode and
clobber the bytes seen through an earlier round's snapshot.

`pin_leaf` likewise uses `shutil.copy` (not `copy2`) so the canonical
file's mtime advances to "now," busting any downstream mtime-based
cache that was keyed against the pre-pin state.

## How to find things

- "Where does PNG X come from?" — search this README's table for the
  basename, or `grep -rn "X.png" kicraft/`.
- "Why is the GUI showing the wrong render?" — `RenderIndex.leaf_render`
  / `parent_render` for the leaf or root in question; check the
  `run_started_at` floor and `pins.json` content.
- "How do I add a new preset view?" — add an entry to `views.VIEWS`;
  every consumer picks it up automatically.
- "Where is `kicad-cli pcb export svg` invoked?" — exactly one
  production call site: `render.pcb_renderer._svg_export`.
