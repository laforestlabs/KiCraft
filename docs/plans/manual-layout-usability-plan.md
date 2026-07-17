# Manual leaf layout on kicraft.io — usability plan

Status: **IMPLEMENTED 2026-07-17** — all of M1–M8 landed as 9 commits on
`placement-streamline` (`6747d88`..`282b4f3`), NOT yet pushed/deployed.
Deploy needs BOTH services restarted (M1/M2/M5/M7 touch the pipeline) plus
`pip install -e ".[server,design]"` for the new cairosvg dependency.

Implementation notes vs the original plan:
- A fifth P0 surfaced during M1 verification: the live host has NO
  ImageMagick 7 (`magick`) and IM6's SVG delegate needs the also-missing
  rsvg-convert, so `render_pcb` had NEVER produced a PNG in production —
  editor leaves were invisible rectangles, monitor previews absent. Fixed
  with a cairosvg (+Pillow) fallback chain (`cab756f`); libcairo was
  already on the host. magick, when present, keeps exact output.
- M2's async open needed `background_tasks.create`, not an awaited
  continuation: the editor opens from a ui.timer living inside the slot it
  clears, so awaiting in that task gets cancelled by its own clear — and
  nicegui's run.io_bound SWALLOWS the CancelledError (silent empty body).
- M4 found the DRC violation positions were ALWAYS None: kicad reports are
  block-oriented and the parser only read the header line. Fixed in
  `_run_kicad_cli_drc` (continuation-line scan); also `courtyards_overlap`
  (not `courtyard`) is the real violation type string.
- M7's post-route round trip needed no code: the fab-ready view + entry
  row + M5 status chip already cover it. Sibling metadata staleness
  (interface_ports/external_nets keep the rep's nets after replication) is
  a KNOWN loose end — harmless today (anchors are remapped; compose stamps
  from solved_layout) but worth a follow-up normalization.
- Verification: unit/UI tests per PR (guard tests in
  test_leaf_replication, test_freerouting_runner, test_layout_ratsnest,
  test_manual_layout_flow, test_web_layout_editor) + $0 end-to-end on a
  copy of project 1/630: finalize heals 6 siblings → 9/9 leaves with PNGs
  → manual save → stamp rc0 DRC-clean; forced-overlap stamp yields 26/26
  positioned violations. Full suite: 2756 passed, 8 failures all
  reproduced on the pre-change commit (parts fixtures / env-dependent).

## Goal / definition of "usable"

A Pro/Max user (or a rescue-path user whose auto-compose failed) can go from solved
leaves → hand-placed board → routed → fab-ready **without reading logs**, on any board
the pipeline produced — including repeated-channel boards — with feedback tight enough
to converge in a few save/route iterations.

Measurable targets:
- Manual-route success rate (rc0 fab) per attempt — today unmeasured; add events first.
- Time-to-first-feedback: placement-quality feedback < 1 s (in-canvas), stamp DRC
  markers ~20 s (already the stamp cost), never "read stamp.log".
- Editor works on 100 % of builds that reach the leaf-solved stage (today it breaks on
  every repeated-channel board — see A1).

## Where the feature lives (map)

- `kicraft/server/layout_panel.py` — web panel (open/save/stamp/route buttons).
- `kicraft/layout_editor/` — shared model/geometry/canvas: `leaves.py` (discovery +
  PNG render), `runner.py` (initial state, save, compose subprocess), `canvas.py`
  (HTML/bootstrap), `static/layout_canvas.js` (the interaction controller),
  `nicegui_panels.py` (outline/holes/view-options/result panels), `model.py`,
  `outline.py`, `holes.py`, `render.py`.
- Entry + route job: `server/web.py` `_open_layout_editor` / `_start_manual_route` /
  rescue banner (~5201–5590); `design/cli_app.py` `_cmd_manual_route` (route +
  promote/verify/fab tail); `cli/compose_subcircuits.py` manual branch (~1576–1702).

## Audit findings (2026-07-17), ranked

### A. Broken today (P0)

**A1. Identical-leaf reuse (`2985ccb`, deployed 2026-07-16) broke the editor for
repeated-channel boards.** `materialize_sibling` / `finalize_leaf_replication`
(`cli/_leaf_replication.py:165,250`) write `solved_layout.json`, `metadata.json`,
mini `layout.kicad_pcb`, `debug.json` — but never `leaf_routed.kicad_pcb`. Everything
editor-side keys on that file:
- `discover_leaves` skips dirs without it (`layout_editor/leaves.py:187-188`) → only
  representatives appear (evidence: project 1/630 FOUR_CHANNEL_RELAY, build DONE +
  fab zip, 3 of 10 leaf dirs have `leaf_routed.kicad_pcb`).
- `leaf_artifacts_exist` (`server/layout_panel.py:65-72`) under-reports for the
  rescue banner.
- A layout saved from the 3 visible leaves is hard-rejected by compose: the manual
  branch raises `ValueError("manual layout missing placements for instance paths…")`
  (`cli/compose_subcircuits.py:1581-1590`), because `load_solved_artifacts` loads all
  10 from `solved_layout.json`. **Stamp can never succeed on such boards.**

Fix at the source: materialize the sibling's `leaf_routed.kicad_pcb` too —
`renumber_pcb_text(rep_leaf_routed, ref_map)` in `materialize_sibling`, refreshed from
the rep's PINNED board in `finalize_leaf_replication` (nets inside the file keep the
rep's names; the file is consumed only by the canvas renderer + `parse_edge_cuts_aabb`,
same contract as the mini_pcb blocker board — note this in a comment). Also copy the
rep's `renders/leaf_canvas.png` + extent sidecar into siblings so the editor does not
run kicad-cli once per byte-identical sibling.

**A2. First editor open can freeze the whole site.** `LayoutEditorPanel.render()` runs
in a `ui.timer` callback on the event loop (`web.py:5238-5254`) and calls
`discover_leaves` → `render_leaf_canvas` → `subprocess.run(kicad-cli …, timeout=30)`
per leaf (`render/pcb_renderer.py:141-152`). Cold cache on an N-leaf board blocks
every session's websocket for N × seconds. Fix: move discovery+render off the loop
(`run.io_bound` / thread executor) with a skeleton/spinner, and pre-render the canvas
PNGs at the build tail (worker process, where blocking is free) so the web path is a
cache hit.

**A3. Dead DOM contract — no selection/coords/size readout on the web.**
`layout_canvas.js` writes `<canvas_id>-selected`, `-coords`, `-outline` elements
(`static/layout_canvas.js:152-154,264-293`) that only the removed offline GUI
rendered; the web panel header (`layout_panel.py:155-163`) never creates them, so the
selected leaf's name, live x/y/rot, and live board W×H silently don't exist on
kicraft.io. Fix: add the info strip to the panel header with those exact ids.

**A4. Outline W/H inputs are one-way.** `outline_controls` pushes Python→canvas only
(`nicegui_panels.py:53-107`); dragging the edge handles updates the board but the
inputs keep the stale numbers. Fix: canvas → server sync (JS `emitEvent` on outline
change, or reuse the A3 `-outline` label for display and demote the inputs to
"set exact size" actions).

### B. The core capability gap (P1): the user places blind

**B1. No connectivity display (ratsnest).** Nothing shows which leaves talk to which,
so placement quality — the #1 determinant of whether FreeRouting succeeds — is
guesswork. All data already exists per leaf: `solved_layout.json.interface_anchors`
(leaf-local pad pos + port name + layer) and `metadata.json.external_nets` /
`interface_ports` (net names, roles, preferred_side). Plan:
- Python side: at panel build, join anchors across leaves by net → list of
  `{net, [{instance_path, local_x, local_y}]}` links; include parent-local anchor
  positions where known. Ship in the canvas cfg.
- JS side: transform anchors by each leaf's live placement (the same CW math as
  `leafBboxParent`) and draw net lines (bundle count = line weight) on every render;
  update live during drag. Toggle in View options; auto-highlight links of the
  selected leaf.
This single feature converts the editor from "arrange colored rectangles" to
"place a circuit".

**B2. Stamp feedback is 4 count pills + a log tail** (`nicegui_panels.py:324-351`).
The stamp already runs full DRC; render the violations *on the canvas* — parse the
stamp DRC report to `{type, pos, msg}` markers, overlay clickable pins at board
coordinates (verify the report's units at impl time; the ERC report's are ×100).

**B3. Route failure dead-ends.** "Route this layout" closes the editor and runs the
normal build tail; on rc6/rc7 the user lands in the generic failed-build view and the
diagnosis (which nets unconnected, which area shorted) never reaches the canvas.
Plan: persist the manual attempt outcome (`.experiments/manual/last_route_result.json`
written by `_cmd_manual_route`), and when the editor reopens show a "last attempt"
overlay: unconnected net list + the B2 marker layer at the failure positions +
affected leaves flagged.

**B4. Partial-leaf honesty.** When some leaf dirs have metadata but no artifacts
(failed leaf solves), the editor shows the subset with no warning and Save→stamp dies
in the compose `ValueError` with a raw log tail. Plan: `discover_leaves` also returns
the *unavailable* set; the panel banners "N of M blocks available — missing: X, Y"
and disables Save (nothing the user does in the editor can fix a missing leaf).

### C. Editor ergonomics (P2) — make it feel like a tool

- **C1. Zoom + pan + fit.** The viewBox is welded to outline+padding
  (`layout_canvas.js:253-262`); precision placement means squinting. Wheel-zoom
  around cursor, drag-pan (space or middle button), "fit" button.
- **C2. Undo/redo + nudge + numeric entry.** Only a global Reset exists. Add a
  bounded undo stack (placements/outline/holes snapshots on gesture end), Ctrl+Z/Y,
  arrow-key nudge (0.1 mm, Shift = 1 mm), and editable x/y/rot fields in the A3 info
  strip.
- **C3. Leaf identification.** Sheet names only appear once a leaf is selected (and
  today not at all, per A3). Add hover tooltip + optional always-on name labels.
- **C4. Render parent-local components read-only.** The stamp adds parent-local parts
  (mounting-hole H refs it maps/synthesizes — visible — but also any other
  parent-local component and its keep-in, `compose_subcircuits.py:1563-1567,1599-1609`)
  that the canvas never shows; `parent_local` is an opaque passthrough
  (`layout_canvas.js:188-192`). Show them (position + courtyard + keep-in) so the
  stamped board matches what the user saw; dragging them is a later step.
- **C5. Touch/pointer events.** Mouse-only today (`mousedown`/`mousemove`); switch to
  pointer events for tablet users.
- **C6. (Optional) multi-select + align/distribute.**

### D. Flow / productization (P3)

- **D1. Manual-layout status chip** on the place/route tab: saved → stamped ok →
  routing (queue position) → fab-ready / failed, with "Continue editing" — today the
  only trace after leaving the editor is a status one-liner.
- **D2. Post-route round trip:** on success show the routed board with "Edit layout
  again" (placements already persist in `manual_layout.json`; make the path visible).
- **D3. Instrumentation:** emit `editor_opened`, `layout_saved`, `stamp_rc`,
  `route_rc` events (events.jsonl + analytics) so the success-rate target is
  measurable.

Out of scope (unchanged): Pro/Max gating; the compose/route engine itself; parent-local
drag editing (C4 read-only first); mobile-phone layouts.

## PR sequencing

| PR | Contents | Risk / notes |
| --- | --- | --- |
| M1 (**ship first**) | A1: sibling `leaf_routed.kicad_pcb` + copied render cache in `materialize_sibling` + `finalize_leaf_replication`; guard test; $0 replay of project 1/630 → editor sees 10/10 leaves, synthetic manual stamp rc0 | Pipeline change → restart **both** services. Fail-safe under existing `cfg['leaf_replication']` |
| M2 | A2: `run.io_bound` open + build-tail prerender; A3 info strip; A4 outline sync | Web-only restart (prerender part touches worker) |
| M3 | B1 ratsnest overlay (+ View-options toggle) | JS + panel cfg; version sentinel handles stale IIFEs |
| M4 | B2 stamp DRC markers on canvas; B4 partial-leaf banner | Web-only |
| M5 | B3 route-failure round trip (`last_route_result.json` + reopen overlay); D1 status chip; D3 events | Touches `_cmd_manual_route` → both services |
| M6 | C1 zoom/pan/fit; C2 undo/nudge/numeric entry; C3 labels | Pure JS, low risk |
| M7 | C4 parent-local read-only render; D2 post-route round trip | Needs a small helper to expose extracted parent-local comps to the panel |
| M8 (opt) | C5 pointer events; C6 multi-select | |

## Verification strategy

- **$0 replays** on frozen projects: 1/630 (repeated channels — the A1 case) plus one
  non-replicated board. Headless check: `discover_leaves` count == metadata dirs,
  write a synthetic `manual_layout.json`, `run_manual_compose --stamp` → rc0, then the
  `verify`-skill build tail for the route path.
- **JS**: the node cross-language geometry test already drives
  `window.kicraftLayoutGeometry` against `outline.py`; add the ratsnest anchor
  transform (same CW convention as `leafBboxParent`) to that harness. Interaction
  features (zoom/undo) get pure-function extraction + node tests where cheap.
- **Live**: after M2, cold-open an N-leaf project and confirm no event-loop stall
  (other session stays responsive).

## Known gotchas that bite this work

- KiCad rotation is CW; the canvas negates it for SVG — any new geometry must use the
  `leafBboxParent` convention or drift from the stamped board.
- ERC report coords are ×100 (1/100 mm); check DRC report units before B2.
- `pins.ensure_applied` can swap leaf content without changing mtimes — the render
  cache already keys on `pins.json` mtime; keep that for sibling-copied caches.
- Never `pkill -f kicraft.server.web`; deploy = restart web (+ worker for M1/M5).
