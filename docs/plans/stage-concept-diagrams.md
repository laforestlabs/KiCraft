# Plan: Concept diagrams for the functional & architecture stages

## Context

The synthesis pipeline (`intent → functional_spec → architecture → bom → wiring`) commits a
JSON slot per stage to `.kicraft/state.json`, and the web app renders each slot as **tables and
lists** in that stage's inspector window:

- **functional_spec** (`web.py:905-920`) → a "Functional blocks" table (name/category/purpose),
  a "Block connections" table (from/to/signal), and an "Assumptions" list.
- **architecture** (`web.py:922-941`) → a "Sheets" table, "Power nets" list, "Rail voltages" and
  "Topologies" key-value blocks, and a "Misc" line that literally reports
  `("inter-sheet nets", <count>)` — i.e. it tells the user *how many* cross-sheet connections
  exist but shows **none of them**.

Both stages are fundamentally about **structure** — which blocks exist and how signals flow
between them — yet the user has to reconstruct that mental model from rows in two tables. A
reader can't see at a glance that power flows USB → LDO → MCU, or that `+3V3` fans out to three
sheets while `SDA` is a two-sheet bus.

**Goal:** add a **conceptual block/connectivity diagram** to the top of the functional_spec and
architecture inspectors so the user *sees* the circuit's structure, without removing the precise
tables that remain the source of truth.

### Why this is low-risk

The data a diagram needs **already exists** in the committed slots — this is a *derived view*, not
new pipeline output:

- functional_spec is already a directed graph: `blocks[]` are nodes, `connections[]`
  (`from_block`, `to_block`, `signal_type`) are edges. (`models.py:119-146`)
- architecture is already a connectivity graph: `sheets[]` are nodes, `inter_sheet_nets[]`
  (`name`, `endpoints: [{sheet, direction}]`) are nets. (`models.py:170-273`)

So this feature is **web-only**: no change to `models.py`, the LLM prompts, the synthesis stages,
or `state.json`'s shape. It rides the **existing** render path and refresh loop, and the deploy
is a single `kicraft-web` restart (no build-worker restart — this is not a pipeline change).

## Decisions

1. **Render with `ui.echart`'s `graph` series — no new dependency.** ECharts is bundled in the
   `server` extra and is the project's mandated chart primitive (AGENTS.md: "Charts in the web app
   use NiceGUI's `ui.echart` (not plotly)"). Its `series.type: "graph"` draws node-link diagrams
   natively with categories, legends, directed edges, tooltips, and pan/zoom (`roam`). `ui.mermaid`
   is **not** used anywhere and its availability in this NiceGUI build is unverified; custom
   SVG/canvas is more code for a worse result. Decision: **ECharts graph series.**

2. **Diagrams are a derived view built on the fly — nothing is stored.** The diagram is computed
   from the already-committed slot inside `_inspector_spec()`. No new `state.json` field, no model
   change, no LLM cost, no pipeline coupling.

3. **Add one new inspector section type, `"graph"`, to the existing contract.** `_inspector_spec()`
   already returns a list of section dicts (`kv`/`list`/`table`) consumed by
   `StagePanel.set_inspector()` → `_render_section()` (`stagetabs.py:452-527`). A `graph` section
   carries a ready-to-use ECharts option dict. Because the page's render timer **already** re-calls
   `set_inspector()` for every stage whenever `state.json`'s mtime changes
   (`web.py:4919-4934`), the diagram **inherits live refresh for free** — when a slot (re)commits,
   the diagram redraws. No new live-update plumbing.

4. **Build the ECharts option in a pure, testable module** — `kicraft/server/stage_diagram.py`
   — mirroring the established "pure module → option dict → `ui.echart`" pattern in
   `kicraft/loadtest/charts.py` and `kicraft/scoring/`. `_render_section()` just does
   `ui.echart(sec["option"])`. This keeps all diagram logic unit-testable with **zero NiceGUI**.

5. **Deterministic layout (`layout: "none"` + Python-computed coordinates), not force.** The
   codebase values determinism (PYTHONHASHSEED, `replay`); ECharts `force` layout jitters and
   re-animates on every refresh. We compute node coordinates in Python so the same slot always
   draws the same picture and refreshes don't shuffle the graph. (`force` is the easy fallback if a
   layout heuristic proves fiddly, but determinism is the default.)

6. **Collapse repeated nodes** — this is essential, not cosmetic. KiCraft's corpus includes large
   arrays (e.g. WS2812 5×9). Rendering 45 replicated sheet nodes is unreadable. Collapse using the
   data the models already provide:
   - functional_spec: a block with `count > 1` → one node with a `×N` badge.
   - architecture: sheets sharing a `replication_group` → one representative node with `×N`
     (use `replication_instance == 1` as the representative; `models.py:186-187`).

7. **Diagram goes first; tables stay.** The `graph` section is prepended to the inspector so the
   user sees the picture, then the existing tables below remain the exact, copy-pasteable data.
   The diagram only appears once the slot is **committed** (the streaming "writing slot…" draft is
   untouched — `set_inspector` only renders sections for validated data; `stagetabs.py:465-480`),
   which is exactly the desired behavior.

## Diagram designs

### functional_spec — directed block diagram

- **Nodes** = `blocks[]`, colored by `category` via ECharts `categories` + a legend. The five
  categories (`sense`, `process`, `drive`, `power`, `interface`; `models.py:121`) map to five fixed
  colors. Node label = block `name`, with `×N` appended when `count > 1`.
- **Edges** = `connections[]`, **directed** (`from_block → to_block`, arrow symbol on the target),
  styled by `signal_type` (`power`/`ground`/`digital`/`analog`/`clock`/`bus`/`rf`/`other`;
  `models.py:140`) — e.g. power/ground thicker or warm-colored, bus dashed. Edge tooltip shows the
  connection `description`.
- **Tooltip** on a node shows the block `purpose`.
- **Layout** = deterministic left-to-right signal flow: assign each node an `x` column from its
  category (`power` left → `sense`/`process` middle → `drive`/`interface` right) and a stable `y`
  within the column (input order). `layout: "none"`, `roam: true` for pan/zoom.
- If `connections` is empty, draw the nodes alone (still a useful "here are the blocks" view) with
  a small note rather than omitting the diagram.

### architecture — sheet connectivity diagram

- **Nodes** = `sheets[]` (replication-collapsed per decision 6). Node label = sheet `name`
  (`×N` when collapsed); tooltip = sheet `function`.
- **Nets** = `inter_sheet_nets[]`. Because a net can touch **more than two** sheets (e.g. `+3V3`
  → LDO_3V3, MCU, SENSOR in the BMP280 fixture), render with a hybrid that stays electrically
  honest:
  - **2-endpoint net** → a single edge between the two sheet nodes, labeled with the net `name`;
    direction from the endpoints' `direction` (`output → input`; bidirectional = no arrow).
  - **≥3-endpoint net, or any power/ground net** → a small **net-hub node** (distinct shape/color)
    that every endpoint sheet connects to. This mirrors reality (a net is one electrical node that
    many sheets tap) and avoids the visual clutter of a clique of pairwise edges.
  - Power/ground nets (matched against `power_nets` / name heuristics) get a distinct hub color so
    rails read differently from signals.
- **Tooltip** on a net-hub lists the net name and its endpoints + directions.

Both diagrams are the same shape (categorized node-link graph), so a single
`build_graph_option(nodes, edges, categories, *, title)` core does the ECharts assembly and two
thin pure adapters produce the nodes/edges. DRY and individually testable.

## Critical files

| File | What changes |
| --- | --- |
| `kicraft/server/stage_diagram.py` *(new)* | Pure builders: `build_graph_option(...)` core + `functional_spec_diagram(slot) -> option\|None` and `architecture_diagram(slot) -> option\|None` adapters (node/edge construction, category colors, replication/count collapse, net-hub logic, deterministic layout). No NiceGUI import. |
| `kicraft/server/web.py` | In `_inspector_spec()`, prepend a `{"type": "graph", "title": ..., "option": ...}` section to the `functional_spec` branch (`web.py:905-920`) and the `architecture` branch (`web.py:922-941`), calling the new builders. Guarded so an empty/None option simply isn't added. |
| `kicraft/server/stagetabs.py` | Handle `kind == "graph"` in `_render_section()` (`stagetabs.py:495-527`): `ui.echart(sec["option"]).classes(...)` sized to the inspector width. Update the `set_inspector` docstring's section-type list (`stagetabs.py:455-458`). |
| `tests/test_stage_diagram.py` *(new)* | Pure-function tests for both adapters against the committed BMP280 fixture and synthetic edge cases. |

No changes to `models.py`, `cli_app.py`, `session.py`, `stage_driver.py`, the synthesis prompts,
or `state.json`'s schema.

## Implementation

### 1. Pure diagram module — `kicraft/server/stage_diagram.py`

Create the module with three pure functions (no NiceGUI), following
`kicraft/loadtest/charts.py`'s "return an ECharts option dict" convention:

- **`build_graph_option(nodes, edges, categories, *, title) -> dict`** — assembles the ECharts
  option: `series[0].type = "graph"`, `layout = "none"`, `roam = true`, `label.show = true`,
  `edgeSymbol = ["none", "arrow"]` for directed edges, a `legend` from `categories`, and a
  `tooltip`. `nodes` carry precomputed `{name, x, y, category, symbol, value(tooltip)}`; `edges`
  carry `{source, target, lineStyle, label}`.
- **`functional_spec_diagram(slot) -> dict | None`** — from `slot["blocks"]` and
  `slot["connections"]`: build category list/colors, one node per block (`×N` on `count>1`,
  deterministic x-by-category / y-by-order coords), directed edges styled by `signal_type`.
  Returns `None` if there are no blocks.
- **`architecture_diagram(slot) -> dict | None`** — from `slot["sheets"]`,
  `slot["inter_sheet_nets"]`, `slot["power_nets"]`: collapse `replication_group`, build sheet
  nodes, and turn each net into either a direct labeled edge (2 endpoints) or a net-hub node +
  spokes (≥3 endpoints or power/ground). Returns `None` if there are no sheets.

Keep the color maps (category → color, signal_type → line style) as module-level constants so the
tests and the legend share one definition.

### 2. Emit `graph` sections — `kicraft/server/web.py`

In `_inspector_spec()`:

- functional_spec branch (`web.py:905-920`): after `sl` is loaded, build
  `opt = stage_diagram.functional_spec_diagram(sl)` and, if non-None, **prepend**
  `{"type": "graph", "title": "Concept diagram", "option": opt}` to `secs`.
- architecture branch (`web.py:922-941`): same with `stage_diagram.architecture_diagram(sl)`.

Add `from . import stage_diagram` (or a direct import) at the top. The diagram is built from the
exact dict already in hand, so it adds negligible cost to a render tick and is `None`-guarded for
malformed/empty slots.

### 3. Render the section — `kicraft/server/stagetabs.py`

In `_render_section()` add:

```python
elif kind == "graph":
    ui.echart(sec["option"]).classes("w-full").style("height:320px")
```

Place it alongside the existing `kv`/`list`/`table` branches and extend the `set_inspector`
docstring (`stagetabs.py:455-458`) to document the new `{"type": "graph", "option": <echarts>}`
section. Height is fixed (~320px) so the diagram has a stable canvas; `roam: true` lets the user
pan/zoom dense graphs.

### 4. Hidden-tab resize (gotcha)

ECharts (like KiCanvas) sizes its canvas to the container; a chart first laid out while its stage
tab is **hidden** can come up 0px. The codebase already solved this for KiCanvas with
`StageTabs.on_show(key, fn)` (`stagetabs.py:745-747`) calling `.refresh()` on reveal
(`web.py:4236-4242`). If the functional_spec/architecture diagrams render undersized on first
reveal, wire the same pattern: on tab show, trigger an ECharts resize
(`echart.run_chart_method("resize")`). Build the simple version first and only add this if testing
shows a sizing problem — NiceGUI's `ui.echart` usually self-resizes via a ResizeObserver.

## Verification / testing

**Unit tests — `tests/test_stage_diagram.py`** (pure Python, no NiceGUI, no network):

- Load `tests/fixtures/bmp280_reader_state.json` and assert `functional_spec_diagram(state["functional_spec"])`
  produces **5 nodes** (USB_INPUT/LDO_3V3/MCU/SENSOR/USER_IO) with the right categories and the
  expected directed edges (e.g. `USB_INPUT → LDO_3V3` typed `power`).
- Assert `architecture_diagram(state["architecture"])` produces one node per sheet, that the
  multi-endpoint `+3V3` net (3 sheets) becomes a **net-hub** with 3 spokes, and that a 2-endpoint
  net (`SDA`) becomes a **single labeled edge**.
- Edge cases: empty slot → `None`; functional_spec with no `connections` → nodes-only option (no
  edges, no crash); a block with `count = 9` → one node labeled `×9`; sheets with a shared
  `replication_group` → collapsed to one `×N` node; a power-net hub gets the power color.
- Determinism: calling a builder twice on the same slot yields **identical** option dicts
  (coordinates included).

Run: `pytest tests/test_stage_diagram.py -q`.

**End-to-end (mock-LLM web driver, per the established recipe in auto-memory):**

1. Drive a design through to the architecture stage with the mock-LLM web driver. Open the
   **functional_spec** tab → a block diagram renders above the tables, colored by category, with
   directed signal edges.
2. Open the **architecture** tab → sheets render as nodes; the `+3V3`-style rail shows as a hub
   touching multiple sheets; a two-sheet bus shows as one labeled edge.
3. Confirm the diagram appears **after** the slot commits (the live "writing … slot" draft is
   unchanged) and **redraws** if the slot is re-committed (e.g. an upstream edit) — i.e. it follows
   the existing mtime-watch refresh.
4. Switch away from and back to each tab → the diagram stays correctly sized (validates the
   §4 resize concern; add the `on_show` hook only if it comes up undersized).
5. NEVER `pkill -f kicraft.server.web` (it kills the user's live :8080) — kill the driver by its
   own port, per auto-memory.

**Deploy note:** web-only change (no `models.py`/pipeline/LLM touch), so a single
`deploy/restart-web.sh` suffices — the build worker does not need restarting.

## Scope / non-goals

- **No new slot data or model fields.** Diagrams are derived from committed slots only.
- **No LLM/prompt changes** and **no synthesis-stage changes.**
- **Tables are not removed** — the diagram augments them; they remain the precise source of truth.
- **No interactivity beyond pan/zoom + tooltips** in v1 (no click-to-edit, no cross-highlight).

## Future enhancements (out of scope for v1)

- Reflect **wiring-stage** net status on the architecture diagram (e.g. dim/flag nets the wiring
  stage couldn't land), turning it into a live connectivity health view.
- Click a block/sheet node to scroll to / highlight its row in the table below.
- Export the diagram as SVG/PNG alongside the fab package.
- Interactive expand/collapse of replication groups.
