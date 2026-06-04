# Plan: web live-view improvements (true-append transcript + state/board windows)

## Context

The kicraft.io web app (`kicraft/server/web.py`) now streams the model's thinking token-by-token (commit `e95617e`). Two follow-ups remain, both web-only (the local `kicraft/gui/` Experiment Manager is intentionally untouched):

1. **The live transcript re-sends the whole growing string every tick.** `render()` calls `transcript.set_text(state["transcript"])` on a 0.2s `ui.timer`, so each tick pushes the entire accumulated transcript over the WebSocket. That is O(n) per tick and O(n squared) over a design, fine for a 30s run, janky and bandwidth-wasteful on a multi-minute BOM. We want a **true append**: only the new bytes cross the wire.
2. **Users can only download a zip.** There is no way to inspect the design in the browser: the committed `state.json` slots, the BOM, or the actual schematic/PCB. We want **viewer windows** (tabs) including a KiCanvas board view.

This is pillar 2 of the web-feedback effort (see memory `kicraft-web-feedback-scope`; broader plan `~/.claude/plans/mighty-toasting-lightning.md`). The streaming infra already exists in `kicraft/server/client.py` (emits `reasoning_delta` / `answer_delta` / `tool` / `tool_result` / `stage_*` progress events) and `drive_chain(..., progress=)`; this plan is UI plumbing on top of that.

## Part 1: true-append transcript

**Current (`web.py` `index()` / `render()`):** a `ui.label` inside a `ui.scroll_area`; `render()` rebuilds `state["transcript"]` from new events and calls `transcript.set_text(full_string)` each tick.

**Target:** keep a `<pre>` feed element and append only the new fragment client-side via `ui.run_javascript`, so per tick we send just the delta, not the whole transcript.

Approach:
- Replace the label with a fixed-id pre element: `ui.html('<pre id="kc-feed" class="whitespace-pre-wrap font-mono text-xs h-96 overflow-auto"></pre>', sanitize=False)` (sanitize=False keeps the id/styling; it is also required later for KiCanvas).
- Keep the worker -> events flow unchanged (the worker thread appends progress events to `state["events"]`).
- In `render()` (runs on the UI loop), process new events into a `pending` string (using the existing `_frag()` formatter), and if non-empty:
  ```python
  ui.run_javascript(
      "const e=document.getElementById('kc-feed');"
      f"e.textContent += {json.dumps(pending)};"
      "e.scrollTop = e.scrollHeight;")
  ```
  Then reset `pending`. Only `pending` (the new bytes) is serialized to the client.
- On a new run (`start()`): `ui.run_javascript("document.getElementById('kc-feed').textContent=''")` and reset `state["rendered"]`.

Notes / edge cases:
- `json.dumps(pending)` handles escaping (newlines, quotes, unicode) safely into the JS string literal.
- One feed per page; the fixed id is fine because each browser connection renders its own page. If we ever want multiple concurrent feeds per page, switch to the NiceGUI element's auto id (`feed.id`).
- Bandwidth drops from O(n squared) to O(n) total; the smooth token feel is preserved because deltas are already small chunks.

Files: `kicraft/server/web.py` only.

## Part 2: state / file / board viewer windows

Add a tabbed panel below the feed (`ui.tabs` + `ui.tab_panels`): **Live | State | BOM | Schematic | PCB | Files**. Everything reads from the per-session design workspace at `state["ws"]` (the `tempfile.mkdtemp(prefix="kicraft_web_")` dir the worker created). The UI handlers run in the same process as the worker, so they share the service's `PrivateTmp` namespace and can read the files directly.

### A. State tab (raw slots)
- Read `state["ws"]/.kicraft/state.json`.
- Show each slot (intent / functional_spec / architecture / bom / wiring) pretty-printed: `ui.code(json.dumps(slot, indent=2))` per `ui.expansion`, or a single `ui.json_editor({"content": {"json": state_dict}})` in read-only mode for a collapsible tree.

### B. BOM tab
- `ui.table` from `state.bom.parts`: columns ref / value / symbol / footprint / sheet, plus the assumptions list. This is the human-readable BOM users will care about.

### C. Files tab
- List `generated/<STEM>/*` (`.kicad_sch`, `.kicad_pcb`, `.kicad_pro`, `_autoplacer.json`) with a per-file `ui.download(path)` (local path works in-process), plus the existing whole-project zip button.

### D. Schematic + PCB tabs (KiCanvas)
KiCanvas renders KiCad files in the browser from a URL, so the app must serve the generated files and inject the web component.

- **Serve the files (authed, path-safe).** Add a route on the NiceGUI FastAPI app:
  ```python
  from nicegui import app
  from starlette.responses import FileResponse, Response
  @app.get("/proj/{token}/{name}")
  def _serve(token: str, name: str):
      ws = _WS_BY_TOKEN.get(token)                 # per-session token -> Path(ws), set when a design starts
      if not ws or not app.storage.user.get("authed"):
          return Response(status_code=404)
      p = (Path(ws) / "generated").resolve()
      f = (p / name).resolve()
      if not str(f).startswith(str(p)) or f.suffix not in {".kicad_sch", ".kicad_pcb", ".kicad_pro"}:
          return Response(status_code=404)        # no traversal, only KiCad files
      return FileResponse(f)
  ```
  Mint a random `token` per design run, store `token -> ws` in a server-side dict and the token in `app.storage.user`, so each user only reaches their own workspace.
- **Self-host KiCanvas** (it is alpha; do not depend on a CDN for a product). Drop `kicanvas.js` under a static dir and `app.add_static_files("/kicanvas", ...)`, then include `<script type="module" src="/kicanvas/kicanvas.js"></script>`.
- **Inject the embed** with sanitize OFF (mandatory, or NiceGUI's DOMPurify strips `<kicanvas-embed>`):
  ```python
  ui.html(f'<kicanvas-embed><kicanvas-source src="/proj/{token}/{stem}.kicad_sch">'
          f'</kicanvas-source></kicanvas-embed>', sanitize=False)
  ```
- Schematic view is meaningful right after `synthesize`. The **PCB view only becomes useful once `kicraft build` (place + route) has run** (synthesize emits an unrouted board); wire the PCB tab to that, which ties this to the other half of pillar 2 (running `kicraft build` from the web app).

Files: `kicraft/server/web.py` (tabs, the serve route, the KiCanvas include), plus a vendored `kicraft/server/static/kicanvas.js`.

## Cross-cutting

- **Temp-workspace cleanup (do alongside this).** Per-design workspaces accumulate in the service `PrivateTmp`. Add a reaper: delete `kicraft_web_*` dirs older than N hours on each new run, or a periodic sweep. Until the viewers exist we could delete the ws after zipping; once viewers read the ws, keep it for the session lifetime then reap.
- **Security.** The serve route must require `authed`, resolve-and-confine paths under the session ws, and whitelist KiCad extensions (no `.env`, no traversal).
- **KiCanvas is alpha.** Pin a known-good build; expect the occasional unsupported feature on complex boards.

## Phasing

1. **Part 1 true-append transcript** (small, high value, removes the bandwidth/jank problem). Ship first.
2. **State + BOM + Files tabs** (pure reads of the ws; no new infra). 
3. **KiCanvas schematic** (serve route + sanitize=False embed). 
4. **KiCanvas PCB** (after the web app can run `kicraft build`).

## Verification

- Transcript: run a multi-minute design; confirm the feed updates smoothly and only the new fragment is sent per tick (watch the browser WS frames; each should be small, not the full transcript). Long sessions stay responsive.
- State/BOM/Files: after a design, the State tab shows the five committed slots, BOM tab lists the parts, Files tab downloads each generated file.
- Schematic: the Schematic tab renders the generated `.kicad_sch` via KiCanvas (zoom/pan works); confirm the serve route 404s for an unauthenticated session and for path traversal attempts.
