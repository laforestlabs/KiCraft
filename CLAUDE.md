# KiCraft — architecture map for agents

Read this first. KiCraft turns a natural-language brief into a fabricable KiCad PCB project:
synthesize a schematic, place + route the board, gate fab-readiness, export the fab package.
This file is the map so you don't have to re-derive the structure every session. (Fix-history
and gotchas live in the auto-memory index; deeper plans in `docs/plans/`.)

## The one thing to internalize: it's files-as-IPC across 3 processes

```
  user ── NiceGUI web app ──(LLM, in-thread)──► synthesis stages ──► state.json + generated/
  (kicraft/server/web.py)                                                     │
        │ enqueues a build_jobs row (SQLite)                                  │ project tree
        ▼                                                                     ▼
  build worker (separate process, kicraft/server/build_worker.py)
        │ runs:  python -m kicraft.design.cli_app build .kicraft/state.json generated   (cwd = the project dir)
        ▼
  place/route pipeline (no LLM): autoexperiment → per-leaf solve → compose parent → freeroute
        │                        (cli/autoexperiment, cli/solve_subcircuits,
        ▼                         cli/compose_subcircuits, autoplacer/*)
  promote routed parent → fab gate (0 shorts / 0 unconnected) → export zip
```

- **Synthesis** (intent → functional_spec → architecture → bom → wiring; LLM-driven) runs
  **in-process** in the web app's `_run_design` background thread; each stage commits a slot
  to `state.json`.
- **Place/route** (`build`, no LLM) runs **out-of-process** via the build worker shelling out
  to `cli_app build`. The web app has an in-process fallback when no worker has heartbeated.
- The processes share **no memory** — they communicate through `state.json` (the design state,
  progressively committed) and the **workspace tree** (`generated/<stem>/` with `.kicad_sch`,
  `.kicad_pcb`, and the heavy `.experiments/`). If you change that on-disk shape, you change a
  contract three processes depend on.

## Storage model (one project dir, build-in-place — Phase 4a done)

- **One project dir, build-in-place** (Phase 4a, `9aa0e39`): a project lives AND builds in
  `~/.kicraft/projects/<uid>/<pid>/` — run metadata under `.kicraft/` (the pipeline's native
  dotted name) plus `generated/` (incl. the heavy `.experiments/`), `events.jsonl`, `brief.txt`,
  the zip. The build runs here directly (cwd = this dir, via `_project_dir`); reopen reads it in
  place. **No scratch workspace, no `copytree` either way** — the workspace↔durable duality is gone.
- `KICRAFT_WORK_DIR` (`~/.kicraft/work/`) now holds only throwaway tempdirs for id-less/admin
  (self-eval) runs, reaped by `_gc_workspaces`.
- **Legacy projects** persisted before build-in-place use `kicraft/` (no dot) + a top-level
  `state.json`. Readers and the build resolve either name via `_kicraft_dir`/`_state_path`
  (`storage.py`), `build_worker._meta_dir_name`, and `cli_app._find_state_json`. See
  `docs/plans/refactor-roadmap.md` Phase 4 + `view-from-durable-refactor-v2.md`.

## Subsystem map (packages)

| Package | Responsibility |
| --- | --- |
| `kicraft/server/` | **The product.** NiceGUI web app (`web.py`), SQLite store (`accounts.py`: users, projects, `build_jobs` queue, likes, fts), the standalone `build_worker.py`, panels (`layout_panel.py`, `rules_panel.py`, `stagetabs.py`), `session.py`/`stage_driver.py` (state.json read/commit). |
| `kicraft/design/` | Synthesis pipeline. `cli_app.py` is the CLI the web/worker drive (`build`, stage-commit, etc.); `synthesis/` holds emitter/validation/router for the schematic + seed PCB + ERC. |
| `kicraft/autoplacer/` | Placement + routing engine (the geometry/DRC core). `brain/placement_solver.py`, `brain/subcircuit_composer.py`, `freerouting_runner.py`, `brain/leaf_routing.py`, `brain/gnd_pour.py`, `brain/breakout_stubs.py`. **Treat as load-bearing; surgical fixes only.** |
| `kicraft/cli/` | Command-line orchestration invoked as subprocesses: `autoexperiment.py` (the optimizing search), `solve_subcircuits.py` (per-leaf), `compose_subcircuits.py` (parent), `inspect_parent.py`, `split_schematic.py`, `generate_report.py`. |
| `kicraft/layout_editor/` | **Shared** manual-layout model/geometry/canvas (model, outline, holes, leaves, rules, nicegui_panels). Used by `server/` and by the compose pipeline — not standalone. |
| `kicraft/parts_library/`, `server/parts_catalog.py` | Part resolution + offline JLC pricing catalog. |
| `kicraft/render/`, `kicraft/leaf_library/` | Board/preview rendering; reusable promoted leaf circuits. |
| `kicraft/eval/`, `kicraft/tuning/`, `kicraft/loadtest/`, `kicraft/security/`, `kicraft/scoring/` | Supporting harnesses: self-eval rubric, CMA-ES config tuner, load/stress, security scans, scoring. |

## "To change X, look in Y"

- **Web UI / open a project / view / build handoff** → `server/web.py` (+ panels `layout_panel.py`,
  `rules_panel.py`, `stagetabs.py`). *(Large monolith — Phase 3 splits it.)*
- **DB / users / projects / build queue / delete** → `server/accounts.py`.
- **The separate build worker** → `server/build_worker.py`.
- **Synthesis (intent…wiring, schematic, ERC)** → `design/cli_app.py` + `design/synthesis/`.
- **Placement / routing / DRC / board geometry** → `autoplacer/` + `cli/{solve,compose}_subcircuits.py`
  + `cli/autoexperiment.py`.
- **Manual layout** → `layout_editor/` (logic) + `server/layout_panel.py` (web wrapper).
- **Parts / pricing** → `parts_library/`, `server/parts_catalog.py`.
- **state.json read/commit** → `server/session.py`, `server/stage_driver.py`.

## Conventions & gotchas

- KiCad rotation is **clockwise**; frame/rotation math is centralized in `autoplacer/brain/geometry.py`.
- ERC report coords in `erc.v1.json` are ×100 (1/100 mm).
- Charts in the web app use NiceGUI's `ui.echart` (not plotly — it's not in the `server` extra).
- Deploy = restart **both** `kicraft-web` and the build worker for pipeline changes
  (`deploy/restart-web.sh`, `deploy/restart-build-worker.sh`).
- `kicraft artifacts` (not glob) is the honest way to find a board / check freshness — see
  `docs/ARTIFACTS.md`.

## Active refactor

`docs/plans/refactor-roadmap.md` is the live plan (legibility-first; Phases 1–2 done). The
Experiment Manager desktop GUI (`kicraft/gui/`) was **removed** 2026-06-22 — recover from git
history if ever needed; its manual-layout logic survives in `kicraft/layout_editor/`.
