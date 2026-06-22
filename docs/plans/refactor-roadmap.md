# KiCraft refactor roadmap — legibility first

**Status:** in progress (Phases 1–2 done; 3–5 planned)
**Date:** 2026-06-22
**Goal:** make the codebase reason-able — for humans *and* coding agents. The driving
pain is concrete: multi-hour agent sessions wasted because the code was too sprawling to
hold in context, so the agent got confused and made mistakes.

**Freedom:** KiCraft is an early prototype with no real users and no projects considered
valuable. **Breaking changes are acceptable** — we can delete and collapse rather than
bridge and preserve.

## Guiding principles

1. **Cheap legibility wins first.** Delete dead surface and write the map before moving code.
2. **Reduce *concepts* before relocating code.** Fewer duplicated/smeared ideas (two UIs, the
   workspace↔durable duality, state spread across four places) beats prettier file splits.
3. **Don't rewrite the place/route core.** `autoplacer/`, `cli/*_subcircuits.py`,
   `cli/autoexperiment.py` are the product's actual value, complex for real domain reasons,
   and hardened by dozens of targeted fixes. High risk, low legibility gain. Leave them;
   only centralize their invariants (geometry conventions are already partly centralized).
4. **Verify at every step.** `pytest --co -q` (import-graph sanity) + targeted suites after
   each structural change. Never land a half-converted monolith.

## Baseline (measured 2026-06-22)

- `kicraft/server/web.py` — **7,292 lines, 157 functions, 34 route handlers** (the monolith).
- `kicraft/design/cli_app.py` — 3,913 lines (the build/synthesis CLI orchestrator).
- `kicraft/autoplacer/brain/placement_solver.py` — 4,232 (core algorithm; leave alone).
- `kicraft/cli/` — ~19k lines / 42 files; `kicraft/server/` — 14.5k / 21 files.
- Runtime is **files-as-IPC**: web process → build-worker process → CLI subprocesses, all
  communicating through `state.json` + the workspace tree. (See `CLAUDE.md` for the full map.)

---

## Phase 1 — Delete the Experiment Manager GUI ✅ DONE

`kicraft/gui/` (4,980 lines) was a standalone desktop NiceGUI "Experiment Manager." It is a
clean leaf — **zero production code imports it** — and its only remaining value was as a
reference for manual-layout functionality. That logic actually lives in the *shared*
`kicraft/layout_editor/` (used by the web app and the compose pipeline), so deleting the
shell keeps the engine.

Removed:
- `kicraft/gui/` (entire package).
- GUI-only tests: `test_pipeline_state.py`, `test_presets_builtin.py`, `test_leaf_status_badge.py`,
  `test_gui_per_component_overrides.py`, `test_run_artifact_cleanup.py`.
- `.claude/commands/launch-gui.md` (the slash command).
- The "Live view (Experiment Manager GUI)" section of `.claude/skills/kicraft/SKILL.md` — it was
  *instructing every agent* to launch a now-deleted module (a direct cause of agent confusion).
- The dead `gui` extra in `pyproject.toml`.

Surgically edited (kept the non-GUI half):
- `tests/test_param_ranges.py` — kept the classes testing real `autoplacer.config` /
  `autoexperiment` invariants; dropped the 5 classes testing `kicraft.gui.state`.

Breadcrumbed (recoverable via git history): README launch instructions removed and marked;
`docs/leaf_library_spec.md` banner-marked superseded.

> Git preserves all of it: `git log -- kicraft/gui/`, `git show <sha>:<path>`,
> `git checkout <sha> -- kicraft/gui/` recover it if ever needed.

## Phase 2 — Architecture map ✅ DONE

Created `CLAUDE.md` at the repo root (auto-loaded into every Claude Code session): the
subsystem map, the 3-process runtime topology, the files-as-IPC contract, the storage model
(`.kicraft` vs `kicraft`), and a "**to change X, look in Y**" index. This is the single
highest-leverage anti-confusion artifact — it stops every agent from re-deriving the
architecture from scratch each session.

---

## Phase 3 — Split the orchestration monoliths 🔜 PLANNED (next)

Target the two files where agents drown because the whole file won't fit in context:
`server/web.py` (7,292) and `design/cli_app.py` (3,913). **Not** `placement_solver.py`
(core algorithm — see principle 3).

Proposed seams for `server/web.py` (extract into `kicraft/server/` submodules, web.py keeps
only the NiceGUI page wiring):
- `storage.py` — workspace + durable lifecycle: `_new_workspace`, `_gc_workspaces`,
  `_rehydrate_workspace`, `_persist_project`, `_discover_generated_dir`,
  `_persisted_generated_dir`, the `.kicraft`/`kicraft` accessors. **Do this first** — it's
  self-contained, high-value, and sets up Phase 4's storage collapse.
- `build_orchestration.py` — `_run_design`, the `build_jobs` enqueue/drive, `_LIVE_RUNS`.
- `project_view.py` — the open/view flow (the open handler + the render loop + panels glue).
- `routes_admin.py` — admin / self-eval / loadtest dashboards (a large, separable surface).
- `prices.py`, `render_serving.py` — the price cache and the tokened `/project/<token>/…`
  endpoint.

Method: one extraction per commit, `pytest --co -q` + targeted tests between each. Stop and
report after the `storage.py` extraction for review before proceeding.

(`design/cli_app.py` gets the same treatment once web.py is tractable: split the `build`
pipeline driver from the stage-commit/CLI-arg surface.)

## Phase 4 — Reduce essential complexity 🔜 PLANNED

These remove *concepts*, not just lines. Bigger and riskier than 1–3; do after the map +
splits make them safe. Detailed spec for (a): `docs/plans/view-from-durable-refactor-v2.md`.

- **(a) Collapse the workspace↔durable storage duality.** One project directory, same layout
  live and at rest — no `.kicraft`-vs-`kicraft`, no two-way `copytree`, no per-reopen 17–29 MB
  copy. Build in place (or in an atomically-promoted `.build/` subdir). This *subsumes* the
  view-from-durable plan: with the freedom to break, we collapse the split instead of bridging
  it, which is strictly simpler. The blank-timeline (`events.jsonl`) bug disappears by
  construction.
- **(b) One source of truth for project state.** Today "what state / is it live?" is smeared
  across the `projects` table, `state.json`, the in-process `_LIVE_RUNS` dict, and the
  `build_jobs` queue. Pick one owner. This is the root of the "reopen is missing things / is
  it still running?" bug family.
- **(c) `events.jsonl` timeline fix — cheap, ship anytime.** It's written at persist but never
  read back on reopen, so the timeline/reasoning panel is blank. Loading it into
  `state["events"]` on open is ~10 lines and independently valuable; can land before (a)/(b).

## Phase 5 — Doc & misc cleanup 🔜 PLANNED / optional

- README still documents GUI-coupled workflows (leaf promotion "GUI-only", the Setup tab,
  searchable-params tab). Re-home or remove now that the GUI is gone.
- `docs/` has loose `HANDOFF_*` / one-off plan / spec files — prune or fold.
- Close the `build_jobs` orphan-row leak (no `DELETE FROM build_jobs` exists anywhere) — see
  the v2 plan's Phase 6.

## What we deliberately will NOT touch

The place/route engine (`autoplacer/`, `cli/{solve,compose}_subcircuits.py`,
`cli/autoexperiment.py`, `freerouting_runner.py`). It's the product value, it's complex for
real reasons, and it's load-bearing. Changes there are surgical bug fixes, not refactors.
