# KiCraft refactor roadmap — legibility first

**Status:** in progress. **Done & verified (zero test regressions — identical 27 pre-existing
env/stale failures; pass count rises only by newly-added tests):** Phase 1, Phase 2, the Phase 3
`storage.py` + `pricing.py` + `render_serving.py` + **`routes_admin.py` (3a, commit `58701d5`)** cuts
(**web.py 7,292 → 5,031**), Phase 4(c), the **Phase 4(a) step 1 storage accessors (`_kicraft_dir`/
`_state_path` + reader conversion, commit `5ae1278`)**, **Phase 4(a) DONE — the workspace↔durable
duality is collapsed:** read-side view-from-durable reopen (`b3b7f44`, flag collapsed `e8d0265`) +
write-side **build-in-place (`9aa0e39`)** — a project lives AND builds in one dir
(`projects_dir/<uid>/<pid>/`), no scratch workspace, no `copytree` either way; verified with a real
design+build through the web lifecycle. Plus Phase 5 (`build_jobs` leak + README install fix).
**Remaining (full per-item plan in `docs/plans/refactor-handoff-remaining.md`):** Phase 4(a)
tidy-up (stale `_gc_workspaces` docstring, vestigial `view_root`/`_read_root`); `project_view`
split; `build_orchestration`
**reassessed → sequence with Phase 4(b)** (its `_LIVE_RUNS`/`_persist_project` rebind-seams make a
clean move depend on the 4b state-consolidation); `cli_app.py` `parts_cli` cut **once its CLI tests
are green** (they fail on env-data today, so the move can't be verified); Phase 4(a)/(b); Phase 5
doc-prose cleanup.
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

## Phase 3 — Split the orchestration monoliths 🔄 IN PROGRESS

Target the two files where agents drown because the whole file won't fit in context:
`server/web.py` (7,292) and `design/cli_app.py` (3,913). **Not** `placement_solver.py`
(core algorithm — see principle 3).

Proposed seams for `server/web.py` (extract into `kicraft/server/` submodules, web.py keeps
only the NiceGUI page wiring):
- `storage.py` — workspace + durable lifecycle. **✅ DONE (commit `b298d81`):** moved
  `_new_workspace`, `_gc_workspaces`, `_rehydrate_workspace`, `_read_project_stem`,
  `_discover_generated_dir`, `_persisted_generated_dir` (behavior-preserving move +
  re-export; web.py 7,292→7,207). Deferred: `_persist_project` (store/notify-coupled — moves
  with `build_orchestration`) and the `.kicraft`/`kicraft` accessors (new Phase-4 work).
- `routes_admin.py` — **✅ DONE (commit `58701d5`):** the `/admin/*` `@ui.page` surface + chart
  helpers + `_SELF_EVAL`/`_LOADTEST`/`_SECURITY` globals; 1,972 lines out. Self-contained (7 narrow
  `from .web import` back-refs, no `common.py` needed); web.py reloads it on its own reload so the
  routes don't 404 in the test harnesses. web.py 7,000→5,029.
- `build_orchestration.py` — `_run_design`, the `build_jobs` enqueue/drive, `_LIVE_RUNS`.
  **Reassessed → sequence with Phase 4(b)** (see handoff 3b): its `_LIVE_RUNS`/`_persist_project`
  rebind-seams + index-page coupling make a clean move depend on the 4b state-consolidation.
- `project_view.py` — the open/view flow (the open handler + the render loop + panels glue).
- `pricing.py` — **✅ DONE (commit `9d78f28`):** pure BOM-pricing helpers (resolution + selection
  + formatting); the live fetch/cache stayed in web.py (the monkeypatch seam tests rely on).
- `render_serving.py` — **✅ DONE (commit `7481748`):** token-gated raw-file/render/part-preview
  serving; routes register via the import. web.py 7,207→7,000 across both.

Method: one extraction per commit, `pytest --co -q` + targeted tests between each.

**Reassessment after the `storage.py` cut (2026-06-22):** storage.py was the clean exception —
leaf helpers, called-not-monkeypatched, no back-deps into web.py. The remaining seams are NOT
clean mechanical moves and should be done deliberately, not in an unattended batch:
- `prices` / `render_serving` — **✅ DONE** (see above). `pricing.py` avoided the test churn by
  keeping the monkeypatched `_safe_fetch`/cache seam in web.py and moving only the *pure* helpers;
  `render_serving.py` was a clean move (zero test churn). The full per-item plan for what's left is
  in `docs/plans/refactor-handoff-remaining.md`.
- `routes_admin` / `project_view` / `build_orchestration` — **coupled.** These `@ui.page`
  handlers and the `_run_design`/view-loop closures reference many web.py internals, so moving
  them risks circular imports. They need a shared-helpers module (or late imports) *first* — an
  untangle, not a move. `routes_admin` (~1,500 lines) is the biggest legibility prize but the
  hardest to extract safely.
- `design/cli_app.py` (3,913) — large, not yet analyzed; same treatment once web.py is tractable.

## Phase 4 — Reduce essential complexity 🔄 PARTIAL (4c done; 4a/4b are behavioral)

These remove *concepts*, not just lines. (a) and (b) **change runtime behavior** of the web app
(open / view / build / delete paths), so they need real app-level verification and should NOT be
done blind in an unattended push. Detailed spec for (a): `docs/plans/view-from-durable-refactor-v2.md`.

- **(a) Collapse the workspace↔durable storage duality.** One project directory, same layout
  live and at rest — no `.kicraft`-vs-`kicraft`, no two-way `copytree`, no per-reopen 17–29 MB
  copy. Build in place (or in an atomically-promoted `.build/` subdir). This *subsumes* the
  view-from-durable plan: with the freedom to break, we collapse the split instead of bridging
  it, which is strictly simpler. The blank-timeline (`events.jsonl`) bug disappears by
  construction.
  - **Step 1 ✅ DONE (commit `5ae1278`, no behavior change).** `_kicraft_dir`/`_state_path`
    accessors in `storage.py`; every in-scope reader (state/synth-check/price in web.py +
    `session.read_state` + `storage._read_project_stem`) routed through them so a durable or
    legacy root reads identically to a workspace root. Parity tests in
    `tests/test_web_storage_accessors.py`. This is the safe prerequisite for the behavioral switch.
  - **Step 2 ✅ DONE — view-from-durable is the reopen path (`b3b7f44`; flag collapsed `e8d0265`).**
    A reopened project READS its durable tree directly — no per-reopen 17–29 MB copy.
    `_read_root(state)` (= ws | view_root); the reopen branch + view loop + `build_edit_panel` +
    `_collect_support_diagnostics` read via it; `_ensure_workspace` rehydrates a scratch workspace
    lazily at every write gate (the §4 data-loss guard — prior slots survive); price cache writes
    through to durable. Built and verified behind `KICRAFT_VIEW_FROM_DURABLE`, then the flag was
    **collapsed** (one reopen semantics, no env var) once proven. Verified: with it as the DEFAULT
    the whole suite holds at the 27 baseline (zero new); `_read_root`/`_ensure_workspace` units +
    an integration reopen (renders from durable, **zero** workspace dirs); mock instance boots+serves.
    `tests/test_web_view_from_durable.py`. NOTE: this is the **read/reopen half** of (a). web.py
    line count is ~flat — `_rehydrate_workspace` stays for legacy + the lazy write path, so the win
    is conceptual.
  - **Step 3 ✅ DONE — build in place (`9aa0e39`).** A project now lives AND builds in ONE dir
    (`projects_dir/<uid>/<pid>/`): `_project_dir(state)` is created at design start and IS the build
    dir; `_design_worker`/`_ensure_workspace`/`open_project` point `state["ws"]` there;
    `_persist_project` detects in-place (`ws == base`) and **skips the copytree**. No scratch
    workspace under `KICRAFT_WORK_DIR`, no forward/backward copy. Build-in-place keeps the
    pipeline's native `.kicraft/`, so durable readers (`_load_persisted_state`,
    `accounts._load_project_state`) + the build invocation (`build_worker._meta_dir_name`,
    `_execute_claimed_job_local` via `_kicraft_dir().name`, `cli_app._find_state_json`) resolve the
    metadata-dir name, so pre-build-in-place `kicraft/` (no-dot) projects still rebuild. Verified: a
    REAL design+build through the web lifecycle (mock LLM, real place/route) built entirely in the
    durable dir with **zero** scratch workspaces; full suite at 27 baseline (zero new). `_ensure_workspace`
    is now just `_project_dir`; `_rehydrate_workspace` survives only for legacy no-`dir_path`
    projects + id-less/admin runs (which still use a throwaway tempdir). **Remaining tidy-up:**
    `_gc_workspaces` docstring is now stale (it reaps only id-less tempdirs); the vestigial
    `view_root`/`_read_root` (always `ws` now) can be inlined; `build_jobs` leak already closed (`ae6199b`).
- **(b) One source of truth for project state.** Today "what state / is it live?" is smeared
  across the `projects` table, `state.json`, the in-process `_LIVE_RUNS` dict, and the
  `build_jobs` queue. Pick one owner. This is the root of the "reopen is missing things / is
  it still running?" bug family.
- **(c) `events.jsonl` timeline fix — ✅ DONE (commit `7b24c0a`).** `_load_events` is loaded into
  `state["events"]` in the reopen path; the render loop replays the cards (display-only:
  `tabs.push` paints, `_reset_view` zeroes the cursor). +tests (`test_web_reopen_events.py`).

## Phase 5 — Doc & misc cleanup 🔄 PARTIAL

- **✅ DONE (commit `ae6199b`):** closed the `build_jobs` orphan-row leak — `delete_project` now
  drops a project's `build_jobs` rows and reaps the workspaces of terminal jobs (+test).
- **✅ DONE (README commit):** fixed the broken install instructions (removed `[gui]` extra,
  corrected `[kicraft]`→`[design]`).
- Remaining: README still has GUI-coupled *feature* prose (leaf promotion "GUI-only", the Setup
  tab, searchable-params tab) — re-home or remove once those features land in the web app; and
  `docs/` has loose `HANDOFF_*` / one-off files to prune or fold.

## What we deliberately will NOT touch

The place/route engine (`autoplacer/`, `cli/{solve,compose}_subcircuits.py`,
`cli/autoexperiment.py`, `freerouting_runner.py`). It's the product value, it's complex for
real reasons, and it's load-bearing. Changes there are surgical bug fixes, not refactors.
