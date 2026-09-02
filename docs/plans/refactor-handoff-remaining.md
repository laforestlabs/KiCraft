# KiCraft legibility refactor — remaining-work handoff

**For:** a future session (or you) implementing the rest of the roadmap.
**Date:** 2026-06-22 (updated after the `routes_admin` cut)
**Branch:** `refactor/legibility-phase-1-2`.
**Companion docs:** `docs/plans/refactor-roadmap.md` (the master plan), `AGENTS.md`
(architecture map), `docs/plans/view-from-durable-refactor-v2.md` (the Phase 4a detail).

> **Run the tests with the repo venv:** `.venv/bin/python -m pytest …` (bare `python` is not on
> PATH → exit 127, a *silent* empty baseline). ruff: `.venv/bin/python -m ruff check`.

## Done so far (verified zero regressions vs `main`)

Phase 1 (delete GUI), Phase 2 (`AGENTS.md`), Phase 4c (`events.jsonl` reopen fix), Phase 5
(`build_jobs` leak + README install fix), and **four Phase 3 web.py extractions**:
`storage.py`, `pricing.py`, `render_serving.py`, and **`routes_admin.py` (Phase 3a, commit
`58701d5`)**. **`web.py`: 7,292 → 5,029 lines** (the admin cut alone removed 1,972).

Verification each step: `ruff check` clean + the full suite diffed shows the **same 27
pre-existing failures** (env-only: missing `matplotlib`/`stripe` → `ModuleNotFound`; the
stage-CLI `unrecognized arguments` + `lookup-lcsc` data-assert failures; plus the 2 stale
`test_pro_and_max_limits`/`test_worker_shutdown_requeues`). Don't chase those 27; they pre-date
this work. **The gate is: new FAILED set ⊆ those 27.**

## The proven recipe (repeat this for every extraction)

This is the most important part of the handoff — it's what made the three clean extractions safe.

1. **Pick a cohesive cluster** and find its symbols + line ranges (`grep -nE "^def |^_[A-Z]"`).
2. **Map the dependency closure**: what the cluster calls, and what calls *it* from outside
   (`grep` each symbol, filter to lines outside the block). A symbol used outside the block goes
   in the **re-export set**.
3. **Check for circular-import risk**: does the cluster call back into web-only helpers
   (`_store`, `_current_user`, page closures, other web globals)? If yes, it is NOT a clean move
   — see "coupled extractions" below.
4. **Check the test seams** — the trap that bites:
   - `grep -rn "web\.<symbol>" tests/`. Symbols tests **call** (`web._vendor_cell(...)`) work via
     re-export. Symbols tests **monkeypatch/rebind** (`web._safe_fetch = ...`,
     `monkeypatch.setattr(web, "_X", ...)`) do **not** — after the move the caller resolves the
     name in the *new* module, bypassing the web-namespace patch. Either keep the patched seam in
     web.py, or retarget the test to patch the new module. (This is why `pricing.py` kept the
     `_safe_fetch`/cache machinery in web.py: tests patch `web._safe_fetch`.)
5. **Move** the symbols to the new `kicraft/server/<name>.py`; add its imports.
6. **Re-export** into web.py: `from .<name> import (a, b, c)`. Names used only by tests (not by
   web.py code) trip ruff F401 — tag them `# noqa: F401  re-exported for tests`.
7. **Drop** now-unused web.py imports (ruff F401 will name them).
8. **Verify**: `ruff check` the two files → clean; `python -c "import kicraft.server.web"` →
   ok; run the cluster's tests; then the **full-suite diff** below. Commit one extraction per
   commit with the before/after `web.py` line count in the message.

### Full-suite regression gate (run before every commit)
```
git stash -u && python -m pytest -q -rf 2>&1 | grep '^FAILED' | sed 's/ -.*//' | sort > /tmp/base.txt; git stash pop
python -m pytest -q -rf 2>&1 | grep '^FAILED' | sed 's/ -.*//' | sort > /tmp/mine.txt
comm -23 /tmp/mine.txt /tmp/base.txt   # MUST be empty (no new failures)
```
(Or diff against a one-time `main` baseline; that's what was used here.)

## Remaining Phase 3 — web.py splits (recommended order)

> **Note the ceiling:** `storage`/`pricing`/`render_serving` were *leaf* clusters. The three
> below are **coupled** — they reference many web.py internals and/or are NiceGUI page closures,
> so a naive move creates circular imports. They need an *untangle*, not just a move.

### 3a. `routes_admin.py` — ✅ DONE (commit `58701d5`, 1,972 lines out)
The `@ui.page("/admin/...")` handlers (tuning, self-eval, loadtest, security, users, invites,
core-components) + their chart helpers and `_SELF_EVAL`/`_LOADTEST`/`_SECURITY` process globals.

**What the plan got wrong (and the simpler truth):** it predicted heavy coupling and prescribed a
`common.py` scaffolding module *first*. An AST closure (do this for every cut — see the script idea
below) showed the admin cluster is **self-contained**: its `_echart_*`/`_admin_header`/palette
helpers are used *nowhere else*, and it needs only **7 narrow back-refs** — `_store`,
`_current_user`, `_require_admin`, `_render_scorecard`, `_render_synth_view`, `_schematic_sources`,
`_signup_code`. So it was a **direct move via `from .web import (those 7)`** — no `common.py`, no
test churn. (`is_admin` comes from `.accounts`, not web.) Only one test imported an admin symbol
(`_build_review_outcome`), retargeted to `routes_admin`.

- **The non-obvious trap (cost me the only red gate):** the 6 web test harnesses
  (`test_web_core_components`, `_index_autoopen`, `_pricing`, `_projects_page`, `_layout_editor`,
  `_support_reports`) do `importlib.reload(web)` to get a fresh app + route table. A plain
  `from . import routes_admin` does **not** re-run an already-imported submodule, so NiceGUI drops
  the unre-registered `/admin/*` pages → **404 only in the full suite** (passes in isolation).
  Fix lives in `web.py`'s registration block: set an `_ADMIN_ROUTES_REGISTERED` flag and
  `importlib.reload(routes_admin)` when the flag is already present (i.e. this is a reload).
  `render_serving` is immune — its `@app.get` routes live on the FastAPI app, which reload doesn't
  clear; only `@ui.page` routes do.
- **Recipe refinement that made this fast:** drive the cut from an AST script, not greps. Parse
  `web.py`, partition module-level defs into in-cluster vs out, then (a) collect `Name` loads inside
  the cluster's line range that resolve to out-of-cluster defs = the exact back-ref set, and
  (b) map remaining used names to their import statement = the exact import block for the new file.
  `ruff` then confirms zero undefined (F821) / unused (F401). This turns "did I miss a dep?" from a
  guess into a proof.

### 3b. `build_orchestration.py` — REASSESSED: do this *after* Phase 4b, not before
`_run_design` (in-thread synthesis driver), `_drive_build_queue`/`_execute_claimed_job_local`,
`_rerun_build_worker`, `_design_worker`, the orphan-reaper group (`_orphan_reaper`,
`_finalize_orphan`, `_reconcile_orphan_projects`, `_drain_build_log`), `_fresh_run_state`,
`_load_events`, `_pick_default_project`. **497 lines across 3 non-contiguous regions.**

I analyzed this for extraction and recommend **deferring it to land with Phase 4b**, reversing the
handoff's original 3b-before-4b order. The evidence (AST closure + a rebind-vs-call seam scan):
- **Bidirectional coupling:** the cluster calls 6 web helpers (`_store`, `_erc_offenders`,
  `_zip_generated`, `_project_spend_usd`, `_file_failure_report`, `_quality_badge_from_ws`); the
  rest of web calls **8** cluster symbols, mostly from **deep inside the index/`project_view`
  closures** (`_run_design` threads at L4118/4234/4254, `_fresh_run_state`, `_load_events`,
  `_LIVE_RUNS.get/items`). So 3b and 3c are entangled.
- **Two *stateful* seams block a clean move:** tests **rebind** `web._LIVE_RUNS` (×1 + ×8 mutations)
  and `web._persist_project` (×2). A re-export can't preserve a rebind (the in-module caller keeps
  the old object/func), so either those two stay pinned in `web` and the moved functions reach them
  as `web._LIVE_RUNS` / `web._persist_project` (a 497-line module sprinkled with `web.*` for hot
  state — *diluted* legibility), or you retarget ~11 test refs. The other 6 cluster symbols are
  call-only (`_run_design`/`_fresh_run_state`/`_load_events`/`_pick_default_project`/
  `_reconcile_orphan_projects`) → re-export is fine.
- **Why after 4b:** `_LIVE_RUNS`'s "one home" *is* the Phase 4b decision (one source of truth for
  project state: `projects` table vs `state.json` vs `_LIVE_RUNS` vs `build_jobs`). Relocating the
  build code first pre-empts that decision and builds a `web.*` tangle 4b then reworks — exactly
  the principle-2 inversion ("reduce concepts before relocating code"). Do 4b, then this cluster
  factors out **clean and contiguous**.
- The safety net here is *good* (unlike cli_app): `test_web_default_project`, `test_web_index_autoopen`,
  `test_web_reopen_events`, `test_build_queue` (mostly) pass in baseline — so when you do it, the
  regression gate has teeth.

### 3c. `project_view.py` — hardest, do last (or defer to after Phase 4)
The `open_project` handler + the render loop + panel glue. This is deep NiceGUI closure code over
`state`/`view`/`tabs`/panels. It does not factor out as a move; it needs genuine decomposition
(extract pure helpers first: status derivation, the events/price/inspector readers). **Consider
deferring** until Phase 4a simplifies the storage model (which removes a lot of this code's
`state["ws"]` branching).

### 3d. `design/cli_app.py` (3,913 lines) — analyzed; clean cut exists, but **fix its tests first**
Structure: a 530-line `main()` argparse surface (L3380–3909) + ~22 `_cmd_*` handlers. The cleanest
seam is a **`parts_cli.py`**: the part-library maintenance commands — `_cmd_add_part` (260L),
`_add_part_from_files`, `_cmd_fetch_3d` (175L), `_cmd_lookup_lcsc_id` (114L), `_cmd_promote_part`,
`_cmd_validate_part`, `_cmd_jlcparts_update` (~900 lines total) — which are conceptually distinct
from the build/stage pipeline the web/worker drive. `main()` re-imports the `_cmd_*` it wires via
`set_defaults(func=…)`.
- **Blocker for safe execution *in this env*:** the validating tests are themselves red in the
  baseline-27, and **not** as collection/import errors — they're real runtime mismatches
  (`test_kicraft_lookup_lcsc`: `assert 'parts-library' == 'easyeda'`, an offline-catalog data
  difference; `test_kicraft_stage_cli`: `kicraft: error: unrecognized arguments: …/state.json`, an
  argparse-surface drift). So the FAILED-set diff can't tell a clean move from a functional break
  here — `verify at every step` is not satisfiable. **Land the cli_app cut only where its tests are
  green** (fix/refresh those stage-CLI + lookup-lcsc tests first), then apply the same AST recipe.

## Phase 4 — behavioral (NEEDS app-level verification; do NOT do blind)

These change runtime behavior of the open/view/build/delete paths. Unit tests are necessary but
**not sufficient** — verify by running the web app (`deploy/restart-web.sh` or the smoketest) and
clicking through reopen / continue / rebuild / manual-layout / delete.

- **4a. Collapse the workspace↔durable storage duality.** Full spec:
  `docs/plans/view-from-durable-refactor-v2.md`. With the storage lifecycle now isolated in
  `storage.py`, this is the natural next home for the `_kicraft_dir`/`_read_root` accessors and
  the "build in place / atomic-promote" change. Biggest essential-complexity win.
- **4b. One source of truth for project state.** Collapse `projects` table vs `state.json` vs
  `_LIVE_RUNS` vs `build_jobs` into one owner. Root of the "is it live / reopen drops things"
  bug family. Sequence after 3b (build_orchestration) so `_LIVE_RUNS` already has one home.

## Phase 5 — doc cleanup (low-risk, anytime)
- README still has GUI-coupled *feature* prose (leaf promotion "GUI-only", the Setup tab,
  searchable-params tab). Re-home or remove once those features exist in the web app.
- `docs/` has loose `HANDOFF_*` / one-off plan / spec files to prune or fold.

## Gotchas catalog (learned this round)
- **`importlib.reload(web)` won't re-run a route submodule** → its `@ui.page` routes 404 *only in
  the full suite* (6 web harnesses reload web for a fresh app; passes in isolation because the first
  load registers normally). Fix in web.py: flag-guard + `importlib.reload(routes_admin)` on reload.
  `@app.get` (FastAPI, e.g. `render_serving`) is immune; only `@ui.page` is cleared by reload.
- **`python` vs `.venv/bin/python`** — bare `python` is exit-127 here and the FAILED-grep then
  reports a *silent* 0-failure baseline. Always use the venv interpreter for the gate.
- **Rebind seams (`web.X = …` / `monkeypatch.setattr(web,"X",…)`) ≠ re-exportable.** A re-export
  binds the *object*; a test rebinding `web.X` won't reach the moved module's copy. Such a symbol
  must stay pinned in `web` and be reached as `web.X` at call time (see 3b's `_LIVE_RUNS`).
- **Re-export ≠ patchable.** Moving a monkeypatched function breaks `web.X = ...` patches; keep
  the seam or retarget the test (see recipe step 4).
- **`@app.get`/`@ui.page` register at import** — `from .module import anything` runs the module
  and registers its routes; you don't need to import the handler names (but tests calling a
  handler by name need it re-exported, with `# noqa: F401`).
- **`_PRICE_CACHE`-style shared mutable globals** are fine to re-export *if tests only mutate*
  them (`.pop`/`[k]=`), not rebind (`= {}`). Verify.
- **ruff is the unused-import oracle** after each move — trust its F401 list to find drops.
