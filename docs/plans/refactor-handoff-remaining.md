# KiCraft legibility refactor — remaining-work handoff

**For:** a future session (or you) implementing the rest of the roadmap.
**Date:** 2026-06-22
**Branch:** `refactor/legibility-phase-1-2` (10 commits on `main`@`33d0917`).
**Companion docs:** `docs/plans/refactor-roadmap.md` (the master plan), `CLAUDE.md` (architecture
map), `docs/plans/view-from-durable-refactor-v2.md` (the Phase 4a detail).

## Done so far (verified zero regressions vs `main`)

Phase 1 (delete GUI), Phase 2 (`CLAUDE.md`), Phase 4c (`events.jsonl` reopen fix), Phase 5
(`build_jobs` leak + README install fix), and **three Phase 3 web.py extractions**:
`storage.py`, `pricing.py`, `render_serving.py`. **`web.py`: 7,292 → 7,000 lines.**

Verification each step: `ruff check` clean + the full suite diffed against `main` shows the
**same 27 pre-existing failures** (env-only: missing `matplotlib`/`stripe` → `ModuleNotFound`,
plus 2 stale tests — `test_pro_and_max_limits`, `test_worker_shutdown_requeues`). Don't chase
those 27; they pre-date this work.

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

### 3a. `routes_admin.py` — biggest win (~1,500 lines), do first
The `@ui.page("/admin/...")` handlers (tuning, self-eval, loadtest, security, users, invites,
core-components) + their globals (`_SELF_EVAL`, `_LOADTEST`, `_SECURITY`, chart palettes
`_CHART_AXIS`/`_CHART_GRID`/`_TUNE_PALETTE`, `_GRADE_COLORS`, `_QUALITY_CHIP`,
`_REVIEW_NONBLOCK_RE`).
- **Coupling:** they call `_store()`, `_current_user()`, `is_admin()`, nav/layout helpers, and
  `_register_project_dir`/`_resolve_project_token` (already in `render_serving`).
- **Approach:** first extract the **shared page scaffolding** (`_store`, `_current_user`, the
  nav/header/`require_admin` helpers) into a `common.py` that both `web.py` and `routes_admin.py`
  import → breaks the cycle. Then move the admin pages; `web.py` does `from . import routes_admin`
  (side-effecting import registers the `@ui.page` routes, same pattern as `render_serving`).
- **Test seams:** admin tests use the NiceGUI `user_simulation`; they hit routes by URL, so
  registration-via-import covers them. Check `tests/test_web_self_eval.py`,
  `tests/test_web_admin*.py`, `tests/test_web_core_components.py` for any `web._<symbol>` rebinds.

### 3b. `build_orchestration.py`
`_run_design` (the in-thread synthesis driver), `_drive_build_queue`/`enqueue` path,
`_rerun_build_worker`, `_LIVE_RUNS`, and `_persist_project` (+ `_load_events`'s sibling write).
- **Coupling:** mutates the shared `state` dict, spawns threads, writes `_LIVE_RUNS`, and
  `_persist_project` is store/notify-coupled. `_run_design` is called by `_continue`/`_submit_answers`
  closures in the index page.
- **Approach:** move the worker functions that take `state` explicitly; leave the page-closure
  callers in web.py. `_persist_project` can move here (it already only needs `_store()`/`notify`,
  which become `common.py` imports). Watch `_LIVE_RUNS` — it's shared module state; either it
  lives in `common.py` or `build_orchestration` owns it and web.py imports it.

### 3c. `project_view.py` — hardest, do last (or defer to after Phase 4)
The `open_project` handler + the render loop + panel glue. This is deep NiceGUI closure code over
`state`/`view`/`tabs`/panels. It does not factor out as a move; it needs genuine decomposition
(extract pure helpers first: status derivation, the events/price/inspector readers). **Consider
deferring** until Phase 4a simplifies the storage model (which removes a lot of this code's
`state["ws"]` branching).

### 3d. `design/cli_app.py` (3,913 lines) — separate effort
Not yet analyzed. Split the `build` pipeline driver from the stage-commit / argparse surface.
Same recipe, but verify against the CLI tests (`tests/test_kicraft_stage_cli.py` — note several
already fail pre-existing due to env).

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
- **Re-export ≠ patchable.** Moving a monkeypatched function breaks `web.X = ...` patches; keep
  the seam or retarget the test (see recipe step 4).
- **`@app.get`/`@ui.page` register at import** — `from .module import anything` runs the module
  and registers its routes; you don't need to import the handler names (but tests calling a
  handler by name need it re-exported, with `# noqa: F401`).
- **`_PRICE_CACHE`-style shared mutable globals** are fine to re-export *if tests only mutate*
  them (`.pop`/`[k]=`), not rebind (`= {}`). Verify.
- **ruff is the unused-import oracle** after each move — trust its F401 list to find drops.
