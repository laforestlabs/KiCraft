# View-from-durable refactor — v2 (revised plan)

> **Now scoped as Phase 4(a) of `docs/plans/refactor-roadmap.md`.** With the freedom to make
> breaking changes, the storage *duality* itself will be collapsed (one project dir, same
> layout live and at rest), which dissolves most of what this doc carefully *bridges*. This
> file remains the detailed reference for the storage/state work.

**Status:** IMPLEMENTED behind `KICRAFT_VIEW_FROM_DURABLE` (default off) — commit `b3b7f44`,
2026-06-23. Done: phase 0 (flag + `_read_root`), phase 1 (events.jsonl, `7b24c0a`), phase 2
(accessors + reader conversion, `5ae1278`), phase 3 (reopen reads durable + view-loop redirect),
phase 4 (`_ensure_workspace` at the write gates), phase 5 (price write-through). Verified flag-off
(byte-identical, full suite) and flag-on (whole web suite green; integration reopen makes zero
workspace dirs). **Remaining:** phase 6 (GC docstring; the `build_jobs` leak is already closed,
`ae6199b`) and phase 7 (flip the default after a manual pass, then delete the dead view-rehydrate
path — where web.py's line count drops). Tests: `tests/test_web_storage_accessors.py`,
`tests/test_web_view_from_durable.py`.
**Date:** 2026-06-22
**Supersedes:** `view-from-durable-refactor.md` (keeps its core idea; corrects motivation,
the reader inventory, the lazy-workspace correctness model, and the delete/GC scope).
**Area:** `kicraft/server/web.py` (open/view, workspace lifecycle), `session.py` (state readers),
`accounts.py` (delete/GC, `build_jobs`).

## What changed from v1 (read this first)

The v1 plan is directionally right and its central mechanism (`_kicraft_dir` accessor +
view-from-durable) is sound. This revision fixes things that are **wrong, stale, or missing**
once checked against the code:

1. **Motivation re-ordered to the truth.** The strongest win is *not* the bug class or the
   delete story — it is that **every reopen runs a blind ~17–29 MB `copytree`** (`generated/`
   incl. `.experiments/`, measured 17 MB of 29 MB on project 1/92). That per-view latency +
   disk churn is the concrete, repeatable cost. Lead with it.
2. **The "complicated delete" motivation is stale.** `delete_project` (`accounts.py:1410-1432`)
   is *already* "3 DB deletes + `rmtree` the durable tree" — it does **not** chase
   `build_jobs.workspace`. v1's payoff narrative double-counts a win that's already banked. The
   real residue is the **`build_jobs` orphan-row leak** (no `DELETE FROM build_jobs` exists
   anywhere). Re-scope phase 5 to that.
3. **The reader inventory was under-inclusive** and would have shipped a half-converted view
   loop. See §"Complete reader inventory."
4. **The lazy-workspace model had a latent data-loss bug.** continue/edit/answer all funnel
   through `_run_design`, which *already* lazily makes an **empty** `kicraft_web_` workspace
   (`web.py:1757-1759`) when `state["ws"]` is falsy. If view-mode sets `state["ws"]=None` and we
   lean on that fallback, a continue/edit on a reopened project **silently loses every
   previously-committed slot and fetched part.** `_ensure_workspace` must use the **rehydrate
   (copytree)** form and run **before** `_run_design`. See §4.
5. **Price-cache-on-view is mis-modeled.** v1 lists "BOM price refresh" as a *mutating action
   needing a workspace.* It isn't user-initiated — the **view render loop** fires it passively
   (`web.py:6990-6991` → background thread → `_save_price_cache`). Minting a workspace to cache
   prices is overkill. Write the cache **through to the durable `kicraft/prices.json`** via the
   accessor instead. See §5.
6. **Two write sites v1's "open handler sets the path keys" model misses:** the render loop
   itself re-derives `project_dir`/`token` (`web.py:7006-7011`), and the heavy layout writes
   live in `LayoutEditorPanel._on_save` (`layout_panel.py:241-243,297`), **not** in
   `_start_manual_route` (which only *reads* `manual_layout.json`). The load-bearing layout gate
   is `_open_layout_editor` (`web.py:6748`), which today doesn't even check `state["ws"]`.
7. **Legacy projects break under the naive accessor.** `_kicraft_dir` as v1 wrote it returns
   `root/"kicraft"` when neither dir exists, so a legacy durable project (only top-level
   `state.json`) reads as empty in view mode — *worse* than "degrade as today," it loses stage
   statuses. State readers need an explicit top-level `state.json` fallback. See §2.

## TL;DR

Make the durable project copy (`projects_dir/<uid>/<pid>/`) the single read-source for the GUI.
Viewing reads it directly — no `copytree`, no scratch workspace. A workspace is materialized
**lazily and synchronously**, only when an action must write (continue, edit+rebuild, manual
layout, placement rules). Price caching writes through to durable (no workspace). Payoffs, in
honest order: (1) kill the 17–29 MB copy on every reopen; (2) fix the blank
timeline/reasoning panel (the one confirmed instance of the "reopen drops things" class) and
remove the whole "did rehydrate copy X?" failure mode; (3) stop minting view-only workspaces
the GC must reap; (4) — already mostly true — keep delete simple, and additionally close the
`build_jobs` row leak.

## The core friction: `.kicraft` (workspace) vs `kicraft` (durable)

A workspace stores run metadata under `ws/.kicraft/` (dotted). Persist writes the durable copy
as `dir_path/kicraft/` (no dot, `web.py:1410-1414`). Every reader hard-coding `<root>/.kicraft/…`
therefore cannot be pointed at `dir_path` unchanged. Resolve with **two small accessors**, not a
layout change (so legacy projects and the clone path `web.py:5052` keep working):

```python
def _kicraft_dir(root: Path) -> Path:
    """Run-metadata dir for a workspace (.kicraft) OR durable project (kicraft, no dot).
    Prefer the existing form; default to the durable name for paths about to be created."""
    for cand in (root / ".kicraft", root / "kicraft"):
        if cand.is_dir():
            return cand
    return root / "kicraft"

def _state_path(root: Path) -> Path:
    """Resolved state.json: .kicraft/ or kicraft/, falling back to a legacy top-level
    state.json (durable projects predating the kicraft/ tree)."""
    p = _kicraft_dir(root) / "state.json"
    return p if p.is_file() else (root / "state.json")
```

> Rejected alternative (same as v1): rewrite persist to emit `.kicraft`. It mutates the on-disk
> contract for every existing project and the clone path; the accessor is back-compatible and
> localizes the blast radius. Keep `kicraft` (no dot) as the durable name.

## `state["ws"]` semantics — Option A, via one read-root accessor

Adopt v1's Option A (so `bool(state["ws"])` cleanly means "a real scratch workspace exists,"
which delete/GC rely on), but implement the read-redirect with a **single** accessor rather than
ad-hoc per-site edits:

```python
def _read_root(state) -> Path | None:
    """Where to READ run metadata + generated artifacts from: the live scratch workspace if
    one exists, else the durable view root. None only for a brand-new run before either."""
    r = state.get("ws") or state.get("view_root")
    return Path(r) if r else None
```

- `state["view_root"]` (new) = durable project root (`p.dir_path`), set on every open in both
  modes. `state["ws"]` = scratch root, set only when a workspace exists.
- `state["project_dir"]` = the **generated** dir (`…/generated/<STEM>/`): durable in view mode,
  scratch after materialization. (Distinct from `view_root`, which is the project root that
  holds `kicraft/` and `events.jsonl`.)
- The audit is now mechanical: in the view loop, replace `Path(state["ws"])` reads with
  `_read_root(state)`, the gate `if state["ws"]:` with `if _read_root(state):`, and every
  `… / ".kicraft" / …` metadata literal with `_kicraft_dir(_read_root(state)) / …` (or
  `_state_path(...)` for state.json + its mtime).

## Complete reader inventory (the part v1 got wrong)

Every `<root>/.kicraft/…` reader that the durable read-path touches. v1 named only
`_synth_check_failures`, `_load_price_cache`, `_save_price_cache`, `read_state`. The bolded ones
are the **misses** — omitting them ships a half-converted view loop.

| Reader | Loc | Reads | Status in v1 |
| --- | --- | --- | --- |
| `_synth_check_failures` | `web.py:580` | `.kicraft/synthesis_check.json` | listed ✓ |
| `_load_price_cache` | `web.py:1067` | `.kicraft/<price file>` | listed ✓ |
| `_save_price_cache` (WRITE) | `web.py:1087` | mkdir + write `.kicraft/<price file>` | listed ✓ |
| `read_state` (session) | `session.py:107` | `.kicraft/state.json` | listed ✓ |
| **`_read_state_json`** | **`web.py:786`** | `.kicraft/state.json` — **the one the view loop actually calls** (`6973,6997,7019,7116`) | **MISS (only "any")** |
| **inline mtime literal** | **`web.py:6970`** | `_mtime(ws/".kicraft"/"state.json")` — a bare literal, not behind any helper | **MISS** |
| **`_erc_offenders`** | **`web.py:531`** | `.kicraft/synthesis_check.json` | **MISS** |
| **`_read_project_stem`** | **`web.py:543`** | `.kicraft/state.json` (called inside `_discover_generated_dir`; its stem fast-path silently dies on durable, falls through to inspection) | **MISS** |
| **`_quality_badge_from_ws`** | **`web.py:4938`** | `.kicraft/synthesis_check.json` (persist-only today; convert for safety) | **MISS** |
| `_derived_statuses` | `web.py:1477` | none directly — reaches `.kicraft` *transitively* via the two above; auto-fixed iff they are | partly noted |

Already-correct durable readers — **do not touch** (they handle the bare name):
`_load_persisted_state` (`web.py:5007`), `accounts.py:1378`, `_persisted_generated_dir`
(`web.py:4949`, delegates to `_discover_generated_dir`).

Out of scope (build-time, always a real `.kicraft` workspace): the relative argv
`".kicraft/state.json"` passed with `cwd=ws` (`web.py:1606-1607`, `build_worker.py:44,49`),
and workspace writers `commit_slot`/`record_answers`/`null_downstream` (`session.py:118-142`),
`stage_driver.py:633-634`, build log `web.py:1655`.

## Phased implementation

### Phase 0 — Feature flag (new)

Gate the behavioral switch (phases 3–4) behind `KICRAFT_VIEW_FROM_DURABLE` (default off until the
manual pass), matching the repo's env-hook convention (cf. `KICRAFT_QUALITY_PRESETS`). Lets us
deploy and roll back the hot reopen path without a code revert. Phases 1–2 ship unconditionally.

### Phase 1 — `events.jsonl` rehydrate (ship FIRST, standalone, no architecture change)

The blank timeline/reasoning panel is the one confirmed "reopen drops things" bug. Confirmed:
`events.jsonl` is written at persist (`web.py:1402-1404`) and **never** read back on the reopen
path — `state["events"]` is seeded `[]` and only ever appended to by worker threads; the only
readers are the self-eval admin dashboards over `run_NN_*` dirs (disjoint). This fix is
**orthogonal** to view-from-durable: load `dir_path/events.jsonl` into `state["events"]` in the
rehydrate branch *today*, deliver value immediately, de-risk the rest.
- In `open_project`'s non-live branch, after building `state`, load `events.jsonl` (best-effort)
  into `state["events"]`.
- Verify the render loop replays events as **display-only** timeline cards (no thread spawn / no
  stage mutation). Question re-rendering is driven by `state["questions"]`, not by events replay
  (`web.py:6960`), so it's independent — but confirm for a `failed`/`awaiting_input` reopen.
- Optional follow-up (user is open to it): trim verbose LLM reasoning text at persist
  (`web.py:1402-1404`), keeping structural/timeline events. Smaller durable trees; defer.

### Phase 2 — Accessors + complete reader conversion (no behavior change)

- Add `_kicraft_dir` and `_state_path` (§"core friction").
- Route **every** reader in the inventory through them (including the four misses + the inline
  `6970` literal). Replace `_read_state_json`/`read_state` bodies to use `_state_path` (giving
  them the legacy top-level fallback for free).
- `grep -n "\.kicraft"` web.py + session.py + accounts.py before landing; reconcile against the
  inventory table; anything new is a gap.
- Tests: each reader returns identical results given a workspace root vs. the durable root for
  the same project, **plus** a legacy root (top-level `state.json`, no `kicraft/`).

### Phase 3 — `_open_for_view` + view-loop redirect (behind the flag)

- New `_open_for_view(p)` for finished/failed/parked projects (the non-live branch):
  - `state["view_root"] = p.dir_path`; `state["ws"] = None`.
  - `state["project_dir"] = str(_persisted_generated_dir(p.dir_path, p.project_stem))`;
    `state["token"] = _register_project_dir(<durable generated dir>)`.
  - statuses via `_derived_statuses` pointed at the durable root.
  - events already loaded (phase 1).
- View-loop redirect (mechanical, per §"read-root accessor"): convert the gates and reads at
  `web.py:6964-7019` (price seed, state-mtime, inspectors, prices re-render, self-heal,
  schematic sources) from `state["ws"]` → `_read_root(state)` / `_state_path(...)`. **Critically**,
  the self-heal at `7006-7011` (which *writes* `project_dir`/`token` during render) must keep
  working off `_read_root(state)` so a failed/mid-state reopen still surfaces the schematic.
- Acceptance: reopen a finished project → gallery, boards, status, synth-check, prices, **and
  timeline** present, with **zero** new dir under `KICRAFT_WORK_DIR`.

### Phase 4 — `_ensure_workspace` at the synchronous mutating gates (behind the flag)

```python
def _ensure_workspace(state, project) -> Path:
    """Materialize a scratch workspace on demand for a WRITE action. Idempotent. MUST run on
    the UI thread, BEFORE _run_design / build enqueue. Uses the rehydrate (copytree) form so
    previously-committed slots + fetched parts are present — NOT _run_design's empty fallback."""
    if state.get("ws"):
        return Path(state["ws"])
    ws = _rehydrate_workspace(project)
    state["ws"] = str(ws)
    pd = _discover_generated_dir(ws)
    if pd is not None:
        state["project_dir"] = str(pd)
        state["token"] = _register_project_dir(pd)
    return ws
```

Two non-negotiable correctness constraints (the v1 model violated both):

1. **Rehydrate-before-`_run_design`.** `_run_design` (`web.py:1757-1759`) makes an *empty*
   `kicraft_web_` workspace when `state["ws"]` is falsy. Call `_ensure_workspace` at the **top of
   each UI-thread handler** so the committed state is restored first; never rely on the empty
   fallback. Otherwise continue/edit on a reopened project silently drops all prior slots.
2. **Synchronous on the UI thread, before enqueue.** The rebuild path enqueues a `build_jobs`
   row with `workspace=str(ws)` (`web.py:1656-1658`) that a **separate worker process** reads
   from the DB. The workspace must exist on disk before enqueue. Never call `_ensure_workspace`
   from inside `_rerun_build_worker` (a daemon thread) or any background context — it mutates
   `state` and does heavy I/O.

Complete set of UI-thread gates to prefix with `_ensure_workspace(state, project)` (v1's list was
incomplete — added entries in **bold**):
- `_continue` (`web.py:6518`) — continue remaining stages.
- `_do_rerun` (`web.py:6480`) — edit a stage + rebuild; note it writes via `commit_slot`/
  `null_downstream` **before** the worker, so the call must precede those.
- **`_submit_answers` (`web.py:~6378`)** — answer a parked question (same `_run_design` write path).
- `_open_layout_editor` (`web.py:6748`) — **the real manual-layout gate**; today it doesn't check
  `state["ws"]` at all and hands `state["project_dir"]` to the panel, whose `_on_save`
  (`layout_panel.py:241-243,297`) writes `manual_layout.json`/preview/`stamp.log`. Without
  `_ensure_workspace` here, those land in the **durable tree**.
- `_start_manual_route` (`web.py:6780`) — only *reads* `manual_layout.json`; still spawns the
  rebuild worker, so it needs a workspace (defensive; the editor-open should have made it).
- `_open_rules_panel` (`web.py:6807`) — placement rules; panel `_apply` writes via `commit_slot`.
- **`_start_replace_build` / `_start_rebuild` (`web.py:~6839/6855`)** — the "Rebuild board" button,
  reachable **without** a stage edit; distinct entry from `_do_rerun`.

After this, a workspace is born only on a brand-new run and on the first *write* action of a
reopened project.

### Phase 5 — Price-cache write-through (replaces v1's "price refresh = mutating workspace")

Do **not** mint a workspace to cache prices. Prices are a cache keyed to the durable project:
- Route `_load_price_cache`/`_save_price_cache` through `_kicraft_dir` so they target the right
  dir name. In view mode `_read_root(state)` is the durable root → the cache reads/writes
  `dir_path/kicraft/<price file>` directly. Small idempotent JSON write; safe for N concurrent
  viewers (last-writer-wins on a cache file); no aliasing risk for a finished project.
- The view-loop call becomes `_ensure_bom_prices(bom_parts, str(_read_root(state)), state)`
  (`web.py:6990-6991`). Its background thread's `_save_price_cache` (`web.py:1119-1120`) then
  persists to durable — the *intended* outcome, not a foot-gun.

### Phase 6 — Delete / GC (re-scoped: delete is already simple)

- `delete_project` (`accounts.py:1410-1432`) is **already** "DROP from 3 tables + `rmtree` the
  durable tree." No change needed for the common case. Drop v1's "complicated delete" framing.
- `_gc_workspaces` (`web.py:151-169`) stays as the backstop for in-flight/abandoned *build*
  workspaces (it reaps everything under `work_dir` by mtime, 2-day, no prefix filter). Update its
  docstring to drop the "reopen creates workspaces" rationale (which the flag now removes).
- **Real residue — close the `build_jobs` orphan-row leak** (confirmed: no `DELETE FROM
  build_jobs` exists). In `delete_project`, additionally `DELETE FROM build_jobs WHERE
  project_id=? AND status IN ('done','failed',<terminal>)` and, for each such row with a
  still-present `workspace`, `rmtree` it. Feasible: `delete_project` already opens `self._conn()`
  and imports `shutil`. **Caveats to honor:** `AccountStore` has no `work_dir` knowledge, so
  `rmtree` the row's stored absolute path with a containment guard; restrict to **terminal**
  status because the `_LIVE_RUNS`-only delete gate (`web.py:2534`) does *not* see worker-side
  `running` builds (separate process), so non-terminal rows could still be active. Ship as its own
  small PR if preferred — it's independent of the view switch.

### Phase 7 — Cleanup

Remove dead view-only rehydrate paths once the flag is default-on; update docstrings
(`_rehydrate_workspace` is now "for edit/resume," not "for view").

## Edge cases & risks

- **Data loss on continue/edit (highest).** Covered by §4 constraint 1 (rehydrate before
  `_run_design`). Add a regression test (below) — this is the easiest thing to get wrong.
- **`.kicraft` vs `kicraft` + legacy.** Covered by `_kicraft_dir` + `_state_path` legacy
  fallback. Test all three roots.
- **Price write into durable on view.** Now intended (§5), via the accessor; not a workspace.
- **Render-loop self-heal writes `project_dir`/`token`** (`web.py:7006-7011`) — a second site that
  sets path keys, kept working via `_read_root`.
- **Render token lifetime — strictly better.** The view token now points at the **durable**
  generated dir (under `projects_dir`), which `_gc_workspaces` never touches; today it points at a
  workspace the 2-day GC can delete under a long-open tab. (Mutate-then-idle still mints a
  `kicraft_resume_` workspace with the old GC hazard — unchanged, pre-existing.)
- **New minor failure mode: delete-while-viewing.** With viewers reading durable directly,
  deleting a project you're viewing in another tab `rmtree`s the read source → next render tick
  hits missing files. Low severity (readers already `try/except → {}`/None). Note it; no fix
  required. (The `_LIVE_RUNS` delete gate is unaffected — viewing isn't a live run.)
- **Read/write aliasing — non-issue in practice.** A viewed project is finished; its only
  in-process writer is the idempotent price cache. A *live* project takes the unchanged
  live-attach branch (reads the workspace, not durable). No concurrent finalize against a viewed
  durable dir.
- **Concurrent viewers.** Read-only durable access is safe for N viewers; no locking.
- **Failed/parked reopen.** Status banners (`web.py:6612-6636`) + `continue_btn` must work in
  view mode; "Continue" then triggers `_ensure_workspace`.
- **Live attach unchanged.** `_LIVE_RUNS` branch (`web.py:6550-6582`) is untouched and still owns
  a real workspace.

## Testing

- Unit: `_kicraft_dir` / `_state_path` resolution (dotted, bare, legacy top-level, neither).
- Unit: each inventory reader — parity across workspace root, durable root, legacy root.
- Unit: `events.jsonl` round-trips into `state["events"]` on open (phase 1, standalone).
- **Regression (the §4 bug): reopen a multi-stage project → Continue/Edit → assert the rebuilt
  workspace contains the previously-committed slots + fetched parts (not an empty `kicraft_web_`).**
- Integration: open a finished project → assert **no** new dir under `KICRAFT_WORK_DIR`; gallery/
  boards/status/timeline populated.
- Integration: open → Continue / Edit-rebuild / Manual-layout / Rebuild-button → assert a
  workspace now exists, rehydrated (has prior slots), action proceeds; manual-layout save writes
  into the **scratch** dir, never durable.
- Integration: view a finished project whose parts aren't process-cached → assert prices populate
  and persist to `dir_path/kicraft/<price file>`, **no** workspace created.
- `tests/test_accounts.py` (near `:186`): delete a finished project → `dir_path` gone, terminal
  `build_jobs` rows + their workspaces gone, live run still refused.

## Manual verification

1. Build to completion; note the workspace under `~/.kicraft/work/`.
2. With the flag on, reopen from the project list. Confirm: timeline + reasoning, render gallery,
   leaf viewers, KiCanvas schematic + PCB — and **no** new `kicraft_resume_*` dir.
3. Edit-a-stage → Rebuild; confirm a fresh workspace appears (rehydrated, prior stages intact)
   and the rebuild runs and produces the same board.
4. Open the layout editor on a reopened project, Save; confirm `manual_layout.json` lands in the
   scratch workspace, **not** the durable tree.
5. Delete the project; confirm `dir_path` and any terminal `build_jobs` workspace are removed.

## Rollout

- Pure server-side; no DB migration (accessors handle old layouts).
- Land phases 1–2 immediately (independently valuable, low risk: the timeline fix + accessor).
- Phases 3–5 behind `KICRAFT_VIEW_FROM_DURABLE`; flip on after the manual pass on a real reopened
  project. Deploy = restart web **and** build worker (pipeline-touching).
- Phase 6 build_jobs cleanup can be its own PR.

## Open questions

- Persist-time reasoning trim (smaller durable trees): do it with phase 1 or defer?
- Ship the `build_jobs` row+workspace cleanup with this refactor or as its own PR? (Independent of
  the view switch; closes a standalone leak.)
- Keep the feature flag permanently as a kill switch, or remove it in phase 7 once proven?
