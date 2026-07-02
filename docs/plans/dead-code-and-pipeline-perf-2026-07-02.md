# Dead-code sweep + pipeline simplification/performance plan

**Date:** 2026-07-02. **Method:** 4 subsystem dead-code audits (whole-repo reference greps,
call-graph reasoning, git -S history checks) + vulture cross-check + a code-level pipeline trace
+ measured wall-clock from the 34-run self-eval sweep `20260630T152811Z`. Spot-verified: the
headline zero-caller claims re-grepped by hand before publishing.

**Relationship to existing plans:** complements `refactor-roadmap.md` (legibility) — that roadmap
deliberately fences the place/route core and contains no performance work. Everything here is
either net-new or explicitly tagged as overlapping an existing phase/lever.

---

## Part 1 — Dead/stale code inventory

### Tier 1: delete now (zero references, verified)

| Item | Where | Notes |
| --- | --- | --- |
| `solve_group` + `_resolve_overlaps_bounded` | `autoplacer/brain/placement_solver.py:142-620` | Entire unreachable "functional group" solver (~480 lines). Zero callers, no dynamic dispatch. |
| `FunctionalGroup`, `GroupSet`, `PlacedGroup`, `PlacementIterationSnapshot`, `DRCScore`(+`from_counts`) | `autoplacer/brain/types.py:353-375, 469-503, 726-791` | Data model only reachable from dead `solve_group` / never instantiated. |
| `hierarchical_placement`, `group_source`, `net_priority` config keys | `autoplacer/config.py:265,418,422` (+ writes in `leaf_size_reduction.py:23,144`) | Write-only / never read. `hierarchical_placement` was the selector for dead `solve_group`. |
| `kicraft/render/index.py` (whole module) | `RenderIndex` / `gather_pipeline_state` | Built for the deleted desktop GUI's pipeline graph; zero importers. |
| `_bbox_overlap` | `autoplacer/brain/placement_utils.py:139-144` | Superseded by `_bbox_overlap_amount`/`_bbox_overlap_xy`. |
| `_placement_model_outline` | `autoplacer/brain/subcircuit_composer.py:1568-1578` | Sibling `_artifact_outline` is the live one. |
| `_tag_leaf_outline_silk` | `autoplacer/hardware/adapter.py:35-49` | Effectively no-op body; reader uses a geometry heuristic instead. |
| `rules.py` persist cluster: `_has_override`, `_build_updated_config`, `_diff_dicts`, `_write_config_with_backup` | `layout_editor/rules.py:238-337` | GUI-era "save rules to project config" feature; web app uses only the read path. |
| `synthesize_project` | `design/synthesis/emitter.py:1217-1236` | Never wired; live path imports `emit_schematic` etc. directly. |
| `_select_preview_image`, `_build_visible_cmd` | `cli/autoexperiment.py:1685-1709, 1916-1940` | Vestiges of the GUI "visible run" mode. |
| `get_store()` | `server/host_metrics.py:258-260` | routes_admin instantiates `HostMetricsStore` directly. |
| `get_sample()` | `server/samples.py:177-179` | Never called since the landing-page commit. |
| `AccountStore.first_admin_id()` | `server/accounts.py:1256-1263` | Caller removed in the self-eval decouple (`bd98991`). |
| `_META` dict | `server/stagetabs.py:54` | Derived from live `PHASES`, never read. |
| Unused imports `_bbox_size`/`_shift_bbox`/`_shift_layer_envelopes`/`_shift_rects` | `cli/compose_subcircuits.py:281` | vulture 90%; the `cli/_compose_geometry.py` functions they point to are also unused → module likely deletable with them. |
| `kicraft/gui/` leftover dir | only stale `__pycache__/*.pyc` | Sources already deleted; rm the dir. |
| Root `SKILL.md` | repo root | Orphaned "KiCad PCB Helper" manifest with broken paths; `stage_driver.py:297` literally says to ignore it. |
| `flask`, `structlog` deps | `pyproject.toml:20-21` (`experiment` extra) | Zero imports anywhere. (`matplotlib` in the same extra IS used — keep.) |
| `program.md` package-data + `.gitignore` line | `pyproject.toml:118` | File doesn't exist; its generator (GUI `experiment_runner`) is deleted. |
| `gui` extra in install line | `CONTRIBUTING.md:17` | Errors on install today. |

### Tier 2: likely safe — remove after a 1-minute product decision

- **`cli/split_schematic.py` (35 KB) + `cli/generate_report.py` (34 KB)** — zero Python
  references; only refs are their `[project.scripts]` entries + the orphaned root SKILL.md +
  a *wrong* CLAUDE.md line. Synthesis emits hierarchical sheets directly now.
  **Fix CLAUDE.md's subsystem table when cutting** (it lists both as pipeline members).
- **`electrical-review` subcommand + `_cmd_electrical_review`** — `design/cli_app.py:507-562,
  3912-3923`. Manual/debug surface; the review that actually runs in `build` is
  `_maybe_electrical_review` → `synthesis/electrical_review.review_design` (keep that module).
- **`_ensure_workspace(project=…)` param** — `server/web.py:1721`; docstring admits it's unused;
  only tests pass it. Drop param + update 2 test call sites.
- **`storage._read_root`** — trivial wrapper post-Phase-4a; inline into its 3 callers
  (roadmap already lists this as tidy-up).
- **`layout_editor/runner.find_latest_parent_pcb`** — `__all__`-exported, zero callers.
- **One-off scripts:** `scripts/rebuild_kc_jbhtjb.py`, `scripts/build_parent_local_conn_fixture.py`,
  the 7-file `scripts/bakeoff_*` cluster (its report `docs/electrical_review_model_bakeoff.{md,pdf}`
  is the durable output; scripts hardcode a gitignored timestamp dir).
- **Docs pruning:** root `HANDOFF_pipeline_footprint_fixes.md`; `docs/HANDOFF_dense_leaf_placement_routing.md`
  ("Status: RESOLVED"), `docs/HANDOFF_array_tasks_4_5.md`, `docs/HANDOFF_routing_bottleneck.md`;
  `docs/web_live_view_plan.md` (claims the GUI is "intentionally untouched" — false);
  `docs/parts_single_source_of_truth_plan.md` (verify superseded). Matches roadmap Phase 5.

### Tier 3: CAUTION — product decisions / stale-not-dead

- **~19 vestigial console-script tools in `kicraft/cli/`** (layout helpers: `add_gnd_zone`,
  `add_group_labels`, `align_components`, `arrange_grid`, `check_trace_widths`, `cleanup_routing`,
  `move_component`, `net_report`, `run_drc`, `list_footprints`, `score_layout`; experiment
  analysis: `diff_rounds`, `plot_results`, `render_failure_heatmap`, `clean_experiments`,
  `watch_status`, `inspect_subcircuits`, `inspect_solved_subcircuits`, `render_pcb`; plus
  `leaf.py`/`kicraft-leaf` whose docstring points at a deleted GUI file). Nothing in the repo
  exercises them beyond `--help` smoke tests, but they're installed entry points a human might
  use. Decide keep-as-toolbox vs cut; cutting must also remove the `pyproject.toml`
  `[project.scripts]` lines + `test_cli_help` cases.
- **`deploy/tuning-i7/`** — the only tuning deploy harness, but named for VOID iteration i7 with
  VOID i10 defaults (`docker-compose.yml:19`, `entrypoint.sh:61,84`, `sync-to-admin.sh`).
  Don't delete; re-point defaults. Same for `scripts/tboard.py:23` (`DEFAULT_DB=/data/runs/i8`).
- **Stale strings/docstrings** (5-min sweep): `routes_admin.py:1271` UI label tells admins to run
  the deleted `kicraft-gui`; `session.py:59,116-120` describes multi-layout state resolution that
  no longer exists; `storage.py:106` "legacy" rationale; `web.py:4526` "view mode (ws=None)"
  comment; GUI-module references in `render/index.py` docstring (moot if deleted),
  `cli/solve_subcircuits.py:701,1236`, `autoplacer/brain/pins.py:66`;
  `.claude/skills/kicraft/SKILL.md:139` still mentions the Experiment Manager.

### Verified NOT dead (don't re-litigate)

- `_optimize_block_rotation`/`_score_rotation_for_block` — the "SA makes pre-SA rotation inert"
  memory was **refuted**: the step-5 `width_mm`/`height_mm` bbox swap survives SA
  (`_update_pad_positions` never touches bbox), and SA never rotates locked blocks. Memory updated.
- Parts CLI subcommands in cli_app (`add-part`, `lookup-lcsc-id`, …) — live via `stage_driver.py:441-484`
  + skills; the roadmap "parts_cli cut" is a move, not a deletion.
- `cli/solve_hierarchy.py` (invoked via non-module-path forms), `render_drc_overlay`,
  `web_cost_report`, `token_report`, `tuning/workspace.py`, `stagetabs.py`, `smoketest.py`/
  `accounts_cli.py` (console entry points), all `@ui.page`/`@app.get` routes, every
  `Settings` field, `tests/test_eval_projects.py` (a regression guard, not a stale test).

**Estimated total: ~1,500–2,000 lines of Python (Tier 1+2) + ~70 KB of one-off scripts + doc noise.**
Suggested order: Tier 1 in 2–3 commits (autoplacer cluster separate, `replay` no-op diff as the
gate), Tier 2 after the two product decisions, Tier 3 strings sweep anytime.

---

## Part 2 — Measured pipeline profile (34-run sweep, 2026-06-30)

- **Median 18.9 min/design: ~6 min LLM synthesis + ~11 min place/route + tail.** Worst 36.4 min.
- Worst-case decomposition (run_19 relay-quad, 12 leaves): 8.5 min synthesis → **16.9 min leaf
  solving** (~9 rounds/leaf, synchronized round barriers) → 5.0 min parent compose+route →
  **3.7 min fab gate/gerbers → 1.7 min 3D render** → zip.
- `.experiments/` ≈ 8–20 MB/run (~50 files/leaf: every round's full board snapshot + PNG renders).

### Where the time actually goes (code-level)

1. **Hundreds of throwaway pcbnew subprocesses per build.** Every atomic board op
   (`clear_traces`, `clear_zones`, `export_dsn`, `import_ses`, `count_board_tracks`,
   `strip_net_copper`=3 spawns, pours, repairs…) launches a fresh interpreter that imports
   pcbnew + `LoadBoard()`s + `Save()`s (`freerouting_runner.py:164` + 8 call sites,
   `_compose_route.py:289,324`). A 4-leaf `good` build ≈ ~200 leaf-phase + ~60 parent-phase
   spawns, each paying ~1–2 s import + board parse. The SWIG-SIGSEGV isolation rationale is real
   but only documented for specific combos (`freerouting_runner.py:334-355`).
2. **Freerouting runs 9× per leaf** (3 experiment rounds × 3 inner attempts, `good` preset) —
   the search scores candidates by routed DRC, so every attempt pays a full route.
   `max_passes=40`, `timeout_s=120` (`freerouting_runner.py:1137`), crash-retry ×2, plus a second
   full parent route attempt on GND-strand (`_compose_route.py:196-216`).
3. **SA does a full-board O(n²) rescore per single-component move** — `_sa_refine`
   (`placement_solver.py:2672`, 300 iterations default) calls `scorer.score().total` each move;
   `score()` runs ~13 sub-scores incl. three O(n²) pair loops and `count_crossings` (O(E²));
   `total_ratsnest_length` and `count_crossings` **each rebuild the same per-net MSTs**
   (`graph.py:211,143`) — 2× MST work per call; improving moves deep-copy all components.
4. **Diagnostic renders nobody reads:** every leaf × every round renders board PNGs + DRC overlay
   + contact sheet via kicad-cli/montage (`leaf_routing.py:219-240`), but headless builds surface
   only the final board.
5. **Sequential rounds:** the 3 experiment rounds and 3 parent rounds run strictly sequentially
   (`autoexperiment.py:2437,2602`); only within-round leaf solving is parallel
   (`solve_subcircuits.py:1610`, ProcessPool cpu−1).
6. **Synthesis side:** stages are inherently sequential (data-dependent), up to 3 model calls per
   stage on retry, and **each commit shells a fresh `cli_app stage-commit` interpreter**
   (`stage_driver.py:35,502`, ~15/run) plus per-part BOM tool subprocesses.
7. **Caching gaps:** `route_cache` only covers fully-array leaves (`leaf_routing.py:709-741`);
   `leaf_library` (promoted leaf reuse) is **never consulted on the build path** — only synthesis
   imports it.

---

## Part 3 — The plan

Verification harness for everything below: `kicraft replay --project <dir> --seed 0` corpus
before/after (byte-diff boards for no-op changes; fab-ready/DRC/wall-clock deltas for behavioral
ones). Autoplacer changes are surgical-only per the standing roadmap principle.

### Wave 1 — free wins (days; low risk)

| # | Change | Expected effect |
| --- | --- | --- |
| 1.1 | Delete Tier-1 dead code (incl. the 480-line `solve_group` subsystem) | −~2k lines; solver file legibility; zero behavior change (replay no-op gate) |
| 1.2 | **Gate intermediate diagnostic renders off for headless builds** — render once at promote (flags exist via the `fast_smoke_mode` path) | Kills per-leaf-per-round kicad-cli/montage spawns; meaningful leaf-phase wall-clock + ~half of `.experiments/` bytes |
| 1.3 | **Share the per-net MST between `total_ratsnest_length` and `count_crossings`** + hoist duplicated `sum(area)` (`placement_scorer.py:105,127`) | ~Halves MST cost of every `score()` call × 300 SA iters × 9N attempts; byte-identical scores |
| 1.4 | Batch the *known-safe* adjacent pcbnew ops into single scripts (first slice: `clear_traces`+`clear_zones` in `prepare_board_for_placement`, `freerouting_runner.py:371-377`; keep documented-crashy combos isolated) | Removes a slice of the ~260 spawns at near-zero risk |
| 1.5 | Stop writing per-round full-board snapshots to `.experiments/` by default (keep best + last; a debug flag restores full history) | ~8–20 MB → ~2 MB per run; less I/O in the round loop |

### Wave 2 — structural performance (1–2 weeks; medium risk, replay-gated)

| # | Change | Expected effect |
| --- | --- | --- |
| 2.1 | **Persistent pcbnew worker**: one long-lived subprocess holding the board in memory, driven by a small stdin command protocol; adjacent strip→pour→repair→count chains become in-memory ops. Keep process isolation as the crash boundary (restart worker on SIGSEGV = today's semantics). | The largest wall-clock lever: eliminates most of the ~1–2 s × hundreds import+parse tax; plausibly minutes/build |
| 2.2 | In-process stage-commit (call the commit function instead of shelling `cli_app stage-commit`; same for BOM part tools where safe) | Removes ~15 interpreter spawns from the synthesis path; retarget the monkeypatch seams |
| 2.3 | Overlap the parent rounds OR pipeline leaf-phase and parent-phase experiment rounds (respect the host `build_slot` flock; per-round `.experiments/` dirs already separate) | Bounded: helps freerouting-bound parent phase |

### Wave 3 — search-strategy wins (bigger prize, higher risk; only behind replay + quality gate)

| # | Change | Expected effect |
| --- | --- | --- |
| 3.1 | **Route less: score placement candidates with a cheap routability proxy, freeroute only the top-k** (today: 9 full routes/leaf). Even top-3-of-9 ≈ ~2× leaf-phase cut. Quality gate: fab-ready rate + DRC deltas on the full replay corpus, N-of-3 medians (the known nondeterminism floor makes single runs non-evidence). | Attacks the single biggest time sink |
| 3.2 | Incremental/delta SA scoring (rescore only pairs touched by the moved component; dirty-set or spatial index for the O(n²) loops; replace deep-copy-on-improve with move journaling) | Big SA speedup; HIGH risk in load-bearing code — do last, replay-gated |
| 3.3 | Wire `leaf_library` into the build path: content-hash promoted leaves and skip solve+route on exact hits (the `route_cache` pattern generalized) | Near-zero leaf cost for repeated/common blocks |

### Simplification (concept count, mostly already-planned — sequence, don't duplicate)

- **Net-new here:** the pcbnew-op consolidation (2.1) *is* the simplification of the
  "files-as-IPC ×20 inside one compose" pattern; Tier-1/2 deletions; killing the
  one-representation-per-hop conversions is subsumed by roadmap Phase 4b — don't start it separately.
- **Already planned — do there:** web.py `project_view`/`build_orchestration` splits (Phase 3,
  sequenced after 4b), state single-source-of-truth (4b), `parts_cli` move (blocked on its tests),
  parent-local-vs-leaf placement-path collapse (simplification doc Lever 2.1, gated).
- **Config surface:** `DEFAULT_CONFIG` is 163 keys; presets vary 4. After the Tier-1 dead-knob
  removals, a "knobs actually read" audit pass would shrink it further — fold into the tuning
  framework's search-space list as the source of truth for what's live.

### Suggested sequencing

1. Wave 1 now (1.1 dead code first — it also de-noises everything after).
2. Wave 2.1 prototyped on the parent-compose chain only (worst spawn density), then leaf phase.
3. Wave 3.1 as an experiment arm in the self-eval harness before making it default.
4. Re-measure the same 34-brief sweep after each wave; the current baseline is
   `20260630T152811Z` (median 18.9 min).
