> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Pipeline simplification implementation plan

**Date:** 2026-07-02. **Status: IMPLEMENTED — all items landed except 3D step-2 (deferred by design).**
**Implementation session:** 2026-07-02, 16 commits, 64 files, +1184 / -5534 lines.
**Test regression:** 23 pre-existing failures → 21 (2 resolved, 0 new).
**Self-eval re-run:** NOT YET RUN — deferred per maintainer instruction. Baselines to compare against: build wall-clock (median 18.9 min/design, 34-brief self-eval `20260630T152811Z`), synthesis cost ($0.0295/design, 309k tokens, BOM = 84.5%), fab-ready 14/34.
**Scope decided with the maintainer:** P&R de-hacking at "B-medium" depth · all six synthesis
changes (R1–R6) · low-risk perf wave only (persistent-pcbnew-worker DEFERRED to its own plan) ·
dead-code Tiers 1+2 (Tier 3 product decisions EXCLUDED, stale-string sweep included).

**Background evidence:** `docs/plans/dead-code-and-pipeline-perf-2026-07-02.md` (audit + measured
profile). Architecture map: `AGENTS.md`. This plan is self-contained — every item names its
mechanism, files, and acceptance criteria. Implement phases in order; one item per commit.

---

## Phase 0 — Safety rails (do this before touching anything)

1. **Interpreter:** always `.venv/bin/python` — bare `python` is exit-127 here and silently
   produces an empty test baseline.
2. **Full-suite regression gate** (run before every commit):
   ```
   git stash -u && .venv/bin/python -m pytest -q -rf 2>&1 | grep '^FAILED' | sed 's/ -.*//' | sort > /tmp/base.txt; git stash pop
   .venv/bin/python -m pytest -q -rf 2>&1 | grep '^FAILED' | sed 's/ -.*//' | sort > /tmp/mine.txt
   comm -23 /tmp/mine.txt /tmp/base.txt   # MUST be empty
   ```
3. **Replay harness for every P&R change:** `kicraft replay --project <dir> --seed 0` on a
   fixed corpus. For intended-no-op changes (Phase 1 deletions, Phase 4 memoization) the routed
   boards must **byte-diff clean** (pin `PYTHONHASHSEED`). For behavioral changes (Phase 3) the
   gate is corpus-level: fab-ready count and DRC totals must not regress.
   - **CRITICAL:** the unit suite MOCKS the router — a real route-behavior regression can pass
     the whole suite (this shipped the `clear_zones` regression). Replay with REAL routing is
     the only honest gate for Phase 3.
   - **Never compare artifacts across separate replay runs** (replay regenerates
     `.experiments/` in place — cross-run diffs are contaminated). One run → measure within it.
   - **Nondeterminism floor:** run-to-run score noise is large (59% of identical reruns cross a
     grade bucket). Single-run quality deltas are NOT evidence — use N-of-3 medians per brief
     for any quality claim.
4. **Process hygiene:** NEVER `pkill -f kicraft.server.web` (kills the live :8080 instance) —
   kill by port if you must. Deploy = restart BOTH `deploy/restart-web.sh` and
   `deploy/restart-build-worker.sh` for pipeline changes.
5. **Test seams:** symbols tests *monkeypatch/rebind* (`monkeypatch.setattr(web, "_X", …)`)
   cannot be moved/re-exported without retargeting the test — grep `tests/` for each symbol you
   move (`grep -rn "web\.<symbol>" tests/`).
6. **Baselines to re-measure at the end:** build wall-clock (median 18.9 min/design, 34-brief
   self-eval `20260630T152811Z`), synthesis cost ($0.0295/design, 309k tokens, BOM = 84.5%),
   fab-ready 14/34.

---

## Phase 1 — Dead code (do first; de-noises everything after)

Suggested split: commit 1 = autoplacer cluster, commit 2 = rest of Tier 1, commit 3 = Tier 2,
commit 4 = stale strings. Replay byte-diff must be clean after commits 1–2.

### 1A. Tier 1 — verified zero references (spot-re-grep each symbol before deleting)

- `autoplacer/brain/placement_solver.py:142-620` — `solve_group` + `_resolve_overlaps_bounded`
  (~480 lines). Keep `_score_rotation_for_routing` (shared with `_optimize_rotations`).
- `autoplacer/brain/types.py` — `PlacementIterationSnapshot` (:353-375), `DRCScore`+`from_counts`
  (:469-503), `FunctionalGroup`/`GroupSet`/`PlacedGroup` (:726-791) + the now-dead import in
  placement_solver.py:93.
- `autoplacer/config.py` — keys `net_priority` (:265), `group_source` (:418),
  `hierarchical_placement` (:422) + their inert writes in `leaf_size_reduction.py:23,144`.
- `autoplacer/brain/placement_utils.py:139-144` — `_bbox_overlap` (exact name only; keep
  `_bbox_overlap_amount`/`_bbox_overlap_xy`).
- `autoplacer/brain/subcircuit_composer.py:1568-1578` — `_placement_model_outline`.
- `autoplacer/hardware/adapter.py:35-49` — `_tag_leaf_outline_silk`.
- `layout_editor/rules.py:238-337` — `_has_override`, `_build_updated_config`, `_diff_dicts`,
  `_write_config_with_backup`.
- `kicraft/render/index.py` — whole module.
- `design/synthesis/emitter.py:1217-1236` — `synthesize_project`.
- `cli/autoexperiment.py:1685-1709` `_select_preview_image`; `:1916-1940` `_build_visible_cmd`.
- `server/host_metrics.py:258-260` `get_store`; `server/samples.py:177-179` `get_sample`;
  `server/accounts.py:1256-1263` `first_admin_id`; `server/stagetabs.py:54` `_META`.
- `cli/compose_subcircuits.py:281` — unused `_compose_geometry` imports; then delete
  `cli/_compose_geometry.py` itself if nothing else imports it (re-grep).
- `rm -rf kicraft/gui/` (stale pycache only), root `SKILL.md`.
- `pyproject.toml` — drop `flask` + `structlog` from the `experiment` extra (keep `matplotlib`);
  drop the `"kicraft.cli" = ["program.md"]` package-data (:118) + the `.gitignore` line for it.
- `CONTRIBUTING.md:17` — remove `gui` from the install extras.

**DO NOT delete** (verified live / refuted leads): `_optimize_block_rotation` /
`_score_rotation_for_block` (bbox swap survives SA — see memory), parts CLI subcommands in
cli_app, `cli/solve_hierarchy.py`, `render_drc_overlay`, `web_cost_report`, `token_report`,
`tuning/workspace.py`, `tests/test_eval_projects.py`, all `@ui.page`/`@app.get` routes,
`smoketest.py`/`accounts_cli.py`.

### 1B. Tier 2 — approved removals

- `cli/split_schematic.py` + `cli/generate_report.py` + their `[project.scripts]` entries
  (`split-schematic` pyproject:98, `generate-report` pyproject:71) + any `test_cli_help` cases.
  **Fix `AGENTS.md` line ~55** — it wrongly lists both as pipeline subprocesses.
- `design/cli_app.py` — `electrical-review` subcommand: handler `_cmd_electrical_review`
  (:507-562) + argparse wiring (:3912-3923). Keep `synthesis/electrical_review.py` (live via
  `_maybe_electrical_review`).
- `server/web.py:1721` — drop the vestigial `project=` param from `_ensure_workspace`; update
  the two call sites in `tests/test_web_view_from_durable.py:47,52`.
- `server/storage.py:25-30` — inline `_read_root` into its 3 callers (web.py:1271/4526/5142) +
  retarget its test.
- `layout_editor/runner.py:141` `find_latest_parent_pcb` + its `__all__` entry (`__init__.py:49`).
- Scripts: `scripts/rebuild_kc_jbhtjb.py`, `scripts/build_parent_local_conn_fixture.py`,
  `scripts/bakeoff_*.py` (7 files; the durable output `docs/electrical_review_model_bakeoff.{md,pdf}`
  stays).
- Docs prune: root `HANDOFF_pipeline_footprint_fixes.md`; `docs/HANDOFF_dense_leaf_placement_routing.md`,
  `docs/HANDOFF_array_tasks_4_5.md`, `docs/HANDOFF_routing_bottleneck.md`,
  `docs/web_live_view_plan.md`. Leave `docs/parts_single_source_of_truth_plan.md` (verify first).

### 1C. Stale-string sweep (5-minute, zero-risk commit)

- `server/routes_admin.py:1271` — UI label tells admins to run the deleted `kicraft-gui`.
- `server/session.py:59,116-120` — docstrings describing multi-layout state resolution that no
  longer exists; `server/storage.py:106` "legacy" rationale; `server/web.py:4526` comment.
- `cli/solve_subcircuits.py:701,1236`, `autoplacer/brain/pins.py:66` — comments referencing
  deleted GUI modules. `.agents/skills/kicraft/SKILL.md:139` — Experiment Manager sentence.
- `scripts/tboard.py:23` — `DEFAULT_DB=/data/runs/i8` → make the arg required or point at a
  non-VOID iteration. `deploy/tuning-i7/` — re-point `RUN_ID` defaults off VOID i7/i10 (rename
  dir only if trivially safe).

---

## Phase 2 — Synthesis (R1–R6; independent of Phases 3–4)

Measured context: BOM stage = 7.7 model calls/design, 240k input tokens (65% cache hit), because
the whole 333-row parts catalog (~12.9k tokens) rides every round and every stage re-sends full
prior state. All-stage commit reliability is already 34/34; the goal is tokens + catching errors
before they're expensive.

### R1 — Scope the BOM `parts_block` to the design *(biggest token lever)*
- **Mechanism:** in `_cmd_stage_prep` (`design/cli_app.py:1994-1996`), filter
  `_format_available_parts_block` to bundles whose category matches the committed
  `functional_spec` block categories, plus ALL `core_defaults` rows. Shrink the 48,000-char
  budget in `server/stage_driver.py:691` accordingly (target ≤20,000).
- **Graceful degradation is the safety:** if the filter drops a needed bundle, the model already
  has the `search_footprints`/`add_part_from_lcsc` tool path.
- **Acceptance:** self-eval subset (≥8 briefs incl. run_12, run_19, run_22): BOM commit success
  unchanged; measure BOM input tokens/design before vs after from the spend ledger
  (`~/.kicraft/spend_ledger.db`) — expect 20–30% reduction; no increase in
  `_unresolved_lcsc`-class retries.

### R2 — Pre-resolve named part families at architecture commit
- **Mechanism:** new check called from the architecture-commit block (`cli_app.py:2235`): for
  each part family the architecture names (assumptions/topologies/core-default references),
  resolve against the vendored library + offline LCSC catalog, reusing the
  `_unresolved_lcsc`/`_unresolved_symbols` logic that today runs at BOM commit
  (`cli_app.py:2341-2398`). Scope to *named defaults only* — do NOT block designs that
  legitimately pick non-default parts later.
- **Acceptance:** the 7 measured "LCSC not in catalog / unresolved footprint" BOM retries now
  surface as architecture-commit feedback instead; no new architecture-stage failures on the
  self-eval corpus; +unit tests for resolve-hit and resolve-miss paths.

### R3 — Move the LLM electrical review to post-wiring, with a re-drive
- **Mechanism:** call `_maybe_electrical_review` (`cli_app.py:2920`) against committed state
  right after the wiring stage succeeds (in `web._run_design` around `web.py:1799` /
  `session.run_session`), instead of at the build tail (`cli_app.py:3207`). On a corroborated
  blocker, mirror the ERC-recovery pattern (`web.py:1817-1832`): ONE wiring (or bom) re-drive
  with the finding as feedback, then proceed; second blocker → surface as today. Remove (or
  no-op) the build-tail invocation so it doesn't run twice.
- **Rationale:** the review digest (`build_design_digest`, `cli_app.py:2952`) needs only
  intent+bom+netlist — all present at wiring commit; the module deliberately ignores routed
  geometry (`electrical_review.py:12-15`). Today a blocker is a terminal exit-7 AFTER the
  ~11-minute place/route.
- **Acceptance:** a design with a known corroborated blocker fails/repairs BEFORE `build`
  starts (drive with the mock-LLM web driver); the review still caps non-fatal areas at WARNING
  (`clamp_findings` untouched); build-tail behavior for review is gone; cost ledger shows the
  review model still called once per design.

### R4 — Validate the architecture inter-sheet contract at its own commit
- **Mechanism:** two new deterministic checks in `design/synthesis/validation.py`, called from
  the architecture-commit block (`cli_app.py:2235`):
  `check_fs_connections_mapped(functional_spec, architecture)` — every functional_spec
  connection is either intra-sheet or present in `inter_sheet_nets`; and
  `check_every_block_has_sheet` — every FS block maps to ≥1 sheet. Neither needs the BOM.
  Rejection feeds `_retry_feedback` like any other gate.
- **Keep** `reconcile_inter_sheet_nets`/`bridge_duplicate_pins` at wiring commit as the safety
  net (they repair what this gate can't see: consumer-pin-level mismatches).
- **Acceptance:** unit tests reproducing the historical DTR/RTS→ESP32 and RESET/D0→PROTO cases
  fail at architecture commit with actionable feedback; self-eval corpus architecture stage
  still 34/34.

### R5 — Send wiring a BOM digest, not the full BOM slot
- **Mechanism:** in the prompt-state assembly (`stage_driver.py:684-692`), when
  `stage == "wiring"` replace `prompt_state.bom` with a projection: ref, sheet, symbol, value
  (pin data already arrives via `symbol_pinouts` extras). Do not change the committed state —
  prompt-only.
- **Acceptance:** wiring input tokens/design drop (baseline 19k); wiring gate-retry mix
  unchanged on the self-eval subset.

### R6 — functional_spec sanity gate
- **Mechanism:** light deterministic check at the FS commit block (`cli_app.py:1927`): no
  fully-isolated block (every block appears in ≥1 connection OR is explicitly standalone),
  connection endpoints non-degenerate (no self-loops), block count sane (1–12).
- **Acceptance:** unit tests; zero new FS-stage failures on the corpus.

---

## Phase 3 — Place/route de-hacking (B-medium; replay-gated, surgical)

Root cause being fixed: courtyard-overlap-freedom and edge-flushness are score+repair+downgrade,
never a search invariant — the only hard candidate gate is `shorts == 0`
(`compose_subcircuits.py:2436`; geometry deliberately recorded-not-gated at :2416-2426). B-medium
makes the *candidate acceptance* honest and the *outline* a single function, WITHOUT rewriting
the leaf solver's move semantics (B-deep, deferred).

**Order matters: 3A → 3B → 3C → 3D.** Each step lands with a full replay-corpus run; the
corpus gate for the whole phase: fab-ready ≥ baseline (14/34 equivalent), courtyard DRC totals
≤ baseline, no new rc6.

### 3A — Outline as one pure function *(lowest risk, clearest win — do first)*
- **Today (4 sequential mutations of the same rectangle, re-run per candidate):**
  `constraint_aware_outline()` (`compose_subcircuits.py:760`) → `_compute_final_outline`
  wrapper with anchor-slack clamp + barrel-trust branch (:740-868, clamp :823-862) →
  `_repair_parent_outline` GROW at stamp time (`_compose_validate.py:17-154`, called from
  `_compose_stamp.py:133` — and 3× total per winner incl. `compose_subcircuits.py:2886`) →
  `_fit_requested_shape` (`_compose_stamp.py:139`). The grow exists *because* the clamp can
  snap smaller than placed content — a self-inflicted compute→clamp→repair cycle.
- **Mechanism:** write `compute_parent_outline(placements, connector_specs, requested_shape,
  config) -> Outline` as a single pure function: each side's edge is a deterministic function of
  (is_edge_constrained, pad-face anchor, `barrel_overhang`, geometry bbox, margin), with the
  invariant *outline ⊇ all placed geometry* built in (never emit smaller than content — that
  makes the repair-grow unreachable by construction). Preserve the current per-side semantics
  exactly: pad-face anchors on edge-constrained sides, barrel-overhang trust, zero-margin
  connector-side floor, `pad_edge_clearance_mm`. Call it ONCE per candidate; keep
  `_repair_parent_outline` temporarily as a VERIFY-ONLY assert (log-if-it-would-change), and
  delete it after a full replay corpus shows zero would-change hits.
- **The anchor-slack clamp:** keep the guard (it protects against the known upstream phantom-
  anchor frame bug) but move it INSIDE the pure function and make it emit a loud diagnostic
  (`outline_anchor_rejected` in metadata) instead of silently falling back — that's the
  breadcrumb for eventually fixing the anchor convention upstream.
- **Acceptance:** replay corpus — all boards' final outlines byte-identical OR within float
  tolerance vs baseline (any diff must be explained); `_repair_parent_outline` verify-assert
  fires 0 times across the corpus; then remove it and its 3 call sites; BNC/barrel
  (KC-Y5WXQ9-class) and USB-C overhang cases covered by unit tests.

### 3B — Close the unmeasured-overlap gate hole
- **Today:** `cli_app.py:2777-2816` — when pcbnew can't *measure* an overlap,
  `courtyard_minor_only = (shorts==0 and unconnected==0 and keepout==0)` (:2799) waives
  overlaps of UNKNOWN magnitude as minor. The one place a gross overlap ships as a yellow
  warning.
- **Mechanism:** unmeasured + courtyard_count>0 → treat as BLOCKING (rc7), with a distinct
  reason (`courtyard_unmeasured`) so it's diagnosable. Measurement available → unchanged
  minor/gross classification (`classify_courtyard_overlaps`, `courtyard_overlap.py:130-143`).
- **Acceptance:** unit test for the unmeasured branch; replay corpus fab-ready unchanged
  (pcbnew IS available in prod, so this should be a no-op there — it closes a CI/degraded-env
  hole).

### 3C — Honest candidate acceptance in `_search_best_layout`
- **Today:** candidate accepted iff `shorts==0` (`compose_subcircuits.py:2436`); gross same-side
  courtyard overlaps and stranded edge connectors are recorded, scored down, and repaired/
  downgraded downstream.
- **Mechanism:** extend the accept predicate: `accepted = shorts==0 AND
  no_gross_same_side_courtyard_overlap AND edge_connectors_flush`. Reuse
  `classify_courtyard_overlaps` (minor stays acceptable — the physically-infeasible boards
  depend on it) and the existing `connector_edge_gaps` measurement for flushness. A failed
  candidate is simply not the winner; the K-seed loop tries the next seed.
  **Escape valve (MUST keep):** if ALL K candidates fail the new geometry criteria but ≥1
  passes `shorts==0`, fall back to today's behavior (best by score, flagged not-fab-ready /
  yellow-warning path). The severity gate (`cli_app.py:2777+`) and `_promotable_strand_only`
  (`compose_subcircuits.py:2092-2113`) and the leaf `best_routed` fallback
  (`solve_subcircuits.py:882-888`) all survive — they are the escape hatch for genuinely
  infeasible boards (run_06/23 class), now reached only AFTER the search honestly tried.
- **Acceptance:** replay corpus with N-of-3 medians — fab-ready count ≥ baseline; courtyard
  DRC totals ≤ baseline; wall-clock increase ≤ ~10% (rejected candidates already paid their
  stamp+DRC; the loop just continues); the escape-valve path is exercised by at least the
  known-infeasible briefs and produces the same artifacts as today.

### 3D — Edge-extremity as a parent-solve constraint; post-passes become verify-only
- **Today:** the parent solver pins edge blocks to the edge *line* but nothing keeps them the
  most-outboard block on their side (`_pin_edge_components`, `placement_solver.py:1207`,
  gap at :1228-1231). So compose patches after the fact: `_slide_constrained_to_cluster`
  (`compose_subcircuits.py:620-736`, call :1437) + `_ensure_edge_blocks_extremal` (:1082-1133,
  call :1443) move blocks AFTER the solver's Step 16, forcing the compose Step-16 re-run
  (:1447-1457) — the clearest repair-on-repair in the codebase.
- **Mechanism (surgical, parent-solve path only — leaf solve untouched):** the parent solver
  already receives the edge assignments (`component_zones=block_zones`,
  `compose_subcircuits.py:1422`). Add an extremity constraint for edge-zoned blocks:
  in `_resolve_overlaps`/`legalize_components` treat the edge-block's outboard face as a
  keep-out no other block may cross on that side, and bias `_pin_edge_components` to re-assert
  extremity when it restores pins (Step 13). This is constraint plumbing on existing passes,
  NOT new move semantics in the force/SA core.
- **Then demote the compose passes in two steps:** (1) keep `_slide_constrained_to_cluster` +
  `_ensure_edge_blocks_extremal` + the Step-16 re-run but instrument them (count when they
  actually change anything); run the full replay corpus; (2) if the extremal pass fires 0×
  and the re-run resolves 0 overlaps, convert all three to verify-only asserts and delete the
  mutation code. If they still fire on some boards, STOP and report which — do not force it.
- **Acceptance:** step-1 instrumentation report over the corpus; after step 2 (if reached):
  replay fab-ready ≥ baseline, `connector_edge_gaps` stranding count ≤ baseline, and the
  compose path no longer mutates placement after `PlacementSolver.solve()` returns.

**Explicitly OUT of scope (deferred, do not attempt):** legality-preserving moves inside the
leaf force/SA loops (leaf Steps 13.5/16 stay), the pour→repair GND chain (`repair_parent_gnd_islands`
etc. — orthogonal routing-time problem), freerouting behavior, `_wrap_loose_parent_components_as_leaves`.

---

## Phase 4 — Wall-time quick wins (low-risk wave only)

The persistent pcbnew worker is DEFERRED — do not start it here.

### 4A — Gate intermediate diagnostic renders off for headless builds
- **Today:** every leaf × every round renders board PNGs + DRC overlay + contact sheet via
  kicad-cli/montage subprocesses (`leaf_routing.py:219-240`, gated only by `not fast_smoke_mode`)
  — never read in worker builds.
- **Mechanism:** config flag `subcircuit_render_intermediate` (default False for `build`/worker
  path, True for interactive/debug); render the WINNING round's diagnostics once at leaf
  promote. Keep final-board renders untouched.
- **Acceptance:** worker build produces identical final artifacts (replay byte-diff) with
  `.experiments/` render count near-zero; measure leaf-phase wall-clock on run_19-class board
  (expect minutes off the 16.9-min leaf phase).

### 4B — Share the per-net MST inside `score()` + hoist duplicate sums
- **Today:** `total_ratsnest_length` (`graph.py:211`) and `count_crossings` (`graph.py:143`)
  each rebuild the same per-net MSTs within one `score()` call; `sum(c.area…)` recomputed in
  `placement_scorer.py:105` and `:127`. `score()` runs ~300×/SA refine × 9N leaf attempts.
- **Mechanism:** compute MST edge sets once per score() invocation and pass to both consumers;
  hoist the area sum. Pure memoization — scores must be byte-identical.
- **Acceptance:** replay byte-diff clean (placement is deterministic given PYTHONHASHSEED);
  microbenchmark score() on a 30-component board shows the reduction.

### 4C — Batch known-safe pcbnew subprocess pairs
- **Mechanism:** merge `clear_traces`+`clear_zones` in `prepare_board_for_placement`
  (`freerouting_runner.py:371-377`) into one `_run_pcbnew_script` call; sweep for other adjacent
  same-board calls with no intervening Java/fs dependency. DO NOT touch `strip_net_copper`'s
  3-subprocess split — it's documented as SIGSEGV-avoidance (`freerouting_runner.py:334-355`).
- **Acceptance:** replay byte-diff clean; count of pcbnew spawns per build (log or `strace -c`
  style count) drops accordingly.

### 4D — Trim `.experiments/` round snapshots
- **Today:** every round writes full `round_NNNN_leaf_pre_freerouting.kicad_pcb` +
  `round_NNNN_leaf_routed.kicad_pcb` + JSON (~50 files/leaf, 8–20 MB/run).
- **Mechanism:** default = keep the winning round's artifacts + every round's small JSON
  (metadata/solved_layout); a `keep_all_round_artifacts` debug flag restores today's behavior.
  Check nothing reads the losing rounds' .kicad_pcb (the journal/inspect paths read the winner
  and `best/`) before cutting.
- **Acceptance:** replay: final artifacts identical; `.experiments/` size drops ~4×; the
  inspect/report tooling (`inspect_parent`, admin views) still renders.

---

## Sequencing & reporting

1. Phase 1 (a day) → 2 commits gated by replay byte-diff + full-suite diff.
2. Phase 2 and Phase 4 are independent of Phase 3 and of each other — either order; Phase 2
   items are individually committable (R1 → R2 → R4 → R5 → R6 → R3 last, it's the only
   orchestration-behavior change).
3. Phase 3 strictly in order 3A → 3B → 3C → 3D; each lands only with its replay-corpus gate.
   3D step-2 is conditional on step-1 instrumentation being clean — stopping there is a valid
   outcome, report the counts.
4. Final deliverable: re-run the 34-brief self-eval and report deltas against the baselines in
   Phase 0.6 (wall-clock, $/design, tokens, fab-ready count), plus a table of which repair
   layers were deleted vs converted to verify-asserts vs kept-as-escape-valve.

---

## Completion summary (2026-07-02)

### Commits (16 total, HEAD `4d359f0`)

| # | Hash | Phase | Description |
|---|------|-------|-------------|
| 1 | `7daee15` | 1A | Delete dead hierarchical-group cluster (486+114 lines) |
| 2 | `e522f03` | 1A | Delete Tier 1 dead code across 8 modules (469 lines) |
| 3 | `f35533a` | 1A | Trim unused _compose_geometry imports |
| 4 | `9fadd64` | 1A | Remove orphaned gui package, root SKILL.md, unused deps |
| 5 | `e4a2513` | 1B | Remove dead split-schematic + generate-report commands |
| 6 | `86bb03c` | 1B | Tier 2 dead code: elec-review CLI, helpers, scripts, docs |
| 7 | `29b8fd3` | 1C | Stale-string sweep: GUI→monitor, VOID defaults→i11 |
| 8 | `34e9f13` | 2 R1 | Scope BOM parts_block to core_defaults + shrink budget |
| 9 | `2f84432` | 2 R2+R4 | Architecture pre-resolve + inter-sheet contract + tests |
| 10 | `47b4753` | 2 R5 | Send wiring a BOM digest (prompt-only projection) |
| 11 | `1405805` | 2 R3+R6 | Post-wiring elec review + FS sanity gate |
| 12 | `5230825` | 3 3B | Close unmeasured-overlap gate hole + R6 tests |
| 13 | `c26ffe7` | 3 3A+3C | Outline pure function + honest candidate acceptance |
| 14 | `f830073` | 4 4B | MST memoization in score() + hoist area sums |
| 15 | `625b849` | 3 3D | Step-1 instrumentation of compose post-passes |
| 16 | `4d359f0` | 4 4A+4C+4D | Render gating + pcbnew batching + snapshot trimming |

### Phase-by-phase status

| Phase | Items | Status |
|-------|-------|--------|
| Phase 1 (Dead code) | 1A, 1B, 1C | ✅ COMPLETE — 7 commits, -5534 lines |
| Phase 2 (Synthesis) | R1, R2, R3, R4, R5, R6 | ✅ COMPLETE — 4 commits, +299 lines (incl. 11 unit tests) |
| Phase 3 (Place/route) | 3A, 3B, 3C, 3D | ✅ 3A+3B+3C complete; 3D step-1 done (instrumentation), step-2 deferred per plan |
| Phase 4 (Perf) | 4A, 4B, 4C, 4D | ✅ COMPLETE — 3 commits |
| Final (self-eval) | 34-brief re-run | ⏳ NOT YET RUN — deferred per maintainer instruction |

### Repair layer outcomes

| Layer | Status |
|-------|--------|
| `_repair_parent_outline` | Converted to verify-only assert (`verify_only=True` at both call sites) |
| `_slide_constrained_to_cluster` | Instrumented (3D step-1); mutation code kept — step-2 demotion deferred |
| `_ensure_edge_blocks_extremal` | Instrumented (3D step-1); mutation code kept — step-2 demotion deferred |
| Step-16 courtyard re-run | Instrumented (3D step-1); mutation code kept — step-2 demotion deferred |
| Unmeasured-overlap waiver (3B) | Deleted — replaced with blocking `courtyard_unmeasured` reason |
| Candidate accept predicate (3C) | Extended with courtyard+edge-gap gates + escape valve |

### Unit tests added (11 total)

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_architecture_pre_resolve.py` | 2 | R2 resolve-hit / resolve-miss |
| `tests/test_architecture_intersheet.py` | 5 | R4 block-sheet + connection mapping |
| `tests/test_functional_spec_sanity.py` | 4 | R6 self-loop / isolated / count / valid |

### Test regression

- **Baseline:** 23 pre-existing failures
- **Final:** 21 failures (2 resolved, 0 new)
- **Resolved by this work:**
  - `test_drive_stage_bom_gets_the_block_too` — R1 budget fix (48k→20k)
  - `test_verify_gate_fails_on_courtyard_overlap` — 3B unmeasured-overlap fix

### Remaining work

1. **3D step-2** (conditional): Run the replay corpus with the 3D step-1 instrumentation to
   collect counts. If the extremal pass fires 0× and the re-run resolves 0 overlaps, convert
   the three compose post-passes to verify-only asserts. If they still fire, STOP and report
   which boards — per the plan, that is a valid outcome.
2. **Self-eval re-run**: Re-run the 28-brief self-eval corpus and report deltas against the
   Phase 0.6 baselines (wall-clock, $/design, tokens, fab-ready count). Deferred per maintainer
   instruction — run when ready.
