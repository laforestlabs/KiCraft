# Plan: stage resource telemetry + BOM LLM-cost reduction

Status: IMPLEMENTED (Phases A + B), remaining work tracked here.
Written 2026-06-24. Audience: the implementing agent completing the work on
the testing deploy box.

## What is already done (this branch)

### Phase A: durable per-stage resource telemetry

The "quantify which stage takes more resources" loop now records wall time,
child CPU, tool rounds, and tool calls durably per stage, not just LLM cost.

- `kicraft/design/models.py` `StageStatus` carries `wall_s / cpu_s / rounds /
  tool_calls`. No prompt cost: `stage_status` is popped before the model sees
  state and is not in `SLOT_MODEL`.
- `kicraft/server/spend_guard.py` gained a `stage_runs` ledger table +
  `record_stage()`. Per-stage wall/CPU/rounds/cost are durable without
  inflating the spend ceiling (cost mirrors LLM spend for the report; the
  `spend` rows still own enforcement).
- `kicraft/server/stage_driver.py` `drive_stage` snapshots `time.monotonic()`
  + `RUSAGE_CHILDREN` at entry and records both to `state.json` and the ledger
  at every terminal path (prep-fail, ok, final-fail). `run_id` threads through
  from `meta_ctx`.
- `kicraft/cli/web_cost_report.py` gained `load_stage_runs` /
  `summarize_stage_runs` / `format_stage_runs`, always appended to the report.
  The `cpu/wall` ratio line is the key signal: BOM is latency-bound (~4%),
  place/route is CPU-bound (~100%).

### Phase B: BOM LLM-cost fixes

Targeted the churn measured in `~/.kicraft/part_queries.jsonl` (395 events:
`lookup_lcsc_id` = 56% of all queries; BMP280 resolved 47x; VL53L1 re-spelled
56x).

- **Server-side per-MPN search budget** (`_bom_executor`): normalizes the MPN
  (case + whitespace + collapses pasted lcsc.com/jlcpcb.com URLs to the bare
  C-number), caps `lookup_lcsc_id` at 2 calls per normalized MPN per stage
  attempt, then returns a terminal "STOP retrying this part" result.
- **Persistent MPN->LCSC cache** (`kicraft/parts_library/mpn_cache.py` + the
  `lookup-lcsc-id` CLI): a part resolved once on a machine resolves instantly,
  offline, on every later run. Stored as `~/.kicraft/mpn_cache.json` (override
  `$KICRAFT_MPN_CACHE`). Verified: BMP280 -> cached -> `bmp280` resolves in
  0.31s with `source: mpn-cache(via parts-library)`, no network.

### Tests

- `tests/test_stage_resource_telemetry.py`: model fields; all three sinks
  (caller return, state.json, ledger); intent-null-rounds; report aggregation.
- `tests/test_bom_search_budget.py`: cache roundtrip/corrupt-file/URL-folding;
  per-MPN cap; per-spelling budget independence.
- `tests/test_spend_guard.py`: `record_stage` writes the row, nulls rounds for
  single-shot stages, does not inflate the spend ceiling.

199 passed across touched + related suites.

## Remaining work

### R1. Roll telemetry onto the production deploy box and capture a baseline

The dev box ledger is legacy (no `stage_runs` rows, `meta=chat` bare strings),
so the new report ships empty there. The fixes only become measurable on the
hosted box where `CappedOpenRouterClient` runs.

Steps:
1. Deploy this branch to the testing box (`deploy/restart-web.sh` +
   `deploy/restart-build-worker.sh`; both processes own the ledger).
2. Run one self-eval or ~3 live designs so `stage_runs` and `spend` populate
   with the new structured `meta` (stage/attempt/round).
3. Capture the baseline:
   - `web-cost-report --by all` (note the new "By stage" block: BOM wall_s +
     cpu/wall% + rounds + tools).
   - `part-query-report --since 2026-06-24` (note `lookup_lcsc_id` total +
     counts-per-MPN + repeat rate).
   - `mpn_cache.json` size (new file; confirm it grows).
4. Save these three outputs as `docs/bench/<date>_baseline_{cost,queries}.txt`
   for the before/after diff.

Acceptance: the "By stage" block shows BOM with the highest `wall_s`,
`rounds`, and `tools`, and `cpu/wall` in the low single digits (latency-bound,
not CPU-bound). This is the measurement contract for every later fix.

### R2. Drive vendoring from real churn, not the fixed slug table

`docs/parts_single_source_of_truth_plan.md` lists 34 parts to vendor but omits
the parts the corpus actually asks for. `part_queries.jsonl` shows the misses:

- `BMP280` / `C83291` (47 resolves, never bundled) — not in the plan at all.
- `VL53L1CXV0FY/1` / `C190004` (20 resolves) — plan vendored `VL53L0X`
  (`vl53l0x`) but designs need the `VL53L1` family. Vendor VL53L1 as a distinct
  bundle; keep VL53L0X too.
- `C8051F320` (18 unresolved) — a SiLabs MCU; likely needs a vendor library or
  a deliberate "advanced: surface a material question" rather than vendoring.
- Re-check `MISSES -> ADD-TO-LIBRARY candidates` after the cache lands: any
  C-number above ~5 resolves is a vendoring candidate.

Steps:
1. After R1's baseline, run `part-query-report --since <deploy-date> --json`
   and sort `unresolved` + `jlcpcb` (resolved-not-bundled) by count.
2. For each candidate above the threshold, follow the existing vendoring
   workflow (plan SS Vendoring workflow): `add-part --from-lcsc C### --into
   vendored --name <slug>` -> author manifest -> `validate-part --update-hash`
   -> `scripts/render_check_bundles.py` -> eyeball -> promote -> flip the
   catalog row from `default_lcsc` to `bundle: <slug>`.
3. Watch the EasyEDA rate limit (403 after ~17 rapid fetches, ~10 min cooldown;
   space ~20s). Batch V1-V4 per the plan.
4. Re-run `part-query-report` after each batch; the `LIBRARY HITS` count for
   the newly-vendored bundles should rise and the `MISSES` count for those
   C-numbers should fall to zero (the `evq-p7a01p` precedent: churned 28x then
  stopped once bundled).

Acceptance: no C-number appears in `MISSES` with >2 resolves after the cache +
vendoring land; `LIBRARY HITS` covers the top-10 churned parts.

### R3. Pre-filter the BOM parts block by architecture categories

`docs/parts_single_source_of_truth_plan.md:373-375` flags it: every BOM run
lists all 64 bundles (~19.8 KB / ~5K tokens) re-sent on every tool round even
for trivial boards. With the parts block growing to 55+ this is a per-round
tax.

Steps:
1. In `kicraft/design/cli_app.py _cmd_stage_prep`, when `stage == "bom"`, read
   `state.architecture`'s functional blocks / categories and pass them to a new
   `_filter_parts_block(parts, arch_categories)` that keeps: all passives
   rows, any bundle whose `core_blocks` category matches an architecture
   block, plus the always-needed USB/connector defaults. Wire `parts_block`
   through this filter.
2. Keep `list_parts` (the tool) unfiltered — the model may still call it for
  an unexpected part; only the proactive `extras.parts_block` is narrowed.
3. Add a `part_queries.jsonl`-style size assertion to
   `tests/test_stage_driver_core_defaults.py` that a trivial board's BOM
   prompt stays under a budget (e.g. extras <= 24K when architecture has <=3
   blocks).

Acceptance: `web-cost-report` per-stage BOM `wall_s` and `cost` drop on small
boards; the prompt-size budget test holds.

### R4. Collapse lookup_lcsc_id + add_part_from_lcsc into one resolve-and-bundle tool

Today resolution and fetch are two tool round-trips, and the model can resolve
(MPN->C-number) but forget to fetch, leaving nothing cached. The
`mpn_cache` (Phase B) fixes the repeat case, but a design's *first* resolve
still costs two calls and can still drop the bundle.

Steps:
1. Add a single BOM tool `resolve_and_bundle(mpn)` whose executor:
   resolves (consulting `mpn_cache` first, then the existing tier order) and,
   on a clean single LCSC hit, *also* runs the `add-part --from-lcsc
   --into home` fetch in one call, returning the exact `<name>:<symbol>` /
   `<name>:<footprint>` strings. Update `mpn_cache` either way.
2. Keep `lookup_lcsc_id` and `add_part_from_lcsc` as separate tools for the
   ambiguous (multi-candidate) path — only the clean single-hit path merges.
3. Update `BOM_TOOLS`, the `_stage_extra("bom")` prose, and `stages/bom.md` to
   point the model at the merged tool as the primary path.
4. The tool-loop convergence caps (`client.py _MAX_*`) and the per-MPN cap
   (Phase B1) still govern; this is a call-count reduction, not a guard change.

Acceptance: `part-query-report` shows `lookup_lcsc_id` + `add_part_from_lcsc`
combined counts drop vs R1 baseline for first-run designs; `mpn-cache` hits
dominate on repeat designs.

### R5. Before/after retest harness

Make the optimize-then-retest loop reproducible.

Steps:
1. Add `scripts/bom_cost_diff.py <baseline.json> <after.json>` that diffs two
   `part-query-report --json` + `web-cost-report --json` snapshots and prints:
   `lookup_lcsc_id` total / repeat-rate / mpn-cache-hit-rate / per-stage BOM
   `wall_s` + `cost`, with per-part deltas for the top churned MPNs.
2. Document the run order in `docs/`: baseline (R1) -> vendoring (R2) ->
   re-snapshot -> diff -> parts-filter (R3) -> re-snapshot -> diff -> merge
   tool (R4) -> re-snapshot -> diff.

Acceptance: one command proves each fix moved the metric it targeted; no manual
eyeballing of two report outputs.

## Ordering and risk

- R1 is the unblocker: nothing else is measurable without a real ledger. Do it
  first on the testing box.
- R2 is the highest-leverage and lowest-risk (the `evq-p7a01p` precedent proves
  vendoring kills the churn). Heavier on human time (EasyEDA fetching, manifest
  authoring, render review) than code.
- R3 and R4 are prompt/tool-shape changes that move the per-round tax; they can
  land in either order after R2. R4 touches the LLM-facing tool spec + the
  stage prose, so it needs a live eval run to confirm the model adopts it.
- Place/route CPU time is explicitly out of scope (separate effort per the
  user). The new `stage_runs.cpu_s` column now makes *that* measurable too when
  the build worker records its phases, but the build worker does not yet call
  `record_stage` for `synthesize`/`place_route`/`fab` — that wiring is a cheap
  follow-up (the `record_stage` API and the report already support build
  phases), tracked here rather than done now to keep this change scoped to the
  design stages the user asked about.

## Build-worker telemetry follow-up (minor, not blocking)

`record_stage` currently fires only from `drive_stage` (the LLM design
stages). To see place/route CPU in the same report, have
`kicraft/server/build_worker.py` (or `cli_app build`'s phase loop) call
`guard.record_stage(stage="place_route", ...)` with a `RUSAGE_CHILDREN` delta
around the FreeRouting JVM. The column and report already render build-phase
rows; only the writer is missing. This is the lever for the
separately-scoped place/route effort.