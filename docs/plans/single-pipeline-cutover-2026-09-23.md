# Single-pipeline cutover (2026-09-23)

**Decision.** One design engine: this checkout. The pinned second engine
(`~/KiCraft-legacy`, commit `bc6a2f8`, "native design") and the whole switch
around it are removed. Owner call, recorded here with its cost.

## What was removed

| path | what it was |
|---|---|
| `kicraft/server/pipeline.py` | the selection/`describe()`/marker/provenance/legacy-env module |
| `kicraft/server/legacy_session_runner.py` | the JSON-over-stdin protocol endpoint run *by* the pinned interpreter |
| `kicraft/server/native_build_runner.py` | snapshot isolation for a legacy-owned canonical state |
| `routing_config.pipeline` + the admin "Design pipeline" card | the runtime knob (`current`/`legacy`) |
| `Settings.pipeline` + `KICRAFT_PIPELINE` | the env default |
| `projects.pipeline` (code + new-schema column) | the per-row stamp; existing rows keep their historical value |
| promote provenance `pipeline*` keys, `write_promote_provenance(pipeline=…)` | per-board stamp |
| self-eval `pipeline_counts`, `native_source`, `native_overrides` | cross-arm cohort grouping |
| `/admin/routing` | renamed `/admin/design` — the page now picks the design *model* only |
| `tests/test_pipeline_switch.py`, `test_native_build_runner.py`, `test_legacy_question_policy.py` | the switch's guard tests |
| 3 legacy-only tests inside `test_self_eval.py`, `test_external_self_eval.py`, `test_web_default_project.py` | state-isolation, reconcile-count accounting, "a legacy design still reaches the build" |

Refusals that were gated on the engine (`current_tree_owns_design_state`) are now
unconditional: BOM reconcile runs for every workspace, post-wiring authorship runs
for every workspace, and every build is `kicraft.design.cli_app` on this tree.

## What it costs (accepted)

The pinned engine was the yield leader on the frozen 34-brief corpus:

| engine | requests producing a finished board |
|---|---|
| `bc6a2f8` (now removed) | 21 / 34 |
| this tree (2026-09-17 era) | 4 / 34 |
| this tree + the 09-23 fixes | not yet measured |

The fork existed to keep the 21/34 arm reachable from `/admin/routing` while the
typed engine's architecture/BOM contracts were repaired. Removing it means the
next canary measures this tree alone; the repair work (BOM work-unit retry,
`2bec9e8`, `aa3193d`, `c822c39`) is what has to close the gap. **No measurement in
this session claims the gap is closed.**

## Preserved, not deleted

The pinned checkout carried uncommitted work in its working tree. It is archived
before removal:

- `/home/kicraft/archives/KiCraft-legacy-bc6a2f8-20260923.tar.gz` (311 MB, whole tree incl. `.git`)
- `/home/kicraft/archives/KiCraft-legacy-bc6a2f8-uncommitted-20260923.patch` (349-line working diff)
- the pinned commit itself is in this repo's history: `git worktree add <path> bc6a2f8`

The archived diff contains three items with no equivalent here (candidates, not
ported — each needs the repo's N-of-3 measurement before it earns a place):

1. a **stream stall bound** in the client (an answerless SSE stream could hang a
   stage for 19 minutes),
2. `serialization_retries` 1 → 3 with a measured justification (BOM is stochastic),
3. a `mechanical` `BlockCategory` (superseded here by `2bec9e8`'s obligation rule —
   the same mounting-hole brief is handled by a different mechanism).

## Historical data

`accounts.db` keeps its `pipeline` column on rows written before this cutover
(14 rows read `legacy`). Nothing reads it any more; it is left in place rather
than dropping a column from the production database. Same for old
`<stem>.provenance.json` files and self-eval `summary.json` bundles, which keep
their `pipeline`/`pipeline_counts` keys as written.
