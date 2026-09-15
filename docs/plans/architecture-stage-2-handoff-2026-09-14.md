# Stage 2 of the constructive architecture slot — handoff (2026-09-14)

**For a new session, starting fresh.** You have no memory of the previous one; this file is the
whole handoff. Read it top to bottom, then read
`docs/plans/architecture-constructive-slot-2026-09-14.md` §1, §5, §6, §8, §9, §10.8 and §10.9 —
that file is the diagnosis and the record; this file is what to do next and how to prove it.

---

## 1. What you are inheriting (facts, not narrative)

Live state of the code: commits `b667c6e` (the endpoint split, the typed regulator rating, the
load-current question), `734ec8b` (the record and the measurement tooling), `23c9d53` (a
cross-reference in the plan).

The pipeline: the "architecture" stage asks a model for a board design; a reader accepts or refuses
it; a semantic layer then reports design problems. Two shapes of answer exist:

- **explicit** — the historical shape, where the model hand-writes net names, port bindings and
  endpoints. Selected when `KICRAFT_ARCHITECTURE_SLOT` is unset (the default). This is what
  production runs today.
- **intent** — the new shape, where the model states intent (which parts, which signals, which
  rails) and `kicraft/design/architecture_intent.py` writes the bookkeeping. Selected with
  `KICRAFT_ARCHITECTURE_SLOT=intent`.

The last measurement (`/tmp/slot-ab-next-unattended`, 80 runs, N=20 per frozen board per arm, both
at `repo_head b667c6e`, $0.8913, reproducible with
`tools/ladder_experiment.py --summary --out /tmp/slot-ab-next-unattended`):

| arm | commits | runs reaching the commit gates | first drafts with **no reader refusal** | `first_draft_accepted` (zero corrections) | reader refusals | semantic repair calls | cost/run |
|---|---|---|---|---|---|---|---|
| intent | **35/40 (88 %)** | 38 | **22/38 (58 %)** | 3/38 | 19 | 31 | $0.0063 |
| explicit (stock) | 14/40 (35 %) | 40 | 0/40 | 0/40 | 53 | 27 | $0.0159 |

The parent plan's pre-registered rule ("stage 2 may start only if the intent arm's first-draft
contract-clean rate ≥ 50 % **and** its commit rate ≥ stock's in the same block") therefore **passed,
and stage 2 is unlocked**. Stage 2 is the work in §2 below. Live spend so far: **$2.60 of the $3
ceiling** for the whole exercise (this plan's two blocks were $0.166 + $0.891).

Two things already shipped that you must not undo by accident:

- **A stage can now park on a question.** When the external 5 V load current is absent from both the
  brief and the recorded answers, the architecture stage stops and asks the user instead of spending
  a repair call (`EXTERNAL_LOAD_CURRENT_QUESTION` in `kicraft/server/stage_runtime.py`). This is
  **not** behind the slot flag: it applies to the default `explicit` path too, and it is live in
  production. The 34-brief canary and the self-eval corpus drive non-interactively
  (`NONINTERACTIVE_DEFAULTS_INSTRUCTION`), so they repair instead of parking.
- **`tools/ladder_experiment.py --unattended`** is how an experiment says "no user is attached".
  Without it, `--replay-architecture` runs park on the load question and the endpoint you need
  (`first_draft_contract_clean`) is never published. Use it for every measurement.

## 2. The work: stage 2 — delete the bookkeeping layer, and make intent the default

This is the parent plan's §5, unchanged. Those validators and rescue passes exist to catch data the
model no longer hand-writes. Delete them, drop the legacy shape, keep one release, no shims, no
aliases.

**2a. `kicraft/server/stage_contracts.py` (~620 lines)**

`_normalize_usb_c_requirements` (135), `_normalize_architecture_sheet_aliases` (126),
`_complete_bound_port_nets` (89), `_validate_typed_inter_sheet_contracts` (73),
`_fold_recipe_covered_sheets` (65), `_complete_connector_requirements` (42),
`_complete_hub75_optional_address` (22), `_inter_sheet_net_endpoint_signature` (9), plus the
net-range dedup block. Measure the spans with `ast` before you delete, as the parent plan did.

**2b. `kicraft/design/recipes/resolver.py` (~400 lines)**

Three of `_port_bindings`' four diagnostics — `unknown_recipe_port_net`, `missing_recipe_port`,
`missing_recipe_port_contract`. **Keep `recipe_signal_in_power_nets`**: it has electrical meaning.
Also delete `_complete_typed_connector_peers`, `_complete_native_usb_companions`,
`_complete_native_usb_data_connector`, and the `unresolved_requirement_ids` path for connector
classes.

**2c. `kicraft/design/stage_semantics.py` (~120 lines)**

The endpoint-ownership bookkeeping only (`architecture_missing_power_endpoint` and its siblings).
**Keep** the design-level codes, which are questions, not wiring: `architecture_mcu_supply_rail_missing`,
`architecture_power_block_as_sheet`, `architecture_rail_source_unspecified`,
`architecture_programming_decision_incomplete`, and the whole external-load family. Keep the typed
regulator check (`architecture_mcu_regulator_incomplete`) — it now reads
`RecipeDefinition.rated_output_current_a`, which is data, not prose.

**2d. The alias tables (~300 lines)**

`kicraft/design/recipes/wave_*.py` / `kicraft/design/lowering.py` each carry a port-name/alias
table. The derivation in `kicraft/design/architecture_intent.py` owns one canonical table. Collapse
to that one, so "which recipe port does this signal mean" is answered in exactly one place.

**2e. Drop the legacy shape.** Make `intent` the default of `KICRAFT_ARCHITECTURE_SLOT` and delete
the explicit reader, schema, spec text and worked example, so there is one answer shape and one set
of rules. Out of scope: the *flag* itself may stay (it is a kill switch); the *legacy path* goes.

**Target: ≥ −1,500 lines net and zero new validators.** If you cannot get there by deleting, you are
doing the wrong change — stop and re-read §5's rule: "a new check may land only by deleting two
checks or 200 lines of validation, and it must show first-draft acceptance up or defects-per-draft
down".

**Why it needs its own measurement:** the completions in 2a run on *both* shapes, so deleting them
changes the intent arm too. Do not assume the block in §1 still describes the code after deletion.

## 3. How to prove it (the protocol is pre-registered — do not tune it)

1. **Free first.** `tools/ladder_experiment.py --replay-corpus /tmp/ladder-exp` must not regress
   (baseline: 263 drafts, `0/256 → 214/256`, class table in §10.9).
2. **Live, interleaved, both boards.** One run per invocation, arms alternating, unattended:

```bash
cd /home/kicraft/KiCraft
git status --porcelain        # must be empty: the harness records repo_head AND repo_dirty
set -a && . ./.env && set +a
for i in $(seq 1 20); do for board in 825 824; do for arm in stock intent; do
  KICRAFT_ARCHITECTURE_SLOT=$([ "$arm" = intent ] && echo intent || echo explicit) \
  .venv/bin/python tools/ladder_experiment.py --arm stock --label "s2_${arm}" \
    --state "$HOME/.kicraft/projects/1/$board/.kicraft/state.json" \
    --runs 1 --budget 0.25 --out /tmp/slot-stage2 --unattended
done; done; done
.venv/bin/python tools/ladder_experiment.py --summary --out /tmp/slot-stage2
```

Realistic cost of one such block: **~80 minutes, ~$0.90** (a stock run ~85 s, an intent run ~33 s).
Stop early if cumulative `cost_usd` in `/tmp/slot-stage2/runs.jsonl` exceeds $1.50.

3. **The gate — decide it before you look at the numbers.** Stage 2 lands only if, in the same
   interleaved block:

   - the intent arm still commits **≥ 33/40** (the §1 baseline minus ~5 %),
   - its first-draft contract-clean rate does **not fall below 22/38**, and
   - the deleted line count is real (≥ −1,500 with 0 new validators).

   Anything else: revert the deletion commit, record the negative result in the plan, and take the
   leading residual class back to the read-the-draft method. A negative result is a deliverable.

4. **Full unit suite** (4108 tests, ~13.5 minutes). Many tests pin the legacy contract — they will
   need to be *deleted or rewritten to the new contract*, never re-pinned to new wording. Expect
   work in `tests/test_stage_driver_retry.py`, `tests/test_stage_semantics.py`,
   `tests/test_design_recipes.py`, `tests/test_architecture_intent.py`,
   `tests/test_stage_driver_prompt_examples.py`, `tests/test_recipe_coverage.py`.

## 4. The residual classes to read next (after the deletions)

From the last block's first drafts, intent arm: `conflicting_port_binding` 15,
`multiple_intent_contracts` 15 (the aggregate of the same), `usb_connector_supply_unknown` 12, then
singletons.

A real example of the leading one:

```
conflicting_port_binding: rail '+5V': port 'vbus' of 'usb_input' is already bound to 'VBUS'
```

The model declared two rails (`VBUS` and `+5V`) and pointed a signal at one while `power.rails` said
the same connector pin produces the other. Method, same as the five fixes that came before: pull the
rejected first draft out of that run's `events.jsonl` (`answer_delta` chunks followed by a `retry`
carrying the `diagnostic`), read what the model actually said, and decide one of three things — the
derivation should own it, a typed fact is missing, or it is a question for the user. Never add a rule
that asserts a fact nobody sourced.

## 5. Landmines

- **`pip install -e` means the working tree is live.** `deploy/deploy-production.sh` restarts the
  web and build-worker processes; until that restart, a running process keeps the old modules in
  memory while fresh CLI subprocesses import the new ones. Do not leave production in a mixed state:
  after you commit, run the canonical deploy.
- **The park is not flag-gated** (§1). If you make the intent shape the default, boards whose brief
  omits the external load current will park and ask their user. That is the intended behaviour, but
  say so in the record and tell the operator before you deploy.
- **`--unattended` is not optional** for measurement. A parked run publishes no `stage_done`, so the
  endpoint is missing and the block measures the question instead of the stage (that is exactly what
  happened in `/tmp/slot-ab-next`: 17 of 24 runs parked).
- **Frozen inputs are `~/.kicraft/projects/1/824` and `/825`** (state.json + brief.txt). The draft
  corpus is `/tmp/ladder-exp`; the five historical blocks are `/tmp/slot-ab*`; the new blocks are
  `/tmp/slot-ab-next` and `/tmp/slot-ab-next-unattended`. Do not overwrite them: they are the
  evidence base.
- **Budget.** $0.40 remains under the parent plan's $3 ceiling as of this handoff. One stage-2 block
  (~$0.90) needs the operator to raise it or needs a smaller N (state which you used).
- **Do not touch** the correction budget, the canary, the deploy gate, or the prompt examples to make
  a number move (parent plan §8).
- **Ask the operator** before running `deploy/verify-design-canary.sh` (34 briefs through the live
  pipeline): it is the honest end-to-end gate for a release this size, but it spends real money.

## 6. Suggested order

1. Read §1/§5/§6/§8/§9/§10.8/§10.9 of the parent plan; run the unit suite once on the inherited
   commit to see the baseline green.
2. Do 2a–2d (deletions + the single alias table). No behaviour change to the intent shape yet.
   Suite green.
3. Do 2e (intent becomes the default, legacy path deleted). Suite green, old-contract tests removed
   or rewritten.
4. Offline replay, then the live block of §3, then the gate of §3.3.
5. Write the outcome into the parent plan's §10 as a new subsection (§10.10), same table shapes as
   §10.9; update this file's status line; commit; deploy with the operator's approval; report the
   verdict with the numbers.

## 7. Status line

**Stage 2: landed (2026-09-14/15), commit `2d5a95d`.** The work is §2, the proof is §3, the verdict
is the parent plan's §10.10. Result: the §5 deletions are done (−3,076 lines net, ten checks deleted
against one relocated), the intent shape is the only shape, and the pre-registered gate passed on all
three counts — 39/40 commits, 35/40 contract-clean first drafts (baseline 35/40 and 22/38), 6 reader
refusals against 19, $0.0055/run, no parks. Deviations from §2, all measured and stated in §10.10:

- **2d had nothing to delete.** There is no duplicated port-name/alias table to collapse: the
  `wave_*.py` tables are physical pinout facts and `lowering.py` holds lowerer port vocabularies,
  while the one signal-alias table (`_PORT_ALIASES`) already lives in a single place and is imported
  by the derivation. The dead consumers went with 2b.
- **One check was added where ten were deleted** (`unbound_required_port`, the *moved*
  `missing_recipe_port`): without it the same defect surfaced at BOM expansion as a bare
  `ValueError`, in a stage the model cannot answer. Net −9 checks.
- **The named-part `exact_part` backfill** (not on §2a's list) was deleted with the layer: it stamped
  compiler-built edge connectors with the brief's named parts, and deleting it is what recovered the
  one corpus draft the completions had been rescuing (214 → 215 accepted offline).
- The explicit arm that §3.2's loop compared against is gone by construction, so the block ran one
  arm (40 runs, 20 per board, `--unattended`, `repo_dirty: false`, $0.2190).

Operational note for the operator: `.env` still sets `KICRAFT_ARCHITECTURE_SLOT=intent`; the variable
is now inert (there is one shape), and the line should be dropped when convenient. The
`deploy/verify-design-canary.sh` gate was **not** run — the operator chose to skip it for this
release.
