# Design completion follow-up — plan (for the next session)

**Purpose.** Continue the work of `docs/plans/live-design-completion-2026-09-17-plan.md` (§12 is the
record of what that session changed and measured). That session removed the *authoring* burden where
the compiler already knew the answer — supply and reference bindings, the obligation list, part-class
vocabulary — and the completion rate quadrupled, but no brief yet commits three times out of three.
This plan is the next four moves, ranked by what the last measurement says they buy, each with the
evidence that motivates it, the exact code it touches, and an acceptance test.

**How to use this document.** §0 is the starting state (read it first, it is the whole context you
need from the previous session). §1 lists the things that are *measured* not to work — do not repeat
them. §2–§5 are the moves. §6 is how to measure. §7–§9 are the environment, the operator's open
questions, and what "done" means. §10 is what is already delivered.

---

## 0. Where the pipeline is

`logs/self_eval/canary_20260917T043323Z` — 34 briefs × 3 repeats, design-only, **$1.86 and 45 min**
(`summary.json`: `total_cost_usd` 1.862, `wall_s` 2 708 — of which `failed_run_cost_usd` is 1.79, i.e.
96 % of the spend bought the runs that failed, and `cost_per_committed_design_usd` is 0.31; the
architecture stage alone is 1.36). Compare with `canary_20260917T004307Z` + `canary_20260917T011637Z`
(runs 3+5 of the previous plan: the same 34 briefs, one repeat each, 68 runs).

| | runs 3+5 (68 runs) | last campaign (102 runs) |
|---|---|---|
| runs committing all five stages | 1 (1.5 %) | 6 (5.9 %) |
| briefs committing in ≥1 repeat | 1/34 | 4/34 |
| briefs committing in ≥2 of 3 repeats | 0 | 2 (`rc-lowpass-bnc`, `esp32-dual-motor`) |
| first failing stage | architecture 41, BOM 25, wiring 1 | architecture 53, BOM 42, functional_spec 1 |

Per-brief failure split in that campaign — this is the map of the remaining work:

- **9 briefs fail only in the architecture stage** (the wiring form):
  `audio-jack-buffer`, `chamfered-badge`, `gpio-expander`, `proto-shield`, `rounded-c3-devboard`,
  `servo-driver-16`, `snowman-ornament`, `star-ornament`, `stepper-a4988`.
- **6 briefs fail only at part resolution in the BOM stage**: `buck-3a`, `esp32-s3-sensor`,
  `highside-switch-10a`, `rp2040-min`, `usb-c-full-breakout`, `speaker-crossover`.
- **15 briefs fail on both** (`can-node`, `daq-8ch`, `dual-rail-supply`, `encoder-oled-panel`,
  `hex-env-sensor`, `led-cc-driver`, `lora-node`, `nrf52-beacon`, `relay-quad`, `round-led-ring`,
  `rs485-terminal`, `stm32-min`, `thermocouple-amp`, `usb-a-power-splitter`, `usb-pd-trigger`).
- **4 briefs commit at least once**: `rc-lowpass-bnc` (2/3), `esp32-dual-motor` (2/3), `r2r-dac`
  (1/3), `fpc-breakout` (1/3). Treat these four as the regression set: they must still commit after
  every change below.

Failure codes per run in that campaign (occurrences ÷ 102 runs), grouped:
model-authored wiring (`conflicting_port_binding` 0.235, `unsupported_lowerer_contract` 0.225,
`declared_signal_port_tied` 0.059, …) ≈ 0.62; BOM/unit (`physical-obligation-unfulfilled` 0.294,
`declared-interface-unrealized` 0.118, `COMMIT_GATE:9.42` 0.069, `missing-requirement-implementation`
0.029) ≈ 0.52; everything else ≈ 0.30. The obligation and supply/reference classes are gone
(`ARCH_SCHEMA` 0.147 → 0.010, `unsupported_supply_port` 0.118 → 0.000).

---

## 1. Non-goals and measured dead ends

Do **not**:

- Add refusal wording or class aliases one brief at a time. Six individually-correct fixes of that
  shape were committed before the 2026-09-17 session and moved the aggregate not at all; the last
  campaign's own code table shows the same two model-authored classes dominating after every alias
  was added.
- Weaken, disable or bypass a check to make a brief commit. Fix either what the model must author or
  what the compiler can decide for it.
- Special-case benchmark slugs, brief wording, or add permissive matching (`startswith`, "close
  enough" comparisons, `try/except` around a refusal).
- Move the deterministic reference corpus off **31/34**: `--reference-replay` must keep exiting 0
  with `buck-3a` refusing (its recorded `specification_conflict`), and `usb-pd-trigger` /
  `speaker-crossover` blocked. `--references` reports exactly those two blocks — that is its
  pre-session state, not a regression.
- Judge a change from one run: under 3 repeats the run-to-run spread is the same size as a fix. Use
  rates over a repeated campaign, never a single headline.
- Run a routing campaign or a heavy full-campaign build alongside the live services: 2 vCPU, and the
  site is production.
- Edit `~/.kicraft` except through documented tooling. The nightly `jlcparts-update` timer truncates
  the part catalogue when upstream's split volumes are missing — that is now guarded (§10), keep it.
- Declare success from `/tmp` artefacts: the campaign directories under `logs/self_eval/` are the
  evidence.

---

## 2. Phase A — split the architecture answer into small steps (do this first)

### Why

The architecture stage is one shape for about thirty rules. 53 of 102 attempts died there, each with
a *different* code — the failures are interactions, not one missing rule: a repair that fixes the
contact list breaks the net binding, the next attempt fixes that and breaks something else. The
stage already has 3 attempts (`_STAGE_MIN_RETRIES` in `kicraft/server/stage_runtime.py`) and gets
16 384 output tokens; more attempts or more tokens will not help, because the *answer* is one large
document.

**The fix is to ask for four small answers instead of one large one**, each with its own validation
and its own repair, then assemble them. The five committed stages, the state keys, the acceptance
gate and the canary contract do **not** change: this is a prompting/normalization strategy inside the
architecture stage.

### Design constraint that makes this safe

**Accept incremental answers, do not require them.** A provider answer that already declares
`signals` and carries no canonical net list is a complete architecture intent — that is exactly what
`_intent_shaped(payload)` in `kicraft/server/stage_contracts.py` already detects — and it must keep
being accepted in a single call, as today. Reasons:

- the reference-corpus replay (`kicraft/loadtest/mockllm.py`, `transcript_from_reference_row`) answers
  the architecture stage with the row's stored *whole* payload, so a section-based protocol must
  accept a complete answer in call 1 and stop — otherwise the corpus gate (and the mock transcript
  mechanism itself) breaks;
- `tests/test_architecture_intent.py::test_intent_slot_commits_on_the_first_draft_through_the_real_driver`
  asserts `contract_name == "kicraft_architecture_response_v2"`;
- a provider that answers everything at once should not be punished for it.

So: the stage becomes *up to four calls*, and the number actually made is decided by what the first
answer contains. Start with the whole-slot call; if the answer is missing a section, ask for that
section alone with a small schema and merge. That also means the first sub-step call carries the
existing contract name, and the section calls get new names (`kicraft_architecture_sheets_response_v1`,
`..._parts_...`, `..._signals_...`, `..._obligations_...`).

### The four answers

Order matters; later steps reference earlier ids.

| step | answer | fields | notes |
|---|---|---|---|
| A1 | board shape | `sheets[]` (name, stem, role, function, optional library/replication fields), `power.rails` (name → voltage + generating `<req>.<port>`), `topologies`, `comms_protocols`, `mcu_present`, `assumptions` | rails must exist before parts and signals reference them |
| A2 | one part per block | `requirements[]` (id, sheet, role, family, exact_part, parameters, supply, programming, functional_blocks) | no ports, no ties, no obligations — the compiler derives or the schema drops them |
| A3 | cross-part wiring | `signals[]` (name, from, to, rails, optional range) | the only place nets are named |
| A4 | requirement ownership | `obligations[]` as `<original_obligation_id> → <requirement_id>` pairs | replaces the "copy the row verbatim onto the requirement" step; the compiler builds the rows from the committed set (it already writes the top-level list) |

Assembly: `{**A1, "requirements": A2.requirements, "signals": A3.signals, ...}` → the existing
`ArchitectureIntent` payload → `derive_architecture` (unchanged) → the existing commit path. A4 is
only needed when `prompt_state` carries obligations at all.

### Where to implement it

- `kicraft/server/stage_runtime.py` — `drive_stage`: give the architecture stage a small sub-step
  plan the way the BOM/wiring stages already drive several provider calls per stage
  (`_drive_work_unit_stage` at ~line 1912 plans units with `plan_stage_work_units`, drives each unit
  as its own interaction with its own validation, and "normalizes and commits once"). Reuse the same
  shape: call 1 = the whole slot (existing `build_stage_response_contract`), then a section call per
  missing piece, each with the stage's feedback machinery (`_lean_retry`, `_retry_feedback`,
  `finalize_stage`) and a bounded per-section retry count (2), with the stage's existing budget as the
  overall cap.
- `kicraft/server/stage_contracts.py` — `build_stage_response_contract` / `_response_schema` /
  `_slot_response_schema`: add a `sub_step: str | None` parameter that selects a small pydantic
  model for the section and names the contract; `_normalize_stage_response` gains the merge (assemble
  the intent payload from whatever the provider returned) before `_derive_intent_payload`.
- `kicraft/server/stage_prompts.py` — `_spec_text` / `_stage_extra` / `_worked_example`: one short
  spec and one worked example per section. The specs live next to the stage spec, e.g.
  `.agents/skills/kicraft/stages/architecture/sheets.md`, `parts.md`, `signals.md`,
  `obligations.md`, with the existing `architecture.md` as the overview and the whole-slot answer.
- `kicraft/server/stage_driver.py` (`kicraft-stage-debug`) — `debug-draft --stage architecture` must
  drive the same sub-step sequence so a single brief can be stepped through without a campaign.
- Tests: `tests/test_architecture_intent.py` (whole-slot still accepted; a section-only answer
  merges), `tests/test_stage_driver_retry.py` (a section failure repairs that section only),
  `tests/test_stage_driver_replay.py` and the mock transcript for the replay path.

### Acceptance

- Reference replay: still 31/34, exit 0.
- Unit tests: `tests/test_architecture_intent.py tests/test_stage_driver_retry.py
  tests/test_stage_driver_replay.py tests/test_stage_work_units.py` green.
- Bounded campaign, 3 repeats, `KICRAFT_CANARY_REPEATS=3 ./deploy/verify-design-canary.sh <slugs>` on
  the 9 architecture-only briefs **plus** the 4 regression briefs:
  - architecture-first-failure occurrences per run at least 40 % below 0.52, and
  - at least 5 of the 9 architecture-only briefs commit in ≥2 of 3 repeats (they are 0/3 today),
    while the 4 regression briefs keep committing at least as often.
- Cost per design ≤ 1.6× the last campaign's (~$0.018 per design-only brief-run; the stage breakdown
  in `summary.json` says the architecture stage was 73 % of it, so a smaller architecture answer
  should also be a cheaper one).

---

## 3. Phase B — the compiler owns the part identity for families we own

### Why

BOM is the other half: 6 briefs fail *only* here and 15 more fail here as well. The dominant code is
`physical-obligation-unfulfilled` (`requires 1 real usb-c-receptacle, found 0`, 0.294/run). The
demanded class is usually one the reviewed vocabulary **has** — what fails is the identity of the
group the draft emitted. Two facts about the current code explain it:

- `deterministic_bom_candidate(unit, prompt_state)` in `kicraft/server/stage_work_units.py` computes
  the compiler-owned groups for a unit (a typed lowerer artifact, or the standard connector sheet) —
  but `_validate_bom_unit` only consults it **when the model emitted no groups** (`if not groups:`).
  So for a family-backed requirement the model's own group wins, and if its `(mpn, symbol,
  footprint)` is a KiCad-stock pair nobody reviewed, every obligation check reads "found 0".
- `_normalize_curated_group_identities` can repair an identity only when the group carries a
  resolvable MPN or a bundle library name; `physical_inventory_record` classifies only an exact
  reviewed pair.

### Steps

1. **Always compute the deterministic candidate for a family-backed unit**, not only when the draft
   is empty. The requirement is family-backed when `lower_requirement(requirement)` returns an
   artifact, or the requirement appears in `architecture.recipe_selections` (expand with
   `kicraft.design.recipes.registry.expand_selections`).
2. **Precedence when the draft also emitted groups**:
   - the model named an *explicit different reviewed part* → refuse, naming both (that is an
     unapproved substitution and the existing `model_authored_protected_identity` path already
     covers part of it);
   - the model's group is *unreviewed* while the compiler has a reviewed/deterministic identity →
     **replace it with the compiler's, record a derived note** ("<group>: identity supplied by the
     <family> recipe; the draft's <symbol>/<mpn> is not a reviewed part"). This is the class that
     produces "found 0" today.
   - the model's group matches → nothing changes.
3. **Extend coverage where the compiler has no artifact but a reviewed bundle exists.** Build a
   family → reviewed-part default table from the curated bundles
   (`kicraft/parts_library/loader.load_all_with_overrides`, the same index
   `_curated_part_indexes` already builds) and
   `kicraft/design/part_identity.REVIEWED_PARTS`, and let `deterministic_bom_candidate` answer from
   it for families with exactly one reviewed realisation (e.g. a family whose bundle manifest names
   one MPN). Where a family has several candidates, keep the model's choice but require it to be a
   reviewed one and name the candidates in the refusal.
4. **Use the improvement already in the tree**: the obligation defect now names the groups the unit
   emitted; the substitution note/replacement above must state the reviewed identity it used, so the
   next diagnosis needs no further instrumentation.

### Acceptance

- The 6 BOM-only briefs commit in ≥2 of 3 repeats; `physical-obligation-unfulfilled` and
  `declared-interface-unrealized` occurrences per run at least halve (0.294 → ≤0.15 and 0.118 →
  ≤0.06).
- The 4 regression briefs still commit; reference replay still 31/34.
- `tests/test_stage_work_units.py` (it already covers the alias map, the deterministic candidate and
  the identity paths) plus `tests/test_bom_*.py` green, with a new test per precedence rule above —
  the replacement is a compiler-owned fact and must be asserted as such (a group whose symbol is
  stock becomes the reviewed bundle's, with the note in `assumptions`).

---

## 4. Phase C — make "fab-ready" mean something (do this before promising 34/34)

### Why

Everything measured so far is design-only: five stages commit, then the pipeline stops. A committed
design is not a routed, DRC-clean, orderable board, and the release gate you actually care about runs
the build tail. Until the build runs on the briefs that now commit, "34/34" would mean "34 forms
accepted", not "34 boards".

### Steps

1. Take the briefs that commit in ≥2 of 3 repeats (currently `rc-lowpass-bnc`, `esp32-dual-motor`,
   plus whatever Phases A and B promote) and run them **through the full pipeline**:
   `.venv/bin/python -m kicraft.eval.self_eval --only <slugs> --repeats 1 --out logs/self_eval/full_<ts>`
   (omit `--design-only`), then
   `.venv/bin/python -m kicraft.eval.design_acceptance logs/self_eval/full_<ts> --mode full --only <slugs>`.
   Start with two briefs and `--build-slots 1`: the box has 2 vCPU and the build is the heavy tail.
2. Read `verify_fulfillment_campaign`'s obligations per brief — the acceptance ids are
   `<slug>.pin-footprint-mapping`, `.complete-required-connections`, `.exported-artifacts`, `.erc`,
   `.drc`, `.programming-when-applicable`, `.geometry-when-applicable` (see the fixtures in
   `tests/fixtures/reference_inputs/`). Every failure is a concrete, named board defect.
3. Record the failing obligation ids per brief in this plan's §0 replacement (or in the handoff), and
   fix them in order of how many briefs they block — this is the *real* fab-ready backlog, and it is
   the first time the pipeline is judged on the artefact rather than on the form.

### Acceptance

- At least one brief passes full artifact acceptance end to end (all obligations `pass`, except the
  ones its contract explicitly defers).
- The remaining failures are recorded per brief as named obligations, not prose.

---

## 5. Phase D — obligations that are not parts

### Why

Five briefs cannot win under the current schema: the user's brief mentions a *fabrication* feature
(printed copper area as a heatsink, thermal vias) or a *negative* constraint ("no microcontroller"),
and the intent stage turns it into a `physical` obligation with a `component_class`. No BOM line can
satisfy it, so `_requirement_obligation_defects` and `check_requirement_physical_realization` refuse
forever. Measured classes: `copper-heatsink-area`, `heatsink-copper-area`, `thermal-via-copper-pour`,
`no-microcontroller` (briefs `led-cc-driver`, `star-ornament`, `buck-3a`, `thermocouple-amp`).

### Steps

1. Add two obligation kinds to the union in `kicraft/design/models.py`
   (`RequirementObligation = Annotated[PhysicalObligation | QuantityObligation | … | Field(discriminator="kind")]`):
   - `fabrication` — `feature` (a canonical lowercase kebab-case fabrication feature, e.g.
     `copper-area`, `thermal-via-field`) plus the quantity/limit the brief states;
   - `negative` — `absent_class` (the canonical class that must **not** appear).
2. Teach the intent stage to use them: the intent spec (`.agents/skills/kicraft/stages/intent.md`) and
   `_stage_extra("intent")` in `kicraft/server/stage_prompts.py` currently say "capture every explicit
   physical component class"; add the rule that a board feature or an absence is never a
   `physical` obligation.
3. Make the consumers behave:
   - `kicraft/server/stage_work_units.py::_requirement_obligation_defects` — skip non-`physical`
     kinds (it already only looks at `physical`; confirm nothing else assumes the union);
   - `kicraft/design/synthesis/validation.py::check_requirement_physical_realization` — same;
   - the architecture stage's ownership rule — a `fabrication`/`negative` row is not implemented by a
     requirement and must be exempt from ownership the way `quantity` is
     (`kicraft/design/architecture_intent.py::_obligations_are_owned_once`, `models.Architecture`'s
     counterpart, and `validate_obligation_retention` in `kicraft/server/stage_contracts.py`);
   - the PCB-side checks stay the owners of a `fabrication` claim; if no check can verify it today,
     record it as **unverified evidence**, never as a BOM defect.
4. Retention is by `(kind, original_obligation_id)`, so the new kinds flow through
   `restore_source_obligations` unchanged — assert that with a test rather than assuming it.

### Acceptance

- The four briefs above stop failing on those classes (their remaining failures are the ordinary
  wiring/part ones), and the new kinds survive intent → functional_spec → architecture verbatim.
- A test asserts a `fabrication` obligation commits through the architecture stage without an owner
  and is not counted as a BOM component demand.

---

## 6. Phase E — how to measure, every time

- **Bounded set while iterating** (slug arguments are positional and go to both the runner and the
  acceptance check):

  ```bash
  KICRAFT_CANARY_REPEATS=3 ./deploy/verify-design-canary.sh \
    audio-jack-buffer proto-shield servo-driver-16 stepper-a4988 gpio-expander \
    rc-lowpass-bnc esp32-dual-motor r2r-dac fpc-breakout        # 9 + regression
  ```

  ≈ **$0.70 and ~17 min** (scaled from the last campaign: $1.86 / 102 runs ≈ $0.018 per design-only
  brief-run; a failed run is not cheaper than a successful one — it burns retries, and the
  architecture stage is 73 % of the spend). Use the whole 34 with 3 repeats (**$1.86, ~45 min**) when
  a bounded result is positive and you want the headline.
- **Read the right things**: `summary.json` (`design_committed`, `stage_outcomes`, `brief_median_mean`,
  `cost`) and the per-run `events.jsonl` for codes. A `retry` event carries the commit gate's
  `commit_gate_codes`/`offenders`; the terminal `stage_done` event under `--lean-events` does **not**
  — read the retry event when a run reports `OTHER:commit_rejected` with no detail.
- **Compare rates**, not headlines: occurrences per participating run, and briefs committing in ≥2 of
  3 repeats. `logs/self_eval/canary_20260917T043323Z` and runs 3+5 are the baselines in §0.
- **Deterministic gates before every campaign**: the reference replay (§1) and the test list in §7.
  They are cheap and they catch the regressions a campaign is too noisy to see.
- **One heavy job at a time.** The replay takes ~18 min, a 34×3 design campaign ~45 min, a full
  (build) campaign much longer; they are all CPU-bound on 2 vCPU.
- **Write the result down**: update §0 of this plan (or append a §11) with the campaign directory,
  the table, and what it means, and update `docs/plans/self-eval-2026-09-16-remediation-handoff.md`
  §0's canary table. A session that changes code without recording the measurement repeats the
  previous session's problem.

---

## 7. Environment, guardrails, commands

This checkout is the production box (`kicraft-web`); see `AGENTS.md`. Web on `127.0.0.1:8080` behind
Caddy, services run detached and are restarted only by `deploy/restart-web.sh` /
`deploy/restart-build-worker.sh` / `deploy/deploy-production.sh` (the systemd units are inactive).

```bash
# deterministic corpus (no provider spend) — must stay 31/34, exit 0
.venv/bin/python -m kicraft.eval.design_acceptance --reference-replay

# the fixtures' contract shape — reports only the two recorded blocks
.venv/bin/python -m kicraft.eval.design_acceptance --references

# the stage/architecture/recipe/work-unit tests
.venv/bin/python -m pytest tests/test_architecture_intent.py tests/test_stage_work_units.py \
  tests/test_design_lowering.py tests/test_stage_driver_retry.py tests/test_part_identity.py -q

# every module that imports a changed file (60 files, ~5 min)
grep -ln "stage_contracts\|architecture_intent\|design.lowering\|stage_work_units\|design.models\|part_identity\|jlcparts" tests/*.py \
  | xargs .venv/bin/python -m pytest -q

# one brief, stage by stage, with a review before each commit
kicraft-stage-debug debug-draft --workspace <dir> --stage architecture --brief-file <brief.txt> --budget 0.25
kicraft-stage-debug run --workspace <dir> --brief-file <brief.txt>      # all five stages, like the site

# spend: ~$0.018 per design-only brief-run ($0.31 per *committed* design in the last campaign,
# because 96 % of the spend went to the runs that failed); caps in .env (project 0.60/run, daily 20, total 250)
.venv/bin/python -c "from kicraft.server.config import Settings; from kicraft.server.spend_guard import SpendGuard; print(SpendGuard(Settings.from_env()).status())"
```

Deploy (only with the operator's go-ahead, and never mid-campaign):

```bash
git push origin <branch>
.venv/bin/pip install -e ".[server,design]"
./deploy/deploy-production.sh          # restarts both services, verifies HTTP 200 + worker ready
```

The live design canary is deliberately **not** part of the deploy path.

---

## 8. Open operator decisions

1. **Approve Phase A's protocol change.** It multiplies the architecture stage's provider calls
   (up to 4 + repairs) for ~1.6× cost per design and a slightly longer run. Nothing else in this plan
   touches the committed contract.
2. **Approve compiler-owned part selection (Phase B).** For families we have reviewed, the board's
   exact part becomes a compiler decision. The alternative — keep the model's choice and require it
   to be a reviewed part — keeps more model freedom but leaves the biggest failure class in place.
3. **Sourcing for the remaining classes** (Phase B step 3 and §5): which orderable parts to
   standardise for a `r-2r-resistor-ladder`, a bare `selector-switch`, an optocoupler variant set,
   and the LoRa module. This is a purchasing decision, not an implementation one; until it is made,
   those classes are documented gaps, not bugs.
4. **Does "fab-ready" acceptance include the build for all 34 briefs?** Phase C starts with the
   briefs that commit; extending it to all 34 is a large daily cost and a long wall-clock tail on
   this box.
5. **Deploy cadence.** Deploying makes live failures legible but does not itself raise completion;
   decide whether to deploy each phase's result or hold until a phase shows a measurable gain.

---

## 9. Success criteria

- **Phase A:** architecture-first-failure rate per run ≥40 % below 0.52 on the bounded repeated set,
  and ≥5 of the 9 architecture-only briefs committing in ≥2 of 3 repeats.
- **Phase B:** the 6 BOM-only briefs committing in ≥2 of 3 repeats, and the two BOM obligation codes
  at least halved per run.
- **Phase C:** at least one brief passing full artifact acceptance (ERC, DRC, exported artifacts,
  connectivity, geometry) end to end, with the remaining failures named per obligation.
- **Phase D:** the fabricating/negative briefs no longer blocked by a component demand that cannot
  exist, with the new kinds retained verbatim across stages.
- **Phase E:** every change judged by a repeated bounded campaign and written down; the reference
  corpus and the test list green before each campaign; the 4 regression briefs never regress.
- **The real gate (unchanged, and not yet close):** `--release` requires **three fresh full
  campaigns** with artifact-verified fulfillment — 3 × 34 briefs, all five stages *and* the build
  tail. Today's ceiling is 4/34 briefs design-committing in at least one of three repeats, so treat
  "34/34 fab-ready" as a programme, not the next session's deliverable.

---

## 10. Delivered inventory (do not redo)

Branch `simplify/bom-wiring-pipeline`, pushed; commit `3689ace` (implementation) and `58c06d3`
(plan bookkeeping), deployed on 2026-09-17 12:22 UTC (`deploy/deploy-production.sh`, web HTTP 200,
worker `ready`).

| area | what is already in the tree |
|---|---|
| derivation | `supply`/reference/required-port bindings derived (`kicraft/design/architecture_intent.py`); the family's reference port binds to GND; lowerer published contacts completed; unpublished authored pins fall back; one physical source port may carry one net under two names; stacking pin maps come from the template |
| obligations | the top level is written from the committed set (`restore_source_obligations`), ownership-only checking, `quantity` may stand alone, identical rows may be shared, paraphrased copies restored, unknown slot keys named (`kicraft/server/stage_contracts.py`, `kicraft/design/models.py`) |
| part classes | one alias vocabulary used by both the work-unit check and the §9.42 gate (`kicraft/design/part_identity.py::canonical_physical_features`), 12 measured classes aliased; lowerer part/parameter contracts published (`required_exact_part`, `reviewed_exact_part`, `parameter_choices`) |
| diagnostics | a physical-obligation failure names the groups the unit emitted; a declared claim with no pin is refused by name; commit-gate codes are in the `retry` events |
| ops | the JLC catalogue truncated by the nightly timer was restored from `~/.kicraft/jlcparts/cache.sqlite3.before-20260916-refresh` (artifact kept as `...truncated-20260917`), and `jlcparts.update()` refuses to install a catalogue with fewer than half the installed rows (`KICRAFT_JLCPARTS_ALLOW_SHRINK=1` to accept one on purpose) |
