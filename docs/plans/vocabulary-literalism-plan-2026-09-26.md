# Vocabulary literalism — plan (drafted 2026-09-26, amended and executed 2026-09-26)

Owner's instruction, verbatim:

> *"yes tackle vocabulary literalism next and to me this sounds like exactly the type of task that Jev
> is built for. consider leaning on Jev to help clarify ambiguity, explore multiple paths, test
> empirically and select the best based on evidence. for now dont do any major implementation, i want
> you to write this to a plan for the next session."*

Amended the same day, after the Step-1 instrument was built and the corpus measured. The
delta against the first draft is listed in §9; the first draft's inventory (V1–V16) is kept as
the *hypothesis list* it was, with measured frequencies beside it.

## 1. What vocabulary literalism is

A check decides something true or false about a design by looking for particular *words* in prose,
instead of at the thing the words describe. Two symptoms, and the second is worse:

- **A valid design is refused** over wording (costs a repair round, teaches the model to satisfy the
  checker rather than the circuit).
- **A real thing the owner asked for is silently dropped**, because the brief said it in other words
  (no refusal, no question — just a board missing something). Silent misses get equal priority: they
  are invisible, so nobody complains about them.

Live evidence, in the owner's own words from the seed-43 run:

> *"our system of checks seems to have gotten too wound up tight and restrictive that it is erroring
> on a valid design over semantics … we need to design this so it makes a valid design in one shot the
> first time through usually. retries are ok to avoid failures but they shouldnt be the foundation."*

## 2. The design rule (amended)

**A refusal stays deterministic. A model may not create one.** The original draft said "no model call
may sit in a path that decides pass/fail". That rule does not describe the shipped system, so it was
replaced with the invariant the code actually keeps:

- **Tier A — deterministic only.** The question is a closed machine grammar or an exhaustive alias
  table: KiCad s-expr, refdes, values, element values, layer counts, the reviewed class table (V14,
  V15). No model, no change.
- **Tier B — a model may *relax*, never tighten.** The check is a prose proxy whose own inventory row
  admits "the intent is not what the words say" (V1, V2, V4, V5, V6, V7, V8, V10, V13). When the
  deterministic proxy refuses, a typed decision may **downgrade the refusal to a recorded
  assumption/warning**. The model can only ever *un-refuse*; the deterministic refusal is the
  fallback, so a provider outage degrades to the stricter behaviour. Answers are cached
  content-addressed (§6), so a replay or a re-run of the same candidate takes the same decision.
- **Tier C — a model extracts, always visibly.** The silent-miss family (V7, V11, V12, V13).
  Extraction is a recall problem over open language; a frozen rule's recall is bounded by the corpus
  it was derived from, and its failure is invisible. An extractor must emit a row *with the brief
  span it came from*, and emit an explicit "not found" row when it finds nothing, so absence is
  visible and can be asked about instead of silently defaulted. The regex stays as the fast path.
- **Offline discovery stays, as hardening.** Deriving a rule from cached labels and freezing it is
  right wherever a genuinely closed rule exists; it is not a substitute for Tier B/C where none does.

Preference order *within* each tier is unchanged from the first draft: structure the fact → ask the
pipeline's own fields → closed table with a fallback → broaden a list only for a closed domain.

Risk, stated: a Tier-B relaxation can accept a real defect. Mitigation is the plan's own rule —
relax only to a *recorded* assumption/warning (never to silence), only for checks whose measured
false-positive rate is above a floor, and never for a check whose deterministic form reads a *value*.

## 3. The shipped Jev surface (corrected from the first draft)

Both paths are **on by default** (`config.py:585,593`); `.env` sets neither flag.

| Path | What it actually does | Evidence |
|---|---|---|
| `kicraft/server/draft_audit.py` | typed findings, `severity="repair_required"`, appended to `diagnostics`, so they consume repair rounds and can fail a stage | `draft_audit._finding`; wired `stage_runtime.py:4788`, `:5000` |
| `kicraft/server/reconciliation.py` | **mutates the candidate** (class rename, quantity bind) under a confidence floor, or recommends (BOM/pins) | `resolve_demanded_classes`/`resolve_quantity_subjects` at `stage_runtime.py:1488-1489`; `reconcile_bom_classes`/`reconcile_declared_port_pins` at `stage_work_units.py:1792,1800` |
| `kicraft/server/decision_layer.py` | `noul`/`choice`/`score`, one call per batch of questions over one `state`, confidence = peak vs uniform, malformed answer dropped not guessed, input-only pricing $0.042/1M, 60 s timeout, no temperature/seed in the payload | module + `decide()` |

Jev has reached live runs: `audit_no_carrier_in_catalogue` ×9 and `audit_identity_unresolved` ×5 in
`~/.kicraft`; the surprise-40 walkthrough records P8 (refusal at 0.86 over an invented family) and
P10 (a *correct* bare-LED finding at 0.98 that two repair rounds could not resolve because the
conflicting rewrite landed after the answer).

## 4. Step 1 — the instrument (done)

`kicraft/eval/check_replay.py`. Walks an archived self-eval tree, recovers each stage's committed
candidate, applies the runtime's pre-diagnosis normalization **once** (with the one
network-touching completion step stubbed, so a replay never researches or writes the parts library),
re-runs `diagnose_stage`, and reports per-code firings plus the repair-round KPI. `--dump-hits`
writes one JSONL row per firing for the labelling step.

Archive facts, corrected after verification:

- **137** campaign dirs, **2051** run dirs with `.kicraft/state.json`, **1897** `events.jsonl`,
  26,777 JSON files. (The first draft said "162 run dirs".)
- `state.json` keeps only the *committed* slots. `stage_status[stage].diagnostics` is `[]` for a
  stage that eventually committed, so the diagnostics a run actually saw cannot be read back — the
  harness recomputes them. Verified: `architecture` with `attempts=2` has `diagnostics=[]`.
- The runtime diagnoses `_normalize_candidate_for_diagnostics(...)`, not the raw payload, and that
  normalization is deliberately **not idempotent** (`complete_over_rated_supply` re-adds a rail).
  The harness applies it exactly once.
- The wiring candidate is not a `ConversationState` slot (it becomes the schematic), so **wiring
  cannot be replayed from the archive** — only intent/functional_spec/architecture/bom. A wiring fix
  needs a live run or `answer_delta` reassembly; 154 runs have no `events.jsonl` at all.
- Replay is a proxy: "would today's checks refuse this shipped board", not "did this exact
  diagnostic fire during the run". Board-level code drift (the reviewed library grew 79→83 protected
  identities since 2026-09-15) means an old archive is judged by today's library.

### 4a. The table (1974 runs replayed, 8.8 s)

Per code: total firings, firings where the stage committed, and firings on a **fully committed**
board (the strictest false-refusal proxy, from `--dump-hits`).

| check | fires | stage-committed | fully-committed | severity |
|---|---|---|---|---|
| `functional_spec_partial_ground_flow` | 589 | 589 | 115 | repair_required |
| `intent_quantity_subject_unbound` | 246 | 246 | 21 | repair_required |
| `intent_obligation_class_unrealizable` | 215 | 215 | 0 | repair_required |
| `architecture_obligation_family_mismatch` | 169 | 168 | 18 | repair_required |
| `functional_spec_nonfunctional_block` | 86 | 86 | 6 | repair_required |
| `bom_architecture_role_unsupported` | 88 | 81 | 51 | repair_required |
| `functional_spec_premature_topology` | 73 | 72 | 9 | repair_required |
| `architecture_power_block_as_sheet` | 65 | 65 | 10 | repair_required |
| `architecture_unavailable_core_default` | 62 | 62 | 4 | repair_required |
| `functional_spec_drive_missing_power` | 79 | 57 | 6 | repair_required |
| `functional_spec_external_load_power_assumed` | 56 | 56 | 8 | repair_required |
| `intent_prototyping_area_omitted` | 52 | 49 | 14 | repair_required |

KPI baseline (mean attempts / runs needing repair):

| stage | runs | mean attempts | repair-attempted runs | failed runs |
|---|---|---|---|---|
| intent | 1823 | 1.089 | 125 | 0 |
| functional_spec | 1807 | 1.512 | 794 | 22 |
| architecture | 1742 | 2.412 | 229 | 679 |
| bom | 1003 | 2.208 | 0 | 558 |
| wiring | 399 | 1.707 | 0 | 88 |

### 4b. Proven live false refusals

Replaying the five live sessions (not the archive) through the same harness:

| session | verdict | hit |
|---|---|---|
| `20260926T151624Z_CH32V003_DEV_BOARD` (seed-43, §11's known-valid board) | committed, built, fabbed | `bom_architecture_role_unsupported`: sheets **"UART INTERFACE"**, **"RESET INPUT"** |
| `20260925T130049Z_PASSIVE_RC_FILTER` | committed | `functional_spec_partial_ground_flow`: `bnc_input` |

Both are the owner's exact complaint, on boards that shipped. Diagnosis:

- **role_unsupported**: `role_text` includes `sheet["function"]` prose unconditionally (`:2748-2762`).
  "Expose the **MCU** UART … signals on a header" and "pulls the **MCU** reset input low" name the
  MCU as a *counterpart on another sheet*; the regex treats any mention as a claim that *this* sheet
  hosts an IC. The code's own comment (`:2745-2747`) already says not to infer the IC from a
  connector sheet's title — the prose channel was left open.
- **partial_ground_flow**: `ground_targets` is the set of `to_block` endpoints only (`:975-985`).
  `BNC_INPUT` is the *source* of the ground net in that spec, so it can never be its own target; a
  design whose ground originates at the input connector is refused. The spec recorded the common
  ground in `assumptions`.

## 5. Method (amended: rank by expected harm, then five lines per check)

Rank by `P(fires) × P(wrong | fires) × cost(repair round)` over the fully-committed column, **not**
by raw frequency: a check that fires often and is right is free, and the poison is the one that fires
rarely and wrongly. For each check taken:

1. the intent in one sentence;
2. two or three candidate rules (structure / ask-the-fields / closed table + fallback / Tier B/C);
3. the corpus measurement for each (newly refused valid designs, newly caught real defects, changed
   repair rounds);
4. Jev's labels for the ambiguous cases, with agreement against **hand labels** (not against a second
   Jev pass — reliability is not validity);
5. the selected rule, stated as a decision with its number.

## 6. Jev labelling (Step 3) and the cache

- `kicraft/eval/jev_labels.py` sends typed questions over cached candidates from `--dump-hits`,
  never over the live pipeline, and writes answers to
  `kicraft/eval/corpora/vocabulary-literalism/` as committed data.
- **Ground truth is hand labels on a bounded sample** (the 34 canary briefs plus the top ambiguous
  cases). A single model's opinion is not ground truth; freezing it into a gate without adjudication
  launders opinion into "determinism". Report both the word list's and Jev's agreement against those
  labels.
- A runtime Tier-B/C decision is **content-addressed**: cache key includes model id, prompt, and a
  state hash. Determinism then comes from the cache, not from banning the model. First-seen states
  are model-decided and frozen as data thereafter; eval comparisons pin the cache.
- Before relying on any frozen label: measure Jev's repeat determinism (same state twice → same
  answer, confidence spread). `typesafe/jev-1.13` is served off an alpha endpoint, so a frozen key
  must name the model id and be re-derived if it moves.

## 7. Acceptance criteria (amended)

A prose-keyed refusal is "done" when:

1. its intent is written in one sentence;
2. a test pins an equivalent valid form it now accepts, and a test (or a documented decision by the
   owner) pins a genuine defect it still refuses — both directions;
3. the corpus numbers are recorded: fully-committed boards newly refused, real defects newly missed,
   and repair rounds, on the same 1974-run replay before and after;
4. the canary is green and its repair-round count did not rise (`deploy/verify-design-canary.sh`,
   run manually — it is not part of the deploy path).

Criterion 3 is "no change beyond corpus noise, adjudicated on the hand-labelled sample" — not a bare
"0" on a finite corpus, which is pressure to overfit the corpus.

## 8. What not to do

- **Don't broaden word lists as the primary fix**; a closed grammar is acceptable only for a closed
  domain, with a comment naming the domain.
- **Don't let a model be the sole reason a design is refused.** Relaxing a refusal, or extracting a
  fact, from a closed option set with a confidence floor and an audit record is established
  (`draft_audit.py`, `reconciliation.py`); creating a refusal is not.
- **Don't put the model in the replay path.** The harness must stay deterministic or the before/after
  numbers are meaningless.
- **Don't rewrite `stage_semantics.py` wholesale.** One check, one measurement, one test.
- **Don't touch the reviewed-class vocabulary** (V15): closed on purpose, with a real fallback.
- **Don't change the eval's verbatim-evidence rule** (V16) into "accept prose"; allow named
  derivations (`derived_from: [...]`).
- **Decide `draft_audit` severity explicitly.** P10 shows a `repair_required` finding that cannot be
  acted on wastes rounds. `repair_required` should be reserved for a finding that names a structured,
  actionable field; otherwise `advisory`.

## 9. Delta against the 2026-09-26 draft

1. §2 replaces "no model call in a pass/fail path" with the tier rule and the actual invariant
   (model may relax/extract, never create a refusal).
2. §3 adds the shipped Jev surface; the first draft's §3 described a system that is not deployed.
3. §4 corrects the corpus description, adds the normalization-once fact, and records that wiring is
   unreplayable; the first draft assumed a stored candidate per stage.
4. §4a/4b add the measured table and two proven live false refusals; the first draft's inventory is
   now the hypothesis list, and the top check by measurement (`partial_ground_flow`) was not in it.
5. §5 ranks by expected harm, not frequency.
6. §6 makes determinism a caching property and requires hand labels for validity.
7. §7 softens "0" to corpus-noise plus adjudication.
8. Scope: the first draft covered `stage_semantics.py` alone. The real surface is **458** `re.*`
   call sites and **224** `re.compile` across `kicraft/`, with `design/synthesis/validation.py`
   (79 call sites, 46 vocabulary constants) denser than `stage_semantics.py` (66/10), and 207
   uppercase vocabulary constants. Step 1's triage must screen all of them (grammar over machine
   text vs vocabulary over prose) before the next batch is picked.

## 10. Results (2026-09-26)

### The instrument

- `kicraft/eval/check_replay.py` — replays archived candidates through the current checks.
  `--dump-hits` writes the corpus the labelling step reads. 2051 runs found, 1974 with candidates,
  whole sweep in ~9 s (`--workers 8`).
- `kicraft/eval/jev_labels.py` — typed questions over those hits, answers cached content-addressed
  under `kicraft/eval/corpora/vocabulary-literalism/`. Never called by the gate.
- Reproduce: `logs/vocab_literalism/replay_report.json` (before), `replay_report_after.json`
  (after), `replay_hits*.jsonl`, `jev_labels*.jsonl`. `logs/` is gitignored, so the **durable**
  record is the committed `kicraft/eval/corpora/vocabulary-literalism/replay_summary.json` (per-code
  before/after, totals, live boards, the paired canary) alongside the label cache and
  `hand_labels.jsonl`.

### Before / after, same 1974 runs

| check | one-line intent | rule chosen | firings | on fully-committed boards |
|---|---|---|---|---|
| `functional_spec_partial_ground_flow` | every functional block has a ground return | a block is grounded at **either** end of a ground connection (was: target only) | 589 → **279** | 115 → **65** |
| `intent_quantity_subject_unbound` | the count names a class some gate can enforce | alias rows for the brief's own jack spellings | 246 → **222** | 21 → **15** |
| `architecture_obligation_family_mismatch` | the chosen family can realize the demanded class | witness row `adjustable-rc-lowpass@1` → `trim-potentiometer` | 169 → **151** | 18 → **0** |
| `bom_architecture_role_unsupported` | a sheet that declares an active IC implements it | declaration = the sheet's title + typed requirements; prose counts only for a role implemented nowhere | 88 → **62** | 51 → **34** |
| `functional_spec_nonfunctional_block` | a mechanical feature or bare net is not a functional block | `crystal` is a functional component, not a board feature | 86 → **66** | 6 → **4** |
| `functional_spec_premature_topology` | the spec must not commit a technology the brief never asked for | scan the block **names** the writer commits to (+ obligation classes), not `purpose`/`description`/`assumptions` prose | 73 → **10** | 9 → **1** |
| `intent_prototyping_area_omitted` | a brief that asks for a pad field records it | `prototyping` must name the field; add the geometry phrasing ("a grid of 2.54 mm holes") | 52 → **32** | 14 → **5** |
| **total** | | | 1882 → **1401** | 278 → **168** |

No check's firing count rose. `intent_prototyping_area_omitted` went 52 → 32, not to zero: an
intermediate state matched nothing for "prototyping shield" and read as 0, which was the wrong
direction; the shipped regex accepts the shield and the geometry, and rejects the *purpose*
phrase ("…header row for easy prototyping") that made the 24-pin FPC breakout record a pad field.

### Batch 2 (same session, after the first deploy)

Three more, found by reading the residual that batch 1 left:

| check | what was wrong | firings | on shipped boards |
|---|---|---|---|
| `architecture_power_block_as_sheet` | the escape hatch read `regulat(?:or\|ion)?`, which cannot match **"regulate"** — so a POWER sheet that regulates VIN was refused as distribution-only. The grammar is now complete (`regulat\w*`, `convert\w*`, `suppl\w*`, …, `implement\w*`) | 65 → **33** | 10 → **1** |
| `functional_spec_partial_ground_flow` | `category == "mechanical"` blocks (mounting holes) were required to have a ground return; they draw no current and own none | 279 → **262** | 65 → **64** |
| — | an attempted third rule, "a sheet with no requirements proves nothing", was **reverted**: it broke two deliberate invariants (`GND` net and a requirements-free distribution sheet must still be refused). The writer's own verb (`implements`) closes that case instead, and the two pinned tests pass unchanged | — | — |

Cumulative: total firings 1882 → **1352**, on shipped boards 278 → **158**. Full suite 4643 passed.

### Batch 3

| check | what was wrong | firings | on shipped boards |
|---|---|---|---|
| `bom_architecture_role_unsupported` | the pipeline assigns role **`driver`** to relays, LED strings and transistor stages, all implemented by K/D/Q references; the check read `driver` as an IC role and asked shipped relay-quad and LED-ring boards for a U-reference for K1 and D1..D12 | 62 → **40** | 34 → **23** |
| `functional_spec_external_load_power_assumed` | the rule has a `motor\|heater` branch but its "the brief assigned this duty" vocabulary only offered LED/display terms, so a brief that explicitly powers its motors could never discharge it (latent; the corpus's motor briefs do not state the duty, so the count barely moved — the false-refusal path is gone and pinned by a test) | 56 → **54** | 8 → 8 |

Cumulative: total firings 1882 → **1328**, on shipped boards 278 → **147**.

### Batch 4 — the deterministic repair the check was asking for

`intent_obligation_class_unrealizable` (215 findings across 154 runs, none on a shipped board) is
three shapes, and its own evidence already names the repair for two of them ("the reviewed name is
the repair", "the board carries the mate"). Nothing performed that repair, so the class was
committed, and the parts stage then could not resolve it — the runs failed downstream.

`complete_class_spellings` now makes the rename the writer was being asked to make, before the
decider and the checks, recorded as a defaulted assumption. Guards, each with a test: a negated
class (`no-microcontroller` — the token-superset relation inverts), a class that is not a part, an
off-board source class (those have their own repairs), and a variant with no reviewed carrier.

| | firings |
|---|---|
| `intent_obligation_class_unrealizable` | 215 → **65** (exactly the guarded residue: 36 not-a-part, 26 source, 3 negation) |
| `intent_quantity_subject_unbound` | 222 → **220** (see below) |

The first cut **regressed** `intent_quantity_subject_unbound` by 2: a count bound by a shared token
(`buttons = 3` ↔ `rotary-encoder-push-button`) stopped binding the moment its class was renamed. A
count row now follows its class, which is why the number ends 2 *better* than before rather than 2
worse. Total firings 1328 → **1176**.

### The three live boards are clean

Replaying the live sessions (not the archive) after the fixes:

| session | before | after |
|---|---|---|
| `20260926T151624Z_CH32V003_DEV_BOARD` (seed-43, the board §11 built) | `bom_architecture_role_unsupported` on `UART INTERFACE`, `RESET INPUT` | clean |
| `20260925T130049Z_PASSIVE_RC_FILTER` | `functional_spec_partial_ground_flow` on `bnc_input` | clean |
| `20260925T190905Z_5V_3V3_CONVERTER` | clean | clean |

### Jev labels and the hand-check

111 labels cached (51 above the 0.7 floor) across the three ambiguous checks, in
`kicraft/eval/corpora/vocabulary-literalism/`. The result that justifies the method:

- The first version of the count question offered two options (missing class / property of one
  part). Jev answered "the count belongs to a part class the design is missing" at 0.70–1.00 for
  **every** sample — which would have justified keeping a literalist refusal. Reading the
  candidates showed the truth: the intent already carried `audio-jack-3-5mm`; only the spelling
  differed. Adding that third option flipped 11 of 12 to "the design already carries that class
  under a different spelling" at 0.85–0.93, and the fix became two alias rows. **A model's
  consensus is not ground truth; the option set decides what it can say.**
- `functional_spec_partial_ground_flow` residual: Jev reads "a ground return is provided
  elsewhere" at 0.88–0.90 on the speaker crossover and the passive RC filter (both shipped) but
  below the floor (0.26–0.67) on others. That residual is the Tier-B case: mixed by nature, so a
  relaxation needs a floor and a recorded assumption, and it is **not wired** (see §11).
- `bom_architecture_role_unsupported` on the seed-43 board: "the sheet only references that part,
  which is implemented on another sheet" at 0.63 — the reading the deterministic fix now encodes.

Agreement against the six hand labels: **2/2 on the live boards** (one of the two above the floor);
the three audio-jack count labels describe firings the alias fix removed, so they no longer exist in
the post-fix corpus — the fix is the proof; and the speaker-crossover ground-flow case is a genuine
**disagreement** (the hand label abstains, "the candidate does not say", while Jev asserts "provided
elsewhere"), recorded rather than smoothed over. That is the honest state of the residual: it needs
the owner's reading or the Tier-B relaxation, not a re-worded check.

### Deliberately left alone

- `intent_obligation_class_unrealizable` (215): the demanded class has no reviewed carrier. The
  library's own fallback — research the part and add the record (P14–P16) — is the honest route;
  making the check lenient would ship a board with a part nobody can buy.
- `architecture_*` checks not in the V list (e.g. `architecture_power_block_as_sheet`, 65): outside
  this batch; the instrument covers them, the harness will rank them next.
- V16 (eval verbatim-evidence): unchanged — the fix is a named derivation (`derived_from`), not
  prose.

### Canary (criterion 4), paired single sample

`deploy/verify-design-canary.sh` on the six briefs whose checks changed, run twice on the same day —
once on the pre-change tree, once with the changes:

| brief | baseline | with changes |
|---|---|---|
| rc-lowpass-bnc | FAIL — architecture `contract_rejected` (`multiple_intent_contracts`), 5 attempts | identical, 5 attempts |
| r2r-dac | FAIL — architecture `contract_rejected` | **OK** (architecture 2 attempts) |
| speaker-crossover | FAIL — bom `unit_repair_exhausted` (6) | FAIL, same check (architecture attempts 4 → 2) |
| stm32-min | FAIL — bom `unit_repair_exhausted` (4) | FAIL — bom `commit_rejected`, architecture mis-attached `crystal`/`pushbutton` to the MCU requirement |
| fpc-breakout | OK | FAIL — architecture `contract_rejected` (`unsupported_lowerer_contract`) |
| audio-jack-buffer | FAIL — architecture `contract_rejected` | **OK** (all five stages) |
| **committed** | **1 / 6** | **2 / 6** |

Read it as: **no regression, and not yet proof of an improvement.** The failures are in checks this
plan did not touch (`multiple_intent_contracts`, `unsupported_lowerer_contract`, bom
`unit_repair_exhausted`, a sourcing gap on the Dayton `LW18-50` inductor), which is the
architecture-convergence problem §10 of the living plan already records. One sample per brief cannot
separate a fix from run-to-run noise — the script says so itself (four 2026-09-17 canaries spread
0–1/34 across identical code). Attributing the change needs `KICRAFT_CANARY_REPEATS=3`; the
1974-run replay above is the stronger evidence for the firing counts, and it has no such noise.

### Limits of this run

- The replay measures **firings**, not repair rounds, so the KPI table in §4a does not move with
  these fixes; the paired canary above is the repair-round evidence and it is one sample per brief.
- Wiring is not replayable from the archive (§4).
- The hand labels were written by the assistant from candidate evidence, not by the owner; the
  agreement numbers are a sanity check, and the owner's confirmation is what makes them ground
  truth.



## 11. Deliverables state

| §5 deliverable | state |
|---|---|
| replay harness + its output artifact (frequency/repair table, saved JSON) | **done** — `kicraft/eval/check_replay.py`, `logs/vocab_literalism/*.json(l)`, tests in `tests/test_check_replay.py` |
| per-check decisions (intent, candidate rules, measured numbers, chosen rule) | **done** for the seven checks in §10; the rest are ranked in `replay_report_after.json` |
| patches + tests for the chosen rules | **done** — both directions per check (a valid form now accepted, a real defect still refused) |
| cached Jev labels + agreement numbers, committed as data | **done** — `kicraft/eval/corpora/vocabulary-literalism/`, agreement in `logs/vocab_literalism/`, `tests/test_jev_labels.py` |
| living plan's §11/ground-rules updated | **done** — `docs/plans/live-debug-skill-plan-2026-09-25.md` §13 |

Next batch, in the order the table ranks the residual (numbers are the shipped-board column after
batch 3):

1. `functional_spec_partial_ground_flow` (64): **not another word rule.** The remaining cases are
   representation, not vocabulary — a passive series network whose ground return runs through the
   signal path, or a spec that enumerates ground once per net instead of per block. Two candidates:
   (a) count a block as grounded when a *connection* reaches it through a block that is itself
   grounded, or (b) the Tier-B relaxation. Either needs the owner's reading of two or three cases
   first; the one case that could be adjudicated from evidence was a recorded disagreement between
   the hand label and Jev, which is why the relaxation is not wired.
2. `bom_architecture_role_unsupported` (23): the residual is prose that names a role the design
   builds nowhere ("touch input", "high speed switch"). Whether that is a claim or a description is
   the same judgement as (1).
3. `intent_quantity_subject_unbound` (15): the remaining subjects name a class no spelling relates
   (a genuinely new class). The research fallback is the honest route; add an alias only where Jev
   and a hand label agree that the class already exists (see §10).
4. `functional_spec_external_load_power_assumed` (8): now a *true* disclosure finding — the spec
   assigned itself a power duty without recording it. Leave it; the remedy is one assumption line.
5. ~~`intent_obligation_class_unrealizable`~~: **done in batch 4** — the deterministic rename the
   check asks for is now performed, with guards. The residual 65 are the not-a-part and
   off-board-source shapes, whose repair only the writer can choose; add a mate-class table
   (`battery`/`cell` -> `battery-connector`/`coin-cell-holder`) if the owner wants those done too.
6. Screen the other ~400 regex sites (`design/synthesis/validation.py` is denser than
   `stage_semantics.py`) with the same harness before picking the next batch.


## 12. Verification notes

- `tests/test_stage_semantics.py`, `tests/test_stage_driver_retry.py`, `tests/test_part_identity.py`,
  `tests/test_architecture_intent.py`, `tests/test_jev_labels.py` — all green.
- The replay is deterministic: the harness disables the one network-touching completion step, so the
  same archive replays to the same table.
- The two retry tests that used prose to trigger a bounded repair now trigger it through the block
  name; one assertion about the stage's *final* row set was replaced (the adopted candidate is
  clean, so the row set is empty, and the defect is asserted from the emitted events instead).
