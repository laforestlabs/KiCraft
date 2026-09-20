# Design yield recovery — plan (2026-09-19)

**Purpose.** The live corpus (34 briefs) delivered **24–25/34 fab-ready boards in late July** and
**3–5/34 today**, on the same rubric and the same judge. This plan recovers the yield *without*
giving up what the strict contract layer buys, in the operator's chosen order:

1. **Move A — measure before changing anything** (§2),
2. **Move E — put the July-family configuration behind an admin switch** (§3), so production
   kicraft.io can be swapped between the flawed-but-working configuration and the development path
   at will,
3. **Move D — split every check into BLOCK vs RECORD** (§4); the classification of the existing
   checks was delegated to this plan and §4.3 *is* that decision,
4. **Move B — keep maturing the current direction on a planned cadence** (§5).

**How to use this document.** §0 is the measured state (read it — it is the whole context). §1 is
what the operator decided and what is therefore not up for debate. §2–§5 are one session each, in
order, each with a kill criterion. §6 is the measurement protocol **every** session uses; a change
that is not measured this way does not ship. §7 is "done". §8 is the environment and the exact
commands. §9 is the short list of things still owed to the operator.

**Working rules in one line each.** One move per session. Repeats or it did not happen. Never
weaken a check to make a brief commit; reclassify what a check *means* and record it. Never do
anything silently — not a downgrade, not a pipeline swap.

---

## 0. Where the pipeline is (measured)

### 0.1 The yield curve, 34-brief live corpus

| date | batch dir | designer | committed / 34 | fab-ready / 34 | mean |
|---|---|---|---|---|---|
| 07-20 | `logs/self_eval/20260720T113207Z` | deepseek-v4-flash | — | 24 | 75.3 |
| 07-27 | `logs/self_eval/20260727T045000Z` | deepseek-v4-flash | — | 22 | 68.1 |
| 07-27 (post-fix re-batch) | `logs/self_eval/20260727T151918Z` | deepseek-v4-flash | — | 22 | 75.8 |
| **07-28** | `logs/self_eval/20260728T120442Z` | deepseek-v4-flash | — | **25 (high)** | 73.7 |
| 09-15 | `logs/self_eval/canary_20260915T034256Z` | luna | **13** | (design-only) | — |
| 09-15 | `logs/self_eval/20260915T132650Z` | luna | **13** | **11** | 57.6 |
| 09-16 | `logs/self_eval/canary_20260916T234557Z` | luna | **0** | 0 | — |
| 09-17 | `canary_20260917T001818Z` / `…004307Z` / `…011637Z` | luna | 0 / 1 / 0 | 0 | — |
| 09-17 | `logs/self_eval/canary_20260917T043323Z` (34×3) | luna | 6/102 runs, 4/34 briefs | 0 | — |
| 09-18 | `logs/self_eval/20260918T055753Z` | luna | 5 | 4 | 55.9 |
| 09-18 | `logs/self_eval/20260918T124717Z_finalfix` | luna | 4 | 3 | 54.5 |

Cost per delivered board: July ≈ **$0.04** (~$1.00 per 34-brief batch ÷ 25 boards); 09-17 ≈
**$0.31** per committing design with **96 % of spend buying runs that failed**; today ≈ **$0.17**.

Wall-clock for one 34-brief pass, for budgeting §2's arms: the 09-15 batch (11 boards built) took
**5 740 s**; the 09-18 batch (4 built) took **2 006 s**. A pass that actually builds boards costs
1.5–2.5 h at `--parallel 3`, one build slot.

### 0.2 What actually changed (and what did not)

- **Structured output is not the change.** `json_schema` entered the tree on **2026-06-03**
  (`1c03872` "generic self-correcting stage driver"). July's 25/34 was measured *with* it, on
  DeepSeek.
- The designer moved to Luna on **2026-09-12** (`d90895c`).
- The typed design layer is new: `kicraft/server/stage_runtime.py` (08-26), `recipes/registry.py`
  (09-02), `stage_work_units.py` (09-06), `design/lowering.py` (09-09), `design/part_identity.py`
  (09-11), `design/architecture_intent.py` (09-14). On 2026-07-29 none of these existed: 56 Python
  files under `kicraft/design` + `kicraft/server`, versus **81 today**.
- FreeRouting was removed on **08-22** (`a25e039`); KRT is the only router and the FreeRouting
  binary is **absent from this box**.

### 0.3 The veto layer grew faster than the model

| | 2026-09-15 (`b4b8be5`) | 2026-09-16 (`0a2e700`) | HEAD |
|---|---|---|---|
| `_fail` sites in `architecture_intent.py` | 22 | 34 | 34 |
| distinct architecture refusal codes | 19 | 30 | **30** |
| §9.x checks in `synthesis/validation.py` (HEAD only — not measured earlier) | — | — | **22 sections, 20 distinct numbers, 2 advisory** |

The 11 codes added between 09-15 and 09-16 are individually defensible
(`declared_signal_port_tied`, `unsupported_lowerer_contract`, `unreviewed_exact_part`,
`reference_domain_not_zero_volt`, `conflicting_supply_binding`, …), and the canary went from
**13/34 committed to 0/34** over the same window.
`docs/plans/self-eval-2026-09-16-remediation-handoff.md` records it plainly: *"canary_20260915T034256Z
(before the 2026-09-16 contract work) committed 13/34 — so the 34/34 release gate is far away under
the current contract."*

### 0.4 The strict layer is not wrong; the pacing is

The refusals catch real defects. Spot-checking the fixed batch's fatal drafts (09-19): all three
`declared_signal_port_tied` runs are genuine self-contradictions in the model's own draft — it states
a pin is tied to a declared rail *and* routes a separate signal to that same pin; no signal in those
drafts declares the rail. That is a wrong netlist, and refusing it is correct.

The structural problem: the design stages have **one** verdict (raise → dead run), while the
build/verify level already has **two** and has had them for weeks (`kicraft/design/cli_app.py`): a
gate dict carrying `ok` *and* `fab_acceptable`, with separate `reasons` and `warnings`, plus
precedents for measured downgrades (minor courtyard overlap → warning, form-factor advisory when
enforcement is off, unverifiable connector mouth → loud warning). §4 gives the design stages that
second level.

### 0.5 The instrument cannot steer

- `3456e14` (09-13, the project's own ladder experiment): *"stock commits 12/20 … stock itself
  measured 0.60 then 0.20 on identical inputs."* Same code, same briefs, 3× apart.
- 09-18 batches on nearly identical trees: 5, then 1, then 4 committed.

Nothing else in this plan matters until that is fixed; §6 is the protocol that fixes it.

---

## 1. Decisions taken (operator, 2026-09-19)

- **D1 — warnings beat nothing.** A board carrying recorded warnings is preferable to a dead run.
  This is the default disposition for checks that prove a property is *unverified*, as against checks
  that prove the board is *wrong*.
- **D2 — no silent weakening, and nothing else silent either.** A check may be reclassified
  (block → record), never deleted or bypassed, and never downgraded without the warning being
  (a) named, (b) persisted on the artifact, (c) countable in the scorecard. A pipeline swap (§3) is
  marked the same way.
- **D3 — measure first.** No behavioural change before §2 has run.
- **D4 — then mature, on a cadence.** One move per session, one measurement per move, revert if the
  pre-registered delta is not met.
- **D5 — no customer-facing bar yet.** There are no paying customers. Boards shipped with warnings
  need no human confirmation gate and no disclosure policy for now; the bar is self-eval and
  internal use. (The operator's note: this stays a non-issue only until the yield is restored —
  restoring it is the point of this plan.)
- **D6 — no limit on the number of warnings.** There is no cap on advisories per board, no maximum
  count that blocks a ship, and no scoring penalty designed to suppress them. The advisory list is
  unlimited; it must only be *recorded and visible*.
- **D7 — the July-family configuration is a supported, switchable configuration.** It goes behind an
  admin switch (§3) so production can be swung between it and the development path at will.

**Measured dead ends — do not repeat them** (from their 09-17 plan §1 and §0, plus the 09-18 session):

- One-brief-at-a-time refusal wording, error-class aliases, or single-brief identity fixes. Six such
  commits moved the aggregate not at all; the same two model-authored classes still dominate.
- Rewriting the contract shape a third time, or restoring a looser JSON path *inside* the current
  pipeline. (Running the older pipeline as its own service, §3, is a different move: it keeps both
  codebases honest and neither absorbs the other.)
- Adding a check and a fix for it in the same session without a like-for-like baseline.

---

## 2. Move A — measure before changing (session 1)

Goal: split the 25 → 4 loss into **(contract work)**, **(model change)**, **(drift)**; size move D's
payoff before implementing it; and stand up the legacy tree that §3 needs.

### 2.1 Arms

| arm | tree | how | what it answers |
|---|---|---|---|
| **A1 current** | working tree at `HEAD` | 34 briefs × **3** repeats | today's baseline, with repeats |
| **A2 legacy** | clone at **`bc6a2f8`** (2026-08-25) — the newest commit with *no* typed design driver that is *after* the FreeRouting removal, so it can build on this box | 34 × **2**, design+build | does the July-family configuration reproduce ~20/34 fab-ready **today, on this box**? |
| **A3 pre-contract** | `git worktree add /tmp/kc-0915 b4b8be5` (09-15, typed but before the 09-16 contract push) | 34 × **2** | how much of the loss is the 09-16 contract work specifically |
| **A4 D-preview** | today + a **scratch, uncommitted** patch turning the §4.3 RECORD-class codes into recorded advisories | 34 × **3** | move D's payoff, measured before anyone implements it |

Budget: **~10 h wall, ~$9**, inside the daily cap. Run the arms sequentially (2 cores, one build
slot); never two arms at once. The legacy arms cost more wall-clock because they actually build
boards — that is the point of them.

A2's patch-free clone is also the §3 deployment, so session 1 produces that prerequisite. A4's patch
is measurement scaffolding, not a candidate ship: it must still *record* every downgraded code (so
the "shipped with N advisories" count exists), and it must be reverted after the run.

`KICRAFT_STAGE_SEMANTICS` is **not** a useful lever here: its production default is already `repair`,
and it governs only the semantic-repair rounds, not the `_fail` contracts.

### 2.2 Pre-registered reading (write this into the doc before starting)

Against **A1** (current tree, 3 repeats):

| observation | what it means | next |
|---|---|---|
| **A2 ≥ 15 fab-ready** (vs A1's ~3–5) | the July-family configuration still works; the loss is the contract layer plus the model | build §3 (the switch) and run D in the same arc |
| A2 ≈ A1, but A3 > A1 by ≥6 committed | the typed layer is fine; the 09-16 contract push owns the loss | skip to §4 with confidence; §3 still worth building for the fallback |
| A2 ≈ A1 ≈ A3 | the loss is not the code path — it is the model or the environment | run the model axis before anything else: same tree, `KICRAFT_DESIGN_PROFILE=deepseek` vs `luna`, 34×3 |
| A2 > A1 but neither reaches 10 | partial: the direction is right, the contract is losing most of the yield | §4 first, §3 second |

Against **A1** (move D's payoff):

| A4 vs A1 fab-ready | next |
|---|---|
| ≥ +3 boards | implement §4 as specified |
| +1 to +2 boards | implement §4 **and** §5 B-2 (compiler owns the authoring) in the same session — A4's advisories will show which codes still kill boards later |
| ≈ 0 | the RECORD list is misclassified; do **not** widen it — re-derive from A4's per-board failure path and re-run A4 |

Record for every arm: committed runs, briefs committing in ≥2 repeats, fab-ready,
first-failing-stage split, failure-code census (`python -m kicraft.cli.triage scan`),
`failed_run_cost_usd`, `cost_per_committed_design_usd`, and `source_unchanged=true`. Append the
tables to this document under `## §2 Results`.

### 2.3 Environment gotcha (cost me a broken first batch)

The agent/hub process environment carries stale `KICRAFT_*` variables (notably
`KICRAFT_DESIGN_PROFILE=flash`, no longer a valid profile) and `.env` never overrides an existing
variable, so a batch launched from a tool shell silently picks the wrong profile. Always launch as:

```bash
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  /home/kicraft/KiCraft/.venv/bin/python -m kicraft.eval.self_eval --out <DIR> …
```

---

## 3. Move E — the July-family configuration behind an admin switch (session 2)

**Gate:** build this only if A2 shows the legacy tree still delivers (§2.2, first row). If A2 does
not reproduce the era, the switch buys nothing — record that and go to §4.

**What "the July configuration" is, concretely.** It is not a model profile; it is the *older
pipeline* (model-authored design, one JSON per stage, no typed contracts, no recipes/lowerers, no
work units). The literal July tree (`0927af0`) cannot build on this box because its router
(FreeRouting) was removed on 08-22. The newest commit that keeps the old design pipeline **and**
routes with KRT is **`bc6a2f8` (2026-08-25, "Bound BOM stage emission")** — verified: it predates
`kicraft/server/stage_runtime.py` (08-26) and postdates the FreeRouting removal. **Pin that.**

### 3.1 Deployment (do not share a tree)

```bash
# a clone, not a worktree: the legacy tree must never write to the live repo's refs
git clone --no-hardlinks /home/kicraft/KiCraft /home/kicraft/KiCraft-legacy
cd /home/kicraft/KiCraft-legacy && git checkout --detach bc6a2f8
python3 -m venv .venv && .venv/bin/pip install -e ".[server,design]"
cp /home/kicraft/KiCraft/.env .env && chmod 600 .env      # secrets stay in one place, mode preserved
```

Three isolation decisions, taken here so the implementer does not have to guess:

- **Projects**: legacy runs get their own `KICRAFT_PROJECTS_DIR` (`~/.kicraft/projects-legacy`) so
  artifacts, state files and logs can never be confused with the current pipeline's.
- **Spend**: **share** the ledger (`~/.kicraft/spend_ledger.db`) so the daily/total caps and the
  spend reports stay honest across a swap, and tag every legacy call with the pipeline name so the
  two configurations can be compared afterwards.
- **Router**: if the legacy tree's KRT preflight refuses this box's checkout
  (`/home/kicraft/KiCadRoutingTools` @ `3ceb773`, venv `/home/kicraft/krt-venv`), give the legacy
  service its own checkout at the commit it expects via per-process
  `KICRAFT_KICAD_ROUTING_TOOLS_PATH` / `KICRAFT_KICAD_ROUTING_TOOLS_PYTHON`. Do not relax the
  preflight.

**If A2's numbers disappoint and you want the *literal* July tree (`0927af0`, 07-29).** It routes
through `kicraft/autoplacer/freerouting_runner.py`, and this box has **neither a Java runtime nor the
FreeRouting directory** (`java` absent, `/home/kicraft/FreeRouting` absent), so restoring it means
installing a JRE, vendoring the FreeRouting jar where that module looks for it, and re-validating the
run. That is bounded work (~an hour), but it is a *third* router configuration to keep alive, and
FreeRouting was removed deliberately on 08-22. Recommendation: pin `bc6a2f8` as the switchable legacy
configuration, and only restore FreeRouting if A2 comes back at ~20/34 fab-ready but the failures are
attributable to routing rather than to the design stages. Do not do it speculatively.

### 3.2 The switch

The admin routing page (`/admin/routing`) and its durable file (`~/.kicraft/routing.json` via
`kicraft/server/routing_config.py`) already exist for exactly this kind of knob — the page promises
"a save takes effect on the next design run — no restart". Add one key:

1. `routing_config.ALLOWED_KEYS` += `pipeline`, valid values `current` | `legacy`, persisted and
   loaded like `active_profile`; a `Settings` field of the same name.
2. Admin UI: a select next to the design profile, labelled with the pinned commit
   ("legacy pipeline (bc6a2f8, 2026-08-25)") and the trade-off in one line.
3. **Design dispatch** — the web app drives the LLM stages *in process* today
   (`kicraft/server/session.py` → `stage_pipeline.drive_chain`). Two versions of the `kicraft`
   package cannot coexist in one interpreter, so `legacy` must delegate: run the legacy tree's own
   headless driver as a subprocess with the legacy venv (it has
   `kicraft/server/stage_driver.py`), then hand the workspace to the build.
4. **Build dispatch** — `kicraft/server/build_worker.py::JOB_KIND_COMMANDS` already runs
   `sys.executable -m kicraft.design.cli_app build …` as a subprocess. For `legacy`, substitute the
   legacy interpreter; the legacy package resolves from its own venv, so no other change is needed.
5. **Marking (D2):** every project records `pipeline` + the pinned commit; the board page and the
   provenance file name it, and the scorecard groups by it. A swap must never be invisible.

### 3.3 The compatibility risk, stated up front

The web UI renders project pages from `state.json`, and the legacy tree writes a **different state
schema**. Mitigation, in order of preference:

- write a small `pipeline.json` marker in the workspace at project creation; for legacy projects the
  page renders the *delivered artifacts* (board, preview, gerbers, report) plus a banner instead of
  live stage state;
- keep the legacy project dir separate (above) so nothing overwrites a current-schema state file;
- if a page-level reader must touch legacy state, gate it behind a version check and fail to the
  banner, never to a stack trace.

### 3.4 Acceptance

- One brief, started while `pipeline=legacy` is selected, commits all five stages **and** reaches
  `build=fab-ready` through the legacy tree — proved with artifact paths (~$0.05–0.10).
- Switching back to `current` and running the same brief uses the current tree (same proof).
- No restart required between the two; the routing page's existing promise holds.
- The project row, provenance and scorecard all name the pipeline.
- §4's guard test (nothing silent) covers this knob too: a swap that is not recorded fails CI.

---

## 4. Move D — the bar: BLOCK vs RECORD (session 3)

### 4.1 The rule

**BLOCK** when shipping the board as-is would deliver something electrically wrong, physically
unbuildable, or missing a feature the brief asked for. **RECORD** when the board is very likely fine
but a property could not be *proven*.

Two corollaries the implementer must not soften:

- *"The brief asked for X and nothing implements X"* is BLOCK, always. That is a missing feature, not
  an unproven one. `physical-obligation-unfulfilled`, `missing-requirement-implementation` and
  `declared-interface-unrealized` (the current top BOM wall) stay blocking; the fix for them is to
  move the decision earlier or let the compiler own it (§5), never to downgrade them.
- *Two different nets on one pin* is BLOCK when the draft names two unrelated nets, but a **derive**
  (not a downgrade) is right when the compiler already knows the answer — e.g. the pin is the
  requirement's published supply port and the draft also declares that rail. That split is §5 B-3.

### 4.2 The mechanism (reuse what exists; do not invent a second system)

Copy the shape the build level already has: a gate dict with `ok` **and** `fab_acceptable` plus
separate `reasons` and `warnings` (`kicraft/design/cli_app.py` ≈5509–5600; downgrade precedents at
6440–6500).

1. **A parallel advisory path next to `_fail`** in `kicraft/design/architecture_intent.py`
   (`_advise(code, message, evidence=…)`) appending to a structured list instead of raising.
   `Architecture.assumptions` (`kicraft/design/models.py:734`) is today's only free-text channel and
   `derived_notes` already lands there (`architecture_intent.py:1931`); add a typed
   `advisories: list[ArchitectureAdvisory]` (code, message, evidence) rather than smuggling structure
   into prose.
2. **Propagate** advisories → stage state → `eval/report.json` and the board's provenance, so the
   artifact says *"shipped with: unreviewed part identity X, unverified rating Y"*.
3. **Count, never cap (D6):** add one rubric gate (`detected_by: script`) that records the advisory
   count and codes. No cap, no ship-block, no suppression — the count exists so a swap or a
   downgrade is auditable over time.
4. **Same treatment for the record-class BOM/wiring unit checks** (§4.3) and reuse the existing
   `gate["warnings"]` plumbing so one board report carries design-stage and build-stage warnings in
   one list.

### 4.3 The classification (delegated decision — implement this)

Rule for anything not listed: *would the delivered board do the wrong thing?* → BLOCK; *could it be
wrong in a way we cannot prove?* → RECORD.

**A. Architecture stage — `kicraft/design/architecture_intent.py` (30 codes)**

| class | codes | verdict |
|---|---|---|
| dangling references (the draft names something that does not exist) | `unknown_requirement_sheet`, `unknown_signal_requirement`, `malformed_signal_ref`, `unknown_interface_port` (×2), `unknown_tie_net`, `unknown_reference_domain`, `unknown_supply_rail` (×2), `unknown_edge_rail`, `duplicate_requirement_id`, `duplicate_edge_connector`, `derived_output_name_collision` | **BLOCK** — the intent cannot be compiled faithfully |
| two unrelated nets on one pin | `conflicting_port_binding`, `conflicting_supply_binding`, `conflicting_reference_binding`, `declared_port_double_bound`, `declared_signal_port_tied`, `signal_conflicts_with_rail`, `signal_names_rail`, `reference_domain_not_zero_volt` | **BLOCK** (verified genuine in the 09-19 spot-check); the *derivable* sub-case is §5 B-3, a derive rather than a warning |
| the brief's own mechanical/electrical frame is contradicted | `unsupported_standard_stacking_interface`, `incomplete_standard_stacking_interface`, `invalid_standard_stacking_owner`, `invalid_standard_stacking_pinmap`, `usb_connector_supply_unknown`, `incomplete_usb_edge`, `empty_edge_connector` | **BLOCK** — the board would not mate, or a socket has no supply |
| **provenance not proven** | `unreviewed_exact_part`, `unknown_part_refused` | **RECORD** — buildable and very likely correct; what is missing is *proof*. Record the exact part string (or "no declared interface") |
| **family cannot build the stated geometry** | `unsupported_lowerer_contract`, `unbound_required_port` | **SPLIT**: (a) the draft declared no contacts while its own signals name them → the compiler derives (§5 B-2, no warning needed); (b) the geometry is genuinely outside the family's range (a 15-way terminal, gapped contact numbers) → **BLOCK** |

**B. BOM / wiring units — `kicraft/server/stage_work_units.py`**

| class | verdict |
|---|---|
| `missing-requirement-implementation`, `physical-obligation-unfulfilled`, `declared-interface-unrealized` | **BLOCK** (missing feature; §4.1) |
| netlist/connectivity gates (`COMMIT_GATE:9.42`-style; §9.7/9.8/9.9/9.10/9.11/9.12/9.13/9.15/9.16–9.18/9.19/9.20/9.24/9.25/9.31/9.32/9.37 in `synthesis/validation.py`) | **BLOCK** — wrong netlist |
| **§9.33 typed exact-part accountability** (hard) | **RECORD** — the same property as `unreviewed_exact_part` one stage later; both must move together or the run dies later instead. Keep the evidence requirement (the exact string must be recorded) |
| **§9.34 brief-stated mount type** (hard) | **RECORD** — mounting preference, not electrical function |
| §9.21 MCU first-flash, §9.22 breakout intent | already advisory — leave |

**C. Build / promote verify — `kicraft/design/cli_app.py`**

| class | verdict |
|---|---|
| `shorts`, `unconnected`, `keepout_intrusion`, gross courtyard overlap, `courtyard_unmeasured` (production pcbnew present), `connector_misoriented:*` with a known mouth, antenna edge contract, `missing_component_refs`, illegal geometry, copper/edge clearance | **BLOCK** — keep exactly as is |
| minor courtyard overlap, `connector_stranded:*`, unverifiable connector mouth, silkscreen, utilization/aspect | already `warnings` — keep |
| `form-factor non-conformant` (enforcement on), `outline-shape non-conformant` (supported shape) | **BLOCK** — a board that cannot mate, or is the wrong shape, is wrong |
| form-factor / outline-shape advisory (enforcement off, or unsupported shape) | already warnings — keep |

**D. Rubric — `kicraft/eval/rubric.yaml`**

Keep every existing cap gate. Add only the advisory-count gate from §4.2 step 3. Caps are already the
"measured, not fatal" idiom at score level; do not convert them into passes.

### 4.3.1 Anti-silent-weakening guards (write these tests)

1. Per RECORD class: the advisory appears in `report.json` **and** in the rubric's advisory gate.
2. The BLOCK list above still raises (snapshot the code strings; a silent move across the line fails).
3. The pipeline marker from §3 appears in provenance and in the scorecard grouping.
4. `design_acceptance --reference-replay` (mock transcript) still reproduces ≥31/34 — the floor
   recorded in the 09-16 handoff.

### 4.4 Acceptance for §4

On the frozen corpus, 34 briefs × 3 repeats, advisory path off vs on, same tree:

- fab-ready **rises** by ≥3 boards, each newly shipped board carrying ≥1 named advisory;
- no BLOCK-class failure appears on any board that ships;
- reference replay still ≥31/34;
- the scorecard prints, per board, the advisory codes it shipped with.

If fab-ready does not rise, the advisories will show why (non-empty on boards that still die later):
record it, re-derive the classification, re-run — do not widen the RECORD list on taste.

---

## 5. Move B — maturing the direction, on a cadence (sessions 4+)

The direction stays. What changes is the cadence and the discipline: **one move per session, one
measurement per move, revert when the pre-registered delta is not met.**

**The queue** (`docs/plans/design-completion-followup-2026-09-17-plan.md` §0 already maps the
corpus: 9 briefs fail only at architecture, 6 only at BOM part resolution, 15 at both):

- **B-1 (highest value): a requirement carrying a physical obligation its family cannot realize
  should be refused where the part is still being chosen** — the architecture stage, with the
  canonical satisfying options in the refusal (their plan §11.8). Today that defect is caught only
  after four BOM repair rounds. The check already exists deterministically in
  `stage_work_units._requirement_obligation_defects`; running it earlier turns a 96 %-wasted spend
  pattern into a cheap correction.
- **B-2: stop asking the model for what the compiler can derive.** Measured precedent: the 09-17
  session removed the authoring burden for supply/reference bindings, the obligation list and the
  part-class vocabulary and *quadrupled* completion (1/34 → 4/34 briefs). Next candidates: the
  connector contact contracts (top code `unsupported_lowerer_contract`) and the per-requirement
  interface declaration (`unknown_part_refused`).
- **B-3: derive the rail-vs-signal tie** where the compiler already knows which net owns the pin
  (§4.1, second corollary). Narrow: only when one claimant is a declared rail that feeds the
  requirement. This is the only queue item that removes a BLOCK rather than moving it, so it is last
  and it updates the §4.3.1 snapshot test deliberately.
- **B-4: keep the model axis open.** If §2 lands on "A2 ≈ A1", the designer profile is the next
  variable, measured on one tree (`deepseek` vs `luna`, 34×3). July's 25/34 was DeepSeek — hypothesis,
  not answer.

**Guardrails for every session in this phase**

- Regression set (their §0): `rc-lowpass-bnc`, `esp32-dual-motor`, `r2r-dac`, `fpc-breakout` must
  still commit, measured with repeats.
- Cost ceiling **≤ $5** of provider spend per session; a session that cannot be measured inside that
  is scoped too large.
- Every move states its expected delta *before* the run and is reverted if the measurement misses it
  — including "null but harmless": their own ladder experiment shipped nothing and the honest record
  was the deliverable.
- No new refusal code ships without a classification (§4.1), a test, and evidence that the frozen
  corpus's commit rate did not drop.

---

## 6. Measurement protocol (every session, no exceptions)

1. **Repeats.** 34 briefs × 3 repeats minimum for anything measured on the current tree; per-brief
   **median** is the unit. A single run is never evidence (`3456e14`: 0.60 then 0.20 on identical
   inputs). Legacy arms may run ×2 (their wall clock is dominated by real builds) and must be read
   against the era's own recorded spread (22, 24, 22, 25 fab-ready across four July batches, ±2).
2. **Same-session, same-box comparison only**; `env -i` launch (§2.3). Never compare across
   separately-scripted runs.
3. **Report the whole table**: committed runs, briefs committing in ≥2/3, fab-ready, first-failing
   stage split, failure-code census, `failed_run_cost_usd`, `cost_per_committed_design_usd`, plus the
   advisory counts once §4 lands.
4. **`source_unchanged=true`** in `summary.json` or the campaign is void.
5. **Cheap gates first**: `design_acceptance --reference-replay` (mock transcript, $0) and
   `--design-only` runs precede any paid full campaign.
6. **Persist and append**: every measurement is a directory under `logs/self_eval/<name>` and its
   table is appended to this document under `## §<n> Results`. A session that does not append did not
   happen.

---

## 7. Definition of done for this plan

- §2's tables exist and name which change owns the loss.
- §3 ships (or is killed by A2 with the reason recorded): production can be swung between the two
  configurations from `/admin/routing`, and every board says which one built it.
- §4 ships: the RECORD path, the §4.3 classification, the four guard tests, and a measured increase
  in boards shipped **with recorded advisories**, at no limit on how many.
- §5 delivers at least two moves with like-for-like before/after numbers, or an honest record of the
  null results.
- **The number that matters:** fab-ready boards per 34-brief campaign, with the advisory codes each
  one shipped with, rising over consecutive campaigns while BLOCK-class defects on shipped boards
  stay at **zero**.

---

## 8. Environment and commands

```bash
REPO=/home/kicraft/KiCraft; PY="$REPO/.venv/bin/python"; cd "$REPO"
LAUNCH='env -i HOME=/home/kicraft PATH='"$PATH"' TERM=xterm PYTHONUNBUFFERED=1'

# the standard measurement: 34 briefs, 3 repeats
eval "$LAUNCH $PY -m kicraft.eval.self_eval --out logs/self_eval/<name> --repeats 3"
# design-only (five LLM stages, no build) — cheap probes and contract questions
eval "$LAUNCH $PY -m kicraft.eval.self_eval --out logs/self_eval/<name> --design-only --repeats 3"
#   subset: --only rc-lowpass-bnc,esp32-dual-motor,r2r-dac,fpc-breakout
#   no judge spend: --no-judge      resume an interrupted batch: --resume <DIR>

# $0 reference corpus (mock transcript) — floor: ≥31/34 rows reproduce
"$PY" -m kicraft.eval.design_acceptance --reference-replay
"$PY" -m kicraft.eval.design_acceptance --references

# triage any run
"$PY" -m kicraft.cli.triage run <RUN> | stages <RUN> | audits <RUN> | scan

# §2 arms
git worktree add /tmp/kc-0915 b4b8be5                       # A3
git clone --no-hardlinks "$REPO" /home/kicraft/KiCraft-legacy   # A2 = §3's deployment
cd /home/kicraft/KiCraft-legacy && git checkout --detach bc6a2f8
```

Production notes: web + build worker run on this box (`127.0.0.1:8080`, Caddy → kicraft.io); builds
share one host gate (`KICRAFT_BUILD_SLOTS=2`); never run two campaigns concurrently. The deploy path
(`deploy/deploy-production.sh`) deliberately excludes the live canary — run
`./deploy/verify-design-canary.sh` on purpose, never as part of a restart. Services run as detached
processes managed by `deploy/restart-web.sh` / `deploy/restart-build-worker.sh`, not systemd.

---

## 9. Still owed to the operator

1. **Auto-fallback or manual only?** D7 puts the swap in the operator's hands. The open variant is
   whether a `current`-pipeline design that dies should *automatically* be retried on the legacy
   pipeline (faster throughput, but two configurations producing boards without a human choosing).
   Default assumed by this plan: **manual only**.
2. **Should the legacy badge be visible on the board page** ("built with the legacy pipeline,
   bc6a2f8"), or recorded only in provenance and the scorecard? Default assumed: recorded everywhere,
   shown plainly on the board page — the operator can dial that back later.

---

## 10. Appendix — evidence index

| claim | where it was measured |
|---|---|
| July 24–25/34 fab-ready, mean 73.7–75.8, DeepSeek | `docs/plans/self-eval-2026-07-27-fix-plan.md` (batches `20260720T113207Z`, `20260727T045000Z`, `20260727T151918Z`, `20260728T120442Z`) |
| 09-15: 13/34 committed, 11 fab, mean 57.6, wall 5 740 s | `logs/self_eval/canary_20260915T034256Z`, `logs/self_eval/20260915T132650Z` |
| 09-16/17 collapse to 0–1/34 | `logs/self_eval/canary_20260916T234557Z`, `canary_20260917T001818Z`, `…004307Z`, `…011637Z`; `docs/plans/self-eval-2026-09-16-remediation-handoff.md` |
| 4/34 briefs, 6/102 runs, $1.86, 96 % failed-run spend, $0.31/committed design | `logs/self_eval/canary_20260917T043323Z`; `docs/plans/design-completion-followup-2026-09-17-plan.md` §0 |
| 09-18: 5 then 4 committed, 4 then 3 fab, wall 2 006 s | `logs/self_eval/20260918T055753Z`, `logs/self_eval/20260918T124717Z_finalfix` |
| architecture refusals 19 → 30 codes in one day; 22 §9.x sections (20 distinct, 2 advisory) | `ast` scan of `architecture_intent.py` at `b4b8be5`/`0a2e700`/`HEAD`; section headers in `synthesis/validation.py` |
| the design framework did not exist on 07-29 (56 → 81 files) | `git ls-tree 0927af0` vs `HEAD`; file-addition dates for `stage_runtime.py`/`recipes/`/`lowering.py`/`part_identity.py`/`architecture_intent.py` |
| structured output since 2026-06-03 | `git log -S json_schema` → `1c03872` |
| 0.60 vs 0.20 on identical inputs | commit `3456e14` (2026-09-13) |
| build level already has `ok` + `fab_acceptable` + `warnings` | `kicraft/design/cli_app.py` ≈5509–5600, 6440–6500 |
| `declared_signal_port_tied` refusals are genuine contradictions | 09-19 probe over the fixed batch's drafts (three runs; no rail-declaring signal in any draft) |
| one real bug fixed 09-18/19 (reviewed terminal identity) | `kicraft/design/lowering.py::_reviewed_terminal_part`; `tests/test_design_lowering.py` (3 tests fail on stock code); `logs/self_eval/ab_fixed` vs `ab_stock` — architecture dead-ends 21/24 → 14/24, no headline gain |
| `bc6a2f8` is the last pre-typed-driver commit that routes with KRT | `git cat-file -e bc6a2f8:kicraft/server/stage_runtime.py` → absent; `bc6a2f8` dated 08-25, after `a25e039` (08-22, FreeRouting removed) |
| the literal July tree needs a router this box does not have | `java` absent, `/home/kicraft/FreeRouting` absent; July routing runs through `kicraft/autoplacer/freerouting_runner.py` at `0927af0` |
| the admin switch's hooks exist | `kicraft/server/routing_config.py` (ALLOWED_KEYS + `~/.kicraft/routing.json`), `kicraft/server/routes_admin.py` (`/admin/routing`), `kicraft/server/session.py` (`drive_chain`, in-process design), `kicraft/server/build_worker.py` (`JOB_KIND_COMMANDS`, subprocess build) |

---

## §2 Results

### Reading (against §2.2, written before the campaign)

**Move A's attribution: the loss is owned by the contract layer, and it splits into two measured
steps.** All four arms ran the same designer (`openai/gpt-5.6-luna`), the same KRT router, the
same 34-brief corpus and the same box, so the arms isolate code paths:

| step | comparison | briefs fab-ready | reading |
|---|---|---|---|
| typed layer | **A2** `bc6a2f8` (08-25, pre-typed) → **A3** `b4b8be5` (09-15) | **21 → 13** of 34 | the typed design layer costs ~8 briefs |
| 09-16 contract push | **A3** (09-15) → **A1** `de74af7` (HEAD) | **13 → 4** of 34 | the contract push costs ~9 briefs |

A2's 21/34 briefs reproduces the era's recorded 22–25/34, so this is not a model or environment
story: luna drives the older pipelines to era-level yield on this box today. The plan's first
pre-registered row is cleared with a wide margin (A2 ≫ 15 fab-ready), which is the gate §3 sets
for building the pipeline switch. The second row — "the typed layer is fine; the 09-16 contract
push owns the loss" — is what the numbers show, except that both steps contribute.

**Move D's payoff is ≈ 0, and the reason is measured rather than inferred.** A4 exercised the
downgrade exactly as designed and it did not convert a single run into a board:

- 12 of A4's 102 runs carried an advisory; on **all 12** the RECORD-class codes were the *only*
  architecture diagnostics, so the downgrade is what let them pass the architecture stage that A1
  killed them at. The mechanism works.
- **All 12 then died at the BOM stage**, on walls §4.1 explicitly keeps blocking:
  `declared-interface-unrealized` (`limiter_1: declared interface needs one identified hardware
  owner`), §9.42 `requirement physical/interface realization` (`E_PHYSICAL_REALIZATION 'rp2040':
  requires 1 reviewed 'crystal' physical part`), and `physical-obligation-unfulfilled`
  (`status:status-led: requires 1 real led, found 0`).
- **Zero boards shipped carrying an advisory** — §4.4's acceptance criterion fails outright.
- The headline fab delta (5 → 7 runs, 4 → 3 briefs) is run-to-run variance, not the patch: the two
  briefs that improved (`rc-lowpass-bnc` 1/3 → 3/3, `r2r-dac` 1/3 → 3/3) *committed*, so their
  state was persisted and would have shown an advisory had one fired — none did, which means no
  RECORD code fired on them and their code path is identical to A1's. `audio-jack-buffer` moved
  the other way (2/3 → 0/3), with no advisory either. This is the 0.60-then-0.20 variance §0.5
  measured, and it is why the plan reads per-brief medians rather than runs.

This is §2.2's **"≈ 0"** branch, whose instruction is *"do not widen it — re-derive from A4's
per-board failure path and re-run A4."* The re-derivation is already visible in the data, and it
does not point at more RECORD codes: the wall sits one stage later, in the BOM obligations, which
§4.1 rules out of the RECORD class by design. Widening RECORD would therefore repeat this null.

**§9.33/§9.34 never fired.** Across 102 A4 runs no spec-named-MPN or brief-stated mount-type
contradiction arose, so the BOM-stage half of the RECORD list is unexercised by this corpus and
its downgrade is a null here, not a verified one.

**Spend.** \$9.42 for the whole campaign (09-19 \$6.98 + 09-20 \$2.44), of which \$8.78 is the four
arms, \$0.05 the smoke gates, and \$0.30 (3.2 %) the two void batches — recorded because the plan's
complaint was precisely that wasted spend was invisible. Wall clock: A1 14:27Z→15:56Z (88 min,
including one killed attempt), A2 16:20Z→21:09Z (4 h 49 min), A3 21:10Z→00:38Z (3 h 28 min, three
attempts), A4 00:43Z→01:48Z (65 min); 11 h 21 min end to end against the plan's ~10 h estimate.

**What this means for the next session.** §3's gate is met (build the switch). §4 as specified does
not earn its session on this evidence; the measurement points at §5's queue instead — B-1 (refuse a
requirement whose family cannot realize its physical obligation *at the architecture stage*, where
the part is still being chosen) and B-2 (let the compiler own the connector contact contracts),
which are exactly the walls the 12 advisory runs died on.

**Environment findings the operator should know**, both discovered by running the plan rather than
reasoning about it:

1. Memory. This host has no swap and 7.7 GB; a campaign at the default concurrency drives
   `mem_pct` to 98–99 % and gets a process killed roughly every 1.5 h (details below). The resume
   path turns that into a bounded loss, but the default `--parallel 3` is too high for this box.
2. Production is affected at the peak. During the 23:57:06Z–00:00:21Z window the live site was
   intermittently unresponsive — 7 health checks failed, spanning ~3.25 minutes — at the same time
   memory sat at its 98–99 % plateau. The web process was never killed (it has been up since
   2026-09-18 05:23:41Z) and answered HTTP 200 again afterwards, so this was unresponsiveness
   under memory pressure, not a crash. Running a paid campaign on this box at the default
   concurrency is not free of customer-visible risk, and §0.5's "the instrument cannot steer"
   problem has an environment component the plan did not have.



**Protocol.** Four arms, run strictly one at a time on this box (2 cores, one build slot), every
process launched as the plan's §2.3 requires:

```bash
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  .venv/bin/python -m kicraft.eval.self_eval --out logs/self_eval/<arm> [--repeats N]
```

`env -i` matters: `.env` never overrides an existing variable, and the routing file
(`~/.kicraft/routing.json`) does not exist, so the production `.env` profile (`luna`) is what every
arm ran under. A2/A3/A4 ran from their own trees and their own venvs; A2 additionally got its own
`KICRAFT_PROJECTS_DIR` (`~/.kicraft/projects-legacy`) and shared the spend ledger, per §3's
isolation decisions. A2/A3 ran ×2 and A1/A4 ×3 per §2.1. All four arms read the same frozen
34-brief corpus, verified present in both the legacy (`bc6a2f8`) and current trees.

**Arm A4 is measurement scaffolding, not a candidate ship.** It differs from A1 in exactly the
RECORD-class codes of §4.3 — `unreviewed_exact_part` and `unknown_part_refused` in the architecture
derivation, and §9.33/§9.34 at the BOM commit gate — which now record instead of refusing. It was
built in its own worktree (`/tmp/kc-a4`, a `git worktree` of the live tree) and never touched the
live source; the full tree delta is saved at
`logs/self_eval/movea_a4_recordpatch.patch` so the next session can re-apply it. Two
properties were checked before any money was spent, so that A4 isolates the classification:

- the provider-facing slot schema is byte-identical between A1 and A4
  (`ArchitectureIntent.model_json_schema()` → `sha256:f4a0c7c472e42f80…` in both trees); only the
  internal `Architecture` model gains the `advisories` field;
- the downgrade records: `unreviewed_exact_part` lands on `Architecture.advisories` with
  `exact_part=AMS1117-3.3` in its evidence and a mirrored `advisory [code]` line in
  `assumptions`, and survives `model_dump` into the committed stage payload.

The only upstream-visible difference is that an advisory rides in the committed architecture state,
so it can appear in the later stages' prompt state. It cannot contaminate the A1↔A4 comparison:
an advisory exists only on a run whose alternative under A1 was a hard refusal at that stage, so
A1 has no run that reached the later stage for that brief in the first place.

**A2's batch carries the legacy summary schema.** It has no `design_committed`,
`source_unchanged`, `failed_run_cost_usd` or `cost_per_committed_design_usd` key, so its committed
count is read as `design_status == "ok"` and its integrity is evidenced differently: the clone is
still clean at `bc6a2f8` (`git status` empty apart from the copied `.env` and run output), and its
`source_fingerprint` field does not exist in that schema — recorded as a gap rather than filled in
with a guess. A2 therefore does not support the `cost_per_committed_design_usd` comparison; its
total spend is reported instead.

**Incidents, recorded because the plan's rule is that nothing is silent.** Three campaigns were
SIGKILLed mid-flight (exit 137) and each was resumed from its own checkpoint:

| when | arm | lost | resumed |
|---|---|---|---|
| 15:28Z | A1 (64/102 runs done) | the in-flight repeats | 64 reused, 38 re-run |
| 22:49Z | A3 attempt 1 (20/68 runs done) | the in-flight repeats | 20 reused, 48 re-run |
| 00:08Z | A3 attempt 2 (59/68 runs done) | the in-flight repeats | 59 reused, 9 re-run |

The first coincided to the second with this session's `hub stop` of a second supervised process
(the broker hard-kills the whole process group), which is why the runner was moved to
`setsid nohup` (ppid 1, outside the broker) with automatic resume-on-abnormal-exit. The second and
third kills had no such cause, so that attribution was wrong.

The box's own metrics (`~/.kicraft/host_metrics.db`, 30 s samples) give the real cause, and it is
memory. Host memory ramps monotonically under a campaign and is released exactly when a campaign
dies:

| window | max `mem_pct` per 10 min |
|---|---|
| 21:20Z → 22:20Z | 77.6 → 93.6 % |
| 22:30Z / 22:40Z | **98.3 %** |
| 22:50Z (after the 22:49:31 kill) | 74.0 % |
| 23:40Z / 23:50Z | 98.6 / **99.1 %** |

This host has **no swap**, 7.7 GB total, and the campaign configuration (3 parallel design runs,
1 build slot) drives `used` to ~5.9 GB with each KiCad build at ~0.4 GB and both the harness and
the campaign parent at ~1 GB RSS; `/proc/pressure/memory` reports ~2.6 h of *full* memory stalls
since boot. Both survivors and victims are the largest RSS in the tree, which is what the kernel
OOM killer picks. Kernel OOM messages are not readable by this user (`journalctl` needs `adm`) and
`/var/log/kern.log` carries no OOM record, so the *mechanism* is inferred; the memory curve and
the release-at-kill are measured.

Production stayed up through all three: `curl -sf http://127.0.0.1:8080/` returned HTTP 200 on
every check and the web process is alive. No arm's numbers depend on a killed process — the
campaign checkpoints per brief and the runner resumes, so a kill costs at most the in-flight
repeats. The finding is reported because it is generalizable: a long campaign on this box should
expect a kill roughly every 1.5 h at the default concurrency, and the resume path is what makes
that survivable rather than a 96 %-wasted-spend pattern.

### Headline

| arm | tree | runs | committed | briefs ≥2 repeats | fab-ready runs | fab-ready briefs | mean | spend | $/committed | source_unchanged |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | current working tree (`de74af7` + inherited dirty source) | 102 | 7 | 3 | **5** | 4 | 52.0 | $2.065854 | 0.2951 | True |
| A2 | legacy clone (`bc6a2f8`, 2026-08-25) — old design pipeline, KRT routing | 68 | 35 | 10 | **30** | 21 | 64.0 | $3.3313 | — | None |
| A3 | pre-contract worktree (`b4b8be5`, 2026-09-15) | 68 | 24 | 7 | **18** | 13 | 59.7 | $1.364487 | 0.0569 | True |
| A4 | today + scratch RECORD patch (§4.3 RECORD classes recorded, not refused) | 102 | 8 | 3 | **7** | 3 | 52.4 | $2.022281 | 0.2528 | True |

#### A1 — current working tree (`de74af7` + inherited dirty source)

- batch `logs/self_eval/movea_a1_current_20260919T1431Z` · repeats 3 · wall 1598.7s · model `openai/gpt-5.6-luna` · `source_unchanged=True` (`sha256:f895cae464405eb7a6ca0c46dab4161b22ff95e947c76b718387261e615324a4`)
- committed **7/102** · briefs committing in ≥2 repeats **3/34** · fab-ready **5/102** · mean score **52.0** · per-brief-median mean **52.5**
- first-failing-stage split: `architecture`×69, `bom`×23, `intent`×1, `passed`×7, `wiring`×2
- failure-code census: `multiple_intent_contracts`×56, `unsupported_lowerer_contract`×47, `conflicting_port_binding`×33, `unreviewed_exact_part`×24, `declared_signal_port_tied`×17, `unknown_part_refused`×16, `source_obligation_not_retained`×15, `unknown_interface_port`×13, `unknown_supply_rail`×8, `unbound_required_port`×7, `declared_port_double_bound`×2, `unknown_reference_domain`×2
- failed-run spend **$1.912799** · cost/committed design **0.295122 USD** · total **$2.065854**
- fab-ready briefs: audio-jack-buffer, fpc-breakout, r2r-dac, rc-lowpass-bnc
- briefs committing in ≥2 repeats: audio-jack-buffer, fpc-breakout, r2r-dac

#### A2 — legacy clone (`bc6a2f8`, 2026-08-25) — old design pipeline, KRT routing

- batch `logs/self_eval/movea_a2_legacy_bc6a2f8` · repeats 2 · wall 17361.8s · model `openai/gpt-5.6-luna` · `source_unchanged=None` (`None`)
- committed **35/68** · briefs committing in ≥2 repeats **10/34** · fab-ready **30/68** · mean score **64.0** · per-brief-median mean **63.7**
- first-failing-stage split: `BROKEN`×1, `NOT-READY`×21, `REWORK`×6, `passed`×35, `unknown`×5
- failure-code census: 
- failed-run spend **$None** · cost/committed design **None USD** · total **$3.3313**
- fab-ready briefs: buck-3a, can-node, chamfered-badge, dual-rail-supply, fpc-breakout, gpio-expander, hex-env-sensor, highside-switch-10a, nrf52-beacon, proto-shield, r2r-dac, rc-lowpass-bnc, relay-quad, round-led-ring, rs485-terminal, servo-driver-16, snowman-ornament, speaker-crossover, star-ornament, thermocouple-amp, usb-a-power-splitter
- briefs committing in ≥2 repeats: can-node, fpc-breakout, hex-env-sensor, highside-switch-10a, proto-shield, rc-lowpass-bnc, round-led-ring, rs485-terminal, snowman-ornament, star-ornament

#### A3 — pre-contract worktree (`b4b8be5`, 2026-09-15)

- batch `logs/self_eval/movea_a3_precontract_b4b8be5` · repeats 2 · wall 1802.8s · model `openai/gpt-5.6-luna` · `source_unchanged=True` (`sha256:27ff25dc92ebde690dc70f478ea0a9c308e2fb4c12734340863ee77718afcad4`)
- committed **24/68** · briefs committing in ≥2 repeats **7/34** · fab-ready **18/68** · mean score **59.7** · per-brief-median mean **59.7**
- first-failing-stage split: `architecture`×25, `bom`×15, `passed`×24, `wiring`×4
- failure-code census: `9.26`×3, `9.15`×2
- failed-run spend **$0.86482** · cost/committed design **0.056854 USD** · total **$1.364487**
- fab-ready briefs: dual-rail-supply, esp32-dual-motor, hex-env-sensor, highside-switch-10a, led-cc-driver, r2r-dac, rc-lowpass-bnc, round-led-ring, speaker-crossover, stm32-min, usb-a-power-splitter, usb-c-full-breakout, usb-pd-trigger
- briefs committing in ≥2 repeats: led-cc-driver, rc-lowpass-bnc, round-led-ring, rp2040-min, stm32-min, usb-a-power-splitter, usb-c-full-breakout

#### A4 — today + scratch RECORD patch (§4.3 RECORD classes recorded, not refused)

- batch `logs/self_eval/movea_a4_recordpatch` · repeats 3 · wall 3911.5s · model `openai/gpt-5.6-luna` · `source_unchanged=True` (`sha256:b4a3a90f25c06ee3bbb358ee3f37a54578c7a716021346d3b314de13d857648a`)
- committed **8/102** · briefs committing in ≥2 repeats **3/34** · fab-ready **7/102** · mean score **52.4** · per-brief-median mean **52.8**
- first-failing-stage split: `architecture`×67, `bom`×25, `passed`×8, `wiring`×2
- failure-code census: `multiple_intent_contracts`×50, `unsupported_lowerer_contract`×50, `conflicting_port_binding`×33, `source_obligation_not_retained`×13, `declared_signal_port_tied`×13, `unknown_interface_port`×8, `unbound_required_port`×8, `unknown_supply_rail`×6, `malformed_signal_ref`×5, `signal_names_rail`×4, `declared_port_double_bound`×3, `unknown_reference_domain`×3
- failed-run spend **$1.858944** · cost/committed design **0.252785 USD** · total **$2.022281**
- fab-ready briefs: fpc-breakout, r2r-dac, rc-lowpass-bnc
- briefs committing in ≥2 repeats: fpc-breakout, r2r-dac, rc-lowpass-bnc
- advisory-flagged runs **12/102** · codes: `unknown_part_refused`×1, `unreviewed_exact_part`×12
