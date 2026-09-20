# Getting to 34/34 — implementation plan for options 1–4 (2026-09-20)

**Purpose.** Turn the four options from the 2026-09-20 session into a concrete, measurable sequence:
recover the boards the September contract work cost (option 1), fix the parts step where the
survivors die (option 2), put the August pipeline in front of users if wanted (option 3), and only
then let boards ship carrying recorded notes (option 4).

**How to use this document.** §0 is the measured state. §1 is the discovery that reshapes the plan
and must be read before any code changes. §2 is prerequisites that all options depend on. §3–§6 are
the work, one move per session. §7 is the measurement protocol every step uses. §8 is budget and
cadence, §9 is what "done" means and how far 34/34 actually is, §10 is what not to redo, §11 is
what needs the operator.

Companion documents (read them, do not restate them):
- `docs/plans/design-yield-recovery-2026-09-20-handoff.md` — the measured state, target list,
  environment recipes, and the red `$0` gate (§12 there).
- `docs/plans/design-yield-recovery-2026-09-19-plan.md` — the authoritative plan; its `## §2
  Results` is the baseline this work moves, and its §4.3 is the BLOCK-vs-RECORD classification
  option 4 implements.

---

## 0. Where we are (measured)

| version | requests producing a finished board | attempts |
|---|---|---|
| current code (`de74af7` + the inherited dirty files) | **4 / 34** | 102 |
| 2026-09-15 (`b4b8be5`) | **13 / 34** | 68 |
| 2026-08-25 (`bc6a2f8`, before the typed design layer) | **21 / 34** | 68 |
| July's best recorded batch | 25 / 34 | recorded, not re-run |
| current code + the RECORD preview patch | **3 / 34** | 102 |

Same designer (`openai/gpt-5.6-luna`), same router pin, same 34-brief corpus, one arm at a time on
this box. Details and per-arm tables: the handoff §1–§3.

Failure anatomy at the current tree (102 attempts): 95 never produce a design at all — 69 die in the
architecture step, 23 in the parts step, and only 5 of the 7 designs that survive the parts step
finish a board. **The design steps, not the build, are where the requests are lost.**

## 1. The discovery that reshapes options 1 and 2

Options 1 and 2 were framed as two projects — architecture strictness, then the parts step. They are
**one**: the parts-step checks that now bind were added by the *same* commit as the architecture
checks. `git log -S` on this tree:

| check | introduced by |
|---|---|
| `unsupported_lowerer_contract` (architecture) | `0a2e700`, 2026-09-16 17:21 |
| the ten other architecture codes added that day | `0a2e700` |
| `declared-interface-unrealized` (parts step) | `0a2e700` |
| `physical-obligation-unfulfilled` (parts step) | `0a2e700` |
| `check_requirement_physical_realization` / §9.42 (parts step) | `0a2e700` |
| `missing-requirement-implementation` (parts step) | pre-existing (09-09) |
| `model_authored_protected_identity` (parts step) | pre-existing (09-09) |

`0a2e700` is *"Make accepted boards provably correct: contracts, references, artefact evidence"* —
one commit carrying both stages' new demands. Later commits in the same window added three more
architecture codes (`declared_port_double_bound`, `declared_signal_port_tied`,
`unreviewed_exact_part`) and removed `unsupported_supply_port`; 34 commits separate `b4b8be5` from
HEAD, and the yield fell 13 → 4 across them.

**Consequence for the plan.** Do not revert `0a2e700` — it also carries legitimate safety work
(artefact evidence, the reference machinery) and the repo's precedent is to revert *targeted moves*
(`834752a`, `5004173`), never a broad window. Instead, triage its checks individually, architecture
and parts step together, using the census below as the order.

### 1.1 The census to work from

Architecture-step refusals, counted as *attempts whose failure text contains the code*, read from each
attempt's `events.jsonl` (the summary's stored code is a bundle representative and hides the members —
see the handoff §7.1):

| code | current code | 09-15 tree | in `0a2e700`? |
|---|---|---|---|
| `unsupported_lowerer_contract` | **32** | **0** | yes |
| `conflicting_port_binding` | 24 | 10 | no (pre-existing, 2.4× more often now) |
| `unreviewed_exact_part` | 18 | 0 | later in the window |
| `declared_signal_port_tied` | 12 | 0 | later in the window |
| `unknown_part_refused` | 11 | 1 | no |
| `unknown_interface_port` | 6 | 5 | no |
| `unknown_supply_rail` | 5 | 0 | no |
| `unbound_required_port` | 5 | 1 | no |

Parts-step deaths (of 23 attempts that reached it): `physical-obligation-unfulfilled` 13,
`declared-interface-unrealized` 9, `missing-requirement-implementation` 6, §9.42 2,
`model_authored_protected_identity` 2 (a single attempt can carry several).

**Read the census this way:** a code that fires often at HEAD and never on the 13/34 tree is a prime
suspect for *cost*; a code that fires on both is *pre-existing* and belongs to the deeper work, not to
recovery. `unsupported_lowerer_contract` is the extreme case: 32 firings, zero on the tree that
delivered 13 boards.

## 2. Prerequisites (session 0 — no behaviour change)

### P1. Repair the free self-check (`--reference-replay`)

It is **red on HEAD** and it is the only zero-cost way to tell real progress from luck, so nothing
else should be measured before it works. Measured state: 30 rows commit, 2 are recorded blocks,
`buck-3a`'s refusal is its contract's own recorded conflict (allowed), and **`proto-shield` refuses
in the first design step while its contract records no conflict** — so it does not reproduce its
boundary. The row's fixture was last touched 2026-09-16 (`adb7597`) and 37 commits have landed since,
none touching `reference_inputs`; a pristine HEAD worktree fails identically, so this is not the
inherited dirty work.

Method (all cheap — the failing row reproduces in **1.2 s**):

```bash
PY=.venv/bin/python; L=logs/self_eval
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  $PY -m kicraft.eval.design_acceptance --reference-replay --only proto-shield
$L/movea_tools_20260919/proto_diag.py    # same row, workspace kept, so the contract reason is readable
```

Decide, with evidence, which side is authoritative: refresh the row's recorded inputs plus its
deferred and obligation ids (likely — the contract reason names an id and deferred-set mismatch, not
a design defect), or repair the contract change. Then confirm the whole gate green
(`--reference-replay`, ~9 min, $0) and record the row count in the plan document.

**Also worth fixing while here:** the refusing step stores an empty `diagnostics` list, which is why
its message is generic and why the reason string had to be recovered by hand. One diagnostic per
refusal would make every later move cheaper.

### P2. Settle the uncommitted work

`git status` carries three modified files, inherited from the previous session and included in **both**
the current-code baseline and the RECORD preview: `kicraft/design/lowering.py` (a contact-derivation
refactor: `_contact_net_map` / `_contact_nets`, plus `port_contract=` and `requirement_ports=` in
`lowerer_contract_diagnostic`), `kicraft/design/architecture_intent.py` (a rewritten
`conflicting_port_binding` message), `tests/test_design_lowering.py` (3 tests).

That work *is* move 1a's derive half. Decide deliberately — land it, or revert it and start clean —
before measuring anything, because every arm in §0 was taken with it applied. If it is landed, the
baseline does not move (it was already in A1) but the move is done; if it is reverted, the next
campaign must be re-baselined.

### P3. Run discipline

- `--parallel 2`, never 3 (the box has 2 cores, no swap; at 3 it reaches 98–99 % memory and the kernel
  kills the campaign roughly every 90 min — three arms died that way).
- One build slot, both services idle-ish, and prefer off-hours: at the memory plateau the live site
  was briefly unresponsive (7 failed health checks over ~3.25 min on 2026-09-19).
- Run every campaign through a resuming runner (`logs/self_eval/movea_tools_20260919/movea_runall2.sh`),
  and after any abnormal exit check `deploy/check-build-slots.sh` — a leak silently removes build
  capacity (the sweep timer is added but not yet installed; see the handoff §6.6).
- Freeze a baseline copy: **do not** overwrite `logs/self_eval/movea_a1_current_20260919T1431Z`.

## 3. Option 1 — recover the architecture checks (moves 1a–1c)

**Target:** the ~9-board gap between the 13/34 tree and the current tree, taken one check at a time.
**Cheap by construction:** 95 of 102 current attempts fail *before* a build, so a **design-only
campaign at 34×3 (~$1.80)** captures nearly all of option 1's signal; only the confirming run needs
builds.

### Move 1a — `unsupported_lowerer_contract` (32 firings, 0 on the 13/34 tree)

Where: `kicraft/design/architecture_intent.py` (the refusal near the required-port walk) and
`kicraft/design/lowering.py::lowerer_contract_diagnostic`.

The plan's §4.3 already classifies this as a **split**, and that split is the whole move:
- when the draft's own signals name the contacts, **derive** them — the compiler owns this (plan §5
  B-2); the inherited `lowering.py` work is the start of exactly that;
- when the geometry is genuinely outside the family's range (a 15-way terminal, gapped contact
  numbers), **keep refusing** — that is a wrong part, not an unproven one.

Experiment: the scratch-patch method the RECORD preview already validated — turn the refusal into a
recorded note for the derive-able sub-case only, run a design-only campaign, and diff against the
baseline. Expect **≥10 more attempts commit a design** and the brief-level commit count to rise by
**≥3 briefs**.
Kill criterion: no rise in the per-brief median commit count ⇒ revert, record the null, and move to 1b.

### Move 1b — `conflicting_port_binding` (24) and `declared_signal_port_tied` (12)

The first is pre-existing but fires **2.4× more often** than on the 13/34 tree (24 vs 10), which makes
it a *shape* regression rather than a new check: the 09-17 commits `618c433` ("Show every
requirement's port menu, and state the one-port-one-net rule") and `09b4f5b` ("Name the declared-port
tie misuse…") changed what the design steps are told, and the drafts now collide more.

The second is entirely new in the window (12 firings, 0 before). The plan's §5 B-3 is the narrow,
correct fix: **derive** the tie when the compiler already knows which net owns the pin — specifically
when one claimant is a declared rail that feeds the requirement. Two genuinely unrelated nets on one
pin stays a refusal.

Experiment: same shape as 1a, but the expectation is modest and split — most of move 1b may be
prompt/contract wording (cheap, no new logic) rather than a new derive. Measure the two codes
separately so a null on one does not hide a gain on the other.

### Move 1c — the rest of the window's codes

`unreviewed_exact_part` (18) is the one case where the RECORD preview already answered the question:
**downgrading it bought nothing**, because the 12 attempts it rescued all died at the parts step. Treat
it as evidence, not as a target: it belongs to option 2's queue (make the parts step see the part the
design emitted), not here. `unknown_supply_rail` (5) and `unbound_required_port` (5) are small; take
them only if 1a/1b leave budget in the session.

## 4. Option 2 — fix the parts step (moves 2a–2b)

**Target:** the 23 attempts that reach the parts step and die, and the 2 of 7 committed designs that
never finish a board. This is the step that converts "nearly finished" into "finished", and it is what
has to work before 21/34 can be passed.

### Move 2a — see the part the design actually emitted

Verbatim from the baseline, and the clearest defect in the dataset:

```
work unit bom-r000 invalid: physical-obligation-unfulfilled=['status:status-led:
requires 1 real led, found 0; the unit emitted: resistor=Device:R mpn=270R | led=Device:LED mpn=green']
```

The design **did** emit an LED. The check cannot recognise it as a real part. That is compiler
bookkeeping, not a model error, and it is the same failure shape as `declared-interface-unrealized`
("needs one identified hardware owner") and `model_authored_protected_identity`. Work: make the
realization checks resolve the emitted entry to a part identity (symbol + MPN → reviewed or orderable
part) before declaring the obligation unmet, and only refuse when it genuinely cannot be resolved —
recording which string failed when it does.

Where: `kicraft/server/stage_work_units.py::_validate_bom_unit`, the physical-obligation and
declared-interface checks, and whatever the identity resolver is called with there.

Expectation: of the 23 parts-step deaths, **≥6 commit a design**, and **≥3 finish a board**.
Kill criterion: fewer than 3 extra finished boards on the confirming full run ⇒ revert, re-derive
from the per-attempt failure text (the plan's "≈0" instruction) rather than widening anything.

### Move 2b — stop demanding parts the library does not hold

```
9.42 requirement physical/interface realization: 6 physical/interface realization contract(s) unproven
  E_PHYSICAL_REALIZATION 'rp2040': requires 1 reviewed 'crystal' physical part
```

An obligation asks for a class the reviewed library has no part for, so no design can satisfy it. Two
legitimate repairs, in the plan's own order: decide it where the part is still being chosen (plan §5
B-1, architecture step, with the satisfying options named), or derive the realizable obligation set
from the reviewed family. What is *not* a repair is silently dropping the obligation — §4.1 of the
plan is explicit that a brief-asked-for feature with nothing implementing it stays a refusal.

Where: `kicraft/design/synthesis/validation.py` (§9.42 / `check_requirement_physical_realization`) and
the obligation bookkeeping in `kicraft/server/stage_work_units.py`.

## 5. Option 4 — ship with recorded notes (moves 4a–4b)

**Only after 1 and 2.** The RECORD preview measured this in isolation and it bought nothing: 12
attempts whose only architecture complaint was a note-instead-of-stop passed that step and **all 12
died at the parts step**, with **zero** boards shipping a note. Relaxing a step moves the failure; it
does not make a board. After 1 and 2 the population changes — the notes then sit on boards that
otherwise finish, which is the case the policy is for.

### Move 4a — land the advisory machinery properly

Start from the preview patch (`logs/self_eval/movea_a4_recordpatch.patch`, validated and
schema-neutral: the provider-facing schema is byte-identical with and without it). It already has:
`ArchitectureAdvisory` on the model, `_advise(...)` beside `_fail(...)` in
`kicraft/design/architecture_intent.py`, the BOM-stage notes for §9.33/§9.34, per-brief
`advisories` in the eval record, and the summary counts. It still needs what the preview deliberately
left out: the rubric gate that counts notes (`kicraft/eval/rubric.yaml`, a `detected_by: script`
gate — never a cap and never a penalty, per the plan's D6), persistence into the board's provenance,
and the four guard tests of the plan's §4.3.1.

### Move 4b — classification, then measure

Implement the plan's §4.3 classification exactly. The boundaries are not negotiable: a brief-asked-for
feature with nothing implementing it, two unrelated nets on one pin, a wrong netlist, and a board
that cannot mate stay refusals. Only *unproven but very likely fine* becomes a note (an unreviewed-but
-plausible part identity; a mounting preference; and the §9.33/§9.34 pair).

Expectation: finished boards rise by **≥3**, and **every newly shipped board carries at least one named
note**. Kill criterion: no rise ⇒ the RECORD list is still misclassified; re-derive from the
per-board failure text rather than widening it.

## 6. Option 3 — the August pipeline in front of users (parallel track)

Independent of §3–§5: it runs a different tree, so it can be done in any session, including in
parallel with option 1 or 2 (but never two campaigns at once).

The design is already written (the plan's §3) and the deployment exists: `/home/kicraft/KiCraft-legacy`
at `bc6a2f8`, its own venv and `.env`, its own projects directory, and it measured **21/34** finished
boards today.

**The prerequisite this session measured, and it is not optional:** that tree cannot talk to the
current model under the production `.env`. Its defaults come from the DeepSeek era — it pins provider
routing to hosts that do not serve the current model and caps the price below it — so **every call
failed in under a second at zero cost**. Its builds are also dead without `--system-site-packages` in
its venv (`pcbnew` lives in the system Python). Run it as:

```bash
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  KICRAFT_PROVIDER_ORDER=openai \
  KICRAFT_MAX_PRICE_PROMPT=0.20 KICRAFT_MAX_PRICE_COMPLETION=1.20 \
  KICRAFT_DESIGN_REASONING_TOKENS=0 \
  KICRAFT_PROJECTS_DIR=/home/kicraft/.kicraft/legacy-projects \
  /home/kicraft/KiCraft-legacy/.venv/bin/python -m kicraft.eval.self_eval --out <abs> --repeats 2
```

Work, in order: the `pipeline` key in `kicraft/server/routing_config.py` (`ALLOWED_KEYS`, persisted
like `active_profile`) and a `Settings` field; the select on `/admin/routing`
(`kicraft/server/routes_admin.py`) labelled with the pinned commit and the trade-off; design dispatch
— the legacy tree must run its own headless driver as a subprocess, because two versions of the
package cannot coexist in one interpreter (`kicraft/server/session.py` drives the chain in-process
today); build dispatch — substitute the legacy interpreter in
`kicraft/server/build_worker.py::JOB_KIND_COMMANDS`; and marking, so every project records which
pipeline built it (project row, provenance, scorecard grouping) — a swap that is not recorded must
fail the guard test.

Acceptance (the plan's §3.4, unchanged): one brief started under `pipeline=legacy` commits five steps
and reaches a finished board through the legacy tree, proved with artifact paths; switching back uses
the current tree; no restart between them; the pipeline is named in the project row, the provenance
and the scorecard. **State the trade-off where the operator will see it:** with one build slot, a
legacy build serializes against a production build, so a user's build can wait up to one build
timeout — and the legacy tree is a fork that will not receive the fixes from §3–§5.

## 7. Measurement protocol (every move, no exceptions)

Inherited from the plan's §6, plus what this session learned:

1. **Repeats.** 34 briefs × 3 minimum; the unit is the **per-brief median**, never a single run —
   identical inputs measured 60 % then 20 % in this repo's own history, and this session saw a brief
   go 2/3 → 0/3 with no code change on its path.
2. **Same session, same box.** Launch every process as `env -i HOME=… PATH=… TERM=xterm
   PYTHONUNBUFFERED=1` (the shell's inherited `KICRAFT_*` variables otherwise silently win, and `.env`
   never overrides an existing variable).
3. **Design-only screening before paid full campaigns.** Option 1's signal is almost entirely
   pre-build, so screen cheap and confirm expensive.
4. **The whole table, every time:** committed designs, briefs committing in ≥2 repeats, finished
   boards (runs **and** briefs), first-failing-step split, failure-code census, failed-attempt spend,
   spend per committed design, `source_unchanged=true`.
5. **Expand bundles before counting.** The summary's stored code is a bundle representative
   (`multiple_intent_contracts` on 43 of 69 architecture deaths at the baseline); the members are the
   actionable content. Read per-attempt `events.jsonl`, not the summary.
6. **Cheap gates first:** the repaired `--reference-replay` ($0), then a one-brief `--only … --no-judge`
   smoke before every campaign — that smoke is what caught the missing `pcbnew` and the legacy route,
   for $0.05 total.
7. **Append results to the plan document** under a new `## §N Results`, and never overwrite a baseline
   batch directory. A session that does not append did not happen.
8. **Regression guard:** the briefs that actually commit at the current tree
   (`audio-jack-buffer`, `fpc-breakout`, `r2r-dac`, `rc-lowpass-bnc`) — **not** the plan's original
   four, which included a brief that does not commit at HEAD.

## 8. Budget and cadence

- **One move per session, one measurement per move**, exactly as the plan's D4 requires: the move
  states its expected delta *before* the run and is reverted if the measurement misses it. A null
  result that is honestly recorded counts as a delivered session.
- **≤ $5 of provider spend per session** (the plan's guardrail; the daily ceiling is $20 and is shared
  with production). A design-only screening is ~$1.80; a full 34×3 campaign ~$2–3. So a session is
  typically one screening plus one confirmation.
- **Wall clock:** a design-only pass is minutes; a full pass that builds boards has taken 1–5 hours.
  Run campaigns off-hours, at `--parallel 2`, through the resuming runner.
- Suggested order and rough shape: session 0 = P1+P2 (no behaviour change); sessions 1–3 = moves
  1a, 1b, (1c); sessions 4–5 = moves 2a, 2b; session 6 = moves 4a+4b; option 3 is one session that can
  slot in anywhere. Re-plan after each measured result — the census will move, and the order should
  follow it rather than this document.

## 9. Definition of done, and how far 34/34 really is

**Done for this plan:** the free self-check is green; each move has a like-for-like before/after
number and an honest null where it failed; finished boards per 34-brief campaign rise over consecutive
campaigns; every shipped board's notes are named and counted; and a swap between pipelines is visible
everywhere it matters.

**Honest trajectory.** 34/34 has **never** been achieved — the best recorded batch is 25/34, and the
plan's own handoff records that the release gate is far away. The measured path is:

| milestone | what it needs |
|---|---|
| 4 → ~13 | recover the checks the September window added (option 1) — cheapest large win |
| 13 → ~21 | the parts step (option 2); or run the August pipeline and get 21 immediately (option 3) |
| 21 → 26+ | pass the best recorded result; requires both the recovery **and** the parts step, plus notes (option 4) |
| 26 → 34 | the long tail: routing failures, unconnected items, design-rule and outline checks, and the briefs that need a part or topology our library does not model. Nothing measured so far says this is impossible, but nothing suggests it is close either |

**The number that matters** is finished boards per 34-request campaign, with the notes each board
carries, rising over consecutive campaigns while boards that ship with a *wrong* or *unbuildable*
defect stay at **zero**.

## 10. What must not be redone

- **Widening the RECORD list.** Measured null (option 4 in isolation bought nothing), and the wall it
  exposed is one step later.
- **One-brief-at-a-time wording fixes, error-class aliases, or a third rewrite of the contract shape.**
  Six such commits moved the aggregate not at all (the plan's §1).
- **Loosening the reference row to make the free gate green.** Decide which side is authoritative;
  the repo's precedent (`834752a`) treats the corpus as the arbiter.
- **Chasing the model.** The August pipeline hits its era's numbers with today's model, so the model is
  not the constraint. Keep the axis available, do not lead with it.
- **Restoring the old router.** Unchanged and unnecessary: all trees pin the same router commit, and
  the July escape hatch needs a runtime this box does not have.

## 11. Still owed to the operator

1. **Pipeline switch (option 3) — in or out, and whether it is automatic.** Manual selection is the
   plan's default. If a current-pipeline design dies, should it be retried on the August pipeline
   automatically? Faster, but two configurations then produce boards with no human choosing.
2. **The uncommitted work in the tree** (§P2): land it or revert it.
3. **Should paid campaigns run on the production box at all?** The measured cost is a memory ceiling
   that kills a campaign every ~90 minutes and a few minutes of site unresponsiveness at the peak.
   Options: run at reduced concurrency off-hours (current practice), move campaigns to a second host,
   or accept it while there are no paying customers.
4. **How long does the August fork live?** Option 3 buys 21/34 today; options 1–2 close that gap on the
   current tree. Carrying both indefinitely has a cost that should be a decision, not a default.
5. **Install the sweep timer** (`sudo cp deploy/kicraft-build-slots.{service,timer} …`, the handoff
   §6.6) so a reboot cannot silently end the leak watch.

## §12 Results — implementation session, 2026-09-20

**What this session was.** The four options as *code*, with the prerequisites settled and every
claim re-measured at $0. **No paid campaign was run**: this box is the production host, the plan's
own §P3 keeps paid campaigns off-hours and the campaign/measurement decision is still owed to the
operator (§8, §11.3). So every number below is a free measurement (`--reference-replay`, a corpus
of real drafts, the unit suite), and each move's *pre-registered paid delta* is handed to the
operator as a command rather than claimed as a result. Nothing here is a re-run of an arm.

### P1 — the free gate is green again (done, verified)

| | before | after |
|---|---|---|
| `--reference-replay` | **FAILED** on `proto-shield`, 30 committed / 2 blocked / 2 refused | **passed: 31/34 rows reproduce their own boundary** (2 recorded blocks reported) |
| the refusal's own record | `diagnostics: []`, message "see the diagnostic" | the refusing contract's code/message/evidence is this stage's one diagnostics row |

The row was stale, not the contract: `proto-shield`'s recorded `functional_spec` reply (last
touched 2026-09-16) predates the 09-18 board-feature capture, so it omits the
`prototyping_area` fabrication obligation the intent commit derives from the brief. The row is
re-recorded with that one compiler-owned row (7 lines); `buck-3a`'s contract-recorded conflict is
still accepted, and nothing else changed. Log: `logs/self_eval/reference_replay_after_1a.log`
(the run that produced 31/34), then re-run on the final tree (§12.5).

### P2 — the inherited work is landed, deliberately

`kicraft/design/lowering.py` (the contact derivation + the reviewed screw-terminal table),
`kicraft/design/architecture_intent.py` (the rail-tie refusal message) and
`tests/test_design_lowering.py` are committed. Rationale recorded in the commit: the work was
present in *every* measured arm (A1/A3/A4), so reverting it would have re-baselined the campaign
for no gain, and it is the derive half of move 1a.

### Move 1a — the derive/refuse split for the compiler-known pin owner

Implemented as the plan's §1 says (one move, not two), on the evidence of what the code actually
refused. Three derives, each where the compiler already knows which net owns the pin:

1. **A domain statement is not a tie.** `supply_rail`/`reference_domain` on a port the design's own
   signals bind is the *signal's domain* ("ground-referenced"), which is what drafts mean by it:
   the compiler takes the signal and records the statement. A **supply port** keeps its rail and
   drops the reference — the precedence this compiler already applies to `supply_bindings`. An
   unrelated net on a pin, or a statement nothing arbitrates, still refuses
   (`conflicting_port_binding`, `declared_port_double_bound`, `declared_signal_port_tied`).
2. **A lowerer publishes its return contact** (`reference_port_keys`): the coin cell's `negative`
   and the TRS jack's `sleeve` are answered with the design's own ground instead of being asked
   for.
3. **A connector contact a signal names after a declared rail is that rail's exposure** — the shape
   `power.rails[net].from_ref` already models — so it binds the rail instead of refusing the signal
   and then refusing the connector again for the contact that refusal left unbound.

Measured on a **corpus of real drafts** (26 per-attempt architecture drafts recovered from the
production projects' `answer_delta` event stream, replayed through the real derivation at $0;
`logs/self_eval/movea_tools_20260919/` shows the extraction idiom):

| observation (distinct codes per draft, 26 drafts) | before | after |
|---|---|---|
| drafts that commit | 3 | **4** |
| `unsupported_lowerer_contract` | 9 | **6** |
| `conflicting_port_binding` | 7 | **6** |
| `declared_port_double_bound` | 3 | **0** |
| `declared_signal_port_tied` | 2 | **0** |
| `signal_names_rail` | 1 | **0** |

**The census needed correcting first, and that correction is a result of its own.** The plan's §1.1
counts attempts whose failure text *contains* a code; the A1 summary's terminal diagnostic is a
bundle (`multiple_intent_contracts`) whose members are the actionable content. Expanded over A1's
69 architecture deaths:

| terminal bundle | runs |
|---|---|
| `source_obligation_not_retained` alone | **11** |
| `unsupported_lowerer_contract` alone | **6** |
| a tie class alone (`conflicting_port_binding` / `declared_*`) | **8** |
| `unreviewed_exact_part` alone | 5 |
| any tie code present (with any others) | 31 |
| `unsupported_lowerer_contract` present (with any others) | 32 |

So `unsupported_lowerer_contract` on its own can convert at most **6 of 102** attempts — the plan's
"expect ≥10 more attempts commit a design" is only reachable for the *bundle* this move addresses
(tie codes + `unsupported_lowerer_contract` + the contact contract), which is what was
implemented. `source_obligation_not_retained` (11, the largest single class) is the architecture
stage's **ownership** rule — the draft must attach each committed obligation to the requirement
that implements it — and is deliberately left blocking (plan §4.1).

### Move 1b — settled inside 1a

`conflicting_port_binding` (24 firings) and `declared_signal_port_tied` (12) were, on the evidence,
not a prompt-wording problem but the same domain-statement misread (1a.1). Measured above. Nothing
was changed in the prompt/contract shape: the plan's §10 forbids another wording rewrite, and the
derive removes the refusal without touching what the draft is told.

### Move 1c — not taken

`unreviewed_exact_part` (18) is now a RECORD-class note (move 4b) and the remaining `1c` codes
(`unknown_supply_rail` 5, `unbound_required_port` 5, `unknown_interface_port` 6) are small, and the
evidence says they are dangling references (the BLOCK class), not bookkeeping.

### Move 2a — the realization checks now see the part the design emitted

`kicraft/server/stage_work_units.py::_requirement_obligation_defects` compared a declared-interface
claim's `pin` against the symbol's pin **numbers** only, so a draft that named the contact the way
the symbol names it was refused for a correct statement. That is 30 of the ~50 BOM defects in A1:

```
mcp23017:vdd: claimed pin 'VDD' is not in mcp23017-soic:MCP23017-E_SO; available pins=['1','10',...]
```

The same symbol publishes `VDD` as pin 9 in the same payload the check already reads, and the
synthesis side already resolved names this way (`validation._declared_port_pin`). A claim is now
resolved by number, else by one unique pin name (case-insensitive), and refused only when it is
neither — naming the string that failed and, when a name matches several pins, which ones.

### Move 2b — the premise did not survive the evidence (recorded, not implemented)

The plan's example ("an obligation asks for a class the reviewed library has no part for") is
**refuted** by the run it quotes. In `logs/self_eval/movea_a4_recordpatch/run_10_rp2040-min__r1`:

```
stage_status.bom = {"attempts": 0, "work_units": 1, "failure_kind": "commit_rejected",
                    "diagnostics": [{"code": "bom_castellation_placeholder", "severity": "fab_gate",
                                     "message": "Board-fabricated castellations were represented as
                                                 assembly headers.", "evidence": ["j2"]}]}
```

The run died on a **castellation-representation gate** over a BOM whose recipe-owned expansion
never reached the stage; the six `E_PHYSICAL_REALIZATION` offenders (`crystal`, `microcontroller`,
`usb-c-connector`, ...) are members of the same commit rejection, and the library **does** hold
those classes (`nx3215sa-32.768k-std-mua-9` carries `crystal`; `rp2040` carries
`microcontroller`). Nothing was derived or downgraded on this premise.

What *was* implemented is the part of 2b the data supports and that P1's rule already requires:
**a rejected commit records every reason it refused.** Before this change the stage status carried
only the semantic diagnostics (`bom_castellation_placeholder`) and the six realization contracts
lived on the transient `retry` event — which is exactly how the plan mis-attributed this death to
a crystal obligation. `_commit_rejection_diagnostics` now puts the gate codes, the errors and the
offenders on the stage's own `diagnostics`, in both the work-unit and the single-payload commit
paths.

### Move 4a/4b — the advisory machinery, and the classification as specified

Landed from the validated preview patch (`logs/self_eval/movea_a4_recordpatch.patch`) plus what the
preview deliberately left out:

- `ArchitectureAdvisory` on the committed architecture, `_advise(...)` beside `_fail(...)`, the
  §9.33/§9.34 BOM notes, per-run `advisories` in the eval record, summary counts, and the
  "Shipped with advisories" table in the rendered report;
- the shared line/marker protocol in `kicraft/design/advisories.py` (the BOM note writer, the
  self-eval reader and the board's promote provenance all read it);
- **persistence into the board's provenance**: `<stem>.provenance.json` now carries
  `advisories: [...]`, so the board itself says what it could not prove;
- **the recording rubric gate** `shipped_with_advisories` (`detected_by: script`). Its cap is
  unreachable (100), so it cannot penalise or stop a ship — no cap, no block, no suppression
  (D6) — and the gate row carries the codes and the count. `meta.version` 2 → 3, hash refreshed,
  `RUBRIC.md` mirror updated.
- **§4.3 A/B applied as written**: `unreviewed_exact_part` and `unknown_part_refused` are RECORD;
  §9.33/§9.34 are RECORD; the three BOM obligation/interface codes stay BLOCK; every architecture
  code the plan lists as BLOCK still raises.

Guard tests (`tests/test_design_advisories.py`) pin both halves: an AST snapshot asserts the
module's `_fail` literals still contain all 28 BLOCK codes while the two RECORD codes appear only
in `_advise`, and the live paths are exercised (the two RECORD classes reach
`architecture.advisories` and the rubric gate; two nets on one pin still refuse end to end).

### Option 3 — the pipeline switch

`kicraft/server/pipeline.py` owns the switch: the pinned commit (`bc6a2f8`), the legacy tree's
paths, the **measured environment the legacy tree needs to reach the current model at all**
(`KICRAFT_PROVIDER_ORDER=openai`, the price caps, reasoning 0 — without these every call fails in
under a second at zero cost, handoff §6.3), the workspace marker, and the provenance fields.

- **Config**: `pipeline` in `routing_config.ALLOWED_KEYS` + `RoutingConfig` + validation
  (`current`|`legacy`), and a `Settings.pipeline` default (`KICRAFT_PIPELINE`). A save needs no
  restart: both dispatch sites re-read it per run/job.
- **Admin**: a "Design pipeline" select on `/admin/routing`, labelled with the pinned commit and
  the measured 21/34, with the trade-off line beside it (a legacy build serializes against a
  production build on a one-slot host; the fork does not receive options 1–2's fixes).
- **Design dispatch**: `session.run_session` writes the marker and, for `legacy`, runs the legacy
  tree's own headless driver as a subprocess (`-m kicraft.server.stage_driver run … --no-build`)
  with its own interpreter, its own cwd and `PYTHONPATH` pinned to the legacy root. Measured: the
  legacy venv's editable install resolves `/home/kicraft/KiCraft-legacy/kicraft` from any directory
  that does not itself hold a `kicraft/` package (a job workspace, the legacy root, `/tmp`), and
  the pin makes the current checkout unable to shadow it whatever the child's cwd is. The result
  rows and status mapping are rebuilt from the workspace's stage status, so the caller sees the
  same shape either way, and an unsupported per-run instruction is reported rather than dropped.
- **Build dispatch**: `build_worker._execute_job` substitutes the legacy interpreter for a legacy
  workspace (from the *marker*, not the config, so a design/build pair can never disagree) and
  pins `PYTHONPATH`; the argv contract is otherwise identical.
- **Marking**: the workspace marker, the `projects.pipeline` column (additive migration), the
  promote provenance, and the eval summary's `pipeline_counts` + rendered line.

Guard tests: `tests/test_pipeline_switch.py` (config round-trip and rejection, the marker as the
build's source of truth, interpreter substitution, the dispatch subprocess's command *and*
environment, the project row, the provenance file, the summary grouping).

### §12.5 Verification run in this session

| check | result |
|---|---|
| `--reference-replay` (34 rows, $0) after 1a | **passed: 31/34** · 2 recorded blocks (floor ≥31 met) — `logs/self_eval/reference_replay_after_1a.log` |
| `--reference-replay` on the final tree (1a + 2a + 4a/4b + option 3) | **passed: see §12.6** — `logs/self_eval/reference_replay_final.log` (run on `9a44ec1`; the commit after it is comment-only) |
| real-draft corpus, 1a before/after | 3 → 4 drafts commit; 6 refusal classes shrink (table above) |
| `tests/` full suite | **4301 passed**, 15 skipped, 1 xfailed, 2 failed — both failures pre-existing and environmental: `test_vendored_bundles_are_not_prototype` (`ams1117-5v0-fixed` still defaults to prototype) and `test_krt_preflight_uses_environment_defaults` (KRT backend unavailable: `No module named 'py_router.startup_checks'`). Both fail identically on the pre-session tree (`b7a379f` worktree). |
| touched test files only (advisories, switch, BOM, intent, routing, lowering) | 315 passed |
| legacy dispatch smoke | `pipeline.describe()` reports `available: true`, `/home/kicraft/KiCraft-legacy`, `bc6a2f8`; the legacy interpreter resolves `kicraft` from the legacy root in a job-shaped cwd; the driver's argv/environment are pinned by `tests/test_pipeline_switch.py` |

**Commits** (in order): `b7a379f` P1 + P2 + moves 1a–1b · `036d8fb` move 2a/2b · `b432205`
moves 4a/4b · `9a44ec1` option 3 · `a8c405b` a comment correction.

### §12.6 What is measured and what is owed

**Not measured (and not claimed):** the plan's per-move paid deltas. Each is one `env -i` command
away, and §7's protocol applies unchanged:

```bash
REPO=/home/kicraft/KiCraft
# Option 1 screening (design-only, 34 briefs x 3): ~$1.80
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  "$REPO/.venv/bin/python" -m kicraft.eval.self_eval --out logs/self_eval/opt1_designonly \
  --design-only --repeats 3
# The confirming full run (builds boards): ~$2-3, off-hours, --parallel 2
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  "$REPO/.venv/bin/python" -m kicraft.eval.self_eval --out logs/self_eval/opt1_full --repeats 3
# Option 3 acceptance: one brief under pipeline=legacy, five stages + a finished board
```

Pre-registered, per the plan: option 1's screen must show the per-brief median commit count rising
(≥3 briefs) or the move is reverted; option 2a must add ≥6 committing designs and ≥3 finished
boards on the confirming run; option 4 must add ≥3 finished boards **with every new board carrying
at least one named note**. The baselines to compare against are frozen and must not be overwritten:
`logs/self_eval/movea_a1_current_20260919T1431Z` (current tree) and
`logs/self_eval/movea_a3_precontract_b4b8be5` (13/34).

**Owed to the operator, unchanged:** §11.1 (option 3 in or out, manual or automatic — this session
implements manual only), §11.3 (may paid campaigns run here at all), §11.4 (how long the legacy
fork lives), §11.5 (install the sweep timer). New, from this session: the `bom_castellation_placeholder`
death of `rp2040-min` is a *real* defect (a castellated edge drawn as an assembly header) and is
now the only named reason on the stage status, but the recipe-owned BOM expansion of that run never
reached the stage (`attempts: 0`, one work unit) — that is option 2's real remaining wall, and it
is a session of its own.
