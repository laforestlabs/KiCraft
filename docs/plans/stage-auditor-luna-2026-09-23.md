# Stage Auditor on luna + live rewriting audit — plan

**Status:** Workstream **A shipped to production 2026-09-23** (see A7 for the evidence).
Workstream **B proposed, not started** — execute it in a later session.
**Owner decision source:** direct instruction on 2026-09-23 — auditor moves to
`openai/gpt-5.6-luna` at medium thinking, across kicraft.io, including the new
functionality; the new functionality is the **rewriting** auditor (option D), not a
flag-only one. Follow-up instruction the same day: **drop the pre-ship bake-off gate and
ship the swap** (see A4).

Read this whole document before touching code. Two independent workstreams:

- **A — auditor model change** — done; the sections are now a record plus rollback notes
- **B — the live rewriting auditor** (the actual feature; depends on A only for the model)

Workstream B is not implemented. `.kicraft/state.json` in
`/tmp/kc-surprise-seed2` holds a clean, **uncommitted** intent candidate; it is a useful
fixture and must survive until B is built.

---

## 1. Why now — the two defects that motivated this

Both found live on 2026-09-23 driving the production pipeline with a Surprise-me brief
(`seed=2`: *"An ESP32-C3 module controller board powered from a 24 V DC screw terminal,
with a 6-pin 0.1 inch header for I/O, an SPI header, and an RGB status LED. Make the input
reverse-polarity protected."*).

**D1 — a correct repair was silently thrown away.** The first candidate named a part class
`microcontroller-module`; the deterministic detector flagged it and suggested the reviewed
spelling `microcontroller`. The repair round produced `microcontroller`. The driver kept the
**pre-repair** candidate.

Cause: the first candidate was normalized before `diagnose_stage`
(`complete_intent_classification` adds the brief's exact token `ESP32-C3` to `named_parts`)
but the repair candidate was diagnosed raw. So the repair was charged a spurious
`intent_named_part_omitted`, both candidates scored 1, and the guard
`_semantic_defect_score(repaired) >= _semantic_defect_score(current)` discarded the fix.

Fixed on the day: `_normalize_candidate_for_diagnostics()` in
`kicraft/server/stage_runtime.py`, applied to both candidates; regression test
`tests/test_stage_driver_retry.py::test_intent_repair_is_scored_under_first_candidate_normalization`
(fails pre-patch, passes post-patch); 422 tests green.

**D2 — nothing enforces a correct *modelling* choice.** Three drafts of the same brief
modelled the requested reverse-polarity protection three ways: physical
`reverse-polarity-protection`, `conversion` (24 V DC → "protected DC rail"), and physical
`microcontroller-module`. Both the physical and the conversion readings pass every
deterministic check *and* commit through the production gate (probed in throwaway
workspaces). `conversion` obligations are consumed by no stage — the only reference outside
the model definition is prompt text — so a conversion row owns no part and the requirement
survives only as prose.

The conversion reading is the wrong model (a protection is not a voltage conversion) and
can cost a component. No deterministic rule can separate protection/filtering/isolation
from a genuine conversion, and the project deliberately allows unfamiliar part classes
(`gps-module`). **This is the gap workstream B exists to close** — an LLM judges the
nuance, and its rewrite is validated deterministically before it is kept.

Recorded evidence: `/tmp/kc-surprise-seed2/.kicraft/debug/findings.json` (`D1` validated,
`D2` unresolved).

---

## 2. Non-goals

- Not replacing the deterministic detectors. They stay the cheap first line and the
  validator for any rewrite.
- Not touching the end-of-build electrical-review gate's *authority*. Workstream A changes
  only which model answers it.
- Not changing the Class-J eval judge. See §4.4 — a judge sharing the author's model
  grades its own output; that is a different role and a separate decision.
- Not making the auditor mandatory for a board to build. Fail-soft, opt-out, always.

---

## 3. Grounded current state

| fact | where |
|---|---|
| design/author model `openai/gpt-5.6-luna`, route `["openai"]`, caps 0.20/1.20 | `kicraft/server/config.py` `DESIGN_PROFILES["luna"]` |
| reviewer `minimax/minimax-m3`, route `["coreweave/fp4"]`, caps 0.30/1.25, `provider_allow_fallbacks=False` | `config.py` `review_model`, `review_provider_order`, `review_max_price_*`, `for_review()` |
| review thinking is **already** effort `medium` | `config.py` `review_reasoning_effort = "medium"`, `review_reasoning()` → `{"effort": "medium"}` |
| review answer budget 24000, reasoning guard named `electrical_review` | `config.py` `review_max_tokens`, `review_reasoning_guard()` |
| review call sites | `kicraft/design/cli_app.py::_maybe_electrical_review` (5698), review-tab preview (5816), silk-plan authoring (5979) |
| review entry point | `kicraft/design/synthesis/electrical_review.py::review_design_corroborated(corroboration=2)`; severity clamped by `_BLOCKER_ELIGIBLE`; fail-soft |
| stage loop, repair-proposal → validate → adopt-if-better | `kicraft/server/stage_runtime.py::drive_stage`, `_semantic_repair_message`, `_semantic_defect_score`, `_MAX_SEMANTIC_REPAIR_ROUNDS = 1` |
| commit-failure fallback to the pre-repair candidate already exists | `stage_runtime.py` ~4877 (`semantic_repair_adopted = False; obj = original_obj`) |
| normalization helpers reusable by the auditor | `kicraft/design/stage_semantics.py::complete_intent_classification`, `named_part_tokens`, `_EXPLICIT_FACT_RE` |
| budgets | `.env`: project $0.60, daily $20, total $250 |
| measured reviewer cost/quality | `docs/electrical_review_model_bakeoff.md` (minimax-m3 medium: 100% blocker recall, 14% clean-board over-block, ~$0.012, ~159 s, p90 388 s) |
| measured pipeline cost | spend ledger: 5-stage run $0.02–0.04, 85–190 s provider time; intent alone $0.0002–0.0006, 5–33 s |

---

## 4. Workstream A — auditor model → luna, medium thinking

### A1. Change the model in code

`kicraft/server/config.py`:
- `review_model: str | None = "minimax/minimax-m3"` → `"openai/gpt-5.6-luna"`.
- Update the comment block above it: it currently says "Runs the inexpensive DESIGN model
  by default" and cites the 2026-06-19 bake-off. After this change the reviewer **is** the
  design model; say so explicitly and link §A4.
- `review_reasoning_effort` stays `"medium"` — no change needed, but see A3.

`Settings.for_review()` is unchanged in shape (still swaps backend/base_url to OpenRouter,
which luna uses).

### A2. Route the reviewer to a provider that serves luna — **this is a hard failure if missed**

`review_provider_order = ["coreweave/fp4"]` with `provider_allow_fallbacks=False`. luna is
served by `openai` (see `DESIGN_PROFILES["luna"]["provider_order"]`). With fallbacks
disabled, leaving the review route as-is makes every review call fail to route — and
because the gate is fail-soft, the failure is **silent**: reviews quietly stop happening.

Required — **two separate places carry this default, and missing the second one is a silent
failure** (`kicraft/server/config.py`):
- line 553, the dataclass default `review_provider_order = ["coreweave/fp4"]` → `["openai"]`;
- line ~751, the `from_env` fallback `os.environ.get("KICRAFT_REVIEW_PROVIDER_ORDER",
  "coreweave/fp4")` → `"openai"`. This is a **separate literal**, not derived from the
  dataclass, so changing only line 553 leaves the production box (which reads env) on the
  old provider — and with fallbacks disabled, every review call then fails to route.
- Confirm the review price caps (0.30/1.25) are at or above luna's (0.20/1.20) — they are;
  leave them, or align them to the profile if you prefer one number.
- Keep the route fields separate (do not delete them): the point of `for_review()` is that
  reviewer and author can be promoted independently. `KICRAFT_EVAL_JUDGE_PROVIDER_ORDER`
  exists for the same reason on the judge route and is **not** part of this change.

### A3. Make "medium thinking" verified, not assumed

`review_reasoning()` returns `{"effort": "medium"}`. Confirm luna accepts the **effort**
form on this route and does not 400 (the bake-off notes some models accept only one shape).
If luna rejects effort, fall back to `review_reasoning_tokens` and record why.
Also confirm `review_max_tokens=24000` is enough for luna's medium-effort reasoning plus
the JSON answer; do **not** lower the answer budget (reasoning models truncate with an
empty answer, a previously-fixed bug).

### A4. Bake-off — **shipped without it, by owner decision**

**Owner decision 2026-09-23: the pre-ship measurement gate is dropped. Ship the swap.**
The whole pipeline is being stabilised; a model swap was not to be blocked on a bake-off
re-run. The sections below are therefore an **optional post-ship regression tool**, not a
gate. Run them only if reviews later look wrong — and read the result as diagnosis, not
permission.

The harness is still worth knowing about because the reviewer was originally chosen by it,
and because it is the only cheap way to tell "the reviewer got worse" from "the pipeline
got worse":

1. Re-run `scripts/bakeoff_*.py` over the frozen corpus
   (`logs/bakeoff/20260618T200126Z/`: labels, digests, gradings) with
   `review_model=openai/gpt-5.6-luna`, effort `medium`.
2. Record: blocker recall, clean-board over-block (FBRc), warning over-block (FBRw),
   $/board, s/board, p90, JSON-ok rate. The previous arm was minimax-m3 at
   100% / 14% / 20% / $0.012 / ~159 s (p90 388 s).
3. If a regression is real, the mitigation the bake-off already recommends is to tighten
   `_BLOCKER_ELIGIBLE` or raise `review_corroboration` — not necessarily a model swap-back.
   A review that over-blocks is a nuisance; a review that stops running is worse, and see
   A2/A6 for how that happens silently.

### A5. Author/reviewer independence — recognised, accepted

luna reviewing luna is self-review. The bake-off's own worst case was a **shared
hallucination** (17/21 runs agreeing on a defect that did not exist), which corroboration
cannot fix. This coupling is now accepted as a known property of the shipped configuration;
the compensating controls are:
- the review reads only the netlist/BOM **digest**, never the author's transcript;
- the deterministic severity clamp (`_BLOCKER_ELIGIBLE`) still owns what may hard-block;
- `review_temperature=0.5` keeps corroboration passes independent;
- **if a review-quality regression ever appears, suspect the shared model first.**

### A6. `.env` + deploy

`.env` on `kicraft-web` currently sets `KICRAFT_REVIEW_MODEL=minimax/minimax-m3`. Set it to
`openai/gpt-5.6-luna`. Also set `KICRAFT_REVIEW_PROVIDER_ORDER=openai` — the env override
exists (see A2) and **the code default alone does not fix the box**, because the box's
`KICRAFT_REVIEW_MODEL` already overrides the dataclass default.

Deploy per `AGENTS.md`: `git pull`, `pip install -e ".[server,design]"`,
`./deploy/deploy-production.sh`. Then prove the review actually ran: it is fail-soft, so
"no errors" is not evidence. Force one review and confirm a `minimax` → `luna` change in
the ledger.

### A7. Workstream A acceptance — **DONE**, not a to-do

Shipped 2026-09-23 (this session):

- `model-preflight --role reviewer` (console script): `ok=true`, model
  `openai/gpt-5.6-luna`, provider `openai`, real smoke reply, $0.00003.
- luna accepts **both** `{"effort": "medium"}` and `{"max_tokens": N}`; the review route
  sends `{"effort": "medium"}`, so medium thinking is live, verified not assumed.
- Real review through the production gate code path
  (`_maybe_electrical_review`) on a committed 45-part/41-net design
  (`/tmp/m1-repro`): `ran=True`, 2 warnings, 0 blockers, **$0.0020, 9.0 s** — vs the
  minimax baseline of ~$0.012 / ~159 s. The swap is also a large latency win.
- `./deploy/deploy-production.sh` green: web HTTP 200, build worker ready.
- Rollback is a one-line env revert (`KICRAFT_REVIEW_MODEL=minimax/minimax-m3`,
  `KICRAFT_REVIEW_PROVIDER_ORDER=coreweave/fp4`) plus a deploy.

Still open from A, deliberately: the eval judge is **unchanged** (`minimax/minimax-m3`,
`KICRAFT_EVAL_JUDGE_MODEL`) — it is a grader, not an auditor, and pointing it at luna would
mean luna grading luna's own designs. Overrule explicitly if that was not the intent.

---

## 5. Workstream B — the live rewriting auditor (option D)

### B0. Framing: this is the existing repair loop with an LLM defect-finder

`drive_stage` already implements exactly the skeleton a rewriting auditor needs:

```
propose candidate → deterministic normalize → diagnose_stage → severe?
   → if severe: ask for a repair, re-check, ADOPT ONLY IF STRICTLY BETTER
   → else: commit, restoring the original candidate if commit fails
```

Today the *proposer* of a repair is the same author model, told exactly what the detectors
found. The auditor generalizes one thing: **a second LLM call that is given the brief and
the candidate but is NOT told what is wrong, and must find and fix defects itself.** Every
existing guard (normalize → diagnose → score → adopt-if-strictly-better → fall back on
commit failure) then applies unchanged. No new authority, no new mutation path.

Do not build a parallel pipeline. Extend this loop.

### B1. Placement

In `drive_stage`, after the candidate has been normalized and diagnosed, and before commit —
i.e. the same place the semantic-repair loop sits. Order:

1. deterministic `diagnose_stage` (unchanged, cheapest, authoritative for what it covers);
2. semantic repair for deterministic findings (unchanged);
3. **auditor pass** (new): runs whether or not step 2 fired, because its whole purpose is
   coverage the detectors do not have;
4. normalize → diagnose → score → adopt/reject (unchanged machinery);
5. commit (unchanged), with the existing fallback to `original_obj` if commit fails.

`review_before_commit` (the debug path) must gain the same step, so `kicraft-stage-debug`
can show an auditor rewrite for review before anything is committed.

### B2. Inputs — what the auditor may see

- the user's brief (verbatim);
- the stage's own contract (the same text the author got);
- the candidate slot JSON;
- `stage_prep` extras (library leaves, catalog, pinouts) — because judging "is this a real
  part class" needs them.

It must **not** see the author's raw response, reasoning, or the deterministic diagnostic
list. Sending the diagnostics would turn the auditor into a second repair call and destroy
the independence this feature is being built for.

### B3. Outputs — typed, auditable

One JSON object, e.g.:

```json
{
  "findings": [
    {"id": "aud-1", "severity": "defect|risk|note",
     "field": "obligations[6].kind", "issue": "...", "rationale": "..."}
  ],
  "rewrite": {"slot": { ...complete slot... }, "changed_fields": ["obligations[6]", "..."]},
  "rewrite_confidence": "high|medium|low"
}
```

Rules:
- `rewrite.slot` is a **complete** replacement slot, exactly like a stage candidate — never
  a JSON patch. Partial patches against a schema-validated slot are how silent corruption
  happens.
- `rewrite_confidence: "low"` must not be adopted (report the finding, keep the candidate).
- Empty `findings` + no `rewrite` is the normal, cheapest outcome and costs one call.

### B4. Adoption rule — all four must hold

1. **Schema + contract**: the rewritten slot validates exactly as a first candidate would.
2. **Deterministic non-regression**: after
   `_normalize_candidate_for_diagnostics()` → `diagnose_stage()`,
   `_semantic_defect_score(rewrite_severe) < _semantic_defect_score(current_severe)`
   (strictly better; the existing rule, now applied to the auditor too).
3. **Brief coverage does not shrink** (new, and the reason D2 cannot silently lose a
   requirement): every token from `named_part_tokens(brief)` and every match of
   `_EXPLICIT_FACT_RE` present in the current candidate must still be present in the
   rewrite. Any drop ⇒ reject the rewrite, keep the candidate, keep the finding.
4. **Commit succeeds**: on commit failure, restore the original candidate exactly as the
   repair path already does, and record why.

Anything else ⇒ the finding is recorded and the candidate stands. The auditor never blocks
a build on its own.

### B5. Bounds

- **Rounds:** one auditor call per stage. Not a loop. (If a second round is ever wanted, it
  is a separate, explicitly-budgeted change.)
- **Budget:** per-call cap surfaced as a setting
  (`stage_audit_max_tokens`, default sized like the review pass, 8192–24000 with reasoning);
  the stage's existing budget client enforces the run cap. Reject the call if the run cap is
  already exhausted; never let the auditor push a run past `KICRAFT_PROJECT_LLM_BUDGET_USD`.
- **Time:** per-call wall cap; on timeout, fail-soft to the candidate. Use the review
  route's existing shape as the template — `review_wall_stall_s = 360` and
  `review_reasoning_max_tokens = 32768` in `config.py` — and give the auditor its own
  named guard rather than reusing the review's.
- **Fail-soft everywhere:** provider error, unparseable JSON, timeout, budget exhaustion ⇒
  skip the auditor, keep the candidate, log the skip with a reason. A broken auditor must
  never stop a board.
- **Kill switch:** `KICRAFT_STAGE_AUDIT=0` (mirrors `KICRAFT_ELECTRICAL_REVIEW`), plus a
  per-stage allowlist so it can be enabled for `intent`/`architecture`/`bom` first and
  `wiring` later.

### B6. Visibility and override — non-negotiable for a rewriting gate

The reason option D was risky is silent wrong edits. These controls are what make it
acceptable:

- **State:** add `audit_findings` and `audit_rewrite` (before/after + `changed_fields`) to
  the stage's `stage_status` row, and mirror a compact form next to the existing
  `review_findings` for the UI. Never a second copy of the slot: the committed slot is the
  single truth, the audit record is a diff.
- **History:** the stage's `--history-message` (or the driver's own summary) must state that
  the auditor changed N fields and name them. The audit trail must survive without the
  debug artifact.
- **UI:** the stage tab shows "auditor changed: obligations[6] conversion → physical
  (reverse-polarity-protection)" with the rationale and a **revert to the authored
  candidate** action. Revert must be a first-class, one-click operation that re-commits the
  authored slot through the normal gate.
- **Debug CLI:** `kicraft-stage-debug` must display the audit facet like any other review
  facet, and `debug-commit` must refuse to commit an audit rewrite the operator has not
  seen.
- **Ledger:** `stage_attempts.call_mode = "audit"` plus `outcome`, `diagnostic_codes`
  (auditor findings mapped to codes), `cost_usd`, `wall_s`; and on the stage row
  `audit_attempted`, `audit_adopted`, `audit_reject_reason`. This is what makes the adopt
  and harm rates measurable.

### B7. Telemetry worth having on day one

Because the whole justification for D is that an LLM can judge what rules cannot, the
feature is only defensible if it is measured:

- **adopt rate** — share of auditor passes that changed the committed slot;
- **harm rate** — adopted rewrites later judged wrong (by the electrical-review gate, by
  `kicraft-eval-batch` grading, or by hand on a sample);
- **D2 closure** — the protection brief must now commit a physical protection part without
  a human instruction;
- **cost/latency delta** per stage and per run.

A small CLI report (there is precedent: `web-cost-report`, `part-query-report`) or an
extension of one, rather than a new dashboard.

### B8. Validation plan

1. **Scripted-client unit tests** (`tests/test_stage_driver_retry.py` style): auditor finds
   a defect the detectors miss and the rewrite is adopted; auditor rewrite that drops a
   named part is **rejected** (coverage rule); auditor error/timeout leaves the candidate
   untouched; kill switch disables; commit-failure restores the original.
2. **Frozen replay, no LLM cost** — the `verify` skill replays a frozen workspace through
   the real build tail. Use it to prove an auditor rewrite still builds, routes and passes
   the fab gates.
3. **The D2 fixture** — `/tmp/kc-surprise-seed2`: with the auditor on and no instruction,
   the protection obligation must be physical, not `conversion`.
4. **Corpus run** — `kicraft-eval-batch` over the curated briefs with the auditor on vs off;
   compare fab-readiness and rubric grades. This is the honest check for "did rewriting
   improve boards, not just plausibility".
5. **Adversarial case** — feed a candidate with a genuine, subtle electrical error and
   confirm the auditor either fixes it or leaves it; either is acceptable, silent
   corruption is not.

### B9. Workstream B acceptance (definition of done)

- Auditor runs on the allowlisted stages, one call each, fail-soft, behind a kill switch.
- A rewrite is adopted only when all four B4 conditions hold, and the committed slot is
  always a schema-valid, detector-clean-or-better, coverage-preserving candidate.
- The debug path can show and review an audit rewrite before commit.
- The audit record (findings + diff + rationale + revert) is visible in state and UI.
- D2 closed on the fixture: protection commits as a physical part with no human instruction.
- Adopt/harm/cost numbers exist for a corpus run.
- Full suite green; this file updated (the bake-off addendum is optional, see A4).

---

## 6. Cost and latency model (real numbers)

| item | cost | wall |
|---|---|---|
| full 5-stage LLM run today | $0.02–0.04 | 85–190 s |
| intent stage (measured, this session) | $0.0002–0.0006 | 5–33 s |
| one auditor pass — luna (stages run 5–10 s to ~170 s by stage) | ≈ one extra stage call per stage | +≈30–120 s/run |
| one end-of-build electrical review — luna (**measured 2026-09-23**, 45-part design) | **$0.0020** | **9.0 s** (minimax baseline: $0.012 / ~159 s) |

Money is not the constraint: the project cap is $0.60 and the auditor adds a handful of
cents even with rework. **Latency and correctness are.** Therefore: one pass per stage,
per-stage wall cap, fail-soft, and a corpus measurement before enabling it for every stage.

---

## 7. Risks

| risk | why it matters | mitigation |
|---|---|---|
| silent wrong rewrite | the D-option's core danger; user never sees it | B4 coverage + strict-improvement rule; B6 diff + revert + debug refusal |
| self-review (luna authors, luna reviews) | correlated blind spots; bake-off showed a 17/21 shared hallucination | **shipped as-is by owner decision**; deterministic severity clamp still owns hard blocks; suspect this first if review quality regresses |
| routed wrong (luna on a provider that does not serve it) | reviews/audits silently stop (both are fail-soft) | done: A2 route fix + a real review observed at $0.0020 / 9.0 s |
| latency on the user's spinner | 5 passes × a slow model is minutes | one pass/stage, wall caps, stage allowlist, kill switch. Note luna measured ~9 s on a real review, so the latency worry is much smaller than the minimax baseline implied |
| review quality (model swap) | bad boards ship or good boards blocked | **gate dropped by owner decision**; A4 re-run is available as post-ship diagnosis; rollback is one env line |
| scope creep into "auditor fixes everything" | unbounded rewrite authority | findings are the product; rewrites only where B4 holds |

---

## 8. Open decisions for the owner

1. **Stages in scope for B.** Recommendation: enable `intent`, `architecture`, `bom` first;
   `functional_spec` next; `wiring` last (largest slot, most expensive, highest blast
   radius).
2. **Eval judge.** `eval_judge_model` is still `minimax/minimax-m3` and is a **separate
   role** (it grades designs); it shipped **unchanged** in workstream A. Recommendation:
   leave it — a judge on the author's own model grades its own work. If "across
   kicraft.io" was meant to include it, say so explicitly.
3. **Revert authority.** Should the *user* be able to revert an audit rewrite after commit
   (B6), or is the debug/operator path enough for now? Recommendation: user-visible revert
   from day one.
4. **Adoption visibility.** Silent-adopt with a visible diff (current proposal), or
   block-on-audit-change and require confirmation? Recommendation: silent adopt + diff +
   revert; blocking on every audit change would re-introduce the over-block problem the
   bake-off already warned about.

---

## 9. Ordered implementation checklist

**Workstream A — ✅ shipped 2026-09-23, keep the record**

1. ✅ `config.py`: `review_model` → `openai/gpt-5.6-luna`; comment rewritten.
2. ✅ `config.py`: `review_provider_order` → `["openai"]` in **both** places (dataclass
   default line ~560 and the `from_env` literal line ~758); caps (0.30/1.25) cover luna.
3. ✅ luna verified to accept `{"effort": "medium"}` on the review route (and
   `{"max_tokens": N}`; see A7).
4. ⏸️ *(dropped as a gate)* bake-off re-run — optional post-ship diagnosis only.
5. ⏸️ `docs/electrical_review_model_bakeoff.md` addendum — do it only if step 4 runs.
6. ✅ `.env` → `KICRAFT_REVIEW_MODEL=openai/gpt-5.6-luna`,
   `KICRAFT_REVIEW_PROVIDER_ORDER=openai`, `KICRAFT_REVIEW_REASONING_EFFORT=medium`;
   `./deploy/deploy-production.sh` green (web 200, worker ready).
7. ✅ real review observed through the production gate: `ran=True`, $0.0020, 9.0 s;
   reviewer calls visible in the spend ledger on luna.
8. ⬜ *(open)* decide whether the eval judge moves too — see §8.2.

**Workstream B**

8. Add the audit prompt + JSON schema + decode path (`kicraft/design/` alongside
   `stage_semantics`; reuse the stage-contract plumbing).
9. Add `_normalize_candidate_for_diagnostics` + `diagnose_stage` + `_semantic_defect_score`
   wrapping for the auditor rewrite, with the brief-coverage check (B4.3).
10. Wire the pass into `drive_stage` after semantic repair; add `call_mode="audit"` ledger
    fields and the `audit_attempted`/`audit_adopted`/`audit_reject_reason` stage columns.
11. Wire the same step into the `review_before_commit` (debug) path; teach
    `debug-commit` to require the audit facet be shown.
12. Persist `audit_findings`/`audit_rewrite` in state and surface the diff + revert in the
    stage tab.
13. Settings: `stage_audit_enabled`, `stage_audit_stages`, `stage_audit_max_tokens`,
    `KICRAFT_STAGE_AUDIT` kill switch, wall cap.
14. Tests (B8.1), then the frozen `verify` replay, then the fixture, then the corpus run.
15. Report/telemetry for adopt/harm/cost; enable per-stage per B8.

---

## 10. Sources

- `docs/electrical_review_model_bakeoff.md` — reviewer quality/cost/over-block measurements.
- `docs/handoff-general-brief-yield-2026-09-22.md`, `docs/backlog.md` — adjacent pipeline
  context worth reading before starting B.
- `/tmp/kc-surprise-seed2/.kicraft/debug/findings.json` — D1/D2 evidence and the re-run
  artifacts referenced above.
- `AGENTS.md` — production box, deploy and verification commands.
