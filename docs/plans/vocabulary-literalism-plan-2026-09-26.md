# Vocabulary literalism — next session's plan (written 2026-09-26)

Paste this as the first message of the next session, or start from the living plan
(`docs/plans/live-debug-skill-plan-2026-09-25.md`), which points here.

## 0. The owner's instruction, verbatim

> *"yes tackle vocabulary literalism next and to me this sounds like exactly the type of task that Jev
> is built for. consider leaning on Jev to help clarify ambiguity, explore multiple paths, test
> empirically and select the best based on evidence. for now dont do any major implementation, i want
> you to write this to a plan for the next session."*

So: **no implementation this session beyond the plan.** The next session does the work below.

## 1. What vocabulary literalism is

A check decides something true or false about a design by looking for particular *words* in prose,
instead of by looking at the thing the words describe. Two symptoms, and the second is worse:

- **A valid design is refused** over wording (costs a repair round, teaches the model to satisfy the
  checker rather than the circuit).
- **A real thing the owner asked for is silently dropped**, because the brief said it in other words
  (no refusal, no question — just a board missing something). *Silent misses get equal priority: they
  are invisible, so nobody complains about them.*

Live evidence, in the owner's own words from the seed-43 run:

> *"our system of checks seems to have gotten too wound up tight and restrictive that it is erroring
> on a valid design over semantics … we need to design this so it makes a valid design in one shot the
> first time through usually. retries are ok to avoid failures but they shouldnt be the foundation."*

## 2. The inventory (verified in the tree, 2026-09-26)

Line references are to `kicraft/design/stage_semantics.py` unless stated otherwise. "Code" is the
diagnostic it feeds where known; `[?]` means confirm the code name in the session.

### 2a. Vocabulary lists that decide refusals

| # | Where | The list | What the check actually intends | How it can be wrong |
|---|---|---|---|---|
| V1 | `_TOPOLOGY_RE` :33-39, used :831 | `ldo\|buck\|boost\|flyback\|charge pump\|esd protection\|direct pwm\|pwm\|dac\|amplif(ier\|ied)\|analog audio\|single-wire\|ws2812\|\d+x\d+\|5v (power\|supply) rail\|powered directly` | "the spec must not choose a technology the brief never asked for" | A *behaviour* described in prose matches a technology word: the live false alarm was a current source into a CDA matching `amplif(ier\|ied)`. The spec's own structured fields (`family`, `role`) are where a technology choice really lives — the prose match is a proxy. |
| V2 | the four-word list :1012 (`"esd protection", "ldo", "buck", "boost"`), same family as `_POWER_RE` :46 | those four words | "a conversion/protection the brief never asked for must be disclosed as an assumption" | A conversion described in other words (e.g. a 12 V→3.3 V step-down written as "derives 3.3 V from the 12 V input") is never seen — so it is never disclosed. This is the seed-43 finding P1. |
| V3 | `_NONFUNCTIONAL_RE` :41, used :998 | `ground\|gnd\|rail\|power[_ -]distribution\|mounting holes?\|decoupling\|crystal\|castellated pads?` | "a mechanical board feature is not a functional block" | `crystal` is in a list of mechanical features, and the *sheet name* is what is matched: a function the sheet is merely named after is treated as a board feature. Earlier session's P7 (the `POWER INDICATOR` false alarm) was this shape, fixed with a role check on one path only. |
| V4 | `_POWER_RE` :46, used :2371 | `power\|vbus\|vcc\|vdd\|3v3\|5v\|1v1\|ldo\|regulat` | "is this name a rail/supply name rather than a signal name" | a signal named `POWER_GOOD`, `REGULATOR_EN`, `VBUS_DETECT` reads as a rail. |
| V5 | `_POWER_ENTRY_WORDS` :458, used :603 | power-entry words | connector vs on-board source | wording decides a wiring property. |
| V6 | :653-661 | `suppl(y\|ies\|ied)\|power(s\|ed\|ing)\|provide(s\|ed)\|feed(s\|ed)` + `both\|external loads?` | "who supplies this rail" | the same rails described differently take different paths. |
| V7 | `_LOAD_WORDS` :1380, used :1482, :1591, :1630 | `motor\|actuator\|solenoid\|heater\|load` | "this clause feeds a load, so the rail must be sized for it" | a load named otherwise ("the pump", "the coil", "the 2 A strip") is not seen. |
| V8 | :1514, :1561 | `convert(s\|ed\|ing)?\|step(s\|ped)? down\|regulat(e\|es\|ed\|or\|ion)?\|buck` | "is this block a conversion?" | a conversion described as "3.3 V from 5 V" or "steps 12 V down to 5 V" — half the vocabulary is there, the other half is not; depends entirely on the verb chosen. |
| V9 | :2308 | `motor\|load\|coil\|actuator` on a requirement *label* | "this requirement needs its own rail" | the label is the pipeline's own field, so this is the least-bad of the family — but a heater labelled `HEAT-1` still slips through. |
| V10 | :2367 | `ldo\|buck\|regulator\|converter\|supply\|input\|sink\|controller\|protection\|connector\|header\|terminal\|battery\|holder` | "is this connector name a power-only name" | naming heuristics in the connector path. |

### 2b. Vocabulary lists that decide *extraction* (the silent-miss side)

| # | Where | The list | What it intends | How it can be wrong |
|---|---|---|---|---|
| V11 | `prototyping_area_requested` (called from `kicraft/design/synthesis/board_features.py`, used :218-240, :722-732) | "prototyping area" and friends | "the brief asked for a pad field; record it" | A brief asking for *"a grid of 2.54 mm holes for soldering"* records nothing, and nothing tells the owner. The board then simply has no pad field. |
| V12 | `external_load_budget_stated` :51-66 | `(5v\|vbus\|hub75\|led string\|external load)` within 80 chars of `\d+ (a\|ma)` | "don't ask the owner for a number they already gave" | a number given in other words ("each panel draws 2.4 A") is not seen — and the pipeline then *asks a question the owner already answered*. Asking for a known fact is the worst UX in the pipeline. |
| V13 | `_EXPLICIT_FACT_RE` :27-32, used :108, :674 | `qfn\|bga\|lqfp\|tqfp\|soic\|usb(-c)?\|i2c\|spi\|qspi\|uart\|gpio\|swd\|jtag\|castellated\|through[- ]hole\|surface[- ]mount\|\d+ (v\|mv\|a\|ma\|hz\|khz\|mhz\|ghz\|mm\|mil\|pins?\|channels?\|pieces?\|pcs?)` | "this brief states a concrete fact" | a fact stated as words ("a couple of amps", "breadboard-friendly") is not a fact; conversely a bare number can match. |
| V14 | `_STACKUP_COUNT_WORDS` :150-186 | four/six/eight + digits | "how many copper layers the brief asks for" | **this one is fine** — a closed grammar over a closed domain, and P2's fix (the pipeline adds the machine-readable row) is the model to copy: extract what you can, then *make the fact structured*. |
| V15 | `quantity_subject_binds`, `reviewed_class_heads`, `_DEMANDED_CLASS_ALIASES` (part_identity/stage_semantics) | the reviewed library's class vocabulary | "which reviewed class does this writer phrasing name" | **this is the model to copy, not to fix**: an explicit phrasing→class table, tested, plus a fallback that *researches the class for real* when nothing matches. Nothing is silently dropped: an unknown class becomes a research job. |

### 2c. The same shape at the reviewer layer

| # | Where | The rule | Intended | Symptom |
|---|---|---|---|---|
| V16 | eval artifact readers (`kicraft/eval/*_artifact_evidence.py`, `evidence_digest.py --evidence grep:case=insufficient`) | "a value counts as evidence only if it appears verbatim in the artifact" | evidence discipline: no invented numbers | a *correct derivation* is rejected — the reviewer refused `VLED = 3.3 − 2.8 = 0.5 V` because no artifact says `0.5`. The owner's original complaint ("the LED is on a 3.3 V rail") is exactly this. Deliberate design choice, so the fix must preserve the discipline: allow a derivation that *names the inputs it used* (`derived_from: [3.3, 2.8]`) rather than allow prose. |

## 3. The design rule (decided, not open)

**Gates stay deterministic and reproducible. No model call may sit in a path that decides pass/fail.**

That said, the fix for a word list is never "more words" as the primary move. In order of preference:

1. **Structure it.** Make the fact a field at the stage that knows it, so the check reads a value, not
   prose. Precedent: P2's copper-layer row; P11's `resolution_id`/lowering unification. A word list
   then becomes only *extraction*, and extraction errors are visible as "no row" instead of as a wrong
   decision.
2. **Ask the intent, over the pipeline's own fields.** E.g. V1's real question is "does it bind a
   component family the brief never asked for?" — `family`/`role`/obligations are structured, so ask
   *those*. Same for V9 (requirement obligations, not the label).
3. **Closed-vocabulary mapping with a fallback** (V15's pattern): an explicit table, a test per
   row, and an action when nothing matches — never a silent "not found ⇒ fine".
4. **Broaden the list** only when the domain is genuinely closed (V14) — and say so in a comment.

**Jev's role: discovery and advisory judgement, never the sole gate.**
`kicraft/server/decision_layer.py` is the client: `Question` (kinds `noul` = probability of yes,
choice, score), `Answer.is_confident(threshold)` where confidence is how far the peak stands above a
uniform distribution, `decide(...)` with a `recorder` so every answer lands in the audit trail, and
`parse_answers` which *drops* a malformed answer rather than guessing it (`KICRAFT_JEV_BASE_URL`, else
derived from the OpenRouter settings). Two precedents already in the tree, and they define the shape to
copy: `kicraft/server/draft_audit.py` — an advisory audit that auto-applies only answers above a
confidence floor — and `kicraft/server/reconciliation.py` — confidence-floored decisions taken during
repair, each one recorded.

That is exactly the hard part of this task: is this phrase a technology choice or a behavioural
description; does this sentence name a part or a family; are two differently-worded statements the same
fact? But for *this* work its answers become **cached labels in the repo** (committed as data), from
which a deterministic rule is derived and then pinned in a test — the shipped gate gains determinism,
not a dependency. Using Jev inside the gate on every run would make the gate non-reproducible and
pay-per-click; using it to *discover* the rule, then freezing the rule, is the point.

## 4. Method (explore paths, measure, then choose)

**Step 1 — the instrument, first.** A read-only dev script (`tools/` or a `kicraft.eval` entry point)
that walks `logs/self_eval/**` (26,778 JSONs, 162 run dirs, each with `.kicraft/state.json` plus
`stage_status[stage].diagnostics` and `attempts` / `repair_attempted` / `rounds`) and replays each
recorded candidate through the checks. Outputs:

- firing frequency per diagnostic code, and per code: how many of those runs committed anyway (the
  false-refusal proxy) vs. how many needed a repair round (true signal);
- the **baseline KPI**: mean repair rounds and repair attempts per stage — the number the owner's
  rule is about ("retries shouldnt be the foundation");
- a saved corpus of (brief, candidate, check, outcome) tuples for the checks under study.

Also available, already built: `deploy/verify-design-canary.sh` (34 curated briefs, live provider,
all-green gate), `tuning_corpus/` (29 briefs), `kicraft/eval/corpora/`, the five live sessions under
`~/.kicraft/**` (including seed-43, which is *known-valid* and must never be refused).

**Step 2 — one check at a time, five lines each.** For the top three by frequency: (a) the intent in
one sentence; (b) two or three candidate rules (structure / ask-the-fields / broaden-with-fallback);
(c) the corpus measurement for each (newly refused valid designs, newly caught real defects, changed
repair rounds); (d) Jev's labels for the ambiguous cases, with inter-rater agreement reported;
(e) the selected rule, stated as a decision with its number.

**Step 3 — the ambiguous cases go to Jev.** A dev-side script sends typed questions over cached
candidates (never over the live pipeline), e.g. *"Does this functional-spec block choose a technology
the brief never asked for? yes/no + probability + what wording in the brief supports it."* Cache the
answers under `kicraft/eval/corpora/`; splitting a derivation set from a hold-out set; report
agreement, and derive the deterministic rule only from the derivation half. If Jev's agreement is not
better than the word list's, that is itself the decision: keep the list, document why.

**Step 4 — implement the winner with tests.** Each rule gets a test pinning an equivalent *valid* form
it now accepts, and (where it is a refusal) one pinning a genuine defect it still refuses — the same
both-directions discipline as §9.42's fix. Silent-miss fixes (V11, V12) get a test that the fact is
*present* afterwards.

**Step 5 — verify end to end.** Full suite; `deploy/verify-design-canary.sh` unchanged-green and no
rise in the canary's repair-round count; the seed-43 board replayed through the changed checks
(known-valid: must stay accepted).

## 5. Session deliverables

- `tools/…` replay harness + its output artifact (frequency/repair table, saved JSON).
- Per-check decisions: intent statement, candidate rules, measured numbers, chosen rule.
- Patches + tests for the chosen rules (start with V2 then V1 — highest frequency, clearest intents).
- Cached Jev labels + the agreement numbers, committed as data.
- The living plan's §11/ground-rules table updated with what was fixed and what was deliberately left
  as a closed vocabulary, with the reason.

## 6. Acceptance criteria

A prose-keyed refusal is only "done" when all four hold:

1. its intent is written in the table above in one sentence;
2. a test pins an equivalent valid form it now accepts (or a documented decision that the vocabulary
   is closed and why);
3. the corpus numbers are recorded: valid designs newly refused **0**, real defects newly missed **0**,
   repair rounds not increased;
4. the canary is green and its repair-round count did not rise.

## 7. What not to do

- **Don't broaden word lists as the primary fix.** It trades one silent failure for another; it is
  acceptable only for closed domains, with a comment saying which domain and why it is closed.
- **Don't make a model call the *sole* decider of validity.** Jev deciding among options the pipeline
  offers (confidence floor + audit record, as `draft_audit.py`/`reconciliation.py` already do) is fine and
  established; Jev being the only reason a design is accepted or refused is not. For this task its
  output is data, cached in the repo, and the gate reads values.
- **Don't rewrite `stage_semantics.py` wholesale.** One check, one measurement, one test at a time.
- **Don't touch the reviewed-class vocabulary** (V15): it is closed *on purpose*, and it already has the
  right fallback (research the class for real). Copy that pattern; do not "free" it.
- **Don't change the eval's verbatim-evidence rule** (V16) into "accept prose". Allow named derivations.

## 8. First actions

1. `git log` for the last state; read `docs/plans/live-debug-skill-plan-2026-09-25.md` §11 + ground rules.
2. Build the Step-1 harness and print the frequency/repair table. Nothing else before that table exists
   — the plan's whole point is to choose by evidence, and this is the evidence.
3. Take the top check, write its five lines, send the ambiguous cases to Jev, measure, implement, test.
4. Report in plain words: what the check was really asking, what it does now, the numbers before and
   after, and which checks were deliberately left alone.
