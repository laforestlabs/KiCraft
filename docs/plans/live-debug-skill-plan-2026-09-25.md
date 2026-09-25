# Live pipeline debugging as a skill — working plan (started 2026-09-25)

**Status: living document.** Assembled during the live Surprise-me walkthrough of seed 37
(board `ESP32_ACTUATOR_DRIVER`). It records what the owner asked for, what the run exposed,
what has been fixed, and what still does not work. Plain language on purpose.

## 1. What the owner wants

1. **Live, step-by-step debugging of the real pipeline** — a real brief, the real provider,
   one design step at a time, with a review before anything is saved. Not a rehearsal, not a
   mock, not a summary after the fact.
2. **Plain-language explanations.** Every explanation must be readable without knowing the
   codebase's internal vocabulary ("obligation", "component class", "semantic repair",
   "candidate_review"). Concrete example beats term of art: *"the list says two JST-XH
   connectors but the part is called jst-xh-connector, so the count was thrown away"*.
   The first two reports of this session were rejected for jargon; that is now a rule.
3. **Working together to improve it slowly.** Small, verified fixes at the source as problems
   appear, each with a test, each explained in plain words; no big-bang rewrites; the owner
   decides the trade-offs that change product behaviour.
4. **A running record of what the owner wants and what does not work.** This file.
5. **Missing parts are made, not refused.** Owner's words, 2026-09-25: *"i cant believe that
   the parts selection will fail if it doesnt have that part available... if something is
   missing you do a bit of research and make it! you are AI!"* A demanded part the library
   cannot answer must trigger research and a new real record, not a refusal.

## 2. What a session looks like (proposed shape)

Per design step, in order, with the owner watching:

1. **Say what this step is for** — one or two plain sentences: what it decides, what it is
   not allowed to decide yet, and what it will hand to the next step.
2. **Draft** against the real provider, paused before saving, with the production
   auto-default behaviour (no parking on questions the live product would answer itself).
3. **Show the answer in plain words**, facet by facet, with the exact lines that matter and
   the machine's own complaints beside them. Skip the internals unless asked.
4. **Say what looks wrong and why**, in plain words, each problem as: what the machine did,
   what it should have done, what it costs, and whether it is the machine's slip or the
   pipeline's fault.
5. **The owner says save, or corrects.** Nothing is saved without that.
6. **Fix the pipeline at the source** when the fault is general, add the smallest test that
   fails without the fix, and note it in the findings list. Never hand-edit the machine's
   answer to paper over a pipeline fault.
7. **Keep the running log** (§4/§5) up to date as we go.

## 3. Ground rules the owner set

- Plain language; concrete examples; no internal jargon in status reports.
- The seed reproduces the sentence, not the design: expect variation between drafts of the
  same brief, and never treat draft-to-draft differences as progress or regressions by
  themselves.
- A missing part is a research task, not an error path (§5, item 3).
- Only the owner approves a stage; "looks plausible" is not approval.

## 4. What works

| # | Observation | Evidence |
|---|---|---|
| 1 | Pausing before save and reviewing by facet catches real defects before they reach a board | this session: two misleading defaults, one vanished requirement, one dropped connector demand found before anything was built |
| 2 | Running the real provider with production's auto-default behaviour makes the debug run match what a real click produces | `debug-draft --budget 0.25` with `auto_default_questions=True`; the draft invented the same 18 V entry path the live product would |
| 3 | Deterministic completion (adding a default instead of asking) is stable and invisible when the writer already did the right thing | the power-entry default is a no-op on draft 3 because the writer named one |
| 4 | A shared comparison used by every checker removes a whole class of silent divergence | count matching now lives in one function used by the parts step and the final gate |
| 5 | A dry run of the save path in a scratch copy tells us whether saving will succeed before the owner says yes | three scratch commits; the third caught that the "was corrected" flags were being dropped |

## 5. What does not work (fixed, open, or wanted)

| # | Problem | State |
|---|---|---|
| 1 | Jargon-heavy status reports | fixed in this session: reports are plain language with examples |
| 2 | The saved note blamed the wrong attempt when a step needed two provider calls, and the review record carried the first call's usage facts | fixed + tested (`F1`), verified live |
| 3 | Saving a corrected step recorded "not corrected" | fixed + tested (`F1b`), verified live |
| 4 | Counts people asked for ("two JST-XH connectors", "two screw terminals") were silently not enforced — 613 of 634 saved boards affected | fixed + tested (`F6`); 186 count lines in the saved history now enforce, 248 name a part class their board's list forgot and are refused instead of dropped |
| 5 | A supply voltage with no stated way in left the requirement with no part | fixed + tested (`F3`): one 2-position screw terminal, recorded as a defaulted choice |
| 6 | **A demanded part the library does not carry is refused instead of researched and added** (owner: unacceptable) | **open, owner's priority.** First case: `jst-xh-connector` — the vendored 2-pin XH bundle `b2b-xh-a-lf-sn` (LCSC C158012, symbol + footprint + manifest already in the repo) has no reviewed record, so the parts step cannot use it. |
| 7 | Nothing checked that a named part can take the stated supply voltage — and the run proved it: the draft powered the DRV8833 (datasheet max 10.8 V) straight from the 18 V input. The generator draws its supply value and its parts independently, so it can ask for an electrically impossible pairing | **fixed** (`F9`): the build-time range check now reads the part's own supply domain (`motor_supply_max_v`), and a new architecture-stage refusal names the rail, the voltage and the limit and asks for a regulator that steps it down or a part rated for the input. Verified firing on the real model reply |
| 8 | The debugger's trace is only as good as the flags the commit path forwards; each gap was found by reading the artifact, not by a test | partially fixed (F1/F1b); no test guards the *class* of gap |
| 9 | Report vocabulary drifts back to internals when the agent is in a hurry | rule restated in §3; needs the owner to keep flagging it |
| 10 | **The architecture step cannot converge.** Four live drafts produced no reviewable answer: stock ladder 3 calls twice, the owner's `preserving,signature,full_feedback` ladder 5 calls twice — and the defect set changed on *every* call (USB data edge with no 5 V rail declared; `usb_dm`/`usb_dp` left unwired; a declared port tied to a rail; a screw-terminal contact spelled "positive"; a header requirement declaring no ports; the prose string "ESP32-C3-based" where a part identity belongs). The ladder does continue while the defect set changes (good), but the stage never lands inside its budget, and each refusal is a *structural contract* that runs before any design-level check | **open, blocks the run.** Two follow-ups identified: (i) the USB-socket rail rule was enforced but never stated in the stage contract — now documented, next run tests it; (ii) the model repeatedly writes prose where a part identity belongs — a repairable detector is the candidate fix |
| 11 | Design-level checks cannot see a draft that a structural contract already refused (the check for the 18 V fault never got to run on the three failed drafts, even though it fires correctly on their text) | consequence of item 10; revisit after the ladder result |
| 12 | Count/consumption bookkeeping: the generation of a stage's answer can consume the whole correction budget on unrelated defects, so a *correct* design can fail for a *different* mistake in the same draft | **fixed** (design-contract rounds): a *changed* contract defect earns up to 2 extra preserving corrections by default, each sanctioning its own call, with the budget scoped so no other lane's ceiling moves. Reproduce-proof in `tests/test_stage_driver_retry.py` (3 calls/fail → 5 calls/commit) |
| 13 | **The research path is broken by configuration, not by the network.** The box has full egress (openrouter.ai, api.github.com, example.com all 200), but the harness's `web_search` providers are all quota-exhausted (Codex 429, ~44 h) or bot-walled from this datacenter IP, and no credentialed provider is configured. Setting `enabledProviders` does **not** change the search chain (`omp search` fails identically in a fresh process) and was reverted. Working now: direct URL fetch (`curl`/`read` against Wikipedia/GitHub/raw.githubusercontent/OpenRouter APIs) and the Chromium browser tool; durable fix is one key (`BRAVE_API_KEY`/`TAVILY_API_KEY`/`EXA_API_KEY`/`JINA_API_KEY`) or a `searxng.endpoint` | **open, owner action** (a key). Workaround in use for all research this session |
| 14 | **Model-shaped work where the answer set is closed is not being asked as a closed question** — the design stages hand a chat model a free-text field (a part class, a family, a part identity) and then spend correction rounds parsing what came back. Jev is exactly the tool for that: it takes state + typed questions and returns typed values with calibrated probabilities, and it is callable **on OpenRouter** as `typesafe/jev-1.13` (input-only pricing $0.042/1M; deliberately absent from the public model list, and it refuses `/chat/completions`, which points at `/api/alpha/decisions`) | **in progress, working**: `kicraft/server/decision_layer.py` is a real Jev client (noul/choice/score, probabilities + confidence, typed parsing, 7 tests) and `kicraft/server/draft_audit.py` asks the rulebook as Jev questions (3 tests). Live on the seed-37 draft: one call, 7 questions, **$0.0003**, and it independently flagged the over-rated DRV8833 (`audit_part_over_rating`, confidence 0.90). Wiring the audit into the correction loop, and the class/part-choice repairs, are next |

## 6. Making a missing part (the owner's directive, in mechanics)

When the parts step needs a class the library cannot answer:

1. **Look** for a part the repo already vendors (the XH bundle above is one).
2. **If none exists**, find a real orderable part (manufacturer part number + distributor
   part number + datasheet) that genuinely implements the demanded class, and vendor its
   symbol/footprint the way `kicraft/parts_library/<bundle>/` does.
3. **Add the reviewed record**: MPN, symbol, footprint, contact count, the features it
   carries (so the demand can be matched), and its provenance link.
4. **Add the test** that fails without the record (the demand resolves; the wrong part is
   still refused — e.g. a 2-pin part must not answer a 4-pin demand).
5. **Never invent** a part number, a pin map, or a footprint. Research means citing a real
   part's real datasheet.
6. Prefer the *same* research path in production (a parts step that can ask for an
   acquisition) over a human adding records by hand — that is the capability this skill
   should make routine.

## 7. Open questions for the owner

- Where should this document and the plain-language glossary live: `docs/plans/` (current
  choice) or beside the skill itself?
- How much should the skill do without being asked once a session is running (draft the next
  step automatically, or wait at every boundary)? Current behaviour: pause for approval
  before saving a step, and pause at each review facet.
- When the parts step finds a class with no carrier, should it (a) research and add the part
  and continue, or (b) stop and show the owner the candidate part first? Owner's directive
  reads as (a) with a visible record.
