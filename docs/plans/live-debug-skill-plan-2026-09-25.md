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
3. **Show the answer in plain words**, one piece at a time, with the exact lines that matter and
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
- Plain language is not a preference. The skill now bans the pipeline's internal vocabulary in
  reports and works an example instead: "Say it plainly" in `.agents/skills/kicraft-debug/SKILL.md`
  (added 2026-09-25, at the owner's request, after "facet" reached a report).

## 4. What works

| # | Observation | Evidence |
|---|---|---|
| 1 | Pausing before save and reviewing one piece at a time catches real defects before they reach a board | this session: two misleading defaults, one vanished requirement, one dropped connector demand found before anything was built |
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
| 6 | **A demanded part the library does not carry is refused instead of researched and added** (owner: unacceptable) | **closed 2026-09-25** (`P14`-`P16`). The pipeline now researches the part itself: it searches the offline JLC/LCSC catalog for the demanded class, ranks on the catalog's own description, stock and single-device evidence, vendors the winner's symbol and footprint into the machine-wide parts library, and writes the reviewed record — so the class is covered here and in every later run. It runs before the architecture and parts steps are diagnosed, and by hand as `kicraft research-part <class>`. Verified live on a varistor (10D471K, LCSC C8760, 211854 in stock, real EasyEDA symbol+footprint fetched in 4.4 s). Two seam bugs it exposed are fixed too: the selection lookups could not see 25 already-reviewed portable records (two Schottky order codes among them), and a one-word class query returned nothing from the catalog. |
| 7 | Nothing checked that a named part can take the stated supply voltage — and the run proved it: the draft powered the DRV8833 (datasheet max 10.8 V) straight from the 18 V input. The generator draws its supply value and its parts independently, so it can ask for an electrically impossible pairing | **fixed** (`F9`): the build-time range check now reads the part's own supply domain (`motor_supply_max_v`), and a new architecture-stage refusal names the rail, the voltage and the limit and asks for a regulator that steps it down or a part rated for the input. Verified firing on the real model reply | **fixed** (`F9`); the stage-level check was DEAD until 2026-09-25 (finding P9: it read `power.rails`/`requirement.supply`, which the architecture response contract derives away before diagnosis) — now live and refusing the 18 V-on-DRV8833 draft |
| 8 | The debugger's trace is only as good as the flags the commit path forwards; each gap was found by reading the artifact, not by a test | partially fixed (F1/F1b); no test guards the *class* of gap |
| 9 | Report vocabulary drifts back to internals when the agent is in a hurry | rule restated in §3; needs the owner to keep flagging it |
| 10 | **The architecture step cannot converge.** Four live drafts produced no reviewable answer: stock ladder 3 calls twice, the owner's `preserving,signature,full_feedback` ladder 5 calls twice — and the defect set changed on *every* call (USB data edge with no 5 V rail declared; `usb_dm`/`usb_dp` left unwired; a declared port tied to a rail; a screw-terminal contact spelled "positive"; a header requirement declaring no ports; the prose string "ESP32-C3-based" where a part identity belongs). The ladder does continue while the defect set changes (good), but the stage never lands inside its budget, and each refusal is a *structural contract* that runs before any design-level check | **resolved 2026-09-25** (findings P9/P11/P12): the stage-level rating check was dead (it read fields the derivation removes) and is now live; the correction could not resolve a legitimate refusal (the library has no dual-H-bridge rated for the stated 18 V), so the pipeline now gives an over-rated load its own regulated rail itself, discloses it as a default, and converges — live re-draft: HBRIDGE_RAIL 10.0 V, 0 diagnostics. Earlier state (open, blocking the run); the two follow-ups identified were: (i) the USB-socket rail rule was enforced but never stated in the stage contract — now documented, next run tests it; (ii) the model repeatedly writes prose where a part identity belongs — a repairable detector is the candidate fix |
| 11 | Design-level checks cannot see a draft that a structural contract already refused (the check for the 18 V fault never got to run on the three failed drafts, even though it fires correctly on their text) | consequence of item 10; revisit after the ladder result |
| 12 | Count/consumption bookkeeping: the generation of a stage's answer can consume the whole correction budget on unrelated defects, so a *correct* design can fail for a *different* mistake in the same draft | **fixed** (design-contract rounds): a *changed* contract defect earns up to 2 extra preserving corrections by default, each sanctioning its own call, with the budget scoped so no other lane's ceiling moves. Reproduce-proof in `tests/test_stage_driver_retry.py` (3 calls/fail → 5 calls/commit) |
| 13 | **The research path is broken by configuration, not by the network.** The box has full egress (openrouter.ai, api.github.com, example.com all 200), but the harness's `web_search` providers are all quota-exhausted (Codex 429, ~44 h) or bot-walled from this datacenter IP, and no credentialed provider is configured. Setting `enabledProviders` does **not** change the search chain (`omp search` fails identically in a fresh process) and was reverted. Working now: direct URL fetch (`curl`/`read` against Wikipedia/GitHub/raw.githubusercontent/OpenRouter APIs) and the Chromium browser tool; durable fix is one key (`BRAVE_API_KEY`/`TAVILY_API_KEY`/`EXA_API_KEY`/`JINA_API_KEY`) or a `searxng.endpoint` | **open, owner action** (a key). Workaround in use for all research this session |
| 14 | **Model-shaped work where the answer set is closed is not being asked as a closed question** — the design stages hand a chat model a free-text field (a part class, a family, a part identity) and then spend correction rounds parsing what came back. Jev is exactly the tool for that: it takes state + typed questions and returns typed values with calibrated probabilities, and it is callable **on OpenRouter** as `typesafe/jev-1.13` (input-only pricing $0.042/1M; deliberately absent from the public model list, and it refuses `/chat/completions`, which points at `/api/alpha/decisions`) | **working, but placement still wrong**: `decision_layer` (real Jev client, 7 tests) and `draft_audit` (rulebook as typed questions, 5 tests) are in the tree and wired into both candidate sites of the driver; a live audit of the seed-37 draft cost **$0.0003** and independently flagged the over-rated DRV8833 (confidence 0.90). **But the audit sits after the derivation, and the derivation runs inside decode** — so a draft that the compiler refuses at decode gets neither the semantic checks nor the audit. Measured: the next live draft was refused at decode in all three rounds (`missing_recipe_requirement`, the prose name `ESP32-C3-based`, identical each time, so the design-contract rounds correctly declined to extend it). Next step: audit the **parsed payload before `_normalize_stage_response` derives it**, which is where "before the compiler sees it" actually is |
| 15 | **A brief that asks for a four-layer stack-up got a two-layer board.** The sentence stayed prose in the intent's constraints box, so nothing enforced it; the builder's board stub, its router configuration, its ground pours and its fab export all assumed two copper layers. Fixed on the owner's call (2026-09-25): the stated stack-up becomes the machine-readable board fact the builder reads (a pipeline completion, from the brief's own words, no invention), the seed board declares the count, a stated count no stack-up can carry is refused instead of rounded, the fab package plots the layers the board declares, and a board with inner layers keeps its signals on the outer pair with ground planes on the inner layers. Verified end-to-end: a real build produced a four-layer board with 0 shorts / 0 unconnected, no track on an inner layer, GND planes on both inner layers, and four copper gerbers. | fixed + tested (`P1`, `P3` in `~/.kicraft/debug/surprise-40-3v3-converter-20260925/.kicraft/debug/findings.json`) |
| 16 | **The regression evidence still counts only two copper layers.** `eval/geometry_artifact_evidence.py` treats F.Cu/B.Cu as "copper", so a four-layer board's inner planes are invisible to a self-eval sweep: a future regression that emptied the inner layers would not show up in the evidence. | open, owner's call (it changes what the rubric measures) |
| 17 | **The debugger stopped the owner to ask a number it could have assumed.** Mid-walkthrough it asked how much current the 3.3 V output must supply, instead of defaulting to the production reading and analysing the consequence. Owner: *"you should have auto defaulted then analyzed that choice automatically. i need the live debug skill to become more automated so i can run it in a loop in the future."* | fixed in the skill: *Default, then analyse* (default it the way the live product would, analyse what a wrong default costs and where it would bite, record it, carry on) and a **loop mode** for unattended runs (self-checked review, save each step when its own checks pass, stop only on provider failure / an unrepairable save refusal / an unjustified repair escalation / the run cap, then a summary and a machine-readable last line) |
| 18 | **Two words of my own shorthand reached a report.** The owner was asked to review "facet 1"; the skill had no rule against the pipeline's internal vocabulary. | fixed in the skill: *Say it plainly* (ban on internal words in reports, machine wording only as quoted evidence, and a worked example), with the same wording fixed in this document; owner's request, 2026-09-25 |
| 19 | **A researched part carried no ratings, on purpose** — `part_research.build_record` left `operating_limits` empty because no rating had been read off a datasheet, so every check that reads a record's own limits treated a researched part as unproven: "in stock" without "right for this rail". | **fixed + tested 2026-09-25** (section 9): the catalog's own parametric fields are read for the chosen part and mapped onto the limit names the reviewed records already use (reverse voltage, Vds, capacitance, positions, pitch, temperature pair, a published supply range), each rating citing the catalog field it came from; a parameter no rating can honestly be read from is kept verbatim, and a part with no such fields says `unverified` rather than leaving the field blank |

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

**Done 2026-09-25:** steps 1-4 are mechanical now (`kicraft research-part <class>`, and
automatically before the architecture and parts steps are diagnosed). The record it writes claims
only what was verified — the class match from the catalog's own description, the contacts read from
the vendored symbol, and (since section 9 below) the ratings the catalog's own parametric fields
state. The bundle keeps the `prototype` badge: machine-researched, not human-vetted.

## 7. Open questions for the owner

- Where should this document and the plain-language glossary live: `docs/plans/` (current
  choice) or beside the skill itself?
- How much should the skill do without being asked once a session is running (draft the next
  step automatically, or wait at every boundary)? Current behaviour: pause for approval
  before saving a step, and pause at each piece of the review. **Answered 2026-09-25** (owner:
  *"i need the live debug skill to become more automated so i can run it in a loop in the
  future"*, after the debugger stopped to ask for a number it could have assumed): the skill now
  has two modes -- interactive (unchanged) and loop (unattended: defaults and analyses anything
  the design can absorb, saves each step as soon as its own checks pass, stops only on provider
  failure, an unrepairable save refusal, an unjustified repair escalation, or the run cap, and
  ends with a summary plus a machine-readable line). The rule *Default, then analyse* applies in
  both modes.
- When the parts step finds a class with no carrier, should it (a) research and add the part
  and continue, or (b) stop and show the owner the candidate part first? Owner's directive
  reads as (a) with a visible record.

## 8. Session of 2026-09-25 (second): the stack-up the builder could not build

Workspace: `~/.kicraft/debug/surprise-40-3v3-converter-20260925`, brief
`generate_brief(40)` — *"A 5 V header from the host board to 3.3 V converter with
reverse-polarity protection, a 6-pin 0.1 inch header, and a power LED. Use a four-layer
stack-up."* Running record: that workspace's `.kicraft/debug/findings.json` (`P1`-`P4`).

What the intent draft showed, and what was done about it:

1. **The four-layer request reached nothing** (`P1`). The writer kept it as one line of prose
   and the whole builder was two-layer: the seed board file, the router configuration, the
   ground pours and the gerber list. Nothing in the pipeline read a stack-up request. Owner's
   call: make the builder deliver four layers.
2. **Found on the way** (`P3`): KiCraft's own copper bookkeeping only knows front/back, and
   the router uses *every* enabled copper layer by default. Declaring four layers without
   touching that would have routed signals onto the inner layers and then rebuilt that copper
   on the front layer. Owner's call: planes inside, signals outside.
3. **Fixed and verified** (findings `P1`, `P3`): the stated stack-up becomes the board fact
   the builder reads; the seed board declares it; a count no stack-up can carry is refused
   instead of rounded; the fab package plots the layers the board declares; the router is
   pinned to the outer pair and the ground plane grows to the inner layers. A real build
   then produced a four-layer board with 0 shorts, 0 unconnected, no track on an inner layer
   and four copper gerbers.
4. **Left alone on purpose** (`P2`): "reverse-polarity protection" stays a stated
   requirement for the architecture/BOM stages plus the end-of-board electrical review; no
   gate was added.
5. **Still open** (`P4`, finding 16 above): the writer also expresses a stack-up as a
   free-form board "feature" row that nothing realizes (inert, but a second spelling of one
   requirement); and the self-eval's copper evidence still counts only F.Cu/B.Cu.

Nothing here is committed to git yet: the patches are in the working tree, so the live site
keeps running its current code until the next restart.

### 8a. Same session, later: automation

Asked mid-run to stop pausing for things it can assume, and to become loopable. The skill now
carries *Default, then analyse* (default the way the live product would, then work out what the
default costs, where it would bite, and how to settle it cheaply -- recorded in the run's
`assumptions_taken`), *Modes* (interactive unchanged; loop unattended with self-checked review
and per-step saving), and *Running in a loop* (one workspace per run, hard caps, four stop
conditions, an end-of-run summary and a machine-readable last line, exit status 0 only when
every step is saved). The same wording was carried into this document and `README.md`. A later
interjection added *Keep the owner oriented*: one short plain-language note per state change,
and never a whole run reported only at the end.

### 8b. What the loop run itself turned up

The unattended run of the same brief (workspace
`~/.kicraft/debug/surprise-40-3v3-converter-20260925`, running record in its
`.kicraft/debug/findings.json`) repaired five more pipeline gaps, each with the focused test that
fails without it:

- **P7** a real circuit sheet refused on the word in its name -- `architecture_power_block_as_sheet`
  fired on "POWER INDICATOR" because the name contains "power"; a rail-named sheet is now
  distribution-only only when nothing else lives on it.
- **P8** no buildable block for reverse-polarity protection, a must-have of the brief: added the
  curated block `reverse-polarity-pmos@1` on the reviewed AONR21357 P-channel MOSFET (drain to the
  input, source to the protected rail, gate to ground through a series resistor).
- **P9** the board's own 5 V input contact silently dropped: the duplicate-statement normalizer
  deleted a signal that was the only statement of how a sourceless rail reaches the board; the
  contact now keeps the rail through a tie.
- **P10** the block that owns the support parts was rewritten away: naming the reviewed LED *and*
  the reviewed `status-led` builder still produced a bare LED across the rail (no series resistor,
  and the typed LED current-path check keyed on the family it had been renamed away from).
- **P14-P16** a demanded class the library could not answer dead-ended: the pipeline could research nothing, the selection lookups could not see 25 already-reviewed portable records, and a one-word class query returned nothing from the catalog. It now researches, vendors and records the part itself (see section 5, item 6).
- **P11** a stated count ignored by the part actually built: "header pins = 6" produced a 2-way
  header because only the used contacts sized the part; the stated count now sizes it and the
  unused contacts become no-connects.

Items 17 and 18 above (loop mode, plain reporting) and these five are the session's output; the
findings file carries the exact evidence and validation for each.

## 9. Session of 2026-09-25 (third): the ratings a researched part can prove

Top item of the handoff, done the same night. A researched part used to be an existence proof —
"this part exists, is in stock and its catalog description names the demanded class" — with every
rating left out on purpose. A part's own numbers are what a check needs to answer "is it right for
this rail", so the record now carries them:

- **Where they come from.** The dump already holds LCSC's parametric table for every part
  (`jlcparts.parameters` reads `attributes`; the field is real, e.g. C64982 B340A publishes
  `Voltage - DC Reverse(Vr) = 40V`, `Current - Rectified = 3A`). That is a published number, quoted,
  not a scrape of prose and not a guess.
- **How they are claimed.** 39 real catalog field names map onto the limit names the reviewed
  records already use, so a researched part is read by the same checks as a hand-reviewed one:
  `reverse_voltage_v`, `vds_max_v`, `capacitance_f`, `voltage_v`, `current_a`, `pitch_mm`,
  `positions`, `output_voltage_v`, the `temperature_min_c`/`temperature_max_c` pair, and a
  published *range* such as `Voltage - Supply = 2V~6V` as the `supply_min_v`/`supply_max_v` pair.
  Each limit names its source (`limits_source`: `catalog parameter 'Inductance' = '10uH'`).
- **What is deliberately not claimed.** A lone `Voltage - Supply: 3.3V` is a nominal on an
  oscillator and a maximum on a regulator, and nothing in the field says which, so it is no rating.
  A range becomes a pair only where the pipeline already names one. Nothing is claimed from a value
  with two magnitudes, a tolerance, or a test-condition suffix it cannot separate. Every parameter
  left over is kept verbatim in `unrated_parameters`, and a record with no readable parameters says
  `limits_review.status: unverified` — explicitly, so an empty field never passes for "no limit
  needed".
- **A trap the mapping avoids.** The input-range check refuses a record that states a voltage input
  without the pin it lands on, and a researched record declares no pin: claiming a lone catalog
  maximum under the input names would have turned every researched regulator into a build refusal.
  The pair spelling (`supply_*_v`) is what the hand-reviewed MCU records use and what that check
  reads only when the record also names its supply port. Pinned by
  `tests/test_electrical_invariants.py::test_a_researched_records_catalog_supply_range_is_not_an_input_declaration`.
- **Verified on the live catalog**, not on fixtures alone: a sweep of 120 000 in-stock rows found
  all 39 mapped field names in real use; a Schottky research yields `reverse_voltage_v 40`,
  `rectified_current_a 1`; a varistor `max_dc_volts_v 385`, `max_ac_volts_v 300`,
  `clamping_voltage_v 775`; a 5.08 mm screw terminal `positions 2`, `pitch_mm 5.08`,
  `voltage_v 250`, `current_a 18` — the same numbers the hand-reviewed screw-terminal records
  carry. Temperature ranges read on 99.95 % of rows; what is left out is what the catalog states
  ambiguously.
- **Tests**: `tests/test_part_research.py` (unit conversion, what is kept rather than claimed, the
  milliamp unit its own records use, a range end that drops its unit, a record with nothing to read,
  and a researched rating reaching the checks) and `tests/test_jlcparts_catalog.py`
  (`parameters()` reads the part's own table, and an older dump without the columns degrades to no
  ratings instead of an error).

Still open from the handoff, unchanged: the class claim is still a description match (item 2), the
copper evidence still counts two layers (3), the layout report still reads as three failures (4),
the internal net names still reach the shipped schematic (5), and protection topology is the
owner's choice (6).


