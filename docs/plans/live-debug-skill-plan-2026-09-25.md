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
- **Checks judge intent, not spelling.** A check must ask what the design *is* — is this pin on the drive path to that rail, does this count hold — not whether the net, word or route matches the one the check happened to expect. A valid design refused over semantics costs a repair round and teaches the model to satisfy the checker instead of the circuit (owner, 2026-09-26).
- **The first pass should usually be valid.** Repair rounds are a safety net, never the mechanism: shape the design right where it is shaped, and let checks confirm it rather than correct it into existence.
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
| 20 | **The parts stage could not be satisfied by either correct answer for a counted connector sheet.** The brief's own count ("two JST-XH connectors") was charged against every requirement carrying it, so a two-connector sheet read `found -1` for one group of two and `found 0` for two single connectors; and the rule that keeps *one connector group per connector requirement* had caps per part, so it silently dropped the second connector. Six repairs exhausted, `unit_repair_exhausted`, stage failed. | **fixed + tested 2026-09-25** (`P2`, `P3`): the design-wide count is compared once against the groups the unit emitted, each requirement then needs one part of its own, and the group rule caps by connector requirement. Live: the same stage went from 6 attempts/failed to 3 attempts/complete, with both J4 and J5 placed. |
| 21 | **An architecture family that builds the demanded class itself was refused as a mismatch.** `switch-input` emits the reset button (`Switch:SW_Push` on the TL3342 footprint), but the check demanded the freshly researched carrier of the `pushbutton` class, spending two repair rounds and asking to replace a working lowerer with a raw part. | **fixed + tested 2026-09-25** (`P4`): the reviewed lowerer-witness table now records that the switch lowerer's graph *is* a pushbutton, and the check accepts a family whose own lowering proves the class. Re-diagnosed on the unchanged candidate: clean. |
| 22 | **The commit gate could not see two things the parts stage could.** A bundle part with no MPN was invisible to §9.42 (`E_PHYSICAL_REALIZATION 'power_led': ... found 0` for the vendored warm-white LED whose pair is exactly the reviewed record's), and the same shared count was summed per requirement there too (`jst1×2, jst2×2 demand 4 distinct part(s)` for a two-connector sheet). | **fixed + tested 2026-09-25** (`P6`, `P7`): a part with no MPN is identified by its own reviewed pair against the whole inventory, and the sheet's count counts each distinct row once and never asks for fewer parts than there are requirements. Live: both complaints gone from the commit. |
| 23 | **The one reviewed CH32V003 order code is out of stock at the LCSC retail storefront**, so a board whose brief names that MCU cannot pass §9.26 and cannot be saved. Same-device alternatives change the package (SOP-16 C5346357, TSSOP-20 C5187096). | **open — owner decision** (`P8`): add a reviewed record and bundle for another variant, point the bundle at a same-device listing if one appears, or accept the stock risk. The run stopped here rather than choosing a package. |
| 24 | **A default the intent recorded is not carried into the design.** "Standard UART pins plus ground" is the intent's own default, yet the architecture declared a two-contact UART header, so the shipped header has transmit and receive and no return. | open (`P5`), with the two honest fixes named: add a contact wired to ground, or send the serial pair to an `edge:` peer whose compiler-built header carries ground automatically. |
| 25 | **A derived function is visible but not marked as a guess.** The functional spec added a power-conversion block (12 V in, 3.3 V logic) without recording it in its assumptions; the check that exists for that duty keys on four literal words (`esd protection`, `ldo`, `buck`, `boost`) and this block said "Convert". | open (`P1`). The machine-readable signal an obvious fix would key on — a block with no requirement attached — does not hold up: seed-40's committed spec leaves its reverse-polarity block unattached for a requirement the brief states in those words, and the replay fixture attaches nothing at all. The durable fix is a marker the pipeline owns. |

| 26 | **A part that is real but out of stock blocked the board.** §9.26 refused a commit for any part whose stock had run dry in either inventory, so a board whose brief names a part that is temporarily unbuyable could not be built at all — no matter that the design was right. | **changed on the owner's call 2026-09-26** (`P8` onward): out of stock is now a *recorded decision*, never a refusal. A part the brief names **by its own order code** is kept (the owner may hold stock or have a second source); a person watching is asked whether to build with it or take an in-stock variation, and an unattended run keeps it. A part the *pipeline* chose for a class the brief names is swapped for an in-stock carrier of the **same family and package**, so no pin or wire moves; a different package is named, never applied silently. The decision, the reading and a substitution ledger entry land on the BOM. Verified live: the seed-43 parts stage saved with the CH32V003 kept and the shortfall recorded, and the run went on to wiring. |
| 27 | **§9.39 could not see a curated recipe's own conversion.** The transfer check read only reviewed *records*' `power_transfer` mapping, and a converter realized by a curated recipe — the sanctioned way to build one — has no record, so a correct AMS1117 board was refused with "no reviewed source-to-load transfer from 'VIN_PROTECTED' to '+3V3'". | **fixed + tested 2026-09-26** (`P9`): the check now adds the recipe's own declared input→output path (its `Port` definitions, bound through the requirement's ports) before walking the transfer graph. |
| 28 | **§9.42 could not see a vendored part with no MPN** — twice over: its declared-interface half required a literal MPN, and then compared the requirement's `exact_part` against that absent MPN. A part the pipeline had just proven was reported as "needs exactly one identity-matched BOM component", hiding the real defect. | **fixed + tested 2026-09-26** (`P10`): a part is identified by its MPN *or* its resolved reviewed identity (the symbol/footprint pair), so the complaint is now the real one — "expected '+3V3' on declared pin selector '1' of D1, found 'GND'". |
| 29 | **An indicator whose declared pin cannot sit on the rail it is bound to** — the architecture names the power LED's `drive` port as the 3.3 V rail and its declared pin as the LED's cathode, while the correct wiring grounds that pin and reaches the LED through its series resistor. The check compared **net names**, so it called a valid circuit a defect. Owner, 2026-09-26: *"you could easily satisfy that if you wanted by moving the series resistor to after the LED but it doesnt matter … our system of checks seems to have gotten too wound up tight and restrictive that it is erroring on a valid design over semantics."* | **fixed + tested 2026-09-26** (`P11`): the declared-interface half now asks whether the declared pin is on the **drive path** — the port's net reached through the requirement's own reviewed two-terminal parts (its interface part and its series element) — instead of demanding the pin sit literally on the port's net. A pin wired to an unrelated net with no path of its own is still refused. No design change, no re-run: the committed architecture and the existing wiring save as they are. |

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




## 10. Session of 2026-09-25 (fourth): a loop run on seed 43, and the part nobody can buy

The first unattended loop run under the new rules, on the brief the live site's next click gets
(`generate_brief(43)`): *"A CH32V003 development board with a 12 V DC barrel jack, a UART header,
two JST-XH connectors, a reset button, and a power LED. Make the input reverse-polarity
protected."* Caps stated before the first call and never raised: **$0.25 per draft, $1.00 for the
run**. Workspace `~/.kicraft/debug/surprise-43-ch32v003-devboard-20260925`; its `findings.json`,
`run-log.md` and per-stage answers carry the evidence for everything below.

**What the run produced.** Three stages committed: the goal and must-haves (five must-haves, seven
typed rows, five recorded guesses), the functional decomposition (seven functions, the two
connectors as one function repeated twice), and the architecture (12 V in, P-channel reverse-
polarity protection, an AMS1117 to 3.3 V, the CH32V003, a UART header, two JST-XH connectors, a
reset button, a power LED, with the rails, nets and bindings derived by the compiler). The parts
list was drafted and refused on sourcing; wiring was not reached. The run also exercised the
capability built earlier the same day: the demanded `pushbutton` class was researched and vendored
for real mid-stage (K2-1109DF-E4SW-04, LCSC C2909684) and its record carries the catalog's own
ratings, so "the part exists" and "the part is right for this rail" now come from one place.

**Five pipeline defects found and fixed at the source, each with the smallest test that fails
without it** (details in `findings.json`, rows 20-22 of section 5):

- **P2** the parts stage charged a count the brief states once against every requirement carrying
  it, so a two-connector sheet could not be satisfied by either correct answer (`found -1`,
  `found 0`) and exhausted six repairs.
- **P3** the rule that keeps one connector group per connector *requirement* capped per part, so it
  silently deleted the second connector of that sheet — a demanded part gone from the board.
- **P4** the architecture stage refused `switch-input` for a `pushbutton` demand although that
  lowerer builds the button itself, and the "repair" it asked for would have replaced a working
  lowerer with a raw part.
- **P6** the commit gate could not see a bundle part with no MPN even when its symbol/footprint
  pair was exactly the reviewed record's, so it refused the LED the parts stage had just proven.
- **P7** the same shared count was summed once per requirement in that gate too: a sheet holding
  exactly the two connectors the brief asks for read "demand 4 distinct part(s)".

**Where it stopped, and why that is the right stop.** With all five fixed, the save reached the
real question: the CH32V003 the library carries (SOP-8, `ch32v003j4m6`, LCSC C5346354) shows **0 in
stock at the lcsc.com retail storefront**, so the board's MCU cannot be bought today.
`kicraft lookup-lcsc-id CH32V003J4M6` confirms it (`retail_stock: 0, min_buy: 5`), and the offline
catalog has no second listing for that order code. Same-device alternatives exist but change the
package (SOP-16 C5346357 and TSSOP-20 C5187096, both in stock). Choosing between a bigger package
and a wait for restock on a part whose SOP-8 already uses most of its six GPIOs is the owner's
call, so the run stopped rather than making it. The retail half of the gate is production
behaviour and was deliberately left on: weakening it would have made the run pass and the board
unbuildable.

**Two findings left open with their evidence** (no fix applied, so nothing is claimed):

- **P5** the intent recorded the default "standard UART pins plus ground", yet the architecture
  declared a two-contact UART header, so the shipped header has transmit, receive and no return.
- **P1** the functional spec introduced the 12 V-to-3.3 V conversion without recording it as a
  guess. The check that exists for that duty keys on four literal words; and the machine-readable
  signal an obvious fix would key on (a block with no requirement attached) was measured and does
  not hold up — seed-40's committed spec leaves its *reverse-polarity* block unattached for a
  requirement the brief states in those words, and the replayed fixture attaches nothing at all.
  The durable fix is a marker the pipeline owns, which is a schema decision for the owner.


## 11. Session of 2026-09-26: a board that gets made even when a part cannot be bought

The owner's directive, in their words: *"if the missing part number was directly called out by the
user in the brief then kicraft should ask the user if they want to continue building with the
specified part (maybe the user has a second source or has some in their own inventory) or if the
part number should be changed to a variation that is in stock … if the run is autodefaulted then
kicraft should automatically choose itself: stick with the out of stock part if it is specifically
named in the brief otherwise swap to an in stock alternative. that goal is to make a valid board at
the end, even if one or two of the parts are out of stock."*

**The rule, in code.** New `kicraft/design/sourcing_policy.py` decides, in one place, what happens
to a part that is real, matches the design and cannot be bought this week:

- **kept** when the brief calls out that part's own order code (a family name is not the part
  number — "a CH32V003 development board" leaves the variant to the pipeline). A person watching
  is **asked** at the parts stage whether to build with it or take an in-stock variation, with the
  variations the pipeline can honour listed; an unattended run keeps it and records the reading.
- **swapped** when the pipeline chose the part for a class the brief names, and a reviewed carrier
  exists in the **same family and the same package** — then every pad number, footprint and wire the
  design already states means the same thing in the replacement. The swap is written as a
  substitution (why the fab BOM differs from the draft) plus a `(defaulted)` assumption.
- **kept, with the reason**, when no such carrier exists — and the reason names the variants that
  *do* exist in other packages, because switching package re-binds the pins and is a design
  decision, never a quiet substitution.

Existence is untouched: a fabricated MPN, a C# the catalog does not hold, or a signature that
contradicts the symbol still refuses the commit. Only *stock* became a decision.

**Proven live, on the run that had stopped.** The seed-43 parts stage — refused the night before on
the CH32V003's retail stock — **saved** with the part kept as designed and this on the BOM:

> `ch32v003j4m6 (CH32V003J4M6) is out of stock at the lcsc.com retail storefront (0 available, min
> buy 5) and the pipeline chose it for a class the brief names, but no in-stock carrier shares its
> package; the library holds no other reviewed variant of this family`

The run then reached the wiring stage, whose first draft was refused by three contracts. Two were
pipeline bugs, found by reading the refusal rather than guessing, and fixed at the source with
tests: **§9.39** could not see a curated recipe's own declared transfer (a converter built the
sanctioned way had no path at all — `P9`), and **§9.42** could not see a vendored part with no MPN,
reporting it as a missing component instead of its real defect (`P10`). The third refusal was real:
the power LED's return went nowhere. The repaired wiring is electrically right — cathode to ground,
anode through its 249 ohm resistor to the rail — and every other gate passes.

**The check that was wrong, and why fixing it was the right trade** (`P11`). The wiring was refused
for one more thing: §9.42's declared-interface half compared **net names** — "does the declared pin
sit on `+3V3`?" — and the LED's declared pin is its cathode, which the correct wiring grounds (the
rail reaches the LED *through* its 249 ohm resistor). The circuit was right; the question was wrong.
The owner's call (2026-09-26) was that this is over-literal:

> *"is it that the architecture stage made a requirement that the status led is on 3.3v rail? i mean
> you could easily satisfy that if you wanted by moving the series resistor to after the LED but it
> doesnt matter. i think the point here is that our system of checks seems to have gotten too wound
> up tight and restrictive that it is erroring on a valid design over semantics. i dont want to pick
> something that caused stages to rerun, its a waste of money and time. we need to design this so it
> makes a valid design in one shot the first time through usually. retries are ok to avoid failures
> but they shouldnt be the foundation of kicraft."*

So the check now asks the question that matters: **is the declared pin on the drive path to the
port's net** — reachable through this requirement's own reviewed two-terminal parts (its interface
part and its series element), not necessarily adjacent to it. Only the requirement's own parts
count (its recipe/lowerer refs, the parts from the same work unit), so a claim wired to an unrelated
net with no path of its own is still refused; the case is pinned from both sides in
`tests/test_electrical_invariants.py`. The committed design saved with **no change and no re-run**,
and the whole board then committed: all five stages.

The rejected alternative was to re-name the LED requirement's family to the reviewed `status-led`
builder so the pin roles become the builder's business. It is the library's own shape and it would
have worked — but it means re-running three stages (~$0.02 and minutes) to satisfy a checker about a
circuit that was already correct, which is exactly the trade the owner ruled out.

**The run finished.** All five stages committed and the deterministic build exited 0:
**BUILD COMPLETE CH32V003_DEV_BOARD** — 0 shorts, 0 unconnected, 166 traces, 12 vias, 15/15
components, and a full fab package (gerbers, drill, CPL, BOM, STEP, 3D render). The MCU on that
board is the order code that is out of stock at retail, kept deliberately and recorded on the BOM —
which is the whole point of §11's rule: the board got made.

The wiring stage's first draft had one genuine defect (the LED's return went nowhere) and needed one
repair round; everything else that refused it was a check reading names instead of intent, and those
are now fixed rather than worked around.

**Cheaper than the failures it replaced:** the whole run cost **$0.035** of its $1.00 cap, and the
two pipeline bugs it exposed would otherwise have refused *every* board whose converter is a curated
recipe and every board with a vendored part in a declared interface.

## 12. Next session: vocabulary literalism (plan written 2026-09-26)

Owner's instruction, verbatim: *"yes tackle vocabulary literalism next and to me this sounds like
exactly the type of task that Jev is built for. consider leaning on Jev to help clarify ambiguity,
explore multiple paths, test empirically and select the best based on evidence. for now dont do any
major implementation, i want you to write this to a plan for the next session."*

The plan is **`docs/plans/vocabulary-literalism-plan-2026-09-26.md`**. It carries the verified
inventory of the 16 places where a check decides by *words* rather than by the thing the words
describe (V1-V16, with file:line), the rule that settles the approach (structure the fact first, ask
the pipeline's own fields second, closed-vocabulary-with-a-fallback third, broaden a list only for a
genuinely closed domain — and never a model call as the sole decider of validity), Jev's exact role
(discovery and advisory labels, cached as data, with `server/draft_audit.py` and
`server/reconciliation.py` as the precedents), the empirical method (a replay harness over
`logs/self_eval` — 26,778 JSONs with per-stage diagnostics and repair counters — plus the 34-brief
canary), the four acceptance criteria, and what not to do. Step 1 is the measurement harness, before
any check is touched: the plan's whole point is to choose by evidence.

## 13. Vocabulary literalism, executed (2026-09-26, evening)

The plan `docs/plans/vocabulary-literalism-plan-2026-09-26.md` was amended and executed; its §10/§11
hold the numbers and the per-check decisions. In plain words:

**What was measured.** A replay harness (`kicraft/eval/check_replay.py`) re-ran today's checks over
1 974 archived boards. Before any fix, 1 882 findings fired on boards that shipped — 278 of them on
**fully committed** boards, i.e. designs the pipeline accepted and a build produced. The top one was
not on the original inventory: `functional_spec_partial_ground_flow`, 115 times on shipped boards.

**What was wrong, and what it is now.** Three of the worst were the same mistake in different
clothes — a check reading *words*:

- `bom_architecture_role_unsupported` refused the seed-43 CH32V003 board because two sheets are
  *described* as carrying the MCU's UART and reset signals. The MCU is on its own sheet. A sheet's
  own claim is now its title and its typed requirements; prose counts only for an active part the
  design builds nowhere.
- `functional_spec_premature_topology` refused a passive crossover because its prose said "accept the
  amplifier input" (an external amplifier) and a buffer because its prose said "analog audio" (a
  signal domain). It now reads the block names the writer commits to build.
- `architecture_obligation_family_mismatch` refused the RC filter because the family
  `adjustable-rc-lowpass` was not recognised as building the trimmer it in fact emits.

Ground flow was a different bug with the same cost: `BNC_INPUT` sources the ground net, so it could
never be its own ground *target*, and a valid design was refused. A block is grounded when it is at
either end of a ground connection.

**The numbers.** Findings on shipped boards 1 882 → 1 176; on fully committed boards 278 → 147.
Eleven checks changed across four batches, each with a test in both directions; no check fires more
than before. The three live boards — including the seed-43 CH32V003 board from §11 — replay clean.
The full suite is 4 643 passed.

Three of the eleven were the same mistake as the words they replaced: `architecture_power_block_as_sheet`
could not match **"regulate"** (the escape hatch read `regulat(?:or|ion)?`), and
`bom_architecture_role_unsupported` read the pipeline's own `driver` role — relays, LED strings,
transistor stages — as an IC role. A word list is not the only place literalism hides; a grammar that
forgets an inflection is the same defect one layer down.

**What Jev contributed, and the trap it exposed.** Jev labelled the ambiguous cases. Its first
consensus on the count question ("the design is missing that part class", 0.70–1.00) was **wrong**:
the class existed, only the spelling differed. The question had offered no way to say so. Adding the
missing option flipped 11 of 12 answers to the correct reading at 0.85–0.93, and the fix became two
alias rows. That is the ground rule for this work: a model decides only among the options the
pipeline offers, and its consensus is not evidence until a human has checked the options.

**Left alone, on purpose.** A demanded part class nothing carries is still researched and added, not
accepted with a substitute. The eval's verbatim-evidence rule is unchanged; a correct derivation
should name its inputs, not argue in prose.
