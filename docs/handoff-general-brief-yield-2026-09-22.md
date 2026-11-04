# Handoff: measuring and fixing "how often does KiCraft deliver a real board?"

**Historical work log.** Current continuation instructions are in
[the standalone October 12 handoff](handoff-surprise-completion-2026-10-12.md).
Use that document to resume; the sections below preserve earlier evidence.

Originally created: 2026-09-22. The filename retains that historical date.
Latest continuation handoff: 2026-10-12 (UTC).
Author: assistant sessions, for the next engineer.
Current status: paused at the user's disconnect request; working tree is not
release-qualified. The standalone October 12 handoff supersedes this document.
Earlier sections below are historical and may be superseded.

This document explains, in plain language, what was built, what was learned, what was
changed, what went wrong along the way (including mistakes I made), and what to do next.

## 2026-10-10 correction: automatic-default evaluation

The historical claims below describe their original campaigns, not a validated
measurement of today's automatic-default Surprise me workflow. External-manifest
campaigns had forced `auto_answer_questions=False`, unlike the intended product
mode. They now enable automatic defaults explicitly across all stages and BOM
reconciliation. Defaults fill unspecified details; they never authorize changing
explicit requirements. Original-text artifact obligations remain independent.

Verification: 191 evaluation/runtime tests passed. A separate four-brief live
diagnostic campaign at
`logs/self_eval/surprise_completion/autodefault_diagnostics_20261010T214659Z/run`
completed without clarification parks: one architecture rejection, two BOM
failures, and one fabrication-ready export, but **zero product successes**.
The export still changed the requested 12 V rail to 5 V, reproducing the separate
semantic-normalization defect. Recorded run cost was $0.060592. The campaign
artifact audit passed. These selected cases are diagnostic evidence, not yield
estimates. The prior 20-case campaign remains unchanged and is not an
automatic-default baseline.

The follow-on correctness patch removes nearest-voltage clamping and exact-part
clearing. Final architecture acceptance checks typed original conversion
obligations again after recipe resolution, including defaults introduced there.
Unsupported exact ordering codes are no longer replaced by registered siblings.
The copied historical 12 V request with a fixed-5 V implementation is refused;
a reviewed TPS54331 2S-to-3.3 V control remains accepted.

Artifact voltage evidence now requires a delivered converter, inductor,
feedback/ground/output-capacitor topology and an output connector. Adjustable
divider calculations use delivered pad connections, not declared rail labels.
Ambiguous multiple exposed outputs remain unverified. On the historical export,
the full artifact extractor observes 5 V: the 5 V obligation passes and the
12 V obligation fails. Disconnected output, missing converter and corrupted
footprint evidence remain unverified. This proves the voltage obligation only,
not whole-product success. Smoke evidence:
`logs/self_eval/surprise_completion/20261010T204651Z/voltage_correctness_smoke.json`.
Integrated verification: 518 tests passed; the only warning was KiCad's existing
SWIG deprecation warning.

---

## 1. The goal in one paragraph

The project's real objective was restated as: **more than 90% of ordinary, reasonable,
few-sentence printed-circuit-board requests should produce a complete, electrically
reviewed, manufacturing-ready board in one normal run, with no engineer stepping in.**

To know whether that is true, you cannot look at a fixed set of 34 example boards that the
system has been tuned against. You need to measure on *new* requests, and you need a strict
definition of success that cannot be satisfied by a broken board. Most of this session went
into building that measuring instrument, proving it is honest, and then using it.

The headline result: **the measuring instrument works; the product currently delivers 0 of
60 boards**, and we now know precisely why — and it is not what anyone would have guessed at
the start.

---

## 2. What was built (the measuring instrument)

### 2.1 A frozen set of 60 new requests, plus a separate answer key

- Location: `kicraft/eval/corpora/general-development-2026-09-22/`
- `briefs.json` — 60 plain-English requests written to resemble real users, spread evenly
  across six families (sensors/acquisition, controller/interface, display/LED, low-voltage
  power, analog, actuators).
- `obligations.json` — 297 "answer key" checks, stored **separately from the requests** so
  the design engine never sees them. Example check: *"the finished board must contain at
  least 4 screw terminals"* or *"the finished board must contain a working power indicator
  light"*.
- The whole set has a fingerprint (`sha256:5520cf99…494c`) so a run can prove it used
  exactly this set.
- The allowed design space was frozen before running: up to 24 V DC, up to 3 A, up to
  100 × 100 mm, two or four copper layers, modules allowed. Per-request limits: $0.60,
  3,600 seconds, 12 recovery rounds, 2,400-second build timeout.

Requests outside that space are reported separately rather than silently dropped, and a
refusal counts as a failure — so a system cannot inflate its score by refusing hard requests.

### 2.2 A strict, evidence-based definition of "delivered"

Success requires **all** of the following, recomputed from the files on disk (never trusted
from a summary the design produced):

1. The board file exists and is the one canonical delivered board (if two candidates exist,
   the run fails rather than picking one).
2. A **fresh** design-rule check is run on that exact board by an independent tool
   (`kicad-cli`), and it must report zero errors and zero unconnected items — including
   checks the design had marked as "excluded".
3. The manufacturing package (Gerber artwork + drill files) is re-opened, every member is
   hashed, and it must match a receipt that ties it to that exact board file. A Gerber "job"
   file inside must describe the right number of copper layers and the board outline.
4. Every electrical terminal that the design's own parts list requires must be present on
   the actual board, on the right net. This is the check that catches "the board looks fine
   but a required wire is missing".
5. Every frozen answer-key check for that request must pass.
6. Cost, duration, and recovery rounds must stay inside the frozen limits.
7. The run must not have needed a human to answer a question.

Missing evidence counts as **not proven**, never as a pass. All 60 requests stay in the
denominator: a request that never ran counts as a failure, not as "not applicable".

### 2.3 A summary that tells you what to fix next

The campaign report now includes a **missing-capability ranking**: for every requested
capability (e.g. "four analog inputs", "a power indicator light", "a Qwiic bus"), it counts
how many runs failed because the board *contradicted* it versus how many failed because
there was simply *no evidence either way*. Those two are very different problems — one is a
bad board, the other is a blind measuring instrument — and the report keeps them apart. It
also reports yield per family, so a good average cannot hide one broken family.

### 2.4 Honest accounting of money and of unfinished work

- Every model call is now admitted individually, with a conservative upper bound written to
  a durable ledger *before* the call goes out. If a call is lost, or the provider never
  reports its cost, the reserved amount stays as "uncertain" rather than being written off
  as spend — so a cap can never be quietly exceeded.
- Runs that were stopped mid-flight are preserved on disk with an `interruption.json`
  recording why, what had started, and what it cost. They are never presented as a smaller
  successful denominator.

---

## 3. The headline measurements

Two full 60-request campaigns ran to completion and are valid evidence. ("Valid" means the
set of requests, the code, and the configuration were all unchanged for the whole run, and
all 60 requests were accounted for.)

| | First run ("run2", before fixes) | Second run ("run7", after most fixes) |
|---|---|---|
| Boards delivered | **0 of 60** | **0 of 60** |
| Any board built at all | 0 | 0 |
| Cost | $0.71 | $0.84 |
| Wall-clock | 6.7 hours | 7.4 hours |
| Failed while writing the functional description | 11 | **2** |
| Failed while writing the system architecture | 3 | **1** |
| Failed while choosing the parts list | **46** | **57** |
| Reason: reply was not usable JSON | 39 | 48 |
| Reason: reply was cut off | 8 | 10 |
| Reason: parts list rejected by a design rule | 13 | 2 |

Two things stand out.

First, **no request ever reached the physical board stage**. The score is not "boards were
built and found faulty"; it is "the design conversation never finished". So the useful
question changed from "is the routing good?" to "why can the system not even write down a
parts list?".

Second, the fixes **did** work — the earlier stages improved a lot (11 → 2 failures, 3 → 1).
The failures simply piled up into one stage, the parts-list stage, which is now effectively
100% of the problem.

---

## 4. The root cause, in plain language

### 4.1 What the parts-list stage actually does

It is the only stage where the model is allowed to *look things up* while it works. It runs
a small loop: the model asks for a part, the system searches a parts library, returns the
result, and the model asks again — up to six rounds — and then it is supposed to write the
finished parts list as one block of structured data.

### 4.2 What goes wrong, with real examples

**(a) The model chats instead of answering.** In 46 of 60 runs the reply contained no usable
structured data at all. Real captured text:

> "I'll research the parts needed for this greenhouse sensor hub and build the BOM. Let me
> research the remaining parts - LEDs, resistors, capacitors, mounting holes, and the I2C
> pull-ups. Let me search for the specific parts… I have enough context from the design brief
> and stage extras to produce the BOM. Let me compile the parts list based on the design
> state."

…and then the reply simply ends. It announces that it is about to write the list and stops.
That is 490 characters of narration where about 8,000 characters of structured data were
required.

**(b) The loop never forced an answer.** The loop only switched off the lookup tools when it
noticed the model repeating *the same* lookup three times. On a new request the model asks
for *different* parts each round, so that safety net never triggered — meaning the model was
still allowed to keep searching on its final round, and did, right up to the end.

**(c) When the model does answer, it sometimes runs away.** One attempt produced **97,479
characters** of output and was cut off at the 32,768-token ceiling. That same runaway also
explains the long freezes: producing that much text takes many minutes.

**(d) The model invents its own field names.** 44 retries were rejected because parts were
missing the required `ref` field — the model had used `designator`, `id` and `part` instead.
It was not following the written specification.

**(e) The conversation grows enormous.** Because every round re-sends the entire
conversation, the input size climbed 19,900 → 24,700 → 31,100 → 35,500 → 40,600 tokens in a
single attempt, and one attempt reached **71,000 input tokens**. Across all campaigns, 1,000
calls consumed **17.6 million input tokens** (median 18,600 per call).

**(f) The safety net was starved.** When the reply could not be read, the system is supposed
to make exactly one extra, tool-free attempt that simply says "re-send that as one block of
structured data". But that single extra attempt was shared across the *whole* stage, so once
it had been used, later attempts had no safety net at all. The baseline logged 101 unusable
replies while that one rescue was already spent.

### 4.3 Two things that were *not* the cause

- **The model is not being asked with a thinking budget.** The configuration currently
  forces reasoning to zero for every design stage, so the model is asked to produce a complex
  parts list with no opportunity to think it through. This is a strong remaining suspect, but
  changing it changes cost and behaviour for production, so it was left as a decision for you.
- **The provider does support enforced structured output.** This was tested directly: asking
  for a strict schema returned exactly the requested shape (`{"parts":[{"ref":"R1","value":…}]}`).
  That is the mechanism the newer codebase already uses; the older pinned engine had no
  support for it at all.

---

## 5. Everything that was changed, and why

### 5.1 Changes to the measuring instrument (current checkout, `~/KiCraft`)

| Change | Why it was needed |
|---|---|
| Frozen 60-request set + separate answer key (`kicraft/eval/corpora/…`, `external_briefs.py`) | To measure on new requests, not the tuned examples |
| Strict delivery check (`product_acceptance.py`) | So a broken or unverifiable board cannot be scored as a success |
| Required-terminal reconciliation against the real board | Previously the "all required connections present" check could pass on *any* pad with *any* part — a board missing a required wire would still pass |
| Missing-capability ranking in the report | To prioritise fixes by measured frequency |
| Per-call budget reservation (`budget_exposure.py`, strict mode) | So a single call cannot cross a spending cap, and lost calls are not silently written off |
| Fresh staging directory for manufacturing export (`fab_export.py`) | Previously an old drill file from an earlier attempt could be re-packaged and certified as new |
| Audit binding (`product_evidence_sha256`, frozen obligations/policy, artifact inventory) | Previously you could add a *second* board file, or replace a passing audit with a failing one, and the saved "success" still stood |
| Preserved interrupted runs (`interruption.json`) | So stopped work is visible and counted, not erased |

### 5.2 Changes to the design engine (pinned checkout, `~/KiCraft-legacy`)

You approved amending the "leave the pinned engine untouched" rule, because *every* measured
failure was in the design stages, which live there. Each change below is tied to a measured
failure and was kept as small as possible.

1. **Force the answer on the last lookup round.** The tools are now switched off on the final
   round, so the model must write the parts list instead of continuing to search. (The newer
   codebase already did this; the pinned revision predated it.)
2. **Allow a "mechanical" block.** Requests that ask for mounting holes could not be written
   down at all: the schema had no category for a purely physical block, and a second rule
   rejected any block with no electrical connection. This caused 30 rejections across 60
   requests.
3. **Say exactly what shape the answer must be.** All three places that demand a final answer
   now state the exact top-level shape and forbid wrapping it in a made-up key such as
   `{"bom_slot": …}`.
4. **Give the rescue attempt a real budget** (1 → 3 per stage). The failure is demonstrably
   random — the *same* request both succeeded and failed on different runs — which is the
   condition the plan set for allowing this.
5. **Ask the provider to enforce structured output** on the two tool-free answer paths, with
   a graceful fallback if the provider ever rejects the request, so the rescue can never be
   lost to an unsupported option.
6. **Bound a frozen connection.** Twice, a provider connection stayed open for ~19 minutes
   with no answer; the normal request timeout does not catch this because the provider keeps
   sending harmless keep-alive traffic. The "no answer yet" phase is now time-bounded and
   fails cleanly into the existing retry paths. This is a genuine production defect, not just
   a measurement one.

### 5.3 Mistakes I made (recorded so they are not repeated)

- **A self-inflicted crash.** When I added the example shapes to the rescue message, the
  message is processed by a formatting function where braces have special meaning. My
  unescaped braces made that function fail, so every rescue attempt died before it was sent.
  The whole fourth campaign failed this way (6 of 6) with a misleading "protocol error". Now
  escaped, with a test that renders the message and checks the examples survive.
- **A wrong attribution.** I blamed a 19-minute freeze on the structured-output option and
  removed it. Freezes then happened *without* it too, so that was wrong; it has been restored.
- **A wrong safety argument.** I tried to narrow a shared lock so one slow call could not
  stall the whole campaign. The existing race test correctly rejected this — two processes
  could both pass the check. I reverted rather than force it through.
- **Targeted tests twice misled me.** Re-running individual previously-failing requests
  suggested 6/6 and 3/3 success; the following full campaign still scored 0/60. Only full
  campaigns are treated as evidence now.

---

## 6. What was verified, and how

| Check | Result |
|---|---|
| Current-checkout test suite (evaluation, acceptance, self-eval, guards, fabrication) | 128 passed |
| Lint on every file this session touched | clean, except `kicraft/eval/electrical_artifact_evidence.py`, which carries 35 pre-existing style findings that predate this session's edit and were deliberately left alone (the change there was a behaviour fix, not a reformat) |
| Legacy-engine client tests (including two new ones for the frozen-connection fix) | passed |
| Legacy-engine rescue-message test (new) | passed |
| Real KiCad export + independent design-rule check on a real board | passed; a stale drill file was correctly excluded; changing the board invalidated its receipt |
| Real board-graph terminal proof | passed with all nets present; correctly withheld after removing one required connection |
| Fault-injected transport test of the retry accounting | two attempts left states "uncertain" and "settled" as intended |
| Provider structured-output support | HTTP 200 for both strict-schema and plain-JSON modes |

---

## 7. Where everything lives

- Frozen requests and answer key: `kicraft/eval/corpora/general-development-2026-09-22/`
- Campaign results (each with its own manifest and summary):
  - `logs/self_eval/m1_general_development_2026-09-22_run2/` — first valid baseline (0/60)
  - `logs/self_eval/m1_general_development_2026-09-22_run7/` — second valid baseline (0/60)
  - `logs/self_eval/*_run3…run9/` — stopped runs, each with `interruption.json`
- Full written plan and running record of every fix: `docs/plans/general-brief-90-percent-yield-plan-2026-09-22.md`
- Spending ledger: `~/.kicraft/spend_ledger.db`

**Money spent:** $2.4337 of the $5.00 authorised for this work (hard ledger ceiling
$138.80079697348). About $0.84 is needed for one more full campaign, so there is room for
one, and a second only if probes are kept small.

### Commands

Check the frozen set without spending anything:

```bash
cd ~/KiCraft && .venv/bin/python -m kicraft.eval.self_eval \
  --brief-manifest kicraft/eval/corpora/general-development-2026-09-22/briefs.json \
  --obligations kicraft/eval/corpora/general-development-2026-09-22/obligations.json \
  --validate-manifest
```

Run one full campaign (use a fresh output directory):

```bash
cd ~/KiCraft && .venv/bin/python -u -m kicraft.eval.self_eval \
  --brief-manifest kicraft/eval/corpora/general-development-2026-09-22/briefs.json \
  --obligations kicraft/eval/corpora/general-development-2026-09-22/obligations.json \
  --campaign-budget-usd <remaining allowance> --parallel 3 --build-slots 1 --no-judge \
  --out logs/self_eval/<new-name>
```

Audit a finished campaign offline (no model calls):

```bash
.venv/bin/python -m kicraft.eval.self_eval --audit-campaign logs/self_eval/<name>
```

---

## 8. What to do next, in priority order

1. **Re-run the full campaign** with the current fixes (recovery budget + enforced structured
   output). This is the only way to know whether the parts-list stage now succeeds. If it
   does, the whole picture changes: you will see the first boards ever to reach the build
   stage, and the next bottleneck will be visible for the first time.
2. **If the parts-list stage still fails, decide the two policy questions** — these are
   yours, not the code's:
   - The model currently gets **no thinking budget** for design stages (forced to zero in
     `LEGACY_ENV`). Giving it one is the most promising untried lever, at higher cost.
   - Alternatively use a **stronger model for the parts-list stage only**.
   Both change production cost, which is why they were not changed unilaterally.
3. **Bound the parts-list conversation size.** Each of six rounds re-sends the whole
   conversation, including 16 search results of up to 4,000 characters each, producing the
   71,000-token prompts and much of the cost. Keeping only the most recent results, or
   reducing the per-result size, is the obvious next saving — but it is a cost/latency fix,
   not a yield fix, so it was deliberately deferred until yield is measured.
4. **Keep the honest bar.** Do not report a "product success" for a board with missing
   evidence, do not drop requests that never ran, and do not re-tune against the 60-request
   set and then present it as unseen — it becomes development material the moment you look
   at it. Fresh validation and a sealed 200-request release sample still need their own
   inputs and their own spending authorisation.

---

## 9. The single most important lesson

Every one of the six design-engine defects fixed this session was invisible to the existing
test suite and to individual re-tests. They only appeared when 60 unseen requests were run
end-to-end with a strict definition of success and honest accounting. The instrument was the
expensive part, and it is the part that keeps paying: it turned "the system is unreliable"
into a specific, ordered, measurable list.

## 2026-10-11: frozen Surprise-me baseline and shared-contract repairs

The auto-default baseline at
`logs/self_eval/surprise_completion/20261010T204651Z/development/run_autodefault_correctness`
completed all 60 frozen requests. Its offline campaign audit passed with no missing,
duplicate, pending, or integrity-failed attempts. Thirteen designs committed; nine builds
reported successful exports; **zero products passed the independent full-brief acceptance
check**. Three of those nine exported boards failed the independent delivered-board DRC.
Total provider spend was $0.845441; recorded campaign runtime was 14,742.4 seconds.
The generator distribution and original obligations were not changed.

Construction remains the largest measured blocker: 47 requests stopped before a build,
including 21 architecture contract rejections and 17 exhausted BOM work units. The
cross-run scan found 11 designs with unsupported lowerer contracts and seven with missing
recipe requirements; these diagnostic counts overlap and are not additive. Missing
acceptance observers remain unverified evidence, not proof that those circuits are wrong.

Shared repairs after this baseline:

- Short device identities such as LM358 are no longer discarded by an arbitrary six-character
  minimum. The saved LM358-to-MCP6001 architecture now fails the named-owner contract.
- Declared physical interfaces use every functional unit of a symbol. Real LM358 pins 5–7
  pass the inventory check; nonexistent pin 9 remains refused.
- Board dimensions expressed in `mm per side` remain board-owned and cannot become a copper
  layer count. An explicit stack-up overrides a contradictory intent draft; unsupported,
  fractional, or conflicting stated counts fail rather than defaulting or rounding.
- Generic JST carrier selection respects real contact cardinality and explicit identities.
  Enable-jumper lowering uses a removable shunt/header and external pull; automatic
  composition currently requires a uniquely identified TLV62569 enable input and its
  reviewed voltage limit. This does **not** yet provide high-voltage enable support for
  TPS54331/AP6320x circuits.
- TL3342 reset/boot producers emit the exact switch MPN. Sourcing rejects a fuse or
  motor-driver IC on that tactile-switch interface while preserving ordinary unregistered
  passive sourcing. Sourceability receipts read the catalog's actual manufacturer field.

Verification: 923 targeted tests passed, with one existing KiCad SWIG deprecation warning.
The real-artifact smoke rejected the saved wrong-amplifier architecture, checked the actual
vendored LM358 pin inventory, wrote and reloaded a four-layer KiCad seed, refused 4.5 layers,
and resolved buttons from six registered MCU recipe variants against the installed catalog
to TL3342F260QG / C2886894 / E Switch. The two previously wrong catalog parts were rejected.
Evidence is in `development/shared_contract_smoke.json` alongside the frozen baseline.
These are contract and artifact checks, **not a new product-yield measurement**; complete
circuit packages, repeated live-stage replays, physical trials, and the development/release
acceptance thresholds still apply. No production restart was performed.

Delivered-board geometry observations now cover copper layer count, strict requested
maximum dimensions, and connectors sharing one edge. Dimensions require a closed outline
and exclude drawing stroke width. Connector placement excludes reference text; a positive
one-edge verdict requires complete component classification and measured connector bounds.
Nonrectangular outlines remain unverified for the one-edge rule.

Verification: 29 geometry regressions passed. A read-only smoke against the exported
development run 13 board measured two copper layers and a 36.887469 × 22.651235 mm outline.
All classified connectors shared the bottom edge. Its geometry checks passed; this does
not establish electrical correctness or change the measured 0/60 full-product result.
The repaired combined power-wave run passed all 1,023 tests in 66.97 seconds.
This verifies the covered contracts, not a live-build or full-product improvement.

Native KiCad smoke verification loaded all four new inductor footprints, checked their
manufacturer land-pattern dimensions and positive courtyard clearance, verified bundle
hashes, then saved and reloaded a board containing them. This caught and repaired a
missing footprint closing delimiter. The actual production sourcing gate subsequently
passed all four inductors and both converter capacitor choices after replacing unavailable
catalogue codes with orderable delivery codes. The catalogue was 27 days old; the
6.8 µH Coilcraft and 100 µH Bourns inductors remained explicitly flagged for low assembly
stock (33 and 46 respectively). Family capacitor DC-bias curves are reference data, not
guaranteed minima for the selected delivery codes.

The subsequent native production smoke expanded seven actual power recipes, loaded
their real KiCad footprints, assigned the compiled pad nets, saved/reloaded each board,
and ran independent electrical extraction plus the production sourcing gate. All seven
passed required-terminal reconciliation; sourcing returned no errors (35 passing
receipts plus explicit keep/warn cases). This exposed and fixed an LM2577 public input
that had incorrectly fed no circuit: its protection input was on an orphaned private
net. The corrected LM2577 and 12 V TPS54331 expansions now independently prove their
source-to-output 12 V binding. The TPS61088 and TPS552892 output qualifications and
all continuous-output-current ratings remain unverified; their support/operating-envelope
evidence is incomplete. No routed/fabrication-ready result is inferred from these
unrouted smoke boards. Full evidence is in
`development/production_power_native_smoke.json` under the campaign root.

The user approved a fresh blind release set after a read-only reviewer accidentally
searched the campaign parent directory and received six snippet ranges from the old
sealed obligation files. That reviewer made no edits or direct agent handoffs; its
engineering report was discarded and it was excluded from further candidate work.
The original corpora remain preserved. The independent custodian generated one new
uniform, no-reroll 200-brief sample, excluding historical, development, and original
holdout seeds, under the unchanged generator and nominal-voltage-v2 obligation policy.
The replacement is `release-sealed-replacement-v1/bundle` under the campaign root,
with canonical bundle hash
`sha256:593f0f2e9cf9d47f1a1934c5fe686a8016f213cb6515832b5f38ee2a53706010`.
The custodian reported 200 briefs, 2,827 obligations, successful offline schema/hash
validation, and preserved lineage/resource caps. Implementation workers have not read
the replacement contents; it must not be opened or run before candidate qualification
and freeze.

Power architecture diagnostic `development/power_wave_stage_replay_v1` committed
0/3 trials; publishing the cross-parameter current limits improved the next frozen
diagnostic (`power_wave_stage_replay_v2`) to 1/3, not three successful trials.
The two remaining failures exposed incomplete fuse input-contact metadata and
invalid connector contacts. A complete distribution owns its fuses and selected
controls, not the separately requested output connectors. These are stage-only
diagnostics and do not change the product yield.

An isolated intermediate source snapshot at
`/tmp/kicraft-power-candidate-r2d7jo2z` has fingerprint
`sha256:0075fa3fa5e314368ae4b6818cd4bbf8c490443c191eeec3839e56e72050d0a3`,
identical to the working source at capture. Its saved architecture was accepted by
the construction-validator entrypoint; this did not establish complete composite
ownership or a routed board. 348 focused artifact-selection, identity,
electrical-invariant and lowering tests passed. This is not the release candidate.

The initial copied-baseline regrade was invalid because absolute promoted-board
paths still named the original runs. The extractor now recognizes an unchanged
run snapshot only when its state, board, KiCad project and custom rules match the
original; it always loads the local copied board. Changed or missing originals,
boards or rules do not acquire a relocated identity. The verified regrade used
the frozen intermediate source and selected all 13 original promoted boards
without relocation errors. All original campaign files remained byte-identical.
`development/regrade_policy_v2/regrade_summary_verified.json` records **0/60**
complete products, no missing/duplicate/pending observations and no audit-integrity
errors. The original invalid regrade summary remains explicitly marked invalid.

Power replay v3 committed 0/3 architectures: private, non-converting distribution
rails lacked nominal-voltage evidence. Validation now propagates declared voltage
only through reviewed non-converting power-transfer contacts; conflicting source
voltages are refused, and resistors or converter switch nodes cannot invent a DC rail.
Replay v4 committed 1/3. Its remaining failures were a KiCad library-pattern name
offered as an exact manufacturer part and an unowned enable-jumper obligation.
The part-choice prompt now excludes the reserved `kicad-` pattern namespace while
retaining real manufacturer parts that use standard KiCad symbols/footprints.
Neither replay meets the three-successful-trial requirement.

The first paid full fresh run of the original development-20 brief,
`development/power_development20_full_v2`, completed all five stages and generated
a routed, ERC/DRC-clean board and fabrication archive in 516.5 seconds for $0.011162.
It remained **0/1 product passes**: the independent check could not verify the
enable jumper, power LED and complete required connections, and sourceability failed.
The earlier v1 diagnostic stopped before paid calls because existing ledger exposure
left too little campaign allowance for a full per-run reservation. V2 used a $1
diagnostic campaign envelope, retaining the original $0.60 per-run policy; no ledger
reservations were removed.

Inspection of that actual board identified two evidence defects, not missing parts:
the LED used two series resistors behind a fused high-side switch, and the MOSFET
had repeated source/drain pads. The observer now requires all reviewed bus pads,
the real source/return, output path, pull-up/pull-down network and correctly oriented
gate clamp when present. A broken switch cannot supply a positive downstream LED
verdict. This discrete enable proof currently covers standard 5 V USB input; other
input ranges remain unverified rather than being inferred from net labels.

The sourcing receipt had passed all stock checks but lost manufacturers absent from
the offline catalogue. The existing live retail response supplies both manufacturer
and exact product identity. These fields are now retained through the retail cache,
and fill a missing receipt manufacturer only when both supplier code and manufacturer
part number match. Old stock-only cache rows are invalidated; unrelated identities
cannot lend their manufacturer to a receipt. The actual supplier listing confirms
[KANGNEX / C8404](https://www.lcsc.com/product-detail/Terminal-Blocks_WJ126V-5-0-2P_C8404.html).

Frozen source `sha256:d22365c306551768af644f52d810faf81c9e713bb5af43be499a6773d25dde1f`
at `/tmp/kicraft-power-candidate-0irm7oky` passed an actual native-board smoke:
the intact delivered board proved the enable, LED and required connections; eight
pad-level defect variants were saved/reloaded and refused the affected proof.
Live sourcing produced 25 passing receipts, no missing manufacturers and no sourcing
errors; the 27-day-old offline catalogue warning remains. Original board/state bytes
were unchanged. Evidence is `development/delivered_power_native_observer_smoke_v1.json`.
This isolated repair proof does not rewrite the saved campaign verdict or establish
a new development-population yield.

The MCU, analog, actuator, display and USB-bridge capability edits have been integrated
but are not yet qualified end to end. The integrated check on frozen source
`sha256:76fefbee56f042b0f1d54f69a003d9d2d401fcd0a190dc0eadfd4dd88e0bea92`
reported 11 failures and 1,170 passes. Failures included a nonexistent CP2102N stock
symbol and obsolete expectations that UART-connected MCUs must also use native USB.
The obsolete expectations were removed without weakening half-pair rejection or
GPIO/programming checks. The next integrated check remains pending. USB bridges
require explicit 5 V VBUS; a native MCU USB PHY must not share their receptacle pair.
No production service restart, secret/configuration change or sealed-release run occurred.

### First freshly generated, independently accepted power product

The fresh v3 diagnostic failed architecture: connector-pruning heuristics removed a
compiler-owned enable jumper because an unrelated output connector existed and the
sheet description lacked a connector keyword. `stage_work_units._validate_bom_unit`
now applies that author-surplus heuristic only to non-compiler candidates, retaining
the independently regenerated compiler inventory. The exact rejected candidate then
passed the full constructive architecture validator without another provider call.

Architecture replay v5 on frozen source `76fefbee…` committed **2/3** trials.
Its remaining LED failure exposed ordinal connector-contact inference applied to
functional device pins. Carrier adoption now limits that inference to reviewed generic
connectors; explicit authored interfaces remain untouched. Three successful live
architecture trials are still required.

The original development-20 brief was then generated again from scratch using frozen
source `76fefbee…`. `development/power_development20_full_v4` completed all five stages,
produced a fabrication-ready export and passed **all original product obligations:
1/1 product success**, in 484.2 seconds for $0.012024. The canonical campaign audit
reported `artifact_integrity_ok=true`, no errors, no missing or duplicate attempts.
This is a selected diagnostic success, not a random-population estimate or a replacement
for any failed attempt. The original full development campaign and unchanged-artifact
regrade remain **0/60**; the 57/60 development exit gate is not yet demonstrated.

### USB bridge and combined-control evidence

Native asset validation of frozen source
`sha256:c6c3654dc78d0baca80f9498f19518d096c65e2e12a42a06d184583e3cf9230b`
loaded 82 symbol/footprint pairs across 49 production recipes and found no missing
declared contacts. This checks real installed assets, not routing or circuit performance.

The CP2102N circuit required more than a symbol-name repair. Manufacturer
[datasheet revision 1.5](https://www.allelcoelec.com/datasheets.f0/CP2102N-A02-GQFN28.pdf),
figures 2.3 and 2.6, requires regulator bypass when externally supplied at 3.3 V,
a protected VBUS sense divider, and the specified supply bypass/reset network.
The recipe now ties VREGIN to external VDD, uses the 22.1 kΩ/47.5 kΩ VBUS divider,
4.7 µF plus 100 nF bypassing per power pin and a 1 kΩ reset pull-up. It uses the
installed `Interface_USB:CP2102N-Axx-xQFN28` symbol while retaining the exact A02
ordering identity. Its integrated USB termination does not need external series
resistors. The unavailable C96433 supplier hint was replaced by the observed,
stocked C964632 A02 reel-pack ordering code.

`development/usb_bridge_native_smoke_v1.json` records a native KiCad save/reload of
the real AP63203 + CP2102N + STM32G030 recipe composition: the intact circuit proved
the bridge path; ten variants with broken supply, divider, reset bias, return,
ESD, USB contact or crossed-UART connections did not. All six explicitly named
parts passed live sourcing. This is not a routed-board, full-passive-BOM or USB
compliance qualification. STM32 USART2 contacts 9/10 are backed by DS12991 Rev. 4,
figure 5 and tables 12–13 ([manufacturer document mirror](https://www.gocarelectronic.es/pdf/010STM32G030ST.pdf)).

Native-board probes also found that the reset observer omitted reviewed momentary
buttons, and that adding programming evidence erased earlier physical gate results.
The observer now recognizes those buttons and preserves independently observed gates.
The same board then retained both reset and programming passes. A separate native
negative case previously called a permanently grounded MCU reset plus a supply-shorting
button a working reset control; it now produces no reset pass.

### Physical baseline status

Read-only triage localized the retained 9101/9102 artifacts. Source inspection found
candidate causes in same-footprint tie clearance, ground-stitch via edge clearance,
the final fabrication predicate, and dense MCU escape placement. These are not yet
current replay-confirmed regressions.

The first six isolated replay invocations stopped at router preflight because the
direct CLI did not load production environment settings. They are preserved under
`development/physical_baseline_v1` and are **not physical baseline trials**.
The corrected harness uses the existing read-only environment loader and router
preflight, preserves project rules and pre-promote seed hashes, and runs three
equal-budget trials per fixture under `development/physical_baseline_v2`.
Those trials are in progress. Original projects remain untouched; the replacement
release holdout remains sealed and unexecuted.

### Named momentary-switch composition

The integrated check on frozen candidate `0fafc05032a8c64469be3ec8b9aa293938f7026c775a6d83394c064098882b06`
passed 1,197 cases and exposed one remaining named-switch composition failure.
Restricting ordinal contact inference to physical connectors correctly stopped
inventing LED pins, but exposed that `switch-input` did not own its named TL3342
button. It now retains that circuit family and the actual ordering code, rather
than rewriting a button-and-bias circuit into a bare two-contact part.

An unspecified switch pull policy now uses an external resistor. Explicit internal
bias remains internal, and a requested resistor cannot be discarded in that mode.
The obsolete test requiring refusal of an unspecified policy was removed, along
with an assertion that assigned the circuit's supply directly to a button contact.

A native KiCad save/reload smoke exercised the same source architecture, including
two header contacts sharing the button signal. Active-low closure reaches ground
with supply-side idle bias; active-high closure reaches the supply with ground-side
idle bias. Both preserve `TL3342F260QG`. Naming an LED instead is refused.
This verifies circuit composition and actual pad nets, not routing or fabrication.

The frozen switch candidate `469eb3d390428ff4a19439325c09b6ea252c9245da8ba21fd7f399d72c6e73ba`
passed all 372 checks in the lowering, architecture-intent and stage-prompt modules.
Its fresh three-trial architecture replay (`power_wave_stage_replay_v6`) remained
2/3: the failed draft assigned the same terminal obligation to four real terminal
requirements and to their terminal-free distribution circuit.

Physical-owner normalization now removes that extra claim only when an existing
owner is independently proven and every part of the lowerer's compiled circuit has
reviewed inventory evidence proving the claimed class absent. Unknown circuits
remain unknown; ambiguous unassigned owners are not invented. Original board-wide
obligations and quantities are untouched. The exact failed draft subsequently
passed real constructive acceptance in a fresh temporary project. The failed trial
remains failed in the retained replay; this local probe is not a replacement run.

### Reviewed ESP analog and memory pin boundaries

An allocation smoke exposed two hardware-capability errors: the ESP32-C3 recipe
accepted GPIO6 as an ADC input, and the ESP32-S3-WROOM-1-N8R8 recipe allocated module
pad 28 despite its internal PSRAM connection. The shared `gpio <= 20` ADC predicate
also excluded real WROOM ADC inputs such as GPIO34.

Module-specific ADC contact sets now follow the manufacturer pin tables:
[C3 table 3-1](https://documentation.espressif.com/esp32-c3-mini-1_datasheet_en.pdf),
[S3 table 3-1](https://documentation.espressif.com/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf),
and [WROOM table 3](https://documentation.espressif.com/esp32-wroom-32e_esp32-wroom-32ue_datasheet_en.pdf).
C3 GPIO5/ADC2 is excluded because the manufacturer's
[ESP-IDF hardware limitation](https://docs.espressif.com/projects/esp-idf/en/v5.3.4/esp32c3/api-reference/peripherals/adc_oneshot.html#hardware-limitations)
states that its conversion results are unstable. S3 N8R8 pads 28/29/30 are left
unconnected per table 3-1 note b, not offered as GPIO35/36/37.

After the fix, the same actual allocator refuses C3 GPIO6 and the PSRAM pads,
while accepting WROOM GPIO34 on physical pad 6. This is allocation-path evidence,
not a newly fabricated or routed product.

### Integrated ADC and RP2040 bridge evidence

The resolver now binds an analog producer's output range to the actual reviewed
ADC consumer, rather than merely checking its own supply. A real resolver/lowerer
smoke on a TLV9062 current front end feeding STM32G0 inputs previously selected
4.95 V full scale. With the consumer's 3.3 V limit propagated, it selects 2.471 V
full scale; the original request remains unchanged. Explicit incompatible gain is
refused, not rewritten. Optional-port label inference also no longer allocates an
already-bound ADC net to a second channel. This is circuit-level evidence, not an
analog accuracy measurement or fabrication result.

Frozen candidate `de85d4a6e9646cfdf78c9b22fd5df4eb48f940518079bed105c753f9a6a0b176`
passed all three fresh selected architecture trials in
`development/power_wave_stage_replay_v7`; the original state and frozen source hashes
were unchanged. The failed v6 trial is retained. This diagnostic does not change
the 60-brief campaign result.

The same frozen candidate passed a native KiCad save/reload smoke of the 72-part
AP63203 / CH340C / RP2040 composition. UART0 uses physical RP2040 pins 2/3, and its
native USB pins 46/47 remain unconnected. Independent SWD recovery reaches actual
header contacts. Disconnecting SWDIO, SWCLK, RUN or header ground each invalidates
recovery without erasing the working serial path. Reversing UART traffic or
disconnecting bridge USB invalidates the serial path without erasing SWD recovery.
These checks prove observed pad connections, not a routed or fabrication-ready board.

The six-module integrated regression run on that candidate passed 672 checks and
failed nine. Six reset-observer fixtures used an unresolved shorthand identity;
the other failures concern USB-PHY ownership, a changed RP2040 button inventory,
and a fixture missing its now-explicit programming mode. Corrections and stronger
analog reset safety evidence are in progress; this candidate is not qualified.

Native analog integration exposed a second, separate defect: the production BOM
work-unit normalizer replaced the reviewed stock TLV9062 symbol/footprint pair with
an alternate library pair that the independent artifact observer could not verify.
Reviewed-pair adoption now remains authoritative after curated-library lookup.
The compiler still uses curated defaults for identities without a portable reviewed
record; no footprint-match gate was relaxed.

Frozen diagnostic `7a9f64dd0497db79dad53cff6482f34f808b8bce353601434ad340b7cada0fb2`
then passed a complete constructive BOM/wiring → native PCB save/reload smoke.
Its two-channel TLV9062/STM32G0 board yielded an independently observed 2.471 V
maximum and actual ADC pad connections. Opening feedback removed voltage evidence;
opening an ADC contact invalidated the composition while preserving the otherwise
valid amplifier evidence.

The same snapshot passed RP2040/CH340C native checks with internally scoped bridge
USB conductors, including rejection of a second PHY connected to only D+. SWD
recovery correctly remains observable when that separate serial path is defective.
Actual intent → recipe → BOM composition now produces one USB receptacle and
independent SWD recovery without inventing an additional RP2040 USB connector.
Explicit contradictory programming modes are refused. CP2102N shares the
internally owned USB-port contract. The integrated five-module run on this snapshot
passed 600 tests, including its intent regression.

Analog power-cycle reset remains under implementation. EN-only analog-switch
control is insufficient: signal reconnection must wait for the switched rail to
recover, and the discharge calculation must include worst-case switch leakage.
It is not a completed capability or a qualified candidate.

### Physical baseline and corrected-candidate verification

All six `development/physical_baseline_v2` trials finished on frozen source
`76fefbee56f042b0f1d54f69a003d9d2d401fcd0a190dc0eadfd4dd88e0bea92`.
None met the physical exit gate. All three 9101 trials returned CLI success while
the selected parent retained a clearance violation and a copper-to-edge violation;
all three 9102 trials returned failure with 14 parent opens. Original project,
state and seed inventories remained unchanged.

The measured producers were a same-footprint power tie, a ground stitching via
whose barrel crossed the board-edge clearance band, and fine-pitch signal exits
blocked by earlier reserved power/ground copper. Corrections enforce full sibling
pad clearance, check via copper against the actual outline and cutouts including
tessellation error, and reserve legal fine-pitch signal exits before plane escapes.
Final verification also refuses `illegal_routed_geometry`; no rule or acceptance
threshold was weakened.

Frozen diagnostic `fed7fa971dbd43ee674acee3b00039bd49321ed4036145e8c7260a3913f3fc64`
passed 321 focused regressions. Six isolated full replays are running in
`development/physical_candidate_v3`; their outcome is not yet established.

Fresh selected RP2040 architecture trials in
`development/rp2040_bridge_stage_replay_v1` all failed and remain retained.
One exposed a real expansion defect: choosing native USB returned before applying
the independently bound UART contacts. UART realization now precedes the recovery
mode split. A 77-part native KiCad save/reload smoke observed the intact serial
path with distinct native USB contacts; uncrossed UART, disconnected bridge USB,
and a second PHY on D+ each failed the serial-path observer. This is not a routed
product or a replacement for the failed provider trials.

Manufacturer review of [CH340 datasheet 3D](https://www.wch-ic.com/download/file?id=79),
sections 4 and 5.1, additionally found that its USB data pins require direct paths
without added series resistors. The CH340 recipe and observer are being corrected
together; earlier CH340 smoke results do not qualify that outdated series topology.
Section 7.6 explicitly supports sharing the MCU's self-powered supply with CH340
while leaving USB VBUS separate; no speculative second power domain is required.

### Native circuit integration: current evidence and limits

The corrected CH340 circuit, with direct protected USB data paths rather than
added series resistors, passed the 75-part native pad-graph smoke: its intact
serial path passed and six damaged or conflicting ownership cases did not.
Fresh RP2040 architecture trials in `development/rp2040_bridge_stage_replay_v3`
still failed **0/3** on frozen source
`db5066ca575ffb68e6e98b50a00793582d6091ee0ecee39e163e3aa28c5823a8`.
All reached gate 9.38 with the WS2812B supplied at 3.3 V, below its 3.7 V minimum.
The required repair must preserve battery-powered operation, not substitute USB
power. These trials are failures, not completion gains.

Native ESP construction/save/reload checks passed for C3, WROOM, S3-WROOM,
S3-MINI and a shared C3 power/data receptacle. Corrections include the C3 GPIO8
download-mode pull-up, complete optional UART contact ownership, MCU-owned native
USB series resistors, and refusal/remapping of unavailable ADC/PSRAM contacts.
This is circuit construction evidence, not routing or manufacturing qualification.

Analog reset now uses TPS22948DCKR with an independently powered TPS3808.
The earlier TPS2553 reverse-leakage specification did not bound forward leakage
while off and cannot support the proposed discharge proof.
[TI's TPS22948 specification](https://www.ti.com/lit/gpn/TPS22948) bounds shutdown
input current at 5 µA with OUT grounded. With the reviewed isolation/supervisor
leakages and 1 kOhm discharge resistor, the calculated worst-case floor is
23.23 mV. The 6 ms passive-discharge bound starts **after the power switch and
signal isolators are off**; it excludes control propagation. The datasheet does
not provide a guaranteed maximum total button-to-discharge delay.
The corrected imported bundle passed `validate-part --update-hash`, yielding
`sha256:37147fd6f822bd77b44b42814c820c94b5d7a606bc0b1547447f2fbb70728a51`.

Actual constructive BOMs and native KiCad save/reload passed three intact reset
compositions: MCP6002 humidity (23 parts), LM358 current sense (25), and TLV9062
temperature (23). All fifteen damaged variants remained unverified: disconnected
switch output, supervisor sensing the input, supervisor powered by the switched
rail, isolation permanently enabled, or isolation powered by the switched rail.
The first native temperature attempt exposed a separate real package error:
MCP9700AT-E/TT is three-pin SOT-23, not a five-pin device.
[Microchip DS20001942K Table 3-1](https://ww1.microchip.com/downloads/aemDocuments/documents/MSLD/ProductDocuments/DataSheets/MCP9700-Family-Data-Sheet-DS20001942.pdf)
establishes VDD1, OUT2, GND3; the record and lowerer now agree.
Thirty focused reset regressions passed before that temperature-package correction.
Original-brief connector ownership, complete architecture integration, routed
boards and broader regression qualification remain outstanding.

Further manufacturer review limits that reset discharge claim to **5 V**:
TPS22948 section 6.5 states that test voltage explicitly. The compiler and
observer now refuse to extrapolate the off-current bound to a 3.3 V reset circuit.
All three selected original analog briefs use 5 V after any required conversion.
The switch circuit now includes the recommended 1 uF input and 18 nF output
bypass capacitors. Its imported symbol has reviewed electrical pin types.
Native footprint inspection confirmed six pads at x = ±1.10 mm and
y = −0.65/0/+0.65 mm, with 0.40 × 0.90 mm lands, matching TI DCK0006A.
The bundle is now `reviewed`, with content hash
`sha256:a8247fdf3b07f4d467c50208a1d4ac15f21deb55bba61d006645beb3e729eb64`;
final manifest validation passed.

Intermediate frozen source
`42c227b7938e9db9224eedc520513f79d63b7f2da9a1a9383d4531e7a6a151cc`
passed 842 tests and failed five. Three failures exposed missing `vplus` supply
binding in typed analog composition; that producer is corrected. The remaining
failures exposed an observer's dependency on an unwanted implicit output header
and a normalization fallback that attempted to generate USB VBUS from the battery.
Those failures require real interface/source-ownership repairs, not re-pinned
expectations. This snapshot is not a qualified candidate.

### Physical candidate v3: same-artifact results

Frozen source
`fed7fa971dbd43ee674acee3b00039bd49321ed4036145e8c7260a3913f3fc64`
completed six equal-budget physical replays in 7,276.52 seconds. Original
projects and seed hashes remained unchanged.

| Fixture | Accepted physical trials | Replay durations (seconds) |
| --- | --- | --- |
| 9101, ESP32 indicator controller | 3/3 | 994.23 / 992.05 / 1053.99 |
| 9102, RP2040 motor controller | 0/3 | 1714.06 / 1263.65 / 1254.41 |

For every 9101 trial, all six leaves have zero local opens. The final promoted
board's own `build_gate.json` is successful and fabrication-acceptable, with no
shorts, opens, courtyard/keepout failures or copper-clearance violations.
Fresh 20261011 fabrication receipts match independently recomputed SHA-256
hashes of both the delivered PCB and ZIP in all three trials. Silkscreen and
dangling-via/track diagnostics remain; this is not a claim of zero total DRC
diagnostics. Trials 1 and 3 share geometry, so these are repeated physical
builds, not three distinct designs.

The triage report's parent section describes its last round, while the selected
board came from the best earlier round. Acceptance therefore rests on the
promoted board's gate and matching fabrication receipt, not that parent summary.
Evidence lives under
`logs/self_eval/surprise_completion/20261010T204651Z/development/physical_candidate_v3/`.
The 9102 first trial still reports six opens, including local 1.1 V and GPIO
connections. No full physical N3 exit, development 57/60 exit or sealed release
qualification is claimed.

The later integrated circuit snapshot
`3e3d0c0c7baf897b2cdc30e432eb357c124fd3507ad9291e70cc2f84a8c10480`
passed 834 tests and failed 21. It remains unqualified. Native construction of
the three original analog interfaces exposed remaining observer/path gaps.
Manufacturer review of SN74LVC1G66 section 5.3 also requires a control transition
of at most 10 ns/V at 5 V: the existing 100 kΩ supervisor pull-up does not
establish that bound. Section 8.3 requires a bypass capacitor at each isolator;
the current-sense input isolators lacked their own capacitors. Earlier static
reset-path positives do not establish compliance with these requirements.

### Native-circuit qualification after reset and RGB repairs

The reset circuit now regenerates the supervisor's slow open-drain transition
through an SN74LVC1G17 Schmitt buffer before driving the SN74LVC1G66 controls.
Current-input isolators have individual bypass capacitors. Frozen revision
`91dff7cd0834c2e3a7a02b0f87bfd8648b5c3033b143e5f78391286f2e46b811`
passed 865 targeted tests and 32 native-pad scenarios: original analog circuits
and reset faults, ADC binding, and RP2040 bridge/recovery variants. These checks
establish construction and electrical observation, not routed exports.

The RP2040's guaranteed 2.62 V GPIO high does not meet the reviewed WS2812B-B/T
V5 2.7 V input-high requirement. The onboard RGB recipe now uses a
SN74AHCT1G125DBVR on the actual 5 V LED rail, with enable held active, input
pulldown, bypass and series resistance. The observer requires the complete
powered controller and this compatible signal path; a direct 3.3 V connection
does not qualify.

Frozen revision
`9239b14fedf2165ccfbd59b09490f5ccfbc25a37a343d61c0d4ed279f3ae2753`
constructed the original saved RP2040 candidate as 94 real native footprints.
Its USB-serial function, RGB status indicator, four-contact JST and 2S source
binding all pass their individual observations. Twelve separately saved and
reloaded native-board fault variants are correctly rejected: disabled or
underpowered buffer, raw-GPIO bypass, missing GPIO/pulldown/bypass, wrong LED
supply, grounded final data output, disconnected flash, floating JST contact,
and disconnected battery positive. This is a deterministic diagnostic repair,
not a new full-brief success or a fabrication-qualified board.

Related corrections preserve legitimate shared connector contacts; distinguish
nominal RP2040 I/O voltage from its 1.62–3.63 V operating range; accept the Hall
sensor's reviewed VCC pin; require the phototransistor's collector supply; and
qualify relay channels by semantic channel number and the actual emitter return.
The A4988 mating footprint is explicitly a composite two-header interface, not
an invented manufacturer-orderable carrier socket.

The same frozen revision passed all 928 tests in the 15-module regression suite
in 336.12 seconds; its sole warning is KiCad's SWIG-interface deprecation.
The three original-brief RP2040 architecture replays subsequently committed
3/3: 183.628, 35.650 and 28.206 seconds, with costs of $0.009975035,
$0.001285620 and $0.001262120. The first needed four attempts; the others
needed one each. Original state and frozen-source hashes remained unchanged.
Evidence: `development/rp2040_bridge_stage_replay_v4` under the campaign root.
These are stage-only diagnostics, not three fulfilled or fabrication-ready
products. Development remains unqualified; the sealed replacement holdout
remains unopened.

### Physical candidate v4: spacing alone does not repair the RP2040

Frozen revision
`91dff7cd0834c2e3a7a02b0f87bfd8648b5c3033b143e5f78391286f2e46b811`
completed all six equal-budget replays in 6,905.92 seconds. Original projects
and input hashes remained unchanged.

| Fixture | Successful build exits | Durations (seconds) |
| --- | --- | --- |
| 9101, ESP32 indicator controller | 3/3 | 933.61 / 980.19 / 965.95 |
| 9102, RP2040 motor controller | 0/3 | 1269.39 / 1402.35 / 1351.26 |

The global 0.761 mm nominal escape-tip spacing filter drops alternate RP2040
pads instead of providing exits for them. Trial 1's CONTROL leaf reports 14
unconnected pads, including QSPI SD0/SD1/SD3, USB DM, MOTOR IN2 and alternating
GPIO contacts. This is worse than the earlier four-to-six-open leaf results;
the aggregate remains 3/6 and does not qualify the physical repair.
Evidence: `development/physical_candidate_v4`, particularly trial 1's
`routed_drc_report.txt` in subcircuit
`b1807844-5c2e-48a4-9af3-0431eb18947b__859487ba1d/renders`.

No routing rule, budget, clearance or final gate was waived. The next candidate
must preserve every proposed signal exit and prove the actual stamped geometry,
not merely improve nominal tip spacing or a router's internal counter.

### Native logger/display integration and clearance-aware escapes

The original-equivalent CH32 logger now constructs through `derive_architecture`,
recipe resolution and the production construction probe. Its 47 delivered native
footprints prove four ambient-light acquisition paths, three buttons, Qwiic,
18 V input and writable EEPROM hardware. SDA disconnection, write protection,
SWIO disconnection, ADC disconnection and a mismatched storage supply each remove
the logging pass. The mismatched-supply case previously passed; the observer now
requires the storage and complete MCU to share actual supply and return nets,
with four distinct supply/return/SDA/SCL nets. This is hardware capability, not
executed logging firmware or a routed export.

Construction fixes:
- Generic CH32 uses the sourced F4U6 variant. F4U6 and exact-only F4P6 now use
  acquired native bundles rather than nonexistent `MCU_WCH` stock symbols.
- The typed Qwiic lowerer preserves the reviewed SH connector's fixed contact
  map; it does not infer ordinal GPIO assignments.
- Storage uses orderable `24LC256T-I/SN` / C5458 with the narrow SOIC footprint,
  not the unavailable wide-package alternative.
- Both HUB75 recipes use ZHOURI `DC3-2.54-16PAS` / C2977594, a sourced 2x8,
  2.54 mm vertical IDC header. The manufacturer drawing specifies 8.4 mm body
  width, 8.6 mm height and 2.54 mm row spacing.

Native RP2040 compositions pass for the external WS2812 interface (75 parts),
four-digit MAX7219 display (80), HUB75 matrix interface (76), and AL8860 power
LED circuit (78). Each loses its pass when its control is disconnected or
disabled. The LED check exposed reversed SS14 wiring and an incorrect reviewed
SS14 pin map; both now match the native cathode-1/anode-2 symbol. The driver no
longer defaults CTRL to the potentially 12 V input: CTRL is a required binding,
and full-current evidence requires a suitable powered control voltage. Diodes
DS39014 Rev. 8-2 specifies a 6 V absolute maximum on CTRL and full current above
2.5 V; the observer conservatively limits fixed control supplies to 5 V.
The sense element is now the sourced 0.1 ohm, 1 W YAGEO
`RL2512FK-070R1L` / C326592 in 2512, not an unspecified 0603 at its nominal
power limit. Optical output, PWM shutdown guarantees and board thermal
performance are not established by these checks.

Verification: 532 tests passed across data logging, display evidence, lowerers,
recipes, converter-current evidence and electrical invariants, with one KiCad
SWIG deprecation warning. Native circuit evidence is recorded in the session
artifacts 732 and 744; this is not a population success-rate claim.

The next physical candidate fans tracks beyond the pad row before staggering
vias, using resolved pad clearance rather than fabrication-floor clearance
alone. It retains all signal exits instead of dropping alternate pads.
Four native escape tests pass, including actual stamping at 0, 90 and 37 degree
rotations. Six equal-budget full replays were launched from frozen source
`7b3855feba124c649d640888a987d54aeed4dcdde194875f9a0b012639a30ecf`
under `development/physical_candidate_v5`; no full-replay verdict is claimed
here. The originals, manufacturing rules and wall budgets remain unchanged.

### Battery-source integration and logger generation replay

The host-plus-battery circuit now constructs through architecture derivation,
recipe resolution and the production BOM/wiring compiler into a 39-part native
KiCad board. Its delivered pads prove the battery path through the source
selector and downstream regulator to a complete powered controller. Opening
the battery contact, forcing the selector disabled, or disconnecting the
downstream regulator input each removes the battery-operation pass. This
explicitly rejects a valid isolated supply stage with no powered controller.
Native evidence: session artifact 765; no routing/fabrication claim.

Construction exposed and repaired four integration defects: recipe-only sheet
metadata passed into the lowerer schema; lost repeated-component indices; an
undeclared selector-current-limit internal net; and unowned JST mounting pads.
The nonexistent polymer-capacitor footprint was replaced in both the recipe
and reviewed identity with installed `CP_EIA-3528-21_Kemet-B` (3528 body,
positive pad 1). Connector mounting pads 3/4 are explicitly unconnected.
The compensation record distinguishes its 192 uF design value from the
240 uF initial, 20 C minimum of the three-capacitor bank. It does not establish
aged/temperature-dependent capacitance, loop stability or board ampacity.

The original development-33 functional specification explicitly omitted a
battery path because chemistry/charging details were unspecified. The semantic
gate now requires a requested battery-powered function to include a battery
source and a power-flow path to a load; a source label or signal-only connection
does not suffice. Five regressions cover missing, isolated, signal-only and
connected sources, plus externally powered battery monitoring. Together with
the existing semantic and dual-source suites, 140 tests passed before the
native asset fixes. Original saved development states were not modified.

Frozen source `c80fa25f422275a7ca0514a92a708650a95eb75af2ea312ea5a2ef074c0db833`
completed the logger development-28 architecture diagnostic with **2/3 commits**.
The third trial was correctly refused: an MCP6002 requirement used the
unsupported `dual-operational-amplifier` family without a declared interface,
rather than the implemented analog-conditioner contract. Evidence is under
`development/logger_architecture_candidate_v1`; original-state and source
fingerprints remained unchanged. This is not N3 acceptance or a campaign yield.

After the native fixes, the combined semantic, dual-source, logger, display,
lowering, recipe, converter-current and electrical-invariant suites passed
**672 tests** (one KiCad SWIG warning; session artifact 767). The four new
CH32V003 F4U6/F4P6, DC3-2.54-16PAS and TPS2113A bundles also pass the real
`validate-part` command without rewriting hashes. The DC3 header reports the
generic missing side-opening warning; its reviewed drawing is top-entry.

Battery functional replay on frozen source
`d9f697599a068fcb880f53876e78b3978895721bfe09ba4ba0d19afd23d7d1c3`
committed **3/3** trials under `development/battery_functional_candidate_v1`,
with unchanged source/state fingerprints. Inspection found two remaining
defects despite those commits: the old assumption still denied a battery
path, and no light-acquisition function was present. New semantic checks now
reject both the contradictory assumption and a processor-only ambient-light
node. A direct invocation on the unchanged original state reports both
omissions. This replay therefore does not establish complete functional or
product acceptance.

### Composition identity preservation and independent board proofs

The logger's third-trial failure was traced to the compiler, not merely the
writer's family choice: `_adopt_carrier_families` changed a supported
`analog-conditioner` with exact MCP6002 into a bare dual opamp because the
published lowerer contract could name only one reviewed exact part. The
registration now exposes plural exact-part choices from the actual supported
amplifier table, and normalization preserves those implemented compositions.
Bare dual-opamp requirements still cannot bypass the conditioner contract.

The new four-screw-terminal I/O observer requires four physical contacts in
groups connected to a complete, observed-powered MCU. Every selected group
must carry GPIO, with any other contacts on the same controller's GPIO,
supply or return. It does not reinterpret four contacts as four GPIO channels.
A saved/reloaded native 30-part controller board passes with two signal+ground
terminal groups; disconnected, rail-only, foreign-return and overvoltage/
unpowered-controller variants fail.

A separate native 85-part two-LED-driver board reproduced an evidence-mixing
false positive: one driver had the valid current loop but disconnected control,
while the other had valid control but a reversed catch diode. The gate
previously passed. Loop and control proof are now bound to the same driver;
the broken pair fails and repairing either complete driver passes.

Removed the bare-IC `vin_max_v` to board-level `input_voltage_max_v` publication.
The native dual-source board had incorrectly reported 12 V despite its
5.25 V maximum host-source contract. An input rating needs actual path and
support-component bounds; until those are proved, that scalar remains absent,
not a successful input-envelope verdict.

Native integration smoke evidence (session artifact 778) covers the two-driver
counterexample, terminal I/O, the 39-part dual-source controller, the 47-part
logger, and all four display compositions. Intact circuits pass their targeted
gates and all exercised fault controls reject them. These are not routed-board,
thermal-characterization or population-yield results.

### Public sensor interfaces and complete logger acquisition

The frozen `1603d3a0eab944b0c4cc30682dac7bf48305408ef75b4649cb2cb0401c1f4188`
candidate passed 932 regression tests when each of the 13 suites ran in a fresh
process. An earlier combined invocation exited 137 without a test failure;
its cause was not established. Provider diagnostics on that candidate committed
all three battery functional trials and all three logger architecture trials.
Neither result is a product-success count. All three logger architectures still
lacked nonvolatile storage, and the chained battery architectures failed 0/3
with conflicting internal sensor bindings (one also chose an unsupported RGB
identity for a single-colour LED circuit).

The logger omission was reproduced against the unchanged development functional
specification: “measurements as logged data” did not match the storage completion
rule. The rule now recognizes that committed logging function. A second native
reproduction found storage completion incorrectly rejected multiple branches
of the same named I2C bus. It now attaches storage to that shared bus while
requiring distinct data/clock nets and rejecting ambiguous bus names.

The analog-conditioner catalogue now exposes supply, return and final consumer
outputs, not compiler-owned sensor/conditioned aliases. Low-level completed
requirements retain those private nodes; a writer cannot expose a pre-isolation
internal output as the requested interface. The existing compiler still derives
the internal wiring and still rejects real voltage/binding conflicts.

`temt6000-readout@1` provides the separate direct-readout circuit: reviewed
TEMT6000X01 collector supply, emitter output, 10 kOhm return load and 1 nF filter.
Its 3.3 V/5 V supply and consumer ceiling follow the
[Vishay datasheet](https://www.vishay.com/docs/81579/temt6000.pdf); the typical
photo-current calculation is not an illuminance calibration. Independent
observation requires a complete powered MCU ADC or an ADS1115 on that MCU's
readable I2C bus, with matching supply/return, decoupling, pull-ups and valid
non-colliding address straps. A powered but unreadable ADC previously passed
the new acquisition check; that false positive was reproduced and removed.
The same ADC-bus proof now supports logger acquisition through an external ADC.

Native saved/reloaded KiCad smoke evidence is session artifact 800: the
original-functional-spec logger works with the 47-part buffered conditioner,
39-part direct-MCU readout and 41-part ADS1115 readout. Each intact composition
passes the five targeted hardware gates: 18 V input, four light channels,
Qwiic, three buttons and writable logging storage. The 18 exercised intact/fault
cases reject broken collector/filter/channel paths, storage connectivity,
write protection, programming access, wrong storage voltage, ADC bus wiring
and floating ADC address as applicable. Losing one sensor correctly reduces the
observed acquisition count to three rather than erasing the surviving channels.
These remain hardware-construction checks, not routed/fabrication-qualified
products, executed firmware or measured analog performance.

## Disconnect handoff: current integration is not qualified

User explicitly requested stopping for disconnect and writing this continuation
handoff. No new provider runs or physical builds were started during wrap-up.
All agents and checks started by this session have finished. The provider and
physical-build slots are free. Nothing was deployed or restarted.

### Goal and non-negotiable custody

- Success means the **unchanged original brief is fulfilled and the actual board
  is fabrication ready**, not merely a stage commit, constructed BOM or passing unit test.
- Development exit is **57/60** complete products; release exit is **188/200**
  independently audited successes. Last full development baseline/regrade is
  **0/60**. No newer full-product success count has been established.
- This checkout is the production box. Follow `AGENTS.md`; do not deploy, restart,
  edit `.env`, or mutate original project states, briefs or evidence ledgers.
- Limit concurrency to **one provider job and one physical-build job**.
  Use disposable copies and frozen source snapshots; symlink `.env`/`.venv`,
  never copy secrets. Original source/state fingerprints must remain unchanged.
- Do not open sealed or quarantined holdouts, `ReviewPowerRatingProof`, or
  broadly search the campaign root. Work only in the development subtree.
  The fresh sealed replacement is `release-sealed-replacement-v1/bundle`,
  200 briefs / 2,827 obligations; custodian `SealedHoldoutObligations`.
  Its canonical hash is
  `593f0f2e9cf9d47f1a1934c5fe686a8016f213cb6515832b5f38ee2a53706010`;
  manifest hash is
  `3b824c9b77a44303b0277bd884a37eb31cc8608b32c11f7d294db2fad514b9f3`.
  Run it only after development qualification and candidate freeze.

Development root used below:
`logs/self_eval/surprise_completion/20261010T204651Z/development/`.

### Verified frozen candidate versus current working tree

Last broadly tested source snapshot:
`/tmp/kicraft-power-candidate-ag3qgkqa`, fingerprint
`9796e7172e032891307877a1c9d6bbba0aaaea68585039bd09edd5e0ac6a78d5`.
It includes the public/private sensor-port cutover, TEMT acquisition proof,
logger storage completion and shared ADS1115/I2C fixes described above.
It **does not include** the latest RGB, GPIO metadata, composition-contract,
edge-datum or pinned-diagnostic edits.

That frozen snapshot passed **1,035 tests in 15 isolated suites**. The earlier
native sensor/logger smoke covered 18 intact/fault cases on 47-, 39- and
41-part compositions. Neither establishes routed/fabrication-qualified products.

Current-tree wrap-up checks:

| Check | Observed result |
| --- | --- |
| Display evidence, electrical evidence, lowering: `pytest -q tests/test_display_artifact_evidence.py tests/test_electrical_artifact_evidence.py tests/test_design_lowering.py` | **269 passed**, one KiCad SWIG warning |
| Architecture, recipes, electrical invariants, auto-pin acceptance, autoexperiment scoring | **546 passed, 1 failed**, one warning |
| `python scripts/smoke_pinned_leaf_triage.py` | **Failed:** expected pinned R2 to report 6 opens, got `None` |
| Native RGB: `python /tmp/kicraft_rgb_indicator_smoke.py /tmp/kicraft-rgb-native-review` | **Failed:** RGB gate absent, `complete_mcu_refs=[]`; programming gate itself reported pass |
| `py_compile` of the seven geometry files listed below | Passed; syntax only |

Use `.venv/bin/python` for these commands from the checkout.
The failing regression is
`tests/test_electrical_invariants.py::test_shared_push_button_count_allocates_one_part_per_owner_and_keeps_total`.
Its three-button candidate is rejected with zero exact reviewed realizations.
**Cause not diagnosed**: do not weaken the invariant or assume the fixture is
wrong without examining the actual inventory/ownership path.
Combined five-suite command:
`.venv/bin/python -m pytest -q tests/test_architecture_intent.py tests/test_design_recipes.py tests/test_electrical_invariants.py tests/test_auto_pin_acceptance.py tests/test_autoexperiment_scoring.py`.

An initial test collection error from a missing required `guideline` argument
in the ESP32-C3 recipe was repaired by restoring the original Espressif URL.
The 269-pass and 546-pass/one-fail results were obtained after that repair.

### Provider diagnostics finished: latest architecture runs are 0/3 each

On the frozen 9796 candidate:

- `logger_architecture_candidate_v3`: **0/3 architecture commits**.
  Trial 1 left TEMT `vminus` unbound despite `reference_v="GND"`.
  Trials 2/3 hit CH32 programming-access detection and three shared button
  quantity rows each being interpreted as requiring three local buttons.
  Trial 2 also incorrectly assigned the light-sensor obligation to its ADC.
- `battery_architecture_candidate_v2`: **0/3 architecture commits**.
  Trials 1/3 reported the generic name “ESP32-C3 module” as missing an owning
  requirement. Trial 2 exceeded the collection limit.
- All original source/state fingerprints were retained unchanged.
  Earlier battery functional 3/3 and logger architecture 3/3 were diagnostic
  stage commits, not successful complete products; the old logger lacked storage.

Latest composition edits, not yet proven by a fresh provider replay:

- `design/analog_conditioner.py`: TEMT `vminus` declared as the real reference
  contact so generic completion can bind it to ground.
- `design/recipes/wave_a_mcus.py`: explicit generic family alias
  “ESP32-C3 module”; no fuzzy acceptance of exact unsupported order codes.
- `design/synthesis/validation.py`: shared quantity ownership allocation and
  WCH-Link recognition with physical SWIO/VDD/GND/NRST connectivity.
- Regressions in architecture, recipes and electrical-invariant tests.
  The shared-button regression remains failing as recorded above.
- ADC ownership and collection-limit failures were deliberately not suppressed.
  Retained manually added two-pin WCH-Link does not prove the complete
  recipe-owned recovery circuit.
- Unexecuted agent smoke: `/tmp/kicraft-composition-contract-smoke.sh`.

### Physical N=3 results and outstanding export custody

`physical_candidate_v5` used snapshot
`/tmp/kicraft-power-candidate-zafu2pwe`, fingerprint
`7b3855feba124c649d640888a987d54aeed4dcdde194875f9a0b012639a30ecf`.
The six sequential trials finished with original rules, seeds and briefs intact:

- `9101_trial_1`, `_2`, `_3`: build rc 0, **3/3 builds completed**.
- `9102_trial_1`, `_2`, `_3`: build rc 7, **0/3**.
- These are not yet independently audited complete-product acceptance counts.
- Six artifact-only `triage audits` commands returned rc 0. No library flags,
  Pass-A suspects, MPN mismatches or silent-substitution suspects were found.
  RP2040's two capacitor voltage upgrades are recorded substitutions, not hidden
  changes. Castellated contacts are fabricated board features, not missing
  purchasable parts.

For all three 9101 builds, actual promoted-board and ZIP SHA-256 matched the
fresh fabrication receipts. The promoted board differs from the resolved
routed-parent hash. Intentional final processing is a possible explanation,
**not established**. Next inspect
`kicraft/design/cli_app.py::_promote_verify_fab` (starts near line 6609;
only its first ~120 lines were read) and
`kicraft/design/synthesis/fab_export.py`, and prove verification/export bind to
the final board. Do not rerun builds merely to inspect these saved artifacts.
Fresh receipt names end `fab_20261011.receipt.json` for trials 1/2 and
`fab_20261012.receipt.json` for trial 3, under the generated ESP32 project.
Ignore inherited older receipts.

RP2040 diagnosis: the canonical best leaf is correctly pinned **round 2**,
with **6 opens**, not the stale debug/triage round 5 with 9 opens.
There was **no demonstrated scoring bug**. Round 2 has three internally failed
nets (QSPI_CS, QSPI_SD2, gpio7); do not confuse internal failures with total opens.
Retained witness:
`physical_candidate_v5/9102_trial_1/generated/RP2040_MOTOR_CONTROLLER/.experiments/subcircuits/b1807844-5c2e-48a4-9af3-0431eb18947b__859487ba1d/`.
Its `debug.json` is large; use narrow JSON selectors, not full dumps.

### Partial physical geometry edits: finish and review before any replay

The 34 required castellations were pinned to opposite sides of a ~212 mm seed
outline, then included as fixed extrema in the tight bounds, imposing a
~211.558 mm width floor. Shrinking translated rather than re-pinned contacts.
Nominal-pitch F.Fab rectangles also produced apparent adjacent-pad bbox overlap.
Do not remove contacts, waive actual overlaps, or increase budgets to hide this.

Agent `FitCastellatedLeaf` stopped on user request with **incomplete,
syntax-checked but runtime-unverified edits**:

- New `kicraft/autoplacer/brain/edge_datum.py`.
- `brain/leaf_geometry.py`, `brain/leaf_size_reduction.py`,
  `brain/subcircuit_extractor.py`, `brain/placement_solver.py`.
- `kicraft/autoplacer/hardware/adapter.py`.
- Geometry-only keyword at the canvas call in `kicraft/cli/solve_subcircuits.py`.

Intended behavior: exclude declared exact-edge interfaces from interior content
bounds, retain their pitch/depth floors, rebase interior content, reapply edge
placement before legality, and always reroute changed geometry. Model exact
castellation physical bounds from pad copper rather than the F.Fab pitch graphic.
Ordinary overlap checks must remain strict.

**Missing:** dedicated geometry/placement regressions and guards for
fixed/manual outline authority and mixed-edge/corner behavior. Agent explicitly
reported these as unfinished. Review before treating this as a safe fix.
Do not blindly execute the agent's proposed copy-and-delete build command:
create a uniquely named disposable workspace through the existing replay
pattern, preserve the original artifacts, freeze the reviewed source, and
honor the one-build slot. First run a small native geometry smoke, then N=3.

### Pinned diagnostic custody edits: unit coverage is not native proof

Agent `PinnedLeafReporting` edited:
`brain/pins.py`, `cli/solve_subcircuits.py`, `cli/autoexperiment.py`,
`cli/triage.py`, `tests/test_auto_pin_acceptance.py`,
`tests/test_autoexperiment_scoring.py`, and added
`scripts/smoke_pinned_leaf_triage.py`.

Intent: canonical debug follows the pinned round's saved validation/gate
evidence, retains all-round history, fails closed if evidence is unavailable,
and pins before parent composition. Triage reconciles old artifacts in memory.
The new unit tests passed in the five-suite run, **but the actual retained
witness smoke returned `None` rather than 6 opens**. No follow-up diagnosis was
performed. Inspect why the real saved snapshot cannot supply the expected
evidence; never invent a gate verdict. The smoke copies the retained leaf and
manifest to a temporary directory and does not modify the originals.
After native proof, remove this throwaway smoke script or retain only a genuine
reusable tool; no unreviewed scaffolding should ship.

### RGB and GPIO drive integration: native failure remains open

New `common-anode-rgb-indicator@1` lowerer in `design/lowering.py` uses exact
**APTF1616LSEEZGKQBKC**, a 5 V common anode, three MMBT3904 sinks, separate
current/base resistors and base pulldowns. Do not alias the hallucinated GQKBKC
order code. Proof is in `eval/display_artifact_evidence.py`, integrated into
`eval/electrical_artifact_evidence.py`.

`design/part_identity.py` adds Espressif loaded GPIO bounds for C3 MINI and
S3 WROOM/MINI modules: 3.3 V, PAD_DRIVER=3, per-pin source 40 mA/sink 28 mA,
VOH 2.64 V / VOL 0.495 V, with primary table citations. These are **conditional
per-pin specifications**, not aggregate simultaneous-output guarantees or
evidence that firmware was generated/configured. No ATtiny guarantee was invented.
RP2040 still lacks matching current-condition metadata.

Parent review removed a hardcoded Espressif-only metadata check that excluded
the existing scalar/ranged STM32/CH32 records. It also removed selection of the
strict RGB proof based on mutable `lowering_role`; the legacy transistor status
path now needs a real grounded emitter and base resistor path to a positive
source, not merely a collector connection. Add/verify native fault coverage for
open bases and stripped role metadata; the unit pass alone is insufficient.

Repaired disposable smoke:
`/tmp/kicraft_rgb_indicator_smoke.py` uses public architecture `ties`/`supply`,
the reviewed dual-source defaults, ESP32-C3, USB programming and RGB.
USB VBUS is separate from the system 5 V rail to avoid backfeeding.
It constructs and saves/reloads the native board, but fails the RGB gate:
`complete_mcu_refs=[]`. It still has **no fault-case loop**.

Next diagnostic lead, **inference not confirmed**:
`eval/mcu_artifact_evidence.py::_module_core_complete` requires
`_one_physical_header_reaches` on the MCU USB nets, which currently checks direct
connector-net membership only. A legitimate USB recipe series-resistor path may
not satisfy that detector. Inspect the actual native graph and every other
core requirement first. Do not weaken recovery proof to accept an open USB path.
The native output did show `gates["programming"] == "pass"`; that is not the same
as the independent complete-core proof.
The `_powered_output_sources` docstring still says “configured”; review that
wording against what is actually observed.

### Resume order and useful local scripts

1. Read this section, relevant `self-eval`, `kicraft-investigate`, and `verify`
   skills, then inspect current source; do not assume agent edits are qualified.
2. Resolve the failing three-button regression, retained pin-custody smoke, and
   native RGB complete-core proof. Preserve fail-closed and exact-identity checks.
3. Finish manual/fixed-outline and mixed-edge safeguards plus actual geometry
   regressions for the partial castellation work. Exercise the native geometry.
4. Finish the 9101 verification/export custody audit using saved artifacts.
5. Run native intact/fault smoke and relevant isolated regressions on the
   integrated source. Update this handoff with observed results, not expectations.
6. Freeze a new source candidate. Then run isolated N=3 provider diagnostics for
   logger and battery, and N=3 physical trials, respecting the two separate slots.
7. Continue ranked shared-capability fixes and rerun all 60 development products
   until at least 57 satisfy original obligations and fabrication readiness.
   Only then freeze and invoke the sealed 200-case release gate once.

Local scripts available on this host (read before executing):
`/tmp/kicraft-snapshot-source.py`,
`/tmp/kicraft-isolated-regressions.py` (15 suites in fresh processes),
`/tmp/kicraft-physical-trials.py`,
`/tmp/kicraft-power-stage-replays.py`,
`/tmp/kicraft-chained-stage-replays.py`,
`/tmp/kicraft-dual-source-native-smoke.py`,
`/tmp/kicraft-display-native-smoke.py`,
`/tmp/smoke_data_logging_storage.py`.
The TEMT logger throwaway script was deleted after its successful native proof.
No Eval variables/kernel state should be assumed to survive the next session.

Additional unresolved engineering limits: no board-level ampacity/copper-weight/
thermal model exists; do not turn an IC current rating into a board rating.
TPS61088 loop stability, capacitor DC-bias qualifications and real board thermal
performance are not established by the construction tests. TEMT's reused
footprint has not yet had a full manufacturer land-pattern audit.
