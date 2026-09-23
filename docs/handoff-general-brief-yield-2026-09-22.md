# Handoff: measuring and fixing "how often does KiCraft deliver a real board?"

Date: 2026-09-22
Author: this session (assistant), for the next engineer
Status: measurement machinery finished and trustworthy; delivery yield still 0%, with the
failure narrowed to one design stage and a clear list of candidate causes.

This document explains, in plain language, what was built, what was learned, what was
changed, what went wrong along the way (including mistakes I made), and what to do next.

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
