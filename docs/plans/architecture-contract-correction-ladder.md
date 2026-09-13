# Draft options + independent live-LLM tests: architecture contract convergence

**Status: options implemented and arm-tested; no arm ships.** Every option from §4 is implemented
behind `KICRAFT_CONTRACT_LADDER` (default `stock`, byte-identical to the shipped behaviour) at
commit `312a69f`, with a routing test per arm and the `tools/ladder_experiment.py` operator
harness. Round 1 ran every arm; round 2 re-ran the two nearest rivals **interleaved** with stock;
the O2 budget control ran interleaved too. **No arm is separable from stock at the pre-registered
sample size, so nothing ships and the default stays `stock`.** The headline result is not the arm
ranking but two corrections to §1 — the baseline is 0.60 (not ≈0/3) and the session drift makes
blocked arm ordering invalid — plus a re-derived option (O9, §8.5) that targets the dominant
defect class directly. Spend: ≈$4.2 of the $10 ceiling. Full record: **§8**.

**Subject:** the pipeline behaviours that make a brief die at `architecture` even when every
individual blocking diagnostic is actionable, the contract text already forbids the defect, and
the model demonstrably fixes what it is told about.

**Test brief (fixed for every arm — no paraphrasing, no variants):**

```
make a board with USB C PD power input (configured for 5V) to an ESP32-S3-WROOM-1-N16R8
that drives a HUB75 display. Include an output for driving addressable LED string and a
speaker.
```

**Boards in evidence:** `KC-M2DW6N` = `~/.kicraft/projects/1/824`, `KC-WGJ6XE` =
`~/.kicraft/projects/1/825` — both this brief, both `architecture` deaths, both after the
`ae3584c` recipe fixes.
**Predecessors:** `kc-2pfpvd-wiring-deadlock-escalation.md` (the clean-slate transition; the
"no second stall detector" rule), `llm-stage-evidence-driven-recovery-2026-09-11.md` (the live
A/B methodology), `recipe-and-deterministic-lowering-expansion-2026-09-09.md` (complete
deterministically instead of asking), `deepseek-v4-flash-json-budget-fix.md`.

---

## 1. Corrected framing: what the failures are *not*

I previously reported this class as "the ladder gives 2 corrections, the brief needs 3". Reading
the *canonical model-facing contract* (`stages/architecture.md`, 319 lines, given to the model
every architecture call) refines that — and it matters, because it changes which options are
worth testing.

**The contract already mandates exactly what the model failed to do.**

| observed defect | contract text the model was given |
|---|---|
| `native_usb_connector_required` — the model declared a power-only USB-C and no data connector | `architecture.md:90` — *"**Native-USB family ⇒ a physical USB data connector is MANDATORY.** … add a USB device requirement (role `connector`, family `usb-c-usb2-device`) on its own USB sheet, bind the MCU's `usb_dm` to `USB_D_N` and `usb_dp` to `USB_D_P`, and declare each of those as an inter-sheet net with TWO bidirectional endpoints"* |
| `unknown_recipe_port_net['HUB75_D']` in rung 2 after rung 1 had declared it | `architecture.md:170-171` — *"For each selected recipe, preserve every required external port listed in `extras.circuit_recipes`"*; `:221` — *"A compact recovery is still a complete architecture … Preserve … required nets"*; `:47` — never *"drop a required net to fit"* |

So these are **not** comprehension failures and **not** missing instructions: the rule, the
recipe family and even the net names are in the prompt, and the model fixed each defect the
moment it was named (`native_usb…` and `LED_DATA_OUT` both cleared on the next rung). What
remains is (a) **non-compliance on the first draft**, and (b) **losing previously-correct
content while regenerating** to fix a different defect.

### 1.1 The ladder (verified, $0)

Reproduce the call accounting with the repo's scripted client — canned replies, no provider
call, so this measures the **driver's control flow**, nothing about the model:

```python
# tests/test_stage_driver_retry.py provides _ScriptedClient + _OK_INTENT.
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "tests")
import test_stage_driver_retry as T
from kicraft.server.session import run_session

BAD = {  # schema-valid; always fails a deterministic contract (missing MCU ports)
    "sheets": [{"name": "MCU", "stem": "MCU", "function": "ESP32-S3-WROOM-1 microcontroller"},
               {"name": "POWER", "stem": "POWER", "function": "USB-C PD 5V input"}],
    "power_nets": ["GND", "+5V", "+3V3", "VBUS"], "rail_voltages": {"+3V3": 3.3, "+5V": 5.0},
    "comms_protocols": [], "mcu_present": True, "topologies": {}, "assumptions": [],
    "requirements": [{"id": "esp32_controller", "sheet": "MCU", "role": "mcu_core",
                      "family": "esp32-s3-module", "exact_part": "ESP32-S3-WROOM-1-N8R8",
                      "ports": {"gnd": "GND"}, "interfaces": []}],
    "inter_sheet_nets": [],
}
reply = lambda p: {"text": json.dumps(p), "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0}
client = T._ScriptedClient([reply(json.loads(T._OK_INTENT))] + [reply(BAD)] * 12)
client.s = T.Settings(api_key="test")
res = run_session(Path(tempfile.mkdtemp()), "USB-C PD 5V to an ESP32-S3-WROOM-1-N16R8",
                  ["intent", "architecture"], client=client)
arch = next(r for r in res["results"] if r["stage"] == "architecture")
print(len(client.calls) - 1, arch["attempts"], arch["failure_kind"])   # -> 3 3 contract_rejected
```

`attempts=3` while `provider_call_budget = 4` (`stage_runtime.py:3404`), because the ladder is
`normal → serialization → clean_slate` and **a rejected clean-slate escape is terminal
unconditionally** on this path (`if was_clean_slate: break`, `stage_runtime.py:3757` — no
signature comparison, unlike the commit path). Measured live on `1/825`: the smoke replay
reproduced `attempts=3` / `contract_rejected` for **$0.0165** in 54s.

### 1.2 The two boards were converging

| board | rung 1 (`normal`) | rung 2 (`serialization`) | rung 3 (`clean_slate`) |
|---|---|---|---|
| `1/824` | `unrealizable_power_requirement` | `multiple_recipe_contracts` | same → rejected → terminal |
| `1/825` | `native_usb_connector_required` + `unknown_recipe_port_net['LED_DATA_OUT']` | `unknown_recipe_port_net['HUB75_D']` | `HUB75_D` → rejected → terminal |

`1/825`: **2 defects → 1 defect → 1 defect.** The last rung is a *fresh* re-emit
(`_lean_retry(None, …)`, armed at `stage_runtime.py:3929-3932`) that discards the candidate,
so the rung that was one field away from committing was replaced by a from-scratch draft.

### 1.3 Breadth and baseline

- **70** of 540 local event streams carry the terminal-by-policy signature (schema-path
  terminal + a serialization rung spent + `attempts >= 3`): `architecture` 63,
  `functional_spec` 7. (Pre-`c410ad3` artifacts: rung deduced; post-`c410ad3`: recorded.)
- This brief's architecture stage: **0/2** committed across the two web runs, and **0/1** in the
  live smoke replay → baseline ≈ 0/3 before any arm.

---

## 2. Already fixed — do not redo

| fix | commit |
|---|---|
| `contract_rejected` split from `invalid_schema` (the label now names the cause) | `ae3584c` |
| reviewed order-code variants resolve instead of hard-blocking | `ae3584c` |
| HUB75 recipe ports are real signals (`r0…oe`), D on pin 12, GND on 4/8/16 | `ae3584c` |
| `retry.call_mode` records the rung; `triage` prints the ladder + `terminal BY POLICY` | `c410ad3` |
| `native_usb_connector_required` names the satisfying recipe | `c410ad3` |

---

## 3. How every option is tested — the shared protocol

Each option is tested **independently**: its own arm, its own knob, identical frozen source,
reviewed on its own primary endpoint. No arm's result may be inferred from another's, and every
arm is reported whether it wins or loses.

### 3.1 Frozen inputs (both are the pre-architecture state: `intent` + `functional_spec` committed)

```bash
S824=$HOME/.kicraft/projects/1/824/.kicraft/state.json   # KC-M2DW6N: named HUB75 + USB Type-C
S825=$HOME/.kicraft/projects/1/825/.kicraft/state.json   # KC-WGJ6XE: named USB-C
```

They are two independent draws of the brief. Every arm runs against **both** (a knob that only
helps one draw is not a general fix).

### 3.2 One option = one arm, one knob

- Source is identical across arms: implement every option behind **one** settings value,
  `KICRAFT_CONTRACT_LADDER` ∈ `stock | no_serialization | signature | preserving |
  dropped_gate | completing | full_feedback` (plus O2's pre-existing `--max-retries`, and O8's
  recipe/setting if adopted). Default stays `stock` until §6.4.
- Only the arm's knob differs from `stock`. Nothing else: same profile, model, temperature,
  prompt, budget cap.
- Arms run **sequentially** (shared spend guard; never concurrent).

### 3.3 Runner — no production change required

`drive_replay` accepts both sinks
(`stage_pipeline.py:179-188`: `budget_usd, max_retries, progress, core_defaults, client,
attempt_observer`). So an operator script gets the full metric set today:

```python
# tools/… (throwaway operator script; not production code)
from kicraft.server.stage_pipeline import drive_replay
events, attempts = [], []
out = drive_replay(
    state_path, "architecture",
    budget_usd=0.25, max_retries=3,          # 3 == production caller value
    progress=events.append,                  # retry/stage_done → call_mode + diagnostics
    attempt_observer=attempts.append,        # sanitized trace (rungs, outcomes)
)
```

Equivalently, per run via the CLI (one run at a time, keeping every trace):

```bash
KICRAFT_CONTRACT_LADDER=<arm> "$PY" -m kicraft.server.stage_driver replay \
  --state "$S825" --stage architecture --max-retries 3 --budget 0.25 \
  --trace-jsonl "/tmp/ladder-exp/<arm>-825-r<N>.jsonl"
```

**Verified measurement facts** (from the smoke replay's trace): `attempt_observer` records
`provider_attempt`, **`call_mode`** (`normal` | `serialization` | `clean_slate`), `outcome`,
`model`, `provider`, and — for commit-path rejections — `commit_result`/`rejection_signature`.
For **schema-path** rejections those three are `null` and there is **no diagnostic field**, so
the trace alone gives rungs and outcomes but **not** which defects each rung hit. The `progress`
sink supplies them: `retry` events carry `call_mode` **and** `diagnostic` since `c410ad3`.
Therefore: **primary endpoint from the trace, defect trajectory from `progress`** — and no
production change is needed before spending. (Adding the diagnostic to the observer record
remains a nice-to-have for `triage`; see §7.)

### 3.4 Metrics

Primary (the only input to §6.3):
- **architecture commit rate** over the arm's N, per state and pooled.

Secondary, for attribution:
- rungs taken; terminating rung; `attempts`; effective budget;
- **defect trajectory**: distinct blocking diagnostics per rung (converging / flat / shuffling);
- **dropped declared content**: `inter_sheet_nets[*].name` (and `requirements[*].ports` values)
  present in rung *i* and absent in rung *i+1* without any diagnostic naming them — this is the
  measured mechanism behind `HUB75_D`;
- cost per run, wall time, and every aborted/failed run.

### 3.5 Sample size and cost

Observed: `architecture` ≈ **$0.017–0.025 per run**, ≈ **54 s** wall.
N=10 per arm per state in round 1 (20 runs/arm ≈ $0.35–0.50, ≈ 18 min).

| round | arms | runs | est. cost |
|---|---|---|---|
| 1 — singles | O1–O8 (8 arms) | 8 × 20 = 160 | ~$3–4 |
| 2 — confirmation | top 2 arms, N=20 | 2 × 40 = 80 | ~$1.5–2 |
| 3 — combination | O5+k best pairing | 20 | ~$0.5 |
| 4 — L2 end-to-end | top 2 arms | 2 × 5 full chains | ~$1–3 |
| | | | **≈ $6–9.5 of the $10 ceiling** |

Hard stop at **$10** total for this plan; report actuals against it. Per-run cap `--budget 0.25`;
daily ceiling $20 and project budget $0.60 (`.env`) already permit this.

### 3.6 Decision rule (pre-registered, before any run)

1. Discard any arm that weakens a deterministic gate, breaks a test, or needs a contract-text
   change not covered by its option.
2. Rank by **architecture commit rate** (pooled over both states). Ties → lower cost → smaller
   diff → fewer new settings.
3. Adopt only if the winner beats `stock` by **≥ 2/N** with the interval reported. Otherwise
   record the negative result and keep `stock`; a negative result is a deliverable.
4. Re-run the winner and its nearest rival at N=20 and adopt on the pooled result.
5. Every arm states its **falsifier** (§4). An arm that wins while its falsifier obtains means
   §1's diagnosis is wrong: stop, revise the diagnosis, re-derive the options.
6. Report all runs, including aborted ones and their cost. No survivor-only reporting.
7. Independence: no cross-arm inference; each arm's verdict is computed from its own runs.

---

## 4. Options

Each is: **Fix** → **Mechanism** (verified `file:line`) → **Independent test** → **Success** →
**Falsifier** → **Risk / cost**.

### O1 — drop the serialization rung for contract rejections
**Fix:** on a schema/contract rejection, spend the next call on a *preserving correction* instead
of the dedicated "re-emit one compact JSON object" call.
**Mechanism:** `serialization_calls(0) >= serialization_budget(0)` on the first failure routes to
the plain-correction branch (`stage_runtime.py:3761`); the production knob already exists
(`Settings.serialization_retries`, `config.py:816`, env `KICRAFT_SERIALIZATION_RETRIES`).
Counted: `stock` = 3 calls (draft, serialization, clean-slate); O1 = **up to 4 calls, all
preserving, no clean-slate** (the clean-slate is armed only inside the serialization sub-path,
`stage_runtime.py:3929-3932`, so O1 also removes the fresh-start escape).
**Independent test:** arm `no_serialization` (`KICRAFT_SERIALIZATION_RETRIES=0`), N=10 × 2 states.
**Success:** commit rate ≥ stock + 2/N, and per-rung defect sets shrink.
**Falsifier:** rate unchanged/worse ⇒ more preserving corrections do not help (the model churns
defects regardless).
**Risk:** low (settings-only; no detector change). **Cost:** $0 code.

### O2 — more budget, same ladder  *(the control)*
**Fix:** none; tests whether the limit is the budget rather than the terminal rule.
**Mechanism:** `provider_call_budget = max_retries + 1` (`stage_runtime.py:3404`).
**Independent test:** arm `--max-retries 5` (budget 6), N=10 × 2 states.
**Success:** — none is expected.
**Falsifier of the diagnosis:** *any* improvement. If more budget helps, §1.1 is wrong (the
clean-slate terminal would not be binding), and every other option's rationale must be re-derived
before adoption.
**Risk:** none. **Cost:** $0 code.

### O3 — signature-aware clean-slate terminal
**Fix:** give the schema path the rule the commit path already has: a rejected clean-slate is
terminal **only when the rejection identity is unchanged**; when the defect set differs, continue
with ordinary preserving corrections.
**Mechanism:** replace the unconditional `if was_clean_slate: break` (`stage_runtime.py:3757`)
with a comparison against the armed rejection identity, reusing
`_commit_rejection_signature` semantics (`stage_runtime.py:4364` region) rather than inventing a
second detector — required by the `kc-2pfpvd` rule. **Substance of the option:** the schema path
has *no* signature today and `contract_rejected`'s message is a constant, so the identity must be
built from the diagnostic codes (e.g. the sorted set of blocking codes + the named nets).
**Independent test:** arm `signature`, N=10 × 2 states; also report how many runs still terminate
with an *unchanged* defect set (those must keep terminating).
**Success:** commit rate up, and the "no progress from a fresh start" behaviour preserved for
flat trajectories.
**Falsifier:** commits rise only where the trajectory was flat/same-defect ⇒ the rescue came from
defeating the no-progress rule, not from recognising progress.
**Risk:** medium (changes a terminal rule). **Cost:** ~1 day incl. tests.

### O4 — preserving clean-slate
**Fix:** keep the last rung but make it *preserving*: send the previous candidate plus every
current blocking diagnostic instead of a from-scratch slot.
**Mechanism:** the arming site builds `_lean_retry(None, …)` (`stage_runtime.py:3929-3932`);
change to `_lean_retry(raw, <all current diagnostics>)` for contract rejections.
**Independent test:** arm `preserving`, N=10 × 2 states.
**Success:** `dropped declared content` = 0 across rungs **and** commit rate up.
**Falsifier:** the same regression recurs with the candidate in the transcript ⇒ the loss is not
an information problem.
**Risk:** medium (message construction only; no new call, no new detector). **Cost:** ~1 day.

### O5 — mechanical dropped-content gate
**Fix:** *enforce* the preservation rule that the contract text already states
(`architecture.md:170-171`, `:221`) instead of asking for it again in prose: after a rejection,
diff the previously seen candidate's declared identities (`inter_sheet_nets[*].name`,
`requirements[*].ports` values) against the new one; any identity that disappeared and is not
named by a current diagnostic is a **regression**, corrected with one targeted message naming the
lost items.
**Mechanism:** new check in the correction path of `drive_stage` (the previous candidate `raw` is
already carried by `_lean_retry`); no gate is weakened — a dropped declared net is *more*
rejections, not fewer.
**Independent test:** arm `dropped_gate`, N=10 × 2 states; measure `dropped declared content` per
rung as the direct endpoint.
**Success:** drops → 0 and commit rate up.
**Falsifier:** the model still drops items when the exact lost names are listed ⇒ detection is not
the missing piece and the option is withdrawn.
**Risk:** low-medium (adds corrections, never accepts a weaker board). **Cost:** ~1 day.
**Why this is the lead hypothesis:** it targets the *measured* mechanism (§1.2) and the contract
already forbids the behaviour, so the gap is enforcement, not instruction.

### O6 — complete the native-USB companion deterministically instead of blocking
**Fix:** when a native-USB MCU recipe is selected and no USB data connector exists, *add* the
`usb-c-usb2-device` requirement bound to the MCU's own DM/DP nets — a deterministic completion,
the same posture the resolver already takes for other contracts — rather than emitting
`native_usb_connector_required` and spending a correction round.
**Mechanism:** `resolver._complete_native_usb_companions` (`resolver.py:1153`) currently blocks
(`:1183`); the rule and the exact recipe family/net names are already contract text
(`architecture.md:90`).
**Independent test:** arm `completing`, N=10 × 2 states; then **L2** (all five stages + build)
for 3 runs to prove the added connector flows through BOM/wiring.
**Success:** architecture commits on rung 1 for this brief, L2 still passes §9.29 programming
access and the board routes.
**Falsifier:** the deterministic connector breaks the programming-path gate, the connector
requirement conflicts with a model-declared power-only sink, or L2 fails where `stock` L2 passed.
**Risk:** medium-high — it turns a design decision into an automatic one. **Cost:** ~1-2 days.
**Design question this arm answers:** should KiCraft *complete* a contract-mandated part, or keep
the model accountable for it? (Precedent: `recipe-and-deterministic-lowering-expansion-2026-09-09.md`
prefers deterministic completion.)

### O7 — full-defect feedback per correction
**Fix:** every correction carries *all* current blocking diagnostics (each with its legal fix),
not only the current one.
**Mechanism:** `_retry_feedback` / `_stage_recovery_message` today carry the single rejection;
`RecipeResolutionError.diagnostics` already holds the full set.
**Independent test:** arm `full_feedback`, N=10 × 2 states.
**Success:** fewer rungs per convergence; commit rate up.
**Falsifier:** no change — **likely**, because the contract already states both mandates in prose
(§1), which makes "more prose" the weakest hypothesis and this arm the cheapest way to falsify it.
**Risk:** low. **Cost:** ~0.5 day.

### O8 — is HUB75's `addr_d` required at all? *(electrical, contract-coupled)*
**Fix:** decide whether the 13th channel is mandatory. `ae3584c` made all 13 HUB75 channels
`required=True` (`wave_c_interfaces.py:528-534`), which *introduced* a mandatory `HUB75_D` net and
was one of `1/825`'s three defects. A 1/16-scan panel does not use D.
**Mechanism:** keep required (complete 1/32 support, 13 buffered channels, contract-consistent
with `architecture.md:170-171`) **or** mark `addr_d` optional and tie the spare '245 channel low
in the recipe (no floating input; D optional at the interface) — the latter also needs the
contract text and the completeness checks updated.
**Independent test:** arm `addr_d_optional` (recipe + setting), N=10 × 2 states.
**Success:** the defect count for this brief drops by one with no new failure class, and no board
that *omits* D fails a socket/buffer/ERC check.
**Falsifier:** boards omitting D then fail completeness, or the tie-off shows an ERC/floating-input
defect.
**Risk:** medium — an electrical contract change needs a source-grade justification and a decided
panel default. **Cost:** ~0.5 day + fixture.

### Ranking hypothesis (to be tested, not assumed)
O5 (targets the measured mechanism, no gate weakening) → O3/O4 (target the terminal/regeneration
mechanisms) → O6 (removes a defect deterministically; highest blast radius) → O1 (blunt) → O7
(likely null) → O2 (control) → O8 (independent, electrical).

---

## 5. Combination round

Options are not mutually exclusive. After round 1, take the best arm and pair it with O5 or O3
(whichever was not in the pair), and test one combined arm under the same protocol.
**Pre-registered rule:** the combination ships only if it beats **both** singles; otherwise ship
the single. Combinations never justify a larger budget cap or a weakened gate.

---

## 6. Execution order, acceptance, rollout

1. **Instrumentation that the experiment genuinely needs** — only what §3.3 lacks: an operator
   script that writes `progress` events and the observer trace per run, plus the arm manifest
   (§3.2). No production change is required for round 1. *(Skip §4.1/§4.2 of the earlier revision:
   the effective-budget and per-rung-diagnostic telemetry is a `triage` improvement, not a
   prerequisite — `progress` already carries both.)*
2. Implement every option behind `KICRAFT_CONTRACT_LADDER` (default `stock`) with a **routing
   test per mode** using the scripted client — no LLM: assert which call each mode makes and
   whether the terminal rule fires. This is what makes a null live result interpretable.
3. Run round 1 (O1–O8), then round 2 (N=20 on the top two), then §5's combination, then L2 for
   the winner. Keep every trace, event log and manifest.
4. Ship the winner (`KICRAFT_CONTRACT_LADDER` default flipped **or** the adopted recipe/contract
   change) only after round 2, with: full suite green; the winner's rungs visible on one fresh
   live run of the brief; `deploy/deploy-production.sh` (web 200 + `[build-worker] ready`). The
   34-brief canary stays out of the deploy path (`AGENTS.md`); run it manually if wanted.
5. Post-deploy: re-scan the corpus after ~20 subsequent `architecture` runs and report the
   terminal-by-policy count before/after (§1.3) — not a single-board anecdote.

## 7. Reader / skill follow-ups still open (not blockers for §6)

| # | gap | fix |
|---|---|---|
| D1 | `triage stages` prints the ladder but not the **defect trajectory** | diff consecutive rungs' diagnostics and print `rung 1: 2 defects -> rung 2: 1 defect (HUB75_D lost)` |
| D2 | `scan` ranks `stage_diag` for the **terminal** rung only | rank diagnostics across all rungs (which contracts the model churns, corpus-wide) |
| D3 | the effective ceiling is derived, not recorded | record `attempt_budget` + rungs in `stage_done`/`StageStatus`; also add the schema-path diagnostic to the observer record so `--trace-jsonl` is self-sufficient |
| D4 | the two `samples.py` hero briefs are this brief class | after the winner ships, add the brief to the **manual** `deploy/verify-design-canary.sh` subset (never the deploy gate) and record its commit rate |

## Non-goals

- No weakening of any deterministic gate; no netlist normaliser to paper over a dropped net.
- No second stall detector: any clean-slate change reuses the existing
  `_commit_rejection_signature`/`next_attempt` semantics (`kc-2pfpvd` rule).
- The **commit** path's ladder already has the signature rule and is out of scope; this plan is
  the schema/contract path.
- No new recipe coverage for this brief (`ae3584c` closed it; `1/784`, `1/700` prove the brief
  commits).
- No prompt-example tuning beyond O7's full-defect feedback.
- No change to the 34-brief canary or the deploy gate. No spend beyond the $10 ceiling.

---

## 8. Execution record (this run)

### 8.1 The $0 measurement that reframed the plan

Each rung of the two historical boards is reconstructible from the `answer_delta` stream, so the
defect trajectory replays offline through `_normalize_stage_response` (no model call):

| `1/825` rung | mode | blocking defect | declared content lost vs the previous rung |
|---|---|---|---|
| 1 | normal | `native_usb_connector_required` (ESP32) + `unknown_recipe_port_net['LED_DATA_OUT']` | — |
| 2 | serialization | `missing_recipe_port` (ws2812 `data_out`) | `port:LED_DATA_OUT`: the binding was deleted instead of the net declared |
| 3 | clean_slate | `unknown_recipe_port_net['HUB75_D']` → terminal | `net:HUB75_D` |

Corrections to §1.2: the content loss happened on the **clean-slate** rung, and that rung's own
diagnostic **did** name the net it lost. The "without any diagnostic naming them" clause holds
for the first loss (`port:LED_DATA_OUT` — the replacement diagnostic names only the port), not
the second, so O5's suppression rule is a no-op on the second loss by construction.

### 8.2 What each arm does (scripted client, no model)

`tests/test_stage_driver_retry.py::test_ladder_*` pins the call sequence per arm and whether the
terminal rule fires; `tests/test_design_recipes.py` pins the two deterministic completions at the
recipe level. Confirmed live: `no_serialization` spent **0 of 62** calls on a serialization rung
and never armed the clean-slate escape; stock spent 20 serialization + 4 clean-slate calls in 21
runs.

Deviations from the drafted options, both to avoid inventing hardware:

* **O6** completes the connector the board *already* declared: a power-only USB-C sink whose own
  sheet carries the MCU's D+/D- pair becomes the `usb-c-usb2-device` the contract mandates
  (verified on the real `825` rung-1 candidate: `native_usb_connector_required` disappears, two
  defects → one). A connector requirement is added only when *no* requirement owns the MCU's USB
  pair, so the arm never puts a second socket on the board.
* **O8** ties the unused 13th HUB75 channel to GND instead of "marking `addr_d` optional": the
  optional flag alone leaves the spare '245 input on a one-pin literal net, which §9.15 rejects
  — the plan's own falsifier, measured before choosing. The tie-off keeps the input off the
  float and the connector position driven, asserted pin-by-pin (`U2.3` on GND, `shifted9` =
  `U2.17` + `J1.12`).

### 8.3 Round 1 — every arm, N=10 per state, run as blocks

| arm | commits/runs | rate | Wilson 95% | per board |
|---|---|---|---|---|
| stock | 12/20 | 0.60 | [0.39, 0.78] | 824 7/10, 825 5/10 |
| O1 no_serialization | 6/20 | 0.30 | [0.15, 0.52] | 824 3/10, 825 3/10 |
| O3 signature | 6/20 | 0.30 | [0.15, 0.52] | 824 3/10, 825 3/10 |
| O4 preserving | 6/20 | 0.30 | [0.15, 0.52] | 824 2/10, 825 4/10 |
| O5 dropped_gate | 4/20 | 0.20 | [0.08, 0.42] | 824 2/10, 825 2/10 |
| O6 completing | 5/20 | 0.25 | [0.11, 0.47] | 824 4/10, 825 1/10 |
| O7 full_feedback | 3/20 | 0.15 | [0.05, 0.36] | 824 0/10, 825 3/10 |
| O8 addr_d_optional | 9/20 | 0.45 | [0.26, 0.66] | 824 5/10, 825 4/10 |

Two premises of §1 are falsified or relocated:

* **The baseline is 0.60, not ≈0/3.** The same frozen `825` state commits about half the time;
  the historical deaths — and this plan's own smoke replay — are the tail of a stochastic
  process, not a systematic death. Every arm's interval overlaps stock's, and all point
  estimates sit *below* it, which in a blocked design is the signature of drift, not of knobs.
* **The measured mechanism (§1.2) is real**: 15 of 21 stock runs lost at least one declared net
  or port binding between rungs, `LED_DATA_OUT` 13 times.

### 8.4 Round 2 — interleaved (drift-controlled), and the O2 control

Round 1 ran arms as blocks, so model/provider drift over the session is indistinguishable from an
arm effect. Round 2 interleaved `stock` with its two nearest rivals run-by-run (N=10 per state,
20 pooled per arm), and the O2 control interleaved stock against stock with `--max-retries 5`
(a provider-call budget of 6):

| label | commits/runs | rate | Wilson 95% | per board |
|---|---|---|---|---|
| interleaved stock | 4/20 | 0.20 | [0.08, 0.42] | 824 1/10, 825 3/10 |
| interleaved O8 addr_d_optional | 10/20 | 0.50 | [0.30, 0.70] | 824 5/10, 825 5/10 |
| interleaved O4 preserving | 9/20 | 0.45 | [0.26, 0.66] | 824 6/10, 825 3/10 |
| O2 stock, budget 4 | 6/20 | 0.30 | [0.15, 0.52] | 824 2/10, 825 4/10 |
| O2 stock, budget 6 | 6/20 | 0.30 | [0.15, 0.52] | 824 3/10, 825 3/10 |

Three results:

1. **Drift is confirmed and large**: stock measures 0.60 in round 1 and 0.20 in round 2 on
   identical inputs (and 0.30 in the O2 window). Round 1's arm ordering is therefore not evidence
   about the arms — the direction reverses when interleaved.
2. **The interleaved ordering cannot be attributed either**: the arm mechanisms do not fire often
   enough to explain +25 to +30 points. Every `missing_recipe_port` in the stock artifacts is the
   ws2812 `data_out` port; `addr_d` appears in 3 of 40 stock artifacts, so O8's tie-off is nearly
   a no-op on this brief. Run-to-run overdispersion dominates: the clean-slate rung is reached in
   4/21 stock runs but 8/20 `preserving` runs, a count the knob cannot change (it only alters the
   message *after* the serialization failure).
3. **O2 is a clean null**: doubling the budget changes nothing (6/20 vs 6/20; the 6 runs that
   used the extra calls all still failed). The binding constraint is the terminal rule and the
   defect mix, not the number of attempts — the O2 falsifier does not obtain.

**Verdict (rule 3 and rule 5):** no arm is separable from stock at N=10 per state
(`addr_d_optional` vs interleaved stock is Fisher p ≈ 0.07; the rest are worse), so the shipped
default stays `stock` and the negative result is recorded. The experiment is *inconclusive*
rather than the options refuted: this sample size cannot resolve a 0.2–0.5 difference under this
overdispersion, and the pre-registered N is the limiting factor. A future round needs the
interleaved design and N ≈ 40 per state per arm to separate an effect that size.

### 8.5 Re-derived diagnosis and the option that follows (rule 5)

The rung-1 blocking code is a **deterministic contract gap**, not a ladder defect:

* in 16 of 21 stock runs the first rejection is `unknown_recipe_port_net` — a requirement binds a
  recipe port to a net the architecture never declares;
* every `missing_recipe_port` in the corpus is the ws2812 `data_out` port: "Include an output for
  driving addressable LED string" means precisely this port, and the model binds it
  (`data_out: LED_DATA_OUT`) without declaring `LED_DATA_OUT` as a net;
* the model's repair for that diagnostic is often to **delete the binding** (§8.1/§8.3), which is
  why the class repeats across rungs.

**Proposed O9 (not implemented, not run — it needs its own pre-registration):** complete the
declaration deterministically, in the posture of O6/O8 — when a requirement binds a recipe port
to a net the architecture does not declare, declare that net with the endpoint the requiring
sheet needs (plus the peer endpoint any other requirement binding the same net implies).
Falsifier: the completion invents a net the model never intended (caught by the downstream
wiring/BOM gates) or a build fails where stock passed. This targets the dominant defect class
itself; the ladder arms only target its consequences.

### 8.6 Steps not run, and why

* **Round 3 (combination)** — §5 conditions it on "the best arm"; no arm beat stock, and the same
  N cannot resolve a pair either.
* **Round 4 (L2)** — §6.4 conditions it on a winner; nothing ships, so no full-chain spend.
* **Deploy** — no winner to ship. The arms and the correction-event telemetry are committed but
  **not deployed**: the running services still execute the previous code, and the default stays
  `stock` until an arm's default is deliberately flipped.
* **Step 5 corpus rescan** — the "before" count is recorded in §8.8; there is no shipped change
  to re-scan after. Re-run it after any default flip.

### 8.7 Spend, artifacts, harness findings

* Spent on this plan ≈ **$4.2 of the $10 ceiling** (round 1 + round 2 + O2 + two validation
  replays; the day's ledger total is $4.39, which includes pre-experiment traffic).
* Artifacts: `/tmp/ladder-exp/` — `runs.jsonl` (one row per run: commit, rungs, per-rung defect
  codes, lost declared content, cost, `at_utc`, artifact paths), `manifest-<label>-<board>.json`
  (arm, knobs, model, settings, repo head), `<label>-<board>-<stamp>.{events,trace}.jsonl`, and
  `analyze.py` (Wilson intervals, ladder shapes, defect trajectory, losses).
* Harness / operations findings:
  1. A supervised process inherits this session's stale `KICRAFT_*` environment, which overrides
     `.env` (`load_dotenv` never overrides) and carried an invalid
     `KICRAFT_DESIGN_PROFILE=flash`. Arm drivers must clear inherited `KICRAFT_*` first; the same
     hazard applies to `deploy/restart-*.sh` when run from such a shell.
  2. `--runs 1` (needed for interleaving) overwrote the previous run's artifacts until the naming
     gained a timestamp. Round 2's per-run events/traces are therefore incomplete (6 of 60
     survive); its `runs.jsonl` rows are intact, and the primary endpoint needs only those.
  3. The serialization rung still emits no `retry` event, so the defect trajectory misses that
     rung (plan item D3). The `declared_identities` key added to correction events makes the
     dropped-content metric computable without the raw reply.

### 8.8 Corpus "before" count (terminal-by-policy)

`tools/ladder_experiment.py --scan` (production `triage` semantics over 48 local project event
streams): `architecture` 2 terminal-by-policy, `wiring` 1, every other stage 0. §1.3's 70-of-540
used a wider corpus and a deduced-rung signature; this is the precise post-`c410ad3` count on the
local project streams, and the reference number for a post-flip rescan.
