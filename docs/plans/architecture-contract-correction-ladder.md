# Architecture contract convergence — the correction ladder

**Status:** plan only. The §2 "already fixed" items are shipped (`ae3584c`, `c410ad3`); the
options in §5 are **not** implemented, and §6/§7/§8 are the work this document authorises.
**Subject:** every brief whose `architecture` stage dies on *sequential, individually
actionable* contract diagnostics — the model fixes the flagged defect, the next attempt
surfaces (or regresses) another, and the ladder runs out before the candidate is clean.
**Test brief (fixed for every experiment here):**

```
make a board with USB C PD power input (configured for 5V) to an ESP32-S3-WROOM-1-N16R8
that drives a HUB75 display. Include an output for driving addressable LED string and a
speaker.
```

**Boards in evidence:** `KC-M2DW6N` = `1/824`, `KC-WGJ6XE` = `1/825` (both this brief).
**Predecessors:** `docs/plans/kc-2pfpvd-wiring-deadlock-escalation.md` (the clean-slate
transition and the "no second stall detector" rule), `docs/plans/llm-stage-evidence-driven-recovery-2026-09-11.md`
(the owning 34/34 loop and its live A/B methodology), `docs/plans/deepseek-v4-flash-json-budget-fix.md`.

---

## 1. Why this is not "the model can't do it"

The mechanism is verified, not inferred. Reproduce the call accounting with the repo's
scripted client (**no LLM, $0**):

```python
# tests/test_stage_driver_retry.py already provides _ScriptedClient + _OK_INTENT.
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "tests")
import test_stage_driver_retry as T
from kicraft.server.session import run_session

BAD = {  # schema-valid, always fails a deterministic contract (missing MCU ports)
    "sheets": [{"name": "MCU", "stem": "MCU", "function": "ESP32-S3-WROOM-1 microcontroller"},
               {"name": "POWER", "stem": "POWER", "function": "USB-C PD 5V input"}],
    "power_nets": ["GND", "+5V", "+3V3", "VBUS"], "rail_voltages": {"+3V3": 3.3, "+5V": 5.0},
    "comms_protocols": [], "mcu_present": True, "topologies": {}, "assumptions": [],
    "requirements": [{"id": "esp32_controller", "sheet": "MCU", "role": "mcu_core",
                      "family": "esp32-s3-module", "exact_part": "ESP32-S3-WROOM-1-N8R8",
                      "ports": {"gnd": "GND"}, "interfaces": []}],
    "inter_sheet_nets": [],
}
reply = lambda p: {"text": json.dumps(p), "reasoning": "", "finish_reason": "stop",
                   "cost_usd": 0.0}
client = T._ScriptedClient([reply(json.loads(T._OK_INTENT))] + [reply(BAD)] * 12)
client.s = T.Settings(api_key="test")
res = run_session(Path(tempfile.mkdtemp()), "USB-C PD 5V to an ESP32-S3-WROOM-1-N16R8",
                  ["intent", "architecture"], client=client)
arch = next(r for r in res["results"] if r["stage"] == "architecture")
print("arch calls        :", len(client.calls) - 1)                      # -> 3
print("modes             :", ["serialization" if c["serialization"] else "normal"
                              for c in client.calls[1:]])              # -> normal, serialization, normal
# NOTE: the client's `serialization` flag is False for BOTH plain and clean-slate
# calls; the third call is the clean-slate rung. The rung is what `call_mode`
# records (trace observer + `retry` events since c410ad3), not this flag.
print("reported attempts :", arch["attempts"], arch["failure_kind"])    # -> 3 contract_rejected
# and because _stage_max_retries("architecture", 2) == 3, provider_call_budget == 4
```

That last line is the whole finding: **4-slot budget, 3 calls, terminal.**

### 1.1 The ladder, verified

`architecture` with the default profile makes **3 provider calls**, in this order:

| call | rung | what it is | code |
|---|---|---|---|
| 1 | `normal` | the first draft | — |
| 2 | `serialization` | one dedicated tool-free "re-emit one compact JSON object" call | `stage_runtime.py:3778` |
| 3 | `clean_slate` | a fresh re-emit from the binding state, previous candidate **discarded** | armed at `stage_runtime.py:3929-3932` |

- `max_retries = _stage_max_retries("architecture", 2) = 3` (`stage_runtime.py:253,263`) →
  `provider_call_budget = 4` (`stage_runtime.py:3404`) and `for attempt in range(4)`
  (`stage_runtime.py:3442`). **The 4th slot is never spendable on this path**: after the
  clean-slate call, `if was_clean_slate: break` (`stage_runtime.py:3757`) ends the stage,
  and — unlike the commit path — **no signature comparison happens first**.
- So a contract-rejection sequence gets **two corrections after the first failure**, and
  `attempts = 3` with `budget = 4` means "terminal by policy", **not** "out of budget".

Measured on the live board (`triage stages KC-WGJ6XE`):

```
attempt ladder: 1 normal(contract_rejected) -> 2 serialization(contract_rejected)
                -> 3 clean_slate(contract_rejected)   [rungs inferred — pre-2026-09-13 artifact]
** terminal BY POLICY: … leaving a slot of its 4-call budget unspent. **
```

### 1.2 The two boards were *converging*

| board | rung 1 defects | rung 2 | rung 3 (`clean_slate`) |
|---|---|---|---|
| `1/824` | `unrealizable_power_requirement` | `multiple_recipe_contracts` (variant + HUB75 endpoint) | same as rung 2 → rejected |
| `1/825` | `native_usb_connector_required` + `unknown_recipe_port_net['LED_DATA_OUT']` | `unknown_recipe_port_net['HUB75_D']` | `HUB75_D` again → rejected |

`1/825` went **2 defects → 1 defect** across rungs, then the fresh-start rung threw away the
nearly-correct candidate and reproduced the remaining one. `HUB75_D` is present in attempt 1's
`available_nets=` and absent from attempt 2's — the model regressed its own correct content
while fixing a different defect.

### 1.3 What is *not* the problem

- **Not coverage.** The brief commits today: `KC-JPYUYS` (`1/784`) and `KC-KJ76KX` (`1/700`)
  built it with one USB-C (`TYPE-C-31-M-12`) serving PD power *and* USB data. The recipe
  coverage for the named part was fixed in `ae3584c`.
- **Not the contract being unsatisfiable.** Each diagnostic is actionable and the model
  demonstrably fixed each one it was shown (`native_usb…` and `LED_DATA_OUT` both cleared).
- **Not budget.** Predictions in §5 are pre-registered precisely so the budget arm (O2) can
  falsify this claim.
- **Not the model choice.** Both boards ran `openai/gpt-5.6-luna` — the production profile.

### 1.4 Breadth

Across the 540 local event streams, **70** carry the terminal-by-policy signature
(schema-path terminal + a serialization rung spent + `attempts >= 3`): `architecture` 63,
`functional_spec` 7. For pre-`c410ad3` artifacts the rung is *deduced* (a failed serialization
is always followed by the clean-slate escape); post-`c410ad3` runs record it exactly.

---

## 2. Already fixed — do not redo

| fix | commit | effect |
|---|---|---|
| `contract_rejected` split from `invalid_schema` | `ae3584c` | the label names the cause; both take the same correction path (`_SCHEMA_REJECTION_KINDS`) |
| reviewed order-code variants resolve instead of hard-blocking | `ae3584c` | `ESP32-S3-WROOM-1-N16R8` binds the WROOM-1 recipe with a recorded deviation |
| HUB75 recipe ports are the real signals (`r0…oe`), D on pin 12 | `ae3584c` | `HUB75_OE`/`HUB75_D` are bindable; pin 12 is no longer grounded |
| `retry.call_mode` records the rung | `c410ad3` | the ladder is observable |
| `triage` reconstructs + prints the ladder; `terminal BY POLICY` | `c410ad3` | `attempts < budget` is no longer misread as breadth |
| `native_usb_connector_required` names the satisfying recipe | `c410ad3` | `canonical choice: recipe=usb-c-usb2-device@1; …` |

---

## 3. The decision this plan exists to make

> Which of O1–O6 (§5) improves the architecture commit rate enough to justify its risk —
> chosen by a pre-registered live A/B on the fixed brief, not by argument.

Three things must be true before any arm can win: it must **not** weaken a deterministic gate,
it must **not** add a second stall detector (`kc-2pfpvd` rule: reuse the existing
exact-repeat/clean-slate transition), and it must be measurable from artifacts.

---

## 4. Instrumentation to land *before* the first paid arm

Without these the comparison is unmeasurable or unfair.

1. **Effective budget in telemetry.** Add `attempt_budget` (and the rungs taken) to the
   `stage_done` event and `StageStatus`, so `triage` stops deriving a floor
   (`stage_runtime.py:3404`; `design/models.py` `StageStatus`). Removes the floor/ceiling
   ambiguity permanently.
2. **Per-rung diagnostics — genuinely missing today.** `--trace-jsonl` is the right instrument
   but it is incomplete for this purpose. `_observe_attempt` (`stage_runtime.py:648-680`)
   records, per provider attempt: `provider_attempt`, **`call_mode`**, `outcome`,
   `candidate`, `commit_result`, `rejection_signature`, `clean_slate_*`, `escalated`,
   `unit_*`. It is invoked for a schema-path failure with **no `candidate` and no
   `commit_result`**, so the trace carries the *rung* and the *outcome kind* but **not the
   diagnostics** that rung was rejected for. The defect trajectory (§1.2) is therefore not
   measurable from the trace as-is. Land the observer record's schema-path diagnostic
   (e.g. pass the normalized `diagnostic`/`schema_error` through `commit_result` or a new
   `diagnostic` field) and read it in `triage`.
   *Fallback if that is deferred:* the replay CLI does not persist progress events (already
   noted in `kc-2pfpvd-wiring-deadlock-escalation.md`), so drive the arms through a short
   operator script calling `drive_replay(..., progress=collector.append)` and capture the
   `retry` events (which now carry `call_mode` + `diagnostic`).
3. **Arm manifest.** One frozen record per arm: source SHA, `KICRAFT_*` env, profile, model,
   `--max-retries`, `--budget`, frozen state path, `--trace-jsonl` path. The repo rule is
   same-code comparison across arms — the *only* permitted difference is the arm's knob.
4. **Confirm the lever is live.** `Settings.serialization_retries` (`config.py:816`, env
   `KICRAFT_SERIALIZATION_RETRIES`, `config.py:588`) already reaches `StageResponsePolicy`;
   the `serialization_retries=1` at `stage_runtime.py:1016` is only the fallback for clients
   without `design_stage_policy`. **Verify on the production profile** before arming O1,
   because the whole arm depends on it.
5. **CLI defaults.** `replay --max-retries` defaults to `2` (`stage_driver.py:74`), which is
   the production caller value → `_stage_max_retries("architecture", 2) = 3` → budget 4. Arms
   must state their value explicitly rather than inherit it silently.

---

## 5. Options

Each is stated as mechanism → predicted effect → **falsifiable prediction** → risk → cost.
The falsifiable prediction is the point: an arm that wins while its prediction fails means the
§1 diagnosis is wrong and the plan must be revised before adopting anything.

### O0 — baseline (control)
Default ladder, default budget. Expected: architecture commit rate = the observed one
(2/2 attempts of this brief died; `1/824`, `1/825`).

### O1 — no serialization rung for contract rejections
`KICRAFT_SERIALIZATION_RETRIES=0` → `serialization_calls(0) >= serialization_budget(0)` is
true on the first failure (`stage_runtime.py:3761`), so every schema-path failure takes the
plain-correction branch and the budget is spent on corrections.
Counted precisely: baseline = **3 calls** (draft, serialization re-emit, clean-slate fresh
start); O1 = **up to 4 calls** (draft + 3 preserving corrections, each with the previous
candidate + the current diagnostics).
**Side effect that must be stated:** the clean-slate rung is armed *only* inside the
serialization sub-path (`stage_runtime.py:3929-3932`), so O1 removes the fresh-start escape
entirely on this path — corrections only, no escalation.
**Predicted:** commit rate rises (3 corrections vs 2 rounds, and no rung discards the
candidate). **Falsified if** commit rate is unchanged or worse — which would mean more
corrections do not help because the model churns defects regardless.
**Risk:** low (settings-only on the production path; no detector changes). **Cost:** $0 code.

### O2 — more budget, same ladder
`stage_driver replay --max-retries 5` (budget 6).
**Predicted: NO improvement** — the terminal rule is `if was_clean_slate: break`, so the extra
slots are unreachable. **This arm is the control for the diagnosis.** If it *does* improve the
commit rate, §1.1 is wrong and every other arm's rationale needs re-deriving.
**Cost:** $0 code.

### O3 — signature-aware clean-slate (reuse the commit path's rule)
Give the schema path the rule the commit path already has: a rejected clean-slate is terminal
**only when its rejection signature is unchanged**; a *different* signature continues with
ordinary preserving corrections (`next_attempt` semantics, `stage_runtime.py:4364` and
`_commit_rejection_signature`). This reuses the existing transition rather than adding a
detector, per the `kc-2pfpvd` constraint.
**Predicted:** commit rate rises for *converging* runs (§1.2) and stays flat for runs that
repeat one defect — the intended "no progress from a fresh start is still terminal" behaviour
is preserved. **Falsified if** the defect-set trajectory is flat (repeat) yet commits rise.
**Risk:** medium — changes a terminal rule; needs the signature to be meaningful on this path
(note: the schema path currently has *no* signature at all, and `contract_rejected`'s error
text is a constant — so the signature must come from the diagnostic codes, not the message.
That is the substance of this option).
**Cost:** ~1 day incl. tests.

### O4 — preserving clean-slate
Keep the rung but make it **preserving**: instead of `_lean_retry(None, …)` (fresh slot), send
the previous candidate plus every current blocking diagnostic and an explicit
"do not drop a declared net" line, i.e. a full-defect correction. Targets §1.2's regression
directly.
**Predicted:** removes the "fixed X, lost Y" defect at the last rung. **Falsified if** the same
regression recurs with the previous candidate in the transcript.
**Risk:** medium (message construction only; no new detector, no new call).
**Cost:** ~1 day incl. tests.

### O5 — feedback completeness (no ladder change)
Send **all** current blocking diagnostics (not only the terminal/current one) in every
correction, with a per-defect fix instruction naming the two legal operations (declare the net
/ drop the binding; bind the data-connector recipe / accept the sink posture), and an explicit
"preserve every previously declared net unless the diagnostic says otherwise".
**Predicted:** fewer rungs per convergence. **Falsified if** the model still regresses content
it was told to preserve.
**Risk:** low. **Cost:** ~0.5 day (messages + a test asserting every blocking code appears).

### O6 — is `addr_d` actually required? (electrical, not ladder)
`ae3584c` made all 13 HUB75 channels `required=True` (`wave_c_interfaces.py:528-534`), which
*introduced* a mandatory `HUB75_D` net — one of `1/825`'s three defects. A 1/16-scan panel does
not use D.
Options: keep `required` (complete 1/32 support, 13 buffered channels); or mark `addr_d`
`required=False` and tie the spare '245 channel low in the recipe (no floating input, D
optional at the interface).
**Predicted:** making it optional removes one defect from the class. **Falsified if** boards
that omit D then fail the socket/buffer completeness checks.
**Risk:** medium — an electrical contract change; needs a datasheet-grade justification either
way, and a decided default for which panels we support.
**Cost:** ~0.5 day + a fixture.

### Ranking hypothesis (to be tested, not assumed)
O5 (cheapest, no routing change) → O4/O3 (target the two measured mechanisms) → O1 (blunt:
more corrections, no escalation) → O2 (control) → O6 (independent).

---

## 6. Live experiment design

### 6.1 Arms and sample

| arm | knob | code change | N (architecture replays) |
|---|---|---|---|
| O0 | none (default ladder) | none | 10 |
| O1 | `KICRAFT_SERIALIZATION_RETRIES=0` | none | 10 |
| O2 | `--max-retries 5` | none | 10 |
| O3 | env/flag for signature-aware clean-slate | §5.O3 | 10 |
| O4 | env/flag for preserving clean-slate | §5.O4 | 10 |
| O5 | env/flag for full-defect feedback | §5.O5 | 10 |

Implement O3/O4/O5 behind **one** settings value (e.g. `KICRAFT_CONTRACT_LADDER` ∈
`stock|signature|preserving|full_feedback`) so all arms run identical source. N=10 is the
starting point; extend the top two arms to 20 before adopting (see 6.5).

### 6.2 Frozen inputs (both are the *pre-architecture* state: intent + functional_spec committed)

```bash
S824=$HOME/.kicraft/projects/1/824/.kicraft/state.json   # KC-M2DW6N
S825=$HOME/.kicraft/projects/1/825/.kicraft/state.json   # KC-WGJ6XE
```

Run each arm against **both** states (they are different draws of the same brief: `1/824`
declared `HUB75` + `USB Type-C`, `1/825` declared `USB-C`), plus the web end-to-end check in 6.4.

### 6.3 Command (per run, per arm)

```bash
REPO=$HOME/KiCraft; PY=$REPO/.venv/bin/python
OUT=$(mktemp -d)/arm-O1-824-r3.jsonl
KICRAFT_SERIALIZATION_RETRIES=0 \
  "$PY" -m kicraft.server.stage_driver replay \
    --state "$S824" --stage architecture --max-retries 3 --budget 0.25 \
    --trace-jsonl "$OUT"
```

`replay` copies the state into a fresh temp workspace, so no run dir is mutated, and
`--trace-jsonl` (`stage_driver.py:135`) writes the sanitized per-attempt records the metrics
come from. **Never** run arms concurrently against the same `--budget`/guard; one run at a time,
retaining every trace.

### 6.4 Metrics (from the trace + `triage`, not from prose)

Primary endpoint, per arm:
- **architecture commit rate** over the arm's N (the only thing the decision uses).

Secondary / diagnostic:
- rungs taken and which rung terminated; `attempts` and the effective budget;
- **defect trajectory**: number of distinct blocking diagnostics at rung 1 vs the terminal rung
  (converging / flat / shuffling);
- regressions: a net or requirement declared in rung 1 and missing in a later rung;
- cost per run, wall time;
- for the top two arms only: **L2 = all five stages commit + `build` reaches layout** for the
  brief through `stage_driver run --brief "<the brief>"`, 5 runs each — the real product
  acceptance, not just the stage. **Report the terminal stage of every L2 run.** This brief
  class has a separate history of dying later (`bom`/`wiring` — `kc-2pfpvd`, `kc-vkut5h`,
  `kc-jmsmve`), so an architecture win must not be read as an end-to-end win, and an L2 failure
  must be attributed to the stage that actually failed rather than to the ladder.

### 6.5 Decision rule (pre-registered)

1. Eliminate any arm that regresses a deterministic gate or a test.
2. Among the rest, pick the highest architecture commit rate; ties → lower cost; then → smaller
   diff.
3. Adopt only if the winner beats O0 by ≥ 2/N with the interval reported; otherwise keep the
   status quo and record the negative result.
4. O2's result is reported regardless: it is the diagnosis's own control.
5. Re-run the winner and its nearest rival at N=20 before the code is adopted; adopt on the
   pooled result.
6. Report every failed/aborted run and its cost (no survivor-only reporting).

### 6.6 Cost

Observed for this brief: `architecture` ≈ **$0.018–0.025/run** (`1/825` $0.0182 for 3 calls,
`1/824` $0.0250). Six arms × 10 runs × 2 states ≈ 120 replays ≈ **$2.5**, plus the L2 escalation
(2 arms × 5 full chains ≈ $1–3, dominated by `bom`/`wiring`). Hard-cap each run at `--budget 0.25`
and stop the whole experiment if spend exceeds **$10**.

---

## 7. Reader / skill follow-ups still open

| # | gap | fix |
|---|---|---|
| D1 | `triage stages` prints the ladder but not the **defect trajectory**; the converging signature (§1.2) is found by hand | diff the diagnostics of consecutive rungs and print `rung 1: 2 defects -> rung 2: 1 defect (HUB75_D retained? no)` |
| D2 | `scan` ranks `stage_diag` for the **terminal** rung only | rank diagnostics across *all* rungs so "which contracts the model churns" is visible corpus-wide |
| D3 | the effective ceiling is derived, not recorded (this plan's §4.1) | record `attempt_budget` + rungs in `stage_done`/`StageStatus` |
| D4 | the two `samples.py` hero briefs are this brief class (`esp32-hub75-controller`, `esp32-robot-controller`); a failing hero brief is a product-visible defect | after the winning arm ships, add the brief to the **manual** `deploy/verify-design-canary.sh` subset (never the deploy gate — see `AGENTS.md`) and record its commit rate |

---

## 8. Acceptance and rollout

1. Land §4 instrumentation first (it is needed to measure anything), with tests.
2. Implement O3/O4/O5 behind `KICRAFT_CONTRACT_LADDER` (default `stock`), with a test per mode
   asserting the *routing* (which call is made, whether the terminal rule fires) — scripted
   client, no LLM.
3. Run the arms per §6; keep every trace and the manifest.
4. Ship the winner with the default flipped **only** after the N=20 re-run, plus:
   - full suite green,
   - the winning arm's `triage` ladder visible on a fresh live run of the brief,
   - `deploy/deploy-production.sh` (web 200 + `[build-worker] ready`). The 34-brief canary stays
     out of the deploy path (see `AGENTS.md`); run it manually if desired.
5. Re-scan the corpus after ~20 subsequent `architecture` runs: the terminal-by-policy
   signature (§1.4) should fall; report the before/after counts rather than a single-board
   anecdote.

## Non-goals

- No weakening of any deterministic gate, and no post-hoc netlist normaliser to paper over a
  dropped net.
- No second stall detector: changes to the clean-slate rule must reuse
  `_commit_rejection_signature`/`next_attempt` semantics (`kc-2pfpvd` rule).
- No change to the **commit** path's ladder (it already has the signature rule); this plan is
  the schema/contract path only.
- No new recipe coverage for this brief — `ae3584c` closed it, and `1/784`/`1/700` prove it.
- No prompt-example tuning of the architecture stage beyond O5's explicit feedback
  completeness.
- No change to the 34-brief canary or the deploy path.
