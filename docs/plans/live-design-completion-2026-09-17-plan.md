# Live design completion — plan (2026-09-17)

**Purpose.** The product goal is that a one-sentence brief on kicraft.io produces a design
end-to-end: the five LLM stages (`intent`, `functional_spec`, `architecture`, `bom`, `wiring`)
each author their slot and commit it. Today that almost never happens. This document records what
was measured, what was already tried and failed, and the concrete work that remains. It supersedes
the "next step" sections of `docs/plans/self-eval-2026-09-16-remediation-handoff.md` for this
problem; that handoff remains the record for the reference-corpus work.

**Read `docs/plans/self-eval-2026-09-15-remediation-plan.md` for the authoritative milestone
program** (milestones 1–5, ranked GAP list). This plan is the delivery path for the specific
"live runs do not complete" failure, which that plan covers only in its acceptance text.

---

## 1. The problem, as measured

### 1.1 Two different tests

- **Deterministic reference corpus** (`tests/fixtures/reference_inputs/`, 34 rows, one per brief):
  `--reference-replay` drives each row's *stored* stage payloads through the real compiler,
  BOM/wiring validators and stage-commit path with a mock provider. **31/34 reproduce their
  boundary**; the other three are recorded supply blocks and one intended refusal. This proves the
  deterministic machinery accepts hand-written payloads. It says nothing about whether the model
  can author them — it never calls the model.
- **Live design canary** (`./deploy/verify-design-canary.sh`, design-only: five LLM stages, real
  provider, no build): the same 34 briefs, model-authored. Four runs on 2026-09-16/17:

  | run | dir | committed all five | architecture fails | BOM fails | wiring fails | cost |
  |---|---|---|---|---|---|---|
  | 1 | `logs/self_eval/canary_20260916T234557Z` | 0/34 | 23 | 11 | 0 | $0.61 |
  | 2 | `logs/self_eval/canary_20260917T001818Z` | 0/34 | 23 | 11 | 0 | $0.57 |
  | 3 | `logs/self_eval/canary_20260917T004307Z` | 1/34 | 19 | 13 | 1 | $0.62 |
  | 5 | `logs/self_eval/canary_20260917T011637Z` | 0/34 | 22 | 12 | 0 | $0.60 |

  (Run 4, `canary_20260917T011459Z`, was cancelled mid-run as superseded.) All runs report
  `source_unchanged: true` and `n_errored: 0`, so each is a valid measurement. Every brief clears
  `intent` and `functional_spec`; failures are concentrated in `architecture` and `bom`.

**Run 3's single completion did not reproduce and must not be treated as progress.**

### 1.2 The model's failures, by diagnostic code

Aggregated across runs 3 and 5 (68 brief-runs, 67 failures: architecture 41, BOM 25, wiring 1).
Codes come from the terminal `stage_done` event's `diagnostic.evidence[].code`, or the `error`
text for BOM unit defects:

| code | n | stage | what it means |
|---|---|---|---|
| `physical-obligation-unfulfilled` | 17 | bom | a required physical class resolved to no reviewed part |
| `unsupported_lowerer_contract` | 13 | arch | the requirement cannot satisfy the chosen family's published contract |
| `ARCH_SCHEMA/VALIDATION` (no codes) | 12 | arch | model output failed the `ArchitectureIntent` schema (obligation ownership etc.) |
| `OTHER:contract_rejected` (no codes) | 10 | arch | contract refusal whose diagnostic carried no evidence rows |
| `conflicting_port_binding` | 10 | arch | one port bound to two nets |
| `declared-interface-unrealized` | 9 | bom | a declared interface has no owning hardware, or its pin claim is unusable |
| `unsupported_supply_port` | 8 | arch | the requirement declares a supply its interface has no port for |
| `OTHER:commit_rejected` | 6 | bom | a commit gate (e.g. `E_PHYSICAL_REALIZATION`) refused the candidate |
| `declared_signal_port_tied` / `unknown_tie_net` / `unbound_required_port` | 3 each | arch | tie/dead-port problems |
| `unknown_reference_port` / `unknown_reference_domain` / `unknown_supply_rail` | 2 each | arch | a named rail/reference does not exist |
| `missing-requirement-implementation` | 2 | bom | a requirement has no implementing group |
| `unknown_part_refused` / `unknown_interface_port` / `unsupported_standard_stacking_interface` | 2 each | arch | uncurated part with no interface; unknown port; form-factor pinmap |
| `declared_port_double_bound`, `invalid_standard_stacking_pinmap`, `malformed_signal_ref` | 1 each | arch | field misuse / format |

`ARCH_SCHEMA/VALIDATION` and `OTHER:contract_rejected` carry no evidence rows; extract the text
from `events.jsonl` (`schema_error`) or `.kicraft/state.json` (`stage_status[stage].error`) when
diagnosing them.

### 1.3 Why the failures persist

The `architecture` stage asks the model to author one large document — sheets, requirements,
per-requirement ports, supply/reference bindings, ties, obligations, signals — that must satisfy
roughly thirty independent checks, with at most three attempts (retry budgets are per stage; see
`_stage_max_retries` in `kicraft/server/stage_runtime.py`). Each run fails a *different* subset.
With one sample per brief, run-to-run spread is the same order as any single fix's effect.

### 1.4 What was already tried, and measured ineffective

Do **not** repeat these; each is committed, tested and in the tree, and each failed to move the
aggregate (see the commit list in §8):

- Port menu: lowerers publish `required_port_keys`; `_Catalog.choices` prints the published pattern
  instead of `(none)`; `conflicting_port_binding` names the menu and the one-port-one-net rule.
- Verified class aliases: `usb-c-connector`→`usb-c-receptacle`,
  `fpc-ffc-connector`→`fpc-connector`/`ffc-connector`, `voltage-regulator-ic`→`voltage-regulator`.
- Identity guard: do not rewrite a group identity the reviewed records already classify.
  **Tried in `a603b1b`, reverted in `834752a`** — skipping the rewrite also skipped the curated
  bundle's enrichment, and two reference rows stopped reproducing
  (`usb-c-full-breakout` at BOM, `rs485-terminal` at wiring). The bundle-identity fallback below
  already covers the same goal more safely, so the guard was redundant as well as harmful.
- Bundle-identity classification: classify a group whose `(symbol, footprint)` is the curated
  bundle's, selecting the record by the bundle manifest's MPN.
- Named refusals for: declared-port tie misuse, obligation ownership, lowerer missing ports.
- Failure reason carried in the `stage_done` progress event.

The lesson: **refusal clarity is not conformance.** Adding messages cannot make the model emit a
document it cannot author.

---

## 2. Goal and non-goals

**Goal.** Raise live 34-brief completion from 0–1/34 to a majority by removing the *authoring*
burden from the model where the compiler can derive it, then re-measure with repeats.

**Non-goals.**

- Do not weaken, disable or bypass any check to make briefs pass. A checker that refuses a wrong
  document is working; the fix must make the model's *input* satisfy it.
- Do not add `try/except` around refusals, permissive matching, or special-casing of benchmark
  slugs or wording.
- Do not edit `~/.kicraft` (production data) except through documented tooling.
- Do not touch the deterministic reference corpus's 31/34 result; it must stay green.
- Do not chase the remaining refusals one at a time.

---

## 3. Phase 0 — measurement that can detect a change

`./deploy/verify-design-canary.sh` now passes `--repeats` through: set
`KICRAFT_CANARY_REPEATS=3`. Use a *bounded* set while iterating (slug arguments are positional and
go to both the runner and the acceptance check), e.g.

```bash
KICRAFT_CANARY_REPEATS=3 ./deploy/verify-design-canary.sh \
  rc-lowpass-bnc stm32-min lora-node usb-c-full-breakout can-node servo-driver-16
```

Cost scales linearly: ~$0.02/brief/stage-set, so 3 repeats × 8 briefs ≈ $0.5 and ~25 min on this
2-core box. Full 34-brief × 3 repeats ≈ $1.8 and ~90 min. Read `summary.json`
(`design_committed`, `stage_outcomes`, `failure_families`, `brief_median_mean`) — not a single
run's headline.

**Acceptance for Phase 0:** you can state, for a chosen fix, whether the same briefs improved
across repeats, not just once.

---

## 4. Phase 1 — derive bindings instead of authoring them (the main lever)

### 4.1 Idea

For a requirement whose family is recipe- or lowerer-backed, the model should name **signals** (and
a supply rail where it matters) and nothing else. The compiler already knows, from the selected
family's published contract, exactly which ports exist and which are supply, reference and signal
ports. It should therefore *derive* the per-port net bindings, instead of requiring the model to
author them and then refusing when the authored set is wrong.

Concretely, the model's authored `ports`, `supply_bindings`, `reference_bindings`, `ties`, and the
`supply_rail`/`reference_domain` fields on `declared_ports` become **optional inputs**: when present
they are still honoured (the reference corpus depends on this), and when absent the compiler
derives them from the signals and the published contract rather than refusing.

### 4.2 Target codes

By occurrence across runs 3+5, this phase should eliminate or drastically reduce:

`conflicting_port_binding` (10), `unsupported_supply_port` (8), `unknown_tie_net` (3),
`declared_signal_port_tied` (3), `unknown_reference_port` (2), `unknown_reference_domain` (2),
`unknown_supply_rail` (2), `declared_port_double_bound` (1) — 31 occurrences, ~46% of failures —
plus the port-driven share of `unsupported_lowerer_contract` (13) and `unbound_required_port` (3),
because a fully derived port set is what that contract check inspects.

Per-brief bucketing (§ of the handoff) gave 37%; by occurrence the target set is ~46%. Expect
something in the 35–50% band; measure it.

Not addressed by this phase: `unknown_interface_port` (a signal naming a port the family does not
publish — the model must choose from the menu), `malformed_signal_ref` (format), and everything in
BOM/obligations.

### 4.3 Implementation steps

All in `kicraft/design/architecture_intent.py` unless stated.

1. **Bind from signals first, then derive the rest.** `derive_architecture` already resolves every
   `signals[]` endpoint through `_lookup` / `_resolve_port` / `_bind` before considering the
   requirement's own bindings. Keep that ordering and add a derivation pass per requirement:
   - **supply**: if `requirement.supply` is set and the catalog has a supply port
     (`_supply_port(catalog, rail)`), bind it to the rail. Only refuse (`unknown_supply_rail`) when
     the named rail is not declared under `power.rails` — that is a genuine error.
   - **reference/ground**: bind the catalog's ground port (`gnd`/`vss`-like key; see
     `_PORT_ALIASES`) to `GND` when unbound. The existing "legacy ground" bind already does this
     for a catalog with a `gnd` direction — generalise it to the family's published reference port
     and to lowerer families.
   - **required ports**: for recipe catalogs, keep the existing "unused port tie" (bind
     `catalog.required & catalog.groundable` to `GND`). For lowerer catalogs, derive the port set
     the lowerer requires from its published contract (see `required_port_keys` in
     `kicraft/design/lowering.py`) and bind each unbound one to the rail/reference it names, else
     leave it and let `unbound_required_port` speak — do **not** invent a signal net.
2. **Stop refusing on absent authored bindings.** Remove the refusals that exist only because the
   model omitted or mis-stated a binding it no longer has to author:
   - `unknown_supply_port`, `unsupported_supply_port`, `unknown_supply_rail` → derive, or refuse
     only for an undeclared rail name.
   - `unknown_reference_port`, `unknown_reference_domain`, `unknown_tie_net` → derive from `GND`
     and declared rails; refuse only when a named net is neither a declared rail, `GND`, nor a
     signal.
   - `declared_signal_port_tied`, `declared_port_double_bound` → these guard against misuse of
     fields the model no longer needs; keep the refusals as a safety net (they cost nothing when
     the fields are absent) but do not let their absence be a failure.
3. **Make `conflicting_port_binding` structurally impossible for derived bindings.** The compiler
   assigns each port once; the check remains for authored inputs only. If a conflict is detected
   where *both* bindings are compiler-derived, that is a compiler bug and must fail loudly rather
   than be swallowed.
4. **Let the lowerer contract check see the derived ports.** `lowerer_contract_diagnostic`
   (`kicraft/design/lowering.py`) is called with `ports=bindings[requirement_id]`. Once derived,
   most `unsupported_lowerer_contract` cases should collapse to genuine parameter/count problems;
   the parameter half needs its own pass (see §4.4).
5. **Update the model-facing contract.** `.agents/skills/kicraft/stages/architecture.md` currently
   instructs the model to author `ports`, bindings and ties. Rewrite that section: signals and
   `supply` are the required inputs; per-port bindings are optional refinements the compiler
   derives. Keep the "one port carries one net" statement as *derivation* behaviour, not as a rule
   the model must satisfy. Do **not** restate the schema in the spec — it is generated from the
   pydantic models (`kicraft/design/models.py`, `kicraft/design/architecture_intent.py`); keep one
   contract.

### 4.4 Follow-ups inside Phase 1

- **Parameter/count half of `unsupported_lowerer_contract`.** After ports are derived, whatever
  remains is a parameter or count mismatch (`rc-lowpass@1` cutoff/resistance outside the accepted
  window; `pin-header@1`/`screw-terminal@1` counts). For each remaining case, decide whether the
  lowerer should accept more inputs or the refusal should name the accepted range. The generic
  message is correct but not actionable — consider publishing the accepted range in
  `lowerer_summaries()` and naming it in the refusal.
- **`unknown_interface_port`.** A signal naming a port the family does not publish. The refusal
  already prints the menu (`_Catalog.choices`). If this survives Phase 1, the next lever is to let
  the model name a *capability* (as MCU application pins already do) and let the compiler resolve
  it — see `_allows_application_port` and `_application_direction` for the existing pattern.

### 4.5 Acceptance for Phase 1

- The target codes in §4.2 fall to near zero across a 3-repeat canary of the bounded set.
- The deterministic corpus stays at 31/34 and its tests stay green.
- Overall completion on the bounded set improves measurably across repeats (state the number; do
  not claim a win from one run).

---

## 5. Phase 2 — obligations and schema validation (12 occurrences, ~18%)

Seven briefs per run fail with `1 validation error for ArchitectureIntent` and no diagnostic rows.
The known members:

- **Obligation ownership.** `ArchitectureIntent._obligations_are_owned_once` requires the top-level
  `obligations` set to *equal* the union of the per-requirement rows. The prompt now states this
  (`kicraft/server/stage_prompts.py`, architecture section) and the refusal names it
  (`c5d6138`), yet the class persists. Options: derive the top-level list from the requirements
  (making the model author obligations once, on the requirement), or accept either and normalise.
  Deriving is preferable and matches Phase 1's philosophy.
- **`questions` and other extra keys.** The provider may emit keys the `ArchitectureIntent` model
  forbids (`extra="forbid"`), producing a schema error whose text does not name the fix. When
  diagnosing, read `schema_error` from the run's `events.jsonl`; make the refusal name the offending
  key.

**Acceptance:** the class falls to zero on the bounded set; the schema errors that remain name the
key and the fix.

---

## 6. Phase 3 — BOM part resolution and library coverage (26 occurrences, ~38%)

Two distinct mechanisms, both measured:

1. **Selection/claim defects.** `physical-obligation-unfulfilled` (17) and
   `declared-interface-unrealized` (9). `physical-obligation-unfulfilled` persists *after* the
   bundle-identity fix (`00114c3`), so the remaining cases are not only identity mismatches —
   re-derive them per brief from the run's BOM unit error. `declared-interface-unrealized` includes
   claims with `pin: null` (`"claimed pin None is not in <symbol>; available pins=[...]"`): the
   model declares a pin function without a pin number, which can never be verified. Decide whether
   the pin is derived from the part's inventory, or the claim must omit `pin` entirely, and make the
   refusal say which.
2. **Genuine coverage gaps.** Classes for which no reviewed part exists — `high-side-load-switch`,
   `opto-isolator`, `status-led`, `power-led`, `thermocouple-input`, `r-2r-resistor-ladder`,
   `voltage-regulator-ic` (no `voltage-regulator` variant for that class), `lora-radio-module`. This
   is library authoring (milestone 2): bundle reviewed parts via
   `kicraft.design.cli_app add-part` / `fetch-3d`, record them in `REVIEWED_PARTS`
   (`kicraft/design/part_identity.py`), vendor them into `kicraft/parts_library/` so the identity
   travels with the repo (tier 2 of the loader outranks the `~/.kicraft` cache; see
   `kicraft/parts_library/__init__.py`). Operator input is needed on which order codes to
   standardise.

**Acceptance:** every obligation class a brief demands resolves to a reviewed part, or the class is
recorded as an explicit, documented gap that blocks that brief.

---

## 7. Phase 4 — re-measure and hand back

1. Full 34-brief canary with `KICRAFT_CANARY_REPEATS=3`.
2. Report per-brief completion across repeats, the code frequency table, and the buckets from §1.2.
3. Update `docs/plans/self-eval-2026-09-16-remediation-handoff.md` (§0 canary table) and this plan's
   status line.
4. Do not declare success from a single run; do not declare success from `/tmp` artifacts — the
   run directories under `logs/self_eval/` are the evidence.

---

## 8. Environment, guardrails and verification

**Production box.** `/home/kicraft/KiCraft` is the live server (see `AGENTS.md`). 2 vCPU. The web
app is NiceGUI on `127.0.0.1:8080` behind Caddy; services run detached (`setsid nohup`), restarted
only via `deploy/restart-web.sh` / `deploy/restart-build-worker.sh` or
`deploy/deploy-production.sh`. Do not run a routing campaign or a heavy build alongside the live
services. Canary runs are network-bound and safe to run; four ran today without disturbing the site.

**Spend.** `.env` sets `KICRAFT_PROJECT_LLM_BUDGET_USD=0.60`, `KICRAFT_DAILY_USD_CEILING=20`,
`KICRAFT_TOTAL_USD_CEILING=250`, profile `luna` (`openai/gpt-5.6-luna`). A full canary costs about
$0.60. Check remaining budget before a multi-repeat run.

**Verification commands.**

```bash
# deterministic corpus (no provider): must stay 31/34
.venv/bin/python -m kicraft.eval.design_acceptance --reference-replay

# the reference fixtures' contract shape
.venv/bin/python -m kicraft.eval.design_acceptance --references

# stage/architecture/recipe/work-unit tests
.venv/bin/python -m pytest tests/test_architecture_intent.py tests/test_stage_work_units.py \
  tests/test_design_lowering.py tests/test_stage_driver_retry.py tests/test_part_identity.py -q

# live, bounded, repeated
KICRAFT_CANARY_REPEATS=3 ./deploy/verify-design-canary.sh <slugs...>

# failure table for a run dir
.venv/bin/python - <<'PY'
import json, pathlib, collections, sys
root = pathlib.Path(sys.argv[-1] if sys.argv[-1].startswith("logs/") else sorted(pathlib.Path("logs/self_eval").glob("canary_*"), key=lambda p: p.stat().st_mtime)[-1])
codes = collections.Counter()
for run in sorted(root.glob("run_*")):
    evs = [json.loads(t) for t in (run/"events.jsonl").read_text().splitlines() if t.strip()]
    bad = [e for e in evs if e.get("kind") == "stage_done" and not e.get("ok")]
    if not bad: continue
    e = bad[-1]
    got = [v["code"] for v in ((e.get("diagnostic") or {}).get("evidence") or []) if isinstance(v, dict) and v.get("code")]
    text = " ".join(str(e.get(k) or "") for k in ("error", "schema_error"))
    if not got:
        got = [k for k in ("physical-obligation-unfulfilled","declared-interface-unrealized","missing-requirement-implementation") if k in text]
    if not got:
        got = ["ARCH_SCHEMA/VALIDATION" if "validation error" in text else f"OTHER:{e.get('failure_kind')}"]
    for c in dict.fromkeys(got): codes[c] += 1
print(dict(codes.most_common()))
PY
```

**Debug tooling.** `skill://kicraft-debug` drives one stage at a time with a mandatory review
before commit; `kicraft-stage-debug debug-draft --workspace W --stage S --brief-file B --budget
0.25` mirrors production question behaviour as of `a23b81c`. `kicraft-stage-debug run` drives all
five stages the way the site does. Use it to inspect one brief's stage without a full canary.

---

## 9. Delivered inventory (do not redo)

Branch `simplify/bom-wiring-pipeline`, all pushed; local == origin.

| commit | content |
|---|---|
| `0a2e700` | accepted-board correctness: acceptance contracts, reference evaluator, artefact evidence |
| `adb7597` | reference replay wired into acceptance; provider-envelope `oneOf`→`anyOf`; fixtures; 6 tests |
| `a1df360` | `nicegui>=3.17,<4` — the sporadic prune-time HTTP 500 and GHSA-p92q-2755-mhgh |
| `f04e35a` | reference corpus completion (`rs485-terminal`, `speaker-crossover`), vendored AMS1117 bundle |
| `a23b81c` | debug harness mirrors production question auto-default |
| `09b4f5b` | declared-port tie misuse refused by name |
| `c5d6138` | obligation-ownership invariant stated and refused actionably |
| `fde8f54`, `94d9ec6` | lowerer refusals name missing ports; nine lowerers publish `required_port_keys` |
| `56aaf9d` | failure reason carried in the `stage_done` progress event |
| `618c433` | port menu shown in every port refusal; one-port-one-net rule stated |
| `a603b1b`, `00114c3` | verified class aliases; bundle-identity classification (`a603b1b`'s identity guard was reverted in `834752a`) |
| `6789922` | canary `--repeats` passthrough |
| `e18e95c` | this plan |
| `834752a` | revert of the identity guard — it regressed two reference rows |
| `77a825b`, `e74faa1`, `037b035`, `1550f12` | canary runs 1–5 recorded, option leverage quantified |

---

## 10. Open operator decisions

1. **Deploy or hold.** The committed fixes are tested but not live: the running web/build-worker
   processes started 2026-09-16 20:58, before every design fix. Deploying makes live failures
   legible and carries the identity/alias fixes, but will not raise the completion rate on its own.
2. **Which phase first.** This plan recommends Phase 1. Phase 3 (library) is independent and can
   run in parallel; Phase 2 is small.
3. **Phase 3 order codes.** Which parts to standardise for the missing classes is a sourcing
   decision, not an implementation one.

---

## 11. Success criteria

- **Phase 0:** a fix's effect is separable from noise (3 repeats, bounded set).
- **Phase 1:** the §4.2 codes fall to near zero; corpus still 31/34; bounded-set completion
  improves across repeats.
- **Phase 3:** every demanded physical class resolves to a reviewed part or is a documented gap.
- **Overall:** live 34-brief × 3 repeats reaches a majority of briefs committing all five stages —
  the prerequisite for the milestone-5 release gate (34/34), which remains far off.
