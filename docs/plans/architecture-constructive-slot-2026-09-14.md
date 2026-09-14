# Architecture first-draft root cause, and the constructive slot (2026-09-14)

**Status: diagnosis complete (measured); plan only, nothing implemented.** This supersedes the
option catalogue in `architecture-contract-correction-ladder.md` §4: those eight options are
implemented and measured flat, and that file's §8 records why. The measurement infrastructure
that file produced (`tools/ladder_experiment.py`, the interleaved driver discipline, the saved
draft corpus) is what this plan needs; no new harness is proposed.

**One-line summary:** the architecture stage has a **0/100 first-draft acceptance rate** because the
model is asked to hand-write wiring that the compiler can derive, and every rejected class has so
far been answered by adding another rule — which lowers the acceptance rate further. The fix is to
stop asking for derived data, derive it, and delete the validators that existed to check it.

---

## 1. The measured problem

Over 100 saved architecture runs (the four experiment rounds in `/tmp/ladder-exp`, both frozen
states of the plan's brief, live provider, stock behaviour):

| measurement | value |
|---|---|
| runs whose **first draft was accepted** (no correction at all) | **0 / 100** |
| distinct blocking defect classes in a rejected first draft: 1 / 2 / 3 | 65 / 22 / 13 runs |
| runs that still commit after corrections (`stock`) | 55–60 % |
| provider calls per committed run | 3.8 |
| first-draft defect classes, by frequency | `unknown_recipe_port_net` 37, `unrealizable_power_requirement` 22, `architecture_unowned_power_support` 21, `missing_recipe_port` 19, `missing_interface_port` 10, `native_usb_connector_required` 9 |

So the pipeline is not "an LLM stage that usually works and sometimes needs a nudge". It is a
**guaranteed rejection followed by a 3–4 call recovery loop**, whose death rate is the tail of that
recovery. Every "fix" of the last three weeks (native USB, HUB75 ports, recipe variants, typed
ownership, the eight ladder arms, my own bound-net completion) changed *which* class is rejected
first, not the fact of rejection: with the LED class completed deterministically, rung 1 simply
rejected on the next class in line (`bound_nets` round: the class vanished from 14/20 runs to 0/20,
and the commit rate stayed flat at 8/20 vs 11/20).

## 2. Root cause

**The slot asks the model to hand-write derived data, and the compiler then validates it.**

`Architecture` requires, in one generation: sheets; per-requirement `ports` maps whose *values*
must exactly equal net names declared elsewhere; every declared net with an endpoint list of
sheets; rails; plus recipe-port names and connector exposure the model can only know by
re-reading `extras.circuit_recipes`. Every one of those is a *consequence* of intent (which
blocks exist, which parts implement them, which signals they exchange), and every one is checked
after the fact — 18 blocking codes in `resolver.py`/`stage_contracts.py` plus ~20
`architecture_*` codes in `stage_semantics.py` (≈38 rules), on top of the schema's own shape
rules. A draft is accepted only if all of them hold at once, so

```
P(accepted) = Π P(rule i holds)  ≈ 0        (measured: 0/100)
```

and the number of defects per draft behaves like a set of independent bookkeeping slips
(1–3 per draft, measured). This is not a prompt-quality problem: the contract text already states
these rules (that is why each rejection is "actionable") — the model simply cannot satisfy a
multi-rule consistency obligation in one pass, and no amount of correction-budget tuning changes
that (the O1–O9 arms all landed flat).

**Why the last three weeks made it worse.** Each corpus defect was answered by adding a rule (and
often an exception that completes one instance of it). Rules are enforced on the *next* draw, so
each fix raised the number of ways to be rejected. Measured churn, all of it in three weeks:

| artifact | size / churn |
|---|---|
| `stage_contracts.py` (new 2026-08-26) | 1,530 lines |
| `resolver.py` (new 2026-09-09) | 2,102 lines |
| `stage_semantics.py` (new 2026-09-02) | 1,231 lines |
| those three together (17 commits, 2026-08-26 → 09-13) | **+5,411 / −548** (all three files are 2–3 weeks old) |
| Python added repo-wide in the same window | **+36,973 / −6,675** (98 files) |
| hard-coded recipe/lowerer definitions (`wave_*.py` + `lowering.py`) | ~3,900 lines |
| model-facing architecture contract | 319 lines |

that buys 0/100 first drafts. The ratio is the diagnosis: ~15 lines of compiler-side validation
per line of contract, none of which reduces the rejection rate, because the rejections are about
consistency the compiler could have established itself.

## 3. The change: make the architecture slot constructive

Split the slot by **who can know the fact**:

| field today | who should own it |
|---|---|
| sheets, sheet roles, `functional_blocks` | **model** (there is no other source) |
| requirement list: `family`, `exact_part`, supply rail, duty/quantity | **model** (part choice and topology are design decisions) |
| power source contract, external-load budget, programming path, rail-source topology | **model** (contract/interview material — keep the validators that check *intent*) |
| signal intent: which requirement exchanges which named signal with which peer or the board edge | **model** (this is the design) |
| canonical net names, `inter_sheet_nets`, endpoint lists and directions | **compiler** (pure bookkeeping; derived from signal intent) |
| `requirements[].ports` (port key → net name) | **compiler** (derived by matching signal intent to recipe port names/aliases) |
| rail nets, connector exposure, support/decoupling parts, HUB75 address ties, USB companions, net-range expansion | **compiler** (already has the code — today it is an exception path, make it the primary path) |

New slot shape (sketch — the point is the *size*, not the exact field names):

```json
{
  "sheets": [{"name": "MCU", "role": "mcu", "blocks": ["ESP32_S3_CONTROLLER"]}],
  "requirements": [
    {"id": "esp32", "sheet": "MCU", "family": "esp32-s3-wroom-1-module",
     "exact_part": "ESP32-S3-WROOM-1-N16R8", "supply": "+3V3", "programming": "native_usb"}
  ],
  "signals": [
    {"name": "USB_D_P", "from": "esp32.usb_dp", "to": "edge:USB_DATA"},
    {"name": "HUB75_R0", "from": "esp32.output_hub75_r0", "to": "hub75.r0"},
    {"name": "LED_DATA", "from": "esp32.output_led", "to": "led.data_in"}
  ],
  "power": {"input": {"port": "edge:USB_C_PD", "contract": "5V"}, "rails": {"+5V": "...", "+3V3": "..."}},
  "assumptions": ["..."]
}
```

`edge:<name>` is the board edge / off-board peer; for an output the compiler emits the connector
requirement (this is exactly the O9 completion, but as the normal path rather than a rescue).

**Derivation** is one pass, `derive_architecture(intent) -> Architecture`, which:
1. resolves family/exact part → recipe (existing registry);
2. maps each `signals[]` entry onto the two requirements' recipe port names (using the alias
   tables that already exist) and emits the `ports` bindings;
3. emits one canonical `inter_sheet_nets` entry per signal, with the endpoints the peer sheets
   imply (edge signals get the connector + endpoint);
4. adds rails, support parts, HUB75 ties, USB companions, expanded ranges;
5. then runs **only** the design-level validators (unknown family/part, incompatible supply or
   programming contract, rail-source completeness, user-ask completeness) — not the bookkeeping
   ones.

**Feasibility, evidenced:** the committed slot after derivation is byte-compatible with today's
valid slot, so **no downstream consumer changes** (`synthesis/validation.py` §9.x gates, wiring
work units, lowerers, form-factor reconcile all keep reading the same fields). And it already
works for the real case: with two completion rules applied to the actual first reply of the board
that died (KC-WGJ6XE), the reply **commits** (`tests/fixtures/ladder/kc-wgj6xe_architecture_rung1.json`);
derivation generalises exactly those rules instead of rescuing their instances one at a time.

## 4. Delete (this is the point of the exercise)

The derivation subsumes these; each is a validator or a rescue for data that will no longer be hand
written. Measured spans:

| file | to delete | lines |
|---|---|---|
| `stage_contracts.py` | `_normalize_usb_c_requirements` (135), `_normalize_architecture_sheet_aliases` (126), `_complete_bound_port_nets` (89), `_validate_typed_inter_sheet_contracts` (73), `_fold_recipe_covered_sheets` (65), `_complete_connector_requirements` (42), `_complete_hub75_optional_address` (22), `_inter_sheet_net_endpoint_signature` (9), plus the net-range dedup block — spans measured with `ast` | ~620 |
| `resolver.py` | three of `_port_bindings`' four diagnostics (`unknown_recipe_port_net`, `missing_recipe_port`, `missing_recipe_port_contract`) — keep `recipe_signal_in_power_nets`, which is electrical meaning — plus `_complete_typed_connector_peers`, `_complete_native_usb_companions`, `_complete_native_usb_data_connector`, and the `unresolved_requirement_ids` path for connector classes | ~400 |
| `stage_semantics.py` | the endpoint-ownership bookkeeping only (`architecture_missing_power_endpoint` and its siblings). Keep the intent-level codes: `architecture_mcu_supply_rail_missing`, `architecture_power_block_as_sheet`, `architecture_rail_source_unspecified`, `architecture_programming_decision_incomplete`, and the load-budget family are design questions, not wiring | estimate ~120 (count the emitters, not the module) |
| `wave_*.py` / `lowering.py` | duplicated port-name/alias tables once the derivation owns name mapping (one canonical table) | estimate ~300 (measured in stage 3) |
| **net target** | | **≥ −1,500 lines, and 0 new validators** |

**Rule for any future architecture change:** a new check may land only by deleting two checks or
200 lines of validation, and it must show first-draft acceptance up or defects-per-draft down.
That single rule is what stops the loop that produced +37k lines and 0/100.

## 5. Measurement (pre-registered, before any code)

Primary endpoint — **first-draft acceptance rate**: share of runs accepted with zero corrections,
N=20 per frozen state, **interleaved** with stock (round 1 of the previous plan proved blocked arm
ordering is invalid under session drift: stock measured 0.60 then 0.20 on identical inputs).

Secondary (all from the same runs): defects per rejected first draft; corrections per committed
run; cost per run; terminal failure kinds. Telemetry to add (cheap, additive): `drafts`,
`first_draft_accepted`, `defect_codes[]` on `stage_done`, and the same on the events stream, so the
number is visible in production, not only in an experiment.

Offline (free, before spending): replay every saved draft in `/tmp/ladder-exp` plus the reconstructed
historical rungs through the new derivation and report the classes that disappear — the same method
that showed O9's class going 14/20 → 0/20.

**Kill criterion:** if stage 1 does not raise first-draft acceptance above **50 %** (from 0 %), the
diagnosis is wrong. Stop; do not proceed to stages 2–3; report and re-derive. Do not "fix" the
guideline by relaxing it.

**Budget:** ≤ $3 of live spend for the whole plan (the arms plus one confirmation block). The
offline work is $0.

## 6. Stages (each independently shippable, each reversible)

0. **Telemetry + corpus measurement** — add the acceptance counters; run the offline replay to
   publish the current per-class distribution. No prompt, slot or validator change. *(hours, $0)*
1. **Derivation becomes the primary path** for the derivable subset (signals → nets → bindings →
   connector exposure), with the new prompt/contract text, while the reader accepts both the new
   intent-shaped slot and the legacy explicit slot. Measure with §5. *(the only stage with real
   risk; flag-gated, default off until measured)*
2. **Delete the bookkeeping layer** — the §4 list, once stage 1 is default. The old slot and its
   validators go in one release; no shims.
3. **Single name-mapping source** — collapse the recipe/lowerer alias tables into the derivation's
   one table, so "which recipe port does this signal mean" is answered once.

## 7. Non-goals / the pattern to stop

- No new validator, no new completion, no new prompt paragraph per corpus defect. That is the loop.
- No second ladder, no extra stage, no new agent, no prompt-example tuning.
- No change to the 34-brief canary or the deploy gate.
- Do not raise the correction budget (measured: doubling it changed nothing, 6/20 vs 6/20).
- Do not "make the messages better" (measured: more prose in corrections moved nothing; the
  full-feedback arm measured 0.15).

## 8. Risks and their checks

| risk | check |
|---|---|
| derivation hides a real design error (an invented net the model did not intend) | design-level validators stay; every derived net is reported in `assumptions`; the wiring/BOM gates still run |
| the model's intent is itself ambiguous (which peer a signal goes to) | `signals[]` requires an explicit peer or `edge:`, and a missing peer is a *question*, not a guess |
| two slot shapes during cutover | one release, flag-gated, legacy reader deleted in stage 2 (no aliases kept) |
| the new slot shape loses information the lowerers need | the committed `Architecture` is unchanged — the derivation emits the same fields, so a stage-1 A/B can diff the two committed slots offline |
| accepting intent-shaped slots breaks strict-schema enforcement | the schema is regenerated from the new model; the provider still gets `response_format: json_schema` |
