# Architecture first-draft root cause, and the constructive slot (2026-09-14)

**Status: stages 0, 1 and 2 implemented and measured; stage 3 (a single name-mapping source) is the
remaining increment, and §10.10 measured that it has nothing left to collapse.** Stage 2 — the §5
deletions and the legacy explicit shape — landed at commit `2d5a95d`: −3,076 lines net, ten checks
deleted against one relocation, and the intent shape is now the only one (the
`KICRAFT_ARCHITECTURE_SLOT` flag and the `completing`/`addr_d_optional`/`bound_nets` ladder arms are
gone with it). Live spend for the whole exercise: **$2.82 of the $3 ceiling** (the operator raised
the ceiling to $4.50 for stage 2's block). See **§10** for the implementation record, the interleaved
live blocks and the class-by-class evidence, and §10.10 for stage 2's gate. This supersedes the
option catalogue in `architecture-contract-correction-ladder.md` §4: those eight options are
implemented and measured flat, and that file's §8 records why. The measurement infrastructure that
file produced (`tools/ladder_experiment.py`, the interleaved driver discipline, the saved draft
corpus) is what this plan needs; no new harness is proposed.

**Update (§10.8–§10.9, same day, next-steps plan Tasks A–C):** `first_draft_accepted` conflated a
reader refusal with a semantic repair, so the kill verdict above was unreadable. With the endpoint
split, the typed regulator rating, the user-fact question and a fresh N=20-per-board-per-arm block,
the intent arm's first drafts are contract-clean **22/38 (58 %)** against stock's **0/40**, and it
commits **35/40** against stock's **14/40** — the next-steps plan's pre-registered rule therefore
**unlocks stage 2**. Live spend then $2.60 of the $3 ceiling.

**Update (§10.10, same day, stage 2):** the §5 deletions are **landed** (`2d5a95d`), the legacy
explicit shape is gone, and the same protocol — one arm now, because the arm it compared against no
longer exists — measures the surviving intent shape at **39/40 commits and 35/40 contract-clean
first drafts** against §1's 35/40 and 22/38, at $0.0055/run. The gate of
`architecture-stage-2-handoff-2026-09-14.md` §3.3 passes on all three counts.

**One-line summary:** the architecture stage has a **0/100 first-draft acceptance rate** because the
model is asked to hand-write wiring that the compiler can derive, and every rejected class has so
far been answered by adding another rule — which lowers the acceptance rate further. The fix is to
stop asking for derived data, derive it, and delete the validators that existed to check it. Parts
the code has never seen are covered by §4, not deferred.

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
1. resolves family/exact part → recipe (existing registry), or to a `declared` interface when
   the part is real but uncurated (§4.2);
2. maps each `signals[]` entry onto the two requirements' recipe port names (using the alias
   tables that already exist) and emits the `ports` bindings;
3. emits one canonical `inter_sheet_nets` entry per signal, with the endpoints the peer sheets
   imply (edge signals get the connector + endpoint);
4. adds rails, support parts, HUB75 ties, USB companions, expanded ranges;
5. then runs **only** the design-level validators (unknown family/part, incompatible supply or
   programming contract, rail-source completeness, user-ask completeness) — not the bookkeeping
   ones.

**Feasibility, evidenced:** the committed slot after derivation is byte-compatible with today's
valid slot, so **no downstream consumer changes** (`synthesis/validation.py`, whose build-gate codes are the `9.x` family; wiring
work units, lowerers, form-factor reconcile all keep reading the same fields). And it already
works for the real case: with two completion rules applied to the actual first reply of the board
that died (KC-WGJ6XE), the reply **commits** (`tests/fixtures/ladder/kc-wgj6xe_architecture_rung1.json`);
derivation generalises exactly those rules instead of rescuing their instances one at a time.

## 4. Parts the code has never seen

The derivation needs to know a part's connections before it can write wiring for it. For a curated
recipe that knowledge is in the repo. For anything else, three cases, handled differently:

**4.1 A curated recipe exists.** Unchanged: the code owns the part's connections and pins.

**4.2 A real part with no recipe (the new-IC case).** Split the knowledge in two, because the two
halves have different sources and different assurance:

| question | source | assurance |
|---|---|---|
| which pins does this package have | the fetched part data (the pipeline already fetches a real part by order code into the local library, and already exposes a machine-readable pin inventory to the wiring stage, which enforces coverage with exact pin numbers) | checkable against vendor data |
| what is each pin *for* (supply, ground, serial data, enable…) | the model states it, at architecture time, as an explicit named interface — the same shape a recipe uses | a **claim**, recorded as such, reviewed, not verified against our table |

With that interface in hand the derivation treats the part exactly like a curated one: name each
wire once, emit its declarations, add the board-edge connector when the function leaves the board,
allocate the host pin, add the support passives the function's rules call for, and register the
part in the BOM. The only difference is provenance, and it is carried explicitly: the requirement
is marked `resolution_source: declared` (vs `recipe`), the interface appears in the review as a
claim, and the part is flagged in the BOM. Nothing about it pretends to have been verified.

What still guards this path, unchanged: the BOM stage refuses anything that does not resolve to a
real symbol, footprint and orderable part number (it already tells the model never to invent a
library prefix or package name); the wiring stage enforces net coverage against the fetched pin
list; the design-level checks (supply, domains, programming path) still run; and the board-edge and
electrical gates still run on the built board.

**4.3 Real part, unsupported function.** Fetchable pin data but no support-circuitry rules for
what it does, or nothing fetchable at all: **refuse, once**, naming the part and what is missing.
That is a capability boundary, not a bookkeeping failure — the point is that it produces one clear
answer instead of one of forty rejection reasons reached three retries in. The refusal is counted
separately in telemetry (§6) so a board that legitimately needs a chip we do not support is not
misread as instability.

**The honest residual risk.** For an uncurated part, a wrong *functional* assignment (two serial
lines swapped, say) is self-consistent and wrong: no amount of derivation or consistency checking
catches it. It is caught by the electrical review if it has something to check for that part, or by
a human looking at the flagged claim. The alternative today is refusing to build the board at all,
which is safe and useless; the trade is a small, visible, reviewable claim instead of a hard
refusal, and it is stated rather than hidden.

**The promotion loop.** When a declared interface is used and the board builds, that part is a
candidate to become a curated recipe: capture the fetched pin list plus the reviewed function
assignment, and every later board using it gets full assurance. The library then grows from the
boards people actually build — the opposite of adding a validator per failure. This promotion step
is human-reviewed work, not automatic, and it is the only place new curated knowledge comes from.

## 5. Delete (this is the point of the exercise)

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

## 6. Measurement (pre-registered, before any code)

Primary endpoint — **first-draft acceptance rate**: share of runs accepted with zero corrections,
N=20 per frozen state, **interleaved** with stock (round 1 of the previous plan proved blocked arm
ordering is invalid under session drift: stock measured 0.60 then 0.20 on identical inputs).

Secondary (all from the same runs): defects per rejected first draft; corrections per committed
run; cost per run; terminal failure kinds. Telemetry to add (cheap, additive): `drafts`,
`first_draft_accepted`, `defect_codes[]` on `stage_done`, plus `declared_interfaces` and
`unknown_part_refused` (§4) so an unsupported part and an unstable stage are never conflated — and
the same on the events stream, so the number is visible in production, not only in an experiment.

Offline (free, before spending): replay every saved draft in `/tmp/ladder-exp` plus the reconstructed
historical rungs through the new derivation and report the classes that disappear — the same method
that showed O9's class going 14/20 → 0/20.

**Kill criterion:** if stage 1 does not raise first-draft acceptance above **50 %** (from 0 %), the
diagnosis is wrong. Stop; do not proceed to stages 2–3; report and re-derive. Do not "fix" the
guideline by relaxing it.

**Budget:** ≤ $3 of live spend for the whole plan (the arms plus one confirmation block). The
offline work is $0.

## 7. Stages (each independently shippable, each reversible)

0. **Telemetry + corpus measurement** — add the acceptance counters; run the offline replay to
   publish the current per-class distribution. No prompt, slot or validator change. *(hours, $0)*
1. **Derivation becomes the primary path** for the derivable subset (signals → nets → bindings →
   connector exposure), including the `declared` interface path for real-but-uncurated parts and
   its single refusal for unsupported ones (§4), with the new prompt/contract text, while the reader
   accepts both the new intent-shaped slot and the legacy explicit slot. Measure with §6. *(the only
   stage with real risk; flag-gated, default off until measured)*
2. **Delete the bookkeeping layer** — the §5 list, once stage 1 is default. The old slot and its
   validators go in one release; no shims.
3. **Single name-mapping source** — collapse the recipe/lowerer alias tables into the derivation's
   one table, so "which recipe port does this signal mean" is answered once.

## 8. Non-goals / the pattern to stop

- No new validator, no new completion, no new prompt paragraph per corpus defect. That is the loop.
- No second ladder, no extra stage, no new agent, no prompt-example tuning.
- No change to the 34-brief canary or the deploy gate.
- Do not raise the correction budget (measured: doubling it changed nothing, 6/20 vs 6/20).
- Do not "make the messages better" (measured: more prose in corrections moved nothing; the
  full-feedback arm measured 0.15).

## 9. Risks and their checks

| risk | check |
|---|---|
| derivation hides a real design error (an invented net the model did not intend) | design-level validators stay; every derived net is reported in `assumptions`; the wiring/BOM gates still run |
| the model's intent is itself ambiguous (which peer a signal goes to) | `signals[]` requires an explicit peer or `edge:`, and a missing peer is a *question*, not a guess |
| two slot shapes during cutover | one release, flag-gated, legacy reader deleted in stage 2 (no aliases kept) |
| the new slot shape loses information the lowerers need | the committed `Architecture` is unchanged — the derivation emits the same fields, so a stage-1 A/B can diff the two committed slots offline |
| accepting intent-shaped slots breaks strict-schema enforcement | the schema is regenerated from the new model; the provider still gets `response_format: json_schema` |
| an uncurated part's functional pin assignment is a claim, and a swapped pin is self-consistent and wrong (§4.2) | the claim is marked (`resolution_source: declared`), shown in review and flagged in the BOM; the fetched pin list still bounds which pins exist; the electrical review and the build gates still run — and the trade versus today's hard refusal is stated in §4, not hidden |

---

## 10. Implementation record (2026-09-14)

Commits `468a178` (intent slot, telemetry, offline replay), `59622bc` (interfaces derived from the
bound ports), `a795ff3` (abbreviated references, connector rail exposure). Everything is behind one
setting, `KICRAFT_ARCHITECTURE_SLOT` (`explicit` by default — the provider contract, schema, spec
text and reader are unchanged for it; `intent` asks for the new slot).

### 10.1 What stage 0 shipped

- `stage_done` now carries the primary endpoint and its attribution, in production and in every
  experiment: `drafts`, `first_draft_accepted`, `defect_codes[]`, `declared_interfaces`,
  `unknown_part_refused` (`stage_runtime.stage_telemetry`, populated from the rejections the drive
  was shown and the committed slot).
- `tools/ladder_experiment.py --replay-corpus` replays every saved architecture draft — the raw
  `answer_delta` stream, so the draft, not a summary — through the real reader before and after the
  derivation, and publishes the per-class counts. No provider call, no new harness.

### 10.2 What stage 1 shipped

`kicraft/design/architecture_intent.py` defines the intent-shaped slot (`ArchitectureIntent`) and
`derive_architecture`, which writes the bookkeeping the model used to hand-write:

- canonical net names, one per declared signal, and `requirements[].ports` bindings on **both**
  ends (aliases like `HUB75_R0` for port `r0` resolve to the recipe's own key);
- `inter_sheet_nets` with the endpoint sheets and directions the parts imply (source direction from
  its own port, sink from its own port), multiplicities preserved, one record per net;
- `power_nets`/`rail_voltages` from `power.rails`, with each rail's producer, consumers and any
  connector that exposes it;
- the physical connector an `edge:` signal needs — a `usb-c-usb2-device` socket for a native-USB
  data pair, otherwise a pin header on the edge's own sheet (or on a sheet the model already
  declared for it), with the rails the peer named;
- the ground tie for a required recipe port the design leaves unused (`allow_ground`, e.g. HUB75's
  spare address channel) — the O8 rule as the normal path;
- the requirement's `interfaces` from the ports it bound (a bus member requests its interface, a
  declared interface with no member is not a request, `parallel_output` takes its count from the
  `parallel_<i>` ports);
- an unambiguous abbreviation of a requirement id in a signal reference;
- a connector's `supply` as a rail it *exposes* (a pin), not one it draws from.

Refusals are one answer per cause, all of them reported together:
`unknown_part_refused` (§4.2's single refusal, naming the part and the missing interface),
`unknown_interface_port` (naming the ports that exist), `unknown_supply_rail`, `unknown_edge_rail`,
`incomplete_usb_edge`, `usb_connector_supply_unknown`, `unknown_signal_requirement`,
`malformed_signal_ref`, `conflicting_port_binding`, `signal_conflicts_with_rail`,
`unknown_requirement_sheet`. The design-level validators (`unrealizable_power_requirement`, rail
source, programming path, block ownership, external-load budget) are untouched and still run.

The declared-interface path of §4.2 is end to end: the requirement keeps its model-stated port map,
`Architecture.declared_interfaces` marks it, `assumptions` says the pin functions are a claim, its
BOM work unit carries `declared_interfaces` and is told to check each function against the sourced
part, and the electrical review names it as a claim to check. Nothing pretends it was verified.

The reader takes either shape (`_intent_shaped`), so a legacy explicit answer during cutover still
commits; the flag only chooses the schema, the spec text and the worked example
(`.agents/skills/kicraft/stages/architecture_intent.md`, `_WORKED_EXAMPLES["architecture_intent"]`).

### 10.3 Offline replay (free, before any spend)

263 saved first drafts (266 event streams, both frozen boards, every experiment round), projected
onto the intent slot and re-derived: **0/254 accepted by the explicit reader → 165/256 after
projection + derivation** (the drafts the projection could not express are excluded and reported
in `replay.jsonl`, never counted as passes).

| blocking class | explicit | after derivation |
|---|---|---|
| `unknown_recipe_port_net` | 165 | **0** |
| `multiple_recipe_contracts` | 51 | 1 |
| `missing_interface_port` | 31 | **0** |
| `native_usb_connector_required` | 13 | 1 |
| `unclassified` (schema-shape) | 7 | **0** |
| `unsupported_recipe_endpoint` | 5 | **0** |
| `missing_mcu_application_contract` | 5 | **0** |
| `unsatisfied_pin_capability` | 1 | **0** |
| `missing_recipe_port` | 50 | 51 |
| `unrealizable_power_requirement` | 41 | 40 |
| `architecture_unowned_power_support` | 37 | 36 |
| residual projection artefacts | 0 | 8 |

Read this as a lower bound: the projection states a draft's design in the new shape, so a class it
cannot express is charged to the derivation. The classes that vanish are exactly the bookkeeping
ones; the classes that stay are the design-level rules §5 keeps.

### 10.4 Live measurement (§6 protocol: frozen source, interleaved, both boards)

One run per invocation, alternating arm and board so session drift is shared, with the slot chosen
by environment and a hard spend guard (`--runs 1` because the harness names its artifacts with a
timestamp, so an interleaved driver never overwrites a previous run's events):

```bash
for i in $(seq 1 10); do for board in 825 824; do for arm in stock intent; do
  KICRAFT_ARCHITECTURE_SLOT=$([ "$arm" = intent ] && echo intent || echo explicit) \
  .venv/bin/python tools/ladder_experiment.py --arm stock --label "ab_${arm}" \
    --state "$HOME/.kicraft/projects/1/$board/.kicraft/state.json" \
    --runs 1 --budget 0.25 --out /tmp/slot-ab
done; done; done
```

| block | source | arm | commits | first-draft accepted | leading first-draft classes |
|---|---|---|---|---|---|
| 1 (N=10×2) | `468a178` | explicit (stock) | 6/20 | **0/20** | `unknown_recipe_port_net` 10, `missing_recipe_port` 6, `unrealizable_power_requirement` 4 |
| 1 (N=10×2) | `468a178` | intent | 7/20 | **0/20** | `invalid_parallel_output_count` 11, `multiple_recipe_contracts` 4, `missing_recipe_port` 4 |
| 2 (N=5×2) | `59622bc` | explicit (stock) | 4/10 | **0/10** | `missing_recipe_port` 8, `unrealizable_power_requirement` 1 |
| 2 (N=5×2) | `59622bc` | intent | **10/10** | **0/10** | `unknown_signal_requirement` 13, `unsupported_supply_port` 2, `usb_connector_supply_unknown` 2 |
| 3 (N=5×2) | `a795ff3` | explicit (stock) | 5/10 | **0/10** | `missing_recipe_port` 7, `unknown_recipe_port_net` 2 |
| 3 (N=5×2) | `a795ff3` | intent | **10/10** | **0/10** | `conflicting_port_binding` 6, `missing_recipe_port` 4 |
| 4 (N=5×2) | `6462cf2` | explicit (stock) | 4/10 | **0/10** | `unknown_recipe_port_net` 7, `unrealizable_power_requirement` 2, `native_usb_connector_required` 2 |
| 4 (N=5×2) | `6462cf2` | intent | **10/10** | **0/10** | `missing_recipe_port` 8 (the driver's unbound continuation output) |
| 5 (N=5×2) | `4b24977` | explicit (stock) | 2/10 | **0/10** | `unknown_recipe_port_net` 5, `missing_recipe_port` 3, `unsupported_recipe_endpoint` 3 |
| 5 (N=5×2) | `4b24977` | intent | **10/10** | **0/10** | `unknown_edge_rail` 1, `conflicting_port_binding` 1 |

Pooled over blocks 2–5 (N=50 runs per arm): intent commits **47/50**, stock **21/50**; intent
`$0.0073`/run against stock `$0.0161`/run.

Attribution, from the same event streams — the classes that matter are not the same kind:

| block | arm | runs | commits | first drafts with **no contract refusal at all** | runs with a semantic repair round |
|---|---|---|---|---|---|
| 1 | stock | 20 | 6 | 0 | 16 |
| 1 | intent | 20 | 7 | 7 | 7 |
| 2 | stock | 10 | 4 | 0 | 10 |
| 2 | intent | 10 | 10 | 4 | 10 |
| 3 | stock | 10 | 5 | 0 | 8 |
| 3 | intent | 10 | 10 | 5 | 10 |
| 4 | stock | 10 | 4 | 0 | 8 |
| 4 | intent | 10 | 10 | 2 | 10 |
| 5 | stock | 10 | 2 | 0 | 8 |
| 5 | intent | 10 | 10 | **8** | 10 |

Pooled: **0/60** stock first drafts were contract-clean; **26/60** intent first drafts were. Every
stock run and every intent run spent at least one semantic repair round in blocks 2–5. (This table
was hand-derived from the retry events; §10.8 re-derives it from the instrument's own counters and
reproduces every row, adding the first-draft vs committed split.)

Each block's leading class was read off its drafts and answered by a *derivation* change, never by a
new rule (the §8 discipline):

| block | leading class | what the model had actually said | change |
|---|---|---|---|
| 1 | `invalid_parallel_output_count` | declared `interfaces: ["parallel_output","pwm"]` while binding `output_*` pins | interfaces are derived from the bound ports (`59622bc`) |
| 2 | `unknown_signal_requirement` | referenced `hub.r0` for its own `hub75` requirement | an unambiguous id abbreviation resolves (`a795ff3`) |
| 2 | `unsupported_supply_port` | set `supply` on a connector, i.e. "expose this rail" | a connector's supply becomes a pin (`a795ff3`) |
| 3 | `conflicting_port_binding` | wired ground explicitly (`MCU_GND` to peers' `gnd`) | a signal out of a ground-carrying port joins ground (`6462cf2`) |
| 3 | `missing_recipe_port` | never said where the driver's `data_out` goes | **no change**: the continuation output's net name is a design statement, and inventing one is exactly what §9 forbids |

### 10.5 The kill criterion fired, and what that says about the diagnosis

§6 pre-registered: *if stage 1 does not raise first-draft acceptance above 50 %, the diagnosis is
wrong; stop, do not proceed to stages 2–3.* Measured: **`first_draft_accepted` is 0 % in every
block, both arms** — the criterion fails, stages 2 and 3 are not landed, and the §5 deletions stay
unspent. That is the literal verdict and it stands.

The attribution makes the verdict more useful than "no":

- **The contract half of the diagnosis is confirmed, and it is the half that moved.** Stock never
  once produced a first draft without a contract refusal (0/60); the intent slot's first drafts were
  contract-clean in 26/60, and 8/10 in the last block. `unknown_recipe_port_net` — the dominant
  class in the corpus (165 drafts) and stock's leading class in blocks 1 and 5 — is zero offline and
  absent from the intent arm's live drafts. Same for `multiple_recipe_contracts`,
  `missing_interface_port`, `native_usb_connector_required`, `unsupported_recipe_endpoint`,
  `unsatisfied_pin_capability`, and after the last fix `missing_recipe_port`. The model is no longer
  asked to hand-write wiring the compiler can derive, and it no longer fails on it.
- **Every remaining first-draft *correction* is a semantic repair round, not a contract refusal.**
  In blocks 2–5 every intent run spent one repair round, on the intent-level codes §5 deliberately
  keeps: `architecture_mcu_regulator_incomplete` (76 emissions), which reads the slot's `topologies`
  for a converter term plus the `3.3V` token plus an ampere figure ≥ 1 A, and
  `architecture_external_load_current_unspecified` (68), which wants the load current the board
  supplies to the HUB75 display and the LED string — a number the brief does not give and only the
  user or the model can supply (`architecture_power_block_as_sheet` 22 is the same family: where a
  distribution block lands). Neither is derivable: the recipe registry states parts and stability
  assertions and no current rating, and the load current is a question, not a computation. The
  endpoint as instrumented therefore measures *that policy* as much as the slot:
  `first_draft_accepted` should be split into "first draft, no contract refusal" (measurable, and it
  moved from 0/60 to 26/60) and "first draft, no semantic repair" (dominated by two design
  statements).
- **The recovery tail — the thing that loses a user's board — improved.** Over blocks 2–5 (N=50 per
  arm, interleaved, same boards): intent commits **47/50** against stock's **21/50**, at
  `$0.0073`/run against `$0.0161`/run. The plan's §1 framing was that the pipeline is a guaranteed
  rejection plus a recovery loop whose death rate is the tail; the constructive slot removes most of
  the loop, and the intent arm's deaths (3/50) are the two derivation refusals and one schema
  shape failure, not bookkeeping churn.
- **The per-draft defect count did not move until the derivation owned more of it.** Block 1's
  intent drafts carried 1–2 classes like stock's; each block's leading class was read off its drafts
  and answered by a derivation change (table above), and the count fell with it.

Re-derivation, per the kill criterion. The diagnosis was right that the compiler should own derived
data and wrong that owning it lifts first-draft acceptance on its own: the residual causes are
statements no derivation may invent (a converter's current rating; a signal whose net name only the
design knows) and one prose check that reads them. Three concrete next steps, in the order the
evidence supports them:

1. **Split the endpoint** in `stage_telemetry` (`first_draft_accepted` /
   `first_draft_contract_clean` / `semantic_repair_rounds`) so the §6 primary endpoint measures the
   slot instead of the semantic policy. The data is already on the events; this is a counter change.
2. **Decide `architecture_mcu_regulator_incomplete` on its merits**: either the registry gains a
   reviewed current rating per regulator recipe (then the derivation can state the topology line and
   the check keeps its meaning), or the check is replaced by the intent-level question it really is
   ("what is the converter rated for?") in the external-load budget family. Do not have the
   derivation assert a floor it cannot source.
3. **Then re-measure at N=20 per board per arm** (the §6 sample size) with the endpoint split. Only
   a first-draft contract-clean rate above 50 % should unlock stage 2, because stage 2 deletes the
   validators that currently catch the 34/60 contract-dirty drafts.

### 10.6 Not done

| item | status |
|---|---|
| §5 deletions (`stage_contracts.py`, `resolver.py`, `stage_semantics.py`, alias tables) | untouched: gated by stage 1 becoming the default, which the measurement does not justify |
| stage 2 (legacy slot and its validators removed in one release) | not started |
| stage 3 (single name-mapping source) | not started; the derivation does own one canonical alias table for signal→port resolution, which the collapse would move to |
| 34-brief canary, deploy gate, correction budget, prompt examples | untouched |

### 10.7 Enabling it on the production box (2026-09-14, after the record above)

The operator asked to test the slot live on kicraft.io. Two things came out of that, and both are
now in the repo:

1. **A live hole in the BOM stage, found on the first real board and fixed.** With the slot on, the
   reference brief committed intent → functional_spec → architecture (2 attempts, $0.009) and then
   died in BOM with **0 provider attempts**: every sheet was recipe- or lowerer-covered, so the BOM
   was assembled entirely deterministically, and §9.33 then failed it outright because the
   architecture names the recipe's identity (`HUB75-SN74HCT245`) while the recipe ships
   `SN74HCT245PWR-JSM`. There was no call left in which to write the substitution ledger the gate
   asks for. Stock survived this only by accident: a model-owned requirement (a speaker amplifier on
   the 2026-09-13 boards) kept one LLM unit alive to paper over it. The fix is in
   `stage_contracts._recipe_parts_with_identity` — each recipe expansion's parts now carry the
   curated identity the design named, as a `sourcing_note` on the part that embodies it, which is
   the honest ledger and not a silenced failure. Regression test:
   `tests/test_stage_driver_prompt_examples.py::test_recipe_covered_bom_records_the_curated_identity`.
2. **The same brief now runs the whole chain with the slot on**: all five stages committed in 86 s
   for $0.013, with BOM and wiring fully deterministic (0 provider calls each). The architecture's
   only first-draft findings were the two design statements §10.5 identifies
   (`architecture_external_load_current_unspecified`, `architecture_mcu_regulator_incomplete`).

Deployment for the live test: `./deploy/deploy-production.sh` (no canary) with
`KICRAFT_ARCHITECTURE_SLOT=intent` in `.env`, which is read once per process
(`Settings.from_env()` → `load_dotenv`, `os.environ.setdefault`), so a restart is required both to
enable and to revert. The switch is global to the web process, not per board: it applies to every
board built while it is set.

### 10.8 The endpoint split, the typed rating and the user question

`docs/plans/architecture-slot-next-steps-2026-09-14.md` is the follow-up this section implements:
its Task A (measurement) and Task B (the two design statements) are landed at commit `b667c6e`,
and its Task C is the re-measurement in §10.9. The §5 deletions are still untouched and stage 2 is
still not unlocked.

**Task A — the endpoint separates the two correction kinds.** `stage_done` now carries, next to the
pre-registered `first_draft_accepted`:

| field | meaning |
|---|---|
| `contract_rejections` | reader refusals, counted once per rejected attempt (a `retry` carrying a `diagnostic`), not per code |
| `semantic_repair_rounds` | repair calls the driver spent on a `repair_required`/`fab_gate` finding |
| `first_draft_contract_clean` | `ok and contract_rejections == 0 and attempts >= 1`: the first draft reached the commit gates without a reader refusal, whether or not a semantic round followed |

`first_draft_accepted`, `drafts`, `defect_codes[]`, `declared_interfaces` and `unknown_part_refused`
are unchanged, and the work-unit stages publish none of the three new counters (their driver does
not know the kinds, so they keep their shape instead of reporting a false zero). Re-deriving the
saved evidence from each run's own `artifacts.events` reproduces §10.4's hand-derived attribution
and moves the verdict from "every run spends a correction" to two distinct numbers:

| block | source | arm | runs | commits | contract rejections (attempts) | first drafts contract-clean | runs with a semantic round |
|---|---|---|---|---|---|---|---|
| 1 | `468a178` | stock | 20 | 6/20 | 26 | 0/20 | 16 |
| 1 | `468a178` | intent | 20 | 7/20 | 26 | 7/20 | 7 |
| 2 | `59622bc` | stock | 10 | 4/10 | 10 | 0/10 | 10 |
| 2 | `59622bc` | intent | 10 | 10/10 | 6 | 4/10 | 10 |
| 3 | `a795ff3` | stock | 10 | 5/10 | 12 | 0/10 | 8 |
| 3 | `a795ff3` | intent | 10 | 10/10 | 5 | 5/10 | 10 |
| 4 | `6462cf2` | stock | 10 | 4/10 | 13 | 0/10 | 8 |
| 4 | `6462cf2` | intent | 10 | 10/10 | 8 | 2/10 | 10 |
| 5 | `4b24977` | stock | 10 | 2/10 | 12 | 0/10 | 8 |
| 5 | `4b24977` | intent | 10 | 10/10 | 2 | 8/10 | 10 |

Pooled: **0/60 stock** first drafts were contract-clean, **26/60 intent** were (blocks 2–5:
**0/40** vs **19/40**); stock spent 73 reader refusals to intent's 47.

The new split also says what the *repair round* bought, which the old endpoints conflated. Counting
the same code in the first draft's diagnostics and in the committed draft's:

| class | first drafts | committed drafts |
|---|---|---|
| `architecture_external_load_current_unspecified` | 39 | **39** |
| `architecture_power_block_as_sheet` | 16 | **16** |
| `architecture_mcu_regulator_incomplete` | 47 | 42 |

Two of the three classes survived the repair call unchanged in every run: the stage paid for a
round and committed the same statement it was told to fix. (These counts also reproduce the plan's
hand-derived emission totals from the event stream itself: over blocks 2–5 the load class is
68 = 34+34, the power-block class 22 = 11+11, and the regulator class 75 against the plan's 76.)

**Task B1 — the 3.3 V rating is typed data.** `RecipeDefinition` gains
`rated_output_current_a: float | None`, filled from each regulating recipe's own cited datasheet:

| recipe | rating | source |
|---|---|---|
| `tlv62569-3v3@1` | 2 A | TI TLV62569 datasheet |
| `ap63203-3v3@1`, `ap63205-5v@1` | 2 A | Diodes AP63200/01/03/05 datasheet |
| `tps54331-adjustable@1` | 3 A | TI TPS54331 datasheet (the recipe's existing `buck_component_bounds` assertion records the inductor/thermal conditionality) |
| `ams1117-3v3@1` | 1 A | AMS DS1117 datasheet, with a new `ldo_thermal_derating` assertion |
| `me6211-3v3@1` | 0.5 A | ME6211 datasheet |
| `mcp1700-3v3@1` | 0.25 A | MCP1700 datasheet |
| `tp4056-1s-charger@1` | 1 A | TP4056 datasheet (charge current) |

`architecture_mcu_regulator_incomplete` no longer reads the model's `topologies` prose: it resolves
the requirement whose recipe port `output`/`vout` is bound to a 3.3 V rail, reads that recipe's
reviewed rating, and requires ≥1 A from a declared, non-MCU sheet. Evidence, offline and free
(`/tmp/ladder-exp`, the throwaway probe): on the 214 derived drafts of the corpus the *old* prose
check fired on **213** of them — it was near-constant, which is why the report read 0 % — while the
typed check fires on **2**, and both are honest (a producer bound to `+5V`/`+3V3_POWER` while the
declared 3.3 V rail has no producer). The spec text, the recipe summaries the model sees and the
schema are untouched: nothing new is asked of the model.

**Task B2 — the load current is the user's fact.** When `architecture_external_load_current_unspecified`
fires and neither the brief (`external_load_budget_stated`) nor the recorded answers state the
number, the stage parks through the same `needs_input` path a model-authored question uses: the
question is attached to the stage in `state.json`, a `question` progress event reaches the UI, and
the stage returns without a commit or a repair call. A brief or an answer that carries the number
keeps today's repair behaviour, as does a non-interactive drive (`allow_questions=False`, the
self-eval/canary instruction) and a run that already has the user's answers (so the same board
cannot be asked twice in a row). This is a deliberate behaviour change on the *default* path, not
flag-gated: a board whose functional spec powers a display and an LED string and whose brief never
says how much current they draw now asks instead of committing a guess. Revert is the commit.

**Task B3 — `architecture_power_block_as_sheet` is bookkeeping, not a statement.** The 16 intent-arm
first drafts that trigger it (and the 16 committed drafts that still carry it) declare a sheet named
`USB POWER` / `POWER` with a `power_input` requirement whose family resolves to
`usb-c-5v-sink@1` — a real connector and its CC pull-downs — and a function line that happens not to
contain one of the check's keywords (`sink`, `input`, `supply`, …). The model's own repair was to
re-title the sheet. The class is therefore a prose test over a model-owned sheet *name*: it does not
name a missing design statement, so per the plan it is left alone here and recorded for stage 2's
derivation work (a typed form would ask "does any requirement on this sheet own a physical power
part", which `architecture_unowned_power_support` already answers at requirement level).

Tests: `tests/test_architecture_intent.py::test_rejected_first_draft_publishes_its_defect_class`
(reader refusal → `contract_rejections == 1`, not contract-clean), `…::test_semantic_repair_round_is_counted_apart_from_contract_rejections`
(reader-clean draft repaired for a 0.25 A regulator → clean but not accepted, one semantic round),
`…::test_missing_external_load_current_parks_with_one_question` (park with one question, one
provider call, no commit, durable `open_questions`; and no park when the brief carries the number),
and the typed regulator cases in
`tests/test_stage_semantics.py::test_architecture_rejects_false_pd_dac_provenance_and_missing_load_budget`
(nonsense prose with a reviewed 2 A recipe passes; a 0.5 A LDO or no producer still fires).

### 10.9 Re-measurement (next-steps plan Task C)

Frozen source: commit `b667c6e` (`repo_head` in every manifest). The surviving manifests also read
`repo_dirty: true`, because the endpoint/CLI/summary/documentation edits below landed while the
blocks ran; the *measured* path was frozen — `git diff b667c6e -- kicraft/design
kicraft/server/stage_runtime.py kicraft/server/stage_contracts.py` is empty. Offline first, then
live.

**Offline replay** (`--replay-corpus /tmp/ladder-exp`, free): the same 263 first drafts of §10.3,
re-read with every landed derivation fix. **0/256 accepted by the explicit reader → 214/256** after
projection + derivation (up from 165/256 when §10.3 was measured, before `59622bc`…`4b24977`):

| blocking class | explicit | after derivation |
|---|---|---|
| `unknown_recipe_port_net` | 165 | **0** |
| `multiple_recipe_contracts` | 51 | **0** |
| `missing_recipe_port` | 50 | 1 |
| `unrealizable_power_requirement` | 41 | 40 |
| `architecture_unowned_power_support` | 37 | 36 |
| `missing_interface_port` | 31 | **0** |
| `native_usb_connector_required` | 13 | 1 |
| `unclassified` | 7 | 1 |
| `architecture_unowned_power_conversion` | 4 | 4 |
| `unsupported_recipe_endpoint` | 5 | **0** |
| `missing_mcu_application_contract` | 5 | **0** |
| `usb_connector_supply_unknown` | 0 | 4 |
| `conflicting_port_binding` / `incomplete_usb_edge` | 0 | 1 / 1 |
| `unsatisfied_pin_capability` | 1 | **0** |

**The pre-registered block, run exactly as §5 writes it** (`/tmp/slot-ab-next`, `KICRAFT_ARCHITECTURE_SLOT`
flipping per arm, `stock` contract ladder, one run per invocation, both boards interleaved). It was
stopped after 6 iterations (12 runs per arm) because the endpoint it is meant to move does not exist
in it: **17 of 24 runs parked**, and the two arms' commits were 2 (intent) and 0 (stock).

| arm | runs | commits | parks | runs reaching the commit gates | first drafts contract-clean | contract rejections | semantic repair rounds | cost/run |
|---|---|---|---|---|---|---|---|---|
| stock | 12 | 0/12 | 7 | 5 | 0/5 | 10 | 0 | $0.0109 |
| intent | 12 | 2/12 | 10 | 2 | 2/2 | 0 | 1 | $0.0029 |

Read this as the product consequence of §4 B2, not as a slot measurement: with questions enabled —
the production posture — a brief that never states the external load current parks *both* arms, and
the stock arm still loses its other runs to the contract path (10 reader refusals, 0 contract-clean
first drafts). The 10 intent parks all diagnosed `architecture_external_load_current_unspecified`
and nothing else before parking; the 7 stock parks had the same finding among a larger set
(`unknown_recipe_port_net` 9, `architecture_missing_power_endpoint` 6, `architecture_duplicate_voltage_rails_unrelated` 6).

**The same protocol, driven unattended** — the command of §5 plus one flag
(`tools/ladder_experiment.py --unattended`), which passes the caller's non-interactive defaults
instruction so a batch with no user attached measures the stage instead of the question.
`/tmp/slot-ab-next-unattended`, N=20 per frozen board per arm, 80 runs, **$0.8913**, every run at
`repo_head b667c6e`:

| arm | runs | commits | runs reaching the commit gates | first drafts contract-clean | `first_draft_accepted` | contract rejections | semantic repair rounds | cost/run |
|---|---|---|---|---|---|---|---|---|
| stock | 40 | 14 (35 %) | 40 | **0/40** | 0/40 | 53 | 27 | $0.0159 |
| intent | 40 | **35 (88 %)** | 38 | **22/38 (58 %)** | 3/38 (8 %) | 19 | 31 | $0.0063 |

**Pre-registered rule, applied as written:** the intent arm's `first_draft_contract_clean` is
**58 % ≥ 50 %** and its commit rate **88 % ≥ stock's 35 %** in the same interleaved block. **Stage 2
is unlocked.** Two things the verdict does *not* say, and they matter for stage 2's own plan:

- The strict zero-correction endpoint is still low (3/38, 8 %) — because the two statements §4
  decides are still diagnosed on nearly every intent draft (the load current 30, the power-sheet
  title 13), and in an unattended drive they are repaired, not asked. That split is exactly what
  Task A made visible; it is not a slot defect.
- The intent arm's *residual* refusals are now a different family: `conflicting_port_binding` 15
  (`rail '+5V': port 'vbus' of 'usb_input' is already bound to 'VBUS'` — the model names one
  producer port for two rails), `multiple_intent_contracts` 15 (the same, aggregated),
  `usb_connector_supply_unknown` 12 (a native-USB edge with no declared ~5 V rail), plus five
  singletons. Those are the class to read from the drafts next, per §2's method.

First-draft classes per arm in the block (the committed-draft counts are identical for every class
except where noted):

| arm | classes |
|---|---|
| stock | `unknown_recipe_port_net` 26, `architecture_missing_power_endpoint` 25, `architecture_duplicate_voltage_rails_unrelated` 23, `architecture_external_load_current_unspecified` 22, `multiple_recipe_contracts` 13, `missing_recipe_port` 12, `missing_interface_port` 9, `native_usb_connector_required` 5, `unrealizable_power_requirement` 5, `unsupported_recipe_endpoint` 4, `architecture_unowned_power_support` 4, `architecture_unavailable_core_default` 3, `missing_mcu_application_contract` 2, `architecture_unowned_power_conversion` 1 |
| intent | `architecture_external_load_current_unspecified` 30, `conflicting_port_binding` 15, `multiple_intent_contracts` 15, `architecture_power_block_as_sheet` 13, `usb_connector_supply_unknown` 12, `architecture_duplicate_voltage_rails_unrelated` 1, `architecture_missing_power_endpoint` 1, `incomplete_usb_edge` 1, `unknown_interface_port` 1, `unsupported_supply_port` 1 |

`architecture_mcu_regulator_incomplete` — the class that fired on 76 drafts of the saved corpus —
does not appear in either arm at all: the typed rating removed it (Task B1). The load-current class
is diagnosed on 30 first drafts of 38 and is still present in the committed draft all 30 times: the
repair round buys nothing for it, which is the measured case for asking the user instead (Task B2).

Live spend for this plan: $0.166 (the pre-registered block) + $0.891 (the unattended block) =
**$1.06**, against the next-steps plan's $1.50 allowance and the parent plan's $3 ceiling
($2.60 total).

**Next:** stage 2 (§5's deletions and the legacy explicit shape dropped) is unlocked. It is
implemented and measured in §10.10.

### 10.10 Stage 2: the bookkeeping layer is gone, and intent is the only shape

`docs/plans/architecture-stage-2-handoff-2026-09-14.md` is the plan this section reports:
its §2 is the work, its §3 is the pre-registered gate. One commit, `2d5a95d` ("Delete the
architecture bookkeeping layer, and make intent the only shape"), 17 files, **+638 / −3,714 =
−3,076 lines net**. No shims, no aliases: the explicit reader, schema, spec text and worked
example are gone with the validators that checked their data.

**What went, and what replaced it.** Ten checks were deleted and one was added:

| deleted check / completion | where it lived | what owns it now |
|---|---|---|
| `unknown_recipe_port_net` | `_port_bindings` | the derivation names every net it binds |
| `missing_recipe_port` (2 sites: `_port_bindings`, the native-USB guard) | `_port_bindings`, `resolve_architecture_recipes` | `unbound_required_port`, raised by `derive_architecture` (see below) |
| `missing_recipe_port_contract` | `_port_bindings` | derived endpoints |
| `recipe_signal_in_power_nets` | `_port_bindings` | **kept** — electrical meaning |
| `native_usb_connector_required`, `native_usb_bus_conflict` | `_complete_native_usb_companions` | `_open_edge` builds the socket per MCU |
| `architecture_missing_power_endpoint`, `architecture_wrong_signal_direction` | `_architecture` | endpoint bookkeeping the compiler writes |
| `architecture_duplicate_voltage_rails_unrelated`, `architecture_fragmented_physical_domain` | `_architecture` + `_power_requirement_diagnostics` helpers | nothing: they were prose/shape tests over compiler-owned nets and sheets |
| sheet aliasing, connector auto-bind, USB-C completion, bound-net completion, hub75 `addr_d`, sheet folding, typed inter-sheet contracts, the net-range representation + dedup, the named-part `exact_part` backfill | `stage_contracts` | `derive_architecture` |
| the `completing` / `addr_d_optional` / `bound_nets` ladder arms, `KICRAFT_ARCHITECTURE_SLOT`, `ARCHITECTURE_SLOTS`, the `slot`/`ladder` reader parameters, `ResolutionResult.completed_nets`, `complete_native_usb` | config / runtime / resolver | one shape, one set of rules |

**The one check added, stated plainly.** `derive_architecture` now refuses a required recipe port
that nothing wires (`unbound_required_port`). This is the *moved* `missing_recipe_port`, and the
move is what keeps the failure at the architecture stage: the deleted diagnostic ran after the
resolver's bindings, and the same defect otherwise surfaced at BOM expansion as a bare
`ValueError` from `registry._validated_parameters` ("recipe … port bindings mismatch"), i.e. a
stage the model cannot answer. Measured on the derived corpus: it fires on 3 of 254 drafts, two of
which the deleted diagnostic also refused and one of which label matching used to patch silently.
Net checks: **−9**.

Two moves the deletion made necessary, both verified as behaviour-preserving:

- **The USB series-resistor decision.** `_complete_native_usb_companions` was the only producer of
  `series_resistors: False` for a socket whose MCU recipe already expands the 22R pair. The
  derivation sets it now. Verified by expanding the same board on both trees: identical
  parameters (`{"series_resistors": false}`) and identical parts (J2/R7/R8/U3, no second pair).
- **The named-part `exact_part` backfill.** Not on §5's list, but provably harmful once the reader
  no longer rescues its output: it stamped compiler-built edge connectors with the brief's own
  named parts, so `hub75_interface_power_edge` acquired `exact_part: "HUB75"` and resolved to the
  HUB75 shift-register recipe. Deleting it is what restored the corpus draft the completions had
  been papering over (214 → 215 accepted, below).

**Part 2d did not exist as the plan estimated it.** §5 budgeted ~300 lines of duplicated
port-name/alias tables in `wave_*.py` / `lowering.py`. Measured: those tables are physical
pinout facts (`_NATIVE_USB_PINS`, `_HUB75_SIGNALS`, `_GPIO_TO_PIN`) and lowerer port vocabularies,
not signal aliases (deleting one breaks recipe construction for every consumer), and the single
signal-alias table, `_PORT_ALIASES`, already lives in exactly one place (`resolver.py`) and is
imported by the derivation. There was nothing to collapse; the dead *consumers* of that table went
with 2b. Reported rather than replaced with invented deletions.

**Offline replay (free, before any spend).** `tools/ladder_experiment.py --replay-corpus
/tmp/ladder-exp`, the same 263 first drafts of §10.9, run again at `2d5a95d` and compared
row-by-row against the baseline replay (produced from `git archive HEAD` in a scratch tree):

| | baseline (`b667c6e`) | stage 2 (`2d5a95d`) |
|---|---|---|
| accepted after projection + derivation | 214/256 | **215/254** |
| drafts the derivation refuses outright | 7 | 9 |
| residual derived classes | `unrealizable_power_requirement` 40, `architecture_unowned_power_support` 36, `missing_recipe_port` 1, `native_usb_connector_required` 1, `unclassified` 1, `usb_connector_supply_unknown` 4, `incomplete_usb_edge` 1, `conflicting_port_binding` 1 | same families, `missing_recipe_port` / `native_usb_connector_required` / `unclassified` **all 0**, `unrealizable_power_requirement` 39, `architecture_unowned_power_support` 35 |

Four rows changed verdict, all in the intended direction: `b_both-824-…` (a bare `ValueError`
refusal, now accepted), and three drafts whose refusal moved from a later layer
(`native_usb_connector_required`, `missing_recipe_port`) or from "the projection cannot express it"
into the derivation's `unbound_required_port` — the same defect, named once, earlier. No row
acquired a new defect, and the "as written" column (the legacy canonical drafts read by the
surviving reader) is reported by the harness as `as-written`, not as an explicit-reader result.

**The pre-registered block (§3.2, adapted: one arm, because 2e deleted the arm it compared
against).** `repo_head 2d5a95d`, `repo_dirty: false`, `KICRAFT_CONTRACT_LADDER=stock`, 20 runs per
frozen board interleaved, one run per invocation, `--unattended`, stop if cumulative spend passed
$1.50. 40 runs, **$0.2190** ($0.0055/run), `2d5a95d` throughout:

| arm | runs | commits | runs reaching the gates | first drafts contract-clean | `first_draft_accepted` | contract rejections | semantic rounds | parks | cost/run |
|---|---|---|---|---|---|---|---|---|---|
| intent (stage 2, §1 protocol) | 40 | **39 (97.5 %)** | 40 | **35/40 (87.5 %)** | 6/40 | 6 | 31 | 0 | $0.0055 |
| intent (§1 baseline, `b667c6e`) | 40 | 35 (88 %) | 38 | 22/38 (58 %) | 3/38 | 19 | 31 | 0 | $0.0063 |

**The gate, applied as written:**

- the intent arm still commits ≥ 33/40 → **39/40** ✓
- its first-draft contract-clean rate does not fall below 22/38 → **35/40 (87.5 %)** ✓
- the deleted line count is real, ≥ −1,500 with 0 new validators → **−3,076 net**, 10 checks
  deleted against 1 relocation (−9 net) ✓

**Stage 2 lands.** The reader refusals that remain are the family §4 named, and they are now
almost all one class: `conflicting_port_binding` 12 of 22 refusal instances
(`multiple_intent_contracts` 5, `usb_connector_supply_unknown` 5), and the single non-committing
run (824's twin, board 825, three attempts: normal → serialization → clean_slate) died on it. The
semantic rounds are unchanged at 31 and are the two prose statements §10.9 already identified —
`architecture_external_load_current_unspecified` 30 and `architecture_power_block_as_sheet` 21 —
repaired in an unattended drive and *asked* in an interactive one (B2), so they remain the
product-level work, not a slot defect.

**Unit suite.** Every file the change touches is green (the four rewritten files:
`test_design_recipes.py` 164, `test_stage_driver_prompt_examples.py` 58,
`test_stage_semantics.py` + `test_stage_driver_retry.py` 154, `test_architecture_intent.py` +
`test_part_identity.py` 50, plus `test_stage_driver_core_defaults.py`). Tests that pinned the
deleted layer are deleted; tests whose behaviour survives were rewritten to the intent contract;
`test_architecture_intent.py` keeps every assertion of the surviving derivation, plus a new one
for the `series_resistors` invariant and one for `unbound_required_port`. In the full run, 18
tests fail — the same 18 at the inherited commit (`test_spend_guard` 1, `test_web_*` 17), an
order/interference cascade in the web/billing simulations; all of those files pass in isolation on
both trees, and the suite stalls at 97 % on both. They are pre-existing and untouched by this
change; the cascade is the next thing to fix in the suite, not in the pipeline.

Live spend for the whole exercise: $2.60 (§10.9) + **$0.219** = **$2.82 of the $3 ceiling** (the
operator raised it to $4.50 for this block).

### 10.11 The 34-brief canary, and the two defects it found

`deploy/verify-design-canary.sh` (§3 of the handoff's "ask the operator" note) was run after the
deploy, `--design-only --no-judge --lean-events`, `--parallel 3`, the production profile
(`KICRAFT_DESIGN_PROFILE=luna`, `openai/gpt-5.6-luna`). Its criterion — *every* selected brief
commits all five stages — has never passed: the last full run (2026-09-11, `deepseek-v4-pro`,
before the intent slot) committed **7/34** with 3 errors for $2.34 in 18 min. Two runs today:

| run | source | briefs | committed all five | errored | cost | wall |
|---|---|---|---|---|---|---|
| 1 | `2d5a95d` (stage 2) | 32 reached (SIGKILL at 32/34) | 9 | 1 (`KeyError`) | $0.601 | ~32 min |
| 2 | `154fa79` (stage 2 + crash fix) | 34 | **13** | **0** | $0.599 | 16 min |

**It found a crash.** `r2r-dac` died with `KeyError: 'logic_power_power_input'`: a signal whose
source port already carries a declared rail and whose peer is `edge:LABEL` opens the edge connector
without binding a signal to it (that is the point — the rail owns the net), and the close-out loop
indexed `bindings[connector_id]` directly. Pre-existing since `468a178`; the shape had no coverage.
Fixed at `154fa79` (+ a regression test); that brief now commits all five stages, which is the
canary verifying its own finding.

**It found an over-refusal, and it is the leading class:** `unsupported_supply_port` fired on 15 of
34 briefs (37 requirement-cases). The rail lookup only knew `vdd`/`vm`/`vin`/`input`/`vdd_5v`, so
the compiler refused a declared supply it *could* place: a part whose pin is `vcc` (5 cases),
`vbus` (2), a qualified `vdd_logic` (1), and every lowerer family, whose published `port_keys` were
treated as an empty vocabulary (23 cases). `8a330a0` binds the pin the design names (exact name
first, then a qualified one, with the rail's own name breaking a tie) and takes a lowerer's
*concrete* published keys as directions; a connector still *exposes* a rail on its own documented
pin rather than drawing it, which is what kept a socket's own VBUS statement authoritative.
Verified: replay held at 215/254, 4007 tests green, three new cases in the derivation suite.

**Still open (diagnosed, not changed):** for a lowerer family, `_catalog` returns the open
vocabulary before reading the model's own `declared_ports`, so a requirement that needs its own
keys (a `status-led` whose `drive` pin takes the rail) cannot bind them, cannot satisfy
`_require_ports`, and is refused with `ports=(none)`. That is a design decision with a measurable
effect — it belongs to the next increment, measured the way stage 2 was, not slipped in here.

**The canary's other refusals** are the family §10.10 and the handoff §4 already name:
`multiple_intent_contracts` 12 briefs, `conflicting_port_binding` 8, `unknown_interface_port` 5,
`missing_recipe_requirement` 3 — the read-the-draft class.

Live spend for the whole exercise: $2.82 + $0.601 (the killed run) + $0.599 = **$4.02 of the
operator's $4.50 ceiling**.
