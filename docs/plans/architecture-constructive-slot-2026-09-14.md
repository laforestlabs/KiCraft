# Architecture first-draft root cause, and the constructive slot (2026-09-14)

**Status: stages 0 and 1 implemented and measured; stages 2 and 3 NOT landed, because the
pre-registered kill criterion (§6) did not pass.** The intent-shaped slot exists and is
flag-gated (`KICRAFT_ARCHITECTURE_SLOT=intent`, default `explicit`, so the provider contract is
unchanged unless an operator asks for the new slot); the telemetry of §6 and the offline replay are
live; the §5 deletions are untouched. Live spend for the whole exercise: **$<AB-TOTAL>** of the $3
ceiling. See **§10** for the implementation record, the three interleaved live blocks and the
class-by-class evidence. This supersedes the option catalogue in
`architecture-contract-correction-ladder.md` §4: those eight options are implemented and measured
flat, and that file's §8 records why. The measurement infrastructure that file produced
(`tools/ladder_experiment.py`, the interleaved driver discipline, the saved draft corpus) is what
this plan needs; no new harness is proposed.

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

`.venv/bin/python tools/ladder_experiment.py --arm stock --label ab_<arm> --runs 1` per run,
alternating arm and board, `KICRAFT_ARCHITECTURE_SLOT` selecting the arm, per-run cap $0.25.

| block | source | arm | commits | first-draft accepted | leading first-draft classes |
|---|---|---|---|---|---|
| 1 (N=10×2) | `468a178` | explicit (stock) | 6/20 | **0/20** | `unknown_recipe_port_net` 10, `missing_recipe_port` 6, `unrealizable_power_requirement` 4 |
| 1 (N=10×2) | `468a178` | intent | 7/20 | **0/20** | `invalid_parallel_output_count` 11, `multiple_recipe_contracts` 4, `missing_recipe_port` 4 |
| 2 (N=5×2) | `59622bc` | explicit (stock) | 4/10 | **0/10** | `missing_recipe_port` 8, `unrealizable_power_requirement` 1 |
| 2 (N=5×2) | `59622bc` | intent | **10/10** | **0/10** | `unknown_signal_requirement` 13, `unsupported_supply_port` 2, `usb_connector_supply_unknown` 2 |
| 3 (N=5×2) | `a795ff3` | explicit (stock) | 5/10 | **0/10** | `missing_recipe_port` 7, `unknown_recipe_port_net` 2 |
| 3 (N=5×2) | `a795ff3` | intent | **10/10** | **0/10** | `conflicting_port_binding` 6, `missing_recipe_port` 4 |
| 4 (N=5×2) | `6462cf2` | explicit (stock) | <AB4-STOCK> | <AB4-STOCK-FIRST> | <AB4-STOCK-CODES> |
| 4 (N=5×2) | `6462cf2` | intent | <AB4-INTENT> | <AB4-INTENT-FIRST> | <AB4-INTENT-CODES> |

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
wrong; stop, do not proceed to stages 2–3.* Measured: **0 % in every block, both arms** — the
criterion fails, so stages 2 and 3 are not landed and the §5 deletions stay unspent.

What the numbers do support:

- **The bookkeeping half of the diagnosis is confirmed.** `unknown_recipe_port_net` — the dominant
  class in the corpus (165 drafts) and the leading stock class in block 1 (10/20) — goes to zero
  offline and disappears from the intent arm's live first drafts. `multiple_recipe_contracts`,
  `missing_interface_port`, `native_usb_connector_required`, `unsupported_recipe_endpoint` and
  `unsatisfied_pin_capability` behave the same way. The model is no longer asked to hand-write
  wiring that the compiler can derive, and it no longer fails on it.
- **The remaining rejections are design-level statements, not bookkeeping.** Every live first draft
  still broke 1–2 of: a required recipe port the design never connected (`led.data_out` — where the
  LED string's continuation output goes is a design decision, and the derivation must not invent a
  net name for it), a rail named on a connector that the design never declared, a converter whose
  rated topology text is missing, an external-load current budget. Those are the rules §5
  deliberately keeps.
- **The per-draft defect count did not move.** Block 1's stock drafts carried 1–2 classes per draft
  and the intent drafts carried 1–2. That is §1's measurement again, now inside the constructive
  slot: the number of ways to be rejected fell, the probability of a clean first draft did not.
- **What did move is the recovery tail**, which is what costs a user a board: block 2's intent arm
  committed **10/10** against stock's 4/10 in the same block, and the intent drafts are cheaper
  (~$0.008 vs ~$0.017 per run). N=10 on one block is not the pre-registered endpoint, so this is
  evidence, not a verdict.

Re-derivation, per the kill criterion: the diagnosis was right about *what* the compiler should own
and wrong that owning it raises first-draft acceptance. The residual rejections are themselves
mostly derivable consequences of what the model *did* say — a recipe's required output with no peer
is by the recipe's own contract an off-board output; a connector exposing a rail the design named
anywhere is the same connector; an interface is a consequence of its ports (that one shipped). The
plan's own rule ("no new validator, no new completion per defect") is what makes the *next* step a
derivation change rather than another rule: measure which residual class each draft hits, derive
that class if the model already said what it needs, and re-measure — with the §6 kill criterion
still the gate on stages 2–3.

### 10.6 Not done

| item | status |
|---|---|
| §5 deletions (`stage_contracts.py`, `resolver.py`, `stage_semantics.py`, alias tables) | untouched: gated by stage 1 becoming the default, which the measurement does not justify |
| stage 2 (legacy slot and its validators removed in one release) | not started |
| stage 3 (single name-mapping source) | not started; the derivation does own one canonical alias table for signal→port resolution, which the collapse would move to |
| 34-brief canary, deploy gate, correction budget, prompt examples | untouched |
