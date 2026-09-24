# Live Surprise-me round 2 (2026-09-24): the reviewed parts the pipeline could not build

## 0. Handoff (read this first)

**Where the loop stands.** Two rounds of live `Surprise me` briefs on kicraft.io (6 briefs) are
**0/6 fab-ready**. Ten root causes were found and fixed at source (§3), each with a guard test.
Production is *running* that tree (web + build worker restarted 2026-09-24 ~13:35). **The tree is
uncommitted** — 24 modified/new files under `kicraft/` and `tests/`, plus this doc.

**Test state.** Focused suites are green throughout (638 → 392 tests across the files touched last;
`tests/test_no_undefined_names.py` green). The last *full* suite run: `2 failed, 4458 passed` —
one was a real defect this work introduced (`Callable` unimported in `examples.py`, **fixed**), the
other is the pre-existing `ams1117-5v0-fixed` maturity failure this repo's baseline already names.
**A full suite has not been re-run since fixes #6–#10** — run it before committing.

**Next three moves, in leverage order.**

1. **Fix `is_board_level_quantitative_obligation` for a board stack-up count — do this first.**
   Live run `KC-CTBW6M` (seed 31, "…Use a four-layer stack-up") was refused at architecture, twice
   with an identical signature, on an obligation **no requirement can own**:
   `{"kind": "quantitative", "original_obligation_id": "pcb-layer-count", "quantity": "PCB copper
   layers", "relation": "equal", "value": 4.0, "unit": "layers"}`.
   `kicraft/design/models.py` classifies a quantitative row as board-level only when it has a
   board/PCB subject **and** an outline-dimension term (`width/height/…`) **and** a length unit
   (`mm/cm/…`). A layer count has none of the latter two, so it is treated as a part limit, and
   `validate_obligation_retention` (`kicraft/server/stage_contracts.py`, the `stage ==
   "architecture"` branch) raises `source_obligation_not_retained` — a code that hits **92 designs
   in the corpus scan**. The refusal's own message contradicts itself: it says board-wide facts may
   stay top-level, then refuses a board-wide fact.
   *Fix:* add a second admissible shape to that predicate — a board/PCB subject with a stack-up
   term (`layer(s)`, `stackup`, `ply/plies`, `copper`) and a build unit (`layers`, `plies`) is
   board-level, on the same three-pieces-of-evidence discipline the docstring sets out. Two of the
   generator's eight trailing requirements are stack-ups, so this gates a large share of surprises.
   *Verify:* `stage_driver replay --state ~/.kicraft/projects/44/917/.kicraft/state.json --stage
   architecture --budget 0.6` N-of-3 → expect `[ok] architecture`.
2. **Give the curated recipes a control port** (§7): `enable` on the switching regulators and a
   feedback/gain node on `mcp6001-follower@1`. The parts have the pins; both regulator recipes tie
   EN to the input rail (always-on), so this needs a port-level default tie (unbound ⇒ tie to the
   rail, today's behaviour preserved) — framework work that changes emitted circuits, so it needs
   its own review and N-of-3.
3. **JST coverage with a count-aware demand** (§7): the vendored `b2b-xh-a-lf-sn` bundle (2-pos XH,
   `C158012`, production) has no reviewed record, and the class cannot be aliased to it without a
   2-pin part answering a 4-pin demand. Fix #10's published contact counts are the prerequisite for
   choosing a count-matched carrier.

**Reproduction kit** (all under `/tmp/kc-live/`, outside the repo): `watch_batch.py <pid>` (per-run
verdict + triage + build log), `replay_n.sh <pid> <n>` (N-of-N stage replay printing each rung's
findings), `ladder_ab.sh` (the control/arm A/B). Live UI creds: `/tmp/ui-cred.json`. Board ids:
914 `KC-YMEWZV`, 915 `KC-5UG8UR`, 916 `KC-W8ZD6D`, 917 `KC-CTBW6M`.

---


**Status: ten fixes landed on the working tree and running in production; round 2 measured (0/3);
round 3's first brief root-caused but not fixed (§0.1).** Every change below carries a guard test;
focused suites are green, the full suite needs one re-run after fixes #6–#10.

Scope: the owner asked for live `Surprise me` briefs on kicraft.io, UI + log monitoring, and a
fix-and-retest loop until three of three are fab-ready. This is the record of round 1 (three
briefs, **0/3** fab-ready), the seven root causes it exposed, and the changes that follow.

It continues `docs/plans/live-surprise-me-yield-2026-09-24.md` (round 1 of that day) and
`docs/plans/yield-movers-2026-09-24.md` (the reviewed-coverage work). The first thing this round
had to do was **deploy that reviewed-coverage work**: the live web process was started at
00:56, three hours *before* the `yield-movers` edits landed at 03:44–04:00, so production was
serving the pre-fix code. Everything below is measured on a restart that loads it.

## 1. The live sample (all through the UI, production, test account)

| # | board | seed | brief | outcome | named causes |
|---|---|---|---|---|---|
| 1 | `KC-HPD3YF` | 25 | 12 V barrel jack → 5 V 3 A regulator, 4 screw terminals, power LED, enable jumper | architecture ✗ (3 rungs) | reviewed `dc005` jack unemittable; LED exact part refused in all 3 rungs; `reg.enable` unknown port; duplicate GND binding |
| 2 | `KC-5CNKJ3` | 26 | CH32V003 + A4988 stepper, 2S Li-ion pack, motor terminals, <40×30 mm | **architecture ✓** → bom ✗ (4 attempts, `unit_repair_exhausted`) | `battery-pack` demanded as a placed component |
| 3 | `KC-P4E2PH` | 27 | CH32V003 relay board, 3 relays on screw terminals, 24 V screw terminal, status LED | architecture ✗ (3 rungs) | LED `drive` unbound; `declared_signal_port_tied` |

**0/3 fab-ready.** Two of three never reached a build; the third reached the BOM unit and died
on an obligation no placed part can implement.

## 2. The root causes

### 2.1 The prompt and the realizer contradicted each other (the largest class)

`_reviewed_class_options_block` tells the architecture stage:

> "REVIEWED PARTS FOR THE DEMANDED CLASSES: a demanded physical class is realized only by one of
> these reviewed identities … so prefer them, and name `exact_part` only from this list"

The model complies — and then `led-current-resistor@1` refuses exactly that with *"does not
implement the exact part 'LTST-C190KGKT'"*, because its build emits a generic `Device:LED` with no
MPN. Run 1 re-emitted the same named part through all three rungs. The same contradiction sat
behind the jack: the prompt offered `dc005` for `barrel-jack-connector`, and no lowerer owned the
family at all, so the requirement was refused as *"no curated recipe and declares no interface"*
— while the reviewed record held its contacts and its symbol/footprint pair.

### 2.2 The reviewed DC005 had no pin map

Its siblings carry one (`132289` → `{"center": "1", "shell": "2"}`, every terminal record), but
`dc005` had `contacts=("1","2","3")` with `port_pins={}`, so nothing could derive which contact is
which — including the draft, which is asked to declare the interface for exactly this case.

### 2.3 A power source demanded as a board component

`li-ion-battery-pack` → `component_class="battery-pack"`. No placed part implements a pack, and a
demand whose class has no reviewed coverage falls back to real-part evidence (MPN + symbol +
footprint) — which the group that honestly implements the input (a generic header/terminal
lowerer) can never carry, because generic lowerer groups have no MPN. Unwinnable by construction:
run 2 spent all four attempts on it. The corpus's battery-input designs that pass carry **no**
such obligation; the connector requirement alone carries the power input.

### 2.4 A refusal naming ≥2 defects wrote an unloadable state

`_derive_intent_payload` wrapped the individual contract findings as bare dicts in `evidence`
with no `severity`. `StageDiagnostic` requires a severity and a `list[str]` evidence, so
`ConversationState.model_validate` refused **every** state saved after an architecture refusal
with two or more defects — 119 states in the corpus, including runs 1 and 3 of this sample. That
is precisely the state `stage_driver replay` exists to iterate on; the tool could not open the
runs this investigation needed.

### 2.5 Reviewed fixtures the build can place, but not prove

Two rows of the reviewed library (`dc005`, and the sample's `led` case) were reachable as
*choices* but not as *parts*. Both fixes below are the same shape: make the reviewed identity
placeable, and keep the reviewed data honest about its contacts.

### 2.6 A lowerer refusal that named nothing (live on run 1's terminal)

`requirement_ports=pin1` — a one-contact screw terminal — was refused with *"cannot realize this
identity/port/parameter combination"*, and the build returned `None` without a reason. A bound
that is real (KiCad ships `Screw_Terminal_01x01` but no 1-position MKDS footprint, so the generic
part cannot be placed) still has to be *stated*, or the draft has nothing to repair.

### 2.7 Surprise briefs could state a requirement twice

Seed 25's brief read *"…and an enable jumper. Add an enable jumper."* — the template's own
requirement plus the same one appended by `BRIEF_TRAILING`. **129 of 3000 seeds (4.3%)** repeated a
requirement this way; a second class came from two slots drawing one subject (seed 257: *"a 3 W
power LED, and a power LED"*).

## 3. The changes

| # | change | where | guard |
|---|---|---|---|
| 1 | A build that **can** place the reviewed record the draft named does: the LED build adopts the record's identity when its pin map (`anode` on 2, `cathode` on 1) matches the contacts it places | `kicraft/design/lowering.py` `_reviewed_group_for_identity`, `_led_resistor` | `test_a_reviewed_led_identity_is_built_not_refused` |
| 2 | `reviewed-connector@1` for `barrel-jack`/`barrel-jack-connector`/`dc-barrel-jack`: wires the draft's two nets onto the record's own contacts, switch contact as a no-connect, polarity from the draft and contact numbers from the record | `lowering.py` `_reviewed_connector` + registration | `test_a_reviewed_dc_jack_is_built_on_its_own_symbol_and_footprint`, `test_dc_jack_accepts_the_spellings_drafts_use_for_its_two_wires` |
| 3 | `dc005` gains `port_pins={"tip":"1","switch":"2","sleeve":"3"}`, cited from the datasheet's own SCHEMATIC sheet | `kicraft/design/part_identity.py` | `test_the_reviewed_dc_jack_maps_its_contact_functions_to_real_contacts`, `test_a_contact_numbered_pin_map_names_contacts_the_record_declares` |
| 4 | An off-board power source is not a part class: `off_board_source_class` + a named `intent_obligation_class_unrealizable` finding telling the model to demand the *mate* instead | `part_identity.py`, `kicraft/design/stage_semantics.py` | `test_a_power_source_obligation_is_named_not_demanded` |
| 5 | The aggregate refusal is a typed row (`severity` + members in `findings`); readers accept both shapes; the historical rows read with `severity=None` and coerced evidence | `kicraft/server/stage_contracts.py`, `kicraft/design/models.py`, `kicraft/design/architecture_intent.py`, `kicraft/server/stage_runtime.py`, `kicraft/cli/triage.py` | `test_a_multi_finding_refusal_stays_a_loadable_stage_diagnostic`, `test_rows_written_before_the_fix_still_load`, `test_an_aggregate_refusal_is_identified_by_its_member_defects` |
| 6 | `_connector` states why it refuses (contact count out of the 2..12 bound, un-numberable contacts, rows that do not divide, reviewed contact-count mismatch) | `lowering.py` | `test_a_connector_refusal_names_the_shape_it_needs` |
| 7 | `BRIEF_TRAILING` entries the body already states are not appended, and two slots may not draw one subject | `kicraft/server/examples.py` | `test_no_brief_states_the_same_requirement_twice` |
| 8 | Reviewed carriers for the two MCUs the brief generator names but the library did not carry (`attiny1604-ssn`, `stm32g030f6p6`) — KiCad's own symbol/footprint pair plus the ordering code the offline catalog resolves | `kicraft/design/part_identity.py` | `test_every_mcu_the_generator_names_has_a_reviewed_carrier` |
| 9 | `trim-potentiometer@1`: the reviewed trim pot is placeable, wired from the record's own wiper/end pin map, accepting the spelling the draft used (and a rheostat wiring leaves the unused end a no-connect) | `lowering.py`, `part_identity.py` (pin map) | `test_a_reviewed_trimmer_is_built_on_its_own_symbol_and_footprint`, `test_a_trimmer_with_no_ports_is_told_which_contact_it_needs` |
| 10 | The reviewed-class options block publishes each carrier's **contact count** and states the rule, so the stage can pick the variant that matches the contacts its requirement declares | `kicraft/server/stage_runtime.py` | `test_reviewed_class_options_publish_the_contact_count_that_decides_the_choice` |

### Fix #10, in full: the choice was blind

The A/B (§5) left one defect behind in **five of six** control replays: `unsupported_lowerer_contract`
on a screw terminal. The evidence is exact — *"the reviewed terminal 'WJ126V-5.0-04P-14-00A' carries
4 contacts and this requirement declares 2"* — and the terminal class is carried by **three** reviewed
blocks that differ in nothing but their contact count:

```
wj126v-5.0-02p-14-00a   contacts=('1','2')
wj126v-5.0-03p-14-00a   contacts=('1','2','3')
wj126v-5.0-04p-14-00a   contacts=('1','2','3','4')
```

The prompt block listed family and ratings but not the count, so the stage kept naming the 4-position
block for a 2- and a 3-contact requirement. Teaching the *refusal* to name the mismatch (fix #6) was
not enough: the refusal can only correct a choice already made, and by then a correction slot is
spent. The fix informs the choice where it is made, which is the block's own stated rationale for
listing ratings. The block now reads:

```
  screw-terminal: wj126v-5.0-02p-14-00a (screw-terminal: 2 contacts); wj126v-5.0-03p-14-00a (screw-terminal: 3 contacts); wj126v-5.0-04p-14-00a (screw-terminal: 4 contacts)
  barrel-jack-connector: dc005 (barrel-jack: 3 contacts)
```

### Fix #8, in full: the generator named parts the library could not build

`BRIEF_SLOTS["mcu"]` offers six MCUs (*ESP32-C3, ESP32-S3, RP2040, STM32G0, ATtiny1604,
CH32V003*). The class they answer (`microcontroller`) **has** reviewed coverage — and that is what
makes the gap fatal: a covered class is realized only by a reviewed part, so a brief that drew
`ATtiny1604` or `STM32G0` died at the BOM with *"requires 1 real microcontroller, found 0"* no
matter what the unit emitted. Live run `KC-YMEWZV` (seed 28) offered the **right ordering code**
(`ATTINY1604-SSN`) on all six attempts and three different symbol candidates, none of which
resolved to a reviewed record — no retry could ever satisfy the demand.

Both records follow the `attiny1614-ssnr` / `attiny412-ssn` precedent (bundle `kicad-standard`,
KiCad's own symbol/footprint, `contacts=1..N`), with the ordering code and ratings taken from the
offline catalog rather than asserted:

| record | symbol / footprint | catalog |
|---|---|---|
| `attiny1604-ssn` | `MCU_Microchip_ATtiny:ATtiny1604-SS` / `Package_SO:SOIC-14_3.9x8.7mm_P1.27mm` | `C614830`, SOIC-14, 1.8–5.5 V, −40…105 °C, 20 MHz |
| `stm32g030f6p6` | `MCU_ST_STM32G0:STM32G030F6Px` / `Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm` | `C724040`, TSSOP-20, 2.0–3.6 V, −40…85 °C, 64 MHz |

Verified with the product's own predicate: a group built from each record's symbol/footprint/ordering
code now answers `microcontroller` (`_group_has_physical_feature` → True), while an unresolvable
symbol/footprint pair is still refused. The guard holds the whole generator vocabulary to the
library, so the next slot added without a carrier fails a test instead of a customer's run.

### Measured on the working tree

* Both refusals reproduce byte-identically offline, then emit the reviewed part:
  `status-led` + `exact_part=LTST-C190KGKT` → group `LTST-C190KGKT`; `barrel-jack` +
  `exact_part=dc005` → `DC005` with tip 1 → `VIN_12V`, sleeve 3 → `GND`, switch 2 no-connect.
* The full chain holds: a draft declaring `jack` + a rail `from: jack.positive` compiles to
  `ports={'positive': 'VIN_12V', 'gnd': 'GND'}` and lowers to that artifact.
* Duplicated requirements: **129/3000 → 0/5000**, with 4818/5000 distinct briefs.
* States that load: 119 previously-unloadable corpus states now load (18 older BOM-unit rows from
  a writer that was already superseded still do not; see §7).
* Focused suites green: 552 → 638 passing across the touched areas; the live web log is clean
  after the restart, and run 3's previously-unloadable page now renders its findings.

## 4. What the frozen-input replay showed

`stage_driver replay --stage architecture` against run 1's own frozen state, **stock ladder, N-of-3: 0/3
before and after** — but the *defects changed*, and that is the point:

| | before (live, round 1) | after (frozen replay) |
|---|---|---|
| rung 1 | `unknown_part_refused` (jack) + `unknown_interface_port` + 2 × `unsupported_lowerer_contract` | `conflicting_port_binding` (converter) + `unsupported_lowerer_contract` (terminals) |
| rung 3 (clean-slate) | `conflicting_port_binding` + `unknown_interface_port` + `unsupported_lowerer_contract` (led) | same two, with the terminal refusal now **stating its cause** |

The jack and the LED defects are **gone**; what remains is a model-authored port binding plus the
terminal declaration. So the binding constraint is no longer a coverage hole — it is that the
ladder spends its two corrections and then ends the stage unconditionally, exactly as `triage`
says: *"The stage needed more CORRECTION rounds than the ladder gives, not more budget."*

## 5. The ladder arm: measured, and it is not the lever

`KICRAFT_CONTRACT_LADDER=signature,full_feedback` continues correcting after a rejected clean-slate
whose defect set changed, and re-sends every blocking defect. The previous session measured it once
(N=1, seed 20: 0/3 → 1/3) and left it disabled as below this repo's N-of-3 bar. This sample is a
better test — the runs' defect sets change from rung to rung, which is the arm's stated
precondition — so it was run as a **self-contained A/B in one script**, control and arm, N-of-3 on
each of the two frozen refusals:

| arm | committed |
|---|---|
| `control:stock` | **0/6** |
| `arm:signature+full_feedback` | **0/6** |

The arm does spend the extra correction it is designed to spend (4 attempts instead of 3 in most
runs) and it changes *which* defects survive, but it converts nothing. **Conclusion: the ladder is
not the lever, and the arm stays disabled.** That is a real result rather than a null one — it
retires the one config change this codebase had measured as promising, on six paired samples
instead of one, and it moves the next fix to where the defects actually live.

## 6. Live re-measurement

Round 2 drew seeds 28–30 from the same counter, on the restarted tree (fixes #1–#5, #7 live at
the time of each run; #6, #8, #9 landed after and are verified separately below).

| # | board | seed | brief | outcome |
|---|---|---|---|---|
| 4 | `KC-YMEWZV` | 28 | ATtiny1604 USB-serial bridge, 5 V header, CH340C, 6-pin header, status LED | architecture ✓ (retry) → **bom ✗** |
| 5 | `KC-5UG8UR` | 29 | LM358 hall-effect amplifier, 5 V header, two JST-XH connectors, trim pot | architecture ✗ (3 rungs) |
| 6 | `KC-W8ZD6D` | 30 | TLV9062 front end, 8 temperature channels, 18 V DC input, 6-pin header, enable jumper | architecture ✗ (3 rungs) |

**0/3 again — and the failures moved past coverage into declaration hygiene.** Run 4 died on the
MCU carrier gap (§3, fix #8). Runs 5 and 6 died at architecture on **clusters** of model-authored
declaration defects, at most two corrections apiece:

| run | defects in the draft |
|---|---|
| 5 (`KC-5UG8UR`) | *rung 1*: `declared_signal_port_tied`, `signal_names_rail`, `unknown_interface_port`, `malformed_signal_ref`, `conflicting_port_binding`, `unbound_required_port`, `unknown_part_refused` (**7**) — *rung 3*: `unknown_part_refused` (the reviewed trim pot, fix #9), `declared_signal_port_tied` |
| 6 (`KC-W8ZD6D`) | `unsupported_lowerer_contract` ×2 (a `switch-input` enable jumper with `['signal','vdd']` unbound; a header declaring no ports), `declared_port_double_bound`, `unknown_supply_rail` |

Both the jack/LED class of contradiction and the *unactionable refusal* class are gone from these
runs: every remaining message names the offending requirement, port and value (`triage` itself
concludes *"the stage needed more CORRECTION rounds than the ladder gives, not more budget"*).
A truncated clean-slate is the last rung, and on this path it ends the stage unconditionally.

### Verified after the round, on the frozen inputs

* **Fix #6** (stated connector refusals), run 5's own state, N-of-3: the terminal refusal now reads
  *"the reviewed terminal 'WJ126V-5.0-04P-14-00A' carries 4 contacts and this requirement declares
  3"* instead of "cannot realize this identity/port/parameter combination".
* **Fix #9** (reviewed trimmer), run 4's/5's shape: the live `gain_pot` requirement
  (`trim-potentiometer` + `3296W-1-103LF`) now lowers to the reviewed part on every corpus
  spelling, with a rheostat wiring leaving the unused end a no-connect.

## 7. Deliberately not done, with the reason

**The curated recipes expose no control port (top remaining gap).** Both live architecture
refusals that are not declaration slips ask for a port the recipe does not publish:

| live evidence | the port asked for | what the recipe publishes |
|---|---|---|
| `KC-HPD3YF` (×2 rungs), brief asks for "an enable jumper" | `reg.enable` | `tlv62569-3v3@1`, `ap63203-3v3@1`: `input`, `gnd`, `output` only |
| `KC-5UG8UR` (×3 rungs), brief asks for a gain trim | `amplifier.feedback` | `mcp6001-follower@1`: `vdd`, `gnd`, `input`, `output` only |

The parts have the pins — `ap63203` wires converter pin 2 (EN) to `input`, `tlv62569` wires pin 1
(EN) to `input`, i.e. always-on — so the recipe is choosing the always-on configuration for the
whole fleet and leaving a brief that asks for the *controllable* configuration with no port to
name. This is systemic on the product side too: "Add an enable jumper" is one of the generator's
eight trailing requirements, so roughly an eighth of surprise briefs ask for a port no regulator
recipe publishes. It is the same class as fix #8 — the product asking for something its own curated
data cannot express — and the same class as the `allow_ground` completion the compiler already has.

**Why it is not landed here.** Exposing `enable` as an ordinary optional port would leave it
dangling when unbound, which is an ERC error on every design that *doesn't* want a jumper — worse
than the refusal it fixes. The honest shape is a port-level default tie (`RecipePort.default_tie`,
unbound ⇒ tie to the input rail, the current always-on behaviour as the default), which is a
framework change to the recipe model and the resolver's binding step, and it changes emitted
circuits for every regulator design. That needs its own review pass and its own N-of-3, not a ride
on this round's diff.

* **One-position screw terminals** (`ports={'pin1'}`): a real demand (4 committed corpus
  architectures), but KiCad ships no 1-position MKDS footprint, so the generic build cannot place
  one. Widening the bound needs a reviewed 1-position part and a deliberate footprint choice;
  the refusal now states the bound instead of hiding it.
* **`jst-connector`** (37 corpus demands): the only vendored JST bundle with no reviewed record is
  a **2-pin** XH (`B2B-XH-A(LF)(SN)`, C158012). Aliasing the class to it — as
  `_DEMANDED_CLASS_ALIASES` does for other classes — would let a 2-pin part answer a 4-pin demand,
  which is why the earlier audit left it alone. It needs a reviewed record *plus* a pin-count-aware
  demand.
* **18 remaining unloadable corpus states**: BOM work-unit rows written by a writer that
  `f79b22a` already replaced (they carry `unit_id`/`defects` and no `code`). Making them load
  would mean inventing a code for a row shape no current writer produces.

## 8. Cost

Provider spend this round: $0.11 across the live sample (3 briefs, $0.009–$0.014 each), ~$0.05
across the frozen-input replays, against a $20/day ceiling. The scarce resource remains reviewed
data and correction rounds, not tokens.
