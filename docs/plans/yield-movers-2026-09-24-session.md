# Yield movers — implementation record (2026-09-24)

**Status: implemented.** Every item in `docs/plans/yield-movers-2026-09-24.md` landed, in the
plan's own order, each with the named diagnostic it removes, a guard test, and a replay
where the frozen input existed. This file is the session record the plan asks for in §8; the
only cell still open is the full-suite regression gate at the end, which was still running
when the rest was written.

Where the plan's diagnosis turned out to be wrong, the correction is stated with the
evidence that contradicted it — three of the four items needed one.

## §1 Reviewed barrel-jack coverage

**Change.** The reviewed library now carries the jack: `dc005`
(`kicraft/design/part_identity.py`), bundle `dc005-barrel-jack` (the vendored easyeda2kicad
pair the loader already resolves for MPN `DC005`), LCSC **C431533**, contacts 1–3,
`physical_features={barrel-jack-connector, power-connector, wire-to-board-connector, barrel-jack}`,
`operating_limits={12 V, 0.5 A}` cited from the vendor datasheet
(`datasheet.lcsc.com/.../c1c86644cdb5448d228bda08350b2b1f.pdf`, note 7: "VOLTAGE/CURRENT
RATING: DC 12V 0.5A"), plus `_DEMANDED_CLASS_ALIASES` entries for `barrel-jack`,
`dc-barrel-jack` and `dc-power-jack`.

**Two plan claims were wrong, and the corrections are what the fix is built on.**

1. *"`C431533` is not in the offline catalog"* — it is: `jlcparts.lcsc_exists("C431533")`
   → `True`, row `DC005`, 64 903 in stock. So the existing `dc005-barrel-jack` bundle was
   never the fabricated-code case; it only lacked a *reviewed record*. Verified with
   `validate-part kicraft/parts_library/dc005-barrel-jack` → `OK dc005-barrel-jack@0.1.0`.
2. *"Re-vendor or delete **both** bundles"* — only one needed deleting. `dc005-barrel-jack`
   resolves and validates; `dc-barrel-jack-5-5-2-1` is the fabricated one
   (`lcsc_exists("C9900003477")` → `False`, MPN `DC005四脚贴片5.5*2.1`, footprint pads
   numbered 2/3/4 with pin 1 missing). It is **deleted**, and its two references updated
   (`scripts/restep_model_frames.py` transform map, `tests/test_edge_zone_derivation.py`
   now pins the surviving bundle's footprint).

**The class is now strictly reviewed.** `_group_has_physical_feature` for a covered class
requires a reviewed record, so the placeholder the plan cites *and* a stock
`Connector_BarrelJack:BarrelJack_Horizontal` pair are both refused, and only the DC005
satisfies `barrel-jack` / `barrel-jack-connector` (executed: 3 shapes, one accepted).

**Guard.** `tests/test_part_identity.py::test_reviewed_physical_part_metadata_requires_complete_pair`
(record fields, portfolio carrier, both spellings resolve) and
`tests/test_stage_work_units.py::test_obligation_class_aliases_match_the_reviewed_feature_vocabulary`
(DC005 answers both classes; a screw terminal does not answer `barrel-jack-connector`).
Plus a new invariant: `test_every_demanded_class_alias_names_an_emittable_reviewed_feature`
— every alias target must have a record that can actually be emitted, because an alias to a
feature with no symbol/footprint pair would turn a class's real-part fallback into a
permanent refusal, which is worse than the gap it closes.

**Live check (seed 22, the brief the plan cites: run 908 / KC-GWVAEU).**
`stage_driver run --brief "<seed 22>" --no-build` → `architecture [ok]`, cost $0.0096.
The architecture now declares `jack | barrel-jack | exact_part=dc005` — the model selected
the reviewed part on its own from the reviewed-candidate list. The subsequent BOM failure
**no longer mentions the jack**; it is:

```
physical-obligation-unfulfilled=['jst:jst-connector: requires 1 real jst-connector, found 0;
 the unit emitted: jst_connector=Connector_Generic:Conn_01x04 mpn=4-pin JST XH female connector']
```

That is a *model-authored* defect, not a coverage hole: the emitted group has a description
string where an order code belongs and no footprint, and the class's real-part fallback
correctly refuses it (`resolved_part_evidence("4-pin JST XH female connector",
"Connector_Generic:Conn_01x04", <footprint>)` → `True`; with an empty footprint → `False`;
executed). The same design reached BOM on other runs with a real part
(`J3 = B4B-XH-A`, symbol `Connector_Generic:Conn_01x04`) — so the honest statement is:
**the barrel-jack refusal is gone; this brief's BOM now stops on an incomplete JST group,
not on the DC input.** §1's acceptance (`[ok] bom`) needs either a reviewed 4-pin JST or a
model that emits a complete group; `jst-connector` is left unaliased on purpose (see the
audit below: aliasing it to the reviewed 2-position PH would let a 2-pin part answer a
4-pin demand).

### The one-or-zero-carrier audit the plan asked for (§1.3)

Executed over **2 826 saved `.kicraft/state.json`** files (`logs/self_eval/**`, `/tmp/**`):
264 demanded `component_class` values, **190 with zero reviewed coverage**.

Fixes: the 14 classes below were aliased, each verified to have at least one *emittable*
record and each a plain second spelling of an already-covered class that
`reviewed_class_variants` does not already repair (a class whose tokens are a superset of a
reviewed feature is told to use the reviewed name at the intent stage; an alias there would
suppress that repair and silently accept a looser demand).

`push-button`, `op-amp`, `gpio-expander`, `gpio-expander-ic`, `dc-dc-converter`,
`qwiic-receptacle`, `uln2003-driver`, `rs-485-transceiver`, `current-limiting-switch`,
`current-limiter`, `load-switch`, `selector-switch`, `audio-jack-3p5mm`,
`temperature-humidity-pressure-sensor`.

Deliberately **not** aliased, with the reason:

| class (demand count) | why not |
|---|---|
| `jst-connector` (37) | the only reviewed JST part is a 2-position PH; aliasing would let a 2-pin part satisfy a 4-pin demand. Needs a reviewed XH/SH record **and** a pin-count-aware demand. |
| `motor-connector`, `jumper`, `coupling-capacitor`, `input-protection`, `reverse-polarity-protection`, `current-sense-sensor`, `seven-segment-display`, `coin-cell-battery`, `resistor-ladder`, `servo-driver-ic`, `switch`, `hang-hole`, `castellated-gpio`, `snowman-shaped-board`, `qfn-56-package` | no reviewed record implements them; `class_is_not_a_part`/`reviewed_class_variants` already route the not-a-part and longer-spelling ones, and the real-part fallback is the honest route for the rest. |
| `qspi-flash-memory`, `microcontroller-module`, `stepper-motor-driver`, `stepper-driver-ic`, `usb-c-pd-controller`, `buck-converter-controller`, `power-output-connector`, `d-sub-9-connector`, `smt-voltage-regulator`, `swd-header`, `stacking-through-hole-header` | already repaired by `reviewed_class_variants` at the intent stage. |

Fragile single-carrier classes (one record, no second source) are listed by `reviewed_parts_for_feature`
sweep, largest corpus share first: `barrel-jack-connector`/`barrel-jack` → `dc005`,
`jst-ph`/`wire-to-board-connector` → `s2b-ph-sm4-tb(lf)(sn)`, `qwiic-connector` → `sm04b-srss-tb(lf)(sn)`,
`bnc-connector` → `kh-bnc50-3511`, `fpc-connector` → `kh-fg0.5-h2.0-24pin`, `usb-pd-controller` → `ch224k`,
`stepper-driver` → `a4988settr-t`, `darlington-array` → `uln2003adr`, `relay` → `srd-05vdc-sl-c`,
`can-transceiver` → `sn65hvd230`, `thermocouple-converter` → `max31855kasa+`,
`usb-uart-bridge` → `ch340c`, `rotary-encoder` → `ec11e15244g1`, `sma-connector` → `132289`,
`current-limited-power-switch` → `tps2553dbvr`, `environmental-sensor` → `bme280`,
`i2c-oled-display` → `hs96l03w2c03`, `quad-operational-amplifier` → `mcp6004-i/sl`,
`optocoupler` → `pc817c-s`, `digital-isolator` → `adum1301arwz-rl`, `chip-antenna` → `h2u38d1e1b0100`,
`air-core-inductor` → `dayton-lw18-50`, `tantalum-capacitor` → `tajb686k010rnj`,
`trim-potentiometer` → `3296w-1-103lf`, `timing-capacitor` → `c0805c103j5gactu`,
`dip-switch`/`three-channel-selector` → `dshp03tsget`, `three-position-selector` → `ss13d07vg4`,
`binding-post` → `keystone-8734`, `usb-a-receptacle` → `u-a-24ss-w-2`, `flash-memory` → `w25q16jvss`.

**A second, distinct wall was found while auditing** (out of the plan's four items, worth
its own look): seed 22's architecture emitted the JST requirement as family **`pin-header`**
carrying a `physical: jst-connector` obligation, while `_required_physical_feature` derives
the unit's typed demand from the *family* (`header`) — so the unit is told to build a header
and is then refused for not being a JST connector. That is an obligation/family mismatch,
not a library gap.

## §2 A header may leave interior pins unconnected

**Change.** `_numbered_connector_ports(..., fill_gaps=True)` for `pin-header@1`: the
physical size is the **highest declared contact**; undeclared positions are emitted as `NC`
and become `LoweringNoConnect`s, and the artifact records which ones
(`"…unused contact(s) 5, 6, 7 are emitted as no-connects"`). The published `port_contract`
text says so, which is also the model-facing hint (`extras.circuit_lowerers` →
`lowerer_summaries()`). Bound preserved: `_connector`'s existing 40-contact cap now bounds
*N* (a stray `pin99` is still refused), a genuinely unknown port still refuses by name, and
`rows`/`per_row` stays a published combination (a 2-row draft whose highest contact is odd
is refused).

**Verified shapes** (executed, `_requirement("pin-header", …)`):
`pin1,pin2,pin3,pin4,pin8` → `Conn_01x08` with 5–7 no-connects (run 4's exact ports);
`pin2..pin8` → `Conn_01x08` with pin 1 no-connect (the seed-18 `main_header` draft);
`pin1,pin2,pin4` → `Conn_01x04` with pin 3 no-connect (the seed-18 `i2c_header` draft);
`pin1,pin2` unchanged; `2 rows × 3 contacts` and `pin1+signal` still refused.

**Guard.** `tests/test_design_lowering.py::test_header_accepts_a_declared_contact_subset_as_no_connects`
(and the stale row that asserted a gapped header is refused was removed from
`test_header_rejects_ambiguous_contact_contracts`, which now only pins the genuinely
ambiguous spellings).

**Replay (this is also §4's replay).** `stage_driver replay --stage architecture`
N-of-3 on the frozen seed-18 input (the only frozen state carrying that contract defect; run
904/907's states were never persisted). The frozen state's `stage_status` predates the
diagnostic-encoding fix, so a minimal state carrying only its committed `intent` +
`functional_spec` is the replay input.

| run | verdict | cost | remaining defects in the draft |
|---|---|---|---|
| 1 | `[FAIL] architecture`, 3 attempts | $0.0071 | `conflicting_port_binding` ×5 (a signal `GND_INPUT` duplicating the ground on five ports), `unbound_required_port` for ADS1115 `io1/io2/io3` |
| 2 | `[FAIL] architecture`, 3 attempts | $0.0045 | the model declared its **own** `usb-c-usb2-device` requirement (via `edge:USB`) on a design with no 5 V rail → `usb_connector_supply_unknown`, plus `conflicting_port_binding` ×2 (`io2`/`io3` ties) and that connector's own `usb_dm`/`usb_dp`/`vbus` unbound |
| 3 | `[FAIL] architecture`, 3 attempts | $0.0039 | `unknown_part_refused` for the jack (`exact_part=dc005`, no `declared_ports`), `unbound_required_port` for ADS1115 `io1/io2/io3` |

**Read it honestly: by the plan's own bar this is 0/3 commits, so §2 and §4 are NOT yet
"moved" on this input — and the two named diagnostics are gone from all three drafts.**
`unsupported_lowerer_contract` for `pin-header@1` (the §2 defect) appears in **none** of the
three; `unbound_required_port` for the RP2040's `usb_dm`/`usb_dp` (the §4 defect, the exact
message live run 1 died on) appears in **none** of the three. What the stage now dies on is
two *other* draft-authored contract defects:

* `conflicting_port_binding` — the draft declares a signal (`GND_INPUT`) that duplicates a
  ground the compiler already bound, five times in one draft; the diagnostic already tells it
  to delete the duplicate.
* `unbound_required_port` for `ads1115-i2c-adc@1`'s `io1`/`io2`/`io3` — **the §4 shape
  again, in a different recipe**: the recipe marks all four analog inputs required and none
  groundable, so a two-channel current-sense brief must bind the two channels it does not
  use. The compiler's existing deterministic completion already ties
  `required & groundable` recipe ports (`allow_ground`); ADS1115's unused inputs are the
  honest next candidate for it (a real design grounds them), and it is the same "reviewed
  data / completion" fix as §4, one recipe over.
* `usb_connector_supply_unknown` (1 of 3 drafts) — the new rail bound doing its job: the
  draft wanted a USB socket on a 12 V-only design, and it is told the rail to declare
  instead of being handed a socket with 12 V on VBUS.

## §3 Wiring: the reviewed transfer graph

**The plan's diagnosis was wrong in two ways, and the fix follows the evidence.**

1. *"no inductor carries any transfer fact"* — irrelevant: `_reviewed_transfer_edges`
   already adds a generic undirected edge for **every `L`-prefixed part with two distinct
   wired nets**, so the series hop `SW/PH → L → VOUT` was never missing. Executed on the
   frozen wiring state: the graph held `+3V3 ↔ <switch net>` and nothing else.
2. The actual hole in that artifact is that the buck the recipe emits — `AP63203WU-7` — had
   **no reviewed record at all**, so no edge from `+18V` to the switch node existed. The
   plan's table lists `tps5430*`/`tps54331ddar`/`tlv62569dbvr` (which do carry `VIN → PH/SW`
   and do resolve), not the part that failed.

**Change.** Reviewed records for `ap63203wu-7` and `ap63205wu-7` (`bundle ap63203`/`ap63205`,
LCSC C780769 / C2071056) with `power_transfer={"from_pin": "VIN", "to_pin": "SW"}`, the
datasheet's pin table (FB 1, EN 2, VIN 3, GND 4, SW 5, BST 6 — verified against the vendored
symbols), `vin_min_v/vin_max_v = 3.8/32`, `output_current_a = 2`, and the mandatory
bootstrap external (`BST → SW`, 100 nF — verified in the Diodes datasheet: *"A 100nF
capacitor is recommended from SW to BST"*). The 5 V order code deliberately does **not**
carry the 3.3 V class. One module-level path added for the isolated converter
`b0509s-1wr3` (`{"isolated": true, "paths": [{"from_pin": "2", "to_pin": "4", …}]}`), which
is item 2 of the plan: a complete module proves the conversion on its own and supplies the
isolated reference-domain model §9.39 reads.

**Not done, with the reason.** A catch-diode hop (`PH → GND`) was NOT added. The recipes'
bucks are synchronous (no catch diode is emitted at all: `_buck` has no diode role), and the
graph is undirected, so a `PH → GND` edge would let *any* input reach the reference net and
then any ground-referenced net — manufacturing reachability the gate deliberately does not
accept ("capacitors, control pins, and unreviewed placeholders are not power paths"). The
series hop the chain needs is already the generic inductor edge.

**Coverage report (item 3).** `kicraft/cli/transfer_coverage_report.py`, entry point
`transfer-coverage-report` (`pyproject.toml`), with `tests/test_transfer_coverage_report.py`.
It lists every reviewed power-path record with its resolved `from → to` pins and status
(edge-producible / dead / no fact), the dead facts, the power families with no fact, and the
recipe-emitted power parts with no reviewed transfer. Its first run found three more real
holes, all still open and now visible: `al8860mp-13`'s `SW → GND` transfer resolves to
**nothing** (the symbol has two pins named `SW` and two named `GND`, and
`_pin_number_named` requires exactly one), the identity-only `tps5430`/`tps5430dda` records
carry a transfer no BOM may exercise, and `ch224k` / `tps2553dbvr` carry none.
Recipe parts without a reviewed transfer: `CH224K`, `TP4056`, `MCP1700T-3302E/TT`,
`AMS1117-3.3`.

**Verify.**
* Deterministic, on the frozen wiring state `/tmp/ab-seed20-sw09mq2l`: §9.39 now **passes**
  (`every typed conversion has a reviewed source-to-load path`), and §9.37 / §9.38 stay
  satisfied (the new `bootstrap` spec and `vin_min_v/vin_max_v` are both met by the recipe's
  own bootstrap capacitor and the 18 V rail).
* `stage_driver replay --stage wiring` N-of-3 on the same frozen state:

| run | result | note |
|---|---|---|
| 1 | `[ok] wiring` | cost $0.0071, 4 attempts |
| 2 | `[ok] wiring` | cost $0.0037 |
| 3 | `[fail]` §9.42 `E_DECLARED_INTERFACE 'indicator'.anode` | cost $0.0032; a model-authored declared-pin defect, unrelated to §9.39 |

So the named diagnostic (`E_POWER_TRANSFER … '+18V' → '+3V3'`) is gone and the stage commits
in 2 of 3 replays — the plan's bar, with the one failure being a different gate.

**Guard.** `tests/test_electrical_invariants.py::test_every_reviewed_buck_proves_its_own_source_to_load_path`
drives the real records against the real symbol lookup: every portable reviewed buck must
reach its load over its own datasheet transfer plus the series inductor, and an unreviewed
converter on the same netlist must still fail §9.39.

## §4 Architecture: deterministic completion for a native-USB MCU

**Change.** After the ties loop in `derive_architecture` (`kicraft/design/architecture_intent.py`),
a requirement that resolved to a native-USB MCU recipe (`rp2040-minimal@2`,
`esp32-s3-*`, `esp32-c3-*`) whose `usb_dm`/`usb_dp` nothing wired and whose design owns no
USB data connector gains the reviewed satisfier through the same `_open_edge` machinery the
declared-signal path uses, with the derived note
`"<req>: native-USB MCU needs a data connector; bound usb-c-usb2-device@1 (<id>) (derived)"`.
A partially wired pair keeps the net the design chose for the line it did state. A design
that declares its own USB connector is untouched.

**A bound the plan did not have, added because the derived part was otherwise electrically
wrong.** `_usb_vbus_rail` used to prefer *any* rail a `power_input` requirement generates,
so a 12 V barrel-jack board (run 904's brief) would have got a USB-C socket with **12 V on
its VBUS**. It now accepts only a rail that comes from the board's own *USB* power input, is
named `VBUS`/`+5V`, or is the single rail at ~5 V; otherwise the completion refuses with
`usb_connector_supply_unknown`, naming the rails the design *did* declare. This also matches
the reviewed reference rows (every one of them declares a `VBUS` rail).

**Verified shapes** (executed through `derive_architecture`):
* RP2040 + USB-C power input, no USB requirement → exactly **one** derived
  `usb-c-usb2-device` (`rp2040_usb`), both data lines bound, `vbus=VBUS`, derived note
  present, and the stage commits (which is itself the proof `unbound_required_port` stopped
  firing: any unbound required port raises).
* The same board **with** its own USB connector and signals → exactly one connector, the
  model's own (`usb_dev`), no derived note.
* 12 V jack, no 5 V rail → `usb_connector_supply_unknown` with evidence `{+12V, 3V3}`.
* 12 V jack **plus** a 5 V rail from its own buck → connector derived with `vbus=+5V`.

**Guard.** `tests/test_design_recipes.py`: `_native_usb_board(...)` plus
`test_native_usb_mcu_gains_exactly_one_derived_data_connector`,
`test_native_usb_mcu_that_wires_its_own_connector_is_left_alone`,
`test_native_usb_completion_refuses_a_rail_that_is_not_a_usb_supply`.

**Replay.** Same N-of-3 as §2 (one seed-18 architecture replay exercises both). Measured:
0/3 commits, with the RP2040 `usb_dm`/`usb_dp` `unbound_required_port` — live run 1's exact
message — absent from all three drafts. One of the three drafts creates its own
`usb-c-usb2-device` requirement through `edge:USB`, i.e. the model reaches for the reviewed
connector by itself; that draft is then refused by the new rail bound (no 5 V rail on a
12 V design), which is the bound working. The remaining refusals are listed under §2's
replay table.

**Policy note (the plan's open question 3).** The plan's Change was followed — the connector
is *added* rather than the port left unbound — and the rail bound is what makes that safe.
A 12 V-powered board now gets a *named* refusal (`declare the rail the USB socket exposes`)
instead of "wire `usb_dm`", which is the actionable version of the same wall: the honest
12 V design either declares the rail the socket exposes or does not fit a socket at all.

## Appendix

**`functional_spec` self-loop is a named diagnostic.**
`functional_spec_self_loop` (`repair_required`) now fires in `stage_semantics._functional_spec`
with the offending pair as evidence, instead of the model only seeing the commit gate's bare
`self-loop connection: 'DISPLAY_DRIVE' → 'DISPLAY_DRIVE'` (which carries no gate id).
Guard: `tests/test_stage_semantics.py::test_functional_spec_names_a_self_loop_connection`.
The commit gate check itself is left in place — it is the last line for the offline
`stage-commit` caller — so a draft that ignores the named finding is still refused.

**Triage rejection signatures carry the detector codes.**
`collect_stage_rejections` now keys on the runtime's gate identity **and** the event's
detector codes (deduped), and `_SIG_ALIAS` names the contract-rejection wrapper. Live
before/after on run KC-SGYXF5:

```
before:  the reply was schema-valid but a deterministic design contract refused it …  x2
           <-- 5 DISTINCT diagnostics under ONE signature: the error text cannot tell them apart
after:   (no gate id: a semantic contract refused a schema-clean candidate)
         + multiple_intent_contracts + declared_signal_port_tied + unknown_interface_port
         + unsupported_lowerer_contract x1
         (no gate id: …) + multiple_intent_contracts + unbound_required_port x1
```

KC-3SRTSK now reads rung-by-rung: `unsupported_lowerer_contract` ×2 → `unknown_interface_port
+ unsupported_lowerer_contract` → `unsupported_lowerer_contract` alone, i.e. convergence is
visible. The now-impossible `distinct_diagnostics` field and its warning were deleted.
Guard: `tests/test_triage_cli.py::test_rejection_signature_carries_the_detector_codes`.

**One owner for the failure block.** The inline failure card owns a stopped run's
affordances; the summary card above it no longer repeats the same button
(`_failure_card_shows` is the single predicate both painters use). Measured on the real
surface (isolated instance, seeded with run 908's failed state):

| | `Continue design` nodes | visible |
|---|---|---|
| before | 3 | 2 (y≈413 summary, y≈967 card) |
| after | 2 | 1 (y≈935 card) |

The hidden node is the composer's own button, shown only in the composer state. Control
case: an `awaiting_input` project (no failure card) still renders its action
("Answer questions") in the summary, so nothing else lost its affordance. The summary keeps
the status headline and the aggregated ISSUES list on purpose: those are the status line and
the cross-stage attention list, not a second failure block.

## Regression gate

**Full suite: `1 failed, 4448 passed, 15 skipped, 1 xfailed` in 11m49s.** The one failure is
the pre-existing `tests/parts_library/test_maturity.py::test_vendored_bundles_are_not_prototype`
(`ams1117-5v0-fixed` still defaulting to `prototype`) — exactly the failure the plan's own
baseline names, and untouched by this work. **No new failures** (baseline 4439 passed → 4448,
the +9 being this session's new tests; one of the ten test functions it touches is a rename).
Focused runs while iterating: the touched areas
(`test_part_identity`, `test_stage_work_units`, `test_design_lowering`, `test_design_recipes`,
`test_electrical_invariants`, `test_stage_semantics`, `test_triage_cli`,
`test_transfer_coverage_report`, `test_edge_zone_derivation`, `test_form_factor_reconcile`,
`test_design_acceptance`, `test_architecture_intent`, `parts_library`) are all green apart
from that same known failure.

Total session spend on provider calls: **$0.058** (one seed-22 brief at $0.0096, three
architecture replays at $0.0155, three wiring replays at $0.0140, plus the two abandoned
replay attempts).
