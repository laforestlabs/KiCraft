# Yield movers: the four changes that should move the number (2026-09-24)

**Status: implemented 2026-09-24** — see `docs/plans/yield-movers-2026-09-24-session.md` for
the per-item result, the guard tests, the replays, and the three places this plan's own
diagnosis turned out to be wrong.

This plan was written after the live Surprise-me session
(`docs/plans/live-surprise-me-yield-2026-09-24.md`) measured the current yield and
ruled out the two knobs (thinking, ladder) with controls. Every item below is a
*fix at the source* of a named refusal, each with the artifact that proves it and a
measurement recipe that produces a comparable before/after.

**The one-sentence thesis.** The pipeline is not short of budget or attempts: it is
short of *reviewed coverage* for the parts and shapes the corpus asks for. Four
specific gaps kill most runs, and three of them are small, citable additions.

## 0. The measured baseline this plan must beat

| population | number | source |
|---|---|---|
| runs with an event stream that never reached a board | **1745 / 2175 (80%)** | `triage scan` |
| top terminal mode | `architecture: contract_rejected` — **589 designs, latest today** | `triage scan` |
| top diagnostic | `architecture: multiple_intent_contracts` — 484 | `triage scan` |
| live briefs driven today (6, seeds 18–23) | **0 boards**; 1 reached BOM | this session |
| cost of a dead run | $0.003–0.021 | this session |

A change "moved the number" only if it converts a *named* refusal into a committed
stage on the same frozen input, N-of-3, and does not regress the full suite.

## 1. Reviewed barrel-jack coverage — cheapest, biggest corpus share

**Why.** `realizable_physical_features("barrel-jack-connector")` is **empty** — none of the
reviewed or stock records (`REVIEWED_PARTS`, `_STANDARD_LIBRARY_PARTS`, `_STOCK_COMMON_PARTS`)
mentions "barrel" — yet "12 V/24 V DC barrel jack" is one of the most common Surprise-me
shapes (seeds 16, 17, 22 today). Every such brief is dead at BOM by construction: the
obligation demands `barrel-jack-connector`, no reviewed part can carry it, and the
no-reviewed-coverage fallback rightly refuses the model's placeholder
(`barrel=Connector:Barrel_Jack_Switch mpn="DC barrel jack"`).

**Two parts already exist** — but neither is a reviewed record, and **neither carries a
resolvable LCSC code**:

| bundle | MPN | LCSC | state |
|---|---|---|---|
| `kicraft/parts_library/dc005-barrel-jack/` | `DC005` | **C431533** | production maturity + an lcsc.com datasheet URL, but the code is **not in the offline catalog** either |
| `kicraft/parts_library/dc-barrel-jack-5-5-2-1/` | `DC005四脚贴片5.5*2.1` | **C9900003477** | run 5's rejected BOM group is *this bundle's own MPN* with *this manifest's code* in `sourcing_note` → `unresolved-sourcing`. The model did not hallucinate the code — it read it from our manifest |

The catalog the pipeline checks against (`~/.kicraft/jlcparts/cache.sqlite3`, 633 k rows)
*does* carry the class — subcategory **"DC Power Connectors"**, with real DC-005 variants at
e.g. `C16214`, `C84007`, **`C130239` (`DC-005-20A`, THT)** — so this is vendoring work, not a
missing category.

**Change.**
1. Vendor a **catalogued** barrel jack as the reviewed record: pick a DC-005 variant present
   in the offline catalog (e.g. `C130239`), confirm its footprint against the KiCad library,
   and add the `ReviewedPart` to `kicraft/design/part_identity.py` with
   `physical_features={"barrel-jack-connector", "power-connector", "wire-to-board-connector"}`,
   the catalog hit as the citation, and the switch-pin note from the bundle's `watch_out_for`.
2. Re-vendor or delete **both** existing bundles: one claims a fabricated-range code, the
   other a code that does not resolve. A vendored bundle whose code is not in the catalog is
   the audits' Pass C `FABRICATED-LCSC` class, and it actively teaches the model a bad code.
3. Same pass, same reason: check every class with one-or-zero carriers
   (`has_reviewed_coverage(...)` over the classes the corpus actually demands) — barrel jack
   is unlikely to be the only hole.

**Guard.** `realizable_physical_features("barrel-jack-connector")` is non-empty; a test that
a BOM group using the DC005 satisfies a `barrel-jack-connector` obligation (same shape as
the `power-connector` test added in `c591ec9`).

**Verify.** Replay `/tmp/ab-seed20-7uhsk6c0` (or the frozen seed-22 BOM state) with
`stage_driver replay --state … --stage bom`, N-of-3 → expect `[ok] bom`; then the full
brief through `stage_driver run` → expect the run past BOM.

**Risk.** Low: additive record, no gate touched. The one judgement is the feature triple —
cite the LCSC category and the datasheet, never a prefix rule.

## 2. Lowering: a header may leave interior pins unconnected

**Why.** Run 4 (seed 21) is the cleanest evidence in the session:

```
requirement_id=header, sheet=HEADER
evidence: port_patterns=pin[1-9][0-9]*
          port_contract=<pin1..pinN: contiguous explicit physical pin numbers>
          requirement_ports=pin1,pin2,pin3,pin4,pin8      ← the gap
          parameters=rows,gender ; parameter_choices=rows=[1, 2],gender=['male','female']
```

The brief asked for an "8-pin 0.1 inch header"; the model declared the pins it *uses*
(1–4 and 8) — a normal board with three unused header pins. `pin-header@1` requires
contiguous `pin1..pinN`, so the architecture stage refused it on all four rungs. The
lowerer *already knows how to leave a pin unconnected*: it emits `LoweringNoConnect` for
any `net == "NC"` (`kicraft/design/lowering.py`, the pin/NC construction below the
`pin-header@1` branch), so the shape exists — only the port-contract validation rejects it.

**Change.** In the pin-header (and socket) lowerer, accept a declared *ordered subset* of
`pin1..pinN`: derive `N` from the requirement's `rows`/`per_row` parameters (or from the
maximum declared pin), fill interior gaps with `LoweringNoConnect` and record the fill in
the group's notes. Update the published `port_contract` text (`lowering.py:1732`) and the
model-facing hint to say a subset is allowed and the rest become no-connects.

**Guard.** Unit test with the exact run-4 ports (`pin1,pin2,pin3,pin4,pin8`, `rows=1`):
`lowerer_contract_diagnostic` returns None and the emitted group has pins 1–8 with 5–7 as
no-connects. Plus the existing `unknown_ports` case still refuses a genuinely unknown port.

**Verify.** `stage_driver replay --state <run 907 state> --stage architecture` N-of-3 →
expect `[ok] architecture` (today: 4 rungs, all `unsupported_lowerer_contract`).

**Risk.** Medium: this widens what a header requirement may declare. Bound it — the
declared pins must be a subset of `pin1..pinN` in ascending order, no gaps *below* the
maximum that are not explicitly `NC`-filled, and `rows`/`per_row` must still be a published
combination.

## 3. Wiring: complete the reviewed transfer graph for switching converters

**Why.** The deepest run of the session died at `§9.39`:
`E_POWER_TRANSFER 'regulator': no reviewed source-to-load transfer from '+18V' to '+3V3'`.
The gate walks a graph of reviewed `power_transfer` facts
(`kicraft/design/synthesis/validation.py:2549-2633`, `_transfer_reaches`). The facts
inventory is the problem:

| part | family | reviewed transfer |
|---|---|---|
| `me6211c33m5g-n`, `ap2112k-3.3trg1` | linear-regulator | `VIN → VOUT` ✅ complete |
| `ams1117-5.0` | linear-regulator | `3 → 2` ✅ complete |
| `tps5430*`, `tps54331ddar`, `tlv62569dbvr` | buck-converter / buck-regulator | `VIN → PH` / `VIN → SW` ❌ **stops at the switch node** |
| *(no inductor carries any transfer fact)* | — | ❌ |

So a design that converts 18 V → 3.3 V with a **buck** can never prove a source-to-load
path: the chain needs `VIN → PH → (inductor) → VOUT` and the inductor contributes no hop.
Only a *linear* regulator can pass, which is the wrong engineering answer for an 18 V input.
This is not a gate bug — it is missing reviewed data, and it blocks every switching-converter
brief in the corpus ("18 V DC input", "24 V DC", "2S Li-ion + 3V3 rail").

**Change.**
1. Add reviewed `power_transfer` facts for the buck chain's passives: the power inductor
   (`SW`/`PH → VOUT`) and the catch diode (`PH → GND`, or `GND → PH` for the return), each
   cited from the reviewed buck's datasheet reference design — the source document the buck
   record already carries.
2. Where the datasheet defines a complete converter (module-style parts), record the
   module-level `paths` entry (`VIN → VOUT`, with `required_externals`) so a reviewed module
   proves the conversion on its own.
3. Add a coverage report (`python -m kicraft.cli.part_query_report` style) listing families
   whose reviewed transfers are *incomplete* (a `from_pin` that reaches no `to_pin`), so this
   class of hole is visible without a live run.

**Guard.** A test that for every reviewed buck, the transfer graph reaches `VOUT` from `VIN`
when the datasheet's required externals are present; and that an unreviewed placeholder still
fails `§9.39`.

**Verify.** Replay the frozen wiring state (`/tmp/ab-seed20-sw09mq2l`) with
`stage_driver replay --state … --stage wiring` N-of-3 → expect `[ok] wiring` (today:
`commit_rejected`, gate 9.39), then the full brief to a board.

**Risk.** Medium-high: it edits reviewed electrical facts. Each hop must be traceable to a
datasheet (pin-level), and the gate's own reference-domain checks must stay satisfied.

## 4. Architecture: deterministic completion for a native-USB MCU

**Why.** `rp2040-minimal@2` marks `usb_dm`/`usb_dp` **required**
(`kicraft/design/recipes/rp2040_minimal_v2.py:39-47`; the GPIO ports are explicitly
`required=False`), and the architecture gate refuses any design that leaves a required port
unbound (`kicraft/design/architecture_intent.py:2074-2085`, `unbound_required_port`).
Live run 1 died exactly there (`unbound_required_port: requirement 'rp2040' port 'usb_dm'
is required by rp2040-minimal@2 and nothing wires it`) — for a brief that never mentioned
USB. The gate is right that an RP2040 needs a programming path; the pipeline is wrong to
leave it to the model, because the reviewed satisfier exists:
`usb-c-usb2-device@1` (`kicraft/design/recipes/wave_b_power.py:148`).

**Change.** Extend the deterministic completion that *already* runs in this module: the
lowerer branch binds published supply ports to their rail and reference ports to GND
(`architecture_intent.py:2030-2050`, with `derived_notes`). Add the recipe analogue: when a
requirement resolved to a native-USB MCU recipe and no requirement in the architecture owns
a USB data connector on those nets, bind the reviewed `usb-c-usb2-device@1` recipe and record
a derived note (`"<req>: native-USB MCU needs a data connector; bound usb-c-usb2-device@1
(derived)"`), the same way the resolver records a substitution assumption.

**Guard.** A test that an RP2040 architecture with no USB requirement gains exactly one
derived USB connector (and that a brief which *does* declare its own USB connector is not
given a second one). The `unbound_required_port` code must stop firing for native-USB MCUs.

**Verify.** `stage_driver replay --state <run 904 state> --stage architecture` N-of-3 →
expect `[ok] architecture`.

**Risk.** Medium: it adds a component to a design. Bound it to (a) a native-USB family and
(b) a genuinely unbound USB port, and record it in the notes so a reader sees why the part
is there.

## 5. Appendix (small, do while in the area)

| item | evidence | change |
|---|---|---|
| `functional_spec` self-loop | run 6: `self-loop connection: 'DISPLAY_DRIVE' → 'DISPLAY_DRIVE'` + `functional_spec_drive_missing_power` | the stage's schema/semantics should reject a self-loop as a *named* diagnostic the model can act on, not as a schema error with no gate id |
| unit-refusal diagnostics lose their gate ids in triage | `triage stages` prints `3 DISTINCT diagnostics under ONE signature` for runs 904/905/907 | include the diagnostic codes in the rejection signature so the operator can tell convergence from ping-pong |
| a failed run renders its failure three times | DOM: three `Continue design` nodes, two visible; run card + red banner + stage tab | pick one owner for the failure block |

## 6. Order, and why

1. **§1 barrel jack** (hours, additive, unlocks the most common brief shape).
2. **§2 header subset** (one contained lowering change with an exact repro).
3. **§3 transfer graph** (highest leverage on power briefs, needs datasheet work).
4. **§4 native-USB completion** (medium; the satisfier already exists).

Items 1–2 are independent and can land in parallel; 3 and 4 both touch reviewed data and
should land one at a time so each keeps its own N-of-3.

## 7. Measurement protocol (every item, same shape)

```bash
REPO=/home/kicraft/KiCraft; PY=$REPO/.venv/bin/python
# 1. paired stage replay against the FROZEN state that failed (N-of-3 — one verdict is a coin flip)
$PY -m kicraft.server.stage_driver replay --state <frozen>/.kicraft/state.json --stage <stage> --budget 0.6
# 2. full brief, LLM stages only (cheap, isolates the change)
$PY -m kicraft.server.stage_driver run --brief "<brief>" --workspace "$(mktemp -d)" --budget 0.6 --no-build
# 3. brief-level A/B on NEW seeds, control arm = the same seeds before the change
$PY -m kicraft.server.stage_driver run --brief "$($PY -c 'from kicraft.server.examples import generate_brief as g; print(g(22))')" \
    --workspace "$(mktemp -d)" --budget 0.6 --no-build
# 4. the honest end-to-end: a live Surprise-me run through the UI, then triage + build log
$PY -m kicraft.cli.triage stages <KC-XXXXXX>; $PY -m kicraft.cli.triage run <KC-XXXXXX>
```

Rules this repo has already paid for: N-of-3 for any LLM verdict; never compare artifacts
across separate replay runs; match `--quality` to the original; a stage verdict is only
"moved" if the **named diagnostic disappears** *and* the stage commits; a fix at the wrong
layer (masking the gate, adding a prefix rule, post-hoc band-aid) is rejected on principle.

**Regression gate per commit:** full suite (`4439 passed, 1 failed` today — the failure is
the pre-existing `ams1117-5v0-fixed` maturity guard) and, for anything touching layout,
`cli_app replay` on a frozen workspace.

## 8. Budget and cadence

Every measurement above is cents: today's whole session — 6 live runs, 4 arms, 6 replays —
cost **$0.26** against a $20/day ceiling. The scarce resource is reviewed-data effort
(datasheet reading), not tokens. One item per commit, each with: the diagnostic that
disappears, the N-of-3 replay, the guard test, and a line in the session record.

## 9. Open questions for the owner

1. **Barrel jack**: re-vendor `dc-barrel-jack-5-5-2-1` or drop it? It is the source of the
   code the model fabricated into a sourcing note.
2. **§3 scope**: pin-level facts for the passives, or module-level `VIN → VOUT` facts on the
   buck records? The former is more honest about the chain; the latter is fewer records.
3. **§4 policy**: should a derived USB connector be *added* to a design that did not ask for
   one, or should the architecture simply be allowed to leave `usb_dm`/`usb_dp` unbound when
   the brief shows no USB intent? Adding it makes the board programmable; leaving it alone
   respects the brief. (The gate's own message says the part "cannot work without it".)
4. **Should the ladder arms be re-measured** with a larger sample before being dismissed
   (today: N=1 positive, control-equivalent on fresh briefs), or is `stock` settled?
