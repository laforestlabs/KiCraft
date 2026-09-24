# Live Surprise-me yield session (2026-09-24)

**Scope.** Drove the production site (kicraft.io, pro test account) through six
`Surprise me` briefs, monitored the UI and logs, measured the two knobs the owner
suggested (thinking, ladder), fixed what the runs exposed, and re-measured on
fresh briefs. **No config lever produced a board.** Two real defects were found
and fixed; the wall is unchanged and now precisely named.

## 1. The live sample (all through the web UI, seeds from the Surprise-me counter)

| # | board | seed | brief shape | config | outcome | cost |
|---|---|---|---|---|---|---|
| 1 | KC-SGYXF5 | 18 | RP2040 current-sense, 12 V barrel jack, I2C + 8-pin header | stock | architecture ✗ (3 attempts) | $0.0093 |
| 2 | KC-3WCVER | 19 | ATtiny1604 + DRV8833, 5 V header, screw terminals | stock | architecture ✗ (3) | $0.0094 |
| 3 | KC-22Z3V5 | 20 | ESP32-S3 + ULN2003, 18 V DC, JST, <80×60 | stock | architecture ✗ (3; `truncated_json`×2 then `contract_rejected`) | $0.0161 |
| 4 | KC-3SRTSK | 21 | CH32V003 current-sense, **24 V DC screw terminal**, 8-pin header | ladder | architecture ✗ (**4 rungs**) | $0.0098 |
| 5 | KC-GWVAEU | 22 | STM32G0 logger, **12 V barrel jack**, Qwiic + JST | ladder | **architecture ✓**, BOM ✗ | $0.0212 |
| 6 | KC-KAHKR7 | 23 | ESP32-C3 7-seg driver, 2S Li-ion, buttons | ladder | functional_spec ✗ (self-loop) | $0.0029 |

Baseline 0/3 reached a build; after 0/3 reached a build. Run 4's four-rung ladder
(`normal → serialization → clean_slate → normal`) is the only *proof* the ladder arm
was live in the deployed process.

## 2. Lever measurements (CLI harness, one brief per seed, `--no-build`)

| arm | seed 18 | seed 19 | seed 20 | cost/run |
|---|---|---|---|---|
| stock (control) | arch ✗3 | arch ✗3 | arch ✗3 | $0.007–0.011 |
| `signature,full_feedback` | arch ✗4 | arch ✗4 | **arch ✓** → bom ✗ | $0.009–0.015 |
| thinking 4096 (stock ladder) | arch ✗4 | arch ✗4 | arch ✓ → **bom ✓** → wiring ✗ §9.39 | $0.018 (2×) |
| ladder + the two fixes below | arch ✗4 | arch ✗4 | arch ✓ → bom ✗ (different defect) | $0.010–0.016 |

**Neither lever survives its control.**

- **Ladder**: on the two fresh briefs (22, 23) a **stock** control reached the same
  stages (seed 22 committed architecture under both). Its only win is the N=1 seed-20
  flip. Reverted in `.env` (documented, not enabled) — N=1 is below this repo's N-of-3 bar.
- **Thinking**: replaying the *same frozen* BOM state 3× with thinking off and 3× with
  4096 tokens on gives **0/3 vs 0/3** (`unit_repair_exhausted`, 4–7 attempts). The
  paired apparent win came from a different architecture candidate, not from thinking
  fixing the BOM. Costs 2–3× money and wall.

Budget was never the constraint: $0.26 spent across the whole session (ceiling $20/day).

## 3. Two defects found and fixed (`c591ec9`, deployed and verified)

1. **`power-connector` was a one-part class.** Only the JST PH 2P record carried it, so an
   obligation spelled with that generic class could be satisfied by *no* screw terminal —
   while briefs keep asking for terminal DC inputs (seeds 19, 20, 21). The reviewed WJ126V
   2P/3P/4P records and the terminal feature pattern now carry `power-connector`
   (LCSC Terminal Blocks, 250 V / 18 A, datasheet cited on the record).
   *Verified*: the defect disappears from the BOM failure it caused (`/tmp/ab-seed20-7uhsk6c0`
   now fails on `missing-requirement-implementation=['jst']` instead), and the class resolves
   to 4 reviewed parts instead of 1.
2. **A unit refusal made the state unreadable.** `stage_status.<stage>.diagnostics` is typed
   `list[StageDiagnostic]` (required fields, `extra="forbid"`), but the unit path wrote
   `{unit_id, defects}` into it: every state saved after a BOM/wiring unit exhausted its
   repair loop failed `ConversationState.model_validate` (5 errors), so `stage_driver replay`
   could not open exactly the runs it exists to iterate on. The row is now a typed
   StageDiagnostic (code = the failing check, evidence = the offender lines) with `unit_id`
   and `defects` kept as new optional fields.
   *Verified*: a state written by the fixed writer loads while the pre-fix shape is rejected
   (4 errors); regression test `test_unit_refusal_is_written_as_a_loadable_stage_diagnostic`.

## 4. UI and log findings

| finding | evidence |
|---|---|
| A failed run renders the SAME failure **three times** — run card ISSUES, the red "did not complete" banner, and the stage tab's PROJECT STATE | DOM: three `Continue design` nodes, two visible (y≈358 and y≈1214); full-page screenshot |
| Recurring client-side NiceGUI/Vue `TypeError: Cannot read properties of null (reading '0')` | 9 occurrences across the session, always after a failure/ISSUES repaint; non-fatal (stepper, live stream and buttons kept working) |
| `triage` collapses 3–5 distinct diagnostics under one rejection signature, so the error text cannot tell them apart | printed by `triage stages` on runs 904/905/907 |
| The four ladder arms are `.env`-only, while `/admin/design` exposes only model/behaviour knobs | `routing_config.ALLOWED_KEYS`; enabling an arm needs a restart |
| No server-side errors | web log clean across all six runs |

## 5. The ranked blockers (each with the evidence that names it)

**GAP 1 — architecture semantic contracts (the wall for 5 of 6 live runs).**
Breadth: 589 designs (`architecture: contract_rejected`, latest today). Members seen live:
- `unsupported_lowerer_contract`: the model declared an 8-pin header as
  `requirement_ports=pin1,pin2,pin3,pin4,pin8` while `pin-header@1` publishes
  `port_contract=<pin1..pinN: contiguous explicit physical pin numbers>`,
  `parameter_choices=rows=[1, 2]`. A header whose middle pins are unused is a normal
  design; the contract has no shape for it.
- `unbound_required_port`: `rp2040-minimal@2` marks `usb_dm`/`usb_dp` **required** (GPIOs are
  explicitly `required=False`), so any RP2040 brief that does not mention USB must still wire
  them — nothing adds the USB connector deterministically, while the lowerer branch *does*
  complete its own published supply/reference ports.
- `declared_signal_port_tied`, `unknown_part_refused` (a generic Schottky needs
  `declared_ports`), `unknown_interface_port`.

**GAP 2 — no reviewed barrel jack.** `realizable_physical_features("barrel-jack-connector")`
is **empty** and no part in the library mentions "barrel", yet the corpus asks for a
"12 V DC barrel jack" constantly (seeds 16, 17, 22). The gate then refuses the model's
placeholder (`Connector:Barrel_Jack_Switch`, `mpn="DC barrel jack"`) — correctly, per the
no-reviewed-coverage fallback — and the model's next move is to invent sourcing
(`sourcing_note claims LCSC C9900003477 which is not in the offline catalog`). Either author
a reviewed barrel-jack record or the BOM stage needs a real part to reach for.

**GAP 3 — wiring gate §9.39 (reviewed source-to-load power transfer).** The deepest run of
the session died here: `E_POWER_TRANSFER 'regulator': no reviewed source-to-load transfer
from '+18V' to '+3V3'`. A reviewed buck exists (`tps5430ddar` → `buck-converter-ic`), so the
question is which transfer records the gate accepts.

Appendix: run 6's `functional_spec` self-loop (`'DISPLAY_DRIVE' → 'DISPLAY_DRIVE'`) plus
`functional_spec_drive_missing_power`, and the fabricated-LCSC sourcing note above (the
audits' Pass C class).

## 6. Reproducing this session

```bash
PY=/home/kicraft/KiCraft/.venv/bin/python
$PY /tmp/kicraft-ab/ab.py --tag <arm> --seeds 18,19,20        # brief-per-seed, JSONL out
KICRAFT_CONTRACT_LADDER=stock $PY /tmp/kicraft-ab/ab.py ...   # control arm (env wins over .env)
KICRAFT_ROUTING_CONFIG=/tmp/kicraft-ab/rc-reasoning.json $PY /tmp/kicraft-ab/ab.py ...  # thinking 4096
$PY -m kicraft.server.stage_driver replay --state <ws>/.kicraft/state.json --stage bom --budget 0.6
$PY /tmp/kicraft-ab/watch.py <project_id>                     # live project -> verdict
```

Artifacts: `/tmp/kicraft-ab/*.jsonl` (arm results), `/tmp/ab-seed20-*` (frozen states),
`/tmp/kc-replay-*` (the N-of-3 workspaces). Live rows 904–909 in `accounts.db`.
