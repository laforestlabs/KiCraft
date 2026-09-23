# Handoff: terminals, design-stage contracts, backend switch (2026-09-23, second session)

**Scope.** This session implemented `docs/plans/next-session-live-run-2026-09-23.md`
(A1 terminal datum, B live run, C backend A/B), reviewed and committed another
session's in-flight work, and pushed the A2 (design-stage contract) queue as far
as it could be proven. Nine commits, all deployed.

**Status in one line.** A live Surprise-me request now ships a fab-ready board
(`KC-MQNE7R`); the terminal blocker is gone; the next production blocker is the
pinned design engine's parts-list completion, which lands as a wiring park.

Read `docs/plans/next-session-live-run-2026-09-23.md` for the runbook (B0–B7) and
the per-item evidence; `docs/plans/terminal-opening-datum-2026-09-23.md` for the
terminal review and its mechanism.

---

## 1. What landed (commit → what → evidence)

| commit | change | proof |
|---|---|---|
| `3e74f3a` | Reviewed wire-entry datum for the vendored Kangnex **WJ128V** 4P/5P terminals (`footprint_opening.py`) | mesh registered on the parts' own pins: wire reaches the clamp 8.6–9.1 mm from +Y vs 1.2–2.0 mm from −Y (datasheet strip length 6~7 mm); cold `replay` of project 892 now composes + routes where it previously rejected every candidate |
| `42e9d4b` | Session record: live run + backend A/B + next terminal task | the run's own artifacts (below) |
| `aa3193d` | Semantic-repair candidates are normalized exactly like the first candidate (the repair was being charged defects the original never faced, and discarded) | new test **fails on a pre-patch scratch tree** (`repair_adopted is False`), passes here |
| `a811624` | Electrical reviewer → `openai/gpt-5.6-luna` on route `["openai"]`, **both** default literals; eval judge deliberately unchanged | live `.env` carries the overrides; rollback = one-line env revert |
| `2bec9e8` | A functional block whose obligations are all board-wide facts needs no implementing requirement (was: every block needed one, so a brief's `quantity` mounting-holes row made a `MOUNTING` block impossible to implement → all 3 architecture attempts refused) | frozen beacon state: 0/1 before → `replay` N-of-3 **2/3 committed** after |
| `70482ae` | A saved commit-gate rejection is readable again (`detector_version` written; `StageDiagnostic` reads additively) | this was what blocked the plan's own `replay` method: `diagnostics.1.detector_version Field required` |
| `dd9342c` | A2 record: what was fixed, what remains, with mechanisms | — |
| `e7e500c` | WJ126V 2P/3P reviewed mouths in the table, so a *markerless* copy still measures (a markerless 3P read 180° on KC-AHW6GA; the reviewed datum is 90°) | tests incl. the markerless cache copies, the stripped-marker gate verdict, and the CLI staying quiet |
| `8036194` | Correction: `KC-MQNE7R`'s terminal verdict *was* reviewed data; terminal-datum inventory | leaf board carries the reviewed `PCB Edge` marker at local `(-3.8, 0)` |

Test state: full suite **4446 passed, 15 skipped, 1 xfailed, 1 failed** — the one
failure is the pre-existing `tests/parts_library/test_maturity.py::
test_vendored_bundles_are_not_prototype` (`ams1117-5v0-fixed` still `prototype`),
documented as environmental in `docs/plans/design-yield-recovery-2026-09-20-options-1-4-handoff.md`.

Deployed with `deploy/deploy-production.sh` after each landing; last check: web
HTTP 200, `[build-worker] ready`.

---

## 2. Live production evidence (2026-09-23)

| board | status | cost | stage outcome |
|---|---|---|---|
| `KC-MQNE7R` (mine, seed 9) | **ok** | $0.0352 | all five stages committed (bom 4 attempts, wiring 3); `verify: shorts=0 unconnected=0 courtyard=0 keepout=0 traces=158 components=16/16`; `build_jobs id=327 rc=0`; fab zip + receipt + STEP + 3D render; FAB tab with the KiCad download, 9m49s |
| `KC-S7DRXZ` (operator) | failed | $0.0418 | intent/fs/architecture/**bom** ok → wiring ✗ `bom_reconcile_exhausted`: "The LM358 amplifier has unused second-channel pins 5, 6, 7, but the BOM provides no defined bias/feedback network" |
| `KC-MTMKKE` (operator) | failed | $0.0433 | same shape → wiring ✗ `bom_reconcile_exhausted`: "The TPS61088RHLR needs its complete mandatory support network…" |

Those two are the *newest* production runs and they name the live blocker
precisely: the **pinned engine's BOM omits required support parts**, wiring asks
for them, the reconcile budget (3 passes) is spent, and the run parks. Same class
this session's A/B measured on the legacy arm, and the class
`docs/handoff-general-brief-yield-2026-09-22.md` already described ("the parked
asks are protocol, not electrical checkers").

Today's project rows, for context: 890 `KC-9N7B8F` failed $0.0537 · 891
`KC-FSXF35` failed $0.0787 · 892 `KC-CG58R4` failed $0.0609 · 894 `KC-GM5UC2`
failed $0.0588 · 895 `KC-Y86YCJ` failed $0.0085 · **896 `KC-MQNE7R` ok $0.0352** ·
897/898 as above.

---

## 3. The backend decision (plan section C) — measured, not flipped

Owner decision: A/B first, then flip; measure the reasoning budget in the same
run. Nine runs, three unseen briefs from the frozen 60, $0.1501, `--no-build`,
production `routing.json` untouched (throwaway `KICRAFT_ROUTING_CONFIG`).

| brief | `cur0` current, reasoning 0 | `curR` current, reasoning 4096 | `leg` pinned `bc6a2f8` |
|---|---|---|---|
| greenhouse sensor hub | **architecture ✗3** (`source_obligation_not_retained`) | arch ✓2, **bom ✗** ("requires 1 real qwiic-receptacle, found 0") | arch ✓, **bom ✓4**, wiring ✗ park |
| line receiver | **architecture ✗3** | arch ✓2, **bom ✗** ("requires 1 real coupling-capacitor") | arch ✓, **bom ✓3**, wiring ✗ park |
| LED beacon | **architecture ✗3** | **architecture ✗3** (block-sheet mapping — the defect fixed in `2bec9e8`) | arch ✓, **bom ✓2**, wiring ✗ park |

**Call: do not flip yet.** The pinned engine still reaches furthest on unseen
briefs (4/5 stages, 3/3), and the current engine's deaths are contract classes
that are now partially fixed. Reasoning measurably helps *architecture*
(0/3 → 2/3) at 4–8× its cost; with architecture solved the current engine then
dies in the **BOM work-unit contract**, which is the next fix.

Method to repeat after the next fixes (production untouched):

```bash
# current arm: reasoning 0 / 4096
KICRAFT_ROUTING_CONFIG=/tmp/ab/rc-current.json \
  .venv/bin/python -m kicraft.server.stage_driver run --brief "$(cat brief.txt)" \
  --workspace "$(mktemp -d)" --budget 0.15 --no-build --quality draft
# legacy arm (pinned engine, its own env)
cd ~/KiCraft-legacy && KICRAFT_PROVIDER_ORDER=openai KICRAFT_MAX_PRICE_PROMPT=0.20 \
  KICRAFT_MAX_PRICE_COMPLETION=1.20 KICRAFT_DESIGN_REASONING_TOKENS=0 \
  .venv/bin/python -m kicraft.server.stage_driver run --brief "$(cat brief.txt)" \
  --workspace "$(mktemp -d)" --budget 0.15 --no-build
```

---

## 4. Open queue (owners + how to settle)

1. **BOM work-unit refusals (biggest live item).** Two shapes, both fatal:
   - *current* engine: a typed work unit demands a class-tagged real part
     ("requires 1 real coupling-capacitor") and the model emits a generic
     `Device:C`; the stage reports `attempts=0`, i.e. the unit is never re-driven
     with that feedback. Candidate fix: one bounded retry per refused unit
     (`kicraft/server/stage_work_units.py`, stage driver).
   - *pinned* engine: missing support parts → wiring park (today's 897/898).
     Fixes land in `~/KiCraft-legacy` (separate repo, operator-approved to patch)
     or in the current tree's reconcile path.
2. **`multiple_intent_contracts`** — a port-binding conflict the architecture
   keeps re-emitting (rail `+5V`: port `vbus` of `usb` already bound to `VBUS`);
   one of three post-fix replays on the beacon state.
3. **Terminals still unmeasured** (blocked at compose, honestly, not guessed):
   `CONN-TH_4P-P5.00_WJ126V-5.0-4P-1`, `CONN-TH_5P-P5.00_WJ126V-5.0-5P` (no
   marker in-repo, no 3D model) and `CONN-TH_5P-P5.00_WJ127-5.0-5P` (no marker;
   mesh does not register on its own outline). Either review them per-part or
   implement the terminal doc's fix 2 (emit a reviewed family from the design
   path so a BOM cannot name an unreviewed terminal).
4. **Edge-zoned non-connector parts** (`connector_orientation_unmeasured:RT1`,
   `connector_misoriented:Q2` on project 892's board): parts whose edge zone is
   an access/field-of-view hint — a trim pot, a phototransistor — are gated as
   directional connectors. `connector_edge_gap._access_only_connector` is the
   existing allowlist; extending it, or gating the facing/stranding verdicts on
   connector identity rather than THT evidence, is a policy decision. The
   *compose* gate does not block them (`_directional_edge_candidate` requires
   `kind == "connector"`), the fab gate has no equivalent.
5. **D2 / the identity-`conversion` modelling gap** (I2C bus → I2C bus, then
   architecture is asked to own it): no deterministic rule separates that from a
   real conversion, so the documented remedy is the rewriting auditor
   (`docs/plans/stage-auditor-luna-2026-09-23.md`, workstream B, not built).
6. **Backend flip** — only after (1)/(3) land; repeat the nine runs with
   reasoning on, then change `~/.kicraft/routing.json` deliberately.

---

## 5. Mechanics and traps (learned the hard way this session)

- **A leaf artifact freezes its components' `opening_direction`.** Re-running the
  compose CLI on an old workspace re-reads the pre-fix `None`, so a place/route
  fix *cannot* be verified that way — re-solve the leaves (`cli_app replay`, or
  **Rebuild board**) with a cold `.experiments`. The plan's B7(c)/(d) order is
  backwards for this class of fix.
- **Replay needs the pre-promote seed.** `rm -rf .experiments` on a workspace
  whose promoted board is an rc6 partial deletes the seed the replay restores
  from; copy `.experiments/pre_promote_seed.kicad_pcb` first.
- **A commit-gate rejection used to make the saved state unloadable** (fixed in
  `70482ae`). If `replay` reports `diagnostics.N.detector_version Field
  required`, that is this defect on a state written before the fix.
- **The home-tier parts cache is stale but shadowed.** `~/.kicraft/parts/
  screw-terminal-5mm-{2p,3p}` hold marker-less copies; the loader prefers curated
  content, so they never reach a board. Do not read a `detect_opening_direction`
  result taken directly from that tier as evidence — it is what produced a wrong
  claim in this session (and the same mistake is easy to repeat).
- **Apply a model's declared transform.** e.g. the WJ126V-2P footprint declares
  `(rotate (xyz 0 0 270))` for its `.wrl`; a mesh review that ignores it measures
  a part that is not there.
- **Use the vendor drawing and photo, not the asymmetry.** For the 2P the mouth
  is on the *shallower* side of the pad row, for the 4P WJ128V on the *deeper*
  one — the family has no reliable asymmetry rule, which is why this module
  refuses to guess.
- **`KICRAFT_ROUTING_CONFIG` gives a throwaway knob set** (pipeline, reasoning,
  temperature, caps) without touching production.
- **Legacy parks are not always user questions**: the wiring deficit park renders
  as "awaiting a clarifying answer", but it is the reconcile-budget-exhausted
  shape; `stage_driver run` carries `auto_default_questions=True`, so a park is a
  real failure of the parts list, not of the answer policy.

---

## 6. Recovering this session's evidence

- Live run: `~/.kicraft/projects/42/896/` (`generated/24V_TO_5V_CONVERTER/
  24V_TO_5V_CONVERTER_fab_20260923.zip` + `.receipt.json`, `.kicraft/build.log`).
- A/B: logs in `/tmp/ab/` (throwaway) — reproduce with the commands in §3;
  the results table above is also in the plan's session log.
- Replays: the beacon/brief states under `/tmp/ab/ws/curR-b3/.kicraft/state.json`
  (throwaway; the *commands* are what matters).
- Budget: $137.01 spent against a $250 total ceiling, $19.22 of $20 left today
  (A/B + replays + the live run cost ≈ $0.22 together).
