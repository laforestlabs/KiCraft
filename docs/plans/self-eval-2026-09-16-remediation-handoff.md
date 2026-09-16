# Self-eval remediation — session handoff (2026-09-16)

**Purpose of this document:** hand the state of the self-eval remediation work to the next
session. It records what was completed, the evidence for it, what is broken or unfinished,
and the exact commands to continue. It does not replace
`docs/plans/self-eval-2026-09-15-remediation-plan.md`, which remains the authoritative plan
(milestones 1–5, ranked GAP list, frozen campaign reference).

## 0. Continuation (2026-09-16 evening)

**The worktree is now committed and pushed.** The uncommitted tail described in §1 became:

- `adb7597` — design-acceptance replay wiring, the provider-envelope `oneOf`→`anyOf` rewrite,
  the pipeline-authored-bank guard, the B_S-1WR3/AMS1117 operating windows, the refreshed
  fixtures, and six new tests.
- `a1df360` — `nicegui>=3.17,<4` (the sporadic prune-time HTTP 500 fix plus
  GHSA-p92q-2755-mhgh). All of it is on `origin/simplify/bom-wiring-pipeline`.

**Corpus status.** `--reference-replay` (mock transcript, no provider spend) now reproduces
**31/34 rows** (was 20/34 when §4 was written), and exits 0:

- 30 rows committed at the 20:21 worktree state, plus `rs485-terminal` after the fix below.
- `buck-3a` refuses — the recorded 5 V/TPS5430 `specification_conflict` requires it to refuse;
  that is acceptance, not a defect.
- `speaker-crossover` and `usb-pd-trigger` remain the two recorded supply blocks.
- `--references` now reports **only those two blocks** and nothing else (previously
  `speaker-crossover` also carried four structural diagnostics).

**Two repairs in this continuation.**

1. `speaker-crossover` was out of contract shape: no `deferred_obligations`, and it carried
   five deferred board-only obligations inside `obligations`. Trimmed `obligations` to the five
   reference obligations, added the seven contract-declared deferred ids, and declared the
   `crossover_hz` range the now-unbounded response check demands (`[2250, 2750]` Hz, the
   ±10 % first-order window around the recorded 2.50 kHz target, documented in the row's
   assumptions).
2. `rs485-terminal` refused at BOM with `missing-requirement-implementation=['field_regulator']`.
   The cause was an identity mismatch, not row data: `_curated_part_indexes` resolves MPN
   `AMS1117-5.0` to the easyeda bundle `ams1117-5v0-fixed` (the user-wide fetch cache), so
   `_normalize_curated_group_identities` rewrites the group to that bundle's symbol/footprint —
   which no longer equalled the reviewed record's `Regulator_Linear:AMS1117-5.0` /
   `Package_TO_SOT_SMD:SOT-223-3_TabPin2` (kicad-standard) pair, and `physical_inventory_record`
   classifies only an exact `(mpn, symbol, footprint)` match. Fixed by vendoring the bundle to
   `kicraft/parts_library/ams1117-5v0-fixed/` (vendored tier 2 outranks the user-wide cache, so
   the identity is portable across machines) and naming that same pair in the reviewed record.

**Still open — unchanged from §4–§6.** The two supply blocks remain policy blocks; milestone 4
(routing / fine-pitch escape / copper custody) still needs the operator's (a)/(b) decision, and
milestone 5 (three paid campaigns + release) still needs approval and spend. §5 item 2
("wire replay into reference acceptance") is done — `verify_reference_replay` and the
`--reference-rows` / `--reference-replay` CLI modes are the landing described there.

## 1. Deployment state (changed this session)

The production box now runs the improved code:

| Service | PID | Started (UTC) | Verification |
|---|---|---|---|
| web (`python -m kicraft.server.web`) | 969462 | 2026-09-16 17:20:20 | `http://127.0.0.1:8080/` → 200; `https://kicraft.io/` → 200 |
| build worker (`python -m kicraft.server.build_worker`) | 969529 | 2026-09-16 17:20:23 | log ends `[build-worker] ready (max 2 concurrent build(s))` |

Restarted with the canonical path: `./deploy/deploy-production.sh` (restarts both services and
verifies health; it does **not** `git pull`, so it loaded the current worktree).
Per operator instruction the 34-brief design canary was **not** run for this restart; the
documented command remains `./deploy/verify-design-canary.sh` when a real-provider gate is wanted.

**The worktree is NOT committed.** `git status` shows ~97 changed/added paths. The running
services load this worktree (editable install), so a future `git pull` on this box would
conflict with or discard it — commit or review before any pull.

> Superseded by §0: that tail is now committed and pushed as `adb7597` and `a1df360`, so a
> `git pull` is safe again.

## 2. Why the restart was justified (measurable improvements)

1. **A real production regression was repaired**: `~/.kicraft/jlcparts/cache.sqlite3` had been
   replaced by a pruned 2,104-row copy (the full catalogue is 633,250 rows). Every sourcing
   decision made while that copy was live was a false negative. Restored; the pruned copy is
   kept as `cache.sqlite3.pruned-20260916-analyst`, the pre-incident copy as
   `cache.sqlite3.before-20260916-refresh`.
2. **Crashes on live code paths were removed.** Examples that would have hit real designs:
   `NameError: pi` in the RC-filter artefact calculation (no `math` import), `NameError:
   SimpleNamespace` in the constant-current port derivation, a missing `except` that made
   `electrical_artifact_evidence` un-importable, a missing `return` that made every fabrication
   export verify as absent, `NameError: inventory`/`_count` in artefact fact extraction, and
   `str(GetFPID())` comparisons that silently unclassified *every* BOM part on a saved board.
3. **Wrong boards are now refused rather than reported as finished.** On the frozen campaign the
   new evaluator fails 9 of the 13 built designs on their own recorded evidence (e.g. #1
   `bnc-count`/`trim-pot`, #4 `physical-parts`, #29 `jacks`, #32 `outline`/`led-placement`) — the
   same defect set the plan documents, now derived from artefacts instead of asserted.
4. **Fabrication can no longer be claimed without evidence**: build rc0 + a verified
   Gerber/drill archive + structured ERC/DRC verdicts (new `.kicraft/build_gate.json`) are
   required before `fabrication: pass`, and software fulfilment requires *every* mandatory
   obligation to pass.

## 3. What was completed

### 3.1 Acceptance contracts (`kicraft/eval/acceptance_contracts.py`)
- All 34 briefs carry mandatory obligations (functional, quantitative, physical, mechanical),
  reviewed against the **original** brief; `validate_contracts()` returns `[]`.
- Per-brief `feasibility`, `sourceability`, `operating_limits`, `assumptions` and a
  `substitutions` ledger. Only one substitution in the corpus exists — the user-approved 5 V
  device change for #15 — and its `consent` record names the approver, date and scope. #4 keeps
  `sourcing_blocked`; #15 keeps its documented `specification_conflict`.
- Obligation reviews fixed checks that could never pass or could pass wrongly: package tokens
  (`LQFP-48`/`QFN-56`) matched against reviewed package text; unbounded numeric checks now
  require the value to fall inside a *declared* range (this is what catches #4's 1 000× inductor
  error); the class counter records zeros so "no MCU fitted" is a real observation; the class
  vocabulary covers every declared class.

### 3.2 Evaluator and harness (`kicraft/eval/design_acceptance.py`, `artifact_evidence.py`, …)
- Reference fixtures validate only the obligations a pre-build reference can prove; board-only
  obligations must be listed exactly in `deferred_obligations`. New entry points:
  `build_reference_fixture`, `verify_reference_fixture(s)`, `verify_reference_rows`, and the CLI
  modes `--references` / `--reference-rows`.
- A corpus regression test asserts all 34 briefs have rows, that every non-blocked row validates,
  and that each blocked row reports its recorded reason.
- Artefact facts: reviewed package *and* footprint are published; `board_loaded` is real;
  nickname-less board footprints resolve; verified export archives returned; gate verdicts
  published into `facts["gates"]`; applicability derived from hardware + contract.

### 3.3 Reviewed parts and library (`kicraft/design/part_identity.py`, `kicraft/parts_library/**`)
- Registry grew to 80+ reviewed records (MAX31855, MAX485, RP2040, SN65HVD230, ESP32-S3/C3,
  MCP23017, PCA9685, ADS1115, DRV8833, relay, ULN2003, MCP6001, CH224K, 0402 R/C/L and crystal
  stock pairs, plus the reference-BOM order codes).
- Encoder bundle vendored (`ec11e15244g1`); Arduino shield stacking sockets carry
  `stacking-header`; zero-padded Phoenix terminal footprints accepted; two-row header symbols
  (`Conn_02xNN_Odd_Even`) accepted; Keystone binding-post symbol repaired (graphics were outside
  the unit, so it had no pins); malformed 3D-model references fixed for 20 bundles with an
  explicit `three_d_model: "unavailable"` marker where no body exists.
- **ADS1115 `address_strap` parameter** (gnd/vdd/sda/scl) makes two ADCs on one bus expressible —
  #24 now commits.

### 3.4 Gates and shared contracts
- §9.38 accepts `input_voltage_*`/`input_*` aliases and no longer flags reviewed parts with no
  voltage-input character; §9.39 accepts a record's own input-ish port names; §9.41 requires
  non-empty feedback metadata; §9.42 is scoped by ownership (recipe-owned proved at BOM commit,
  model-owned at the wiring commit, both stages emitting the same error vocabulary).
- Router wall-clock deadlines are treated as budgets, not verdicts: #10 now reaches the honest
  rc7 verify verdict instead of a false rc6 "engine aborted".

### 3.5 Test tree
Every block verified this session is green: work-units 95, recipes 173, lowering 85, stage-driver
131, prompt examples 60, self-eval + web 66, stage-cli 33, acceptance 21, part identity 45,
electrical invariants 19, geometry 20, architecture intent 33, electrical review 22, sourcing gate
44, parts library, loadtest 44, post-wiring review 7, undefined-name check 1, web/wiring tail 207.
A repo-wide run reached 75 % with zero failures before being reaped; the tail was verified
separately.

### 3.6 New analysis artefacts (under `logs/self_eval/remediation_20260916T013532Z/`)
- `frozen_baseline_obligation_verdicts.json` — per-brief honest verdicts for the frozen campaign
  (13 built, 9 failing a mandatory obligation).
- `reference_realization_stages.json` — realization stage per brief; a stage is claimed only from a
  retained artefact (`routed`/`artifact_fulfilled` are false everywhere).
- `reference_replay_probe.json` equivalents: `reference_replay_all.py` +
  `reference_replay_verdicts.json` — replays each row's *own stored inputs* through the real
  five-stage chain (merges verdicts per slug).
- `reference_routing_probe.json` — milestone-4 entry probe: a reference row commits all five
  stages and reaches ERC-clean synthesis; the build then needs the production `.env`
  (`KICRAFT_KICAD_ROUTING_TOOLS_*`) and a serialized, wall-clock-sensitive router run.

## 4. Reference corpus status — the main open work

`tests/fixtures/reference_inputs/` holds 34 rows; **32 validate** (`--references`), the other two
are the recorded supply blocks. But validating is not the same as reproducible: replaying each
row's stored inputs through the real chain shows

- **20 rows reproduce every stage** (committed).
- **11 refused**, each with a named cause:
  - architecture payload predates the current contract (6): `audio-jack-buffer` (provider schema),
    `esp32-s3-sensor`, `lora-node`, `nrf52-beacon`, `rs485-terminal`, `round-led-ring`
    (block-sheet/functional-connection defects).
  - `usb-c-full-breakout`: wiring payload never assigns the USB-C receptacle's own pads.
  - `thermocouple-amp`: §9.42 `needs exactly one identity-matched BOM component with resolved pin
    inventory` (its missing wiring coverage was already repaired).
  - `servo-driver-16`: BOM leaves a declared sheet (`SERVO`) empty.
  - `usb-a-power-splitter`: declares `port1/port2/switch1/switch2` interfaces with no owning group.
  - `buck-3a`: **correctly** refused by the documented 5 V/TPS5430 conflict — leave it.
- **1 unreplayable**: `rounded-c3-devboard` ships no `bom_candidate`/`wiring_candidate`.
- **2 supply-blocked**: `speaker-crossover` (no air-core inductor in the catalogue; policy forbids
  substitutes), `usb-pd-trigger` (CH224K C970725: 7 051 assembly stock, 0 retail, min buy 1).

Two systematic row defects were found and fixed in passing (use these patterns for the rest):
1. **Unreviewed order codes in BOM groups** — e.g. `gpio-expander` asserted `ERJ-3EKF1002V`
   (not stocked); it now names the reviewed in-stock `FRC0603F1002TS` with the substitution
   recorded in `bom.substitutions`.
2. **Wiring payloads that omit decoupling passives** — `gpio-expander` and `thermocouple-amp`
   both failed net coverage on unassigned capacitor pins; every pin of every BOM part must be
   assigned (or declared no-connect).

## 5. Next steps, in order

1. **Finish the corpus repairs** (section 4 list), then re-run the replay:
   ```bash
   .venv/bin/python logs/self_eval/remediation_20260916T013532Z/reference_replay_all.py
   .venv/bin/python logs/self_eval/remediation_20260916T013532Z/reference_replay_all.py <slug>   # focused
   ```
2. **Wire replay into reference acceptance** so a row only counts as realized when its own inputs
   reproduce the boundary (the failure mode this session caught: a row claiming a boundary its
   stored wiring could not reach). Natural home: next to `--references`.
3. **Routing (milestone 4)** — decision needed:
   - (a) route the reproducible set first (each run ≈ 10 min on this 2-core box, one at a time,
     and it must not overlap the live web/worker services), or
   - (b) first resolve the fine-pitch escape defect: the build path has **no** signal fanout step,
     so #6's 0.5 mm-class USB-C pads and #10's QFN-56 pads have no legal escape. A dog-bone
     fanout was implemented and measured DRC-clean, but as wired it made the parent route fail
     harder because KRT's issue-#220 strip removes pre-existing GND copper from the output and
     trips KiCraft's copper-custody gate. Options (a)–(d) for that custody interaction are in the
     routing agent's report (`logs/self_eval/.../fanout_failure_evidence/`).
4. **Remaining verification items**: `_catalog`/lowerer interface-claim completeness;
   #12's `D1/R9` singleton repair routing; and the four architecture-payload rewrites above.
5. **Release (milestone 5)**: three fresh 34-brief campaigns (paid, hours) and deployment — gated
   by the two supply blocks and by operator approval. `reference_realization_stages.json` and
   `frozen_baseline_obligation_verdicts.json` are the honest measurement baselines to compare
   against; do not deploy off the design-only canary.

## 6. Decisions the operator still owns

- Routing campaign timing vs. fixing the fine-pitch escape / copper-custody interaction.
- The two supply blocks: the strict sourcing policy is what blocks #4 (air-core inductor) and #5
  (retail-dry but assembly-stocked CH224K). Relaxing the dual-inventory rule would unblock #5;
  nothing in the pipeline changed that rule.
- Approval to spend on the three release campaigns.

## 7. Environment cautions

- **2-core box, production**: routing is serialized and wall-clock-sensitive; do not run a
  campaign alongside the live services. `.env` (mode 600) supplies provider keys, caps, and the
  pinned KiCadRoutingTools path (`/home/kicraft/KiCadRoutingTools` @ `3ceb7737`); an unpackaged
  shell will not have it (that is why the milestone-4 probe stopped at
  `kicad_routing_tools_path is unset`).
- **Never write to `~/.kicraft`** except through the documented tooling; the catalogue there is
  production data (its accidental replacement is recorded above).
- The reference-replay harness (`reference_replay_all.py`) is scratch tooling living in the log
  directory; it now merges verdicts per slug rather than overwriting, and it costs nothing
  (mock transcript, no provider calls).
