# USB-PD / ESP32-S3 / HUB75 travel handoff — 2026-11-06

## Stop verdict

The user superseded the continuous campaign with a request to stop within 30 minutes and write this handoff. That request was recorded at **20:14:08 UTC**, deadline **20:44:08 UTC**.

**The requested fabrication-ready GUI result is NOT complete.** There are **zero qualified frozen full-board fabrication-tail passes**. No production restart/deployment or fresh GUI submission was performed during the wrap-up. Do not describe a partial preview, a clean unit suite, or an interrupted leaf replay as a completed board.

The application patch is retained in the working tree, with targeted verification and immutable failed diagnostic workspaces. Resume the campaign from this evidence rather than repeating initial investigation or asking the user for credentials/information already available locally.

## Locations and original requirement

- Production checkout: `/home/kicraft/KiCraft`, serving `https://kicraft.io`.
- Session evidence root, called `S` below: `/home/kicraft/.kicraft/debug/hub75-gui-20261106T042027Z`.
- Original approved plan: `local://hub75-gui-convergence-plan.md` in the originating session. This document preserves the operational acceptance requirements without depending on that URI surviving a new session.
- Exact immutable brief:

> make a board with USB C PD power input to an ESP32-S3-WROOM-1-N16R8 that drives a 64x32 p2.5 HUB75 display. Include header outputs for driving addressable LED string and a speaker.

Every fresh acceptance attempt must submit that literal brief through the deployed production GUI with **Auto default questions? checked**. Do not append hints, answer around a broken defaulting path, substitute an MCU ordering code, or accept a CLI replay as GUI acceptance.

Read the matching `kicraft`, `kicraft-debug`, `kicraft-investigate`, and `verify` skills before resuming their respective work. Repository operating instructions identify this as the live production box, not a disposable development host.

## Account and production state

A new account was created through the supported product flow after the previous credential failed. No authentication storage was forged and no existing account password was reset.

- Account/user ID: **45**.
- Email: `hub75-gui-20261106042445@kicraft-test.dev`.
- Private credential file: `S/gui-account-cred.json`, mode 600. Read values privately into normal login actions; never print the password or put it in command arguments.
- Supported tester account, normal user role, Max allowance of 20 projects/week. Do not alter quota rows.
- Existing campaign GUI project: **1155 / KC-NBTT32**, `https://kicraft.io/?project=1155`.
- Project directory: `/home/kicraft/.kicraft/projects/45/1155`.
- This GUI attempt failed at architecture on older, unrestarted service code; intent and functional specification committed. Its auto-default flag is 1, attempt generation 0, recorded cost $0.013883666. Preserve it as a failure, not the final deliverable.

At the wrap-up production database read: project statuses were failed=213 / ok=72; all 83 build jobs were done. There were no active production designs/builds. Recheck before restarting or submitting anything.

Services are detached processes managed by deployment scripts, not active systemd services. **They were not restarted in this campaign.** Working-tree edits are present on the production filesystem; newly spawned CLI subprocesses can read current source even before a service restart. A loaded old web/worker process is not proof that all child execution uses old code.

Last observed effective provider settings, unchanged by this session: luna / `openai/gpt-6-luna`, 4096 call tokens, $0.60/project, reasoning disabled with 4096 reasoning guard, $20/day, $250 lifetime, kill switch false. Do not clear accounting, uncertain reservations, or quotas.

## Financial evidence

The read-only ledger total at wrap-up was **$154.95422338948** versus session start **$154.77605591448**: ledger delta **$0.178167475**. Native CAD/routing diagnostics did not call the provider.

Outstanding records were retained unchanged:

- `strict_budget_exposure`: reserved 13 / $0.2852716; uncertain 2 / $0.0889552.
- `provider_exposure`: uncertain 19 / $0.217543388.

These are distinct tables, not an instruction to add or clear them manually. Let the existing budget policy calculate headroom. Recheck current totals before paid work. Assistant subscription use is separate from this application ledger.

## Electrical/compiler/library work already completed in source

The current source supports a real controller rather than a nominal block diagram. These are implemented capabilities and earlier diagnostic proofs, not a final routed-board approval:

- Exact **ESP32-S3-WROOM-1-N16R8** identity, 16 MB flash / 8 MB PSRAM; no N8R8 substitution. Reserved module pads 28/29/30 remain unused. Native USB uses GPIO19/20, module pads 13/14, through 22-ohm series resistors.
- Separate 5 V programming-USB VBUS is not the PD input or a source powering the controller. PD connector is power-only; CH224K D+/D- handling does not short programming data.
- Fixed-9-V `ch224k-pd-trigger@2`: real CFG support, 6.8 kOhm selection, CFG2/3 NC, local D+/D- short for the PD-only input, VDD feed/decoupling and voltage-sense support. Raw-path support is qualified for 4–22 V; it is not passive USB power mislabeled as PD.
- TPS26631RGER controlled input path with actual IN/OUT/GND/EP pins, UVLO/OVP, ILIM, controlled startup and power-good/enable support. Nominal path allocation 1.8 A; no invented external reverse FET or programming-VBUS backfeed. Component/energy/startup assumptions are in the retained electrical evidence.
- AP63205 5 V and AP63203 3.3 V reviewed constructions, actual FXL0630-4R7-M inductors, correctly rated input/output/HF/bootstrap capacitors and device input limits. Output-capacitor placement/current paths originate at the **inductor output**, not feedback.
- HUB75 13-signal 5 V HCT buffering, safe OE/enable transistor network, input pulldowns, OE pullup and decoupling. Old recipe versions must regenerate; do not patch an old accepted state by hand.
- Addressable LED output is a real 3-pin header with 5 V buffering and series resistor, preserving unspecified-load intent with an explicit **0.25 A default allocation**. Explicit user loads remain authoritative.
- PAM8302AADCR/C112137 BTL speaker driver with actual support/filter/shutdown/decoupling/output network and two-pin floating speaker header. Default load is **8 ohms, speaker rated at least 2 W**; that is a speaker rating, not a claim of 2 W delivered audio. Firmware PWM/filter and short floating-lead constraints remain material.
- Default panel power is external 5 V; controller drives the HUB75 interface. A reference panel's 5 V / 2.5 A data is not a measured specification for the user's panel.
- Typed allocations retain approximately 0.97 A on 5 V and 1 A on 3.3 V, with conservative conversion/path allowances and honest PD headroom. Unknown/series-path capacities are not silently reduced to make the design fit.
- Architecture objects are deep-copied across completion/retry boundaries; ordinary automatic-default questions do not park the run, while actual upstream reconciliation remains required.
- Real source identities and provenance are checked. The unpruned 3,732,372-entry catalog was atomically installed, with original database retained at `S/catalog-before-fix.sqlite3`. No stock was fabricated. Some non-passive availability remains unverified/zero in snapshots; this is not an assembly/procurement guarantee.

Primary manufacturer and JLC references, downloaded data and conditional thermal calculations are retained under `S`, notably `converter-thermal-reference-and-estimates.json` and `jlcpcb-rendered-manufacturing-capabilities.json`. AP reference theta-JA/efficiency estimates are conditional, not measured board thermal qualification or an automatic 2 A guarantee.

## Physical pipeline repairs already in source

- Pinned external KiCadRoutingTools remains unchanged at `/home/kicraft/KiCadRoutingTools`, commit `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`, interpreter `/home/kicraft/krt-venv/bin/python`.
- Strict native input-copper custody is preserved through every router pass, including temporary no-rip locks and restoration of authored lock state. No blanket dead-end or GND copper-loss exemption.
- Failed parent outputs cannot be promoted as fresh successful fabrication. A partial preview is not an accepted routed parent. Post-silkscreen source/final provenance is retained.
- Native local-current auditing includes finite pad contacts, layers/PTH/vias, exact native pad/drill boundaries, real physical route length and cumulative exposed narrow escapes. Tiny tangencies or 40-micrometre nubs cannot masquerade as a qualified current path. The 1.5 mm exposed-neck limit is unchanged.
- Canonical thermal-owner metadata must match actual architecture/BOM/wiring at emission, prelayout and final qualification. There are three thermal owners, 43 local contracts and 19 explicit plane-backed ground-return contracts in synthesis12.
- Mandatory two-layer stack is 1.600 mm, with 0.070 mm / 2 oz outer copper, 0.010 mm masks and 1.440 mm dielectric. Actual track/space floor 0.170 mm; component PTH annulus at least 0.254 mm; drilled-hole edge spacing at least 0.200 mm. No gate was lowered to accept narrower copper.
- USB-C shell and IDC native lands were corrected without moving holes/pins or changing identities; PAM/CH224K courtyards were corrected against actual native pads and manifests rehashed.
- Ground finishing retains input copper and requires real F/B filled regions, local connected area and qualified nearby vias. Via-in-pad requires appropriate resin-fill/cap/planarization manufacturing instructions, not a claim that an ordinary cheap 2-layer order automatically supplies it.
- Placement now uses compact world geometry, cached compatibility, MRV/forward checking and ownership-family ordering while retaining numeric candidates, keepouts, fixed obstacles and hard legality. MCU/PAM/CH and converter bypasses use actual positive supply pins and explicit plane-backed ground ownership rather than impossible centre-distance constraints to buried EP lands.

## Latest repair set and exact causes

### Routing width and ground ordering

`kicraft/autoplacer/kicad_routing_tools.py` runs width-homogeneous passes using the pinned router's real parser/rules/pattern expansion. Non-ground groups run widest first; configured plane-ground groups run last at their original width. Every preceding trace/via, actual rule sidecar and shared whole-route deadline is retained. Ambiguous suffix/pattern selection fails before routing.

Reason: pinned cleanup otherwise narrowed power segments to Default width, and ground-first routing created genuine output-bank detours. A native ground-last diagnostic completed in 13.68 s with all 18 5 V current checks passing and DRC zero errors/unconnected. This is not three full-board reliability passes.

Evidence: `S/ground-last-native-5v`, `S/native-ground-last-bank-audit.json`.

### Fine-pitch thermal ground bonds

`kicraft/autoplacer/brain/gnd_pour.py` now defaults GND-zone clearance to the fabrication floor, clamps smaller configured values, and preserves explicitly larger values. Power-zone policy and thermal gates are unchanged.

Native retained P6 geometry proved the old 0.30 mm zone default excluded TPS26631 GND pins 8/11: 116/118 GND pad centres covered. At the actual unchanged 0.17 mm project/netclass fabrication clearance, all 118 were covered. U8.8 had 0.1392 square mm positive pad/fill overlap, including with the optional higher-priority power pour. The original R1 finished board was overwritten, so do not claim its exact bytes were rechecked.

A native regression fixture initially failed because its via was 3.0717 mm from pin 11, outside the real 3 mm limit. The fixture gained a real symmetric via at 2.0088 mm; no gate/assertion was relaxed. Final targeted tests pass.

### Real interface access, not merely net names

Files: `signal_escape.py`, `leaf_routing.py`, `leaf_acceptance.py`, `solve_subcircuits.py`, `tests/test_signal_escape.py`.

Independent native review established no contact loss during composition across 208 signal-pad identities. MCU U4.19 was already bare; DISPLAY U6.4 had two 0.2 mm x 0.04 mm nubs and R22.1 was bare before composition. Later 0.5 mm +3V3/GND copper caged their exits. A centre-point-only claim that U6.4 was bare was incorrect: finite native shapes really touch.

The guarded planner now reserves actual cross-leaf SMT signals, including wide module pads, before GND escapes and power ties. All original native pad/copper/hole/mask/keepout/outline guards remain. Canonical `power_nets` are excluded, not just voltage-looking names. This matters: a first diagnostic incorrectly generated signal-width stubs on aliases such as V5/PD_PROTECTED. The typed exclusion is fixed and tested; final pre-escape counts were MCU18/18, DISPLAY27/27 and REG5 1/1, all with zero blocked reservations.

Every disconnected external signal component now needs an actual connected via/PTH landing, including singleton ports with no local DRC open. One via elsewhere on the same net is insufficient. Native connected-track vectors expose vias through `PCB_TRACK` wrappers; recognition uses `item.Type() == pcbnew.PCB_VIA_T`, not Python `isinstance`. Trivial `no_internal_nets` rounds no longer bypass structured acceptance. Test fixtures provide the real extraction's required `external_net_names` field rather than weakening production with a fallback.

### Size-reduction acceptance and provenance

Files: `leaf_size_reduction.py`, `solve_subcircuits.py`, `tests/test_content_canvas.py`.

P6 POWER INPUT round15 was reduced to a board with 395 traces / 40 vias and actual internal CFG1/VDD opens. Reduction accepted only lightweight validation; the original 357/35 clean round record was then persisted against the overwritten reduced snapshot. The report parser was not the cause.

Reduction now applies the same structured leaf gates as normal rounds, records its fresh verdict, restores canonical/numbered bytes on rejection, and replaces the winning object in `all_rounds` when a reduced round is accepted. Independent review found no critical scoped defects. An actual accepted MCU reduction produced matching canonical, numbered and persisted counts: **171 traces / 42 vias**, one reduction pass. Evidence: `S/native-mcu-reduced-round-provenance.json`.

## Verification at this stop

- **217 passed in 20.65 s**, 15 focused suites, final frozen application source. Command is below. Earlier 160/164 and 215/217 outcomes were intermediate failures, fixed before the final pass.
- Earlier broad pre-latest-repair verification: 1970 passed across 38 distinct suites in 230.60 s, one KiCad SWIG deprecation warning. Do not attribute that older broad run to the final latest patch.
- Actual MCU first-full-phase-seed native leaf completed in **85.30 s**, accepted with one size reduction, 18/18 escapes and no blocked reservation. The final all-interface audit was then exercised against that real routed board: no unlanded components, strict accepted true. Evidence: `S/native-mcu-final-interface-audit.json`.
- Earlier native MCU/speaker plane-backed-bypass diagnostics completed in 50.782 / 78.028 s with native ground audits reporting zero errors. These precede the latest interface changes and are not new full-board passes.
- Final bounded REG5 and DISPLAY CLI runs each reached the outer **180 s diagnostic limit**, not a completed solve. Their three KRT pass outputs exist; do not assume the outer deadline identifies a router search timeout. REG5 had progressed through numbered rounds; DISPLAY retained `leaf_routed.kicad_pcb.pre_signal_repair`.
- Retained final rendered DRC reports show zero unconnected pads: REG5 one via-dangling warning; DISPLAY only via-dangling warnings in the report read. These rendered reports are not a guarantee that every later interrupted mutation matches them.
- Read-only audit of the actual final retained boards: DISPLAY's two local current contracts pass; REG5's 18 contracts still include **`local contract U2.6/C31.1: no qualified current-carrying copper path`**. This bootstrap path is a concrete unresolved problem. Evidence: `S/travel-final-native-current-audit.json`.
- No full fabrication package has passed final native electrical, ground, thermal, manufacturing, freshness and GUI-download acceptance.

Final focused command:

```bash
.venv/bin/python -m pytest -q tests/test_kicad_routing_tools.py tests/test_power_pour.py tests/test_signal_escape.py tests/test_content_canvas.py tests/test_auto_pin_acceptance.py tests/test_leaf_acceptance_unconnected.py tests/test_trivial_leaf_stamp.py tests/test_solve_subcircuits_layout_persistence.py tests/test_leaf_geometry.py tests/test_leaf_best_effort.py tests/test_autoexperiment_leaf_recovery.py tests/test_leaf_place_quality_gate.py tests/test_leaf_acceptance_courtyard.py tests/test_leaf_acceptance_area.py tests/test_routing_board.py
```

## Failed full physical attempts — preserve, never promote as success

All are under `S`.

| Workspace | Actual outcome |
| --- | --- |
| `physical-probe-1` | rc0/export with DRC0 errors, but all parent rounds lost 521 GND traces / 117 vias; stale/partial promotion and old physical defects made it unqualified. |
| `physical-replay-2`, `physical-replay-3` | Superseded and cancelled; retained. |
| `physical-replay-4` | Cancelled after 18,505 s in placement; source ownership/CSP defects subsequently repaired. |
| `physical-replay-5` | Cancelled after more than 4,687 s; actual missing leaves were MCU/speaker, not the converters. Plane-backed bypass/courtyard fixes subsequently produced native accepted leaves. |
| `physical-replay-6` | Completed rc6 after 3,674.26 s. All five logical stages/109 parts/355 wired pins/ERC0, seven of eight leaves pinned, no qualified parent. Reg5 current paths failed; parent R1 reached finishing then failed U8.8 ground bond, R2/R3 timed out in signal pass at 404.615/403.932 s. Fresh partial preview only. |

P6 root: `S/physical-replay-6/generated/ESP32_HUB75_CONTROLLER`.

P6 leaf identifiers:

- REG5: `026565ee-ebfb-4a4b-a84e-cbafeb7273c8__5172e1e64b`.
- MCU: `462053c9-222b-4588-ae3c-d348bfcd8599__7d8752b523`.
- DISPLAY: `9d8cc3f8-1504-4927-9d45-47fb744cc705__1797988f50`.
- POWER: `dad04265-9037-456c-ad76-f141c9379a29__7ca713d5b1`.
- Parent: `subcircuit__8a5edab282`.

Selected MCU/display snapshots matched canonical bytes during review; every pad across their 10/23 footprints matched the exact rigid transform. Do not reopen the disproven composition-registration hypothesis without new evidence.

## Resume in this order

1. Read this handoff and skills, inspect current source fingerprint and existing session evidence. Do not reset/pull/stash over this patch. Do not edit failed state, candidate JSON, PCB geometry or target routing manually.
2. Diagnose the retained **U2.6/C31.1 bootstrap current-path failure** from actual native copper and canonical width/length contract. Distinguish real insufficient contact/neck/route from an auditor error with native geometry. Fix the producing generic source and earliest detector; do not relax the current gate.
3. Inspect final REG5/DISPLAY interrupted run logs, pass logs, numbered boards and repair stages to identify what consumed the bounded run. At least some routing completed; do not blindly increase router timeouts or equate a 180 s outer kill with unroutability. Source aliases are now correctly typed as power.
4. Prove normal REG5, DISPLAY and POWER leaf acceptance with actual strict current/landing/DRC/provenance evidence. Recheck accepted reduced rounds, not an old clean report against newer bytes. Verify all affected callsites/tests and independent review of new fixes. No tests/builds while source owners edit.
5. Cold-copy complete **`S/synthesis-probe-12`** into a new `S/physical-replay-7` (it does not yet exist). Keep all generated sidecars and state. Do not use interrupted boards or hand-edit state. If a new fix changes emitted schematic/asset metadata, regenerate through synthesis first; ordinary route-only changes do not justify new paid architecture calls.
6. Run full frozen quality-good seed0 build tails without an arbitrary short outer diagnostic deadline. Require **three completed qualified full tails** before a stochastic routing-reliability claim. Currently zero. Inspect every actual ERC/DRC, current path, landing, F/B ground/thermal proof, copper custody, selected/promoted identity, freshness and package content.
7. Independently audit final exact N16R8, programming USB, fixed PD power path/limits/startup, current headroom, all 13 HUB75 buffers and safe OE, LED output, BTL speaker, RF keepout and manufacturing process. Do not call CAD verification a fabricated-and-powered hardware test.
8. Recheck all live project/build jobs and budget/disk headroom. Drain production, then use canonical `./deploy/deploy-production.sh`, not systemctl. Require local/public HTTP200 and a new worker-ready line. The unrelated 34-brief live canary is not a prerequisite.
9. Log in normally to account45. Create a **fresh project**, paste the exact literal brief and check Auto default questions. Verify stored brief/flag and all five committed stages plus actual fabrication. Open actual schematic/PCB GUI views, capture evidence, download the actual GUI ZIP and inspect its freshness/content. Only then deliver project link/code/package/defaults/evidence/spend/source counts as the completed result.

## Exact diagnostic entry points

The native CLI `--only` selector matches **sheet name, complete sheet filename, or instance path**, not a guessed filename stem. `REGULATOR_5V` alone failed before work; `REGULATOR_5V.kicad_sch` is valid.

`S/profile_placement.py` loads the production dotenv and executes the actual `kicraft/cli/solve_subcircuits.py`. First argument `-` disables profiling/watchdog; it is not a mock. Example on a fresh complete scratch copy:

```bash
.venv/bin/python "$S/profile_placement.py" - "$P/ESP32_HUB75_CONTROLLER.kicad_sch" --pcb "$P/ESP32_HUB75_CONTROLLER.kicad_pcb" --config "$P/ESP32_HUB75_CONTROLLER_autoplacer.json" --rounds 3 --seed 1654617007 --only REGULATOR_5V.kicad_sch --route --workers 1 --experiment-round 1 --json
```

Selecting one leaf changes its enumeration index. Correct full-first-phase bases: POWER1654615998, REG5**1654617007**, REG3**1654618016**, MCU**1654619025**, DISPLAY**1654620034**, LED1654621043, SPEAKER1654622052, USB1654623061. `S/physical5-native-representative-seeds.json` records the mapping. A default index0 diagnostic is not an identical full-run seed comparison.

Full replay helper, after a fresh complete copy:

```bash
.venv/bin/python "$S/physical_replay.py" --project "$S/physical-replay-7/generated/ESP32_HUB75_CONTROLLER" --quality good --seed 0 --no-archive
```

Use a named/background tool job with no short outer timeout for a full proof. Preserve failed workspaces and ensure at least 4 GiB beyond estimated writes. Last observed filesystem had about 20 GiB available.

Latest bounded directories:

- `S/native-mcu-plane-bypass`, `S/native-speaker-plane-bypass`: earlier accepted ground diagnostics.
- `S/native-interface-mcu-1`: accepted actual MCU source-run/provenance witness.
- `S/native-interface-display-1`: earlier 240 s interruption, before typed power-alias exclusion.
- `S/native-reg5-ground-last-2`: earlier 480 s interruption, before that exclusion.
- `S/native-reg5-travel-final`, `S/native-display-travel-final`: final 180 s interrupted cases, typed power aliases fixed.

`S/capture_parent_boards.py` is a session-only event-driven read-only observer for a future replay root. It captures changing `parent_routed.kicad_pcb` bytes and matching rule sidecars outside the replay root, deduplicated by SHA256. **It was not started or runtime-verified.** Use it only as diagnostic evidence capture, not an acceptance gate. It exists because parent R1 finished bytes were overwritten by later rounds. Capture the associated frozen config too. Other session helper scripts and failed data remain intentionally available for resumption, not installed as production services.

## Source custody and records

- Original tracked-source inventory: `S/source-start-sha256.json`, 2,220 paths; clean at session start.
- Exact original bytes: `S/baseline/`. Every changed original must match the initial hash before counting or rollback.
- Frozen earlier inventories: `S/frozen-source-before-physical4.json`, `...physical5.json`, `...physical6.json`.
- Final wrap-up fingerprint and exact session-only additions/removals are recorded in `S/source-frozen-travel.json` and `S/session-source-change-counts-travel.json`.
- Counts compare preserved original bytes with current source, **not HEAD diff**. CAD/manifest text and opaque STEP/model assets are separate categories. The new requested handoff itself is included as documentation.
- Main chronological/machine evidence: `S/run-log.md`, `S/findings.json`, `S/assumptions.json`, `S/summary.json`.
- Three correct real-provider architecture replay outcomes were obtained in replay18/19/20, each with two attempts and exact architecture; later physical-only metadata changes do not require paid replay repetition. Never treat those CLI provider diagnostics as the final deployed GUI run.

No final project, final GUI package, fabricated hardware, assembly stock guarantee or reliable full-board routing success is claimed by this handoff.
