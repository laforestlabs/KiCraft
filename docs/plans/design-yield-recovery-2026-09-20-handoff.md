# Design yield recovery — session handoff (2026-09-20)

**Purpose of this document.** Hand the state of the yield-recovery work to the next session. It
records what was measured, what the measurement *decided*, the exact target list with evidence,
the environment recipes that took several attempts to get right, and what must not be redone. It
does not replace `docs/plans/design-yield-recovery-2026-09-19-plan.md`, which remains the
authoritative plan — its `## §2 Results` section is the measurement this document interprets.

**Read order.** §1 (state in one screen) → §4 (the target list) → §5 (what to do next). §6–§8 are
reference. **Convention:** `§N` unqualified is a section of *this* document; a section of the
authoritative plan is written *the plan's §N*.

---

## 0. The state in one screen

| | value |
|---|---|
| Campaign | Move A of the plan, four arms, all run 2026-09-19/20 on this box |
| Spend | **$9.42** total (09-19 $6.98 + 09-20 $2.44); $8.78 arms, $0.05 smoke gates, **$0.30 void** |
| Wall | 11 h 21 min end to end (plan estimated ~10 h) |
| Headline | current tree **4/34 briefs fab-ready**; legacy pipeline **21/34**; pre-contract typed tree **13/34**; RECORD-preview **3/34** |
| Verdict | the loss is the contract layer (§2); **Move D pays ≈ 0** (§3); the binding wall is the BOM obligation/interface layer |
| Tree state | **no source change**: HEAD + the three inherited dirty files; the A4 scaffolding stayed in its own worktree |
| Production impact | web process never died (HTTP 200 throughout, 7 failed health checks over ~3.25 min at the memory plateau); **a SIGKILLed campaign orphaned a build that held a host build slot for 12 h 24 m**, halving the build worker's concurrency until it was reaped; `KICRAFT_BUILD_SLOTS` is now **1** (was 2) and a supervisor reclaims leaked slots (§6.6, §11) |
| Committed | `a879461` — the plan document with the measured `## §2 Results` |
| Next | attack `unsupported_lowerer_contract` (§5 Move 1), then the BOM obligation layer (§5 Move 3) |

---

## 1. What was measured

Four arms, run strictly one at a time (one build slot, 2 cores), every process launched exactly as
the plan's §2.3 requires (`env -i HOME=… PATH=… TERM=xterm PYTHONUNBUFFERED=1`), same designer
`openai/gpt-5.6-luna`, same KRT pin (`3ceb773`, VERSION `0.20.2`), same frozen 34-brief corpus.

| arm | tree | runs | committed | briefs ≥2 repeats | fab-ready | mean | $/committed | spend |
|---|---|---|---|---|---|---|---|---|
| **A1** | `de74af7` + inherited dirty source, 34×3 | 102 | 7 | 3 | **5 runs / 4 briefs** | 52.0 | 0.2951 | $2.07 |
| **A2** | `bc6a2f8` (08-25, pre-typed), 34×2 | 68 | 35 | 10 | **30 runs / 21 briefs** | 64.0 | — | $3.33 |
| **A3** | `b4b8be5` (09-15, typed pre-contract), 34×2 | 68 | 24 | 7 | **18 runs / 13 briefs** | 59.7 | 0.0569 | $1.36 |
| **A4** | today + scratch RECORD patch, 34×3 | 102 | 8 | 3 | **7 runs / 3 briefs** | 52.4 | 0.2528 | $2.02 |

Batch dirs are listed in §8. `source_unchanged=true` for A1/A3/A4; A2's batch carries the legacy
summary schema, which has no such field (§7).

---

## 2. What the measurement decided: the loss is the contract layer

| step | comparison | briefs fab-ready | attribution |
|---|---|---|---|
| typed design layer | A2 `bc6a2f8` → A3 `b4b8be5` | 21 → 13 of 34 | ~8 briefs |
| 09-16 contract push | A3 `b4b8be5` → A1 `de74af7` | 13 → 4 of 34 | ~9 briefs |

**A2's 21/34 reproduces the era's recorded 22–25/34** with the *same designer* as the other arms.
That kills two hypotheses at once: the model is not the constraint (Luna drives the pre-typed
pipeline to era yield), and neither is the box. The plan's §3 gate ("build the switch only if
A2 shows the legacy tree still delivers", threshold ≥15 fab-ready) is cleared with margin.

The regression set in the plan's §5 needs rebasing: it lists `rc-lowpass-bnc`, `esp32-dual-motor`,
`r2r-dac`, `fpc-breakout`, but at HEAD only `rc-lowpass-bnc`, `r2r-dac` and `fpc-breakout` commit
(`esp32-dual-motor` is 0/3 fab at HEAD, so it is a target, not a guard). Use the briefs that
actually commit at HEAD as the guard set, plus the $0 reference replay.

---

## 3. What the measurement decided: Move D pays ≈ 0, and the wall moved one stage later

A4 exercised the RECORD downgrade exactly as specified and it converted nothing:

- 12 of 102 runs carried an advisory; on **all 12** the RECORD-class codes were the *only*
  architecture diagnostics, so the downgrade is what let them pass the stage A1 killed them at.
- **All 12 then died at the BOM stage.** **Zero boards shipped carrying an advisory** — the plan's
  the plan's §4.4 acceptance criterion fails outright.
- The headline delta is variance: the two briefs that improved (`rc-lowpass-bnc` 1/3→3/3,
  `r2r-dac` 1/3→3/3) *committed*, so an advisory would have been persisted had one fired — none
  did, so no RECORD code fired and their code path is identical to A1's. `audio-jack-buffer`
  regressed the other way (2/3→0/3), also with no advisory.
- §9.33/§9.34 never fired on this corpus: that half of the RECORD list is unexercised, not verified.

This is the plan's own "≈ 0" branch, and its re-derivation instruction ("re-derive from A4's
per-board failure path") resolves cleanly: the wall is the **BOM obligation/interface layer**,
which the plan's §4.1 rules out of the RECORD class by design.

**Do not run the plan's §4 as specified.** Widening RECORD would repeat a measured null.

---

## 4. The target list (read this before planning any move)

### 4.1 Architecture refusals — measured, bundles expanded

Counts are **architecture-failing runs** whose `stage_done.schema_error` contains the code,
read from each run's `events.jsonl` (works across all three tree schemas; the summary's
`terminal_diagnostic` only exists on the current tree and hides members inside bundles).

| refusal code | A1 | A3 | A4 | added by |
|---|---|---|---|---|
| **`unsupported_lowerer_contract`** | **32** | **0** | **41** | 09-16 push |
| `conflicting_port_binding` | 24 | 10 | 21 | pre-existing |
| `unreviewed_exact_part` | 18 | 0 | 0 (downgraded) | 09-17/18 |
| `declared_signal_port_tied` | 12 | 0 | 9 | 09-17/18 |
| `unknown_part_refused` | 11 | 1 | 0 (downgraded) | pre-existing |
| `unknown_interface_port` | 6 | 5 | 4 | pre-existing |
| `unknown_supply_rail` | 5 | 0 | 2 | pre-existing |
| `unbound_required_port` | 5 | 1 | 2 | pre-existing |

Totals: A1 69 architecture-failing runs / 13 distinct codes; A3 25 / 7; A4 67 / 12.

**`unsupported_lowerer_contract` is the single highest-value target in the dataset**: it was
introduced by the 09-16 push, it fires on 32 of A1's 102 runs (46 % of A1's architecture deaths),
and it is **absent at A3** — the tree that delivers 13/34. The plan already prescribes its split
(the plan's §4.3 A, last row): *derive* the contacts when the draft's own signals name them
(the plan's §5 B-2), and
refuse only when the geometry is genuinely outside the family's range.

### 4.2 BOM walls — the stage that binds once architecture passes

| BOM death reason | A1 (23 deaths) | A3 (15) | A4 (25) |
|---|---|---|---|
| `physical-obligation-unfulfilled` | 13 | — | 10 |
| `declared-interface-unrealized` | 9 | — | 11 |
| `missing-requirement-implementation` | 6 | 10 | 4 |
| §9.42 `requirement physical/interface realization` | 2 | — | 6 |
| `model_authored_protected_identity` | 2 | 1 | 2 |

The most informative message in the campaign, verbatim:

```
work unit bom-r000 invalid: physical-obligation-unfulfilled=['status:status-led:
requires 1 real led, found 0; the unit emitted: resistor=Device:R mpn=270R | led=Device:LED mpn=green']
```

The model **did** emit the LED; the deterministic check cannot see it as a real part. And:

```
9.42 requirement physical/interface realization: 6 physical/interface realization contract(s) unproven
  E_PHYSICAL_REALIZATION 'rp2040': requires 1 reviewed 'crystal' physical part
```

Neither is "the model was wrong" — both are the compiler's own bookkeeping. That is the plan's
§5 B-2's
territory ("stop asking the model for what the compiler can derive"), whose precedent quadrupled
completion (1/34 → 4/34 briefs on 09-17).

### 4.3 What the 09-16 push contains

- `0a2e700` (2026-09-16 17:21) *"Make accepted boards provably correct: contracts, references,
  artefact evidence"* added **11** architecture refusal codes: `conflicting_reference_binding`,
  `conflicting_supply_binding`, `incomplete_standard_stacking_interface`,
  `invalid_standard_stacking_owner`, `invalid_standard_stacking_pinmap`,
  `reference_domain_not_zero_volt`, `unknown_reference_domain`, `unknown_reference_port`,
  `unknown_supply_port`, `unsupported_lowerer_contract`,
  `unsupported_standard_stacking_interface`; §9.x sections went 19 → 20 (§9.37 added).
- `0a2e700` → HEAD added three more (`declared_port_double_bound`, `declared_signal_port_tied`,
  `unreviewed_exact_part`) and removed `unsupported_supply_port`.
- 34 commits separate `b4b8be5` from HEAD; the yield fell 13/34 → 4/34 across them.

---

## 5. The next session, in order

**Move 1 (recommended). Land or revert the in-flight `unsupported_lowerer_contract` work, then
measure it.** The live tree already carries uncommitted work aimed at exactly this code (see §9):
a contact-derivation refactor (`_contact_nets` / `_contact_net_map`) and evidence enrichment
(`port_contract=`, `requirement_ports=`) for `lowerer_contract_diagnostic`. It was present in
*both* A1 and A4, so the baseline already includes it — which is why A1's 32 firings are what
remains *after* that partial work.
- Pre-registered expectation: A1's 69 architecture deaths drop by ≥15, and per-brief commit count
  rises by ≥3 briefs.
- Protocol: 34 briefs × 3 repeats, same `env -i` launch, same corpus; compare against
  `logs/self_eval/movea_a1_current_20260919T1431Z` (the frozen A1 baseline).
- Kill criterion: no rise in the per-brief median commit count ⇒ revert and record the null.
- First action, before spending: **read the diagnostic the model is actually getting.** A1's
  failure texts are in each run's `events.jsonl` (`schema_error`), and expanding them is one
  command (§8). Do not guess which sub-case the drafts hit.

**Move 2. `conflicting_port_binding` (24) + `declared_signal_port_tied` (12).** Note
`conflicting_port_binding` is pre-existing but fires 2.4× more often at HEAD than at A3 (24 vs 10),
so the regression is in prompt/contract shape (09-17 commits `618c433`, `09b4f5b` are in that
area), not only in the check.

**Move 3. The BOM obligation/interface layer (§5 B-2).** Make the realization checks see the part
the BOM emitted (identity resolution), and derive the obligation set from the reviewed family
instead of demanding classes it does not hold. This is the wall that still stands after Move 1.

**Move 4. §3's switch.** Gate cleared (A2 ≫ 15). It now has a measured environment prerequisite —
the legacy tree cannot talk to Luna at all under the production `.env` (§6.3 without the overrides
gives every call a `<1 s` "provider error" at $0.00). Production could serve 21/34 boards through
it; the cost is a maintained fork of an 08-25 tree.

**Do not:** run the plan's §4 (measured null), widen RECORD, chase the model axis (the plan's §5 B-4 — A2/A3 show Luna
is not the constraint), or restore FreeRouting (§6.4 unchanged: `java` absent, no such directory,
and the legacy tree routes with KRT fine).

---

## 6. Environment: the recipes that took several attempts

### 6.1 Launch
```bash
REPO=/home/kicraft/KiCraft
cd "$REPO"
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  "$REPO/.venv/bin/python" -m kicraft.eval.self_eval \
  --out logs/self_eval/<name> --repeats 3
```
`env -i` is mandatory: `.env` never overrides an existing variable, and this box has no
`~/.kicraft/routing.json`, so the production `.env` profile (`luna`) is what runs — but only if the
inherited `KICRAFT_*` variables are cleared first.

### 6.2 Arm venvs must be `--system-site-packages`
`pcbnew` lives in `/usr/lib/python3/dist-packages`; the production venv is created with
`--system-site-packages` and plain venvs are not. A plain venv gives
`ModuleNotFoundError: No module named 'pcbnew'` on the **build** path only — designs still run, so
the failure is silent until you read the build rc. This voided one arm.

### 6.3 The legacy tree (A2 / §3) needs a route, not a patch
`/home/kicraft/KiCraft-legacy` @ `bc6a2f8` is intact, venv-ready, `.env` copied (mode 600). Its
config has no `DESIGN_PROFILES`, so `.env`'s `KICRAFT_DESIGN_PROFILE=luna` is ignored and the
DeepSeek-era defaults apply: `provider.order = novita/siliconflow/streamlake` and a $0.35/Mtok
completion ceiling, which reject every Luna call in <1 s. Run it as:

```bash
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  KICRAFT_PROVIDER_ORDER=openai \
  KICRAFT_MAX_PRICE_PROMPT=0.20 KICRAFT_MAX_PRICE_COMPLETION=1.20 \
  KICRAFT_DESIGN_REASONING_TOKENS=0 \
  KICRAFT_PROJECTS_DIR=/home/kicraft/.kicraft/projects-legacy \
  /home/kicraft/KiCraft-legacy/.venv/bin/python -m kicraft.eval.self_eval --out <abs> --repeats 2
```
The legacy `self_eval` has **no `--design-only`** (argparse exits 2); smoke it with
`--only rc-lowpass-bnc --no-judge --repeats 1`, which also exercises build + pcbnew. The `--out`
path must be absolute: the legacy run's cwd differs from the repo.

### 6.4 Router
Unchanged and unproblematic: legacy and HEAD pin the *same* KRT commit (`3ceb773`, VERSION
`0.20.2`, native `0.20.1`) as this box's checkout, so no second KRT checkout is needed.

### 6.5 Memory — the campaign is not free of production risk
This host has **no swap** (7.7 GB total) and a campaign at the default `--parallel 3` drives
`mem_pct` to 98–99 % in about 1.5 h, at which point the kernel kills the largest process. Three
campaigns died this way (exit 137); `host_metrics.db` shows memory released at the kill instant
(98.3 % → 74.0 % across the 22:49:31 kill). Consequences and mitigations:
- **Run at `--parallel 2`.** The loss per kill is only the in-flight repeats, so always run
  through a resuming runner — `logs/self_eval/movea_tools_20260919/movea_runall2.sh` is the one
  used here (detached `setsid nohup`, auto-`--resume`, no-progress stop, budget abort).
- **Expect degradation of the live site at the plateau.** During 23:57:06Z–00:00:21Z the web app
  failed 7 health checks over ~3.25 min with the process never dying (up since 2026-09-18
  05:23:41Z). Prefer running campaigns outside business hours, and health-check
  `curl -sf http://127.0.0.1:8080/` as part of any watcher.
- Kernel OOM messages are **not readable** to this user (`journalctl` needs `adm`); the OOM
  attribution is inferred from the curve, not proven.

### 6.6 Build slots — the monitor, the leak it watches for, and the config change

**Reap after every kill; the monitor now does it for you.** A SIGKILLed campaign orphans its build
children, and the build timeout lives in the *parent*, so nothing bounds an orphan's lifetime.
`build_slot()`'s `flock` is released only when the holder exits, so a hung-but-alive orphan holds a
**host** slot indefinitely. Measured instance: A1's 15:28Z kill left `cli_app build` pid 1454563
(ppid 1, **12 h 24 m elapsed, 1 s CPU, 407 MB RSS**) holding `slot_0.lock`, with its own child left
a zombie. Production's build worker, which sizes its concurrency from the same host gate, ran
against **one usable slot instead of two** for 12 hours. A leaked slot is silent — the next build
just takes another slot and looks healthy.

```bash
deploy/check-build-slots.sh                     # report; exit 1 when a strict leak is present
deploy/check-build-slots.sh --reap              # TERM then KILL strict leaks; exit 2
deploy/check-build-slots.sh --json              # machine-readable verdict
deploy/check-build-slots.sh --loop 300 --reap   # supervise until stopped
deploy/box-diag.sh                              # remote read-only report now includes a slot line
```

A **strict leak** is a process that is orphaned (`ppid 1`) *and* whose command line is a build/route
invocation — the detached services (`kicraft.server.web`, `kicraft.server.build_worker`) are ppid 1
by design and never match, so they are never reaped. `--min-age-s` (default 120) gives a restart's
in-flight orphans time to exit on their own first. Detector and reaper are tested end-to-end against
a deliberately orphaned holder (§11), and a supervisor is running on this box:

```bash
hub ps                                   # process name: build-slot-monitor
cat logs/build_slots_monitor.log         # one line per pass, 300 s interval
```

**Durable version.** A detached process does not survive a reboot, so the repo's provisioning
pattern (unit files in `deploy/` + installer lines — the jlcparts nightly is the precedent) now
carries `deploy/kicraft-build-slots.service` + `.timer`: a 5-minute sweep, `Persistent=true`,
logging to the same file. Installing needs root, and privileged actions on this box stay gated,
so it is not installed yet:

```bash
cd ~/KiCraft && sudo cp deploy/kicraft-build-slots.{service,timer} /etc/systemd/system/ \
  && sudo systemctl daemon-reload && sudo systemctl enable --now kicraft-build-slots.timer
```

(`deploy/03-service-setup.sh` installs it on a fresh box, next to the jlcparts timer.) Once the
timer is live the detached loop is redundant — stop it with `hub stop build-slot-monitor` or
`pkill -f 'check-build-slots.sh --loop'`.

**Root cause is NOT fixed by the monitor.** Nothing makes a build child exit when its parent dies;
the monitor only reclaims the slot afterwards (up to one interval late). The real fix is a
parent-death bound on the spawned build — `prctl(PR_SET_PDEATHSIG, SIGTERM)` in the child, or a
watchdog that exits the child when `getppid()` changes. That is a code change in the build spawn
path (`cli_app` / `build_worker` / `self_eval`'s build call) and belongs in a session of its own;
until then the monitor is the mitigation.

**Configuration changed 2026-09-20 ~04:00Z: `KICRAFT_BUILD_SLOTS` 2 → 1** (`.env`, mode preserved
at 600), with both services restarted through the documented scripts
(`deploy/restart-build-worker.sh`, `deploy/restart-web.sh`). Evidence for the change:
`kicraft/build_slots.py`'s own docstring states the rule (`slots * 6 ≈ cores`, since one build fans
out to ~6 leaf solvers) and names the 2-on-a-2-core-host setting as the cause of the 2026-07-08
`rc=-9` timeout cluster; the harness already used 1 (`default_build_slots()` deliberately ignores
the variable), so the host gate and the harness disagreed — after `load_dotenv()`
`slot_count()` returned 2 while `default_build_slots()` returned 1. On a no-swap box that is the
configuration in which a production build overlapping a campaign reaches the 98–99 % memory ceiling
of §6.5.

Verified after restart: `[build-worker] ready (max 1 concurrent build(s))` (was `max 2`), web HTTP
200, `slot_count()` = 1 inside a `.env`-loaded process, and `check-build-slots.sh` reports
`0 held / 1 total`. There is still **no admin control** for this value — the only `build_slots`
field in `routes_admin.py` (≈2306/2353/2401) belongs to the loadtest launcher, and neither
`routing_config.ALLOWED_KEYS` nor `Settings` carries it. If the operator wants to change it again,
it is `.env` plus the two restart scripts.

**Consequence of one slot, stated because it changes the blast radius.** With a single host slot a
leak no longer halves build capacity — it removes *all* of it until the sweep runs, which makes the
monitor load-bearing rather than a nicety (worst case ≈7 min: 5 min cadence + the 120 s age guard).
The same slot also serializes a campaign build against a production build, so a campaign run during
business hours can delay a user's build by up to one build timeout (2400 s). Both are acceptable
against what they replace (a silent 12 h leak, and two CPU-saturating builds on two cores), but they
are reasons to keep running campaigns off-hours — and reasons to prefer the parent-death fix below
over the sweep.



---

## 7. Measurement-protocol deltas learned this session

1. **Expand refusal bundles before counting.** `multiple_intent_contracts` is a bundle
   representative: at A1 it is the terminal code on 43 of 69 architecture deaths, and the members
   are the actionable content. Counting the terminal code alone makes the census useless.
2. **A3's schema has no `terminal_diagnostic`.** Its codes are only recoverable from
   `events.jsonl` → `stage_done.schema_error`. Any cross-tree census must read events, not
   summaries.
3. **A2's batch carries the legacy summary schema**: no `design_committed`, `source_unchanged`,
   `failed_run_cost_usd` or `cost_per_committed_design_usd`. Read committed as
   `design_status == "ok"`, and do not fill the missing fields with a guess.
4. **`summary.json.wall_s` is per-attempt for a resumed batch**, not end-to-end. End-to-end
   windows: A1 14:27Z→15:56Z (88 min, one kill), A2 16:20Z→21:09Z (4 h 49 min), A3 21:10Z→00:38Z
   (3 h 28 min, three attempts), A4 00:43Z→01:48Z (65 min).
5. **`triage scan`'s tier classifier disagrees with `build_rc == 0`** (A2: 27 "clean" vs 30
   fab-ready). Report the harness's own `fab_ready` for cross-arm comparisons and the triage tier
   beside it; do not silently pick one.
6. **$0 gates first.** `--reference-replay` (mock transcript) before any paid run, and a 1-brief
   `--only <slug> --no-judge` smoke before every arm — the smoke is what caught both the missing
   `pcbnew` and the provider route, for $0.05 total.
7. **Void batches are quarantined, not resumed.** `--resume` reuses any record with a report file,
   so a batch produced by a broken harness would be reused silently. Both void batches here are
   renamed `*_void_provider` / `*_void_pcbnew`.

---

## 8. Artifacts

**Batches** (`logs/self_eval/`, all gitignored):

| path | what |
|---|---|
| `movea_a1_current_20260919T1431Z` | A1 baseline — the comparison for any Move 1 change |
| `movea_a2_legacy_bc6a2f8` | A2 (legacy pipeline) |
| `movea_a3_precontract_b4b8be5` | A3 (typed, pre-09-16) |
| `movea_a4_recordpatch` | A4 (RECORD preview) |
| `movea_a*_smoke` | per-arm smoke gates |
| `movea_a2_legacy_bc6a2f8_void_provider`, `movea_a3_precontract_b4b8be5_void_pcbnew` | void batches, kept as evidence |
| `movea_a4_recordpatch.patch` | the A4 scratch patch (618 lines, includes the inherited dirty diff) |
| `movea_runall2_20260919.log`, `movea_watch2_20260919.log` | runner + watcher logs (kill/resume history, web health) |
| `movea_tools_20260919/` | the scripts used: runner, arm report/render/append, census, bundle expansion, venv repair, plan restore |

**Trees** (left in place deliberately):
- `/home/kicraft/KiCraft-legacy` — A2 / §3's deployment, at `bc6a2f8`, venv + `.env`.
- `/tmp/kc-0915` — A3 worktree at `b4b8be5` (`git worktree remove` when done).
- `/tmp/kc-a4` — A4 worktree: HEAD + the scratch patch (the patch is also saved as a file).

**Commands** used to turn batches into the tables:
```bash
PY=/home/kicraft/KiCraft/.venv/bin/python
T=logs/self_eval/movea_tools_20260919
$PY $T/arm_report.py A1 logs/self_eval/movea_a1_current_20260919T1431Z   # one arm, full table
$PY $T/census_events.py                                                  # refusal census, all arms
$PY $T/append_s2.py --plan docs/plans/design-yield-recovery-2026-09-19-plan.md \
    --prose <prose.md> A1=<dir> A2=<dir> A3=<dir> A4=<dir>                # idempotent §2 Results
$PY -m kicraft.cli.triage scan logs/self_eval/<arm>                      # per-arm failure census
$PY -m kicraft.eval.design_acceptance --reference-replay                 # $0 gate, floor ≥31/34
```

---

## 9. Uncommitted work in the tree — decide this first

`git status` at HEAD shows three modified files, **all inherited from the previous session** and
**present in both A1 and A4** (so the measured baseline includes them):

- `kicraft/design/architecture_intent.py` — `conflicting_port_binding`'s message now offers the
  rail-tie repair ("delete the connection that duplicates the rail") instead of telling the draft
  to port-shop.
- `kicraft/design/lowering.py` — a contact-derivation refactor: `_contact_net_map` / `_contact_nets`
  (contiguous `pinN`/`pN`, one alphabet, no gaps or duplicates), used by `_connector` (terminal and
  generic), `_numbered_connector_ports`, and `_reviewed_terminal_part` (moving the reviewed
  screw-terminal table onto `part_identity.reviewed_part`); plus `port_contract=` and
  `requirement_ports=` in `lowerer_contract_diagnostic`'s evidence.
- `tests/test_design_lowering.py` — 3 new tests (the plan's appendix records that they fail on
  stock code).

That is `unsupported_lowerer_contract` work: the §4.3 split's *derive* half and the evidence the
next session needs. It is measured-but-unlanded. **Land it or revert it deliberately before
Move 1**, and record which — the A1 baseline was taken with it applied.

---

## 10. Still owed to the operator

From the plan's §9, plus what this session added:

1. **Auto-fallback or manual only?** (plan §9.1; plan's default: manual.) Unchanged, but now
   material: A2's 21/34 means a `pipeline=legacy` switch would ship boards production cannot
   currently produce.
2. **Legacy badge on the board page** (plan §9.2; plan's default: shown plainly).
3. **New: should paid campaigns run on this production box at all?** The 98–99 % memory plateau
   and the 3.25 min window of intermittent unresponsiveness are a customer-visible cost the plan
   did not price. Options seen from here: run at `--parallel 2` off-hours, move campaigns to a
   second host, or accept the risk while there are no paying customers (plan D5).
4. **New: how long does the legacy fork live?** §3's switch keeps an 08-25 tree alive. It buys
   21/34 today; the alternative is closing the 13→21 gap on the current tree (§4 Moves 1–3).
5. **Resolved 2026-09-20: `KICRAFT_BUILD_SLOTS`.** Was `2` on a 2-core, no-swap host, against the
   module's own sizing rule (`slots * 6 ≈ cores`) and against the harness's own value of 1. Now
   `1` in `.env`, both services restarted and verified, and a supervisor
   (`deploy/check-build-slots.sh --loop 300 --reap`, hub name `build-slot-monitor`) reclaims leaked
   host slots. **Open sub-item:** the root cause — a build child outliving its parent — is still
   unfixed (§6.6); the monitor is only the mitigation.

---

## 11. How the monitor was verified

Claiming a detector works without reproducing the failure is how a monitor ends up useless. The
leak was reproduced deliberately and the tool exercised against it end-to-end:

| step | result |
|---|---|
| clean box | `0 held / 2 total`, exit 0, JSON `{"findings":[]}` |
| simulated leak: orphaned process (`exec -a "python -m kicraft.design.cli_app build …"`), holding `flock` on `slot_0`, `ppid=1` | detected; inside the age guard reported `orphan-young` and **not** counted — nothing reaped |
| same, `--min-age-s 1` | `slot 1/2 leak pid=1950481 ppid=1 age=15s rss=10564KB`, `1 held / 2 total · strict leaks: 1`, exit 1 |
| same, `--json` | `{"slots_total":2,"findings":[{"kind":"leak","pid":1950481,…}]}`, exit 1 |
| same, `--reap` | `reaping pid=1950481 (leak)`, exit 2; process gone, slot free |
| afterwards | `{"slots_total":2,"findings":[]}`, exit 0 |

After the config change the same checks report `0 held / 1 total`, and the running services show
`[build-worker] ready (max 1 concurrent build(s))` with `slot_count()` = 1 in a `.env`-loaded
process.

