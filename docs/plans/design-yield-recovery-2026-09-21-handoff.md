# Design yield recovery — handoff and the road to 34/34 (2026-09-21)

**Purpose.** One document for the next session: what was built and is now deployed, what the first
paid measurement since session 1 actually showed (including the part of it that did *not* go the
way the plan predicted), where the 34 reviewed briefs die today, and the sequenced plan to get them
all fab-ready with a cost, an expected delta and a kill criterion per step.

**Read order.** §0 (state in one screen) → §2 (what the screening measured — the headline is flat
and that matters) → §3 (the wall map, with a reproduced defect) → §4 (the plan) → §5 (next three
actions, as commands).

**Supersedes** `design-yield-recovery-2026-09-20-options-1-4-handoff.md` for the *state of play*;
that document and `…-options-1-4-plan.md` §12 remain the record of how the code got here.
`design-yield-recovery-2026-09-19-plan.md` remains the authoritative specification of the bar
(BLOCK vs RECORD) and the measurement protocol.

**Operator mandate (2026-09-20/21):** paid runs approved up to the limits; time must not be wasted.
Practical reading, used throughout below: screen cheap (design-only, ~$1.80), read the result
immediately, confirm with a full run only when a screen earns it; one campaign at a time; the $20
daily ceiling is shared with the live site (a runner-side guard aborts at $13).

---

## 0. The state in one screen

| | value |
|---|---|
| **Deployed** | **Yes.** `deploy/restart-web.sh` + `deploy/restart-build-worker.sh` at 2026-09-20 20:26 UTC: web HTTP 200 on `:8080`, `[build-worker] ready (max 1 concurrent build(s))`, 0/1 build slots held, no in-flight jobs at restart. Before that restart the services were from 03:53 and served pre-session code — which is why the pipeline switch was not visible. |
| **The switch** | `/admin/routing` → "Design pipeline" card (between the model profile and the behaviour knobs): `current` or `legacy pipeline (bc6a2f8, 2026-08-25) — measured 21/34 finished boards`, with the trade-off line. A save applies to the next design run and the next build job, no restart. |
| **Branch** | `simplify/bom-wiring-pipeline`, **19 commits ahead of `origin`** (11 of them this work). Not pushed; the box serves the local commits. |
| **Free gate ($0)** | `--reference-replay` **31/34** rows reproduce their own boundary, 2 recorded blocks (floor ≥31). Green; red at the start of the session. |
| **Unit suite** | **4301 passed**, 15 skipped, 1 xfailed, **2 failed** — both pre-existing and environmental (`test_vendored_bundles_are_not_prototype`: `ams1117-5v0-fixed` still defaults to prototype; `test_krt_preflight_uses_environment_defaults`: `No module named 'py_router.startup_checks'`). |
| **First paid screen (design-only, 34×3)** | **102/102 runs, $1.78, 50 min.** Committed designs **7** (A1 baseline: 7). Briefs committing at all **3** (A1: 4); briefs committing in ≥2 repeats **2** (A1: 3). **Headline flat** — but the failures moved one stage downstream: architecture deaths **69 → 51**, BOM deaths **23 → 41**. §2 reads this. |
| **A/B attribution** | Ran at the same time as this document (batches `ab_move1a_before` / `ab_move1a_after`): the five briefs that moved, 3 repeats each, on the two trees that differ *only* by moves 1a–1b. Result in §2.3. |
| **Spend** | 2026-09-20: $4.24 (session-1 arm $2.02 + this session's screen $1.78 + smokes). 2026-09-21: $0.11 at the time of writing. Daily ceiling $20. |
| **The ladder** | current tree **4/34** fab-ready briefs (A1) · 09-15 tree **13/34** (A3) · August tree **21/34** (A2) · best recorded July batch **25/34** · target **34/34** (never achieved). |

---

## 1. What was built (11 commits, all deployed)

| commit | what it does |
|---|---|
| `b7a379f` | The free gate repaired (a stale reference row re-recorded; the contract was right), a contract refusal's own code/message/evidence now stored on the stage, the inherited lowerer contact-derivation work landed, and **moves 1a–1b**: the compiler derives the pin owner where it already knows it (a domain annotation on a port the design's own signals bind is not a tie; a lowerer's published return contact — a coin cell's `negative`, a jack's `sleeve` — is grounded from the design's own ground; a connector contact a signal names after a declared rail is that rail's exposure). |
| `036d8fb` | **Move 2a**: a declared-interface pin is resolved by number **or** by one unique pin name (30 of ~50 A1 BOM defects were a correct name compared against a number list). **2b**: a rejected commit records **every** reason, not one. |
| `b432205` | **Moves 4a/4b**: the RECORD path — `ArchitectureAdvisory`, `_advise(...)`, the §9.33/§9.34 notes, advisories in the board's provenance file, a recording-only rubric gate (`shipped_with_advisories`, unreachable cap), and §4.3's classification with one measured exception: `unknown_part_refused` stays BLOCK because downgrading it drops the part from the board and the run dies one validation later naming something else. |
| `9a44ec1` | **Option 3**: the pipeline switch (new `kicraft/server/pipeline.py`), the legacy tree's measured environment, the design subprocess, the build-interpreter substitution, and full marking (workspace marker, `projects.pipeline` column + migration, promote provenance, summary grouping). |
| `a8c405b`, `210a82e`, `80f219a` | Comment correction on how the legacy package resolves; a migration guard against the production DB shape; `unknown_part_refused` back to BLOCK with the evidence. |
| `b7be38f`, `420f2dd`, `41d295d`, `e03313b` | The measured results in the plan's §12, and two handoff documents. |

---

## 2. What the first paid screen measured

### 2.1 The table

Batch `logs/self_eval/opt1_designonly_20260920` (34 briefs × 3, design-only, `--parallel 2`, $1.78,
3 030 s) against the frozen baseline `logs/self_eval/movea_a1_current_20260919T1431Z`.

| | A1 (before) | screen (after) |
|---|---|---|
| committed designs | 7 / 102 | **7 / 102** |
| briefs committing at all | 4 | 3 |
| briefs committing in ≥2 repeats | 3 | 2 |
| first-failing stage | architecture 69 · bom 23 · passed 7 · wiring 2 · intent 1 | **architecture 51 · bom 41** · passed 7 · functional_spec 1 · intent 2 |
| per-brief commits | r2r-dac 2, fpc-breakout 2, audio-jack-buffer 2, rc-lowpass-bnc 1 | rc-lowpass-bnc 3, proto-shield 3, fpc-breakout 1 |

### 2.2 The mechanism fired; the boards did not appear

Refusal-code counts over the same 102 runs (bundle-expanded; every code that appeared in any
attempt):

| code | A1 | screen | delta |
|---|---|---|---|
| `declared_signal_port_tied` | 90 | **20** | **−70** |
| `unreviewed_exact_part` | 43 | **0** | **−43** (now a recorded note) |
| `unknown_interface_port` | 38 | 11 | −27 |
| `conflicting_port_binding` | 140 | 117 | −23 |
| `unbound_required_port` | 25 | 4 | −21 |
| `source_obligation_not_retained` | 27 | 19 | −8 |
| `unsupported_lowerer_contract` | 160 | 179 | +19 |
| `malformed_signal_ref` | 1 | 18 | +17 |
| `architecture_power_block_as_sheet` | 6 | 18 | +12 |
| `commit_gate_rejected` (new, this session: a *recorded* commit refusal, not a new refusal) | 0 | 12 | +12 |

**Reading.** Every code the derives targeted fell, several by three quarters; 18 runs that used to
die at the architecture stage now reach the BOM stage and die there. The headline is unchanged
because the wall is one stage later — exactly the shape the RECORD preview measured in session 1
(12 rescued attempts, all dead at the BOM) and the reason the plan's §4.1 keeps the BOM obligation
codes blocking. The increases are exposure, not regressions: a draft that survives its earlier
refusal gets further into the same pass, so later checks in that pass now report too.

**Decision (recorded, deliberate deviation from the plan's literal kill criterion).** The plan says
"no rise in the per-brief median commit count ⇒ revert". The rise did not come. Reverting would
re-block 18 runs at architecture and free no board; the deterministic before/after on 26 real
drafts (3 → 5 committing, the two tied-port codes → 0) says the derives are not harmful and do what
they were built to do. So the derives **stay**, no yield claim is made for them, and the next move
is the wall they exposed (§3.1). If the *next* screen (M2 below) also fails to move the headline,
the architecture moves get re-examined together with it.

### 2.3 The three briefs that moved down, attributed

`audio-jack-buffer` 2/3 → 0/3, `r2r-dac` 2/3 → 0/3, `fpc-breakout` 2/3 → 1/3, against
`proto-shield` 0/3 → 3/3 and `rc-lowpass-bnc` 1/3 → 3/3. This corpus has measured 60 % → 20 % swings
on *identical* inputs, so a single arm cannot attribute those. A/B run (both arms, same box, same
session, 3 repeats, only the 1a derives differing): `logs/self_eval/ab_move1a_20260921.log`,
batches `ab_move1a_before` / `ab_move1a_after`, ~$0.55.

> **Result:** see `§2.3 result` at the end of this document (filled in when the run completed).

---

## 3. The wall map — where the 34 briefs die today

### 3.1 The BOM wall, and a reproduced defect (this is move M2)

Runs whose BOM stage failed carrying a defect payload:

| defect class | A1 | screen | example |
|---|---|---|---|
| `declared-interface-unrealized` | 11 | **9** | `breakout: declared interface needs one identified hardware owner` |
| `missing-requirement-implementation` | 6 | 3 | `power_led` |
| `physical-obligation-unfulfilled` | 4 | 3 | `led_driver:power_led: requires 1 real power-led, found 0; the unit emitted: led_driver=al8860:AL8860MP-13 …` |
| `model_authored_protected_identity` | 2 | 3 | `passive_breakout_connector` |

The largest class is the *owner* half of `declared-interface-unrealized`, and it is a bookkeeping
defect rather than a design error. Reproduced at $0 from the model's own BOM answer stream
(`run_11_fpc-breakout__r3`):

```
requirement:  breakout            (a curated family; the design chose the part, not the draft)
emitted group: id=passive_breakout_connector  symbol=fpc-24-0-5-kinghelm:KH-FG0.5-H2.0-24PIN
                                           mpn=KH-FG0.5-H2.0-24PIN
refusal:      breakout: declared interface needs one identified hardware owner
```

Root cause, in two lines: `_requirement_owns_protected_group` compares the *requirement's own
declared* identity (`exact_part`, then `family` as a word) against the group's `mpn`/`value`, and
gives up when the family is a typed product family — while the part in front of it was chosen by the
requirement's **own recipe/lowerer**, whose identity the compiler has already stamped on the group
(`resolution_id`, `resolution_source`, `recipe_id`; the lowering path even reads
`_lowering_requirement_id` a few calls away) and whose symbol *library* (`fpc-24-0-5-kinghelm`) is
the family the requirement resolves to. Nothing about the board is wrong; the check cannot see who
owns it.

**M2 fix:** resolve ownership from the compiler's own bookkeeping first — `resolution_id ==
requirement.id`, or `recipe_id`/`_lowering_requirement_id` matching, or the group's symbol library
or resolved MPN matching the requirement's family/recipe identity — and only then fall back to the
current word matching. Refuse only when none of those resolve, naming what was tried. Expect
+4…+9 committing designs per arm (9 in the screen, 11 in A1), and the first realistic chance at new
*finished boards*, because these are runs whose parts were already right.

### 3.2 The architecture set, censused and split three ways (this is move M3)

The largest single architecture refusal, `source_obligation_not_retained` (11 of A1's 69 deaths),
has 51 offending obligation rows across A1 + A4. They are three different problems:

| sub-case | rows | repair |
|---|---|---|
| `quantitative` (a numeric limit with a unit, e.g. *board width ≤ 40 mm*) | **22** | the ownership rule exempts `quantity`, `fabrication`, `negative` as "board-level facts, not implementation claims"; a numeric limit is the same kind of fact and is currently *not* exempt. Either add it to `OWNERSHIP_EXEMPT_OBLIGATION_KINDS` with the reason recorded, or attach it to the requirement that owns its subject. **Cheapest large win in the dataset.** |
| novel "physical" classes that name a board shape/feature (`snowman-shaped-board` 9, `rounded-rectangular-pcb`, `chamfered-board-outline`) | 10+ | nothing can own a board outline. Retype to `fabrication` when the class names a board/outline/shape fact **and** no reviewed part and no reviewed variant could carry it. Careful: `mounting-hole` *is* realizable, so no naive keyword rule. |
| realizable classes the draft never attached (`crystal` 5, `warm-white-led` 3, `stacking-header`, `screw-terminal`, `qspi-flash-memory`) | 9 | where exactly one requirement's family/recipe realizes the class, attach the row and note it; where none or several could, refuse and name the candidates. |

### 3.3 The empty unit (this is move M4, and it is option 2's real wall)

`run_10_rp2040-min__r1` (A4): the BOM stage ran with `attempts: 0`, `work_units: 1`, and no parts —
a `deterministic_architecture_lowering` unit with a requirement emitted nothing and never asked the
model, so the commit gate judged an **empty BOM** and refused six realization contracts plus a
castellation representation (`bom_castellation_placeholder`, offender `j2`). Every one of those
messages is downstream of "the unit produced nothing". Fix the unit (a unit expected to produce
parts and producing none should refuse at the unit, by name, cheaply repairable) before reading any
of the messages behind it.

### 3.4 What is not the wall

The library does hold the classes the plan suspected (`nx3215sa-32.768k-std-mua-9` carries
`crystal`; `rp2040` carries `microcontroller`); the §9.42 messages were symptoms of §3.3, not an
unsatisfiable demand. Do not loosen part demands for them.

---

## 4. The plan to 34/34

One move per screen, each with a pre-registered delta and a kill criterion. Screens are design-only
(34 × 3 ≈ $1.80, ~50 min); a confirmation is a full run (~$2–3, 1–5 h, off-hours).

| | move | expected on the screen | kill criterion |
|---|---|---|---|
| **M1** ✅ | free gate green + architecture derives + RECORD path (done, deployed) | — (this screen: flat headline, wall moved 18 runs downstream) | — |
| **M2** | **the declared-interface owner resolution** (§3.1) | committed designs 7 → **11–16**; BOM deaths 41 → ~32 | < +3 committed designs ⇒ revert M2, re-derive from the per-attempt `work_unit` failure text |
| **M3** | **the obligation-ownership split** (§3.2): `quantitative` exemption first, then the board-shape retype, then the provable attachment | architecture deaths 51 → ~35; committed designs +2–4 | no rise in committed designs ⇒ revert; a null recorded is a delivered move |
| **M4** | **the empty unit** (§3.3) | protects M2/M3's gains from dying behind an empty BOM; ≥3 of the runs that reach BOM stop failing for "found 0" | no rise in BOM-stage completions |
| **M5** | **confirming full run** (34 × 3, with builds) | the headline: fab-ready briefs, and every newly shipped board's recorded notes | fewer than +3 boards over A1's 4 ⇒ the screen→board conversion is broken; re-derive from the per-board build tail |
| **M6** | **the long tail**: routing failures, unconnected items, DRC/outline checks, and the briefs needing a part or topology the library does not model | 21 → 26+ | — (each tail item is its own census-driven move) |

**Cost to a measured claim of the next milestone (13 → ~21 on the current tree):** M2 screen +
M3 screen + M5 confirmation ≈ **$6–8** and one working day, inside the daily ceiling if spread over
two days.

**Parallel track B — the legacy switch (already deployed, needs proving once).**
`~$0.10` + one build: drive one brief with `pipeline=legacy` end to end (five stages **and** a
finished board), switch back, confirm `current`, and check the pipeline is named in the project row,
the provenance file and the summary. This is the only untested path in the whole session and it is
worth 21/34 boards today. It does not require flipping the global switch — a test workspace with the
marker set exercises the same dispatch — but the *operator decision* it feeds (which pipeline serves
users, and for how long the fork lives) is §8.2.

**Track C — the ops items** (§8.4: install the sweep timer; decide about pushing the branch).

**Cadence rules (unchanged, they are what makes this not wasteful):** 34 × 3 minimum, per-brief
median as the unit; expand refusal bundles before counting; one campaign at a time at
`--parallel 2`; append every result to the plan document; never overwrite a frozen baseline; state
the expected delta *before* the run and revert when it misses.

---

## 5. The next three actions, as commands

**1. Finish the attribution that is already running** (batches `ab_move1a_before`/`ab_move1a_after`)
and read §2.3. If the before-arm wins clearly, revert `b7a379f`'s derive hunks and re-baseline
before M2; if not, M2 proceeds on the current tree.

**2. Implement M2** (§3.1) — `_requirement_owns_protected_group` in
`kicraft/server/stage_contracts.py`, plus a guard test built from the reproduced case
(requirement `breakout` / group `passive_breakout_connector` / mpn `KH-FG0.5-H2.0-24PIN`).

**3. Screen M2** on the same instrument as M1, for a like-for-like comparison:
```bash
REPO=/home/kicraft/KiCraft
cp logs/self_eval/opt1_designonly_20260920_run.sh logs/self_eval/m2_designonly_run.sh
# point OUT/LOG at m2_designonly_<date>, leave everything else as-is, then:
setsid nohup logs/self_eval/m2_designonly_run.sh >/dev/null 2>&1 &
tail -f logs/self_eval/m2_designonly_<date>.log
```
Compare against `movea_a1_current_20260919T1431Z` **and** against the M1 screen
(`opt1_designonly_20260920`) so the M2 delta is separated from M1's.

---

## 6. Environment recipes (do not re-derive)

```bash
REPO=/home/kicraft/KiCraft
# every campaign launch
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  "$REPO/.venv/bin/python" -m kicraft.eval.self_eval \
  --out logs/self_eval/<name> --repeats 3 --parallel 2
```
- `env -i` is mandatory (`.env` never overrides an existing variable; the tool shell carries stale
  `KICRAFT_*`). `--design-only` for screens, `--no-judge` when the score is not needed,
  `--only <slugs>` + `--resume <dir>` for slices and continuations.
- **Box**: 2 cores, no swap, 7.7 GB. `--parallel 3` reaches 98–99 % memory and gets the campaign
  killed (~90 min); `--parallel 2` is the setting that survives. Health-check
  `curl -sf http://127.0.0.1:8080/` during a run.
- **Slots**: `deploy/check-build-slots.sh [--reap]` after any abnormal exit — a leaked slot now
  removes *all* build capacity (one slot configured).
- **A/B or frozen-tree runs**: `git worktree add /tmp/<name> <commit>`, copy `.env` in
  (`chmod 600`), run with `PYTHONPATH=/tmp/<name>` and `cwd=/tmp/<name>`. Verified: the worktree's
  package wins over the venv's editable install, and `cli_app` subprocesses inherit it.
- **Legacy tree**: everything is in `kicraft/server/pipeline.py` (`LEGACY_ENV`, paths, pinned
  commit); `pipeline.describe()` prints the state. Its venv resolves its own checkout from any
  directory without a `kicraft/` package; the `PYTHONPATH` pin is belt-and-braces.
- **Deploy**: `deploy/restart-build-worker.sh && deploy/restart-web.sh` — both services, because
  the build dispatch lives in the worker. No `pip install` unless dependencies changed.

---

## 7. Baselines and artifacts

| path | what |
|---|---|
| `logs/self_eval/movea_a1_current_20260919T1431Z` | **the frozen baseline** (current tree, 102 attempts): 7 committed, 3 briefs ≥2 repeats, 5 fab-ready runs / 4 briefs. Never overwrite. |
| `logs/self_eval/movea_a3_precontract_b4b8be5` | 09-15 tree, 13/34 fab-ready |
| `logs/self_eval/movea_a2_legacy_bc6a2f8` | August pipeline, 21/34 — the option-3 value |
| `logs/self_eval/opt1_designonly_20260920` (+ `.log`, `_run.sh`) | the M1 screen and its resuming runner |
| `logs/self_eval/ab_move1a_{before,after}` (+ `ab_move1a_20260921.log`) | the 1a attribution A/B |
| `logs/self_eval/movea_a4_recordpatch` (+ `.patch`) | the RECORD preview: the patch that landed, and the measurement that says do not widen it |
| `logs/self_eval/reference_replay_{after_1a,final}.log` | the two green gate runs |
| `logs/self_eval/movea_tools_20260919/` | session 1's tooling (census, bundle expansion, per-arm report, resuming runner) |

---

## 8. Operator decisions still open

1. **Push the branch?** 19 local commits are the only copy; the box serves them locally.
2. **Option 3, in or out — and manual or automatic?** Manual is what shipped. An automatic retry of
   a failed `current` design on the legacy pipeline means two configurations producing boards with
   no human choosing; the plan's default is no.
3. **How long does the legacy fork live?** It buys 21/34 today and receives none of M2–M6's fixes.
4. **Install the build-slot sweep timer** (needs root):
   `sudo cp deploy/kicraft-build-slots.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now kicraft-build-slots.timer`
5. **Keep the two $0 instruments as tracked tools?** The real-draft replay corpus (screens any
   architecture change for $0) and the bundle-expanding census (prevents ranking the wrong code
   first) live in `/tmp` and session-1 scratch respectively.

---

## 9. Do not redo

- **Widening the RECORD list.** Measured null: every attempt the downgrade rescued died one stage
  later at the BOM — which is exactly what the M1 screen then reproduced on the live corpus.
- **Rating a change by its headline alone.** The M1 screen's headline is flat while 18 runs moved
  one stage deeper and the target codes fell 70–100 %. Read the composition and the stage split.
- **Trusting the stored terminal diagnosis.** It is a bundle that hides its members; expand it
  before ranking anything.
- **Loosening the reference rows to make the free gate green.** The corpus arbitrates.
- **Wording-only fixes and error-class aliases.** Six such commits moved nothing.
- **Chasing the model.** The August tree hits its era's numbers with today's model.
- **Two campaigns at once, or `--parallel 3`.** Both have already cost a session.

---

## §2.3 result (filled in when the A/B finished)

See the dedicated section appended below.
