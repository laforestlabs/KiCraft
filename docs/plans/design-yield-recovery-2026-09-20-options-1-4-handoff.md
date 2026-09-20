# Design yield recovery — options 1–4 handed over (2026-09-20, session 2)

**Purpose.** Hand the state of the options-1–4 implementation to the next session: what landed,
what is deployed and therefore visible, what the free measurements decided, what is running right
now, and the sequenced plan to 34/34 with its costs and kill criteria. It interprets, and does not
replace, `docs/plans/design-yield-recovery-2026-09-20-options-1-4-plan.md` (§12 there is this
session's measured results).

**Read order.** §0 (state in one screen) → §2 (what the free measurements decided — two of the
plan's own premises were wrong) → §4 (the plan forward) → §5 (environment recipes). §1, §3, §6–§8
are reference.

**Operator mandate for the next session (given 2026-09-20):** paid runs are approved up to the
limits, and time must not be wasted. Practical reading: screen cheap first (design-only, ~$1.80),
read the result immediately, and only then spend the confirming run; one campaign at a time, never
two; the daily ceiling is $20 and is shared with the live site.

---

## 0. The state in one screen

| | value |
|---|---|
| **Deployed?** | **Yes, since 2026-09-20 20:26 UTC.** Both services restarted through `deploy/restart-web.sh` / `deploy/restart-build-worker.sh`: web HTTP 200 on `:8080`, `[build-worker] ready (max 1 concurrent build(s))`, 0/1 build slots held, no in-flight jobs at restart time. Before that restart the running processes were from 03:53 and served pre-session code — that is why the switch was not visible. |
| **Where the switch is** | `/admin/routing`, behind admin login → a "Design pipeline" card between the model profile and the behaviour knobs, labelled `legacy pipeline (bc6a2f8, 2026-08-25) — measured 21/34 finished boards`, with the trade-off line under it. A save takes effect on the next design run and the next build job, with no restart. |
| **Branch** | `simplify/bom-wiring-pipeline`, **19 commits ahead of `origin`** (this session added 8). Nothing pushed; the box runs the local commits. |
| **Free gate** | `--reference-replay` **green: 31/34 rows reproduce their own boundary**, 2 recorded blocks (floor ≥31). Red at session start. |
| **Unit suite** | **4301 passed**, 15 skipped, 1 xfailed, **2 failed** — both pre-existing and environmental (`test_vendored_bundles_are_not_prototype`: `ams1117-5v0-fixed` still defaults to prototype; `test_krt_preflight_uses_environment_defaults`: `No module named 'py_router.startup_checks'`). Both fail identically on the pre-session tree. |
| **Running right now** | The option-1 design-only screening (34 briefs × 3, `--design-only --parallel 2`): batch `logs/self_eval/opt1_designonly_20260920`, log `…_20260920.log`, pid in `…_20260920.pid`. See §3 for how to read and how to stop it. |
| **Spend so far** | Session 1 campaign: $9.42. This session: $0 of campaign money until the screening started (the free gate and the draft replays are $0). |
| **The wall that matters** | A1 baseline (`logs/self_eval/movea_a1_current_20260919T1431Z`): 102 attempts, **7 committed designs, 3 briefs committing in ≥2 repeats, 5 fab-ready runs / 4 briefs**. Milestone ladder in §4. |

---

## 1. What landed (8 commits, in order)

| commit | what | where |
|---|---|---|
| `b7a379f` | P1: gate repaired (stale `proto-shield` row re-recorded) + a contract refusal's own code/message/evidence now recorded on the stage. P2: the inherited lowerer contact-derivation work landed. Moves 1a–1b: the compiler now derives what it already knows (see §2). | `kicraft/server/stage_runtime.py`, `kicraft/design/architecture_intent.py`, `kicraft/design/lowering.py`, `tests/fixtures/reference_inputs/interface_references.json`, `tests/test_design_lowering.py` |
| `036d8fb` | Move 2a: a declared-interface pin is resolved by number **or** by one unique pin name (30 of ~50 BOM defects in A1 were a name compared against a number list). Move 2b: a rejected commit records **every** reason, not one. | `kicraft/server/stage_work_units.py`, `kicraft/server/stage_runtime.py` |
| `b432205` | Moves 4a/4b: the RECORD-class advisory path — `ArchitectureAdvisory`, `_advise(...)`, the §9.33/§9.34 BOM notes, advisories in the board's provenance, the recording rubric gate `shipped_with_advisories` (rubric v2→v3, hash + `RUBRIC.md` refreshed), and the §4.3 classification with one measured exception (§2.3). | `kicraft/design/advisories.py` (new), `kicraft/design/models.py`, `kicraft/design/cli_app.py`, `kicraft/eval/{artifacts,scoring,self_eval}.py`, `kicraft/eval/rubric.yaml`, `tests/skill-eval/RUBRIC.md`, `tests/test_design_advisories.py` (new) |
| `9a44ec1` | Option 3: the pipeline switch — `kicraft/server/pipeline.py` (new), the `pipeline` routing key + `Settings.pipeline`, the admin select, the legacy design subprocess with its measured environment, the build-interpreter substitution, and the marking (workspace marker, `projects.pipeline` column + migration, promote provenance, summary grouping). | `kicraft/server/{pipeline,routing_config,config,session,build_worker,routes_admin,accounts,web}.py`, `kicraft/cli/artifact_paths.py`, `tests/test_pipeline_switch.py` (new) |
| `a8c405b` | Comment correction: the legacy venv resolves its own package from any directory without a `kicraft/` package; the `PYTHONPATH`/cwd pin is belt-and-braces. | `kicraft/server/pipeline.py` |
| `210a82e` | Guard for the `projects.pipeline` migration against the production DB shape (pre-pipeline schema → row survives with `NULL`). | `tests/test_pipeline_switch.py` |
| `80f219a` | `unknown_part_refused` **stays BLOCK**: the downgrade drops the requirement from the netlist entirely and the run dies one validation later naming something else (§2.3). | `kicraft/design/architecture_intent.py`, `tests/` |
| `b7be38f` | `§12 Results` in the plan document — the measured state, the two premise corrections, the verification table, and the paid commands still owed. | `docs/plans/design-yield-recovery-2026-09-20-options-1-4-plan.md` |

---

## 2. What the free measurements decided (read this before planning anything)

### 2.1 The census denominator was wrong — the plan's per-code ranking overstates its top entry

The plan's §1.1 counted "attempts whose failure text contains the code". The stored terminal
diagnostic is a *bundle* (`multiple_intent_contracts`) that hides its members, so I expanded every
one of A1's 69 architecture deaths:

| terminal bundle | runs |
|---|---|
| `source_obligation_not_retained` **alone** | **11** |
| `unsupported_lowerer_contract` alone | 6 |
| a tie code alone (`conflicting_port_binding`, `declared_port_double_bound`, `declared_signal_port_tied`) | 8 |
| `unreviewed_exact_part` alone | 5 |
| any tie code present (with any others) | 31 |
| `unsupported_lowerer_contract` present (with any others) | 32 |

Consequence: move 1a's "expect ≥10 more attempts commit a design" is reachable only for the
*bundle* (tie codes + the contact contract), which is what was implemented — not for
`unsupported_lowerer_contract` alone, whose ceiling is 6.

### 2.2 Move 2b's premise was refuted by the run the plan quotes

`run_10_rp2040-min__r1` (batch `movea_a4_recordpatch`) died with
`bom_castellation_placeholder` ("board-fabricated castellations were represented as assembly
headers", offender `j2`) over a BOM whose recipe-owned expansion never reached the stage
(`attempts: 0`, `work_units: 1`, no parts). The six `E_PHYSICAL_REALIZATION` offenders the plan
quotes are members of that same rejection, and the library **does** hold those classes
(`nx3215sa-32.768k-std-mua-9` carries `crystal`; `rp2040` carries `microcontroller`). Nothing was
derived or downgraded on that premise. What *was* implemented is the rule that made the misreading
possible: a rejected commit now records all its reasons.

### 2.3 One RECORD downgrade was reverted, with evidence

§4.3's table lists `unknown_part_refused` as RECORD. Measured, the downgrade does not do what the
table assumes: with no curated recipe and no declared interface the requirement has no catalog, so
it never enters `requirements` — the part is **absent from the netlist**, and the obligation it
owned then fails the ownership check one validation later under a message that no longer names the
cause. Three of three production drafts whose only advisory was this code died exactly that way
(`1_870`, `1_867`, `1_862`). §4.1's boundary wins: the code stays BLOCK. The other RECORD code
(`unreviewed_exact_part`) is kept, where the requirement *is* carried and the note names the
identity that was unproven.

### 2.4 What the free measurements showed, in numbers

| measurement | before | after |
|---|---|---|
| `--reference-replay` (34 reviewed rows, $0) | **FAILED** (1 row refused at `functional_spec`) | **31/34 pass**, 2 recorded blocks |
| real-draft corpus (26 per-attempt architecture drafts recovered from production `answer_delta` events, replayed at $0 through the real derivation) | 3 commit | **5 commit**; `unsupported_lowerer_contract` 9→6, `conflicting_port_binding` 7→6, `declared_port_double_bound` 3→0, `declared_signal_port_tied` 2→0, `signal_names_rail` 1→0 |
| unit suite | — | 4301 passed, 2 pre-existing environmental failures |

The draft-corpus instrument is reusable and cheap: it is how any future architecture-stage change
can be screened before spend. The extraction idiom (events → per-attempt drafts → real derivation)
is in this session's history; a script worth recreating lives at
`logs/self_eval/movea_tools_20260919/` for the census half.

---

## 3. What is running right now, and how to read it

**Batch:** `logs/self_eval/opt1_designonly_20260920` · **log:** `…_20260920.log` ·
**runner:** `logs/self_eval/opt1_designonly_20260920_run.sh` (resuming: up to 8 attempts, smoke
gate first, aborts if today's spend passes $13, health-checks the web app each attempt)
· **pid:** `logs/self_eval/opt1_designonly_20260920_run.sh` writes it to `…_20260920.pid`.

**Pre-registered reading** (write it down before looking at the result):

| observation | verdict |
|---|---|
| ≥17/102 runs committed (A1: 7) **and** ≥6 briefs committing in ≥2 repeats (A1: 3) | moves 1a/1b/2a/4a hold — keep them and spend the confirming full run |
| 8–16/102 committed | partial: keep the derives, and read *which* codes still kill runs before spending more |
| ≤7/102 committed (no rise) | the architecture moves do not pay on the live corpus ⇒ revert moves 1a–1b (they are one commit, `b7a379f`) and re-derive from the per-attempt failure text |

**How to stop it** (it is `setsid`, so it survives the session): `kill $(cat logs/self_eval/opt1_designonly_20260920.pid)`, then
`deploy/check-build-slots.sh --reap` (a killed campaign can orphan a build child; the sweep timer
is still not installed, see §7.4).

**Cost:** ~$1.80 expected; the smoke is ~$0.02–0.05. Design-only means no builds, so memory stays
far below the 98–99 % plateau the full campaigns hit.

---

## 4. The plan forward — measurable progress toward 34/34

The ladder, from the measured baselines: current tree **4/34** fab-ready briefs → 09-15 tree
**13/34** → August tree **21/34** → best recorded July batch **25/34**. 34/34 has never been
reached. Each step below is one campaign and one decision, in the order that buys the most boards
per dollar and per hour.

### Step 1 (in flight) — does §3's screening pass?

Cost ~$1.80, ~30–60 min, running now. Decides whether the architecture-stage derives stay. If it
passes, the current tree's design commit rate is up by 2–3× on the only metric that predicts
finished boards.

### Step 2 (same sitting) — prove the pipeline switch end to end

Cost ~$0.05–0.10 + one build (minutes to ~40 min). The only untested path in the whole session, and
it is worth 21/34 boards *today* if it works.

1. `/admin/routing` → pipeline = `legacy` → Save.
2. Start one brief (the plan's acceptance uses `rc-lowpass-bnc`; any brief with a known-good shape
   works) and confirm five stages commit **and** a board reaches `fab-ready`, through the legacy
   tree.
3. Switch back to `current`, run the same brief, confirm it uses the current tree; no restart in
   between.
4. Confirm the pipeline is named in the project row, in `<stem>.provenance.json`, and in the
   summary's `pipeline_counts`.

If it fails: the failure is in the dispatch, not in the plan — read the job log
(`~/.kicraft/work/*/build.log` or `logs/kicraft_build_worker.log`) and the tail the legacy path
returns (`stdout_tail`/`stderr_tail` on the session result). The legacy tree's environment is
already pinned in `kicraft/server/pipeline.py::LEGACY_ENV`; the handoff §5.5 has the reference
recipe it came from.

### Step 3 — the confirming full run on the current tree

Cost ~$2–3, 1–5 h, off-hours, `--parallel 2`, one build slot. This is the number the plan actually
tracks: **finished boards per 34-brief campaign**, with the notes each carries. Pre-registered:
move 1a/1b must raise committed designs ≥3 briefs over A1; move 2a is expected to convert ≥6
committing designs and ≥3 finished boards (its 30-defect class); move 4a/4b must add no *unproven*
boards — every newly shipped board that carries a note names it.

### Step 4 — the largest single remaining architecture class

`source_obligation_not_retained`: 11 of A1's 69 architecture deaths, more than any code this
session touched, and in none of the four options. The draft must attach each committed obligation
to the requirement that implements it. Two directions, cheapest first:
(a) make the refusal actionable (name *which* requirement should own it, from the design's own
sheets/families) — wording alone will not pay, so measure it as a derive: where exactly one
requirement's family can realize the obligation's class, the compiler attaches it and notes it;
(b) otherwise leave it blocking and record the null.

### Step 5 — option 2's real wall

The empty-BOM defect behind Step 4's neighbour: in `rp2040-min`, a `deterministic_architecture_lowering`
unit with a requirement produced **no** parts and no provider call (`attempts: 0`), so the commit
gate judged an empty BOM and refused the whole board. Fix the unit's emptiness at the unit (a cheap,
repairable, named refusal) and then the castellation representation the run actually died on. This
is where 13 → 21 lives on the current tree.

### Step 6 — the long tail (21 → 26+)

Routing failures, unconnected items, DRC and outline checks, and the briefs that need a part or
topology the library does not model. Nothing measured says this is impossible; nothing says it is
close either. Run it as: full campaign → census the *first-failing-stage split* and the expanded
failure codes → pick the top two by count → one move each.

**Cadence rules that keep this from wasting time** (from the plan's §7–§8, unchanged): one move per
session with its delta pre-registered and reverted if missed; 34 briefs × 3 minimum, per-brief
median as the unit; expand bundles before counting; append every result to the plan document; never
overwrite a baseline batch directory; alternate cheap screens with expensive confirmations.

---

## 5. Environment recipes (they cost several attempts to get right — do not re-derive)

### 5.1 Launch
```bash
REPO=/home/kicraft/KiCraft
env -i HOME=/home/kicraft PATH="$PATH" TERM=xterm PYTHONUNBUFFERED=1 \
  "$REPO/.venv/bin/python" -m kicraft.eval.self_eval --out logs/self_eval/<name> --repeats 3
```
`env -i` is mandatory: `.env` never overrides an existing variable, and the shell that runs these
commands carries stale `KICRAFT_*`. Add `--design-only` for the cheap screen, `--parallel 2` (never
3), `--only <slug>` + `--no-judge` for a smoke, `--resume <dir>` to continue a batch.

### 5.2 Money
The daily ceiling is **$20**, shared with production; the runner aborts at **$13** spent today. The
campaign runner is the only thing that should spend. Read today's spend any time with:
```bash
.venv/bin/python - <<'PY'
import datetime, sqlite3
day = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
con = sqlite3.connect("file:/home/kicraft/.kicraft/spend_ledger.db?mode=ro", uri=True)
print(con.execute("select coalesce(sum(cost_usd),0) from spend where substr(ts,1,10)=?", (day,)).fetchone())
PY
```

### 5.3 The box under a campaign
2 cores, no swap, 7.7 GB. `--parallel 3` drives memory to 98–99 % and the kernel kills a campaign
roughly every 90 minutes; `--parallel 2` is the setting that survives. Always run through a resuming
runner (`opt1_designonly_20260920_run.sh` is this session's; `movea_runall2.sh` is session 1's).
Health-check the site during the run (`curl -sf http://127.0.0.1:8080/`); at the memory plateau the
live site was briefly unresponsive for ~3.25 min on 2026-09-19. After any abnormal exit:

### 5.4 Build slots
```bash
deploy/check-build-slots.sh            # report; exit 1 on a strict leak
deploy/check-build-slots.sh --reap     # TERM then KILL strict leaks
```
One slot is configured (`KICRAFT_BUILD_SLOTS=1`). A leak now removes *all* build capacity, so reap
after every kill. The 5-minute sweep timer is still **not installed** (§7.4).

### 5.5 The legacy tree (option 3)
Everything the switch needs is in `kicraft/server/pipeline.py`: the pinned commit `bc6a2f8`, the
interpreter `/home/kicraft/KiCraft-legacy/.venv/bin/python`, and `LEGACY_ENV`
(`KICRAFT_PROVIDER_ORDER=openai`, `KICRAFT_MAX_PRICE_PROMPT=0.20`,
`KICRAFT_MAX_PRICE_COMPLETION=1.20`, `KICRAFT_DESIGN_REASONING_TOKENS=0`) — without those, its
DeepSeek-era defaults reject every call in under a second at $0.00. `pipeline.describe()` prints the
whole picture, including whether the tree is available. A legacy design runs
`-m kicraft.server.stage_driver run … --no-build` in a subprocess; the build then runs the legacy
interpreter on the same `.kicraft/state.json`. Projects go to their own dir only for the offline
recipe (`KICRAFT_LEGACY_PROJECTS_DIR`, default `~/.kicraft/projects-legacy`); the web path uses the
normal per-project workspace so the site can serve it.

### 5.6 Deploy
```bash
cd ~/KiCraft && deploy/restart-build-worker.sh && deploy/restart-web.sh
curl -sf -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/
tail -2 logs/kicraft_build_worker.log      # expect: [build-worker] ready (max 1 concurrent build(s))
```
Both scripts restart from the repo root, so a local commit is live after them — no `pip install`
needed unless dependencies changed. **Both** services must be restarted after a change that touches
the build path (the build dispatch lives in the worker).

---

## 6. Artifacts and baselines

| path | what |
|---|---|
| `logs/self_eval/movea_a1_current_20260919T1431Z` | **the frozen baseline.** 102 attempts, 7 committed, 3 briefs ≥2 repeats, 5 fab-ready runs / 4 briefs. Never overwrite. |
| `logs/self_eval/movea_a3_precontract_b4b8be5` | 09-15 tree, 13/34 fab-ready — the pre-contract reference |
| `logs/self_eval/movea_a2_legacy_bc6a2f8` | August pipeline, 21/34 — the option-3 value |
| `logs/self_eval/movea_a4_recordpatch` + `.patch` | the RECORD preview: the patch this session landed, and the measurement that says do not widen it |
| `logs/self_eval/opt1_designonly_20260920*` | this session's screening (running) |
| `logs/self_eval/reference_replay_{after_1a,final}.log` | the two green gate runs (31/34) |
| `logs/self_eval/movea_tools_20260919/` | session 1's tooling: runner, census, bundle expansion, per-arm report |
| `logs/self_eval/opt1_designonly_20260920_run.sh` | this session's resuming runner (adapt for the next batch) |

---

## 7. Still owed to the operator

1. **Option 3, in or out, and manual or automatic.** Manual selection is what shipped. If a
   current-pipeline design dies, should it be retried on the legacy pipeline automatically? That
   means two configurations producing boards with no human choosing — the plan's default is no.
2. **How long does the legacy fork live?** It buys 21/34 today and receives none of the fixes from
   options 1–2. Carrying it forever is a cost that should be a decision.
3. **Should paid campaigns run on this production box at all?** Approved as of 2026-09-20, with the
   caveats in §5.3 (a memory ceiling that can kill a campaign every ~90 min, and minutes of site
   unresponsiveness at the plateau). The alternative is a second host.
4. **Install the build-slot sweep timer** (needs root):
   `sudo cp deploy/kicraft-build-slots.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now kicraft-build-slots.timer`
5. **Push the branch?** 19 local commits are ahead of `origin`; the box serves them locally, but
   they exist in one place only.
6. **Whether to keep the campaign's cheap instrument** (the real-draft replay corpus): it screened
   move 1a for $0 and could screen every future architecture change the same way, but it is a
   throwaway script today (`/tmp`, not tracked). Say the word and it becomes a tracked tool.

---

## 8. Do not redo (measured nulls and self-inflicted traps)

- **Widening the RECORD list.** Measured null in session 1: the 12 attempts the downgrade rescued
  all died one stage later at the BOM. The list is §4.3's, minus `unknown_part_refused` (§2.3).
- **Loosening the reference row to make the gate green.** The corpus arbitrates (`834752a`); the
  contract was right and the row was stale.
- **One-brief-at-a-time wording fixes or error-class aliases.** Six such commits moved nothing.
- **Chasing the model.** The August pipeline hits its era's numbers with today's model; the model is
  not the constraint.
- **Trusting the stored terminal diagnosis without expanding it.** Every census in this plan's
  history that skipped the bundle expansion ranked the wrong code first (§2.1).
- **Running two campaigns at once, or at `--parallel 3`.** Both have already cost a session.
