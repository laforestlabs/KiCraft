# Next steps for the constructive architecture slot (2026-09-14)

**For the implementing agent.** The parent plan is
`docs/plans/architecture-constructive-slot-2026-09-14.md`; its §10 is the implementation and
measurement record you are extending. Read §1, §6, §8, §9 and §10 of that file before changing
anything: the diagnosis, the pre-registered endpoint, the non-goals and the measured verdict are all
there, and this file assumes them.

**Where things stand.** Stages 0 and 1 of the parent plan are landed and measured (commits
`468a178`, `59622bc`, `a795ff3`, `6462cf2`, `4b24977`, docs `4f18490`). The intent-shaped
architecture slot exists behind `KICRAFT_ARCHITECTURE_SLOT` (default `explicit`, i.e. the historical
provider contract is untouched). The parent plan's kill criterion — *first-draft acceptance above
50 %* — measured **0 %** across five interleaved live blocks, so stages 2 and 3 (deleting the
bookkeeping validators, collapsing the alias tables) are **not** landed and must not be started
until this plan's Task C says so. Live spend so far: $1.54 of the parent plan's $3 ceiling.

**Implemented (commit `b667c6e`), and where its record is.** Tasks A (the endpoint split), B1 (the
typed regulator rating) and B2 (the load current is a question) are landed; Task B3's evidence is
recorded and no code changed for it; Task C's measurement and the stage-2 verdict are written up in
the parent plan **§10.8** (what landed, with the re-derived attribution and the class-by-class
evidence) and **§10.9** (the offline replay, the pre-registered live block, and the verdict). The
two operational consequences of B2 — a board can now park on the load-current question, and the
`stage_driver replay` rehearsal prints that question — are folded into §6.4/§6.5 below.

**Task C verdict: stage 2 is unlocked.** Ran as written, the §5 block parks both arms (17 of 24 runs
in 6 iterations, commits 2 stock-less), so it was stopped early and the block re-run with
`tools/ladder_experiment.py --unattended` — the same protocol for a driver with no user attached,
N=20 per frozen board per arm, 80 runs, $0.8913, all at `repo_head b667c6e`. There, the intent arm's
first drafts are contract-clean **22/38 (58 %)** against stock's **0/40** and it commits **35/40
(88 %)** against stock's **14/40 (35 %)** — both halves of the pre-registered rule pass. The strict
zero-correction endpoint stays low (3/38) because the two statements §4 decides are still diagnosed
on nearly every draft; that is the split Task A exists to show, and both are now decided (B1 typed,
B2 asked). The leading *residual* class in the intent arm is `conflicting_port_binding` (15, the
model naming one producer port for two rails) with `usb_connector_supply_unknown` (12) — §2's method
reads those from the drafts next. Stage 2 itself (the §5 deletions) is **not** started here: this
plan's Task C decides, the parent plan's stage 2 is its own measured increment.

---

## 1. What the measurement actually showed (this is your starting evidence)

Offline replay of 263 saved drafts (`tools/ladder_experiment.py --replay-corpus`, free):
`0/254 → 214/256` accepted after projection + derivation; `unknown_recipe_port_net` 165 → 0,
`missing_interface_port` 31 → 0, `missing_recipe_port` 51 → 1, `multiple_recipe_contracts` 51 → 0.

Live, interleaved, both frozen boards (`~/.kicraft/projects/1/{824,825}`), real provider:

| block | source | arm | commits | `first_draft_accepted` | first drafts with **no contract refusal** |
|---|---|---|---|---|---|
| 1 (N=20) | `468a178` | stock | 6/20 | 0/20 | 0/20 |
| 1 (N=20) | `468a178` | intent | 7/20 | 0/20 | 7/20 |
| 2–5 (N=40) | …`4b24977` | stock | 13/40 | 0/40 | 0/40 |
| 2–5 (N=40) | …`4b24977` | intent | **40/40** | 0/40 | 19/40 (8/10 in block 5) |

Pooled blocks 2–5: intent commits **47/50** at `$0.0073`/run; stock **21/50** at `$0.0161`/run.

The important number is the last column: **the wiring half of the problem is fixed** (stock never
produced a contract-clean first draft in 60 runs; the intent slot did in 26/60). The 0 % headline
comes from somewhere else. Every intent run in blocks 2–5 spent one *semantic repair round* on the
intent-level checks §5 of the parent plan deliberately keeps:

| semantic code | emissions across 40 intent runs | what it wants |
|---|---|---|
| `architecture_mcu_regulator_incomplete` | 76 | the slot's `topologies` text to name a converter for the 3.3 V rail **and** a current rating ≥ 1 A |
| `architecture_external_load_current_unspecified` | 68 | the current the board supplies to the display / LED string |
| `architecture_power_block_as_sheet` | 22 | where a distribution block physically lands |

Neither number can be computed by the compiler: the recipe registry carries parts and stability
assertions but **no current rating**, and the load current is a fact only the brief, the user, or
the model has. The measurement therefore cannot distinguish "the slot is wrong" from "the design
statement is missing" — which is Task A.

**One live finding, already fixed (see §1.1).** Enabling the slot on the production
box surfaced a *pre-existing* hole the constructive slot made reachable: a BOM whose every sheet is
recipe- or lowerer-covered has no provider call, and §9.33 then failed outright (0 attempts) when a
recipe's identity (`HUB75-SN74HCT245`) is not the part it ships (`SN74HCT245PWR-JSM`). The BOM now
records the recipe→part pairing on the part itself, and the same brief runs all five stages in 86 s
for $0.013 with BOM and wiring fully deterministic. Read §1.1 before touching the BOM stage.

### 1.1 The BOM hole the live test found (and its fix)

Reproduce: any architecture whose sheets are all covered by recipe selections or lowerers. Before
the fix, `stage_contracts._expand_bom_groups` assembled the BOM from recipe expansions and the §9.33
gate (`check_spec_named_mpn_substitutions`) reported *"spec/architecture names 'HUB75-SN74HCT245'
but the BOM neither ships it nor records a substitution"*. With zero work units there was no
provider call to write the ledger, so the stage died at `attempts=0` with no recovery. The old
behaviour survived only because a model-owned requirement (a speaker amplifier, in the 2026-09-13
boards) kept an LLM unit alive to paper over it.

Fix, in `stage_contracts._recipe_parts_with_identity`: each recipe expansion's parts carry the
curated identity the design named, as a `sourcing_note` on the part that embodies it
(`curated recipe hub75-sn74hct245-interface@1 ships SN74HCT245PWR-JSM for identity
HUB75-SN74HCT245`). Nothing is substituted — the recipe *is* the part — so this is the honest
ledger §9.33 asks for, not a silenced failure. Regression test:
`tests/test_stage_driver_prompt_examples.py::test_recipe_covered_bom_records_the_curated_identity`.

---

## 2. Ground rules (do not break these)

1. **No new validator, no new completion, and no new contract paragraph per corpus defect.** That
   loop is what the parent plan exists to stop. Every change here is either a *measurement* change
   (Task A), a *typed-data* change replacing a prose check (Task B), or a *question* where the
   missing fact is the user's (Task B2).
2. **Never let the compiler assert a fact it cannot source.** No invented net names, no invented
   current ratings, no "the design surely means X". If the model did not say it and the registry
   does not hold it, refuse or ask — never fill it in.
3. **Do not weaken a gate to make a number move.** The parent plan's §8: no raising the correction
   budget, no "better messages", no prompt-example tuning, no canary/deploy-gate changes.
4. **Measure before you claim.** Any behaviour change needs the interleaved live protocol of §3
   below, and the offline replay (`--replay-corpus`) must not regress.
5. **One class at a time, read from the drafts.** The method that produced the last five fixes: take
   the rejected first draft out of the event stream, name what the model *did* say, and derive from
   that. `/tmp/slot-ab*` holds 120 runs of evidence; a fresh run's draft is in
   `<run dir>/events.jsonl` as `answer_delta` chunks followed by a `retry` event carrying the
   `diagnostic`.

Where the relevant code lives:

| concern | location |
|---|---|
| intent slot + derivation | `kicraft/design/architecture_intent.py` |
| model-facing contract text | `.agents/skills/kicraft/stages/architecture_intent.md` (+ `_WORKED_EXAMPLES["architecture_intent"]` in `kicraft/server/stage_prompts.py`) |
| flag + settings | `kicraft/server/config.py` (`architecture_slot`, `parse_architecture_slot`, `KICRAFT_ARCHITECTURE_SLOT`) |
| slot schema, reader, refuse→retry bridge | `kicraft/server/stage_contracts.py` (`_slot_response_schema`, `_response_schema`, `spec_name`, `_intent_shaped`, `_derive_intent_payload`) |
| telemetry + prompt assembly | `kicraft/server/stage_runtime.py` (`stage_telemetry`, `finalize_stage`, `drive_stage`) |
| intent-level checks | `kicraft/design/stage_semantics.py` (`_architecture`, `architecture_power_requirement_diagnostics`) |
| recipe data | `kicraft/design/recipes/models.py` (`RecipeDefinition`), `wave_b_power.py` / `wave_c_interfaces.py` |
| tests | `tests/test_architecture_intent.py`, `tests/test_stage_driver_prompt_examples.py`, `tests/test_design_recipes.py`, `tests/test_stage_semantics.py` |

---

## 3. Task A — split the endpoint so it measures the slot, not the repair policy

**Why.** `first_draft_accepted` as emitted today is `ok and attempts == 1`. A semantic repair round
is a provider call, so it counts as a correction — and since *every* architecture run spends one
repair round for the two statements in §1, the endpoint reads 0 % for any slot, including a perfect
one. The parent plan's §6 primary endpoint is unusable until the correction kinds are separated.

**What to change** (all in `kicraft/server/stage_runtime.py`):

1. `stage_telemetry(...)` (currently `ok, attempts, defect_codes, outcome`) gains explicit inputs for
   the two correction kinds and emits:
   - `contract_rejections: int` — provider attempts rejected by the *contract* path (a `retry` event
     carrying a `diagnostic`, i.e. the reader refused the draft: bookkeeping, recipe resolution,
     missing ports, intent refusals);
   - `semantic_repair_rounds: int` — repair rounds the driver spent because `diagnose_stage` returned
     a `repair_required`/`fab_gate` finding (the loop that consumes `provider_call_budget`);
   - `first_draft_contract_clean: bool` — `ok and contract_rejections == 0 and attempts >= 1`
     (the first draft reached the commit gates without a contract refusal, whether or not a semantic
     round followed);
   - keep `first_draft_accepted`, `drafts`, `defect_codes[]`, `declared_interfaces`,
     `unknown_part_refused` exactly as they are (their consumers are the parent plan's §6 numbers and
     the operator harness).
2. Feed both counters where the drive already knows the answer, so no new detector is introduced:
   - contract rejections: the accumulator `defect_codes` in `drive_stage` is appended from
     `_diagnostic_codes(last.get("diagnostic"))` on the schema path (~line 3976) and from
     `provider_diagnostic_codes` for semantic findings (~line 4317). Split these two appends into
     two accumulators (`contract_codes`, `semantic_codes`) — `defect_codes` stays the union so no
     existing reader breaks — and count a contract rejection once per *rejected attempt*, not per
     code (increment where the retry event is emitted, ~line 3990).
   - semantic rounds: `semantic_repair_rounds` and `attempts` already exist as locals of `drive_stage`
     (the loop increments `attempts` before `run_serialization_recovery` at ~line 4294); pass
     `semantic_repair_rounds` through the two `finalize_stage` calls that already receive
     `defect_codes=tuple(defect_codes)` (~lines 4596 success, ~4721 terminal). For the work-unit
     path (`_drive_work_unit_stage`, ~line 2810/2940/3140) leave the default so those `stage_done`
     events keep their current shape.
3. Emit the same three fields on the events stream (they are `stage_done` fields; nothing else is
   needed).

**Acceptance criteria**

- `tests/test_architecture_intent.py::test_rejected_first_draft_publishes_its_defect_class` and the
  new-driver test still pass; add one scripted-client test per case with the `_ScriptedClient` from
  `tests/test_stage_driver_retry.py`:
  - a half-USB intent first draft (rejected by the reader) → `contract_rejections == 1`,
    `first_draft_contract_clean is False`, `first_draft_accepted is False`;
  - a first draft the reader accepts but a semantic finding follows → `contract_rejections == 0`,
    `first_draft_contract_clean is True`, `first_draft_accepted is False`,
    `semantic_repair_rounds == 1`.
- Re-deriving the existing evidence must reproduce the parent plan's §10.4 attribution table:
  ```
  .venv/bin/python - <<'PY'
  import json, pathlib
  for block in ("slot-ab", "slot-ab2", "slot-ab3", "slot-ab4", "slot-ab5"):
      for line in (pathlib.Path("/tmp") / block / "runs.jsonl").read_text().splitlines():
          row = json.loads(line)
          events = [json.loads(l) for l in pathlib.Path(row["artifacts"]["events"]).read_text().splitlines()]
          done = next(e for e in events if e.get("kind") == "stage_done" and e.get("stage") == "architecture")
          print(block, row["label"], row["board"], {k: done.get(k) for k in
                ("first_draft_accepted", "first_draft_contract_clean", "contract_rejections", "semantic_repair_rounds")})
  PY
  ```
  (the four new fields are absent for runs made before your change — that is expected; the *old*
  runs' `retry`/`stage_diagnostic` events are what the table above was computed from.)
- Paste the resulting table into the parent plan §10.4/§10.5, replacing the hand-derived
  attribution with the instrument's own numbers.

**Do not** change the meaning of `first_draft_accepted` (it is the pre-registered endpoint), and do
not count a semantic repair as a contract rejection.

---

## 4. Task B — decide the two design statements the model keeps omitting

Both must be decided on their merits, with the parent plan's §9 risk row in mind ("the derivation
hides a real design error"). Neither may be silenced by having the compiler assert something it
cannot source.

### B1. `architecture_mcu_regulator_incomplete` (the 3.3 V converter's rating)

Today the check (`kicraft/design/stage_semantics.py`, the `_architecture` block that starts
`if re.search(r"esp32[- ]?s3", intent_text...)`) joins the `topologies` entries that mention
`3v3|3.3v|3 v3`, requires a regulator term (`ldo|regulat|buck|convert`) plus an ampere figure ≥ 1 A,
and additionally requires a non-MCU sheet whose name/function carries both. That is a prose regex
over the model's own words, and the model does not reliably write it.

**Preferred fix — typed data, not prose.** The fact the check needs is a property of the *part*:

1. Add a reviewed rating to the recipe model (`kicraft/design/recipes/models.py`), e.g.
   `rated_output_current_a: float | None = None`, documented as "the regulator's continuous output
   current from its datasheet, the source being the recipe's own `source_documents`".
2. Fill it for every regulating recipe, from the datasheet the recipe already cites. As of
   `4f18490` that is at least: `tlv62569-3v3@1`, `ap63203-3v3@1`, `me6211-3v3@1`, `mcp1700-3v3@1`,
   `ams1117-3v3@1`, `ap63205-5v@1`, `tps54331-adjustable@1` (and any boost/charger recipe that
   generates a rail). Where a datasheet rating is conditional (thermal, inductor-limited), record
   the conservative continuous figure and say so in the recipe's `electrical_assertions`.
3. Rewrite the check to read typed data instead of text: for each declared rail whose voltage is
   ~3.3 V, find the requirement that generates it (the `ports` binding on that net whose recipe port
   direction is `power`/`output`, or the `power.rails[net].from` reference the intent slot emits),
   resolve that requirement's recipe, and require `rated_output_current_a >= 1` for an ESP32-S3
   design plus a producer on its own non-MCU sheet. The check keeps its meaning ("this rail must be
   generated by a converter rated for the MCU") and stops depending on how the model phrases things.
4. If the *derivation* should state the topology line (`{"POWER": "TLV62569DBVR, VBUS to 3.3V, 1 A
   (recipe rating)"}`), it may — because the number is now sourced and the assumption names the
   source. This is optional; do it only if Task C shows the check still fires on drafts whose design
   is fine.

**Fallback (only if a rating cannot be sourced honestly).** Keep the check but stop spending a repair
round on a fact the model has to guess: turn the missing-rating case into one `blocking` question to
the user ("What output current must the 3.3 V rail supply?"), following the mechanism in B2.

**Acceptance.** Unit tests in `tests/test_stage_semantics.py`: a design whose 3.3 V rail is generated
by a recipe with a reviewed rating ≥ 1 A passes with no diagnostic, regardless of how `topologies`
is phrased; a design whose rail is generated by a 0.5 A regulator, or by no named part, still fires.
No prose regex remains in this check. `architecture_rail_source_unspecified` and
`architecture_mcu_supply_rail_missing` stay as they are (they also check intent).

### B2. `architecture_external_load_current_unspecified` (the load's current)

This is a **user fact**: the brief "USB-C PD 5 V to an ESP32-S3 driving a HUB75 display, plus an LED
string and a speaker output" never says how much current the display or the string draws. The
model-facing contract already instructs the model to return one `blocking: true` question when the
brief does not supply it; live drafts instead omit it and get repaired.

**Change:** when this finding fires and the number is absent from both the brief and the recorded
answers, the stage must **park with one question** rather than spend a repair call — the same
`needs_input` path the driver already uses for a model-authored question
(`_questions_need_input`, `attach_questions`, and the `questions` progress event), so the user sees
the question in the UI. Keep today's repair behaviour as the fallback when questions are disabled
(`allow_questions=False`, i.e. the non-interactive default instruction), and keep
`architecture_external_load_source_capacity_unspecified` /
`architecture_external_load_source_has_no_headroom` as findings (they check the *design's* answer,
not the user's).

**Acceptance.** A scripted run whose brief lacks the current parks with exactly one question, one
provider call, no repair round, and no commit; a run whose brief or answers carry the number does not
park. `tests/test_stage_driver_retry.py` has the question-parking harness to copy. Note in the
parent plan's §10 that this makes some briefs *ask* instead of guessing — that is the intended
behaviour, not a regression in `first_draft_accepted`.

### B3. `architecture_power_block_as_sheet` (22 emissions) — evidence only

Do **not** act on this one yet. Read the drafts that trigger it first and write down what the model
actually said; if it turns out to be the same class as B1/B2 (a statement, not bookkeeping), fold it
into the same decision. If it is bookkeeping, it belongs to the derivation, not to a new rule.

---

## 5. Task C — re-measure, then decide stages 2–3

**Protocol (the parent plan's §6, unchanged):** frozen source (commit your work first — the harness
records `repo_head`/`repo_dirty`), interleaved arms and boards, one run per invocation, spend guard.

```bash
cd /home/kicraft/KiCraft && set -a && . ./.env && set +a
for i in $(seq 1 20); do for board in 825 824; do for arm in stock intent; do
  KICRAFT_ARCHITECTURE_SLOT=$([ "$arm" = intent ] && echo intent || echo explicit) \
  .venv/bin/python tools/ladder_experiment.py --arm stock --label "ab_${arm}" \
    --state "$HOME/.kicraft/projects/1/$board/.kicraft/state.json" \
    --runs 1 --budget 0.25 --out /tmp/slot-ab-next
done; done; done
```

80 runs ≈ 70 min ≈ $0.6. Stop early if cumulative `cost_usd` in `/tmp/slot-ab-next/runs.jsonl`
exceeds $1.50 (`spent` in the manifest tracks the process guard too).

Also re-run the offline replay and paste both tables into the parent plan §10:

```bash
.venv/bin/python tools/ladder_experiment.py --replay-corpus /tmp/ladder-exp
```

**Pre-registered decision rule (do not change it after seeing the numbers):**

- Stage 2 (delete the §5 list of bookkeeping validators, drop the legacy explicit slot) may start
  **only if** the intent arm's `first_draft_contract_clean` ≥ 50 % **and** its commit rate ≥ stock's
  commit rate in the same block.
- Otherwise: record the negative result in §10, keep the flag default `explicit`, and take the
  leading residual class back to §2's method. A negative result is a deliverable; do not tune the
  gate or the prompt to pass it.
- Whatever the outcome: report `contract_rejections`, `semantic_repair_rounds`, commit rate and cost
  per arm, and the class list per arm, in the same table format §10.4 uses.

---

## 6. Operator section — deploying, and testing live

Who does what: the implementing agent runs **6.2** first (deploy the landed work, no canary) so the
operator can start testing immediately; Tasks A–C then continue on top. §6.3–§6.6 are for the
operator's live test.

### 6.1 The switch

The intent-shaped slot is off by default. It is selected per process by:

```
KICRAFT_ARCHITECTURE_SLOT=intent      # explicit (default) | intent
```

`kicraft/server/config.py:load_dotenv` uses `os.environ.setdefault`, so a real environment variable
beats `.env` — which is how the single-board rehearsal below can flip it without touching `.env`.
Only the web process drives LLM stages; the build worker runs the deterministic build, so the flag
is irrelevant to it (restart both anyway, via the canonical script).

### 6.2 Deploy (no canary)

The commits are already on this box's checkout, so a deploy is a dependency check plus a restart.
Per `AGENTS.md`:

```bash
cd /home/kicraft/KiCraft
.venv/bin/pip install -e ".[server,design]"
./deploy/deploy-production.sh          # restarts both services + verifies web 200 and [build-worker] ready
```

`deploy/verify-design-canary.sh` (the real-provider 34-brief gate) is deliberately **not** part of
this and stays unrun. Verify by hand:

```bash
curl -sf -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/     # 200
tail -5 logs/kicraft_build_worker.log                                # [build-worker] ready
```

Performed on 2026-09-14 against `4f18490`: web restarted (new pid), build worker restarted (new
pid), `production health passed`. `KICRAFT_ARCHITECTURE_SLOT` was **not** set in the web process's
environment afterwards, so the live behaviour is the historical `explicit` slot until §6.3 is done —
the deploy alone changes nothing for users.

### 6.3 Turn the slot on for your own boards

1. Add the line to `.env` (mode 600, next to the other `KICRAFT_*` settings):
   `KICRAFT_ARCHITECTURE_SLOT=intent`
2. Restart the web app so the flag is read: `./deploy/restart-web.sh` (verifies it serves before
   returning).
3. Confirm it took effect on the next board: the architecture contract name in that run's events is
   `kicraft_architecture_intent_response_v2` (the explicit slot is `…_response_v2`).

### 6.4 A one-board rehearsal first (≈ $0.01–0.02, ≈ 1 minute)

Before spending a full board, drive the architecture stage alone from a frozen pre-architecture
state, with the slot forced for that one command:

```bash
cd /home/kicraft/KiCraft && set -a && . ./.env && set +a
KICRAFT_ARCHITECTURE_SLOT=intent .venv/bin/python -m kicraft.server.stage_driver \
  replay --state ~/.kicraft/projects/1/825/.kicraft/state.json \
  --stage architecture --max-retries 3 --budget 0.25 \
  --trace-jsonl /tmp/slot-rehearsal.trace.jsonl
```

Expect: **either** the stage commits (attempts 1–3 with the typed 3.3 V rating landed) **or** it
parks with one question — the external 5 V load current, which neither this brief nor the recorded
answers state. The CLI prints the question and exits nonzero; the park is the intended answer for a
fact only the user has, and the rehearsal is a probe, not the answering path: answer it on the live
board, where the UI renders the question and re-drives the stage. A brief that already states the
current never parks. The first attempt is no longer redrafted for the two design statements §10.5
lists; the committed slot has the same shape as before, so BOM/wiring/layout run unchanged.

### 6.5 What to watch on a live board

Per run directory (`~/.kicraft/projects/<account>/<board>/events.jsonl`), the architecture
`stage_done` event now carries:

| field | meaning |
|---|---|
| `drafts` | provider calls that drafted a slot |
| `first_draft_accepted` | the first draft committed with **zero** corrections (the pre-registered endpoint) |
| `first_draft_contract_clean` | the first draft reached the commit gates **without a reader refusal** — true even when a semantic repair followed |
| `contract_rejections` | reader refusals, counted once per rejected attempt |
| `semantic_repair_rounds` | repair calls spent on a design statement |
| `defect_codes[]` | the blocking classes the drafts were rejected for |
| `declared_interfaces` | parts whose pin functions the model declared instead of a curated recipe (a claim — the review says so) |
| `unknown_part_refused` | the §4.3 refusal count (an unsupported part, counted apart from instability) |

A **park** publishes no `stage_done`: the run's status is `awaiting_input`, the question is in
`state.json`'s `open_questions`, and the answering UI re-drives the stage. The three new counters
appear only for the stages whose driver knows the two correction kinds (architecture, intent,
functional_spec), never as a false zero on a work-unit stage.

```bash
# the counters for the newest board, and what caused each redraft
python3 - <<'PY'
import glob, json, pathlib
run = sorted(glob.glob(str(pathlib.Path.home() / ".kicraft/projects/*/*/events.jsonl")))[-1]
print(run)
for line in open(run):
    e = json.loads(line)
    if e.get("kind") == "stage_done" and e.get("stage") == "architecture":
        print({k: e.get(k) for k in (
            "attempts", "first_draft_accepted", "first_draft_contract_clean",
            "contract_rejections", "semantic_repair_rounds", "defect_codes",
            "declared_interfaces", "unknown_part_refused")})
    elif e.get("kind") == "retry":
        print("  redraft:", (e.get("diagnostic") or {}).get("code"), (e.get("diagnostic") or {}).get("message", "")[:120])
    elif e.get("kind") == "question":
        print("  parked:", [q.get("text") for q in e.get("questions") or []])
PY
```

`python -m kicraft.cli.triage stages KC-XXXXXX` (board code, `uid/pid`, or a run path; no argument
means the most recent run) gives the same stage verdict from the CLI.

### 6.6 Revert

Delete or comment `KICRAFT_ARCHITECTURE_SLOT` in `.env` and run `./deploy/restart-web.sh`. The
default is `explicit`, so nothing else needs undoing; the five implementation commits are inert
without the flag. Reporting a problem: the run's `events.jsonl` plus the board code is enough —
§2's method (read the draft, name the class) works from that alone.
