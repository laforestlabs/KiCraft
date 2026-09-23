# Next session: live kicraft.io run — blockers, runbook, decisions

**2026-09-23.** Plan for the next session, written after the Surprise-me
walkthrough that produced commit `6166b68`. Two parts: what to fix first (A),
how to drive the live product again (B), and the one configuration decision to
settle (C).

Deliverable of the previous session (context, not work to redo):

- `6166b68` — eight defects fixed with regression tests: raw machine dumps shown
  as the user-facing failure; the dead "Rebuild board" action; a live run's row
  marked `interrupted` mid-flight; the screen-reader alert; **legacy designs were
  never built**; "Rebuild board" could not start a *first* build; §9.9 rejected a
  pinless mounting-hole sheet; legacy snapshots were synthesized by the pinned
  tree. Suite: 336 passed / 4 skipped. Deployed and healthy.
- Live evidence: four surprise runs, all `failed` — `KC-83KWVK` (wiring `9.15`),
  `KC-9N7B8F` (wiring `9.11`), `KC-FSXF35` (BOM `9.33`), `KC-CG58R4` (BOM `9.33`
  → Continue → wiring committed → build → `rc=6 no routed parent`). ~$0.26 LLM.
- Two further runs landed on the box afterwards (owner: the admin account, so
  the operator's own testing, both `legacy`): `KC-GM5UC2` (11:32) failed at
  wiring `commit_rejected` ×5; `KC-Y86YCJ` (11:58) failed at architecture
  `commit_rejected` ×4. Still no `ok` board today — the blockers in A1/A2
  reproduce outside the Surprise-me path too.
- Open blockers and their evidence: `docs/plans/terminal-opening-datum-2026-09-23.md`
  (unreviewed terminal footprint) and `docs/plans/stage-auditor-luna-2026-09-23.md`
  (design-stage contracts; peer workstream, uncommitted).

---

## A. Blocker queue

### A1. Review the terminal opening datum  *(no deploy needed to test)*

**Symptom.** Parent compose rejects every candidate before routing:

```
[candidate-search] cand=3 rejected before routing: connector_orientation_unmeasured:J2
error: failed to compose subcircuits: every one of 4 candidate(s) was rejected
       before routing for connector orientation: connector_orientation_unmeasured:J2
```

**Cause.** `KC-CG58R4`'s J2 is `screw-terminal-5mm-4p-128:WJ128V-4P-5.0-14-00A`,
chosen by the pinned legacy backend. `kicraft/parts_library/footprint_opening.py`
answers "which way does the wire enter" only from reviewed family metadata keyed
on the exact footprint name, and refuses to guess (see its docstring, the
KC-DZQ76R story). The WJ128V family is not in `_REVIEWED_OPENINGS`.

**Do.** Read the part's STEP model / drawing for the wire-entry face, then add a
row per vendored name (siblings live under `~/.kicraft/parts/`:
`screw-terminal-5mm-4p-128`, `-4p-wj128v`, `-5p-wj128v`), citing the evidence in
the comment exactly as the MKDS entries do. Full measurements and the
why-not-a-heuristic reasoning are in
`docs/plans/terminal-opening-datum-2026-09-23.md`.

**Verify.** Re-run the compose command from that doc. Expect the
`connector_orientation_unmeasured` line to disappear and the parent round to
reach routing. Then rebuild the board in the UI (Rebuild board on `KC-CG58R4`)
and read the next gate, if any.

### A2. Design-stage contracts  *(peer workstream — coordinate before touching)*

Surprise briefs fail in the LLM stages, so the build never runs:

| stage | failure_kind | diagnostic / gate |
|---|---|---|
| architecture | `commit_rejected` | Two shapes, both contract/prompt alignment: (a) `source_obligation_not_retained` — obligation `four-layer-stackup` (PCB layer count = 4) dropped from `requirements[].obligations` (reproduced on the LM358 brief via `stage_driver run`); (b) `slot validation failed: 3 validation errors for Architecture` — the model emitted `sheets` / `power_nets` / `inter_sheet_nets` **nested under `architecture`** while the committed slot wanted them at the top level, plus `InterSheetNet 'USB_D+' needs at least 2 endpoints` and one `no JSON in reply`. `triage` on `KC-Y86YCJ`: 4 attempts, 4 *different* rejections — died on breadth, gates still moving |
| bom | `commit_rejected` | `9.33 spec-named part accountability` — a brief-named part (e.g. TLV9062) missing from the BOM |
| wiring | `commit_rejected` | `9.11 net coverage` (no connections) / `9.15 no dangling signal nets` (single-pin net). Reproduces on `KC-GM5UC2` (5 attempts) |

Read the per-run detail with the repo's own triage before touching a contract:

```bash
.venv/bin/python -m kicraft.cli.triage run    KC-Y86YCJ
.venv/bin/python -m kicraft.cli.triage stages KC-Y86YCJ   # attempts, kinds, diagnostics
.venv/bin/python -m kicraft.cli.triage scan                # systematic vs per-design
```

**Do.** Replay the failing stage against the frozen state, N-of-3 (LLM verdicts
are stochastic), then fix the contract that refused it — not the prompt:

```bash
cd ~/KiCraft
.venv/bin/python -m kicraft.server.stage_driver replay \
  --state <run>/.kicraft/state.json --stage architecture --budget 0.25
```

**Note.** `config.py` / `stage_runtime.py` / `tests/test_stage_driver_retry.py`
were modified but **uncommitted** when this plan was written; that is another
session's in-flight work. Check `git status` and coordinate before editing them.

### A3. Then re-run the live flow (section B)

Pick a **terminal-free** surprise draw if A1 is still pending — the corpus varies
(headers, battery packs, trim pots), and the plumbing fixed in `6166b68` is
enough for those to reach the fab package. Record which seed/draw was used.

---

## B. Runbook: driving the live product again

Everything below was exercised this session; the commands are verbatim.

### B0. Pre-flight (1 minute)

```bash
cd ~/KiCraft
git log --oneline -1 && git status --short          # know what is deployed
curl -s -o /dev/null -w "web:%{http_code}\n" http://127.0.0.1:8080/   # expect 200
tail -1 logs/kicraft_build_worker.log                # expect "[build-worker] ready"
python3 -c "import sqlite3;c=sqlite3.connect('/home/kicraft/.kicraft/accounts.db');\
print([tuple(r) for r in c.execute(\"select id,board_code,status from projects where status='running'\")])"
```

If code changed: `./deploy/deploy-production.sh` (restarts both services, then
verifies web 200 + worker ready). **A deploy logs everyone out** — re-login.

### B1. Log in as the smoke user

- URL `https://kicraft.io/login`
- `live-smoke-1790139495@kicraft.io` / `KiCraftLiveSmoke!2026`
  (email verification was completed out-of-band via the DB flag by the session
  that created it; the UI may still show "Resend verification link".
  `is_admin`/quota gates do not apply: the account is a plain Free-tier `user`.)
- NiceGUI needs a moment: wait ~2.5 s after `goto` before touching widgets, or
  the click lands before handlers are wired.
- Quota: Free tier = 1 design/week. `running` / `ok` / `awaiting_input` hold the
  slot; a `failed` run frees it. "0 of 1 left" while a run is open is normal.

### B2. Start a run the way a user does

- Open `/` → click **Surprise me** (it composes a brief from the persistent seed
  counter and starts the run immediately, no confirmation).
- Click by text; a synthetic `.click()` can be lost against the 1 s repaint's
  node replacement — if nothing happens, re-query and click once more.
- Capture: the brief text on screen, the board code (`KC-XXXXXX`), and the seed:

```bash
grep -n "\[surprise\]" ~/KiCraft/logs/kicraft_web.log | tail -1
```

### B3. Watch both surfaces

- **Page (what the user sees):** stage pills, headline, ISSUES with evidence.
- **DB (ground truth for status/timing):**

```bash
python3 -c "
import sqlite3
a=sqlite3.connect('/home/kicraft/.kicraft/accounts.db')
print('project', a.execute('select id,status,finished_at,cost_usd from projects order by id desc limit 1').fetchone())
print('job    ', a.execute('select id,status,rc from build_jobs order by id desc limit 1').fetchone())
s=sqlite3.connect('/home/kicraft/.kicraft/spend_ledger.db')
print('stages ', [tuple(r) for r in s.execute('select stage,ok,attempts,round(wall_s,1),round(cost_usd,4),failure_kind from stage_runs order by id desc limit 5')])"
```

Screenshot the failure/complete card as evidence, and grep invisible a11y text
when checking announcements: `[...document.querySelectorAll('[aria-live]')]`.

### B4. Watch-outs learned the hard way

- A frozen/closed tab does **not** stop the run (it runs in the web process /
  build worker). Reopen it from **My projects**; the row may legitimately stay
  `running`.
- The orphan reaper needs an *idle workspace* (300 s grace on the live journal)
  before it closes a `running` row, and logs `[reap] project <id> …` when it
  does — grep that before calling a row stuck.
- Restarting mid-run is safe (that is the point of the fixes), but a build in the
  host queue is what survives; the design stages do not.
- Never accept "it failed" without the gate sentence: the UI, the durable
  transcript (`events.jsonl`) and `state.json:stage_status[].diagnostics` all
  carry it now.

### B5. Evidence per run (paste into the session report)

board code · seed · brief · pipeline (`legacy`/`current`) · per-stage
`attempts/wall_s/cost` · the UI gate sentence · `generated/<stem>/` artifacts
(`.experiments/`, `build.log`) · build job rc · total cost.

### B6. Exit criteria

- **Success:** a FAB tab with a `kicraft_project.zip` download (routed, zero
  shorts, zero unconnected), or
- **Honest stop:** a specific gate sentence plus the owning module named, and the
  next action written down. Never a vague failure.

### B7. Verification without repeating the LLM spend

```bash
# a) A design stage, against a frozen state (LLM, capped, run N-of-3)
.venv/bin/python -m kicraft.server.stage_driver replay \
  --state <run>/.kicraft/state.json --stage <stage> --budget 0.25

# b) A whole brief on the CURRENT backend, in a throwaway workspace (LLM)
.venv/bin/python -m kicraft.server.stage_driver run \
  --brief "<brief text>" --workspace "$(mktemp -d)" --budget 0.25 --quality draft

# c) Placement/routing from a frozen workspace — always on a COPY, never in place
cp -a <run>/generated/<stem> /tmp/replay-$$ && \
.venv/bin/python -m kicraft.design.cli_app replay --project /tmp/replay-$$ --quality draft

# d) Parent compose alone (the terminal-mouth gate)
.venv/bin/python kicraft/cli/compose_subcircuits.py --project <gen_dir> \
  --parent / --spacing-mm 4.0 --pcb <gen_dir>/<stem>.kicad_pcb --route \
  --output /tmp/parent_pp.json --seed <round seed> --config <round_config.json>

# e) The repo's own failure triage, by board code
.venv/bin/python -m kicraft.cli.triage run    KC-XXXXXX
.venv/bin/python -m kicraft.cli.triage stages KC-XXXXXX
.venv/bin/python -m kicraft.cli.triage scan
```

---

## C. Decision to settle: which design backend is active

`~/.kicraft/routing.json` (admin-controlled, written 2026-09-21) currently pins:

```json
{ "active_profile": "luna", "pipeline": "legacy", "design_temperature": 0.0,
  "max_reasoning_tokens": 0, "max_tokens_per_call": 4096,
  "project_llm_budget_usd": 0.6 }
```

Consequences observed: `legacy` designs are authored by the pinned tree
(`bc6a2f8`) and then manufactured by the current tree, so their BOMs can select
parts outside the current *reviewed* set (A1) and their snapshots meet current
manufacturing gates (D7/D8 from the previous session). Every surprise brief this
session was authored by `legacy` and none produced a board.

**Decide:** keep `legacy` active while its outputs are reconciled with current
manufacturing (the terminal datum and any further snapshot gaps), or switch the
active pipeline back to `current` and compare the same brief on both backends
(`stage_driver run` is a per-brief A/B that does not touch production settings).

Whichever way it goes, do not flip a production setting to run an experiment —
use the `stage_driver run` harness for the comparison and change
`routing.json` only as a deliberate product decision.
