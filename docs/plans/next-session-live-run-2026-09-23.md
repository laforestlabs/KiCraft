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

**Status 2026-09-23 (second session): DONE.** The datum was reviewed from the
vendored 3D mesh, not from the two disagreeing footprint heuristics: a wire
enters `8.6..9.1 mm` from `+Y` (matching the datasheet's `6~7 mm` strip length)
and stops on the clamp cage after `1.2..2.0 mm` from `-Y`. `+Y = 90 deg`, marker
on the wire-entry wall at `y = +5.27 mm`. Landed in
`kicraft/parts_library/footprint_opening.py` for both vendored WJ128V footprint
names, with tests in `tests/test_screw_terminal_orientation.py`. Measurements,
the drawing cross-check and the verification caveat are in
`docs/plans/terminal-opening-datum-2026-09-23.md` → "Resolution".

Verification (no LLM, no cost — `cli_app replay --quality draft --seed 0` on a
cold copy of project 892's workspace, pre-promote seed restored):

| step | before the datum | after |
|---|---|---|
| `detect_opening_direction` on J2's board footprint | `None` | `90.0` |
| persisted `opening_direction` in the re-solved leaf | `None` | `90.0` (`in_plane`) |
| parent compose | every candidate rejected `connector_orientation_unmeasured:J2` | composed + routed |
| promote / verify | `rc=6`, `no routed parent` | `3/5 promoted routed parent`, `4/5 verify: shorts=0 unconnected=0 courtyard=0 keepout=0 traces=243` |

The same cold replay on the *frozen* (pre-fix) leaf artifacts still rejects
`connector_orientation_unmeasured:J2`, and clearing only the two WJ128V rows
in-process turns J2's measurement back into `None` — so the datum, not a
stochastic placement difference, is what unblocked it.

**Next gate found by that replay** (fab export still refused, so no board yet):
the design zones two *non-connector* parts at board edges — `RT1` (`3296W-1-103LF`
trim pot, zone `top`) and `Q2` (SOT-23 phototransistor, zone `bottom`) — and the
fab-facing gate treats both as directional connectors:

```
connector_orientation_unmeasured:RT1        # THT part, no measured mouth -> blocking
connector_misoriented:Q2(mouth 180deg vs bottom outward 90deg)   # heuristic mouth on a SOT-23
connector_stranded:Q2@-1.23mm(bottom)
```

Owning modules: `kicraft/autoplacer/brain/connector_edge_gap.py`
(`connector_facings`, `_access_only_connector`) and
`kicraft/design/cli_app.py::_connector_misoriented`. Their zones are access /
field-of-view hints (a trim pot the user turns, a phototransistor that must see
light); neither has an in-plane mating mouth to aim. The existing escape hatch
is `_access_only_connector`, an identity allowlist (coin cells, prog/debug
headers) — extending it to these parts, or gating the facing/stranding verdicts
on a connector identity rather than on THT evidence, is a policy decision for
the next session (a wrong allowlist silently skips a real connector's mouth
check). Note the *compose* gate did not block them: `_directional_edge_candidate`
requires `kind == "connector"`, which the fab gate has no equivalent of.

Note the `Verify` above as written is not sufficient: a leaf artifact persists
its components' `opening_direction`, so the compose CLI on the frozen workspace
re-reads the pre-fix `None` and still rejects. Re-solve the leaves
(`cli_app replay` / **Rebuild board**) to see the datum take effect; the plan's
B7(c)/(d) order should be reversed accordingly.

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
(2026-09-23, second session: that work is now reviewed and committed — `aa3193d`,
`a811624` — and the tree is clean, so A2 can be edited freely.)

**Status 2026-09-23 (second session): two contract defects fixed and measured;
the rest of A2 stands.**

The A/B in section C reproduced two of these classes on the *current* engine,
which is what made them fixable (`stage_driver run`, `--no-build`, reasoning
4096, three unseen briefs).

1. **A board-wide functional block no requirement can implement — fixed.**
   `validation._functional_block_sheets` demanded an implementing requirement
   from every functional block, while the sibling obligation-retention gate
   already exempted board-wide rows (`quantity` / `fabrication` / `negative`, and
   `quantitative` only when it measures the board outline). A brief whose "two
   mounting holes" arrives as a `quantity` row (no part) declared a `MOUNTING`
   block that *nothing* could implement, so every architecture attempt was
   refused with `functional block 'MOUNTING' has no implementation requirement on
   a sheet` and the design died at the stage. The gate now reads the same
   predicate (`obligation_requires_requirement_owner`); a block with no
   obligations at all, or citing a row the architecture does not carry, is still
   refused — the gate cannot be cleared by declaring a block and leaving the work
   out.
   *Measured* on the frozen beacon-brief state (`/tmp/ab/ws/curR-b3/.kicraft/state.json`,
   current tree, reasoning 4096): before, the mapping refusal was terminal (3
   attempts); after, `stage_driver replay` N-of-3 → **2/3 committed** (3 and 2
   attempts, $0.022/$0.013), and the single failure died on a *different*
   contract (`multiple_intent_contracts`).

2. **A saved commit rejection could not be re-read — fixed** (this is what
   blocked the A2 method itself). `_commit_rejection_diagnostics` wrote
   `gate_codes` and no `detector_version`, but `StageDiagnostic` forbids extra
   keys and required that field, so `ConversationState.model_validate` refused
   *every* state saved after a deterministic commit gate rejected a candidate —
   precisely the states `replay` exists to reopen. The first replay attempt died
   with `stage_status.architecture.diagnostics.1.detector_version Field required`
   and `gate_codes Extra inputs are not permitted`. The row now carries
   `DETECTOR_VERSION`, and `StageDiagnostic` reads additively (an unrecorded
   version is `None`, like `StageStatus`), so historical states load.

**Remaining A2 classes, measured and unfixed:**

- `multiple_intent_contracts` — a port-binding conflict the architecture keeps
  re-emitting (rail `+5V`: port `vbus` of `usb` already bound to `VBUS`); one of
  the three post-fix replays above.
- BOM work-unit refusals (`physical-obligation-unfulfilled`: the typed work unit
  demands a class-tagged real part — *"requires 1 real coupling-capacitor"* —
  while the model emits a generic `Device:C`). The stage reports `attempts=0`, so
  it never re-drives the unit with that feedback; one bounded retry is the
  candidate fix.
- The upstream identity-`conversion` modelling oddity (`I2C bus` → `I2C bus`)
  that the architecture stage is then asked to own. No deterministic rule
  separates that from a real conversion (level shifting, filtering), so the
  documented remedy is the stage-auditor's *rewriting auditor*
  (`docs/plans/stage-auditor-luna-2026-09-23.md`, workstream B), not a gate.
- The pinned engine's wiring **deficit park** — unchanged, and only
  production-visible while `pipeline: legacy`.

`config.py`, `stage_runtime.py` and `tests/test_stage_driver_retry.py` are now
committed, so the "coordinate before touching" caveat above is spent.

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

---

## Session log 2026-09-23 (second session)

### B. The live run — **a board shipped**

One Surprise-me run on the deployed code (datum included), driven through
`https://kicraft.io` as the smoke user:

| field | value |
|---|---|
| board code | `KC-MQNE7R` (project 896) |
| seed / brief | `seed=9` — *"A 24 V DC screw terminal to 5 V converter with reverse-polarity protection, a 6-pin 0.1 inch header, and a power LED. Use a four-layer stack-up."* |
| pipeline | `legacy` (`bc6a2f8` design + current manufacturing), `luna`, temperature 0, reasoning 0 |
| stages | intent 1/$0.0007/3.8s · functional_spec 1/$0.0012/7.6s · architecture 1/$0.0025/14.7s · bom 4/$0.0229/170.4s · wiring 3/$0.0079/22.8s — all `ok` |
| gate sentence | `4/5 verify: shorts=0 unconnected=0 courtyard=0 keepout=0 traces=158 components=16/16 util=36.1% aspect=1.46` |
| build job | `build_jobs.id=327 rc=0` (fab export ran) |
| artifacts | `generated/24V_TO_5V_CONVERTER/24V_TO_5V_CONVERTER_fab_20260923.zip` + `.receipt.json` (Gerbers per layer, PTH/NPTH drill + maps, CPL, BOM, STEP, 3D render) |
| UI | "Design complete", stage pills INTENT→FAB all green, **Download KiCad project (.zip)**, 3D render; elapsed 9m 49s |
| cost | **$0.0352** LLM (budget $0.60) |
| issues shown | warnings only, all resolved by retry: BOM §9.25 capacitor polarity ×2, §9.5 stock, wiring §9.11/§9.15 — no blocking finding |

So B6's success criterion is met for the first time in this stretch: a real
Surprise-me request produced a fab package, and the terminal datum shipped in
this session was in the deployed build that ran it.

**One thing that run exposed — and a correction.** That board's `J1` is the
*other* vendored Kangnex family — `CONN-TH_WJ126V-5.0-2P`, not the WJ128V this
session reviewed. It is edge-zoned `left` and the fab gate certified it
`status=ok, opening_board_deg=180`, which is the **reviewed** datum, not a
heuristic guess: both the repo bundle (`kicraft/parts_library/
screw-terminal-5mm-2p`) and the pinned tree's copy carry a vendor-style
`PCB Edge` marker at footprint-local `(-3.8, 0)`, added with the KC-YJ7Q69 fix
(`7c82a8d`) — and the leaf board's J1 carries exactly that marker (local
`-3.8, 0`), so the placer and the gate both read it. The same holds for the 3P
(`CONN-TH_3P-P5.00_WJ126V-5.0-3P`, marker at `(0, 4.13)` → 90°).

What *is* true, and is only a documentation risk: the **home-tier fetch cache**
(`~/.kicraft/parts/screw-terminal-5mm-{2p,3p}`) holds older copies of those two
bundles with **no marker** — same manifest `version`, different content. That is
harmless because the loader's tier order prefers curated over cache
(`loader.py`: "curated repo content beats auto-fetched caches"), so those copies
never reach a board; do not "fix" them by editing the cache, and do not read a
`detect_opening_direction` result taken directly from that tier as evidence
(it returns `None` there — which is what made this session mis-read the shipped
board's verdict at first).

Terminal-datum inventory after this session (the vendored 5.00 mm families the
legacy backend can pick):

| footprint | datum | evidence |
|---|---|---|
| `CONN-TH_4P-P5.00_WJ128V-4P-5.0-14-00A` | **90°**, marker `(0, 5.27)` | reviewed this session (mesh + drawing), `_REVIEWED_OPENINGS` |
| `CONN-TH_5P-P5.00_WJ128V-5P-5.0-14-00A` | **90°**, marker `(0, 5.27)` | same family; 5P mesh corroborates |
| `CONN-TH_WJ126V-5.0-2P` | **180°**, marker `(-3.8, 0)` | footprint marker, `7c82a8d` (the shipped `KC-MQNE7R` J1) |
| `CONN-TH_3P-P5.00_WJ126V-5.0-3P` | **90°**, marker `(0, 4.13)` | footprint marker, `7c82a8d` |
| `CONN-TH_4P-P5.00_WJ126V-5.0-4P-1` | **none** | no marker in-repo, no 3D model → unmeasured, blocked at compose (honest) |
| `CONN-TH_5P-P5.00_WJ126V-5.0-5P` | **none** | same |
| `CONN-TH_5P-P5.00_WJ127-5.0-5P` | **none** | no marker anywhere; mesh does not register cleanly on its own outline |

So the remaining terminal work is the last three rows (a reviewed family for them
— the terminal doc's fix 2 — or a per-part drawing review), not the 2P/3P.

### C. Backend A/B — results (9 runs, $0.1501, no build, no production change)

Three unseen briefs from the frozen 60 (`5 V greenhouse sensor hub`, `dual-channel
line receiver`, `42 mm LED beacon`); every arm `rc=1` — **no arm completed all
five stages, and nothing reached the build**:

| brief | `cur0` current, reasoning 0 | `curR` current, reasoning 4096 | `leg` pinned `bc6a2f8` |
|---|---|---|---|
| 1 greenhouse hub | intent ✓2 · fs ✓1 · **architecture ✗3** (`source_obligation_not_retained`) | intent ✓1 · fs ✓1 · arch ✓2 · **bom ✗** work unit `qwiic_1`: *requires 1 real qwiic-receptacle, found 0* (emitted `Conn_01x04`/`PinSocket_1x04`) | intent ✓1 · fs ✓1 · arch ✓2 · **bom ✓4** · **wiring ✗** deficit park |
| 2 line receiver | intent ✓1 · fs ✓1 · **architecture ✗3** (contract refused) | intent ✓1 · fs ✓1 · arch ✓2 · **bom ✗** work unit `input_condition_left`: *requires 1 real coupling-capacitor / input-protection, found 0* (emitted generic `Device:R`/`Device:C`) | intent ✓1 · fs ✓1 · arch ✓1 · **bom ✓3** · **wiring ✗** park |
| 3 LED beacon | intent ✓2 · fs ✓1 · **architecture ✗3** | intent ✓1 · fs ✓1 · **architecture ✗3** (functional-block mapping: `MOUNTING` block has no implementation requirement on a sheet) | intent ✓1 · fs ✓1 · arch ✓1 · **bom ✓2** · **wiring ✗** park |

Three measured facts:

1. **The pinned engine still reaches further on unseen briefs**: 4/5 stages on
   3/3 (bom committed after 2–4 attempts), and its only failure is the deficit
   *park* at wiring — "awaiting a clarifying answer" — the model-authored
   protocol park the 34/34 record already describes, with the reconcile budget
   spent.
2. **Reasoning measurably helps the current engine's architecture stage**:
   reasoning 0 → 0/3 committed (3 attempts each, `source_obligation_not_retained`
   / block-sheet mapping classes); reasoning 4096 → 2/3 committed, at 4–8× the
   architecture cost ($0.022–0.026 vs $0.002–0.006). With architecture solved,
   the current engine then dies in the **bom work-unit contract**: the typed
   work unit demands a class-tagged real part ("requires 1 real
   coupling-capacitor") and the model emits a generic `Device:C` — a refusal
   that costs nothing but blocks the run 2/2.
3. **Neither side is shippable on these briefs**, so this is not evidence for a
   flip. It is evidence for the two current-side contract classes to fix next
   (`architecture` obligation retention / block-sheet mapping;
   `bom` work-unit class resolution), which is exactly the plan's A2 workstream
   — now with a measured reproduction on the current side rather than only the
   legacy-authored one.

**Decision (owner: A/B first, then flip): do not flip yet.** Keep `pipeline:
legacy` while the two current-side contract classes above are fixed and the
same nine runs are repeated; turn the reasoning budget on for the `current`
arm of that re-measurement (it is the single knob this A/B shows moving
architecture, and the throwaway-config mechanism (`KICRAFT_ROUTING_CONFIG`)
means it can be measured without touching production). Also worth carrying:
the reasoning arm's cost is per-attempt, so the BOM stage (10–70× architecture
in wall time) is where a reasoning default would be felt.

Budget note: the A/B spent $0.1501; the ledger stands at $137.01 of a $250
total ceiling (today $19.22 of $20 left), so the re-measurement fits.

Method, for the re-measurement (production `routing.json` untouched
throughout):

| arm | tree | knobs |
|---|---|---|
| `cur0` | current checkout | `KICRAFT_ROUTING_CONFIG=/tmp/ab/rc-current.json` (luna, temp 0, reasoning 0) |
| `curR` | current checkout | same file + `design_reasoning_tokens: 4096` (a throwaway routing config) |
| `leg` | `~/KiCraft-legacy` @ `bc6a2f8` | `LEGACY_ENV` (openai route, 0.20/1.20 caps, reasoning 0) |

Each run: `stage_driver run --brief <file> --workspace <tmp> --budget 0.15
--no-build --quality draft`; logs in `/tmp/ab/<arm>-b<N>.log`, per-stage lines
`[ok|FAIL] <stage> cost=$… attempts=…`.
